"""
Download LibriSpeech dataset with subset support.

This script downloads LibriSpeech data and caches it locally, with options to:
- Download only specific splits
- Download a percentage of the data (uses split slicing to avoid downloading unnecessary files)
- Cache for reuse across different tasks

Available splits:
    Training:   train.clean.100 (28,539 samples, ~100h)
                train.clean.360 (104,014 samples, ~360h)
                train.other.500 (148,688 samples, ~500h)
    Validation: dev.clean (2,703 samples)
                dev.other (2,864 samples)
    Test:       test.clean (2,620 samples)
                test.other (2,939 samples)

Usage:
    # Download small subset for development (only downloads needed parquet files)
    python download_data.py --subset 0.01 --splits train.clean.100

    # Download full training data
    python download_data.py --splits train.clean.100 train.clean.360 train.other.500

    # Download validation and test sets
    python download_data.py --splits dev.clean test.clean
"""

import os
import sys  # Add this
import argparse
import gc
import datasets

datasets.config.STREAMING_READ_MAX_RETRIES = 40  # default
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10  # default

from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm


def download_librispeech(
    splits=None,
    subset_percentage=1.0,
    cache_dir="data/hf_cache",
):
    """
    Download LibriSpeech dataset splits and cache them locally.

    Args:
        splits: List of splits to download (e.g., ["train.clean.100", "test.clean"])
        subset_percentage: Percentage of each split to download (0.0-1.0)
        cache_dir: Directory to cache the downloaded data

    Returns:
        Dictionary mapping split names to dataset objects
    """
    if splits is None:
        splits = [
            "train.clean.100",
            "train.clean.360",
            "train.other.500",
            "dev.clean",
            "dev.other",
            "test.clean",
            "test.other",
        ]

    os.makedirs(cache_dir, exist_ok=True)

    datasets = {}

    print("=" * 80)
    print("LibriSpeech Dataset Download")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print(f"Subset percentage: {subset_percentage * 100:.1f}%")
    print(f"Splits to download: {', '.join(splits)}")
    print()

    for split in splits:
        print(f"\nDownloading {split}...")

        # Determine configuration and actual split name
        # User format: train.clean.100 -> config='clean', split_name='train.100'
        # User format: dev.clean -> config='clean', split_name='validation'
        # User format: test.other -> config='other', split_name='test'

        if "clean" in split:
            config = "clean"
            actual_split = split.replace(".clean", "")
        elif "other" in split:
            config = "other"
            actual_split = split.replace(".other", "")
        else:
            config = "clean"  # Default
            actual_split = split

        # Map dev -> validation
        if actual_split == "dev":
            actual_split = "validation"

        try:
            # For subsets, use streaming mode to avoid downloading all parquet files
            if subset_percentage < 1.0:
                # Known sizes for LibriSpeech splits (using dot format)
                split_sizes = {
                    "train.clean.100": 28539,
                    "train.clean.360": 104014,
                    "train.other.500": 148688,
                    "dev.clean": 2703,
                    "dev.other": 2864,
                    "test.clean": 2620,
                    "test.other": 2939,
                }

                total_size = split_sizes.get(split, 10000)
                subset_size = max(1, int(total_size * subset_percentage))
                print(
                    f"  Downloading subset: {subset_size}/{total_size} samples ({subset_percentage*100:.1f}%)"
                )

                # Use streaming to download only needed samples
                print(f"  Using streaming mode to minimize downloads...")
                ds_stream = load_dataset(
                    "librispeech_asr",
                    config,
                    split=actual_split,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                    streaming=True,
                )
                ds_stream = ds_stream.cast_column("audio", Audio(decode=False))

                # Take only the subset we need
                samples = []
                for i, sample in enumerate(ds_stream):
                    if i >= subset_size:
                        break
                    samples.append(sample)
                    if (i + 1) % 50 == 0:
                        print(
                            f"  Downloaded {i + 1}/{subset_size} samples...", end="\r"
                        )

                print(f"  Downloaded {len(samples)}/{subset_size} samples     ")

                # Convert to regular dataset
                if not samples:  # Handle case where no samples were downloaded
                    print(f"  Warning: No samples downloaded for {split}")
                    continue

                ds = Dataset.from_dict(
                    {
                        key: [sample[key] for sample in samples]
                        for key in samples[0].keys()
                    }
                )

                # Important: Delete the streaming dataset to free resources
                del ds_stream

                print(f"  Loaded {len(ds)} samples")
            else:
                # Download full dataset normally
                ds = load_dataset(
                    "librispeech_asr",
                    config,
                    split=actual_split,
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                )
                print(f"  Loaded {len(ds)} samples")

            datasets[split] = ds
            ds.save_to_disk(f"{cache_dir}/{split}_subset")

        except Exception as e:
            print(f"  Error downloading {split}: {e}")
            # Force cleanup of any open streaming connections
            gc.collect()
            continue

    print()
    print("=" * 80)
    print("✓ Download Complete!")
    print("=" * 80)
    print(f"Total splits downloaded: {len(datasets)}")
    print(f"Cache location: {cache_dir}")
    print()

    return datasets


def save_text_data(datasets, output_file="data/librispeech_text.txt"):
    """
    Extract and save all text transcriptions from downloaded datasets.

    This is useful for tokenizer training - you can download once and
    train the tokenizer separately.

    Args:
        datasets: Dictionary of dataset objects
        output_file: Path to save the text data
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_samples = sum(len(ds) for ds in datasets.values())

    print(f"\nExtracting text data from {len(datasets)} splits...")
    print(f"Total samples: {total_samples}")

    with open(output_file, "w", encoding="utf-8") as f:
        for split_name, ds in datasets.items():
            print(f"  Processing {split_name}...")
            for item in tqdm(ds, desc=f"  {split_name}"):
                # Normalize to uppercase following ESPnet convention
                normalized_text = item["text"].upper().strip()
                f.write(normalized_text + "\n")

    print(f"\n✓ Text data saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech dataset with subset support"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,  # Will use all splits from function default
        help="Splits to download (e.g., train.clean.100 test.clean). Default: all splits",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Percentage of data to download (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/hf_cache",
        help="Directory to cache downloaded data",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Extract and save text data for tokenizer training",
    )
    parser.add_argument(
        "--text-output",
        type=str,
        default="data/librispeech_text.txt",
        help="Output file for text data (used with --save-text)",
    )

    args = parser.parse_args()

    # Validate subset percentage
    if not 0.0 < args.subset <= 1.0:
        print("Error: --subset must be between 0.0 and 1.0")
        return

    # Download datasets
    datasets = download_librispeech(
        splits=args.splits,
        subset_percentage=args.subset,
        cache_dir=args.cache_dir,
    )

    # Optionally save text data
    if args.save_text and datasets:
        save_text_data(datasets, args.text_output)
        print()
        print("Next step: Train tokenizer using:")
        print(f"python train_tokenizer.py --text-file {args.text_output}")
        # Force cleanup and exit
    if datasets:
        datasets.clear()
    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    main()