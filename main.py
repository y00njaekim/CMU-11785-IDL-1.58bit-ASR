"""
Main script for data preparation and testing.
This script demonstrates the complete data preprocessing pipeline:
1. Check tokenizer (train with: python train_tokenizer.py)
2. Compute CMVN statistics
3. Create dataloaders
4. Test data loading
"""

import os
import torch
from src.data.dataset import compute_cmvn_stats, get_dataloaders
import wandb

def main():
    """
    Main function to prepare data and test the pipeline.
    """
    # Configuration
    tokenizer_path = "src/data/tokenizer.model"
    cmvn_stats_path = "src/data/cmvn_stats.pt"

    batch_size = 4  # Small batch size for testing

    print("=" * 80)
    print("LibriSpeech ASR Data Preparation Pipeline")
    print("=" * 80)

    # Step 1: Check tokenizer exists
    if not os.path.exists(tokenizer_path):
        print("\n[Step 1/3] ERROR: Tokenizer not found!")
        print("-" * 80)
        print(f"Tokenizer not found at: {tokenizer_path}")
        print("Please train the tokenizer first:")
        print("  python train_tokenizer.py")
        print()
        return
    else:
        print(f"\n[Step 1/3] ✓ Tokenizer found: {tokenizer_path}")

    # Step 2: Compute CMVN statistics if not exists
    if not os.path.exists(cmvn_stats_path):
        print("\n[Step 2/3] Computing CMVN statistics...")
        print("-" * 80)
        compute_cmvn_stats(
            tokenizer_path=tokenizer_path,
            output_path=cmvn_stats_path,
            num_samples=1000,  # Use 1000 samples for computing statistics
        )
    else:
        print(f"\n[Step 2/3] CMVN statistics already exist: {cmvn_stats_path}")

    # # Step 3: Create dataloaders
    # print("\n[Step 3/3] Creating dataloaders...")
    # print("-" * 80)

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     tokenizer_path=tokenizer_path,
    #     cmvn_stats_path=cmvn_stats_path,
    #     batch_size=batch_size,
    #     num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    # )

    # # Test data loading
    # print("\n" + "=" * 80)
    # print("Testing Data Loading")
    # print("=" * 80)

    # # Test train loader
    # print("\n[Train Loader]")
    # train_batch = next(iter(train_loader))
    # print(f"  FBank shape: {train_batch['fbank'].shape}")
    # print(f"    - Batch size: {train_batch['fbank'].shape[0]}")
    # print(f"    - Max time steps: {train_batch['fbank'].shape[1]}")
    # print(f"    - Feature dim: {train_batch['fbank'].shape[2]}")
    # print(f"  Labels shape: {train_batch['labels'].shape}")
    # print(f"  FBank lengths: {train_batch['fbank_lengths']}")
    # print(f"  Label lengths: {train_batch['label_lengths']}")
    # print(
    #     f"  FBank value range: [{train_batch['fbank'].min():.3f}, {train_batch['fbank'].max():.3f}]"
    # )

    # # Test validation loader
    # print("\n[Validation Loader]")
    # val_batch = next(iter(val_loader))
    # print(f"  FBank shape: {val_batch['fbank'].shape}")
    # print(f"  Labels shape: {val_batch['labels'].shape}")
    # print(f"  FBank lengths: {val_batch['fbank_lengths']}")
    # print(f"  Label lengths: {val_batch['label_lengths']}")

    # # Test test loader
    # print("\n[Test Loader]")
    # test_batch = next(iter(test_loader))
    # print(f"  FBank shape: {test_batch['fbank'].shape}")
    # print(f"  Labels shape: {test_batch['labels'].shape}")
    # print(f"  FBank lengths: {test_batch['fbank_lengths']}")
    # print(f"  Label lengths: {test_batch['label_lengths']}")

    # print("\n" + "=" * 80)
    # print("Data preparation pipeline completed successfully!")
    # print("=" * 80)


if __name__ == "__main__":
    main()