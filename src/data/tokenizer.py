import os
import sentencepiece as spm
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def train_tokenizer(
    output_dir: str = "src/data",
    vocab_size: int = 5000,
    model_prefix: str = "tokenizer",
    train_splits: list = None,
):
    """
    Train a SentencePiece BPE tokenizer on LibriSpeech training data.
    
    Args:
        output_dir: Directory to save the trained tokenizer model
        vocab_size: Size of the vocabulary (default: 5000, following ESPnet recipe)
        model_prefix: Prefix for the output model files
        train_splits: List of LibriSpeech splits to use for training
    
    Returns:
        Path to the trained tokenizer model file
    """
    if train_splits is None:
        # Use all training splits as specified in the paper
        train_splits = ["train.clean.100", "train.clean.360", "train.other.500"]
    
    print("Loading LibriSpeech training data...")
    datasets = []
    for split in train_splits:
        print(f"  Loading {split}...")
        ds = load_dataset("librispeech_asr", "clean" if "clean" in split else "other", 
                         split=split.replace(".", "-"), trust_remote_code=True)
        datasets.append(ds)
    
    # Concatenate all training datasets
    train_dataset = concatenate_datasets(datasets)
    print(f"Total training samples: {len(train_dataset)}")
    
    # Extract and normalize text data
    # Following ESPnet convention: uppercase text
    print("Extracting and normalizing text data...")
    text_file = os.path.join(output_dir, "librispeech_train_text.txt")
    
    with open(text_file, "w", encoding="utf-8") as f:
        for item in tqdm(train_dataset, desc="Processing text"):
            # Normalize to uppercase following ESPnet convention
            normalized_text = item["text"].upper().strip()
            f.write(normalized_text + "\n")
    
    print(f"Text data saved to {text_file}")
    
    # Train SentencePiece model
    # Parameters follow ESPnet recipe configuration
    print(f"Training SentencePiece BPE model with vocab_size={vocab_size}...")
    model_path = os.path.join(output_dir, model_prefix)
    
    spm.SentencePieceTrainer.train(
        f"--input={text_file} "
        f"--model_prefix={model_path} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=1.0 "
        f"--pad_id=0 "
        f"--unk_id=1 "
        f"--bos_id=2 "
        f"--eos_id=3 "
        f"--pad_piece=<blank> "
        f"--unk_piece=<unk> "
        f"--bos_piece=<sos> "
        f"--eos_piece=<eos>"
    )
    
    print(f"Tokenizer trained successfully!")
    print(f"Model saved to: {model_path}.model")
    print(f"Vocab saved to: {model_path}.vocab")
    
    # Clean up temporary text file
    os.remove(text_file)
    print(f"Cleaned up temporary file: {text_file}")
    
    return f"{model_path}.model"


if __name__ == "__main__":
    # Train tokenizer with default parameters
    tokenizer_path = train_tokenizer()
    print(f"\nTokenizer training complete: {tokenizer_path}")

