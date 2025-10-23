import os
import sys
import torch
import matplotlib.pyplot as plt

# Add src directory to path to import custom modules
# This allows the script to find the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import LibriSpeechDataset


def generate_spectrogram_from_test_sample(sample_index=0, output_dir="test"):
    """
    Loads a sample from the test dataset, generates its FBank spectrogram,
    and saves it as an image.

    Args:
        sample_index (int): The index of the sample to use from the test set.
        output_dir (str): The directory to save the output spectrogram image.
    """
    tokenizer_path = "src/data/tokenizer.model"
    cmvn_stats_path = "src/data/cmvn_stats.pt"

    # --- 1. Check for prerequisite files ---
    if not os.path.exists(tokenizer_path) or not os.path.exists(cmvn_stats_path):
        print("="*80)
        print("ERROR: Tokenizer or CMVN stats file not found.")
        print(f"  - Searched for tokenizer at: {tokenizer_path}")
        print(f"  - Searched for CMVN stats at: {cmvn_stats_path}")
        print("\nPlease run 'python main.py' first to generate these necessary files.")
        print("="*80)
        return

    # --- 2. Load the dataset ---
    print("Loading test dataset...")
    try:
        # Load CMVN statistics
        cmvn_stats = torch.load(cmvn_stats_path)

        # Create an instance of the test dataset
        test_dataset = LibriSpeechDataset(
            split='test',
            tokenizer_path=tokenizer_path,
            cmvn_stats=cmvn_stats,
            apply_spec_augment=False,  # No augmentation for visualization
        )
    except FileNotFoundError as e:
        print(f"ERROR: Could not load the dataset. Have you downloaded the data?")
        print(f"Details: {e}")
        return

    if len(test_dataset) == 0:
        print("Test dataset is empty. Please check your data directory and ensure data is present.")
        return

    # --- 3. Get a sample and its data ---
    if sample_index >= len(test_dataset):
        print(f"ERROR: sample_index {sample_index} is out of bounds for the test dataset of size {len(test_dataset)}.")
        return
        
    print(f"Fetching sample {sample_index} from the test set...")
    sample = test_dataset[sample_index]

    fbank = sample['fbank']
    labels = sample['labels']
    
    # Decode the labels back to text to see the transcription
    tokenizer = test_dataset.tokenizer
    transcription = tokenizer.decode(labels.tolist())

    print(f"\n--- Sample Information ---")
    print(f"  Transcription: '{transcription}'")
    print(f"  FBank (Spectrogram) Shape: {fbank.shape} (Time Steps, Frequency Bins)")
    print(f"---")

    # --- 4. Visualize and save the spectrogram ---
    print("Generating spectrogram plot...")
    plt.figure(figsize=(12, 5))
    
    # Transpose the FBank features for standard visualization (Time on x-axis)
    plt.imshow(fbank.T, aspect='auto', origin='lower', cmap='viridis')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'FBank Spectrogram (Sample #{sample_index})')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"spectrogram_sample_{sample_index}.png")
    
    # Save the plot to a file
    plt.savefig(output_path)
    print(f"\nSpectrogram saved successfully to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # You can change the sample index here if you want to see a different sample
    generate_spectrogram_from_test_sample(sample_index=42)
