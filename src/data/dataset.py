import os
# Force SentencePiece to use a single thread per process.
# This fixes SegFaults (Exit -11) when running tokenizers in DDP.
os.environ.setdefault("SPM_NUM_THREADS", "1")
import json
import io
import torch
import torchaudio
import sentencepiece as spm
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_from_disk, concatenate_datasets, Audio
from tqdm import tqdm


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset with ESPnet-style preprocessing.
    
    Following preprocessing steps:
    1. Audio resampling to 16kHz
    2. 80-dimensional FBank feature extraction
    3. CMVN (Cepstral Mean and Variance Normalization)
    4. SpecAugment data augmentation (training only)
    5. Text tokenization using BPE
    """
    
    def __init__(
        self,
        split: str,
        tokenizer_path: str,
        cmvn_stats: Optional[Dict[str, torch.Tensor]] = None,
        apply_spec_augment: bool = False,
        spec_augment_config: Optional[Dict] = None,
    ):
        """
        Initialize LibriSpeech dataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            tokenizer_path: Path to the trained SentencePiece model
            cmvn_stats: Dictionary containing 'mean' and 'std' for CMVN normalization
            apply_spec_augment: Whether to apply SpecAugment (only for training)
            spec_augment_config: Configuration for SpecAugment
        """
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.cmvn_stats = cmvn_stats
        self.apply_spec_augment = apply_spec_augment
        
        # Lazy initialization of tokenizer to avoid DDP/multiprocessing issues
        # Initialize tokenizer only when first needed (in __getitem__)
        self._tokenizer = None
        
        self.dataset = self._load_dataset()
        
        if self.apply_spec_augment:
            if spec_augment_config is None:
                spec_augment_config = {
                    'freq_mask_param': 27,
                    'time_mask_param': 100,
                    'num_freq_mask': 2,
                    'num_time_mask': 2,
                }
            self.spec_augment = SpecAugment(**spec_augment_config)
        else:
            self.spec_augment = None
    
    def _load_dataset(self):
        """
        Load and concatenate LibriSpeech dataset splits from local disk.
        """
        if self.split == 'train':
            splits = ["train.clean.100", "train.clean.360", "train.other.500"]
        elif self.split == 'validation':
            splits = ["dev.clean", "dev.other"]
        elif self.split == 'test':
            splits = ["test.clean", "test.other"]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        data_dir = "data"
        print(f"Loading {self.split} data from {data_dir}...")
        datasets = []
        
        for split_name in splits:
            dataset_path = os.path.join(data_dir, f"{split_name}_subset")
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path not found: {dataset_path}. Skipping.")
                continue

            print(f"  Loading {split_name} from {dataset_path}...")
            ds = load_from_disk(dataset_path)
            # Use decode=False to avoid torchcodec dependency, we'll decode manually with soundfile
            ds = ds.cast_column("audio", Audio(decode=False, sampling_rate=16000))
            
            print(f"    Loaded {len(ds)} samples")
            datasets.append(ds)
        
        if not datasets:
            raise FileNotFoundError(f"No datasets found for split '{self.split}' in '{data_dir}'. Please check data paths.")

        combined_dataset = concatenate_datasets(datasets)
        print(f"  Total samples after combining: {len(combined_dataset)}")
        
        return combined_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def tokenizer(self):
        """Lazy initialization of tokenizer to avoid DDP/multiprocessing issues."""
        if self._tokenizer is None:
            # Use a lock-free approach: each process initializes its own tokenizer
            # This avoids race conditions in SentencePiece's C++ extension
            # Add a small process-specific delay to avoid simultaneous initialization
            import time
            import os
            rank = int(os.environ.get('RANK', 0))
            time.sleep(0.01 * rank)  # Stagger initialization by rank
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.load(self.tokenizer_path)
        return self._tokenizer
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - fbank: FBank features (T, 80)
                - labels: Tokenized text (L,)
                - fbank_length: Length of FBank features
                - label_length: Length of labels
        """
        item = self.dataset[idx]
        
        # Decode audio manually using soundfile to avoid torchcodec dependency
        audio_data = item['audio']
        
        if isinstance(audio_data, dict):
            if 'bytes' in audio_data and audio_data['bytes']:
                # Audio is stored as bytes in Arrow format, decode it
                audio_bytes = audio_data['bytes']
                waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
                waveform = torch.FloatTensor(waveform)
            elif 'path' in audio_data and audio_data['path']:
                # Try to read from path if it's an absolute path
                audio_path = audio_data['path']
                if os.path.isabs(audio_path) and os.path.exists(audio_path):
                    waveform, sample_rate = sf.read(audio_path)
                    waveform = torch.FloatTensor(waveform)
                else:
                    # Path is relative or doesn't exist, try bytes or fall back to torchaudio
                    if 'bytes' in audio_data and audio_data['bytes']:
                        audio_bytes = audio_data['bytes']
                        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
                        waveform = torch.FloatTensor(waveform)
                    else:
                        raise ValueError(f"Audio path '{audio_path}' doesn't exist and no bytes available")
            elif 'array' in audio_data:
                # Already decoded (shouldn't happen with decode=False, but handle it)
                waveform = torch.FloatTensor(audio_data['array'])
                sample_rate = audio_data.get('sampling_rate', 16000)
            else:
                raise ValueError(f"Audio data format not recognized: {list(audio_data.keys())}")
        else:
            raise ValueError(f"Audio data is not a dict: {type(audio_data)}")
        
        # Ensure waveform is 1D, then add channel dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            # If stereo, take first channel
            waveform = waveform[0:1]
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=16000,
        )
        
        if self.cmvn_stats is not None:
            fbank = (fbank - self.cmvn_stats['mean']) / self.cmvn_stats['std']
        
        # Apply SpecAugment if enabled (training only)
        if self.spec_augment is not None:
            fbank = self.spec_augment(fbank)
        
        # Tokenize text (uppercase following ESPnet convention)
        text = item['text'].upper().strip()
        labels = self.tokenizer.encode(text, out_type=int)
        labels = torch.LongTensor(labels)
        
        return {
            'fbank': fbank,
            'labels': labels,
            'fbank_length': torch.tensor(fbank.size(0)),
            'label_length': torch.tensor(labels.size(0)),
        }


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_mask: int = 2,
        num_time_mask: int = 2,
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            num_freq_mask: Number of frequency masks to apply
            num_time_mask: Number of time masks to apply
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to FBank features.
        
        Args:
            x: FBank features (T, F) where T is time and F is frequency
        
        Returns:
            Augmented FBank features
        """
        x = x.clone()
        
        # Apply frequency masking
        for _ in range(self.num_freq_mask):
            x = self._mask_along_axis(x, mask_param=self.freq_mask_param, axis=1)
        
        # Apply time masking
        for _ in range(self.num_time_mask):
            x = self._mask_along_axis(x, mask_param=self.time_mask_param, axis=0)
        
        return x
    
    @staticmethod
    def _mask_along_axis(x: torch.Tensor, mask_param: int, axis: int) -> torch.Tensor:
        """Mask along specified axis."""
        if axis == 0:
            # Time axis
            max_frames = x.size(0)
            mask_width = min(mask_param, max_frames)
            mask_start = torch.randint(0, max(1, max_frames - mask_width), (1,)).item()
            x[mask_start:mask_start + mask_width, :] = 0.0
        else:
            # Frequency axis
            num_bins = x.size(1)
            mask_width = min(mask_param, num_bins)
            mask_start = torch.randint(0, max(1, num_bins - mask_width), (1,)).item()
            x[:, mask_start:mask_start + mask_width] = 0.0
        
        return x


class CollateFunction:
    """
    Collate function for batching variable-length sequences.
    Applies dynamic padding to FBank features and labels.
    """
    
    def __init__(self, pad_value: float = 0.0, label_pad_value: int = 0):
        """
        Args:
            pad_value: Padding value for FBank features
            label_pad_value: Padding value for labels (should match tokenizer pad_id)
        """
        self.pad_value = pad_value
        self.label_pad_value = label_pad_value
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with dynamic padding.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Batched dictionary with padded tensors
        """
        # Get maximum lengths in batch
        max_fbank_len = max(item['fbank'].size(0) for item in batch)
        max_label_len = max(item['labels'].size(0) for item in batch)
        
        batch_size = len(batch)
        fbank_dim = batch[0]['fbank'].size(1)
        
        # Initialize padded tensors
        padded_fbanks = torch.full(
            (batch_size, max_fbank_len, fbank_dim),
            self.pad_value,
            dtype=torch.float32
        )
        padded_labels = torch.full(
            (batch_size, max_label_len),
            self.label_pad_value,
            dtype=torch.long
        )
        fbank_lengths = torch.zeros(batch_size, dtype=torch.long)
        label_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill padded tensors
        for i, item in enumerate(batch):
            fbank_len = item['fbank'].size(0)
            label_len = item['labels'].size(0)
            
            padded_fbanks[i, :fbank_len, :] = item['fbank']
            padded_labels[i, :label_len] = item['labels']
            fbank_lengths[i] = fbank_len
            label_lengths[i] = label_len
        
        return {
            'fbank': padded_fbanks,
            'labels': padded_labels,
            'fbank_lengths': fbank_lengths,
            'label_lengths': label_lengths,
        }


def compute_cmvn_stats(
    tokenizer_path: str,
    output_path: str = "src/data/cmvn_stats.json",
    num_samples: int = 1000,
) -> Dict[str, torch.Tensor]:
    """
    Compute CMVN (Cepstral Mean and Variance Normalization) statistics
    from the training data.
    
    DEPENDENCY: Requires tokenizer to be trained first.
    
    Args:
        tokenizer_path: Path to tokenizer (required for dataset initialization)
        output_path: Path to save computed statistics
        num_samples: Number of samples to use for computing statistics
    
    Returns:
        Dictionary containing 'mean' and 'std' tensors
    
    Raises:
        FileNotFoundError: If tokenizer model does not exist
    """
    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Please train the tokenizer first using train_tokenizer()."
        )
    
    print("Computing CMVN statistics from training data...")
    
    # Load training data
    train_dataset = LibriSpeechDataset(
        split='train',
        tokenizer_path=tokenizer_path,
        cmvn_stats=None,  # Don't apply normalization yet
        apply_spec_augment=False,
    )
    
    # Collect FBank features
    all_fbanks = []
    print(f"Collecting FBank features from {min(num_samples, len(train_dataset))} samples...")
    
    for i in tqdm(range(min(num_samples, len(train_dataset)))):
        item = train_dataset[i]
        all_fbanks.append(item['fbank'])
    
    # Concatenate all features
    all_fbanks = torch.cat(all_fbanks, dim=0)  # (Total_frames, 80)
    
    # Compute mean and std
    mean = all_fbanks.mean(dim=0)  # (80,)
    std = all_fbanks.std(dim=0)    # (80,)
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    cmvn_stats = {
        'mean': mean,
        'std': std,
    }
    
    # Save statistics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(cmvn_stats, output_path)
    print(f"CMVN statistics saved to {output_path}")
    print(f"  Mean shape: {mean.shape}, range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std shape: {std.shape}, range: [{std.min():.3f}, {std.max():.3f}]")
    
    return cmvn_stats

    
from torch.utils.data import Sampler
from typing import List

# length_sampler.py

import math
import random
from typing import List, Iterator, Sequence
from torch.utils.data import Sampler


class LengthAwareBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups examples of similar length, then shuffles
    the resulting batches each epoch.

    - Uses per-example lengths (e.g., fbank_length) to sort indices.
    - Forms contiguous batches in that sorted order -> length-homogeneous.
    - Shuffles the list of batches every epoch (and optionally shuffles
      the order of samples within each batch).

    This gives:
      * good padding efficiency
      * still-randomized training order (at the batch level)
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        shuffle_within_batch: bool = True,
    ) -> None:
        """
        Args:
            lengths: A sequence of lengths, one per dataset index (len = len(dataset)).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the order of batches each epoch.
            drop_last: Drop the last batch if it's smaller than batch_size.
            shuffle_within_batch: Whether to shuffle sample order inside each batch.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(lengths) == 0:
            raise ValueError("lengths must be non-empty")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.shuffle_within_batch = shuffle_within_batch

        # Sort indices by length once (ascending).
        self.sorted_indices = sorted(range(len(self.lengths)),
                                     key=lambda i: self.lengths[i])

        # Precompute how many batches we'd get.
        self._num_batches = self._compute_num_batches()

    def _compute_num_batches(self) -> int:
        n = len(self.sorted_indices)
        if self.drop_last:
            return n // self.batch_size
        else:
            return math.ceil(n / self.batch_size)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Copy sorted indices so we don’t mutate the original list.
        indices = self.sorted_indices.copy()

        # Form contiguous batches from the sorted indices.
        batches: List[List[int]] = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)

        # Shuffle the batch order each epoch.
        if self.shuffle:
            random.shuffle(batches)

        # Optionally shuffle inside each batch as well.
        if self.shuffle_within_batch:
            for b in batches:
                random.shuffle(b)

        # Yield batches.
        for b in batches:
            yield b


def get_dataloaders(
    tokenizer_path: str,
    cmvn_stats_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    DEPENDENCIES: Requires both tokenizer and CMVN statistics to be prepared first.
    
    Args:
        tokenizer_path: Path to the trained tokenizer model
        cmvn_stats_path: Path to precomputed CMVN statistics
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_fraction: Fraction of training data to use (0.0-1.0, default: 1.0 for 100%)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Raises:
        FileNotFoundError: If tokenizer or CMVN statistics do not exist
    """
    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Please train the tokenizer first using train_tokenizer()."
        )
    
    # Check if CMVN statistics exist
    if not os.path.exists(cmvn_stats_path):
        raise FileNotFoundError(
            f"CMVN statistics not found at {cmvn_stats_path}. "
            f"Please compute CMVN statistics first using compute_cmvn_stats()."
        )
    
    # Load CMVN statistics
    cmvn_stats = torch.load(cmvn_stats_path)
    
    # Create datasets
    train_dataset = LibriSpeechDataset(
        split='train',
        tokenizer_path=tokenizer_path,
        cmvn_stats=cmvn_stats,
        apply_spec_augment=True,  # Apply SpecAugment for training
    )
    
    val_dataset = LibriSpeechDataset(
        split='validation',
        tokenizer_path=tokenizer_path,
        cmvn_stats=cmvn_stats,
        apply_spec_augment=False,  # No augmentation for validation
    )
    
    test_dataset = LibriSpeechDataset(
        split='test',
        tokenizer_path=tokenizer_path,
        cmvn_stats=cmvn_stats,
        apply_spec_augment=False,  # No augmentation for testing
    )
    
    # Create collate function
    collate_fn = CollateFunction(pad_value=0.0, label_pad_value=0)

    # Slice training dataset BEFORE computing lengths to avoid processing unused data
    if train_fraction < 1.0:
        total_samples = len(train_dataset)
        subset_size = int(total_samples * train_fraction)
        
        # Create a deterministic random permutation
        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(total_samples, generator=g)[:subset_size].tolist()
        
        # Create a Subset BEFORE computing lengths
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {subset_size}/{total_samples} training samples ({train_fraction*100:.1f}%)")

    # 1. Precompute lengths (one-time cost at startup) - now only for the subset
    train_lengths = []
    for i in tqdm(range(len(train_dataset))):
        # LibriSpeechDataset.__getitem__ returns 'fbank' [T, 80]
        # but we don't want to actually compute fbank for all samples here
        # (that would be expensive). Instead, you can:
        #
        # Option A (cheap, if you have lengths stored in HF dataset):
        #   train_lengths.append(int(train_dataset.dataset[i]["audio"]["array"].shape[0] / hop_size_estimate))
        #
        # Option B (simpler but more expensive): just call __getitem__:
        item = train_dataset[i]
        train_lengths.append(int(item["fbank"].size(0)))

    # 2. Create the sampler
    train_batch_sampler = LengthAwareBatchSampler(
        lengths=train_lengths,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        shuffle_within_batch=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        batch_sampler=train_batch_sampler,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

