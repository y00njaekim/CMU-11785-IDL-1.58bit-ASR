
# LibriSpeech Data Loader Interface :
# --------------------------------------------------
# Provide a class `LibriSpeechDataModule` with the following minimal API:
#   class LibriSpeechDataModule:
#       def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4): ...
#       def train_dataloader(self) -> torch.utils.data.DataLoader: ...
#       def valid_dataloader(self) -> torch.utils.data.DataLoader: ...
#       def vocab_size(self) -> int: ...
#       def special_ids(self) -> dict:
#           return { 'bos_id': int, 'eos_id': int, 'pad_id': int, 'blank_id': int }
#
# Each batch from the dataloaders must be a dict with keys:
#   {
#       'feats': FloatTensor [B, T, F],    # log-mel filterbank or MFCC features
#       'feat_lens': LongTensor [B],       # valid lengths of each feature sequence
#       'tokens': LongTensor [B, U],       # target token ids (no BOS/EOS inside)
#       'token_lens': LongTensor [B],      # target lengths (<= U)
#   }
# The train loop will add BOS/EOS for the attention decoder. CTC blank is given by special_ids()['blank_id'].

# --------------------------------------------------------------
# File: dataloader_stub.py
# --------------------------------------------------------------

from typing import Dict, Optional, Iterator, Any, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import os
import random

# Use the real dataset/dataloaders from src.data.dataset
try:
    from src.data.dataset import get_dataloaders
    import sentencepiece as spm
except Exception as e:
    raise e
    get_dataloaders = None 
    spm = None


class _DummyLibriSpeechDataset(Dataset):
    """A tiny synthetic dataset that returns fixed-shape dummy batches.

    Each item is a dict with keys:
      - 'feats': FloatTensor [T, F]
      - 'feat_lens': LongTensor [] (scalar length T)
      - 'tokens': LongTensor [U]
      - 'token_lens': LongTensor [] (scalar length U)
    Default collate will stack these into the expected batched shapes.
    """

    def __init__(
        self,
        num_samples: int,
        T: int,
        F: int,
        U: int,
        vocab_size: int,
        special_ids: Dict[str, int],
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.T = T
        self.F = F
        self.U = U
        self.vocab_size = vocab_size
        self.special_ids = special_ids

        # Pre-generate deterministic tensors for simplicity and speed.
        g = torch.Generator().manual_seed(seed)
        self._feats = torch.randn(num_samples, T, F, generator=g, dtype=torch.float32)
        # Exclude special ids (0..3) from inside-target tokens
        lo = max(special_ids.get('blank_id', 3) + 1, 4)
        hi = max(vocab_size, lo + 1)
        self._tokens = torch.randint(lo, hi, (num_samples, U), generator=g, dtype=torch.long)
        self._feat_lens = torch.full((num_samples,), T, dtype=torch.long)
        self._token_lens = torch.full((num_samples,), U, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'feats': self._feats[idx],           # [T, F]
            'feat_lens': self._feat_lens[idx],   # scalar
            'tokens': self._tokens[idx],         # [U]
            'token_lens': self._token_lens[idx], # scalar
        }


class LibriSpeechDataModuleDummy:
    """Minimal dummy implementation

    This does not read any data from disk. It simply returns small, synthetic
    examples with fixed shapes so downstream code can run end-to-end.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        # Keep vocab tiny and simple. Reserve first four ids for special tokens.
        self._vocab_size = 32
        self._special_ids = {'bos_id': 1, 'eos_id': 2, 'pad_id': 0, 'blank_id': 3}

        # Fixed feature/label shapes for simplicity.
        T, F, U = 160, 80, 40
        self._train_ds = _DummyLibriSpeechDataset(
            num_samples=256,
            T=T,
            F=F,
            U=U,
            vocab_size=self._vocab_size,
            special_ids=self._special_ids,
            seed=123,
        )
        self._valid_ds = _DummyLibriSpeechDataset(
            num_samples=64,
            T=T,
            F=F,
            U=U,
            vocab_size=self._vocab_size,
            special_ids=self._special_ids,
            seed=456,
        )

    def train_dataloader(self) -> DataLoader:
        # Returns batches with keys:
        # 'feats': FloatTensor [B, T, F]
        # 'feat_lens': LongTensor [B]
        # 'tokens': LongTensor [B, U]
        # 'token_lens': LongTensor [B]
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self._valid_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def vocab_size(self) -> int:
        return self._vocab_size

    def special_ids(self) -> Dict[str, int]:
        return self._special_ids

class LibriSpeechDataModule:
    """DataModule that adapts src.data.dataset loaders to the expected training interface."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 4,
        tokenizer_path: Optional[str] = None,
        cmvn_stats_path: Optional[str] = None,
        train_fraction: float = 1.0,  # <--- NEW ARGUMENT
    ) -> None:
        if get_dataloaders is None or spm is None:
            raise ImportError(
                "Failed to import dataset utilities. Ensure src is on PYTHONPATH."
            )

        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        # Default artifact paths
        self._tokenizer_path = tokenizer_path or os.path.join("src", "data", "tokenizer.model")
        self._cmvn_stats_path = cmvn_stats_path or os.path.join("src", "data", "cmvn_stats.pt")

        # Lazy initialization of tokenizer to avoid DDP/multiprocessing issues
        # Initialize tokenizer only when first needed (in vocab_size() or special_ids())
        self._sp = None
        if not os.path.exists(self._tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at: {self._tokenizer_path}")

        # Reserve first 4 ids for specials and offset real tokens by +4
        self._token_offset = 4
        # Vocab size and special ids will be computed lazily when needed
        self._vocab_size = None
        self._special_ids = None

        # Instantiate underlying dataloaders
        # Pass train_fraction to get_dataloaders so slicing happens BEFORE computing lengths
        train_dl, val_dl, _test_dl = get_dataloaders(
            tokenizer_path=self._tokenizer_path,
            cmvn_stats_path=self._cmvn_stats_path,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            train_fraction=train_fraction,
        )

        # Wrap loaders to map keys and shift token ids
        self._train_dl = _MappedLoader(train_dl, token_offset=self._token_offset)
        self._valid_dl = _MappedLoader(val_dl, token_offset=self._token_offset)

    def train_dataloader(self) -> Any:
        return self._train_dl

    def valid_dataloader(self) -> Any:
        return self._valid_dl

    def _ensure_tokenizer_loaded(self):
        """Lazy initialization of tokenizer to avoid DDP/multiprocessing issues."""
        if self._sp is None:
            # Add a small process-specific delay to avoid simultaneous initialization
            import time
            import os
            rank = int(os.environ.get('RANK', 0))
            time.sleep(0.01 * rank)  # Stagger initialization by rank
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self._tokenizer_path)
            # Compute vocab_size and special_ids once tokenizer is loaded
            self._vocab_size = int(self._sp.get_piece_size()) + self._token_offset
            self._special_ids = {
                'pad_id': 0,
                'bos_id': 1,
                'eos_id': 2,
                'blank_id': 3,
            }

    def vocab_size(self) -> int:
        self._ensure_tokenizer_loaded()
        return self._vocab_size

    def special_ids(self) -> Dict[str, int]:
        self._ensure_tokenizer_loaded()
        return self._special_ids


class _MappedLoader:
    """Thin wrapper that maps batch keys from src.data.dataset to training API."""

    def __init__(self, base: DataLoader, token_offset: int) -> None:
        self._base = base
        self._token_offset = token_offset

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self._base:
            labels = batch['labels']
            if self._token_offset:
                # Preserve pads (0), shift only non-pad labels
                labels = torch.where(labels != 0, labels + self._token_offset, labels)
            yield {
                'feats': batch['fbank'],
                'feat_lens': batch['fbank_lengths'],
                'tokens': labels,
                'token_lens': batch['label_lengths'],
            }

    def __len__(self) -> int:
        return len(self._base)

if __name__ == "__main__":

    dm = LibriSpeechDataModule(data_dir="nothing", batch_size=16, num_workers=4)
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()

    print(dm.vocab_size(), dm.special_ids())  # e.g., 32, {'bos_id':1,'eos_id':2,'pad_id':0,'blank_id':3}
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(k, v.shape if hasattr(v, "shape") else v)