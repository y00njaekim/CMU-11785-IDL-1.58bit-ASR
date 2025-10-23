
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


from typing import Dict
import torch
from torch.utils.data import DataLoader, Dataset

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


class LibriSpeechDataModule:
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


if __name__ == "__main__":

    dm = LibriSpeechDataModule(data_dir="data/hf_cache", batch_size=16, num_workers=4)
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()

    print(dm.vocab_size(), dm.special_ids())  # e.g., 32, {'bos_id':1,'eos_id':2,'pad_id':0,'blank_id':3}
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(k, v.shape if hasattr(v, "shape") else v)