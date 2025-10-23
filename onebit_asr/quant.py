
# LibriSpeech Data Loader Interface:
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
#
# Usage:
#   python -m train \
#       --data_dir /path/to/LibriSpeech \
#       --save_dir ./checkpoints \
#       --epochs 40 --batch_size 16 --lr 2e-3
#
# --------------------------------------------------------------
# File: onebit_asr/quant.py
# --------------------------------------------------------------
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _QuantizeSTE(torch.autograd.Function):
    """
    Quantize weights to 1-bit (binary {-1, +1}) or 2-bit (ternary {-1, 0, +1})
    with a learnable tensor-wise scale alpha. Implements the STE; and the custom
    d/dalpha from Eq. (3) of the paper.
    """
    @staticmethod
    def forward(ctx, W: torch.Tensor, alpha: torch.Tensor, bitwidth: int):
        # Alpha is scalar (tensor-wise). W_hat = alpha * Pi_Qn( clip(W/alpha, minQ, maxQ) )
        # For simplicity, clip range is [-1, 1] for both binary (Q1) and ternary (Q2).
        
        Wa = W / alpha
        Wa_clipped = Wa.clamp(-1.0, 1.0)
        
        if bitwidth == 1:  # binary {-1, +1}
            Q = Wa_clipped.sign()
            # map zeros to +1 (by convention). Avoid zeros by epsilon.
            Q[Q == 0] = 1.0
        elif bitwidth == 2:  # ternary {-1, 0, +1}
            # nearest to {-1, 0, +1} -> threshold 0.5 to decide going to 0 vs +/-1
            Q = Wa_clipped.clone()
            absWa = Wa_clipped.abs()
            Q = torch.where(absWa < 0.5, torch.zeros_like(Q), Q.sign())
        elif bitwidth == 32:
            # full precision passthrough
            ctx.save_for_backward(Wa, alpha.detach(), torch.tensor(bitwidth, device=W.device))
            return W
        else:
            raise ValueError("bitwidth must be one of {1,2,32}")

        W_hat = alpha * Q
        ctx.save_for_backward(Wa, alpha.detach(), torch.tensor(bitwidth, device=W.device))
        return W_hat

    @staticmethod
    def backward(ctx, grad_out): # pyright: ignore[reportIncompatibleMethodOverride]
        Wa, alpha, bitwidth_tensor = ctx.saved_tensors
        bitwidth = int(bitwidth_tensor.item())
        if bitwidth == 32:
            # standard FP gradient
            return grad_out, grad_out.new_zeros(()), None

        # STE for dW: pass-through inside clip range, zero outside
        indicator = (Wa.abs() <= 1.0).to(grad_out.dtype)
        grad_W = grad_out * indicator

        # d/dalpha per Eq. (3):
        # ∂W_hat/∂α = -W/α + Pi_Qn(W/α), if |W/α| < max(|Qn|)=1; else sign(W/α)
        term = torch.where(
            Wa.abs() < 1.0,
            -Wa + Wa.sign().where(Wa.abs() >= 0.5, torch.zeros_like(Wa)) if bitwidth == 2 else -Wa + Wa.sign(),
            Wa.sign(),
        )
        grad_alpha = (grad_out * term).sum()
        return grad_W, grad_alpha, None


def quantize_weight(W: torch.Tensor, alpha: torch.Tensor, bitwidth: int) -> torch.Tensor:
    return _QuantizeSTE.apply(W, alpha, bitwidth) # pyright: ignore[reportReturnType]


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Scale weights up by 2x for quantization-friendly range
        # This puts most weights in range where |W/alpha| > 0.5 with alpha~0.1
        with torch.no_grad():
            self.weight.data *= 2.0
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, bitwidth: int) -> torch.Tensor:
        if bitwidth == 32:
            W_used = self.weight
        else:
            W_used = quantize_weight(self.weight, self.alpha.abs() + 1e-8, bitwidth)
        
        output = F.linear(x, W_used, self.bias)
        return output
