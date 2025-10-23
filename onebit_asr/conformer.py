

# --------------------------------------------------------------
# File: onebit_asr/conformer.py
# --------------------------------------------------------------
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import QuantizedLinear


def swish(x):
    return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.ln(x)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.lin1 = QuantizedLinear(d_model, d_ff)
        self.lin2 = QuantizedLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, bitwidth: int, mask=None):
        y = self.ln(x)
        y = self.lin1(y, bitwidth)
        y = swish(y)
        y = self.dropout(y)
        y = self.lin2(y, bitwidth)
        y = self.dropout(y)
        # Zero out padded positions
        if mask is not None:
            seq_mask = mask[:, :, 0].unsqueeze(-1)  # [B, T, 1]
            y = y * seq_mask
        return x + 0.5 * y  # macaron scaling 0.5


class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln = LayerNorm(d_model)
        self.q_proj = QuantizedLinear(d_model, d_model)
        self.k_proj = QuantizedLinear(d_model, d_model)
        self.v_proj = QuantizedLinear(d_model, d_model)
        self.out_proj = QuantizedLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, bitwidth: int):
        B, T, C = x.shape
        y = self.ln(x)
        Q = self.q_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, :, :]==0, float('-inf'))
        A = torch.softmax(attn_scores, dim=-1)
        # Replace NaN with 0 (happens when all attention scores are -inf for padded positions)
        A = torch.nan_to_num(A, nan=0.0)
        A = self.dropout(A)
        H = A @ V  # [B, h, T, d]
        H = H.transpose(1, 2).contiguous().view(B, T, C)
        H = self.out_proj(H, bitwidth)
        H = self.dropout(H)
        # Zero out padded positions in H to prevent NaN propagation
        if mask is not None:
            # mask is [B, T, T], we need [B, T, 1] for the diagonal (self positions)
            seq_mask = mask[:, :, 0].unsqueeze(-1)  # [B, T, 1]
            H = H * seq_mask
        return x + H


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2*d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model, track_running_stats=False)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # x: [B, T, C]
        y = self.ln(x)
        y = y.transpose(1, 2)  # [B, C, T]
        y = self.pw1(y)
        y = self.glu(y)
        y = self.dw(y)
        y = self.bn(y)
        y = swish(y)
        y = self.pw2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        # Zero out padded positions
        if mask is not None:
            seq_mask = mask[:, :, 0].unsqueeze(-1)  # [B, T, 1]
            y = y * seq_mask
        return x + y


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel: int, dropout: float,
                 block_index: int):
        super().__init__()
        self.block_index = block_index
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MHSA(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
    def forward(self, x, src_mask, bitwidth_linear: int):
        x = self.ff1(x, bitwidth_linear, src_mask)
        x = self.mhsa(x, src_mask, bitwidth_linear)
        x = self.conv(x, src_mask)  # kept full-precision per paper recommendation
        x = self.ff2(x, bitwidth_linear, src_mask)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 10000, d_model) * 0.01)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, n_heads, conv_kernel, dropout, i)
            for i in range(n_layers)
        ])
        self.ln_out = LayerNorm(d_model)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor,
                precision: int, sp_mask: Optional[List[int]] = None):
        x = self.in_proj(feats)
        T = x.size(1)
        pos_enc_slice = self.pos_enc[:, :T, :]
        x = x + pos_enc_slice
        
        # Build masks...
        B = x.size(0)
        device = x.device
        src_key_mask = torch.arange(T, device=device)[None, :].expand(B, T)
        src_key_mask = (src_key_mask < feat_lens[:, None]).int()
        attn_mask = src_key_mask[:, None, :] * src_key_mask[:, :, None]

        for i, blk in enumerate(self.blocks):
            bitw = precision
            if sp_mask is not None:
                bitw = 1 if sp_mask[i] == 1 else 2
            x = blk(x, attn_mask, bitw if bitw in (1, 2) else 32)
        
        x = self.ln_out(x)
        return x, src_key_mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, dropout: float, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.ln = LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_inp, memory, memory_mask, tgt_key_padding_mask):
        # build square subsequent mask for causal decoding
        Tt = tgt_inp.size(1)
        causal = torch.triu(torch.ones(Tt, Tt, device=tgt_inp.device), diagonal=1).bool()
        causal = causal.float().masked_fill(causal, float('-inf'))
        y = self.emb(tgt_inp)
        y = self.dec(y, memory, tgt_mask=causal,
                     memory_key_padding_mask=(memory_mask==0),
                     tgt_key_padding_mask=tgt_key_padding_mask)
        y = self.ln(y)
        logits = self.out(y)
        return logits


class ConformerASR(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int,
                 enc_d_model=256, enc_layers=12, enc_heads=4, enc_d_ff=1024,
                 enc_conv_kernel=31, enc_dropout=0.1,
                 dec_layers=2, dec_heads=4, dec_d_ff=1024, dec_dropout=0.1,
                 pad_id=0):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, enc_d_model, enc_layers, enc_heads,
                                        enc_d_ff, enc_conv_kernel, enc_dropout)
        self.decoder = TransformerDecoder(vocab_size, enc_d_model, dec_layers, dec_heads,
                                          dec_d_ff, dec_dropout, pad_id)
        self.ctc_head = nn.Linear(enc_d_model, vocab_size)

    def forward(self, batch, precision: int, sp_mask=None):
        # batch must contain: feats [B,T,F], feat_lens [B], tokens [B,U], token_lens [B]
        enc_out, enc_mask = self.encoder(batch['feats'], batch['feat_lens'], precision, sp_mask)
        logits_ctc = self.ctc_head(enc_out)  # [B,T,V]
        return enc_out, enc_mask, logits_ctc

    def decode_logits(self, enc_out, enc_mask, tgt_inp, tgt_pad_mask):
        return self.decoder(tgt_inp, enc_out, enc_mask, tgt_pad_mask)
