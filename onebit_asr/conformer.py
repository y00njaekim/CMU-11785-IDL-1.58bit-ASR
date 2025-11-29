

# --------------------------------------------------------------
# File: onebit_asr/conformer.py
# --------------------------------------------------------------
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from onebit_asr.quant import QuantizedLinear


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


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise RuntimeError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}. Increase max_len in RelPositionalEncoding.")
        pos_emb = self.pe[:, :seq_len].clone()
        return self.dropout(x), pos_emb


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
        self.pos_proj = QuantizedLinear(d_model, d_model)
        self.out_proj = QuantizedLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.pos_bias_u = nn.Parameter(torch.randn(self.n_heads, self.d_head) * 0.01)
        self.pos_bias_v = nn.Parameter(torch.randn(self.n_heads, self.d_head) * 0.01)

    def rel_shift(self, x):
        B, H, T1, T2 = x.shape
        zero_pad = torch.zeros((B, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(B, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        return x

    def forward(self, x, mask, bitwidth: int, pos_emb: torch.Tensor):
        B, T, C = x.shape
        assert C == self.d_model, f"Expected {self.d_model}, got {C}"

        y = self.ln(x)
        Q = self.q_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(y, bitwidth).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        p = self.pos_proj(pos_emb, bitwidth).view(1, T, self.n_heads, self.d_head).transpose(1, 2)

        Q_with_u = Q + self.pos_bias_u.view(1, self.n_heads, 1, self.d_head)
        Q_with_v = Q + self.pos_bias_v.view(1, self.n_heads, 1, self.d_head)

        matrix_ac = torch.matmul(Q_with_u, K.transpose(-2, -1))
        matrix_bd = torch.matmul(Q_with_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        attn_scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_head)
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


class Conv2dSubsampling(nn.Module):
    """
    2D convolutional subsampling (to ~1/4 time length), ESPnet/WeNet style.

    Input:  x [B, T, F]  (e.g. log-Mel features)
    Output: y [B, T', d_model], where T' ~ T // 4
    """
    def __init__(self, idim: int, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Two 3x3 convs with stride 2 (time and freq both downsampled)
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),  # (B,1,T,F) -> (B,d_model,T1,F1)
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),  # (B,d_model,T2,F2)
            nn.ReLU(),
        )

        # Compute resulting frequency size after two convs (no padding, k=3,s=2)
        # Matches ESPnet formula: (((idim - 1) // 2 - 1) // 2)
        out_freq = (((idim - 1) // 2 - 1) // 2)
        if out_freq <= 0:
            raise ValueError(f"Input dim too small for Conv2dSubsampling: idim={idim}")

        self.out = nn.Linear(d_model * out_freq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape
        # Treat as image: [B, 1, T, F]
        x = x.unsqueeze(1)
        x = self.conv(x)                          # [B, C=d_model, T', F']
        B, C, T_sub, F_sub = x.size()

        # Move time to dim 1 and flatten (C * F')
        x = x.transpose(1, 2).contiguous().view(B, T_sub, C * F_sub)  # [B, T', C*F']
        x = self.out(x)                          # [B, T', d_model]
        return x



class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel: int, dropout: float,
                 block_index: int):
        super().__init__()
        self.block_index = block_index
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MHSA(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ln = LayerNorm(d_model)
    def forward(self, x, src_mask, bitwidth_linear: int, pos_emb: torch.Tensor):
        x = self.ff1(x, bitwidth_linear)
        x = self.mhsa(x, src_mask, bitwidth_linear, pos_emb)
        x = self.conv(x)  # kept full-precision per paper recommendation
        x = self.ff2(x, bitwidth_linear)
        x = self.ln(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, conv_kernel: int, dropout: float, use_checkpoint: bool = True):
        super().__init__()
        self.subsample = Conv2dSubsampling(input_dim, d_model)
        self.pos_enc = RelPositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, n_heads, conv_kernel, dropout, i)
            for i in range(n_layers)
        ])
        self.ln_out = LayerNorm(d_model)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor,
                precision: int, sp_mask: Optional[List[int]] = None):
        # feats: [B, T, F]
        B, T_in, F = feats.shape
        device = feats.device
        # 1) 2D conv subsampling: [B, T, F] -> [B, T_sub, d_model]
        x = self.subsample(feats)
        T_sub = x.size(1)

        # Approximate new lengths (two stride-2 convs -> ~ /4)
        enc_lens = feat_lens // 4

        # 2) Relative positional encoding on subsampled sequence
        x, pos_emb = self.pos_enc(x)  # pos_emb: [1, T_sub, d_model]

        # 3) Build src mask: [B, T_sub]
        t_idx = torch.arange(T_sub, device=device)[None, :].expand(B, T_sub)
        src_key_mask = (t_idx < enc_lens[:, None])  # bool [B, T_sub]

        # 4) Build attention mask [B, T_sub, T_sub] for encoder self-attn
        attn_mask = src_key_mask[:, :, None] & src_key_mask[:, None, :]

        for i, blk in enumerate(self.blocks):
            bitw = precision
            if sp_mask is not None:
                bitw = 1 if sp_mask[i] == 1 else 2
            x = blk(x, attn_mask, bitw if bitw in (1, 2) else 32, pos_emb)

        x = self.ln_out(x)
        return x, src_key_mask  # src_key_mask now matches subsampled time


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
                 pad_id=0, use_checkpoint=True):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, enc_d_model, enc_layers, enc_heads,
                                        enc_d_ff, enc_conv_kernel, enc_dropout, use_checkpoint=use_checkpoint)
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


def test_conformer():
    # Use the same architecture numbers as defaults in train.py
    input_dim = 256
    vocab_size = 100
    # ConformerASR defaults already match train.py; instantiate with defaults
    model = ConformerASR(input_dim, vocab_size)

    # Dummy input batch
    B = 4
    T = 1000
    F = input_dim
    feats = torch.randn(B, T, F)
    feat_lens = torch.tensor([500, 500, 500, 500])

    batch = {
        'feats': feats,
        'feat_lens': feat_lens
    }

    # Minimal torchinfo summary (no wrappers/fallbacks)
    from torchinfo import summary  # requires: pip install torchinfo
    print("\nModel summary (torchinfo):")
    summary(
        model,
        input_data=({'feats': feats, 'feat_lens': feat_lens}, 32),
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=4,
        device="cpu",
    )

    print("\nRunning forward pass...")
    try:
        enc_out, enc_mask, logits = model(batch, precision=32)
        print("Forward pass successful!")
        print("Enc out shape:", enc_out.shape)
        print("Logits shape:", logits.shape)
    except Exception as e:
        print("Forward pass failed!")
        print(e)
        raise e

if __name__ == "__main__":
    test_conformer()