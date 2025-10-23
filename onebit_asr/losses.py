
# --------------------------------------------------------------
# File: onebit_asr/losses.py
# --------------------------------------------------------------
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_att_targets(tokens: torch.Tensor, bos_id: int, eos_id: int, pad_id: int):
    # tokens: [B, U] (no BOS/EOS inside). Create decoder inputs/targets.
    B, U = tokens.size()
    BOS = torch.full((B, 1), bos_id, dtype=tokens.dtype, device=tokens.device)
    EOS = torch.full((B, 1), eos_id, dtype=tokens.dtype, device=tokens.device)
    tgt_inp = torch.cat([BOS, tokens], dim=1)  # [B, U+1]
    tgt_out = torch.cat([tokens, EOS], dim=1)  # [B, U+1]
    tgt_pad_mask = (tgt_inp == pad_id)
    return tgt_inp, tgt_out, tgt_pad_mask


def att_ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, label_smoothing: float = 0.0):
    # logits: [B, Tt, V], targets: [B, Tt]
    if label_smoothing > 0:
        # CrossEntropy with label smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        n_class = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(label_smoothing / (n_class - 1))
            true_dist.scatter_(2, targets.unsqueeze(-1), 1.0 - label_smoothing)
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        # mask pads
        mask = (targets != pad_id).float()
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=pad_id)
        return loss


def ctc_loss_from_logits(ctc_logits: torch.Tensor, feat_lens: torch.Tensor,
                         tokens: torch.Tensor, token_lens: torch.Tensor, blank_id: int):
    # ctc_logits: [B,T,V] -> [T,B,V]
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    loss = loss_fn(log_probs, tokens, feat_lens, token_lens)
    return loss


def kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, pad_mask: torch.Tensor):
    # KL( SG(p_t) || p_s ). student_logits: [B, T, V] ; pad_mask: True where pad
    with torch.no_grad():
        p_t = F.softmax(teacher_logits, dim=-1)
    log_p_s = F.log_softmax(student_logits, dim=-1)
    # mask pads by zeroing loss at pad positions
    kl = F.kl_div(log_p_s, p_t, reduction='none')  # [B,T,V]
    kl = kl.sum(dim=-1)  # [B,T]
    mask = (~pad_mask).float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)

