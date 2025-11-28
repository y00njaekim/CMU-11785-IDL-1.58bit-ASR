import math
from typing import List, Tuple, Dict, Optional

import torch


def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_wer(refs: List[str], hyps: List[str]) -> Tuple[int, int]:
    total_dist = 0
    total_words = 0
    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()
        total_dist += levenshtein_distance(r_words, h_words)
        total_words += len(r_words)
    return total_dist, total_words


def ids_to_text(ids, spm_processor, token_offset: int = 4, pad_id: int = 0):
    """
    Convert a sequence of model token IDs (0=pad,1=bos,2=eos,3=blank, >=4 = SPM+offset)
    to a string using the SentencePiece processor.
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    # Keep only vocab tokens (>= token_offset), drop pad/bos/eos/blank
    piece_ids = [int(i - token_offset) for i in ids if i >= token_offset]

    if len(piece_ids) == 0:
        return ""

    return spm_processor.decode(piece_ids)


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 3) -> List[int]:
    # logits: [T, V]
    pred = torch.argmax(logits, dim=-1).tolist()
    out: List[int] = []
    prev = None
    for t in pred:
        if t != blank_id and t != prev:
            out.append(t)
        prev = t
    return out


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def ctc_beam_search(logits: torch.Tensor, beam_size: int = 10, blank_id: int = 3, top_k_per_t: Optional[int] = 20) -> List[int]:
    """
    Prefix beam search for CTC without LM.
    logits: [T, V] (unnormalized). Returns best path token ids (without blanks/repeats).
    """
    T, V = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1)

    # state: prefix(tuple)->(p_blank, p_non_blank) in log space
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}

    for t in range(T):
        lp_t = log_probs[t]
        if top_k_per_t is not None and top_k_per_t < V:
            topk = torch.topk(lp_t, k=top_k_per_t)
            idxs = topk.indices.tolist()
            vals = topk.values.tolist()
            candidates = list(zip(idxs, vals))
        else:
            candidates = [(i, float(lp_t[i])) for i in range(V)]

        new_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (p_b, p_nb) in beams.items():
            # Extend by blank
            p_b_new = _logsumexp(new_beams.get(prefix, (-math.inf, -math.inf))[0], _logsumexp(p_b + lp_t[blank_id].item(), p_nb + lp_t[blank_id].item()))
            p_nb_new = new_beams.get(prefix, (-math.inf, -math.inf))[1]
            new_beams[prefix] = (p_b_new, p_nb_new)

            last = prefix[-1] if len(prefix) > 0 else None
            for c, lp in candidates:
                if c == blank_id:
                    continue
                c = int(c)
                lp_c = float(lp)
                new_prefix = prefix + (c,)

                # If same as last char, only non-blank transitions from blank
                if c == last:
                    # stay on same prefix (don't add repeated char)
                    p_nb_same = _logsumexp(new_beams.get(prefix, (-math.inf, -math.inf))[1], p_b + lp_c)
                    p_b_same = new_beams.get(prefix, (-math.inf, -math.inf))[0]
                    new_beams[prefix] = (p_b_same, p_nb_same)
                else:
                    # extend prefix
                    prev_pb, prev_pnb = new_beams.get(new_prefix, (-math.inf, -math.inf))
                    p_nb_ext = _logsumexp(prev_pnb, _logsumexp(p_b + lp_c, p_nb + lp_c))
                    new_beams[new_prefix] = (prev_pb, p_nb_ext)

        # prune to beam_size
        def total_score(state: Tuple[float, float]) -> float:
            return _logsumexp(state[0], state[1])

        beams = dict(sorted(new_beams.items(), key=lambda kv: total_score(kv[1]), reverse=True)[:beam_size])

    # pick best final prefix
    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]

    # Collapse repeats already handled; return as list
    return list(best_prefix)


def ctc_beam_search_batch(logits: torch.Tensor, valid_lens: torch.Tensor, beam_size: int = 10, blank_id: int = 3) -> List[List[int]]:
    """
    Batch beam search. logits: [B, T, V], valid_lens: [B] number of valid timesteps per sample.
    Returns list of token id sequences per sample.
    """
    B, T, V = logits.shape
    outs: List[List[int]] = []
    for b in range(B):
        t_len = int(valid_lens[b].item())
        outs.append(ctc_beam_search(logits[b, :t_len, :], beam_size=beam_size, blank_id=blank_id))
    return outs
