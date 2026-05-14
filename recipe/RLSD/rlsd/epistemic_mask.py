"""
Epistemic token mask generation for uncertainty-aware self-distillation.

Based on Kim et al. (2026) — epistemic token set T:
  {wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}

Two masking strategies:
  - Token-identity mask (方案 A): mask out positions where the generated token
    belongs to the epistemic token set.
  - Entropy-percentile mask (方案 B): mask out the top-K% highest-entropy
    positions in each response.
"""

from __future__ import annotations

import torch
from typing import Set


# ── Epistemic token set (from the paper, Section 3) ──────────────
EPISTEMIC_TOKENS: list[str] = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]

# Surface-form variants for each base token.
# Each variant is encoded with tokenizer.encode(…, add_special_tokens=False).
# If encoding yields >1 token, we only collect the *last* token ID
# (the preceding tokens are typically whitespace/punctuation artifacts).
_VARIANTS: dict[str, list[str]] = {
    "wait":          ["Wait", "Wait,", "Wait.", "Wait ", " wait", " wait,", " wait.", " wait ",
                      "wait", "wait,", "wait.", "wait "],
    "hmm":           ["Hmm", "Hmm,", "Hmm.", "Hmm ", " hmm", " hmm,", " hmm.", " hmm ",
                      "hmm", "hmm,", "hmm.", "hmm "],
    "perhaps":       ["Perhaps", "Perhaps,", "Perhaps.", "Perhaps ",
                      " perhaps", " perhaps,", " perhaps.", " perhaps ",
                      "perhaps", "perhaps,", "perhaps.", "perhaps "],
    "maybe":         ["Maybe", "Maybe,", "Maybe.", "Maybe ",
                      " maybe", " maybe,", " maybe.", " maybe ",
                      "maybe", "maybe,", "maybe.", "maybe "],
    "actually":      ["Actually", "Actually,", "Actually.", "Actually ",
                      " actually", " actually,", " actually.", " actually ",
                      "actually", "actually,", "actually.", "actually "],
    "alternatively": ["Alternatively", "Alternatively,", "Alternatively.", "Alternatively ",
                      " alternatively", " alternatively,", " alternatively.", " alternatively ",
                      "alternatively", "alternatively,", "alternatively.", "alternatively "],
    "seems":         ["Seems", "Seems,", "Seems.", "Seems ",
                      " seems", " seems,", " seems.", " seems ",
                      "seems", "seems,", "seems.", "seems "],
    "might":         ["Might", "Might,", "Might.", "Might ",
                      " might", " might,", " might.", " might ",
                      "might", "might,", "might.", "might "],
    "likely":        ["Likely", "Likely,", "Likely.", "Likely ",
                      " likely", " likely,", " likely.", " likely ",
                      "likely", "likely,", "likely.", "likely "],
    "check":         ["Check", "Check,", "Check.", "Check ",
                      " check", " check,", " check.", " check ",
                      "check", "check,", "check.", "check "],
}


def build_epistemic_token_ids(tokenizer) -> Set[int]:
    """
    Scan the tokenizer vocabulary for all surface-form variants of the
    10 epistemic tokens. Returns a set of token IDs.

    The result is cached in the tokenizer object (``_epistemic_token_ids``)
    to avoid repeated computation.
    """
    cache_attr = "_epistemic_token_ids"
    if hasattr(tokenizer, cache_attr):
        return getattr(tokenizer, cache_attr)

    token_ids: Set[int] = set()
    for base, variants in _VARIANTS.items():
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            # If encoding yields multiple tokens, only take the last one.
            # The preceding tokens (e.g. a leading-space token or BOS) are
            # not the epistemic token itself.
            token_ids.add(ids[-1])

    # Safety: remove any special tokens that might have leaked in
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    token_ids -= special_ids

    setattr(tokenizer, cache_attr, token_ids)
    return token_ids


def build_token_identity_mask(
    response_ids: torch.Tensor,         # (B, T_resp)  — generated token IDs
    epistemic_ids: Set[int],
) -> torch.Tensor:
    """
    Build a boolean mask for token-identity masking (方案 A).

    Returns (B, T_resp) tensor where:
        True  = this position is NOT an epistemic token → SHOULD be trained
        False = this position IS an epistemic token     → SHOULD be masked

    Args:
        response_ids: (B, T_resp) token IDs of the student's generated response.
        epistemic_ids: set of token IDs identified as epistemic markers.
    """
    mask = torch.ones_like(response_ids, dtype=torch.bool)
    for tid in epistemic_ids:
        mask = mask & (response_ids != tid)
    return mask  # True = keep (trainable), False = masked


def build_entropy_percentile_mask(
    entropies: torch.Tensor,            # (B, T_resp)  — per-position entropy
    percentile: float = 0.8,
) -> torch.Tensor:
    """
    Build a boolean mask for entropy-percentile masking (方案 B).

    For each sample in the batch independently, positions whose entropy
    exceeds the `percentile`-th quantile are masked out.

    Returns (B, T_resp) tensor where:
        True  = entropy ≤ percentile → SHOULD be trained
        False = entropy > percentile → SHOULD be masked

    Args:
        entropies: (B, T_resp) per-token entropy values.
        percentile: float in (0, 1). Top (1-percentile) fraction is masked.
                    Default 0.8 means top 20% by entropy are protected.
    """
    B, T = entropies.shape
    device = entropies.device

    # Per-sample quantile threshold
    # quantile over last dim (T), keepdim for broadcasting
    thresholds = torch.quantile(
        entropies.float(), percentile, dim=1, keepdim=True,
    )  # (B, 1)

    mask = entropies <= thresholds  # (B, T) — True where entropy is NOT in top (1-p)%
    return mask
