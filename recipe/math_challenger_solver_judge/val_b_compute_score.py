# Copyright 2026 the verl recipe authors
"""Validation scoring for model B: GSM8K via RLSD ``is_correct``; MBPP unchanged.

Avoids edits under ``verl/``; GSM8K eval aligns with
``recipe.RLSD.rlsd.verifier`` (last \\boxed{} / \\fbox{} + math_verify).
"""

from __future__ import annotations

from recipe.my_project.mbpp_exec import compute_mbpp_score_dict

MBPP_DATA_SOURCE = "google-research-datasets/mbpp"
GSM8K_DATA_SOURCE = "openai/gsm8k"


def _normalize_non_mbpp_result(res):
    if isinstance(res, dict):
        out = dict(res)
        out.setdefault("mbpp_ok", float("nan"))
        out.setdefault("mbpp_err", "")
        return out
    return {"score": float(res), "mbpp_ok": float("nan"), "mbpp_err": ""}


def val_b_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    if data_source == MBPP_DATA_SOURCE:
        return compute_mbpp_score_dict(solution_str, ground_truth, extra_info)

    if data_source == GSM8K_DATA_SOURCE:
        try:
            from recipe.RLSD.rlsd.verifier import is_correct

            score = 1.0 if is_correct(solution_str, str(ground_truth)) else 0.0
            return _normalize_non_mbpp_result(score)
        except Exception:
            pass

    from verl.utils.reward_score import default_compute_score

    res = default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
    )
    return _normalize_non_mbpp_result(res)
