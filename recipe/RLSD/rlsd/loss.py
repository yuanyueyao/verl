"""
RLSD Self-Distillation Loss。

SD 分支：full-distribution clipped KL (D_clip^KL(p_T || p_S))

    p_T = p_ref(·|x, y*, ŷ_{<n})  ← frozen ref 在特权 context (含 GT) 下的分布
    p_S = p_θ(·|x, ŷ_{<n})        ← student 在无特权 context 下的分布

    D_clip^KL(p_T || p_S) = Σ_v min(p_T(v) · log(p_T(v)/p_S(v)), τ)

梯度只通过 p_S 传播。GRPO 分支直接复用 verl 原生 update_policy。

Uncertainty-aware variant:
  通过 token_mask 可选地 mask 掉 epistemic / high-entropy token 位置，
  不对这些位置施加 teacher 分布约束，保护认知不确定性表达。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_sd_loss_chunked(
    stu_full_logits: torch.Tensor,      # (B, seq_stu, V) 模型原始输出 logits
    ref_full_logits: torch.Tensor,      # (B, seq_ref, V) ref 模型原始输出 logits
    T_resp: int,                        # response 长度
    response_mask: torch.Tensor,        # (B, T_resp)
    temperature: float = 1.0,
    kl_clip: float = 10.0,
    chunk_size: int = 128,
    token_mask: torch.Tensor | None = None,   # (B, T_resp) bool, True=应训练
    return_entropy: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    内存友好的 full-distribution clipped KL loss。

    核心思路：不一次性切出 (B, T_resp, V) 的完整 logit 张量，
    而是每次只取 chunk_size 个 token 位置做 softmax + KL，
    峰值显存降低 T_resp/chunk_size 倍。

    对于 B=2, chunk_size=128, V=150K：
      每 chunk 峰值 ≈ 2×128×150K×4 bytes × 3 tensors ≈ 440MB（可接受）

    Args:
        token_mask: (B, T_resp) bool tensor. True = this position should
                    contribute to the loss (i.e. is NOT masked).
                    If None, all response_mask positions contribute.
        return_entropy: if True, also returns per-token student entropy
                        in the metrics dict under key "sd/entropy_tensor".
    """
    # ── Compute effective denominator ──────────────────────────
    resp_mask_f = response_mask.float()
    if token_mask is not None:
        effective_mask = resp_mask_f * token_mask.float()  # (B, T_resp)
    else:
        effective_mask = resp_mask_f
    denom = effective_mask.sum().clamp(min=1.0)

    total_kl = torch.tensor(0.0, device=stu_full_logits.device)

    # Optional: collect per-token entropy for mask generation (方案 B)
    all_entropies: list[torch.Tensor] = [] if return_entropy else None

    for t_start in range(0, T_resp, chunk_size):
        t_end = min(t_start + chunk_size, T_resp)

        # ── chunk masks ────────────────────────────────────────
        chunk_resp_mask = resp_mask_f[:, t_start:t_end]   # (B, chunk)
        if token_mask is not None:
            chunk_token_mask = effective_mask[:, t_start:t_end]  # already combined
        else:
            chunk_token_mask = chunk_resp_mask

        if chunk_token_mask.sum() == 0:
            continue

        # ── logit slicing ──────────────────────────────────────
        # logits[:, pos, :] 预测 token[pos+1]
        # response tokens 占 input_ids 的最后 T_resp 个位置
        # chunk [t_start, t_end) 对应:
        #   start_idx = seq_len - T_resp - 1 + t_start  (即 -(T_resp + 1 - t_start))
        #   end_idx   = seq_len - T_resp - 1 + t_end    (即 -(T_resp + 1 - t_end))
        seq_stu = stu_full_logits.shape[1]
        seq_ref = ref_full_logits.shape[1]
        s_start = seq_stu - T_resp - 1 + t_start
        s_end = seq_stu - T_resp - 1 + t_end
        r_start = seq_ref - T_resp - 1 + t_start
        r_end = seq_ref - T_resp - 1 + t_end

        stu_chunk = stu_full_logits[:, s_start:s_end, :]
        ref_chunk = ref_full_logits[:, r_start:r_end, :]

        if temperature != 1.0:
            stu_chunk = stu_chunk / temperature
            ref_chunk = ref_chunk / temperature

        # ── ref distribution (no_grad) ─────────────────────────
        with torch.no_grad():
            ref_lp = F.log_softmax(ref_chunk.float(), dim=-1)
            ref_p = ref_lp.exp()

        # ── student distribution (有梯度) ──────────────────────
        student_lp = F.log_softmax(stu_chunk.float(), dim=-1)
        student_p = student_lp.exp()

        # ── entropy (optional, for mask generation) ────────────
        if return_entropy:
            H_chunk = -(student_p * student_lp).sum(dim=-1)  # (B, chunk)
            all_entropies.append(H_chunk.detach())

        # ── per-vocab KL with clip ─────────────────────────────
        kl = ref_p * (ref_lp - student_lp)  # (B, chunk, V)
        if kl_clip > 0:
            kl = kl.clamp(max=kl_clip)      # per-vocab-item clip

        per_token_kl = kl.sum(dim=-1)       # (B, chunk)
        total_kl = total_kl + (per_token_kl * chunk_token_mask).sum()

        del ref_lp, ref_p, student_lp, student_p, kl, per_token_kl, stu_chunk, ref_chunk

    loss = total_kl / denom

    with torch.no_grad():
        metrics = {
            "sd/kl_loss": loss.item(),
            "sd/kl_per_token_mean": loss.item(),
            "sd/n_tokens": denom.item(),
        }
        if token_mask is not None:
            n_total = resp_mask_f.sum().item()
            metrics["sd/n_masked_tokens"] = n_total - denom.item()
        if return_entropy and all_entropies:
            ent_tensor = torch.cat(all_entropies, dim=1)  # (B, T_resp)
            metrics["sd/entropy_tensor"] = ent_tensor
            metrics["sd/entropy_mean"] = ent_tensor.mean().item()

    return loss, metrics
