"""
RLSD 自定义 Actor。

继承 verl DataParallelPPOActor，覆写 update_policy 实现两分支：
  - SD 分支：full-distribution clipped KL(p_T || p_S)
  - GRPO 分支：标准 clipped policy gradient

通过 DataProto.meta_info["rlsd_mode"] 标记使用哪个分支：
  - "sd": Self-Distillation（worker 会设置 self._ref_module）
  - "grpo": 标准 GRPO（需要 advantages + old_log_probs）
"""

from __future__ import annotations

import torch
from typing import Dict

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor


class RLSDPPOActor(DataParallelPPOActor):
    """
    RLSD Actor：按 meta_info["rlsd_mode"] 分流到 SD 或 GRPO loss。
    _ref_module: 由 worker 在 update_actor 时注入的 frozen ref model。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref_module = None

    def update_policy(self, data: DataProto) -> Dict:
        """根据 rlsd_mode 分流。"""
        mode = data.meta_info.get("rlsd_mode", "sd")
        if mode == "grpo":
            return self._update_grpo(data)
        else:
            return self._update_sd(data)

    # ──────────────────────────────────────────────────────────────
    # SD 分支：per-micro-batch 计算 ref_logits + student_logits → clipped KL
    # ──────────────────────────────────────────────────────────────

    def _update_sd(self, data: DataProto) -> Dict:
        self.actor_module.train()

        from recipe.RLSD.rlsd.loss import compute_sd_loss_chunked

        temperature = data.meta_info.get("temperature", 1.0)
        kl_clip: float = float(data.meta_info.get("kl_clip", 10.0))
        mask_mode: str = data.meta_info.get("sd_mask_mode", "none")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                       "ref_input_ids", "ref_attention_mask", "ref_position_ids", "response_mask"]
        batch = data.select(batch_keys=select_keys).batch

        micro_bsz = self.config.ppo_micro_batch_size_per_gpu
        if self.config.use_dynamic_bsz:
            from verl.utils.seqlen_balancing import rearrange_micro_batches
            max_token_len = self.config.ppo_max_token_len_per_gpu * getattr(self, "ulysses_sequence_parallel_size", 1)
            micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_bsz)

        n_micro_batches = len(micro_batches)
        metrics: Dict = {}
        self.actor_optimizer.zero_grad()

        ref_module = self._ref_module
        assert ref_module is not None, "SD branch requires _ref_module (set by worker)"
        ref_module.eval()

        # ── Token-identity mask: read pre-built IDs from meta_info ────
        if mask_mode == "token_identity":
            from recipe.RLSD.rlsd.epistemic_mask import build_token_identity_mask
            ep_ids_list = data.meta_info.get("epistemic_token_ids", [])
            epistemic_ids = set(ep_ids_list)
        else:
            epistemic_ids = None

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                mb = micro_batch.batch.to(self.device_name)
            elif hasattr(micro_batch, "to"):
                mb = micro_batch.to(self.device_name)
            else:
                mb = {k: (v.to(self.device_name) if isinstance(v, torch.Tensor) else v)
                      for k, v in micro_batch.items()}

            responses = mb["responses"]
            T_resp = responses.shape[1]
            response_mask = mb["response_mask"].float()[:, :T_resp]

            # ── Build per-micro-batch token mask ────────────────────
            token_mask = None
            if mask_mode == "token_identity" and epistemic_ids is not None:
                token_mask = build_token_identity_mask(responses, epistemic_ids)
            elif mask_mode == "entropy_percentile":
                # Two-pass: first get entropy, then build mask
                pass  # implemented below

            # ── Teacher forward（no_grad）— 使用特权 context (含 GT) ──
            with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                ref_output = ref_module(
                    input_ids=mb["ref_input_ids"],
                    attention_mask=mb["ref_attention_mask"],
                    position_ids=mb["ref_position_ids"],
                    use_cache=False,
                )
            ref_full_logits = ref_output.logits  # (B, seq_ref, V)

            # ── Student forward（有梯度）— 无特权 context ────────────
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                output = self.actor_module(
                    input_ids=mb["input_ids"],
                    attention_mask=mb["attention_mask"],
                    position_ids=mb["position_ids"],
                    use_cache=False,
                )
            stu_full_logits = output.logits  # (B, seq_stu, V)

            # ── Entropy-percentile mask: two-pass strategy ──────────
            # First pass: compute entropy (no backprop) to build mask
            if mask_mode == "entropy_percentile":
                from recipe.RLSD.rlsd.epistemic_mask import build_entropy_percentile_mask
                # Pass 1: get entropy only (loss is discarded; no backward)
                _, pass1_metrics = compute_sd_loss_chunked(
                    stu_full_logits=stu_full_logits,
                    ref_full_logits=ref_full_logits,
                    T_resp=T_resp,
                    response_mask=response_mask,
                    temperature=temperature,
                    kl_clip=kl_clip,
                    chunk_size=128,
                    token_mask=None,
                    return_entropy=True,
                )
                entropies = pass1_metrics.get("sd/entropy_tensor")
                if entropies is not None:
                    pct = float(data.meta_info.get("sd_mask_entropy_percentile", 0.8))
                    token_mask = build_entropy_percentile_mask(entropies, percentile=pct)
                # Clear metrics from pass 1 (don't mix with pass 2)
                # Note: pass1's logits are NOT freed — pass 2 reuses them

            # ── Chunked clipped KL loss ─────────────────────────────
            loss, step_metrics = compute_sd_loss_chunked(
                stu_full_logits=stu_full_logits,
                ref_full_logits=ref_full_logits,
                T_resp=T_resp,
                response_mask=response_mask,
                temperature=temperature,
                kl_clip=kl_clip,
                chunk_size=128,
                token_mask=token_mask,
            )

            # Free ref tensors before backward — they come from no_grad
            # and aren't needed for gradient computation
            del ref_full_logits, ref_output

            loss = loss / n_micro_batches
            loss.backward()
            del stu_full_logits, output
            append_to_dict(metrics, step_metrics)

        grad_norm = self._optimizer_step()
        append_to_dict(metrics, {"sd/grad_norm": grad_norm.detach().item()})
        return metrics

    # ──────────────────────────────────────────────────────────────
    # GRPO 分支：直接复用 verl 原生 update_policy
    # entropy 由 trainer 在 update 前调 compute_log_prob 时一并拿到
    # ──────────────────────────────────────────────────────────────

    def _update_grpo(self, data: DataProto) -> Dict:
        return super().update_policy(data)
