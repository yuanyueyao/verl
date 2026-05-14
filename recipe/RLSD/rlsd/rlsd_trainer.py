"""
RLSD Ray Trainer。

继承 RayPPOTrainer，覆写 fit() 实现 RLSD 混合训练：
  - SD 分支：本步对该题 k 条 student rollout 全部判错 → full-distribution clipped KL(p_ref || p_student)
  - GRPO 分支：有对有错 → 标准 clipped policy gradient

init_workers / _save_checkpoint / _load_checkpoint 全部复用官方实现。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.utils.tracking import Tracking

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from recipe.RLSD.rlsd.dataset import RLSDDataset, RLSDProblem
from recipe.RLSD.rlsd.prompt import (
    build_student_messages,
    build_teacher_privileged_messages,
    question_from_verl_prompt,
)
from recipe.RLSD.rlsd.verifier import is_correct


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _build_gen_batch(tokenizer, messages_list, max_prompt_len):
    """将 messages_list 编码为 DataProto 用于 rollout。"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc = tokenizer(texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return DataProto.from_single_dict({"input_ids": ids, "attention_mask": mask, "position_ids": pos})


def _build_sd_train_batch(tokenizer, student_msgs, teacher_msgs, responses_text, max_prompt_len, max_resp_len):
    """
    构建 SD 分支训练 batch。

    Student 和 Teacher 拥有不同的 prompt（Teacher 含 GT 特权信息），
    但 response tokens 相同。

    返回的 DataProto 包含：
      - input_ids / attention_mask / position_ids: student 完整序列
      - ref_input_ids / ref_attention_mask / ref_position_ids: teacher 完整序列
      - responses: 共享的 response token ids
      - response_mask: response 区域掩码
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode student prompts
    s_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in student_msgs]
    enc_sp = tokenizer(s_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # Encode teacher prompts (with GT privilege)
    t_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in teacher_msgs]
    enc_tp = tokenizer(t_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # Encode shared response tokens
    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses_text, return_tensors="pt", max_length=max_resp_len,
                      truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    responses_tensor = enc_r["input_ids"]
    response_mask = enc_r["attention_mask"]

    # Student full sequence
    s_full_ids = torch.cat([enc_sp["input_ids"], enc_r["input_ids"]], dim=1)
    s_full_mask = torch.cat([enc_sp["attention_mask"], enc_r["attention_mask"]], dim=1)
    s_pos = (s_full_mask.cumsum(-1) - 1).clamp(min=0)

    # Teacher full sequence
    t_full_ids = torch.cat([enc_tp["input_ids"], enc_r["input_ids"]], dim=1)
    t_full_mask = torch.cat([enc_tp["attention_mask"], enc_r["attention_mask"]], dim=1)
    t_pos = (t_full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": s_full_ids,
        "attention_mask": s_full_mask,
        "position_ids": s_pos,
        "ref_input_ids": t_full_ids,
        "ref_attention_mask": t_full_mask,
        "ref_position_ids": t_pos,
        "responses": responses_tensor,
        "response_mask": response_mask,
    })


def _build_logprob_batch(tokenizer, messages_list, responses_text, max_prompt_len, max_resp_len):
    """构建仅用于 compute_log_prob 的 batch（单 prompt + response 文本）。"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    tokenizer.padding_side = "right"
    enc_r = tokenizer(responses_text, return_tensors="pt", max_length=max_resp_len,
                      truncation=True, padding=True, add_special_tokens=False)
    tokenizer.padding_side = "left"

    full_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": enc_r["input_ids"],
    })


def _build_logprob_batch_from_tokens(tokenizer, messages_list, responses_tokens,
                                      response_mask, max_prompt_len):
    """
    构建用于 compute_log_prob / compute_ref_log_prob 的 batch。
    使用原始生成 token ids（不重新编码文本），确保与 old_log_probs 严格对齐。
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    full_ids = torch.cat([enc_p["input_ids"], responses_tokens], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], response_mask], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    return DataProto.from_single_dict({
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": responses_tokens,
    })


def _build_grpo_train_batch(tokenizer, messages_list, responses_tokens, response_mask,
                            rewards, old_log_probs, group_ids, max_prompt_len,
                            ref_log_probs=None):
    """
    构建 GRPO 分支训练 batch。

    使用原始生成的 token IDs（不重新编码文本），确保 old_log_probs 与
    responses token 位置严格对齐，与官方 verl GRPO 一致。

    Args:
        responses_tokens: (B, T_r) 原始生成 token ids（来自 rollout 输出）
        response_mask:    (B, T_r) 对应 attention mask（非 padding 位置为 1）
        rewards:          list[float], 每条 response 的 reward (0/1)
        old_log_probs:    (B, T_r) rollout 时的 per-token log probs
        group_ids:        list[int], 每条 response 归属的 problem index（用于组内归一化）
        ref_log_probs:    (B, T_r) or None，参考模型的 per-token log probs；
                          use_kl_loss=True 时必须提供
    """
    from collections import defaultdict

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    enc_p = tokenizer(prompt_texts, return_tensors="pt", max_length=max_prompt_len, truncation=True, padding=True)

    # 直接使用原始 token ids，不再重新编码文本（避免 round-trip tokenization 错位）
    T_r = responses_tokens.shape[1]
    full_ids = torch.cat([enc_p["input_ids"], responses_tokens], dim=1)
    full_mask = torch.cat([enc_p["attention_mask"], response_mask], dim=1)
    pos = (full_mask.cumsum(-1) - 1).clamp(min=0)

    # 按 problem (group) 内部归一化 advantage — 与官方 GRPO 完全一致：
    #   单样本 group: mean=0, std=1 → advantage = raw_reward（保留梯度信号）
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    g2scores = defaultdict(list)
    for i, gid in enumerate(group_ids):
        g2scores[gid].append(rewards_t[i])
    g2mean, g2std = {}, {}
    for gid, scores in g2scores.items():
        st = torch.stack(scores)
        if len(scores) == 1:
            g2mean[gid] = torch.tensor(0.0)   # 与官方一致：单样本不减均值
            g2std[gid] = torch.tensor(1.0)
        else:
            g2mean[gid] = st.mean()
            g2std[gid] = st.std()

    normed = torch.zeros_like(rewards_t)
    for i, gid in enumerate(group_ids):
        normed[i] = (rewards_t[i] - g2mean[gid]) / (g2std[gid] + 1e-6)

    advantages = normed.unsqueeze(1).expand(-1, T_r) * response_mask.float()

    batch_dict = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": pos,
        "responses": responses_tokens,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if ref_log_probs is not None:
        batch_dict["ref_log_prob"] = ref_log_probs

    return DataProto.from_single_dict(batch_dict)


# ══════════════════════════════════════════════════════════════════════
# RLSDTrainer
# ══════════════════════════════════════════════════════════════════════

class RLSDTrainer(RayPPOTrainer):
    """
    RLSD Trainer：继承 RayPPOTrainer，覆写 fit()。

    **模式**（``rlsd.grpo_only`` 与 ``rlsd.opsd_only`` 不能同时为 true；默认二者 false = SD+GRPO 混合）
      - ``grpo_only``：mixed 走 GRPO；全错分支不跑 SD。
      - ``opsd_only``：每题固定 1 次 rollout；**不判对错**、不更新 RLSDDataset 统计；每题必经 SD（特权 teacher）。题池始终为全集，无毕业语义。

    **单步**：采样 problems → student rollout → 按上表分流 → ``update_actor``。
    """

    def __init__(self, *args, rlsd_dataset: Optional[RLSDDataset] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlsd_dataset = rlsd_dataset

        rlsd_cfg = OmegaConf.select(self.config, "rlsd", default=OmegaConf.create({}))
        self.student_k = int(OmegaConf.select(rlsd_cfg, "student_rollout_per_problem", default=8))
        self.kl_clip = float(OmegaConf.select(rlsd_cfg, "kl_clip", default=10.0))
        self.problems_per_step = int(OmegaConf.select(rlsd_cfg, "problems_per_step", default=32))
        self.max_prompt_len = int(OmegaConf.select(self.config, "data.max_prompt_length", default=2048))
        self.max_resp_len = int(OmegaConf.select(self.config, "data.max_response_length", default=3072))
        # GRPO-Only：跳过 SD；OPSD-Only：仅无条件 SD，且关闭 GRPO / 判题统计
        self.grpo_only = bool(OmegaConf.select(rlsd_cfg, "grpo_only", default=False))
        self.opsd_only = bool(OmegaConf.select(rlsd_cfg, "opsd_only", default=False))
        if self.grpo_only and self.opsd_only:
            raise ValueError("rlsd.grpo_only 与 rlsd.opsd_only 不能同时为 true")
        # ── Uncertainty-aware SD mask config ────────────────────────
        self.sd_mask_mode = str(OmegaConf.select(rlsd_cfg, "sd_mask_mode", default="none"))
        self.sd_mask_entropy_percentile = float(
            OmegaConf.select(rlsd_cfg, "sd_mask_entropy_percentile", default=0.8)
        )
        self._epistemic_token_ids = None  # lazy-init after tokenizer is set
        if self.opsd_only:
            if self.student_k != 1:
                print(
                    f"[RLSDTrainer] opsd_only=true：忽略 rlsd.student_rollout_per_problem={self.student_k}，强制为 1"
                )
            self.student_k = 1

    def init_workers(self):
        """
        覆写 init_workers：使用 role="actor_rollout_ref" 使得 ref model 与 actor 共享 GPU。
        这样 update_actor 时可以在本地计算 ref_logits，无需跨进程传输。
        """
        from verl.trainer.ppo.ray_trainer import RayClassWithInitArgs, create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout_ref",  # 关键：加载 ref model
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    # ──────────────────────────────────────────────────────────────────
    # 推理工具
    # ──────────────────────────────────────────────────────────────────

    def _generate(self, messages_list, n_samples):
        """对一批 messages 生成 n_samples 条回复，返回 list[list[str]]。"""
        repeated = [m for m in messages_list for _ in range(n_samples)]
        gen_batch = _build_gen_batch(self.tokenizer, repeated, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        all_texts = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        n = len(messages_list)
        return [all_texts[i * n_samples:(i + 1) * n_samples] for i in range(n)]

    def _generate_with_logprobs(self, messages_list, n_samples):
        """
        生成回复，返回 (grouped_texts, old_log_probs, all_responses_tokens, all_response_masks)。

        - grouped_texts:        list[list[str]]，按 problem 分组的解码文本
        - old_log_probs:        (total_samples, T) per-token log probs，或 None
        - all_responses_tokens: (total_samples, T) 原始 response token ids
        - all_response_masks:   (total_samples, T) 对应 attention mask（非 padding 为 1）
        """
        repeated = [m for m in messages_list for _ in range(n_samples)]
        gen_batch = _build_gen_batch(self.tokenizer, repeated, self.max_prompt_len)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)

        all_texts = self.tokenizer.batch_decode(
            out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # vLLM rollout with calculate_log_probs=True stores "rollout_log_probs"
        old_log_probs = None
        for key in ("rollout_log_probs", "old_log_probs", "log_probs"):
            if key in out.batch:
                old_log_probs = out.batch[key]
                break

        # 保留原始 token ids 和 response mask（用于 GRPO batch 构建，避免重新编码）
        all_responses_tokens = out.batch["responses"]          # (B, T)
        resp_len = all_responses_tokens.shape[1]
        full_attn = out.batch["attention_mask"]                # (B, prompt+resp)
        all_response_masks = full_attn[:, -resp_len:]          # (B, T) response 部分

        n = len(messages_list)
        grouped_texts = [all_texts[i * n_samples:(i + 1) * n_samples] for i in range(n)]
        return grouped_texts, old_log_probs, all_responses_tokens, all_response_masks

    def _compute_log_probs(self, messages_list, responses):
        """在 messages_list[i] 的 context 下计算 responses[i] 的 per-token log-probs。"""
        data = _build_logprob_batch(self.tokenizer, messages_list, responses, self.max_prompt_len, self.max_resp_len)
        data.meta_info["micro_batch_size"] = 4
        data.meta_info["temperature"] = float(
            OmegaConf.select(self.config, "actor_rollout_ref.rollout.temperature", default=1.0)
        )
        data.meta_info["use_dynamic_bsz"] = False
        data.meta_info["max_token_len"] = 8192
        data_padded, pad_size = pad_dataproto_to_divisor(data, self.actor_rollout_wg.world_size)
        out_padded = self.actor_rollout_wg.compute_log_prob(data_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        batch = out.batch
        for key in ("old_log_probs", "log_probs", "response_log_probs"):
            if key in batch:
                return batch[key]
        raise KeyError(f"compute_log_prob 返回 keys: {list(batch.keys())}")

    # ──────────────────────────────────────────────────────────────────
    # 单步训练：RLSD 核心
    # ──────────────────────────────────────────────────────────────────

    def _rlsd_step(self, problems: list[RLSDProblem], sample_file=None, step_num=0) -> dict:
        """
        RLSD 单步：
          - opsd_only：每题 1 条 rollout → 无条件 SD
          - 否则：全错→SD；mixed→GRPO；全对跳过
        """
        metrics = {}
        t0 = time.time()
        rollout_temp = float(
            OmegaConf.select(self.config, "actor_rollout_ref.rollout.temperature", default=1.0)
        )

        # ── Step 1: Student Rollout ──────────────────────────────────
        student_msgs = [build_student_messages(p.question) for p in problems]
        student_resps_grouped, all_old_log_probs, all_resp_tokens, all_resp_masks = \
            self._generate_with_logprobs(student_msgs, n_samples=self.student_k)

        # ── Step 2: 计算 reward + 分类 ──────────────────────────────
        sd_student_msgs, sd_teacher_msgs, sd_resps, sd_lp_indices = [], [], [], []
        grpo_msgs, grpo_resps, grpo_rewards, grpo_lp_indices, grpo_group_ids = [], [], [], [], []

        flat_idx = 0
        if self.opsd_only:
            for _pi, (prob, resps) in enumerate(zip(problems, student_resps_grouped)):
                r = resps[0]
                sd_student_msgs.append(build_student_messages(prob.question))
                sd_teacher_msgs.append(
                    build_teacher_privileged_messages(
                        prob.question,
                        prob.ground_truth,
                        prob.reference_solution or None,
                    )
                )
                sd_resps.append(r)
                sd_lp_indices.append(flat_idx)
                flat_idx += 1
        else:
            for _pi, (prob, resps) in enumerate(zip(problems, student_resps_grouped)):
                correctness = [is_correct(r, prob.ground_truth) for r in resps]
                n_correct = sum(correctness)

                if n_correct == 0:
                    for ri, r in enumerate(resps):
                        sd_student_msgs.append(build_student_messages(prob.question))
                        sd_teacher_msgs.append(
                            build_teacher_privileged_messages(
                                prob.question,
                                prob.ground_truth,
                                prob.reference_solution or None,
                            )
                        )
                        sd_resps.append(r)
                        sd_lp_indices.append(flat_idx + ri)
                elif n_correct < len(resps):
                    for ri, (r, c) in enumerate(zip(resps, correctness)):
                        grpo_msgs.append(build_student_messages(prob.question))
                        grpo_resps.append(r)
                        grpo_rewards.append(1.0 if c else 0.0)
                        grpo_lp_indices.append(flat_idx + ri)
                        grpo_group_ids.append(_pi)

                flat_idx += self.student_k

        metrics["rlsd/n_sd_samples"] = float(len(sd_resps))
        metrics["rlsd/n_grpo_samples"] = float(len(grpo_resps))
        metrics["rlsd/n_problems"] = float(len(problems))

        if self.opsd_only:
            n_all_wrong = n_mixed = n_solved = 0
            metrics["rlsd/opsd_no_dataset_reward_routing"] = 1.0
        else:
            n_all_wrong = sum(
                1
                for prob, resps in zip(problems, student_resps_grouped)
                if all(not is_correct(r, prob.ground_truth) for r in resps)
            )
            n_mixed = sum(
                1
                for prob, resps in zip(problems, student_resps_grouped)
                if 0 < sum(is_correct(r, prob.ground_truth) for r in resps) < len(resps)
            )
            n_solved = len(problems) - n_all_wrong - n_mixed

        metrics["rlsd/n_all_wrong_problems"] = float(n_all_wrong)
        metrics["rlsd/n_mixed"] = float(n_mixed)
        metrics["rlsd/n_all_correct"] = float(n_solved)
        metrics["rlsd/grpo_only_mode"] = float(self.grpo_only)
        metrics["rlsd/opsd_only_mode"] = float(self.opsd_only)

        # ── Response length 统计（全部 rollout）────────────────────
        resp_lens = all_resp_masks.sum(dim=-1).float()  # (total_samples,)
        metrics["rollout/resp_len_min"]  = resp_lens.min().item()
        metrics["rollout/resp_len_mean"] = resp_lens.mean().item()
        metrics["rollout/resp_len_max"]  = resp_lens.max().item()

        # 各分支的长度均值（便于对比 SD/GRPO rollout 长短差异）
        # 默认 0，无对应样本的步骤在 wandb 仍有完整 key
        metrics["rollout/grpo_resp_len_mean"] = 0.0
        metrics["rollout/sd_resp_len_mean"] = 0.0
        if grpo_lp_indices and not self.opsd_only:
            grpo_lens = resp_lens[torch.tensor(grpo_lp_indices, dtype=torch.long)]
            metrics["rollout/grpo_resp_len_mean"] = grpo_lens.mean().item()
        if sd_lp_indices and not self.grpo_only:
            sd_lens = resp_lens[torch.tensor(sd_lp_indices, dtype=torch.long)]
            metrics["rollout/sd_resp_len_mean"] = sd_lens.mean().item()

        if self.opsd_only:
            mode_tag = "opsd-only"
        elif self.grpo_only:
            mode_tag = "grpo-only"
        else:
            mode_tag = "rlsd"
        if self.opsd_only:
            print(
                f"  [{mode_tag}] problems={len(problems)}"
                f"  (1 rollout → OPSD/sample, 全错/混合 计数不适用)"
            )
        else:
            print(
                f"  [{mode_tag}] problems={len(problems)}  "
                f"all_wrong={n_all_wrong}  mixed={n_mixed}  all_correct={n_solved}"
            )
        sys.stdout.flush()

        # ── Step 3: SD 分支训练（GRPO-Only 模式下跳过）────────────────
        if sd_resps and not self.grpo_only:
            sd_data = _build_sd_train_batch(
                self.tokenizer, sd_student_msgs, sd_teacher_msgs, sd_resps,
                self.max_prompt_len, self.max_resp_len
            )
            sd_data.meta_info["rlsd_mode"] = "sd"
            sd_data.meta_info["temperature"] = rollout_temp
            sd_data.meta_info["kl_clip"] = self.kl_clip
            sd_data.meta_info["global_token_num"] = (
                sd_data.batch["attention_mask"].sum(dim=-1).tolist()
            )
            # ── Uncertainty-aware mask ──────────────────────────────
            sd_data.meta_info["sd_mask_mode"] = self.sd_mask_mode
            sd_data.meta_info["sd_mask_entropy_percentile"] = self.sd_mask_entropy_percentile
            if self.sd_mask_mode == "token_identity":
                # Lazy-build epistemic token IDs (requires tokenizer)
                if self._epistemic_token_ids is None:
                    from recipe.RLSD.rlsd.epistemic_mask import build_epistemic_token_ids
                    self._epistemic_token_ids = list(build_epistemic_token_ids(self.tokenizer))
                sd_data.meta_info["epistemic_token_ids"] = self._epistemic_token_ids
            # ────────────────────────────────────────────────────────
            sd_padded, pad_size = pad_dataproto_to_divisor(sd_data, self.actor_rollout_wg.world_size)
            sd_out_padded = self.actor_rollout_wg.update_actor(sd_padded)
            sd_out = unpad_dataproto(sd_out_padded, pad_size=pad_size)
            sd_metrics = reduce_metrics(sd_out.meta_info.get("metrics", {}))
            metrics.update(sd_metrics)

        # ── Step 4: GRPO 分支训练（OPSD-Only 永不进入）────────────────
        # 预设默认值：GRPO 未触发的步骤仍能在 wandb 看到完整 key
        metrics["actor/entropy"] = 0.0
        metrics["actor/kl_loss"] = 0.0
        metrics["actor/kl_coef"] = float(OmegaConf.select(
            self.config, "actor_rollout_ref.actor.kl_loss_coef", default=0.0
        ))
        if grpo_resps and not self.opsd_only:
            lp_indices = torch.tensor(grpo_lp_indices, dtype=torch.long)

            # 取出对应的原始 token ids 和 response mask
            grpo_resp_tokens = all_resp_tokens[lp_indices]   # (B_grpo, T)
            grpo_resp_masks  = all_resp_masks[lp_indices]    # (B_grpo, T)

            # ── 与官方 verl 对齐：用 actor 的 compute_log_prob 重算 old_log_probs
            # 官方注释："we should always recompute old_log_probs when it is HybridEngine"
            # compute_log_prob 在 worker 层硬编码 calculate_entropy=True，
            # 因此 old_log_probs 与 entropys 一次前向同时拿到，无额外开销。
            lp_batch = _build_logprob_batch_from_tokens(
                self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                self.max_prompt_len,
            )
            lp_batch.meta_info["micro_batch_size"] = (
                self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
            )
            lp_batch.meta_info["temperature"] = rollout_temp
            lp_batch.meta_info["use_dynamic_bsz"] = bool(
                self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
            )
            lp_batch.meta_info["max_token_len"] = int(
                self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
            )
            lp_padded, lp_pad_size = pad_dataproto_to_divisor(lp_batch, self.actor_rollout_wg.world_size)
            lp_out_padded = self.actor_rollout_wg.compute_log_prob(lp_padded)
            lp_out = unpad_dataproto(lp_out_padded, pad_size=lp_pad_size)

            grpo_old_lp = lp_out.batch["old_log_probs"]   # (B_grpo, T)

            # entropy：update 前的策略熵，与官方 agg_loss 聚合方式一致
            from verl.trainer.ppo.core_algos import agg_loss as _agg_loss
            entropys = lp_out.batch["entropys"]            # (B_grpo, T)
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = _agg_loss(
                loss_mat=entropys,
                loss_mask=grpo_resp_masks.float(),
                loss_agg_mode=loss_agg_mode,
            )
            metrics["actor/entropy"] = entropy_agg.item()

            # ref_log_probs：use_kl_loss=True 时 dp_actor 需要此字段
            use_kl = OmegaConf.select(self.config, "actor_rollout_ref.actor.use_kl_loss", default=False)
            grpo_ref_lp = None
            if use_kl:
                ref_lp_data = _build_logprob_batch_from_tokens(
                    self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                    self.max_prompt_len,
                )
                ref_lp_data.meta_info["micro_batch_size"] = 4
                ref_lp_data.meta_info["temperature"] = rollout_temp
                ref_lp_data.meta_info["use_dynamic_bsz"] = False
                ref_lp_data.meta_info["max_token_len"] = 8192
                ref_lp_data_padded, pad_size_ref = pad_dataproto_to_divisor(
                    ref_lp_data, self.actor_rollout_wg.world_size
                )
                ref_out_padded = self.actor_rollout_wg.compute_ref_log_prob(ref_lp_data_padded)
                ref_out = unpad_dataproto(ref_out_padded, pad_size=pad_size_ref)
                grpo_ref_lp = ref_out.batch["ref_log_prob"]  # (B_grpo, T)

            grpo_data = _build_grpo_train_batch(
                self.tokenizer, grpo_msgs, grpo_resp_tokens, grpo_resp_masks,
                grpo_rewards, grpo_old_lp, grpo_group_ids,
                self.max_prompt_len,
                ref_log_probs=grpo_ref_lp,
            )
            grpo_data.meta_info["rlsd_mode"] = "grpo"
            grpo_data.meta_info["temperature"] = rollout_temp
            grpo_data.meta_info["global_token_num"] = (
                grpo_data.batch["attention_mask"].sum(dim=-1).tolist()
            )
            grpo_padded, pad_size = pad_dataproto_to_divisor(grpo_data, self.actor_rollout_wg.world_size)
            grpo_out_padded = self.actor_rollout_wg.update_actor(grpo_padded)
            grpo_out = unpad_dataproto(grpo_out_padded, pad_size=pad_size)
            grpo_metrics = reduce_metrics(grpo_out.meta_info.get("metrics", {}))
            metrics.update(grpo_metrics)

        metrics["rlsd/step_time_s"] = time.time() - t0

        # ── 定期保存训练输出采样 ────────────────────────────────────
        if sample_file and len(problems) > 0:
            import json as _json
            from recipe.RLSD.rlsd.verifier import extract_boxed_answer

            n_show = min(3, len(problems))
            with open(sample_file, "a") as _f:
                for _pi in range(n_show):
                    prob = problems[_pi]
                    resps = student_resps_grouped[_pi]
                    correct_flags = [is_correct(r, prob.ground_truth) for r in resps]
                    for _ri, _r in enumerate(resps):
                        _f.write(_json.dumps({
                            "step": step_num,
                            "problem_idx": prob.index,
                            "question": prob.question[:500],
                            "ground_truth": prob.ground_truth,
                            "rollout": _ri,
                            "correct": int(correct_flags[_ri]),
                            "extracted": (extract_boxed_answer(_r) or ""),
                            "response": _r,
                            "resp_len_tokens": len(_r),
                        }, ensure_ascii=False) + "\n")
                    _f.flush()

        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 验证（data.val_files parquet；AIME*: acc@n / avg@n 随机采样（同值双键）；其它: greedy pass@1）
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_val_parquet_paths(val_files):
        """将 ``data.val_files`` 规范为若干 .parquet 路径列表（支持单路径或列表）。"""
        from omegaconf import ListConfig

        if val_files is None:
            return []
        if isinstance(val_files, (list, tuple, ListConfig)):
            raw = [str(p).strip() for p in val_files]
        else:
            raw = [str(val_files).strip()]
        paths = [p for p in raw if p]
        bad = [p for p in paths if not p.endswith(".parquet")]
        if bad:
            raise ValueError(f"[eval] 验证集仅支持 .parquet，收到: {bad}")
        return paths

    @staticmethod
    def _eval_benchmark_is_aime(bench: str) -> bool:
        return "aime" in bench.lower()

    def _evaluate(self, step: int, logger: Tracking) -> dict:
        """在多份 val parquet 上评测；文件名含 ``aime`` 的基准每题随机采样 n 次（``rlsd.eval_aime_avg_at_n``），
        指标为「每题正确率均值」（macro）；同时写入 ``acc_at_{n}`` 与 ``avg_at_{n}`` 两个键，数值相同。"""
        import json

        val_spec = OmegaConf.select(self.config, "data.val_files", default=None)
        val_paths = self._normalize_val_parquet_paths(val_spec)
        if not val_paths:
            return {}

        raw_max = OmegaConf.select(self.config, "rlsd.val_max_samples", default=64)
        if raw_max is None:
            max_val_cap = -1
        else:
            max_val_cap = int(raw_max)

        ckpt_dir = Path(self.config.trainer.default_local_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_file = ckpt_dir / "eval_samples.jsonl"

        merged: dict[str, float] = {}
        macro_scores: list[float] = []

        benchmark_leaf: dict[str, str] = {}

        from recipe.RLSD.rlsd.verifier import extract_boxed_answer

        def _stripped_benchmark_stem(val_path: str) -> str:
            b = Path(val_path).stem
            return b[len("val_") :] if b.startswith("val_") else b

        with open(sample_file, "a") as fout:
            for val_path in val_paths:
                bench = _stripped_benchmark_stem(val_path)

                df = pd.read_parquet(val_path)
                n_all = len(df)
                if max_val_cap > 0:
                    df = df.head(max_val_cap)
                print(
                    f"\n[eval] step={step}  benchmark={bench}  file={val_path}\n"
                    f"[eval] 评测行数: {len(df)}/{n_all}"
                    + ("" if max_val_cap > 0 else "  (全量)")
                    + f"  [rlsd.val_max_samples={raw_max}]"
                )

                messages_list = []
                ground_truths = []
                questions = []
                for _, row in df.iterrows():
                    raw_prompt = row["prompt"]
                    msgs = raw_prompt if isinstance(raw_prompt, list) else list(raw_prompt)
                    rm = row["reward_model"]
                    gt = rm["ground_truth"] if isinstance(rm, dict) else dict(rm)["ground_truth"]
                    question = question_from_verl_prompt(msgs)
                    questions.append(question)
                    messages_list.append(build_student_messages(question))
                    ground_truths.append(gt)

                n_probs = len(ground_truths)
                is_aime = self._eval_benchmark_is_aime(bench)
                k_aime = int(OmegaConf.select(self.config, "rlsd.eval_aime_avg_at_n", default=12))

                if is_aime and k_aime > 1:
                    temp_ev = float(OmegaConf.select(self.config, "rlsd.eval_aime_temperature", default=1.0))
                    top_p_ev = float(OmegaConf.select(self.config, "rlsd.eval_aime_top_p", default=0.95))
                    top_k_ev = int(OmegaConf.select(self.config, "rlsd.eval_aime_top_k", default=-1))
                    expanded = [msg for msg in messages_list for _ in range(k_aime)]
                    gen_batch = _build_gen_batch(self.tokenizer, expanded, self.max_prompt_len)
                    gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    gen_padded.meta_info["do_sample"] = True
                    gen_padded.meta_info["temperature"] = temp_ev
                    gen_padded.meta_info["top_p"] = top_p_ev
                    gen_padded.meta_info["top_k"] = top_k_ev
                    out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                    out = unpad_dataproto(out_padded, pad_size=pad_size)
                    responses = self.tokenizer.batch_decode(
                        out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    # 宏平均：每题 score = (该题 n 次采样中判对次数)/n，再在题上平均；与「全体 trial 微平均」一般不等。
                    mleaf_avg = f"avg_at_{k_aime}"
                    mleaf_acc = f"acc_at_{k_aime}"
                    benchmark_leaf[bench] = mleaf_acc
                    per_q_frac_sum = 0.0
                    total_correct_trials = 0
                    examine_q = min(1, n_probs)
                    for qi in range(n_probs):
                        chunk = responses[qi * k_aime : (qi + 1) * k_aime]
                        gt = ground_truths[qi]
                        qtxt = questions[qi]
                        q_correct = 0
                        for trial, r in enumerate(chunk):
                            correct = is_correct(r, gt)
                            extracted = extract_boxed_answer(r) or ""
                            if correct:
                                q_correct += 1
                                total_correct_trials += 1
                            if qi < examine_q and trial < min(3, k_aime):
                                status = "+" if correct else "-"
                                print(f"  [{bench}][Q{qi} t{trial}{status}] gt={gt}  pred={extracted or '(none)'}")
                                sys.stdout.flush()
                            fout.write(
                                json.dumps(
                                    {
                                        "step": step,
                                        "benchmark": bench,
                                        "metric": mleaf_acc,
                                        "problem_idx": qi,
                                        "trial": trial,
                                        "question": qtxt,
                                        "ground_truth": gt,
                                        "response": r,
                                        "extracted": extracted,
                                        "correct": correct,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        frac_q = q_correct / k_aime
                        per_q_frac_sum += frac_q

                    avg_metric = per_q_frac_sum / max(n_probs, 1)
                    macro_scores.append(avg_metric)
                    merged[f"val/{bench}/{mleaf_acc}"] = avg_metric
                    merged[f"val/{bench}/{mleaf_avg}"] = avg_metric
                    merged[f"val/{bench}/n_correct_trials"] = float(total_correct_trials)
                    merged[f"val/{bench}/n_total_trials"] = float(n_probs * k_aime)
                    merged[f"val/{bench}/n_questions"] = float(n_probs)
                    micro_trial_acc = total_correct_trials / max(n_probs * k_aime, 1)
                    merged[f"val/{bench}/trial_micro_acc"] = float(micro_trial_acc)
                    print(
                        f"[eval] step={step}  {bench}  acc@{k_aime}=avg@{k_aime}={avg_metric:.3f} "
                        f"(macro；trial 微平均={micro_trial_acc:.3f})  "
                        f"判对 trial {total_correct_trials}/{n_probs * k_aime}"
                    )
                else:
                    gen_batch = _build_gen_batch(self.tokenizer, messages_list, self.max_prompt_len)
                    gen_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    gen_padded.meta_info["do_sample"] = False
                    gen_padded.meta_info["temperature"] = 0.0
                    gen_padded.meta_info["top_p"] = 1.0
                    gen_padded.meta_info["top_k"] = 1
                    out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
                    out = unpad_dataproto(out_padded, pad_size=pad_size)

                    responses = self.tokenizer.batch_decode(
                        out.batch["responses"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    benchmark_leaf[bench] = "pass@1"
                    n_correct = 0
                    n_examine = min(3, len(responses))
                    for i, (r, gt, q) in enumerate(zip(responses, ground_truths, questions)):
                        correct = is_correct(r, gt)
                        extracted = extract_boxed_answer(r) or ""
                        if correct:
                            n_correct += 1
                        if i < n_examine:
                            status = "+" if correct else "-"
                            print(f"  [{bench}][{status}] Q{i} gt={gt}  pred={extracted or '(none)'}")
                            sys.stdout.flush()
                        fout.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "benchmark": bench,
                                    "metric": "pass@1",
                                    "problem_idx": i,
                                    "trial": 0,
                                    "question": q,
                                    "ground_truth": gt,
                                    "response": r,
                                    "extracted": extracted,
                                    "correct": correct,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    pass1 = n_correct / max(len(ground_truths), 1)
                    macro_scores.append(pass1)
                    n_tot = float(len(ground_truths))
                    merged[f"val/{bench}/pass@1"] = pass1
                    merged[f"val/{bench}/n_correct"] = float(n_correct)
                    merged[f"val/{bench}/n_total"] = n_tot
                    print(f"[eval] step={step}  {bench}  pass@1={pass1:.3f}  ({n_correct}/{len(ground_truths)})")

        if len(val_paths) > 1:
            # 各 basename 评测主指标的简单平均（命名沿用；含 MATH pass@1 与 AIME acc@n/avg@n 混比时仅作粗参考）
            merged["val/pass@1_macro_mean"] = sum(macro_scores) / len(macro_scores)
        else:
            b = _stripped_benchmark_stem(val_paths[0])
            leaf = benchmark_leaf[b]
            if leaf == "pass@1":
                merged["val/pass@1"] = merged[f"val/{b}/pass@1"]
                merged["val/n_correct"] = merged[f"val/{b}/n_correct"]
                merged["val/n_total"] = merged[f"val/{b}/n_total"]
            else:
                merged[f"val/{leaf}"] = merged[f"val/{b}/{leaf}"]
                if leaf.startswith("acc_at_"):
                    legacy_leaf = leaf.replace("acc_at_", "avg_at_", 1)
                    merged[f"val/{legacy_leaf}"] = merged[f"val/{b}/{legacy_leaf}"]
                merged["val/n_correct_trials"] = merged[f"val/{b}/n_correct_trials"]
                merged["val/n_total_trials"] = merged[f"val/{b}/n_total_trials"]
                merged["val/n_questions_eval"] = merged[f"val/{b}/n_questions"]
        logger.log(data=merged, step=step)
        print(f"[eval] 样本已追加到 {sample_file}\n")
        return merged

    # ──────────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """RLSD 训练循环。"""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        total_steps = self.total_training_steps
        save_freq = int(OmegaConf.select(self.config.trainer, "save_freq", default=50))
        test_freq = int(OmegaConf.select(self.config.trainer, "test_freq", default=10))
        ckpt_dir = Path(self.config.trainer.default_local_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_steps = 0
        self._load_checkpoint()

        print(f"\n[RLSDTrainer] 开始  total_steps={total_steps}  n_problems={len(self.rlsd_dataset)}")
        if not bool(OmegaConf.select(self.config, "rlsd.skip_initial_eval", default=False)):
            self._evaluate(step=0, logger=logger)
        else:
            print("[RLSDTrainer] 已跳过初试验证（rlsd.skip_initial_eval=true）")

        progress = tqdm(total=total_steps, initial=self.global_steps, desc="RLSD")

        def _scalar(v):
            return float(v[0]) if isinstance(v, (list, tuple)) else float(v)

        sample_file = ckpt_dir / "train_samples.jsonl"
        sample_freq = int(OmegaConf.select(self.config, "rlsd.sample_dump_freq", default=10))

        while self.global_steps < total_steps:
            batch = self.rlsd_dataset.sample_batch(self.problems_per_step)
            if not batch:
                continue

            dump = sample_file if self.global_steps % sample_freq == 0 else None
            step_metrics = self._rlsd_step(batch, sample_file=dump, step_num=self.global_steps)
            self.global_steps += 1
            step_metrics["train/global_step"] = float(self.global_steps)
            logger.log(data=step_metrics, step=self.global_steps)
            progress.update(1)
            if self.opsd_only:
                progress.set_postfix({
                    "sd": int(_scalar(step_metrics.get("rlsd/n_sd_samples", 0))),
                })
            else:
                progress.set_postfix({
                    "sd": int(_scalar(step_metrics.get("rlsd/n_sd_samples", 0))),
                    "grpo": int(_scalar(step_metrics.get("rlsd/n_grpo_samples", 0))),
                    "all_wrong": int(_scalar(step_metrics.get("rlsd/n_all_wrong_problems", 0))),
                })

            if self.global_steps % test_freq == 0:
                self._evaluate(step=self.global_steps, logger=logger)

            if self.global_steps % save_freq == 0:
                self._save_checkpoint()

        progress.close()
        self._save_checkpoint()

        print(f"\n[RLSDTrainer] 完成  total_steps={self.global_steps}")
