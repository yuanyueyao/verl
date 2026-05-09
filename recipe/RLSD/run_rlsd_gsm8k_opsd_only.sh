#!/bin/bash
# Qwen2.5-3B-Instruct · OpenThoughts 训练集 · MATH-500/AIME24/AIME25 评测 · OPSD-Only（关闭 GRPO）
# 用法：bash recipe/RLSD/run_rlsd_gsm8k_opsd_only.sh [额外 hydra overrides]
#
# - 训练：OpenThoughts parquet；MRSD 题池从 train_files 构建（data.mrsd_problems_path=null）
# - 评测：data/math 下 val_*.parquet（需事先 export_math_val_parquets.py）
# - mrsd.opsd_only=true → 每题 1×rollout → 必走 SD；不判对错/不更新题池/不毕业检查；mixed 无 GRPO（trainer 仍会忽略配置把 k 强制为 1）
# - 与 mrsd.grpo_only 互斥；actor 超参与 grpo_only 脚本同骨架

set -euo pipefail

# ── 环境设置 ────────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export VERL_TMP_ROOT=/data3/yyy/tmp
export TMPDIR="${VERL_TMP_ROOT}"
export RAY_TMPDIR="${VERL_TMP_ROOT}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV=verl

# ── 默认参数（绝对路径）───────────────────────────────────────────────────────
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
OPS_DIR=/data3/yyy/verl/data/Openthoughts_math_30k_opsd
TRAIN_PARQUET="${OPS_DIR}/data/train.parquet"
MATH_EVAL_DIR=/data3/yyy/verl/data/math
VAL_FILES="[${MATH_EVAL_DIR}/val_MATH-500.parquet,${MATH_EVAL_DIR}/val_aime_2024.parquet,${MATH_EVAL_DIR}/val_aime_2025.parquet]"
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_openthoughts_opsd_only

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "Qwen2.5-3B-Instruct · OpenThoughts · OPSD-Only（禁用 GRPO）"
echo "  模型: ${MODEL_PATH}"
echo "  训练 parquet: ${TRAIN_PARQUET}"
echo "  评测 parquet: ${MATH_EVAL_DIR}/val_{MATH-500,aime_2024,aime_2025}.parquet"
echo "  检查点: ${CKPT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────────
cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.top_p=0.9 \
        data.train_files="${TRAIN_PARQUET}" \
        data.val_files="${VAL_FILES}" \
        data.mrsd_problems_path=null \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="openthoughts-opsd-only-qwen25-3b-instruct-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=100 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.default_local_dir="${CKPT_DIR}" \
        mrsd.student_rollout_per_problem=1 \
        mrsd.problems_per_step=32 \
        mrsd.val_max_samples=-1 \
        mrsd.opsd_only=true \
        mrsd.skip_initial_eval=true \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
