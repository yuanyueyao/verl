#!/bin/bash
# GRPO-Only 对比实验（与 RLSD 做对照）
# 同数据集、同模型、同超参，唯一区别：跳过 SD 分支（仅 GRPO）
# 用法：bash recipe/RLSD/run_grpo_only.sh [额外 hydra overrides]

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV=verl

# ── 默认参数 ────────────────────────────────────────────────────────────────
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
CKPT_DIR=/data3/yyy/verl/checkpoints/grpo-only

# ── 日志目录 ────────────────────────────────────────────────────────────────
LOG_DIR="${VERL_ROOT}/logs/grpo-only"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "========================================================"
echo "GRPO-Only 对比训练配置"
echo "  模型:     ${MODEL_PATH}"
echo "  数据:     ${TRAIN_DATA}"
echo "  检查点:   ${CKPT_DIR}"
echo "  日志:     ${LOG_FILE}"
echo "  说明:     SD 分支已禁用（grpo_only=true）"
echo "========================================================"

cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="[/data3/yyy/verl/data/math/val_MATH-500.parquet, /data3/yyy/verl/data/math/val_aime_2024.parquet, /data3/yyy/verl/data/math/val_aime_2025.parquet]" \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="grpo-only-qwen25-3b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        rlsd.problems_per_step=32 \
        rlsd.student_rollout_per_problem=8 \
        rlsd.grpo_only=true \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到 ${LOG_FILE}"
