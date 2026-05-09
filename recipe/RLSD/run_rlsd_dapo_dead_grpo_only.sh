#!/bin/bash
# DAPO dead-zone（pass@64≈0）难题集 · GRPO-Only（对照组：无 SD）
#
# 协议：与 run_rlsd_dapo_dead_full.sh 相同数据与评测；仅 mrsd.grpo_only=true。
#
# 用法：bash recipe/RLSD/run_rlsd_dapo_dead_grpo_only.sh [hydra overrides]

set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export PYTHONHASHSEED=42

export VERL_TMP_ROOT=/data3/yyy/tmp
export TMPDIR="${VERL_TMP_ROOT}"
export RAY_TMPDIR="${VERL_TMP_ROOT}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV=verl

DATA_ROOT=/data3/yyy/verl/data
JSONL="${DATA_ROOT}/dapo_dead_pass64_qwen2.5instruct_split_10pct.jsonl"
PARQUET="${DATA_ROOT}/dapo_dead_pass64_qwen2.5instruct_split_10pct.parquet"
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_dapo_dead_grpo_only_qwen25_3b

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/dapo_dead_grpo_only_${TIMESTAMP}.log"

echo "========================================================"
echo "DAPO dead-zone · GRPO-Only · Qwen2.5-Instruct"
echo "  模型: ${MODEL_PATH}"
echo "  题池 jsonl: ${JSONL}"
echo "  train/val parquet（同一文件）: ${PARQUET}"
echo "  检查点: ${CKPT_DIR}"
echo "========================================================"

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
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.top_p=0.9 \
        data.train_files="${PARQUET}" \
        data.val_files="${PARQUET}" \
        data.shuffle=false \
        data.mrsd_problems_path="${JSONL}" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="dapo-dead-grpo-only-qwen25-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=100 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        mrsd.dataset_seed=42 \
        mrsd.student_rollout_per_problem=8 \
        mrsd.problems_per_step=16 \
        mrsd.val_max_samples=-1 \
        mrsd.grpo_only=true \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志: ${LOG_FILE}"
