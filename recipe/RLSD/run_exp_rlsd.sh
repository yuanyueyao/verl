#!/bin/bash
# 实验2：RLSD（SD + GRPO），4 GPU
# 用法：bash recipe/RLSD/run_exp_rlsd.sh

set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export VERL_TMP_ROOT=/data3/yyy/tmp
export TMPDIR="${VERL_TMP_ROOT}"
export RAY_TMPDIR="${VERL_TMP_ROOT}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV=verl
MODEL_PATH=/data3/yyy/models/Qwen2.5-3B-Instruct
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
MATH_DIR=/data3/yyy/verl/data/math
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_rlsd

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/exp_rlsd_${TIMESTAMP}.log"

echo "========================================================"
echo " 实验2：RLSD（SD + GRPO）"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${TRAIN_DATA}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

cd "${VERL_ROOT}"

conda run -n ${CONDA_ENV} --no-capture-output \
    python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=0.9 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="[${MATH_DIR}/val_MATH-500.parquet, ${MATH_DIR}/val_aime_2024.parquet, ${MATH_DIR}/val_aime_2025.parquet]" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="rlsd-qwen25-3b-${TIMESTAMP}" \
        trainer.total_training_steps=500 \
        trainer.save_freq=100 \
        trainer.test_freq=10 \
        trainer.resume_mode=auto \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        rlsd.problems_per_step=12 \
        rlsd.student_rollout_per_problem=8 \
        rlsd.kl_clip=10.0 \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "实验2完成，日志已保存到 ${LOG_FILE}"
