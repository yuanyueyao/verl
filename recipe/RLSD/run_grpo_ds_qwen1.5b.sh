#!/bin/bash
# GRPO baseline: DeepSeek-R1-Distill-Qwen-1.5B (8 GPU)
# Pure GRPO, no self-distillation — compare against Masked SD
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
    echo "ERROR: 必须激活 conda env 'verl'"
    exit 1
fi

export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export VERL_TMP_ROOT=/data3/yyy/tmp
export TMPDIR="${VERL_TMP_ROOT}"
export RAY_TMPDIR="${VERL_TMP_ROOT}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_PATH=/data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
MATH_DIR=/data3/yyy/verl/data/math
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_grpo_ds_qwen1.5b

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
LOG_FILE="${LOG_DIR}/grpo_ds_qwen1.5b_${TIMESTAMP}.log"

echo "========================================================"
echo " GRPO baseline: DS-R1-Distill-Qwen-1.5B (8 GPU)"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${TRAIN_DATA}"
echo "  步数: 100, test_freq=10"
echo "  输出: ${CKPT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

cd "${VERL_ROOT}"

# Pick 8 free GPUs
TRAIN_GPUS=$(python3 -c "
import subprocess
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().split('\n')
free = [l.split(',')[0].strip() for l in out if int(l.split(',')[1].strip().split()[0]) < 1000]
if len(free) < 8:
    raise SystemExit(f'need at least 8 free GPUs, found {len(free)}: {free}')
print(','.join(free[:8]))
")
echo "  Train GPUs: ${TRAIN_GPUS}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

python recipe/RLSD/main_rlsd.py \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name="grpo-dsr1-1.5b-${TIMESTAMP}" \
    trainer.total_training_steps=100 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    rlsd.problems_per_step=16 \
    rlsd.student_rollout_per_problem=8 \
    rlsd.grpo_only=true \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "=== Done @ $(date) ==="
