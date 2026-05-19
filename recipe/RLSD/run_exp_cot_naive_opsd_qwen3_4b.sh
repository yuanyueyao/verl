#!/bin/bash
# 实验：COT Naive OPSD on Qwen3-4B-Instruct (8 GPUs)
# 用法：bash recipe/RLSD/run_exp_cot_naive_opsd_qwen3_4b.sh
#
# 假设：COT reference_solution 会抑制 epistemic verbalization → 推理退化
# 与 masked 串行运行，先跑 naive。

set -euo pipefail

# ── 强制检查 conda env ──────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
    echo "ERROR: 必须激活 conda env 'verl' 再跑此脚本"
    echo "  conda activate verl"
    exit 1
fi

export TORCH_COMPILE_DISABLE=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export VERL_TMP_ROOT=/data3/yyy/tmp
export TMPDIR="${VERL_TMP_ROOT}"
export RAY_TMPDIR="${VERL_TMP_ROOT}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=/data3/yyy/models/Qwen3-4B-Instruct-2507
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
MATH_DIR=/data3/yyy/verl/data/math
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_cot_naive_opsd_qwen3_4b

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/exp_cot_naive_opsd_qwen3_4b_${TIMESTAMP}.log"

echo "========================================================"
echo " 实验：COT Naive OPSD on Qwen3-4B-Instruct (8 GPUs)"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${TRAIN_DATA}"
echo "  Ref:  COT_Reason (完整思考链，含 epistemic tokens)"
echo "  Mask: NONE (negative control — expects degradation)"
echo "  日志: ${LOG_FILE}"
echo "  步数: 100"
echo "========================================================"

cd "${VERL_ROOT}"

python recipe/RLSD/main_rlsd.py \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=28672 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="[${MATH_DIR}/val_MATH-500.parquet, ${MATH_DIR}/val_aime_2024.parquet, ${MATH_DIR}/val_aime_2025.parquet]" \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.project_name=rlsd \
    trainer.experiment_name="cot-naive-opsd-qwen3-4b-${TIMESTAMP}" \
    trainer.total_training_steps=100 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    rlsd.problems_per_step=12 \
    rlsd.student_rollout_per_problem=1 \
    rlsd.grpo_only=false \
    rlsd.opsd_only=true \
    rlsd.sd_mask_mode=none \
    rlsd.reference_column=COT_Reason \
    rlsd.skip_initial_eval=false \
    rlsd.eval_aime_avg_at_n=12 \
    rlsd.eval_aime_temperature=1.0 \
    rlsd.eval_aime_top_p=0.95 \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "实验完成，日志已保存到 ${LOG_FILE}"
