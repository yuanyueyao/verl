#!/bin/bash
# SFT baseline: DeepSeek-R1-Distill-Qwen-1.5B (8-GPU train, then 8-GPU eval)
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
    echo "ERROR: 必须激活 conda env 'verl'"
    exit 1
fi

export TORCH_COMPILE_DISABLE=1
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_PATH=/data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
OUTPUT_ROOT=/data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b
EVAL_ROOT=/data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b_eval
OUTPUT_DIR="${OUTPUT_ROOT}/run_${TIMESTAMP}"
EVAL_OUTPUT_DIR="${EVAL_ROOT}/run_${TIMESTAMP}"
TMP_DIR=/data3/yyy/verl/checkpoints/sft_tmp

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${TMP_DIR}"
LOG_FILE="${LOG_DIR}/sft_ds_qwen1.5b_${TIMESTAMP}.log"

echo "========================================================"
echo " SFT: DS-R1-Distill-Qwen-1.5B (8-GPU DDP)"
echo "  步数: 100"
echo "  Save: every 10 steps"
echo "  Eval: all saved checkpoints after training with 8 vLLM shards"
echo "  输出: ${OUTPUT_DIR}"
echo "  Eval输出: ${EVAL_OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "========================================================"

cd "${VERL_ROOT}"

# Pick 8 free GPUs for training and final sharded eval.
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

# ── Train ──
echo ">>> Training @ $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=8 recipe/RLSD/sft_train.py \
    --model "${MODEL_PATH}" \
    --data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_output_dir "${EVAL_OUTPUT_DIR}" \
    --response_column COT_Reason \
    --prompt_column problem \
    --total_steps 100 \
    --warmup_steps 10 \
    --lr 5e-6 \
    --batch_size 64 \
    --micro_batch_size 1 \
    --max_length 24576 \
    --save_every 10 \
    --eval_every 0 \
    --tmp_dir "${TMP_DIR}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "<<< Training done @ $(date)"

# ── Eval ──
echo ">>> 8-GPU sharded vLLM eval @ $(date)" | tee -a "${LOG_FILE}"
IFS=',' read -r -a GPU_ARRAY <<< "${TRAIN_GPUS}"
for STEP in $(seq 10 10 100); do
    CKPT_DIR="${OUTPUT_DIR}/step_${STEP}"
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "ERROR: missing checkpoint ${CKPT_DIR}" | tee -a "${LOG_FILE}"
        exit 1
    fi
    echo ">>> Eval step ${STEP} @ $(date)" | tee -a "${LOG_FILE}"
    PIDS=()
    for SHARD_ID in $(seq 0 7); do
        GPU_ID="${GPU_ARRAY[$SHARD_ID]}"
        SHARD_LOG="${LOG_DIR}/sft_ds_qwen1.5b_eval_step${STEP}_shard${SHARD_ID}_${TIMESTAMP}.log"
        echo "  step ${STEP} shard ${SHARD_ID}/8 on GPU ${GPU_ID}: ${SHARD_LOG}" | tee -a "${LOG_FILE}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python recipe/RLSD/sft_eval.py \
            --model "${CKPT_DIR}" \
            --output_dir "${EVAL_OUTPUT_DIR}" \
            --step "${STEP}" \
            --max_samples 64 \
            --include_gsm8k \
            --gsm8k_max_samples 1000 \
            --shard_id "${SHARD_ID}" \
            --num_shards 8 \
            > "${SHARD_LOG}" 2>&1 &
        PIDS+=("$!")
    done

    for PID in "${PIDS[@]}"; do
        wait "${PID}"
    done

    python recipe/RLSD/sft_eval.py \
        --model "${CKPT_DIR}" \
        --output_dir "${EVAL_OUTPUT_DIR}" \
        --step "${STEP}" \
        --num_shards 8 \
        --aggregate_only \
        2>&1 | tee -a "${LOG_FILE}"
done

echo "=== Done @ $(date) ==="
