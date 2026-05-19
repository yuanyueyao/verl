#!/bin/bash
# SFT baseline: DeepSeek-R1-Distill-Qwen-7B
# 100 steps, COT_Reason as response, aligned with SD experiments

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
    echo "ERROR: 必须激活 conda env 'verl'"
    echo "  conda activate verl"
    exit 1
fi

export TORCH_COMPILE_DISABLE=1
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MODEL_PATH=/data3/yyy/models/DeepSeek-R1-Distill-Qwen-7B
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
OUTPUT_DIR=/data3/yyy/verl/checkpoints/sft_exp_ds_qwen7b

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/sft_ds_qwen7b_${TIMESTAMP}.log"

echo "========================================================"
echo " SFT baseline: DS-R1-Distill-Qwen-7B"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${TRAIN_DATA}"
echo "  输出: ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "  步数: 100"
echo "========================================================"

cd "${VERL_ROOT}"

# Auto-detect free GPUs
GPUS=$(python3 -c "
import subprocess
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().split('\n')
free = [l.split(',')[0].strip() for l in out if int(l.split(',')[1].strip().split()[0]) < 1000]
print(','.join(free))
")
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "  可用 GPU (${N_GPUS}): ${GPUS}"

export CUDA_VISIBLE_DEVICES=$GPUS

accelerate launch \
    --num_processes="${N_GPUS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    recipe/RLSD/sft_train.py \
    --model "${MODEL_PATH}" \
    --data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --response_column COT_Reason \
    --prompt_column problem \
    --total_steps 100 \
    --warmup_steps 10 \
    --lr 5e-6 \
    --batch_size 64 \
    --micro_batch_size 1 \
    --max_length 24576 \
    --eval_every 10 \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo "SFT 7B 完成，日志已保存到 ${LOG_FILE}"
