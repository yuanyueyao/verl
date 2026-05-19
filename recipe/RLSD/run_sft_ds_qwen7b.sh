#!/bin/bash
# SFT baseline: DeepSeek-R1-Distill-Qwen-7B (veRL FSDP SFT trainer)
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

MODEL_PATH=/data3/yyy/models/DeepSeek-R1-Distill-Qwen-7B
TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
OUTPUT_DIR=/data3/yyy/verl/checkpoints/sft_exp_ds_qwen7b

LOG_DIR="${VERL_ROOT}/logs/rlsd"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/sft_ds_qwen7b_${TIMESTAMP}.log"

echo "========================================================"
echo " SFT: DS-R1-Distill-Qwen-7B (8 GPU, FSDP)"
echo "  步数: 100"
echo "  日志: ${LOG_FILE}"
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

# veRL SFT trainer via torchrun
# Steps per epoch = len(dataset) / train_batch_size
# With 100 steps and ~29K samples, we need ~0.34 epochs
# Use trainer.total_training_steps=100 to control
torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${TRAIN_DATA}" \
    data.prompt_key=problem \
    data.response_key=COT_Reason \
    data.max_length=24576 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=64 \
    model.partial_pretrain="${MODEL_PATH}" \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bfloat16 \
    model.strategy=fsdp2 \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=constant \
    trainer.project_name=ope-sft \
    trainer.experiment_name="ds-r1-qwen7b-sft-${TIMESTAMP}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.total_training_steps=100 \
    trainer.logger="['console','wandb']" \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    2>&1 | tee "${LOG_FILE}"

echo "<<< Training done @ $(date)"

# ── Eval ──
echo ">>> Eval @ $(date)"
EVAL_LOG="${LOG_DIR}/sft_ds_qwen7b_eval_${TIMESTAMP}.log"

for ckpt in $(ls -d "${OUTPUT_DIR}"/global_step_* 2>/dev/null | sort -V); do
    STEP=$(basename "$ckpt" | sed 's/global_step_//')
    echo "--- Eval step ${STEP} ---" | tee -a "${EVAL_LOG}"
    python recipe/RLSD/sft_eval.py \
        --model "$ckpt/actor" --output_dir "$OUTPUT_DIR" --step "$STEP" \
        2>&1 | tee -a "${EVAL_LOG}"
done

echo "=== Done @ $(date) ==="
