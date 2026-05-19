#!/bin/bash
# 从 Phase 2 恢复：自动检测可用 GPU，串行跑 1.7B Masked → 4B Naive → 4B Masked
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo " COT OPSD Resume: Phase 2/4 → 4/4 (auto GPU detection)"
echo " 开始: $(date)"
echo "============================================================================"

cleanup_ray() {
    echo ">>> 清理 Ray 集群..."
    ray stop --force 2>/dev/null || true
    sleep 5
}

# ── Auto-detect free GPUs ──
detect_gpus() {
    # Returns comma-separated list of GPU indices with < 1 GB used
    python3 -c "
import subprocess, sys
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip().split('\n')
free = [l.split(',')[0].strip() for l in out if int(l.split(',')[1].strip().split()[0]) < 1000]
if len(free) < 2:
    print('ERROR: < 2 free GPUs', file=sys.stderr)
    sys.exit(1)
print(','.join(free))
"
}

# ── Run a phase with auto GPU detection ──
run_phase() {
    local phase_num=$1
    local phase_name=$2
    local script=$3
    local lr=${4:-5e-6}

    cleanup_ray

    GPUS=$(detect_gpus)
    N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
    echo ">>> 检测到 ${N_GPUS} 个可用 GPU: ${GPUS}"

    # Compute train_batch_size that's divisible by n_gpus and >= 64
    TRAIN_BSZ=$(python3 -c "n=$N_GPUS; b=64; print(b + (n - b % n) % n)")
    MINI_BSZ=$TRAIN_BSZ

    echo "    train_batch_size=${TRAIN_BSZ}, ppo_mini_batch_size=${MINI_BSZ}"

    CUDA_VISIBLE_DEVICES=$GPUS python recipe/RLSD/main_rlsd.py \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr="${lr}" \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.use_kl_loss=false \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BSZ}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=0.9 \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_model_len=28672 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="[${MATH_DIR}/val_MATH-500.parquet, ${MATH_DIR}/val_aime_2024.parquet, ${MATH_DIR}/val_aime_2025.parquet]" \
        data.train_batch_size="${TRAIN_BSZ}" \
        data.max_prompt_length=8192 \
        data.max_response_length=16384 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.project_name=rlsd \
        trainer.experiment_name="${EXP_NAME}-${TIMESTAMP}" \
        trainer.total_training_steps=100 \
        trainer.save_freq=50 \
        trainer.test_freq=10 \
        trainer.resume_mode=disable \
        trainer.n_gpus_per_node="${N_GPUS}" \
        trainer.nnodes=1 \
        rlsd.problems_per_step=12 \
        rlsd.student_rollout_per_problem=1 \
        rlsd.grpo_only=false \
        rlsd.opsd_only=true \
        rlsd.sd_mask_mode="${MASK_MODE}" \
        rlsd.reference_column=COT_Reason \
        rlsd.skip_initial_eval=false \
        rlsd.eval_aime_avg_at_n=12 \
        rlsd.eval_aime_temperature=1.0 \
        rlsd.eval_aime_top_p=0.95 \
        2>&1

    echo "<<< [${phase_num}/4] ${phase_name} 完成 @ $(date)"
}

TRAIN_DATA=/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet
MATH_DIR=/data3/yyy/verl/data/math
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Phase 2: Qwen3-1.7B Masked ──
echo ""
echo ">>> [2/4] Qwen3-1.7B Masked OPSD @ $(date)"
MODEL_PATH=/data3/yyy/models/Qwen3-1.7B
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_cot_masked_opsd_qwen3_1.7b
EXP_NAME="cot-masked-opsd-qwen3-1.7b"
MASK_MODE="token_identity"
GPU_MEM="0.6"
run_phase 2 "Qwen3-1.7B Masked OPSD" "masked_1.7b" 5e-6

# ── Phase 3: Qwen3-4B Naive ──
echo ""
echo ">>> [3/4] Qwen3-4B Naive OPSD @ $(date)"
MODEL_PATH=/data3/yyy/models/Qwen3-4B-Instruct-2507
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_cot_naive_opsd_qwen3_4b
EXP_NAME="cot-naive-opsd-qwen3-4b"
MASK_MODE="none"
GPU_MEM="0.6"
run_phase 3 "Qwen3-4B Naive OPSD" "naive_4b" 5e-6

# ── Phase 4: Qwen3-4B Masked ──
echo ""
echo ">>> [4/4] Qwen3-4B Masked OPSD @ $(date)"
MODEL_PATH=/data3/yyy/models/Qwen3-4B-Instruct-2507
CKPT_DIR=/data3/yyy/verl/checkpoints/rlsd_exp_cot_masked_opsd_qwen3_4b
EXP_NAME="cot-masked-opsd-qwen3-4b"
MASK_MODE="token_identity"
GPU_MEM="0.6"
run_phase 4 "Qwen3-4B Masked OPSD" "masked_4b" 5e-6

echo ""
echo "============================================================================"
echo " Phase 2-4 全部完成 @ $(date)"
echo "============================================================================"
