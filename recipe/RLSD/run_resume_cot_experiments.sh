#!/bin/bash
# COT OPSD Resume: 从 [2/4] 继续（[1/4] Naive 1.7B 已完成）
# gpu_memory_utilization 已从 0.6 降至 0.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo " COT OPSD Resume — 从 [2/4] 开始"
echo " gpu_memory_utilization=0.5 (降 0.1)"
echo " 开始: $(date)"
echo "============================================================================"

# ── Phase 2: Qwen3-1.7B Masked (restart from scratch, 0.5 gpu_mem) ──
ray stop --force 2>/dev/null || true
sleep 5

echo ""
echo ">>> [2/4] Qwen3-1.7B Masked OPSD @ $(date)"
bash "${SCRIPT_DIR}/run_exp_cot_masked_opsd_qwen3_1.7b.sh"
echo "<<< [2/4] Qwen3-1.7B Masked OPSD 完成 @ $(date)"

# ── Phase 3: Qwen3-4B Naive ──
ray stop --force 2>/dev/null || true
sleep 5

echo ""
echo ">>> [3/4] Qwen3-4B Naive OPSD @ $(date)"
bash "${SCRIPT_DIR}/run_exp_cot_naive_opsd_qwen3_4b.sh"
echo "<<< [3/4] Qwen3-4B Naive OPSD 完成 @ $(date)"

# ── Phase 4: Qwen3-4B Masked ──
ray stop --force 2>/dev/null || true
sleep 5

echo ""
echo ">>> [4/4] Qwen3-4B Masked OPSD @ $(date)"
bash "${SCRIPT_DIR}/run_exp_cot_masked_opsd_qwen3_4b.sh"
echo "<<< [4/4] Qwen3-4B Masked OPSD 完成 @ $(date)"

echo ""
echo "============================================================================"
echo " [2/4]–[4/4] 完成 @ $(date)"
echo "============================================================================"
