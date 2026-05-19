#!/bin/bash
# COT OPSD 全量对照实验：Qwen3-1.7B → Qwen3-4B
# 用法：bash recipe/RLSD/run_all_cot_experiments.sh
#
# 串行运行 4 组：
#   1. Qwen3-1.7B Naive OPSD   (100 steps)
#   2. Qwen3-1.7B Masked OPSD  (100 steps)
#   3. Qwen3-4B   Naive OPSD   (100 steps)
#   4. Qwen3-4B   Masked OPSD  (100 steps)
#
# 预计总时长 ~13h

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo " COT OPSD 全量对照实验"
echo " 模型: Qwen3-1.7B → Qwen3-4B"
echo " 条件: Naive → Masked × 2"
echo " 开始: $(date)"
echo "============================================================================"

# ── Phase 1: Qwen3-1.7B Naive ──

ray stop --force 2>/dev/null || true
sleep 5

echo ""
echo ">>> [1/4] Qwen3-1.7B Naive OPSD @ $(date)"
bash "${SCRIPT_DIR}/run_exp_cot_naive_opsd_qwen3_1.7b.sh"
echo "<<< [1/4] Qwen3-1.7B Naive OPSD 完成 @ $(date)"

# ── Phase 2: Qwen3-1.7B Masked ──

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
echo " 全部 4 组实验完成 @ $(date)"
echo "============================================================================"
