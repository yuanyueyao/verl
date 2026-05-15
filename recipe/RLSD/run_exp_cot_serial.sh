#!/bin/bash
# COT 对照实验：串行运行 naive → masked
# 用法：bash recipe/RLSD/run_exp_cot_serial.sh
#
# 先跑 naive OPSD (100 steps, ~30-40 min)，完成后自动跑 masked OPSD。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " COT OPSD Serial: Naive → Masked"
echo " $(date)"
echo "============================================================"

echo ""
echo "[Phase 1/2] Naive OPSD (no mask, expects degradation)..."
bash "${SCRIPT_DIR}/run_exp_cot_naive_opsd_1.5b.sh"
echo "[Phase 1/2] Naive OPSD 完成 @ $(date)"

echo ""
echo "[Phase 2/2] Masked OPSD (token_identity mask, expects stability)..."
bash "${SCRIPT_DIR}/run_exp_cot_masked_opsd_1.5b.sh"
echo "[Phase 2/2] Masked OPSD 完成 @ $(date)"

echo ""
echo "============================================================"
echo " 全部完成 @ $(date)"
echo "============================================================"
