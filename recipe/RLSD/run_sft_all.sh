#!/bin/bash
# SFT 全量实验：1.5B → 7B 串行
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo " SFT Baseline Experiments: DS-R1-1.5B → DS-R1-7B"
echo " 开始: $(date)"
echo "============================================================================"

echo ""
echo ">>> [1/2] SFT DS-R1-Distill-Qwen-1.5B @ $(date)"
bash "${SCRIPT_DIR}/run_sft_ds_qwen1.5b.sh"
echo "<<< [1/2] SFT DS-R1-Distill-Qwen-1.5B 完成 @ $(date)"

echo ""
echo ">>> [2/2] SFT DS-R1-Distill-Qwen-7B @ $(date)"
bash "${SCRIPT_DIR}/run_sft_ds_qwen7b.sh"
echo "<<< [2/2] SFT DS-R1-Distill-Qwen-7B 完成 @ $(date)"

echo ""
echo "============================================================================"
echo " SFT 全部完成 @ $(date)"
echo "============================================================================"
