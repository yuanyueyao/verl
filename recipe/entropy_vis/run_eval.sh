#!/usr/bin/env bash
# 多模型逐 token 熵评测启动脚本（vLLM 加速版）。
# 前置条件：先 `conda activate verl`（参见仓库根目录 CLAUDE.md）。
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl" ]]; then
    echo "ERROR: 必须先 'conda activate verl' 再启动本脚本（当前 env: ${CONDA_DEFAULT_ENV:-<none>}）。" >&2
    echo "       CUDA 12.8 toolchain 和 vLLM 依赖都在该 env 内。" >&2
    exit 1
fi

# 避免 vLLM 子进程 fork 时重复初始化 CUDA
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# 默认跑全部模型，可通过环境变量 MODELS 只跑子集
# 例：MODELS="Qwen2.5-3B-Instruct Qwen3-4B-Base" bash run_eval.sh
EXTRA_ARGS=()
if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    _models=($MODELS)
    EXTRA_ARGS+=(--models "${_models[@]}")
fi
if [[ -n "${N_PROBLEMS:-}" ]]; then
    EXTRA_ARGS+=(--n-problems "$N_PROBLEMS")
fi
if [[ -n "${MAX_NEW:-}" ]]; then
    EXTRA_ARGS+=(--max-new "$MAX_NEW")
fi
if [[ -n "${GPU_MEM_UTIL:-}" ]]; then
    EXTRA_ARGS+=(--gpu-mem-util "$GPU_MEM_UTIL")
fi

python eval_entropy_all_models.py "${EXTRA_ARGS[@]}" "$@"
