#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "用法：bash scripts/vbench/inference.sh <checkpoint.pt> [inference.py 参数...]" >&2
    exit 2
fi
if [[ ! -f "$1" ]]; then
    echo "Checkpoint 不存在或不是文件：$1" >&2
    exit 1
fi
checkpoint=$(realpath -- "$1")
shift

exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path "$checkpoint" \
    --data_path prompts/vbench/all_dimension.txt \
    --extended_prompt_path prompts/vbench/all_dimension_extended.txt \
    --num_samples 5 \
    --output_folder outputs/vbench/vbench_14b_500 \
    "$@"
