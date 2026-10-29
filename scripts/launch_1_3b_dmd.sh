#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

logdir=${LOGDIR:-logs/self_forcing_dmd_1_3b}
if compgen -G "$logdir/checkpoints/step_*.pt" > /dev/null; then
    echo "已有 checkpoint：$logdir；请指定新的 LOGDIR，避免覆盖（训练不支持断点续跑）" >&2
    exit 1
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir "$logdir" --no_visualize
