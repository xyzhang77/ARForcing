#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

caption_path=${CAPTION_PATH:-prompts/vidprom_filtered_extended.txt}
output_folder=${OUTPUT_FOLDER:-outputs/ode_pairs}

if [[ ! -f "$caption_path" ]]; then
    echo "提示词文件不存在：$caption_path；请设置 CAPTION_PATH（每行一条提示词）" >&2
    exit 1
fi
if compgen -G "$output_folder/*.pt" > /dev/null; then
    echo "已有 ODE pairs：$output_folder；请指定新的 OUTPUT_FOLDER，避免覆盖（不支持断点续跑）" >&2
    exit 1
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --module scripts.data.generate_ode_pairs \
    --caption_path "$caption_path" \
    --output_folder "$output_folder" \
    --guidance_scale "${GUIDANCE_SCALE:-6.0}"
