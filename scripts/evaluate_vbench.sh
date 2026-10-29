#!/usr/bin/env bash
# Usage (Self-Forcing root, on the pod): bash scripts/evaluate_vbench.sh <videos_dir> [output_dir]
# Copied from ../VBench/scripts/evaluate_self_forcing.sh; runs VBench's evaluate.py from the sibling VBench repo.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root=/apdcephfs_tj5/share_303570626/xiyuzhang/code/VBench
videos_path="$(realpath -- "${1:?用法: bash scripts/evaluate_vbench.sh <videos_dir> [output_dir]}")"
output_path="$(realpath -m -- "${2:-${videos_path}_eval}")"

cd -- "$repo_root"
export PYTHONUNBUFFERED=1

dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

python - "$videos_path" "${dimensions[@]}" <<'PY'
import json
import sys
from pathlib import Path

video_dir = Path(sys.argv[1])
if not video_dir.is_dir():
    raise SystemExit(f"视频目录不存在: {video_dir}")

full_info = json.loads(Path("vbench/VBench_full_info.json").read_text())
assert set(sys.argv[2:]) == set(json.loads(Path("dimension_to_folder.json").read_text()))
expected = {f"{row['prompt_en']}-{i}.mp4" for row in full_info for i in range(5)}
missing = sorted(name for name in expected if not (video_dir / name).is_file())
if missing:
    raise SystemExit(f"缺少 {len(missing)} 个标准样本，停止评估。示例: {missing[:5]}")

try:
    import cv2
except ImportError as error:
    raise SystemExit(f"vbench2 的 OpenCV 无法导入: {error}\n请先修复环境依赖，再运行脚本。") from error

print(f"标准样本检查通过: {len(expected)} 个视频；OpenCV: {cv2.__version__}")
PY

for dimension in "${dimensions[@]}"; do
    if compgen -G "$output_path/$dimension/*_eval_results.json" > /dev/null; then
        printf '\n跳过已完成: %s\n' "$dimension"
        continue
    fi
    printf '\n开始评估: %s\n' "$dimension"
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" "$repo_root/evaluate.py" \
        --videos_path "$videos_path" \
        --full_json_dir "$repo_root/vbench/VBench_full_info.json" \
        --mode vbench_standard \
        --dimension "$dimension" \
        --output_path "$output_path/$dimension"
done

printf '\n全部评估完成，结果目录: %s\n' "$output_path"
python "$script_dir/cal_final_score.py" --file "$output_path" | tee "$output_path/final_score.txt"
