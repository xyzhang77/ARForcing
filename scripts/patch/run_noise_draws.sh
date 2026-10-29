#!/bin/bash
# Usage (repo root, on a pod): bash scripts/patch/run_noise_draws.sh <prompt> [<prompt> ...]
# Draws 1-4: baseline + all 192 patches at scale 0. Draws 5-16: baseline only.
# Jobs are spread round-robin over the 8 local GPUs.
set -u
PY=/apdcephfs_tj5/share_303570626/xiyuzhang/conda_env/dev/bin/python
# FULL="0 1 2 3 4" also scans draw 0 (original noise; writes attention.npz) into $O/draw_0.
O=${O:-outputs/patch_importance/ar_noise_draws_20261026}
FULL=${FULL:-"1 2 3 4"}
mkdir -p $O/logs
jobs=()
for p in "$@"; do for s in 1 2; do for d in $FULL; do jobs+=("$p $s $d full"); done; done; done
for p in "$@"; do for s in 1 2; do for d in $(seq 5 16); do jobs+=("$p $s $d base"); done; done; done
for g in 0 1 2 3 4 5 6 7; do
  (
    for ((i = g; i < ${#jobs[@]}; i += 8)); do
      read -r p s d kind <<< "${jobs[$i]}"
      extra=""; [ "$kind" = base ] && extra="--baseline-only"
      log=$O/logs/p${p}_s${s}_d${d}.log
      CUDA_VISIBLE_DEVICES=$g $PY -u scripts/patch/ar_patch_attention.py --output-dir $O/draw_$d --prompt-index $p \
        --step $s --noise-draw $d --sigma-scales 0 $extra > $log 2>&1 || echo "FAILED $log" >> $O/logs/failures.txt
    done
  ) &
done
wait
echo "done $(date)" > $O/logs/host_$1.done
