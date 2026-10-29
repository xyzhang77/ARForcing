#!/bin/bash
# Usage (repo root, on each pod): bash scripts/vbench/run_wan_vbench.sh <node_rank> <num_nodes> [gen_wan_vbench.py args...]
set -u
NODE=$1 NNODES=$2; shift 2
PY=/apdcephfs_tj5/share_303570626/xiyuzhang/conda_env/dev/bin/python
O=outputs/vbench/vbench_wan_14b
mkdir -p $O/logs
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$((NODE * 8 + g)) WORLD_SIZE=$((NNODES * 8)) PYTHONPATH=. \
    $PY -u scripts/vbench/gen_wan_vbench.py --task t2v-14B --output_folder $O "$@" \
    > $O/logs/rank$((NODE * 8 + g)).log 2>&1 || echo "FAILED rank$((NODE * 8 + g))" >> $O/logs/failures.txt &
done
wait
echo "done $(date)" > $O/logs/node$NODE.done
