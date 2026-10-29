"""Generate native (bidirectional) Wan2.1 T2V videos for selected VBench dimensions.

Files are named "<short prompt>-<k>.mp4" (VBench standard mode); the line-aligned extended
prompt is what gets fed, matching inference.py. Launch: torchrun --nproc_per_node=8 this.py ...
"""
import argparse
import json
import os

import imageio.v2 as imageio
import torch

import wan
from wan.configs import WAN_CONFIGS

QUALITY_DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
                "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]

p = argparse.ArgumentParser()
p.add_argument("--task", default="t2v-1.3B", choices=["t2v-1.3B", "t2v-14B"])
p.add_argument("--ckpt_dir", default=None, help="defaults to wan_models/Wan2.1-T2V-{1.3B,14B}")
p.add_argument("--dimensions", nargs="+", default=QUALITY_DIMS)
p.add_argument("--info_path", default="prompts/VBench_full_info.json")
p.add_argument("--prompt_path", default="prompts/vbench/all_dimension.txt")
p.add_argument("--extended_prompt_path", default="prompts/vbench/all_dimension_extended.txt")
p.add_argument("--output_folder", required=True)
p.add_argument("--num_samples", type=int, default=5)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guide_scale", type=float, default=5.0)
p.add_argument("--shift", type=float, default=5.0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# No process group on purpose: WanT2V.generate barriers when dist is initialized, which would
# deadlock ranks that get uneven prompt counts.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank, world = int(os.environ.get("RANK", local_rank)), int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)

with open(args.info_path) as f:
    wanted = {e["prompt_en"] for e in json.load(f) if set(e["dimension"]) & set(args.dimensions)}
with open(args.prompt_path) as f1, open(args.extended_prompt_path) as f2:
    extended = {}
    for s, e in zip(f1.read().splitlines(), f2.read().splitlines()):
        extended.setdefault(s, e)
missing = wanted - extended.keys()
assert not missing, f"{len(missing)} VBench prompts missing from {args.prompt_path}"
prompts = sorted(wanted)
if rank == 0:
    print(f"{len(prompts)} prompts x {args.num_samples} samples for {args.dimensions}", flush=True)
    os.makedirs(args.output_folder, exist_ok=True)

ckpt_dir = args.ckpt_dir or f"wan_models/Wan2.1-T2V-{args.task.split('-')[1]}"
model = wan.WanT2V(config=WAN_CONFIGS[args.task], checkpoint_dir=ckpt_dir, device_id=local_rank, rank=0)

for i in range(rank, len(prompts), world):
    pr = prompts[i]
    for k in range(args.num_samples):
        out = os.path.join(args.output_folder, f"{pr}-{k}.mp4")
        if os.path.exists(out):
            continue
        video = model.generate(extended[pr], size=(832, 480), frame_num=81, shift=args.shift,
                               sampling_steps=args.steps, guide_scale=args.guide_scale,
                               seed=args.seed + i * 100 + k, offload_model=False)
        frames = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        imageio.mimwrite(out + ".tmp.mp4", frames, fps=16, codec="libx264", quality=None, macro_block_size=1)
        os.replace(out + ".tmp.mp4", out)
    print(f"[rank {rank}] {i + 1}/{len(prompts)} {pr}", flush=True)
