"""Paired local-noise interventions on a frozen causal Self-Forcing generator.

The global conditioning timestep is unchanged, as in Mask Forcing. This is
a sensitivity experiment, not a reconstruction-quality or training benchmark.
Run from the repository root: python -m scripts.patch.patch_noise_importance --help
"""

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import torch

from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--weights", choices=("generator", "generator_ema"), default="generator")
    p.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    p.add_argument("--prompt-indices", type=int, nargs="+", default=[158])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2], help="Zero-based denoising indices; exclude initial pure noise")
    p.add_argument("--source-chunk", type=int, default=0)
    p.add_argument("--source-frame", type=int, default=1, help="Frame within the source chunk")
    p.add_argument("--grid-rows", type=int, default=4)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--sigma-scales", type=float, nargs="+", default=[0.9, 0.75])
    p.add_argument("--num-output-frames", type=int, default=6, help="Latent frames; 6 decodes to 21 video frames")
    p.add_argument("--rollout-top-k", type=int, default=2)
    p.add_argument("--rollout-random-k", type=int, default=2)
    p.add_argument("--ranking-metric", choices=("normalized_outside_energy", "normalized_other_frames_energy"),
                   default="normalized_outside_energy", help="One-step metric used to select top rollout sources")
    p.add_argument("--log-every", type=int, default=1, help="Print progress every N source interventions")
    p.add_argument("--halo", type=int, default=2, help="Latent-pixel exclusion margin")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--save-previews", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error("Require 0 <= shard-index < num-shards")
    if args.log_every < 1:
        p.error("log-every must be >=1")
    if not args.sigma_scales or any(not math.isfinite(s) or not 0 < s < 1 for s in args.sigma_scales):
        p.error("sigma-scales must lie strictly between 0 and 1")
    if any(v < 0 for v in (args.source_chunk, args.source_frame, args.rollout_top_k, args.rollout_random_k, args.halo)):
        p.error("Chunk/frame, rollout counts and halo must be nonnegative")
    if any(s < 1 for s in args.steps) or any(i < 0 for i in args.prompt_indices):
        p.error("steps must be >=1; prompt indices must be >=0")
    for name in ("prompt_indices", "seeds", "steps", "sigma_scales"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            p.error(f"{name} must not contain duplicates")
    return args


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def make_noise_bank(shape, chunks, steps, seed, dtype, device):
    """Materialize all randomness before branching, identically on every worker."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(shape, generator=rng, dtype=torch.float32).to(device=device, dtype=dtype)
    transition = {}
    for chunk, size in enumerate(chunks):
        for step in range(1, steps):
            transition[chunk, step] = torch.randn(
                (shape[0], size, *shape[2:]), generator=rng, dtype=torch.float32
            ).to(device=device, dtype=dtype)
    return initial, transition


class PairedRollout:
    """Original few-step recurrence with an explicit, reusable noise bank.

    Full attention is required: prefix KV is immutable, and each branch overwrites
    all current/future keys before reading them. Resetting indices to the source
    chunk boundary therefore isolates branches without copying a 6 GB cache.
    """

    def __init__(self, pipeline, conditional, initial, transition, chunks):
        self.p = pipeline
        self.conditional = conditional
        self.initial = initial
        self.transition = transition
        self.chunks = chunks
        self.starts = [sum(chunks[:i]) for i in range(len(chunks))]
        self.steps = pipeline.denoising_step_list
        if pipeline.local_attn_size != -1:
            raise ValueError("Branch index reset requires full attention (local_attn_size=-1)")
        if initial.shape[1] * pipeline.frame_seq_length > 32760:
            raise ValueError("Requested video exceeds the pipeline KV capacity")
        if pipeline.kv_cache1 is None:
            pipeline._initialize_kv_cache(initial.shape[0], initial.dtype, initial.device)
            pipeline._initialize_crossattn_cache(initial.shape[0], initial.dtype, initial.device)
        self.reset_to_chunk(0)
        for cache in pipeline.crossattn_cache:
            cache["is_init"] = False

    def reset_to_chunk(self, chunk):
        token_start = self.starts[chunk] * self.p.frame_seq_length
        for cache in self.p.kv_cache1:
            cache["global_end_index"].fill_(token_start)
            cache["local_end_index"].fill_(token_start)

    def timestep(self, chunk, value):
        # Multiplication preserves the float warped timestep used by inference.py.
        return torch.ones((self.initial.shape[0], self.chunks[chunk]),
                          device=self.initial.device, dtype=torch.int64) * value

    def predict(self, noisy, chunk, step=None, context=False):
        value = self.p.args.context_noise if context else self.steps[step]
        return self.p.generator(
            noisy_image_or_video=noisy, conditional_dict=self.conditional,
            timestep=self.timestep(chunk, value), kv_cache=self.p.kv_cache1,
            crossattn_cache=self.p.crossattn_cache,
            current_start=self.starts[chunk] * self.p.frame_seq_length,
        )[1]

    def renoise(self, clean, chunk, step):
        return self.p.scheduler.add_noise(
            clean.flatten(0, 1), self.transition[chunk, step].flatten(0, 1),
            self.timestep(chunk, self.steps[step]).flatten(),
        ).unflatten(0, clean.shape[:2])

    def baseline(self, source_chunk, source_step):
        self.reset_to_chunk(0)
        completed = []
        state = None
        for chunk, size in enumerate(self.chunks):
            start = self.starts[chunk]
            noisy = self.initial[:, start:start + size]
            previous = None
            for step in range(len(self.steps)):
                if step:
                    noisy = self.renoise(previous, chunk, step)
                if (chunk, step) == (source_chunk, source_step):
                    state = {"noisy": noisy.clone(), "clean": previous.clone(),
                             "noise": self.transition[chunk, step]}
                previous = self.predict(noisy, chunk, step)
                if (chunk, step) == (source_chunk, source_step):
                    state["prediction"] = previous.clone()
            completed.append(previous)
            self.predict(previous, chunk, context=True)
        if state is None:
            raise ValueError("Source intervention is outside the rollout")
        state["output"] = torch.cat(completed[source_chunk:], dim=1)
        return state

    def branch(self, noisy, source_chunk, source_step, complete=False):
        self.reset_to_chunk(source_chunk)
        prediction = self.predict(noisy, source_chunk, source_step)
        first = prediction.clone()
        if not complete:
            return first, None
        for step in range(source_step + 1, len(self.steps)):
            prediction = self.predict(self.renoise(prediction, source_chunk, step), source_chunk, step)
        completed = [prediction]
        self.predict(prediction, source_chunk, context=True)
        for chunk in range(source_chunk + 1, len(self.chunks)):
            start = self.starts[chunk]
            noisy = self.initial[:, start:start + self.chunks[chunk]]
            for step in range(len(self.steps)):
                if step:
                    noisy = self.renoise(prediction, chunk, step)
                prediction = self.predict(noisy, chunk, step)
            completed.append(prediction)
            self.predict(prediction, chunk, context=True)
        return first, torch.cat(completed, dim=1)


def load_pipeline(args):
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.mixed_precision else torch.float32
    print("Loading frozen generator and text encoder", flush=True)
    pipeline = CausalInferencePipeline(config, device=device, vae=torch.nn.Identity())
    checkpoint_path = args.checkpoint_path / "model.pt" if args.checkpoint_path.is_dir() else args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if args.weights not in checkpoint:
        raise ValueError(f"Checkpoint has no {args.weights}; available keys: {list(checkpoint)}")
    pipeline.generator.load_state_dict(checkpoint[args.weights], strict=True)
    del checkpoint
    gc.collect()
    pipeline.generator.model.independent_first_frame = config.independent_first_frame
    pipeline.to(dtype=dtype).eval().requires_grad_(False)
    pipeline.generator.to(device)
    patch = tuple(pipeline.generator.model.patch_size)
    if patch != (1, 2, 2) or len(pipeline.generator.model.blocks) != 30:
        raise ValueError("This probe follows the existing Wan 1.3B pipeline: patch (1,2,2), 30 layers")
    shape = list(config.image_or_video_shape)
    if shape[0] != 1 or shape[2:] != [16, 60, 104]:
        raise ValueError("Expected single-sample Wan latents [1,F,16,60,104]")
    shape[1] = args.num_output_frames
    block = config.num_frame_per_block
    first = int(config.independent_first_frame)
    if shape[1] < 1 or shape[1] > 21 or (shape[1] - first) % block:
        raise ValueError("Output frames must fit the model's chunk structure and be <=21")
    chunks = ([1] if first else []) + [block] * ((shape[1] - first) // block)
    if args.source_chunk >= len(chunks) or args.source_frame >= chunks[args.source_chunk]:
        raise ValueError("Source chunk/frame out of range")
    if (getattr(args, "ranking_metric", "normalized_outside_energy") == "normalized_other_frames_energy"
            and chunks[args.source_chunk] < 2):
        raise ValueError("other-frames ranking requires a source chunk containing at least two frames")
    if max(args.steps) >= len(pipeline.denoising_step_list):
        raise ValueError("Denoising step index out of range")
    return pipeline, config, shape, chunks, checkpoint_path


def save_preview(pipeline, latent, destination, source_frame=1):
    from utils.wan_wrapper import WanVAEWrapper
    import imageio.v2 as imageio

    if isinstance(pipeline.vae, torch.nn.Identity):
        pipeline.vae = WanVAEWrapper().to(device=latent.device, dtype=latent.dtype).eval()
    pixels = (pipeline.vae.decode_to_pixel(latent, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
    video = pixels[0].permute(0, 2, 3, 1).mul(255).to(torch.uint8).cpu().numpy()
    imageio.mimwrite(str(destination.with_suffix(".mp4")), video, fps=16,
                    codec="libx264", quality=7, macro_block_size=1)
    frame = min(source_frame * 4, len(video) - 1)
    imageio.imwrite(str(destination.with_suffix(".png")), video[frame])
    pipeline.vae.model.clear_cache()


@torch.inference_mode()
def run_case(args, pipeline, config, shape, chunks, checkpoint_path, prompt_index, prompt, seed, step):
    case_dir = args.output_dir / f"prompt_{prompt_index:06d}_seed_{seed:06d}_chunk_{args.source_chunk}_step_{step}"
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
        "source_chunk": args.source_chunk, "source_frame": args.source_frame, "step_index": step,
        "grid_rows": args.grid_rows, "grid_cols": args.grid_cols, "sigma_scales": args.sigma_scales,
        "checkpoint_path": str(checkpoint_path.resolve()), "weights": args.weights,
        "checkpoint_size": checkpoint_path.stat().st_size, "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "num_output_frames": args.num_output_frames, "halo": args.halo,
        "rollout_top_k": args.rollout_top_k, "rollout_random_k": args.rollout_random_k,
        "ranking_metric": args.ranking_metric,
        "config": __import__("omegaconf").OmegaConf.to_container(config, resolve=True),
        "mechanism": "Self-Forcing checkpoint; local actual sigma lowered; global timestep unchanged",
        "clean_source": "previous denoising prediction; no ground-truth information",
    }
    result_path = case_dir / "metrics.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if not args.resume:
            raise FileExistsError(f"{result_path} exists; use a new output directory or --resume")
        defaults = {"ranking_metric": "normalized_outside_energy"}
        if any(existing["metadata"].get(k, defaults.get(k)) != v for k, v in metadata.items()):
            raise ValueError(f"Resume configuration differs for {case_dir}")
        print(f"SKIP {case_dir.name}", flush=True)
        return
    print(f"START {case_dir.name}: {prompt}", flush=True)
    started = time.monotonic()
    device = next(pipeline.generator.parameters()).device
    dtype = next(pipeline.generator.parameters()).dtype
    pipeline.text_encoder.to(device)
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list), seed, dtype, device)
    replay = PairedRollout(pipeline, conditional, initial, transition, chunks)
    state = replay.baseline(args.source_chunk, step)
    global_t = float(replay.steps[step])
    sigma_index = (pipeline.scheduler.timesteps - global_t).abs().argmin()
    sigma = float(pipeline.scheduler.sigmas[sigma_index])
    regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
    metadata.update(global_t=global_t, base_sigma=sigma,
                    schedule=[float(v) for v in replay.steps], torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(), host=platform.node(),
                    baseline_sha256=tensor_hash(state["output"]), initial_noise_sha256=tensor_hash(initial))
    # Exercise the exact intervention path at sigma'=sigma, not just a repeated forward.
    identity_input = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                         sigma, regions[0], args.source_frame)
    if not torch.equal(identity_input, state["noisy"]):
        raise RuntimeError("Identity intervention changed input (dtype/sigma mismatch)")
    same_step, same_rollout = replay.branch(identity_input, args.source_chunk, step, complete=True)
    identity = {"one_step_max_abs": float((same_step.float() - state["prediction"].float()).abs().max()),
                "rollout_max_abs": float((same_rollout.float() - state["output"].float()).abs().max())}
    if max(identity.values()) > 1e-5:
        raise RuntimeError(f"Paired replay failed identity control: {identity}")
    print(f"IDENTITY {identity}; t={global_t:.6f}, sigma={sigma:.6f}", flush=True)
    records, difference_maps = [], {}
    total_interventions = len(args.sigma_scales) * len(regions)
    for intervention_index, (scale, region) in enumerate(itertools.product(args.sigma_scales, regions), start=1):
        changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                      sigma * scale, region, args.source_frame)
        prediction, _ = replay.branch(changed, args.source_chunk, step)
        input_delta = changed.float() - state["noisy"].float()
        metrics = measure_influence(state["prediction"], prediction, input_delta, region,
                                    args.source_frame, regions, halo=args.halo)
        records.append({"source_id": region["id"], "sigma_scale": scale, "source_sigma": sigma * scale,
                        "one_step": metrics, "rollout": None})
        difference_maps[f"scale_{scale}_source_{region['id']}_one_step"] = (
            prediction.float() - state["prediction"].float()).square().mean(dim=(0, 2)).cpu()
        if intervention_index % args.log_every == 0 or intervention_index == total_interventions:
            print(f"PATCH {intervention_index}/{total_interventions} scale={scale} source={region['id']} "
                  f"outside={metrics['outside_source_mse']:.6g} "
                  f"{args.ranking_metric}={metrics[args.ranking_metric]:.6g}", flush=True)
    # Select on one-step non-source sensitivity, and keep deterministic random controls.
    scale = min(args.sigma_scales)
    ranked = sorted((r for r in records if r["sigma_scale"] == scale),
                    key=lambda r: r["one_step"][args.ranking_metric], reverse=True)
    top = [r["source_id"] for r in ranked[:args.rollout_top_k]]
    rest = [r["source_id"] for r in ranked if r["source_id"] not in top]
    controls = random.Random(seed + 191).sample(rest, min(args.rollout_random_k, len(rest)))
    selected = set(top + controls)
    torch.save({"baseline": state["output"].cpu(), "one_step_baseline": state["prediction"].cpu(),
                "previous_prediction": state["clean"].cpu(), "selected_sources": sorted(selected)},
               case_dir / "baseline.pt")
    if args.save_previews:
        save_preview(pipeline, state["output"], case_dir / "baseline", args.source_frame)
    for record in records:
        if record["sigma_scale"] != scale or record["source_id"] not in selected:
            continue
        region = regions[record["source_id"]]
        changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                      record["source_sigma"], region, args.source_frame)
        _, output = replay.branch(changed, args.source_chunk, step, complete=True)
        record["selection"] = "top" if record["source_id"] in top else "random_control"
        record["rollout"] = measure_influence(state["output"], output,
            changed.float() - state["noisy"].float(), region, args.source_frame, regions, halo=args.halo)
        record["rollout"]["next_chunk_mse"] = float((
            output[:, chunks[args.source_chunk]:].float() - state["output"][:, chunks[args.source_chunk]:].float()
        ).square().mean()) if len(chunks) > args.source_chunk + 1 else None
        difference_maps[f"scale_{scale}_source_{region['id']}_rollout"] = (
            output.float() - state["output"].float()).square().mean(dim=(0, 2)).cpu()
        torch.save(output.cpu(), case_dir / f"source_{region['id']:03d}_rollout.pt")
        if args.save_previews:
            save_preview(pipeline, output, case_dir / f"source_{region['id']:03d}", args.source_frame)
        print(f"ROLLOUT source={region['id']} selection={record['selection']} "
              f"next_chunk_mse={record['rollout']['next_chunk_mse']}", flush=True)
    # Verify reset still works after every perturbation has modified the cache.
    end_step, end_rollout = replay.branch(state["noisy"], args.source_chunk, step, complete=True)
    identity["post_sweep_one_step_max_abs"] = float((end_step.float() - state["prediction"].float()).abs().max())
    identity["post_sweep_rollout_max_abs"] = float((end_rollout.float() - state["output"].float()).abs().max())
    if max(identity.values()) > 1e-5:
        raise RuntimeError(f"Post-sweep cache contamination: {identity}")
    torch.save(difference_maps, case_dir / "difference_maps.pt")
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__), Path("scripts/patch/patch_noise_utils.py"), Path("pipeline/causal_inference.py"),
         Path("utils/wan_wrapper.py"), Path("wan/modules/causal_model.py")]}
    atomic_json(result_path, {"schema_version": 1, "metadata": metadata, "regions": regions,
                             "identity": identity, "records": records})
    print(f"DONE {case_dir} ({metadata['elapsed_seconds']:.1f}s)", flush=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment in the GPU machine's dev environment")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    prompts = args.data_path.read_text().splitlines()
    if max(args.prompt_indices) >= len(prompts) or any(not prompts[i].strip() for i in args.prompt_indices):
        raise ValueError("Invalid or empty selected prompt")
    jobs = list(itertools.product(args.prompt_indices, args.seeds, args.steps))[args.shard_index::args.num_shards]
    if not jobs:
        print("No cases assigned to this shard", flush=True)
        return
    pipeline, config, shape, chunks, checkpoint = load_pipeline(args)
    for prompt_index, seed, step in jobs:
        run_case(args, pipeline, config, shape, chunks, checkpoint, prompt_index, prompts[prompt_index], seed, step)


if __name__ == "__main__":
    main()
