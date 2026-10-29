"""Small paired AR rollout comparing global and locally matched timesteps.

Only the source denoising call changes. Later calls reuse the original AR
schedule, transition noise, and clean-context cache update implementation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.ar_patch_rollout import (
    TracedRollout, check_identity, save_trace, save_video, source_state,
)
from scripts.patch.patch_noise_importance import atomic_json, load_pipeline, make_noise_bank, tensor_hash
from scripts.patch.patch_noise_strength import decode_pixels, pixel_metrics, scale_name
from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise
from scripts.patch.patch_timestep_utils import predict_with_patch_timestep


class PatchTimestepRollout(TracedRollout):
    """Override one prediction, keeping PairedRollout's recurrence intact."""

    def __init__(self, *args, **kwargs):
        self.intervention = None
        super().__init__(*args, **kwargs)

    def predict(self, noisy, chunk, step=None, context=False):
        spec = self.intervention
        if context or spec is None or (chunk, step) != (0, spec["step"]):
            return super().predict(noisy, chunk, step, context)
        matched = spec["mode"] == "matched"
        prediction = predict_with_patch_timestep(
            self.p, noisy, self.conditional, self.timestep(chunk, self.steps[step]),
            region=spec["region"] if matched else None,
            source_frame=spec["source_frame"],
            source_sigma=spec["source_sigma"] if matched else None,
            current_start=self.starts[chunk] * self.p.frame_seq_length,
        )
        if self.capture:
            index = (self.p.scheduler.timesteps - float(self.steps[step])).abs().argmin()
            self.trace.append({
                "chunk_index": chunk, "step_index": step,
                "global_t": float(self.steps[step]),
                "sigma": float(self.p.scheduler.sigmas[index]),
                "start_frame": self.starts[chunk], "frame_count": self.chunks[chunk],
                "mode": spec["mode"], "source_sigma": spec["source_sigma"],
                "source_frame": spec["source_frame"], "region": spec["region"],
                "noisy": noisy.detach().to("cpu", copy=True),
                "clean_prediction": prediction.detach().to("cpu", copy=True),
            })
        return prediction

    def probe(self, noisy, step, mode, region, source_frame, source_sigma):
        if self.intervention is not None:
            raise RuntimeError("Nested timestep intervention")
        if mode not in ("global", "token_global", "matched"):
            raise ValueError(f"Unknown mode: {mode}")
        if mode != "global":
            self.intervention = dict(step=step, mode=mode, region=region,
                                     source_frame=source_frame, source_sigma=source_sigma)
        try:
            return self.branch(noisy, 0, step, complete=True)
        finally:
            self.intervention = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ar_patch_timestep_probe_20261022"))
    parser.add_argument("--checkpoint-path", type=Path,
                        default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    parser.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    parser.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    parser.add_argument("--reference-dir", type=Path, default=Path("outputs/ar_patch_rollout_20261022/pilot"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--source-id", type=int, default=95)
    parser.add_argument("--source-frame", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--grid-rows", type=int, default=12)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument("--sigma-scales", type=float, nargs="+", default=[0.5, 0.0])
    args = parser.parse_args(argv)
    if (not args.sigma_scales or len(set(args.sigma_scales)) != len(args.sigma_scales)
            or any(not math.isfinite(s) or not 0 <= s < 1 for s in args.sigma_scales)):
        parser.error("sigma-scales must be distinct values in [0,1)")
    if min(args.prompt_index, args.seed, args.source_id) < 0 or min(args.grid_rows, args.grid_cols) < 1:
        parser.error("Invalid prompt, seed, source, or grid")
    args.weights, args.num_output_frames, args.source_chunk, args.halo = "generator", 6, 0, 2
    args.steps = [args.step]
    return args


def differences(a, b):
    delta = a.float() - b.float()
    return {"max_abs": float(delta.abs().max()), "mse": float(delta.square().mean())}


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    pipeline, config, shape, chunks, checkpoint = load_pipeline(args)
    if chunks != [3, 3]:
        raise ValueError("Expected two three-frame AR chunks")
    regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
    if args.source_id >= len(regions):
        raise ValueError("source-id is outside the grid")
    region = regions[args.source_id]
    prompt = args.data_path.read_text().splitlines()[args.prompt_index].strip()
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                          args.seed, parameter.dtype, parameter.device)
    replay = PatchTimestepRollout(pipeline, conditional, initial, transition, chunks)
    print("BASELINE", flush=True)
    baseline_output = replay.baseline(0, args.step)["output"]
    baseline_trace = replay.take_trace()
    state = source_state(baseline_trace, transition, 0, args.step, parameter.device)
    base_sigma = next(e["sigma"] for e in baseline_trace
                      if (e["chunk_index"], e["step_index"]) == (0, args.step))
    global_t = float(replay.steps[args.step])
    identity = check_identity(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
    print("IDENTITY exact", flush=True)

    # Check that expanding frame timesteps to all tokens is numerically neutral.
    pred, output = replay.probe(state["noisy"], args.step, "token_global", region,
                                args.source_frame, base_sigma)
    replay.take_trace()
    uniform = {"one_step": differences(pred, state["prediction"]),
               "rollout": differences(output, baseline_output)}
    print("UNIFORM_TOKEN_CONTROL " + json.dumps(uniform), flush=True)
    modes = ["global", "matched"]
    if uniform["one_step"]["max_abs"] or uniform["rollout"]["max_abs"]:
        # Preserve a same-layout control if bf16 kernels differ with batch size.
        modes.insert(1, "token_global")

    files = [Path(__file__), Path("scripts/patch/patch_timestep_utils.py"),
             Path("scripts/patch/patch_noise_importance.py"), Path("scripts/patch/ar_patch_rollout.py"),
             Path("utils/wan_wrapper.py"), Path("wan/modules/causal_model.py")]
    protocol = {
        "prompt": prompt, "prompt_index": args.prompt_index, "seed": args.seed,
        "checkpoint": str(checkpoint.resolve()), "weights": args.weights,
        "config": OmegaConf.to_container(config, resolve=True),
        "source_id": args.source_id, "source_chunk": 0, "source_frame": args.source_frame,
        "step": args.step, "region": region, "global_t": global_t, "base_sigma": base_sigma,
        "sigma_scales": args.sigma_scales, "modes": modes, "shape": shape, "chunks": chunks,
        "baseline_sha256": tensor_hash(baseline_output), "initial_noise_sha256": tensor_hash(initial),
        "transition_noise_sha256": {f"{c}:{s}": tensor_hash(n) for (c, s), n in transition.items()},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "code_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "matched_policy": "Only source call: token timestep=sigma*1000; x0=xt-sigma_map*flow. Later calls unchanged.",
        "clean_source": "Previous model prediction, not ground truth",
    }
    reference_path = args.reference_dir / "baseline/final_latent.pt"
    if reference_path.exists():
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        protocol["old_baseline_comparison"] = differences(baseline_output.cpu(), reference)
    baseline_pixels = decode_pixels(pipeline, baseline_output)
    baseline_dir = args.output_dir / "baseline"
    baseline_dir.mkdir()
    save_video(baseline_pixels, baseline_dir)
    save_trace(pipeline, baseline_trace, baseline_trace, baseline_dir, "latent")
    torch.save(baseline_output.cpu(), baseline_dir / "final_latent.pt")
    summary = dict(protocol=protocol, identity=identity, uniform_token_control=uniform, records=[])
    atomic_json(args.output_dir / "summary.json", summary)

    for scale in args.sigma_scales:
        sigma = base_sigma * scale
        changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                      sigma, region, args.source_frame)
        delta = changed.float() - state["noisy"].float()
        for mode in modes:
            branch_started = time.monotonic()
            print(f"BRANCH scale={scale:g} mode={mode} local_t={sigma * pipeline.scheduler.num_train_timesteps:g}", flush=True)
            prediction, output = replay.probe(changed, args.step, mode, region, args.source_frame, sigma)
            trace = replay.take_trace()
            one_step = measure_influence(state["prediction"], prediction, delta,
                                         region, args.source_frame, regions, args.halo)
            rollout = measure_influence(baseline_output, output, delta,
                                        region, args.source_frame, regions, args.halo)
            rollout["next_chunk_mse"] = float((output[:, chunks[0]:].float()
                                                - baseline_output[:, chunks[0]:].float()).square().mean())
            pixels = decode_pixels(pipeline, output)
            destination = args.output_dir / f"scale_{scale_name(scale)}" / mode
            destination.mkdir(parents=True)
            save_video(pixels, destination)
            save_trace(pipeline, trace, baseline_trace, destination, "latent")
            torch.save(output.cpu(), destination / "final_latent.pt")
            record = dict(mode=mode, sigma_scale=scale, source_sigma=sigma,
                          local_timestep=sigma * pipeline.scheduler.num_train_timesteps if mode == "matched" else global_t,
                          directory=str(destination.relative_to(args.output_dir)),
                          input_sha256=tensor_hash(changed), output_sha256=tensor_hash(output),
                          one_step=one_step, rollout=rollout, pixel=pixel_metrics(baseline_pixels, pixels),
                          elapsed_seconds=time.monotonic() - branch_started)
            if mode == "global":
                old = args.reference_dir / f"step_{args.step}" / f"source_{args.source_id:03d}" / f"scale_{scale_name(scale)}" / "final_latent.pt"
                if old.exists():
                    record["old_global_comparison"] = differences(output.cpu(), torch.load(old, map_location="cpu", weights_only=True))
            atomic_json(destination / "record.json", record)
            summary["records"].append(record)
            atomic_json(args.output_dir / "summary.json", summary)
            print(f"RESULT scale={scale:g} mode={mode} pixel_mse={record['pixel']['all_video_mse']:.9g} "
                  f"next_chunk_mse={rollout['next_chunk_mse']:.9g}", flush=True)
    summary["post_identity"] = check_identity(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["complete"] = True
    atomic_json(args.output_dir / "summary.json", summary)
    print(f"DONE {args.output_dir} seconds={summary['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
