"""Replay frozen fine-patch candidates at stronger local sigma reductions.

All pixel metrics precede uint8 quantization and video encoding. A zero local
sigma substitutes the previous denoising prediction, never ground-truth pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.patch_group_importance import validate_model_source
from scripts.patch.patch_noise_importance import (
    PairedRollout, atomic_json, load_pipeline, make_noise_bank, tensor_hash,
)
from scripts.patch.patch_noise_utils import measure_influence, replace_patch_noise


def frozen_candidates(data):
    """Use saved rollout membership, without re-ranking at any new strength."""
    records = [r for r in data["records"] if r.get("rollout") is not None
               and math.isclose(r["sigma_scale"], 0.75, rel_tol=0, abs_tol=1e-8)]
    if (len(records) != 4 or len({r["source_id"] for r in records}) != 4
            or sum(r.get("selection") == "top" for r in records) != 2
            or sum(r.get("selection") == "random_control" for r in records) != 2):
        raise ValueError("Expected exactly two saved top and two saved random .75 rollouts")
    regions = {r["id"]: r for r in data["regions"]}
    return [{"source_id": r["source_id"], "selection": r["selection"],
             "region": dict(regions[r["source_id"]])}
            for r in sorted(records, key=lambda r: r["source_id"])]


def pixel_metrics(baseline, variant):
    """Compare decoded float [0,1] tensors [T,H,W,3], before saving previews."""
    if baseline.shape != variant.shape or baseline.ndim != 4 or baseline.shape[-1] != 3:
        raise ValueError("Pixels must have equal [T,H,W,3] shapes")
    for value in (baseline, variant):
        if (not value.is_floating_point() or not torch.isfinite(value).all()
                or value.numel() == 0 or value.min() < 0 or value.max() > 1):
            raise ValueError("Pixel metrics require finite decoded floats in [0,1]")
    mse, mae = [], []
    for ref, changed in zip(baseline, variant):
        delta = changed.float() - ref.float()
        mse.append(float(delta.square().mean()))
        mae.append(float(delta.abs().mean()))
    result = {"all_video_mse": sum(mse) / len(mse),
              "all_video_mae": sum(mae) / len(mae),
              "per_frame_mse": mse, "per_frame_mae": mae,
              "per_frame_rmse": [math.sqrt(v) for v in mse],
              "pixel_domain": "decoded float [0,1], before uint8 quantization or video encoding"}
    result["all_video_rmse"] = math.sqrt(result["all_video_mse"])
    if len(mse) > 12:
        result.update(later_frames=list(range(12, len(mse))),
                      later_frames_mse=sum(mse[12:]) / len(mse[12:]),
                      later_frames_mae=sum(mae[12:]) / len(mae[12:]))
        result["later_frames_rmse"] = math.sqrt(result["later_frames_mse"])
    return result


def validate_source_code(metadata):
    hashes = metadata.get("code_sha256")
    if not hashes:
        raise ValueError("Original experiment is missing source code hashes")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original experiment code changed: {name}")


def scale_name(scale):
    return format(scale, ".8g").replace(".", "p")


@torch.inference_mode()
def decode_pixels(pipeline, latent):
    from utils.wan_wrapper import WanVAEWrapper
    if isinstance(pipeline.vae, torch.nn.Identity):
        pipeline.vae = WanVAEWrapper().to(device=latent.device, dtype=latent.dtype).eval()
    pipeline.vae.model.clear_cache()
    try:
        pixels = (pipeline.vae.decode_to_pixel(latent, use_cache=False).float() * 0.5 + 0.5).clamp(0, 1)
        return pixels[0].permute(0, 2, 3, 1).contiguous().cpu()
    finally:
        pipeline.vae.model.clear_cache()


def save_pixels(pixels, destination, write_video=True):
    video = pixels.mul(255).round().to(torch.uint8).numpy()
    npz = destination.with_name(destination.name + "_pixels.npz")
    # Unencoded uint8 samples allow synchronized reports without codec errors.
    np.savez(npz, pixels=video)
    artifacts = {"pixels_npz": npz.name}
    if write_video:
        import imageio.v2 as imageio
        mp4 = destination.with_suffix(".mp4")
        imageio.mimwrite(str(mp4), video, fps=16, codec="libx264", quality=7,
                        macro_block_size=1, ffmpeg_params=["-threads", "2"])
        artifacts["mp4"] = mp4.name
    return artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    parser.add_argument("--sigma-scales", type=float, nargs="+", default=[0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard configuration")
    if (len(set(args.sigma_scales)) != len(args.sigma_scales)
            or 0.75 not in args.sigma_scales
            or any(not math.isfinite(s) or not 0 <= s <= 0.75 for s in args.sigma_scales)):
        parser.error("Scales must be unique within [0,.75], and include .75 for exact replay")
    # Check the original strength before any stronger branch.
    args.sigma_scales = sorted(args.sigma_scales, reverse=True)
    return args


def load_cases(args):
    cases = []
    identities = set()
    for path in sorted(args.input_dir.rglob("metrics.json")):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        if metadata["grid_rows"] != 12 or metadata["grid_cols"] != 16:
            raise ValueError("This follow-up expects the original 12x16 fine grid")
        if metadata["source_chunk"] != 0:
            raise ValueError("Full-video decoding requires the original source chunk zero")
        identity = tuple(metadata[k] for k in ("prompt_index", "seed", "source_chunk", "step_index"))
        if identity in identities:
            raise ValueError("Duplicate source case identity")
        identities.add(identity)
        candidates = frozen_candidates(data)
        for candidate in candidates:
            rollout_path = path.parent / f"source_{candidate['source_id']:03d}_rollout.pt"
            if not rollout_path.is_file():
                raise FileNotFoundError(rollout_path)
        cases.append({"path": path, "data": data, "candidates": candidates})
    if not cases:
        raise ValueError("No fine sweep cases found")
    return cases


@torch.inference_mode()
def run_case(args, pipeline, shape, chunks, case):
    source = case["data"]["metadata"]
    case_dir = args.output_dir / case["path"].parent.name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "strength_metrics.json"
    protocol = {"source_metrics_path": str(case["path"].resolve()),
                "source_metrics_sha256": hashlib.sha256(case["path"].read_bytes()).hexdigest(),
                "sigma_scales": args.sigma_scales, "fixed_candidates": case["candidates"],
                "write_videos": not args.no_videos}
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"{result_path} exists; use --resume")
        existing = json.loads(result_path.read_text())
        if any(existing["metadata"].get(k) != v for k, v in protocol.items()):
            raise ValueError("Resume protocol differs")
        for item in [existing["metadata"]["baseline_artifacts"]] + [r["artifacts"] for r in existing["records"]]:
            if any(not (case_dir / name).is_file() for name in item.values()):
                raise FileNotFoundError("Resume artifacts missing")
        print(f"SKIP {case_dir.name}", flush=True)
        return
    print(f"START {case_dir.name}", flush=True)
    started = time.monotonic()
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[source["prompt"]])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                         source["seed"], parameter.dtype, parameter.device)
    replay = PairedRollout(pipeline, conditional, initial, transition, chunks)
    chunk, frame, step = source["source_chunk"], source["source_frame"], source["step_index"]
    state = replay.baseline(chunk, step)
    if (tensor_hash(state["output"]) != source["baseline_sha256"]
            or tensor_hash(initial) != source["initial_noise_sha256"]):
        raise RuntimeError("Baseline or initial-noise hash differs from fine sweep")
    original_baseline = torch.load(case["path"].parent / "baseline.pt", map_location="cpu", weights_only=False)
    if tensor_hash(original_baseline["baseline"]) != source["baseline_sha256"]:
        raise RuntimeError("Original saved baseline differs from metadata")
    if sorted(original_baseline["selected_sources"]) != sorted(c["source_id"] for c in case["candidates"]):
        raise RuntimeError("Original saved candidate membership differs")
    del original_baseline
    sigma_index = (pipeline.scheduler.timesteps - float(replay.steps[step])).abs().argmin()
    base_sigma = float(pipeline.scheduler.sigmas[sigma_index])
    if base_sigma != source["base_sigma"] or float(replay.steps[step]) != source["global_t"]:
        raise RuntimeError("Global conditioning or actual sigma differs")
    regions = case["data"]["regions"]
    unchanged = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                    base_sigma, regions[0], frame)
    if not torch.equal(unchanged, state["noisy"]):
        raise RuntimeError("Unchanged-sigma input must be exact")
    same_step, same_rollout = replay.branch(unchanged, chunk, step, complete=True)
    identity = {"one_step_max_abs": float((same_step.float() - state["prediction"].float()).abs().max()),
                "rollout_max_abs": float((same_rollout.float() - state["output"].float()).abs().max())}
    if any(v != 0 for v in identity.values()):
        raise RuntimeError(f"Nonzero identity: {identity}")
    del same_step, same_rollout, unchanged
    baseline_pixels = decode_pixels(pipeline, state["output"])
    baseline_artifacts = save_pixels(baseline_pixels, case_dir / "baseline", not args.no_videos)
    records = []
    for candidate in case["candidates"]:
        region, source_id = candidate["region"], candidate["source_id"]
        for scale in args.sigma_scales:
            changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                          base_sigma * scale, region, frame)
            prediction, output = replay.branch(changed, chunk, step, complete=True)
            input_delta = changed.float() - state["noisy"].float()
            one_step = measure_influence(state["prediction"], prediction, input_delta, region,
                                         frame, regions, halo=source["halo"])
            rollout = measure_influence(state["output"], output, input_delta, region,
                                        frame, regions, halo=source["halo"])
            next_squared = (output[:, chunks[chunk]:].float() - state["output"][:, chunks[chunk]:].float()).square()
            next_energy = float(next_squared.sum())
            rollout.update(next_chunk_mse=float(next_squared.mean()), next_chunk_energy=next_energy,
                           next_chunk_normalized_energy=next_energy / rollout["input_energy"])
            replay_check = None
            if scale == 0.75:
                original = torch.load(case["path"].parent / f"source_{source_id:03d}_rollout.pt",
                                      map_location="cpu", weights_only=False)
                current = output.cpu()
                replay_check = {"max_abs": float((current.float() - original.float()).abs().max()),
                                "original_sha256": tensor_hash(original), "current_sha256": tensor_hash(current)}
                if replay_check["max_abs"] != 0 or replay_check["original_sha256"] != replay_check["current_sha256"]:
                    raise RuntimeError(f"Original .75 rollout failed exact replay: {replay_check}")
                del original, current
            pixels = decode_pixels(pipeline, output)
            pixel_result = pixel_metrics(baseline_pixels, pixels)
            stem = f"source_{source_id:03d}_scale_{scale_name(scale)}"
            artifacts = save_pixels(pixels, case_dir / stem, not args.no_videos)
            if scale == 0:
                torch.save(output.cpu(), case_dir / f"{stem}_rollout.pt")
                artifacts["rollout_pt"] = f"{stem}_rollout.pt"
            records.append({**candidate, "sigma_scale": scale, "source_sigma": base_sigma * scale,
                            "num_tokens": (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4,
                            "one_step": one_step, "rollout": rollout, "pixels": pixel_result,
                            "artifacts": artifacts, "original_replay_check": replay_check})
            print(f"STRENGTH source={source_id} selection={candidate['selection']} scale={scale} "
                  f"next_chunk_mse={rollout['next_chunk_mse']:.7g} pixel_rmse={pixel_result['all_video_rmse']:.7g}", flush=True)
            del pixels, changed, prediction, output, input_delta, next_squared
    end_step, end_rollout = replay.branch(state["noisy"], chunk, step, complete=True)
    identity.update(post_sweep_one_step_max_abs=float((end_step.float() - state["prediction"].float()).abs().max()),
                    post_sweep_rollout_max_abs=float((end_rollout.float() - state["output"].float()).abs().max()))
    if any(v != 0 for v in identity.values()):
        raise RuntimeError(f"Post-sweep identity failed: {identity}")
    metadata = {**source, **protocol, "original_metadata": source,
                "baseline_artifacts": baseline_artifacts, "fps": 16,
                "decoded_shape": list(baseline_pixels.shape),
                "pixel_temporal_domain": "All decoded frames; later_frames denotes indices >=12, not an exact latent chunk mapping",
                "zero_endpoint": "Previous denoising prediction inside source patch only; no ground truth",
                "candidate_selection": "Frozen .75 fine-sweep top2 and random2; no new ranking",
                "host": platform.node(), "torch_version": torch.__version__,
                "elapsed_seconds": time.monotonic() - started,
                "strength_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "transition_noise_sha256": {f"{c}:{s}": tensor_hash(value) for (c, s), value in transition.items()}}
    atomic_json(result_path, {"schema_version": 1, "metadata": metadata, "regions": regions,
                             "identity": identity, "records": records})
    print(f"DONE {case_dir.name} ({metadata['elapsed_seconds']:.1f}s)", flush=True)


def main():
    from omegaconf import OmegaConf
    args = parse_args()
    cases = load_cases(args)[args.shard_index::args.num_shards]
    if not cases:
        print("No cases assigned", flush=True)
        return
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    first = cases[0]["data"]["metadata"]
    checkpoint = Path(first["checkpoint_path"])
    for case in cases:
        metadata = case["data"]["metadata"]
        validate_model_source(metadata, config, checkpoint)
        validate_source_code(metadata)
        if metadata["weights"] != first["weights"]:
            raise ValueError("Mixed checkpoint weights")
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the GPU machine's dev environment")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    model_args = SimpleNamespace(config_path=args.config_path, checkpoint_path=checkpoint,
                                 weights=first["weights"], num_output_frames=6, source_chunk=0,
                                 source_frame=max(c["data"]["metadata"]["source_frame"] for c in cases),
                                 steps=[c["data"]["metadata"]["step_index"] for c in cases])
    pipeline, loaded_config, shape, chunks, loaded_checkpoint = load_pipeline(model_args)
    for case in cases:
        validate_model_source(case["data"]["metadata"], loaded_config, loaded_checkpoint)
        run_case(args, pipeline, shape, chunks, case)


if __name__ == "__main__":
    main()
