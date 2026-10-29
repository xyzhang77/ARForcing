"""Dense paired few-step AR patch continuations, with exact denoising traces.

No production sampler is changed. A trace hook wraps the established
PairedRollout, and every source patch is continued to the final video.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.patch_group_importance import validate_model_source
from scripts.patch.patch_noise_importance import (
    PairedRollout, atomic_json, load_pipeline, make_noise_bank, tensor_hash,
)
from scripts.patch.patch_noise_strength import decode_pixels, pixel_metrics, scale_name
from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise


SCHEMA_VERSION = "ar_patch_rollout_v1"
CODE_FILES = (
    "scripts/patch/ar_patch_rollout.py", "scripts/patch/patch_noise_importance.py",
    "scripts/patch/patch_noise_strength.py", "scripts/patch/patch_noise_utils.py",
    "scripts/patch/patch_group_importance.py", "pipeline/causal_inference.py",
    "utils/wan_wrapper.py", "utils/scheduler.py", "wan/modules/causal_model.py",
    "wan/modules/vae.py",
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TracedRollout(PairedRollout):
    """Record actual inputs and outputs without duplicating recurrence logic."""

    def __init__(self, *args, **kwargs):
        self.trace = []
        self.capture = True
        super().__init__(*args, **kwargs)

    def predict(self, noisy, chunk, step=None, context=False):
        prediction = super().predict(noisy, chunk, step, context)
        if self.capture and not context:
            index = (self.p.scheduler.timesteps - float(self.steps[step])).abs().argmin()
            self.trace.append({
                "chunk_index": chunk, "step_index": step,
                "global_t": float(self.steps[step]),
                "sigma": float(self.p.scheduler.sigmas[index]),
                "start_frame": self.starts[chunk], "frame_count": self.chunks[chunk],
                "noisy": noisy.detach().to("cpu", copy=True),
                "clean_prediction": prediction.detach().to("cpu", copy=True),
            })
        return prediction

    def take_trace(self):
        trace, self.trace = self.trace, []
        return trace


def trace_key(entry):
    return entry["chunk_index"], entry["step_index"]


def source_state(trace, transition, source_chunk, source_step, device):
    """Recover every source step from one unmodified baseline pass."""
    by_key = {trace_key(entry): entry for entry in trace}
    if source_step < 1:
        raise ValueError("Local sigma reduction requires a previous clean prediction")
    current, previous = by_key[source_chunk, source_step], by_key[source_chunk, source_step - 1]
    return {
        "noisy": current["noisy"].to(device),
        "prediction": current["clean_prediction"].to(device),
        "clean": previous["clean_prediction"].to(device),
        "noise": transition[source_chunk, source_step],
    }


def trajectory_metadata(trace):
    return [{**{key: value for key, value in entry.items()
                if key not in ("noisy", "clean_prediction")},
             "noisy_sha256": tensor_hash(entry["noisy"]),
             "clean_sha256": tensor_hash(entry["clean_prediction"])} for entry in trace]


def trajectory_maps(trace, baseline_trace):
    baseline = {trace_key(entry): entry for entry in baseline_trace}
    maps = {name: [] for name in ("noisy_rms", "clean_rms", "noisy_delta_rms", "clean_delta_rms")}
    for entry in trace:
        reference = baseline[trace_key(entry)]
        for kind, key in (("noisy", "noisy"), ("clean", "clean_prediction")):
            value, ref = entry[key].float(), reference[key].float()
            maps[f"{kind}_rms"].append(value.square().mean(dim=(0, 2)).sqrt().numpy())
            maps[f"{kind}_delta_rms"].append((value - ref).square().mean(dim=(0, 2)).sqrt().numpy())
    return {name: np.stack(values) for name, values in maps.items()}


def save_video(pixels, destination):
    """Persist display samples; numerical metrics are computed before this."""
    from PIL import Image
    import imageio.v2 as imageio
    video = pixels.mul(255).round().to(torch.uint8).numpy()
    np.savez_compressed(destination / "final_pixels.npz", pixels=video)
    imageio.mimwrite(str(destination / "final.mp4"), video, fps=16, codec="libx264",
                    quality=7, macro_block_size=1, ffmpeg_params=["-threads", "2"])
    Image.fromarray(video[-1]).save(destination / "final.jpg", quality=95)
    return {"pixels_npz": "final_pixels.npz", "mp4": "final.mp4", "last_frame": "final.jpg"}


def save_trace(pipeline, trace, baseline_trace, destination, preview):
    torch.save({"schema_version": SCHEMA_VERSION, "entries": trace}, destination / "trajectory.pt")
    np.savez_compressed(destination / "trajectory_maps.npz", **trajectory_maps(trace, baseline_trace))
    artifacts = {"trajectory_pt": "trajectory.pt", "trajectory_maps_npz": "trajectory_maps.npz"}
    if preview == "vae":
        from PIL import Image
        parameter = next(pipeline.generator.parameters())
        previews = {"noisy": [], "clean": []}
        for entry in trace:
            for name, key in (("noisy", "noisy"), ("clean", "clean_prediction")):
                decoded = decode_pixels(pipeline, entry[key].to(parameter.device))
                if decoded.shape[0] <= 4:
                    raise ValueError("Trace preview requires three latent frames per chunk")
                frame = decoded[4].mul(255).round().to(torch.uint8).numpy()
                previews[name].append(np.asarray(Image.fromarray(frame).resize((416, 240), Image.Resampling.LANCZOS)))
        np.savez_compressed(destination / "trajectory_preview.npz",
                            **{name: np.stack(values) for name, values in previews.items()})
        artifacts["trajectory_preview_npz"] = "trajectory_preview.npz"
    return artifacts


def impact_maps(baseline_latent, changed_latent, baseline_pixels, changed_pixels):
    latent = (changed_latent.float() - baseline_latent.float()).square().mean(dim=(0, 2)).cpu().numpy()
    # Pixel area averages preserve the mean MSE, and keep all 21 frames small.
    pixel = (changed_pixels - baseline_pixels).square().mean(dim=-1)
    frames, height, width = pixel.shape
    if height % 8 or width % 8:
        raise ValueError("Pixel impact maps require dimensions divisible by eight")
    pixel = pixel.reshape(frames, height // 8, 8, width // 8, 8).mean(dim=(2, 4)).numpy()
    return {"latent_mse": latent, "pixel_mse_downsample": pixel}


def artifact_sizes(directory, artifacts):
    return {name: (directory / relative).stat().st_size for name, relative in artifacts.items()}


def validate_completed(path, protocol_sha256, identity=None):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION or data.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Completed record has incompatible protocol: {path}")
    if identity and any(data.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Completed record has a different branch identity: {path}")
    for name, relative in data["artifacts"].items():
        artifact = Path(path).parent / relative
        if not artifact.is_file() or artifact.stat().st_size != data["artifact_sizes"][name]:
            raise ValueError(f"Completed record has a missing or truncated artifact: {artifact}")
    return data


def branch_identity(step, region, scale):
    return {"step_index": step, "source_id": region["id"], "sigma_scale": scale, "region": region}


def branch_directory(root, step, source_id, scale):
    return Path(root) / f"step_{step}" / f"source_{source_id:03d}" / f"scale_{scale_name(scale)}"


def check_identity(replay, baseline_trace, baseline_output, steps, regions, source_frame):
    replay.capture = False
    result = {}
    for step in steps:
        state = source_state(baseline_trace, replay.transition, 0, step, replay.initial.device)
        entry = next(entry for entry in baseline_trace if trace_key(entry) == (0, step))
        unchanged = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                        entry["sigma"], regions[0], source_frame)
        if not torch.equal(unchanged, state["noisy"]):
            raise RuntimeError("Identity local-sigma input differs")
        prediction, output = replay.branch(unchanged, 0, step, complete=True)
        result[str(step)] = {
            "one_step_max_abs": float((prediction.float() - state["prediction"].float()).abs().max()),
            "rollout_max_abs": float((output.float() - baseline_output.float()).abs().max()),
            "rollout_sha256": tensor_hash(output),
        }
        if any(result[str(step)][key] != 0 for key in ("one_step_max_abs", "rollout_max_abs")):
            raise RuntimeError(f"Identity replay failed: {result[str(step)]}")
    replay.capture = True
    replay.take_trace()
    return result


def old_case_path(reference_dir, args, step):
    return Path(reference_dir) / (f"prompt_{args.prompt_index:06d}_seed_{args.seed:06d}"
                                  f"_chunk_0_step_{step}")


def read_reference(path, required=False):
    """Only an absent reference may be skipped; partial references are errors."""
    if not path.exists():
        if required:
            raise ValueError(f"Required reference is missing: {path}")
        if path.parent.is_dir() and any(path.parent.iterdir()):
            raise ValueError(f"Reference artifacts exist but metadata is missing: {path}")
        return None
    data = json.loads(path.read_text())
    if (not isinstance(data.get("metadata"), dict) or not isinstance(data.get("regions"), list)
            or not isinstance(data.get("records"), list) or not data["records"]):
        raise ValueError(f"Incomplete reference metadata/regions/records: {path}")
    identity = data.get("identity", {})
    if any(identity.get(key) != 0 for key in ("one_step_max_abs", "rollout_max_abs")):
        raise ValueError(f"Reference identity is absent or failed: {path}")
    return data


def load_references(args, config, checkpoint, prompt, regions):
    """Validate every available old case before running the new baseline."""
    policy = getattr(args, "reference_policy", "required")
    bundle, summary = {}, {"policy": policy}
    for kind, directory, filename, steps in (
            ("fine", args.reference_dir, "metrics.json", sorted(set(args.steps) | {1})),
            ("strength", args.strength_reference_dir, "strength_metrics.json", args.steps)):
        bundle[kind], summary[kind] = {}, {}
        for step in steps:
            path = old_case_path(directory, args, step) / filename
            required = policy == "required" and (step == 1 if kind == "fine" else step in (1, 2))
            data = read_reference(path, required)
            bundle[kind][step] = data
            summary[kind][str(step)] = {"status": "available" if data is not None else "not_available",
                                       "path": str(path.resolve()), "sha256": file_hash(path) if data is not None else None}
            if data is None:
                continue
            metadata = data["metadata"]
            validate_model_source(metadata, config, checkpoint)
            expected = {"prompt_index": args.prompt_index, "prompt": prompt, "seed": args.seed,
                        "source_chunk": args.source_chunk, "source_frame": args.source_frame,
                        "step_index": step, "weights": args.weights,
                        "grid_rows": args.grid_rows, "grid_cols": args.grid_cols}
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Source/prompt/seed/geometry differs from reference: {path}")
            if data["regions"] != regions:
                raise ValueError(f"Source patch geometry differs from reference: {path}")
            # Missing selected artifacts must not turn a partial old run into a skipped check.
            for record in data["records"]:
                if kind == "fine" and record["sigma_scale"] == 0.75 and record["rollout"] is not None:
                    artifact = path.parent / f"source_{record['source_id']:03d}_rollout.pt"
                    if not artifact.is_file():
                        raise ValueError(f"Reference rollout artifact is missing: {artifact}")
                if kind == "strength":
                    pixel = record.get("pixels", record.get("pixel"))
                    if not isinstance(pixel, dict) or "all_video_mse" not in pixel:
                        raise ValueError(f"Reference float pixel metric is missing: {path}")
                    artifacts = [path.parent / record["artifacts"]["pixels_npz"]]
                    if record["sigma_scale"] == 0:
                        artifacts.append(path.parent / f"source_{record['source_id']:03d}_scale_0_rollout.pt")
                    if any(not artifact.is_file() for artifact in artifacts):
                        raise ValueError(f"Reference strength artifact is missing: {path}")
    return bundle, summary


def validate_reference_baselines(bundle, baseline_hash, initial_hash, schedule, sigmas, transition_hashes):
    for kind, cases in bundle.items():
        for step, data in cases.items():
            if data is None:
                continue
            metadata = data["metadata"]
            if metadata["baseline_sha256"] != baseline_hash or metadata["initial_noise_sha256"] != initial_hash:
                raise RuntimeError(f"Baseline/initial-noise hash differs from original {kind} step {step}")
            if (metadata["schedule"] != schedule or metadata["global_t"] != schedule[step]
                    or metadata["base_sigma"] != sigmas[step]):
                raise ValueError(f"Denoising schedule differs from original {kind} step {step}")
            if kind == "strength" and metadata["transition_noise_sha256"] != transition_hashes:
                raise RuntimeError("Transition-noise hash differs from original strength sweep")


def validate_old_rollout(args, step, source_id, scale, output, pixels, pixel_result):
    """Exact replay where old artifacts exist; absence is explicitly recorded."""
    policy = getattr(args, "reference_policy", "required")
    result = {}
    if policy == "if-present":
        result["reference_status"] = {
            kind: "available" if (old_case_path(directory, args, step) / filename).is_file() else "not_available"
            for kind, directory, filename in (("fine", args.reference_dir, "metrics.json"),
                                              ("strength", args.strength_reference_dir, "strength_metrics.json"))}
    if scale == 0.75:
        old = old_case_path(args.reference_dir, args, step) / f"source_{source_id:03d}_rollout.pt"
        if old.is_file():
            expected = torch.load(old, map_location="cpu", weights_only=False)
            difference = float((output.cpu().float() - expected.float()).abs().max())
            result["fine_latent"] = {"path": str(old), "max_abs": difference,
                                      "sha256": tensor_hash(expected)}
            if difference != 0 or tensor_hash(output) != tensor_hash(expected):
                raise RuntimeError(f"Old fine continuation differs: {old}")
    strength_case = old_case_path(args.strength_reference_dir, args, step)
    strength_metrics = strength_case / "strength_metrics.json"
    if policy == "required" and step in (1, 2) and not strength_metrics.is_file():
        raise ValueError(f"Expected existing strength reference is missing: {strength_metrics}")
    if strength_metrics.is_file():
        old_data = json.loads(strength_metrics.read_text())
        candidates = [record for record in old_data["records"]
                      if record["source_id"] == source_id and record["sigma_scale"] == scale]
        if candidates:
            record = candidates[0]
            old_pixel = record.get("pixels", record.get("pixel"))
            if old_pixel is None:
                raise ValueError(f"Old strength float pixel metric is missing: {strength_metrics}")
            difference = abs(pixel_result["all_video_mse"] - old_pixel["all_video_mse"])
            result["strength_pixel_metric"] = {"path": str(strength_metrics), "mse_abs_difference": difference}
            if difference != 0:
                raise RuntimeError("Old strength float pixel metric differs")
            old_npz = strength_case / record["artifacts"]["pixels_npz"]
            with np.load(old_npz) as archive:
                expected = archive["pixels"]
            equal = np.array_equal(expected, pixels.mul(255).round().to(torch.uint8).numpy())
            result["strength_uint8"] = {"path": str(old_npz), "exact": equal}
            if not equal:
                raise RuntimeError(f"Old strength display pixels differ: {old_npz}")
            if scale == 0:
                old_latent = strength_case / f"source_{source_id:03d}_scale_0_rollout.pt"
                expected = torch.load(old_latent, map_location="cpu", weights_only=False)
                difference = float((output.cpu().float() - expected.float()).abs().max())
                result["strength_latent"] = {"path": str(old_latent), "max_abs": difference}
                if difference != 0:
                    raise RuntimeError("Old strength latent differs")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ar_patch_rollout_20261022"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    parser.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    parser.add_argument("--reference-dir", type=Path, default=Path("outputs/anchor_probe_step600_20261021/fine_sweep"))
    parser.add_argument("--strength-reference-dir", type=Path, default=Path("outputs/anchor_probe_step600_20261021/strength_sweep"))
    parser.add_argument("--reference-policy", choices=("required", "if-present"), default="required",
                        help="if-present permits absent old cases, but validates every available reference")
    parser.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma-scales", type=float, nargs="+", default=[0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--grid-rows", type=int, default=12)
    parser.add_argument("--grid-cols", type=int, default=16)
    parser.add_argument("--source-frame", type=int, default=1)
    parser.add_argument("--trace-preview", choices=("latent", "vae"), default="vae")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--source-ids", type=int, nargs="+", help="Execution filter; protocol still covers the complete grid")
    parser.add_argument("--only-step", type=int, help="Execution filter for pilot runs")
    parser.add_argument("--max-records", type=int, help="Stop this invocation after N newly completed branches")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.prompt_index < 0 or args.seed < 0:
        parser.error("prompt-index and seed must be nonnegative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard assignment")
    if not args.steps or len(set(args.steps)) != len(args.steps) or any(step not in (1, 2, 3) for step in args.steps):
        parser.error("steps must be distinct values in 1,2,3")
    if (not args.sigma_scales or len(set(args.sigma_scales)) != len(args.sigma_scales)
            or any(not math.isfinite(scale) or not 0 <= scale < 1 for scale in args.sigma_scales)):
        parser.error("sigma scales must be distinct and in [0,1)")
    if args.only_step is not None and args.only_step not in args.steps:
        parser.error("only-step must occur in steps")
    if args.max_records is not None and args.max_records < 1:
        parser.error("max-records must be positive")
    args.weights, args.num_output_frames, args.source_chunk, args.halo = "generator", 6, 0, 2
    args.ranking_metric = "normalized_other_frames_energy"
    return args


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a GPU machine in the dev environment")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    pipeline, config, shape, chunks, checkpoint = load_pipeline(args)
    if chunks != [3, 3]:
        raise ValueError("This protocol requires two three-frame AR chunks")
    prompt = args.data_path.read_text().splitlines()[args.prompt_index].strip()
    regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
    references, reference_summary = load_references(args, config, checkpoint, prompt, regions)
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                          args.seed, parameter.dtype, parameter.device)
    replay = TracedRollout(pipeline, conditional, initial, transition, chunks)
    print("BASELINE", flush=True)
    baseline_state = replay.baseline(0, 1)
    baseline_output, baseline_trace = baseline_state["output"], replay.take_trace()
    baseline_hash, initial_hash = tensor_hash(baseline_output), tensor_hash(initial)
    schedule = [float(value) for value in replay.steps]
    sigmas = [entry["sigma"] for entry in baseline_trace if entry["chunk_index"] == 0]
    transition_hashes = {f"{chunk}:{step}": tensor_hash(noise) for (chunk, step), noise in transition.items()}
    validate_reference_baselines(references, baseline_hash, initial_hash, schedule, sigmas, transition_hashes)
    selected_ids = args.source_ids if args.source_ids is not None else [r["id"] for r in regions][args.shard_index::args.num_shards]
    if len(set(selected_ids)) != len(selected_ids) or any(sid < 0 or sid >= len(regions) for sid in selected_ids):
        raise ValueError("Invalid source IDs")
    stats = checkpoint.stat()
    protocol = {
        "schema_version": SCHEMA_VERSION, "prompt_index": args.prompt_index, "prompt": prompt, "seed": args.seed,
        "source_chunk": 0, "source_frame": args.source_frame, "steps": args.steps,
        "sigma_scales": args.sigma_scales, "grid_rows": args.grid_rows, "grid_cols": args.grid_cols,
        "regions": regions, "shape": shape, "chunks": chunks, "halo": args.halo,
        "schedule": schedule, "sigmas": sigmas,
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_size": stats.st_size,
        "checkpoint_mtime_ns": stats.st_mtime_ns, "weights": args.weights,
        "config": OmegaConf.to_container(config, resolve=True), "trace_preview": args.trace_preview,
        "baseline_sha256": baseline_hash, "initial_noise_sha256": initial_hash,
        "transition_noise_sha256": transition_hashes,
        "code_sha256": {name: file_hash(name) for name in CODE_FILES},
        "torch_version": torch.__version__, "reference_metadata_sha256": reference_summary["fine"]["1"]["sha256"],
        "mechanism": "Local actual sigma lowered using previous predicted clean and fixed transition epsilon; global timestep unchanged",
        "clean_source": "previous model prediction; no ground truth",
        "expected_branches": len(regions) * len(args.steps) * len(args.sigma_scales),
        "trajectory_preview": {"kind": args.trace_preview, "local_latent_frame": 1, "decoded_frame_index": 4,
                               "height": 240, "width": 416, "diagnostic_only": True,
                               "description": "Each three-latent-frame chunk decoded independently; noisy previews are diagnostic, not generated video"},
    }
    if args.reference_policy != "required":
        protocol.update(reference_policy=args.reference_policy, reference_availability=reference_summary)
    protocol_hash = json_hash(protocol)
    identity = check_identity(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
    print("IDENTITY exact for every source step", flush=True)
    baseline_pixels = decode_pixels(pipeline, baseline_output)
    with (args.output_dir / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if (existing["protocol_sha256"] != protocol_hash or existing["protocol"] != protocol
                    or existing.get("references") != reference_summary):
                raise ValueError("Output directory belongs to a different experiment protocol")
        else:
            atomic_json(manifest_path, {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash,
                                       "protocol": protocol, "references": reference_summary,
                                       "baseline_record": "baseline/record.json"})
        baseline_dir = args.output_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        if (baseline_dir / "record.json").exists():
            validate_completed(baseline_dir / "record.json", protocol_hash)
        else:
            artifacts = save_video(baseline_pixels, baseline_dir)
            artifacts.update(save_trace(pipeline, baseline_trace, baseline_trace, baseline_dir, args.trace_preview))
            torch.save(baseline_output.cpu(), baseline_dir / "final_latent.pt")
            artifacts["final_latent_pt"] = "final_latent.pt"
            atomic_json(baseline_dir / "record.json", {"schema_version": SCHEMA_VERSION,
                        "protocol_sha256": protocol_hash, "baseline_sha256": baseline_hash,
                        "trajectory": trajectory_metadata(baseline_trace), "identity": identity,
                        "artifacts": artifacts, "artifact_sizes": artifact_sizes(baseline_dir, artifacts)})
        fcntl.flock(lock, fcntl.LOCK_UN)
    completed, resumed, records = 0, 0, []
    for step in args.steps:
        if args.only_step is not None and step != args.only_step:
            continue
        state = source_state(baseline_trace, transition, 0, step, parameter.device)
        base_sigma = protocol["sigmas"][step]
        for sid in selected_ids:
            region = regions[sid]
            for scale in args.sigma_scales:
                destination = branch_directory(args.output_dir, step, sid, scale)
                record_path = destination / "record.json"
                key = branch_identity(step, region, scale)
                if record_path.exists():
                    if not args.resume:
                        raise ValueError(f"Completed result exists; use --resume: {record_path}")
                    validate_completed(record_path, protocol_hash, key)
                    resumed += 1
                    continue
                branch_started = time.monotonic()
                destination.mkdir(parents=True, exist_ok=True)
                changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                              base_sigma * scale, region, args.source_frame)
                prediction, output = replay.branch(changed, 0, step, complete=True)
                trace = replay.take_trace()
                delta = changed.float() - state["noisy"].float()
                one_step = measure_influence(state["prediction"], prediction, delta, region,
                                             args.source_frame, regions, args.halo)
                rollout = measure_influence(baseline_output, output, delta, region,
                                            args.source_frame, regions, args.halo)
                future = (output[:, chunks[0]:].float() - baseline_output[:, chunks[0]:].float()).square()
                rollout.update(next_chunk_mse=float(future.mean()), next_chunk_energy=float(future.sum()),
                               normalized_next_chunk_energy=float(future.sum()) / rollout["input_energy"])
                pixels = decode_pixels(pipeline, output)
                pixel = pixel_metrics(baseline_pixels, pixels)
                old_replay = validate_old_rollout(args, step, sid, scale, output, pixels, pixel)
                artifacts = save_video(pixels, destination)
                artifacts.update(save_trace(pipeline, trace, baseline_trace, destination, args.trace_preview))
                torch.save(output.cpu(), destination / "final_latent.pt")
                artifacts["final_latent_pt"] = "final_latent.pt"
                np.savez_compressed(destination / "impact_maps.npz",
                                    **impact_maps(baseline_output, output, baseline_pixels, pixels))
                artifacts["impact_maps_npz"] = "impact_maps.npz"
                record = {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash, **key,
                          "source_chunk": 0, "source_frame": args.source_frame,
                          "num_tokens": (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4,
                          "global_t": float(replay.steps[step]), "base_sigma": base_sigma,
                          "source_sigma": base_sigma * scale, "baseline_sha256": baseline_hash,
                          "input_sha256": tensor_hash(changed), "input_delta_sha256": tensor_hash(delta),
                          "output_sha256": tensor_hash(output), "one_step": one_step, "rollout": rollout,
                          "pixel": pixel, "trajectory": trajectory_metadata(trace),
                          "shared_prefix": "baseline/trajectory.pt", "old_replay": old_replay,
                          "artifacts": artifacts, "artifact_sizes": artifact_sizes(destination, artifacts),
                          "elapsed_seconds": time.monotonic() - branch_started,
                          "host": socket.gethostname(), "shard_index": args.shard_index}
                atomic_json(record_path, record)
                records.append(str(record_path.relative_to(args.output_dir)))
                completed += 1
                print(f"BRANCH step={step} source={sid:03d} scale={scale:g} "
                      f"pixel_rmse={pixel['all_video_rmse']:.8g} seconds={record['elapsed_seconds']:.2f}", flush=True)
                atomic_json(args.output_dir / f"progress_shard_{args.shard_index:03d}.json",
                            {"completed_this_invocation": completed, "resumed": resumed, "last_record": records[-1],
                             "protocol_sha256": protocol_hash})
                if args.max_records is not None and completed >= args.max_records:
                    break
            if args.max_records is not None and completed >= args.max_records:
                break
        if args.max_records is not None and completed >= args.max_records:
            break
    post_identity = check_identity(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
    result = {"protocol_sha256": protocol_hash, "shard_index": args.shard_index, "num_shards": args.num_shards,
              "source_ids": selected_ids, "only_step": args.only_step, "max_records": args.max_records,
              "completed_this_invocation": completed, "resumed": resumed, "identity": identity,
              "post_identity": post_identity, "records": records, "elapsed_seconds": time.monotonic() - started}
    atomic_json(args.output_dir / f"worker_{args.shard_index:03d}_result.json", result)
    print(f"DONE completed={completed} resumed={resumed} seconds={result['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
