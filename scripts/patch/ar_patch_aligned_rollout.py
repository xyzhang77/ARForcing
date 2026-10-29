"""Paired AR patch continuations with source-call noise/timestep alignment.

The established probe supplies token conditioning and spatial flow conversion.
This runner adds complete coverage, immutable old-input pairing, and storage;
it never changes the production sampler or previously completed experiments.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch import ar_patch_rollout as old
from scripts.patch.ar_patch_timestep_probe import PatchTimestepRollout
from scripts.patch.patch_noise_importance import atomic_json, load_pipeline, make_noise_bank, tensor_hash
from scripts.patch.patch_noise_strength import decode_pixels, pixel_metrics
from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise

SCHEMA_VERSION = old.SCHEMA_VERSION
MODE, TIMESTEP_POLICY = "matched", "source_call_only"
DEFAULT_ROOT = Path("outputs/ar_patch_aligned_rollout_20261022")
PROBE_ROOT = Path("outputs/ar_patch_timestep_probe_20261022")
CODE_FILES = (*old.CODE_FILES, "scripts/patch/ar_patch_aligned_rollout.py",
              "scripts/patch/ar_patch_timestep_probe.py", "scripts/patch/patch_timestep_utils.py")
CORE_CODE_FILES = tuple(name for name in old.CODE_FILES if name != "scripts/patch/ar_patch_rollout.py")
REQUIRED_ARTIFACTS = {"pixels_npz", "mp4", "last_frame", "trajectory_pt",
                      "trajectory_maps_npz", "final_latent_pt"}


def reference_directory(prompt_index, seed=0):
    if prompt_index == 0 and seed == 0:
        return Path("outputs/ar_patch_rollout_20261022/data")
    return Path("outputs/ar_patch_rollout_step1_multiprompt_20261022/cases") / (
        f"prompt_{prompt_index:06d}_seed_{seed:06d}") / "data"


def parse_args(argv=None):
    # Reuse the public old CLI without inheriting its output or three-step defaults.
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--global-reference-dir", type=Path)
    values, rest = extra.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = old.parse_args(["--output-dir", str(DEFAULT_ROOT / "data"), "--steps", "1",
                           "--num-shards", "4", *rest])
    if args.steps != [1] or args.source_frame != 1:
        raise ValueError("Aligned protocol requires steps=[1] and source_frame=1")
    args.global_reference_dir = values.global_reference_dir or reference_directory(args.prompt_index, args.seed)
    reference = args.global_reference_dir.resolve()
    output = args.output_dir.resolve()
    protected = [reference, Path("outputs/ar_patch_rollout_20261022").resolve(),
                 Path("outputs/ar_patch_rollout_step1_multiprompt_20261022").resolve(), PROBE_ROOT.resolve()]
    if any(output == root or root in output.parents or output in root.parents for root in protected):
        raise ValueError("Output must be separate from every previous experiment")
    return args


def load_global_reference(args, config, checkpoint, prompt, regions, shape, chunks):
    path = args.global_reference_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    protocol = manifest["protocol"]
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("protocol_sha256") != old.json_hash(protocol)):
        raise ValueError("Old AR manifest schema/hash is invalid")
    stats = checkpoint.stat()
    expected = dict(prompt_index=args.prompt_index, prompt=prompt, seed=args.seed,
                    source_chunk=0, source_frame=args.source_frame, grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols, regions=regions, shape=list(shape), chunks=chunks,
                    halo=args.halo, weights=args.weights, config=config,
                    checkpoint_path=str(checkpoint.resolve()), checkpoint_size=stats.st_size,
                    checkpoint_mtime_ns=stats.st_mtime_ns)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Old AR reference differs in {key}")
    if protocol.get("intervention_mode", "global") != "global":
        raise ValueError("Reference must be the previous global-timestep noise-only scan")
    if not set(args.steps).issubset(protocol["steps"]) or args.sigma_scales != protocol["sigma_scales"]:
        raise ValueError("Old AR source steps/scales differ")
    for name in CORE_CODE_FILES:
        if protocol["code_sha256"].get(name) != old.file_hash(name):
            raise ValueError(f"Generation code changed since old AR reference: {name}")
    return manifest, {"data_dir": str(args.global_reference_dir.resolve()),
                      "manifest_path": str(path.resolve()), "manifest_sha256": old.file_hash(path),
                      "protocol_sha256": manifest["protocol_sha256"]}


def validate_reference_baseline(reference, reference_dir, baseline_output, baseline_trace,
                                initial_hash, transition_hashes, schedule, sigmas):
    protocol = reference["protocol"]
    for key, value in dict(baseline_sha256=tensor_hash(baseline_output), initial_noise_sha256=initial_hash,
                           transition_noise_sha256=transition_hashes, schedule=schedule, sigmas=sigmas).items():
        if protocol.get(key) != value:
            raise RuntimeError(f"Old AR baseline pairing differs in {key}")
    record_path = Path(reference_dir) / reference["baseline_record"]
    record = old.validate_completed(record_path, reference["protocol_sha256"])
    if record.get("baseline_sha256") != protocol["baseline_sha256"]:
        raise ValueError("Old baseline record hash differs")
    if record["trajectory"] != old.trajectory_metadata(baseline_trace):
        raise RuntimeError("Old AR baseline trajectory differs")
    expected = torch.load(record_path.parent / record["artifacts"]["final_latent_pt"],
                          map_location="cpu", weights_only=True)
    if not torch.equal(expected, baseline_output.cpu()) or tensor_hash(expected) != protocol["baseline_sha256"]:
        raise RuntimeError("Old AR baseline tensor differs")
    return {"exact": True, "max_abs": 0.0, "record_path": str(record_path.resolve()),
            "record_sha256": old.file_hash(record_path), "trajectory_exact": True}


def check_aligned_controls(replay, baseline_trace, baseline_output, steps, regions, source_frame):
    result = {"identity": old.check_identity(replay, baseline_trace, baseline_output,
                                             steps, regions, source_frame),
              "uniform_token_control": {}, "matched_identity": {}}
    previous_capture = replay.capture
    replay.capture = False
    try:
        for step in steps:
            state = old.source_state(baseline_trace, replay.transition, 0, step, replay.initial.device)
            sigma = next(entry["sigma"] for entry in baseline_trace if old.trace_key(entry) == (0, step))
            unchanged = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                            sigma, regions[0], source_frame)
            if not torch.equal(unchanged, state["noisy"]):
                raise RuntimeError("Matched identity input differs")
            for mode, name in (("token_global", "uniform_token_control"), (MODE, "matched_identity")):
                prediction, output = replay.probe(unchanged, step, mode, regions[0], source_frame, sigma)
                values = {"one_step_max_abs": float((prediction.float() - state["prediction"].float()).abs().max()),
                          "rollout_max_abs": float((output.float() - baseline_output.float()).abs().max()),
                          "rollout_sha256": tensor_hash(output)}
                if (values["one_step_max_abs"] != 0 or values["rollout_max_abs"] != 0
                        or values["rollout_sha256"] != tensor_hash(baseline_output)):
                    raise RuntimeError(f"{name} failed exact control: {values}")
                result[name][str(step)] = values
    finally:
        replay.capture = previous_capture
        replay.take_trace()
    return result


def branch_identity(step, region, scale, source_frame=1):
    return {**old.branch_identity(step, region, scale), "source_chunk": 0, "source_frame": source_frame,
            "intervention_mode": MODE, "timestep_policy": TIMESTEP_POLICY}


def validate_paired_input(reference, reference_dir, step, region, scale, source_frame, changed, delta):
    path = old.branch_directory(reference_dir, step, region["id"], scale) / "record.json"
    identity = {**old.branch_identity(step, region, scale), "source_chunk": 0, "source_frame": source_frame,
                "baseline_sha256": reference["protocol"]["baseline_sha256"]}
    record = old.validate_completed(path, reference["protocol_sha256"], identity)
    if (record["input_sha256"] != tensor_hash(changed)
            or record["input_delta_sha256"] != tensor_hash(delta)):
        raise RuntimeError(f"Aligned input differs from the old noise-only branch: {path}")
    if (not record.get("trajectory") or record["trajectory"][0]["noisy_sha256"] != record["input_sha256"]
            or old.trace_key(record["trajectory"][0]) != (0, step)):
        raise ValueError(f"Old input/trajectory metadata differs: {path}")
    return {"record_path": str(path.resolve()), "record_sha256": old.file_hash(path),
            "protocol_sha256": reference["protocol_sha256"], "comparison_method": "sha256_exact",
            "input_sha256": record["input_sha256"], "input_delta_sha256": record["input_delta_sha256"],
            "input_exact": True, "input_delta_exact": True, "output_sha256": record["output_sha256"],
            "one_step": record["one_step"], "rollout": record["rollout"], "pixel": record["pixel"]}


def annotate_trace(trace, step, region, source_frame, source_sigma, local_timestep, baseline_trace):
    expected = [old.trace_key(entry) for entry in baseline_trace
                if entry["chunk_index"] > 0 or entry["step_index"] >= step]
    if [old.trace_key(entry) for entry in trace] != expected:
        raise RuntimeError("Aligned branch trajectory has missing or reordered calls")
    overridden = [entry for entry in trace if entry.get("mode") is not None]
    if len(overridden) != 1 or old.trace_key(overridden[0]) != (0, step):
        raise RuntimeError("Token timestep override leaked beyond the source call")
    first = trace[0]
    if any(first.get(key) != value for key, value in dict(mode=MODE, source_sigma=source_sigma,
                                                        source_frame=source_frame, region=region).items()):
        raise RuntimeError("Source call conditioning metadata differs")
    first.update(local_timestep=local_timestep, timestep_policy=TIMESTEP_POLICY)
    baseline_next = next(entry for entry in baseline_trace if old.trace_key(entry) == (1, 0))
    next_chunk = next(entry for entry in trace if old.trace_key(entry) == (1, 0))
    if not torch.equal(next_chunk["noisy"], baseline_next["noisy"]):
        raise RuntimeError("Next chunk initial noise is no longer paired")
    return trace


def completed_record(path, protocol_hash, baseline_hash, identity=None, expected_keys=None, preview="vae"):
    """Reject incompatible records; recompute only absent/incomplete artifacts."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (data.get("schema_version") != SCHEMA_VERSION or data.get("protocol_sha256") != protocol_hash
            or data.get("baseline_sha256") != baseline_hash):
        raise ValueError(f"Completed record has incompatible protocol/baseline: {path}")
    if identity and any(data.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Completed record has incompatible branch identity: {path}")
    trajectory = data.get("trajectory")
    if not trajectory:
        return None
    if expected_keys is not None and [old.trace_key(entry) for entry in trajectory] != list(expected_keys):
        raise ValueError(f"Completed record has incompatible trajectory: {path}")
    required = REQUIRED_ARTIFACTS | ({"trajectory_preview_npz"} if preview == "vae" else set())
    if identity:
        required |= {"impact_maps_npz"}
        paired = data.get("paired_global", {})
        if paired.get("input_exact") is not True or paired.get("input_delta_exact") is not True:
            raise ValueError("Completed matched branch lacks old-input pairing")
        first = trajectory[0]
        expected_source = dict(mode=MODE, timestep_policy=TIMESTEP_POLICY,
                               source_frame=data["source_frame"], region=data["region"],
                               source_sigma=data["source_sigma"], local_timestep=data["local_timestep"])
        if any(first.get(key) != value for key, value in expected_source.items()):
            raise ValueError("Completed source trajectory conditioning differs")
        if any("mode" in entry for entry in trajectory[1:]):
            raise ValueError("Completed trajectory overrides later calls")
        if (first.get("noisy_sha256") != data.get("input_sha256")
                or paired.get("input_sha256") != data.get("input_sha256")
                or paired.get("input_delta_sha256") != data.get("input_delta_sha256")):
            raise ValueError("Completed old-input pairing hash differs")
    artifacts, sizes = data.get("artifacts", {}), data.get("artifact_sizes", {})
    if not required.issubset(artifacts) or not set(artifacts).issubset(sizes):
        return None
    for name, relative in artifacts.items():
        file = path.parent / relative
        if file.resolve().parent != path.parent.resolve():
            raise ValueError("Artifact path escapes record directory")
        if not file.is_file() or file.stat().st_size <= 0 or file.stat().st_size != sizes[name]:
            return None
    return data


def retire_incomplete(path):
    path = Path(path)
    if path.exists():
        path.replace(path.with_name("record.incomplete.json"))


def validate_matched_probe(args, step, sid, scale, changed, output, pixels, pixel_result):
    """Only the two actually completed matched probe branches are references."""
    if (args.prompt_index, args.seed, step, sid) != (0, 0, 1, 95) or scale not in (0.5, 0.0):
        return {"available": False}
    path = PROBE_ROOT / "summary.json"
    summary = json.loads(path.read_text())
    if summary.get("complete") is not True:
        raise ValueError("Matched probe reference is incomplete")
    candidates = [record for record in summary["records"]
                  if record["mode"] == MODE and record["sigma_scale"] == scale]
    if len(candidates) != 1:
        raise ValueError("Matched probe reference branch missing or duplicated")
    record = candidates[0]
    if record["input_sha256"] != tensor_hash(changed) or record["output_sha256"] != tensor_hash(output):
        raise RuntimeError("Matched probe input/output hashes differ")
    if record["pixel"] != pixel_result:
        raise RuntimeError("Matched probe float pixel metrics differ")
    directory = PROBE_ROOT / record["directory"]
    expected = torch.load(directory / "final_latent.pt", map_location="cpu", weights_only=True)
    if not torch.equal(output.cpu(), expected):
        raise RuntimeError("Matched probe latent tensor differs")
    with np.load(directory / "final_pixels.npz") as archive:
        if not np.array_equal(archive["pixels"], pixels.mul(255).round().to(torch.uint8).numpy()):
            raise RuntimeError("Matched probe display pixels differ")
    return {"available": True, "summary_path": str(path.resolve()), "summary_sha256": old.file_hash(path),
            "directory": str(directory.resolve()), "input_exact": True, "output_exact": True,
            "latent_max_abs": 0.0, "pixel_metric_exact": True, "display_pixels_exact": True}


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a GPU machine in the dev environment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    # The lifetime lock also prevents accidentally duplicating a resumed worker.
    with (args.output_dir / f".worker_{args.shard_index:03d}.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline, config, shape, chunks, checkpoint = load_pipeline(args)
        if chunks != [3, 3] or float(pipeline.scheduler.num_train_timesteps) != 1000:
            raise ValueError("Protocol requires two three-frame chunks and 1000 training timesteps")
        prompt = args.data_path.read_text().splitlines()[args.prompt_index].strip()
        regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
        config_dict = OmegaConf.to_container(config, resolve=True)
        reference, reference_summary = load_global_reference(args, config_dict, checkpoint, prompt,
                                                             regions, shape, chunks)
        parameter = next(pipeline.generator.parameters())
        pipeline.text_encoder.to(parameter.device)
        conditional = pipeline.text_encoder(text_prompts=[prompt])
        pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()
        initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                              args.seed, parameter.dtype, parameter.device)
        replay = PatchTimestepRollout(pipeline, conditional, initial, transition, chunks)
        print("BASELINE", flush=True)
        baseline_output = replay.baseline(0, 1)["output"]
        baseline_trace = replay.take_trace()
        baseline_hash, initial_hash = tensor_hash(baseline_output), tensor_hash(initial)
        schedule = [float(value) for value in replay.steps]
        sigmas = [entry["sigma"] for entry in baseline_trace if entry["chunk_index"] == 0]
        transition_hashes = {f"{c}:{s}": tensor_hash(n) for (c, s), n in transition.items()}
        baseline_pairing = validate_reference_baseline(reference, args.global_reference_dir, baseline_output,
                                                       baseline_trace, initial_hash, transition_hashes, schedule, sigmas)
        selected_ids = args.source_ids if args.source_ids is not None else [r["id"] for r in regions][args.shard_index::args.num_shards]
        if len(set(selected_ids)) != len(selected_ids) or any(sid < 0 or sid >= len(regions) for sid in selected_ids):
            raise ValueError("Invalid source IDs")
        stats = checkpoint.stat()
        protocol = {"schema_version": SCHEMA_VERSION, "prompt_index": args.prompt_index, "prompt": prompt,
                    "seed": args.seed, "source_chunk": 0, "source_frame": args.source_frame, "steps": args.steps,
                    "sigma_scales": args.sigma_scales, "grid_rows": args.grid_rows, "grid_cols": args.grid_cols,
                    "regions": regions, "shape": shape, "chunks": chunks, "halo": args.halo,
                    "schedule": schedule, "sigmas": sigmas, "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_size": stats.st_size, "checkpoint_mtime_ns": stats.st_mtime_ns,
                    "weights": args.weights, "config": config_dict, "trace_preview": args.trace_preview,
                    "baseline_sha256": baseline_hash, "initial_noise_sha256": initial_hash,
                    "transition_noise_sha256": transition_hashes,
                    "code_sha256": {name: old.file_hash(name) for name in CODE_FILES},
                    "torch_version": torch.__version__, "intervention_mode": MODE,
                    "timestep_policy": TIMESTEP_POLICY, "local_timestep_rule": "source_sigma * 1000",
                    "global_reference": reference_summary,
                    "mechanism": "Local actual sigma, token timestep, and flow-to-clean sigma map aligned at source call only; later calls use original schedule",
                    "model_conditioning": "Token times [B,F*H/2*W/2], F,H,W order; source token t=source_sigma*1000; x0=xt-sigma_map*flow; exact sigma zero supported",
                    "clean_source": "previous model prediction; no ground truth",
                    "normalization_note": "Input-energy denominator includes latent input change only, excluding timestep-conditioning change",
                    "expected_branches": len(regions) * len(args.steps) * len(args.sigma_scales),
                    "trajectory_preview": {"kind": args.trace_preview, "local_latent_frame": 1,
                        "decoded_frame_index": 4, "height": 240, "width": 416, "diagnostic_only": True,
                        "description": "Each three-latent-frame chunk decoded independently; noisy previews are diagnostic, not generated video"}}
        protocol_hash = old.json_hash(protocol)
        controls = check_aligned_controls(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
        print("CONTROLS old identity, uniform token, matched scale1 all exact", flush=True)
        baseline_pixels = decode_pixels(pipeline, baseline_output)
        with (args.output_dir / ".manifest.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest_path = args.output_dir / "manifest.json"
            manifest = {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash,
                        "protocol": protocol, "global_reference": reference_summary,
                        "baseline_record": "baseline/record.json"}
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != manifest:
                    raise ValueError("Output directory belongs to a different experiment protocol")
            else:
                atomic_json(manifest_path, manifest)
            destination = args.output_dir / "baseline"
            destination.mkdir(exist_ok=True)
            path = destination / "record.json"
            baseline_keys = [old.trace_key(entry) for entry in baseline_trace]
            if completed_record(path, protocol_hash, baseline_hash, expected_keys=baseline_keys,
                                preview=args.trace_preview) is None:
                retire_incomplete(path)
                artifacts = old.save_video(baseline_pixels, destination)
                artifacts.update(old.save_trace(pipeline, baseline_trace, baseline_trace, destination, args.trace_preview))
                torch.save(baseline_output.cpu(), destination / "final_latent.pt")
                artifacts["final_latent_pt"] = "final_latent.pt"
                atomic_json(path, {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash,
                                  "baseline_sha256": baseline_hash, "trajectory": old.trajectory_metadata(baseline_trace),
                                  **controls, "global_baseline_pairing": baseline_pairing,
                                  "artifacts": artifacts, "artifact_sizes": old.artifact_sizes(destination, artifacts)})
            fcntl.flock(lock, fcntl.LOCK_UN)
        completed, resumed, records = 0, 0, []
        for step in args.steps:
            state = old.source_state(baseline_trace, transition, 0, step, parameter.device)
            base_sigma = sigmas[step]
            expected_keys = [key for key in baseline_keys if key[0] > 0 or key[1] >= step]
            for sid in selected_ids:
                region = regions[sid]
                for scale in args.sigma_scales:
                    branch_started = time.monotonic()
                    destination = old.branch_directory(args.output_dir, step, sid, scale)
                    record_path = destination / "record.json"
                    key = branch_identity(step, region, scale, args.source_frame)
                    sigma, local_timestep = base_sigma * scale, base_sigma * scale * 1000
                    key.update(source_sigma=sigma, local_timestep=local_timestep)
                    changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"],
                                                  sigma, region, args.source_frame)
                    delta = changed.float() - state["noisy"].float()
                    paired = validate_paired_input(reference, args.global_reference_dir, step, region, scale,
                                                   args.source_frame, changed, delta)
                    if record_path.exists():
                        if not args.resume:
                            raise ValueError(f"Completed result exists; use --resume: {record_path}")
                        existing = completed_record(record_path, protocol_hash, baseline_hash, key,
                                                    expected_keys, args.trace_preview)
                        if existing is not None:
                            if existing["paired_global"] != paired:
                                raise ValueError("Resumed branch old-reference record changed")
                            resumed += 1
                            records.append(str(record_path.relative_to(args.output_dir)))
                            continue
                        retire_incomplete(record_path)
                    destination.mkdir(parents=True, exist_ok=True)
                    prediction, output = replay.probe(changed, step, MODE, region, args.source_frame, sigma)
                    trace = annotate_trace(replay.take_trace(), step, region, args.source_frame, sigma,
                                           local_timestep, baseline_trace)
                    one_step = measure_influence(state["prediction"], prediction, delta, region,
                                                 args.source_frame, regions, args.halo)
                    rollout = measure_influence(baseline_output, output, delta, region,
                                                args.source_frame, regions, args.halo)
                    future = (output[:, chunks[0]:].float() - baseline_output[:, chunks[0]:].float()).square()
                    energy = float(future.sum())
                    rollout.update(next_chunk_mse=float(future.mean()), next_chunk_energy=energy,
                                   normalized_next_chunk_energy=energy / rollout["input_energy"] if rollout["input_energy"] else 0.0)
                    pixels = decode_pixels(pipeline, output)
                    pixel = pixel_metrics(baseline_pixels, pixels)
                    matched_replay = validate_matched_probe(args, step, sid, scale, changed, output, pixels, pixel)
                    artifacts = old.save_video(pixels, destination)
                    artifacts.update(old.save_trace(pipeline, trace, baseline_trace, destination, args.trace_preview))
                    torch.save(output.cpu(), destination / "final_latent.pt")
                    artifacts["final_latent_pt"] = "final_latent.pt"
                    np.savez_compressed(destination / "impact_maps.npz",
                                        **old.impact_maps(baseline_output, output, baseline_pixels, pixels))
                    artifacts["impact_maps_npz"] = "impact_maps.npz"
                    record = {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash, **key,
                              "num_tokens": (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4,
                              "global_t": schedule[step], "base_sigma": base_sigma, "baseline_sha256": baseline_hash,
                              "input_sha256": tensor_hash(changed), "input_delta_sha256": tensor_hash(delta),
                              "output_sha256": tensor_hash(output), "one_step": one_step, "rollout": rollout,
                              "pixel": pixel, "trajectory": old.trajectory_metadata(trace),
                              "shared_prefix": "baseline/trajectory.pt", "shared_prefix_steps": [[0, s] for s in range(step)],
                              "paired_global": paired, "matched_probe_replay": matched_replay,
                              "artifacts": artifacts, "artifact_sizes": old.artifact_sizes(destination, artifacts),
                              "elapsed_seconds": time.monotonic() - branch_started, "host": socket.gethostname(),
                              "shard_index": args.shard_index}
                    atomic_json(record_path, record)
                    records.append(str(record_path.relative_to(args.output_dir)))
                    completed += 1
                    print(f"BRANCH step={step} source={sid:03d} scale={scale:g} local_t={local_timestep:g} "
                          f"pixel_rmse={pixel['all_video_rmse']:.8g} seconds={record['elapsed_seconds']:.2f}", flush=True)
                    atomic_json(args.output_dir / f"progress_shard_{args.shard_index:03d}.json",
                                {"completed_this_invocation": completed, "resumed": resumed,
                                 "last_record": records[-1], "protocol_sha256": protocol_hash})
                    if args.max_records is not None and completed >= args.max_records:
                        break
                if args.max_records is not None and completed >= args.max_records:
                    break
            if args.max_records is not None and completed >= args.max_records:
                break
        after = check_aligned_controls(replay, baseline_trace, baseline_output, args.steps, regions, args.source_frame)
        expected_worker_records = len(selected_ids) * len(args.steps) * len(args.sigma_scales)
        result = {"protocol_sha256": protocol_hash, "shard_index": args.shard_index, "num_shards": args.num_shards,
                  "source_ids": selected_ids, "only_step": args.only_step, "max_records": args.max_records,
                  "completed_this_invocation": completed, "resumed": resumed, "completed": completed + resumed,
                  "expected_worker_records": expected_worker_records,
                  "status": "complete" if completed + resumed == expected_worker_records else "partial",
                  "exit_code": 0,
                  **controls, "post_identity": after["identity"],
                  "post_uniform_token_control": after["uniform_token_control"],
                  "post_matched_identity": after["matched_identity"], "records": records,
                  "global_baseline_pairing": baseline_pairing, "elapsed_seconds": time.monotonic() - started}
        atomic_json(args.output_dir / f"worker_{args.shard_index:03d}_result.json", result)
        print(f"DONE completed={completed} resumed={resumed} seconds={result['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
