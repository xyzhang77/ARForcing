#!/usr/bin/env python3
"""Capture all four AR denoiser calls from the frozen first-step experiment.

Each state_stepN is the noisy input before call N, and each x0_stepN is
that call's clean estimate. The schedule and random noise bank are unchanged.
The original artifacts are read-only references: any hash or replay mismatch
stops the run before the new case is published.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "outputs/denoising/first_step_noise_20261023"
SCHEMA_VERSION = "denoising_steps_ar_v1"
PROMPTS = [42, 158, 300]
SEEDS = list(range(8))
SHAPE = [1, 6, 16, 60, 104]
CHUNKS = [3, 3]
LATENT_SHAPE = [1, 3, 16, 60, 104]
VIEW_SHAPE = (3, 240, 416, 3)
KEYS = {*(f"x0_step{i}" for i in range(1, 5)),
        *(f"state_step{i}" for i in range(1, 5)), "final"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    # Match the original compare_first_step_noise protocol digest exactly.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Original experiment directory; new files go under denoising_steps/ar")
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS,
                        help="Original seeds to replay (default: 0 1 2 3 4 5 6 7)")
    parser.add_argument("--prompt-index", type=int, nargs="+", default=PROMPTS,
                        help="Original prompt indices (default: 42 158 300)")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device, e.g. cuda:0; CUDA_VISIBLE_DEVICES is also supported")
    args = parser.parse_args(argv)
    for name, allowed in (("seed", SEEDS), ("prompt_index", PROMPTS)):
        values = getattr(args, name)
        if len(set(values)) != len(values) or any(value not in allowed for value in values):
            parser.error(f"--{name.replace('_', '-')} must contain distinct values from {allowed}")
    if args.device != "cuda" and not (args.device.startswith("cuda:")
                                      and args.device[5:].isdigit()):
        parser.error("--device must be cuda or cuda:<index>")
    args.root = args.root.resolve()
    if not args.root.is_dir():
        parser.error(f"Original experiment root does not exist: {args.root}")
    return args


def load_source(root, prompt_index, seed):
    """Verify the original record and both original artifact files before use."""
    directory = root / "ar" / f"prompt_{prompt_index:06d}" / f"seed_{seed:06d}"
    path = directory / "record.json"
    record = json.loads(path.read_text())
    require(record.get("complete") is True, f"Original case is incomplete: {path}")
    require((record.get("model"), record.get("prompt_index"), record.get("seed"))
            == ("ar", prompt_index, seed), f"Original case identity differs: {path}")
    protocol = record["protocol"]
    require(record.get("protocol_sha256") == json_hash(protocol),
            f"Original protocol digest differs: {path}")
    for name in ("model", "prompt_index", "seed", "prompt", "shape", "chunks"):
        require(record.get(name) == protocol.get(name), f"Original protocol {name} differs: {path}")
    require(record["shape"] == SHAPE and record["chunks"] == CHUNKS,
            f"Original case has a different latent/chunk layout: {path}")
    require(len(record["timesteps"]) == len(record["sigmas"]) == 4,
            f"Original case must have exactly four denoising calls: {path}")
    for name in ("latents.pt", "views.npz"):
        artifact = directory / name
        require(artifact.is_file() and file_hash(artifact) == record["artifact_sha256"][name],
                f"Original artifact missing or changed: {artifact}")
    source = {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path),
              "artifact_sha256": dict(record["artifact_sha256"])}
    return {"directory": directory, "record": record, "source_record": source}


def validate_sources(args, config, torch_version):
    """Freeze each selected case's source, configuration, code and checkpoint."""
    experiment = json.loads((args.root / "experiment.json").read_text())
    cases = []
    code_hashes = {}
    model_source = None
    for index in args.prompt_index:
        for seed in args.seed:
            case = load_source(args.root, index, seed)
            record, protocol = case["record"], case["record"]["protocol"]
            require(record["prompt"] == experiment["prompts"][str(index)],
                    f"Frozen prompt differs for prompt={index}")
            require(protocol["config"] == config, "Frozen inference configuration differs from original")
            require(protocol["torch_version"] == torch_version,
                    f"PyTorch version differs: original={protocol['torch_version']}, current={torch_version}")
            source = protocol["model_source"]
            require(all(record.get(key) == value for key, value in source.items()),
                    "Original checkpoint metadata differs between record and protocol")
            if model_source is None:
                model_source = source
            require(source == model_source, "Selected cases refer to different checkpoints")
            for name, digest in protocol["source_sha256"].items():
                if name not in code_hashes:
                    code_hashes[name] = file_hash(REPO_ROOT / name)
                require(code_hashes[name] == digest, f"Original inference source changed: {name}")
            cases.append(case)
    checkpoint = Path(model_source["checkpoint_path"])
    stat = checkpoint.stat()
    require(stat.st_size == model_source["checkpoint_size"]
            and stat.st_mtime_ns == model_source["checkpoint_mtime_ns"],
            f"Original checkpoint size or mtime changed: {checkpoint}")
    require(model_source["weights"] == experiment["ar_weights"], "Frozen checkpoint weights differ")
    expected_checkpoint = Path(experiment["ar_checkpoint"])
    if not expected_checkpoint.is_absolute():
        expected_checkpoint = REPO_ROOT / expected_checkpoint
    require(checkpoint.resolve() == expected_checkpoint.resolve(), "Frozen checkpoint path differs")
    return cases, model_source


def verify_latents(latents):
    import torch

    require(set(latents) == KEYS, "New latent artifact has unexpected keys")
    for key, value in latents.items():
        require(isinstance(value, torch.Tensor) and list(value.shape) == LATENT_SHAPE
                and value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()),
                f"Invalid first-chunk latent: {key}")
    require(torch.equal(latents["final"], latents["x0_step4"]), "Final must equal step-four clean estimate")


def check_tensor(actual, expected, label):
    import torch
    from scripts.patch.patch_noise_importance import tensor_hash

    actual_hash, expected_hash = tensor_hash(actual), tensor_hash(expected)
    require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual, expected) and actual_hash == expected_hash,
            f"Exact replay failed for {label}: actual={actual_hash}, original={expected_hash}")
    return {"equal": True, "max_abs": 0.0, "sha256": actual_hash}


def capture_case(pipeline, conditional, case):
    import torch
    from scripts.patch.ar_patch_rollout import TracedRollout, trajectory_metadata
    from scripts.patch.patch_noise_importance import make_noise_bank, tensor_hash

    record = case["record"]
    parameter = next(pipeline.generator.parameters())
    initial, transition = make_noise_bank(SHAPE, CHUNKS, 4, record["seed"],
                                         parameter.dtype, parameter.device)
    noise = {"initial_noise_sha256": tensor_hash(initial),
             "first_chunk_noise_sha256": tensor_hash(initial[:, :3]),
             "transition_noise_sha256": {f"{chunk}:{step}": tensor_hash(value)
                                         for (chunk, step), value in transition.items()}}
    for key, value in noise.items():
        require(value == record[key], f"Original random noise bank differs: {key}")
    replay = TracedRollout(pipeline, conditional, initial, transition, CHUNKS)
    require([float(t) for t in replay.steps] == record["timesteps"], "Original AR timesteps differ")
    # Use the original full two-chunk recurrence, including clean-context writes.
    baseline = replay.baseline(0, 1)
    trace = replay.take_trace()
    chunk0 = [entry for entry in trace if entry["chunk_index"] == 0]
    require([entry["step_index"] for entry in chunk0] == list(range(4)),
            "AR trace has missing or duplicated denoising calls")
    require([entry["sigma"] for entry in chunk0] == record["sigmas"], "Original AR sigmas differ")
    latents = {}
    for step, entry in enumerate(chunk0, 1):
        latents[f"x0_step{step}"] = entry["clean_prediction"]
        latents[f"state_step{step}"] = entry["noisy"]
    latents["final"] = baseline["output"][:, :3].detach().to("cpu", copy=True)
    verify_latents(latents)
    original = torch.load(case["directory"] / "latents.pt", map_location="cpu", weights_only=False)
    checks = {"first": check_tensor(latents["x0_step1"], original["first"], "first"),
              "second": check_tensor(latents["x0_step2"], original["second"], "second"),
              "next_state": check_tensor(latents["state_step2"], original["next_state"], "next_state"),
              "final_first_chunk": check_tensor(latents["final"], original["final"][:, :3], "final first chunk"),
              "final_full_rollout": check_tensor(baseline["output"].cpu(), original["final"], "full final"),
              "state_step1_initial_noise": check_tensor(latents["state_step1"], initial[:, :3].cpu(), "initial state"),
              "final_equals_x0_step4": True, "noise_bank_matches_original": True}
    return latents, dict(noise, timesteps=record["timesteps"], sigmas=record["sigmas"],
                         trajectory=trajectory_metadata(chunk0), checks=checks)


def make_protocol(args, case, model_source, config):
    return {"schema_version": SCHEMA_VERSION, "model": "ar",
            "prompt_index": case["record"]["prompt_index"], "prompt": case["record"]["prompt"],
            "seed": case["record"]["seed"], "source_record": case["source_record"],
            "model_source": model_source, "config": config,
            "frozen_config_sha256": file_hash(args.root / "frozen_config.yaml"),
            "capture_source_sha256": file_hash(__file__),
            "latent_shape": LATENT_SHAPE, "keys": sorted(KEYS),
            "view_protocol": case["record"]["protocol"]["view_protocol"],
            "definition": "Original four-call trajectory; state is pre-call input; x0 is clean estimate; final is first chunk"}


def completed(destination, protocol):
    import numpy as np
    import torch
    from scripts.patch.patch_noise_importance import tensor_hash

    path = destination / "record.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    require(record.get("complete") is True and record.get("protocol") == protocol
            and record.get("protocol_sha256") == json_hash(protocol),
            f"Existing capture is incomplete or has a different protocol: {path}")
    for name in ("latents.pt", "views.npz"):
        require((destination / name).is_file()
                and file_hash(destination / name) == record["artifact_sha256"][name],
                f"Existing captured artifact changed: {destination / name}")
    latents = torch.load(destination / "latents.pt", map_location="cpu", weights_only=False)
    verify_latents(latents)
    require({key: tensor_hash(value) for key, value in latents.items()} == record["tensor_sha256"],
            f"Existing latent hashes differ: {path}")
    with np.load(destination / "views.npz", allow_pickle=False) as views:
        require(set(views.files) == KEYS and all(views[key].shape == VIEW_SHAPE
                and views[key].dtype == np.uint8 for key in views.files), f"Invalid captured views: {path}")
    return True


def publish_case(pipeline, args, case, destination, protocol, latents, metadata, started):
    import numpy as np
    import torch
    from scripts.denoising.compare_first_step_noise import render_views
    from scripts.patch.patch_noise_importance import atomic_json, tensor_hash

    views = render_views(pipeline, {key: value for key, value in latents.items() if key != "final"})
    views["final"] = views["x0_step4"].copy()
    with np.load(case["directory"] / "views.npz", allow_pickle=False) as old_views:
        for current, old in (("x0_step1", "first"), ("x0_step2", "second"), ("final", "final")):
            require(np.array_equal(views[current], old_views[old]), f"Replayed decoded view differs: {old}")
    metadata["checks"]["original_decoded_views_match"] = True
    # Recheck the reference artifacts immediately before publishing this result.
    checked = load_source(args.root, case["record"]["prompt_index"], case["record"]["seed"])
    require(checked["source_record"] == case["source_record"], "Original record changed during capture")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f"latents.{os.getpid()}.tmp"
    torch.save(latents, temporary)
    temporary.replace(destination / "latents.pt")
    temporary = destination / f"views.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **views)
    temporary.replace(destination / "views.npz")
    stages = {}
    for step, (timestep, sigma) in enumerate(zip(metadata["timesteps"], metadata["sigmas"]), 1):
        for prefix, kind in (("x0", "clean_prediction"), ("state", "noisy_state_before_call")):
            stages[f"{prefix}_step{step}"] = {"step_index": step - 1, "step_number": step,
                "timestep": timestep, "sigma": sigma, "kind": kind}
    source = case["record"]
    record = dict(metadata, schema_version=SCHEMA_VERSION, model="ar", prompt_index=source["prompt_index"],
                  prompt=source["prompt"], seed=source["seed"], shape=LATENT_SHAPE, chunks=[3],
                  source_record=case["source_record"], checkpoint_path=source["checkpoint_path"],
                  weights=source["weights"], stages=stages, protocol=protocol, protocol_sha256=json_hash(protocol),
                  final_definition="First chunk of the original final rollout; bitwise equal to x0_step4",
                  artifacts={"latents": "latents.pt", "views": "views.npz"},
                  view_path="views.npz", view_keys=sorted(KEYS),
                  tensor_sha256={key: tensor_hash(value) for key, value in latents.items()},
                  artifact_sha256={name: file_hash(destination / name) for name in ("latents.pt", "views.npz")},
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  elapsed_seconds=time.monotonic() - started, complete=True)
    atomic_json(destination / "record.json", record)
    return record


def run(args):
    # Keep --help usable in the control environment without the GPU dependencies.
    import torch
    from omegaconf import OmegaConf

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.patch.patch_noise_importance import load_pipeline

    os.chdir(REPO_ROOT)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                             OmegaConf.load(args.root / "frozen_config.yaml"))
    resolved = OmegaConf.to_container(config, resolve=True)
    cases, model_source = validate_sources(args, resolved, torch.__version__)
    pending = []
    for case in cases:
        source = case["record"]
        destination = args.root / "denoising_steps/ar" / f"prompt_{source['prompt_index']:06d}" / f"seed_{source['seed']:06d}"
        protocol = make_protocol(args, case, model_source, resolved)
        if completed(destination, protocol):
            print(f"SKIP ar prompt={source['prompt_index']} seed={source['seed']}", flush=True)
        else:
            pending.append((case, destination, protocol))
    if not pending:
        return
    require(torch.cuda.is_available(), "Capture requires the original CUDA/PyTorch GPU environment")
    device = torch.device(args.device)
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(4)
    loader_args = SimpleNamespace(config_path=args.root / "frozen_config.yaml",
        checkpoint_path=Path(model_source["checkpoint_path"]), weights=model_source["weights"],
        num_output_frames=6, source_chunk=0, source_frame=1, steps=[0, 1, 2, 3],
        ranking_metric="normalized_other_frames_energy")
    with torch.inference_mode():
        pipeline, loaded_config, shape, chunks, _ = load_pipeline(loader_args)
        require(shape == SHAPE and chunks == CHUNKS and len(pipeline.denoising_step_list) == 4,
                "Frozen pipeline must use two three-frame chunks and exactly four denoising calls")
        require(OmegaConf.to_container(loaded_config, resolve=True) == resolved, "Loaded config differs")
        cached_prompt, conditional = None, None
        for case, destination, protocol in pending:
            source = case["record"]
            started = time.monotonic()
            print(f"START ar prompt={source['prompt_index']} seed={source['seed']}", flush=True)
            if cached_prompt != source["prompt"]:
                parameter = next(pipeline.generator.parameters())
                pipeline.text_encoder.to(parameter.device)
                conditional = pipeline.text_encoder(text_prompts=[source["prompt"]])
                pipeline.text_encoder.cpu()
                torch.cuda.empty_cache()
                cached_prompt = source["prompt"]
            latents, metadata = capture_case(pipeline, conditional, case)
            record = publish_case(pipeline, args, case, destination, protocol, latents, metadata, started)
            print(f"DONE ar prompt={source['prompt_index']} seed={source['seed']} "
                  f"seconds={record['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
