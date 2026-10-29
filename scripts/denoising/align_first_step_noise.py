#!/usr/bin/env python3
"""Supplement the first-step experiment at the AR post-update noise level.

AR uses its saved re-noised state and second clean estimate. Wan replays only
the first twelve native Euler updates, then probes at sigma=0.9375. Matching
sigma does not imply that the two samplers follow the same random path.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.denoising.compare_first_step_noise import CachedNativeWan, SHAPE, VIEW_SHAPE, render_views
from scripts.patch.patch_noise_importance import atomic_json, tensor_hash


TARGET_SIGMA = 0.9375
LATENT_SHAPE = [1, 3, 16, 60, 104]
SCHEMA_VERSION = "first_step_noise_aligned_v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("ar", "wan"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards and num-shards >= 1")
    return args


def load_source(path, model):
    record = json.loads(path.read_text())
    if record.get("model") != model or record.get("complete") is not True:
        raise ValueError(f"Invalid or incomplete source record: {path}")
    for name in ("latents.pt", "views.npz"):
        artifact = path.parent / name
        expected = record.get("artifact_sha256", {}).get(name)
        if not expected or not artifact.is_file() or file_hash(artifact) != expected:
            raise ValueError(f"Missing or changed source artifact: {artifact}")
    stage_name = "second" if model == "ar" else "quarter"
    expected_index = 1 if model == "ar" else 12
    stage = record["stages"][stage_name]
    if stage["sigma"] != TARGET_SIGMA or stage["step_index"] != expected_index:
        raise ValueError(f"Source stage does not match target sigma/index: {path}")
    return record, file_hash(path)


def valid_completed(destination, source, source_hash):
    path = destination / "record.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    expected = {"schema_version": SCHEMA_VERSION, "complete": True,
                "model": source["model"], "prompt_index": source["prompt_index"],
                "seed": source["seed"], "sigma": TARGET_SIGMA,
                "source_record_sha256": source_hash,
                "source_noise_sha256": source["initial_noise_sha256"]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing aligned record belongs to another protocol: {path}")
    for name in ("latents.pt", "views.npz"):
        artifact = destination / name
        if not artifact.is_file() or file_hash(artifact) != record["artifact_sha256"][name]:
            raise ValueError(f"Missing or changed aligned artifact: {artifact}")
    keys = {"aligned_state", "aligned_x0"}
    if source["model"] == "wan":
        keys.add("aligned_cfg1")
    with np.load(destination / "views.npz") as arrays:
        if set(arrays.files) != keys or any(arrays[key].shape != VIEW_SHAPE or
                                            arrays[key].dtype != np.uint8 for key in keys):
            raise ValueError(f"Invalid aligned view schema: {destination}")
    return True


def initial_noise(seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=rng, dtype=torch.float32).to(torch.bfloat16)


def ar_case(path, source):
    saved = torch.load(path.parent / "latents.pt", map_location="cpu", weights_only=True)
    latents = {"aligned_state": saved["next_state"], "aligned_x0": saved["second"]}
    with np.load(path.parent / "views.npz") as arrays:
        views = {"aligned_state": arrays["next_state"].copy(),
                 "aligned_x0": arrays["second"].copy()}
    initial = initial_noise(source["seed"])
    if tensor_hash(initial) != source["initial_noise_sha256"]:
        raise ValueError(f"Reconstructed initial noise differs: {path}")
    checks = {"initial_noise_match": True, "source_latent_mapping": {
        "aligned_state": "next_state", "aligned_x0": "second"}}
    return latents, views, checks


def wan_case(pipeline, path, source):
    sigmas = source["sigmas"]
    if len(sigmas) != 49 or float(sigmas[12]) != TARGET_SIGMA:
        raise ValueError("Expected the original 48-update Wan schedule with target at index 12")
    initial = initial_noise(source["seed"])
    if tensor_hash(initial) != source["initial_noise_sha256"]:
        raise ValueError(f"Reconstructed initial noise differs: {path}")
    if tensor_hash(initial[:, :3]) != source["first_chunk_noise_sha256"]:
        raise ValueError(f"Reconstructed first three noise frames differ: {path}")
    # Recreate the original bf16 initial tensor BEFORE promotion to float32.
    state = initial.to(pipeline.device).float()
    for index in range(12):
        sigma = float(sigmas[index])
        flow = pipeline.predict(state, sigma)
        state = state + (float(sigmas[index + 1]) - sigma) * flow
    positive, negative = pipeline.predict_parts(state, TARGET_SIGMA)
    flow = negative + pipeline.guidance * (positive - negative)
    x0 = state - TARGET_SIGMA * flow
    latents = {"aligned_state": state[:, :3].cpu(),
               "aligned_x0": x0[:, :3].cpu(),
               "aligned_cfg1": (state - TARGET_SIGMA * positive)[:, :3].cpu()}
    saved = torch.load(path.parent / "latents.pt", map_location="cpu", weights_only=True)
    original = saved["quarter"][:, :3].float()
    delta = float((latents["aligned_x0"].float() - original).abs().max())
    if delta != 0.0:
        raise RuntimeError(f"Wan replay differs from saved quarter prediction: max_abs={delta}")
    checks = {"initial_noise_match": True, "first_chunk_noise_match": True,
              "source_quarter_x0_max_abs": delta,
              "source_quarter_x0_sha256": tensor_hash(original),
              "replayed_aligned_x0_sha256": tensor_hash(latents["aligned_x0"])}
    return latents, render_views(pipeline, latents), checks


def save_case(args, path, source, source_hash, destination, latents, views, checks, started):
    for key, value in latents.items():
        if list(value.shape) != LATENT_SHAPE or not torch.isfinite(value).all():
            raise ValueError(f"Invalid aligned latent: {key}")
    for key, value in views.items():
        if value.shape != VIEW_SHAPE or value.dtype != np.uint8:
            raise ValueError(f"Invalid aligned view: {key}")
    if file_hash(path) != source_hash:
        raise ValueError(f"Source record changed during sampling: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_latents = destination / f"latents.{os.getpid()}.tmp"
    torch.save(latents, temporary_latents)
    temporary_latents.replace(destination / "latents.pt")
    temporary_views = destination / f"views.{os.getpid()}.tmp"
    with temporary_views.open("wb") as stream:
        np.savez_compressed(stream, **views)
    temporary_views.replace(destination / "views.npz")
    updates = 12 if args.model == "wan" else 1
    stages = {key: {"sigma": TARGET_SIGMA, "step_index": updates,
                    "completed_updates": updates,
                    "kind": "noisy_state" if key == "aligned_state" else "clean_prediction"}
              for key in latents}
    record = {
        "schema_version": SCHEMA_VERSION, "model": args.model,
        "prompt_index": source["prompt_index"], "prompt": source["prompt"],
        "seed": source["seed"], "sigma": TARGET_SIGMA,
        "step_index": updates, "completed_updates": updates,
        "denoiser_probe_ordinal": updates + 1,
        "shape": LATENT_SHAPE, "stages": stages,
        "source_record": str(path.resolve()), "source_record_sha256": source_hash,
        "source_artifact_sha256": source["artifact_sha256"],
        "source_protocol_sha256": source["protocol_sha256"],
        "source_noise_sha256": source["initial_noise_sha256"],
        "initial_noise_sha256": source["initial_noise_sha256"],
        "first_chunk_noise_sha256": source["first_chunk_noise_sha256"],
        "guidance": source["protocol"]["guidance"],
        "alignment": "same nominal sigma, prompt and initial noise; different sampler paths",
        "sampler": ("native Wan CFG6 Euler trajectory; 12 completed Euler updates; 13th denoiser probe"
                    if args.model == "wan" else
                    "AR saved first-chunk state after one update and re-noising; second denoiser probe"),
        "cfg1_scope": ("conditional x0 probe on the current CFG6 trajectory state; not a full CFG1 trajectory"
                       if args.model == "wan" else None),
        "view_protocol": source["protocol"]["view_protocol"],
        "replay_checks": checks,
        "artifacts": {"latents": "latents.pt", "views": "views.npz"},
        "view_path": "views.npz", "view_keys": sorted(views),
        "artifact_sha256": {name: file_hash(destination / name)
                            for name in ("latents.pt", "views.npz")},
        "latent_sha256": {key: tensor_hash(value) for key, value in latents.items()},
        "code_sha256": {str(Path(__file__).resolve()): file_hash(__file__)},
        "host": socket.gethostname(), "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "complete": True,
    }
    atomic_json(destination / "record.json", record)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    paths = sorted((args.output_dir / args.model).glob("prompt_*/seed_*/record.json"))
    if not paths:
        raise ValueError(f"No base records found in {args.output_dir / args.model}")
    pending = []
    for path in paths[args.shard_index::args.num_shards]:
        source, source_hash = load_source(path, args.model)
        destination = args.output_dir / "aligned" / args.model / path.parent.parent.name / path.parent.name
        if valid_completed(destination, source, source_hash):
            print(f"SKIP {args.model} prompt={source['prompt_index']} seed={source['seed']}", flush=True)
        else:
            pending.append((path, source, source_hash, destination))
    if not pending:
        print("No unfinished cases assigned to this shard", flush=True)
        return
    pipeline = None
    if args.model == "wan":
        if not torch.cuda.is_available():
            raise RuntimeError("Wan supplementation requires a CUDA GPU")
        protocol = pending[0][1]["protocol"]
        settings = (protocol["config"]["negative_prompt"], protocol["guidance"], protocol["shift"])
        if settings[1] != 6.0:
            raise ValueError("This aligned diagnostic expects the original CFG6 trajectory")
        for _, source, _, _ in pending:
            item = source["protocol"]
            if (item["config"]["negative_prompt"], item["guidance"], item["shift"]) != settings:
                raise ValueError("Assigned Wan cases have incompatible model settings")
        pipeline = CachedNativeWan([source["prompt"] for _, source, _, _ in pending], *settings)
    for path, source, source_hash, destination in pending:
        started = time.monotonic()
        print(f"START {args.model} prompt={source['prompt_index']} seed={source['seed']}", flush=True)
        if args.model == "ar":
            latents, views, checks = ar_case(path, source)
        else:
            pipeline.select_prompt(source["prompt"])
            latents, views, checks = wan_case(pipeline, path, source)
            torch.cuda.synchronize()
        save_case(args, path, source, source_hash, destination, latents, views, checks, started)
        print(f"DONE {args.model} prompt={source['prompt_index']} seed={source['seed']} "
              f"sigma={TARGET_SIGMA} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
