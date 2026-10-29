#!/usr/bin/env python3
"""Sample paired-seed AR/Wan x0 distributions at sigma=0.9375, without final rollouts.

Only the first three latent frames are exported. Wan follows its original six
frame Euler trajectory through 12 updates; AR follows its first update and
re-noising. Their next clean predictions therefore have the same nominal sigma.
"""

import argparse
import gc
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

from scripts.denoising.compare_first_step_noise import (
    CachedNativeWan, CHUNKS, CODE_FILES, SHAPE, file_hash, json_hash,
)
from scripts.patch.patch_noise_importance import (
    PairedRollout, atomic_json, load_pipeline, make_noise_bank, tensor_hash,
)
from scripts.patch.wan_patch_probe import model_manifest, shifted_sigmas


SCHEMA_VERSION = "diversity_moments_sampling_v1"
TARGET_SIGMA = 0.9375
FEATURE_SPACE = "wan_vae_raw_x0_first3latents"
LATENT_SHAPE = [3, 16, 60, 104]
TEACHER_MODEL = "Wan2.1-T2V-14B"
NATIVE_MODELS = ("wan", "wan14b")


class TrainingTeacherWan(CachedNativeWan):
    """The DMD real-score model with the training text encoder and native CFG.

    Prompt selection and flow prediction are inherited unchanged from the old
    native Wan baseline. Only the model loader is different.
    """

    def __init__(self, prompts, negative_prompt, guidance, shift):
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

        self.device = torch.device("cuda")
        self.guidance = guidance
        print("LOAD and cache training WanTextEncoder embeddings", flush=True)
        encoder = WanTextEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.embeddings = {}
        for prompt in dict.fromkeys([*prompts, negative_prompt]):
            self.embeddings[prompt] = encoder(text_prompts=[prompt])["prompt_embeds"].cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        self.unconditional = self.embeddings[negative_prompt].to(self.device)
        self.conditional = self.embeddings[prompts[0]].to(self.device)
        print(f"LOAD training teacher {TEACHER_MODEL}", flush=True)
        self.generator = WanDiffusionWrapper(model_name=TEACHER_MODEL, is_causal=False,
                                            timestep_shift=shift).to(device=self.device, dtype=torch.bfloat16)
        self.generator.eval().requires_grad_(False)
        self.generator.seq_len = SHAPE[1] * (SHAPE[-2] // 2) * (SHAPE[-1] // 2)
        self.vae = torch.nn.Identity()


def training_teacher_alignment(args, config):
    if getattr(config, "real_name", None) != TEACHER_MODEL:
        raise ValueError(f"wan14b requires training real_name={TEACHER_MODEL}")
    key = "real_guidance_scale" if "real_guidance_scale" in config else "guidance_scale"
    training_guidance = float(config[key])
    if not math.isfinite(training_guidance) or training_guidance < 0:
        raise ValueError("Training real guidance must be finite and nonnegative")
    native_guidance = training_guidance + 1.0
    if args.guidance is not None and args.guidance != native_guidance:
        raise ValueError(f"Training teacher alignment requires native guidance={native_guidance}, "
                         f"derived from {key}={training_guidance}")
    negative_prompt = str(config.negative_prompt)
    return {
        "teacher_model": TEACHER_MODEL,
        "training_guidance_key": key,
        "training_guidance_scale": training_guidance,
        "training_cfg_formula": "conditional_x0 + training_guidance_scale * (conditional_x0 - unconditional_x0)",
        "native_guidance_scale": native_guidance,
        "native_cfg_formula": "unconditional_flow + native_guidance_scale * (conditional_flow - unconditional_flow)",
        "negative_prompt": negative_prompt,
        "negative_prompt_sha256": hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
        "text_encoder": "utils.wan_wrapper.WanTextEncoder (same loader as training)",
        "scope": "teacher-native 6-frame Euler trajectory at matched sigma; not a replay of DMD score-loss inputs",
    }


def teacher_model_manifest():
    """Validate the same local sharded teacher used by WanDiffusionWrapper."""
    directory = Path("wan_models") / TEACHER_MODEL
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("Invalid sharded teacher weight index")
    for name in shards:
        if not (directory / name).is_file():
            raise ValueError(f"Missing training teacher shard: {name}")
    if not (directory / "config.json").is_file():
        raise ValueError("Missing training teacher config.json")
    files = []
    for path in sorted(directory.iterdir()):
        if path.suffix not in (".safetensors", ".pth", ".bin", ".json"):
            continue
        stat = path.stat()
        record = {"filename": path.name, "path": str(path.resolve()),
                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".json":
            record["sha256"] = file_hash(path)
        files.append(record)
    return files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("ar", "ode", "cf_init", "cf", "sgf", "wan", "wan14b"), required=True,
                        help="ar/ode/cf_init/cf/sgf use the same causal sampler with --checkpoint-path")
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=[42, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(64)))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--config-path", type=Path,
                        default=Path("outputs/denoising/first_step_noise_20261023/frozen_config.yaml"))
    parser.add_argument("--checkpoint-path", type=Path,
                        default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    parser.add_argument("--weights", choices=("generator", "generator_ema"), default="generator")
    parser.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    parser.add_argument("--guidance", type=float,
                        help="Native CFG; defaults to 6 for wan, training real guidance + 1 for wan14b")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--wan-steps", type=int, default=48)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--reuse-only", action="store_true",
                        help="Validate/export reusable cases on CPU; do not start models")
    args = parser.parse_args(argv)
    if args.guidance is None and args.model != "wan14b":
        args.guidance = 6.0
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards and num-shards >= 1")
    for key in ("prompt_indices", "seeds"):
        values = getattr(args, key)
        if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
            parser.error(f"{key} must contain distinct nonnegative integers")
    if any(seed >= 2 ** 63 for seed in args.seeds):
        parser.error("seeds must be smaller than 2**63")
    if (args.wan_steps != 48 or args.shift != 5.0 or (args.guidance is not None
            and (not math.isfinite(args.guidance) or args.guidance < 0))):
        parser.error("This aligned protocol requires wan-steps=48, shift=5, and finite guidance >=0")
    if not args.config_path.is_file() or not args.data_path.is_file():
        parser.error("config-path and data-path must exist")
    prompts = args.data_path.read_text().splitlines()
    if any(index >= len(prompts) or not prompts[index].strip() for index in args.prompt_indices):
        parser.error("prompt index is out of range or empty")
    checkpoint = args.checkpoint_path / "model.pt" if args.checkpoint_path.is_dir() else args.checkpoint_path
    if args.model in ("ar", "ode", "cf_init", "cf", "sgf") and not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    args.num_output_frames, args.source_chunk, args.source_frame = 6, 0, 1
    args.steps = [0, 1]
    args.ranking_metric = "normalized_other_frames_energy"
    return args


def initial_noise(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(SHAPE, generator=generator, dtype=torch.float32).to(torch.bfloat16)


def noise_metadata(initial):
    digest = tensor_hash(initial)
    return {"noise_sha256": digest, "initial_noise_sha256": digest,
            "first_chunk_noise_sha256": tensor_hash(initial[:, :3])}


def common_protocol(args, config):
    from omegaconf import OmegaConf

    checkpoint = args.checkpoint_path / "model.pt" if args.checkpoint_path.is_dir() else args.checkpoint_path
    if args.model in ("ar", "ode", "cf_init", "cf", "sgf"):
        stat = checkpoint.stat()
        source = {"checkpoint_path": str(checkpoint.resolve()), "checkpoint_size": stat.st_size,
                  "checkpoint_mtime_ns": stat.st_mtime_ns, "weights": args.weights}
        if args.model == "cf_init":
            source["model_label"] = "Causal Forcing init"
            source["checkpoint_role"] = "chunkwise causal_ode Stage 2 initialization"
        elif args.model in ("cf", "sgf"):
            source["model_label"] = {"cf": "Causal Forcing final", "sgf": "Self Gradient Forcing final"}[args.model]
            source["checkpoint_role"] = {
                "cf": "chunkwise causal_forcing final DMD checkpoint",
                "sgf": "chunkwise ar final checkpoint",
            }[args.model]
            source["evaluation_scope"] = "shared first 3-latent-frame causal block; no historical context"
            source["evaluation_config"] = "same frozen first-chunk configuration and noise protocol as AR"
    elif args.model == "wan14b":
        source = {"checkpoint_path": str((Path("wan_models") / TEACHER_MODEL /
                                          "diffusion_pytorch_model.safetensors.index.json").resolve()),
                  "weights": "pretrained", "model_name": TEACHER_MODEL,
                  "model_manifest": teacher_model_manifest(),
                  "text_encoder": "utils.wan_wrapper.WanTextEncoder (same loader as training)"}
    else:
        source = {"checkpoint_path": str(Path(
            "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors").resolve()),
            "weights": "pretrained", "model_manifest": model_manifest()}
    protocol = {
        "schema_version": SCHEMA_VERSION, "model": args.model,
        "shape": SHAPE, "chunks": CHUNKS, "feature_shape": LATENT_SHAPE,
        "feature_space": FEATURE_SPACE, "sigma": TARGET_SIGMA,
        "model_source": source, "config": OmegaConf.to_container(config, resolve=True),
        "config_sha256": file_hash(args.config_path),
        "guidance": args.guidance if args.model in NATIVE_MODELS else 1.0,
        "wan_steps": args.wan_steps if args.model in NATIVE_MODELS else None,
        "shift": args.shift if args.model in NATIVE_MODELS else None,
        "noise_protocol": "CPU float32 randn cast to bfloat16, independent per seed",
        "ar_transition_protocol": "same CPU generator after initial 6-frame noise; original noise bank order",
        "sampler": ("native Wan CFG Euler; 12 completed updates; 13th denoiser probe"
                    if args.model in NATIVE_MODELS else "AR first chunk; 1 update and re-noising; 2nd denoiser probe"),
        "completed_updates": 12 if args.model in NATIVE_MODELS else 1,
        "source_sha256": {name: file_hash(name) for name in [*CODE_FILES, str(Path(__file__).relative_to(Path.cwd()))]},
        "torch_version": torch.__version__,
    }
    if args.model == "wan14b":
        protocol["teacher_training_alignment"] = training_teacher_alignment(args, config)
    return protocol


def validate_completed(destination, protocol):
    path = destination / "record.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    if record.get("protocol") != protocol or record.get("protocol_sha256") != json_hash(protocol):
        raise ValueError(f"Existing case belongs to another protocol: {path}")
    if not record.get("complete"):
        return False
    for name, expected in record.get("artifact_sha256", {}).items():
        artifact = destination / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise ValueError(f"Missing or changed completed artifact: {artifact}")
    if "features.npz" not in record.get("artifact_sha256", {}):
        raise ValueError(f"Missing feature artifact hash: {path}")
    with np.load(destination / "features.npz") as features:
        latent = features["latent"]
        if set(features.files) != {"latent"} or list(latent.shape) != LATENT_SHAPE or latent.dtype != np.float32:
            raise ValueError(f"Invalid feature schema: {destination}")
        if not np.isfinite(latent).all():
            raise ValueError(f"Nonfinite features: {destination}")
    expected_noise = noise_metadata(initial_noise(protocol["seed"]))
    if any(record.get(key) != value for key, value in expected_noise.items()):
        raise ValueError(f"Resume noise differs: {path}")
    return True


def checked_artifacts(directory, record, names):
    for name in names:
        path = directory / name
        expected = record.get("artifact_sha256", {}).get(name)
        if not expected or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Missing or modified reuse artifact: {path}")


def load_reusable(args, protocol):
    # Aligned legacy exports contain the trained AR and pretrained Wan only.
    # Additional checkpoints must be sampled from their own weights.
    if args.reuse_root is None or args.model in ("ode", "cf_init", "cf", "sgf", "wan14b"):
        return None
    index, seed = protocol["prompt_index"], protocol["seed"]
    relative = Path(args.model) / f"prompt_{index:06d}" / f"seed_{seed:06d}"
    directory = args.reuse_root / "aligned" / relative
    path = directory / "record.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if (record.get("schema_version") != "first_step_noise_aligned_v1"
            or not record.get("complete") or record.get("sigma") != TARGET_SIGMA
            or record.get("completed_updates") != protocol["completed_updates"]
            or any(record.get(key) != protocol[key] for key in ("model", "prompt_index", "prompt", "seed", "guidance"))):
        raise ValueError(f"Reuse case identity or aligned stage differs: {path}")
    checked_artifacts(directory, record, ("latents.pt", "views.npz"))
    source_path = args.reuse_root / relative / "record.json"
    source = json.loads(source_path.read_text())
    if file_hash(source_path) != record.get("source_record_sha256") or not source.get("complete"):
        raise ValueError(f"Reuse original source record was modified or incomplete: {source_path}")
    source_protocol = source["protocol"]
    if json_hash(source_protocol) != source.get("protocol_sha256"):
        raise ValueError(f"Reuse source protocol hash differs: {source_path}")
    for key in ("model", "prompt_index", "prompt", "seed", "shape", "chunks", "model_source",
                "config", "guidance", "wan_steps", "shift", "noise_protocol", "torch_version"):
        if source_protocol.get(key) != protocol.get(key):
            raise ValueError(f"Reuse source parameter {key} differs: {source_path}")
    for name, digest in source_protocol.get("source_sha256", {}).items():
        if name not in protocol["source_sha256"] or protocol["source_sha256"][name] != digest:
            raise ValueError(f"Reuse model/sampler code differs: {name}")
    checked_artifacts(source_path.parent, source, ("latents.pt", "views.npz"))
    metadata = noise_metadata(initial_noise(seed))
    for key in ("initial_noise_sha256", "first_chunk_noise_sha256"):
        if record.get(key) != metadata[key] or source.get(key) != metadata[key]:
            raise ValueError(f"Reuse initial noise differs: {path}")
    saved = torch.load(directory / "latents.pt", map_location="cpu", weights_only=True)
    x0 = saved["aligned_x0"]
    if tensor_hash(x0) != record.get("latent_sha256", {}).get("aligned_x0"):
        raise ValueError(f"Reuse aligned_x0 hash differs: {path}")
    if args.model == "ar":
        saved_source = torch.load(source_path.parent / "latents.pt", map_location="cpu", weights_only=True)
        if not torch.equal(x0, saved_source["second"]):
            raise ValueError(f"AR reuse second-probe mapping differs: {path}")
        metadata["transition_noise_sha256"] = {"0:1": source["transition_noise_sha256"]["0:1"]}
    else:
        if source["sigmas"] != [float(sigma) for sigma in shifted_sigmas(48, 5.0)]:
            raise ValueError(f"Reuse Wan schedule differs: {source_path}")
        if record["replay_checks"].get("source_quarter_x0_max_abs") != 0:
            raise ValueError(f"Reuse Wan replay was not exact: {path}")
    preview = None
    if seed < 8:
        with np.load(directory / "views.npz") as views:
            preview = views["aligned_x0"][1].copy()
        if preview.shape != (240, 416, 3) or preview.dtype != np.uint8:
            raise ValueError(f"Invalid reuse preview: {directory}")
    metadata["reuse_source"] = {
        "record": str(path.resolve()), "record_sha256": file_hash(path),
        "artifact_sha256": record["artifact_sha256"],
        "original_record": str(source_path.resolve()), "original_record_sha256": file_hash(source_path),
        "checks": "original and aligned artifact hashes, model source/config/code, exact sigma, prompt and paired noise",
    }
    return x0, preview, metadata


def save_case(destination, protocol, x0, metadata, preview, started):
    latent = x0.detach().float().cpu().numpy()
    if list(latent.shape) == [1, *LATENT_SHAPE]:
        latent = latent[0]
    if list(latent.shape) != LATENT_SHAPE or not np.isfinite(latent).all():
        raise ValueError("Invalid clean latent prediction")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f"features.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, latent=latent)
    temporary.replace(destination / "features.npz")
    artifacts = {"features.npz": file_hash(destination / "features.npz")}
    if preview is not None:
        from PIL import Image
        temporary_image = destination / f"preview.{os.getpid()}.tmp"
        Image.fromarray(preview).save(temporary_image, format="JPEG", quality=90)
        temporary_image.replace(destination / "preview.jpg")
        artifacts["preview.jpg"] = file_hash(destination / "preview.jpg")
    record = dict(metadata, schema_version=SCHEMA_VERSION, model=protocol["model"],
                  prompt_index=protocol["prompt_index"], prompt=protocol["prompt"], seed=protocol["seed"],
                  sigma=TARGET_SIGMA, feature_space=FEATURE_SPACE, shape=LATENT_SHAPE,
                  completed_updates=protocol["completed_updates"],
                  denoiser_probe_ordinal=protocol["completed_updates"] + 1,
                  guidance=protocol["guidance"], model_source=protocol["model_source"],
                  code_sha256=protocol["source_sha256"], protocol=protocol,
                  protocol_sha256=json_hash(protocol), artifact_sha256=artifacts,
                  latent_sha256=hashlib.sha256(np.ascontiguousarray(latent).tobytes()).hexdigest(),
                  host=socket.gethostname(), torch_version=torch.__version__,
                  elapsed_seconds=time.monotonic() - started, complete=True)
    atomic_json(destination / "record.json", record)


def run_ar_case(pipeline, conditional, seed):
    parameter = next(pipeline.generator.parameters())
    if parameter.dtype != torch.bfloat16:
        raise ValueError("Paired-noise protocol requires bfloat16")
    initial, transition = make_noise_bank(SHAPE, CHUNKS, len(pipeline.denoising_step_list),
                                         seed, parameter.dtype, parameter.device)
    replay = PairedRollout(pipeline, conditional, initial, transition, CHUNKS)
    index = (pipeline.scheduler.timesteps - float(replay.steps[1])).abs().argmin()
    sigma = float(pipeline.scheduler.sigmas[index])
    if sigma != TARGET_SIGMA:
        raise ValueError(f"AR next-probe sigma differs: {sigma}")
    first = replay.predict(initial[:, :3], 0, 0)
    state = replay.renoise(first, 0, 1)
    x0 = replay.predict(state, 0, 1)
    metadata = noise_metadata(initial)
    metadata["transition_noise_sha256"] = {"0:1": tensor_hash(transition[0, 1])}
    metadata["timesteps"] = [float(step) for step in replay.steps]
    return x0, metadata


def run_wan_case(pipeline, seed, sigmas):
    initial = initial_noise(seed)
    metadata = noise_metadata(initial)
    state = initial.to(pipeline.device).float()
    if float(sigmas[12]) != TARGET_SIGMA:
        raise ValueError("Wan stage 12 must have sigma=0.9375")
    for index in range(12):
        sigma = float(sigmas[index])
        flow = pipeline.predict(state, sigma)
        state = state + (float(sigmas[index + 1]) - sigma) * flow
    flow = pipeline.predict(state, TARGET_SIGMA)
    x0 = (state - TARGET_SIGMA * flow)[:, :3]
    metadata["sigmas"] = [float(sigma) for sigma in sigmas]
    return x0, metadata


def preview_frame(pipeline, x0):
    from scripts.patch.patch_noise_strength import decode_pixels
    import torch.nn.functional as F

    parameter = next(pipeline.generator.parameters())
    pixels = decode_pixels(pipeline, x0.to(parameter.device, parameter.dtype))
    frame = F.interpolate(pixels[4:5].permute(0, 3, 1, 2), size=(240, 416), mode="area")
    return frame[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    if not config.mixed_precision:
        raise ValueError("Paired noise requires mixed_precision=true")
    if args.model == "wan14b":
        alignment = training_teacher_alignment(args, config)
        args.guidance = alignment["native_guidance_scale"]
    prompts = args.data_path.read_text().splitlines()
    common = common_protocol(args, config)
    cases = [(index, seed) for index in args.prompt_indices for seed in args.seeds]
    pending = []
    for index, seed in cases[args.shard_index::args.num_shards]:
        protocol = dict(common, prompt_index=index, prompt=prompts[index].strip(), seed=seed)
        destination = args.output_dir / args.model / f"prompt_{index:06d}" / f"seed_{seed:06d}"
        if validate_completed(destination, protocol):
            print(f"SKIP {args.model} prompt={index} seed={seed}", flush=True)
            continue
        started = time.monotonic()
        reused = load_reusable(args, protocol)
        if reused is not None:
            x0, preview, metadata = reused
            save_case(destination, protocol, x0, metadata, preview, started)
            print(f"REUSE {args.model} prompt={index} seed={seed}", flush=True)
        else:
            pending.append((index, seed, protocol, destination))
    if args.reuse_only or not pending:
        print(f"FINISHED model={args.model} pending={len(pending)} reuse_only={args.reuse_only}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run pending sampling in the GPU dev environment")
    if args.model in ("ar", "ode", "cf_init", "cf", "sgf"):
        pipeline, _, shape, chunks, _ = load_pipeline(args)
        if shape != SHAPE or chunks != CHUNKS:
            raise ValueError("Expected original six-frame / two-chunk AR protocol")
    else:
        native_class = TrainingTeacherWan if args.model == "wan14b" else CachedNativeWan
        pipeline = native_class([case[2]["prompt"] for case in pending],
                                config.negative_prompt, args.guidance, args.shift)
    current_prompt, conditional = None, None
    sigmas = shifted_sigmas(args.wan_steps, args.shift)
    for index, seed, protocol, destination in pending:
        started = time.monotonic()
        print(f"START {args.model} prompt={index} seed={seed}", flush=True)
        if args.model in ("ar", "ode", "cf_init", "cf", "sgf"):
            if current_prompt != protocol["prompt"]:
                parameter = next(pipeline.generator.parameters())
                pipeline.text_encoder.to(parameter.device)
                conditional = pipeline.text_encoder(text_prompts=[protocol["prompt"]])
                pipeline.text_encoder.cpu()
                torch.cuda.empty_cache()
                current_prompt = protocol["prompt"]
            x0, metadata = run_ar_case(pipeline, conditional, seed)
        else:
            pipeline.select_prompt(protocol["prompt"])
            x0, metadata = run_wan_case(pipeline, seed, sigmas)
        preview = preview_frame(pipeline, x0) if seed < 8 else None
        save_case(destination, protocol, x0, metadata, preview, started)
        print(f"DONE {args.model} prompt={index} seed={seed} seconds={time.monotonic()-started:.2f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
