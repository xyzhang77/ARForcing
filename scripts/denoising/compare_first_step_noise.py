#!/usr/bin/env python3
"""Paired-noise AR / native bidirectional Wan first-step experiment.

Each seed uses the same CPU float32 -> bfloat16 initial noise in both models.
The clean predictions and the actual next noisy state are distinct artifacts.
All previews independently decode exactly the first three latent frames.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.ar_patch_rollout import TracedRollout
from scripts.patch.patch_noise_importance import atomic_json, load_pipeline, make_noise_bank, tensor_hash
from scripts.patch.patch_noise_strength import decode_pixels
from scripts.patch.wan_patch_probe import NativeWan, model_manifest, shifted_sigmas


SHAPE = [1, 6, 16, 60, 104]
CHUNKS = [3, 3]
VIEW_SHAPE = (3, 240, 416, 3)
SCHEMA_VERSION = 1
BASE_KEYS = {"first", "second", "final", "next_state"}
WAN_KEYS = {"first_cfg1", "first_horizon3", "first_horizon3_cfg1",
            "quarter", "half", "threequarter"}
AR_KEYS = {"final_alt_transition"}
CODE_FILES = [
    "scripts/denoising/compare_first_step_noise.py", "scripts/patch/ar_patch_rollout.py",
    "scripts/patch/patch_noise_importance.py", "scripts/patch/patch_noise_strength.py",
    "scripts/patch/wan_patch_probe.py", "pipeline/causal_inference.py",
    "utils/wan_wrapper.py", "wan/modules/causal_model.py", "wan/modules/model.py",
    "wan/modules/vae.py",
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("ar", "wan"), required=True)
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=[42, 158, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    parser.add_argument("--checkpoint-path", type=Path,
                        default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    parser.add_argument("--weights", choices=("generator", "generator_ema"), default="generator_ema")
    parser.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    parser.add_argument("--wan-steps", type=int, default=48)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    for name in ("prompt_indices", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values) or any(value < 0 for value in values):
            parser.error(f"{name.replace('_', '-')} must contain distinct nonnegative integers")
    if any(seed >= 2 ** 63 for seed in args.seeds):
        parser.error("seeds must be smaller than 2**63")
    if args.wan_steps < 4:
        parser.error("wan-steps must be >= 4 to capture three distinct intermediate stages")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        parser.error("guidance must be finite and nonnegative")
    if not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("shift must be finite and positive")
    if not args.config_path.is_file() or not args.data_path.is_file():
        parser.error("config-path and data-path must be existing files")
    checkpoint = args.checkpoint_path / "model.pt" if args.checkpoint_path.is_dir() else args.checkpoint_path
    if args.model == "ar" and not checkpoint.is_file():
        parser.error(f"AR checkpoint not found: {checkpoint}")
    prompts = args.data_path.read_text().splitlines()
    if any(index >= len(prompts) or not prompts[index].strip() for index in args.prompt_indices):
        parser.error("prompt index is out of range or refers to an empty prompt")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("output-dir must be a directory")
    # The shared loader validates the original inference configuration.
    args.num_output_frames, args.source_chunk, args.source_frame = 6, 0, 1
    args.steps = [0, 1]
    args.ranking_metric = "normalized_other_frames_energy"
    return args


class IdentityTracedRollout(TracedRollout):
    """Repeat only the first denoiser call while preserving the baseline trace."""

    def __init__(self, *args, check_identity=False, **kwargs):
        self.check_identity = check_identity
        self.identity = None
        super().__init__(*args, **kwargs)

    def predict(self, noisy, chunk, step=None, context=False):
        prediction = super().predict(noisy, chunk, step, context)
        if self.check_identity and self.identity is None and not context:
            self.reset_to_chunk(0)
            self.capture = False
            try:
                repeated = super().predict(noisy, chunk, step, context)
            finally:
                self.capture = True
            delta = float((prediction.float() - repeated.float()).abs().max())
            self.identity = {"first_call_repeat_max_abs": delta,
                             "scope": "first chunk, first denoiser call; no rollout repeat"}
            if not math.isfinite(delta):
                raise RuntimeError("Nonfinite AR identity difference")
        return prediction


class CachedNativeWan(NativeWan):
    """NativeWan with one encoder load and cached embeddings for all prompts."""

    def __init__(self, prompts, negative_prompt, guidance, shift):
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

        self.device = torch.device("cuda")
        self.guidance = guidance
        print("LOAD and cache native Wan text embeddings", flush=True)
        encoder = WanTextEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.embeddings = {}
        for prompt in dict.fromkeys([*prompts, negative_prompt]):
            self.embeddings[prompt] = encoder(text_prompts=[prompt])["prompt_embeds"].cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        self.unconditional = self.embeddings[negative_prompt].to(self.device)
        self.conditional = self.embeddings[prompts[0]].to(self.device)
        print("LOAD pretrained bidirectional Wan2.1-T2V-1.3B", flush=True)
        self.generator = WanDiffusionWrapper(model_name="Wan2.1-T2V-1.3B", is_causal=False,
                                            timestep_shift=shift).to(device=self.device, dtype=torch.bfloat16)
        self.generator.eval().requires_grad_(False)
        self.generator.seq_len = SHAPE[1] * (SHAPE[-2] // 2) * (SHAPE[-1] // 2)
        self.vae = torch.nn.Identity()

    def select_prompt(self, prompt):
        self.conditional = self.embeddings[prompt].to(self.device)

    def predict_parts(self, state, sigma):
        # Exactly the native flow path used by NativeWan.predict. Sharing these
        # outputs makes the CFG comparison independent of extra forward calls.
        model_input = state.to(torch.bfloat16).permute(0, 2, 1, 3, 4)
        timestep = torch.full((state.shape[0],), sigma * 1000,
                              device=state.device, dtype=torch.float32)
        seq_len = state.shape[1] * (state.shape[-2] // 2) * (state.shape[-1] // 2)
        positive = self.generator.model(model_input, t=timestep, context=self.conditional,
                                        seq_len=seq_len).permute(0, 2, 1, 3, 4).float()
        negative = self.generator.model(model_input, t=timestep, context=self.unconditional,
                                        seq_len=seq_len).permute(0, 2, 1, 3, 4).float()
        return positive, negative


def run_ar_case(pipeline, conditional, seed, check_identity):
    parameter = next(pipeline.generator.parameters())
    initial, transition = make_noise_bank(SHAPE, CHUNKS, len(pipeline.denoising_step_list),
                                         seed, parameter.dtype, parameter.device)
    replay = IdentityTracedRollout(pipeline, conditional, initial, transition, CHUNKS,
                                   check_identity=check_identity)
    baseline = replay.baseline(0, 1)
    trace = replay.take_trace()
    chunk0 = [entry for entry in trace if entry["chunk_index"] == 0]
    if [entry["step_index"] for entry in chunk0] != list(range(len(replay.steps))):
        raise RuntimeError("AR trace has missing or duplicated denoising steps")
    latents = {"first": chunk0[0]["clean_prediction"], "second": chunk0[1]["clean_prediction"],
               "next_state": chunk0[1]["noisy"], "final": baseline["output"].cpu()}
    metadata = {"initial_noise_sha256": tensor_hash(initial),
                "first_chunk_noise_sha256": tensor_hash(initial[:, :3]),
                "transition_noise_sha256": {f"{chunk}:{step}": tensor_hash(noise)
                                            for (chunk, step), noise in transition.items()},
                "sigmas": [entry["sigma"] for entry in chunk0],
                "timesteps": [float(step) for step in replay.steps], "identity": replay.identity}
    _, alternate_transition = make_noise_bank(
        SHAPE, CHUNKS, len(pipeline.denoising_step_list), seed + 1000003,
        parameter.dtype, parameter.device)
    alternate_replay = TracedRollout(pipeline, conditional, initial, alternate_transition, CHUNKS)
    alternate = alternate_replay.baseline(0, 1)
    alternate_trace = alternate_replay.take_trace()
    alternate_first = next(entry["clean_prediction"] for entry in alternate_trace
                           if entry["chunk_index"] == 0 and entry["step_index"] == 0)
    first_delta = float((alternate_first.float() - latents["first"].float()).abs().max())
    if first_delta != 0.0:
        raise RuntimeError(f"Changing later noise changed first prediction: {first_delta}")
    latents["final_alt_transition"] = alternate["output"].cpu()
    metadata["alternate_transition"] = {
        "seed": seed + 1000003, "initial_noise_unchanged": True,
        "first_prediction_max_abs": first_delta,
        "transition_noise_sha256": {f"{chunk}:{step}": tensor_hash(noise)
                                    for (chunk, step), noise in alternate_transition.items()}}
    metadata["stages"] = {"first": {"step_index": 0, "sigma": chunk0[0]["sigma"]},
                          "second": {"step_index": 1, "sigma": chunk0[1]["sigma"]}}
    return latents, metadata


def run_wan_case(pipeline, seed, sigmas, check_identity):
    # Match make_noise_bank without materializing unused AR transition noise.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(SHAPE, generator=rng, dtype=torch.float32).to(torch.bfloat16).to(pipeline.device)
    state = initial.float()
    positive, negative = pipeline.predict_parts(state, float(sigmas[0]))
    first_flow = negative + pipeline.guidance * (positive - negative)
    identity = None
    if check_identity:
        repeated = pipeline.predict(state, float(sigmas[0]))
        delta = float((first_flow - repeated).abs().max())
        identity = {"first_call_repeat_max_abs": delta,
                    "scope": "first native CFG flow; no rollout repeat"}
        if not math.isfinite(delta):
            raise RuntimeError("Nonfinite Wan identity difference")
    latents = {"first": (state - float(sigmas[0]) * first_flow)[:, :3].cpu(),
               "first_cfg1": (state - float(sigmas[0]) * positive)[:, :3].cpu()}
    stage_indices = {max(1, min(len(sigmas) - 2, int((len(sigmas) - 1) * fraction))): name
                     for name, fraction in [("quarter", .25), ("half", .5), ("threequarter", .75)]}
    stages = {"first": {"step_index": 0, "sigma": float(sigmas[0])},
              "second": {"step_index": 1, "sigma": float(sigmas[1])}}
    for index in range(len(sigmas) - 1):
        sigma = float(sigmas[index])
        flow = first_flow if index == 0 else pipeline.predict(state, sigma)
        if index == 1:
            latents["second"] = (state - sigma * flow)[:, :3].cpu()
        if index in stage_indices:
            name = stage_indices[index]
            latents[name] = (state - sigma * flow)[:, :3].cpu()
            stages[name] = {"step_index": index, "sigma": sigma}
        state = state + (float(sigmas[index + 1]) - sigma) * flow
        if index == 0:
            latents["next_state"] = state[:, :3].cpu()
    latents["final"] = state.cpu()
    # The shared first three noise frames hold the temporal horizon fixed when
    # contrasting causal first-chunk and bidirectional first-step predictions.
    state3 = initial[:, :3].float()
    positive3, negative3 = pipeline.predict_parts(state3, float(sigmas[0]))
    flow3 = negative3 + pipeline.guidance * (positive3 - negative3)
    latents["first_horizon3"] = (state3 - float(sigmas[0]) * flow3).cpu()
    latents["first_horizon3_cfg1"] = (state3 - float(sigmas[0]) * positive3).cpu()
    metadata = {"initial_noise_sha256": tensor_hash(initial),
                "first_chunk_noise_sha256": tensor_hash(initial[:, :3]),
                "sigmas": [float(sigma) for sigma in sigmas],
                "timesteps": [float(sigma) * 1000 for sigma in sigmas[:-1]], "identity": identity,
                "stages": stages}
    return latents, metadata


def render_views(pipeline, latents):
    parameter = next(pipeline.generator.parameters())
    views = {}
    for key, latent in latents.items():
        # Slice BEFORE decoding even final. Wan VAE is temporally causal, but
        # identical decoder horizons make comparison independent of that fact.
        pixels = decode_pixels(pipeline, latent[:, :3].to(parameter.device, parameter.dtype))
        selected = pixels[[0, 4, 8]].permute(0, 3, 1, 2)
        selected = F.interpolate(selected, size=VIEW_SHAPE[1:3], mode="area")
        views[key] = selected.permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).numpy()
        if views[key].shape != VIEW_SHAPE:
            raise RuntimeError(f"Unexpected decoded view shape: {key} {views[key].shape}")
    return views


def validate_completed(destination, protocol):
    path = destination / "record.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("protocol") != protocol or record.get("protocol_sha256") != json_hash(protocol):
        raise ValueError(f"Existing case has a different protocol: {path}")
    if record.get("complete") is not True:
        raise ValueError(f"Existing record is incomplete: {path}")
    for name in ("latents.pt", "views.npz"):
        artifact = destination / name
        if not artifact.is_file() or file_hash(artifact) != record["artifact_sha256"][name]:
            raise ValueError(f"Missing or changed artifact: {artifact}")
    keys = BASE_KEYS | (WAN_KEYS if protocol["model"] == "wan" else AR_KEYS)
    with np.load(destination / "views.npz") as arrays:
        if set(arrays.files) != keys or any(arrays[key].shape != VIEW_SHAPE or
                                            arrays[key].dtype != np.uint8 for key in arrays.files):
            raise ValueError(f"Invalid view schema: {destination}")
    return True


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment in the GPU dev environment")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    if not config.mixed_precision:
        raise ValueError("Paired-noise protocol requires bfloat16 model inputs (mixed_precision=true)")
    prompts = args.data_path.read_text().splitlines()
    cases = [(index, seed) for index in args.prompt_indices for seed in args.seeds]
    cases = cases[args.shard_index::args.num_shards]
    if not cases:
        print("No cases assigned", flush=True)
        return
    checkpoint = args.checkpoint_path / "model.pt" if args.checkpoint_path.is_dir() else args.checkpoint_path
    if args.model == "ar":
        stat = checkpoint.stat()
        source = {"checkpoint_path": str(checkpoint.resolve()), "checkpoint_size": stat.st_size,
                  "checkpoint_mtime_ns": stat.st_mtime_ns, "weights": args.weights}
    else:
        source = {"checkpoint_path": str(Path("wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors").resolve()),
                  "weights": "pretrained", "model_manifest": model_manifest()}
    common = {"schema_version": SCHEMA_VERSION, "model": args.model, "shape": SHAPE, "chunks": CHUNKS,
              "model_source": source, "config": OmegaConf.to_container(config, resolve=True),
              "guidance": args.guidance if args.model == "wan" else 1.0,
              "wan_steps": args.wan_steps if args.model == "wan" else None,
              "shift": args.shift if args.model == "wan" else None,
              "noise_protocol": "CPU float32 randn cast to bfloat16, independent per seed",
              "view_protocol": {"latent_frames": [0, 1, 2], "decoded_frames": [0, 4, 8],
                                "shape": list(VIEW_SHAPE), "resize": "area", "independent_decode": True},
              "source_sha256": {name: file_hash(name) for name in CODE_FILES},
              "torch_version": torch.__version__}
    pending = []
    for index, seed in cases:
        prompt = prompts[index].strip()
        protocol = dict(common, prompt_index=index, prompt=prompt, seed=seed)
        destination = args.output_dir / args.model / f"prompt_{index:06d}" / f"seed_{seed:06d}"
        if validate_completed(destination, protocol):
            print(f"SKIP {args.model} prompt={index} seed={seed}", flush=True)
        else:
            pending.append((index, seed, prompt, protocol, destination))
    if not pending:
        return
    if args.model == "ar":
        pipeline, _, shape, chunks, _ = load_pipeline(args)
        if shape != SHAPE or chunks != CHUNKS or len(pipeline.denoising_step_list) < 2:
            raise ValueError("Protocol requires [1,6,16,60,104], two chunks of three, and >=2 steps")
    else:
        pipeline = CachedNativeWan([case[2] for case in pending], config.negative_prompt,
                                   args.guidance, args.shift)
    cached_prompt, conditional = None, None
    sigmas = shifted_sigmas(args.wan_steps, args.shift)
    for number, (index, seed, prompt, protocol, destination) in enumerate(pending):
        started = time.monotonic()
        print(f"START {args.model} prompt={index} seed={seed}", flush=True)
        if args.model == "ar":
            if cached_prompt != prompt:
                parameter = next(pipeline.generator.parameters())
                pipeline.text_encoder.to(parameter.device)
                conditional = pipeline.text_encoder(text_prompts=[prompt])
                pipeline.text_encoder.cpu()
                torch.cuda.empty_cache()
                cached_prompt = prompt
            latents, metadata = run_ar_case(pipeline, conditional, seed, number == 0)
        else:
            pipeline.select_prompt(prompt)
            latents, metadata = run_wan_case(pipeline, seed, sigmas, number == 0)
        expected_keys = BASE_KEYS | (WAN_KEYS if args.model == "wan" else AR_KEYS)
        if set(latents) != expected_keys:
            raise RuntimeError("Unexpected latent artifact keys")
        for key, value in latents.items():
            expected = [1, 6 if key in ("final", "final_alt_transition") else 3, 16, 60, 104]
            if list(value.shape) != expected or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid latent shape or nonfinite values: {key}")
        views = render_views(pipeline, latents)
        destination.mkdir(parents=True, exist_ok=True)
        temporary_latents = destination / f"latents.{os.getpid()}.tmp"
        torch.save(latents, temporary_latents)
        temporary_latents.replace(destination / "latents.pt")
        temporary_views = destination / f"views.{os.getpid()}.tmp"
        with temporary_views.open("wb") as target:
            np.savez_compressed(target, **views)
        temporary_views.replace(destination / "views.npz")
        torch.cuda.synchronize()
        record = dict(metadata, schema_version=SCHEMA_VERSION, model=args.model, prompt_index=index,
                      prompt=prompt, seed=seed, shape=SHAPE, chunks=CHUNKS, **source,
                      protocol=protocol, protocol_sha256=json_hash(protocol),
                      artifacts={"latents": "latents.pt", "views": "views.npz"},
                      view_path="views.npz", view_keys=sorted(views),
                      artifact_sha256={name: file_hash(destination / name) for name in ("latents.pt", "views.npz")},
                      elapsed_seconds=time.monotonic() - started,
                      gpu=torch.cuda.get_device_name(), complete=True)
        atomic_json(destination / "record.json", record)
        print(f"DONE {args.model} prompt={index} seed={seed} seconds={record['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
