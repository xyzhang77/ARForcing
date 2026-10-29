"""Capture Wan 14B x0 distributions at steps 2, 3 and 4 of a frozen trajectory."""

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

from scripts.denoising.compare_first_step_noise import CODE_FILES, SHAPE, file_hash, json_hash
from scripts.diversity.eval_diversity_moments import (
    FEATURE_SPACE, LATENT_SHAPE, TEACHER_MODEL, TrainingTeacherWan,
    initial_noise, noise_metadata, preview_frame, teacher_model_manifest,
    training_teacher_alignment,
)
from scripts.patch.patch_noise_importance import atomic_json
from scripts.patch.wan_patch_probe import shifted_sigmas


SCHEMA_VERSION = "teacher_steps_sampling_v1"
STAGE_KEYS = ("step_2", "step_3", "step_4")
DEFAULT_REFERENCE = Path("outputs/diversity/diversity_teacher14b_20261024")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-mode", choices=("ar_grid", "ar_noise_levels", "native_ordinals"), default="ar_grid")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_REFERENCE / "frozen_config.yaml")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=[42, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(64)))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    for name in ("prompt_indices", "seeds"):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
            parser.error(f"{name} must contain distinct nonnegative integers")
    if any(seed >= 2**63 for seed in args.seeds):
        parser.error("Seeds must be smaller than 2**63")
    if not args.config_path.is_file() or not args.data_path.is_file():
        parser.error("Frozen config and prompt files must exist")
    return args


def sampling_schedule(stage_mode, config):
    """Derive the same warped grid as BaseModel, without loading a student."""
    from utils.scheduler import FlowMatchScheduler

    shift = float(config.get("model_kwargs", {}).get("timestep_shift", 8.0))
    scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    raw_steps = torch.tensor(config.denoising_step_list, dtype=torch.long)
    if raw_steps.numel() != 4 or torch.any(raw_steps < 0) or torch.any(raw_steps > 1000):
        raise ValueError("This experiment requires the frozen four-step AR schedule")
    if config.warp_denoising_step:
        table = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        timesteps = table[1000 - raw_steps]
    else:
        timesteps = raw_steps.float()
    ar_sigmas = timesteps / 1000
    if not torch.all(ar_sigmas[:-1] > ar_sigmas[1:]) or float(ar_sigmas[0]) != 1.0:
        raise ValueError("AR grid must descend strictly from sigma=1")
    derivation = dict(source="model.base.BaseModel.__init__ warped denoising_step_list",
                      raw_denoising_step_list=raw_steps.tolist(),
                      warp_denoising_step=bool(config.warp_denoising_step),
                      generator_timestep_shift=shift, training_table_steps=1000,
                      warped_timesteps=timesteps.tolist(), ar_sigmas=ar_sigmas.tolist())
    if stage_mode == "ar_grid":
        sigmas = torch.cat((ar_sigmas, torch.tensor([0.0])))
        indices = [1, 2, 3]
        derivation["trajectory"] = "four-step native Euler on the AR sigma grid; no AR re-noising"
    else:
        if shift != 5.0:
            raise ValueError("Native diagnostic modes require the original shift=5 schedule")
        sigmas = shifted_sigmas(48, shift)
        indices = [12, 24, 36] if stage_mode == "ar_noise_levels" else [1, 2, 3]
        derivation["trajectory"] = "48-step native Euler trajectory"
        if stage_mode == "ar_noise_levels" and not torch.equal(sigmas[indices], ar_sigmas[1:]):
            raise ValueError("Native trajectory probes differ from frozen AR sigma grid")
    stages = {key: dict(stage=stage, step_index=index, denoiser_probe_ordinal=index + 1,
                        completed_updates=index, sigma=float(sigmas[index]))
              for key, stage, index in zip(STAGE_KEYS, (2, 3, 4), indices)}
    return sigmas, stages, derivation, shift


def common_protocol(args, config, sigmas, stages, derivation):
    from omegaconf import OmegaConf

    alignment = training_teacher_alignment(args, config)
    source = dict(checkpoint_path=str((Path("wan_models") / TEACHER_MODEL /
                                      "diffusion_pytorch_model.safetensors.index.json").resolve()),
                  weights="pretrained", model_name=TEACHER_MODEL,
                  model_manifest=teacher_model_manifest(),
                  text_encoder="utils.wan_wrapper.WanTextEncoder (same loader as training)")
    code_files = [*CODE_FILES, "utils/scheduler.py", "scripts/diversity/eval_diversity_moments.py",
                  str(Path(__file__).resolve().relative_to(Path.cwd()))]
    return dict(schema_version=SCHEMA_VERSION, model="wan14b", stage_mode=args.stage_mode,
                shape=SHAPE, feature_shape=LATENT_SHAPE, feature_space=FEATURE_SPACE,
                stages=stages, sigmas=[float(value) for value in sigmas],
                schedule_derivation=derivation, guidance=alignment["native_guidance_scale"],
                sampler="native Wan CFG Euler; x0 = state - sigma * flow; no AR re-noising",
                noise_protocol="CPU float32 randn cast to bfloat16, independent per seed",
                model_source=source, teacher_training_alignment=alignment,
                config=OmegaConf.to_container(config, resolve=True), config_sha256=file_hash(args.config_path),
                source_sha256={name: file_hash(name) for name in dict.fromkeys(code_files)},
                torch_version=torch.__version__)


def reference_validation(args, protocol, metadata, captures=None):
    directory = args.reference_root / "wan14b" / f"prompt_{protocol['prompt_index']:06d}" / f"seed_{protocol['seed']:06d}"
    path = directory / "record.json"
    if not path.is_file():
        raise ValueError(f"Missing paired Wan14B baseline record: {path}")
    record = json.loads(path.read_text())
    old = record["protocol"]
    if not record.get("complete") or record.get("protocol_sha256") != json_hash(old):
        raise ValueError(f"Incomplete or changed reference record: {path}")
    for key in ("prompt_index", "prompt", "seed", "guidance", "model_source"):
        if record.get(key) != protocol[key]:
            raise ValueError(f"Reference {key} differs: {path}")
    for key in ("config", "config_sha256", "teacher_training_alignment"):
        if old.get(key) != protocol[key]:
            raise ValueError(f"Reference protocol {key} differs: {path}")
    for key, value in metadata.items():
        if record.get(key) != value:
            raise ValueError(f"Paired reference initial noise differs: {path}, {key}")
    result = dict(record=str(path.resolve()), record_sha256=file_hash(path),
                  same_initial_noise=True, teacher_model_source_match=True,
                  training_config_match=True, guidance_and_negative_prompt_match=True,
                  trajectory_differs=args.stage_mode != "ar_noise_levels")
    if captures is not None and args.stage_mode == "ar_noise_levels":
        feature_path = directory / "features.npz"
        if file_hash(feature_path) != record["artifact_sha256"]["features.npz"]:
            raise ValueError(f"Modified reference features: {feature_path}")
        with np.load(feature_path, allow_pickle=False) as archive:
            expected = archive["latent"]
        actual = captures["step_2"].numpy()
        if not np.array_equal(actual, expected):
            raise ValueError("Same-trajectory step2 prediction differs from prior Wan14B probe")
        result["step2_x0_bitwise_equal"] = True
    return result


def validate_completed(directory, protocol):
    path = directory / "record.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    if record.get("protocol") != protocol or record.get("protocol_sha256") != json_hash(protocol):
        raise ValueError(f"Existing case has a different protocol: {path}")
    if record.get("complete") is not True:
        return False
    for name, digest in record.get("artifact_sha256", {}).items():
        if not (directory / name).is_file() or file_hash(directory / name) != digest:
            raise ValueError(f"Missing or changed artifact: {directory / name}")
    if "features.npz" not in record.get("artifact_sha256", {}):
        raise ValueError(f"Missing feature artifact identity: {path}")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(STAGE_KEYS):
            raise ValueError(f"Unexpected feature keys: {directory}")
        for key in STAGE_KEYS:
            array = archive[key]
            if list(array.shape) != LATENT_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid feature array: {directory}, {key}")
            if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != record["latent_sha256"][key]:
                raise ValueError(f"Feature tensor hash differs: {directory}, {key}")
    expected = noise_metadata(initial_noise(protocol["seed"]))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Paired initial noise differs: {directory}")
    return True


def capture_steps(pipeline, seed, sigmas, stages):
    initial = initial_noise(seed)
    metadata = noise_metadata(initial)
    state = initial.to(pipeline.device).float()
    wanted = {value["step_index"]: key for key, value in stages.items()}
    captures = {}
    for index in range(max(wanted) + 1):
        sigma = float(sigmas[index])
        flow = pipeline.predict(state, sigma)
        if index in wanted:
            captures[wanted[index]] = (state - sigma * flow)[0, :3].float().cpu()
        if index < max(wanted):
            state = state + (float(sigmas[index + 1]) - sigma) * flow
    return captures, metadata


def save_case(directory, protocol, captures, metadata, reference, previews, started):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {key: captures[key].numpy() for key in STAGE_KEYS}
    if any(list(value.shape) != LATENT_SHAPE or value.dtype != np.float32 or not np.isfinite(value).all()
           for value in arrays.values()):
        raise ValueError("Invalid captured x0 features")
    temporary = directory / f"features.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(directory / "features.npz")
    artifacts = {"features.npz": file_hash(directory / "features.npz")}
    if previews:
        from PIL import Image
        for key, pixels in previews.items():
            name = f"preview_{key}.jpg"
            temporary = directory / f"{name}.{os.getpid()}.tmp"
            Image.fromarray(pixels).save(temporary, format="JPEG", quality=90)
            temporary.replace(directory / name)
            artifacts[name] = file_hash(directory / name)
    record = dict(metadata, schema_version=SCHEMA_VERSION, model="wan14b", complete=True,
                  prompt_index=protocol["prompt_index"], prompt=protocol["prompt"], seed=protocol["seed"],
                  stage_mode=protocol["stage_mode"], stages=protocol["stages"], shape=LATENT_SHAPE,
                  feature_space=FEATURE_SPACE, guidance=protocol["guidance"],
                  model_source=protocol["model_source"], protocol=protocol, protocol_sha256=json_hash(protocol),
                  artifact_sha256=artifacts,
                  latent_sha256={key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                 for key, value in arrays.items()},
                  same_noise_reference_validation=reference,
                  host=socket.gethostname(), torch_version=torch.__version__,
                  elapsed_seconds=time.monotonic() - started)
    atomic_json(directory / "record.json", record)


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    if not config.mixed_precision:
        raise ValueError("Paired teacher protocol requires mixed_precision=true")
    alignment = training_teacher_alignment(args, config)
    args.guidance = alignment["native_guidance_scale"]
    sigmas, stages, derivation, shift = sampling_schedule(args.stage_mode, config)
    common = common_protocol(args, config, sigmas, stages, derivation)
    prompts = args.data_path.read_text().splitlines()
    if any(index >= len(prompts) or not prompts[index].strip() for index in args.prompt_indices):
        raise ValueError("Prompt index out of range or empty")
    cases = [(index, seed) for index in args.prompt_indices for seed in args.seeds]
    pending = []
    for index, seed in cases[args.shard_index::args.num_shards]:
        protocol = dict(common, prompt_index=index, prompt=prompts[index].strip(), seed=seed)
        directory = args.output_dir / "wan14b" / f"prompt_{index:06d}" / f"seed_{seed:06d}"
        if validate_completed(directory, protocol):
            print(f"SKIP wan14b prompt={index} seed={seed}", flush=True)
            continue
        reference_validation(args, protocol, noise_metadata(initial_noise(seed)))
        pending.append((protocol, directory))
    if not pending:
        print("FINISHED all assigned cases already complete", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Pending teacher sampling requires the GPU dev environment")
    pipeline = TrainingTeacherWan([p[0]["prompt"] for p in pending], config.negative_prompt, args.guidance, shift)
    for protocol, directory in pending:
        started = time.monotonic()
        print(f"START wan14b prompt={protocol['prompt_index']} seed={protocol['seed']}", flush=True)
        pipeline.select_prompt(protocol["prompt"])
        captures, metadata = capture_steps(pipeline, protocol["seed"], sigmas, stages)
        reference = reference_validation(args, protocol, metadata, captures)
        previews = {key: preview_frame(pipeline, captures[key].unsqueeze(0)) for key in STAGE_KEYS} if protocol["seed"] < 8 else {}
        save_case(directory, protocol, captures, metadata, reference, previews, started)
        print(f"DONE wan14b prompt={protocol['prompt_index']} seed={protocol['seed']} seconds={time.monotonic()-started:.2f}", flush=True)
    print(f"FINISHED model=wan14b stage_mode={args.stage_mode}", flush=True)


if __name__ == "__main__":
    run(parse_args())
