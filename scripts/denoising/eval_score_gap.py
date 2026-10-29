"""Latent-only teacher/critic diagnostic; run with python -m scripts.denoising.eval_score_gap."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean

import torch

from model.dmd import frame_diff_energy, motion_ratio
from utils.scheduler import FlowMatchScheduler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path("prompts/MovieGenVideoBench.txt"))
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.5, 0.8],
                        help="Actual noise mixing coefficients, without an additional timestep shift")
    args = parser.parse_args(argv)
    if args.num_prompts < 1 or not 0 <= args.seed < 2**32 - args.num_prompts:
        parser.error("num_prompts must be positive and seeds must fit in uint32")
    if any(not 0 < sigma < 1 for sigma in args.sigmas) or len(set(args.sigmas)) != len(args.sigmas):
        parser.error("sigmas must be distinct finite values strictly between 0 and 1")
    for name in ("config_path", "checkpoint_path", "data_path"):
        if not getattr(args, name).is_file():
            parser.error(f"{name} is not a file: {getattr(args, name)}")
    if args.output_path.exists():
        parser.error(f"Refusing to overwrite {args.output_path}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run one process on one GPU, not multi-process torchrun")
    return args


def make_eval_scheduler(sigmas):
    scheduler = FlowMatchScheduler(shift=1.0)
    # Share exact sigmas between corruption and both models' flow-to-x0 conversion.
    scheduler.sigmas = torch.tensor(sigmas, dtype=torch.float32)
    scheduler.timesteps = scheduler.sigmas * scheduler.num_train_timesteps
    return scheduler


def gap_metrics(clean, teacher, critic):
    gap = critic.float() - teacher.float()
    mae = gap.abs().mean().item()
    reference = (clean.float() - teacher.float()).abs().mean().item()
    metrics = {
        "x0_mse": gap.square().mean().item(),
        "x0_mae": mae,
        "normalized_mae": mae / max(reference, 1e-6),
        "teacher_residual_mae": reference,
        # gap is the (unnormalized) DMD grad; compare its motion share with the sample's own.
        "grad_motion_ratio": motion_ratio(gap).item(),
        "student_motion_ratio": motion_ratio(clean).item(),
        "motion_energy_student": frame_diff_energy(clean).item(),
        "motion_energy_teacher": frame_diff_energy(teacher).item(),
        "motion_energy_critic": frame_diff_energy(critic).item(),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite score-gap metrics; check latents and model predictions")
    return metrics


@torch.inference_mode()
def generate_samples(config, weights, prompts, seed, device, dtype):
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.misc import set_seed

    shape = [1, *config.image_or_video_shape[1:]]
    block_size = config.num_frame_per_block
    if (len(shape) != 5 or shape[2:] != [16, 60, 104] or not 0 < shape[1] <= 21
            or block_size < 1 or (shape[1] - int(config.independent_first_frame)) % block_size):
        raise ValueError("Expected up to 21 latent frames at 16x60x104, aligned to num_frame_per_block")
    pipeline = CausalInferencePipeline(config, device=device, vae=torch.nn.Identity())
    pipeline.generator.load_state_dict(weights, strict=True)
    pipeline.generator.model.independent_first_frame = config.independent_first_frame
    pipeline.to(device=device, dtype=dtype).eval().requires_grad_(False)
    unconditional = pipeline.text_encoder(text_prompts=[config.negative_prompt])
    unconditional = {key: value.cpu() for key, value in unconditional.items()}
    samples = []
    for index, prompt in enumerate(prompts):
        sample_seed = seed + index
        set_seed(sample_seed)
        print(f"Generating latent {index + 1}/{len(prompts)}, seed={sample_seed}", flush=True)
        noise = torch.randn(shape, device=device, dtype=dtype)
        latent = pipeline.inference(noise=noise, text_prompts=[prompt], latent_only=True)
        conditional = pipeline.text_encoder(text_prompts=[prompt])
        samples.append({
            "prompt_index": index,
            "prompt": prompt,
            "seed": sample_seed,
            "noise_seed": sample_seed + 1_000_000,
            "latent": latent.cpu(),
            "conditional": {key: value.cpu() for key, value in conditional.items()},
        })
    return samples, unconditional


def predict_x0(model, noisy, timestep, conditional, unconditional, guidance):
    _, pred_cond = model(noisy_image_or_video=noisy, conditional_dict=conditional, timestep=timestep)
    pred_cond = pred_cond.float()
    if guidance == 0:
        return pred_cond, pred_cond
    _, pred_uncond = model(noisy_image_or_video=noisy, conditional_dict=unconditional, timestep=timestep)
    # Match DMD's convention: cond + scale * (cond - uncond), not uncond + scale * delta.
    return pred_cond, pred_cond + guidance * (pred_cond - pred_uncond.float())


def iter_predictions(model, samples, unconditional, scheduler, guidance, device, dtype):
    model.scheduler = scheduler
    unconditional = {key: value.to(device=device, dtype=dtype) for key, value in unconditional.items()}
    for index, sample in enumerate(samples):
        print(f"Scoring latent {index + 1}/{len(samples)}", flush=True)
        with torch.inference_mode():
            clean = sample["latent"].to(device=device, dtype=dtype)
            conditional = {key: value.to(device=device, dtype=dtype)
                           for key, value in sample["conditional"].items()}
            rng = torch.Generator(device=device).manual_seed(sample["noise_seed"])
            noise = torch.randn(clean.shape, generator=rng, device=device, dtype=dtype)
            predictions = []
            for time in scheduler.timesteps.tolist():
                timestep = torch.full(clean.shape[:2], time, device=device, dtype=torch.float32)
                noisy = scheduler.add_noise(clean.flatten(0, 1), noise.flatten(0, 1),
                                            timestep.flatten()).unflatten(0, clean.shape[:2])
                conditional_x0, guided_x0 = predict_x0(
                    model, noisy, timestep, conditional, unconditional, guidance)
                predictions.append((conditional_x0.cpu(), guided_x0.cpu()))
        yield predictions


def summarize(results, levels):
    summary = []
    for index, level in enumerate(levels):
        row = {**level, "num_samples": len(results)}
        for mode in ("conditional", "guided"):
            row[mode] = {
                metric: fmean(result["by_sigma"][index][mode][metric] for result in results)
                for metric in results[0]["by_sigma"][index][mode]
            }
        summary.append(row)
    return summary


def main():
    args = parse_args()
    from omegaconf import OmegaConf
    from utils.dataset import TextDataset

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    if not getattr(config, "denoising_step_list", None) or getattr(config, "i2v", False):
        raise ValueError("This diagnostic requires a few-step T2V student config")
    teacher_guidance = config.get("real_guidance_scale", config.get("guidance_scale", 0.0))
    critic_guidance = config.get("fake_guidance_scale", 0.0) if "real_guidance_scale" in config else 0.0
    if not all(math.isfinite(scale) for scale in (teacher_guidance, critic_guidance)):
        raise ValueError("Guidance scales must be finite")
    dataset = TextDataset(args.data_path)
    prompts = [dataset[index]["prompts"] for index in range(min(args.num_prompts, len(dataset)))]
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("Evaluation prompts must be non-empty")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(key in checkpoint for key in ("generator", "critic")):
        raise ValueError("Checkpoint must contain both generator and its matching critic")
    checkpoint = {key: checkpoint[key] for key in ("generator", "critic")}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if config.mixed_precision else torch.float32
    samples, unconditional = generate_samples(
        config, checkpoint.pop("generator"), prompts, args.seed, device, dtype)
    torch.cuda.empty_cache()
    scheduler = make_eval_scheduler(args.sigmas)
    levels = [{"sigma": sigma, "timestep": sigma * scheduler.num_train_timesteps} for sigma in args.sigmas]

    from utils.wan_wrapper import WanDiffusionWrapper

    print("Evaluating checkpoint critic", flush=True)
    critic = WanDiffusionWrapper(model_name=config.get("fake_name", "Wan2.1-T2V-1.3B"), is_causal=False)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    del checkpoint
    critic.to(device=device, dtype=dtype).eval().requires_grad_(False)
    critic_predictions = list(iter_predictions(
        critic, samples, unconditional, scheduler, critic_guidance, device, dtype))
    del critic
    torch.cuda.empty_cache()

    # shortcut: one full score model must fit on one GPU; add sharding only if needed.
    print("Evaluating frozen teacher", flush=True)
    teacher = WanDiffusionWrapper(model_name=config.get("real_name", "Wan2.1-T2V-1.3B"), is_causal=False)
    teacher.to(device=device, dtype=dtype).eval().requires_grad_(False)
    results = []
    for index, teacher_predictions in enumerate(iter_predictions(
            teacher, samples, unconditional, scheduler, teacher_guidance, device, dtype)):
        sample = samples[index]
        result = {key: sample[key] for key in ("prompt_index", "prompt", "seed", "noise_seed")}
        result["by_sigma"] = []
        for level, teacher_pair, critic_pair in zip(levels, teacher_predictions, critic_predictions[index]):
            row = dict(level)
            for mode, teacher_x0, critic_x0 in zip(("conditional", "guided"), teacher_pair, critic_pair):
                row[mode] = gap_metrics(sample["latent"], teacher_x0, critic_x0)
            result["by_sigma"].append(row)
        results.append(result)
    summary = summarize(results, levels)
    report = {
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "generator_weights": "generator",
        "data_path": str(args.data_path.resolve()),
        "config": OmegaConf.to_container(config, resolve=True),
        "dtype": str(dtype),
        "latent_shape": list(samples[0]["latent"].shape),
        "teacher_guidance_scale": teacher_guidance,
        "critic_guidance_scale": critic_guidance,
        "guidance_convention": "conditional + scale * (conditional - unconditional)",
        "normalizer_floor": 1e-6,
        "interpretation": "Score-gap diagnostic, not KL or an independent video-quality metric. Compare per sigma.",
        "summary": summary,
        "samples": results,
    }
    with args.output_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved diagnostic to {args.output_path}")


if __name__ == "__main__":
    main()
