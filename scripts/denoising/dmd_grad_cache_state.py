"""DMD gradient on Step-1 x0 in the training state vs the inference state; run with python -m scripts.denoising.dmd_grad_cache_state.

States, all from the same initial noise:
  A train_step1: every chunk's Step-1 x0 read from a cache written with Step-1 x0 (Self Forcing exit=1).
  B infer_step1: every chunk's Step-1 x0 read from the clean cache (Step-4 x0), i.e. what inference calls.
  C final:       the inference output (Step-4 x0), as a reference.
Per chunk and DMD noise level, reports the DMD update |fake - real| / mean|x - real| (paper eq. 7-8),
the critic-free teacher residual |x - real|, and cos(grad_A, grad_B). Chunk 0 has no history, so A == B there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.denoising.eval_score_gap import iter_predictions, make_eval_scheduler

STATES = ("train_step1", "infer_step1", "final")


@torch.no_grad()
def rollout(pipe, noise, cond, cache_step):
    """Plain inference with the cache written from Step `cache_step` x0; returns the x0 of every step."""
    B, T = noise.shape[:2]
    n, L, steps = pipe.num_frame_per_block, pipe.frame_seq_length, pipe.denoising_step_list
    pipe._initialize_kv_cache(B, noise.dtype, noise.device)
    pipe._initialize_crossattn_cache(B, noise.dtype, noise.device)
    x0s = torch.zeros(len(steps), *noise.shape, device=noise.device, dtype=noise.dtype)
    for start in range(0, T, n):
        def call(inp, ts):
            timestep = torch.ones([B, n], device=noise.device, dtype=torch.int64) * ts
            return pipe.generator(noisy_image_or_video=inp, conditional_dict=cond, timestep=timestep,
                                  kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                                  current_start=start * L)[1]
        x = noise[:, start:start + n]
        for i, ts in enumerate(steps):
            x0s[i, :, start:start + n] = x0 = call(x, ts)
            if i < len(steps) - 1:
                x = pipe.scheduler.add_noise(
                    x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                    steps[i + 1] * torch.ones([B * n], device=noise.device, dtype=torch.long)).unflatten(0, x0.shape[:2])
        call(x0s[cache_step, :, start:start + n], pipe.args.context_noise)
    return x0s


def generate(config, weights, prompts, num_samples, seed, device, dtype):
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.misc import set_seed

    pipe = CausalInferencePipeline(config, device=device, vae=torch.nn.Identity())
    pipe.generator.load_state_dict(weights, strict=True)
    pipe.to(device=device, dtype=dtype).eval().requires_grad_(False)
    uncond = {k: v.cpu() for k, v in pipe.text_encoder(text_prompts=[config.negative_prompt]).items()}
    samples = []
    for pi, prompt in enumerate(prompts):
        set_seed(seed + pi)
        noise = torch.randn([num_samples, 21, 16, 60, 104], device=device, dtype=dtype)
        cond = pipe.text_encoder(text_prompts=[prompt] * num_samples)
        clean, noisy = rollout(pipe, noise, cond, cache_step=-1), rollout(pipe, noise, cond, cache_step=0)
        states = {"train_step1": noisy[0], "infer_step1": clean[0], "final": clean[-1]}
        n = pipe.num_frame_per_block
        gap0 = (states["train_step1"][:, :n] - states["infer_step1"][:, :n]).abs().max().item()
        assert gap0 < 1e-3, f"chunk 0 has no history, so A and B must match there: {gap0}"
        for s in range(num_samples):
            for state in STATES:
                samples.append({"prompt_index": pi, "seed_index": s, "state": state,
                                "noise_seed": 1_000_000 + pi * 100 + s,  # paired DMD noise across states
                                "latent": states[state][s:s + 1].cpu(),
                                "conditional": {k: v[s:s + 1].cpu() for k, v in cond.items()}})
        print(f"generated prompt {pi + 1}/{len(prompts)}", flush=True)
    del pipe
    torch.cuda.empty_cache()
    return samples, uncond


def score(model_name, weights, samples, uncond, scheduler, guidance, device, dtype):
    from utils.wan_wrapper import WanDiffusionWrapper

    model = WanDiffusionWrapper(model_name=model_name, is_causal=False)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    # keep the guided x0 (== conditional when guidance is 0), as DMD does
    out = [[pair[1] for pair in preds]
           for preds in iter_predictions(model, samples, uncond, scheduler, guidance, device, dtype)]
    del model
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600/model.pt"))
    p.add_argument("--prompts", type=Path, default=Path("outputs/denoising/cache_step_20261028/prompts_extended.txt"))
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    # uniform t in {0.2, 0.5, 0.8} after the DMD timestep shift of 5
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.5556, 0.8333, 0.9524])
    p.add_argument("--output", type=Path, default=Path("outputs/denoising/cache_step_20261028/dmd_grad_cache_state.json"))
    args = p.parse_args()
    assert not args.output.exists(), f"refusing to overwrite {args.output}"

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    device, dtype = torch.device("cuda"), torch.bfloat16
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert "generator" in ckpt and "critic" in ckpt, "checkpoint must hold the generator and its matching critic"
    prompts = args.prompts.read_text().splitlines()
    samples, uncond = generate(config, ckpt.pop("generator"), prompts, args.num_samples, args.seed, device, dtype)

    scheduler = make_eval_scheduler(args.sigmas)
    real_g = config.get("real_guidance_scale", config.guidance_scale)
    fake_g = config.get("fake_guidance_scale", 0.0) if "real_guidance_scale" in config else 0.0
    print("scoring with critic", flush=True)
    fake = score(config.get("fake_name", "Wan2.1-T2V-1.3B"), ckpt.pop("critic"), samples, uncond, scheduler, fake_g, device, dtype)
    print("scoring with teacher", flush=True)
    real = score(config.real_name, None, samples, uncond, scheduler, real_g, device, dtype)

    n, num_chunks = config.num_frame_per_block, 21 // config.num_frame_per_block
    rows = []  # one row per (sample, state, sigma, chunk)
    grads = {}
    for i, s in enumerate(samples):
        x = s["latent"].float()
        for k, sigma in enumerate(args.sigmas):
            g = fake[i][k] - real[i][k]
            resid = (x - real[i][k]).abs()
            normalizer = resid.mean().item()  # DMD normalizes over the whole video
            grads[(s["prompt_index"], s["seed_index"], s["state"], k)] = g
            for c in range(num_chunks):
                sl = slice(c * n, (c + 1) * n)
                rows.append({"prompt_index": s["prompt_index"], "seed_index": s["seed_index"], "state": s["state"],
                             "sigma": sigma, "chunk": c,
                             "dmd_update": g[:, sl].abs().mean().item() / normalizer,
                             "raw_grad_mae": g[:, sl].abs().mean().item(),
                             "teacher_residual": resid[:, sl].mean().item()})
    cos = []
    for (pi, si, state, k), g in grads.items():
        if state != "train_step1":
            continue
        h = grads[(pi, si, "infer_step1", k)]
        for c in range(num_chunks):
            a, b = g[:, c * n:(c + 1) * n].flatten(), h[:, c * n:(c + 1) * n].flatten()
            cos.append({"sigma": args.sigmas[k], "chunk": c,
                        "cos": torch.nn.functional.cosine_similarity(a, b, dim=0).item()})

    def agg(rs, key):
        return float(np.mean([r[key] for r in rs]))
    summary = {}
    for sigma in args.sigmas:
        for group, chunks in (("chunk0", [0]), ("chunk1-6", list(range(1, num_chunks)))):
            entry = {state: {key: agg([r for r in rows if r["sigma"] == sigma and r["state"] == state
                                       and r["chunk"] in chunks], key)
                             for key in ("dmd_update", "raw_grad_mae", "teacher_residual")} for state in STATES}
            entry["cos_grad_train_vs_infer"] = agg([r for r in cos if r["sigma"] == sigma and r["chunk"] in chunks], "cos")
            summary[f"sigma{sigma}_{group}"] = entry
    report = {"description": __doc__, "checkpoint": str(args.checkpoint), "sigmas": args.sigmas,
              "real_guidance": real_g, "fake_guidance": fake_g, "num_prompts": len(prompts),
              "num_samples": args.num_samples, "summary": summary, "rows": rows, "cos": cos}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
