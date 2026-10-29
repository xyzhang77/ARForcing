"""Step-1 x0 of every chunk read from a clean (Step-4 x0) vs noisy (Step-1 x0) KV cache.

The rollout is the original inference (clean cache). A second cache holds the Step-1 x0 of the
same chunks, so both caches describe the same history at different denoising depth. Each chunk's
first denoiser call is replayed on the noisy cache with the identical input noise.
Run: python -m scripts.denoising.cache_step1_compare
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.misc import set_seed

KEYS = ("step1_clean_ctx", "step1_noisy_ctx", "final")


def spearman(a, b):
    a, b = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


@torch.no_grad()
def rollout(pipe, noise, cond):
    B, T = noise.shape[:2]
    n, L, steps = pipe.num_frame_per_block, pipe.frame_seq_length, pipe.denoising_step_list
    caches = []
    for _ in range(2):
        pipe._initialize_kv_cache(B, noise.dtype, noise.device)
        caches.append(pipe.kv_cache1)
    pipe._initialize_crossattn_cache(B, noise.dtype, noise.device)
    out = {k: torch.zeros_like(noise) for k in KEYS}
    for start in range(0, T, n):
        def call(inp, ts, cache):
            timestep = torch.ones([B, n], device=noise.device, dtype=torch.int64) * ts
            return pipe.generator(noisy_image_or_video=inp, conditional_dict=cond, timestep=timestep, kv_cache=cache,
                                  crossattn_cache=pipe.crossattn_cache, current_start=start * L)[1]

        x = noise[:, start:start + n]
        # No RNG is consumed here, so the clean rollout's transition noise matches plain inference.
        noisy_x0 = call(x, steps[0], caches[1])
        for i, ts in enumerate(steps):
            x0 = call(x, ts, caches[0])
            if i == 0:
                step1 = x0
            if i < len(steps) - 1:
                x = pipe.scheduler.add_noise(
                    x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                    steps[i + 1] * torch.ones([B * n], device=noise.device, dtype=torch.long)).unflatten(0, x0.shape[:2])
        for key, value in zip(KEYS, (step1, noisy_x0, x0)):
            out[key][:, start:start + n] = value
        call(x0, pipe.args.context_noise, caches[0])
        call(step1, pipe.args.context_noise, caches[1])
    return out


def token_mse(a, b):
    # [B, T, C, 60, 104] -> per-token MSE on the 2x2 patch grid [B, T, 30*52]
    d = (a.float() - b.float()).pow(2).mean(2)
    return F.avg_pool2d(d, 2).flatten(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600/model.pt")
    p.add_argument("--prompts", default="outputs/denoising/cache_step_20261028/prompts_extended.txt")
    p.add_argument("--out", type=Path, default=Path("outputs/denoising/cache_step_20261028/step1_ctx_compare"))
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame", type=int, default=40, help="pixel frame for the grid (40 = latent 10, chunk 3)")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(config, device="cuda")
    pipe.generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["generator"])
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
    n = pipe.num_frame_per_block
    args.out.mkdir(parents=True, exist_ok=True)

    prompts = Path(args.prompts).read_text().splitlines()
    records, rows = [], []
    for pi, prompt in enumerate(prompts):
        noise = torch.randn([args.num_samples, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16)
        cond = pipe.text_encoder(text_prompts=[prompt] * args.num_samples)
        out = rollout(pipe, noise, cond)
        diff0 = (out["step1_clean_ctx"][:, :n] - out["step1_noisy_ctx"][:, :n]).abs().max().item()
        assert diff0 < 1e-3, f"chunk 0 has no history and must not depend on the cache: {diff0}"
        torch.save({k: v.cpu() for k, v in out.items()}, args.out / f"prompt_{pi:02d}_latents.pt")

        d = token_mse(out["step1_noisy_ctx"], out["step1_clean_ctx"])   # cache effect on Step-1 x0
        e_clean = token_mse(out["step1_clean_ctx"], out["final"])       # Step-1 -> final gap (= cache content gap)
        e_noisy = token_mse(out["step1_noisy_ctx"], out["final"])
        chunks = []
        for c in range(1, 21 // n):
            s, prev = slice(c * n, (c + 1) * n), slice((c - 1) * n, c * n)
            chunks.append({
                "chunk": c,
                "mse_noisy_vs_clean": d[:, s].mean().item(),
                "mse_clean_vs_final": e_clean[:, s].mean().item(),
                "mse_noisy_vs_final": e_noisy[:, s].mean().item(),
                # does the cache-induced change sit where the previous chunk's cache content differed?
                "spearman_vs_prev_cache_gap": spearman(d[:, s].mean(1), e_clean[:, prev].mean(1)).mean().item(),
            })
        records.append({"prompt_index": pi, "prompt": prompt[:80],
                        "chunk0_mse_step1_vs_final": e_clean[:, :n].mean().item(), "chunks": chunks})
        print(f"[{pi + 1}/{len(prompts)}] c0 gap {e_clean[:, :n].mean().item():.3f} | " + " ".join(
            f"c{x['chunk']}:{x['mse_noisy_vs_clean']:.3f}/{x['mse_clean_vs_final']:.3f}/{x['mse_noisy_vs_final']:.3f}"
            for x in chunks), flush=True)

        frames = []
        for key in KEYS:
            v = pipe.vae.decode_to_pixel(out[key][:1], use_cache=False)[0, args.frame]
            frames.append(((v.float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        rows.append(frames)

    pd = [np.abs(r[1].astype(np.float32) - r[0].astype(np.float32)).mean(-1) for r in rows]
    scale = max(np.percentile(np.stack(pd), 99), 1e-6)
    grid = []
    for (clean, noisy, final), m in zip(rows, pd):
        heat = cv2.applyColorMap((np.clip(m / scale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        cells = [cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), heat,
                 cv2.cvtColor(final, cv2.COLOR_RGB2BGR)]
        grid.append(np.hstack([cv2.resize(x, (312, 180)) for x in cells]))
    cv2.imwrite(str(args.out / f"grid_frame{args.frame}.jpg"), np.vstack(grid))

    agg = {}
    for c in range(1, 21 // n):
        xs = [ch for r in records for ch in r["chunks"] if ch["chunk"] == c]
        agg[c] = {k: float(np.mean([x[k] for x in xs])) for k in xs[0] if k != "chunk"}
    summary = {"description": __doc__, "checkpoint": args.checkpoint, "num_samples": args.num_samples,
               "grid_columns": ["step1 x0, clean cache", "step1 x0, noisy cache", "|diff|", "final x0"],
               "per_chunk_mean": agg, "records": records}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
