"""Exp 4: does releasing noise in high-uncertainty patches give more diverse results?

For selected patches (high vs low U from analyze_patch_uncertainty.py, one token-count class
so the expected input energy sigma^2 * E|eps' - eps|^2 is identical), the intervened call's
transition noise is redrawn inside the patch only (same sigma, same global timestep, no x0
injected), K times; everything else, including all later noise, is the original bank.
Diversity = mean pairwise distance among the K outputs: decoded video RMSE, latent MSE outside
the patch (+halo on its frame), next-chunk latent MSE.
  run:       python scripts/patch/ar_patch_local_resample.py --prompt-index 158 --step 1
  summarize: python scripts/patch/ar_patch_local_resample.py --summarize
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path("outputs/patch_importance/ar_patch_local_resample_20261027")
UNCERTAINTY = Path("outputs/patch_importance/ar_noise_draws_20261026/uncertainty_summary.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT)
    p.add_argument("--uncertainty", type=Path, default=UNCERTAINTY)
    p.add_argument("--checkpoint-path", type=Path,
                   default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    p.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    p.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    p.add_argument("--prompt-index", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, choices=(1, 2, 3))
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--per-group", type=int, default=10)
    p.add_argument("--token-class", type=int, default=9, help="Only patches with this many tokens")
    p.add_argument("--grid-rows", type=int, default=12)
    p.add_argument("--grid-cols", type=int, default=16)
    p.add_argument("--source-frame", type=int, default=1)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args(argv)
    if not args.summarize and (args.prompt_index is None or args.step is None):
        p.error("--prompt-index and --step are required unless --summarize")
    if args.draws < 2 or args.per_group < 1:
        p.error("Need >= 2 draws and >= 1 patch per group")
    args.weights, args.num_output_frames, args.source_chunk, args.halo = "generator", 6, 0, 2
    args.steps = [args.step]
    return args


def select(entry, per_group, token_class):
    """Lowest / highest U among patches of one token count (ties by id)."""
    u, tokens = np.array(entry["patch_maps"]["U"]), np.array(entry["num_tokens"])
    candidates = sorted((i for i in range(len(u)) if tokens[i] == token_class), key=lambda i: (u[i], i))
    if len(candidates) < 2 * per_group:
        raise ValueError(f"Only {len(candidates)} patches with {token_class} tokens")
    return {"low": candidates[:per_group], "high": candidates[-per_group:]}


def pairwise_mean(items, distance):
    return float(np.mean([distance(a, b) for a, b in itertools.combinations(items, 2)]))


def fresh_noise(like, seed, step, source_id, draw):
    rng = torch.Generator(device="cpu").manual_seed(1_000_003 * seed + 104_729 * source_id + 7_919 * draw + 31 * step)
    return torch.randn(like.shape, generator=rng, dtype=torch.float32).to(like)


@torch.inference_mode()
def run(args):
    from PIL import Image
    from scripts.patch import ar_patch_rollout as old
    from scripts.patch.patch_noise_importance import (PairedRollout, atomic_json, load_pipeline, make_noise_bank,
                                                tensor_hash)
    from scripts.patch.patch_noise_strength import decode_pixels
    from scripts.patch.patch_noise_utils import build_regions, replace_patch_noise

    entry = next(e for e in json.loads(args.uncertainty.read_text())["cases"]
                 if e["prompt_index"] == args.prompt_index and e["step_index"] == args.step)
    groups = select(entry, args.per_group, args.token_class)
    case_dir = args.output_dir / f"prompt_{args.prompt_index:06d}_seed_{args.seed:06d}" / f"step_{args.step}"
    case_dir.mkdir(parents=True, exist_ok=True)
    pipeline, _, shape, chunks, checkpoint = load_pipeline(args)
    prompt = args.data_path.read_text().splitlines()[args.prompt_index].strip()
    regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                          args.seed, parameter.dtype, parameter.device)
    replay = old.TracedRollout(pipeline, conditional, initial, transition, chunks)
    baseline = replay.baseline(0, args.step)["output"]
    trace = replay.take_trace()
    replay.capture = False
    state = old.source_state(trace, transition, 0, args.step, parameter.device)
    sigma = next(e["sigma"] for e in trace if (e["chunk_index"], e["step_index"]) == (0, args.step))
    unchanged = replace_patch_noise(state["noisy"], state["clean"], state["noise"], sigma, regions[0], args.source_frame)
    _, identity = replay.branch(unchanged, 0, args.step, complete=True)
    if not torch.equal(unchanged, state["noisy"]) or not torch.equal(identity, baseline):
        raise RuntimeError("Identity replay failed")
    baseline_pixels = decode_pixels(pipeline, baseline)
    frame = args.source_frame
    records = []
    for group, ids in groups.items():
        for sid in ids:
            started = time.monotonic()
            region = regions[sid]
            mask = torch.ones(shape[1], shape[-2], shape[-1], dtype=torch.bool)
            mask[frame, max(0, region["y0"] - args.halo):region["y1"] + args.halo,
                 max(0, region["x0"] - args.halo):region["x1"] + args.halo] = False
            latents, videos, energies, last_frames = [], [], [], []
            for draw in range(1, args.draws + 1):
                changed = replace_patch_noise(state["noisy"], state["clean"],
                                              fresh_noise(state["noise"], args.seed, args.step, sid, draw),
                                              sigma, region, args.source_frame)
                energies.append(float((changed.float() - state["noisy"].float()).square().sum()))
                _, output = replay.branch(changed, 0, args.step, complete=True)
                latents.append(output[0].float().cpu())
                pixels = decode_pixels(pipeline, output)
                videos.append(pixels)
                last_frames.append(pixels[-1].mul(255).round().to(torch.uint8).numpy())
            sheet = np.concatenate([np.concatenate(last_frames[i:i + 4], axis=1) for i in range(0, len(last_frames), 4)])
            Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(
                case_dir / f"{group}_source_{sid:03d}_last_frames.jpg", quality=88)
            m = mask[:, None]
            records.append({
                "group": group, "source_id": sid, "num_tokens": entry["num_tokens"][sid],
                "U": entry["patch_maps"]["U"][sid], "B": entry["patch_maps"]["B"][sid],
                "seed0_gap": entry["patch_maps"]["seed0_gap"][sid],
                "importance_mean": entry["patch_maps"]["importance_mean"][sid],
                "input_energy_mean": float(np.mean(energies)),
                "spread_video_rmse": pairwise_mean(videos, lambda a, b: float((a - b).square().mean().sqrt())),
                "spread_outside_latent_mse": pairwise_mean(latents, lambda a, b: float((a - b).square()[
                    m.expand_as(a)].mean())),
                "spread_next_chunk_mse": pairwise_mean(latents, lambda a, b: float(
                    (a[chunks[0]:] - b[chunks[0]:]).square().mean())),
                "mean_video_rmse_to_baseline": float(np.mean([(v - baseline_pixels).square().mean().sqrt() for v in videos])),
                "seconds": time.monotonic() - started})
            print(f"{group:4s} source {sid:03d} U={records[-1]['U']:.4f} spread_rmse={records[-1]['spread_video_rmse']:.4f} "
                  f"outside={records[-1]['spread_outside_latent_mse']:.4f} {records[-1]['seconds']:.0f}s", flush=True)
            atomic_json(case_dir / "records.json", {"prompt_index": args.prompt_index, "prompt": prompt, "seed": args.seed,
                                                     "step_index": args.step, "sigma": sigma, "draws": args.draws,
                                                     "token_class": args.token_class, "groups": groups,
                                                     "baseline_sha256": tensor_hash(baseline),
                                                     "checkpoint_path": str(checkpoint.resolve()), "records": records})
    print(f"DONE {case_dir}", flush=True)


def summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, spearmanr

    metrics = ["spread_video_rmse", "spread_outside_latent_mse", "spread_next_chunk_mse"]
    cases = []
    for path in sorted(args.output_dir.glob("prompt_*/step_*/records.json")):
        data = json.loads(path.read_text())
        if len(data["records"]) != sum(len(v) for v in data["groups"].values()):
            raise ValueError(f"Incomplete case: {path}")
        by = {g: [r for r in data["records"] if r["group"] == g] for g in ("low", "high")}
        case = {"prompt_index": data["prompt_index"], "step_index": data["step_index"], "metrics": {},
                "input_energy_high_over_low": float(np.mean([r["input_energy_mean"] for r in by["high"]])
                                                    / np.mean([r["input_energy_mean"] for r in by["low"]]))}
        for metric in metrics:
            low, high = [r[metric] for r in by["low"]], [r[metric] for r in by["high"]]
            all_r = by["low"] + by["high"]
            case["metrics"][metric] = {
                "low_median": float(np.median(low)), "high_median": float(np.median(high)),
                "high_over_low_median": float(np.median(high) / np.median(low)),
                "mann_whitney_p_one_sided": float(mannwhitneyu(high, low, alternative="greater").pvalue),
                "spearman_U": float(spearmanr([r["U"] for r in all_r], [r[metric] for r in all_r]).statistic)}
        cases.append({**case, "_by": by})
        v = case["metrics"]["spread_video_rmse"]
        print(f"#{case['prompt_index']:3d} s{case['step_index']}: video spread low {v['low_median']:.4f} high "
              f"{v['high_median']:.4f} x{v['high_over_low_median']:.2f} p={v['mann_whitney_p_one_sided']:.4f} "
              f"rho(U)={v['spearman_U']:+.2f} | outside x{case['metrics']['spread_outside_latent_mse']['high_over_low_median']:.2f}"
              f" | energy high/low {case['input_energy_high_over_low']:.3f}", flush=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6))
    for ax, metric in zip(axes, metrics):
        positions = np.arange(len(cases))
        for offset, group, color in ((-0.18, "low", "tab:blue"), (0.18, "high", "tab:red")):
            data = [[r[metric] for r in c["_by"][group]] for c in cases]
            parts = ax.boxplot(data, positions=positions + offset, widths=0.3, patch_artist=True, showfliers=False)
            for box in parts["boxes"]:
                box.set_facecolor(color), box.set_alpha(0.45)
            for x, values in zip(positions + offset, data):
                ax.scatter(np.full(len(values), x), values, s=9, color=color)
        for x, c in zip(positions, cases):
            ax.text(x, ax.get_ylim()[1], f"p={c['metrics'][metric]['mann_whitney_p_one_sided']:.3f}",
                    ha="center", va="top", fontsize=7)
        ax.set_xticks(positions, [f"#{c['prompt_index']}\ns{c['step_index']}" for c in cases], fontsize=8)
        ax.set_title(f"{metric}: low U (blue) vs high U (red)", fontsize=10)
    fig.suptitle("Resampling only one patch's noise: diversity among K outputs (one-sided Mann-Whitney)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "local_resample_diversity.png", dpi=100)
    plt.close(fig)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"description": __doc__, "cases": [{k: v for k, v in c.items() if k != "_by"} for c in cases]},
        ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.summarize:
        summarize(arguments)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Run on a GPU machine")
        run(arguments)
