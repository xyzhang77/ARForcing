"""Patch uncertainty from resampled future noise; gap decomposition; what drives importance.

Inputs
  --draws   outputs of scripts/patch/run_noise_draws.sh: draw_<d>/prompt_*/step_<s>/ with
            baseline_latent.pt (all draws) and importance_shard_00.json (draws 1-4: 192 patches, scale 0)
  --orig    outputs/patch_importance/ar_patch_importance_20261026: draw-0 importance, attention.npz, x0_step_gap/x0_steps.pt
Draw d of step_index s keeps everything up to the Step s x0 (1-based) and redraws chunk-0 noise
of calls >= s, so the final chunk-0 x0 samples are conditional on the injected x0 (anchor).

Exp 1/2  per patch on latent frame 1, over the draws (draw 0 = original noise included):
  U = variance of final x0 (uncertainty), B = |mean final - anchor|^2 (systematic correction),
  mean gap = U(ddof=0) + B exactly; seed-0 gap is one sample of the left side.
Exp 0    importance reliability over draws 0..4: mean pairwise Spearman, ICC(1) on log values,
         Spearman-Brown reliability of the 5-draw mean.
Exp 3    targets = 5-draw mean of importance (video RMSE) and outside-patch change; partial
         Spearman of U, B, attention, gap against the target with the others plus tokens,
         input energy and texture removed; p from 767 spatial maps of the tested predictor only.
Run: python scripts/patch/analyze_patch_uncertainty.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.analyze_x0_gap_importance import (p_value, partial_rho, percentile, ranks, spatial_null,
                                               texture_map)
from scripts.patch.ar_x0_step_gap import region_means
from scripts.patch.summarize_ar_patch_attention import attention_scores, rho

TARGETS = {"importance": ("pixel", "all_video_rmse"), "outside_halo": ("rollout", "outside_halo_mse")}
LABELS = {0: "stop sign", 42: "oasis", 158: "marching", 300: "cat"}


def decompose(samples, anchor):
    """samples [D,...], anchor [...] -> elementwise (variance ddof=0, bias^2, mean squared gap)."""
    mean = samples.mean(axis=0)
    var, bias = samples.var(axis=0), (mean - anchor) ** 2
    gap = ((samples - anchor) ** 2).mean(axis=0)
    if not np.allclose(gap, var + bias, rtol=1e-4, atol=1e-6):
        raise RuntimeError("gap != variance + bias")
    return var, bias, gap


def icc1(x):
    """One-way random-effects ICC(1) for x [k raters, n subjects]."""
    k, n = x.shape
    msb = k * x.mean(axis=0).var(ddof=1)
    msw = ((x - x.mean(axis=0)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def records(path):
    data = json.loads(path.read_text())
    if not data.get("complete"):
        raise ValueError(f"Incomplete shard: {path}")
    return data, [r for r in data["records"] if r["sigma_scale"] == 0.0]


def load_case(draws_root, orig_root, prompt, step, n_draws, n_scans):
    name = f"prompt_{prompt:06d}_seed_000000"
    orig = orig_root / name / f"step_{step}"
    shards = [json.loads(p.read_text()) for p in sorted(orig.glob("importance_shard_*.json"))]
    protocol = shards[0]["protocol"]
    draw0 = sorted((r for d in shards for r in d["records"] if r["sigma_scale"] == 0.0), key=lambda r: r["source_id"])
    x0 = torch.load(orig_root / name / "x0_step_gap" / "x0_steps.pt", weights_only=True)["x0"].float().numpy()
    frame, regions = protocol["source_frame"], protocol["regions"]
    finals, scans, floors = [x0[-1][frame]], [draw0], []
    for d in range(1, n_draws + 1):
        directory = draws_root / f"draw_{d}" / name / f"step_{step}"
        data, recs = records(directory / "importance_shard_00.json")
        if data["protocol"]["draw0_baseline_sha256"] != protocol["baseline_sha256"]:
            raise ValueError(f"Draw {d} is not paired with the original baseline: {directory}")
        if data["protocol"]["resampled_noise"] != [f"0:{c}" for c in range(step, 4)]:
            raise ValueError(f"Unexpected resampled noise in {directory}")
        latent = torch.load(directory / "baseline_latent.pt", weights_only=True).float().numpy()
        finals.append(latent[0, frame])
        floors.append(data["noise_floor"])
        if d <= n_scans:
            if len(recs) != len(regions):
                raise ValueError(f"Draw {d} scan incomplete: {directory}")
            scans.append(sorted(recs, key=lambda r: r["source_id"]))
    return protocol, x0, np.stack(finals), scans, floors


def patch_means(values, regions):
    return region_means(values.mean(axis=0)[None], regions, 0)[0]


def analyze_case(draws_root, orig_root, prompt, step, n_draws, n_scans):
    protocol, x0, finals, scans, floors = load_case(draws_root, orig_root, prompt, step, n_draws, n_scans)
    regions, frame, (rows, cols) = protocol["regions"], protocol["source_frame"], protocol["grid"]
    anchor = x0[step - 1][frame]
    var, bias, gap_mean = decompose(finals, anchor)
    d = len(finals)
    u = patch_means(var, regions) * d / (d - 1)
    b = patch_means(bias, regions)
    gap0 = patch_means((x0[-1][frame] - anchor) ** 2, regions)
    texture = texture_map(x0[-1][frame], regions)
    tokens = np.array([r["num_tokens"] for r in scans[0]], dtype=float)
    energy = np.mean([[r["rollout"]["input_energy"] for r in s] for s in scans], axis=0)
    attention = attention_scores(np.load(orig_root / f"prompt_{prompt:06d}_seed_000000" / f"step_{step}" /
                                         "attention.npz")["region_flow"].astype(np.float64),
                                 tokens, rows * cols, frame)["received_all"].mean(axis=0)
    importance = np.array([[r["pixel"]["all_video_rmse"] for r in s] for s in scans])
    log_imp = np.log(importance)
    k = len(scans)
    pairs = [rho(importance[i], importance[j]) for i in range(k) for j in range(i + 1, k)]
    reliability = {"draws": k, "mean_pairwise_spearman": float(np.mean(pairs)),
                   "min_pairwise_spearman": float(np.min(pairs)), "icc1_log": icc1(log_imp)}
    reliability["spearman_brown_mean_of_draws"] = k * reliability["icc1_log"] / (1 + (k - 1) * reliability["icc1_log"])
    perms = spatial_null(rows, cols)
    controls = [tokens, energy, texture]
    entry = {"prompt_index": prompt, "step_index": step, "anchor": f"Step {step} x0", "num_final_samples": d,
             "mean_U": float(u.mean()), "mean_B": float(b.mean()), "mean_seed0_gap": float(gap0.mean()),
             "U_share_of_mean_gap": float(u.mean() * (d - 1) / d / patch_means(gap_mean, regions).mean()),
             "noise_floor_video_rmse_mean": float(np.mean([f["pixel"]["all_video_rmse"] for f in floors])),
             "mean_importance": float(importance.mean()), "importance_reliability": reliability,
             "rho": {"seed0_gap~U": rho(gap0, u), "seed0_gap~B": rho(gap0, b), "U~B": rho(u, b),
                     "U~texture": rho(u, texture), "B~texture": rho(b, texture), "U~attention": rho(u, attention)},
             "targets": {}}
    for name, (group, key) in TARGETS.items():
        y = np.exp(np.mean(np.log([[r[group][key] for r in s] for s in scans]), axis=0))
        tests = {"U": (u, [b, attention] + controls), "B": (b, [u, attention] + controls),
                 "attention": (attention, [u, b] + controls), "gap|controls": (gap0, controls),
                 "gap|U+controls": (gap0, [u] + controls)}
        out = {"raw_rho": {"U": rho(u, y), "B": rho(b, y), "attention": rho(attention, y), "gap": rho(gap0, y)}}
        for label, (x, cov) in tests.items():
            value = partial_rho(x, y, cov)
            out[label] = {"partial_rho": value, "p_spatial": p_value(value, [partial_rho(x[p], y, cov) for p in perms])}
        ru, ra = ranks(u) - ranks(u).mean(), ranks(attention) - ranks(attention).mean()
        value = partial_rho(ru * ra, y, [u, attention, b] + controls)
        out["U_x_attention"] = {"partial_rho": value,
                                "p_spatial": p_value(value, [partial_rho(ru[p] * ra, y, [u[p], attention, b] + controls)
                                                             for p in perms])}
        entry["targets"][name] = out
    maps = {"U": u, "B": b, "seed0_gap": gap0, "importance_mean": np.exp(log_imp.mean(axis=0))}
    entry["patch_maps"] = {key: value.tolist() for key, value in maps.items()}
    entry["num_tokens"] = tokens.tolist()
    return entry, maps, (rows, cols)


def figures(entries, maps, grid, orig_root, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    rows, cols = grid
    keys = [("U", "uncertainty U", "viridis"), ("B", "systematic correction B", "cividis"),
            ("seed0_gap", "seed-0 gap", "viridis"), ("importance_mean", "importance (5-draw mean)", "magma")]
    fig, axes = plt.subplots(len(entries), 4, figsize=(17, 2.75 * len(entries)), squeeze=False)
    for i, (entry, m) in enumerate(zip(entries, maps)):
        background = np.asarray(Image.open(orig_root / f"prompt_{entry['prompt_index']:06d}_seed_000000" /
                                           "x0_step_gap" / "x0_step4.png"))
        for j, (key, title, cmap) in enumerate(keys):
            ax = axes[i, j]
            ax.imshow(background, extent=(0, cols, rows, 0))
            ax.imshow(percentile(m[key]).reshape(rows, cols), cmap=cmap, vmin=0, vmax=100, alpha=0.7,
                      extent=(0, cols, rows, 0))
            extra = ""
            if key == "U":
                extra = f"  share of gap {entry['U_share_of_mean_gap']:.0%}"
            elif key in ("B", "seed0_gap"):
                extra = f"  rho vs U {entry['rho']['U~B' if key == 'B' else 'seed0_gap~U']:+.2f}"
            elif key == "importance_mean":
                extra = f"  rho vs U {entry['targets']['importance']['raw_rho']['U']:+.2f}"
            ax.set_title(f"#{entry['prompt_index']} s{entry['step_index']}: {title} (pct){extra}", fontsize=8.5)
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_maps.png", dpi=90)
    plt.close(fig)

    tests = ["U", "B", "attention", "gap|controls", "gap|U+controls", "U_x_attention"]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(8.5 * len(TARGETS), 4.6), squeeze=False)
    width = 0.8 / len(tests)
    for ax, target in zip(axes[0], TARGETS):
        for t, test in enumerate(tests):
            values = [e["targets"][target][test]["partial_rho"] for e in entries]
            bars = ax.bar(np.arange(len(entries)) + (t - len(tests) / 2 + 0.5) * width, values, width, label=test)
            for bar, e in zip(bars, entries):
                if e["targets"][target][test]["p_spatial"] < 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*", ha="center",
                            va="bottom" if bar.get_height() >= 0 else "top", fontsize=9)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(entries)),
                      [f"#{e['prompt_index']} {LABELS.get(e['prompt_index'], '')}\ns{e['step_index']}" for e in entries],
                      fontsize=8)
        ax.set_ylabel("partial Spearman rho")
        ax.set_title(f"target: {target} (5-draw mean); * spatial p < 0.05")
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_partial_rho.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=Path, default=Path("outputs/patch_importance/ar_noise_draws_20261026"))
    parser.add_argument("--orig", type=Path, default=Path("outputs/patch_importance/ar_patch_importance_20261026"))
    parser.add_argument("--prompts", type=int, nargs="+", default=[0, 42, 158, 300])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-draws", type=int, default=16)
    parser.add_argument("--num-scans", type=int, default=4)
    args = parser.parse_args()
    entries, maps = [], []
    for prompt in args.prompts:
        for step in args.steps:
            entry, m, grid = analyze_case(args.draws, args.orig, prompt, step, args.num_draws, args.num_scans)
            entries.append(entry)
            maps.append(m)
            rel, tgt = entry["importance_reliability"], entry["targets"]
            print(f"#{prompt:3d} s{step}: U-share {entry['U_share_of_mean_gap']:.0%}  gap~U {entry['rho']['seed0_gap~U']:+.2f}"
                  f"  gap~B {entry['rho']['seed0_gap~B']:+.2f}  U~tex {entry['rho']['U~texture']:+.2f}"
                  f"  B~tex {entry['rho']['B~texture']:+.2f} | reliab pair {rel['mean_pairwise_spearman']:.2f}"
                  f" ICC {rel['icc1_log']:.2f} mean-rel {rel['spearman_brown_mean_of_draws']:.2f}"
                  f" | floor {entry['noise_floor_video_rmse_mean']:.3f} imp {entry['mean_importance']:.3f}", flush=True)
            for name in TARGETS:
                print(f"      {name:12s} raw U {tgt[name]['raw_rho']['U']:+.2f} | partial " + "  ".join(
                    f"{k}={tgt[name][k]['partial_rho']:+.2f}(p{tgt[name][k]['p_spatial']:.3f})"
                    for k in ("U", "B", "attention", "gap|controls", "gap|U+controls", "U_x_attention")), flush=True)
    figures(entries, maps, grid, args.orig, args.draws)
    (args.draws / "uncertainty_summary.json").write_text(
        json.dumps({"description": __doc__, "cases": entries}, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {args.draws / 'uncertainty_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
