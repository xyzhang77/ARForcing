"""Matched x0 step gap vs patch importance, with confound control and spatial null.

Pairing: intervention at step_index s injects the x0 predicted at Step s (1-based), so the
matched gap is Step s -> Step 4 (final): step_index 1 -> "1-4", step_index 2 -> "2-4".
Partial Spearman removes rank-linear effects of tokens per patch, input energy and final-x0
gradient energy (texture) from both variables. p-values use torus shifts x 4 flips of the
12x16 gap map (767 non-identity maps), which keeps neighbouring patches together.
Targets split the effect: inside the patch (source_mse), outside it (outside_halo_mse),
next chunk (next_chunk_mse), whole video (importance).
Figures (scale 0, all in within-prompt percentiles so prompts share one axis):
  x0_gap_vs_importance_trend.png  gap quintile -> mean importance / outside-patch percentile
  x0_gap_vs_importance_maps.png   gap and importance percentile maps + top-20 overlap
  x0_gap_matched_scatter.png      percentile scatter with decile means
Needs outputs of ar_patch_attention.py and ar_x0_step_gap.py. CPU only.
Run: python scripts/patch/analyze_x0_gap_importance.py --root outputs/patch_importance/ar_patch_importance_20261026
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

from scripts.patch.ar_x0_step_gap import region_means
from scripts.patch.summarize_ar_patch_attention import load_case, rho

MATCHED = {1: (1, 4), 2: (2, 4)}
TARGETS = {"importance": ("pixel", "all_video_rmse"), "source_patch": ("rollout", "source_mse"),
           "outside_halo": ("rollout", "outside_halo_mse"), "next_chunk": ("rollout", "next_chunk_mse")}
TOP_K = 20
LABELS = {0: "stop sign", 42: "oasis", 158: "marching", 300: "cat"}


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def percentile(x):
    return ranks(x) / (len(x) - 1) * 100


def binned(gap, y, bins):
    """Mean and standard error of y's percentile within equal-count gap bins (low -> high)."""
    p = percentile(y)
    groups = np.array_split(np.argsort(gap, kind="stable"), bins)
    return (np.array([p[g].mean() for g in groups]),
            np.array([p[g].std(ddof=1) / np.sqrt(len(g)) for g in groups]))


def partial_rho(x, y, covariates):
    z = np.column_stack([np.ones(len(x))] + [ranks(c) for c in covariates])
    residual = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
    a, b = residual(ranks(x)), residual(ranks(y))
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def spatial_null(rows, cols):
    grid = np.arange(rows * cols).reshape(rows, cols)
    maps = [np.roll(f, (dy, dx), (0, 1)).ravel() for f in (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1])
            for dy in range(rows) for dx in range(cols)]
    return np.array(maps[1:])


def top_jaccard(a, b, k=TOP_K):
    sa, sb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(sa & sb) / len(sa | sb)


def p_value(observed, null):
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))


def texture_map(final_frame, regions):
    gy = np.diff(final_frame, axis=1, append=final_frame[:, -1:])
    gx = np.diff(final_frame, axis=2, append=final_frame[:, :, -1:])
    return region_means((gx ** 2 + gy ** 2).mean(axis=0)[None], regions, 0)[0]


def analyze_case(step_dir, x0):
    protocol, by_scale, _ = load_case(step_dir)
    step, regions, frame = protocol["step_index"], protocol["regions"], protocol["source_frame"]
    a, b = MATCHED[step]
    gap = region_means(((x0[b - 1] - x0[a - 1]) ** 2).mean(axis=1), regions, frame)[0]
    texture = texture_map(x0[-1][frame], regions)
    perms = spatial_null(*protocol["grid"])
    entry = {"prompt_index": protocol["prompt_index"], "step_index": step, "gap_pair": f"{a}-{b}",
             "gap_vs_texture_rho": rho(gap, texture), "null_size": len(perms), "scales": {}}
    plot = None
    for scale, recs in by_scale.items():
        tokens = np.array([r["num_tokens"] for r in recs], dtype=float)
        energy = np.array([r["rollout"]["input_energy"] for r in recs])
        out = {"gap_vs_input_energy_rho": rho(gap, energy), "gap_vs_tokens_rho": rho(gap, tokens), "targets": {}}
        values = {}
        for name, (group, key) in TARGETS.items():
            y = values[name] = np.array([r[group][key] for r in recs])
            raw, part = rho(gap, y), partial_rho(gap, y, [tokens, energy, texture])
            out["targets"][name] = {
                "rho": raw, "p_spatial": p_value(raw, [rho(gap[p], y) for p in perms]),
                "partial_rho": part,
                "p_partial_spatial": p_value(part, [partial_rho(gap[p], y, [tokens, energy, texture[p]])
                                                    for p in perms])}
        importance = values["importance"]
        null = [top_jaccard(gap[p], importance) for p in perms]
        out["top_k"] = {"k": TOP_K, "jaccard": top_jaccard(gap, importance), "null_mean": float(np.mean(null))}
        out["top_k"]["p_spatial"] = p_value(out["top_k"]["jaccard"], null)
        out["quintile_mean_importance_percentile"] = binned(gap, importance, 5)[0].tolist()
        out["quintile_mean_outside_halo_percentile"] = binned(gap, values["outside_halo"], 5)[0].tolist()
        entry["scales"][str(scale)] = out
        if scale == 0.0:
            plot = {"gap": gap, "tokens": tokens, "grid": protocol["grid"], **values}
    return entry, plot


def plot_trend(plt, plots, path):
    steps = sorted({s for _, s in plots})
    fig, axes = plt.subplots(2, len(steps), figsize=(6.4 * len(steps), 8.8), squeeze=False)
    centers = np.arange(1, 6)
    for col, step in enumerate(steps):
        cases = sorted(p for p, s in plots if s == step)
        for row, (target, label) in enumerate((("importance", "whole-video RMSE (importance)"),
                                               ("outside_halo", "change outside the patch (latent MSE)"))):
            ax = axes[row, col]
            curves = []
            for prompt in cases:
                d = plots[prompt, step]
                mean, se = binned(d["gap"], d[target], 5)
                curves.append(mean)
                ax.errorbar(centers, mean, yerr=se, marker="o", capsize=3, alpha=0.75,
                            label=f"#{prompt} {LABELS.get(prompt, '')} (rho {rho(d['gap'], d[target]):+.2f})")
            pooled_gap = np.concatenate([percentile(plots[p, step]["gap"]) for p in cases])
            pooled_y = np.concatenate([percentile(plots[p, step][target]) for p in cases])
            ax.plot(centers, np.mean(curves, axis=0), color="k", lw=3,
                    label=f"mean of prompts (pooled rank rho {rho(pooled_gap, pooled_y):+.2f})")
            ax.axhline(50, color="grey", ls="--", lw=1)
            ax.text(5.05, 51, "no relation", color="grey", fontsize=8, ha="right")
            ax.set_xticks(centers, ["lowest\n20%", "2", "3", "4", "highest\n20%"])
            ax.set_ylim(0, 100)
            a, b = MATCHED[step]
            ax.set_xlabel(f"x0 gap Step {a}->{b} quintile (38-39 patches each)")
            ax.set_ylabel(f"mean percentile of {label}")
            ax.set_title(f"intervene at step_index {step} (inject Step {a} x0), scale 0", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Patches whose x0 still changes a lot (high gap) get higher importance", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_maps(plt, plots, root, path):
    from PIL import Image
    prompts = sorted({p for p, _ in plots})
    steps = sorted({s for _, s in plots})
    fig, axes = plt.subplots(len(prompts), 3 * len(steps), figsize=(4.3 * 3 * len(steps), 2.9 * len(prompts)),
                             squeeze=False)
    colors = {"both": (0.85, 0.1, 0.1, 0.85), "gap": (0.15, 0.35, 0.95, 0.75), "imp": (1.0, 0.6, 0.0, 0.75)}
    for i, prompt in enumerate(prompts):
        background = np.asarray(Image.open(root / f"prompt_{prompt:06d}_seed_000000" / "x0_step_gap" / "x0_step4.png"))
        for k, step in enumerate(steps):
            d = plots[prompt, step]
            rows, cols = d["grid"]
            extent = (0, cols, rows, 0)
            top_gap = set(np.argsort(-d["gap"])[:TOP_K])
            top_imp = set(np.argsort(-d["importance"])[:TOP_K])
            overlap = np.zeros((rows * cols, 4))
            for idx in range(rows * cols):
                key = "both" if idx in top_gap and idx in top_imp else "gap" if idx in top_gap else (
                    "imp" if idx in top_imp else None)
                if key:
                    overlap[idx] = colors[key]
            a, b = MATCHED[step]
            panels = [(f"x0 gap {a}->{b} percentile", percentile(d["gap"]), "viridis"),
                      ("importance percentile", percentile(d["importance"]), "magma")]
            for m, (title, value, cmap) in enumerate(panels):
                ax = axes[i, 3 * k + m]
                ax.imshow(background, extent=extent)
                ax.imshow(value.reshape(rows, cols), cmap=cmap, vmin=0, vmax=100, alpha=0.7, extent=extent)
                ax.set_title(f"#{prompt} s{step}: {title}", fontsize=9)
            ax = axes[i, 3 * k + 2]
            ax.imshow(background, extent=extent)
            ax.imshow(overlap.reshape(rows, cols, 4), extent=extent)
            ax.set_title(f"#{prompt} s{step}: top-{TOP_K} red=both ({len(top_gap & top_imp)}), "
                         f"blue=gap only, orange=importance only", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Where the x0 gap and the importance agree (scale 0; chance overlap of two top-20 sets ~2 patches)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_scatter(plt, plots, path):
    prompts = sorted({p for p, _ in plots})
    steps = sorted({s for _, s in plots})
    fig, axes = plt.subplots(len(steps), len(prompts), figsize=(4.4 * len(prompts), 4.4 * len(steps)),
                             squeeze=False)
    for i, step in enumerate(steps):
        for j, prompt in enumerate(prompts):
            d, ax = plots[prompt, step], axes[i, j]
            x, y = percentile(d["gap"]), percentile(d["importance"])
            ax.scatter(x, y, s=10, alpha=0.45, color="tab:blue")
            groups = np.array_split(np.argsort(x), 10)
            ax.plot([x[g].mean() for g in groups], [y[g].mean() for g in groups], "o-", color="crimson", lw=2.2,
                    label="decile mean")
            ax.plot([0, 100], [0, 100], ls=":", color="grey", lw=1)
            ax.axhline(50, color="grey", ls="--", lw=0.8)
            ax.set_xlim(0, 100), ax.set_ylim(0, 100)
            a, b = MATCHED[step]
            ax.set_xlabel(f"x0 gap {a}->{b} percentile")
            ax.set_ylabel("importance percentile")
            ax.set_title(f"#{prompt} {LABELS.get(prompt, '')}, step_index {step}: rho {rho(x, y):+.2f}", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/patch_importance/ar_patch_importance_20261026"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted(args.root.glob("prompt_*_seed_*"))
    if not cases:
        raise SystemExit("No cases found")
    results, plots = [], {}
    for case in cases:
        x0 = torch.load(case / "x0_step_gap" / "x0_steps.pt", weights_only=True)["x0"].float().numpy()
        for step_dir in sorted(case.glob("step_*")):
            entry, plot = analyze_case(step_dir, x0)
            results.append(entry)
            plots[entry["prompt_index"], entry["step_index"]] = plot
            for scale in ("0.0", "0.75"):
                v = entry["scales"][scale]
                print(f"#{entry['prompt_index']:3d} s{entry['step_index']} {entry['gap_pair']} x{scale:<4} " + " ".join(
                    f"{n}={x['rho']:+.2f}/{x['partial_rho']:+.2f}(p{x['p_partial_spatial']:.3f})"
                    for n, x in v["targets"].items())
                    + f" | top{TOP_K} J={v['top_k']['jaccard']:.2f} null={v['top_k']['null_mean']:.2f}"
                      f" p={v['top_k']['p_spatial']:.3f} | quintiles "
                      + "/".join(f"{q:.0f}" for q in v["quintile_mean_importance_percentile"]), flush=True)
    plot_trend(plt, plots, args.root / "x0_gap_vs_importance_trend.png")
    plot_maps(plt, plots, args.root, args.root / "x0_gap_vs_importance_maps.png")
    plot_scatter(plt, plots, args.root / "x0_gap_matched_scatter.png")
    (args.root / "x0_gap_matched_analysis.json").write_text(
        json.dumps({"description": __doc__, "cases": results}, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {args.root / 'x0_gap_matched_analysis.json'} and 3 figures", flush=True)


if __name__ == "__main__":
    main()
