"""Merge ar_patch_attention shards: importance by noise scale, and importance vs attention.

importance_by_scale.png / summary.json "importance_by_scale": full-video RGB RMSE
per source patch for every local-noise scale, on one shared colour range.

Attention scores per source-frame region (latent frame 1 of chunk 0), from the
baseline call at the same denoising step, per token of the region:
  received_all    all 4680 chunk-0 queries -> region keys
  received_other  queries outside the region itself
  cross_frame     queries in latent frames 0 and 2 only
Spearman rho is reported per layer and for the 30-layer mean.
Run: python scripts/patch/summarize_ar_patch_attention.py --root outputs/patch_importance/ar_patch_attention_20261026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

SCORES = ("received_all", "received_other", "cross_frame")


def attention_scores(flow, counts, regions, source_frame):
    """flow [L, G, G] query-group x key-group attention mass -> {score: [L, R]} per token."""
    keys = slice(source_frame * regions, (source_frame + 1) * regions)
    column = flow[:, :, keys]                                      # [L, G, R]
    diag = column[:, keys, :].diagonal(axis1=1, axis2=2)           # self-region mass
    same_frame = column[:, keys, :].sum(1)
    out = {"received_all": column.sum(1), "received_other": column.sum(1) - diag,
           "cross_frame": column.sum(1) - same_frame}
    return {k: v / counts[None] for k, v in out.items()}


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def load_case(case_dir):
    shards = sorted(case_dir.glob("importance_shard_*.json"))
    data = [json.loads(p.read_text()) for p in shards]
    if not data or not all(d.get("complete") for d in data):
        raise ValueError(f"Incomplete shards in {case_dir}")
    protocol = data[0]["protocol"]
    if any(d["protocol"] != protocol for d in data):
        raise ValueError(f"Shard protocols differ in {case_dir}")
    scales = protocol.get("sigma_scales", [0.0])
    by_scale = {s: sorted((r for d in data for r in d["records"] if r.get("sigma_scale", 0.0) == s),
                          key=lambda r: r["source_id"]) for s in scales}
    for s, records in by_scale.items():
        if [r["source_id"] for r in records] != list(range(len(protocol["regions"]))):
            raise ValueError(f"Missing or duplicated sources at scale {s} in {case_dir}")
    return protocol, by_scale, np.load(case_dir / "attention.npz")


def scale_overview(case_dir, protocol, by_scale):
    """Importance maps for every local-noise scale on one shared colour range."""
    rows, cols = protocol["grid"]
    maps = {s: np.array([r["pixel"]["all_video_rmse"] for r in rec]) for s, rec in by_scale.items()}
    reference = maps[min(maps)]
    stats = {str(s): {"local_sigma": protocol["base_sigma"] * s, "mean_rmse": float(m.mean()),
                      "median_rmse": float(np.median(m)), "max_rmse": float(m.max()),
                      "top_source_ids": np.argsort(-m)[:5].tolist(),
                      "spearman_vs_lowest_scale": rho(m, reference)} for s, m in maps.items()}
    frame = np.asarray(Image.open(case_dir / "baseline_source_frame.png"))
    vmax = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(5.2 * (len(maps) + 1), 4.2))
    axes[0].imshow(frame)
    axes[0].set_title("baseline, decoded frame 4")
    for ax, (s, m) in zip(axes[1:], sorted(maps.items(), reverse=True)):
        ax.imshow(frame, alpha=0.35, extent=(0, cols, rows, 0))
        im = ax.imshow(m.reshape(rows, cols), cmap="magma", vmin=0, vmax=vmax, alpha=0.75, extent=(0, cols, rows, 0))
        ax.set_title(f"local sigma = {s:g} x {protocol['base_sigma']:.4f}\nmean RMSE {m.mean():.4f}", fontsize=10)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axes[1:].tolist(), fraction=0.02)
    fig.suptitle(f"#{protocol['prompt_index']} \"{protocol['prompt']}\"  step_index {protocol['step_index']}: "
                 "full-video RGB RMSE after lowering one patch's noise")
    fig.savefig(case_dir / "importance_by_scale.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/patch_importance/ar_patch_attention_20261026"))
    parser.add_argument("--scale", type=float, default=0.0, help="Noise scale used for the attention comparison")
    args = parser.parse_args()
    cases = []
    for case_dir in sorted(args.root.glob("prompt_*_seed_*/step_*")):
        protocol, by_scale, attn = load_case(case_dir)
        records = by_scale[args.scale]
        rows, cols = protocol["grid"]
        counts = np.array([r["num_tokens"] for r in records], dtype=float)
        importance = np.array([r["pixel"]["all_video_rmse"] for r in records])
        scores = attention_scores(attn["region_flow"].astype(np.float64), counts, rows * cols,
                                  protocol["source_frame"])
        entry = {"prompt_index": protocol["prompt_index"], "prompt": protocol["prompt"], "seed": protocol["seed"],
                 "step_index": protocol["step_index"], "base_sigma": protocol["base_sigma"],
                 "importance_by_scale": scale_overview(case_dir, protocol, by_scale),
                 "attention_scale": args.scale,
                 "importance_vs_num_tokens_rho": rho(importance, counts),
                 "attention_max_rel_error": float(attn["rel_error"].max()), "spearman": {}}
        for name, value in scores.items():
            entry["spearman"][name] = {"layer_mean": rho(importance, value.mean(0)),
                                       "per_layer": [rho(importance, layer) for layer in value]}
        cases.append({"dir": case_dir, "entry": entry, "importance": importance.reshape(rows, cols),
                      "scores": {k: v.reshape(len(v), rows, cols) for k, v in scores.items()}})
        print(f"prompt {protocol['prompt_index']:3d} step {protocol['step_index']}: mean RMSE by scale " + "  ".join(
            f"{s}:{v['mean_rmse']:.4f}(rho {v['spearman_vs_lowest_scale']:+.2f})"
            for s, v in entry["importance_by_scale"].items()), flush=True)
        print(f"prompt {protocol['prompt_index']:3d} step {protocol['step_index']}: " + "  ".join(
            f"{k}={entry['spearman'][k]['layer_mean']:+.3f}" for k in SCORES)
            + f"  tokens={entry['importance_vs_num_tokens_rho']:+.3f}")
    if not cases:
        raise SystemExit("No cases found")

    pooled = {}
    for step in sorted({c["entry"]["step_index"] for c in cases}):
        sel = [c for c in cases if c["entry"]["step_index"] == step]
        # Rank within each prompt first, so prompt-level magnitude differences do not dominate.
        ranks = lambda xs: np.concatenate([np.argsort(np.argsort(x.ravel())) for x in xs])
        pooled[str(step)] = {k: rho(ranks([c["importance"] for c in sel]),
                                    ranks([c["scores"][k].mean(0) for c in sel])) for k in SCORES}
    summary = {"description": __doc__, "cases": [c["entry"] for c in cases],
               "pooled_within_prompt_rank_spearman": pooled}
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    for c in cases:
        e = c["entry"]
        frame = np.asarray(Image.open(c["dir"] / "baseline_source_frame.png"))
        fig, axes = plt.subplots(1, 5, figsize=(26, 4.6))
        axes[0].imshow(frame)
        axes[0].set_title("baseline, decoded frame 4 (latent frame 1)")
        panels = [("importance (full-video RMSE)", c["importance"], "magma")] + [
            (f"{k}, layer mean  rho={e['spearman'][k]['layer_mean']:+.2f}", c["scores"][k].mean(0), "viridis")
            for k in SCORES]
        for ax, (title, value, cmap) in zip(axes[1:], panels):
            ax.imshow(frame, alpha=0.35, extent=(0, 16, 12, 0))
            im = ax.imshow(value, cmap=cmap, alpha=0.75, extent=(0, 16, 12, 0))
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.035)
        for ax in axes:
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"#{e['prompt_index']} \"{e['prompt']}\"  seed {e['seed']}  step_index {e['step_index']} "
                     f"(sigma={e['base_sigma']:.4f})")
        fig.tight_layout()
        fig.savefig(c["dir"] / "importance_vs_attention.png", dpi=110)
        plt.close(fig)

    steps = sorted({c["entry"]["step_index"] for c in cases})
    fig, axes = plt.subplots(len(steps), len(SCORES), figsize=(6 * len(SCORES), 3.6 * len(steps)), squeeze=False)
    for i, step in enumerate(steps):
        for j, k in enumerate(SCORES):
            ax = axes[i, j]
            for c in (c for c in cases if c["entry"]["step_index"] == step):
                ax.plot(c["entry"]["spearman"][k]["per_layer"], marker=".", label=f"#{c['entry']['prompt_index']}")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-1, 1)
            ax.set_title(f"step_index {step}: {k}")
            ax.set_xlabel("layer")
            ax.set_ylabel("Spearman rho vs importance")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.root / "per_layer_spearman.png", dpi=110)
    plt.close(fig)
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
