"""Per-patch gap between AR x0 predictions at different denoising steps, vs patch importance.

Step naming is 1-based: "Step k" = k-th denoiser call of chunk 0 (step_index k-1).
Step 1 is at sigma=1; Step 4 is the last call, whose x0 is the chunk's final output.
The importance scan (scripts/patch/ar_patch_attention.py) intervened at step_index 1/2 = Step 2/3.
Hypothesis: a large x0 gap means the region is still undetermined, so changing its
noise level changes the result more (higher importance).

Only the unperturbed baseline is re-run per prompt; its hash must equal the importance
protocol's baseline hash. Gap = per-patch mean over channels/pixels of (x0_b - x0_a)^2.
Run: python scripts/patch/ar_x0_step_gap.py --root outputs/patch_importance/ar_patch_importance_20261026
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

from scripts.patch.summarize_ar_patch_attention import load_case, rho

PAIRS = [(1, 4), (2, 4), (3, 4), (1, 2), (2, 3)]
FOCUS = {1: "2-4", 2: "3-4"}  # change still to come after the intervened call


def region_means(gap, regions, frame):
    """gap [F,H,W] -> (mean on the intervened frame, mean on all frames) per region."""
    one = np.array([gap[frame, r["y0"]:r["y1"], r["x0"]:r["x1"]].mean() for r in regions])
    every = np.array([gap[:, r["y0"]:r["y1"], r["x0"]:r["x1"]].mean() for r in regions])
    return one, every


@torch.inference_mode()
def capture(root, cases):
    from PIL import Image
    from scripts.patch.ar_patch_attention import parse_args
    from scripts.patch.ar_patch_rollout import TracedRollout
    from scripts.patch.patch_noise_importance import load_pipeline, make_noise_bank, tensor_hash
    from scripts.patch.patch_noise_strength import decode_pixels

    args = parse_args(["--prompt-index", "0", "--step", "1"])
    pipeline, _, shape, chunks, checkpoint = load_pipeline(args)
    lines = args.data_path.read_text().splitlines()
    parameter = next(pipeline.generator.parameters())
    for case in cases:
        protocols = [load_case(d)[0] for d in sorted(case.glob("step_*"))]
        prompt_index, seed = protocols[0]["prompt_index"], protocols[0]["seed"]
        prompt = lines[prompt_index].strip()
        if any(p["prompt"] != prompt or p["seed"] != seed or p["checkpoint_path"] != str(checkpoint.resolve())
               for p in protocols):
            raise ValueError(f"Importance protocol differs from this run: {case}")
        pipeline.text_encoder.to(parameter.device)
        conditional = pipeline.text_encoder(text_prompts=[prompt])
        pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()
        initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                              seed, parameter.dtype, parameter.device)
        replay = TracedRollout(pipeline, conditional, initial, transition, chunks)
        output = replay.baseline(0, 1)["output"]
        trace = sorted((e for e in replay.take_trace() if e["chunk_index"] == 0), key=lambda e: e["step_index"])
        if any(tensor_hash(output) != p["baseline_sha256"] for p in protocols):
            raise RuntimeError(f"Baseline differs from the importance scan: {case}")
        x0 = torch.cat([e["clean_prediction"] for e in trace])
        if not torch.equal(x0[-1:], output[:, :chunks[0]].cpu()):
            raise RuntimeError("Step 4 x0 is not the chunk output")
        out = case / "x0_step_gap"
        out.mkdir(exist_ok=True)
        for k in range(len(x0)):
            frame = decode_pixels(pipeline, x0[k:k + 1].to(parameter.device))[4 * protocols[0]["source_frame"]]
            Image.fromarray(frame.mul(255).round().to(torch.uint8).numpy()).save(out / f"x0_step{k + 1}.png")
        torch.save({"x0": x0, "sigmas": [e["sigma"] for e in trace], "baseline_sha256": tensor_hash(output)},
                   out / "x0_steps.pt")
        print(f"CAPTURED {case.name} sigmas={[round(e['sigma'], 4) for e in trace]}", flush=True)


def patch_percentiles(values):
    """Rank patches on a 0–100 scale, assigning tied values their average rank."""
    from scipy.stats import rankdata
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite patch values")
    return (rankdata(values, method="average") - 1) / (values.size - 1) * 100


def plot_percentiles(case):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    protocol, by_scale, _ = load_case(case / "step_1")
    saved = torch.load(case / "x0_step_gap" / "x0_steps.pt", map_location="cpu", weights_only=True)
    if saved["baseline_sha256"] != protocol["baseline_sha256"]:
        raise ValueError(f"Gap and importance use different baselines: {case}")
    x0 = saved["x0"].float().numpy()
    gap = region_means(((x0[3] - x0[0]) ** 2).mean(axis=1),
                       protocol["regions"], protocol["source_frame"])[0]
    importance = np.array([r["pixel"]["all_video_rmse"] for r in by_scale[0.0]])
    rows, cols = protocol["grid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, step in zip(axes[0], (1, 4)):
        ax.imshow(Image.open(case / "x0_step_gap" / f"x0_step{step}.png"))
        ax.set_title(f"Step {step} clean prediction")
        ax.axis("off")
    for ax, title, values in zip(axes[1],
                                ("Step 1 -> 4 gap", "Importance (intervene at Step 2, scale 0)"),
                                (gap, importance)):
        im = ax.imshow(patch_percentiles(values).reshape(rows, cols), cmap="magma",
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_title(f"{title}\nWithin-map patch percentile", fontsize=11)
        ax.set_xlabel("Patch column (0-based)")
        ax.set_ylabel("Patch row (0-based)")
        ax.set_xticks(range(0, cols, 2))
        ax.set_yticks(range(0, rows, 2))
    fig.colorbar(im, ax=axes[1].tolist(), ticks=[0, 20, 40, 60, 80, 100],
                 format="%d%%", shrink=0.85)
    fig.suptitle(f"Prompt #{protocol['prompt_index']} | Same baseline trajectory | "
                 f"gap vs importance: rho={rho(gap, importance):+.3f}", fontsize=13)
    fig.supxlabel("Each map is ranked separately: 0% = lowest, 100% = highest. "
                  "Colors indicate relative rank, not probability.", fontsize=10)
    path = case / "x0_step_gap" / "x0_gap_vs_importance_percentile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"WROTE {path}", flush=True)
    return path


def analyze(root, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    results, ranks = [], {}
    for case in cases:
        saved = torch.load(case / "x0_step_gap" / "x0_steps.pt", weights_only=True)
        x0 = saved["x0"].float()
        steps = {}
        for d in sorted(case.glob("step_*")):
            protocol, by_scale, _ = load_case(d)
            steps[protocol["step_index"]] = {s: np.array([r["pixel"]["all_video_rmse"] for r in recs])
                                             for s, recs in by_scale.items()}
        regions, frame, (rows, cols) = protocol["regions"], protocol["source_frame"], protocol["grid"]
        tokens = np.array([r["num_tokens"] for r in by_scale[0.0]], dtype=float)
        gaps = {}
        for a, b in PAIRS:
            g = (x0[b - 1] - x0[a - 1]).square().mean(dim=1).numpy()
            gaps[f"{a}-{b}"] = dict(zip(("source_frame", "all_frames"), region_means(g, regions, frame)))
        entry = {"prompt_index": protocol["prompt_index"], "prompt": protocol["prompt"], "sigmas": saved["sigmas"],
                 "gap_mean": {p: float(v["source_frame"].mean()) for p, v in gaps.items()},
                 "gap_vs_tokens_rho": {p: rho(v["source_frame"], tokens) for p, v in gaps.items()},
                 "gap_maps_source_frame": {p: v["source_frame"].tolist() for p, v in gaps.items()},
                 "spearman": {}}
        for step, by in steps.items():
            for scale, imp in by.items():
                key = f"step_index{step}_scale{scale:g}"
                entry["spearman"][key] = {f"{p}/{v}": rho(imp, g) for p, gv in gaps.items() for v, g in gv.items()}
                for p, gv in gaps.items():
                    ranks.setdefault((key, p), []).append((imp, gv["source_frame"]))
        results.append(entry)
        print(f"#{entry['prompt_index']:3d}  rho(importance, gap on source frame), scale 0:")
        for step in sorted(steps):
            print(f"   step_index {step}: " + "  ".join(
                f"{p}={entry['spearman'][f'step_index{step}_scale0'][p + '/source_frame']:+.2f}"
                for p in ("1-4", FOCUS[step])))

        fig, axes = plt.subplots(2, 5, figsize=(26, 7.6))
        for k in range(4):
            axes[0, k].imshow(Image.open(case / "x0_step_gap" / f"x0_step{k + 1}.png"))
            axes[0, k].set_title(f"x0 at Step {k + 1} (sigma in={saved['sigmas'][k]:.4f})")
        axes[0, 4].axis("off")
        panels = [(f"gap Step {p} (source frame)", gaps[p]["source_frame"], "viridis") for p in ("1-4", "2-4", "3-4")]
        panels += [(f"importance step_index {s} (=Step {s + 1}), scale 0", steps[s][0.0], "magma") for s in sorted(steps)]
        background = np.asarray(Image.open(case / "x0_step_gap" / "x0_step4.png"))
        for ax, (title, value, cmap) in zip(axes[1], panels):
            ax.imshow(background, alpha=0.3, extent=(0, cols, rows, 0))
            im = ax.imshow(value.reshape(rows, cols), cmap=cmap, alpha=0.75, extent=(0, cols, rows, 0))
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.035)
        for ax in axes.ravel():
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"#{entry['prompt_index']} \"{entry['prompt']}\": x0 gap across AR steps vs patch importance")
        fig.tight_layout()
        fig.savefig(case / "x0_step_gap" / "x0_gap_vs_importance.png", dpi=100)
        plt.close(fig)

    def within_rank(pairs):
        r = lambda x: np.argsort(np.argsort(x))
        return rho(np.concatenate([r(i) for i, _ in pairs]), np.concatenate([r(g) for _, g in pairs]))
    pooled = {}
    for (key, p), pairs in ranks.items():
        pooled.setdefault(key, {})[p] = within_rank(pairs)
    print("pooled within-prompt rank rho (source frame):")
    for key, values in pooled.items():
        print(f"   {key}: " + "  ".join(f"{p}={v:+.2f}" for p, v in values.items()))
    (root / "x0_step_gap_summary.json").write_text(json.dumps(
        {"description": __doc__, "cases": results, "pooled_within_prompt_rank_spearman_source_frame": pooled},
        ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/patch_importance/ar_patch_importance_20261026"))
    parser.add_argument("--percentile", action="store_true",
                        help="Plot saved Step 1->4 gap and Step 2 importance as percentiles; no inference")
    args = parser.parse_args()
    cases = sorted(args.root.glob("prompt_*_seed_*"))
    if not cases:
        raise SystemExit("No importance cases found")
    if args.percentile:
        for case in cases:
            plot_percentiles(case)
        return
    missing = [c for c in cases if not (c / "x0_step_gap" / "x0_steps.pt").is_file()]
    if missing:
        if not torch.cuda.is_available():
            raise RuntimeError("Capturing x0 requires a GPU")
        capture(args.root, missing)
    analyze(args.root, cases)


if __name__ == "__main__":
    main()
