#!/usr/bin/env python3
"""Compare paired fine/coarse noise sweeps on a fixed, untouched target domain.

No torch import is needed. ``--html-only`` recomputes JSON/HTML without importing
matplotlib; existing figures in the output directory are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = ("normalized", "raw")
LIMITATIONS = [
    "主指标为 one-step 预测中除整个 source frame 外的所有帧：本实验固定为帧 0、2。粗细两轮均从 per_frame_mse 重算，区别于旧报告排除 source box 的 outside_source 指标；因此旧排名可能改变。",
    "归一化分数 = 固定目标输出变化的平方和 / 实际输入扰动的平方和；同时报告 raw MSE。输入能量受 patch 面积和内容影响，归一化并不等同于控制所有内容差异。",
    "分数份额及有效区域数只描述单独干预分数的集中程度，不是可加的因果贡献比例；未测试的联合干预可能非线性。",
    "每个细 patch 仍包含多个 transformer tokens。本实验不能排除更细的单 token anchor，也不能证明或否定语义机制。",
    "这是固定 checkpoint、prompt、seed、去噪阶段和扰动方向下的噪声敏感度；变化更大不代表信息更重要或生成质量更好。",
    "跨分辨率同时改变了干预区域大小与几何形状。比较集中程度可以定位热点，但不能把效应差异完全归因于某个 token。",
    "baseline.png 是 source latent frame 对应的解码视频帧（source_frame × 4）。叠加框是 latent 坐标到像素图的空间投影，不代表 VAE 精确感受野或物体边界。",
    "选中细 patch 的完整 rollout 是探索性可视化；随机候选未做面积匹配，不能将其直接作为匹配随机组对照。",
    "候选来自同一批探索性样本，尚未经过独立 seed/prompt 的选择与验证。checkpoint 校验使用路径、大小、mtime 指纹，不是 checkpoint 文件内容 SHA256。",
]


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def number(value, digits=4):
    value = finite(value)
    return "—" if value is None else f"{value:.{digits}g}"


def percent(value):
    return "—" if finite(value) is None else f"{100 * float(value):.1f}%"


def stats(values):
    values = [value for item in values if (value := finite(item)) is not None]
    return ({"n": 0, "min": None, "median": None, "max": None} if not values else
            {"n": len(values), "min": min(values), "median": float(np.median(values)), "max": max(values)})


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def relative_url(path, output_dir):
    return Path(os.path.relpath(path, output_dir)).as_posix()


def key(metadata):
    return tuple(metadata[field] for field in (
        "prompt_index", "seed", "source_chunk", "source_frame", "step_index"))


def case_name(metadata):
    return (f"prompt_{metadata['prompt_index']:06d}_seed_{metadata['seed']:06d}"
            f"_chunk_{metadata['source_chunk']}_step_{metadata['step_index']}")


def case_label(case):
    metadata = case["metadata"]
    return f"prompt {metadata['prompt_index']} · seed {metadata['seed']} · step {metadata['step_index']}"


def mean_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def spearman(first, second):
    first, second = mean_ranks(first), mean_ranks(second)
    first -= first.mean()
    second -= second.mean()
    return ratio(float(first @ second), float(np.linalg.norm(first) * np.linalg.norm(second)))


def concentration(values, areas=None):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("concentration requires finite, nonnegative scores")
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    areas = np.asarray(areas if areas is not None else np.ones(len(values)), dtype=float)
    if len(areas) != len(values) or not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("areas must contain one finite positive value per score")
    total = float(ordered.sum())
    result = {"n": len(values), "sum_scores": total, "median_score": float(np.median(values)),
              "max_score": float(ordered[0]), "max_over_median": ratio(ordered[0], np.median(values)),
              "effective_n": ratio(total * total, float(values @ values))}
    result["effective_n_over_n"] = (result["effective_n"] / len(values)
                                    if result["effective_n"] is not None else None)
    for label, count in [("top1", 1), ("top5pct", math.ceil(len(values) * .05)),
                         ("top10pct", math.ceil(len(values) * .10)),
                         ("top6_25pct", math.ceil(len(values) / 16)),
                         ("top25pct", math.ceil(len(values) / 4))]:
        result[label] = {"k": count, "fraction_of_regions": count / len(values),
                         "fraction_of_source_area": float(areas[order[:count]].sum() / areas.sum()),
                         "score_share": ratio(float(ordered[:count].sum()), total)}
    return result


def load_cases(directory):
    cases = {}
    for path in sorted(directory.glob("*/metrics.json")):
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported schema: {path}")
        identity = data.get("identity") or {}
        if not identity or any(finite(value) is None or abs(float(value)) > 1e-5
                               for value in identity.values()):
            raise ValueError(f"Missing or failed identity control: {path}")
        case_key = key(data["metadata"])
        if case_key in cases:
            raise ValueError(f"Duplicate case key: {path}")
        data["_path"] = path
        cases[case_key] = data
    return cases


def geometry(data):
    regions = data["regions"]
    height, width = max(r["y1"] for r in regions), max(r["x1"] for r in regions)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    if len({r["id"] for r in regions}) != len(regions):
        raise ValueError("duplicate region IDs")
    for region in regions:
        y0, y1, x0, x1 = [region[field] for field in ("y0", "y1", "x0", "x1")]
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("invalid region bounds")
        if any(value % 2 for value in (y0, y1, x0, x1)):
            raise ValueError("expected 2×2 latent-pixel transformer tokens")
        occupancy[y0:y1, x0:x1] += 1
    if not (occupancy == 1).all():
        raise ValueError("regions must cover each latent position exactly once")
    counts = Counter((r["y1"] - r["y0"]) * (r["x1"] - r["x0"]) // 4 for r in regions)
    return {"height": height, "width": width, "n_regions": len(regions),
            "token_shape": [2, 2], "tokens_per_region_counts": dict(sorted(counts.items())),
            "total_tokens": height * width // 4}


def verify_pair(fine, coarse):
    first, second = fine["metadata"], coarse["metadata"]
    fields = ("baseline_sha256", "initial_noise_sha256", "prompt", "weights", "num_output_frames",
              "global_t", "base_sigma", "schedule", "source_chunk", "source_frame", "step_index")
    for field in fields:
        if field not in first or first[field] != second.get(field):
            raise ValueError(f"Fine/coarse {field} mismatch: {fine['_path']}")
    config_hash = canonical_hash(first["config"])
    if config_hash != canonical_hash(second["config"]):
        raise ValueError(f"Fine/coarse config mismatch: {fine['_path']}")
    checkpoint_fields = ("checkpoint_path", "checkpoint_size", "checkpoint_mtime_ns")
    checkpoint = {field: first[field] for field in checkpoint_fields}
    if checkpoint != {field: second[field] for field in checkpoint_fields}:
        raise ValueError(f"Fine/coarse checkpoint metadata mismatch: {fine['_path']}")
    stable_code = ("pipeline/causal_inference.py", "utils/wan_wrapper.py", "wan/modules/causal_model.py")
    for path in stable_code:
        if path not in first.get("code_sha256", {}) or path not in second.get("code_sha256", {}) or first["code_sha256"][path] != second["code_sha256"][path]:
            raise ValueError(f"Fine/coarse model code mismatch for {path}")
    return {"baseline_and_noise_hashes_match": True, "config_sha256": config_hash,
            "checkpoint_metadata_sha256": canonical_hash(checkpoint), "checkpoint_metadata": checkpoint,
            "model_code_hashes_match": True,
            "identity_max_abs": max(abs(float(value)) for data in (fine, coarse)
                                    for value in data["identity"].values())}


def score_records(data, scale, geometry_data):
    metadata = data["metadata"]
    regions = {region["id"]: region for region in data["regions"]}
    shape = metadata["config"]["image_or_video_shape"]
    if list(shape[-2:]) != [geometry_data["height"], geometry_data["width"]]:
        raise ValueError("config latent shape differs from region coverage")
    elements_per_frame = shape[0] * shape[2] * shape[-2] * shape[-1]
    scores = []
    seen = set()
    for record in data["records"]:
        if float(record["sigma_scale"]) != scale:
            continue
        source = record["source_id"]
        if source in seen or source not in regions:
            raise ValueError("duplicate or unknown source record")
        seen.add(source)
        metrics = record["one_step"]
        frames = metrics["per_frame_mse"]
        selected_frames = [index for index in range(len(frames)) if index != metadata["source_frame"]]
        if not selected_frames:
            raise ValueError("no fixed untouched target frames")
        if any(finite(frames[index]) is None for index in selected_frames):
            raise ValueError("nonfinite target MSE")
        energy = sum(frames[index] for index in selected_frames) * elements_per_frame
        raw = sum(frames[index] for index in selected_frames) / len(selected_frames)
        input_energy = finite(metrics["input_energy"])
        if input_energy is None or input_energy <= 0:
            raise ValueError("nonpositive intervention energy")
        normalized = energy / input_energy
        for field, computed in (("other_frames_mse", raw), ("other_frames_energy", energy),
                                ("normalized_other_frames_energy", normalized)):
            if field in metrics and not np.isclose(float(metrics[field]), computed, rtol=2e-5, atol=1e-9):
                raise ValueError(f"stored {field} differs from fixed-domain reconstruction")
        region = regions[source]
        area = (region["y1"] - region["y0"]) * (region["x1"] - region["x0"])
        scores.append({**region, "num_tokens": area // 4, "latent_area": area,
                       "raw": raw, "normalized": normalized, "input_energy": input_energy,
                       "target_frames": selected_frames})
    if seen != set(regions):
        raise ValueError(f"Incomplete grid for sigma scale {scale}: {data['_path']}")
    return sorted(scores, key=lambda score: score["id"])


def hierarchy(fine_scores, coarse_scores):
    fine_to_coarse = {}
    rows = []
    ordered = sorted(coarse_scores, key=lambda item: item["normalized"], reverse=True)
    coarse_rank = {item["id"]: rank + 1 for rank, item in enumerate(ordered)}
    for coarse in coarse_scores:
        children = [item for item in fine_scores
                    if coarse["y0"] <= item["y0"] < item["y1"] <= coarse["y1"]
                    and coarse["x0"] <= item["x0"] < item["x1"] <= coarse["x1"]]
        if sum(item["latent_area"] for item in children) != coarse["latent_area"]:
            raise ValueError("fine regions are not nested within coarse grid")
        fine_to_coarse.update({item["id"]: coarse["id"] for item in children})
        row = {"coarse_id": coarse["id"], "coarse_normalized_rank": coarse_rank[coarse["id"]],
               "fine_ids": [item["id"] for item in children]}
        for metric in METRICS:
            scores = [item[metric] for item in children]
            best = max(children, key=lambda item: item[metric])
            row[metric] = {"coarse_score": coarse[metric], "fine_mean": float(np.mean(scores)),
                           "fine_max": best[metric], "max_over_mean": ratio(best[metric], np.mean(scores)),
                           "best_fine_id": best["id"]}
        rows.append(row)
    if len(fine_to_coarse) != len(fine_scores):
        raise ValueError("unassigned fine regions")
    top = sorted(fine_scores, key=lambda item: item["normalized"], reverse=True)[:math.ceil(len(fine_scores) * .05)]
    hidden = [{"fine_id": item["id"], "coarse_id": fine_to_coarse[item["id"]],
               "coarse_normalized_rank": coarse_rank[fine_to_coarse[item["id"]]]} for item in top
              if coarse_rank[fine_to_coarse[item["id"]]] > len(coarse_scores) / 2]
    correlations = {metric: {aggregation: spearman([row[metric]["coarse_score"] for row in rows],
                                                  [row[metric][aggregation] for row in rows])
                             for aggregation in ("fine_mean", "fine_max")} for metric in METRICS}
    return {"regions": rows, "top5pct_fine_in_bottom_half_coarse": hidden,
            "top5pct_fine_count": len(top), "coarse_vs_fine_rank_spearman": correlations}


def build_case(fine, coarse, output_dir):
    verification = verify_pair(fine, coarse)
    fine_geo, coarse_geo = geometry(fine), geometry(coarse)
    if fine_geo["height"] != coarse_geo["height"] or fine_geo["width"] != coarse_geo["width"]:
        raise ValueError("fine/coarse spatial domains differ")
    fine_scales = sorted({float(item["sigma_scale"]) for item in fine["records"]}, reverse=True)
    coarse_scales = sorted({float(item["sigma_scale"]) for item in coarse["records"]}, reverse=True)
    if fine_scales != coarse_scales:
        raise ValueError("fine/coarse sigma scales differ")
    result = {"name": case_name(fine["metadata"]), "metadata": fine["metadata"],
              "verification": verification, "fine_geometry": fine_geo, "coarse_geometry": coarse_geo,
              "fine_metrics_url": relative_url(fine["_path"], output_dir),
              "coarse_metrics_url": relative_url(coarse["_path"], output_dir), "by_scale": {}}
    baseline = coarse["_path"].parent / "baseline.png"
    result["baseline_image_url"] = relative_url(baseline, output_dir) if baseline.exists() else None
    fine_baseline = fine["_path"].parent / "baseline.mp4"
    result["baseline_video_url"] = relative_url(fine_baseline, output_dir) if fine_baseline.exists() else None
    result["rollout_previews"] = []
    regions_by_id = {region["id"]: region for region in fine["regions"]}
    for record in fine["records"]:
        if not record.get("rollout"):
            continue
        source = record["source_id"]
        region = regions_by_id[source]
        path = fine["_path"].parent / f"source_{source:03d}.mp4"
        result["rollout_previews"].append({
            "source_id": source, "selection": record.get("selection"), "sigma_scale": record["sigma_scale"],
            "num_tokens": (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4,
            "bounds": {field: region[field] for field in ("y0", "y1", "x0", "x1")},
            "next_chunk_mse": record["rollout"].get("next_chunk_mse"),
            "video_url": relative_url(path, output_dir) if path.exists() else None,
        })
    for scale in fine_scales:
        fine_scores, coarse_scores = score_records(fine, scale, fine_geo), score_records(coarse, scale, coarse_geo)
        if fine_scores[0]["target_frames"] != coarse_scores[0]["target_frames"]:
            raise ValueError("fine/coarse target frames differ")
        result["by_scale"][str(scale)] = {
            "fine": {"scores": fine_scores, **{metric: concentration([item[metric] for item in fine_scores], [item["latent_area"] for item in fine_scores]) for metric in METRICS}},
            "coarse": {"scores": coarse_scores, **{metric: concentration([item[metric] for item in coarse_scores], [item["latent_area"] for item in coarse_scores]) for metric in METRICS}},
            "hierarchy": hierarchy(fine_scores, coarse_scores),
        }
    result["across_strength_rank_spearman"] = {}
    if len(fine_scales) == 2:
        first, second = [result["by_scale"][str(scale)] for scale in fine_scales]
        result["across_strength_rank_spearman"] = {
            grid: {metric: spearman([item[metric] for item in first[grid]["scores"]],
                                   [item[metric] for item in second[grid]["scores"]]) for metric in METRICS}
            for grid in ("fine", "coarse")}
    return result


def aggregate(cases):
    fields = ("max_over_median", "effective_n_over_n", "median_score")
    shares = ("top1", "top5pct", "top10pct", "top6_25pct", "top25pct")
    by_scale = {}
    for scale in cases[0]["by_scale"]:
        grids = {}
        for grid in ("fine", "coarse"):
            grids[grid] = {}
            for metric in METRICS:
                values = [case["by_scale"][scale][grid][metric] for case in cases]
                grids[grid][metric] = {field: stats(value[field] for value in values) for field in fields}
                grids[grid][metric].update({field: stats(value[field]["score_share"] for value in values) for field in shares})
                grids[grid][metric]["source_area_fractions"] = {field: stats(value[field]["fraction_of_source_area"] for value in values) for field in shares}
        grids["within_coarse_max_over_mean"] = {
            metric: stats(row[metric]["max_over_mean"] for case in cases
                          for row in case["by_scale"][scale]["hierarchy"]["regions"]) for metric in METRICS}
        grids["top5pct_fine_in_bottom_half_coarse_count"] = stats(
            len(case["by_scale"][scale]["hierarchy"]["top5pct_fine_in_bottom_half_coarse"]) for case in cases)
        by_scale[scale] = grids
    strength = {grid: {metric: stats(case["across_strength_rank_spearman"].get(grid, {}).get(metric)
                                    for case in cases) for metric in METRICS} for grid in ("fine", "coarse")}
    by_pair = {}
    for case in cases:
        metadata = case["metadata"]
        pair = tuple(metadata[field] for field in ("prompt_index", "seed", "source_chunk", "source_frame"))
        by_pair.setdefault(pair, []).append(case)
    stages = []
    for pair, selected in by_pair.items():
        selected.sort(key=lambda case: case["metadata"]["step_index"])
        for first, second in zip(selected, selected[1:]):
            for scale in first["by_scale"]:
                row = {"prompt_index": pair[0], "seed": pair[1], "steps": [first["metadata"]["step_index"], second["metadata"]["step_index"]], "scale": scale}
                for grid in ("fine", "coarse"):
                    a, b = first["by_scale"][scale][grid]["scores"], second["by_scale"][scale][grid]["scores"]
                    row[grid] = {}
                    for metric in METRICS:
                        top_a, top_b = [sorted(scores, key=lambda item: item[metric], reverse=True) for scores in (a, b)]
                        count = math.ceil(len(a) * .05)
                        row[grid][metric] = {"rho": spearman([v[metric] for v in a], [v[metric] for v in b]),
                                             "top1_same": top_a[0]["id"] == top_b[0]["id"], "top5pct_k": count,
                                             "top5pct_overlap_count": len({v["id"] for v in top_a[:count]} & {v["id"] for v in top_b[:count]})}
                stages.append(row)
    return {"by_scale": by_scale, "across_strength_rank_spearman": strength, "across_step_pairs": stages,
            "across_step_rank_spearman": {scale: {grid: {metric: stats(row[grid][metric]["rho"] for row in stages if row["scale"] == scale)
                                                         for metric in METRICS} for grid in ("fine", "coarse")} for scale in by_scale},
            "identity_max_abs": max(case["verification"]["identity_max_abs"] for case in cases),
            "all_fine_coarse_pairs_verified": True}


def plot_scores(axis, scores, metric, *, baseline=None, labels=3, title=None):
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    xedges = sorted({item[key] for item in scores for key in ("x0", "x1")})
    yedges = sorted({item[key] for item in scores for key in ("y0", "y1")})
    grid = np.full((len(yedges) - 1, len(xedges) - 1), np.nan)
    average = np.mean([item[metric] for item in scores])
    for item in scores:
        grid[yedges.index(item["y0"]), xedges.index(item["x0"])] = item[metric] / average if average else 0
    positive = grid[grid > 0]
    lower = max(float(positive.min()) if positive.size else 1e-3, 1e-4)
    upper = max(float(grid.max()), lower * 1.01)
    if baseline is not None:
        axis.imshow(baseline, extent=[0, xedges[-1], yedges[-1], 0], aspect="equal")
    mesh = axis.pcolormesh(xedges, yedges, grid, cmap="magma", norm=LogNorm(vmin=lower, vmax=upper),
                           shading="flat", alpha=.52 if baseline is not None else 1)
    for item in sorted(scores, key=lambda score: score[metric], reverse=True)[:labels]:
        axis.add_patch(Rectangle((item["x0"], item["y0"]), item["x1"] - item["x0"], item["y1"] - item["y0"],
                                 fill=False, edgecolor="cyan", linewidth=1.4))
        axis.text((item["x0"] + item["x1"]) / 2, (item["y0"] + item["y1"]) / 2, str(item["id"]),
                  color="white", ha="center", va="center", fontsize=7,
                  bbox={"facecolor": "black", "alpha": .6, "edgecolor": "none", "pad": 1})
    axis.set(xlim=(0, xedges[-1]), ylim=(yedges[-1], 0), title=title, aspect="equal")
    axis.set_xlabel("latent x")
    axis.set_ylabel("latent y")
    return mesh


def plot_report(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    images = []
    for case in summary["cases"]:
        scales = list(case["by_scale"])
        figure, axes = plt.subplots(len(scales), 3, figsize=(16, 4 * len(scales)), squeeze=False, constrained_layout=True)
        for row, scale in enumerate(scales):
            for col, (grid, metric) in enumerate((("coarse", "normalized"), ("fine", "normalized"), ("fine", "raw"))):
                scores = case["by_scale"][scale][grid]["scores"]
                mesh = plot_scores(axes[row, col], scores, metric, title=f"{grid}: {metric}, sigma x {scale}")
                figure.colorbar(mesh, ax=axes[row, col], label="score / within-grid mean (log)", shrink=.8)
        figure.suptitle(case_label(case) + " | untouched target frames, one-step response")
        destination = output_dir / f"sensitivity_{case['name']}.png"
        figure.savefig(destination, dpi=150)
        plt.close(figure)
        case["sensitivity_image"] = destination.name
        if case["metadata"]["seed"] == 0 and case["metadata"]["step_index"] == 1 and case["baseline_image_url"]:
            baseline = plt.imread(output_dir / case["baseline_image_url"])
            figure, axes = plt.subplots(1, 1 + len(scales), figsize=(16, 4), constrained_layout=True)
            axes[0].imshow(baseline)
            axes[0].set_title(f"Baseline preview: prompt {case['metadata']['prompt_index']}")
            axes[0].axis("off")
            for axis, scale in zip(axes[1:], scales):
                mesh = plot_scores(axis, case["by_scale"][scale]["fine"]["scores"], "normalized", baseline=baseline,
                                   labels=5, title=f"Fine sensitivity, sigma x {scale}")
                figure.colorbar(mesh, ax=axis, label="score / mean (log)", shrink=.8)
            destination = output_dir / f"overlay_{case['name']}.png"
            figure.savefig(destination, dpi=170)
            plt.close(figure)
            case["overlay_image"] = destination.name
            images.append(destination.name)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    definitions = [("top6_25pct", "Top 6.25% of regions: score share"), ("top25pct", "Top 25% of regions: score share"),
                   ("effective_n_over_n", "Effective regions / N"), ("max_over_median", "Maximum / median")]
    colors = ("#2265ad", "#d57625", "#2265ad", "#d57625")
    entries = [(grid, scale) for grid in ("coarse", "fine") for scale in summary["cases"][0]["by_scale"]]
    for axis, (field, title) in zip(axes.flat, definitions):
        for xpos, ((grid, scale), color) in enumerate(zip(entries, colors)):
            values = [case["by_scale"][scale][grid]["normalized"][field] for case in summary["cases"]]
            if field.startswith("top"):
                values = [value["score_share"] for value in values]
            values = [value for value in values if value is not None]
            axis.scatter(np.full(len(values), xpos) + np.linspace(-.12, .12, len(values)), values, color=color, s=25, alpha=.65)
            if values:
                axis.plot([xpos - .25, xpos + .25], [np.median(values)] * 2, color="black", linewidth=2)
        axis.set_xticks(range(len(entries)), [f"{grid}\nx {scale}" for grid, scale in entries])
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Fixed-target normalized sensitivity | each dot is one case; line = median")
    destination = output_dir / "concentration.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    summary["overview_images"] = [destination.name] + images


def esc(value):
    return html.escape(str(value), quote=True)


def image_html(filename, caption):
    return f"<figure><a href='{esc(filename)}'><img src='{esc(filename)}' loading='lazy' alt='{esc(caption)}'></a><figcaption>{esc(caption)}</figcaption></figure>"


def concentration_row(label, metric):
    columns = [esc(label), str(metric["n"])]
    for field in ("top1", "top5pct", "top10pct", "top6_25pct", "top25pct"):
        item = metric[field]
        columns.append(f"{percent(item['score_share'])} <small>({item['k']}/{metric['n']}; 面积 {percent(item['fraction_of_source_area'])})</small>")
    columns.extend([number(metric["max_over_median"]), percent(metric["effective_n_over_n"])])
    return "<tr>" + "".join(f"<td>{column}</td>" for column in columns) + "</tr>"


def write_html(summary, output_dir):
    pieces = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
              "<title>细 patch 噪声敏感度实验</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:auto;padding:28px;color:#203042;background:#f7f9fc}section{background:white;border:1px solid #dde3ed;border-radius:12px;padding:22px;margin:22px 0}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}td,th{padding:8px;border-bottom:1px solid #dde3ed;text-align:left}th{background:#eef3f9}img{max-width:100%;height:auto}video{width:340px;max-width:100%}.videos{display:flex;flex-wrap:wrap;gap:14px}.videos figure{max-width:340px}figure{margin:18px 0}small,figcaption{color:#607084}summary{cursor:pointer;font-weight:600}a{color:#165fba}.scroll{overflow:auto}.note{border-left:4px solid #d29335;padding-left:14px}code{word-break:break-all}</style>",
              "<h1>细 patch 噪声敏感度：与粗网格的配对比较</h1>",
              f"<p>{len(summary['cases'])} 个条件，{summary['n_fine_interventions']} 次细 patch 单独干预。<a href='summary.json'>完整统计 JSON</a></p>",
              "<p class='note'>主指标测量固定未干预帧的 one-step 预测变化。所有“份额”都是单独干预分数之和的占比，不能理解为因果贡献率或质量增益。</p>",
              "<section><h2>设置与校验</h2>"]
    first = summary["cases"][0]
    if summary.get("coarse_report_url"):
        pieces.append(f"<p><a href='{esc(summary['coarse_report_url'])}'>原粗 patch 实验报告</a>（使用旧 outside_source 主指标）</p>")
    tokens = "、".join(f"{count} tokens × {number_of_regions} 块" for count, number_of_regions in first["fine_geometry"]["tokens_per_region_counts"].items())
    pieces.append(f"<p>细网格 {first['metadata']['grid_rows']} × {first['metadata']['grid_cols']}，共 {first['fine_geometry']['n_regions']} 块；{tokens}。粗网格共 {first['coarse_geometry']['n_regions']} 块。每个模型 token 对应 2 × 2 latent 像素。</p>")
    pieces.append(f"<p>checkpoint step600，source chunk 0 / frame 1；step 为零起始索引。局部 sigma × 0.9 / 0.75，全局 timestep 保持不变。所有粗细配对 baseline、初始 noise、config 和 checkpoint 元信息均一致；identity 最大误差 {number(summary['aggregate']['identity_max_abs'])}。</p>")
    pieces.append("<p>固定目标为同一 chunk 的帧 0、2；既包含 source frame 前，也包含之后的帧，不等同于下一 chunk 的远期影响。原粗报告包含 source frame 内其余像素，本报告重新计算粗网格分数，以保证目标域一致。</p></section>")
    pieces.append("<section><h2>集中程度</h2><p>下表为各条件统计量的中位数。N<sub>eff</sub> = (Σ score)² / Σ score²；N<sub>eff</sub>/N 越接近 1，分数越均匀。Top 5% / 10% 按区域数量向上取整；精确相同区域数量比例的跨网格比较见 Top 6.25% / 25%。细块面积不等，数量比例不等于面积比例；每个 case 的表格同时列出所选集合的实际面积占比。</p>")
    pieces.append("<div class='scroll'><table><tr><th>sigma scale</th><th>网格 / 分数</th><th>Top 1</th><th>Top 5%</th><th>Top 10%</th><th>Top 6.25%</th><th>Top 25%</th><th>最大/中位</th><th>N_eff/N</th></tr>")
    for scale, grids in summary["aggregate"]["by_scale"].items():
        for grid in ("coarse", "fine"):
            for metric in METRICS:
                values = grids[grid][metric]
                cells = [esc(scale), f"{grid} / {metric}"]
                cells += [percent(values[field]["median"]) for field in ("top1", "top5pct", "top10pct", "top6_25pct", "top25pct")]
                cells += [number(values["max_over_median"]["median"]), percent(values["effective_n_over_n"]["median"])]
                pieces.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    pieces.append("</table></div>")
    if (output_dir / "concentration.png").exists():
        pieces.append(image_html("concentration.png", "每个点对应一个条件，黑线为中位数；固定目标域、归一化分数。"))
    pieces.append("</section><section><h2>排名稳定性与粗块内部差异</h2><table><tr><th>比较</th><th>coarse normalized</th><th>fine normalized</th><th>fine raw</th></tr>")
    strength = summary["aggregate"]["across_strength_rank_spearman"]
    pieces.append(f"<tr><td>跨 sigma strength Spearman 中位数</td><td>{number(strength['coarse']['normalized']['median'])}</td><td>{number(strength['fine']['normalized']['median'])}</td><td>{number(strength['fine']['raw']['median'])}</td></tr>")
    for scale, grids in summary["aggregate"]["across_step_rank_spearman"].items():
        pieces.append(f"<tr><td>跨去噪 step，sigma × {esc(scale)}</td><td>{number(grids['coarse']['normalized']['median'])}</td><td>{number(grids['fine']['normalized']['median'])}</td><td>{number(grids['fine']['raw']['median'])}</td></tr>")
    pieces.append("</table><table><tr><th>sigma scale</th><th>粗块内细分数 max/mean 中位数（normalized）</th><th>Top 5% 细块落在粗排名后半区：每 case 中位数</th></tr>")
    for scale, grids in summary["aggregate"]["by_scale"].items():
        pieces.append(f"<tr><td>{esc(scale)}</td><td>{number(grids['within_coarse_max_over_mean']['normalized']['median'])}</td><td>{number(grids['top5pct_fine_in_bottom_half_coarse_count']['median'])}</td></tr>")
    pieces.append("</table><p>粗块内 max/mean 比较 12 个子块单独干预的分数，不是子块联合干预与父块的分解。“隐藏热点”仅按本批数据的相对排名定义。</p></section>")
    pieces.append("<section><h2>空间定位</h2><p>颜色显示本图分数 / 均值（对数色标）；各图色标独立，颜色不可直接用于比较绝对强度。青框与数字为排名最高的细 patch ID。下方每个条件也包含 raw MSE 热图。</p>")
    for case in summary["cases"]:
        filename = f"overlay_{case['name']}.png"
        if (output_dir / filename).exists():
            pieces.append(image_html(filename, case_label(case)))
    pieces.append("</section>")
    for case in summary["cases"]:
        pieces.append(f"<section><h2>{esc(case_label(case))}</h2><p>{esc(case['metadata']['prompt'])}</p><p><a href='{esc(case['fine_metrics_url'])}'>细网格原始 JSON</a> · <a href='{esc(case['coarse_metrics_url'])}'>粗网格原始 JSON</a></p>")
        filename = f"sensitivity_{case['name']}.png"
        if (output_dir / filename).exists():
            pieces.append(image_html(filename, "粗 normalized / 细 normalized / 细 raw；框出每图 Top 3。"))
        if case["baseline_video_url"] or case["rollout_previews"]:
            pieces.append("<details><summary>查看 baseline 与选中细 patch 的完整 rollout 视频</summary><p>候选按 one-step 主指标选择；下方 next-chunk MSE 测量后续 chunk。随机候选没有面积匹配，不作配对优劣结论。</p><div class='videos'>")
            if case["baseline_video_url"]:
                url = esc(case["baseline_video_url"])
                pieces.append(f"<figure><video controls preload='none' src='{url}'></video><figcaption><a href='{url}'>Baseline</a></figcaption></figure>")
            for preview in case["rollout_previews"]:
                url = esc(preview["video_url"]) if preview["video_url"] else None
                label = (f"ID {preview['source_id']} · {preview['selection']} · {preview['num_tokens']} tokens · "
                         f"sigma × {preview['sigma_scale']} · next-chunk MSE {number(preview['next_chunk_mse'])}")
                if url:
                    pieces.append(f"<figure><video controls preload='none' src='{url}'></video><figcaption><a href='{url}'>{esc(label)}</a></figcaption></figure>")
                else:
                    pieces.append(f"<p>{esc(label)}（无视频文件）</p>")
            pieces.append("</div></details>")
        pieces.append("<details><summary>查看每个强度的集中程度与 Top 10 细 patch</summary>")
        for scale, result in case["by_scale"].items():
            pieces.append(f"<h3>sigma × {esc(scale)}</h3><div class='scroll'><table><tr><th>网格 / 分数</th><th>N</th><th>Top 1</th><th>Top 5%</th><th>Top 10%</th><th>Top 6.25%</th><th>Top 25%</th><th>最大/中位</th><th>N_eff/N</th></tr>")
            for grid in ("coarse", "fine"):
                for metric in METRICS:
                    pieces.append(concentration_row(f"{grid} / {metric}", result[grid][metric]))
            pieces.append("</table></div><table><tr><th>排名</th><th>ID</th><th>latent bounds [y0:y1, x0:x1)</th><th>tokens / source面积</th><th>normalized</th><th>raw MSE</th><th>input energy</th></tr>")
            for rank, item in enumerate(sorted(result["fine"]["scores"], key=lambda score: score["normalized"], reverse=True)[:10], 1):
                pieces.append(f"<tr><td>{rank}</td><td>{item['id']}</td><td>[{item['y0']}:{item['y1']}, {item['x0']}:{item['x1']})</td><td>{item['num_tokens']} / {percent(item['num_tokens'] / case['fine_geometry']['total_tokens'])}</td><td>{number(item['normalized'])}</td><td>{number(item['raw'])}</td><td>{number(item['input_energy'])}</td></tr>")
            hidden = result["hierarchy"]["top5pct_fine_in_bottom_half_coarse"]
            pieces.append("</table><p>Top 5% 细块落在粗排名后半区：" + ("；".join(f"fine {item['fine_id']} → coarse {item['coarse_id']}（rank {item['coarse_normalized_rank']}）" for item in hidden) or "无") + "。</p>")
            pieces.append("<details><summary>粗块内部细分结果</summary><table><tr><th>coarse ID / rank</th><th>fine IDs</th><th>normalized max/mean</th><th>raw max/mean</th><th>最佳 fine ID（normalized）</th></tr>")
            for row in result["hierarchy"]["regions"]:
                pieces.append(f"<tr><td>{row['coarse_id']} / {row['coarse_normalized_rank']}</td><td>{esc(', '.join(map(str, row['fine_ids'])))}</td><td>{number(row['normalized']['max_over_mean'])}</td><td>{number(row['raw']['max_over_mean'])}</td><td>{row['normalized']['best_fine_id']}</td></tr>")
            pieces.append("</table></details>")
        pieces.append("</details></section>")
    pieces.append("<section><h2>解释边界</h2><ul>" + "".join(f"<li>{esc(item)}</li>" for item in LIMITATIONS) + "</ul></section></html>")
    (output_dir / "report.html").write_text("\n".join(pieces))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--coarse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    fine, coarse = load_cases(args.input_dir.resolve()), load_cases(args.coarse_dir.resolve())
    if not fine:
        parser.error("no completed fine metrics.json cases found")
    missing = set(fine) - set(coarse)
    if missing:
        parser.error(f"missing coarse matches: {sorted(missing)}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [build_case(data, coarse[case_key], args.output_dir) for case_key, data in sorted(fine.items())]
    summary = {"schema_version": 1, "input_dir": str(args.input_dir.resolve()), "coarse_dir": str(args.coarse_dir.resolve()),
               "n_cases": len(cases), "n_fine_interventions": sum(len(data["records"]) for data in fine.values()),
               "target_definition": "one-step prediction; all frames except entire source_frame; reconstructed from per_frame_mse",
               "primary_metric": "normalized_other_frames_energy", "secondary_metric": "other_frames_mse",
               "limitations": LIMITATIONS, "cases": cases, "aggregate": aggregate(cases)}
    coarse_report = args.coarse_dir.resolve().parent / "report" / "report.html"
    summary["coarse_report_url"] = relative_url(coarse_report, args.output_dir) if coarse_report.exists() else None
    if not args.html_only:
        plot_report(summary, args.output_dir)
    else:
        summary["overview_images"] = []
        if (args.output_dir / "concentration.png").exists():
            summary["overview_images"].append("concentration.png")
        for case in cases:
            for field, prefix in (("sensitivity_image", "sensitivity"), ("overlay_image", "overlay")):
                filename = f"{prefix}_{case['name']}.png"
                if (args.output_dir / filename).exists():
                    case[field] = filename
                    if field == "overlay_image":
                        summary["overview_images"].append(filename)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    write_html(summary, args.output_dir)
    print(json.dumps({"report": str(args.output_dir / "report.html"), "n_cases": len(cases),
                      "n_fine_interventions": summary["n_fine_interventions"],
                      "identity_max_abs": summary["aggregate"]["identity_max_abs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
