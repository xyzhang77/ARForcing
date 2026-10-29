#!/usr/bin/env python3
"""Summarize joint patch interventions using the next chunk as a fixed target.

python scripts/patch/summarize_patch_group_importance.py --input-dir outputs/.../groups \
    --output-dir outputs/.../group_report

Use --skip-plots for a JSON/HTML report without matplotlib, or --html-only to
refresh an existing report from summary.json without recomputing any figures.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np


LIMITATIONS = [
    "主目标固定为干预帧之后的下一 chunk，所有 group 使用同一目标区域。当前 chunk 的 common_target 仅为辅助指标，本报告不拿它替代 next_chunk。",
    "top 按同一 case 的单 patch 排名选出；这是候选组合探索，不是独立验证。排序与干预的 sigma scale 均以 metadata 为准。",
    "随机组与 top 组匹配实际面积，允许包含部分 top 成员，但排除完全相同的 top 集合。bottom 取单 patch 排名最低的 k 块，面积可能不同；表中单独标记，不把它自动当作等面积对照。",
    "随机对照没有匹配空间连通性和 mask 边界形状；即使 top 优于 random，也尚未分离位置内容与 mask 几何的作用。输出姿态或身份变化本身不能证明 source 携带语义 anchor。",
    "相同面积并不保证相同输入扰动能量，因此同时报告下一 chunk 的原始 MSE 与输出变化能量 / 输入扰动能量。后者也不是质量分数。",
    "full 是同一 source frame 全部粗 patch 同时降噪的参照干预，并非全部因果影响。group / full 响应比可以大于 1，不能当作因果贡献百分比、解释率或语义方差占比。",
    "joint effects 不可默认相加。要判断协同或抵消，需要在相同目标和噪声条件下比较带符号向量 ΔG 与各单 patch Δi 之和；仅比较能量或现有单 patch 排名不足以验证交互。",
    "语义分布与稀疏性不是互斥解释。16 个粗 patch 中，每块包含 91 或 104 个模型 tokens；单 patch 效果或这些候选 group 的结果不能排除更细 token、未测试组合或隐藏的联合协同。",
    "本轮使用 Self-Forcing checkpoint，测试局部实际降噪的传播机制；不是 MaskForcing 训练效果或生成质量改善验证。语义 anchor 还需要对象/部件对齐以及新样本验证。",
    "多个 case 共享 prompt、seed 或去噪轨迹；这里展示配对结果与描述性统计，不把所有 case 当作独立样本进行显著性检验。",
]

PRIMARY_FIELDS = (
    "next_chunk_mse", "next_chunk_energy", "next_chunk_normalized_energy",
    "next_chunk_delta_vs_full_relative_l2", "next_chunk_cosine_to_full",
    "next_chunk_energy_relative_to_full",
)
KINDS = {"full": "full", "identity": "empty", "top": "top", "bottom": "bottom",
         "random": "random", "complement": "complement"}


def finite(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def safe_ratio(numerator, denominator):
    numerator, denominator = finite(numerator), finite(denominator)
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def stats(values):
    values = [value for item in values if (value := finite(item)) is not None]
    if not values:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {"n": len(values), "min": min(values), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)), "max": max(values)}


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def number(value, digits=4):
    value = finite(value)
    return "—" if value is None else f"{value:.{digits}g}"


def case_label(metadata):
    return f"P{metadata.get('prompt_index', '?')} S{metadata.get('seed', '?')} T{metadata.get('step_index', '?')}"


def relative_url(path, output_dir):
    return quote(os.path.relpath(path, output_dir).replace(os.sep, "/"), safe="/")


def group_from_record(record):
    rollout = record.get("rollout") or {}
    group = {key: record.get(key) for key in (
        "name", "kind", "k", "source_ids", "num_patches", "latent_area", "num_tokens",
        "source_frame_area_fraction", "top_group_latent_area", "matches_top_area", "sigma_scale", "source_sigma",
    )}
    group["input_energy"] = finite(rollout.get("input_energy"))
    group.update({key: finite(rollout.get(key)) for key in PRIMARY_FIELDS})
    group["missing_primary_fields"] = [key for key in PRIMARY_FIELDS if key not in rollout]
    group["secondary_common_target_mse"] = finite(rollout.get("common_target_mse"))
    return group


def area_matches(group, top):
    if group.get("matches_top_area") is False:
        return False
    area, top_area = finite(group.get("latent_area")), finite(top.get("latent_area"))
    return area is not None and top_area is not None and area == top_area


def paired_comparisons(groups):
    comparisons = []
    for top in groups:
        if top.get("kind") != "top":
            continue
        k = top["k"]
        peers = [group for group in groups if group.get("k") == k]
        random_all = [group for group in peers if group.get("kind") == "random"]
        random = [group for group in random_all if area_matches(group, top)]
        bottom = next((group for group in peers if group.get("kind") == "bottom"), None)
        complement = next((group for group in peers if group.get("kind") == "complement"), None)
        raw_random = stats(group["next_chunk_mse"] for group in random)
        normalized_random = stats(group["next_chunk_normalized_energy"] for group in random)
        raw_top, normalized_top = top["next_chunk_mse"], top["next_chunk_normalized_energy"]
        comparison = {
            "k": k, "top": top, "bottom": bottom, "complement": complement,
            "n_random_total": len(random_all), "n_random_area_matched": len(random),
            "random_names": [group["name"] for group in random],
            "random_mse": raw_random, "random_normalized_energy": normalized_random,
            "top_over_random_median_mse": safe_ratio(raw_top, raw_random["median"]),
            "top_over_random_median_normalized_energy": safe_ratio(normalized_top, normalized_random["median"]),
            "top_exceeds_random_median_mse": (raw_top > raw_random["median"])
                if raw_top is not None and raw_random["median"] is not None else None,
            "top_exceeds_random_median_normalized_energy": (normalized_top > normalized_random["median"])
                if normalized_top is not None and normalized_random["median"] is not None else None,
            "bottom_area_matches_top": area_matches(bottom, top) if bottom is not None else None,
            "random_geometry": {field: stats(group[field] for group in random) for field in (
                "next_chunk_delta_vs_full_relative_l2", "next_chunk_cosine_to_full", "next_chunk_energy_relative_to_full")},
        }
        comparisons.append(comparison)
    return sorted(comparisons, key=lambda item: item["k"])


def load_cases(input_dir):
    cases, skipped = [], []
    for path in sorted(input_dir.rglob("group_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("metadata"), dict) or not isinstance(data.get("records"), list):
                raise ValueError("missing metadata or records")
            if not data["records"]:
                raise ValueError("no group records")
            names = [record["name"] for record in data["records"]]
            if len(set(names)) != len(names):
                raise ValueError("duplicate group names")
            groups = [group_from_record(record) for record in data["records"]]
            identity = data.get("identity") or {}
            identity_values = [finite(value) for value in identity.values()]
            metadata = data["metadata"]
            original = metadata.get("original_metadata") or {}
            same_hash = (metadata["baseline_sha256"] == original["baseline_sha256"]
                         if metadata.get("baseline_sha256") and original.get("baseline_sha256") else None)
            cases.append({"metrics_path": str(path), "metadata": metadata, "identity": identity,
                          "identity_max_abs": max((value for value in identity_values if value is not None), default=None),
                          "baseline_matches_original": same_hash, "groups": groups,
                          "comparisons": paired_comparisons(groups), "runner_limitations": data.get("limitations", [])})
        except (OSError, ValueError, TypeError, KeyError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    return cases, skipped


def aggregate_cases(cases):
    result = {"n_cases": len(cases), "n_groups": sum(len(case["groups"]) for case in cases),
              "n_prompts": len({case["metadata"].get("prompt_index") for case in cases}),
              "n_prompt_seed_pairs": len({(case["metadata"].get("prompt_index"), case["metadata"].get("seed")) for case in cases}),
              "identity_max_abs": max((case["identity_max_abs"] for case in cases if case["identity_max_abs"] is not None), default=None),
              "baseline_hash_comparisons": {str(value): sum(case["baseline_matches_original"] is value for case in cases)
                                            for value in (True, False, None)},
              "n_groups_missing_primary_fields": sum(bool(group["missing_primary_fields"]) for case in cases for group in case["groups"]),
              "by_k": []}
    ks = sorted({comparison["k"] for case in cases for comparison in case["comparisons"]})
    for k in ks:
        comparisons = [comparison for case in cases for comparison in case["comparisons"] if comparison["k"] == k]
        item = {"k": k, "n_cases": len(comparisons)}
        for metric in ("mse", "normalized_energy"):
            ratio_key = f"top_over_random_median_{metric}"
            win_key = f"top_exceeds_random_median_{metric}"
            valid = [comparison[win_key] for comparison in comparisons if comparison[win_key] is not None]
            item[ratio_key] = stats(comparison[ratio_key] for comparison in comparisons)
            item[f"top_exceeds_random_{metric}"] = {"wins": sum(valid), "n": len(valid)}
        item["n_bottom_area_matched"] = sum(comparison["bottom_area_matches_top"] is True for comparison in comparisons)
        item["n_bottom_present"] = sum(comparison["bottom"] is not None for comparison in comparisons)
        item["geometry"] = {}
        for kind in ("top", "bottom", "complement", "random"):
            geometry = {}
            for metric in ("next_chunk_mse", "next_chunk_normalized_energy", "next_chunk_delta_vs_full_relative_l2",
                           "next_chunk_cosine_to_full", "next_chunk_energy_relative_to_full"):
                if kind == "random":
                    if metric == "next_chunk_mse":
                        values = [comparison["random_mse"]["median"] for comparison in comparisons]
                    elif metric == "next_chunk_normalized_energy":
                        values = [comparison["random_normalized_energy"]["median"] for comparison in comparisons]
                    else:
                        values = [comparison["random_geometry"][metric]["median"] for comparison in comparisons]
                else:
                    values = [comparison[kind][metric] for comparison in comparisons if comparison[kind] is not None]
                geometry[metric] = stats(values)
            item["geometry"][kind] = geometry
        result["by_k"].append(item)
    return result


def concentration_summary(path):
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in data.get("aggregate_concentration", [])
            if row.get("stage") == "one_step" and row.get("metric") == "normalized_outside_energy"
            and row.get("step_index") == "all" and row.get("coverage_complete")]
    return {"path": str(path), "n_cases": data.get("n_cases"), "rows": rows,
            "token_grid": data.get("token_grid"), "uniform_reference": data.get("uniform_16_reference"),
            "notes": data.get("notes", [])}


def factual_findings(aggregate):
    findings = [f"读取 {aggregate['n_cases']} 个 case、{aggregate['n_groups']} 次 group 干预；"
                f"覆盖 {aggregate['n_prompts']} 个 prompt、{aggregate['n_prompt_seed_pairs']} 个 prompt/seed 配对。"]
    for item in aggregate["by_k"]:
        k = item["k"]
        raw = item["top_over_random_median_mse"]
        normalized = item["top_over_random_median_normalized_energy"]
        wins = item["top_exceeds_random_mse"]
        findings.append(f"top{k} 相对等面积 random 中位数：下一 chunk MSE 的配对比值中位数为 {number(raw['median'])}×"
                        f"（范围 {number(raw['min'])}–{number(raw['max'])}×），在 {wins['wins']}/{wins['n']} 个可比较 case 更高；"
                        f"输入能量归一化后的配对比值中位数为 {number(normalized['median'])}×。")
        top = item["geometry"]["top"]
        complement = item["geometry"]["complement"]
        findings.append(f"top{k} 与 full 的目标差分相对 L2 残差中位数为 "
                        f"{number(top['next_chunk_delta_vs_full_relative_l2']['median'])}，cosine 为 "
                        f"{number(top['next_chunk_cosine_to_full']['median'])}；"
                        f"complement(top{k}) / full 的响应能量比中位数为 "
                        f"{number(complement['next_chunk_energy_relative_to_full']['median'])}×。这些是响应几何关系，不是贡献比例。")
    return findings


def make_plots(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = summary["cases"]
    ks = [item["k"] for item in summary["aggregate"]["by_k"]]
    labels = [case_label(case["metadata"]) for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(cases) * 0.9), 5.2))
    for axis, metric, title in zip(axes, ("mse", "normalized_energy"),
                                   ("Raw next-chunk MSE", "Next-chunk energy / input energy")):
        has_positive = False
        for k in ks:
            values = []
            for case in cases:
                comparison = next((item for item in case["comparisons"] if item["k"] == k), None)
                value = comparison.get(f"top_over_random_median_{metric}") if comparison else None
                values.append(value if value is not None and value > 0 else np.nan)
            if np.isfinite(values).any():
                has_positive = True
            axis.plot(np.arange(len(cases)), values, marker="o", label=f"top{k} / matched random median", linewidth=1.3)
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        if has_positive:
            axis.set_yscale("log")
        axis.set_xticks(np.arange(len(cases)), labels, rotation=65, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Within-case response ratio (log scale)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Joint groups: same next-chunk target; ratio > 1 means greater sensitivity, not better quality")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_next_chunk.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"top": "#2864bc", "random": "#da8a28", "bottom": "#7d8085", "complement": "#29836f"}
    categories = [(k, kind) for k in ks for kind in ("top", "random", "bottom", "complement")]
    for axis, field, title in zip(axes, ("next_chunk_delta_vs_full_relative_l2", "next_chunk_cosine_to_full"),
                                  ("Relative L2 residual to full", "Cosine with full response")):
        for x, (k, kind) in enumerate(categories):
            values = []
            for case in cases:
                comparison = next((item for item in case["comparisons"] if item["k"] == k), None)
                if not comparison:
                    continue
                value = (comparison["random_geometry"][field]["median"] if kind == "random"
                         else comparison[kind][field] if comparison[kind] else None)
                if value is not None:
                    values.append(value)
            if values:
                axis.scatter(x + np.linspace(-0.13, 0.13, len(values)), values, s=24, color=colors[kind], alpha=0.7)
                axis.plot([x - 0.24, x + 0.24], [np.median(values)] * 2, color="black", linewidth=2.2)
        axis.set_xticks(range(len(categories)), [f"{kind}{k}" for k, kind in categories], rotation=40, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if field.endswith("relative_l2"):
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_ylabel("||delta_group - delta_full|| / ||delta_full||")
        else:
            axis.set_ylim(-1.05, 1.05)
            axis.axhline(1, color="black", linestyle="--", linewidth=0.8)
            axis.set_ylabel("Cosine(delta_group, delta_full)")
    fig.suptitle("Each point = one case; random = within-case median; black bar = across-case median\n"
                 "complement k = all source patches except top k; bottom may have a different area")
    fig.tight_layout()
    fig.savefig(output_dir / "full_response_geometry.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    return ["paired_next_chunk.png", "full_response_geometry.png"]


def table(headers, rows):
    rendered = ["<div class='scroll'><table><thead><tr>"]
    rendered.extend(f"<th>{html.escape(str(header))}</th>" for header in headers)
    rendered.append("</tr></thead><tbody>")
    for row in rows:
        rendered.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    rendered.append("</tbody></table></div>")
    return "".join(rendered)


def write_report(summary, output_dir):
    esc = html.escape
    aggregate = summary["aggregate"]
    body = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'><title>关键 patch 组合实验</title>",
            "<style>body{font:16px/1.65 system-ui,sans-serif;background:#f7f8fa;color:#20252b;max-width:1320px;margin:auto;padding:24px}",
            "section{background:white;border:1px solid #dfe3e7;border-radius:10px;margin:22px 0;padding:22px}h1,h2,h3{line-height:1.3}",
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}",
            "th{white-space:nowrap}.scroll{overflow:auto}img{max-width:100%;height:auto}.muted{color:#606b76}code{word-break:break-all}",
            "details{margin:14px 0}summary{cursor:pointer;font-weight:600}.finding{border-left:4px solid #2864bc;padding-left:15px}",
            "</style></head><body><h1>关键 patch 组合与分布式响应</h1>",
            "<p>在固定基线、后续噪声与全局 timestep 下，对多个 source patch 同时降低实际噪声。"
            "本报告用下一 chunk 的同一 latent 区域，比较 top、等面积 random、bottom、complement 和 full。</p>",
            "<p><a href='summary.json'>结构化分析</a></p><section><h2>本轮观测</h2>"]
    body.extend(f"<p class='finding'>{esc(finding)}</p>" for finding in summary["findings"])
    body.append("<p>较强的少数组合响应可以支持候选集合的重要性，但单独不足以证明必要性、充分性或语义 anchor。"
                "需要同时查看等面积对照、complement 和与 full 的方向/残差关系，并在新样本上验证。</p></section>")
    body.append("<section><h2>对照与测量口径</h2>")
    body.append(f"<p>所有 identity 检查的最大绝对误差：<strong>{number(aggregate['identity_max_abs'])}</strong>。"
                f"与原单 patch 扫描基线 hash 一致 / 不一致 / 未记录："
                f"{aggregate['baseline_hash_comparisons'].get('True', 0)} / {aggregate['baseline_hash_comparisons'].get('False', 0)} / "
                f"{aggregate['baseline_hash_comparisons'].get('None', 0)}。</p>")
    if aggregate["n_groups_missing_primary_fields"]:
        body.append(f"<p>{aggregate['n_groups_missing_primary_fields']} 组缺少部分 next_chunk 字段；对应统计标为缺失，不使用其他 target 代替。</p>")
    settings = sorted({(case["metadata"].get("ranking_source_scale"), case["metadata"].get("test_scale"))
                       for case in summary["cases"]}, key=str)
    body.append("<p>实际 scale 设置（排序 / group 干预）：" + "; ".join(f"{number(a)} / {number(b)}" for a, b in settings) + "。</p>")
    body.append("<p>令 ΔG=YG−Ybaseline，Δfull=Yfull−Ybaseline，全部仅取下一 chunk。"
                "相对残差为 ‖ΔG−Δfull‖ / ‖Δfull‖，越小表示更接近 full 的变化；cosine 表示方向一致性。"
                "能量比为 ‖ΔG‖² / ‖Δfull‖²，不具有可加贡献解释；它们都不能判断哪段视频质量更好。</p><ul>")
    body.extend(f"<li>{esc(note)}</li>" for note in LIMITATIONS)
    body.append("</ul></section>")
    for filename in summary.get("figures", []):
        body.append(f"<section><a href='{esc(filename, quote=True)}'><img src='{esc(filename, quote=True)}' alt='Group experiment comparison'></a></section>")
    body.append("<section><h2>逐 case 的固定目标比较</h2><p>random 列使用该 case 内所有面积匹配随机组的中位数；括号为最小–最大值。"
                "比值先在每个 case 内计算，再汇总，不用两个总体中位数相除。bottom 的面积差异见最后一列。</p>")
    rows = []
    for case in summary["cases"]:
        for comparison in case["comparisons"]:
            top, bottom, complement = (comparison[key] for key in ("top", "bottom", "complement"))
            random = comparison["random_mse"]
            rows.append([case_label(case["metadata"]), comparison["k"], ",".join(map(str, top["source_ids"])),
                         number(top["next_chunk_mse"]), f"{number(random['median'])} ({number(random['min'])}–{number(random['max'])})",
                         number(bottom["next_chunk_mse"]) if bottom else "—", number(complement["next_chunk_mse"]) if complement else "—",
                         number(comparison["top_over_random_median_mse"]) + "×",
                         number(comparison["top_over_random_median_normalized_energy"]) + "×",
                         f"{comparison['n_random_area_matched']}/{comparison['n_random_total']}",
                         "相同" if comparison["bottom_area_matches_top"] else "不同或未记录"])
    body.append(table(["case", "k", "top IDs", "top MSE", "random MSE", "bottom MSE", "complement MSE",
                       "top/random 原始", "top/random 归一化", "匹配 random 数", "bottom 面积"], rows))
    body.append("</section>")
    concentration = summary.get("concentration")
    if concentration:
        body.append("<section><h2>已有单 patch 分数的集中度</h2><p>以下是先前单 patch 一步预测的 outside_source 分数分布，"
                    "与本次 group 的下一 chunk 指标不是同一 target。top-k share 是单独测得分数之和的占比，不能解释为联合干预贡献或语义解释率。</p>")
        rows = []
        for item in sorted(concentration["rows"], key=lambda row: row["sigma_scale"]):
            rows.append([number(item["sigma_scale"]), item["n_cases"],
                         *(f"{100 * item[field]['median']:.1f}%" for field in ("top1_share", "top2_share", "top4_share")),
                         number(item["effective_n"]["median"]), number(item["gini"]["median"])])
        body.append(table(["sigma scale", "cases", "top1 分数占比", "top2 分数占比", "top4 分数占比", "effective_n", "Gini"], rows))
        body.append("<p>表中均为 case 中位数。16 块完全均匀的参考：top1=6.25%、top2=12.5%、top4=25%，effective_n=16，Gini=0。"
                    "每块含 91 或 104 tokens，无法据此排除块内少数关键 tokens。</p>")
        body.append(f"<a href='{esc(relative_url(Path(concentration['path']), output_dir), quote=True)}'>完整集中度统计</a></section>")
    body.append("<section><h2>完整 group 数值与可追溯结果</h2>")
    for case in summary["cases"]:
        metadata = case["metadata"]
        body.append(f"<details><summary>{esc(case_label(metadata))} — {esc(str(metadata.get('prompt', '')))}</summary>")
        body.append(f"<p><a href='{esc(relative_url(Path(case['metrics_path']), output_dir), quote=True)}'>原始 group_metrics.json</a>；"
                    f"identity max={number(case['identity_max_abs'])}；checkpoint=<code>{esc(str(metadata.get('checkpoint_path', '未记录')))}</code></p>")
        rows = []
        for group in case["groups"]:
            rows.append([group["name"], ",".join(map(str, group.get("source_ids") or [])), group.get("num_tokens"),
                         f"{100 * group['source_frame_area_fraction']:.2f}%" if group.get("source_frame_area_fraction") is not None else "—",
                         number(group["input_energy"]), number(group["next_chunk_mse"]), number(group["next_chunk_normalized_energy"]),
                         number(group["next_chunk_delta_vs_full_relative_l2"]), number(group["next_chunk_cosine_to_full"]),
                         number(group["next_chunk_energy_relative_to_full"]) + "×"])
        body.append(table(["group", "source IDs", "tokens", "source frame 面积", "输入扰动能量", "下一chunk MSE",
                           "归一化能量", "相对full L2残差", "与full cosine", "响应能量/full"], rows))
        body.append("</details>")
    body.append("</section>")
    if summary.get("skipped_files"):
        body.append("<section><h2>未纳入文件</h2><ul>")
        body.extend(f"<li>{esc(item['path'])}: {esc(item['reason'])}</li>" for item in summary["skipped_files"])
        body.append("</ul></section>")
    body.append("</body></html>")
    (output_dir / "report.html").write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concentration-json", type=Path)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if args.html_only:
        path = output_dir / "summary.json"
        if not path.is_file():
            parser.error(f"missing {path}")
        write_report(json.loads(path.read_text(encoding="utf-8")), output_dir)
        print(f"Rewrote HTML only: {output_dir / 'report.html'}", flush=True)
        return
    if not input_dir.is_dir():
        parser.error(f"not a directory: {input_dir}")
    cases, skipped = load_cases(input_dir)
    for item in skipped:
        print(f"Skipping {item['path']}: {item['reason']}", file=sys.stderr)
    if not cases:
        parser.error("no compatible group_metrics.json found")
    aggregate = aggregate_cases(cases)
    concentration_path = args.concentration_json or input_dir.parent / "concentration_analysis.json"
    summary = {"schema_version": 1, "input_dir": str(input_dir), "primary_target": "next_chunk",
               "aggregate": aggregate, "findings": factual_findings(aggregate), "cases": cases,
               "concentration": concentration_summary(concentration_path), "limitations": LIMITATIONS,
               "skipped_files": skipped, "figures": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        summary["figures"] = make_plots(summary, output_dir)
    summary = clean_json(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, output_dir)
    print(f"Summarized {len(cases)} group cases: {output_dir / 'report.html'}", flush=True)
    for finding in summary["findings"]:
        print(finding, flush=True)


if __name__ == "__main__":
    main()
