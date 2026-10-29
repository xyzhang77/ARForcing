#!/usr/bin/env python3
"""CPU-only analysis of paired initial-noise experiments (NumPy and Pillow)."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ALIGNED_SIGMA = 0.9375
FALLBACK_STAGES = {"ar": "second", "wan": "quarter"}
STAGE_ORDER = ["aligned_x0", "first", "aligned_cfg1", "second", "quarter", "half", "threequarter", "final",
               "first_cfg1", "first_horizon3", "first_horizon3_cfg1",
               "final_alt_transition", "aligned_state", "next_state"]
WARNINGS = [
    "主比较按实际输入状态的 noise level 对齐：AR 首次更新/重加噪后 σ=0.9375（second，第2次 x0 probe）；Wan 完成12次 Euler 更新后 σ=0.9375（quarter，第13次 x0 probe）。probe 只读当前状态，不代表又完成一次更新。",
    "本实验比较 AR 蒸馏四步与原生 Wan 双向 Euler 48 步、CFG=6 的完整系统；权重、训练/蒸馏、调度器、CFG、上下文与时域长度均可能影响结果，不能归因为 attention 单变量。实际参数请查每个 record.json。",
    "aligned_x0 是 σ=0.9375 状态的 clean x0 预测经 VAE 解码。aligned_state 和 next_state 是带噪状态，仅用于诊断，不能作为 x0 或成像质量的证据。同 σ 不要求两模型 state 张量相同。",
    "first@σ=1 仅作首步预测锚点；Wan 字面 first 与 second 只作补充，不能据此得出同 noise level 下 Wan 场景更模糊的结论。AR second 与 Wan second 的 σ 不同。",
    "结构代理为解码图像降采样到 15×26 的 RGB；它混合布局、颜色、亮度和 VAE 解码偏差。无法证明语义场景结构或真实分布相同。三个观察帧为 [0,4,8]，只覆盖第一 latent chunk。",
    "置信区间只重采样当前 prompt 内的 seeds；少量 prompt/seed 只能支持探索性结论。nearest-final 检索是同 prompt 内 N 选 1，平分最小距离的 ties 按 1/并列数计分，机会水平为 1/N。",
    "同一初噪跨模型配对仅在 first_chunk_noise_sha256 一致时成立。AR 与 Wan first 的原生时间上下文不同，horizon3 控制只匹配第一步的时域长度；CFG=1 控制也仅作用于第一步 x0，并非一条 CFG=1 的完整轨迹。",
    "AR 第一 chunk 内部三个 latent 帧互相可见：在这个观察窗口内它并不是逐帧严格 causal。Wan horizon3 即使匹配输入时长，差异也不能单独归因于 causal / bidirectional attention。",
    "AR final_alt_transition 若存在，固定初噪并替换后续 transition noise；其与 final 的差异相对于跨初噪 final 差异用于衡量后续噪声的作用。此干预本身不能验证第一步 x0 的语义正确性。",
    "Wan aligned_cfg1 仅在 CFG=6 已走过的轨迹上，把 σ=0.9375 当前 probe 的 CFG 改为1；它不是 CFG=1 采样轨迹，不能用于宣称轨迹/CFG 的单变量因果效果。",
]


def lowpass(views):
    """Area-average RGB features, one vector spanning all three observed frames."""
    return np.stack([np.asarray(Image.fromarray(frame).resize((26, 15), Image.Resampling.BOX),
                               dtype=np.float64) / 255.0 for frame in views])


def centered_cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale <= 1e-20:
        # Both have no centered signal: identical structure by this proxy.
        return 1.0 if np.linalg.norm(a) <= 1e-10 and np.linalg.norm(b) <= 1e-10 else 0.0
    return float(np.clip(np.dot(a, b) / scale, -1, 1))


def edges(a):
    gray = a @ np.array([0.299, 0.587, 0.114])
    return np.concatenate([np.diff(gray, axis=1).ravel(), np.diff(gray, axis=2).ravel()])


def distance_matrices(left, right=None):
    right = left if right is None else right
    a, b = np.stack(left), np.stack(right)
    af, bf = a.reshape(len(a), -1), b.reshape(len(b), -1)
    mse = np.maximum(((af * af).mean(1)[:, None] + (bf * bf).mean(1)[None, :]
                      - 2 * (af @ bf.T) / af.shape[1]), 0)
    # Direct diagonal differences avoid roundoff driving self-distance above zero.
    if right is left:
        np.fill_diagonal(mse, 0)
    cosine = np.array([[centered_cosine(x, y) for y in b] for x in a])
    edge = np.array([[centered_cosine(edges(x), edges(y)) for y in b] for x in a])
    return {"mse": mse, "rmse": np.sqrt(mse), "centered_cosine": cosine,
            "edge_correlation": edge}


def retrieval_scores(matrix):
    minimum = matrix.min(axis=1, keepdims=True)
    ties = np.isclose(matrix, minimum, rtol=1e-8, atol=1e-12)
    return np.diag(ties).astype(float) / ties.sum(axis=1)


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(values, [2.5, 97.5])] if len(values) else None


def safe_ratio(a, b):
    return float(a / b) if b > 1e-15 else None


def seed_weights(n, samples=2000):
    rng = np.random.default_rng(20261023)
    draws = rng.integers(0, n, (samples, n))
    return (draws[:, :, None] == np.arange(n)[None, None, :]).sum(axis=1).astype(float)


def weighted_offdiag(matrix, weights):
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    numerator = np.einsum("bi,ij,bj->b", weights, matrix, weights)
    denominator = weights.sum(axis=1) ** 2 - (weights ** 2).sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def pairwise_summary(features):
    n = len(features)
    if n < 2:
        return {"n_seeds": n, "n_pairs": 0, "metrics": {}}
    matrices, mask, weights = distance_matrices(features), ~np.eye(n, dtype=bool), seed_weights(n)
    stats = {}
    for name, matrix in matrices.items():
        values = matrix[mask]
        stats[name] = {"mean": float(values.mean()), "std": float(values.std()),
                       "ci95_seed_bootstrap": interval(weighted_offdiag(matrix, weights))}
    return {"n_seeds": n, "n_pairs": n * (n - 1) // 2, "metrics": stats}


def retention_summary(first, final):
    n = len(first)
    matrices = distance_matrices(first, final)
    matrix = matrices["mse"]
    matched = np.diag(matrix)
    result = {"n_seeds": n, "matched_mse": float(matched.mean()),
              "matched_rmse": float(np.sqrt(matched).mean()),
              "matched_centered_cosine": float(np.diag(matrices["centered_cosine"]).mean()),
              "matched_edge_correlation": float(np.diag(matrices["edge_correlation"]).mean()),
              "mse_matrix": matrix.tolist()}
    if n < 2:
        return dict(result, mismatched_mse=None, matched_over_mismatched_mse=None,
                    retrieval_accuracy=None, retrieval_chance=None, ratio_ci95=None,
                    retrieval_ci95=None)
    mismatch = float(matrix[~np.eye(n, dtype=bool)].mean())
    scores, weights = retrieval_scores(matrix), seed_weights(n)
    boot_match = weights @ matched / n
    boot_mismatch = weighted_offdiag(matrix, weights)
    boot_ratio = np.divide(boot_match, boot_mismatch, out=np.full(len(weights), np.nan),
                           where=boot_mismatch > 1e-15)
    return dict(result, mismatched_mse=mismatch,
                matched_over_mismatched_mse=safe_ratio(matched.mean(), mismatch),
                retrieval_accuracy=float(scores.mean()), retrieval_chance=1 / n,
                retrieval_per_seed=scores.tolist(), ratio_ci95=interval(boot_ratio),
                retrieval_ci95=interval(weights @ scores / n))


def stage_sort(stage):
    return (STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 100, stage)


def load_cases(root):
    cases, problems = [], []
    for path in sorted(root.glob("*/prompt_*/seed_*/record.json")):
        if path.parts[-4] not in ("ar", "wan"):
            continue
        try:
            record = json.loads(path.read_text())
            if not record.get("complete"):
                raise ValueError("record 未完成")
            view_path = path.parent / "views.npz"
            expected_hash = record.get("artifact_sha256", {}).get("views.npz")
            if expected_hash and hashlib.sha256(view_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("views.npz SHA256 不匹配")
            with np.load(view_path, allow_pickle=False) as arrays:
                views = {key: arrays[key] for key in arrays.files}
            for key, array in views.items():
                if array.shape != (3, 240, 416, 3) or array.dtype != np.uint8:
                    raise ValueError(f"{key}: 非预期 view schema {array.shape}/{array.dtype}")
            if not {"first", "final"} <= set(views):
                raise ValueError("缺少 first/final")
            model = record["model"]
            if model != path.parts[-4]:
                raise ValueError("model 与目录不一致")
            capture_stages = dict(record.get("stages", {}))
            alignment_source = None
            supplement_path = root / "aligned" / path.relative_to(root)
            try:
                if not supplement_path.is_file():
                    raise ValueError("同σ补充记录缺失；若原记录 second/quarter σ已验证，将使用其 x0")
                supplement = json.loads(supplement_path.read_text())
                if not supplement.get("complete"):
                    raise ValueError("同σ补充 record 未完成")
                if supplement.get("source_record_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("同σ补充 source_record_sha256 与 base record 不匹配")
                if any(supplement.get(key) != record.get(key) for key in ("model", "prompt_index", "seed")):
                    raise ValueError("同σ补充样本身份与 base record 不匹配")
                supplement_view_path = supplement_path.parent / "views.npz"
                expected = supplement.get("artifact_sha256", {}).get("views.npz")
                if not expected:
                    raise ValueError("同σ补充缺少 views.npz artifact SHA256")
                if hashlib.sha256(supplement_view_path.read_bytes()).hexdigest() != expected:
                    raise ValueError("同σ补充 views.npz SHA256 不匹配")
                with np.load(supplement_view_path, allow_pickle=False) as arrays:
                    extra_views = {key: arrays[key] for key in arrays.files}
                if not {"aligned_state", "aligned_x0"} <= set(extra_views):
                    raise ValueError("同σ补充缺少 aligned_state/aligned_x0")
                for key, array in extra_views.items():
                    if key not in ("aligned_state", "aligned_x0", "aligned_cfg1"):
                        raise ValueError(f"非预期补充 stage {key}")
                    if array.shape != (3, 240, 416, 3) or array.dtype != np.uint8:
                        raise ValueError(f"{key}: 非预期 view schema {array.shape}/{array.dtype}")
                    sigma = supplement.get("stages", {}).get(key, {}).get("sigma")
                    if sigma is None or not np.isclose(sigma, ALIGNED_SIGMA, atol=1e-7, rtol=0):
                        raise ValueError(f"{key}: 补充 sigma 未验证为 {ALIGNED_SIGMA}")
                views.update(extra_views)
                capture_stages.update(supplement["stages"])
                alignment_source = "validated_aligned_supplement"
            except (ValueError, KeyError, OSError, TypeError) as error:
                problems.append(f"{supplement_path.relative_to(root)}: {error}")
            if "aligned_x0" not in views:
                fallback = FALLBACK_STAGES[model]
                sigma = capture_stages.get(fallback, {}).get("sigma")
                if fallback in views and sigma is not None and np.isclose(sigma, ALIGNED_SIGMA, atol=1e-7, rtol=0):
                    views["aligned_x0"] = views[fallback]
                    capture_stages["aligned_x0"] = dict(capture_stages[fallback], source_stage=fallback)
                    alignment_source = f"verified_base_{fallback}"
                else:
                    problems.append(f"{path.relative_to(root)}: 无可验证的同σ x0；不纳入主比较")
            cases.append({"record": record, "model": model,
                          "prompt_index": int(record["prompt_index"]), "seed": int(record["seed"]),
                          "capture_stages": capture_stages, "alignment_source": alignment_source,
                          "views": views, "features": {key: lowpass(value) for key, value in views.items()}})
        except (ValueError, KeyError, OSError) as error:
            problems.append(f"{path.relative_to(root)}: {error}")
    return cases, problems


def analyze(cases):
    rows, groups = [], []
    for (prompt, model), grouped in itertools.groupby(
            sorted(cases, key=lambda c: (c["prompt_index"], c["model"], c["seed"])),
            key=lambda c: (c["prompt_index"], c["model"])):
        group = list(grouped)
        item = {"prompt_index": prompt, "model": model, "prompt": group[0]["record"]["prompt"],
                "seeds": [c["seed"] for c in group], "stages": {}, "retention": {},
                "capture_stages": group[0].get("capture_stages", group[0]["record"].get("stages", {})),
                "primary_stage": "aligned_x0", "primary_sigma": ALIGNED_SIGMA,
                "alignment_sources": {str(c["seed"]): c.get("alignment_source") for c in group},
                "run_parameters": {key: group[0]["record"].get("protocol", {}).get(key)
                                   for key in ("guidance", "wan_steps", "shift", "shape", "chunks", "model_source")}}
        stages = sorted(set.union(*(set(c["features"]) for c in group)), key=stage_sort)
        for stage in stages:
            selected = [c for c in group if stage in c["features"]]
            result = pairwise_summary([c["features"][stage] for c in selected])
            result["seeds"] = [c["seed"] for c in selected]
            result["diagnostic_only"] = stage in ("next_state", "aligned_state")
            result["sigma"] = item["capture_stages"].get(stage, {}).get("sigma")
            item["stages"][stage] = result
            for metric, stat in result["metrics"].items():
                ci = stat["ci95_seed_bootstrap"] or [None, None]
                rows.append(dict(prompt_index=prompt, model=model, stage=stage, analysis="across_seed",
                                 metric=metric, value=stat["mean"], ci_low=ci[0], ci_high=ci[1],
                                 n_seeds=len(selected)))
            if stage not in ("final", "final_alt_transition", "next_state", "aligned_state"):
                retention = retention_summary([c["features"][stage] for c in selected],
                                              [c["features"]["final"] for c in selected])
                retention["seeds"] = [c["seed"] for c in selected]
                item["retention"][stage] = retention
                for metric in ("matched_mse", "mismatched_mse", "matched_over_mismatched_mse",
                               "retrieval_accuracy", "matched_centered_cosine", "matched_edge_correlation"):
                    ci_key = {"matched_over_mismatched_mse": "ratio_ci95", "retrieval_accuracy": "retrieval_ci95"}.get(metric)
                    ci = retention.get(ci_key) or [None, None]
                    rows.append(dict(prompt_index=prompt, model=model, stage=stage, analysis="to_final",
                                     metric=metric, value=retention.get(metric), ci_low=ci[0], ci_high=ci[1],
                                     n_seeds=len(selected)))
        alternate = [c for c in group if "final_alt_transition" in c["features"]]
        if alternate:
            baseline = [c["features"]["final"] for c in alternate]
            alt = [c["features"]["final_alt_transition"] for c in alternate]
            matrices = distance_matrices(baseline, alt)
            delta = np.diag(matrices["mse"])
            denom = pairwise_summary(baseline)["metrics"].get("mse", {}).get("mean", 0)
            weights = seed_weights(len(alternate))
            boot_denom = weighted_offdiag(distance_matrices(baseline)["mse"], weights)
            boot_delta = weights @ delta / len(alternate)
            ratios = np.divide(boot_delta, boot_denom, out=np.full(len(weights), np.nan), where=boot_denom > 1e-15)
            item["transition_noise_intervention"] = {
                "seeds": [c["seed"] for c in alternate], "same_initial_changed_transition_mse": float(delta.mean()),
                "across_initial_final_mse": denom, "transition_over_initial_mse": safe_ratio(delta.mean(), denom),
                "ratio_ci95": interval(ratios), "per_seed_mse": delta.tolist(),
                "matched_centered_cosine": float(np.diag(matrices["centered_cosine"]).mean()),
                "matched_edge_correlation": float(np.diag(matrices["edge_correlation"]).mean())}
            for metric, value in item["transition_noise_intervention"].items():
                if isinstance(value, (int, float)) or value is None:
                    ci = interval(ratios) if metric == "transition_over_initial_mse" else None
                    ci = ci or [None, None]
                    rows.append(dict(prompt_index=prompt, model=model, stage="final_alt_transition",
                                     analysis="transition_noise_intervention", metric=metric, value=value,
                                     ci_low=ci[0], ci_high=ci[1], n_seeds=len(alternate)))
        groups.append(item)
    cross = []
    for prompt in sorted({c["prompt_index"] for c in cases}):
        ar = {c["seed"]: c for c in cases if c["prompt_index"] == prompt and c["model"] == "ar"}
        wan = {c["seed"]: c for c in cases if c["prompt_index"] == prompt and c["model"] == "wan"}
        seeds = sorted(ar.keys() & wan.keys())
        if not seeds:
            continue
        hashes_match = all(ar[s]["record"].get("first_chunk_noise_sha256") is not None and
                           ar[s]["record"].get("first_chunk_noise_sha256") ==
                           wan[s]["record"].get("first_chunk_noise_sha256") for s in seeds)
        comparisons = {}
        stage_pairs = [("aligned_x0", "aligned_x0", "primary_same_sigma"),
                       ("second", "quarter", "same_sigma_base_check"),
                       ("aligned_x0", "aligned_cfg1", "same_sigma_probe_cfg1_on_cfg6_trajectory"),
                       ("first", "first", "supplementary_initial_sigma_anchor"),
                       ("second", "second", "supplementary_different_sigma"),
                       ("final", "final", "supplementary_final"),
                       ("first", "first_cfg1", "supplementary_initial_probe"),
                       ("first", "first_horizon3", "supplementary_initial_probe"),
                       ("first", "first_horizon3_cfg1", "supplementary_initial_probe")]
        for ar_stage, stage, role in stage_pairs:
            selected = [seed for seed in seeds if ar_stage in ar[seed]["features"] and stage in wan[seed]["features"]]
            if not selected:
                continue
            comparison = retention_summary([ar[s]["features"][ar_stage] for s in selected],
                                           [wan[s]["features"][stage] for s in selected])
            comparison.update(role=role, seeds=selected,
                              ar_sigma=ar[selected[0]].get("capture_stages", {}).get(ar_stage, {}).get("sigma"),
                              wan_sigma=wan[selected[0]].get("capture_stages", {}).get(stage, {}).get("sigma"),
                              interpretation="跨模型图像配对，不是到 final 的保留度；同σ状态张量可不同")
            comparison["same_sigma"] = (comparison["ar_sigma"] is not None and comparison["wan_sigma"] is not None
                                         and bool(np.isclose(comparison["ar_sigma"], comparison["wan_sigma"], atol=1e-7, rtol=0)))
            if role.startswith("primary_same_sigma") and not comparison["same_sigma"]:
                raise ValueError("主比较 sigma 未对齐")
            if len(selected) > 1:
                ar_mse = distance_matrices([ar[s]["features"][ar_stage] for s in selected])["mse"]
                wan_mse = distance_matrices([wan[s]["features"][stage] for s in selected])["mse"]
                mask = ~np.eye(len(selected), dtype=bool)
                weights = seed_weights(len(selected))
                ar_boot, wan_boot = weighted_offdiag(ar_mse, weights), weighted_offdiag(wan_mse, weights)
                boot = np.divide(ar_boot, wan_boot, out=np.full(len(weights), np.nan), where=wan_boot > 1e-15)
                comparison["ar_over_wan_across_seed_mse"] = safe_ratio(ar_mse[mask].mean(), wan_mse[mask].mean())
                comparison["ar_over_wan_across_seed_mse_ci95"] = interval(boot)
            comparisons[f"ar_{ar_stage}_vs_wan_{stage}"] = comparison
        cross.append({"prompt_index": prompt, "seeds": seeds, "initial_first_chunk_hashes_match": hashes_match,
                      "comparisons": comparisons})
    return {"schema_version": 2, "n_completed_cases": len(cases), "warnings": list(WARNINGS),
            "primary_comparison": {"stage": "aligned_x0", "sigma": ALIGNED_SIGMA,
                                   "ar_base_stage": "second", "wan_base_stage": "quarter",
                                   "ar_completed_updates": 1, "wan_completed_updates": 12,
                                   "ar_probe_number": 2, "wan_probe_number": 13},
            "metric_protocol": {"lowpass_shape": [3, 15, 26, 3], "rgb_range": [0, 1],
                                "resize": "Pillow BOX", "bootstrap_samples": 2000,
                                "constant_correlation": "both centered-zero => 1, only one zero => 0",
                                "zero_denominator_ratio": None},
            "groups": groups, "cross_model": cross}, rows


def save_visuals(root, cases):
    assets, sheets, manifest = root / "report_assets", root / "contact_sheets", []
    assets.mkdir(exist_ok=True)
    sheets.mkdir(exist_ok=True)
    for case in cases:
        prefix = f"{case['model']}_p{case['prompt_index']}_s{case['seed']}"
        paths = {}
        for stage, views in case["views"].items():
            paths[stage] = []
            for frame, pixels in enumerate(views):
                relative = f"report_assets/{prefix}_{stage}_f{frame}.jpg"
                Image.fromarray(pixels).save(root / relative, quality=90)
                paths[stage].append(relative)
        manifest.append({"model": case["model"], "prompt_index": case["prompt_index"],
                         "prompt": case["record"]["prompt"], "seed": case["seed"], "paths": paths,
                         "capture_stages": case.get("capture_stages", {}),
                         "alignment_source": case.get("alignment_source")})
    links = []
    for (prompt, model), grouped in itertools.groupby(
            sorted(cases, key=lambda c: (c["prompt_index"], c["model"], c["seed"])),
            key=lambda c: (c["prompt_index"], c["model"])):
        group = list(grouped)
        stages = sorted(set.union(*(set(c["views"]) for c in group)), key=stage_sort)
        for frame in range(3):
            sheet = Image.new("RGB", (120 + 208 * len(stages), 32 + 144 * len(group)), "#151a23")
            draw = ImageDraw.Draw(sheet)
            for j, stage in enumerate(stages):
                draw.text((120 + j * 208, 8), stage, fill="white")
            for i, case in enumerate(group):
                y = 32 + i * 144
                draw.text((8, y + 48), f"seed {case['seed']}", fill="white")
                for j, stage in enumerate(stages):
                    if stage in case["views"]:
                        thumb = Image.fromarray(case["views"][stage][frame]).resize((208, 120), Image.Resampling.BOX)
                        sheet.paste(thumb, (120 + j * 208, y))
            relative = f"contact_sheets/{model}_prompt_{prompt:06d}_frame_{[0,4,8][frame]}.jpg"
            sheet.save(root / relative, quality=91)
            links.append({"path": relative, "label": f"{model.upper()} prompt {prompt} frame {[0,4,8][frame]}"})
    return manifest, links


def fmt(value):
    return "—" if value is None else f"{value:.4g}"


def write_html(root, summary, manifest, sheets):
    from scripts.denoising.first_step_noise_report import write_report
    write_report(root, summary, manifest, sheets)


def existing_visuals(root, summary):
    """Rebuild the report manifest from published images, without raw data IO."""
    assets = root / "report_assets"
    filenames = {path.name for path in assets.iterdir() if path.suffix == ".jpg"}
    manifest = []
    for group in summary["groups"]:
        model, prompt = group["model"], group["prompt_index"]
        for seed in group["seeds"]:
            paths = {}
            for stage in group["stages"]:
                names = [f"{model}_p{prompt}_s{seed}_{stage}_f{frame}.jpg" for frame in range(3)]
                if all(name in filenames for name in names):
                    paths[stage] = [f"report_assets/{name}" for name in names]
            if not all(stage in paths for stage in ("aligned_x0", "final")):
                raise ValueError(f"Missing published main images: {model}, prompt={prompt}, seed={seed}")
            manifest.append({"model": model, "prompt_index": prompt, "seed": seed,
                             "prompt": group["prompt"], "paths": paths,
                             "capture_stages": group.get("capture_stages", {}),
                             "alignment_source": group.get("alignment_sources", {}).get(str(seed))})
    sheets = [{"path": str(path.relative_to(root)), "label": path.stem}
              for path in sorted((root / "contact_sheets").glob("*.jpg"))]
    (root / "visual_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    return manifest, sheets


def self_test():
    # Perfect correspondence, reversed correspondence and all-distance ties.
    assert np.allclose(retrieval_scores(np.array([[0., 2.], [2., 0.]])), [1, 1])
    assert np.allclose(retrieval_scores(np.array([[2., 0.], [0., 2.]])), [0, 0])
    assert np.allclose(retrieval_scores(np.zeros((8, 8))), np.full(8, 1 / 8))
    constant = [np.zeros((3, 15, 26, 3))] * 8
    stats = pairwise_summary(constant)
    retention = retention_summary(constant, constant)
    assert retention["retrieval_accuracy"] == 1 / 8
    assert retention["matched_over_mismatched_mse"] is None
    assert stats["metrics"]["mse"]["mean"] == 0
    assert centered_cosine(np.ones(3), np.ones(3)) == 1
    json.dumps([stats, retention], allow_nan=False)
    rng = np.random.default_rng(3)
    features = list(rng.random((8, 3, 15, 26, 3)))
    retention = retention_summary(features, features)
    assert retention["retrieval_accuracy"] == 1
    assert retention["matched_over_mismatched_mse"] < 1e-12
    json.dumps(retention, allow_nan=False)
    # Alignment comes from the recorded sigma, never a stage's English name.
    with tempfile.TemporaryDirectory(prefix="first_step_report_test_") as temporary:
        root = Path(temporary)
        for model, fallback in FALLBACK_STAGES.items():
            for seed in range(2):
                directory = root / model / "prompt_000000" / f"seed_{seed:06d}"
                directory.mkdir(parents=True)
                view = np.full((3, 240, 416, 3), seed * 120, dtype=np.uint8)
                np.savez_compressed(directory / "views.npz", first=view, final=view, **{fallback: view})
                record = {"complete": True, "model": model, "prompt_index": 0, "prompt": "test", "seed": seed,
                          "first_chunk_noise_sha256": f"noise{seed}",
                          "stages": {"first": {"sigma": 1.}, fallback: {"sigma": ALIGNED_SIGMA}},
                          "artifact_sha256": {"views.npz": hashlib.sha256((directory / "views.npz").read_bytes()).hexdigest()}}
                (directory / "record.json").write_text(json.dumps(record))
        cases, problems = load_cases(root)
        assert len(cases) == 4 and len(problems) == 4
        assert all(c["alignment_source"].startswith("verified_base_") for c in cases)
        summary, _ = analyze(cases)
        assert summary["cross_model"][0]["comparisons"]["ar_aligned_x0_vs_wan_aligned_x0"]["same_sigma"]
        base = root / "wan/prompt_000000/seed_000000/record.json"
        aligned = root / "aligned/wan/prompt_000000/seed_000000"
        aligned.mkdir(parents=True)
        np.savez_compressed(aligned / "views.npz", aligned_x0=view, aligned_state=view, aligned_cfg1=view)
        extra = {**json.loads(base.read_text()), "source_record_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                 "stages": {key: {"sigma": ALIGNED_SIGMA} for key in ("aligned_x0", "aligned_state", "aligned_cfg1")},
                 "artifact_sha256": {"views.npz": hashlib.sha256((aligned / "views.npz").read_bytes()).hexdigest()}}
        (aligned / "record.json").write_text(json.dumps(extra))
        cases, problems = load_cases(root)
        valid = next(c for c in cases if c["model"] == "wan" and c["seed"] == 0)
        assert valid["alignment_source"] == "validated_aligned_supplement" and len(problems) == 3
        summary, _ = analyze(cases)
        assert all("aligned_state" not in g["retention"] for g in summary["groups"])
        assert summary["cross_model"][0]["comparisons"]["ar_aligned_x0_vs_wan_aligned_cfg1"]["role"] == "same_sigma_probe_cfg1_on_cfg6_trajectory"
        json.dumps(summary, allow_nan=False)
        extra["source_record_sha256"] = "wrong"
        (aligned / "record.json").write_text(json.dumps(extra))
        cases, problems = load_cases(root)
        rejected = next(c for c in cases if c["model"] == "wan" and c["seed"] == 0)
        assert rejected["alignment_source"] == "verified_base_quarter" and "aligned_state" not in rejected["views"]
        assert any("source_record_sha256" in p for p in problems)
        extra["source_record_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
        extra["artifact_sha256"]["views.npz"] = "wrong"
        (aligned / "record.json").write_text(json.dumps(extra))
        _, problems = load_cases(root)
        assert any("views.npz SHA256" in p for p in problems)
        # A named quarter with the wrong sigma must not be admitted as aligned.
        record = json.loads(base.read_text())
        record["stages"]["quarter"]["sigma"] = 0.95
        base.write_text(json.dumps(record))
        cases, _ = load_cases(root)
        rejected = next(c for c in cases if c["model"] == "wan" and c["seed"] == 0)
        assert "aligned_x0" not in rejected["views"]
        write_html(root, summary, [], [])
        page = (root / "report.html").read_text()
        assert "同 σ 图像对照" in page and "PRIMARY_ROWS" not in page and "MANIFEST" not in page
    print("self-test passed: retrieval/bootstrap, recorded-sigma alignment, supplement source/artifact hashes, diagnostic exclusion, report rendering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render-only", action="store_true",
                        help="Render existing summary and JPEG assets without recalculating experiment metrics")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.output_dir is None:
            return
    if args.output_dir is None:
        parser.error("--output-dir is required unless only running --self-test")
    root = args.output_dir.resolve()
    if args.render_only:
        summary = json.loads((root / "summary.json").read_text())
        manifest, sheets = existing_visuals(root, summary)
        write_html(root, summary, manifest, sheets)
        print(json.dumps({"render_only": True, "cases": len(manifest), "report": str(root / "report.html")}))
        return
    cases, problems = load_cases(root)
    if not cases:
        parser.error(f"No valid completed cases under {root}")
    summary, rows = analyze(cases)
    summary["problems"] = problems
    sources = [c.get("alignment_source") for c in cases]
    summary["alignment_coverage"] = {
        "validated_supplements": sources.count("validated_aligned_supplement"),
        "verified_base_fallbacks": sum(bool(s and s.startswith("verified_base_")) for s in sources),
        "missing_aligned_x0": sources.count(None),
        "validation": "Supplement source_record_sha256 and views.npz artifact hash; latent tensors are not read by this CPU report."}
    if summary["alignment_coverage"]["verified_base_fallbacks"]:
        summary["warnings"].append("部分同σ补充缺失或未通过检查，已回退到原记录中明确标注 σ=0.9375 的 AR second / Wan quarter；这些样本没有补充 aligned_state 诊断或 aligned_cfg1 控制，详见问题列表。")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (root / "summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["prompt_index", "model", "stage", "analysis", "metric", "value", "ci_low", "ci_high", "n_seeds"])
        writer.writeheader()
        writer.writerows(rows)
    manifest, sheets = save_visuals(root, cases)
    write_html(root, summary, manifest, sheets)
    print(json.dumps({"completed_cases": len(cases), "groups": len(summary["groups"]),
                      "reported_problems": len(problems), "alignment_coverage": summary["alignment_coverage"],
                      "report": str(root / "report.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
