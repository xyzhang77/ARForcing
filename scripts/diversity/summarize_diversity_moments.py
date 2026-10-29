#!/usr/bin/env python3
"""Small paired-seed diversity evaluation in the shared, raw Wan VAE space.

Input: ROOT/{ar,wan}/prompt_000042/seed_000000/{record.json,features.npz}.
The NPZ must contain `latent` (x0 at the aligned sigma; no normalization).
Only NumPy and Python's standard library are required.
"""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import quote

import numpy as np

PROMPT_LABELS = {42: "绿洲", 300: "猫"}


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def gram(samples):
    """Inner products / feature dimension; accumulate in float64."""
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    if x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be nonempty and finite")
    return (x @ x.T) / x.shape[1]


def pooled_gram(ar, wan):
    if ar.shape != wan.shape or len(ar) < 2:
        raise ValueError("AR/Wan must have the same shape and at least two seeds")
    return gram(np.concatenate((ar, wan)))


def moments_from_gram(g):
    n = len(g)
    if n < 2:
        raise ValueError("Sample variance needs at least two seeds")
    mean_squared = float(g.mean())
    variance = float((np.trace(g) - n * mean_squared) / (n - 1))
    if variance < -1e-10 * max(1, float(np.diag(g).max())):
        raise ValueError("Negative variance beyond numerical tolerance")
    return {"mean_tensor_rms": math.sqrt(max(0.0, mean_squared)),
            "mean_coordinate_variance_unbiased": max(0.0, variance)}


def squared_distances(g):
    diagonal = np.diag(g)
    result = np.maximum(diagonal[:, None] + diagonal[None, :] - 2 * g, 0)
    np.fill_diagonal(result, 0)
    return result


def pilot_bandwidth_squared(g, n, pilot_indices):
    """ell^2 = median pooled off-diagonal squared distance / 2."""
    selected = list(pilot_indices) + [n + i for i in pilot_indices]
    distance = squared_distances(g[np.ix_(selected, selected)])
    values = distance[np.triu_indices(len(selected), k=1)]
    median = float(np.median(values))
    # A constant pilot cannot define an RBF bandwidth; do not invent a scale.
    return median / 2 if median > 0 else None


def mmd_from_gram(g, n, bandwidth_squared):
    if bandwidth_squared is None:
        return {"mmd2_paired_u": None, "mmd2_v_descriptive": None}
    kernel = np.exp(-squared_distances(g) / (2 * bandwidth_squared))
    aa, bb, ab = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    # The same seed couples AR and Wan. Every cross i=j must be excluded,
    # just as the within-model diagonals are excluded, in the paired U form.
    h = aa + bb - ab - ab.T
    u = float((h.sum() - np.trace(h)) / (n * (n - 1)))
    v = float(h.mean())
    return {"mmd2_paired_u": u, "mmd2_v_descriptive": max(0.0, v)}


def variance_ratio(ar_variance, wan_variance):
    return ar_variance / wan_variance if wan_variance > 0 else None


def bootstrap_variance_ratio(gaa, gbb, draws=2000, random_seed=20261023):
    """Ordinary paired-seed bootstrap, including duplicate sampled seeds.

    Counts are merely a memory-saving evaluation of the actual bootstrap
    sample mean and ddof=1 variance, not off-diagonal/pair reweighting.
    """
    n = len(gaa)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, n, (draws, n))
    counts = np.zeros((draws, n), dtype=np.float64)
    for row, sampled in enumerate(indices):
        counts[row] = np.bincount(sampled, minlength=n)

    def variances(g):
        mean_squared = np.einsum("bi,ij,bj->b", counts, g, counts, optimize=True) / n**2
        raw_second = counts @ np.diag(g) / n
        return np.maximum((raw_second - mean_squared) * n / (n - 1), 0)

    a, b = variances(gaa), variances(gbb)
    ratios = np.divide(a, b, out=np.full(draws, np.nan), where=b > 0)
    valid = ratios[np.isfinite(ratios)]
    return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "draws": draws, "valid_draws": len(valid), "seed": random_seed}


def summarize_gram(g, n, bandwidth_squared, bootstrap_draws=2000):
    aa, bb, ab = g[:n, :n], g[n:, n:], g[:n, n:]
    ar, wan = moments_from_gram(aa), moments_from_gram(bb)
    mean_difference = max(0.0, float(aa.mean() + bb.mean() - 2 * ab.mean()))
    result = {"n_seeds": n, "ar": ar, "wan": wan,
              "variance_ratio_ar_over_wan": variance_ratio(
                  ar["mean_coordinate_variance_unbiased"], wan["mean_coordinate_variance_unbiased"]),
              "mean_difference_mse": mean_difference,
              **mmd_from_gram(g, n, bandwidth_squared)}
    if bootstrap_draws:
        result["variance_ratio_bootstrap"] = bootstrap_variance_ratio(aa, bb, bootstrap_draws)
    return result


def compare_samples(ar, wan, seeds, pilot_seeds, bootstrap_draws=2000):
    g, n = pooled_gram(ar, wan), len(ar)
    pilot_complete = set(pilot_seeds).issubset(seeds)
    ell2 = pilot_bandwidth_squared(g, n, [seeds.index(s) for s in pilot_seeds]) if pilot_complete else None
    result = summarize_gram(g, n, ell2, bootstrap_draws)
    result["bandwidth"] = {"squared": ell2, "pilot_seeds": pilot_seeds,
                           "pilot_complete": pilot_complete,
                           "rule": "pooled pilot median off-diagonal squared distance / 2",
                           "distance": "raw flattened latent squared Euclidean distance / dimension"}
    result["stability"] = []
    # Reuse exactly the same pilot bandwidth at every prefix size.
    for size in [32, 64]:
        if n < size:
            continue
        selected = list(range(size)) + list(range(n, n + size))
        prefix = summarize_gram(g[np.ix_(selected, selected)], size, ell2, 0)
        result["stability"].append(prefix)
    if n >= 64:
        before, after = result["stability"][:2]
        result["stability_32_to_64"] = {
            "variance_ratio_relative_change": (
                after["variance_ratio_ar_over_wan"] / before["variance_ratio_ar_over_wan"] - 1
                if before["variance_ratio_ar_over_wan"] not in (None, 0) else None),
            "mmd2_paired_u_absolute_change": (
                after["mmd2_paired_u"] - before["mmd2_paired_u"]
                if after["mmd2_paired_u"] is not None else None),
            "ar_variance_relative_change": (
                after["ar"]["mean_coordinate_variance_unbiased"] / before["ar"]["mean_coordinate_variance_unbiased"] - 1
                if before["ar"]["mean_coordinate_variance_unbiased"] else None),
            "wan_variance_relative_change": (
                after["wan"]["mean_coordinate_variance_unbiased"] / before["wan"]["mean_coordinate_variance_unbiased"] - 1
                if before["wan"]["mean_coordinate_variance_unbiased"] else None)}
    return result


def load_case(directory, model, prompt_index, seed, expected_sigma):
    with (directory / "record.json").open() as stream:
        record = json.load(stream)
    if record.get("complete") is not True:
        raise ValueError(f"{directory}: record does not declare complete=True")
    if record.get("schema_version") != "diversity_moments_sampling_v1":
        raise ValueError(f"{directory}: unsupported sampling schema")
    expected_hash = record.get("artifact_sha256", {}).get("features.npz")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"{directory}: missing features.npz artifact SHA-256")
    actual_hash = hashlib.sha256((directory / "features.npz").read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{directory}: features.npz artifact SHA-256 mismatch")
    for key, expected in [("model", model), ("prompt_index", prompt_index), ("seed", seed)]:
        if record.get(key) != expected:
            raise ValueError(f"{directory}: record {key} does not match {expected!r}")
    sigma = record.get("sigma", record.get("aligned_sigma"))
    if sigma is None or not math.isclose(float(sigma), expected_sigma, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"{directory}: sigma {sigma!r} != expected {expected_sigma}")
    noise_hash = record.get("noise_sha256", record.get("initial_noise_sha256"))
    if not isinstance(noise_hash, str) or not noise_hash:
        raise ValueError(f"{directory}: missing initial noise hash")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        latent = np.asarray(archive["latent"], dtype=np.float32)
    if latent.size == 0 or not np.isfinite(latent).all():
        raise ValueError(f"{directory}: non-finite or empty latent")
    return {"record": record, "latent": latent, "noise_sha256": noise_hash, "directory": directory}


def load_inputs(root, prompts, expected_seeds, seed_start, sigma, allow_partial):
    expected = list(range(seed_start, seed_start + expected_seeds))
    groups, completeness = [], []
    reference_shape = None
    for prompt_index in prompts:
        cases = {model: {} for model in ("ar", "wan")}
        prompt_text = None
        for model in cases:
            prompt_directory = root / model / f"prompt_{prompt_index:06d}"
            available = []
            if prompt_directory.exists():
                for directory in sorted(prompt_directory.glob("seed_*")):
                    if not directory.is_dir():
                        continue
                    try:
                        seed = int(directory.name.removeprefix("seed_"))
                    except ValueError as error:
                        raise ValueError(f"Unexpected sample directory: {directory}") from error
                    if seed not in expected:
                        raise ValueError(f"Unexpected seed {seed} outside planned seeds: {directory}")
                    if not (directory / "record.json").exists() or not (directory / "features.npz").exists():
                        continue
                    case = load_case(directory, model, prompt_index, seed, sigma)
                    current_prompt = case["record"].get("prompt")
                    if not isinstance(current_prompt, str) or not current_prompt:
                        raise ValueError(f"{directory}: missing prompt text")
                    if prompt_text is None:
                        prompt_text = current_prompt
                    if current_prompt != prompt_text:
                        raise ValueError(f"Prompt text changes across seeds/models: {directory}")
                    if reference_shape is None:
                        reference_shape = case["latent"].shape
                    if case["latent"].shape != reference_shape:
                        raise ValueError(f"Latent shape differs: {directory}")
                    cases[model][seed] = case
                    available.append(seed)
            missing = sorted(set(expected) - set(available))
            completeness.append({"prompt_index": prompt_index, "model": model,
                                 "expected": expected_seeds, "completed": len(available), "missing_seeds": missing})
        common = sorted(set(cases["ar"]) & set(cases["wan"]))
        for seed in common:
            a, b = cases["ar"][seed], cases["wan"][seed]
            # Prefer the common-prefix hash if both writers include it. Whole
            # noise tensors may differ in shape between causal and full-video runs.
            compared_hash = False
            for field in ("first_chunk_noise_sha256", "noise_sha256", "initial_noise_sha256"):
                if field in a["record"] and field in b["record"]:
                    compared_hash = True
                    if a["record"][field] != b["record"][field]:
                        raise ValueError(f"Prompt {prompt_index}, seed {seed}: {field} mismatch")
            if not compared_hash and a["noise_sha256"] != b["noise_sha256"]:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: noise hash mismatch")
            if a["record"].get("prompt") != b["record"].get("prompt"):
                raise ValueError(f"Prompt text mismatch: prompt {prompt_index}, seed {seed}")
        groups.append({"prompt_index": prompt_index, "seeds": common, "cases": cases})
    complete = all(not row["missing_seeds"] for row in completeness)
    if not complete and not allow_partial:
        description = "; ".join(f"{row['model']}/prompt{row['prompt_index']}: {row['completed']}/{expected_seeds}" for row in completeness)
        raise ValueError(f"Incomplete experiment ({description}). Use --allow-partial only for a provisional report.")
    return groups, completeness, complete, reference_shape


def direction(item):
    interval = item.get("variance_ratio_bootstrap", {}).get("ci95")
    if interval and interval[1] < 1:
        return "AR 方差更小"
    if interval and interval[0] > 1:
        return "AR 方差更大"
    return "区间跨 1，方向未定"


def format_number(value, precision=4):
    return "—" if value is None else f"{value:.{precision}g}"


def report_html(summary, root):
    rows, diagnostics, images = [], [], []
    for item in summary["prompts"]:
        p, n = item["prompt_index"], item["n_seeds"]
        label = f"{PROMPT_LABELS.get(p, 'Prompt')} #{p}"
        if n < 2:
            rows.append(f'<tr><td>{label}</td><td>{n}</td><td colspan="5">等待至少 2 个配对 seed</td></tr>')
            continue
        ci = item["variance_ratio_bootstrap"]["ci95"]
        ci_text = "—" if ci is None else f"[{format_number(ci[0], 3)}, {format_number(ci[1], 3)}]"
        prompt = html.escape(item.get("prompt", ""))
        rows.append(f'''<tr><td><strong>{label}</strong><small>{prompt[:150]}</small></td><td>{n}</td>
<td>{format_number(item['ar']['mean_coordinate_variance_unbiased'])}</td>
<td>{format_number(item['wan']['mean_coordinate_variance_unbiased'])}</td>
<td><strong>{format_number(item['variance_ratio_ar_over_wan'], 3)}</strong><small>95% CI {ci_text}<br>{direction(item)}</small></td>
<td>{format_number(item['mean_difference_mse'])}</td>
<td>{format_number(item['mmd2_paired_u'])}</td></tr>''')
        stability = item.get("stability", [])
        stability_rows = "".join(f"<tr><td>{x['n_seeds']}</td><td>{format_number(x['variance_ratio_ar_over_wan'])}</td><td>{format_number(x['mmd2_paired_u'])}</td></tr>" for x in stability)
        diagnostics.append(f'''<h3>Prompt #{p}</h3><p>均值张量 RMS：AR {format_number(item['ar']['mean_tensor_rms'])}；Wan {format_number(item['wan']['mean_tensor_rms'])}。
两模型均值之差 MSE：{format_number(item['mean_difference_mse'])}。RBF 带宽 ℓ²：{format_number(item['bandwidth']['squared'])}。</p>
<table><tr><th>配对 seed 数</th><th>AR / Wan 方差比</th><th>MMD²</th></tr>{stability_rows or '<tr><td colspan="3">累计到 32 个 seed 后显示稳定性。</td></tr>'}</table>''')
        model_images = []
        for model in ("ar", "wan"):
            thumbnails = []
            for seed in item["seeds"][:8]:
                image_path = root / model / f"prompt_{p:06d}" / f"seed_{seed:06d}" / "preview.jpg"
                if image_path.exists():
                    url = quote(image_path.relative_to(root).as_posix(), safe="/._-")
                    thumbnails.append(f'<a href="{url}"><img src="{url}" loading="lazy" alt="{model} seed {seed}"><small>seed {seed}</small></a>')
            if thumbnails:
                model_images.append(f'<h4>{model.upper()}</h4><div class="gallery">{"".join(thumbnails)}</div>')
        if model_images:
            images.append(f'<h3>Prompt #{p}：前 8 个 seed，点击看大图</h3>{"".join(model_images)}')
    status = "完整结果" if summary["complete"] else "运行中 · 临时结果"
    completed = sum(row["completed"] for row in summary["completeness"])
    total = sum(row["expected"] for row in summary["completeness"])
    conclusion = "；".join(f"{PROMPT_LABELS.get(x['prompt_index'], 'Prompt')} #{x['prompt_index']}：{direction(x)}" for x in summary["prompts"] if x["n_seeds"] >= 2)
    missing = "" if summary["complete"] else '<p class="pending">样本尚未齐全；当前只分析两模型共同完成的 seed，暂不作为最终结论。</p>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AR / Wan diversity：方差与 MMD</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.65 system-ui,sans-serif;margin:40px auto;max-width:1120px;padding:0 24px;color:#173238;background:#fafcfc}}
h1{{font-size:28px;line-height:1.3}}h2{{font-size:21px;margin-top:32px}}h3{{font-size:18px}}p{{max-width:1000px}}small{{display:block;color:#667b80;font-size:12px;font-weight:400}}a{{color:#086a78}}
.status{{color:#587078}}.lead{{background:#e6f2f0;padding:18px 22px;border-radius:12px;font-size:18px}}.pending{{color:#89590f;background:#fff3d5;padding:12px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:white}}th,td{{padding:14px 12px;text-align:left;border-bottom:1px solid #dce7e7;vertical-align:top}}th{{font-size:13px;color:#587078}}td:first-child{{max-width:280px}}details{{margin-top:24px}}summary{{cursor:pointer;font-weight:650}}
.gallery{{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:8px}}.gallery img{{width:100%;border-radius:6px;display:block}}.gallery a{{text-decoration:none}}h4{{margin:12px 0 6px}}@media(max-width:700px){{body{{margin:22px auto;padding:0 12px}}.gallery{{grid-template-columns:repeat(4,minmax(0,1fr))}}}}
</style><body><p class="status">{status} · {completed}/{total} 个样本 · σ={summary['sigma']} · 每个 prompt 计划 {summary['expected_seeds']} 个 seed / 模型</p>
<h1>同一 prompt，AR 和 Wan 的 noise 多样性差多少？</h1>
<p>在相同 noise level 取两模型的 x̂₀，共享 Wan VAE 原始坐标、相同前 3 个 latent 帧。主指标只看跨 seed 的方差。</p>
{missing}<div class="lead">{conclusion or '等待样本。'}<small>方差比 = AR / Wan。小于 1 表示 AR 的 latent 更集中；大于 1 表示更分散。</small></div>
<h2>主结果</h2><div class="scroll"><table><thead><tr><th>Prompt</th><th>配对 seeds</th><th>AR 方差</th><th>Wan 方差</th><th>方差比与不确定性</th><th>均值差 MSE</th><th>MMD²<br><small>分布差异</small></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><strong>读法：</strong>方差回答“谁更分散”；均值差反映分布中心偏移；MMD 描述两组分布差异的大小，不能说明谁更多样。这里只衡量共享 latent 空间的分散程度，不等同于语义覆盖或生成质量。</p>
<details><summary>均值、32 → 64 seed 稳定性与计算方式</summary>{''.join(diagnostics)}
<p>对每个模型的 n 个原始 latent 展平为 D 维向量。方差 = Σᵢ‖xᵢ−μ‖² / [(n−1)D]；均值 RMS = √(‖μ‖²/D)；均值差 MSE = ‖μ_AR−μ_Wan‖²/D。不做逐样本归一化或逐模型标准化。95% 区间为 {summary['bootstrap_draws']} 次普通配对 seed bootstrap：每次同时重采样相同 AR/Wan seed，并重新计算均值、ddof=1 方差及其比值。</p>
<p>RBF 核使用每坐标平方距离 d²=‖x−y‖²/D，k=exp[−d²/(2ℓ²)]。每个 prompt 用固定 pilot seeds {', '.join(map(str, summary['pilot_seeds']))} 的两模型 pooled 样本估计 ℓ²=非对角 d² 中位数/2，再固定用于 32/64 seed。MMD² 使用配对 U 形式：Σᵢ≠ⱼ[k(aᵢ,aⱼ)+k(bᵢ,bⱼ)−k(aᵢ,bⱼ)−k(aⱼ,bᵢ)]/[n(n−1)]。同 seed 的跨模型项也去除；有限样本估计允许为负。由于带宽 pilot 也包含在最终样本中，此处不宣称数据自适应估计严格无偏，不作显著性检验。</p>
<p>已检查两模型 prompt、seed、noise SHA-256、σ 和 latent shape 一致。<a href="summary.json">完整 JSON</a> · <a href="summary.csv">CSV</a></p></details>
{''.join(images)}<p><small>MMD 定义参考：<a href="https://www.jmlr.org/papers/v13/gretton12a.html">Gretton et al., A Kernel Two-Sample Test (2012)</a>。</small></p></body></html>'''


def write_outputs(root, summary):
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fields = ["prompt_index", "n_seeds", "ar_variance", "wan_variance", "variance_ratio_ar_over_wan", "ratio_ci95_low", "ratio_ci95_high", "ar_mean_rms", "wan_mean_rms", "mean_difference_mse", "mmd2_paired_u", "rbf_bandwidth_squared", "complete"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in summary["prompts"]:
            if item["n_seeds"] < 2:
                continue
            ci = item["variance_ratio_bootstrap"]["ci95"] or [None, None]
            writer.writerow(dict(prompt_index=item["prompt_index"], n_seeds=item["n_seeds"],
                ar_variance=item["ar"]["mean_coordinate_variance_unbiased"], wan_variance=item["wan"]["mean_coordinate_variance_unbiased"],
                variance_ratio_ar_over_wan=item["variance_ratio_ar_over_wan"], ratio_ci95_low=ci[0], ratio_ci95_high=ci[1],
                ar_mean_rms=item["ar"]["mean_tensor_rms"], wan_mean_rms=item["wan"]["mean_tensor_rms"],
                mean_difference_mse=item["mean_difference_mse"], mmd2_paired_u=item["mmd2_paired_u"],
                rbf_bandwidth_squared=item["bandwidth"]["squared"], complete=summary["complete"]))
    (root / "report.html").write_text(report_html(summary, root))


def self_test():
    rng = np.random.default_rng(831)
    a = rng.normal(size=(64, 128)).astype(np.float32)
    seeds, pilot = list(range(64)), list(range(8))
    same = compare_samples(a, a, seeds, pilot, 200)
    assert math.isclose(same["variance_ratio_ar_over_wan"], 1, abs_tol=1e-12)
    assert abs(same["mmd2_paired_u"]) < 1e-12
    assert np.allclose(same["variance_ratio_bootstrap"]["ci95"], [1, 1])
    changed = compare_samples(a, a * 2, seeds, pilot, 200)
    assert math.isclose(changed["variance_ratio_ar_over_wan"], .25, rel_tol=1e-12)
    assert np.allclose(changed["variance_ratio_bootstrap"]["ci95"], [.25, .25])
    assert changed["mmd2_paired_u"] > .05
    assert math.isclose(changed["ar"]["mean_coordinate_variance_unbiased"], float(a.astype(np.float64).var(axis=0, ddof=1).mean()), rel_tol=1e-12)
    assert math.isclose(changed["mean_difference_mse"], float(np.mean(a.astype(np.float64).mean(axis=0)**2)), rel_tol=1e-12)
    constant = compare_samples(np.ones((8, 4)), np.ones((8, 4)), list(range(8)), pilot, 40)
    assert constant["variance_ratio_ar_over_wan"] is None
    assert constant["bandwidth"]["squared"] is None
    assert constant["variance_ratio_bootstrap"]["valid_draws"] == 0
    bad = a.copy(); bad[0, 0] = np.nan
    try:
        compare_samples(bad, a, seeds, pilot, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN input was accepted")
    # Check exact equivalence of count-based bootstrap to explicit resampling.
    direct_rng = np.random.default_rng(20261023)
    b = rng.normal(size=a.shape)
    indices = direct_rng.integers(0, len(a), (100, len(a)))
    explicit = [a[i].astype(np.float64).var(axis=0, ddof=1).mean() / b[i].var(axis=0, ddof=1).mean() for i in indices]
    via_gram = bootstrap_variance_ratio(gram(a), gram(b), 100)["ci95"]
    assert np.allclose(via_gram, np.quantile(explicit, [.025, .975]), rtol=1e-12)
    # Exercise paired MMD exclusion against its literal h(i,j) definition.
    g = pooled_gram(a[:8], b[:8]); ell2 = pilot_bandwidth_squared(g, 8, list(range(8)))
    k = np.exp(-squared_distances(g) / (2 * ell2))
    literal = sum(k[i,j]+k[8+i,8+j]-k[i,8+j]-k[j,8+i] for i in range(8) for j in range(8) if i != j) / (8*7)
    assert math.isclose(mmd_from_gram(g, 8, ell2)["mmd2_paired_u"], literal, abs_tol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        groups, coverage, complete, shape = load_inputs(root, [42], 4, 0, .9375, True)
        assert not complete and not groups[0]["seeds"] and shape is None
        try:
            load_inputs(root, [42], 4, 0, .9375, False)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete inputs silently accepted")
        for model in ("ar", "wan"):
            for seed in range(4):
                sample = root / model / "prompt_000042" / f"seed_{seed:06d}"
                sample.mkdir(parents=True)
                np.savez(sample / "features.npz", latent=a[seed])
                record = dict(complete=True, schema_version="diversity_moments_sampling_v1",
                              model=model, prompt_index=42, seed=seed, prompt="test oasis", sigma=.9375,
                              noise_sha256=f"seed-{seed}", artifact_sha256={"features.npz": hashlib.sha256((sample / "features.npz").read_bytes()).hexdigest()})
                (sample / "record.json").write_text(json.dumps(record))
        groups, coverage, complete, shape = load_inputs(root, [42], 4, 0, .9375, False)
        assert complete and len(groups[0]["seeds"]) == 4
        sample = root / "ar/prompt_000042/seed_000000"
        with (sample / "features.npz").open("ab") as stream:
            stream.write(b"changed after generation")
        try:
            load_inputs(root, [42], 4, 0, .9375, False)
        except ValueError as error:
            assert "SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("Modified features silently accepted")
    print("Self-test passed: moments, paired bootstrap, paired MMD, constants, NaN, completeness, artifact integrity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-indices", default="42,300")
    parser.add_argument("--expected-seeds", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=.9375)
    parser.add_argument("--pilot-seeds", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is set")
    if args.expected_seeds < 2 or args.pilot_seeds < 2 or args.bootstrap_draws < 1:
        parser.error("expected/pilot seeds must be >=2 and bootstrap draws >=1")
    if args.pilot_seeds > args.expected_seeds:
        parser.error("pilot seeds cannot exceed expected seeds")
    prompts = [int(value) for value in args.prompt_indices.split(",")]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups, coverage, complete, shape = load_inputs(root, prompts, args.expected_seeds, args.seed_start, args.sigma, args.allow_partial)
    pilot = list(range(args.seed_start, args.seed_start + args.pilot_seeds))
    result = {"complete": complete, "sigma": args.sigma, "expected_seeds": args.expected_seeds,
              "feature_space": "raw_shared_wan_vae_x0_first3_latent_frames", "feature_shape": shape,
              "bootstrap_draws": args.bootstrap_draws, "pilot_seeds": pilot,
              "completeness": coverage, "prompts": [],
              "interpretation": "Variance measures latent spread; MMD measures distribution discrepancy. Neither measures semantic coverage or quality."}
    for group in groups:
        seeds = group["seeds"]
        item = {"prompt_index": group["prompt_index"], "seeds": seeds, "n_seeds": len(seeds)}
        if seeds:
            item["prompt"] = group["cases"]["ar"][seeds[0]]["record"].get("prompt", "")
        if len(seeds) >= 2:
            arrays = [np.stack([group["cases"][model][seed]["latent"] for seed in seeds]) for model in ("ar", "wan")]
            item.update(compare_samples(*arrays, seeds, pilot, args.bootstrap_draws))
        result["prompts"].append(item)
    write_outputs(root, result)
    print(json.dumps({"complete": complete, "output": str(root / "report.html"), "prompts": [
        {"prompt": p["prompt_index"], "n": p["n_seeds"], "variance_ratio": p.get("variance_ratio_ar_over_wan"),
         "ratio_ci95": p.get("variance_ratio_bootstrap", {}).get("ci95"), "mmd2": p.get("mmd2_paired_u")} for p in result["prompts"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
