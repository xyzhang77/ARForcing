"""Summarize cross-noise dispersion and paired changes along one teacher trajectory."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diversity.summarize_diversity_moments import (
    bootstrap_variance_ratio, gram, mmd_from_gram, moments_from_gram,
    squared_distances, variance_ratio,
)
from scripts.diversity.summarize_diversity_threeway import canonical_hash


SCHEMA = "teacher_steps_summary_v1"
SAMPLING_SCHEMA = "teacher_steps_sampling_v1"
STEPS = ("step_2", "step_3", "step_4")
PAIRS = (("step_2", "step_3"), ("step_3", "step_4"), ("step_2", "step_4"))
FEATURE_SHAPE = [3, 16, 60, 104]
PROMPT_LABELS = {42: "沙漠绿洲", 300: "猫喝水"}
BOOTSTRAP_SEED = 20261024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_case(directory, prompt_index, seed):
    record = json.loads((directory / "record.json").read_text())
    require(record.get("complete") is True, f"{directory}: incomplete record")
    require(record.get("schema_version") == SAMPLING_SCHEMA, f"{directory}: unknown schema")
    require(record.get("model") == "wan14b" and record.get("prompt_index") == prompt_index
            and record.get("seed") == seed, f"{directory}: identity mismatch")
    protocol = record.get("protocol", {})
    require(record.get("protocol_sha256") == canonical_hash(protocol), f"{directory}: protocol hash mismatch")
    for key in ("model", "prompt_index", "prompt", "seed", "model_source", "guidance", "stages"):
        require(record.get(key) == protocol.get(key), f"{directory}: record/protocol {key} mismatch")
    require(record.get("guidance") == 4.0, f"{directory}: expected training-aligned native CFG 4")
    require(record.get("model_source", {}).get("model_name") == "Wan2.1-T2V-14B",
            f"{directory}: teacher model identity mismatch")
    require(record.get("shape") == protocol.get("feature_shape") == FEATURE_SHAPE,
            f"{directory}: unexpected feature shape")
    require(record.get("feature_space") == protocol.get("feature_space") == "wan_vae_raw_x0_first3latents",
            f"{directory}: unexpected feature space")
    require(set(record.get("stages", {})) == set(STEPS), f"{directory}: stage coverage mismatch")
    for key in STEPS:
        stage = record["stages"][key]
        require(stage.get("stage") == int(key[-1]) and 0 <= float(stage.get("sigma", -1)) <= 1,
                f"{directory}: invalid stage {key}")
        require(stage.get("denoiser_probe_ordinal") == stage.get("step_index", -2) + 1
                and stage.get("completed_updates") == stage.get("step_index"),
                f"{directory}: invalid denoiser/update mapping {key}")
    noise_hash = record.get("noise_sha256")
    require(isinstance(noise_hash, str) and len(noise_hash) == 64
            and record.get("initial_noise_sha256") == noise_hash,
            f"{directory}: missing or inconsistent paired noise identity")
    artifact_hash = record.get("artifact_sha256", {}).get("features.npz")
    require(artifact_hash and file_hash(directory / "features.npz") == artifact_hash,
            f"{directory}: feature artifact hash mismatch")
    arrays = {}
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(STEPS), f"{directory}: unexpected feature arrays")
        for key in STEPS:
            latent = archive[key]
            require(list(latent.shape) == FEATURE_SHAPE and latent.dtype == np.float32
                    and np.isfinite(latent).all(), f"{directory}: invalid {key} latent")
            require(hashlib.sha256(np.ascontiguousarray(latent).tobytes()).hexdigest()
                    == record.get("latent_sha256", {}).get(key), f"{directory}: latent hash mismatch for {key}")
            arrays[key] = latent
    return record, arrays


def load_groups(root, prompts, seeds, allow_partial=False):
    groups, coverage, shared_protocol = [], [], None
    for prompt in prompts:
        records, arrays, present, prompt_text = {}, {key: [] for key in STEPS}, [], None
        for seed in seeds:
            directory = root / "wan14b" / f"prompt_{prompt:06d}" / f"seed_{seed:06d}"
            if not (directory / "record.json").is_file() or not (directory / "features.npz").is_file():
                continue
            record, features = load_case(directory, prompt, seed)
            common = {key: value for key, value in record["protocol"].items()
                      if key not in ("prompt_index", "prompt", "seed")}
            if shared_protocol is None:
                shared_protocol = common
            require(common == shared_protocol, f"{directory}: mixed teacher sampling protocols")
            if prompt_text is None:
                prompt_text = record["prompt"]
            require(record["prompt"] == prompt_text, f"{directory}: mixed prompt text")
            present.append(seed)
            records[seed] = record
            for key in STEPS:
                arrays[key].append(features[key])
        missing = sorted(set(seeds) - set(present))
        coverage.append(dict(prompt_index=prompt, completed=len(present), expected=len(seeds), missing_seeds=missing))
        if missing and not allow_partial:
            raise ValueError(f"Prompt {prompt}: missing seeds {missing}")
        groups.append(dict(prompt_index=prompt, prompt=prompt_text, seeds=present, records=records,
                           arrays={key: np.stack(value) for key, value in arrays.items()} if present else {}))
        print(f"VALIDATED teacher steps prompt={prompt} {len(present)}/{len(seeds)}", flush=True)
    return groups, coverage, shared_protocol


def bootstrap_mean(values, draws, random_seed=BOOTSTRAP_SEED):
    if not draws:
        return None
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(values), (draws, len(values)))
    means = values[sampled].mean(axis=1)
    return dict(ci95=np.quantile(means, [0.025, 0.975]).tolist(), draws=draws, seed=random_seed)


def compare_steps(arrays, seeds, bootstrap_draws=2000, pilot_seeds=tuple(range(8))):
    n = len(seeds)
    require(n >= 2 and set(arrays) == set(STEPS), "Need at least two complete paired noises for all stages")
    require(len({arrays[key].shape for key in STEPS}) == 1 and len(arrays[STEPS[0]]) == n,
            "All stages must have equal feature shape and paired seed count")
    dimension = int(np.prod(arrays[STEPS[0]].shape[1:]))
    pooled = gram(np.concatenate([arrays[key] for key in STEPS]))
    blocks = {key: list(range(index * n, (index + 1) * n)) for index, key in enumerate(STEPS)}
    pilot_complete = set(pilot_seeds).issubset(seeds)
    ell2 = None
    if pilot_complete:
        selected = [blocks[key][seeds.index(seed)] for key in STEPS for seed in pilot_seeds]
        distances = squared_distances(pooled[np.ix_(selected, selected)])
        median = float(np.median(distances[np.triu_indices(len(selected), k=1)]))
        ell2 = median / 2 if median > 0 else None
    result = dict(n_seeds=n, seeds=list(seeds), dimension=dimension, steps={}, pairs={},
                  bandwidth=dict(squared=ell2, pilot_complete=pilot_complete, pilot_seeds=list(pilot_seeds),
                                 pilot_stages=list(STEPS), rule="pooled three-stage pilot median off-diagonal squared distance / 2",
                                 distance="raw latent squared Euclidean distance / dimension"))
    for key in STEPS:
        moments = moments_from_gram(pooled[np.ix_(blocks[key], blocks[key])])
        moments["total_variance_trace_unbiased"] = dimension * moments["mean_coordinate_variance_unbiased"]
        moments["cross_noise_pairwise_rms"] = math.sqrt(2 * moments["mean_coordinate_variance_unbiased"])
        result["steps"][key] = moments
    for earlier, later in PAIRS:
        index = blocks[earlier] + blocks[later]
        g = pooled[np.ix_(index, index)]
        aa, bb, ab = g[:n, :n], g[n:, n:], g[:n, n:]
        delta_gram = aa + bb - ab - ab.T
        paired_mse = np.maximum(np.diag(delta_gram), 0)
        mean_drift = max(0.0, float(delta_gram.mean()))
        mean_paired = float(paired_mse.mean())
        centered_population = max(0.0, mean_paired - mean_drift)
        earlier_variance = result["steps"][earlier]["mean_coordinate_variance_unbiased"]
        later_variance = result["steps"][later]["mean_coordinate_variance_unbiased"]
        centered_cross_covariance = float((np.trace(ab) - n * ab.mean()) / (n - 1))
        rho = centered_cross_covariance / math.sqrt(earlier_variance * later_variance) if earlier_variance > 0 and later_variance > 0 else None
        if rho is not None:
            require(abs(rho) <= 1 + 1e-9, "Centered trace correlation outside numerical tolerance")
            rho = float(np.clip(rho, -1, 1))
        distance = dict(mean=mean_paired, median=float(np.median(paired_mse)),
                        p05=float(np.quantile(paired_mse, 0.05)), p95=float(np.quantile(paired_mse, 0.95)),
                        rms=math.sqrt(mean_paired), mean_per_sample_rmse=float(np.sqrt(paired_mse).mean()),
                        per_seed_mse=paired_mse.tolist(),
                        mean_bootstrap=bootstrap_mean(paired_mse, bootstrap_draws))
        pair = dict(earlier=earlier, later=later, numerator=later, denominator=earlier,
                    variance_ratio=variance_ratio(result["steps"][later]["mean_coordinate_variance_unbiased"], earlier_variance),
                    mean_difference_mse=mean_drift, mean_difference_rms=math.sqrt(mean_drift),
                    paired_distance=distance, centered_delta_population_mse=centered_population,
                    centered_delta_unbiased_mse=centered_population * n / (n - 1),
                    centered_cross_covariance_trace_per_dimension=centered_cross_covariance,
                    centered_trace_correlation=rho,
                    paired_mse_over_earlier_variance=mean_paired / earlier_variance if earlier_variance > 0 else None,
                    decomposition_residual=mean_paired - mean_drift - centered_population,
                    **mmd_from_gram(g, n, ell2))
        if bootstrap_draws:
            pair["variance_ratio_bootstrap"] = bootstrap_variance_ratio(bb, aa, bootstrap_draws,
                                                                         random_seed=BOOTSTRAP_SEED)
        result["pairs"][f"{earlier}_to_{later}"] = pair
    return result


def build_previews(root, groups, stages):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    manifest = dict(schema_version="teacher_steps_previews_v1", decoded_frame_index=4, prompts={})
    for group in groups:
        prompt, rows = group["prompt_index"], []
        for seed in group["seeds"]:
            if seed >= 8:
                continue
            source = root / "wan14b" / f"prompt_{prompt:06d}" / f"seed_{seed:06d}"
            record, images = group["records"][seed], {}
            for key in STEPS:
                path = source / f"preview_{key}.jpg"
                require(path.is_file() and file_hash(path) == record["artifact_sha256"].get(path.name),
                        f"Missing or changed preview: {path}")
                images[key] = dict(file=str(path.relative_to(root)), sha256=file_hash(path))
            rows.append(dict(seed=seed, stages=images))
        entry = dict(rows=rows)
        if rows:
            width, height, gap, title, row_header = 416, 240, 12, 82, 28
            sheet = Image.new("RGB", (3 * (width + gap) + gap, title + len(rows) * (height + row_header + gap) + gap), "#eef2f6")
            draw = ImageDraw.Draw(sheet)
            draw.text((gap, 7), f"Wan14B CFG4 | prompt {prompt} | decoded frame index 4", fill="#172334", font=font)
            for column, key in enumerate(STEPS):
                draw.text((gap + column * (width + gap), 42), f"{key} | sigma={stages[key]['sigma']:.8f}", fill="#172334", font=font)
            for row_index, row in enumerate(rows):
                y = title + gap + row_index * (height + row_header + gap)
                for column, key in enumerate(STEPS):
                    x = gap + column * (width + gap)
                    draw.text((x, y), f"seed {row['seed']}", fill="#172334", font=font)
                    with Image.open(root / row["stages"][key]["file"]) as image:
                        image = image.convert("RGB")
                        image.thumbnail((width, height))
                        sheet.paste(image, (x + (width - image.width) // 2, y + row_header + (height - image.height) // 2))
            path = root / "comparisons" / f"prompt_{prompt:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, quality=90)
            entry["contact_sheet"] = str(path.relative_to(root))
        manifest["prompts"][str(prompt)] = entry
    (root / "comparisons").mkdir(exist_ok=True)
    write_json(root / "comparisons/manifest.json", manifest)
    return manifest


def number(value, digits=5):
    return "—" if value is None else f"{value:.{digits}g}"


def interval(value):
    ci = (value or {}).get("ci95")
    return f"[{number(ci[0], 4)}, {number(ci[1], 4)}]" if ci else "—"


def gallery(entry, stages):
    if not entry.get("rows"):
        return "<p>预览尚未生成。</p>"
    result = [f'<p><a href="{html.escape(entry["contact_sheet"], quote=True)}">打开完整同 seed 对照图</a></p>']
    result.append('<div class="images">' + "".join(f'<strong>Step {key[-1]} · σ={number(stages[key]["sigma"], 8)}</strong>' for key in STEPS) + '</div>')
    for row in entry["rows"]:
        result.append(f'<p class="seed">seed {row["seed"]}</p><div class="images">')
        for key in STEPS:
            path = html.escape(row["stages"][key]["file"], quote=True)
            result.append(f'<a href="{path}"><img src="{path}" alt="seed {row["seed"]}, {key}" loading="lazy"></a>')
        result.append('</div>')
    return "".join(result)


def report_html(summary, previews=None):
    protocol, stages = summary.get("protocol") or {}, summary.get("stages") or {}
    mode = protocol.get("stage_mode", "unknown")
    mode_note = ("使用 AR 四步 σ 网格驱动 Wan 14B 做 Euler 更新，观察第 2、3、4 次 denoiser 的预测 x₀。"
                 "这是原始 teacher 的四步诊断轨迹；该 teacher 没有经过四步蒸馏。它与上轮 48-step 原生轨迹在相同 σ 下的 x₀ 不相同。"
                 if mode == "ar_grid" else "沿所声明的 Wan 原生采样轨迹观察三个 x₀ 快照；步号及物理 σ 见下表。")
    stage_rows = "".join(f'<tr><td>Step {key[-1]}</td><td>{number(stages[key]["sigma"], 9)}</td><td>{stages[key]["completed_updates"]}</td><td>{stages[key]["denoiser_probe_ordinal"]}</td></tr>' for key in STEPS) if stages else ""
    sections = []
    for item in summary["prompts"]:
        prompt = item["prompt_index"]
        if item["n_seeds"] < 2:
            sections.append(f'<section><h2>{PROMPT_LABELS.get(prompt,"Prompt")} · #{prompt}</h2><p>等待至少两个完整 noise。</p></section>')
            continue
        moment_rows = "".join(f'<tr><td>Step {key[-1]}</td><td><strong>{number(item["steps"][key]["mean_coordinate_variance_unbiased"])}</strong></td><td>{number(item["steps"][key]["total_variance_trace_unbiased"], 8)}</td><td>{number(item["steps"][key]["mean_tensor_rms"])}</td></tr>' for key in STEPS)
        pair_rows, extra_rows = [], []
        for earlier, later in PAIRS:
            pair = item["pairs"][f"{earlier}_to_{later}"]
            d = pair["paired_distance"]
            label = f'{earlier[-1]} → {later[-1]}'
            pair_rows.append(f'<tr><td>{label}</td><td>{number(pair["variance_ratio"])}<small>95% CI {interval(pair.get("variance_ratio_bootstrap"))}</small></td><td><strong>{number(d["rms"])}</strong><small>平均 MSE {number(d["mean"])}</small></td><td>{number(pair["mean_difference_mse"])}</td><td>{number(pair["mmd2_paired_u"])}</td></tr>')
            extra_rows.append(f'<tr><td>{label}</td><td>{number(d["median"])}<small>P05–P95 [{number(d["p05"])}, {number(d["p95"])}]</small></td><td>{interval(d.get("mean_bootstrap"))}</td><td>{number(pair["centered_delta_population_mse"])}</td><td>{number(pair["paired_mse_over_earlier_variance"])}</td><td>{number(pair["centered_trace_correlation"])}</td></tr>')
        first, second = item["pairs"]["step_2_to_step_3"]["paired_distance"]["mean"], item["pairs"]["step_3_to_step_4"]["paired_distance"]["mean"]
        change_ratio = second / first if first > 0 else None
        note = f'相邻步同 noise 的平均 MSE：2→3 为 {number(first)}，3→4 为 {number(second)}；后者 / 前者 = {number(change_ratio)}。不同相邻步的 Δσ 不同，这个比例描述实际更新差距，不能单独证明轨迹已收敛。'
        entry = (previews or {}).get("prompts", {}).get(str(prompt), {})
        sections.append(f'''<section><h2>{PROMPT_LABELS.get(prompt,"Prompt")} · #{prompt}</h2><p>{html.escape(item.get("prompt") or "")}</p><small>{item["n_seeds"]} 个共同 noise；D={item["dimension"]:,}。</small>
<h3>每一步的跨 noise 方差</h3><div class="table"><table><tr><th>输出</th><th>整体方差 / D</th><th>整体总方差 V</th><th>平均张量 RMS</th></tr>{moment_rows}</table></div>
<h3>步间相差多少</h3><div class="table"><table><tr><th>步间比较</th><th>方差比：后 / 前</th><th>同 noise 样本距离 RMS</th><th>均值之差 MSE</th><th>RBF MMD²</th></tr>{"".join(pair_rows)}</table></div><p>{note}</p>
<details><summary>距离分位数、置信区间及变化分解</summary><div class="table"><table><tr><th>比较</th><th>同 noise MSE 中位数及范围</th><th>平均 MSE 的 95% CI</th><th>去除平均位移后的变化 MSE</th><th>平均 MSE / 前一步跨 noise 方差</th><th>去均值后的 trace 相关性 ρ</th></tr>{"".join(extra_rows)}</table></div></details>
<details><summary>同 noise 的三个 step 图片（前 8 个 noise）</summary><small>显示相同 VAE 解码 frame index 4；指标使用前三个完整 latent 帧。</small>{gallery(entry,stages)}</details>
<small>本 prompt 的三个 MMD 共用同一个带宽 ℓ²={number(item["bandwidth"]["squared"], 8)}，来自三个 stage 的 pilot seeds 0–7。</small></section>''')
    done, total = sum(row["completed"] for row in summary["completeness"]), sum(row["expected"] for row in summary["completeness"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan 14B · Step 2 / 3 / 4</title><style>
body{{margin:0;background:#f3f6f9;color:#172334;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:28px 20px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:0}}h3{{font-size:17px;margin-bottom:6px}}section{{margin:20px 0;padding:24px;background:white;border:1px solid #dce4ed;border-radius:12px}}p{{margin:10px 0}}small{{display:block;color:#5c6d80;font-size:13px}}table{{width:100%;border-collapse:collapse;text-align:left}}td,th{{padding:11px;border-bottom:1px solid #e2e8ef;vertical-align:top}}th{{background:#f1f5f9}}.table{{overflow:auto;margin:14px 0}}a{{color:#1764af}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.images img{{width:100%;border-radius:5px}}.images strong,.seed{{font-size:13px}}@media(max-width:700px){{main{{padding:15px 10px}}section{{padding:14px}}td,th{{font-size:13px;padding:9px}}}}
</style><main><h1>Wan 14B：Step 2 / 3 / 4 的分布与变化</h1><p>{"完整结果" if summary["complete"] else "临时结果 · 样本尚未齐全"} · {done}/{total} 个 noise 轨迹 · {done*3} 个 x₀ 快照 · CFG=4</p>
<p>{mode_note}</p><p>使用训练相同的 Wan 14B 权重、文本编码器和负向提示词；每个 noise 沿同一条轨迹得到三个输出。全部指标使用同一个原始 Wan VAE latent 空间，不逐样本归一化。</p>
<details><summary>步号与 noise level</summary><div class="table"><table><tr><th>输出</th><th>σ</th><th>此前已完成 Euler 更新</th><th>denoiser 调用序号</th></tr>{stage_rows}</table></div></details>{"".join(sections)}
<details><summary>指标定义与结论范围</summary><p><strong>整体方差：</strong>每个 noise 的完整 latent 展平为 xᵢ，μ=(1/N)Σxᵢ，V=Σ‖xᵢ−μ‖²/(N−1)=tr(Σ̂)。V/D 为同一个跨 noise 离散度的按维度平均，不是单个 latent 内部的方差。方差比为后一步 / 前一步。</p>
<p><strong>同 noise 距离：</strong>dᵢ=‖xᵢ,后−xᵢ,前‖²/D；表中 RMS=√mean(dᵢ)，均值差 MSE=‖μ后−μ前‖²/D。mean(dᵢ)=均值差 MSE + 去均值后的变化 MSE；后者使用经验平均（ddof=0），JSON 另存 ddof=1 版本。距离 / 前一步跨 noise 方差的分母为 V前/D（ddof=1）；该无量纲量不表示多样性的百分比。</p>
<p><strong>MMD²：</strong>RBF paired U 估计，剔除同 noise 的跨 stage 对角项；有限样本值可略小于零。三个配对使用同一个 pooled-stage pilot 带宽。MMD 衡量边缘分布差异，同 noise 距离衡量轨迹中的样本移动；两者回答不同问题。方差比及平均距离的 CI 使用 {summary["bootstrap_draws"]} 次配对 seed bootstrap。</p>
<p><strong>跨 step 相关性：</strong>ρ=tr(Cov(x前,x后))/√(V前V后)，先在每个 step 对全部 noise 去均值，再比较同 seed 的完整 latent 偏移；它不是单张图内部的相关性。ρ 接近 1 表示不同 noise 相对各自均值的变化方向更相似，不能单独等同于语义结构固定。</p>
<p>这些结果只描述当前两个 prompt 的早期 x₀ latent，且 step 间 σ 在变化；不能直接代表最终视频的语义 diversity，或证明某一步之后场景结构已完全固定。</p></details>
<p><a href="summary.json">完整 JSON</a> · <a href="moments.csv">各 step 方差和均值</a> · <a href="pairwise.csv">步间指标</a> · <a href="paired_distances.csv">逐 noise 步间距离</a> · <a href="mean_tensors.npz">完整均值张量</a></p></main></html>'''


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(root, summary, means, previews):
    moments, pairs, distances = [], [], []
    for item in summary["prompts"]:
        if item["n_seeds"] < 2:
            continue
        for key in STEPS:
            moments.append(dict(prompt_index=item["prompt_index"], stage=key, sigma=summary["stages"][key]["sigma"],
                                n_seeds=item["n_seeds"], **item["steps"][key]))
        for key, pair in item["pairs"].items():
            d, ratio_ci = pair["paired_distance"], pair.get("variance_ratio_bootstrap", {}).get("ci95")
            distance_ci = (d.get("mean_bootstrap") or {}).get("ci95")
            pairs.append(dict(prompt_index=item["prompt_index"], pair=key, variance_ratio_later_over_earlier=pair["variance_ratio"],
                              variance_ratio_ci95_low=ratio_ci[0] if ratio_ci else None, variance_ratio_ci95_high=ratio_ci[1] if ratio_ci else None,
                              mean_difference_mse=pair["mean_difference_mse"], mmd2_paired_u=pair["mmd2_paired_u"],
                              paired_mse_mean=d["mean"], paired_rms=d["rms"], paired_mse_median=d["median"],
                              paired_mse_p05=d["p05"], paired_mse_p95=d["p95"],
                              paired_mse_mean_ci95_low=distance_ci[0] if distance_ci else None,
                              paired_mse_mean_ci95_high=distance_ci[1] if distance_ci else None,
                              centered_delta_population_mse=pair["centered_delta_population_mse"],
                              centered_trace_correlation=pair["centered_trace_correlation"],
                              paired_mse_over_earlier_variance=pair["paired_mse_over_earlier_variance"]))
            for seed, mse in zip(item["seeds"], d["per_seed_mse"]):
                distances.append(dict(prompt_index=item["prompt_index"], pair=key, seed=seed, mse=mse, rmse=math.sqrt(mse)))
    write_csv(root / "moments.csv", moments)
    write_csv(root / "pairwise.csv", pairs)
    write_csv(root / "paired_distances.csv", distances)
    np.savez_compressed(root / "mean_tensors.npz", **means)
    write_json(root / "summary.json", summary)
    (root / "report.html").write_text(report_html(summary, previews))


def self_test():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(16, 3, 2))
    same = compare_steps({key: a.copy() for key in STEPS}, list(range(16)), 100)
    for pair in same["pairs"].values():
        assert abs(pair["variance_ratio"] - 1) < 1e-12
        assert pair["paired_distance"]["mean"] < 1e-12
        assert abs(pair["mmd2_paired_u"]) < 1e-12
    shifted = compare_steps(dict(step_2=a, step_3=a + 2, step_4=2 * a + 2), list(range(16)), 100)
    pair = shifted["pairs"]["step_2_to_step_3"]
    np.testing.assert_allclose(pair["paired_distance"]["per_seed_mse"], 4, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pair["mean_difference_mse"], 4, rtol=1e-12)
    assert pair["centered_delta_population_mse"] < 1e-12
    np.testing.assert_allclose(pair["centered_trace_correlation"], 1, rtol=1e-12)
    assert abs(pair["variance_ratio"] - 1) < 1e-12 and pair["mmd2_paired_u"] > 0
    for earlier, later in PAIRS:
        values = dict(step_2=a, step_3=a + 2, step_4=2 * a + 2)
        pair = shifted["pairs"][f"{earlier}_to_{later}"]
        delta = values[later] - values[earlier]
        direct = np.mean(delta ** 2, axis=(1, 2))
        np.testing.assert_allclose(pair["paired_distance"]["per_seed_mse"], direct, rtol=1e-12, atol=1e-12)
        centered = delta - delta.mean(axis=0)
        np.testing.assert_allclose(pair["centered_delta_population_mse"], np.mean(centered ** 2), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pair["paired_distance"]["mean"], pair["mean_difference_mse"] + pair["centered_delta_population_mse"], rtol=1e-12)
    np.testing.assert_allclose(shifted["pairs"]["step_2_to_step_4"]["variance_ratio"], 4, rtol=1e-12)
    permuted = a[rng.permutation(len(a))]
    marginal_same = compare_steps(dict(step_2=a, step_3=permuted, step_4=a), list(range(16)), 0)
    assert marginal_same["pairs"]["step_2_to_step_3"]["paired_distance"]["mean"] > 0
    assert marginal_same["pairs"]["step_2_to_step_3"]["mmd2_v_descriptive"] < 1e-12
    for key in STEPS:
        flat = dict(step_2=a, step_3=a+2, step_4=2*a+2)[key].reshape(16,-1)
        np.testing.assert_allclose(shifted["steps"][key]["total_variance_trace_unbiased"], np.var(flat, axis=0, ddof=1).sum(), rtol=1e-12)
    print("Teacher-stage same/shifted/scaled marginals, direct paired distances, variance axis and mean/centered decomposition self-tests passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-indices", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.output_dir is None or args.bootstrap_draws < 0:
        parser.error("Need --output-dir and nonnegative --bootstrap-draws")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summary = json.loads((root / "summary.json").read_text())
        require(summary.get("schema_version") == SCHEMA, "Unknown summary schema")
        previews = json.loads((root / "comparisons/manifest.json").read_text()) if (root / "comparisons/manifest.json").is_file() else None
        (root / "report.html").write_text(report_html(summary, previews))
        print(root / "report.html")
        return
    plan = json.loads((root / "experiment.json").read_text()) if (root / "experiment.json").is_file() else {}
    prompts = args.prompt_indices or plan.get("prompt_indices", [42,300])
    seeds = args.seeds or plan.get("seeds", list(range(64)))
    require(len(set(prompts)) == len(prompts) and len(set(seeds)) == len(seeds) and len(seeds) >= 2,
            "Prompt indices and seeds must be distinct, with at least two seeds")
    groups, coverage, protocol = load_groups(root, prompts, seeds, args.allow_partial)
    if protocol and plan.get("stage_mode"):
        require(protocol.get("stage_mode") == plan["stage_mode"], "Sampling stage mode differs from experiment plan")
    summary = dict(schema_version=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(),
                   complete=all(not row["missing_seeds"] for row in coverage), completeness=coverage,
                   expected_seeds=seeds, protocol=protocol, stages=(protocol or {}).get("stages", {}),
                   bootstrap_draws=args.bootstrap_draws, prompts=[])
    means = {}
    for group in groups:
        item = dict(prompt_index=group["prompt_index"], prompt=group["prompt"], n_seeds=len(group["seeds"]), seeds=group["seeds"])
        if len(group["seeds"]) >= 2:
            item.update(compare_steps(group["arrays"], group["seeds"], args.bootstrap_draws))
            for key in STEPS:
                means[f"prompt_{group['prompt_index']:06d}_{key}"] = group["arrays"][key].mean(axis=0, dtype=np.float64)
            print(f"SUMMARIZED teacher steps prompt={group['prompt_index']} noises={len(group['seeds'])}", flush=True)
        summary["prompts"].append(item)
    previews = None if args.skip_previews else build_previews(root, groups, summary["stages"])
    write_outputs(root, summary, means, previews)
    print(json.dumps(dict(complete=summary["complete"], report=str(root / "report.html")), ensure_ascii=False))


if __name__ == "__main__":
    main()
