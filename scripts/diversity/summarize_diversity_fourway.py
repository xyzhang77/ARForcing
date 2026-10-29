#!/usr/bin/env python3
"""Compare Causal Forcing init, ODE init, AR, and Wan at a matched sigma.

The official chunkwise causal_ode initialization uses the same causal sampler
as AR; its protocol and re-noising draws are checked explicitly.
Existing two- and three-model reports are only read and remain unchanged.
"""
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
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diversity.summarize_diversity_moments import (
    PROMPT_LABELS, bootstrap_variance_ratio, format_number, gram,
    mmd_from_gram, moments_from_gram, pilot_bandwidth_squared, variance_ratio,
)
from scripts.diversity.summarize_diversity_threeway import (
    FEATURE_SPACE, canonical_hash, checked_case, model_directory,
    validate_causal_protocols,
)

MODELS = ("cf_init", "ode", "ar", "wan")
LABELS = {"cf_init": "Causal Forcing init", "ode": "Self-Forcing ODE init", "ar": "AR", "wan": "Wan bidirectional"}
PAIRS = tuple(combinations(MODELS, 2))


def cf_directory(root):
    root = Path(root)
    return root if root.name == "cf_init" else root / "cf_init"


def load_fourway(baseline_root, ode_root, cf_root, prompts, seeds, sigma=.9375,
                 allow_partial=False, ode_checkpoint=None, cf_checkpoint=None):
    """Load each feature once; checkpoint path resolution is cached per source."""
    ode_path, ode_label = model_directory(ode_root, "ode")
    locations = {"cf_init": (cf_directory(cf_root), "cf_init"), "ode": (ode_path, ode_label),
                 "ar": (Path(baseline_root) / "ar", "ar"), "wan": (Path(baseline_root) / "wan", "wan")}
    if ode_label == "ar" and ode_checkpoint is None:
        raise ValueError("ODE artifacts labeled ar require an explicit ODE checkpoint")
    expected_paths = {name: Path(path).resolve() for name, path in
                      (("ode", ode_checkpoint), ("cf_init", cf_checkpoint)) if path is not None}
    resolved_sources, common_protocols, sources = {}, {}, {}
    groups, coverage, complete, shape = [], [], True, None
    started = time.monotonic()
    for prompt_index in prompts:
        cases = {name: {} for name in MODELS}
        prompt_text = None
        for name in MODELS:
            model_root, record_label = locations[name]
            for seed in seeds:
                directory = model_root / f"prompt_{prompt_index:06d}" / f"seed_{seed:06d}"
                if not (directory / "record.json").is_file() or not (directory / "features.npz").is_file():
                    continue
                case = checked_case(directory, record_label, prompt_index, seed, sigma)
                record = case["record"]
                if prompt_text is None:
                    prompt_text = record["prompt"]
                if record["prompt"] != prompt_text:
                    raise ValueError(f"{directory}: prompt text differs across models/seeds")
                if shape is None:
                    shape = case["latent"].shape
                if case["latent"].shape != shape:
                    raise ValueError(f"{directory}: feature shape differs across models/seeds")
                if name == "cf_init":
                    # Exported x0 must be probed at exactly the requested sigma,
                    # rather than at a nearby schedule point.
                    if record["sigma"] != sigma or record["protocol"]["sigma"] != sigma:
                        raise ValueError(f"{directory}: CF probe sigma must equal target exactly")
                    for key in ("native_probe_sigma", "target_probe_sigma"):
                        for owner in (record, record["protocol"]):
                            if key in owner and owner[key] != sigma:
                                raise ValueError(f"{directory}: {key} differs from target sigma")
                common = {k: v for k, v in record["protocol"].items()
                          if k not in ("prompt_index", "prompt", "seed")}
                if name in common_protocols and common_protocols[name] != common:
                    raise ValueError(f"{directory}: mixed sampling protocols for {name}")
                common_protocols[name], sources[name] = common, record["model_source"]
                if name in expected_paths:
                    actual = record["model_source"].get("checkpoint_path", "")
                    if not actual:
                        raise ValueError(f"{directory}: missing checkpoint path")
                    if actual not in resolved_sources:
                        resolved_sources[actual] = Path(actual).resolve()
                    if resolved_sources[actual] != expected_paths[name]:
                        raise ValueError(f"{directory}: {name} checkpoint identity differs")
                cases[name][seed] = case
            missing = sorted(set(seeds) - cases[name].keys())
            coverage.append(dict(model=name, prompt_index=prompt_index, completed=len(cases[name]),
                                 expected=len(seeds), missing_seeds=missing))
            complete = complete and not missing
            print(f"VALIDATED {name} prompt={prompt_index} {len(cases[name])}/{len(seeds)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        paired = sorted(set.intersection(*(set(cases[name]) for name in MODELS)))
        for seed in paired:
            if len({cases[name][seed]["noise_sha256"] for name in MODELS}) != 1:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: initial noise differs across models")
            ar = cases["ar"][seed]["record"]
            for name in ("ode", "cf_init"):
                other = cases[name][seed]["record"]
                if not ar.get("transition_noise_sha256") or ar["transition_noise_sha256"] != other.get("transition_noise_sha256"):
                    raise ValueError(f"Prompt {prompt_index}, seed {seed}: {name}/AR transition noise differs")
                if ar.get("timesteps") is not None and other.get("timesteps") is not None and ar["timesteps"] != other["timesteps"]:
                    raise ValueError(f"Prompt {prompt_index}, seed {seed}: {name}/AR timesteps differ")
        arrays = {name: np.stack([cases[name][seed]["latent"] for seed in paired])
                  for name in MODELS} if paired else {}
        groups.append(dict(prompt_index=prompt_index, prompt=prompt_text, seeds=paired, arrays=arrays))
    for name in ("ode", "cf_init"):
        if name in common_protocols and "ar" in common_protocols:
            try:
                validate_causal_protocols(common_protocols[name], common_protocols["ar"])
            except ValueError as error:
                raise ValueError(f"{name}/AR protocol mismatch: {error}") from error
    if not complete and not allow_partial:
        raise ValueError("Incomplete four-model samples; use --allow-partial for a progress report")
    return groups, coverage, complete, shape, sources, common_protocols


def compare_fourway(arrays, seeds, pilot_seeds=tuple(range(8)), bootstrap_draws=2000):
    n = len(seeds)
    if n < 2 or any(len(arrays[name]) != n for name in MODELS):
        raise ValueError("Need at least two paired samples from every model")
    if len({arrays[name].shape for name in MODELS}) != 1:
        raise ValueError("All model arrays must have identical shape")
    dimension = int(np.prod(arrays["ar"].shape[1:]))
    g = gram(np.concatenate([arrays[name] for name in MODELS]))
    blocks = {name: list(range(i*n, (i+1)*n)) for i, name in enumerate(MODELS)}
    old_pair = blocks["ar"] + blocks["wan"]
    pilot_complete = set(pilot_seeds).issubset(seeds)
    ell2 = pilot_bandwidth_squared(g[np.ix_(old_pair, old_pair)], n,
                                  [seeds.index(seed) for seed in pilot_seeds]) if pilot_complete else None
    result = dict(n_seeds=n, seeds=list(seeds), dimension=dimension, models={}, pairs={},
                  bandwidth=dict(squared=ell2, pilot_complete=pilot_complete,
                                 pilot_seeds=list(pilot_seeds), pilot_models=["ar", "wan"],
                                 rule="pooled AR/Wan pilot median off-diagonal squared distance / 2",
                                 distance="raw latent squared Euclidean distance / dimension"))
    for name in MODELS:
        moments = moments_from_gram(g[np.ix_(blocks[name], blocks[name])])
        moments["total_variance_trace_unbiased"] = dimension * moments["mean_coordinate_variance_unbiased"]
        result["models"][name] = moments
    for a, b in PAIRS:
        indices = blocks[a] + blocks[b]
        pair_g = g[np.ix_(indices, indices)]
        aa, bb, ab = pair_g[:n, :n], pair_g[n:, n:], pair_g[:n, n:]
        pair = dict(numerator=a, denominator=b,
                    variance_ratio=variance_ratio(result["models"][a]["mean_coordinate_variance_unbiased"],
                                                  result["models"][b]["mean_coordinate_variance_unbiased"]),
                    mean_difference_mse=max(0.0, float(aa.mean()+bb.mean()-2*ab.mean())),
                    **mmd_from_gram(pair_g, n, ell2))
        if bootstrap_draws:
            pair["variance_ratio_bootstrap"] = bootstrap_variance_ratio(aa, bb, bootstrap_draws)
        result["pairs"][f"{a}_vs_{b}"] = pair
    return result


def build_comparisons(output_dir, baseline_root, ode_root, cf_root, prompts, seeds, sigma=.9375):
    """Copy hash-checked previews and make a four-column contact sheet."""
    from PIL import Image, ImageDraw, ImageFont

    roots = {"cf_init": cf_directory(cf_root), "ode": model_directory(ode_root, "ode")[0],
             "ar": Path(baseline_root)/"ar", "wan": Path(baseline_root)/"wan"}
    manifest = dict(schema_version="diversity_fourway_previews_v1", models=list(MODELS),
                    sigma=sigma, decoded_frame_index=4, prompts={})
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",20)
    except OSError:
        font = ImageFont.load_default()
    destination_root = output_dir / "comparisons"
    destination_root.mkdir(parents=True, exist_ok=True)
    for prompt_index in prompts:
        rows = []
        for seed in seeds[:8]:
            previews = {}
            for name in MODELS:
                case_root = roots[name] / f"prompt_{prompt_index:06d}" / f"seed_{seed:06d}"
                source = case_root / "preview.jpg"
                record_path = case_root / "record.json"
                if not source.is_file() or not record_path.is_file():
                    continue
                record = json.loads(record_path.read_text())
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if record.get("artifact_sha256", {}).get("preview.jpg") != digest:
                    raise ValueError(f"{source}: preview SHA-256 mismatch")
                if record.get("prompt_index") != prompt_index or record.get("seed") != seed:
                    raise ValueError(f"{source}: preview record identity mismatch")
                target = destination_root / f"prompt_{prompt_index:06d}" / name / f"seed_{seed:06d}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                previews[name] = dict(file=str(target.relative_to(output_dir)), source=str(source), sha256=digest)
            if previews:
                rows.append(dict(seed=seed, models=previews))
        entry = dict(rows=rows, model_order=list(MODELS))
        if rows:
            width, height, gutter, header, title_height = 416, 240, 12, 28, 68
            sheet = Image.new("RGB", (len(MODELS)*(width+gutter)+gutter,
                                      title_height+len(rows)*(height+header+gutter)+gutter), "#eef2f6")
            draw = ImageDraw.Draw(sheet)
            draw.text((gutter,7),f"prompt {prompt_index} | sigma={sigma} | decoded frame index 4 | same seed in each row",fill="#172334",font=font)
            for col, name in enumerate(MODELS):
                draw.text((gutter+col*(width+gutter), 38), LABELS[name], fill="#172334",font=font)
            for row_index, row in enumerate(rows):
                y = title_height+gutter+row_index*(height+header+gutter)
                for col, name in enumerate(MODELS):
                    x = gutter+col*(width+gutter)
                    draw.text((x, y), f"seed {row['seed']}", fill="#172334",font=font)
                    preview = row["models"].get(name)
                    if preview:
                        with Image.open(output_dir/preview["file"]) as im:
                            im = im.convert("RGB")
                            im.thumbnail((width, height))
                            sheet.paste(im, (x+(width-im.width)//2, y+header+(height-im.height)//2))
            path = destination_root/f"comparison_prompt_{prompt_index:06d}.jpg"
            sheet.save(path, quality=90)
            entry["contact_sheet"] = str(path.relative_to(output_dir))
        manifest["prompts"][str(prompt_index)] = entry
    (destination_root/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def gallery_html(prompt_index, comparisons):
    entry = (comparisons or {}).get("prompts", {}).get(str(prompt_index), {})
    rows = entry.get("rows", [])
    if not rows:
        return "<small>尚无四模型预览。指标只依赖 latent，不依赖 JPEG。</small>"
    columns = "".join(f"<strong>{LABELS[name]}</strong>" for name in MODELS)
    result = [f'<details class="gallery"><summary>看同 seed 图像（前 {len(rows)} 个 noise）</summary>']
    result.append(f'<small>预览：σ = {format_number((comparisons or {}).get("sigma"),6)}，解码 frame index 4；数值指标使用完整前三个 latent 帧。</small>')
    if entry.get("contact_sheet"):
        result.append(f'<p><a href="{html.escape(entry["contact_sheet"], quote=True)}" target="_blank">打开四模型完整对照图</a></p>')
    result.append(f'<div class="model-head">{columns}</div>')
    for row in rows:
        result.append(f'<p class="seed-label">seed {row["seed"]}</p><div class="image-row">')
        for name in MODELS:
            item = row["models"].get(name)
            if item:
                path = html.escape(item["file"], quote=True)
                label = html.escape(f"{LABELS[name]}, prompt {prompt_index}, seed {row['seed']}", quote=True)
                result.append(f'<a href="{path}" target="_blank"><img src="{path}" alt="{label}" loading="lazy"></a>')
            else:
                result.append('<div class="empty-image">预览未生成</div>')
        result.append("</div>")
    result.append("</details>")
    return "".join(result)


def report_html(summary, comparisons=None):
    sections = []
    for item in summary["prompts"]:
        p = item["prompt_index"]
        title = f"{PROMPT_LABELS.get(p, 'Prompt')} · #{p}"
        if item["n_seeds"] < 2:
            sections.append(f"<section><h2>{title}</h2><p>等待至少 2 个四模型共同完成的 noise。</p></section>")
            continue
        rows = "".join(f"<tr><td><strong>{LABELS[name]}</strong></td><td>{format_number(item['models'][name]['mean_tensor_rms'])}</td><td>{format_number(item['models'][name]['mean_coordinate_variance_unbiased'])}</td><td>{format_number(item['models'][name]['total_variance_trace_unbiased'],6)}</td></tr>" for name in MODELS)
        pair_rows = []
        for a, b in PAIRS:
            pair = item["pairs"][f"{a}_vs_{b}"]
            ci = pair.get("variance_ratio_bootstrap", {}).get("ci95")
            ci_text = f"[{format_number(ci[0],3)}, {format_number(ci[1],3)}]" if ci else "—"
            pair_rows.append(f"<tr><td>{LABELS[a]} / {LABELS[b]}</td><td><strong>{format_number(pair['variance_ratio'])}</strong><small>95% CI {ci_text}</small></td><td>{format_number(pair['mean_difference_mse'])}</td><td>{format_number(pair['mmd2_paired_u'])}</td></tr>")
        sections.append(f"""<section><h2>{title}</h2><p>{html.escape(item.get('prompt') or '')}</p>
<p>共同 noise：<strong>{item['n_seeds']}</strong>；维度 D = {item['dimension']:,}。</p>
<div class="table"><table><tr><th>模型</th><th>Mean tensor RMS</th><th>跨 noise 方差 V / D</th><th>整体总方差 V</th></tr>{rows}</table></div>
<div class="table"><table><tr><th>比较（分子 / 分母）</th><th>方差比及 95% CI</th><th>均值之差 MSE</th><th>RBF MMD²</th></tr>{''.join(pair_rows)}</table></div>
<small>六对比较共用原 AR/Wan seed 0–7 确定的带宽 ℓ² = {format_number(item['bandwidth']['squared'],6)}。</small>
{gallery_html(p,comparisons)}</section>""")
    protocol_rows = []
    for name in MODELS:
        protocol = summary.get("sampling_protocols", {}).get(name, {})
        source = summary.get("model_sources", {}).get(name, {})
        details = f"guidance={protocol.get('guidance', '—')}；completed updates={protocol.get('completed_updates', '—')}；initial shape={protocol.get('shape', '—')}"
        schedule = protocol.get("native_schedule_steps", protocol.get("wan_steps"))
        if schedule is not None:
            details += f"；schedule steps={schedule}"
        role = f"；role={source['checkpoint_role']}" if source.get("checkpoint_role") else ""
        protocol_rows.append(f"<tr><td>{LABELS[name]}</td><td>{html.escape(str(protocol.get('sampler','—')))}<small>{html.escape(details)}</small></td><td class='path'>{html.escape(str(source.get('checkpoint_path','—')))}<small>weights={html.escape(str(source.get('weights','—')))}{html.escape(role)}</small></td></tr>")
    completed = sum(row["completed"] for row in summary["completeness"])
    expected = sum(row["expected"] for row in summary["completeness"])
    status = "完整结果" if summary["complete"] else "临时结果 · 样本尚未齐全"
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Causal Forcing init / Self-Forcing ODE init / AR / Wan</title><style>
body{{font:16px/1.65 system-ui,sans-serif;background:#f3f6f9;color:#172334;margin:0}}main{{max-width:1200px;margin:auto;padding:28px 20px}}h1{{font-size:27px;line-height:1.3}}h2{{font-size:21px;margin-top:0}}section{{background:white;border:1px solid #dde4ed;border-radius:12px;margin:22px 0;padding:24px}}p{{margin:10px 0}}.table{{overflow:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;text-align:left}}th,td{{padding:12px;border-bottom:1px solid #e4e9ef;vertical-align:top}}th{{background:#f2f6fa}}small{{display:block;color:#5c6d80;font-size:13px}}a{{color:#1764af}}.status{{color:#3e5974}}.path{{overflow-wrap:anywhere;font-size:13px}}summary{{cursor:pointer;font-weight:600}}.gallery{{margin-top:20px}}.model-head,.image-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}.model-head{{font-size:13px;margin-top:15px}}.image-row img{{display:block;width:100%;border-radius:5px}}.empty-image{{min-height:90px;background:#eff2f6;text-align:center;color:#7a8797}}.seed-label{{font-size:13px;font-weight:600;margin:16px 0 5px}}@media(max-width:700px){{main{{padding:16px 10px}}section{{padding:14px}}th,td{{padding:9px;font-size:13px}}.model-head{{font-size:10px}}}}
</style><main><h1>Causal Forcing init / Self-Forcing ODE init / AR / Wan</h1>
<p class="status">{status} · {completed}/{expected} 个模型样本 · σ = {summary['sigma']}</p>
<p>同 prompt、同初始 noise、同 noise level，比较预测 x₀ 的前三个 latent 帧。使用共同的原始 Wan VAE latent 空间，不对单个样本归一化。</p>
{''.join(sections)}
<section><h2>指标怎么读</h2><p><strong>整体方差 V：</strong>每个 noise 的完整 latent 展平为 xᵢ，平均张量 μ = (1/N) Σᵢ xᵢ；V = [1/(N−1)] Σᵢ ‖xᵢ−μ‖² = tr(Σ̂)。V/D 是同一个整体离散度按维度平均，模型排序相同。方差比 &lt; 1 表示分子模型更集中。</p>
<p><strong>Mean：</strong>RMS = ‖μ‖/√D，只表示平均张量大小；均值之差 MSE = ‖μₐ−μᵦ‖²/D 表示两个平均张量的距离。完整均值见 <a href="mean_tensors.npz">mean_tensors.npz</a>。</p>
<p><strong>MMD²：</strong>RBF 核衡量分布差异，六对共用同一 prompt 的固定带宽。采用 paired U 估计，同 seed 的跨模型 i=j 项也剔除；有限样本估计可以略小于零。MMD 本身不能判断哪个模型更 diverse。方差比区间来自 {summary['bootstrap_draws']} 次配对 seed bootstrap。</p>
<p>以上只描述当前早期 x₀ latent 分布，不能直接代表最终完整视频的语义多样性。各模型的原生采样过程见下表；相同 σ 不表示经历了相同计算步数。</p></section>
<details><summary>Checkpoint 与各模型实际采样协议</summary><div class="table"><table><tr><th>模型</th><th>原生 sampler / 参数</th><th>Checkpoint</th></tr>{''.join(protocol_rows)}</table></div></details>
<p><a href="summary.json">完整 JSON</a> · <a href="moments.csv">各模型 moments</a> · <a href="pairwise.csv">六对分布比较</a> · <a href="https://www.jmlr.org/papers/v13/gretton12a.html">MMD 参考</a></p></main></html>"""


def write_report(output_dir, summary):
    path = output_dir/"comparisons/manifest.json"
    comparisons = json.loads(path.read_text()) if path.is_file() else None
    (output_dir/"report.html").write_text(report_html(summary, comparisons))


def write_outputs(output_dir, summary, means):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/"mean_tensors.npz", **means)
    (output_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    moment_fields = ["prompt_index", "n_seeds", "model", "mean_tensor_rms", "variance_per_dimension", "total_variance_trace", "complete"]
    pair_fields = ["prompt_index", "n_seeds", "numerator", "denominator", "variance_ratio", "ratio_ci95_low", "ratio_ci95_high", "mean_difference_mse", "mmd2_paired_u", "mmd2_v_descriptive", "rbf_bandwidth_squared", "complete"]
    with (output_dir/"moments.csv").open("w", newline="") as moment_stream, (output_dir/"pairwise.csv").open("w", newline="") as pair_stream:
        moment_writer, pair_writer = csv.DictWriter(moment_stream, moment_fields), csv.DictWriter(pair_stream, pair_fields)
        moment_writer.writeheader()
        pair_writer.writeheader()
        for item in summary["prompts"]:
            if item["n_seeds"] < 2:
                continue
            identity = dict(prompt_index=item["prompt_index"], n_seeds=item["n_seeds"], complete=summary["complete"])
            for name in MODELS:
                moments = item["models"][name]
                moment_writer.writerow(dict(identity, model=name, mean_tensor_rms=moments["mean_tensor_rms"],
                                             variance_per_dimension=moments["mean_coordinate_variance_unbiased"],
                                             total_variance_trace=moments["total_variance_trace_unbiased"]))
            for pair in item["pairs"].values():
                ci = pair.get("variance_ratio_bootstrap", {}).get("ci95") or [None,None]
                row = {field:pair[field] for field in ("numerator", "denominator", "variance_ratio", "mean_difference_mse", "mmd2_paired_u", "mmd2_v_descriptive")}
                pair_writer.writerow(dict(identity, **row, ratio_ci95_low=ci[0], ratio_ci95_high=ci[1],
                                          rbf_bandwidth_squared=item["bandwidth"]["squared"]))
    # A conventional entrypoint for tools that expect summary.csv.
    (output_dir/"summary.csv").write_bytes((output_dir/"pairwise.csv").read_bytes())
    write_report(output_dir, summary)


def self_test():
    from scripts.diversity.summarize_diversity_threeway import compare_threeway
    rng = np.random.default_rng(724)
    a = rng.normal(size=(16, 2, 3)).astype(np.float32)
    seeds = list(range(16))
    same = compare_fourway(dict.fromkeys(MODELS, a), seeds, bootstrap_draws=100)
    assert len(same["pairs"]) == 6
    for pair in same["pairs"].values():
        assert abs(pair["mmd2_paired_u"]) < 1e-12
        assert math.isclose(pair["variance_ratio"], 1, abs_tol=1e-12)
        assert np.allclose(pair["variance_ratio_bootstrap"]["ci95"], [1,1])
    arrays = dict(cf_init=a.astype(np.float64)*3, ode=a*2, ar=a, wan=a*.5)
    changed = compare_fourway(arrays, seeds, bootstrap_draws=100)
    assert math.isclose(changed["pairs"]["cf_init_vs_ode"]["variance_ratio"], 9/4, rel_tol=1e-12)
    assert math.isclose(changed["pairs"]["cf_init_vs_ar"]["variance_ratio"], 9, rel_tol=1e-12)
    assert np.allclose(changed["models"]["cf_init"]["total_variance_trace_unbiased"], (a.astype(np.float64)*3).var(axis=0, ddof=1).sum())
    old = compare_threeway({name:arrays[name] for name in ("ode","ar","wan")}, seeds, bootstrap_draws=100)
    assert changed["bandwidth"] == old["bandwidth"]
    for name, pair in old["pairs"].items():
        for key in ("variance_ratio", "mean_difference_mse", "mmd2_paired_u"):
            assert math.isclose(changed["pairs"][name][key], pair[key], abs_tol=1e-12)
    constants = compare_fourway(dict.fromkeys(MODELS,np.ones((8,4))),list(range(8)),bootstrap_draws=20)
    assert constants["bandwidth"]["squared"] is None
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline, ode, cf = root/"baseline", root/"ode", root/"cf"
        for name in MODELS:
            for seed in range(8):
                run_root = cf if name == "cf_init" else ode if name == "ode" else baseline
                path = run_root/name/"prompt_000042"/f"seed_{seed:06d}"
                path.mkdir(parents=True)
                latent = a[seed]
                np.savez_compressed(path/"features.npz",latent=latent)
                protocol = dict(model=name,prompt_index=42,prompt="test",seed=seed,sigma=.9375,
                                feature_space=FEATURE_SPACE,feature_shape=list(latent.shape),
                                model_source=dict(checkpoint_path=f"/{name}.pt",weights="generator"),
                                sampler="same causal",completed_updates=1)
                record = dict(protocol, complete=True,schema_version="diversity_moments_sampling_v1",shape=list(latent.shape),
                              noise_sha256=f"noise-{seed}",initial_noise_sha256=f"noise-{seed}",
                              protocol=protocol,protocol_sha256=canonical_hash(protocol),
                              latent_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),
                              artifact_sha256={"features.npz":hashlib.sha256((path/"features.npz").read_bytes()).hexdigest()})
                if name in ("ar","ode","cf_init"):
                    record["transition_noise_sha256"]={"0:1":f"transition-{seed}"}
                if seed == 0:
                    from PIL import Image
                    Image.new("RGB",(416,240),(30,90,140)).save(path/"preview.jpg")
                    record["artifact_sha256"]["preview.jpg"]=hashlib.sha256((path/"preview.jpg").read_bytes()).hexdigest()
                (path/"record.json").write_text(json.dumps(record))
        args = (baseline,ode,cf,[42],list(range(8)))
        groups,coverage,complete,shape,sources,protocols = load_fourway(*args,cf_checkpoint="/cf_init.pt")
        assert complete and len(groups[0]["seeds"])==8 and shape==(2,3)
        try:
            load_fourway(*args,cf_checkpoint="/wrong_checkpoint.pt")
        except ValueError as error:
            assert "checkpoint identity differs" in str(error)
        else:
            raise AssertionError("Expected CF checkpoint identity rejection")
        record_path = cf/"cf_init/prompt_000042/seed_000000/record.json"
        original = json.loads(record_path.read_text())
        for key,value,expected in (("sigma",.9375+1e-9,"CF probe sigma"),("noise_sha256","bad","initial noise"),
                                   ("prompt_index",300,"prompt_index"),("latent_sha256","bad","latent array")):
            record_path.write_text(json.dumps(dict(original,**{key:value})))
            try:
                load_fourway(*args,cf_checkpoint="/cf_init.pt")
            except ValueError as error:
                assert expected in str(error),str(error)
            else:
                raise AssertionError(f"Expected rejection for {key}")
        record_path.write_text(json.dumps(original))
        # Report rendering must work from the saved summary without features.
        item = dict(prompt_index=42,prompt="test",**changed)
        summary = dict(prompts=[item],complete=True,sigma=.9375,bootstrap_draws=100,
                       completeness=coverage,model_sources=sources,sampling_protocols=protocols)
        output = root/"output"
        previews=build_comparisons(output,baseline,ode,cf,[42],list(range(8)))
        assert len(previews["prompts"]["42"]["rows"])==1
        assert set(previews["prompts"]["42"]["rows"][0]["models"])==set(MODELS)
        assert (output/previews["prompts"]["42"]["contact_sheet"]).is_file()
        write_outputs(output,summary,{"test":a.mean(axis=0)})
        assert "Causal Forcing init" in (output/"report.html").read_text()
        for path in cf.glob("cf_init/prompt_*/seed_*/features.npz"):
            path.unlink()
        write_report(output,json.loads((output/"summary.json").read_text()))
    print("Four-way metrics, old-metric compatibility, CF identity/sigma, and report-only self-tests passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root",type=Path,default=Path("outputs/diversity/diversity_moments_20261023"))
    parser.add_argument("--ode-root",type=Path,default=Path("outputs/diversity/diversity_ode_init_20261023"))
    parser.add_argument("--cf-root",type=Path,default=Path("outputs/diversity/diversity_causal_forcing_init_20261023"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/diversity/diversity_causal_forcing_init_20261023"))
    parser.add_argument("--cf-checkpoint",type=Path)
    parser.add_argument("--ode-checkpoint",type=Path,default=Path("checkpoints/ode_init.pt"))
    parser.add_argument("--prompt-indices",type=int,nargs="+",default=[42,300])
    parser.add_argument("--expected-seeds",type=int,default=64)
    parser.add_argument("--seed-start",type=int,default=0)
    parser.add_argument("--sigma",type=float,default=.9375)
    parser.add_argument("--bootstrap-draws",type=int,default=2000)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-previews",action="store_true")
    parser.add_argument("--report-only",action="store_true",help="Render saved summary without loading latent samples")
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.report_only:
        write_report(args.output_dir,json.loads((args.output_dir/"summary.json").read_text()))
        print(args.output_dir/"report.html")
        return
    if args.cf_checkpoint is None:
        parser.error("--cf-checkpoint is required to verify Causal Forcing initialization identity")
    if args.expected_seeds<2 or args.bootstrap_draws<0 or len(set(args.prompt_indices))!=len(args.prompt_indices):
        parser.error("Need >=2 seeds, nonnegative bootstrap draws, and unique prompts")
    if args.output_dir.resolve() in (args.baseline_root.resolve(),args.ode_root.resolve()):
        parser.error("Use a new output directory to preserve previous reports")
    seeds = list(range(args.seed_start,args.seed_start+args.expected_seeds))
    groups,coverage,complete,shape,sources,protocols = load_fourway(
        args.baseline_root,args.ode_root,args.cf_root,args.prompt_indices,seeds,args.sigma,
        args.allow_partial,args.ode_checkpoint,args.cf_checkpoint)
    summary = dict(schema_version="diversity_fourway_v1",generated_at=datetime.now(timezone.utc).isoformat(),
                   complete=complete,sigma=args.sigma,expected_seeds=seeds,feature_space=FEATURE_SPACE,
                   feature_shape=list(shape) if shape else None,model_order=list(MODELS),
                   model_sources=sources,sampling_protocols=protocols,completeness=coverage,
                   roots={"baseline":str(args.baseline_root),"ode":str(args.ode_root),"cf_init":str(args.cf_root)},
                   bootstrap_draws=args.bootstrap_draws,prompts=[])
    means = {}
    for group in groups:
        item = {k:v for k,v in group.items() if k!="arrays"}
        if len(group["seeds"])>=2:
            item.update(compare_fourway(group["arrays"],group["seeds"],bootstrap_draws=args.bootstrap_draws))
            for name in MODELS:
                means[f"prompt_{group['prompt_index']:06d}_{name}"]=group["arrays"][name].mean(axis=0,dtype=np.float64)
            print(f"SUMMARIZED prompt={group['prompt_index']} seeds={len(group['seeds'])}",flush=True)
        else:
            item["n_seeds"]=len(group["seeds"])
        summary["prompts"].append(item)
    if not args.skip_previews:
        build_comparisons(args.output_dir,args.baseline_root,args.ode_root,args.cf_root,args.prompt_indices,seeds,args.sigma)
    write_outputs(args.output_dir,summary,means)
    print(json.dumps({"complete":complete,"report":str(args.output_dir/"report.html"),
                      "paired_seeds":{item["prompt_index"]:item["n_seeds"] for item in summary["prompts"]}},ensure_ascii=False))


if __name__=="__main__":
    main()
