#!/usr/bin/env python3
"""Compare ODE initialization, trained AR, and Wan at the same noise level.

Only reads sampling artifacts. The original two-model report is left intact.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diversity.summarize_diversity_moments import (
    PROMPT_LABELS, bootstrap_variance_ratio, format_number, gram, load_case,
    mmd_from_gram, moments_from_gram, pilot_bandwidth_squared, variance_ratio,
)

MODELS = ("ode", "ar", "wan")
LABELS = {"ode": "ODE init", "ar": "AR", "wan": "Wan bidirectional"}
PAIRS = (("ode", "ar"), ("ode", "wan"), ("ar", "wan"))
FEATURE_SPACE = "wan_vae_raw_x0_first3latents"


def canonical_hash(value):
    # Match the sampling script's JSON representation, including Unicode and
    # default separators, rather than importing its torch-dependent helper.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def model_directory(root, model):
    """Accept both a run root and a direct model directory for ODE samples."""
    root = Path(root)
    if model != "ode":
        return root / model, model
    for name in ("ode", "ar"):
        if (root / name).is_dir():
            return root / name, name
    if root.name in ("ode", "ar") and root.is_dir():
        return root, root.name
    return root / "ode", "ode"


def checked_case(directory, record_model, prompt_index, seed, sigma):
    case = load_case(directory, record_model, prompt_index, seed, sigma)
    record, latent = case["record"], case["latent"]
    protocol = record.get("protocol")
    if not isinstance(protocol, dict) or record.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError(f"{directory}: protocol SHA-256 mismatch or missing protocol")
    expected = dict(model=record_model, prompt_index=prompt_index, seed=seed,
                    prompt=record.get("prompt"), feature_space=FEATURE_SPACE,
                    model_source=record.get("model_source"))
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"{directory}: record/protocol {key} mismatch")
    if not isinstance(record.get("prompt"), str) or not record["prompt"]:
        raise ValueError(f"{directory}: missing prompt text")
    if not record.get("model_source") or record.get("feature_space") != FEATURE_SPACE:
        raise ValueError(f"{directory}: missing model source or incorrect feature space")
    if record.get("shape") != list(latent.shape) or protocol.get("feature_shape") != list(latent.shape):
        raise ValueError(f"{directory}: declared feature shape does not match artifact")
    if not math.isclose(float(protocol.get("sigma", -1)), sigma, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"{directory}: protocol sigma mismatch")
    for key in ("initial_noise_sha256", "noise_sha256"):
        if record.get(key) != case["noise_sha256"]:
            raise ValueError(f"{directory}: inconsistent initial noise hashes")
    digest = hashlib.sha256(np.ascontiguousarray(latent).tobytes()).hexdigest()
    if record.get("latent_sha256") != digest:
        raise ValueError(f"{directory}: latent array SHA-256 mismatch")
    return case


def validate_causal_protocols(ode, ar):
    for key in ("schema_version", "shape", "chunks", "feature_shape", "feature_space", "sigma",
                "config", "guidance", "wan_steps", "shift", "noise_protocol", "ar_transition_protocol",
                "sampler", "completed_updates", "torch_version"):
        if ode.get(key) != ar.get(key):
            raise ValueError(f"ODE/AR causal sampling protocol differs: {key}")
    if ode.get("model_source", {}).get("weights") != ar.get("model_source", {}).get("weights"):
        raise ValueError("ODE/AR checkpoint weights selection differs")
    # The sampling entry point gained the ODE label/branch in this experiment.
    # Every shared model and sampler implementation must remain identical.
    excluded = {"scripts/diversity/eval_diversity_moments.py"}
    if any(not isinstance(protocol.get("source_sha256", {}), dict) for protocol in (ode, ar)):
        raise ValueError("Invalid source SHA-256 mapping")
    ode_code = {k:v for k,v in ode.get("source_sha256", {}).items() if k not in excluded}
    ar_code = {k:v for k,v in ar.get("source_sha256", {}).items() if k not in excluded}
    if ode_code != ar_code:
        raise ValueError("ODE/AR model or sampler source SHA-256 differs")


def load_threeway(baseline_root, ode_root, prompts, seeds, sigma=.9375,
                  allow_partial=False, ode_checkpoint=None):
    locations = {name: model_directory(ode_root if name == "ode" else baseline_root, name)
                 for name in MODELS}
    if locations["ode"][1] == "ar" and ode_checkpoint is None:
        raise ValueError("ODE artifacts labeled ar require --ode-checkpoint to establish checkpoint identity")
    expected_checkpoint = Path(ode_checkpoint).resolve() if ode_checkpoint is not None else None
    resolved_sources = {}
    common_protocols, sources, shape = {}, {}, None
    groups, coverage, complete = [], [], True
    for prompt_index in prompts:
        cases = {name: {} for name in MODELS}
        prompt_text = None
        for name in MODELS:
            model_root, record_model = locations[name]
            for seed in seeds:
                directory = model_root / f"prompt_{prompt_index:06d}" / f"seed_{seed:06d}"
                if not (directory / "record.json").is_file() or not (directory / "features.npz").is_file():
                    continue
                case = checked_case(directory, record_model, prompt_index, seed, sigma)
                record = case["record"]
                if prompt_text is None:
                    prompt_text = record["prompt"]
                if record["prompt"] != prompt_text:
                    raise ValueError(f"{directory}: prompt text mismatch across models/seeds")
                if shape is None:
                    shape = case["latent"].shape
                if case["latent"].shape != shape:
                    raise ValueError(f"{directory}: feature shape mismatch across models/seeds")
                common = {k: v for k, v in record["protocol"].items()
                          if k not in ("prompt_index", "prompt", "seed")}
                if name in common_protocols and common_protocols[name] != common:
                    raise ValueError(f"{directory}: mixed sampling protocols for {name}")
                common_protocols[name] = common
                sources[name] = record["model_source"]
                if name == "ode" and ode_checkpoint is not None:
                    actual = record["model_source"].get("checkpoint_path", "")
                    if actual not in resolved_sources:
                        resolved_sources[actual] = Path(actual).resolve()
                    if resolved_sources[actual] != expected_checkpoint:
                        raise ValueError(f"{directory}: ODE checkpoint differs from requested checkpoint")
                cases[name][seed] = case
            missing = sorted(set(seeds) - cases[name].keys())
            coverage.append(dict(model=name, prompt_index=prompt_index, completed=len(cases[name]),
                                 expected=len(seeds), missing_seeds=missing))
            complete = complete and not missing
        paired = sorted(set.intersection(*(set(cases[name]) for name in MODELS)))
        for seed in paired:
            if len({cases[name][seed]["noise_sha256"] for name in MODELS}) != 1:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: initial noise differs across models")
            # The two causal checkpoints must also share the re-noising draw.
            ar_noise = cases["ar"][seed]["record"].get("transition_noise_sha256")
            ode_noise = cases["ode"][seed]["record"].get("transition_noise_sha256")
            if not ar_noise or ar_noise != ode_noise:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: AR/ODE transition noise mismatch")
            ar_steps = cases["ar"][seed]["record"].get("timesteps")
            ode_steps = cases["ode"][seed]["record"].get("timesteps")
            # Earlier reused records have steps in their source provenance,
            # while freshly sampled records expose them directly.
            if ar_steps is not None and ode_steps is not None and ar_steps != ode_steps:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: AR/ODE timesteps differ")
        arrays = {name: np.stack([cases[name][seed]["latent"] for seed in paired])
                  for name in MODELS} if paired else {}
        groups.append(dict(prompt_index=prompt_index, prompt=prompt_text, seeds=paired, arrays=arrays))
    if "ar" in common_protocols and "ode" in common_protocols:
        validate_causal_protocols(common_protocols["ode"], common_protocols["ar"])
    if not complete and not allow_partial:
        raise ValueError("Incomplete three-model sample set; use --allow-partial only for a progress report")
    return groups, coverage, complete, shape, sources


def compare_threeway(arrays, seeds, pilot_seeds=tuple(range(8)), bootstrap_draws=2000):
    n = len(seeds)
    if n < 2 or any(len(arrays[name]) != n for name in MODELS):
        raise ValueError("Need at least two paired samples for all three models")
    shapes = {arrays[name].shape for name in MODELS}
    if len(shapes) != 1:
        raise ValueError("All model arrays must have the same shape")
    dimension = int(np.prod(arrays["ar"].shape[1:]))
    g = gram(np.concatenate([arrays[name] for name in MODELS]))
    blocks = {name: list(range(i*n, (i+1)*n)) for i, name in enumerate(MODELS)}
    # Fix the bandwidth from exactly the original AR/Wan pilot, including when
    # ODE samples are added; all three MMD values then have the same scale.
    ar_wan = blocks["ar"] + blocks["wan"]
    pilot_complete = set(pilot_seeds).issubset(seeds)
    ell2 = pilot_bandwidth_squared(g[np.ix_(ar_wan, ar_wan)], n,
                                  [seeds.index(seed) for seed in pilot_seeds]) if pilot_complete else None
    result = dict(n_seeds=n, seeds=seeds, dimension=dimension, models={}, pairs={},
                  bandwidth=dict(squared=ell2, pilot_seeds=list(pilot_seeds),
                                 pilot_complete=pilot_complete, pilot_models=["ar", "wan"],
                                 rule="pooled AR/Wan pilot median off-diagonal squared distance / 2",
                                 distance="raw latent squared Euclidean distance / dimension"))
    for name in MODELS:
        moments = moments_from_gram(g[np.ix_(blocks[name], blocks[name])])
        moments["total_variance_trace_unbiased"] = dimension * moments["mean_coordinate_variance_unbiased"]
        result["models"][name] = moments
    for a, b in PAIRS:
        index = blocks[a] + blocks[b]
        pair_g = g[np.ix_(index, index)]
        aa, bb, ab = pair_g[:n, :n], pair_g[n:, n:], pair_g[:n, n:]
        result["pairs"][f"{a}_vs_{b}"] = dict(
            numerator=a, denominator=b,
            variance_ratio=variance_ratio(result["models"][a]["mean_coordinate_variance_unbiased"],
                                          result["models"][b]["mean_coordinate_variance_unbiased"]),
            mean_difference_mse=max(0.0, float(aa.mean() + bb.mean() - 2*ab.mean())),
            **mmd_from_gram(pair_g, n, ell2))
        if bootstrap_draws:
            result["pairs"][f"{a}_vs_{b}"]["variance_ratio_bootstrap"] = bootstrap_variance_ratio(aa, bb, bootstrap_draws)
    return result


def direction(pair):
    interval = pair.get("variance_ratio_bootstrap", {}).get("ci95")
    if interval and interval[1] < 1:
        return f"{LABELS[pair['numerator']]} 方差更小"
    if interval and interval[0] > 1:
        return f"{LABELS[pair['numerator']]} 方差更大"
    return "区间跨 1，方向未定"


def comparison_gallery(prompt_index, comparisons):
    entry = next((item for item in (comparisons or {}).get("prompts", [])
                  if item["prompt_index"] == prompt_index), None)
    if not entry or not entry.get("images"):
        return ""
    seeds = entry["seeds"]
    first = seeds[0]
    images = entry["images"]
    buttons = "".join(f'<button type="button" data-seed="{seed}" aria-pressed="{str(seed == first).lower()}">Seed {seed}</button>' for seed in seeds)
    cards = "".join(f'''<figure><figcaption>{LABELS[model]}</figcaption>
<a data-model="{model}" href="{html.escape(images[str(first)][model], quote=True)}" target="_blank" rel="noopener">
<img src="{html.escape(images[str(first)][model], quote=True)}" alt="{LABELS[model]}, seed {first}" width="416" height="240" loading="lazy"></a></figure>''' for model in MODELS)
    encoded = html.escape(json.dumps(images), quote=True)
    sheet = html.escape(entry["sheet"], quote=True)
    return f'''<div class="comparison-gallery" data-images="{encoded}">
<h3>同一个 noise，三个模型并排看</h3>
<div class="seed-buttons" aria-label="选择同一个 seed">{buttons}</div>
<div class="comparison-cards">{cards}</div>
<small>σ = 0.9375 的预测 x₀，经 VAE 解码后取相同中间帧（帧索引 4）；这是早期预测图。固定展示 seed 0–7，指标使用全部 64 个 noise。</small>
<details class="comparison-overview"><summary>展开全部 {len(seeds)} 个 noise 的对比拼图</summary>
<p><a href="{sheet}" target="_blank" rel="noopener">打开大图</a> · <a href="{sheet}" download>下载拼图</a></p>
<a href="{sheet}" target="_blank" rel="noopener"><img src="{sheet}" alt="Prompt {prompt_index}：三模型全部 noise 对比" loading="lazy"></a></details></div>'''


def report_html(summary, comparisons=None):
    gallery_style = """
.comparison-gallery{margin:24px 0;padding:20px 0;border-top:1px solid #e3e7ec;border-bottom:1px solid #e3e7ec}
.comparison-gallery h3{margin:0 0 12px;font-size:18px}.seed-buttons{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 18px}
.seed-buttons button{border:1px solid #c6d2df;border-radius:7px;padding:7px 12px;background:white;color:#35506b;cursor:pointer;font:inherit;font-size:14px}
.seed-buttons button[aria-pressed=true]{background:#1765b7;border-color:#1765b7;color:white}.seed-buttons button:focus-visible{outline:3px solid #8bb8ed;outline-offset:2px}
.comparison-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:12px}
.comparison-cards figure{margin:0;min-width:0}.comparison-cards figcaption{font-weight:650;margin-bottom:8px}
.comparison-cards img{display:block;width:100%;height:auto;border-radius:7px;background:#e8edf3}
.comparison-overview{margin-top:16px}.comparison-overview summary{cursor:pointer;color:#1765b7}.comparison-overview img{display:block;width:100%;height:auto}
@media(max-width:700px){.comparison-cards{grid-template-columns:1fr}.comparison-cards figure{max-width:416px}.seed-buttons button{padding:6px 9px}}
"""
    gallery_script = """<script>
document.querySelectorAll('.comparison-gallery').forEach(gallery => {
  const images = JSON.parse(gallery.dataset.images);
  gallery.querySelectorAll('button[data-seed]').forEach(button => {
    button.addEventListener('click', () => {
      const seed = button.dataset.seed;
      gallery.querySelectorAll('button[data-seed]').forEach(other =>
        other.setAttribute('aria-pressed', String(other === button)));
      gallery.querySelectorAll('a[data-model]').forEach(link => {
        const model = link.dataset.model;
        const image = link.querySelector('img');
        link.href = images[seed][model];
        image.src = images[seed][model];
        image.alt = model + ', seed ' + seed;
      });
    });
  });
});
</script>"""
    sections = []
    for item in summary["prompts"]:
        title = f"{PROMPT_LABELS.get(item['prompt_index'], 'Prompt')} · #{item['prompt_index']}"
        if item["n_seeds"] < 2:
            sections.append(f"<section><h2>{title}</h2><p>等待至少 2 个三模型共同完成的 noise。</p></section>")
            continue
        rows = "".join(f"<tr><td><strong>{LABELS[name]}</strong></td><td>{format_number(item['models'][name]['mean_tensor_rms'])}</td><td>{format_number(item['models'][name]['mean_coordinate_variance_unbiased'])}</td><td>{format_number(item['models'][name]['total_variance_trace_unbiased'], 6)}</td></tr>" for name in MODELS)
        pair_rows = []
        for a, b in PAIRS:
            pair = item["pairs"][f"{a}_vs_{b}"]
            ci = pair.get("variance_ratio_bootstrap", {}).get("ci95")
            ci_text = f"[{format_number(ci[0], 3)}, {format_number(ci[1], 3)}]" if ci else "—"
            pair_rows.append(f"<tr><td>{LABELS[a]} / {LABELS[b]}</td><td><strong>{format_number(pair['variance_ratio'])}</strong><small>95% CI {ci_text}<br>{direction(pair)}</small></td><td>{format_number(pair['mean_difference_mse'])}</td><td>{format_number(pair['mmd2_paired_u'])}</td></tr>")
        sections.append(f"""<section><h2>{title}</h2><p>{html.escape(item.get('prompt') or '')}</p>
<p>共同 noise 数：<strong>{item['n_seeds']}</strong>；维度 D = {item['dimension']:,}。</p>
{comparison_gallery(item['prompt_index'], comparisons)}
<div class="table"><table><tr><th>模型</th><th>均值张量 RMS</th><th>跨 noise 方差 V / D</th><th>整体总方差 V = tr(Σ̂)</th></tr>{rows}</table></div>
<div class="table"><table><tr><th>比较（分子 / 分母）</th><th>方差比及 95% CI</th><th>均值之差 MSE</th><th>RBF MMD²</th></tr>{''.join(pair_rows)}</table></div>
<small>本 prompt 三对比较共用 ℓ² = {format_number(item['bandwidth']['squared'], 6)}；带宽仅由原 AR/Wan 的 seed 0–7 确定。</small></section>""")
    sources = "".join(f"<li><strong>{LABELS[name]}</strong>：{html.escape(source.get('checkpoint_path', str(source)))}；weights={html.escape(str(source.get('weights', '')))}</li>" for name, source in summary["model_sources"].items())
    done = sum(x["completed"] for x in summary["completeness"])
    expected = sum(x["expected"] for x in summary["completeness"])
    status = "完整结果" if summary["complete"] else "临时结果 · 样本尚未齐全"
    visual_note = ('<p><strong>读图提示：</strong>本次预览中，ODE init 的早期预测明显更模糊。解释方差差距时需要同时考虑清晰度与细节强度，不能直接将较低 latent 方差等同于较少的场景模式。</p>'
                   if comparisons else "")
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ODE init / AR / Wan：均值、方差与 MMD</title><style>
{gallery_style}
body{{font:16px/1.65 system-ui,sans-serif;color:#172334;background:#f4f6f8;margin:0}}main{{max-width:1100px;margin:auto;padding:32px 20px}}h1{{font-size:27px;line-height:1.3}}h2{{font-size:21px;margin-top:0}}section{{background:white;padding:24px;margin:22px 0;border:1px solid #dde3ea;border-radius:12px}}p{{margin:10px 0}}.table{{overflow-x:auto;margin:20px 0}}table{{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}}th,td{{padding:12px;border-bottom:1px solid #e3e7ec;vertical-align:top}}th{{background:#f3f6fa}}small{{display:block;color:#5e6d7e;font-size:13px}}code{{background:#edf1f5;padding:2px 5px}}a{{color:#1765b7}}li{{overflow-wrap:anywhere}}.status{{color:#36526f}}</style>
<main><h1>ODE init / AR / Wan：均值、方差与 MMD</h1>
<p class="status">{status} · {done}/{expected} 个模型样本 · σ = {summary['sigma']}</p>
<p>相同 prompt、相同初始 noise，在相同 noise level 比较预测 x₀。三个模型均取前三个 latent 帧，使用同一个原始 Wan VAE latent 空间。</p>
{visual_note}
{''.join(sections)}
<section><h2>指标怎么读</h2><p><strong>方差：</strong>将每个 noise 的完整 latent 展平为 xᵢ，先求平均张量 μ = (1/N) Σᵢ xᵢ，再计算 V = [1/(N−1)] Σᵢ ‖xᵢ−μ‖²。整体总方差是 V；V/D 只用于消除维度尺度。二者的模型大小关系相同。</p>
<p><strong>Mean：</strong>表中 RMS = ‖μ‖/√D，只描述平均张量的大小；两模型平均张量是否接近，看均值之差 MSE = ‖μₐ−μᵦ‖²/D。完整平均张量保存在 <a href="mean_tensors.npz">mean_tensors.npz</a>。</p>
<p><strong>MMD²：</strong>衡量两个分布的差异，越大表示差异越明显。使用 RBF 核、同一 prompt 固定带宽和 paired U 估计；相同 seed 的三模型配对，跨模型 i=j 项也剔除。有限样本值可略小于零。它不单独判断哪个模型更 diverse。</p>
<p>方差比 &lt; 1 表示分子模型跨 noise 更集中；置信区间使用 {summary['bootstrap_draws']} 次配对 seed bootstrap。这些指标只衡量当前早期 x₀ latent 分布，不能直接代表最终完整视频的语义多样性。</p></section>
<details><summary>Checkpoint 与采样范围</summary><ul>{sources}</ul><p>AR 与 ODE init 使用相同 causal sampler、初始 noise 与 re-noising noise；Wan 使用原 48 步 Euler schedule，到 σ = 0.9375 时读取预测 x₀。AR / ODE init 的上下文为前三个 latent 帧，Wan 上下文为六帧后截取前三帧。未对每个样本单独归一化。</p></details>
<p><a href="summary.json">完整 JSON</a> · <a href="summary.csv">CSV</a> · <a href="https://www.jmlr.org/papers/v13/gretton12a.html">MMD 参考</a></p></main>{gallery_script}</html>"""


def write_report(output_dir, summary):
    manifest_path = output_dir / "comparisons" / "manifest.json"
    comparisons = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    (output_dir / "report.html").write_text(report_html(summary, comparisons))


def write_outputs(output_dir, summary, means):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "mean_tensors.npz", **means)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fields = ["prompt_index", "n_seeds", "model", "mean_tensor_rms", "variance_per_dimension", "total_variance_trace", "compared_to", "variance_ratio", "ratio_ci95_low", "ratio_ci95_high", "mean_difference_mse", "mmd2_paired_u", "rbf_bandwidth_squared", "complete"]
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in summary["prompts"]:
            if item["n_seeds"] < 2:
                continue
            for a, b in PAIRS:
                pair, model = item["pairs"][f"{a}_vs_{b}"], item["models"][a]
                ci = pair.get("variance_ratio_bootstrap", {}).get("ci95") or [None, None]
                writer.writerow(dict(prompt_index=item["prompt_index"], n_seeds=item["n_seeds"], model=a,
                                     mean_tensor_rms=model["mean_tensor_rms"], variance_per_dimension=model["mean_coordinate_variance_unbiased"],
                                     total_variance_trace=model["total_variance_trace_unbiased"], compared_to=b,
                                     variance_ratio=pair["variance_ratio"], ratio_ci95_low=ci[0], ratio_ci95_high=ci[1],
                                     mean_difference_mse=pair["mean_difference_mse"], mmd2_paired_u=pair["mmd2_paired_u"],
                                     rbf_bandwidth_squared=item["bandwidth"]["squared"], complete=summary["complete"]))
            # Include the remaining model's moments even though it is only a pair denominator.
            model = item["models"]["wan"]
            writer.writerow(dict(prompt_index=item["prompt_index"], n_seeds=item["n_seeds"], model="wan",
                                 mean_tensor_rms=model["mean_tensor_rms"], variance_per_dimension=model["mean_coordinate_variance_unbiased"],
                                 total_variance_trace=model["total_variance_trace_unbiased"], complete=summary["complete"]))
    write_report(output_dir, summary)


def self_test():
    rng = np.random.default_rng(52)
    a = rng.normal(size=(16, 2, 3)).astype(np.float32)
    seeds = list(range(16))
    same = compare_threeway(dict.fromkeys(MODELS, a), seeds, bootstrap_draws=100)
    for pair in same["pairs"].values():
        assert math.isclose(pair["variance_ratio"], 1, abs_tol=1e-12)
        assert abs(pair["mmd2_paired_u"]) < 1e-12
        assert np.allclose(pair["variance_ratio_bootstrap"]["ci95"], [1, 1])
    changed = compare_threeway(dict(ode=a*2, ar=a, wan=a), seeds, bootstrap_draws=100)
    assert math.isclose(changed["pairs"]["ode_vs_ar"]["variance_ratio"], 4, rel_tol=1e-12)
    assert np.allclose(changed["pairs"]["ode_vs_wan"]["variance_ratio_bootstrap"]["ci95"], [4, 4])
    assert changed["pairs"]["ode_vs_ar"]["mmd2_paired_u"] > 0
    assert direction(changed["pairs"]["ode_vs_ar"]) == "ODE init 方差更大"
    literal = a.astype(np.float64).reshape(len(a), -1).var(axis=0, ddof=1)
    assert np.allclose(changed["models"]["ar"]["total_variance_trace_unbiased"], literal.sum())
    assert np.allclose(changed["models"]["ar"]["mean_tensor_rms"], np.sqrt((a.astype(np.float64).mean(axis=0)**2).mean()))
    assert changed["bandwidth"] == same["bandwidth"]
    constant = compare_threeway(dict.fromkeys(MODELS, np.ones((8, 4))), list(range(8)), bootstrap_draws=20)
    assert constant["bandwidth"]["squared"] is None
    assert all(pair["mmd2_paired_u"] is None and pair["variance_ratio"] is None for pair in constant["pairs"].values())
    for changed_key in ("config", "torch_version", "chunks", "source_sha256"):
        try:
            validate_causal_protocols({changed_key:"changed"}, {})
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected rejection for cross-checkpoint {changed_key}")
    validate_causal_protocols({"source_sha256":{"scripts/diversity/eval_diversity_moments.py":"new"}},
                             {"source_sha256":{"scripts/diversity/eval_diversity_moments.py":"old"}})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline, ode = root / "baseline", root / "ode_run"
        for model in MODELS:
            for seed in range(8):
                destination = (ode if model == "ode" else baseline) / model / "prompt_000042" / f"seed_{seed:06d}"
                destination.mkdir(parents=True)
                x = a[seed]
                np.savez_compressed(destination / "features.npz", latent=x)
                protocol = dict(model=model, prompt_index=42, seed=seed, prompt="test", sigma=.9375,
                                feature_space=FEATURE_SPACE, feature_shape=list(x.shape), model_source=dict(checkpoint_path=f"/{model}.pt"))
                record = dict(complete=True, schema_version="diversity_moments_sampling_v1", **protocol,
                              shape=list(x.shape), noise_sha256=f"noise-{seed}", initial_noise_sha256=f"noise-{seed}",
                              transition_noise_sha256={"0:1":f"transition-{seed}"}, protocol=protocol,
                              protocol_sha256=canonical_hash(protocol), latent_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                              artifact_sha256={"features.npz":hashlib.sha256((destination / "features.npz").read_bytes()).hexdigest()})
                (destination / "record.json").write_text(json.dumps(record))
        args = (baseline, ode, [42], list(range(8)))
        groups, coverage, complete, shape, sources = load_threeway(*args)
        assert complete and shape == (2, 3) and len(groups[0]["seeds"]) == 8
        record_path = ode / "ode/prompt_000042/seed_000000/record.json"
        original = json.loads(record_path.read_text())
        for key, changed_value, expected_error in (("noise_sha256", "changed", "inconsistent initial noise"),
                                                  ("prompt_index", 300, "prompt_index"),
                                                  ("sigma", .9, "sigma"),
                                                  ("latent_sha256", "changed", "latent array"),
                                                  ("protocol_sha256", "changed", "protocol SHA-256")):
            altered = dict(original, **{key:changed_value})
            record_path.write_text(json.dumps(altered))
            try:
                load_threeway(*args)
            except ValueError as error:
                assert expected_error in str(error), str(error)
            else:
                raise AssertionError(f"Expected rejection for {key}")
        record_path.write_text(json.dumps(original))
        # Also verify the older sampler layout with model=ar can be relabeled ODE.
        (ode / "ode").rename(ode / "ar")
        for path in (ode / "ar").glob("prompt_*/seed_*/record.json"):
            record = json.loads(path.read_text())
            record["model"] = record["protocol"]["model"] = "ar"
            record["protocol_sha256"] = canonical_hash(record["protocol"])
            path.write_text(json.dumps(record))
        assert load_threeway(*args, ode_checkpoint="/ode.pt")[2]
    print("Three-way moments/MMD and sampling identity self-tests passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/diversity/diversity_moments_20261023"))
    parser.add_argument("--ode-root", type=Path, default=Path("outputs/diversity/diversity_ode_init_20261023"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diversity/diversity_ode_init_20261023"))
    parser.add_argument("--ode-checkpoint", type=Path)
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=[42, 300])
    parser.add_argument("--expected-seeds", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=.9375)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Render existing summary and comparison images without rereading latent samples")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.report_only:
        write_report(args.output_dir, json.loads((args.output_dir / "summary.json").read_text()))
        print(args.output_dir / "report.html")
        return
    if args.expected_seeds < 2 or args.bootstrap_draws < 0 or len(set(args.prompt_indices)) != len(args.prompt_indices):
        parser.error("Need >=2 seeds, nonnegative bootstrap draws, and unique prompts")
    if args.output_dir.resolve() == args.baseline_root.resolve():
        parser.error("Use a separate output directory to preserve the original AR/Wan report")
    seeds = list(range(args.seed_start, args.seed_start + args.expected_seeds))
    groups, coverage, complete, shape, sources = load_threeway(
        args.baseline_root, args.ode_root, args.prompt_indices, seeds, args.sigma,
        args.allow_partial, args.ode_checkpoint)
    summary = dict(schema_version="diversity_threeway_v1", generated_at=datetime.now(timezone.utc).isoformat(),
                   complete=complete, sigma=args.sigma, expected_seeds=seeds, feature_space=FEATURE_SPACE,
                   feature_shape=list(shape) if shape else None, baseline_root=str(args.baseline_root.resolve()),
                   ode_root=str(args.ode_root.resolve()), completeness=coverage, model_sources=sources,
                   bootstrap_draws=args.bootstrap_draws, prompts=[])
    means = {}
    for group in groups:
        identity = {k:v for k,v in group.items() if k != "arrays"}
        if len(group["seeds"]) >= 2:
            identity.update(compare_threeway(group["arrays"], group["seeds"], bootstrap_draws=args.bootstrap_draws))
            for name in MODELS:
                means[f"prompt_{group['prompt_index']:06d}_{name}"] = group["arrays"][name].mean(axis=0, dtype=np.float64)
        else:
            identity["n_seeds"] = len(group["seeds"])
        summary["prompts"].append(identity)
    write_outputs(args.output_dir, summary, means)
    print(json.dumps({"complete":complete, "report":str(args.output_dir / "report.html"),
                      "paired_seeds":{item["prompt_index"]:item["n_seeds"] for item in summary["prompts"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
