#!/usr/bin/env python3
"""Evaluate a manifest of checkpoint distributions without changing old reports.

Manifest paths are relative to the repository root. Each model declares:
  id, label, root, directory, checkpoint, weights; optional record_model,
  expected_source, protocol_reference, and transition_noise_reference.
Top-level fields include models, primary_models, prompt_indices, seeds, sigma,
feature_shape, pilot_models (AR/Wan by default), and pilot_seeds (0..7).
Weights and native sampler relationships are explicit model metadata: a new
checkpoint is never assumed to use AR's weights key or re-noising protocol.
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
import re
import sys
import tempfile
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPOSITORY))

from scripts.diversity.summarize_diversity_moments import (
    PROMPT_LABELS, bootstrap_variance_ratio, format_number, gram,
    mmd_from_gram, moments_from_gram, pilot_bandwidth_squared, variance_ratio,
)
from scripts.diversity.summarize_diversity_threeway import (
    FEATURE_SPACE, canonical_hash, checked_case, validate_causal_protocols,
)

SCHEMA = "diversity_models_v1"
DEFAULT_OUTPUT = "outputs/diversity/diversity_forcing_final_20261023"


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY/path


def validate_manifest(data):
    """Validate and fill defaults without resolving every sample path."""
    data = json.loads(json.dumps(data, allow_nan=False))
    models = data.get("models")
    if not isinstance(models, list) or len(models)<2:
        raise ValueError("Manifest needs at least two model definitions")
    ids = []
    for model in models:
        name = model.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or "_vs_" in name or name in ids:
            raise ValueError(f"Invalid or duplicate model id: {name!r}")
        ids.append(name)
        for key in ("label", "root", "directory", "checkpoint", "weights"):
            if not isinstance(model.get(key),str) or not model[key]:
                raise ValueError(f"{name}: explicitly specify nonempty {key}")
        model.setdefault("record_model", name)
        if not isinstance(model["record_model"], str) or not model["record_model"]:
            raise ValueError(f"{name}: invalid sampling record model")
        model.setdefault("expected_source", {})
        if not isinstance(model["expected_source"], dict):
            raise ValueError(f"{name}: expected_source must be an object")
        model.setdefault("expected_protocol", {})
        if not isinstance(model["expected_protocol"], dict):
            raise ValueError(f"{name}: expected_protocol must be an object")
    for model in models:
        for key in ("protocol_reference", "transition_noise_reference"):
            if key in model and model[key] not in ids:
                raise ValueError(f"{model['id']}: unknown {key} {model[key]!r}")
            if model.get(key) == model["id"]:
                raise ValueError(f"{model['id']}: {key} cannot reference itself")
    data.setdefault("primary_models", [name for name in ("cf","sgf","ar","wan") if name in ids] or ids[:4])
    primary = data["primary_models"]
    if not primary or len(set(primary)) != len(primary) or not set(primary).issubset(ids):
        raise ValueError("primary_models must contain unique known model ids")
    if data.get("reference_model") is not None and data["reference_model"] not in primary:
        raise ValueError("reference_model must name a primary model")
    data.setdefault("prompt_indices", [42,300])
    data.setdefault("seeds", list(range(64)))
    data.setdefault("sigma", .9375)
    data.setdefault("feature_shape", [3,16,60,104])
    data.setdefault("pilot_models", ["ar","wan"])
    data.setdefault("pilot_seeds", list(range(8)))
    for key in ("prompt_indices", "seeds", "pilot_seeds"):
        values = data[key]
        if not isinstance(values,list) or not values or any(not isinstance(x,int) or isinstance(x,bool) for x in values) or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list of unique integers")
    if len(data["seeds"])<2 or not math.isfinite(float(data["sigma"])):
        raise ValueError("Need >=2 planned seeds and finite sigma")
    if len(data["pilot_models"]) != 2 or len(set(data["pilot_models"])) != 2 or not set(data["pilot_models"]).issubset(ids):
        raise ValueError("pilot_models must name two distinct known models")
    if not set(data["pilot_seeds"]).issubset(data["seeds"]):
        raise ValueError("Every pilot seed must be a planned seed")
    if not isinstance(data["feature_shape"],list) or not data["feature_shape"] or any(not isinstance(x,int) or x<=0 for x in data["feature_shape"]):
        raise ValueError("feature_shape must contain positive integers")
    data["seeds"].sort()
    return data


def model_root(spec):
    directory = Path(spec["directory"])
    return directory if directory.is_absolute() else repository_path(spec["root"])/directory


def validate_sampler_reference(protocol, reference):
    # The weights key belongs to checkpoint identity and is independently
    # checked against the manifest. For example, EMA and generator checkpoints
    # can legitimately use the exact same native sampling algorithm.
    comparable = dict(protocol)
    comparable["model_source"] = dict(protocol["model_source"], weights=reference["model_source"].get("weights"))
    validate_causal_protocols(comparable, reference)


def load_models(manifest, allow_partial=False):
    specs, seeds, sigma = manifest["models"], manifest["seeds"], manifest["sigma"]
    expected_paths = {s["id"]: repository_path(s["checkpoint"]).resolve() for s in specs}
    resolved_sources, protocols, sources = {}, {}, {}
    groups, coverage, complete = [], [], True
    started = time.monotonic()
    for prompt_index in manifest["prompt_indices"]:
        cases = {s["id"]:{} for s in specs}
        prompt_text = None
        for spec in specs:
            name, directory = spec["id"], model_root(spec)
            for seed in seeds:
                case_dir = directory/f"prompt_{prompt_index:06d}"/f"seed_{seed:06d}"
                if not (case_dir/"record.json").is_file() or not (case_dir/"features.npz").is_file():
                    continue
                case = checked_case(case_dir,spec["record_model"],prompt_index,seed,sigma)
                record, protocol = case["record"], case["record"]["protocol"]
                if record["sigma"] != sigma or protocol["sigma"] != sigma:
                    raise ValueError(f"{case_dir}: probe sigma must equal target exactly")
                for key in ("native_probe_sigma","target_probe_sigma"):
                    if any(key in owner and owner[key] != sigma for owner in (record,protocol)):
                        raise ValueError(f"{case_dir}: {key} differs from target sigma")
                if list(case["latent"].shape) != manifest["feature_shape"]:
                    raise ValueError(f"{case_dir}: feature shape differs from manifest")
                if prompt_text is None:
                    prompt_text = record["prompt"]
                if record["prompt"] != prompt_text:
                    raise ValueError(f"{case_dir}: prompt text differs across models/seeds")
                source = record["model_source"]
                actual = source.get("checkpoint_path", "")
                if not actual:
                    raise ValueError(f"{case_dir}: missing checkpoint path")
                if actual not in resolved_sources:
                    resolved_sources[actual] = repository_path(actual).resolve()
                if resolved_sources[actual] != expected_paths[name] or source.get("weights") != spec["weights"]:
                    raise ValueError(f"{case_dir}: checkpoint or weights identity differs from manifest")
                for key,value in spec["expected_source"].items():
                    if source.get(key) != value:
                        raise ValueError(f"{case_dir}: expected model_source.{key} differs")
                for key,value in spec["expected_protocol"].items():
                    if protocol.get(key) != value:
                        raise ValueError(f"{case_dir}: expected protocol.{key} differs")
                common = {k:v for k,v in protocol.items() if k not in ("prompt_index","prompt","seed")}
                if name in protocols and protocols[name] != common:
                    raise ValueError(f"{case_dir}: mixed sampling protocols for {name}")
                protocols[name], sources[name], cases[name][seed] = common, source, case
            missing = sorted(set(seeds)-cases[name].keys())
            coverage.append(dict(model=name,prompt_index=prompt_index,completed=len(cases[name]),expected=len(seeds),missing_seeds=missing))
            complete = complete and not missing
            print(f"VALIDATED {name} prompt={prompt_index} {len(cases[name])}/{len(seeds)} elapsed={time.monotonic()-started:.1f}s",flush=True)
        paired = sorted(set.intersection(*(set(cases[s["id"]]) for s in specs)))
        for seed in paired:
            if len({cases[s["id"]][seed]["noise_sha256"] for s in specs}) != 1:
                raise ValueError(f"Prompt {prompt_index}, seed {seed}: initial noise differs across models")
            for spec in specs:
                reference = spec.get("transition_noise_reference")
                if reference:
                    current, other = cases[spec["id"]][seed]["record"], cases[reference][seed]["record"]
                    if not current.get("transition_noise_sha256") or current["transition_noise_sha256"] != other.get("transition_noise_sha256"):
                        raise ValueError(f"Prompt {prompt_index}, seed {seed}: {spec['id']}/{reference} transition noise differs")
                    if current.get("timesteps") is not None and other.get("timesteps") is not None and current["timesteps"] != other["timesteps"]:
                        raise ValueError(f"Prompt {prompt_index}, seed {seed}: {spec['id']}/{reference} timesteps differ")
        arrays = {s["id"]:np.stack([cases[s["id"]][seed]["latent"] for seed in paired]) for s in specs} if paired else {}
        groups.append(dict(prompt_index=prompt_index,prompt=prompt_text,seeds=paired,arrays=arrays))
    for spec in specs:
        reference = spec.get("protocol_reference")
        if reference and spec["id"] in protocols and reference in protocols:
            try:
                validate_sampler_reference(protocols[spec["id"]],protocols[reference])
            except ValueError as error:
                raise ValueError(f"{spec['id']}/{reference} sampler mismatch: {error}") from error
    if not complete and not allow_partial:
        raise ValueError("Incomplete model samples; use --allow-partial for a progress report")
    return groups,coverage,complete,sources,protocols


def compare_models(arrays,seeds,model_order,pilot_models=("ar","wan"),pilot_seeds=tuple(range(8)),bootstrap_draws=2000):
    n = len(seeds)
    if n<2 or len(model_order)<2 or len(set(model_order)) != len(model_order) or any(len(arrays[name]) != n for name in model_order):
        raise ValueError("Need >=2 paired samples from every distinct model")
    if len({arrays[name].shape for name in model_order}) != 1:
        raise ValueError("All model arrays must have identical shape")
    dimension = int(np.prod(arrays[model_order[0]].shape[1:]))
    g = gram(np.concatenate([arrays[name] for name in model_order]))
    blocks = {name:list(range(i*n,(i+1)*n)) for i,name in enumerate(model_order)}
    pilot_complete = set(pilot_seeds).issubset(seeds)
    pilot_indices = blocks[pilot_models[0]]+blocks[pilot_models[1]]
    ell2 = pilot_bandwidth_squared(g[np.ix_(pilot_indices,pilot_indices)],n,
                                  [seeds.index(seed) for seed in pilot_seeds]) if pilot_complete else None
    result = dict(n_seeds=n,seeds=list(seeds),model_order=list(model_order),dimension=dimension,models={},pairs={},
                  bandwidth=dict(squared=ell2,pilot_complete=pilot_complete,pilot_models=list(pilot_models),
                                 pilot_seeds=list(pilot_seeds),rule="pooled pilot median off-diagonal squared distance / 2",
                                 distance="raw latent squared Euclidean distance / dimension"))
    for name in model_order:
        moments = moments_from_gram(g[np.ix_(blocks[name],blocks[name])])
        moments["total_variance_trace_unbiased"] = dimension*moments["mean_coordinate_variance_unbiased"]
        result["models"][name] = moments
    for a,b in combinations(model_order,2):
        indices = blocks[a]+blocks[b]
        pair_g = g[np.ix_(indices,indices)]
        aa,bb,ab = pair_g[:n,:n],pair_g[n:,n:],pair_g[:n,n:]
        pair = dict(numerator=a,denominator=b,
                    variance_ratio=variance_ratio(result["models"][a]["mean_coordinate_variance_unbiased"],result["models"][b]["mean_coordinate_variance_unbiased"]),
                    reverse_variance_ratio=variance_ratio(result["models"][b]["mean_coordinate_variance_unbiased"],result["models"][a]["mean_coordinate_variance_unbiased"]),
                    mean_difference_mse=max(0.0,float(aa.mean()+bb.mean()-2*ab.mean())),**mmd_from_gram(pair_g,n,ell2))
        if bootstrap_draws:
            pair["variance_ratio_bootstrap"] = bootstrap_variance_ratio(aa,bb,bootstrap_draws)
            # Quantiles of reciprocal draws do not exactly equal reciprocals
            # of quantiles under linear interpolation. Store both directly so
            # comparisons keep identical CIs when model display order changes.
            pair["reverse_variance_ratio_bootstrap"] = bootstrap_variance_ratio(bb,aa,bootstrap_draws)
        result["pairs"][f"{a}_vs_{b}"] = pair
    return result


def display_pair(item,a,b):
    """Render any requested direction regardless of manifest model order."""
    if f"{a}_vs_{b}" in item["pairs"]:
        return item["pairs"][f"{a}_vs_{b}"]
    original = item["pairs"][f"{b}_vs_{a}"]
    result = dict(original,numerator=a,denominator=b)
    result["variance_ratio"] = original["reverse_variance_ratio"]
    result["reverse_variance_ratio"] = original["variance_ratio"]
    if "reverse_variance_ratio_bootstrap" in original:
        result["variance_ratio_bootstrap"] = original["reverse_variance_ratio_bootstrap"]
        result["reverse_variance_ratio_bootstrap"] = original["variance_ratio_bootstrap"]
    return result


def build_previews(output_dir,manifest):
    from PIL import Image,ImageDraw,ImageFont
    result = dict(schema_version="diversity_models_previews_v1",sigma=manifest["sigma"],decoded_frame_index=4,prompts={})
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",19)
    except OSError:
        font = ImageFont.load_default()
    output_dir.mkdir(parents=True,exist_ok=True)
    primary = manifest["primary_models"]
    secondary = [s["id"] for s in manifest["models"] if s["id"] not in primary]
    labels = {s["id"]:s["label"] for s in manifest["models"]}
    for prompt in manifest["prompt_indices"]:
        rows = []
        for seed in manifest["seeds"][:8]:
            models = {}
            for spec in manifest["models"]:
                directory = model_root(spec)/f"prompt_{prompt:06d}"/f"seed_{seed:06d}"
                path,record_path = directory/"preview.jpg",directory/"record.json"
                if not path.is_file() or not record_path.is_file():
                    continue
                record,data = json.loads(record_path.read_text()),path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                expected = dict(model=spec["record_model"],prompt_index=prompt,seed=seed,sigma=manifest["sigma"])
                if any(record.get(k) != v for k,v in expected.items()) or record.get("artifact_sha256",{}).get("preview.jpg") != digest:
                    raise ValueError(f"{path}: preview identity or SHA-256 differs")
                target = output_dir/"comparisons"/f"prompt_{prompt:06d}"/spec["id"]/f"seed_{seed:06d}.jpg"
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
                models[spec["id"]] = dict(file=str(target.relative_to(output_dir)),source=str(path),sha256=digest)
            if models:
                rows.append(dict(seed=seed,models=models))
        entry = dict(rows=rows,contact_sheets={})
        for group_name,order in (("primary",primary),("initializations",secondary)):
            if not order or not rows:
                continue
            width,height,gap,title,row_header = 416,240,12,68,28
            sheet = Image.new("RGB",(len(order)*(width+gap)+gap,title+len(rows)*(height+row_header+gap)+gap),"#eef2f6")
            draw = ImageDraw.Draw(sheet)
            draw.text((gap,7),f"prompt {prompt} | sigma={manifest['sigma']} | decoded frame index 4",fill="#172334",font=font)
            for col,name in enumerate(order):
                draw.text((gap+col*(width+gap),38),labels[name],fill="#172334",font=font)
            for row_index,row in enumerate(rows):
                y = title+gap+row_index*(height+row_header+gap)
                for col,name in enumerate(order):
                    x = gap+col*(width+gap)
                    draw.text((x,y),f"seed {row['seed']}",fill="#172334",font=font)
                    if name in row["models"]:
                        with Image.open(output_dir/row["models"][name]["file"]) as image:
                            image = image.convert("RGB"); image.thumbnail((width,height))
                            sheet.paste(image,(x+(width-image.width)//2,y+row_header+(height-image.height)//2))
            path = output_dir/"comparisons"/f"{group_name}_prompt_{prompt:06d}.jpg"
            sheet.save(path,quality=90)
            entry["contact_sheets"][group_name] = str(path.relative_to(output_dir))
        result["prompts"][str(prompt)] = entry
    target = output_dir/"comparisons/manifest.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def moments_table(item,order,labels):
    rows = "".join(f"<tr><td><strong>{html.escape(labels[name])}</strong></td><td>{format_number(item['models'][name]['mean_tensor_rms'])}</td><td>{format_number(item['models'][name]['mean_coordinate_variance_unbiased'])}</td><td>{format_number(item['models'][name]['total_variance_trace_unbiased'],6)}</td></tr>" for name in order)
    return f'<div class="table"><table><tr><th>模型</th><th>Mean tensor RMS</th><th>跨 noise 方差 V / D</th><th>整体总方差 V</th></tr>{rows}</table></div>'


def pairs_table(item,pairs,labels):
    rows = []
    for a,b in pairs:
        pair = display_pair(item,a,b)
        ci = pair.get("variance_ratio_bootstrap",{}).get("ci95")
        ci_text = f"[{format_number(ci[0],3)}, {format_number(ci[1],3)}]" if ci else "—"
        rows.append(f"<tr><td>{html.escape(labels[a])} / {html.escape(labels[b])}</td><td><strong>{format_number(pair['variance_ratio'])}</strong><small>95% CI {ci_text}</small></td><td>{format_number(pair['mean_difference_mse'])}</td><td>{format_number(pair['mmd2_paired_u'])}</td></tr>")
    return f'<div class="table"><table><tr><th>比较（分子 / 分母）</th><th>方差比及 95% CI</th><th>均值之差 MSE</th><th>RBF MMD²</th></tr>{"".join(rows)}</table></div>'


def gallery_html(prompt,order,labels,previews,group_name):
    entry = (previews or {}).get("prompts",{}).get(str(prompt),{})
    rows = entry.get("rows",[])
    if not rows:
        return "<small>预览尚未生成。</small>"
    columns = len(order)
    pieces = [f'<div class="model-head" style="grid-template-columns:repeat({columns},minmax(0,1fr))">'+"".join(f"<strong>{html.escape(labels[name])}</strong>" for name in order)+"</div>"]
    sheet = entry.get("contact_sheets",{}).get(group_name)
    if sheet:
        pieces.append(f'<p><a href="{html.escape(sheet,quote=True)}" target="_blank">打开完整对照图</a></p>')
    for row in rows:
        pieces.append(f'<p class="seed-label">seed {row["seed"]}</p><div class="image-row" style="grid-template-columns:repeat({columns},minmax(0,1fr))">')
        for name in order:
            image = row["models"].get(name)
            if image:
                path = html.escape(image["file"],quote=True)
                alt = html.escape(f"{labels[name]}, prompt {prompt}, seed {row['seed']}",quote=True)
                pieces.append(f'<a href="{path}" target="_blank"><img src="{path}" alt="{alt}" loading="lazy"></a>')
            else:
                pieces.append('<div class="empty-image">预览未生成</div>')
        pieces.append("</div>")
    return "".join(pieces)


def report_html(summary,previews=None):
    manifest = summary["manifest"]
    labels = {s["id"]:s["label"] for s in manifest["models"]}
    order,primary = [s["id"] for s in manifest["models"]],manifest["primary_models"]
    secondary = [name for name in order if name not in primary]
    reference = manifest.get("reference_model")
    main_pairs = [(name,reference) for name in primary if name != reference] if reference else list(combinations(primary,2))
    main_pair_sets = {frozenset(pair) for pair in main_pairs}
    extra_pairs = [(a,b) for a,b in combinations(order,2) if frozenset((a,b)) not in main_pair_sets]
    comparison_note = f'<p><strong>主比较的分母统一为 {html.escape(labels[reference])}。</strong>方差比小于 1 表示 student 的跨 noise 离散度小于该 teacher；MMD² 只表示分布差异。</p>' if reference else ""
    sections = []
    for item in summary["prompts"]:
        p = item["prompt_index"]
        title = f"{PROMPT_LABELS.get(p,'Prompt')} · #{p}"
        if item["n_seeds"]<2:
            sections.append(f"<section><h2>{title}</h2><p>等待至少 2 个所有模型共同完成的 noise。</p></section>")
            continue
        extra = ""
        if secondary or extra_pairs:
            other_moments = moments_table(item,secondary,labels) if secondary else ""
            other_images = f'<details><summary>其他参照模型同 seed 图片</summary>{gallery_html(p,secondary,labels,previews,"initializations")}</details>' if secondary else ""
            extra = f'<details><summary>其他参照模型及其余 {len(extra_pairs)} 对比较</summary>{other_moments}{pairs_table(item,extra_pairs,labels)}{other_images}</details>'
        sections.append(f"""<section><h2>{title}</h2><p>{html.escape(item.get('prompt') or '')}</p>
<p>共同 noise：<strong>{item['n_seeds']}</strong>；维度 D = {item['dimension']:,}。</p>
{moments_table(item,primary,labels)}{comparison_note}{pairs_table(item,main_pairs,labels)}
<small>所有模型配对共用 {html.escape(' / '.join(labels[x] for x in manifest['pilot_models']))} 的 seed {html.escape(', '.join(map(str,manifest['pilot_seeds'])))} 确定的固定带宽 ℓ² = {format_number(item['bandwidth']['squared'],6)}。</small>
<details><summary>看同 seed 图片（前 8 个 noise）</summary><small>预览显示解码 frame index 4；数值使用完整前三个 latent 帧。</small>{gallery_html(p,primary,labels,previews,'primary')}</details>{extra}</section>""")
    protocol_rows = []
    specs = {spec["id"]:spec for spec in manifest["models"]}
    for name in order:
        protocol,source = summary.get("sampling_protocols",{}).get(name,{}),summary.get("model_sources",{}).get(name,{})
        details = f"guidance={protocol.get('guidance','—')}；completed updates={protocol.get('completed_updates','—')}；initial shape={protocol.get('shape','—')}"
        download = specs[name].get("download",{})
        role_value = download.get("role",source.get("checkpoint_role"))
        role = f"；role={role_value}" if role_value else ""
        filename = download.get("filename",download.get("source_filename"))
        revision = download.get("revision",download.get("resolved_revision"))
        origin = "；".join(f"{key}={value}" for key,value in (("source",filename),("revision",revision)) if value)
        transform = specs[name].get("checkpoint_transform",source.get("checkpoint_transform"))
        if isinstance(transform,dict):
            transform = transform.get("note") or json.dumps(transform,ensure_ascii=False)
        extra = (f"<small>{html.escape(origin)}</small>" if origin else "")+(f"<small>checkpoint transform：{html.escape(str(transform))}</small>" if transform else "")
        protocol_rows.append(f"<tr><td>{html.escape(labels[name])}</td><td>{html.escape(str(protocol.get('sampler','—')))}<small>{html.escape(details)}</small></td><td class='path'>{html.escape(str(source.get('checkpoint_path','—')))}<small>weights={html.escape(str(source.get('weights','—')))}{html.escape(role)}</small>{extra}</td></tr>")
    done,total = sum(x["completed"] for x in summary["completeness"]),sum(x["expected"] for x in summary["completeness"])
    title = " / ".join(labels[name] for name in primary)
    status = "完整结果" if summary["complete"] else "临时结果 · 样本尚未齐全"
    reference_note = manifest.get("reference_note")
    reference_scope = f'<section><h2>参照模型与结论范围</h2><p>{html.escape(reference_note)}</p></section>' if reference_note else ""
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#172334;background:#f3f6f9;margin:0}}main{{max-width:1200px;margin:auto;padding:28px 20px}}h1{{font-size:27px;line-height:1.3}}h2{{font-size:21px;margin-top:0}}section{{background:white;border:1px solid #dce4ed;border-radius:12px;margin:22px 0;padding:24px}}p{{margin:10px 0}}.table{{overflow:auto;margin:18px 0}}table{{width:100%;border-collapse:collapse;text-align:left}}td,th{{padding:12px;border-bottom:1px solid #e3e9ef;vertical-align:top}}th{{background:#f2f6fa}}small{{display:block;color:#5c6d80;font-size:13px}}a{{color:#1764af}}.path{{font-size:13px;overflow-wrap:anywhere}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}.model-head,.image-row{{display:grid;gap:10px}}.model-head{{font-size:13px;margin-top:15px}}.image-row img{{display:block;width:100%;border-radius:5px}}.empty-image{{min-height:80px;background:#edf1f6;color:#7a8797;text-align:center}}.seed-label{{font-size:13px;font-weight:600;margin:16px 0 5px}}@media(max-width:700px){{main{{padding:16px 10px}}section{{padding:14px}}td,th{{font-size:13px;padding:9px}}.model-head{{font-size:10px}}}}
</style><main><h1>{html.escape(title)}</h1><p>{status} · {done}/{total} 个模型样本 · σ = {manifest['sigma']}</p>
<p>同 prompt、同初始 noise、同 noise level，比较预测 x₀ 的前三个 latent 帧；使用同一个原始 Wan VAE latent 空间。</p>{reference_scope}{''.join(sections)}
<section><h2>指标怎么读</h2><p><strong>整体方差：</strong>每个 noise 的完整 latent 展平为 xᵢ，先求平均张量 μ = (1/N) Σᵢ xᵢ；V = [1/(N−1)] Σᵢ ‖xᵢ−μ‖² = tr(Σ̂)。V/D 是按维度平均的同一离散度。方差比 &lt; 1 表示分子模型更集中。</p>
<p><strong>Mean：</strong>RMS = ‖μ‖/√D 描述平均张量大小；均值之差 MSE = ‖μₐ−μᵦ‖²/D 描述两个平均张量的距离。完整均值见 <a href="mean_tensors.npz">mean_tensors.npz</a>。</p>
<p><strong>MMD²：</strong>采用 RBF 核和 paired U 估计，同 seed 的跨模型 i=j 项也剔除；有限样本值可以略小于零。MMD 描述分布差异，不能单独判断哪个模型更 diverse。方差比使用 {summary['bootstrap_draws']} 次配对 seed bootstrap。</p>
<p>不对每个样本单独归一化。这些结果只描述当前早期 x₀ latent 分布，不能直接代表最终完整视频的语义多样性。不同 checkpoint 的采样参数与权重选择见下表。</p></section>
<details><summary>Checkpoint 与实际采样协议</summary><div class="table"><table><tr><th>模型</th><th>Sampler / 参数</th><th>Checkpoint</th></tr>{''.join(protocol_rows)}</table></div></details>
<p><a href="summary.json">全部模型 JSON</a> · <a href="moments.csv">各模型 moments</a> · <a href="pairwise.csv">全部配对 CSV</a> · <a href="https://www.jmlr.org/papers/v13/gretton12a.html">MMD 参考</a></p></main></html>"""


def write_report(output_dir,summary):
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("Report-only mode requires a diversity_models_v1 summary")
    path = output_dir/"comparisons/manifest.json"
    previews = json.loads(path.read_text()) if path.is_file() else None
    (output_dir/"report.html").write_text(report_html(summary,previews))


def write_outputs(output_dir,summary,means):
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_dir/"mean_tensors.npz",**means)
    (output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    moment_fields = ["prompt_index","n_seeds","model","label","mean_tensor_rms","variance_per_dimension","total_variance_trace","complete"]
    pair_fields = ["prompt_index","n_seeds","numerator","denominator","variance_ratio","ratio_ci95_low","ratio_ci95_high","reverse_variance_ratio","reverse_ratio_ci95_low","reverse_ratio_ci95_high","mean_difference_mse","mmd2_paired_u","mmd2_v_descriptive","rbf_bandwidth_squared","complete"]
    labels = {s["id"]:s["label"] for s in summary["manifest"]["models"]}
    with (output_dir/"moments.csv").open("w",newline="") as ms,(output_dir/"pairwise.csv").open("w",newline="") as ps:
        mw,pw = csv.DictWriter(ms,moment_fields),csv.DictWriter(ps,pair_fields)
        mw.writeheader()
        pw.writeheader()
        for item in summary["prompts"]:
            if item["n_seeds"]<2:
                continue
            identity = dict(prompt_index=item["prompt_index"],n_seeds=item["n_seeds"],complete=summary["complete"])
            for name,moments in item["models"].items():
                mw.writerow(dict(identity,model=name,label=labels[name],mean_tensor_rms=moments["mean_tensor_rms"],variance_per_dimension=moments["mean_coordinate_variance_unbiased"],total_variance_trace=moments["total_variance_trace_unbiased"]))
            for pair in item["pairs"].values():
                ci = pair.get("variance_ratio_bootstrap",{}).get("ci95") or [None,None]
                reverse_ci = pair.get("reverse_variance_ratio_bootstrap",{}).get("ci95") or [None,None]
                row = {field:pair[field] for field in ("numerator","denominator","variance_ratio","reverse_variance_ratio","mean_difference_mse","mmd2_paired_u","mmd2_v_descriptive")}
                pw.writerow(dict(identity,**row,ratio_ci95_low=ci[0],ratio_ci95_high=ci[1],reverse_ratio_ci95_low=reverse_ci[0],reverse_ratio_ci95_high=reverse_ci[1],rbf_bandwidth_squared=item["bandwidth"]["squared"]))
    (output_dir/"summary.csv").write_bytes((output_dir/"pairwise.csv").read_bytes())
    write_report(output_dir,summary)


def self_test():
    from scripts.diversity.summarize_diversity_fourway import compare_fourway
    rng = np.random.default_rng(310)
    a = rng.normal(size=(16,2,3)).astype(np.float32)
    order = ["cf_init","ode","ar","wan","cf","sgf"]
    arrays = {name:a.astype(np.float64)*(i+1) for i,name in enumerate(order)}
    seeds = list(range(16))
    result = compare_models(arrays,seeds,order,bootstrap_draws=100)
    assert len(result["models"])==6 and len(result["pairs"])==15
    old = compare_fourway({name:arrays[name] for name in order[:4]},seeds,bootstrap_draws=100)
    for name,pair in old["pairs"].items():
        for field in ("variance_ratio","mean_difference_mse","mmd2_paired_u"):
            assert math.isclose(result["pairs"][name][field],pair[field],abs_tol=1e-12)
    assert math.isclose(display_pair(result,"cf","ar")["variance_ratio"],25/9,rel_tol=1e-12)
    assert np.allclose(display_pair(result,"cf","ar")["variance_ratio_bootstrap"]["ci95"],[25/9,25/9])
    same = compare_models(dict.fromkeys(order,a),seeds,order,bootstrap_draws=20)
    assert all(abs(pair["mmd2_paired_u"])<1e-12 and math.isclose(pair["variance_ratio"],1,abs_tol=1e-12) for pair in same["pairs"].values())
    varied = {name:rng.normal(size=a.shape).astype(np.float32) for name in order}
    reordered = compare_models(varied,seeds,["cf","sgf","ar","wan","cf_init","ode"],bootstrap_draws=100)
    legacy = compare_fourway({name:varied[name] for name in order[:4]},seeds,bootstrap_draws=100)
    for pair in legacy["pairs"].values():
        shown = display_pair(reordered,pair["numerator"],pair["denominator"])
        for field in ("variance_ratio","mean_difference_mse","mmd2_paired_u"):
            assert math.isclose(shown[field],pair[field],abs_tol=1e-12)
        assert np.allclose(shown["variance_ratio_bootstrap"]["ci95"],pair["variance_ratio_bootstrap"]["ci95"],rtol=1e-12,atol=1e-12)
    with tempfile.TemporaryDirectory() as temporary:
        from PIL import Image
        root = Path(temporary)
        specs = []
        for name in order:
            spec = dict(id=name,label=name,root=str(root),directory=name,checkpoint=str(root/f"{name}.pt"),weights="generator_ema" if name=="sgf" else "generator")
            if name in ("cf_init","ode","cf","sgf"):
                spec.update(protocol_reference="ar",transition_noise_reference="ar")
            if name=="sgf":
                spec.update(download=dict(role="official final EMA",filename="model.pt",revision="test-revision"),checkpoint_transform=dict(note="Remove wrapper prefix only; tensor values unchanged."))
            specs.append(spec)
            for seed in range(8):
                directory = root/name/"prompt_000042"/f"seed_{seed:06d}"
                directory.mkdir(parents=True)
                latent = a[seed]
                np.savez_compressed(directory/"features.npz",latent=latent)
                protocol = dict(model=name,prompt_index=42,prompt="test",seed=seed,sigma=.9375,
                                feature_space=FEATURE_SPACE,feature_shape=list(latent.shape),
                                sampler="same causal",completed_updates=1,
                                model_source=dict(checkpoint_path=spec["checkpoint"],weights=spec["weights"]))
                record = dict(protocol,complete=True,schema_version="diversity_moments_sampling_v1",shape=list(latent.shape),
                              noise_sha256=f"noise-{seed}",initial_noise_sha256=f"noise-{seed}",
                              transition_noise_sha256={"0:1":f"transition-{seed}"},protocol=protocol,
                              protocol_sha256=canonical_hash(protocol),latent_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),
                              artifact_sha256={"features.npz":hashlib.sha256((directory/"features.npz").read_bytes()).hexdigest()})
                if seed==0:
                    Image.new("RGB",(416,240),(30,90,140)).save(directory/"preview.jpg")
                    record["artifact_sha256"]["preview.jpg"]=hashlib.sha256((directory/"preview.jpg").read_bytes()).hexdigest()
                (directory/"record.json").write_text(json.dumps(record))
        manifest = validate_manifest(dict(models=specs,primary_models=["cf","sgf","ar","wan"],prompt_indices=[42],seeds=list(range(8)),feature_shape=[2,3]))
        groups,coverage,complete,sources,protocols = load_models(manifest)
        assert complete and len(groups[0]["seeds"])==8
        # SGF's independently declared EMA weights can use the same sampler.
        altered = json.loads(json.dumps(manifest))
        altered["models"][-1]["weights"]="generator"
        try:
            load_models(altered)
        except ValueError as error:
            assert "weights identity" in str(error)
        else:
            raise AssertionError("Expected checkpoint weights identity rejection")
        out = root/"out"
        build_previews(out,manifest)
        summary = dict(schema_version=SCHEMA,manifest=manifest,complete=True,completeness=coverage,
                       model_sources=sources,sampling_protocols=protocols,bootstrap_draws=100,
                       prompts=[dict(prompt_index=42,prompt="test",**result)])
        write_outputs(out,summary,{"test":a.mean(axis=0)})
        assert "其他参照模型及其余 9 对比较" in (out/"report.html").read_text()
        assert "test-revision" in (out/"report.html").read_text() and "tensor values unchanged" in (out/"report.html").read_text()
        for path in root.glob("*/prompt_*/seed_*/features.npz"):
            path.unlink()
        write_report(out,json.loads((out/"summary.json").read_text()))
    print("Generic six-model metrics, old-metric compatibility, independent weights/sampler identity, galleries and report-only self-tests passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-manifest",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-draws",type=int,default=2000)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--skip-previews",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    output_dir = repository_path(args.output_dir)
    if args.report_only:
        write_report(output_dir,json.loads((output_dir/"summary.json").read_text()))
        print(output_dir/"report.html")
        return
    if args.models_manifest is None or args.bootstrap_draws<0:
        parser.error("Need --models-manifest and nonnegative bootstrap draws")
    previous = output_dir/"summary.json"
    if previous.is_file() and json.loads(previous.read_text()).get("schema_version") != SCHEMA:
        parser.error("Use a new output directory to preserve previous reports")
    manifest = validate_manifest(json.loads(repository_path(args.models_manifest).read_text()))
    groups,coverage,complete,sources,protocols = load_models(manifest,args.allow_partial)
    summary = dict(schema_version=SCHEMA,generated_at=datetime.now(timezone.utc).isoformat(),manifest=manifest,
                   complete=complete,completeness=coverage,model_sources=sources,sampling_protocols=protocols,
                   model_order=[s["id"] for s in manifest["models"]],primary_models=manifest["primary_models"],
                   bootstrap_draws=args.bootstrap_draws,prompts=[])
    means = {}
    order = [s["id"] for s in manifest["models"]]
    for group in groups:
        item = {k:v for k,v in group.items() if k!="arrays"}
        if len(group["seeds"])>=2:
            item.update(compare_models(group["arrays"],group["seeds"],order,manifest["pilot_models"],manifest["pilot_seeds"],args.bootstrap_draws))
            for name in order:
                means[f"prompt_{group['prompt_index']:06d}_{name}"]=group["arrays"][name].mean(axis=0,dtype=np.float64)
            print(f"SUMMARIZED prompt={group['prompt_index']} seeds={len(group['seeds'])} models={len(order)}",flush=True)
        else:
            item["n_seeds"]=len(group["seeds"])
        summary["prompts"].append(item)
    if not args.skip_previews:
        build_previews(output_dir,manifest)
    write_outputs(output_dir,summary,means)
    print(json.dumps({"complete":complete,"report":str(output_dir/"report.html"),
                      "paired_seeds":{item["prompt_index"]:item["n_seeds"] for item in summary["prompts"]}},ensure_ascii=False))


if __name__=="__main__":
    main()
