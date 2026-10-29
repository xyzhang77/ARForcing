"""Build an offline report for the four denoising calls of a fixed AR rollout.

All plotted stages are clean predictions, not decoded noisy solver states.  The
four AR stages are paired with Wan probes after 0, 12, 24 and 36 Euler updates.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAGES = ("step1", "step2", "step3", "step4", "final")
AR_KEYS = ("x0_step1", "x0_step2", "x0_step3", "x0_step4", "final")
WAN_KEYS = ("first", "quarter", "half", "threequarter", "final")
SIGMAS = (1.0, 0.9375, 5 / 6, 0.625, 0.0)
FRAMES = (0, 4, 8)
DIFF_CAP = 0.25
COLORS = {"ar": "#13796c", "wan": "#b05c17"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_number(value):
    if value is None:
        return "—"
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.5f}" if abs(value) >= 0.001 else f"{value:.3g}"


def load_views(path, keys):
    record = read_json(path / "record.json")
    if not record.get("complete"):
        raise ValueError(f"Incomplete record: {path}")
    expected = record.get("artifact_sha256", {}).get("views.npz")
    if not expected or sha256(path / "views.npz") != expected:
        raise ValueError(f"Missing or changed views: {path}")
    with np.load(path / "views.npz", allow_pickle=False) as arrays:
        result = {stage: arrays[key].copy() for stage, key in zip(STAGES, keys)}
    for stage, value in result.items():
        if value.shape != (3, 240, 416, 3) or value.dtype != np.uint8:
            raise ValueError(f"Invalid view schema {path}/{stage}: {value.shape}/{value.dtype}")
    return record, result


def layout_rgb(frame):
    """The same BOX 26x15 reduction used by the numerical layout features."""
    return np.asarray(Image.fromarray(frame).resize((26, 15), Image.Resampling.BOX))


def heatmap(values):
    """A fixed, shared sequential color map: zero is dark; 0.25 RGB is yellow."""
    colors = np.array([[15, 23, 42], [35, 86, 123], [18, 143, 139],
                       [123, 188, 93], [255, 224, 102]], dtype=np.float64)
    position = np.clip(values / DIFF_CAP, 0, 1) * (len(colors) - 1)
    lower = np.minimum(position.astype(int), len(colors) - 2)
    fraction = position - lower
    return np.round(colors[lower] * (1 - fraction[..., None])
                    + colors[lower + 1] * fraction[..., None]).astype(np.uint8)


def caption_font():
    try:
        return ImageFont.load_default(size=17)
    except TypeError:
        return ImageFont.load_default()


def stage_strip(views, frame_index, mode):
    width, height, left, gutter, title, row_gap = 416, 240, 58, 8, 33, 12
    canvas = Image.new("RGB", (left + 5 * (width + gutter), 2 * (height + title + row_gap)), "#f8fafc")
    draw, font = ImageDraw.Draw(canvas), caption_font()
    for row, model in enumerate(("ar", "wan")):
        y = row * (height + title + row_gap)
        draw.text((9, y + 115), model.upper(), font=font, fill=COLORS[model])
        for column, stage in enumerate(STAGES):
            x = left + column * (width + gutter)
            label = f"{stage} x0 | sigma_in={SIGMAS[column]:.6f}"
            if model == "wan" and stage != "final":
                label += f" | {12 * column} updates"
            if stage == "final":
                label = "final x0 = AR step4" if model == "ar" else "final x0 | after 48 Euler updates"
            draw.text((x + 3, y + 7), label, font=font, fill="#17202c")
            array = views[model][stage][frame_index]
            if mode == "layout":
                array = layout_rgb(array)
            tile = Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y + title))
    return canvas


def difference_strip(views, frame_index, mode):
    width, height, left, gutter, title, row_gap = 416, 240, 58, 8, 33, 12
    canvas = Image.new("RGB", (left + 4 * (width + gutter), 2 * (height + title + row_gap)), "#f8fafc")
    draw, font = ImageDraw.Draw(canvas), caption_font()
    for row, model in enumerate(("ar", "wan")):
        y = row * (height + title + row_gap)
        draw.text((9, y + 115), model.upper(), font=font, fill=COLORS[model])
        for column, (before, after) in enumerate(zip(STAGES[:-1], STAGES[1:])):
            x = left + column * (width + gutter)
            label = f"{before} -> {after}"
            if model == "ar" and after == "final":
                label += " | identical by construction"
            draw.text((x + 3, y + 7), label, font=font, fill="#17202c")
            a, b = views[model][before][frame_index], views[model][after][frame_index]
            if mode == "layout":
                a, b = layout_rgb(a), layout_rgb(b)
            difference = np.mean(np.abs(a.astype(float) - b.astype(float)), axis=-1) / 255
            tile = Image.fromarray(heatmap(difference)).resize((width, height), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y + title))
    return canvas


def chart_svg(series, labels, title, y_title):
    """A standalone SVG with shared axes and explicit zero values."""
    width, height = 760, 340
    left, right, top, bottom = 90, 24, 50, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [float(v) for item in series for v in item["values"] if v is not None]
    ceiling = max(values, default=0) * 1.12 or 1.0
    x = lambda index: left + index * plot_width / max(1, len(labels) - 1)
    y = lambda value: top + plot_height * (1 - value / ceiling)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{left}" y="25" font-size="17" fill="#17202c">{html.escape(title)}</text>']
    for fraction in np.linspace(0, 1, 5):
        value, py = fraction * ceiling, y(fraction * ceiling)
        elements.extend([f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#e2e8f0"/>',
                         f'<text x="{left-9}" y="{py+4}" text-anchor="end" font-size="12" fill="#64748b">{display_number(value)}</text>'])
    for index, label in enumerate(labels):
        elements.append(f'<text x="{x(index)}" y="{height-bottom+24}" text-anchor="middle" font-size="12" fill="#475569">{html.escape(label)}</text>')
    for index, item in enumerate(series):
        points = [(x(i), y(value)) for i, value in enumerate(item["values"]) if value is not None]
        point_string = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        elements.append(f'<polyline points="{point_string}" fill="none" stroke="{item["color"]}" stroke-width="2.5"/>')
        for i, value in enumerate(item["values"]):
            if value is None:
                continue
            elements.append(f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="4" fill="{item["color"]}"><title>{html.escape(item["label"])} {html.escape(labels[i])}: {value:.8g}</title></circle>')
        legend_x = left + index * 145
        elements.append(f'<text x="{legend_x}" y="{height-13}" font-size="13" fill="{item["color"]}">● {html.escape(item["label"])}</text>')
    elements.append(f'<text transform="translate(17,{top+plot_height/2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#475569">{html.escape(y_title)}</text></svg>')
    return "".join(elements)


def metric_table(prompt):
    rows = []
    for model in ("ar", "wan"):
        for item in prompt["models"][model]["stage_metrics"]:
            retention = item["to_final"]
            diversity = item["diversity"]["metrics"].get("mse", {}).get("mean")
            identity = item.get("construction_identity_with_final", False)
            label = item["stage"] + (" †" if identity else "")
            retrieval = retention.get("retrieval_accuracy")
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in (
                model.upper(), html.escape(label), display_number(item.get("sigma")),
                display_number(retention["matched_mse"]),
                display_number(retention.get("normalized_to_final_interseed_mse")),
                f"{100*retrieval:.1f}%" if retrieval is not None else "—",
                display_number(diversity))) + "</tr>")
    return '<div class="table-scroll"><table><thead><tr><th>模型</th><th>阶段</th><th>输入 σ</th><th>到 final 的布局 MSE</th><th>相对跨 seed final 距离</th><th>final seed 检索率</th><th>跨 seed 布局 MSE</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def adjacent_table(prompt):
    rows = []
    for model in ("ar", "wan"):
        for item in prompt["models"][model]["adjacent_metrics"]:
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in (
                model.upper(), html.escape(f'{item["from_stage"]} → {item["to_stage"]}'),
                display_number(item["lowpass_mse"]),
                display_number(item["high_frequency_rgb_mse"]),
                display_number(item["rgb_mse"]),
                display_number(item.get("normalized_to_final_interseed_mse")))) + "</tr>")
    return '<div class="table-scroll"><table><thead><tr><th>模型</th><th>相邻阶段</th><th>布局变化 MSE</th><th>高频残差变化 MSE</th><th>完整 RGB 变化 MSE</th><th>布局变化 / final 跨 seed 距离</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def cross_table(prompt):
    rows = []
    for item in prompt.get("cross_model", []):
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in (
            html.escape(item["stage"]), display_number(item.get("sigma")),
            display_number(item["lowpass_mse"]), display_number(item["high_frequency_rgb_mse"]),
            display_number(item["rgb_mse"]))) + "</tr>")
    return '<div class="table-scroll"><table><thead><tr><th>阶段</th><th>输入 σ</th><th>AR/Wan 布局 MSE</th><th>高频残差 MSE</th><th>完整 RGB MSE</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def write_prompt_charts(prompt, assets):
    output = {}
    by_model = {model: {entry["stage"]: entry for entry in prompt["models"][model]["stage_metrics"]}
                for model in ("ar", "wan")}
    for name, stages in (("distance_all", STAGES), ("distance_late", STAGES[1:])):
        series = [{"label": model.upper(), "color": COLORS[model],
                   "values": [by_model[model][stage]["to_final"].get("normalized_to_final_interseed_mse")
                              for stage in stages]} for model in ("ar", "wan")]
        title = "各阶段到自身 final 的布局距离" if name == "distance_all" else "第 2–4 步到自身 final 的布局距离"
        filename = f'prompt_{prompt["prompt_index"]:06d}_{name}.svg'
        (assets / filename).write_text(chart_svg(series, list(stages), title, "MSE / final inter-seed MSE"), encoding="utf-8")
        output[name] = "assets/" + filename
    pairs = list(zip(STAGES[:-1], STAGES[1:]))
    for name, field, title in (("adjacent_layout", "lowpass_mse", "相邻阶段的布局变化"),
                                ("adjacent_highfreq", "high_frequency_rgb_mse", "相邻阶段的高频 RGB 残差变化")):
        series = []
        for model in ("ar", "wan"):
            mapping = {(entry["from_stage"], entry["to_stage"]): entry[field]
                       for entry in prompt["models"][model]["adjacent_metrics"]}
            series.append({"label": model.upper(), "color": COLORS[model],
                           "values": [mapping.get(pair) for pair in pairs]})
        filename = f'prompt_{prompt["prompt_index"]:06d}_{name}.svg'
        (assets / filename).write_text(chart_svg(series, [f"{a}→{b}" for a, b in pairs], title, "RGB MSE"), encoding="utf-8")
        output[name] = "assets/" + filename
    return output


CSS = """
:root{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#17202c;background:#f2f5f7;line-height:1.6}*{box-sizing:border-box}body{margin:0}main{max-width:1480px;padding:32px 28px 80px;margin:auto}a{color:#126b9c}h1{font-size:30px;line-height:1.3;margin:0 0 14px}h2{font-size:23px;margin:28px 0 12px}h3{font-size:18px;margin:18px 0 10px}p{margin:10px 0}.subtitle,.muted{color:#596b7d}.lead{font-size:17px}.card{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:22px;margin:20px 0;box-shadow:0 3px 14px #17202c05}.note{background:#eef6f4;border-left:4px solid #13796c;padding:14px 18px;border-radius:4px;margin:16px 0}.warning{background:#fff5e9;border-left-color:#b05c17}.controls{display:flex;gap:18px;align-items:end;flex-wrap:wrap;background:#f4f7f9;padding:14px;border-radius:8px;margin:14px 0}label{font-size:13px;color:#53677b;display:flex;flex-direction:column;gap:4px}select{font:inherit;font-size:15px;padding:8px 12px;background:white;border:1px solid #bdcbd6;border-radius:6px;min-width:140px}.media{width:100%;display:block;height:auto;border:1px solid #dfe5eb;border-radius:5px}.media-wrap{overflow-x:auto}.legend{display:flex;align-items:center;gap:10px;font-size:13px;color:#596b7d}.bar{height:13px;width:200px;background:linear-gradient(90deg,#0f172a,#23567b,#128f8b,#7bbc5d,#ffe066);border-radius:2px}.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart-grid img{width:100%;height:auto;border:1px solid #edf1f5;border-radius:8px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap;margin:12px 0}th,td{text-align:right;padding:8px 12px;border-bottom:1px solid #e2e8f0}th{background:#f4f7f9;color:#43556a}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f7fafb}.chip{padding:3px 9px;background:#e7eeef;border-radius:20px;font-size:12px;display:inline-block;margin-right:6px}details{border-top:1px solid #e2e8f0;margin-top:18px;padding-top:12px}summary{cursor:pointer;font-weight:600}.prompt-title{margin-top:0}.code{font-family:ui-monospace,monospace;font-size:12px}#case-meta{font-size:14px;color:#53677b}.download-row{display:flex;flex-wrap:wrap;gap:18px;margin-top:14px}.noscript-sheet{max-width:100%}@media(max-width:800px){main{padding:20px 12px}.card{padding:16px}.chart-grid{grid-template-columns:1fr}h1{font-size:25px}select{min-width:110px}.controls{gap:10px}}@media print{.controls{display:none}.card{break-inside:avoid;box-shadow:none}.chart-grid{grid-template-columns:1fr 1fr}body{background:white}}
"""


SCRIPT = """
const data=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id);
const promptSelect=$('prompt-select'), seedSelect=$('seed-select'), frameSelect=$('frame-select'), modeSelect=$('mode-select');
function updateSeeds(){const item=data.prompts.find(p=>String(p.prompt_index)===promptSelect.value);const old=seedSelect.value;seedSelect.replaceChildren(...item.seeds.map(seed=>new Option(String(seed),String(seed))));if(item.seeds.map(String).includes(old))seedSelect.value=old;render();}
function render(){const prompt=data.prompts.find(p=>String(p.prompt_index)===promptSelect.value);const seed=seedSelect.value;const frame=frameSelect.value;const mode=modeSelect.value;const item=prompt.cases[seed][frame];$('filmstrip').src=item[mode];$('filmstrip-link').href=item[mode];$('difference').src=item['diff_'+mode];$('difference-link').href=item['diff_'+mode];$('case-meta').textContent=`Prompt ${prompt.prompt_index} · seed ${seed} · decoded frame ${frame} · ${mode==='rgb'?'完整 RGB':'26×15 低通布局'} · 上：AR，下：Wan`;$('prompt-text').textContent=prompt.prompt;}
promptSelect.addEventListener('change',updateSeeds);for(const element of [seedSelect,frameSelect,modeSelect])element.addEventListener('change',render);updateSeeds();
"""


def assessment_html(assessment):
    if assessment is None:
        return ""
    sections = assessment.get("sections")
    if not isinstance(sections, list):
        raise ValueError("assessment.json requires sections: [{title: str, text: str}]")
    paragraphs = []
    for section in sections:
        if not isinstance(section, dict) or not isinstance(section.get("title"), str) or not isinstance(section.get("text"), str):
            raise ValueError("Each assessment section requires string title and text")
        paragraphs.append(f'<h3>{html.escape(section["title"])}</h3><p>'
                          + html.escape(section["text"]).replace("\n", "<br>") + "</p>")
    return '<section class="card" id="observations"><h2>这批样本的观察</h2>' + "".join(paragraphs) + "</section>"


def build_report(root):
    root = Path(root).resolve()
    destination = root / "denoising_steps"
    summary_path = destination / "summary.json"
    summary = read_json(summary_path)
    prompts = summary["per_prompt"]
    assessment_path = destination / "assessment.json"
    assessment = read_json(assessment_path) if assessment_path.is_file() else None
    observations = assessment_html(assessment)
    if not prompts:
        raise ValueError("No completed prompts in denoising-step summary")
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = {"summary_sha256": sha256(summary_path), "script_sha256": sha256(__file__),
                "assessment_sha256": sha256(assessment_path) if assessment is not None else None,
                "assessment": assessment,
                "difference_metric": "mean absolute RGB difference / 255, shared cap 0.25",
                "prompts": []}
    for prompt in prompts:
        index = int(prompt["prompt_index"])
        item = {"prompt_index": index, "prompt": prompt["prompt"], "seeds": prompt["seeds"], "cases": {},
                "charts": write_prompt_charts(prompt, assets)}
        for seed in prompt["seeds"]:
            suffix = Path(f"prompt_{index:06d}") / f"seed_{int(seed):06d}"
            ar_record, ar = load_views(destination / "ar" / suffix, AR_KEYS)
            wan_record, wan = load_views(root / "wan" / suffix, WAN_KEYS)
            for model, record in (("ar", ar_record), ("wan", wan_record)):
                if int(record["prompt_index"]) != index or int(record["seed"]) != int(seed):
                    raise ValueError(f"Wrong sample identity: {model}/{suffix}")
            if not np.array_equal(ar["step4"], ar["final"]):
                raise ValueError(f"AR step4 must equal first-chunk final: {suffix}")
            views = {"ar": ar, "wan": wan}
            case_assets = {}
            for frame_index, frame in enumerate(FRAMES):
                frame_assets = {}
                for mode in ("rgb", "layout"):
                    stem = f"prompt_{index:06d}_seed_{int(seed):06d}_frame_{frame}_{mode}"
                    image_name = stem + (".jpg" if mode == "rgb" else ".png")
                    kwargs = {"quality": 90, "subsampling": 0} if mode == "rgb" else {"optimize": True}
                    stage_strip(views, frame_index, mode).save(assets / image_name, **kwargs)
                    diff_name = stem + "_diff.png"
                    difference_strip(views, frame_index, mode).save(assets / diff_name, optimize=True)
                    frame_assets[mode] = "assets/" + image_name
                    frame_assets["diff_" + mode] = "assets/" + diff_name
                case_assets[str(frame)] = frame_assets
            item["cases"][str(seed)] = case_assets
        manifest["prompts"].append(item)
    manifest_path = destination / "visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    default_prompt = manifest["prompts"][0]
    default_case = default_prompt["cases"][str(default_prompt["seeds"][0])]["4"]
    options = "".join(f'<option value="{p["prompt_index"]}">Prompt {p["prompt_index"]}</option>' for p in manifest["prompts"])
    default_seeds = "".join(f'<option value="{seed}">{seed}</option>' for seed in default_prompt["seeds"])
    prompt_sections = []
    for prompt, visual in zip(prompts, manifest["prompts"]):
        charts = visual["charts"]
        chart_html = "".join(f'<a href="{charts[name]}" target="_blank"><img loading="lazy" src="{charts[name]}" alt="{label}"></a>'
                             for name, label in (("distance_all", "各阶段到 final 的布局距离"),
                                                 ("distance_late", "第2至4步到 final 的布局距离"),
                                                 ("adjacent_layout", "相邻阶段布局变化"),
                                                 ("adjacent_highfreq", "相邻阶段高频残差变化")))
        prompt_sections.append(f'''<section class="card" id="prompt-{prompt['prompt_index']}">
<h2 class="prompt-title">Prompt {prompt['prompt_index']} · {len(prompt['seeds'])} 个 seed</h2>
<p class="muted">{html.escape(prompt['prompt'])}</p><div class="chart-grid">{chart_html}</div>
<p class="muted">曲线与下表均对同一 prompt 的 seed 取平均，并同时使用 decoded frame 0/4/8。不同 prompt 不合并为独立重复样本。AR 的 step4 到 final 距离为 0 是构造事实。</p>
<h3>每步保留了多少最终布局</h3>{metric_table(prompt)}
<p class="muted">相对距离 = 同 seed 的阶段→final 布局 MSE / 不同 seed 的 final→final 平均布局 MSE。0 表示与自身 final 相同，数值不表示“结构确定百分比”。检索率是在本 prompt 的 8 个 final 中，以布局 MSE 找到对应 seed 的比例。</p>
<h3>相邻步骤改变了什么</h3>{adjacent_table(prompt)}
<p class="muted">布局 MSE 比较 BOX 26×15 RGB；高频残差是完整 RGB 减去低通图经双线性放大的结果。两者均会受到颜色、亮度和 VAE 解码影响，不等同于语义结构与语义细节，且不能直接相加恢复完整 RGB MSE。</p>
<details><summary>AR 与 Wan 同 seed、同 σ 的直接差异</summary>{cross_table(prompt)}
<p class="muted">同 seed 只配对初始噪声。DMD 没有强制 teacher/student 的逐样本构图一致；跨模型差异不能单独解释为蒸馏误差。</p></details></section>''')
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    sample_count = sum(len(p["seeds"]) for p in prompts)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AR / Wan 第 2–4 步与相邻变化</title><style>{CSS}</style></head>
<body><main><header><p class="subtitle">固定四步轨迹 · 同 σ 对照 · 首个 chunk</p><h1>第 2、3、4 步怎样改变图片</h1>
<p class="lead">沿用同一批 {len(prompts)} 个 prompt × {len(prompts[0]['seeds'])} 个 seed，比较 AR 四次去噪调用及 Wan 的对应噪声阶段。这里分析同一条轨迹的中间预测，没有重新设置总采样步数。</p>
<p><span class="chip">{sample_count} 对样本</span><span class="chip">decoded frames 0 / 4 / 8</span><span class="chip">全部离线可读</span></p>
<div class="download-row"><a href="../report.html">原首步分析报告</a><a href="summary.json">完整统计 JSON</a><a href="summary.csv">指标 CSV</a><a href="visual_manifest.json">图像清单</a></div></header>
{observations}
<section class="card"><h2>先明确每一列的含义</h2><div class="table-scroll"><table><thead><tr><th>列</th><th>step1</th><th>step2</th><th>step3</th><th>step4</th><th>final</th></tr></thead><tbody><tr><td>调用前输入 σ</td><td>1</td><td>0.9375</td><td>0.833333</td><td>0.625</td><td>0（输出域）</td></tr><tr><td>AR 调用序号</td><td>1</td><td>2</td><td>3</td><td>4</td><td>就是第 4 次输出</td></tr><tr><td>Wan 已完成 Euler 更新数</td><td>0</td><td>12</td><td>24</td><td>36</td><td>48</td></tr></tbody></table></div>
<div class="note">每张阶段图片都是在该输入 σ 下预测的干净 latent（x̂₀）的独立 VAE 解码。图片本身不是对应噪声水平的 state；因此不能把图像看起来更清晰直接解释成残留噪声更少。</div>
<div class="note warning">AR step4 就是首 chunk 的 final，所以它与 final 的零误差不能作为独立收敛证据。Wan 的 step4 仍是第 36 次更新后的 clean probe，距离 final 还有 12 次更新；最后单列 final。</div></section>
<section class="card"><h2>逐个 seed 查看阶段变化</h2><p id="prompt-text" class="muted">{html.escape(default_prompt['prompt'])}</p>
<div class="controls"><label>Prompt<select id="prompt-select">{options}</select></label><label>Seed<select id="seed-select">{default_seeds}</select></label><label>Decoded frame<select id="frame-select"><option value="4">frame 4</option><option value="0">frame 0</option><option value="8">frame 8</option></select></label><label>显示方式<select id="mode-select"><option value="rgb">完整 RGB</option><option value="layout">低通布局（26×15）</option></select></label></div>
<p id="case-meta">Prompt {default_prompt['prompt_index']} · seed {default_prompt['seeds'][0]} · decoded frame 4 · 上：AR，下：Wan</p>
<div class="media-wrap"><a id="filmstrip-link" href="{default_case['rgb']}" target="_blank"><img id="filmstrip" class="media" src="{default_case['rgb']}" alt="AR 与 Wan 的 step1、step2、step3、step4、final 阶段图；上方AR，下方Wan"></a></div>
<h3>相邻变化的绝对差分</h3><p class="muted">从左到右：1→2、2→3、3→4、4→final。上方 AR，下方 Wan。所有 prompt、seed、帧和模型共用下方色阶。</p>
<div class="legend"><span>0</span><span class="bar"></span><span>0.25 及以上</span><span>逐像素 RGB 绝对差的通道均值，RGB 归一化为 0–1</span></div>
<div class="media-wrap"><a id="difference-link" href="{default_case['diff_rgb']}" target="_blank"><img id="difference" class="media" src="{default_case['diff_rgb']}" alt="相邻阶段的绝对RGB差分图，共用0至0.25色阶"></a></div><p class="muted">点击图片可打开原始大图。切换到低通布局会先按同一 26×15 BOX 特征降采样，再画绝对差；两种显示的色阶相同。</p>
<noscript><p>当前脚本被禁用，显示第一个 prompt / seed 的 frame 4。其他样本可通过图像清单访问。</p></noscript></section>
{''.join(prompt_sections)}
<section class="card"><h2>如何解释这些变化</h2><p>阶段→自身 final 距离描述后续还会改变多少；相邻变化描述这一步实际改了多少；跨 seed 距离描述这批样本之间有多不同。三者一起看，可以区分较早形成布局、后续继续调整，以及样本间差异的变化。</p><p>同 σ 只对齐名义加噪尺度。AR 通过 clean prediction 后重新加噪进入下一步，Wan 则沿 Euler 轨迹更新。两个模型即使共享初始噪声，也不要求中间结果或最终构图一致。</p><p>结果范围为本批三个 prompt、每个八个 seed 和首 chunk。低通 RGB 不是语义距离，跨 seed 方差也不是语义模式数量。训练 teacher、CFG、attention 上下文和采样算法差异需要另做对照，当前数据不能单独归因某个因素。</p><p class="code">summary SHA256: {manifest['summary_sha256']}<br>report builder SHA256: {manifest['script_sha256']}</p></section>
</main><script type="application/json" id="report-data">{payload}</script><script>{SCRIPT}</script></body></html>'''
    temporary = destination / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(destination / "report.html")
    return {"report": str(destination / "report.html"), "n_paired_cases": sample_count,
            "n_image_assets": sample_count * len(FRAMES) * 4, "n_chart_assets": len(prompts) * 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Original first_step_noise experiment directory")
    args = parser.parse_args()
    print(json.dumps(build_report(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
