"""Summarize paired local-sigma interventions and render synchronized previews.

Numeric pixel metrics come from the runner's decoded float [0, 1] videos.
Quantized uint8 NPZ artifacts are used only to render the display videos.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np


SCALES = (0.75, 0.5, 0.25, 0.0)
LIMITATIONS = [
    "干预只降低选定 patch 的实际 sigma；模型接收的全局 timestep、初始噪声和后续噪声均保持配对。这里的强弱指局部干预幅度，而不是基础噪声大小。",
    "sigma × 0 表示将该 patch 替换为模型前一步预测，不是真实无噪声图像，也不是 ground truth。大幅干预可能偏离模型熟悉的输入分布。",
    "每个 case 固定使用上一轮细网格实验的 Top-2 和 2 个随机候选；强度间比较不重新挑选 source。Top 候选仍来自同一 case，结果不是独立验证或全网格扫描。",
    "像素 RMSE/MAE 由编码 MP4 前的 decoded float [0,1] 全视频计算；latent next-chunk MSE 仅度量未被直接干预的下一 chunk。两种度量的单位和目标域不同。",
    "展示视频使用 uint8 NPZ 缩小至每格 416×240；差分面板为逐 RGB 通道 abs(sigma×0 − baseline) × 固定增益，并截断至显示范围，不按帧或样本自动拉伸。它不是用于正式像素统计的 float 视频。",
    "位置示意图取基线帧索引 min(source_frame×4, T−1)，青色框仅为 latent patch 的空间投影，并非精确 VAE 感受野；框不是物体跟踪，也不是各输出帧的实际干预边界。",
    "较大的输出变化不代表质量改善，不能据此证明 semantic 机制或决定性的 anchor；48 个候选共享 12 个样本条件，并非 48 个独立样本。",
]


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return "—" if value is None else f"{value:.4g}"


def stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": float(np.median(values)), "min": min(values), "max": max(values)}


def stat_text(value):
    return f"{number(value['median'])} [{number(value['min'])}, {number(value['max'])}]"


def ratio(value, reference):
    return float(value / reference) if reference > 0 else None


def relative_url(path, output_dir):
    return quote(os.path.relpath(path, output_dir).replace(os.sep, "/"), safe="/")


def artifact_path(case_dir, artifacts, key, default=None):
    name = artifacts.get(key, default)
    if not name:
        raise ValueError(f"Missing artifact {key} in {case_dir}")
    path = Path(name)
    return path if path.is_absolute() else case_dir / path


def scalar(value, label):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative finite metric {label}: {value}")
    return result


def load_summary(input_dir, output_dir, diff_gain):
    cases = []
    for path in sorted(input_dir.glob("*/strength_metrics.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        metadata = raw["metadata"]
        records_by_source = {}
        for record in raw["records"]:
            source_id = int(record["source_id"])
            scale = float(record["sigma_scale"])
            if scale not in SCALES:
                raise ValueError(f"Unexpected scale {scale} in {path}")
            records = records_by_source.setdefault(source_id, {})
            if scale in records:
                raise ValueError(f"Duplicate source/scale: {source_id}/{scale} in {path}")
            records[scale] = record
        candidates = []
        for source_id, records in sorted(records_by_source.items(),
                                         key=lambda pair: (pair[1][0.75].get("selection") != "top", pair[0])):
            if set(records) != set(SCALES):
                raise ValueError(f"Incomplete strength series for source {source_id} in {path}")
            reference = records[0.75]
            reference_pixel_mse = scalar(reference["pixels"]["all_video_mse"], "pixel MSE")
            reference_latent = scalar(reference["rollout"]["next_chunk_mse"], "latent MSE")
            candidate = {
                "source_id": source_id, "selection": reference.get("selection", "unknown"),
                "region": reference["region"], "num_tokens": reference["num_tokens"],
                "records": [],
            }
            for scale in SCALES:
                record = records[scale]
                if record["region"] != reference["region"] or record.get("selection") != reference.get("selection"):
                    raise ValueError(f"Changed candidate across strengths: {path}, source {source_id}")
                pixel_mse = scalar(record["pixels"]["all_video_mse"], "pixel MSE")
                pixel_mae = scalar(record["pixels"]["all_video_mae"], "pixel MAE")
                latent = scalar(record["rollout"]["next_chunk_mse"], "next chunk MSE")
                pixels_path = artifact_path(path.parent, record["artifacts"], "pixels_npz")
                item = {"sigma_scale": scale, "source_sigma": record["source_sigma"],
                        "pixel_mse": pixel_mse, "pixel_rmse": math.sqrt(pixel_mse), "pixel_mae": pixel_mae,
                        "next_chunk_mse": latent, "pixel_mse_ratio_vs_0p75": ratio(pixel_mse, reference_pixel_mse),
                        "pixel_rmse_ratio_vs_0p75": ratio(math.sqrt(pixel_mse), math.sqrt(reference_pixel_mse)),
                        "next_chunk_mse_ratio_vs_0p75": ratio(latent, reference_latent),
                        "pixels_path": str(pixels_path.resolve()),
                        "original_replay_check": record.get("original_replay_check")}
                if record["artifacts"].get("mp4"):
                    item["original_video_url"] = relative_url(artifact_path(path.parent, record["artifacts"], "mp4"), output_dir)
                candidate["records"].append(item)
            for metric in ("pixel_rmse", "next_chunk_mse"):
                values = [item[metric] for item in candidate["records"]]
                candidate[f"{metric}_nondecreasing"] = bool(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))
                candidate[f"{metric}_endpoint_larger"] = bool(values[-1] > values[0] + 1e-12)
            stem = f"{path.parent.name}_source_{source_id:03d}"
            candidate["comparison_video"] = f"{stem}.mp4"
            candidate["comparison_poster"] = f"{stem}.jpg"
            candidate["source_locator"] = f"{stem}_source.png"
            candidates.append(candidate)
        if not candidates:
            raise ValueError(f"No candidate records in {path}")
        baseline_path = artifact_path(path.parent, metadata.get("baseline_artifacts", {}),
                                      "pixels_npz", "baseline_pixels.npz")
        cases.append({"name": path.parent.name, "metadata": metadata, "identity": raw.get("identity"),
                      "metrics_url": relative_url(path, output_dir), "metrics_path": str(path.resolve()),
                      "baseline_pixels_path": str(baseline_path.resolve()), "candidates": candidates})
    if not cases:
        raise ValueError(f"No */strength_metrics.json found in {input_dir}")
    candidates = [candidate for case in cases for candidate in case["candidates"]]
    by_scale = {}
    for index, scale in enumerate(SCALES):
        records = [candidate["records"][index] for candidate in candidates]
        by_scale[str(scale)] = {
            metric: stats([record[metric] for record in records])
            for metric in ("pixel_mse", "pixel_rmse", "pixel_mae", "next_chunk_mse", "pixel_mse_ratio_vs_0p75",
                           "pixel_rmse_ratio_vs_0p75", "next_chunk_mse_ratio_vs_0p75")
        }
    aggregate = {"n_cases": len(cases), "n_sources": len(candidates),
                 "n_interventions": len(candidates) * len(SCALES), "by_scale": by_scale}
    identity_values = [float(value) for case in cases for value in (case["identity"] or {}).values()]
    replay_checks = [candidate["records"][0]["original_replay_check"] for candidate in candidates]
    aggregate["identity_max_abs"] = max(identity_values) if identity_values else None
    aggregate["original_replay_checks_count"] = sum(check is not None for check in replay_checks)
    aggregate["original_replay_exact_count"] = sum(
        check is not None and check["max_abs"] == 0 and check["current_sha256"] == check["original_sha256"]
        for check in replay_checks
    )
    for metric in ("pixel_rmse", "next_chunk_mse"):
        for suffix in ("nondecreasing", "endpoint_larger"):
            aggregate[f"{metric}_{suffix}_count"] = sum(candidate[f"{metric}_{suffix}"] for candidate in candidates)
    aggregate["by_selection"] = {}
    for selection in sorted({candidate["selection"] for candidate in candidates}):
        selected = [candidate for candidate in candidates if candidate["selection"] == selection]
        group = {"n_sources": len(selected), "by_scale": {}}
        for index, scale in enumerate(SCALES):
            group["by_scale"][str(scale)] = {
                metric: stats([candidate["records"][index][metric] for candidate in selected])
                for metric in ("pixel_rmse", "next_chunk_mse", "pixel_rmse_ratio_vs_0p75",
                               "pixel_mse_ratio_vs_0p75", "next_chunk_mse_ratio_vs_0p75")
            }
        for metric in ("pixel_rmse", "next_chunk_mse"):
            for suffix in ("nondecreasing", "endpoint_larger"):
                group[f"{metric}_{suffix}_count"] = sum(candidate[f"{metric}_{suffix}"] for candidate in selected)
        aggregate["by_selection"][selection] = group
    return {"schema_version": 1, "input_dir": str(input_dir.resolve()), "sigma_scales": list(SCALES),
            "pixel_metric_units": "decoded float [0,1], before uint8 quantization and MP4 encoding",
            "difference_display_gain": diff_gain, "aggregate": aggregate, "cases": cases, "limitations": LIMITATIONS}


def load_pixels(path):
    with np.load(path, allow_pickle=False) as data:
        pixels = data["pixels"]
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected uint8 [T,H,W,3]: {path}, {pixels.shape}, {pixels.dtype}")
    return pixels


def render_media(summary, output_dir, fps=16, overwrite=False):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    tile_width, tile_height, header_height = 416, 240, 24
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    gain = summary["difference_display_gain"]
    for case in summary["cases"]:
        baseline = load_pixels(case["baseline_pixels_path"])
        latent_h, latent_w = case["metadata"]["config"]["image_or_video_shape"][-2:]
        for candidate in case["candidates"]:
            video_path = output_dir / candidate["comparison_video"]
            poster_path = output_dir / candidate["comparison_poster"]
            locator_path = output_dir / candidate["source_locator"]
            candidate["display_frame_count"] = len(baseline)
            candidate["poster_frame_index"] = len(baseline) - 1
            candidate["source_locator_frame_index"] = min(case["metadata"]["source_frame"] * 4, len(baseline) - 1)
            region = candidate["region"]
            box = (region["x0"] / latent_w * tile_width, region["y0"] / latent_h * tile_height,
                   region["x1"] / latent_w * tile_width, region["y1"] / latent_h * tile_height)
            locator = Image.fromarray(baseline[candidate["source_locator_frame_index"]]).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            ImageDraw.Draw(locator).rectangle(box, outline="#15ffff", width=2)
            locator.save(locator_path)
            if not overwrite and video_path.is_file() and video_path.stat().st_size > 0 and poster_path.is_file():
                continue
            conditions = [load_pixels(record["pixels_path"]) for record in candidate["records"]]
            if any(pixels.shape != baseline.shape for pixels in conditions):
                raise ValueError(f"Unpaired video shapes in {case['name']}, source {candidate['source_id']}")
            temporary = video_path.with_name(video_path.stem + ".partial.mp4")
            try:
                with imageio.get_writer(temporary, fps=fps, codec="libx264", macro_block_size=16,
                                        ffmpeg_params=["-crf", "18", "-preset", "fast", "-threads", "2"]) as writer:
                    for frame_index in range(len(baseline)):
                        difference = np.clip(np.abs(conditions[-1][frame_index].astype(np.int16) -
                                                    baseline[frame_index].astype(np.int16)) * gain, 0, 255).astype(np.uint8)
                        frames = [baseline[frame_index]] + [pixels[frame_index] for pixels in conditions] + [difference]
                        labels = [f"Baseline | source {candidate['source_id']}",
                                  "Local sigma x 0.75", "Local sigma x 0.50", "Local sigma x 0.25",
                                  "Local sigma x 0 (previous prediction)", f"Abs diff: x0 vs baseline | gain {gain:g}"]
                        canvas = Image.new("RGB", (tile_width * 3, (tile_height + header_height) * 2), "#10131a")
                        draw = ImageDraw.Draw(canvas)
                        for index, (pixels, label) in enumerate(zip(frames, labels)):
                            x = index % 3 * tile_width
                            y = index // 3 * (tile_height + header_height)
                            tile = Image.fromarray(pixels).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                            canvas.paste(tile, (x, y + header_height))
                            draw.text((x + 7, y + 3), label, fill="white", font=font)
                        draw.rectangle((canvas.width - 110, canvas.height - 21, canvas.width, canvas.height), fill="black")
                        draw.text((canvas.width - 106, canvas.height - 20), f"frame {frame_index + 1}/{len(baseline)}", fill="white", font=font)
                        writer.append_data(np.asarray(canvas))
                        if frame_index == len(baseline) - 1:
                            canvas.save(poster_path, quality=92)
                temporary.replace(video_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            print(f"RENDERED {case['name']} source={candidate['source_id']}", flush=True)


def plot_summary(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [candidate for case in summary["cases"] for candidate in case["candidates"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    definitions = [("pixel_rmse_ratio_vs_0p75", "Decoded-video RMSE / RMSE at 0.75"),
                   ("next_chunk_mse_ratio_vs_0p75", "Next-chunk latent MSE / MSE at 0.75")]
    for axis, (metric, title) in zip(axes, definitions):
        values = []
        for candidate in candidates:
            series = [record[metric] for record in candidate["records"]]
            if any(value is None or value <= 0 for value in series):
                continue
            values.append(series)
            axis.plot(range(4), series, alpha=.24, linewidth=.8, color="#2265ad")
        if values:
            axis.plot(range(4), np.median(values, axis=0), linewidth=2.5, marker="o", color="#ce5d13", label="Median")
        axis.set_xticks(range(4), ["0.75", "0.50", "0.25", "0"])
        axis.set_xlabel("Remaining local sigma fraction (stronger intervention →)")
        axis.set_ylabel("Ratio to the same source at 0.75 (log)")
        axis.set_title(title)
        axis.set_yscale("log")
        axis.grid(alpha=.2)
        axis.legend()
    figure.savefig(output_dir / "strength_response.png", dpi=160)
    plt.close(figure)


def case_label(case):
    metadata = case["metadata"]
    return f"prompt {metadata['prompt_index']} · seed {metadata['seed']} · step {metadata['step_index']}"


def write_html(summary, output_dir):
    aggregate = summary["aggregate"]
    endpoint = aggregate["by_scale"]["0.0"]
    preferred = [(case_index, source_index) for case_index, case in enumerate(summary["cases"])
                 for source_index, candidate in enumerate(case["candidates"])
                 if case["metadata"]["prompt_index"] == 42 and case["metadata"]["seed"] == 0
                 and case["metadata"]["step_index"] == 1 and candidate["source_id"] == 17]
    default_choice = next(((case_index, source_index) for case_index, case in enumerate(summary["cases"])
                           for source_index, candidate in enumerate(case["candidates"])
                           if candidate["selection"] == "top"), (0, 0))
    if preferred:
        default_choice = preferred[0]
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>细 patch：局部降噪强度对照</title><style>body{max-width:1280px;margin:28px auto;padding:0 20px;font:16px/1.65 system-ui,sans-serif;color:#202733;background:#fafbfc}h1,h2{line-height:1.3}section{background:white;border:1px solid #dce1e8;border-radius:9px;padding:20px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce1e8;padding:9px;text-align:left}th{background:#f0f3f7}video,img{max-width:100%;height:auto}small,.muted{color:#647084}.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:16px 0}select{font:inherit;padding:6px;max-width:100%}.candidate[hidden]{display:none}.scroll{overflow:auto}.key{font-size:18px;background:#edf5ff;padding:15px;border-radius:8px}a{color:#165eaa}.caption{font-size:14px;color:#647084}summary{cursor:pointer}</style>",
             "<h1>细 patch：更强的局部降噪，是否让影响更明显？</h1>",
             f"<p>{aggregate['n_cases']} 个条件 × {aggregate['n_sources']//aggregate['n_cases']} 个固定候选 × 4 档强度 = {aggregate['n_interventions']} 次干预。每个候选始终使用同一基线和同一组后续噪声。</p>",
             f"<p class='key'>从 σ×0.75 增强到 σ×0：全视频像素 RMSE 的配对倍率中位数为 <b>{number(endpoint['pixel_rmse_ratio_vs_0p75']['median'])}×</b>，下一 chunk latent MSE 为 <b>{number(endpoint['next_chunk_mse_ratio_vs_0p75']['median'])}×</b>。",
             f"像素 RMSE 在 {aggregate['pixel_rmse_endpoint_larger_count']}/{aggregate['n_sources']} 个候选中增大；四档依次不下降的有 {aggregate['pixel_rmse_nondecreasing_count']}/{aggregate['n_sources']} 个。</p>",
             "<p>σ×0.75 是原细网格实验的较强档；本轮再比较 ×0.5、×0.25、×0。数值越小，剩余局部噪声越少，干预越强。σ×0 仍然是模型前一步预测，不是真值。输出变化大小不能视作质量提升。</p>",
             "<section><h2>同步视频</h2><p>选择一个条件和 patch，在同一段视频中同步比较 5 个输出。右下角仅展示 σ×0 的差分。</p>",
             ("<p class='muted'>默认展示此前细网格发现的热点：绿洲 prompt 42 / seed 0 / step 1 / patch 17。该例在本轮增强强度前已确定，并非按本轮变化最大挑选。</p>" if preferred else "<p class='muted'>默认展示此前细网格选出的第一个 Top 候选。</p>"),
             "<div class='controls'><label>条件 <select id='case-select'>"]
    for index, case in enumerate(summary["cases"]):
        parts.append(f"<option value='{index}'>{esc(case_label(case))}</option>")
    parts.append("</select></label><label>候选 <select id='source-select'></select></label></div>")
    for case_index, case in enumerate(summary["cases"]):
        metadata = case["metadata"]
        for source_index, candidate in enumerate(case["candidates"]):
            label = f"patch {candidate['source_id']} · {candidate['selection']} · {candidate['num_tokens']} tokens"
            parts.append(f"<article class='candidate' data-case='{case_index}' data-source='{source_index}' data-label='{esc(label)}' hidden>")
            parts.append(f"<h3>{esc(label)}</h3><p>{esc(metadata['prompt'])}</p>")
            parts.append(f"<p class='muted'>基础 σ={number(metadata['base_sigma'])}；全局 timestep={number(metadata['global_t'])} 保持不变；source chunk {metadata['source_chunk']} / latent frame {metadata['source_frame']}。region: {esc(candidate['region'])}</p>")
            parts.append(f"<details><summary>查看 source 位置示意</summary><img src='{esc(candidate['source_locator'])}' loading='lazy' alt='Source patch spatial projection on baseline'><p class='caption'>基线解码帧索引 {candidate.get('source_locator_frame_index', '—')}（从 0 开始）上的 latent 空间投影，并非精确 VAE 感受野；干预发生在 latent frame {metadata['source_frame']}。此框不表示物体跟踪，也不是后续每帧的真实干预位置。</p></details>")
            parts.append(f"<video controls loop playsinline preload='none' poster='{esc(candidate['comparison_poster'])}'><source src='{esc(candidate['comparison_video'])}' type='video/mp4'></video>")
            parts.append(f"<p class='caption'>上排：baseline、σ×0.75、σ×0.5；下排：σ×0.25、σ×0、abs(σ×0 − baseline) × {summary['difference_display_gain']:g}。差分增益在所有帧和候选中固定；显示经过 uint8 量化、缩小和视频编码，亮度可饱和。海报统一取最后一帧（索引 {candidate.get('poster_frame_index', '—')}）。</p>")
            parts.append("<div class='scroll'><table><thead><tr><th>剩余 σ 倍率</th><th>实际 σ</th><th>像素 RMSE [0,1]</th><th>像素 MAE [0,1]</th><th>RMSE / ×0.75</th><th>下一 chunk MSE</th><th>latent MSE / ×0.75</th><th>原视频</th></tr></thead><tbody>")
            for record in candidate["records"]:
                link = f"<a href='{esc(record['original_video_url'])}'>MP4</a>" if record.get("original_video_url") else "—"
                cells = [number(record[field]) for field in ("sigma_scale", "source_sigma", "pixel_rmse", "pixel_mae",
                                                            "pixel_rmse_ratio_vs_0p75", "next_chunk_mse", "next_chunk_mse_ratio_vs_0p75")]
                parts.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + f"<td>{link}</td></tr>")
            parts.append(f"</tbody></table></div><p><a href='{esc(case['metrics_url'])}'>该条件原始指标与 identity/replay 校验</a> · <a href='{esc(candidate['comparison_video'])}'>同步 MP4</a> · <a href='{esc(candidate['comparison_poster'])}'>最后一帧</a></p></article>")
    parts.append("</section><section><h2>强度与响应</h2><p>表格为固定 source 内先计算配对倍率，再跨候选取中位数 [最小值, 最大值]。RMSE 与 MSE 的倍率不同，表中分别标明。</p>")
    parts.append("<div class='scroll'><table><thead><tr><th>剩余 σ 倍率</th><th>像素 RMSE</th><th>像素 RMSE 倍率</th><th>像素 MSE 倍率</th><th>下一 chunk latent MSE 倍率</th></tr></thead><tbody>")
    for scale in SCALES:
        values = aggregate["by_scale"][str(scale)]
        cells = [number(scale)] + [stat_text(values[field]) for field in ("pixel_rmse", "pixel_rmse_ratio_vs_0p75",
                                                                         "pixel_mse_ratio_vs_0p75", "next_chunk_mse_ratio_vs_0p75")]
        parts.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in cells) + "</tr>")
    parts.append("</tbody></table></div>")
    selection_summaries = [
        f"{esc(selection)}（{group['n_sources']} 个候选）{number(group['by_scale']['0.0']['pixel_rmse_ratio_vs_0p75']['median'])}×"
        for selection, group in aggregate.get("by_selection", {}).items()
    ]
    if selection_summaries:
        parts.append("<p>按此前选源类别分组，σ×0 相比 ×0.75 的像素 RMSE 配对倍率中位数："
                     + "；".join(selection_summaries)
                     + "。这些数值描述各候选自身随干预强度的变化，不是 Top 与 random 的匹配优劣比较。</p>")
    parts.append(f"<p>下一 chunk latent MSE：σ×0 相比 ×0.75 增大 {aggregate['next_chunk_mse_endpoint_larger_count']}/{aggregate['n_sources']}；四档依次不下降 {aggregate['next_chunk_mse_nondecreasing_count']}/{aggregate['n_sources']}。</p>")
    parts.append("<a href='strength_response.png'><img src='strength_response.png' alt='Paired response ratios over local sigma strength'></a><p class='caption'>每条蓝线为一个固定 source 的配对倍率，橙线为中位数；纵轴为对数。四档数据并不预设单调。</p></section>")
    parts.append(f"<section><h2>解释边界与数据</h2><p>Identity 最大绝对误差：{number(aggregate.get('identity_max_abs'))}；σ×0.75 与上一轮保存 latent 输出精确重放一致：{aggregate.get('original_replay_exact_count', 0)}/{aggregate['n_sources']}。</p><ul>" + "".join(f"<li>{esc(item)}</li>" for item in summary["limitations"]) + "</ul><p><a href='summary.json'>完整汇总 JSON</a></p></section>")
    parts.append(f"<script>const defaultCase='{default_choice[0]}', defaultSource='{default_choice[1]}';</script>")
    parts.append("""<script>
const cases=document.getElementById('case-select'), sources=document.getElementById('source-select');
const cards=[...document.querySelectorAll('.candidate')];
function show(){for(const card of cards){const active=card.dataset.case===cases.value&&card.dataset.source===sources.value;card.hidden=!active;if(!active)card.querySelector('video').pause();}}
function refresh(){sources.replaceChildren();for(const card of cards.filter(c=>c.dataset.case===cases.value)){const option=document.createElement('option');option.value=card.dataset.source;option.textContent=card.dataset.label;sources.append(option);}show();}
cases.addEventListener('change',refresh);sources.addEventListener('change',show);cases.value=defaultCase;refresh();sources.value=defaultSource;show();
</script></html>""")
    (output_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--difference-gain", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--overwrite-media", action="store_true")
    parser.add_argument("--html-only", action="store_true", help="Refresh HTML from an existing summary.json without video/plot libraries")
    args = parser.parse_args()
    if not math.isfinite(args.difference_gain) or args.difference_gain <= 0 or args.fps <= 0:
        parser.error("difference gain and fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if args.html_only:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = load_summary(args.input_dir, args.output_dir, args.difference_gain)
        render_media(summary, args.output_dir, fps=args.fps, overwrite=args.overwrite_media)
        plot_summary(summary, args.output_dir)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(summary, args.output_dir)
    print(json.dumps({"report": str(args.output_dir / "report.html"), "aggregate": summary["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
