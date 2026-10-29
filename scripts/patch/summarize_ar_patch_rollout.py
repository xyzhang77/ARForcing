#!/usr/bin/env python3
"""Build a server-free, progressively refreshable AR patch rollout explorer.

Only completed record.json files enter the index. Display atlases are uint8
previews; quantitative values always come from the runner's float metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VERSION = 1
SCALES = (0.75, 0.5, 0.25, 0.0)
METRICS = {
    "pixel_rmse": {"label": "最终视频 · Pixel RMSE", "unit": "RGB [0,1]", "description": "完整最终视频、编码前浮点像素相对 baseline 的 RMSE"},
    "next_chunk_mse": {"label": "下一 chunk · Latent MSE", "unit": "latent²", "description": "完整 rollout 下一 chunk 的 latent 均方差"},
    "normalized_next_chunk_energy": {"label": "下一 chunk · 归一化响应", "unit": "输出能量 / 输入能量", "description": "下一 chunk 输出变化平方和 / 实际输入扰动平方和"},
}
PREVIEW_SIZE = (416, 240)
MAP_SIZE = (208, 120)
ATLAS_COLUMNS = 7
DIFFERENCE_GAIN = 4
RMS_LIMIT = 4.0
DELTA_RMS_LIMIT = 1.0
PIXEL_MAP_LIMIT = 0.15
LATENT_MAP_LIMIT = 1.0
PALETTE = np.array([[236, 245, 244], [160, 210, 202], [65, 160, 153], [14, 103, 115], [247, 178, 78], [203, 68, 41]], dtype=np.float32)


def finite(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Expected a finite nonnegative metric, got {value!r}")
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def url(path, output_dir):
    return quote(os.path.relpath(path, output_dir).replace(os.sep, "/"), safe="/")


def artifact(directory, record, key, default=None):
    value = record.get("artifacts", {}).get(key, default)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else directory / path


def existing_url(path, output_dir):
    return url(path, output_dir) if path and path.is_file() else None


def scale_key(scale):
    return format(float(scale), ".2f").rstrip("0").rstrip(".") or "0"


def branch_key(step, source, scale):
    return f"{int(step)}:{int(source)}:{scale_key(scale)}"


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def colorize(values, maximum):
    positions = np.clip(np.asarray(values, dtype=np.float32) / maximum, 0, 1) * (len(PALETTE) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(PALETTE) - 1)
    fraction = (positions - lower)[..., None]
    return np.round(PALETTE[lower] * (1 - fraction) + PALETTE[upper] * fraction).astype(np.uint8)


def save_atlas(frames, path, size=PREVIEW_SIZE, columns=ATLAS_COLUMNS, quality=87):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [frames,height,width,3], got {frames.shape}/{frames.dtype}")
    rows = math.ceil(len(frames) / columns)
    atlas = Image.new("RGB", (columns * size[0], rows * size[1]), (242, 245, 244))
    for index, pixels in enumerate(frames):
        preview = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
        atlas.paste(preview, ((index % columns) * size[0], (index // columns) * size[1]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.jpg")
    atlas.save(temporary, "JPEG", quality=quality, subsampling=0)
    temporary.replace(path)
    return {"file": path.name, "count": len(frames), "columns": columns, "width": size[0], "height": size[1]}


def atlas_descriptor(description, directory, output_dir):
    return {**description, "url": url(directory / description["file"], output_dir)}


def load_pixels(path):
    with np.load(path, allow_pickle=False) as archive:
        pixels = archive["pixels"]
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Invalid final pixels: {path}")
    return pixels


def validate_record_artifacts(directory, record):
    for name, relative in record.get("artifacts", {}).items():
        path = Path(relative)
        path = path if path.is_absolute() else directory / path
        if not path.is_file():
            raise ValueError(f"Completed record has a missing artifact: {path}")
        expected = record.get("artifact_sizes", {}).get(name)
        if expected is not None and path.stat().st_size != expected:
            raise ValueError(f"Completed record has a truncated artifact: {path}")


def fingerprint(paths, extra):
    parts = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths if path and path.is_file()]
    return hashlib.sha256(json.dumps([VERSION, parts, extra], sort_keys=True).encode()).hexdigest()


def normalize_trajectory(entries, matched=False):
    result = []
    for item in entries:
        result.append({
            "chunk": int(item.get("chunk_index", item.get("chunk", 0))),
            "step": int(item.get("step_index", item.get("step", 0))),
            "global_t": item.get("global_t", item.get("timestep")),
            "sigma": item.get("sigma"),
            "preview_frame_index": item.get("preview_frame_index", 4),
        })
        if matched:
            for name in ("mode", "source_sigma", "source_frame", "region", "local_timestep", "timestep_policy"):
                if name in item:
                    result[-1][name] = item[name]
    if len({(x["chunk"], x["step"]) for x in result}) != len(result):
        raise ValueError("Duplicate trajectory chunk/step")
    return result


def validate_matched_intervention(protocol, record):
    """Reject a report that would mislabel a different conditioning policy."""
    if protocol.get("intervention_mode") != "matched":
        return
    if protocol.get("timestep_policy") != "source_call_only":
        raise ValueError("Matched report requires timestep_policy=source_call_only")
    if record.get("intervention_mode") != "matched" or record.get("timestep_policy") != "source_call_only":
        raise ValueError("Matched branch conditioning metadata differs from protocol")
    sigma, local_t = finite(record.get("source_sigma")), finite(record.get("local_timestep"))
    if sigma is None or local_t is None or not math.isclose(local_t, sigma * 1000, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Matched branch local_timestep must equal source_sigma * 1000")
    entries = normalize_trajectory(record.get("trajectory", []), matched=True)
    source = (int(protocol.get("source_chunk", 0)), int(record.get("step_index", record.get("source_step"))))
    overrides = [entry for entry in entries if entry.get("mode") == "matched"]
    if len(overrides) != 1 or (overrides[0]["chunk"], overrides[0]["step"]) != source:
        raise ValueError("Matched timestep override must occur only at the source call")
    entry = overrides[0]
    if (entry.get("source_frame") != protocol["source_frame"] or entry.get("source_sigma") != sigma or
            entry.get("local_timestep") != local_t or entry.get("timestep_policy") != "source_call_only"):
        raise ValueError("Matched source trajectory metadata differs from branch")
    region = protocol["regions"][int(record["source_id"])]
    if any(entry.get("region", {}).get(name) != region.get(name) for name in ("x0", "x1", "y0", "y1")):
        raise ValueError("Matched source trajectory region differs from protocol")
    if any(item.get("local_timestep") is not None for item in entries if item is not entry):
        raise ValueError("Later calls must restore the original schedule")


def make_media(directory, record, output_dir, asset_dir, baseline_pixels=None, overwrite=False):
    """Create cached display images; no metrics are derived from JPEGs."""
    pixels_path = artifact(directory, record, "pixels_npz", "final_pixels.npz")
    maps_path = artifact(directory, record, "trajectory_maps_npz", "trajectory_maps.npz")
    preview_path = artifact(directory, record, "trajectory_preview_npz", "trajectory_preview.npz")
    impact_path = artifact(directory, record, "impact_maps_npz", "impact_maps.npz")
    identity_path = directory / "record.json"
    stamp = fingerprint([identity_path, pixels_path, maps_path, preview_path, impact_path], [DIFFERENCE_GAIN, RMS_LIMIT, DELTA_RMS_LIMIT, PREVIEW_SIZE])
    cache_path = asset_dir / "media.json"
    if not overwrite and cache_path.is_file():
        cached = load_json(cache_path)
        if cached.get("fingerprint") == stamp and (asset_dir / "last.jpg").is_file() and all((asset_dir / item["file"]).is_file() for item in cached.get("atlases", {}).values()):
            return {key: atlas_descriptor(item, asset_dir, output_dir) for key, item in cached["atlases"].items()}, cached.get("map_statistics", {})
    asset_dir.mkdir(parents=True, exist_ok=True)
    atlases, statistics = {}, {}
    if pixels_path and pixels_path.is_file():
        pixels = load_pixels(pixels_path)
        atlases["final"] = save_atlas(pixels, asset_dir / "final.jpg")
        Image.fromarray(pixels[-1]).save(asset_dir / "last.jpg", quality=90)
        if baseline_pixels is not None:
            if pixels.shape != baseline_pixels.shape:
                raise ValueError(f"Baseline/branch pixel shape mismatch: {directory}")
            difference = np.clip(np.abs(pixels.astype(np.int16) - baseline_pixels.astype(np.int16)) * DIFFERENCE_GAIN, 0, 255).astype(np.uint8)
            atlases["difference"] = save_atlas(difference, asset_dir / "difference.jpg")
    if maps_path and maps_path.is_file():
        with np.load(maps_path, allow_pickle=False) as archive:
            for name in ("noisy_rms", "clean_rms", "noisy_delta_rms", "clean_delta_rms"):
                if name not in archive:
                    continue
                values = archive[name]
                if values.ndim != 4 or values.shape[0] != len(record.get("trajectory", [])) or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid trajectory RMS array: {maps_path}:{name}")
                limit = DELTA_RMS_LIMIT if "delta" in name else RMS_LIMIT
                n, f, h, w = values.shape
                atlases[name] = save_atlas(colorize(values.reshape(n * f, h, w), limit), asset_dir / f"{name}.jpg", MAP_SIZE)
                statistics[name] = {"frames_per_entry": f, "means": values.mean(axis=(-1, -2)).tolist(), "maxima": values.max(axis=(-1, -2)).tolist(), "display_max": limit}
    if preview_path and preview_path.is_file():
        with np.load(preview_path, allow_pickle=False) as archive:
            for name in ("noisy", "clean"):
                if name in archive:
                    if archive[name].shape[0] != len(record.get("trajectory", [])):
                        raise ValueError(f"Trajectory preview count mismatch: {preview_path}")
                    atlases[f"{name}_pixels"] = save_atlas(archive[name], asset_dir / f"{name}_pixels.jpg")
    if impact_path and impact_path.is_file():
        with np.load(impact_path, allow_pickle=False) as archive:
            for name, limit in (("latent_mse", LATENT_MAP_LIMIT), ("pixel_mse_downsample", PIXEL_MAP_LIMIT)):
                if name not in archive:
                    continue
                values = archive[name]
                if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid final impact map: {impact_path}:{name}")
                atlases[name] = save_atlas(colorize(np.sqrt(values), limit), asset_dir / f"{name}.jpg", MAP_SIZE)
    atomic_json(cache_path, {"fingerprint": stamp, "atlases": atlases, "map_statistics": statistics})
    return {key: atlas_descriptor(item, asset_dir, output_dir) for key, item in atlases.items()}, statistics


def metric_values(record, manifest):
    pixels = record.get("pixels", record.get("pixel", {}))
    rollout = record.get("rollout", {})
    one_step = record.get("one_step", {})
    mse = finite(pixels.get("all_video_mse"))
    next_mse = finite(rollout.get("next_chunk_mse"))
    input_energy = finite(one_step.get("input_energy", rollout.get("input_energy", record.get("input_energy"))))
    normalized = rollout.get("normalized_next_chunk_energy", rollout.get("next_chunk_normalized_energy"))
    if normalized is None and next_mse is not None and input_energy:
        shape = manifest.get("protocol", {}).get("shape", [1, 6, 16, 60, 104])
        chunk_size = manifest.get("protocol", {}).get("chunks", [3, 3])[1]
        normalized = next_mse * int(shape[0]) * int(chunk_size) * math.prod(shape[-3:]) / input_energy
    return {"pixel_rmse": math.sqrt(mse) if mse is not None else None,
            "next_chunk_mse": next_mse,
            "normalized_next_chunk_energy": finite(normalized),
            "input_energy": input_energy,
            "pixel_mae": finite(pixels.get("all_video_mae"))}


def normalize_region(region, fallback_id, rows, cols):
    source_id = int(region.get("id", region.get("source_id", fallback_id)))
    row, col = divmod(source_id, cols)
    result = dict(region)
    result.update(id=source_id, row=int(region.get("row", row)), col=int(region.get("col", col)))
    if "num_tokens" not in result:
        bounds = region.get("bounds")
        if bounds and len(bounds) == 4:
            result["num_tokens"] = ((bounds[1] - bounds[0]) // 2) * ((bounds[3] - bounds[2]) // 2)
        elif all(key in region for key in ("y0", "y1", "x0", "x1")):
            result["num_tokens"] = ((region["y1"] - region["y0"]) // 2) * ((region["x1"] - region["x0"]) // 2)
    return result


def collect(input_dir, output_dir, workers=4, overwrite=False):
    manifest = load_json(input_dir / "manifest.json")
    protocol = manifest.get("protocol", manifest)
    matched = protocol.get("intervention_mode") == "matched"
    if matched and protocol.get("timestep_policy") != "source_call_only":
        raise ValueError("Matched report requires timestep_policy=source_call_only")
    rows = int(protocol.get("grid_rows", manifest.get("grid_rows", 12)))
    cols = int(protocol.get("grid_cols", manifest.get("grid_cols", 16)))
    regions = [normalize_region(r, i, rows, cols) for i, r in enumerate(manifest.get("regions", protocol.get("regions", [])))]
    if not regions:
        regions = [normalize_region({}, i, rows, cols) for i in range(rows * cols)]
    if len(regions) != rows * cols or {r["id"] for r in regions} != set(range(rows * cols)):
        raise ValueError("Manifest regions do not cover the grid exactly")
    steps = [int(x) for x in protocol.get("steps", protocol.get("source_steps", [1, 2, 3]))]
    scales = [float(x) for x in protocol.get("sigma_scales", manifest.get("sigma_scales", SCALES))]
    baseline_dir = input_dir / "baseline"
    baseline_record = load_json(baseline_dir / "record.json")
    validate_record_artifacts(baseline_dir, baseline_record)
    protocol_hash = manifest.get("protocol_sha256")
    if protocol_hash and baseline_record.get("protocol_sha256") != protocol_hash:
        raise ValueError("Baseline protocol hash differs from manifest")
    baseline_pixels = load_pixels(artifact(baseline_dir, baseline_record, "pixels_npz", "final_pixels.npz"))
    baseline_media, baseline_stats = make_media(baseline_dir, baseline_record, output_dir, output_dir / "media" / "baseline", overwrite=overwrite)
    baseline = {"media": baseline_media, "map_statistics": baseline_stats, "trajectory": normalize_trajectory(baseline_record.get("trajectory", [])),
                "video_url": existing_url(artifact(baseline_dir, baseline_record, "mp4", "final.mp4"), output_dir),
                "record_url": url(baseline_dir / "record.json", output_dir), "latent_url": existing_url(artifact(baseline_dir, baseline_record, "trajectory_pt", "trajectory.pt"), output_dir)}
    paths = sorted(input_dir.glob("step_*/source_*/scale_*/record.json"))
    seen = set()
    tasks = []
    for path in paths:
        record = load_json(path)
        if record.get("status", "complete") not in ("complete", "completed", "ok"):
            continue
        validate_record_artifacts(path.parent, record)
        if protocol_hash and record.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Branch protocol hash differs from manifest: {path}")
        if baseline_record.get("baseline_sha256") and record.get("baseline_sha256") != baseline_record["baseline_sha256"]:
            raise ValueError(f"Branch baseline hash differs: {path}")
        step = int(record.get("step_index", record.get("source_step")))
        source = int(record["source_id"])
        scale = float(record["sigma_scale"])
        key = branch_key(step, source, scale)
        if key in seen:
            raise ValueError(f"Duplicate branch: {key}")
        if step not in steps or scale not in scales or source not in range(rows * cols):
            raise ValueError(f"Unexpected branch: {key}")
        validate_matched_intervention(protocol, record)
        seen.add(key)
        tasks.append((path, record, key, step, source, scale))

    def process(task):
        path, record, key, step, source, scale = task
        asset_dir = output_dir / "media" / f"step_{step}" / f"source_{source:03d}" / f"scale_{scale_key(scale).replace('.', 'p')}"
        media, map_statistics = make_media(path.parent, record, output_dir, asset_dir, baseline_pixels, overwrite)
        item = {"step": step, "source_id": source, "sigma_scale": scale, "base_sigma": record.get("base_sigma"),
                     "source_sigma": record.get("source_sigma"), "global_t": record.get("global_t"),
                     "metrics": metric_values(record, manifest), "trajectory": normalize_trajectory(record.get("trajectory", []), matched=matched),
                     "media": media, "map_statistics": map_statistics,
                     "video_url": existing_url(artifact(path.parent, record, "mp4", "final.mp4"), output_dir),
                     "record_url": url(path, output_dir),
                     "latent_url": existing_url(artifact(path.parent, record, "trajectory_pt", "trajectory.pt"), output_dir),
                     "_asset_dir": str(asset_dir), "_record": str(path)}
        if matched:
            item.update(local_timestep=record["local_timestep"], intervention_mode="matched", timestep_policy="source_call_only")
        return key, item

    branches = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (key, item) in enumerate(pool.map(process, tasks), 1):
            branches[key] = item
            if index % 32 == 0 or index == len(tasks):
                print(f"REPORT_MEDIA {index}/{len(tasks)}", flush=True)
    expected = len(regions) * len(steps) * len(scales)
    schedules = protocol.get("schedule", manifest.get("timesteps", protocol.get("timesteps", [])))
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "input_dir": str(input_dir.resolve()), "manifest_url": url(input_dir / "manifest.json", output_dir),
               "protocol": protocol, "prompt": manifest.get("prompt", protocol.get("prompt", "")),
               "rows": rows, "cols": cols, "regions": sorted(regions, key=lambda r: r["id"]), "steps": steps, "scales": scales,
               "timesteps": schedules, "baseline": baseline, "branches": branches, "metrics": METRICS,
               "expected": expected, "completed": len(branches), "complete": len(branches) == expected,
               "display": {"difference_gain": DIFFERENCE_GAIN, "rms_max": RMS_LIMIT, "delta_rms_max": DELTA_RMS_LIMIT,
                           "pixel_map_max": PIXEL_MAP_LIMIT, "latent_map_max": LATENT_MAP_LIMIT, "fps": 16, "frames": len(baseline_pixels)},
               "metric_maxima": {metric: max([item["metrics"][metric] for item in branches.values() if item["metrics"][metric] is not None], default=0) for metric in METRICS},
               "static_heatmaps": {}, "posters": {}}
    return summary, baseline_pixels


def write_heatmaps(summary, output_dir):
    directory = output_dir / "heatmaps"
    directory.mkdir(exist_ok=True)
    for step in summary["steps"]:
        for scale in summary["scales"]:
            for metric, description in METRICS.items():
                maximum = summary["metric_maxima"][metric] or 1.0
                image = Image.new("RGB", (1100, 760), "white")
                draw = ImageDraw.Draw(image)
                draw.text((32, 20), f"Prompt {summary['protocol'].get('prompt_index', 0)} / seed {summary['protocol'].get('seed', 0)} | step {step} | sigma x {scale_key(scale)} | {metric}", font=font(21), fill=(22, 49, 53))
                draw.text((32, 54), f"Logical patch grid | fixed color range: 0 .. {maximum:.5g} | grey = missing, never zero", font=font(16), fill=(70, 90, 92))
                count = 0
                for region in summary["regions"]:
                    branch = summary["branches"].get(branch_key(step, region["id"], scale))
                    value = branch["metrics"][metric] if branch else None
                    fill = (228, 230, 232) if value is None else tuple(colorize(np.array(value), maximum).tolist())
                    count += value is not None
                    x = 32 + region["col"] * 64
                    y = 100 + region["row"] * 47
                    draw.rectangle((x, y, x + 62, y + 45), fill=fill)
                    text_color = (20, 35, 39) if sum(fill) > 350 else (255, 255, 255)
                    draw.text((x + 5, y + 3), f"{region['id']:03d}", font=font(13), fill=text_color)
                    draw.text((x + 4, y + 23), "--" if value is None else f"{value:.2g}", font=font(12), fill=text_color)
                draw.text((32, 690), f"Completed {count}/{len(summary['regions'])}; independent interventions, not additive causal shares.", font=font(15), fill=(70, 90, 92))
                if summary["protocol"].get("intervention_mode") == "matched":
                    draw.text((32, 719), "Noise-level aligned: local sigma + token timestep + flow-to-x0 sigma map; source call only.", font=font(15), fill=(70, 90, 92))
                path = directory / f"step_{step}_scale_{scale_key(scale).replace('.', 'p')}_{metric}.png"
                image.save(path)
                summary["static_heatmaps"][f"{step}:{scale_key(scale)}:{metric}"] = url(path, output_dir)


def write_posters(summary, baseline_pixels, output_dir):
    directory = output_dir / "posters"
    directory.mkdir(exist_ok=True)
    width, height, head = 320, 185, 28
    baseline_last = Image.fromarray(baseline_pixels[-1]).resize((width, height), Image.Resampling.BILINEAR)
    prompt_index = int(summary["protocol"].get("prompt_index", 0))
    seed = int(summary["protocol"].get("seed", 0))
    source_chunk = int(summary["protocol"].get("source_chunk", 0))
    for step in summary["steps"]:
        for source in range(summary["rows"] * summary["cols"]):
            selected = [summary["branches"].get(branch_key(step, source, scale)) for scale in summary["scales"]]
            if not any(selected):
                continue
            path = directory / f"prompt_{prompt_index:06d}_seed_{seed:06d}_chunk_{source_chunk}_step_{step}_source_{source:03d}.jpg"
            stamp = fingerprint([Path(item["_record"]) for item in selected if item], [VERSION, "poster", prompt_index, seed, source_chunk, summary["baseline"]["media"]["final"]["url"]])
            stamp_path = path.with_suffix(".json")
            if not path.is_file() or not stamp_path.is_file() or load_json(stamp_path).get("fingerprint") != stamp:
                canvas = Image.new("RGB", (5 * width, 2 * (height + head) + 62), (247, 249, 249))
                draw = ImageDraw.Draw(canvas)
                draw.text((14, 11), f"Prompt {prompt_index} / seed {seed} / source step {step} / patch {source:03d} / last frame {summary['display']['frames'] - 1}", font=font(19), fill=(24, 54, 58))
                poster_note = "Top: final generated frame. Bottom: absolute pixel difference x4. Display JPEGs; metrics use original float pixels."
                if summary["protocol"].get("intervention_mode") == "matched":
                    poster_note = "Noise-level aligned (source call only): local sigma, token timestep and flow-to-x0 sigma map. Top: final frame. Bottom: pixel diff x4."
                draw.text((14, 36), poster_note, font=font(14), fill=(80, 96, 98))
                canvas.paste(baseline_last, (0, 62 + head))
                draw.text((10, 68), "Baseline", font=font(15), fill=(24, 54, 58))
                draw.text((10, 62 + height + head + 8), "Baseline difference = 0", font=font(15), fill=(24, 54, 58))
                canvas.paste(Image.new("RGB", (width, height), "black"), (0, 62 + height + 2 * head))
                for column, (scale, branch) in enumerate(zip(summary["scales"], selected), 1):
                    draw.text((column * width + 10, 68), f"Sigma x {scale_key(scale)}", font=font(15), fill=(24, 54, 58))
                    if branch is None:
                        draw.text((column * width + 20, 150), "Not completed", font=font(19), fill=(100, 110, 110))
                        continue
                    frame = Image.open(Path(branch["_asset_dir"]) / "last.jpg").convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                    canvas.paste(frame, (column * width, 62 + head))
                    descriptor = branch["media"]["difference"]
                    index = descriptor["count"] - 1
                    x = (index % descriptor["columns"]) * descriptor["width"]
                    y = (index // descriptor["columns"]) * descriptor["height"]
                    with Image.open(Path(branch["_asset_dir"]) / descriptor["file"]) as atlas:
                        difference = atlas.crop((x, y, x + descriptor["width"], y + descriptor["height"])).resize((width, height), Image.Resampling.BILINEAR)
                    canvas.paste(difference, (column * width, 62 + height + 2 * head))
                    rmse = branch["metrics"]["pixel_rmse"]
                    draw.text((column * width + 10, 62 + height + head + 8), f"Pixel RMSE = {rmse:.5f}" if rmse is not None else "Pixel RMSE unavailable", font=font(14), fill=(24, 54, 58))
                canvas.save(path, quality=90, subsampling=0)
                atomic_json(stamp_path, {"fingerprint": stamp})
            summary["posters"][f"{step}:{source}"] = url(path, output_dir)


def write_html(summary, output_dir):
    assets = Path(__file__).with_name("ar_patch_report_assets")
    css = (assets / "report.css").read_text(encoding="utf-8")
    javascript = (assets / "report.js").read_text(encoding="utf-8")
    template = (assets / "report.html").read_text(encoding="utf-8")
    exported = json.loads(json.dumps(summary))
    navigation_path = output_dir.parent / "report_navigation.json"
    if navigation_path.is_file():
        exported["navigation"] = load_json(navigation_path)
    if output_dir.name == "report":
        exported["influence_summary"] = {
            kind: existing_url(output_dir.parent / f"final_influence_summary.{kind}", output_dir)
            for kind in ("json", "txt")
        }
    for branch in exported["branches"].values():
        branch.pop("_asset_dir", None)
        branch.pop("_record", None)
    encoded = json.dumps(exported, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    document = template.replace("__REPORT_CSS__", css).replace("__REPORT_DATA__", encoded).replace("__REPORT_JS__", javascript)
    temporary = output_dir / "report.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_dir / "report.html")
    atomic_json(output_dir / "report_data.json", exported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/ar_patch_rollout_20261022"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Label a validation subset without changing protocol coverage")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, pixels = collect(args.input_dir, output_dir, args.workers, args.overwrite)
    if args.pilot:
        summary["report_context"] = {"phase": "pilot", "source_ids": sorted({item["source_id"] for item in summary["branches"].values()}),
                                     "full_scan_url": url(output_dir.parent / "report.html", output_dir)}
    write_heatmaps(summary, output_dir)
    write_posters(summary, pixels, output_dir)
    write_html(summary, output_dir)
    print(f"REPORT_DONE completed={summary['completed']}/{summary['expected']} path={output_dir / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
