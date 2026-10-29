#!/usr/bin/env python3
"""Build verified contact sheets from real ODE init / AR / Wan previews.

Individual preview JPEGs are copied byte-for-byte. The sheets only add labels
and arrange the decoded frames; no image content is cropped, resized, or edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


MODELS = ("ode", "ar", "wan")
LABELS = {"ode": "ODE init", "ar": "AR", "wan": "Wan bidirectional"}
COLORS = {"ode": "#137c73", "ar": "#2456a6", "wan": "#b35d14"}
PROMPTS = {42: "Desert oasis", 300: "Cat drinking water"}
SIGMA = 0.9375
PREVIEW_SIZE = (416, 240)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    # Match compare_first_step_noise.json_hash, imported by the sampler.
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def checked_preview(root, model, prompt_index, seed, ode_checkpoint):
    directory = root / model / f"prompt_{prompt_index:06d}" / f"seed_{seed:06d}"
    record_path, preview_path = directory / "record.json", directory / "preview.jpg"
    record = json.loads(record_path.read_text())
    if record.get("complete") is not True:
        raise ValueError(f"Incomplete case: {record_path}")
    if record.get("schema_version") != "diversity_moments_sampling_v1":
        raise ValueError(f"Unexpected sampling schema: {record_path}")
    expected = dict(model=model, prompt_index=prompt_index, seed=seed)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Incorrect {key}: {record_path}")
    if not math.isclose(float(record.get("sigma", -1)), SIGMA, abs_tol=1e-10, rel_tol=0):
        raise ValueError(f"Incorrect sigma: {record_path}")
    if record.get("feature_space") != "wan_vae_raw_x0_first3latents":
        raise ValueError(f"Incorrect feature space: {record_path}")
    protocol = record.get("protocol")
    if not isinstance(protocol, dict) or canonical_hash(protocol) != record.get("protocol_sha256"):
        raise ValueError(f"Protocol hash mismatch: {record_path}")
    for key in (*expected, "prompt", "sigma", "model_source"):
        if record.get(key) != protocol.get(key):
            raise ValueError(f"Record/protocol {key} mismatch: {record_path}")
    noise_hash = record.get("initial_noise_sha256")
    if not noise_hash or record.get("noise_sha256") != noise_hash:
        raise ValueError(f"Initial noise hash mismatch: {record_path}")
    if model == "ode" and ode_checkpoint is not None:
        source = record.get("model_source", {}).get("checkpoint_path", "")
        if Path(source).resolve() != ode_checkpoint.resolve():
            raise ValueError(f"ODE checkpoint mismatch: {record_path}")
    digest = file_hash(preview_path)
    if record.get("artifact_sha256", {}).get("preview.jpg") != digest:
        raise ValueError(f"Preview artifact hash mismatch: {preview_path}")
    with Image.open(preview_path) as image:
        if image.size != PREVIEW_SIZE or image.mode != "RGB":
            raise ValueError(f"Unexpected preview size/mode: {preview_path}")
        pixels = image.copy()
    return dict(record=record, record_path=record_path, source_path=preview_path,
                source_hash=digest, pixels=pixels)


def font(size, bold=False, font_dir=None):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    directories = ([font_dir] if font_dir else []) + [
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    for directory in directories:
        path = Path(directory) / filename
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("DejaVuSans fonts unavailable; specify --font-dir")


def make_sheet(prompt_index, seeds, cases, font_dir=None):
    margin, row_label, gap, top, footer = 16, 64, 12, 152, 50
    tile_w, tile_h = PREVIEW_SIZE
    width = margin*2 + row_label + tile_w*len(MODELS) + gap*(len(MODELS)-1)
    height = top + tile_h*len(seeds) + gap*(len(seeds)-1) + footer
    sheet = Image.new("RGB", (width, height), "#f2f4f7")
    draw = ImageDraw.Draw(sheet)
    title_font, text_font = font(27, True, font_dir), font(15, False, font_dir)
    header_font, seed_font = font(20, True, font_dir), font(18, True, font_dir)
    draw.text((margin, 15), f"Prompt #{prompt_index} - {PROMPTS.get(prompt_index, 'Comparison')}",
              fill="#172334", font=title_font)
    draw.text((margin, 55), "Predicted x0 at sigma = 0.9375 | decoded frame 4 (zero-based), latent frame 1",
              fill="#35475b", font=text_font)
    draw.text((margin, 80), "Each row shares initial noise; ODE init and AR also share re-noising noise.",
              fill="#35475b", font=text_font)
    for column, model in enumerate(MODELS):
        left = margin + row_label + column*(tile_w+gap)
        draw.rounded_rectangle((left, 111, left+tile_w, 143), radius=5, fill=COLORS[model])
        bbox = draw.textbbox((0, 0), LABELS[model], font=header_font)
        draw.text((left+(tile_w-bbox[2])/2, 113), LABELS[model], fill="white", font=header_font)
    for row, seed in enumerate(seeds):
        y = top + row*(tile_h+gap)
        draw.text((margin, y+8), "seed", fill="#5e6d7e", font=text_font)
        draw.text((margin, y+30), str(seed), fill="#172334", font=seed_font)
        for column, model in enumerate(MODELS):
            left = margin + row_label + column*(tile_w+gap)
            sheet.paste(cases[seed][model]["pixels"], (left, y))
    draw.text((margin, height-32), "Real early x0 previews; these are not final video frames. Individual source JPEGs are provided unchanged.",
              fill="#5e6d7e", font=font(13, False, font_dir))
    return sheet


def build(args):
    roots = {model: args.ode_root if model == "ode" else args.baseline_root for model in MODELS}
    all_cases = {}
    # Validate all inputs before creating comparison artifacts.
    for prompt_index in args.prompt_indices:
        cases, prompt_text = {}, None
        for seed in args.seeds:
            cases[seed] = {}
            for model in MODELS:
                case = checked_preview(roots[model], model, prompt_index, seed, args.ode_checkpoint)
                record = case["record"]
                if prompt_text is None:
                    prompt_text = record["prompt"]
                if record["prompt"] != prompt_text:
                    raise ValueError(f"Prompt text mismatch: prompt={prompt_index}, seed={seed}")
                cases[seed][model] = case
            records = [cases[seed][model]["record"] for model in MODELS]
            for key in ("initial_noise_sha256", "first_chunk_noise_sha256"):
                hashes = {record.get(key) for record in records}
                if None in hashes or len(hashes) != 1:
                    raise ValueError(f"{key} differs: prompt={prompt_index}, seed={seed}")
            transition = records[0].get("transition_noise_sha256")
            if not transition or transition != records[1].get("transition_noise_sha256"):
                raise ValueError(f"ODE/AR re-noising differs: prompt={prompt_index}, seed={seed}")
        all_cases[prompt_index] = (prompt_text, cases)
    manifest = dict(schema_version="diversity_comparison_images_v1",
                    generated_at=datetime.now(timezone.utc).isoformat(), sigma=SIGMA,
                    models=list(MODELS), labels=LABELS, preview_size=list(PREVIEW_SIZE),
                    decoded_frame_index=4, latent_frame_index=1, frame_indices_zero_based=True,
                    source_images_copied_byte_for_byte=True, prompts=[])
    output = args.output_root / "comparisons"
    for prompt_index, (prompt_text, cases) in all_cases.items():
        directory = output / f"prompt_{prompt_index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        item = dict(prompt_index=prompt_index, prompt=prompt_text, seeds=args.seeds,
                    images={}, artifacts={})
        for seed in args.seeds:
            item["images"][str(seed)] = {}
            item["artifacts"][str(seed)] = {}
            for model in MODELS:
                case = cases[seed][model]
                destination = directory / f"{model}_seed_{seed:06d}.jpg"
                shutil.copyfile(case["source_path"], destination)
                digest = file_hash(destination)
                if digest != case["source_hash"]:
                    raise ValueError(f"Copied image hash mismatch: {destination}")
                item["images"][str(seed)][model] = str(destination.relative_to(args.output_root))
                record = case["record"]
                item["artifacts"][str(seed)][model] = dict(
                    source_preview=str(case["source_path"].resolve()), source_preview_sha256=digest,
                    copied_preview_sha256=digest, source_record=str(case["record_path"].resolve()),
                    source_record_sha256=file_hash(case["record_path"]),
                    model=model, seed=seed, prompt_index=prompt_index,
                    initial_noise_sha256=record["initial_noise_sha256"],
                    first_chunk_noise_sha256=record["first_chunk_noise_sha256"],
                    transition_noise_sha256=record.get("transition_noise_sha256"),
                    model_source=record["model_source"])
        sheet = make_sheet(prompt_index, args.seeds, cases, args.font_dir)
        path = output / f"comparison_prompt_{prompt_index:06d}.jpg"
        sheet.save(path, quality=96, subsampling=0)
        item.update(sheet=str(path.relative_to(args.output_root)), sheet_size=list(sheet.size),
                    sheet_sha256=file_hash(path))
        manifest["prompts"].append(item)
    manifest["validated_image_count"] = len(args.prompt_indices)*len(args.seeds)*len(MODELS)
    manifest["checks"] = ["completed case identity and protocol hash", "source preview artifact SHA-256",
                          "copied preview SHA-256", "paired initial and first-chunk noise hashes",
                          "paired ODE/AR transition noise hashes", "same prompt and sigma"]
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / "manifest.json")
    print(json.dumps(dict(manifest=str(output / "manifest.json"),
                          validated_image_count=manifest["validated_image_count"],
                          sheets=[dict(path=p["sheet"], size=p["sheet_size"]) for p in manifest["prompts"]])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/diversity/diversity_moments_20261023"))
    parser.add_argument("--ode-root", type=Path, default=Path("outputs/diversity/diversity_ode_init_20261023"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/diversity/diversity_ode_init_20261023"))
    parser.add_argument("--ode-checkpoint", type=Path, default=Path("checkpoints/ode_init.pt"))
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=[42, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args(argv)
    for key in ("prompt_indices", "seeds"):
        values = getattr(args, key)
        if len(set(values)) != len(values) or any(value < 0 for value in values):
            parser.error(f"{key} must contain distinct nonnegative integers")
    build(args)


if __name__ == "__main__":
    main()
