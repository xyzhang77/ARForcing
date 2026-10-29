#!/usr/bin/env python3
"""CLIP/DINOv3 diversity across samples of each prompt, not across video frames.

Input: one model's flat directory of <prompt>-<sample>.mp4 or
<index>-<sample>_{regular,ema}.mp4, as written by inference.py/gen_wan_vbench.py.
Compare matching, uniformly sampled frame positions; average over pairs, frames,
then prompts. --num-frames 1 evaluates only the first frame. Higher is more diverse.
Run directly or with accelerate launch for prompt-sharded multi-GPU evaluation.
Reference: https://github.com/Multimedia-Analytics-Laboratory/dpdmd/blob/main/test_diversity.py
"""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import fmean

import numpy as np


def discover_videos(video_dir):
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    groups, variants = {}, set()
    for path in sorted(video_dir.glob("*.mp4")):
        match = re.fullmatch(r"(.+)-([0-9]+)(?:_(regular|ema))?", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized video filename (including unfinished files): {path.name}")
        prompt, sample, variant = match.groups()
        sample = int(sample)
        variants.add(variant)
        samples = groups.setdefault(prompt, {})
        if sample in samples:
            raise ValueError(f"Duplicate sample {sample} for prompt {prompt!r}")
        samples[sample] = path
    if not groups:
        raise ValueError(f"No MP4 videos found in {video_dir}")
    if len(variants) != 1:
        raise ValueError("Do not mix regular/EMA weights or filename formats in one directory")
    expected = set(next(iter(groups.values())))
    for prompt, samples in groups.items():
        if len(samples) < 2:
            raise ValueError(f"Prompt {prompt!r} needs at least two samples")
        if set(samples) != expected:
            raise ValueError(f"Prompt {prompt!r} has different sample IDs; finish generation before evaluation")
    return {prompt: dict(sorted(samples.items())) for prompt, samples in groups.items()}


def uniform_frame_indices(total_frames, num_frames):
    if not 1 <= num_frames <= total_frames:
        raise ValueError(f"Need 1 <= num_frames <= video length, got {num_frames} and {total_frames}")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()


def read_video_frames(path, num_frames):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {path}")
        indices = uniform_frame_indices(total_frames, num_frames)
        frames = []
        # Sequential decoding avoids codec-dependent inaccuracies when seeking.
        for index in range(indices[-1] + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index} of {total_frames}: {path}")
            if index == indices[len(frames)]:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, {"total_frames": total_frames, "fps": fps, "frame_indices": indices}
    finally:
        cap.release()


def select_features(outputs, kind):
    if kind == "clip":
        return outputs.last_hidden_state[:, 0, :]
    if kind == "dino":
        pooled = getattr(outputs, "pooler_output", None)
        return pooled if pooled is not None else outputs.last_hidden_state.mean(dim=1)
    raise ValueError(f"Unknown encoder: {kind}")


def extract_embeddings(images, model, processor, kind, batch_size, device):
    import torch

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            inputs = processor(images=images[start:start + batch_size], return_tensors="pt")
            pixels = inputs["pixel_values"].to(device=device, dtype=model.dtype)
            outputs = model(pixel_values=pixels, return_dict=True, output_hidden_states=False)
            embeddings.append(select_features(outputs, kind).float().cpu().numpy())
    return np.concatenate(embeddings)


def compute_diversity(embeddings):
    """Embeddings [samples, frames, dimensions]; exclude self/cross-time pairs."""
    features = np.asarray(embeddings, dtype=np.float64)
    if features.ndim != 3 or features.shape[0] < 2 or min(features.shape[1:]) < 1:
        raise ValueError("Expected [samples >= 2, frames >= 1, dimensions >= 1]")
    if not np.isfinite(features).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embedding norms must be finite and nonzero")
    features = features / norms
    similarities = np.einsum("ntd,mtd->tnm", features, features)
    i, j = np.triu_indices(len(features), k=1)
    frame_cosines = similarities[:, i, j].clip(-1, 1).mean(axis=1)
    avg_cosine = float(frame_cosines.mean())
    return {"avg_cosine_similarity": avg_cosine, "diversity": 1.0 - avg_cosine,
            "per_frame_diversity": (1.0 - frame_cosines).tolist()}


def load_scorers(clip_name, dino_name, device):
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModel

    scorers = {}
    for kind, name, model_class, processor_class in (
        ("clip", clip_name, CLIPVisionModel, CLIPImageProcessor),
        ("dino", dino_name, AutoModel, AutoImageProcessor),
    ):
        processor = processor_class.from_pretrained(name)
        model = model_class.from_pretrained(name).to(device=device, dtype=torch.float32)
        scorers[kind] = (model.eval().requires_grad_(False), processor)
    return scorers


def evaluate_prompt(prompt, samples, scorers, num_frames, batch_size, device):
    images, metadata = [], None
    try:
        for path in samples.values():
            frames, info = read_video_frames(path, num_frames)
            images.extend(frames)
            if metadata is not None and (info["total_frames"] != metadata["total_frames"] or
                                        not math.isclose(info["fps"], metadata["fps"], rel_tol=1e-4)):
                raise ValueError(f"Videos must have matching lengths and FPS within prompt {prompt!r}: {path}")
            metadata = info
        result = {"prompt": prompt, "sample_ids": list(samples), "num_samples": len(samples),
                  "num_pairs": len(samples) * (len(samples) - 1) // 2,
                  "videos": [str(path) for path in samples.values()], **metadata}
        for kind, (model, processor) in scorers.items():
            features = extract_embeddings(images, model, processor, kind, batch_size, device)
            result[kind] = compute_diversity(features.reshape(len(samples), num_frames, -1))
        return result
    finally:
        for image in images:
            image.close()


def aggregate_scores(records):
    if not records:
        raise ValueError("No prompt results to aggregate")
    summary = {}
    for kind in ("clip", "dino"):
        cosine = fmean(record[kind]["avg_cosine_similarity"] for record in records)
        summary[kind] = {"avg_cosine_similarity": cosine, "diversity": 1.0 - cosine}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True, help="New JSON file; never overwrite results")
    parser.add_argument("--num-frames", type=int, default=8, help="Uniform frame positions; 1 means first frame")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per encoder forward pass")
    parser.add_argument("--clip-name", default="openai/clip-vit-large-patch14", help="HF model ID or local directory")
    parser.add_argument("--dino-name", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HF model ID or local directory; DINOv3 needs transformers>=4.56 and model access")
    parser.add_argument("--cpu", action="store_true", help="Run scorers on CPU instead of GPU (slow)")
    args = parser.parse_args(argv)
    if args.num_frames < 1 or args.batch_size < 1:
        parser.error("num-frames and batch-size must be positive")
    groups = discover_videos(args.video_dir.resolve())
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_path}")

    import torch
    import transformers
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from tqdm import tqdm

    accelerator = Accelerator(cpu=args.cpu, mixed_precision="no")
    shard = list(groups.items())[accelerator.process_index::accelerator.num_processes]
    scorers = load_scorers(args.clip_name, args.dino_name, accelerator.device) if shard else {}
    records = []
    for prompt, samples in tqdm(shard, desc="Diversity (rank 0)", disable=not accelerator.is_main_process):
        records.append(evaluate_prompt(prompt, samples, scorers, args.num_frames,
                                       args.batch_size, accelerator.device))
    records = gather_object(records)
    if not accelerator.is_main_process:
        return
    records.sort(key=lambda record: record["prompt"])
    summary = aggregate_scores(records)
    result = {
        "schema_version": "video_diversity_clip_dino_v1",
        "video_dir": str(args.video_dir.resolve()),
        "clip_name": args.clip_name, "dino_name": args.dino_name,
        "feature_selection": {"clip": "last_hidden_state[:, 0, :]",
                              "dino": "pooler_output, else last_hidden_state.mean(dim=1)"},
        "frame_sampling": "round(linspace(0, total_frames - 1, num_frames)); first frame if num_frames=1",
        "aggregation": "L2 normalize; mean cosine over i<j at matching frames; equal mean over frames and prompts",
        "encoder_dtype": "float32", "metric_dtype": "float64",
        "num_frames": args.num_frames, "batch_size": args.batch_size,
        "num_prompts": len(records), "num_videos": sum(record["num_samples"] for record in records),
        "world_size": accelerator.num_processes,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "summary": summary, "per_prompt": records,
    }
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(records)} prompts to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
