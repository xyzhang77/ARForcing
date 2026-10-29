#!/usr/bin/env python3
"""Exploratory, paired group interventions selected from a single-patch sweep.

All groups are compared on the same target: every frame except the complete
source frame. Changing intervention strength is not independent validation;
latent response energy is neither quality nor additive causal contribution.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.patch_noise_importance import (
    PairedRollout, atomic_json, load_pipeline, make_noise_bank, tensor_hash,
)
from scripts.patch.patch_noise_utils import replace_patch_noise


LIMITATIONS = [
    "Exploratory selection on the same prompt, seed, and denoising state; a different sigma strength is not independent validation.",
    "Ranking uses single-patch normalized outside-source energy; group evaluation excludes the entire source frame for every group.",
    "Latent response magnitude measures sensitivity, not output quality, semantic correctness, or additive causal contribution.",
    "Full-frame intervention is a response reference, not ground truth or a quality target; response cancellation and nonlinear interactions are possible.",
    "Random controls match both patch count and latent area; bottom groups may have different areas, explicitly recorded.",
    "Coarse patches contain many tokens. This experiment cannot identify individual anchor tokens or establish semantic mechanisms.",
]


def build_group_plan(regions, ranking, top_ks=(2, 4), random_groups=4, seed=191):
    """Create reproducible distinct controls matching each top group's exact area."""
    by_id = {r["id"]: r for r in regions}
    if len(by_id) != len(regions) or len(set(ranking)) != len(ranking) or set(ranking) != set(by_id):
        raise ValueError("ranking must contain every distinct region ID exactly once")
    if not top_ks or len(set(top_ks)) != len(top_ks):
        raise ValueError("top_ks must be nonempty and distinct")
    if any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k < len(regions) for k in top_ks):
        raise ValueError("each k must be an integer between 1 and number of regions - 1")
    if isinstance(random_groups, bool) or not isinstance(random_groups, int) or random_groups < 1:
        raise ValueError("random_groups must be a positive integer")
    areas = {i: (r["y1"] - r["y0"]) * (r["x1"] - r["x0"]) for i, r in by_id.items()}
    if any(a <= 0 or a % 4 for a in areas.values()):
        raise ValueError("regions must have positive areas containing whole 2x2 tokens")
    all_ids = sorted(by_id)
    total_area = sum(areas.values())

    def record(name, kind, source_ids, k=None, matched_area=None):
        ids = sorted(source_ids)
        area = sum(areas[i] for i in ids)
        return {"name": name, "kind": kind, "k": k, "source_ids": ids,
                "num_patches": len(ids), "latent_area": area, "num_tokens": area // 4,
                "source_frame_area_fraction": area / total_area,
                "top_group_latent_area": matched_area,
                "matches_top_area": None if matched_area is None else area == matched_area}

    plan = [record("full", "full", all_ids), record("empty", "identity", [])]
    for k in top_ks:
        top, bottom = tuple(sorted(ranking[:k])), tuple(sorted(ranking[-k:]))
        area = sum(areas[i] for i in top)
        candidates = [c for c in itertools.combinations(all_ids, k)
                      if c != top and sum(areas[i] for i in c) == area]
        if len(candidates) < random_groups:
            raise ValueError(f"k={k} has only {len(candidates)} distinct area-matched controls")
        controls = random.Random(seed + 104729 * k).sample(candidates, random_groups)
        plan.extend([record(f"top_{k}", "top", top, k, area),
                     record(f"bottom_{k}", "bottom", bottom, k, area)])
        plan.extend(record(f"random_{k}_{j}", "random", ids, k, area)
                    for j, ids in enumerate(controls))
        plan.append(record(f"complement_{k}", "complement", set(all_ids) - set(top), k, area))
    return plan


def group_metrics(baseline, output, input_delta, source_frame, full_output, next_chunk_start=None):
    """Measure every group on one fixed domain, including full-response similarity."""
    if baseline.ndim != 5 or baseline.shape != output.shape or baseline.shape != full_output.shape:
        raise ValueError("baseline, output, full_output must share [B,F,C,H,W] shape")
    if not 0 <= source_frame < baseline.shape[1] or baseline.shape[1] < 2:
        raise ValueError("source frame must exist and at least one other frame is required")
    if input_delta.ndim != 5 or (input_delta.shape[0], *input_delta.shape[2:]) != (baseline.shape[0], *baseline.shape[2:]):
        raise ValueError("input_delta must match batch, channel and spatial dimensions")
    if next_chunk_start is not None and not 0 < next_chunk_start <= baseline.shape[1]:
        raise ValueError("next_chunk_start must follow a nonempty source chunk")
    delta = output.float() - baseline.float()
    full_delta = full_output.float() - baseline.float()
    frames = [i for i in range(baseline.shape[1]) if i != source_frame]
    target, reference = delta[:, frames], full_delta[:, frames]
    energy = float(target.square().sum())
    reference_energy = float(reference.square().sum())
    input_energy = float(input_delta.float().square().sum())
    residual_energy = float((target - reference).square().sum())
    cosine = (float((target * reference).sum()) / math.sqrt(energy * reference_energy)
              if energy > 0 and reference_energy > 0 else None)
    result = {
        "common_target_frames": frames,
        "common_target_mse": float(target.square().mean()),
        "common_target_energy": energy,
        "input_energy": input_energy,
        "normalized_common_energy": energy / input_energy if input_energy else (0.0 if energy == 0 else None),
        "per_frame_mse": [float(delta[:, i].square().mean()) for i in range(delta.shape[1])],
        "delta_vs_full_relative_l2": (math.sqrt(residual_energy / reference_energy)
                                      if reference_energy else (0.0 if residual_energy == 0 else None)),
        "cosine_to_full": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "common_energy_relative_to_full": energy / reference_energy if reference_energy else None,
    }
    if next_chunk_start is not None:
        following = delta[:, next_chunk_start:]
        following_full = full_delta[:, next_chunk_start:]
        following_energy = float(following.square().sum())
        following_full_energy = float(following_full.square().sum())
        following_residual_energy = float((following - following_full).square().sum())
        following_cosine = (float((following * following_full).sum()) /
                            math.sqrt(following_energy * following_full_energy)
                            if following_energy > 0 and following_full_energy > 0 else None)
        result["next_chunk_mse"] = float(following.square().mean()) if following.numel() else None
        result["next_chunk_energy"] = following_energy if following.numel() else None
        result["next_chunk_normalized_energy"] = (following_energy / input_energy if input_energy
                                                    else (0.0 if following_energy == 0 else None))
        result["next_chunk_delta_vs_full_relative_l2"] = (
            math.sqrt(following_residual_energy / following_full_energy) if following_full_energy
            else (0.0 if following_residual_energy == 0 else None))
        result["next_chunk_cosine_to_full"] = max(-1.0, min(1.0, following_cosine)) if following_cosine is not None else None
        result["next_chunk_energy_relative_to_full"] = following_energy / following_full_energy if following_full_energy else None
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite group metrics")
    return result


def replace_group(state, source_ids, regions, sigma, source_frame):
    changed = state["noisy"].clone()
    for source_id in source_ids:
        changed = replace_patch_noise(changed, state["clean"], state["noise"],
                                      sigma, regions[source_id], source_frame)
    return changed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--sigma-scale", type=float, default=0.9)
    p.add_argument("--ranking-scale", type=float, default=None,
                   help="Single-patch selection strength, default: same as sigma-scale")
    p.add_argument("--top-ks", type=int, nargs="+", default=[2, 4])
    p.add_argument("--random-groups", type=int, default=4)
    p.add_argument("--prompt-indices", type=int, nargs="+")
    p.add_argument("--steps", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error("Require 0 <= shard-index < num-shards")
    if not math.isfinite(args.sigma_scale) or not 0 < args.sigma_scale < 1:
        p.error("sigma-scale must lie strictly between 0 and 1")
    if args.ranking_scale is None:
        args.ranking_scale = args.sigma_scale
    if not math.isfinite(args.ranking_scale) or not 0 < args.ranking_scale < 1:
        p.error("ranking-scale must lie strictly between 0 and 1")
    return args


def load_cases(args):
    cases = []
    for path in sorted(args.input_dir.rglob("metrics.json")):
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        if "baseline_sha256" not in metadata or "records" not in data:
            continue
        filters = ((args.prompt_indices, "prompt_index"), (args.seeds, "seed"), (args.steps, "step_index"))
        if any(values is not None and metadata[key] not in values for values, key in filters):
            continue
        records = [r for r in data["records"] if math.isclose(r["sigma_scale"], args.ranking_scale, abs_tol=1e-8)]
        if any(not math.isfinite(r["one_step"]["normalized_outside_energy"]) for r in records):
            raise ValueError(f"Invalid ranking values in {path}")
        ranking = [r["source_id"] for r in sorted(records,
                   key=lambda r: (-r["one_step"]["normalized_outside_energy"], r["source_id"]))]
        control_seed = 191 + metadata["seed"] + 1000003 * metadata["prompt_index"] + 10007 * metadata["step_index"]
        plan = build_group_plan(data["regions"], ranking, args.top_ks, args.random_groups, control_seed)
        cases.append({"path": path, "data": data, "ranking": ranking, "plan": plan, "control_seed": control_seed})
    if not cases:
        raise ValueError("No matching single-patch cases found")
    identities = [(c["data"]["metadata"]["prompt_index"], c["data"]["metadata"]["seed"],
                   c["data"]["metadata"]["source_chunk"], c["data"]["metadata"]["step_index"]) for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Input contains duplicate case identities; select a single sweep directory")
    return sorted(cases, key=lambda c: (c["data"]["metadata"]["prompt_index"], c["data"]["metadata"]["seed"],
                                       c["data"]["metadata"]["source_chunk"], c["data"]["metadata"]["step_index"]))


def validate_model_source(metadata, config, checkpoint_path):
    from omegaconf import OmegaConf
    if OmegaConf.to_container(config, resolve=True) != metadata["config"]:
        raise ValueError("Resolved model configuration differs from the single-patch experiment")
    stat = checkpoint_path.stat()
    if (str(checkpoint_path.resolve()) != metadata["checkpoint_path"] or
            stat.st_size != metadata["checkpoint_size"] or stat.st_mtime_ns != metadata["checkpoint_mtime_ns"]):
        raise ValueError("Checkpoint path, size or mtime differs from the single-patch experiment")
    if metadata["num_output_frames"] != 6:
        raise ValueError("This group probe requires the original six-latent-frame rollout")


@torch.inference_mode()
def run_case(args, pipeline, shape, chunks, case):
    data, source = case["data"], case["data"]["metadata"]
    chunk, frame, step = source["source_chunk"], source["source_frame"], source["step_index"]
    case_dir = args.output_dir / f"prompt_{source['prompt_index']:06d}_seed_{source['seed']:06d}_chunk_{chunk}_step_{step}"
    case_dir.mkdir(parents=True, exist_ok=True)
    destination = case_dir / "group_metrics.json"
    specification = {"source_metrics_sha256": hashlib.sha256(case["path"].read_bytes()).hexdigest(),
                     "ranking_source_scale": args.ranking_scale, "test_scale": args.sigma_scale,
                     "top_ks": args.top_ks, "random_groups": args.random_groups,
                     "control_seed": case["control_seed"], "group_plan": case["plan"]}
    if destination.exists():
        if not args.resume:
            raise FileExistsError(f"{destination} exists; use --resume or a new output directory")
        existing = json.loads(destination.read_text())
        if any(existing["metadata"].get(k) != v for k, v in specification.items()):
            raise ValueError(f"Resume specification differs for {case_dir}")
        print(f"SKIP {case_dir.name}", flush=True)
        return
    print(f"START {case_dir.name}: {source['prompt']}", flush=True)
    started = time.monotonic()
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[source["prompt"]])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, transition = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                          source["seed"], parameter.dtype, parameter.device)
    replay = PairedRollout(pipeline, conditional, initial, transition, chunks)
    state = replay.baseline(chunk, step)
    baseline_hash = tensor_hash(state["output"])
    if baseline_hash != source["baseline_sha256"] or tensor_hash(initial) != source["initial_noise_sha256"]:
        raise RuntimeError("Baseline or noise hash differs from the original single-patch experiment")
    sigma_index = (pipeline.scheduler.timesteps - float(replay.steps[step])).abs().argmin()
    base_sigma = float(pipeline.scheduler.sigmas[sigma_index])
    if base_sigma != source["base_sigma"] or [float(v) for v in replay.steps] != source["schedule"]:
        raise RuntimeError("Scheduler differs from the original single-patch experiment")
    regions = {r["id"]: r for r in data["regions"]}
    full_ids = case["plan"][0]["source_ids"]
    # Also exercise the replacement arithmetic at unchanged sigma over the full source frame.
    unchanged = replace_group(state, full_ids, regions, base_sigma, frame)
    if not torch.equal(unchanged, state["noisy"]):
        raise RuntimeError("Unchanged-sigma replacement does not reproduce the baseline input")
    identical_step, identical_rollout = replay.branch(unchanged, chunk, step, complete=True)
    identity = {"one_step_max_abs": float((identical_step.float() - state["prediction"].float()).abs().max()),
                "rollout_max_abs": float((identical_rollout.float() - state["output"].float()).abs().max())}
    if any(v != 0 for v in identity.values()):
        raise RuntimeError(f"Identity replay must be exact: {identity}")
    del identical_step, identical_rollout, unchanged
    source_sigma = base_sigma * args.sigma_scale
    changed_full = replace_group(state, full_ids, regions, source_sigma, frame)
    full_step, full_rollout = replay.branch(changed_full, chunk, step, complete=True)
    records, maps = [], {}
    for group in case["plan"]:
        if group["kind"] == "full":
            changed, prediction, output = changed_full, full_step, full_rollout
        else:
            changed = replace_group(state, group["source_ids"], regions, source_sigma, frame)
            prediction, output = replay.branch(changed, chunk, step, complete=True)
        input_delta = changed.float() - state["noisy"].float()
        one_step = group_metrics(state["prediction"], prediction, input_delta, frame, full_step)
        rollout = group_metrics(state["output"], output, input_delta, frame, full_rollout,
                                next_chunk_start=chunks[chunk])
        records.append({**group, "sigma_scale": args.sigma_scale, "source_sigma": source_sigma,
                        "one_step": one_step, "rollout": rollout})
        if group["kind"] == "identity" and (any(one_step["per_frame_mse"]) or any(rollout["per_frame_mse"])):
            raise RuntimeError("Empty-group identity changed output")
        if group["kind"] in ("top", "full") or (group["kind"] == "random" and group["name"].endswith("_0")):
            maps[group["name"]] = {
                "one_step": (prediction.float() - state["prediction"].float()).square().mean(dim=(0, 2)).cpu(),
                "rollout": (output.float() - state["output"].float()).square().mean(dim=(0, 2)).cpu(),
            }
            torch.save(output.cpu(), case_dir / f"{group['name']}_rollout.pt")
        print(f"GROUP {group['name']} area={group['latent_area']} common_mse={rollout['common_target_mse']:.6g} "
              f"normalized={rollout['normalized_common_energy']:.6g} next_chunk={rollout['next_chunk_mse']}", flush=True)
    end_step, end_rollout = replay.branch(state["noisy"], chunk, step, complete=True)
    identity.update(post_sweep_one_step_max_abs=float((end_step.float() - state["prediction"].float()).abs().max()),
                    post_sweep_rollout_max_abs=float((end_rollout.float() - state["output"].float()).abs().max()))
    if any(v != 0 for v in identity.values()):
        raise RuntimeError(f"Post-sweep identity must be exact: {identity}")
    torch.save(maps, case_dir / "group_difference_maps.pt")
    torch.save({"baseline": state["output"].cpu(), "one_step_baseline": state["prediction"].cpu()}, case_dir / "baseline.pt")
    metadata = {**source, **specification, "source_metrics_path": str(case["path"].resolve()),
                "ranking": case["ranking"], "original_metadata": source,
                "baseline_sha256": baseline_hash,
                "common_target_definition": "All output frames except the complete source frame; identical spatial domain for all groups",
                "primary_target": "next_chunk; common_target is secondary",
                "next_chunk_definition": "All frames after the source chunk, excluding every frame within the source chunk",
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "host": platform.node(), "elapsed_seconds": time.monotonic() - started,
                "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path("scripts/patch/patch_noise_importance.py"), Path("scripts/patch/patch_noise_utils.py")]}}
    atomic_json(destination, {"schema_version": 1, "metadata": metadata, "limitations": LIMITATIONS,
                              "regions": data["regions"], "identity": identity, "records": records})
    print(f"DONE {case_dir.name} ({metadata['elapsed_seconds']:.1f}s)", flush=True)


def main():
    from omegaconf import OmegaConf
    args = parse_args()
    cases = load_cases(args)[args.shard_index::args.num_shards]
    if not cases:
        print("No cases assigned to this shard", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the GPU machine's dev environment")
    first = cases[0]["data"]["metadata"]
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    checkpoint = Path(first["checkpoint_path"])
    for case in cases:
        source = case["data"]["metadata"]
        validate_model_source(source, config, checkpoint)
        if source["weights"] != first["weights"]:
            raise ValueError("All selected cases must use the same checkpoint weights")
    model_args = SimpleNamespace(config_path=args.config_path, checkpoint_path=checkpoint,
                                 weights=first["weights"], num_output_frames=6,
                                 source_chunk=max(c["data"]["metadata"]["source_chunk"] for c in cases),
                                 source_frame=max(c["data"]["metadata"]["source_frame"] for c in cases),
                                 steps=[c["data"]["metadata"]["step_index"] for c in cases])
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    pipeline, loaded_config, shape, chunks, loaded_checkpoint = load_pipeline(model_args)
    for case in cases:
        validate_model_source(case["data"]["metadata"], loaded_config, loaded_checkpoint)
        run_case(args, pipeline, shape, chunks, case)


if __name__ == "__main__":
    main()
