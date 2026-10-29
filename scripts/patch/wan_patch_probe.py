"""Paired local interventions in pretrained, bidirectional Wan 2.1 T2V 1.3B.

This research runner uses a memoryless Euler ODE sampler, so a branch needs only
the recorded state, never a hidden multistep-solver history. The local sigma
operation follows the baseline CFG flow towards its current clean estimate;
it is not a substitution of ground-truth clean content. Pixel metrics precede
preview quantization and compression.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch.patch_noise_importance import atomic_json, tensor_hash
from scripts.patch.patch_noise_strength import decode_pixels, pixel_metrics, save_pixels
from scripts.patch.patch_noise_utils import build_regions, measure_influence


def shifted_sigmas(steps, shift=5.0):
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps and shift must be positive")
    # Float64 construction makes shared fractions exactly agree across N.
    values = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return (shift * values / (1 + (shift - 1) * values)).float()


def stage_index(steps, fraction):
    position = steps * fraction
    index = round(position)
    if not 0 <= index < steps or not math.isclose(position, index, abs_tol=1e-9):
        raise ValueError("stage_fraction must identify an exact nonterminal grid point")
    return index


def noise_tensor(shape, seed):
    """Match the saved Self-Forcing initial bank's CPU float32 -> bf16 path."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=rng, dtype=torch.float32).to(torch.bfloat16)


def patch_slice(state, region, source_frame):
    if state.ndim != 5 or not 0 <= source_frame < state.shape[1]:
        raise ValueError("Expected [B,F,C,H,W] and an existing source frame")
    y0, y1, x0, x1 = (region[k] for k in ("y0", "y1", "x0", "x1"))
    if not 0 <= y0 < y1 <= state.shape[-2] or not 0 <= x0 < x1 <= state.shape[-1]:
        raise ValueError("Source region lies outside the latent")
    return (slice(None), source_frame, slice(None), slice(y0, y1), slice(x0, x1))


def intervene(state, flow, sigma, region, source_frame, scale=0.0, donor=None):
    """Change exactly one spatial box in one frame; leave all other bits intact."""
    if state.dtype != torch.float32 or state.shape != flow.shape:
        raise ValueError("ODE state must be float32 with a matching flow")
    box = patch_slice(state, region, source_frame)
    changed = state.clone()
    if donor is not None:
        if donor.shape != state.shape or donor.device != state.device or donor.dtype != state.dtype:
            raise ValueError("Donor shape/device/dtype must match the ODE state")
        if sigma != 1.0:
            raise ValueError("Independent noise replacement is defined only at sigma=1")
        changed[box] = donor[box]
    else:
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("sigma scale must lie in [0,1]")
        if scale != 1.0:
            changed[box] += ((scale - 1.0) * sigma) * flow[box]
    return changed


def token_count(region):
    return (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4


def choose_candidates(records, regions, seed):
    """Two hotspots and two reproducible distinct controls of matching area."""
    by_id = {r["id"]: r for r in regions}
    ranked = sorted(records, key=lambda r: (-r["one_step"]["normalized_other_frames_energy"], r["source_id"]))
    top = [r["source_id"] for r in ranked[:2]]
    if len(top) != 2 or len(set(top)) != 2:
        raise ValueError("Need two distinct top sources")
    chosen = set(top)
    candidates = [{"source_id": sid, "selection": "top", "matched_top_id": sid,
                   "region": dict(by_id[sid]), "num_tokens": token_count(by_id[sid])} for sid in top]
    rng = random.Random(seed + 873121)
    for sid in top:
        available = sorted(k for k, r in by_id.items()
                           if k not in chosen and token_count(r) == token_count(by_id[sid]))
        if not available:
            raise ValueError("No remaining area-matched random control")
        control = rng.choice(available)
        chosen.add(control)
        candidates.append({"source_id": control, "selection": "random_control", "matched_top_id": sid,
                           "region": dict(by_id[control]), "num_tokens": token_count(by_id[control])})
    return candidates


def frozen_candidates(selection, regions, prompt_index, seed, fraction, intervention):
    metadata = selection["metadata"]
    if metadata["prompt_index"] != prompt_index or metadata["seed"] != seed or metadata["steps"] != 48:
        raise ValueError("Selection must come from the same prompt/seed at 48 steps")
    expected_fraction = 0.25 if intervention == "initial_noise_replace" else fraction
    if (metadata["intervention"] != "sigma_reduce"
            or not math.isclose(metadata["stage_fraction"], expected_fraction, abs_tol=1e-9)):
        raise ValueError("Selection stage or intervention does not match the frozen protocol")
    result = metadata["fixed_candidates"]
    if (len(result) != 4 or len({r["source_id"] for r in result}) != 4
            or sum(r["selection"] == "top" for r in result) != 2
            or sum(r["selection"] == "random_control" for r in result) != 2):
        raise ValueError("Selection must contain two distinct top and two random controls")
    by_id = {r["id"]: r for r in regions}
    top_ids = {r["source_id"] for r in result if r["selection"] == "top"}
    for record in result:
        sid, matched = record["source_id"], record["matched_top_id"]
        if (record["region"] != by_id.get(sid) or matched not in top_ids
                or token_count(by_id[sid]) != token_count(by_id[matched])):
            raise ValueError("Frozen candidate region or matched area differs")
    return [dict(r, region=dict(r["region"])) for r in result]


def euler_rollout(predict, state, sigmas, start=0, capture=None, first_flow=None):
    """Deterministic float32 state evolution; returns a requested baseline state."""
    state = state.float().clone()
    snapshot = None
    for index in range(start, len(sigmas) - 1):
        sigma = float(sigmas[index])
        flow = first_flow if index == start and first_flow is not None else predict(state, sigma)
        flow = flow.float()
        if index == capture:
            snapshot = {"state": state.clone(), "flow": flow.clone(), "x0": state - sigma * flow}
        state = state + (float(sigmas[index + 1]) - sigma) * flow
    return state, snapshot


class NativeWan:
    def __init__(self, prompt, negative_prompt, shape, guidance, shift):
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

        self.device = torch.device("cuda")
        self.guidance = guidance
        print("LOAD text encoder", flush=True)
        encoder = WanTextEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.conditional = encoder(text_prompts=[prompt])["prompt_embeds"]
        self.unconditional = encoder(text_prompts=[negative_prompt])["prompt_embeds"]
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("LOAD pretrained bidirectional Wan2.1-T2V-1.3B", flush=True)
        self.generator = WanDiffusionWrapper(model_name="Wan2.1-T2V-1.3B", is_causal=False,
                                            timestep_shift=shift).to(device=self.device, dtype=torch.bfloat16)
        self.generator.eval().requires_grad_(False)
        self.generator.seq_len = shape[1] * (shape[-2] // 2) * (shape[-1] // 2)
        self.vae = torch.nn.Identity()

    def predict(self, state, sigma):
        # Use the native flow directly. The wrapper's approximate scheduler
        # lookup for x0 is deliberately not part of this sampler.
        model_input = state.to(torch.bfloat16).permute(0, 2, 1, 3, 4)
        timestep = torch.full((state.shape[0],), sigma * 1000,
                              device=state.device, dtype=torch.float32)
        positive = self.generator.model(model_input, t=timestep, context=self.conditional,
                                        seq_len=self.generator.seq_len).permute(0, 2, 1, 3, 4).float()
        negative = self.generator.model(model_input, t=timestep, context=self.unconditional,
                                        seq_len=self.generator.seq_len).permute(0, 2, 1, 3, 4).float()
        return negative + self.guidance * (positive - negative)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_manifest(model_dir=Path("wan_models/Wan2.1-T2V-1.3B")):
    model_dir = Path(model_dir)
    files = []
    for path in sorted(model_dir.iterdir()):
        if path.suffix not in (".safetensors", ".pth", ".bin", ".json"):
            continue
        stat = path.stat()
        record = {"filename": path.name, "path": str(path.resolve()),
                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".json":
            record["sha256"] = file_hash(path)
        files.append(record)
    if not (model_dir / "diffusion_pytorch_model.safetensors").is_file():
        raise RuntimeError("Pretrained native Wan weight file was not found")
    return files


def source_case(args):
    old_step = 2 if args.intervention == "sigma_reduce" and args.stage_fraction == .5 else 1
    path = args.fine_dir / f"prompt_{args.prompt_index:06d}_seed_{args.seed:06d}_chunk_0_step_{old_step}" / "metrics.json"
    data = json.loads(path.read_text())
    metadata = data["metadata"]
    if metadata["prompt_index"] != args.prompt_index or metadata["seed"] != args.seed:
        raise ValueError("Fine-sweep source identity mismatch")
    return path, data


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    source_path, source = source_case(args)
    shape = (1, 6, 16, 60, 104)
    index = stage_index(args.steps, args.stage_fraction)
    sigmas = shifted_sigmas(args.steps, args.shift)
    sigma = float(sigmas[index])
    initial_bf16 = noise_tensor(shape, args.seed)
    if tensor_hash(initial_bf16) != source["metadata"]["initial_noise_sha256"]:
        raise RuntimeError("Reconstructed initial noise differs from the original fine sweep")
    regions = build_regions(shape[-2], shape[-1], 12, 16)
    selection = json.loads(args.selection_path.read_text()) if args.selection_path else None
    candidates = (frozen_candidates(selection, regions, args.prompt_index, args.seed,
                                   args.stage_fraction, args.intervention) if selection else None)
    if args.intervention == "initial_noise_replace" and (index != 0 or candidates is None):
        raise ValueError("Initial replacement requires stage zero and a frozen 48-step .25 selection")
    if args.mode == "replay" and candidates is None:
        raise ValueError("Replay requires --selection-path")
    if selection is not None:
        selected_metadata = selection["metadata"]
        expected = {"initial_noise_sha256": tensor_hash(initial_bf16), "source_frame": args.source_frame,
                    "cfg_scale": args.guidance, "shift": args.shift, "sigma_scale": args.sigma_scale}
        for key, value in expected.items():
            if selected_metadata.get(key) != value:
                raise ValueError(f"Frozen selection differs on {key}")
    if args.output_dir.joinpath("metrics.json").exists():
        raise FileExistsError("Refusing to overwrite a completed case; use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": args.mode, "prompt_index": args.prompt_index, "prompt": source["metadata"]["prompt"],
        "seed": args.seed, "steps": args.steps, "stage_fraction": args.stage_fraction,
        "stage_index": index, "actual_sigma": sigma, "global_t": sigma * 1000,
        "intervention": args.intervention, "sigma_scale": args.sigma_scale,
        "source_frame": args.source_frame, "grid_rows": 12, "grid_cols": 16,
        "latent_shape": list(shape), "seq_len": shape[1] * shape[-2] * shape[-1] // 4,
        "cfg_scale": args.guidance, "shift": args.shift, "sigmas": sigmas.tolist(),
        "solver": "explicit Euler, no solver history, float32 ODE state",
        "model_name": "Wan2.1-T2V-1.3B", "model_type": "original pretrained bidirectional T2V, no distilled checkpoint",
        "model_dtype": "bfloat16", "model_files": model_manifest(), "fps": 16,
        "source_fine_metrics_path": str(source_path.resolve()), "source_fine_metrics_sha256": file_hash(source_path),
        "selection_path": str(args.selection_path.resolve()) if args.selection_path else None,
        "selection_sha256": file_hash(args.selection_path) if args.selection_path else None,
        "initial_noise_sha256": tensor_hash(initial_bf16), "initial_noise_dtype": "bf16, promoted to float32 for ODE",
        "donor_seed": args.seed + 1000003 if args.intervention == "initial_noise_replace" else None,
        "ranking_metric": "normalized_other_frames_energy",
        "ranking_domain": f"all five latent frames except source frame {args.source_frame}",
        "one_step_prediction": "x - sigma * CFG_flow at unchanged global timestep",
        "intervention_definition": ("replace source box by independent Gaussian donor at sigma=1"
                                    if args.intervention == "initial_noise_replace" else
                                    "x_source += (scale-1)*sigma*baseline_CFG_flow_source; estimated clean endpoint, no ground truth"),
        "scope": "six latent frames / 21 decoded frames; shorter than the standard 81-frame Wan example",
        "negative_prompt": source["metadata"]["config"]["negative_prompt"],
        "torch_version": torch.__version__, "host": platform.node(), "gpu": torch.cuda.get_device_name(),
        "code_sha256": {str(p): file_hash(p) for p in [Path(__file__), Path("utils/wan_wrapper.py"),
                         Path("wan/modules/model.py"), Path("scripts/patch/patch_noise_utils.py"),
                         Path("scripts/patch/patch_noise_strength.py")]},
    }
    pipeline = NativeWan(metadata["prompt"], metadata["negative_prompt"], shape, args.guidance, args.shift)
    initial = initial_bf16.to(device=pipeline.device, dtype=torch.float32)
    donor_bf16 = noise_tensor(shape, metadata["donor_seed"]) if metadata["donor_seed"] is not None else None
    metadata["donor_noise_sha256"] = tensor_hash(donor_bf16) if donor_bf16 is not None else None
    donor = donor_bf16.to(device=pipeline.device, dtype=torch.float32) if donor_bf16 is not None else None
    baseline_start = time.monotonic()
    baseline, snapshot = euler_rollout(pipeline.predict, initial, sigmas, capture=index)
    print(f"BASELINE steps={args.steps} stage={index} sigma={sigma:.9f}", flush=True)
    metadata["baseline_seconds"] = time.monotonic() - baseline_start
    metadata["baseline_sha256"] = tensor_hash(baseline)
    metadata["stage_state_sha256"] = tensor_hash(snapshot["state"])
    metadata["stage_flow_sha256"] = tensor_hash(snapshot["flow"])
    same_input = intervene(snapshot["state"], snapshot["flow"], sigma, regions[0], args.source_frame, scale=1)
    same_flow = pipeline.predict(same_input, sigma)
    same_x0 = same_input - sigma * same_flow
    same_rollout, _ = euler_rollout(pipeline.predict, same_input, sigmas, start=index, first_flow=same_flow)
    identity = {"input_max_abs": float((same_input - snapshot["state"]).abs().max()),
                "one_step_max_abs": float((same_x0 - snapshot["x0"]).abs().max()),
                "rollout_max_abs": float((same_rollout - baseline).abs().max())}
    if any(value != 0 for value in identity.values()):
        raise RuntimeError(f"Nonzero exact replay identity: {identity}")
    del same_input, same_flow, same_x0, same_rollout
    print("IDENTITY exact", flush=True)
    torch.save({"baseline": baseline.cpu(), "initial_noise": initial_bf16,
                "stage_state": snapshot["state"].cpu(), "stage_flow": snapshot["flow"].cpu(),
                "one_step_baseline": snapshot["x0"].cpu(), "sigmas": sigmas}, args.output_dir / "baseline.pt")
    # Lazy decoder is bf16; preserve full float32 latents for numerical metrics.
    baseline_pixels = decode_pixels(pipeline, baseline.to(torch.bfloat16))
    metadata["baseline_artifacts"] = save_pixels(baseline_pixels, args.output_dir / "baseline", not args.no_videos)
    metadata["baseline_artifacts"]["latent_pt"] = "baseline.pt"
    metadata["decoded_shape"] = list(baseline_pixels.shape)
    scan_records = []
    scan_regions = regions if args.mode == "scan" else [r["region"] for r in candidates]
    scan_start = time.monotonic()
    for count, region in enumerate(scan_regions, start=1):
        changed = intervene(snapshot["state"], snapshot["flow"], sigma, region, args.source_frame,
                            scale=args.sigma_scale, donor=donor)
        flow = pipeline.predict(changed, sigma)
        prediction = changed - sigma * flow
        delta = changed - snapshot["state"]
        scan_records.append({"source_id": region["id"], "num_tokens": token_count(region),
                             "one_step": measure_influence(snapshot["x0"], prediction, delta, region,
                                                           args.source_frame, regions),
                             "input_delta_sha256": tensor_hash(delta),
                             "intervention_input_sha256": tensor_hash(changed)})
        if count % 16 == 0 or count == len(scan_regions):
            print(f"SCAN {count}/{len(scan_regions)} source={region['id']}", flush=True)
            atomic_json(args.output_dir / "scan_progress.json", {"completed": count, "total": len(scan_regions),
                        "metadata": metadata, "scan_records": scan_records, "identity": identity})
    metadata["scan_seconds"] = time.monotonic() - scan_start
    ranked = sorted(scan_records, key=lambda r: (-r["one_step"]["normalized_other_frames_energy"], r["source_id"]))
    metadata["native_top_ids"] = [r["source_id"] for r in ranked[:2]] if args.mode == "scan" else None
    if candidates is None:
        candidates = choose_candidates(scan_records, regions, args.seed)
    metadata["fixed_candidates"] = candidates
    metadata["selection_policy"] = ("frozen 48-step sources; native ranking retained only as diagnostic"
                                    if selection else "native top2 plus independent seeded same-area random control for each top")
    old_top_ids = [r["source_id"] for r in source["records"]
                   if r.get("selection") == "top" and r.get("sigma_scale") == 0.75]
    rank_map = {r["source_id"]: i + 1 for i, r in enumerate(ranked)}
    metadata["old_self_forcing_top_ranks"] = {str(sid): rank_map.get(sid) for sid in old_top_ids}
    by_id = {r["source_id"]: r for r in scan_records}
    records = []
    for candidate in candidates:
        region = candidate["region"]
        changed = intervene(snapshot["state"], snapshot["flow"], sigma, region, args.source_frame,
                            scale=args.sigma_scale, donor=donor)
        output, _ = euler_rollout(pipeline.predict, changed, sigmas, start=index)
        delta = changed - snapshot["state"]
        record = {**candidate, **by_id[region["id"]],
                  "rollout": measure_influence(baseline, output, delta, region, args.source_frame, regions)}
        pixels = decode_pixels(pipeline, output.to(torch.bfloat16))
        record["pixels"] = pixel_metrics(baseline_pixels, pixels)
        record["artifacts"] = save_pixels(pixels, args.output_dir / f"source_{region['id']:03d}", not args.no_videos)
        latent_name = f"source_{region['id']:03d}_rollout.pt"
        torch.save(output.cpu(), args.output_dir / latent_name)
        record["artifacts"]["latent_pt"] = latent_name
        record["rollout_sha256"] = tensor_hash(output)
        records.append(record)
        print(f"ROLLOUT source={region['id']} selection={candidate['selection']} "
              f"pixel_rmse={record['pixels']['all_video_rmse']:.8f}", flush=True)
        atomic_json(args.output_dir / "rollout_progress.json", {"metadata": metadata, "identity": identity,
                    "scan_records": scan_records, "records": records, "regions": regions})
    final_flow = pipeline.predict(snapshot["state"], sigma)
    final_x0 = snapshot["state"] - sigma * final_flow
    identity["post_sweep_one_step_max_abs"] = float((final_x0 - snapshot["x0"]).abs().max())
    if any(value != 0 for value in identity.values()):
        raise RuntimeError(f"Post sweep identity failed: {identity}")
    metadata["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.output_dir / "metrics.json", {"schema_version": 1, "metadata": metadata, "regions": regions,
                "identity": identity, "scan_records": scan_records, "records": records})
    print(f"DONE {args.output_dir} seconds={metadata['elapsed_seconds']:.1f}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scan", "replay"), default="scan")
    parser.add_argument("--steps", type=int, choices=(12, 48), default=48)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-fraction", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Complete destination case directory")
    parser.add_argument("--fine-dir", type=Path, default=Path("outputs/anchor_probe_step600_20261021/fine_sweep"))
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--intervention", choices=("sigma_reduce", "initial_noise_replace"), default="sigma_reduce")
    parser.add_argument("--sigma-scale", type=float, default=0.0)
    parser.add_argument("--source-frame", type=int, default=1)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", "--guidance", dest="guidance", type=float, default=6.0)
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use the GPU machine's dev environment")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
