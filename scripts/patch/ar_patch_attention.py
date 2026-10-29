"""AR patch importance and baseline self-attention at the same source denoising call.

Importance follows the aligned AR protocol: at one denoising step, one 12x16
grid region in chunk 0, latent frame 1 gets local sigma = base sigma * scale
(scale 0 = previous clean prediction), with matched token timestep; the rest of
the video is then sampled with the original noise bank. Score = full-video
decoded RGB RMSE against the unperturbed baseline.

Attention is read from the unperturbed call at the same (chunk 0, step):
softmax(QK^T/sqrt(d)) of all 30 self-attention layers, recomputed in fp32 from
the exact q/k fed to flash attention and checked against its output.
Only metrics are stored unless --save-media (video, last frame, impact map per branch).

--noise-draw k>0 redraws the chunk-0 transition noise of the intervened call and all
later chunk-0 calls (resample_after); everything before the intervened call, the
injected x0, and chunk-1 noise stay identical. Baseline and branches of draw k are
paired with each other. The shard JSON then records the noise floor: the change of
the draw-k baseline against the original (draw 0) baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patch import ar_patch_rollout as old
from scripts.patch.ar_patch_aligned_rollout import MODE, check_aligned_controls
from scripts.patch.ar_patch_timestep_probe import PatchTimestepRollout
from scripts.patch.patch_noise_importance import atomic_json, load_pipeline, make_noise_bank, tensor_hash
from scripts.patch.patch_noise_strength import decode_pixels, pixel_metrics, scale_name
from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/patch_importance/ar_patch_attention_20261026"))
    p.add_argument("--checkpoint-path", type=Path,
                   default=Path("logs/self_forcing_dmd_1_3b_teacher_14b/checkpoint_model_000600"))
    p.add_argument("--config-path", type=Path, default=Path("configs/self_forcing_dmd.yaml"))
    p.add_argument("--data-path", type=Path, default=Path("prompts/vbench/all_dimension.txt"))
    p.add_argument("--prompt-index", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, choices=(1, 2, 3), required=True, help="Zero-based denoising index")
    p.add_argument("--grid-rows", type=int, default=12)
    p.add_argument("--grid-cols", type=int, default=16)
    p.add_argument("--source-frame", type=int, default=1)
    p.add_argument("--sigma-scales", type=float, nargs="+", default=[0.75, 0.5, 0.25, 0.0],
                   help="Local sigma = base sigma * scale (original aligned protocol)")
    p.add_argument("--save-media", action="store_true", help="Per branch: final.mp4, last frame, impact maps")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--noise-draw", type=int, default=0,
                   help=">0: redraw chunk-0 noise from the intervened call onward (see resample_after)")
    p.add_argument("--baseline-only", action="store_true", help="Save the baseline (and noise floor) only")
    args = p.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        p.error("Require 0 <= shard-index < num-shards")
    if len(set(args.sigma_scales)) != len(args.sigma_scales) or any(not 0 <= s < 1 for s in args.sigma_scales):
        p.error("sigma-scales must be distinct values in [0,1)")
    if args.noise_draw < 0:
        p.error("noise-draw must be >= 0")
    args.weights, args.num_output_frames, args.source_chunk, args.halo = "generator", 6, 0, 2
    args.steps = [args.step]
    return args


def resample_after(transition, step, seed, draw):
    """Copy of the noise bank with chunk-0 transition noise of calls >= step redrawn (draw > 0)."""
    if draw == 0:
        return transition, []
    rng = torch.Generator(device="cpu").manual_seed(10_000_019 * draw + 1009 * seed + step)
    result, keys = dict(transition), sorted(k for k in transition if k[0] == 0 and k[1] >= step)
    for key in keys:
        old_noise = transition[key]
        result[key] = torch.randn(old_noise.shape, generator=rng, dtype=torch.float32).to(old_noise)
    return result, [f"{c}:{s}" for c, s in keys]


def token_groups(regions, frames, height, width):
    """Group id (frame * R + region) of every token, in the model's F,H,W order."""
    rows, cols = height // 2, width // 2
    plane = torch.full((rows, cols), -1, dtype=torch.long)
    for r in regions:
        plane[r["y0"] // 2:r["y1"] // 2, r["x0"] // 2:r["x1"] // 2] = r["id"]
    if (plane < 0).any():
        raise ValueError("Regions do not cover every token")
    groups = torch.cat([plane.flatten() + f * len(regions) for f in range(frames)])
    counts = torch.bincount(groups, minlength=frames * len(regions))
    expected = [(r["y1"] - r["y0"]) * (r["x1"] - r["x0"]) // 4 for r in regions] * frames
    if counts.tolist() != expected:
        raise ValueError("Token grouping disagrees with region token counts")
    return groups


class AttentionRecorder:
    """Temporarily wraps causal_model.attention (self-attention only) to log probabilities."""

    def __init__(self, groups, num_groups):
        self.onehot = torch.nn.functional.one_hot(groups, num_groups).float()
        self.received, self.flow, self.rel_error = [], [], []

    def __enter__(self):
        import wan.modules.causal_model as cm
        self.module, self.original = cm, cm.attention
        cm.attention = self.wrapped
        return self

    def __exit__(self, *exc):
        self.module.attention = self.original

    def wrapped(self, q, k, v, **kwargs):
        out = self.original(q, k, v, **kwargs)
        if kwargs or q.shape[0] != 1 or q.shape[1] != k.shape[1] or q.shape[1] != len(self.onehot):
            raise ValueError("Recorder expects the first chunk with no KV prefix")
        qf, kf, vf = (t[0].float().transpose(0, 1) for t in (q, k, v))  # [N, L, D]
        probs = torch.softmax(qf @ kf.transpose(1, 2) * qf.shape[-1] ** -0.5, dim=-1)
        reconstructed = (probs @ vf).transpose(0, 1)
        self.rel_error.append(float((reconstructed - out[0].float()).abs().max() / out.float().abs().max()))
        self.received.append(probs.sum(dim=1).cpu())  # [N, Lk]: attention each key receives
        onehot = self.onehot.to(probs.device)
        self.flow.append((onehot.T @ (probs.mean(0) @ onehot)).cpu())  # [query group, key group]
        del probs, reconstructed
        return out


@torch.inference_mode()
def run(args):
    from omegaconf import OmegaConf
    started = time.monotonic()
    torch.set_num_threads(4)
    case_dir = args.output_dir / f"prompt_{args.prompt_index:06d}_seed_{args.seed:06d}" / f"step_{args.step}"
    case_dir.mkdir(parents=True, exist_ok=True)
    pipeline, config, shape, chunks, checkpoint = load_pipeline(args)
    if chunks != [3, 3]:
        raise ValueError("Protocol requires two three-frame chunks")
    prompt = args.data_path.read_text().splitlines()[args.prompt_index].strip()
    regions = build_regions(shape[-2], shape[-1], args.grid_rows, args.grid_cols)
    parameter = next(pipeline.generator.parameters())
    pipeline.text_encoder.to(parameter.device)
    conditional = pipeline.text_encoder(text_prompts=[prompt])
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    initial, original = make_noise_bank(shape, chunks, len(pipeline.denoising_step_list),
                                        args.seed, parameter.dtype, parameter.device)
    transition, resampled = resample_after(original, args.step, args.seed, args.noise_draw)
    noise_floor = None
    if args.noise_draw:
        reference = PatchTimestepRollout(pipeline, conditional, initial, original, chunks)
        reference_output = reference.baseline(0, args.step)["output"]
        reference.take_trace()
        reference_pixels = decode_pixels(pipeline, reference_output)
    replay = PatchTimestepRollout(pipeline, conditional, initial, transition, chunks)
    baseline_output = replay.baseline(0, args.step)["output"]
    trace = replay.take_trace()
    sigmas = [e["sigma"] for e in trace if e["chunk_index"] == 0]
    state = old.source_state(trace, transition, 0, args.step, parameter.device)
    base_sigma = sigmas[args.step]
    protocol = {"prompt_index": args.prompt_index, "prompt": prompt, "seed": args.seed, "step_index": args.step,
                "base_sigma": base_sigma, "sigmas": sigmas, "schedule": [float(s) for s in replay.steps],
                "source_chunk": 0, "source_frame": args.source_frame, "sigma_scales": args.sigma_scales, "mode": MODE,
                "grid": [args.grid_rows, args.grid_cols], "regions": regions, "shape": shape,
                "checkpoint_path": str(checkpoint.resolve()), "weights": args.weights,
                "config": OmegaConf.to_container(config, resolve=True),
                "baseline_sha256": tensor_hash(baseline_output), "initial_noise_sha256": tensor_hash(initial),
                "importance_metric": "pixel.all_video_rmse over 21 decoded float RGB frames"}
    baseline_pixels = decode_pixels(pipeline, baseline_output)
    if args.noise_draw:
        protocol.update(noise_draw=args.noise_draw, resampled_noise=resampled,
                        draw0_baseline_sha256=tensor_hash(reference_output))
        frame = (slice(None), args.source_frame)
        noise_floor = {"pixel": pixel_metrics(reference_pixels, baseline_pixels),
                       "chunk0_latent_mse": float((baseline_output[:, :chunks[0]].float()
                                                   - reference_output[:, :chunks[0]].float()).square().mean()),
                       "source_frame_latent_mse": float((baseline_output[frame].float()
                                                         - reference_output[frame].float()).square().mean()),
                       "next_chunk_mse": float((baseline_output[:, chunks[0]:].float()
                                                - reference_output[:, chunks[0]:].float()).square().mean())}
        print(f"NOISE_FLOOR draw={args.noise_draw} video_rmse={noise_floor['pixel']['all_video_rmse']:.6g}", flush=True)
    path = case_dir / f"importance_shard_{args.shard_index:02d}.json"
    if args.shard_index == 0:
        from PIL import Image
        torch.save(baseline_output.cpu(), case_dir / "baseline_latent.pt")
        Image.fromarray(baseline_pixels[4 * args.source_frame].mul(255).round().to(torch.uint8).numpy()).save(
            case_dir / "baseline_source_frame.png")
    if args.baseline_only:
        atomic_json(path, {"protocol": protocol, "noise_floor": noise_floor, "records": [], "baseline_only": True,
                           "complete": True, "elapsed_seconds": time.monotonic() - started})
        print(f"DONE baseline-only {path}", flush=True)
        return
    controls = check_aligned_controls(replay, trace, baseline_output, [args.step], regions, args.source_frame)
    replay.capture = False
    print(f"CONTROLS exact; step={args.step} sigma={base_sigma:.6f}", flush=True)

    if args.shard_index == 0 and args.noise_draw == 0:
        groups = token_groups(regions, chunks[0], shape[-2], shape[-1])
        replay.reset_to_chunk(0)
        with AttentionRecorder(groups, chunks[0] * len(regions)) as recorder:
            prediction = replay.predict(state["noisy"], 0, args.step)
        if not torch.equal(prediction, state["prediction"]):
            raise RuntimeError("Attention capture call differs from the baseline call")
        if len(recorder.flow) != len(pipeline.generator.model.blocks) or max(recorder.rel_error) > 2e-2:
            raise RuntimeError(f"Attention reconstruction failed: {recorder.rel_error}")
        flow = torch.stack(recorder.flow)
        if not torch.allclose(flow.sum(dim=(1, 2)), torch.full((len(flow),), float(len(groups))), rtol=1e-3):
            raise RuntimeError("Region attention flow does not conserve query mass")
        np.savez_compressed(case_dir / "attention.npz", received=torch.stack(recorder.received).numpy(),
                            region_flow=flow.numpy(), token_group=groups.numpy(),
                            rel_error=np.array(recorder.rel_error))
        print(f"ATTENTION saved; max relative reconstruction error {max(recorder.rel_error):.3g}", flush=True)

    records = []
    if path.exists():
        existing = json.loads(path.read_text())
        if not args.resume or existing["protocol"] != protocol:
            raise FileExistsError(f"{path} exists with {'another protocol' if args.resume else 'no --resume'}")
        records = existing["records"]
    done = {(r["source_id"], r["sigma_scale"]) for r in records}
    jobs = [(region, scale) for region in regions[args.shard_index::args.num_shards] for scale in args.sigma_scales]
    if args.save_media and args.shard_index == 0:
        save_media(baseline_pixels, baseline_pixels, case_dir / "baseline")
    for region, scale in jobs:
        if (region["id"], scale) in done:
            continue
        t0 = time.monotonic()
        sigma = base_sigma * scale
        changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"], sigma, region, args.source_frame)
        delta = changed.float() - state["noisy"].float()
        prediction, output = replay.probe(changed, args.step, MODE, region, args.source_frame, sigma)
        rollout = measure_influence(baseline_output, output, delta, region, args.source_frame, regions, args.halo)
        rollout["next_chunk_mse"] = float((output[:, chunks[0]:].float()
                                           - baseline_output[:, chunks[0]:].float()).square().mean())
        pixels = decode_pixels(pipeline, output)
        if args.save_media:
            save_media(baseline_pixels, pixels, case_dir / f"source_{region['id']:03d}" / f"scale_{scale_name(scale)}")
        records.append({"source_id": region["id"], "sigma_scale": scale, "source_sigma": sigma,
                        "row": region["row"], "col": region["col"],
                        "num_tokens": (region["y1"] - region["y0"]) * (region["x1"] - region["x0"]) // 4,
                        "one_step": measure_influence(state["prediction"], prediction, delta, region,
                                                      args.source_frame, regions, args.halo),
                        "rollout": rollout, "pixel": pixel_metrics(baseline_pixels, pixels),
                        "seconds": time.monotonic() - t0})
        atomic_json(path, {"protocol": protocol, "controls": controls, "noise_floor": noise_floor,
                           "records": records})
        print(f"PATCH {region['id']:03d} scale={scale:g} rmse={records[-1]['pixel']['all_video_rmse']:.6g} "
              f"({len(records)}/{len(jobs)}) {records[-1]['seconds']:.1f}s", flush=True)
    post = check_aligned_controls(replay, trace, baseline_output, [args.step], regions, args.source_frame)
    atomic_json(path, {"protocol": protocol, "controls": controls, "post_controls": post, "noise_floor": noise_floor,
                       "records": records, "complete": len(records) == len(jobs),
                       "elapsed_seconds": time.monotonic() - started})
    print(f"DONE {path}", flush=True)


def save_media(baseline_pixels, pixels, destination):
    """Display-only: H.264 video, last frame, and 8x-downsampled per-frame RGB MSE map."""
    import imageio.v2 as imageio
    from PIL import Image
    destination.mkdir(parents=True, exist_ok=True)
    video = pixels.mul(255).round().to(torch.uint8).numpy()
    imageio.mimwrite(str(destination / "final.mp4"), video, fps=16, codec="libx264", quality=7,
                     macro_block_size=1, ffmpeg_params=["-threads", "2"])
    Image.fromarray(video[-1]).save(destination / "final.jpg", quality=92)
    mse = (pixels - baseline_pixels).square().mean(dim=-1)
    t, h, w = mse.shape
    np.savez_compressed(destination / "impact_map.npz",
                        pixel_mse_downsample=mse.reshape(t, h // 8, 8, w // 8, 8).mean(dim=(2, 4)).numpy())


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a GPU machine")
    run(parse_args())
