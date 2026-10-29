"""Evaluation-only token timestep conditioning for a local sigma intervention.

The causal Wan inference path accepts one time embedding per token.  The public
wrapper's flow-to-clean conversion is frame-wise, so this module supplies the
matching spatial sigma map without changing the model or training code.
"""

import math
from numbers import Integral

import torch


def build_patch_timestep_maps(noisy, timestep, scheduler, patch_size, region=None,
                             source_frame=0, source_sigma=None):
    """Return token times [B,L] and exact conversion sigmas [B,F,1,H,W].

    Tokens follow Wan's F,H,W flattening order.  Outside the source region,
    conversion uses the wrapper's nearest scheduler entry.  Inside it, the
    supplied sigma is used exactly, including zero; t = sigma * training_steps
    because the scheduler timesteps already include the sigma shift.

    With ``region=None`` all tokens retain their frame timestep, providing the
    same token-conditioning execution path for an unmodified-time control.
    """
    if noisy.ndim != 5 or not noisy.is_floating_point():
        raise ValueError("noisy must be floating [B,F,C,H,W]")
    batch, frames, channels, height, width = noisy.shape
    if min(batch, frames, channels, height, width) < 1:
        raise ValueError("noisy dimensions must be nonempty")
    if tuple(timestep.shape) != (batch, frames) or not torch.isfinite(timestep).all():
        raise ValueError("timestep must contain finite [B,F] values")
    if (len(patch_size) != 3 or any(not isinstance(p, Integral) or p < 1 for p in patch_size)
            or patch_size[0] != 1):
        raise ValueError("token conditioning requires patch_size=(1, positive H, positive W)")
    _, patch_height, patch_width = patch_size
    if height % patch_height or width % patch_width:
        raise ValueError("latent dimensions must align with model patch_size")

    time_dtype = torch.float64 if timestep.dtype == torch.float64 else torch.float32
    frame_times = timestep.to(device=noisy.device, dtype=time_dtype)
    token_times = frame_times[:, :, None, None].expand(
        batch, frames, height // patch_height, width // patch_width).clone()
    schedule_times = scheduler.timesteps.to(device=noisy.device, dtype=torch.float64)
    schedule_sigmas = scheduler.sigmas.to(device=noisy.device, dtype=torch.float64)
    indices = (schedule_times[None] - frame_times.double().reshape(-1, 1)).abs().argmin(dim=1)
    frame_sigmas = schedule_sigmas[indices].reshape(batch, frames, 1, 1, 1)
    sigma_map = frame_sigmas.expand(batch, frames, 1, height, width).clone()

    if region is not None:
        if (not isinstance(source_frame, Integral) or isinstance(source_frame, bool)
                or not 0 <= source_frame < frames):
            raise ValueError("source_frame must index the current latent chunk")
        if source_sigma is None or not math.isfinite(float(source_sigma)) or not 0 <= float(source_sigma) <= 1:
            raise ValueError("source_sigma must be finite and within [0,1]")
        coordinates = [region.get(key) for key in ("y0", "y1", "x0", "x1")]
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in coordinates):
            raise ValueError("region bounds must be integers")
        y0, y1, x0, x1 = coordinates
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("region must be nonempty and inside the latent frame")
        if y0 % patch_height or y1 % patch_height or x0 % patch_width or x1 % patch_width:
            raise ValueError("region must align with model patch_size")
        train_steps = float(scheduler.num_train_timesteps)
        if not math.isfinite(train_steps) or train_steps <= 0:
            raise ValueError("num_train_timesteps must be finite and positive")
        token_times[:, source_frame, y0 // patch_height:y1 // patch_height,
                    x0 // patch_width:x1 // patch_width] = float(source_sigma) * train_steps
        sigma_map[:, source_frame, :, y0:y1, x0:x1] = float(source_sigma)

    return token_times.flatten(1), sigma_map


@torch.inference_mode()
def predict_with_patch_timestep(pipeline, noisy, conditional, timestep, region=None,
                               source_frame=0, source_sigma=None, current_start=0):
    """Predict clean latents using matched local conditioning and conversion.

    Caches are intentionally the live rollout caches, exactly as in the regular
    wrapper call.  Callers should reset/replay them between paired branches.
    """
    generator = pipeline.generator
    if getattr(generator, "uniform_timestep", False):
        raise ValueError("patch timestep evaluation requires a causal generator")
    if pipeline.kv_cache1 is None:
        raise ValueError("patch timestep evaluation requires initialized KV caches")
    token_times, sigma_map = build_patch_timestep_maps(
        noisy, timestep, generator.scheduler, generator.model.patch_size,
        region=region, source_frame=source_frame, source_sigma=source_sigma)
    flow = generator.model(
        noisy.permute(0, 2, 1, 3, 4), t=token_times,
        context=conditional["prompt_embeds"], seq_len=generator.seq_len,
        kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start, cache_start=None,
    ).permute(0, 2, 1, 3, 4)
    if flow.shape != noisy.shape:
        raise ValueError("model flow shape differs from noisy latents")
    # Match the original wrapper's double-precision conversion and output dtype.
    return (noisy.double() - sigma_map * flow.double()).to(flow.dtype)
