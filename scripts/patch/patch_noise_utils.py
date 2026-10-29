"""Paired patch-noise interventions and latent influence measurements.

Coordinates and ``halo`` are in latent pixels. A source region occupies only
``source_frame``; excluding it never excludes the same coordinates in other
frames. Metrics quantify sensitivity, not improvements in sample quality.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _video(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if tensor.ndim != 5 or any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must have nonempty shape [B, F, C, H, W]")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains nonfinite values")


def _bounds(region, height, width):
    if not isinstance(region, Mapping):
        raise ValueError("region must be a mapping")
    try:
        y0, y1, x0, x1 = (
            _integer(region[key], key) for key in ("y0", "y1", "x0", "x1")
        )
    except KeyError as error:
        raise ValueError(f"region is missing {error.args[0]}") from error
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("region must be nonempty and within the latent dimensions")
    return y0, y1, x0, x1


def build_regions(height, width, rows=4, cols=4, token_height=2, token_width=2):
    """Partition a latent into nonempty, token-aligned, row-major regions.

    The latent dimensions must be multiples of the token dimensions. Region
    heights/widths differ by at most one token and cover every pixel once.
    """
    height = _integer(height, "height", 1)
    width = _integer(width, "width", 1)
    rows = _integer(rows, "rows", 1)
    cols = _integer(cols, "cols", 1)
    token_height = _integer(token_height, "token_height", 1)
    token_width = _integer(token_width, "token_width", 1)
    if height % token_height or width % token_width:
        raise ValueError("latent dimensions must be divisible by token dimensions")
    token_rows, token_cols = height // token_height, width // token_width
    if rows > token_rows or cols > token_cols:
        raise ValueError("the grid cannot contain more regions than tokens")
    regions = []
    for row in range(rows):
        for col in range(cols):
            regions.append({
                "id": row * cols + col,
                "row": row,
                "col": col,
                "y0": (row * token_rows // rows) * token_height,
                "y1": ((row + 1) * token_rows // rows) * token_height,
                "x0": (col * token_cols // cols) * token_width,
                "x1": ((col + 1) * token_cols // cols) * token_width,
            })
    return regions


def replace_patch_noise(noisy, clean, noise, source_sigma, region, source_frame):
    """Clone ``noisy`` and re-corrupt one spatial region in one frame.

    ``source_sigma`` is a scalar in [0, 1]. As in
    ``FlowMatchScheduler.add_noise``, float32 sigma coefficients are broadcast
    over the sample and the result is cast back to the noise dtype. In
    particular, multiplying bf16 samples directly by Python scalars would
    produce different rounding and invalidate an unchanged-sigma control.
    """
    for name, tensor in (("noisy", noisy), ("clean", clean), ("noise", noise)):
        _video(tensor, name)
    if clean.shape != noisy.shape or noise.shape != noisy.shape:
        raise ValueError("noisy, clean, and noise must have the same shape")
    if clean.device != noisy.device or noise.device != noisy.device:
        raise ValueError("noisy, clean, and noise must be on the same device")
    if clean.dtype != noisy.dtype or noise.dtype != noisy.dtype:
        raise ValueError("noisy, clean, and noise must have the same dtype")
    source_frame = _integer(source_frame, "source_frame")
    if source_frame >= noisy.shape[1]:
        raise ValueError("source_frame is outside the input video")
    y0, y1, x0, x1 = _bounds(region, noisy.shape[-2], noisy.shape[-1])
    if isinstance(source_sigma, torch.Tensor):
        if source_sigma.numel() != 1 or not source_sigma.is_floating_point():
            raise ValueError("source_sigma must be a floating-point scalar")
        scalar_sigma = source_sigma.item()
    elif isinstance(source_sigma, Real) and not isinstance(source_sigma, bool):
        scalar_sigma = float(source_sigma)
    else:
        raise ValueError("source_sigma must be a real scalar")
    if not math.isfinite(scalar_sigma) or not 0 <= scalar_sigma <= 1:
        raise ValueError("source_sigma must be finite and within [0, 1]")
    # A four-dimensional coefficient preserves scheduler dtype promotion.
    sigma = torch.tensor(scalar_sigma, device=noisy.device, dtype=torch.float32)
    sigma = sigma.reshape(1, 1, 1, 1)
    index = (slice(None), source_frame, slice(None), slice(y0, y1), slice(x0, x1))
    replacement = ((1 - sigma) * clean[index] + sigma * noise[index]).to(noisy.dtype)
    if not torch.isfinite(replacement).all().item():
        raise ValueError("patch replacement contains nonfinite values")
    result = noisy.clone()
    result[index] = replacement
    return result


def _json_number(value):
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("influence metrics overflowed float32")
    return result


def _mean_or_none(values):
    return _json_number(values.mean()) if values.numel() else None


def measure_influence(
    baseline, perturbed, input_delta, region, source_frame, target_regions, halo=2
):
    """Measure paired output changes in float32 and return JSON-ready values.

    ``baseline`` and ``perturbed`` have shape [B, F, C, H, W]. ``input_delta``
    has matching B/C/H/W but may contain fewer frames than the output, e.g.
    when the output includes future rollout chunks. Its full squared norm is
    the intervention energy. ``source_frame`` uses the same frame origin in
    both tensors.

    ``outside_source_*`` excludes only the source box on ``source_frame``.
    ``outside_halo_mse`` excludes that box expanded by ``halo`` latent pixels,
    clipped to image bounds, on that same frame. All other frames remain in
    both outside measurements. ``other_frames_*`` excludes the entire source
    frame, providing the same target domain for every spatial source region.
    ``future_mse`` includes frames strictly after ``source_frame``. Empty regions have ``None`` metrics; zero input energy
    gives normalized energy 0.0. ``target_mse[frame][region]`` averages over
    batch, channels, and the target box, preserving the provided region order.
    """
    for name, tensor in (
        ("baseline", baseline), ("perturbed", perturbed), ("input_delta", input_delta)
    ):
        _video(tensor, name)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline and perturbed must have the same shape")
    if baseline.device != perturbed.device or baseline.device != input_delta.device:
        raise ValueError("all tensors must be on the same device")
    if (input_delta.shape[0], *input_delta.shape[2:]) != (
        baseline.shape[0], *baseline.shape[2:]
    ):
        raise ValueError("input_delta must match output batch/channel/spatial dimensions")
    source_frame = _integer(source_frame, "source_frame")
    if source_frame >= min(baseline.shape[1], input_delta.shape[1]):
        raise ValueError("source_frame must exist in both the input and output")
    halo = _integer(halo, "halo")
    height, width = baseline.shape[-2:]
    y0, y1, x0, x1 = _bounds(region, height, width)
    if not isinstance(target_regions, (list, tuple)):
        raise ValueError("target_regions must be a list or tuple of region mappings")
    targets = [_bounds(target, height, width) for target in target_regions]

    squared = (perturbed.float() - baseline.float()).square()
    input_energy_tensor = input_delta.float().square().sum()
    input_energy = _json_number(input_energy_tensor)
    output_energy = _json_number(squared.sum())
    outside_source_mask = torch.ones(
        (baseline.shape[1], height, width), dtype=torch.bool, device=baseline.device
    )
    outside_source_mask[source_frame, y0:y1, x0:x1] = False
    outside_halo_mask = outside_source_mask.clone()
    outside_halo_mask[
        source_frame,
        max(0, y0 - halo):min(height, y1 + halo),
        max(0, x0 - halo):min(width, x1 + halo),
    ] = False
    # Flatten only after broadcasting so each B/C element contributes once.
    outside_source = squared.masked_select(outside_source_mask[None, :, None])
    outside_halo = squared.masked_select(outside_halo_mask[None, :, None])
    outside_energy_tensor = outside_source.sum() if outside_source.numel() else None
    outside_energy = (
        _json_number(outside_energy_tensor) if outside_energy_tensor is not None else None
    )
    if input_energy == 0:
        normalized = 0.0
    elif outside_energy_tensor is None:
        normalized = None
    else:
        normalized = _json_number(outside_energy_tensor / input_energy_tensor)

    other_frames = torch.cat((squared[:, :source_frame], squared[:, source_frame + 1:]), dim=1)
    other_energy_tensor = other_frames.sum() if other_frames.numel() else None
    other_energy = _json_number(other_energy_tensor) if other_energy_tensor is not None else None
    normalized_other = (
        0.0 if input_energy == 0 else
        None if other_energy_tensor is None else
        _json_number(other_energy_tensor / input_energy_tensor)
    )

    # A dense source sweep evaluates every target box for every intervention.
    # Batch rectangle queries instead of synchronizing the GPU once per box.
    # Double-precision prefix sums prevent cancellation from obscuring small
    # box responses near a large response elsewhere in the image. The paired
    # difference and its square still use float32, as all other metrics do.
    frames = squared.shape[1]
    if targets:
        spatial_mean = squared.mean(dim=(0, 2), dtype=torch.float64)
        prefix = torch.zeros((frames, height + 1, width + 1),
                             dtype=torch.float64, device=baseline.device)
        prefix[:, 1:, 1:] = spatial_mean.cumsum(dim=-2).cumsum(dim=-1)
        boxes = torch.tensor(targets, dtype=torch.long, device=baseline.device)
        ty0, ty1, tx0, tx1 = boxes.unbind(dim=1)
        target_values = (prefix[:, ty1, tx1] - prefix[:, ty0, tx1]
                         - prefix[:, ty1, tx0] + prefix[:, ty0, tx0])
        target_values = target_values / ((ty1 - ty0) * (tx1 - tx0))[None]
        target_mse = target_values.cpu().tolist()
        if not all(math.isfinite(value) for frame in target_mse for value in frame):
            raise ValueError("influence metrics overflowed float32")
    else:
        target_mse = [[] for _ in range(frames)]
    per_frame_mse = squared.mean(dim=(0, 2, 3, 4)).cpu().tolist()
    if not all(math.isfinite(value) for value in per_frame_mse):
        raise ValueError("influence metrics overflowed float32")
    return {
        "total_mse": _mean_or_none(squared),
        "source_mse": _mean_or_none(squared[:, source_frame, :, y0:y1, x0:x1]),
        "outside_source_mse": _mean_or_none(outside_source),
        "outside_halo_mse": _mean_or_none(outside_halo),
        "future_mse": _mean_or_none(squared[:, source_frame + 1:]),
        "input_energy": input_energy,
        "output_energy": output_energy,
        "outside_source_energy": outside_energy,
        "normalized_outside_energy": normalized,
        "other_frames_mse": _mean_or_none(other_frames),
        "other_frames_energy": other_energy,
        "normalized_other_frames_energy": normalized_other,
        "per_frame_mse": per_frame_mse,
        "target_mse": target_mse,
    }
