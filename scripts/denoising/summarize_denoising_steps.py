"""Compare clean predictions along paired AR and Wan denoising trajectories.

The original first-step experiment is read-only. Results are written beneath
its denoising_steps/ directory; noisy solver states are never image metrics.
"""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.denoising.summarize_first_step_noise import (
    distance_matrices, interval, lowpass, pairwise_summary, retention_summary,
    retrieval_scores, safe_ratio, seed_weights, weighted_offdiag,
)


SIGMAS = (1.0, 0.9375, 0.8333333134651184, 0.625)
WAN_KEYS = ("first", "quarter", "half", "threequarter")
VIEW_SHAPE = (3, 240, 416, 3)
SCHEMA_VERSION = "denoising_steps_summary_v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_record(path, model, prompt, seed, inventory):
    """Require both latent and view hashes, even though metrics use views."""
    record = json.loads(path.read_text())
    expected = {"model": model, "prompt_index": prompt, "seed": seed, "complete": True}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Record identity/incomplete status differs: {path}")
    hashes = record.get("artifact_sha256", {})
    for name in ("latents.pt", "views.npz"):
        artifact = path.parent / name
        if not hashes.get(name) or not artifact.is_file() or file_hash(artifact) != hashes[name]:
            raise ValueError(f"Missing or changed artifact: {artifact}")
    inventory.append({"record": str(path), "record_sha256": file_hash(path),
                      "artifact_sha256": {name: hashes[name] for name in ("latents.pt", "views.npz")}})
    return record


def load_views(path, keys):
    with np.load(path.parent / "views.npz", allow_pickle=False) as arrays:
        if not set(keys) <= set(arrays.files):
            raise ValueError(f"Missing view keys {set(keys) - set(arrays.files)}: {path}")
        result = {key: arrays[key].copy() for key in keys}
    for key, value in result.items():
        if value.shape != VIEW_SHAPE or value.dtype != np.uint8:
            raise ValueError(f"Invalid {key} view schema {value.shape}/{value.dtype}: {path}")
    return result


def validate_stages(record, keys, indices, path):
    for key, sigma, index in zip(keys, SIGMAS, indices):
        stage = record.get("stages", {}).get(key, {})
        if (not np.isclose(stage.get("sigma", -1), sigma, rtol=0, atol=1e-7)
                or stage.get("step_index") != index):
            raise ValueError(f"Invalid sigma or index for {key}: {path}")


def load_cases(root):
    """Fail closed on partial samples, changed sources, or mismatched noise."""
    root = Path(root).resolve()
    experiment = json.loads((root / "experiment.json").read_text())
    prompts, seeds = experiment["prompt_indices"], experiment["seeds"]
    if len(set(prompts)) != 3 or len(prompts) != 3 or len(set(seeds)) != 8 or len(seeds) != 8:
        raise ValueError("Expected the original three prompts and eight seeds per prompt")
    expected = {(int(prompt), int(seed)) for prompt in prompts for seed in seeds}
    for folder in (root / "ar", root / "wan", root / "denoising_steps" / "ar"):
        paths = list(folder.glob("prompt_*/seed_*/record.json"))
        actual = {(int(path.parent.parent.name.split("_")[1]), int(path.parent.name.split("_")[1]))
                  for path in paths}
        if actual != expected or len(paths) != 24:
            raise ValueError(f"Expected exactly 24 cases under {folder}; missing={expected-actual}, extra={actual-expected}")
    cases, inventory = [], []
    for prompt, seed in sorted(expected):
        suffix = Path(f"prompt_{prompt:06d}") / f"seed_{seed:06d}" / "record.json"
        original_path = root / "ar" / suffix
        wan_path = root / "wan" / suffix
        ar_path = root / "denoising_steps" / "ar" / suffix
        original = checked_record(original_path, "ar", prompt, seed, inventory)
        wan = checked_record(wan_path, "wan", prompt, seed, inventory)
        ar = checked_record(ar_path, "ar", prompt, seed, inventory)
        source = ar.get("source_record", {})
        if (source.get("sha256") != file_hash(original_path)
                or source.get("artifact_sha256") != original["artifact_sha256"]):
            raise ValueError(f"AR source record or source artifacts differ: {ar_path}")
        if ar.get("schema_version") != "denoising_steps_ar_v1":
            raise ValueError(f"Unexpected AR trace schema: {ar_path}")
        if any(ar.get(key) != original.get(key) for key in ("checkpoint_path", "weights")):
            raise ValueError(f"AR rerun checkpoint or weights differ: {ar_path}")
        if (ar.get("checks", {}).get("final_equals_x0_step4") is not True
                or ar.get("checks", {}).get("noise_bank_matches_original") is not True):
            raise ValueError(f"AR trajectory replay checks are missing: {ar_path}")
        text = experiment["prompts"][str(prompt)]
        if any(record.get("prompt") != text for record in (original, wan, ar)):
            raise ValueError(f"Prompt text differs for prompt={prompt}, seed={seed}")
        for key in ("initial_noise_sha256", "first_chunk_noise_sha256"):
            if not original.get(key) or any(record.get(key) != original[key] for record in (ar, wan)):
                raise ValueError(f"Paired noise mismatch ({key}) for prompt={prompt}, seed={seed}")
        ar_keys = tuple(f"x0_step{number}" for number in range(1, 5))
        validate_stages(ar, ar_keys, range(4), ar_path)
        validate_stages(wan, WAN_KEYS, (0, 12, 24, 36), wan_path)
        ar_views = load_views(ar_path, (*ar_keys, "final"))
        old_views = load_views(original_path, ("first", "second", "final"))
        for new_key, old_key in ((ar_keys[0], "first"), (ar_keys[1], "second"), ("final", "final")):
            if not np.array_equal(ar_views[new_key], old_views[old_key]):
                raise ValueError(f"AR rerun differs from original {old_key} view: {ar_path}")
        if not np.array_equal(ar_views[ar_keys[-1]], ar_views["final"]):
            raise ValueError(f"AR step4 is not identical to final: {ar_path}")
        wan_views = load_views(wan_path, (*WAN_KEYS, "final"))
        for model, record, path, views, source_keys in (
                ("ar", ar, ar_path, ar_views, ar_keys),
                ("wan", wan, wan_path, wan_views, WAN_KEYS)):
            ordered_views = {f"step{number}": views[key] for number, key in enumerate(source_keys, 1)}
            ordered_views["final"] = views["final"]
            cases.append({"model": model, "prompt_index": prompt, "prompt": text, "seed": seed,
                          "record": record, "path": str(path), "views": ordered_views,
                          "source_keys": dict(zip(ordered_views, (*source_keys, "final"))),
                          "features": {key: lowpass(value) for key, value in ordered_views.items()}})
    return cases, inventory


def high_frequency(views):
    """RGB residual after BOX 15x26 lowpass and bilinear reconstruction."""
    coarse = lowpass(views)
    reconstructed = np.empty(views.shape, dtype=np.float32)
    for frame in range(len(views)):
        for channel in range(3):
            plane = Image.fromarray(coarse[frame, :, :, channel].astype(np.float32))
            reconstructed[frame, :, :, channel] = np.asarray(
                plane.resize((views.shape[2], views.shape[1]), Image.Resampling.BILINEAR))
    return views.astype(np.float32) / 255.0 - reconstructed


def mean_and_ci(values, weights):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), interval(weights @ values / len(values))


def normalized_change(values, final_matrix, weights):
    mask = ~np.eye(len(final_matrix), dtype=bool)
    denominator = float(final_matrix[mask].mean())
    boot_denominator = weighted_offdiag(final_matrix, weights)
    boot_numerator = weights @ np.asarray(values) / len(values)
    boot_ratio = np.divide(boot_numerator, boot_denominator,
                           out=np.full(len(weights), np.nan), where=boot_denominator > 1e-15)
    return {"final_interseed_lowpass_mse": denominator,
            "normalized_to_final_interseed_mse": safe_ratio(float(np.mean(values)), denominator),
            "normalized_to_final_interseed_mse_ci95": interval(boot_ratio)}


def paired_change(left, right, left_stage, right_stage, final_matrix=None):
    """One change per matched seed; CIs resample seeds within one prompt."""
    if [case["seed"] for case in left] != [case["seed"] for case in right]:
        raise ValueError("Paired change requires identical seed order")
    weights = seed_weights(len(left))
    per_seed = {name: [] for name in ("lowpass_mse", "rgb_mse", "high_frequency_rgb_mse")}
    for before, after in zip(left, right):
        a, b = before["views"][left_stage], after["views"][right_stage]
        low_delta = before["features"][left_stage] - after["features"][right_stage]
        per_seed["lowpass_mse"].append(float(np.mean(low_delta ** 2)))
        rgb_delta = a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0
        per_seed["rgb_mse"].append(float(np.mean(rgb_delta.astype(np.float64) ** 2)))
        high_delta = high_frequency(a) - high_frequency(b)
        per_seed["high_frequency_rgb_mse"].append(float(np.mean(high_delta.astype(np.float64) ** 2)))
    result = {"n_seeds": len(left), "seeds": [case["seed"] for case in left], "per_seed": per_seed}
    for name, values in per_seed.items():
        result[name], result[f"{name}_ci95"] = mean_and_ci(values, weights)
    if final_matrix is not None:
        result.update(normalized_change(per_seed["lowpass_mse"], final_matrix, weights))
    return result


def to_final_summary(group, stage, final_matrix):
    features = [case["features"][stage] for case in group]
    finals = [case["features"]["final"] for case in group]
    result = retention_summary(features, finals)
    # Use direct per-seed deltas so exact identities remain exactly zero.
    matched = np.array([float(np.mean((a - b) ** 2)) for a, b in zip(features, finals)])
    weights = seed_weights(len(group))
    result["matched_mse"], result["matched_mse_ci95"] = mean_and_ci(matched, weights)
    result["matched_rmse"] = float(np.sqrt(matched).mean())
    matrix = np.array(result["mse_matrix"])
    np.fill_diagonal(matrix, matched)
    result["mse_matrix"] = matrix.tolist()
    result["matched_over_mismatched_mse"] = safe_ratio(result["matched_mse"], result["mismatched_mse"])
    scores = retrieval_scores(matrix)
    result["retrieval_accuracy"] = float(scores.mean())
    result["retrieval_per_seed"] = scores.tolist()
    result["retrieval_ci95"] = interval(weights @ scores / len(group))
    bootstrap_mismatch = weighted_offdiag(matrix, weights)
    bootstrap_match = weights @ matched / len(group)
    result["ratio_ci95"] = interval(np.divide(bootstrap_match, bootstrap_mismatch,
                                             out=np.full(len(weights), np.nan), where=bootstrap_mismatch > 1e-15))
    result["per_seed_lowpass_mse"] = matched.tolist()
    result.update(normalized_change(matched, final_matrix, weights))
    return result


def analyze(cases, inventory):
    per_prompt, rows = [], []
    for prompt in sorted({case["prompt_index"] for case in cases}):
        selected = [case for case in cases if case["prompt_index"] == prompt]
        item = {"prompt_index": prompt, "prompt": selected[0]["prompt"],
                "seeds": sorted({case["seed"] for case in selected}), "models": {}, "cross_model": []}
        for model in ("ar", "wan"):
            group = sorted((case for case in selected if case["model"] == model), key=lambda case: case["seed"])
            finals = [case["features"]["final"] for case in group]
            final_matrix = distance_matrices(finals)["mse"]
            stages, adjacent = [], []
            for number, stage in enumerate(("step1", "step2", "step3", "step4", "final"), 1):
                identity = stage == "final" or (model == "ar" and stage == "step4")
                stage_item = {"stage": stage, "source_key": group[0]["source_keys"][stage],
                              "sigma": SIGMAS[number - 1] if stage != "final" else 0.0,
                              "step_number": number if stage != "final" else None,
                              "construction_identity_with_final": identity,
                              "to_final": to_final_summary(group, stage, final_matrix),
                              "diversity": pairwise_summary([case["features"][stage] for case in group])}
                if identity:
                    stage_item["identity_note"] = (
                        "AR step4 is the final first-chunk prediction by construction; zero distance is not independent evidence of structural convergence."
                        if model == "ar" else "Final is the reference output itself; zero distance is a construction identity.")
                if stages:
                    previous = stages[-1]["to_final"]["matched_mse"]
                    current = stage_item["to_final"]["matched_mse"]
                    stage_item["to_final"]["mse_change_from_previous_stage"] = current - previous
                    stage_item["to_final"]["mse_ratio_to_previous_stage"] = safe_ratio(current, previous)
                    stage_item["to_final"]["mse_increased_from_previous_stage"] = current > previous + 1e-12
                stages.append(stage_item)
                append_scalars(rows, prompt, model, "to_final", stage, "final", stage_item["sigma"],
                               stage_item["to_final"], identity)
                for metric, stats in stage_item["diversity"]["metrics"].items():
                    rows.append(make_row(prompt, model, "diversity", stage, None, stage_item["sigma"],
                                         metric, stats["mean"], stats["ci95_seed_bootstrap"], len(group), identity))
            pairs = [("step1", "step2"), ("step2", "step3"), ("step3", "step4")]
            if model == "wan":
                pairs.append(("step4", "final"))
            for before, after in pairs:
                change = paired_change(group, group, before, after, final_matrix)
                change.update(from_stage=before, to_stage=after)
                if adjacent:
                    for name in ("lowpass_mse", "rgb_mse", "high_frequency_rgb_mse"):
                        change[f"{name}_ratio_to_previous_transition"] = safe_ratio(change[name], adjacent[-1][name])
                adjacent.append(change)
                append_scalars(rows, prompt, model, "adjacent", before, after, None, change)
            mask = ~np.eye(len(group), dtype=bool)
            item["models"][model] = {"n_seeds": len(group), "stage_metrics": stages,
                                      "adjacent_metrics": adjacent,
                                      "convergence": convergence_description(stages, adjacent, model),
                                      "final_interseed_lowpass_mse": float(final_matrix[mask].mean()),
                                      "run_parameters": group[0]["record"].get("protocol", {}),
                                      "records": [case["path"] for case in group]}
        ar = sorted((case for case in selected if case["model"] == "ar"), key=lambda case: case["seed"])
        wan = sorted((case for case in selected if case["model"] == "wan"), key=lambda case: case["seed"])
        for number, stage in enumerate(("step1", "step2", "step3", "step4", "final"), 1):
            change = paired_change(ar, wan, stage, stage)
            change.update(stage=stage, sigma=SIGMAS[number - 1] if stage != "final" else 0.0)
            if stage == "final":
                change["sigma_note"] = "Both outputs are final, reached by different samplers; AR uses its step4 x0 directly."
            item["cross_model"].append(change)
            append_scalars(rows, prompt, "ar_vs_wan", "cross_model", stage, stage, change["sigma"], change)
        per_prompt.append(item)
    return {"schema_version": SCHEMA_VERSION,
            "n_cases": len(cases), "n_prompts": len(per_prompt), "n_seeds_per_prompt": 8,
            "metric_protocol": {
                "view_shape": list(VIEW_SHAPE), "decoded_frames": [0, 4, 8],
                "lowpass_shape": [3, 15, 26, 3], "lowpass": "Pillow BOX resize; RGB range [0,1]",
                "high_frequency_rgb": "Original 240x416 RGB minus bilinear upsampling of the BOX 15x26 lowpass; float32 reconstruction, MSE accumulated in float64",
                "normalization": "Matched lowpass MSE divided by mean off-diagonal final-vs-final lowpass MSE within the same model and prompt; null if denominator is zero",
                "retention_ratio": "matched_over_mismatched_mse retains the old stage-vs-final off-diagonal denominator; it differs from normalized_to_final_interseed_mse",
                "bootstrap": "2000 seed resamples independently within each prompt, fixed RNG seed 20261023; no pooling of the 24 samples as independent prompts",
                "retrieval": "Same-prompt 8-way nearest-final retrieval; ties share credit; chance 1/8",
                "interpretation": "Lowpass RGB is a layout/color proxy. High-frequency RGB includes texture, edges, fine displacement and decoder effects; neither establishes semantic structure."},
            "alignment": [{"step_number": i + 1, "sigma": sigma, "ar_probe_number": i + 1,
                            "wan_probe_number": i * 12 + 1, "wan_source_key": WAN_KEYS[i]}
                           for i, sigma in enumerate(SIGMAS)],
            "warnings": [
                "AR step4 equals its final first-chunk clean prediction by construction; its zero-to-final error is not independent convergence evidence.",
                "Only clean x0 predictions and final outputs are analyzed. Saved noisy state_step* tensors are excluded.",
                "The aligned predictions come from the same respective trajectory within each seed, but AR and Wan sampler paths, learned weights, conditioning and temporal contexts differ.",
                "Three fixed prompts and eight seeds each support descriptive comparisons; no pooled population confidence interval is reported.",
                "Lowpass changes cannot by themselves distinguish semantic layout changes from color, brightness or decoder changes."],
            "validation": {"complete": True, "original_ar_cases": 24, "original_wan_cases": 24,
                           "new_ar_cases": 24, "all_latent_and_view_hashes_valid": True,
                           "all_source_record_hashes_valid": True, "all_paired_initial_noise_hashes_match": True,
                           "all_ar_step1_step2_final_views_match_original": True,
                           "all_ar_step4_views_equal_final": True, "inventory": inventory},
            "per_prompt": per_prompt}, rows


def convergence_description(stages, adjacent, model):
    """Describe observed sequences without assuming denoising is monotonic."""
    errors = [stage["to_final"]["matched_mse"] for stage in stages[:4]]
    errors_before_identity = errors[:3] if model == "ar" else errors
    return {
        "step1_to_step4_to_final_lowpass_mse": errors,
        "step1_to_step4_to_final_lowpass_mse_relative_to_step1": [safe_ratio(value, errors[0]) for value in errors],
        "to_final_lowpass_mse_nonincreasing_step1_to_step4": all(b <= a + 1e-12 for a, b in zip(errors, errors[1:])),
        "to_final_lowpass_mse_nonincreasing_before_forced_identity": all(
            b <= a + 1e-12 for a, b in zip(errors_before_identity, errors_before_identity[1:])),
        "adjacent_lowpass_mse": [change["lowpass_mse"] for change in adjacent],
        "adjacent_high_frequency_rgb_mse": [change["high_frequency_rgb_mse"] for change in adjacent],
        "adjacent_lowpass_mse_nonincreasing": all(
            b["lowpass_mse"] <= a["lowpass_mse"] + 1e-12 for a, b in zip(adjacent, adjacent[1:])),
        "adjacent_high_frequency_rgb_mse_nonincreasing": all(
            b["high_frequency_rgb_mse"] <= a["high_frequency_rgb_mse"] + 1e-12 for a, b in zip(adjacent, adjacent[1:])),
        "interpretation": "These are descriptive sequences for one prompt. Decreasing x0-to-own-final errors are not proof that semantic structure is fixed; AR step4-to-final equality is constructed.",
    }


def make_row(prompt, model, analysis, stage, to_stage, sigma, metric, value, ci, n, identity=False):
    ci = ci or [None, None]
    return dict(prompt_index=prompt, model=model, analysis=analysis, stage=stage, to_stage=to_stage,
                sigma=sigma, metric=metric, value=value, ci_low=ci[0], ci_high=ci[1], n_seeds=n,
                construction_identity_with_final=identity)


def append_scalars(rows, prompt, model, analysis, stage, to_stage, sigma, values, identity=False):
    excluded = {"n_seeds", "sigma", "step_number"}
    for key, value in values.items():
        if key in excluded or not (value is None or isinstance(value, (int, float))):
            continue
        ci = values.get(f"{key}_ci95")
        if key == "matched_over_mismatched_mse":
            ci = values.get("ratio_ci95")
        elif key == "retrieval_accuracy":
            ci = values.get("retrieval_ci95")
        rows.append(make_row(prompt, model, analysis, stage, to_stage, sigma, key, value, ci,
                             values["n_seeds"], identity))


def self_test():
    """Check meaningful identities, residual decomposition, and zero denominators."""
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, VIEW_SHAPE, dtype=np.uint8)
    constant = np.full(VIEW_SHAPE, 71, dtype=np.uint8)
    assert np.max(np.abs(high_frequency(constant))) < 1e-6
    cases = [{"seed": i, "views": {"a": value, "b": value},
              "features": {"a": lowpass(value), "b": lowpass(value)}}
             for i, value in enumerate((image, constant))]
    change = paired_change(cases, cases, "a", "b")
    assert all(change[key] == 0 for key in ("lowpass_mse", "rgb_mse", "high_frequency_rgb_mse"))
    weights = seed_weights(2)
    assert normalized_change([0., 0.], np.zeros((2, 2)), weights)["normalized_to_final_interseed_mse"] is None
    matrix = np.array([[0., .2], [.2, 0.]])
    assert np.isclose(normalized_change([.1, .1], matrix, weights)["normalized_to_final_interseed_mse"], .5)
    # A uniform brightness shift changes lowpass RGB, but has no high-frequency change.
    a, b = np.full(VIEW_SHAPE, 70, dtype=np.uint8), np.full(VIEW_SHAPE, 90, dtype=np.uint8)
    assert np.mean((high_frequency(a) - high_frequency(b)) ** 2) < 1e-12
    assert np.mean((lowpass(a) - lowpass(b)) ** 2) > 0
    with tempfile.TemporaryDirectory(prefix="denoising_step_summary_test_") as temporary:
        folder = Path(temporary)
        for name in ("latents.pt", "views.npz"):
            (folder / name).write_bytes(b"source artifact")
        record = {"model": "ar", "prompt_index": 42, "seed": 0, "complete": True,
                  "artifact_sha256": {name: file_hash(folder / name) for name in ("latents.pt", "views.npz")}}
        path = folder / "record.json"
        path.write_text(json.dumps(record))
        assert checked_record(path, "ar", 42, 0, [])["complete"]
        (folder / "latents.pt").write_bytes(b"changed artifact")
        try:
            checked_record(path, "ar", 42, 0, [])
        except ValueError:
            pass
        else:
            raise AssertionError("Changed latent artifact was accepted")
    fake_stages = [{"to_final": {"matched_mse": error}} for error in (.2, .3, .1, 0.)]
    fake_changes = [{"lowpass_mse": error, "high_frequency_rgb_mse": error} for error in (.1, .2, .05)]
    assert not convergence_description(fake_stages, fake_changes, "ar")["to_final_lowpass_mse_nonincreasing_step1_to_step4"]
    assert not convergence_description(fake_stages, fake_changes, "ar")["adjacent_lowpass_mse_nonincreasing"]
    json.dumps(change, allow_nan=False)
    print("self-test passed: exact identities, residual definition, zero denominators, latent artifact hashes, nonmonotonic sequences")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Original experiment root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        if args.output_dir is None:
            return
    if args.output_dir is None:
        parser.error("--output-dir is required unless only running --self-test")
    cases, inventory = load_cases(args.output_dir)
    summary, rows = analyze(cases, inventory)
    destination = args.output_dir.resolve() / "denoising_steps"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(cases), "prompts": len(summary["per_prompt"]),
                      "summary": str(destination / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
