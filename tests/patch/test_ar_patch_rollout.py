"""Exact trace/continuation and storage-contract checks on a tiny AR model."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.patch.ar_patch_rollout import (
    SCHEMA_VERSION, TracedRollout, artifact_sizes, branch_directory,
    check_identity, impact_maps, load_references, parse_args, read_reference, source_state, trace_key,
    trajectory_maps, trajectory_metadata, validate_completed, validate_old_rollout,
    validate_reference_baselines,
)
from scripts.patch.patch_noise_importance import PairedRollout, make_noise_bank
from scripts.patch.patch_noise_strength import pixel_metrics
from scripts.patch.patch_noise_utils import replace_patch_noise

_spec = importlib.util.spec_from_file_location("ar_trace_fixtures", Path(__file__).with_name("test_patch_noise_replay.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


class ARPatchRolloutTests(unittest.TestCase):
    def make_rollout(self, traced=True):
        pipeline = _fixtures.FakePipeline()
        pipeline.denoising_step_list = torch.tensor([1000.0, 937.5, 833.3333129882812, 625.0])
        pipeline.scheduler = _fixtures.FakeScheduler(pipeline.denoising_step_list)
        chunks = [3, 3]
        initial, transition = make_noise_bank((1, 6, 1, 2, 2), chunks, 4, 0,
                                              torch.float32, torch.device("cpu"))
        cls = TracedRollout if traced else PairedRollout
        return cls(pipeline, {"offset": 0.03125}, initial, transition, chunks)

    def test_hook_preserves_original_outputs_and_all_source_states(self):
        replay, original = self.make_rollout(), self.make_rollout(False)
        output = replay.baseline(0, 1)["output"]
        trace = replay.take_trace()
        self.assertEqual([trace_key(e) for e in trace], [(c, s) for c in range(2) for s in range(4)])
        for step in (1, 2, 3):
            expected = original.baseline(0, step)
            state = source_state(trace, replay.transition, 0, step, "cpu")
            for name in ("noisy", "prediction", "clean", "noise"):
                self.assertTrue(torch.equal(state[name], expected[name]), name)
            self.assertTrue(torch.equal(output, expected["output"]))
        self.assertEqual(len(trajectory_metadata(trace)), 8)
        self.assertEqual(replay.trace, [])
        with self.assertRaisesRegex(ValueError, "previous clean"):
            source_state(trace, replay.transition, 0, 0, "cpu")

    def test_trace_lengths_and_future_initial_input_are_correct(self):
        replay = self.make_rollout()
        baseline = replay.baseline(0, 1)["output"]
        trace = replay.take_trace()
        region = {"id": 0, "y0": 0, "y1": 1, "x0": 0, "x1": 1}
        before = {trace_key(e): (e["noisy"].clone(), e["clean_prediction"].clone()) for e in trace}
        for step in (1, 2, 3):
            state = source_state(trace, replay.transition, 0, step, "cpu")
            changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"], 0, region, 1)
            _, output = replay.branch(changed, 0, step, complete=True)
            branch = replay.take_trace()
            self.assertEqual(len(branch), 8 - step)
            self.assertEqual([trace_key(e) for e in branch],
                             [(0, s) for s in range(step, 4)] + [(1, s) for s in range(4)])
            future = next(e for e in branch if trace_key(e) == (1, 0))
            self.assertTrue(torch.equal(future["noisy"], before[1, 0][0]))
            self.assertFalse(torch.equal(future["clean_prediction"], before[1, 0][1]))
            self.assertFalse(torch.equal(output, baseline))
            for entry in trace:
                self.assertTrue(torch.equal(entry["noisy"], before[trace_key(entry)][0]))
                self.assertTrue(torch.equal(entry["clean_prediction"], before[trace_key(entry)][1]))
        checks = check_identity(replay, trace, baseline, [1, 2, 3], [region], 1)
        self.assertTrue(all(v["rollout_max_abs"] == 0 and v["one_step_max_abs"] == 0 for v in checks.values()))
        self.assertEqual(replay.trace, [])

    def test_trace_maps_use_channels_and_fixed_baseline(self):
        replay = self.make_rollout()
        replay.baseline(0, 1)
        trace = replay.take_trace()
        maps = trajectory_maps(trace, trace)
        self.assertEqual(maps["noisy_rms"].shape, (8, 3, 2, 2))
        self.assertTrue(np.all(maps["noisy_delta_rms"] == 0))
        self.assertTrue(np.all(maps["clean_delta_rms"] == 0))
        changed = dict(trace[0], noisy=trace[0]["noisy"] + 2)
        self.assertTrue(np.allclose(trajectory_maps([changed], trace)["noisy_delta_rms"], 2))

    def test_spatial_maps_preserve_full_float_mse(self):
        baseline = torch.zeros(1, 6, 2, 2, 3)
        changed = torch.arange(baseline.numel()).reshape_as(baseline).float() / 1000
        bp = torch.zeros(5, 16, 24, 3)
        cp = torch.arange(bp.numel()).reshape_as(bp).float() / 100000
        maps = impact_maps(baseline, changed, bp, cp)
        self.assertEqual(maps["latent_mse"].shape, (6, 2, 3))
        self.assertEqual(maps["pixel_mse_downsample"].shape, (5, 2, 3))
        self.assertAlmostEqual(float(maps["latent_mse"].mean()), float(changed.square().mean()), places=7)
        self.assertAlmostEqual(float(maps["pixel_mse_downsample"].mean()), float(cp.square().mean()), places=8)

    def test_resume_rejects_changed_protocol_identity_and_truncated_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "final.pt").write_bytes(b"abcd")
            artifacts = {"final_latent_pt": "final.pt"}
            record = {"schema_version": SCHEMA_VERSION, "protocol_sha256": "abc", "source_id": 95,
                      "artifacts": artifacts, "artifact_sizes": artifact_sizes(directory, artifacts)}
            path = directory / "record.json"
            path.write_text(json.dumps(record))
            self.assertEqual(validate_completed(path, "abc", {"source_id": 95}), record)
            with self.assertRaisesRegex(ValueError, "protocol"):
                validate_completed(path, "other")
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_completed(path, "abc", {"source_id": 96})
            (directory / "final.pt").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "truncated"):
                validate_completed(path, "abc")

    def test_strength_reference_filename_and_missing_reference_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(reference_dir=root / "fine", strength_reference_dir=root / "strength",
                                   prompt_index=0, seed=0)
            output = torch.zeros(1, 6, 1, 2, 2)
            pixels = torch.zeros(2, 8, 8, 3)
            metric = pixel_metrics(pixels, pixels)
            with self.assertRaisesRegex(ValueError, "strength reference is missing"):
                validate_old_rollout(args, 1, 95, 0.5, output, pixels, metric)
            case = args.strength_reference_dir / "prompt_000000_seed_000000_chunk_0_step_1"
            case.mkdir(parents=True)
            np.savez(case / "saved.npz", pixels=pixels.to(torch.uint8).numpy())
            (case / "strength_metrics.json").write_text(json.dumps({"records": [
                {"source_id": 95, "sigma_scale": 0.5, "pixels": metric, "artifacts": {"pixels_npz": "saved.npz"}}
            ]}))
            result = validate_old_rollout(args, 1, 95, 0.5, output, pixels, metric)
            self.assertTrue(result["strength_uint8"]["exact"])
            self.assertEqual(result["strength_pixel_metric"]["mse_abs_difference"], 0)
            np.savez(case / "saved.npz", pixels=np.ones(pixels.shape, dtype=np.uint8))
            with self.assertRaisesRegex(RuntimeError, "display pixels differ"):
                validate_old_rollout(args, 1, 95, 0.5, output, pixels, metric)

    def test_pilot_filters_do_not_change_complete_matrix(self):
        args = parse_args(["--source-ids", "95", "--only-step", "1", "--max-records", "4"])
        self.assertEqual(args.steps, [1, 2, 3])
        self.assertEqual(args.sigma_scales, [0.75, 0.5, 0.25, 0.0])
        self.assertEqual(args.grid_rows * args.grid_cols * len(args.steps) * len(args.sigma_scales), 2304)
        self.assertEqual(str(branch_directory("out", 1, 95, 0.75)), "out/step_1/source_095/scale_0p75")

    def test_step_one_protocol_and_new_prompt_policy(self):
        args = parse_args(["--prompt-index", "300", "--seed", "2", "--steps", "1",
                           "--reference-policy", "if-present"])
        self.assertEqual((args.prompt_index, args.seed, args.steps, args.reference_policy),
                         (300, 2, [1], "if-present"))
        self.assertEqual(args.grid_rows * args.grid_cols * len(args.steps) * len(args.sigma_scales), 768)
        self.assertEqual(parse_args([]).reference_policy, "required")

    def test_optional_absence_is_explicit_and_partial_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(reference_dir=root / "fine", strength_reference_dir=root / "strength",
                                   prompt_index=300, seed=0, reference_policy="if-present")
            path = root / "fine" / "prompt_000300_seed_000000_chunk_0_step_1" / "metrics.json"
            self.assertIsNone(read_reference(path))
            with self.assertRaisesRegex(ValueError, "Required reference is missing"):
                read_reference(path, required=True)
            result = validate_old_rollout(args, 1, 95, 0.75, torch.zeros(1), torch.zeros(1), {})
            self.assertEqual(result, {"reference_status": {"fine": "not_available", "strength": "not_available"}})
            path.parent.mkdir(parents=True)
            path.write_text("{")
            with self.assertRaises(json.JSONDecodeError):
                read_reference(path)
            path.write_text(json.dumps({"metadata": {}}))
            with self.assertRaisesRegex(ValueError, "Incomplete reference"):
                read_reference(path)
            path.unlink()
            (path.parent / "orphan.pt").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "metadata is missing"):
                read_reference(path)

    def test_available_reference_is_strict_under_optional_policy(self):
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"test checkpoint")
            args = parse_args(["--prompt-index", "300", "--steps", "1", "--reference-policy", "if-present",
                               "--reference-dir", str(root / "fine"), "--strength-reference-dir", str(root / "strength")])
            config = OmegaConf.create({"model": "fake"})
            regions = [{"id": 0, "y0": 0, "y1": 2, "x0": 0, "x1": 2}]
            schedule, sigmas = [1000., 937.5, 833.3, 625.], [1., .9375, .8333, .625]
            metadata = {"config": {"model": "fake"}, "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_size": checkpoint.stat().st_size, "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                        "num_output_frames": 6, "prompt_index": 300, "prompt": "A cat drinking water", "seed": 0,
                        "source_chunk": 0, "source_frame": 1, "step_index": 1, "weights": "generator",
                        "grid_rows": 12, "grid_cols": 16, "baseline_sha256": "baseline", "initial_noise_sha256": "initial",
                        "schedule": schedule, "global_t": 937.5, "base_sigma": .9375}
            data = {"metadata": metadata, "regions": regions,
                    "identity": {"one_step_max_abs": 0, "rollout_max_abs": 0},
                    "records": [{"source_id": 0, "sigma_scale": .75, "rollout": None}]}
            path = args.reference_dir / "prompt_000300_seed_000000_chunk_0_step_1" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(data))
            bundle, summary = load_references(args, config, checkpoint, metadata["prompt"], regions)
            self.assertEqual(summary["fine"]["1"]["status"], "available")
            self.assertEqual(summary["strength"]["1"], {"status": "not_available", "sha256": None,
                             "path": str((args.strength_reference_dir / path.parent.name / "strength_metrics.json").resolve())})
            validate_reference_baselines(bundle, "baseline", "initial", schedule, sigmas, {})
            with self.assertRaisesRegex(RuntimeError, "Baseline/initial-noise"):
                validate_reference_baselines(bundle, "wrong", "initial", schedule, sigmas, {})
            with self.assertRaisesRegex(ValueError, "Denoising schedule"):
                validate_reference_baselines(bundle, "baseline", "initial", [999.] + schedule[1:], sigmas, {})
            with self.assertRaisesRegex(ValueError, "Source/prompt/seed"):
                load_references(args, config, checkpoint, "different prompt", regions)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_references(args, OmegaConf.create({"model": "different"}), checkpoint, metadata["prompt"], regions)
            args.reference_policy = "required"
            with self.assertRaisesRegex(ValueError, "Required reference is missing"):
                load_references(args, config, checkpoint, metadata["prompt"], regions)
            args.reference_policy = "if-present"
            data["records"][0]["rollout"] = {"total_mse": 0}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "rollout artifact is missing"):
                load_references(args, config, checkpoint, metadata["prompt"], regions)


if __name__ == "__main__":
    unittest.main()
