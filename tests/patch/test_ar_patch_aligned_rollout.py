"""CPU rollout/control/storage tests; spatial token arithmetic has its own tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.patch import ar_patch_aligned_rollout as aligned
from scripts.patch import ar_patch_timestep_probe as probe
from scripts.patch import ar_patch_rollout as old
from scripts.patch.patch_noise_importance import make_noise_bank, tensor_hash
from scripts.patch.patch_noise_utils import replace_patch_noise

_spec = importlib.util.spec_from_file_location("aligned_replay_fixture", Path(__file__).with_name("test_patch_noise_replay.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def toy_token_prediction(pipeline, noisy, conditional, timestep, region=None,
                         source_frame=0, source_sigma=None, current_start=0):
    """Only a model stub: real local token/sigma arithmetic is tested separately.

    Keep the fixture's real KV recurrence, and introduce a local response to a
    changed source timestep. Uniform-time calls have exactly the old output.
    """
    prediction = pipeline.generator(noisy_image_or_video=noisy, conditional_dict=conditional,
        timestep=timestep, kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start)[1]
    if region is not None:
        prediction = prediction.clone()
        sigma = float(pipeline.scheduler.sigmas[(pipeline.scheduler.timesteps - timestep[0, source_frame]).abs().argmin()])
        prediction[:, source_frame, :, region["y0"]:region["y1"], region["x0"]:region["x1"]] += source_sigma - sigma
    return prediction


class AlignedRolloutTests(unittest.TestCase):
    region = {"id": 0, "row": 0, "col": 0, "y0": 0, "y1": 1, "x0": 0, "x1": 1}

    def make_rollout(self):
        pipeline = _fixture.FakePipeline()
        pipeline.denoising_step_list = torch.tensor([1000., 937.5, 833.3333129882812, 625.])
        pipeline.scheduler = _fixture.FakeScheduler(pipeline.denoising_step_list)
        pipeline.scheduler.num_train_timesteps = 1000
        initial, transition = make_noise_bank((1, 6, 1, 2, 2), [3, 3], 4, 0,
                                              torch.float32, torch.device("cpu"))
        replay = probe.PatchTimestepRollout(pipeline, {"offset": .03125}, initial, transition, [3, 3])
        baseline = replay.baseline(0, 1)["output"]
        return replay, baseline, replay.take_trace()

    def test_cli_defaults_never_target_old_experiments(self):
        args = aligned.parse_args([])
        self.assertEqual(args.steps, [1])
        self.assertEqual(args.num_shards, 4)
        self.assertEqual(args.trace_preview, "vae")
        self.assertEqual(args.output_dir, aligned.DEFAULT_ROOT / "data")
        self.assertEqual(args.global_reference_dir, Path("outputs/ar_patch_rollout_20261022/data"))
        args = aligned.parse_args(["--prompt-index", "300", "--source-ids", "95", "--shard-index", "3"])
        self.assertEqual(args.source_ids, [95])
        self.assertIn("prompt_000300_seed_000000", str(args.global_reference_dir))
        for output in ("outputs/ar_patch_rollout_20261022/new", "outputs/ar_patch_timestep_probe_20261022", "outputs"):
            with self.assertRaisesRegex(ValueError, "separate"):
                aligned.parse_args(["--output-dir", output])
        with self.assertRaisesRegex(ValueError, "steps"):
            aligned.parse_args(["--steps", "2"])

    @patch.object(probe, "predict_with_patch_timestep", side_effect=toy_token_prediction)
    def test_three_exact_controls_and_post_branch_cache_isolation(self, predictor):
        replay, baseline, trace = self.make_rollout()
        before = aligned.check_aligned_controls(replay, trace, baseline, [1], [self.region], 1)
        self.assertEqual(set(before), {"identity", "uniform_token_control", "matched_identity"})
        for result in before.values():
            self.assertEqual(result["1"], {"one_step_max_abs": 0., "rollout_max_abs": 0.,
                                           "rollout_sha256": tensor_hash(baseline)})
        state = old.source_state(trace, replay.transition, 0, 1, "cpu")
        changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"], 0, self.region, 1)
        replay.probe(changed, 1, "matched", self.region, 1, 0)
        replay.take_trace()
        after = aligned.check_aligned_controls(replay, trace, baseline, [1], [self.region], 1)
        self.assertEqual(before, after)
        self.assertTrue(replay.capture)
        self.assertEqual(replay.trace, [])
        self.assertIsNone(replay.intervention)

    def test_nonzero_uniform_control_fails_and_clears_override(self):
        replay, baseline, trace = self.make_rollout()
        def wrong(*args, **kwargs):
            return toy_token_prediction(*args, **kwargs) + .01
        with patch.object(probe, "predict_with_patch_timestep", side_effect=wrong):
            with self.assertRaisesRegex(RuntimeError, "uniform_token_control failed"):
                aligned.check_aligned_controls(replay, trace, baseline, [1], [self.region], 1)
        self.assertTrue(replay.capture)
        self.assertEqual(replay.trace, [])
        self.assertIsNone(replay.intervention)

    @patch.object(probe, "predict_with_patch_timestep", side_effect=toy_token_prediction)
    def test_exactly_one_override_seven_states_and_fixed_next_chunk_noise(self, predictor):
        replay, baseline, baseline_trace = self.make_rollout()
        state = old.source_state(baseline_trace, replay.transition, 0, 1, "cpu")
        original_initial = replay.initial.clone()
        original_transitions = {key: value.clone() for key, value in replay.transition.items()}
        for scale in (.75, .5, .25, 0):
            sigma = .9375 * scale
            changed = replace_patch_noise(state["noisy"], state["clean"], state["noise"], sigma, self.region, 1)
            calls_before = predictor.call_count
            _, output = replay.probe(changed, 1, "matched", self.region, 1, sigma)
            branch = aligned.annotate_trace(replay.take_trace(), 1, self.region, 1, sigma, sigma * 1000, baseline_trace)
            self.assertEqual(predictor.call_count - calls_before, 1)
            self.assertEqual([old.trace_key(entry) for entry in branch], [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)])
            self.assertTrue(torch.equal(branch[0]["noisy"], changed))
            self.assertEqual(branch[0]["local_timestep"], sigma * 1000)
            self.assertTrue(all("mode" not in entry for entry in branch[1:]))
            self.assertTrue(torch.equal(branch[3]["noisy"], baseline_trace[4]["noisy"]))
            self.assertFalse(torch.equal(branch[3]["clean_prediction"], baseline_trace[4]["clean_prediction"]))
            self.assertFalse(torch.equal(output, baseline))
        self.assertTrue(torch.equal(original_initial, replay.initial))
        for key, value in original_transitions.items():
            self.assertTrue(torch.equal(value, replay.transition[key]))
        bad = copy.deepcopy(branch)
        bad[-1]["mode"] = "matched"
        with self.assertRaisesRegex(RuntimeError, "leaked"):
            aligned.annotate_trace(bad, 1, self.region, 1, 0, 0, baseline_trace)

    def test_paired_input_strict_hashes_but_outputs_are_allowed_to_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = old.branch_directory(root, 1, 0, .5) / "record.json"
            path.parent.mkdir(parents=True)
            changed, delta = torch.ones(1, 3, 1, 2, 2), torch.full((1, 3, 1, 2, 2), .5)
            (path.parent / "artifact").write_bytes(b"ok")
            record = {"schema_version": old.SCHEMA_VERSION, "protocol_sha256": "old", "baseline_sha256": "baseline",
                      **old.branch_identity(1, self.region, .5), "source_chunk": 0, "source_frame": 1,
                      "input_sha256": tensor_hash(changed), "input_delta_sha256": tensor_hash(delta),
                      "output_sha256": "deliberately-old-output", "one_step": {"input_energy": 3.},
                      "rollout": {"total_mse": .4}, "pixel": {"all_video_mse": .2},
                      "trajectory": [{"chunk_index": 0, "step_index": 1, "noisy_sha256": tensor_hash(changed)}],
                      "artifacts": {"test": "artifact"}, "artifact_sizes": {"test": 2}}
            path.write_text(json.dumps(record))
            reference = {"protocol_sha256": "old", "protocol": {"baseline_sha256": "baseline"}}
            result = aligned.validate_paired_input(reference, root, 1, self.region, .5, 1, changed, delta)
            self.assertTrue(result["input_exact"] and result["input_delta_exact"])
            self.assertEqual(result["output_sha256"], "deliberately-old-output")
            self.assertEqual(result["record_sha256"], old.file_hash(path))
            for input_, delta_ in ((changed + 1, delta), (changed, delta + 1)):
                with self.assertRaisesRegex(RuntimeError, "input differs"):
                    aligned.validate_paired_input(reference, root, 1, self.region, .5, 1, input_, delta_)

    def make_completed(self, root):
        key = aligned.branch_identity(1, self.region, .5)
        key.update(source_sigma=.46875, local_timestep=468.75)
        artifacts = {name: name + ".bin" for name in aligned.REQUIRED_ARTIFACTS | {"impact_maps_npz", "trajectory_preview_npz"}}
        for filename in artifacts.values():
            (root / filename).write_bytes(b"complete")
        keys = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        trajectory = [{"chunk_index": c, "step_index": s, "noisy_sha256": "input"} for c, s in keys]
        trajectory[0].update(mode="matched", timestep_policy="source_call_only", source_frame=1,
                             region=self.region, source_sigma=.46875, local_timestep=468.75)
        record = {"schema_version": old.SCHEMA_VERSION, "protocol_sha256": "new", "baseline_sha256": "base",
                  **key, "trajectory": trajectory, "input_sha256": "input", "input_delta_sha256": "delta",
                  "paired_global": {"input_exact": True, "input_delta_exact": True,
                                    "input_sha256": "input", "input_delta_sha256": "delta"},
                  "artifacts": artifacts, "artifact_sizes": old.artifact_sizes(root, artifacts)}
        path = root / "record.json"
        path.write_text(json.dumps(record))
        return path, key, keys, record

    def test_resume_requires_full_artifacts_and_atomic_incomplete_retirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, key, keys, record = self.make_completed(root)
            self.assertEqual(aligned.completed_record(path, "new", "base", key, keys), record)
            (root / record["artifacts"]["trajectory_preview_npz"]).write_bytes(b"short")
            self.assertIsNone(aligned.completed_record(path, "new", "base", key, keys))
            aligned.retire_incomplete(path)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads((root / "record.incomplete.json").read_text()), record)
            self.assertIsNone(aligned.completed_record(path, "new", "base", key, keys))

    def test_resume_rejects_wrong_protocol_source_sequence_or_late_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, key, keys, original = self.make_completed(Path(temporary))
            for ph, bh, identity in (("wrong", "base", key), ("new", "wrong", key),
                                     ("new", "base", dict(key, local_timestep=0))):
                with self.assertRaises(ValueError):
                    aligned.completed_record(path, ph, bh, identity, keys)
            for change in ("sequence", "source", "later", "paired"):
                record = copy.deepcopy(original)
                if change == "sequence": record["trajectory"][1]["step_index"] = 3
                if change == "source": record["trajectory"][0]["local_timestep"] = 937.5
                if change == "later": record["trajectory"][-1]["mode"] = "matched"
                if change == "paired": record["paired_global"]["input_exact"] = False
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError, msg=change):
                    aligned.completed_record(path, "new", "base", key, keys)
            record = copy.deepcopy(original)
            record["trajectory"] = []
            path.write_text(json.dumps(record))
            self.assertIsNone(aligned.completed_record(path, "new", "base", key, keys))

    def test_no_matched_probe_reference_is_explicit(self):
        args = SimpleNamespace(prompt_index=300, seed=0)
        self.assertEqual(aligned.validate_matched_probe(args, 1, 95, .5, None, None, None, None), {"available": False})
        args.prompt_index = 0
        self.assertEqual(aligned.validate_matched_probe(args, 1, 95, .75, None, None, None, None), {"available": False})


if __name__ == "__main__":
    unittest.main()
