"""Checks of intervention support and the paired sampling protocol; no weights."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.patch.wan_patch_probe import (
    choose_candidates, euler_rollout, frozen_candidates, intervene, noise_tensor,
    model_manifest, shifted_sigmas, stage_index, token_count,
)
from scripts.patch.patch_noise_importance import tensor_hash
from scripts.patch.patch_noise_utils import build_regions


class WanPatchProbeTests(unittest.TestCase):
    def test_huggingface_snapshot_symlinks_keep_logical_weight_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            # HF caches store extensionless blobs behind named snapshot links.
            blob = root / "0123456789abcdef"
            blob.write_bytes(b"test pretrained weight bytes")
            weight = snapshot / "diffusion_pytorch_model.safetensors"
            weight.symlink_to(blob)
            config_bytes = b'{"num_layers":30}'
            (snapshot / "config.json").write_bytes(config_bytes)
            result = {r["filename"]: r for r in model_manifest(snapshot)}
            self.assertEqual(result[weight.name]["path"], str(blob.resolve()))
            self.assertEqual(result[weight.name]["size"], blob.stat().st_size)
            self.assertEqual(result["config.json"]["sha256"], hashlib.sha256(config_bytes).hexdigest())
            weight.unlink()
            with self.assertRaises(RuntimeError):
                model_manifest(snapshot)

    def test_matched_grid_and_restart_identity(self):
        for fraction in (0, .25, .5):
            self.assertEqual(float(shifted_sigmas(12)[stage_index(12, fraction)]),
                             float(shifted_sigmas(48)[stage_index(48, fraction)]))
        self.assertEqual(float(shifted_sigmas(48)[12]), .9375)
        with self.assertRaises(ValueError):
            stage_index(12, .2)
        sigmas = shifted_sigmas(12)
        initial = noise_tensor((1, 3, 2, 8, 12), 42).float()
        def predict(state, sigma):
            # A state-dependent flow exposes accidental history, indexing,
            # integration precision or first-flow reuse mismatches.
            return state.to(torch.bfloat16).float() * .4 + sigma
        baseline, snapshot = euler_rollout(predict, initial, sigmas, capture=3)
        replay, _ = euler_rollout(predict, snapshot["state"], sigmas, start=3)
        reuse, _ = euler_rollout(predict, snapshot["state"], sigmas, start=3,
                                 first_flow=snapshot["flow"])
        self.assertTrue(torch.equal(baseline, replay))
        self.assertTrue(torch.equal(replay, reuse))
        self.assertEqual(baseline.dtype, torch.float32)

    def test_local_sigma_intervention_support_and_endpoint(self):
        state = noise_tensor((1, 3, 2, 8, 12), 3).float()
        flow = noise_tensor(state.shape, 9).float()
        region = {"id": 0, "y0": 2, "y1": 6, "x0": 4, "x1": 10}
        identity = intervene(state, flow, .9375, region, 1, scale=1)
        self.assertTrue(torch.equal(state, identity))
        self.assertNotEqual(state.data_ptr(), identity.data_ptr())
        changed = intervene(state, flow, .9375, region, 1, scale=0)
        box = (slice(None), 1, slice(None), slice(2, 6), slice(4, 10))
        self.assertTrue(torch.equal(changed[box], (state - .9375 * flow)[box]))
        mask = torch.ones_like(state, dtype=torch.bool)
        mask[box] = False
        self.assertTrue(torch.equal(state[mask], changed[mask]))
        with self.assertRaises(ValueError):
            intervene(state, flow, .9375, region, 3)

    def test_donor_same_across_steps_and_exact_source_support(self):
        shape = (1, 3, 2, 8, 12)
        initial = noise_tensor(shape, 1).float()
        donor12 = noise_tensor(shape, 1 + 1000003).float()
        donor48 = noise_tensor(shape, 1 + 1000003).float()
        self.assertEqual(tensor_hash(donor12), tensor_hash(donor48))
        self.assertFalse(torch.equal(initial, donor12))
        region = {"id": 1, "y0": 0, "y1": 4, "x0": 2, "x1": 6}
        a = intervene(initial, initial, 1, region, 1, donor=donor12)
        b = intervene(initial, initial * 3, 1, region, 1, donor=donor48)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, 1, :, :4, 2:6], donor12[:, 1, :, :4, 2:6]))
        restored = a.clone()
        restored[:, 1, :, :4, 2:6] = initial[:, 1, :, :4, 2:6]
        self.assertTrue(torch.equal(restored, initial))
        with self.assertRaises(ValueError):
            intervene(initial, initial, .9375, region, 1, donor=donor12)

    def test_area_matched_random_and_frozen_sources(self):
        regions = build_regions(60, 104, 12, 16)
        records = [{"source_id": r["id"], "one_step": {"normalized_other_frames_energy":
                    200 if r["id"] == 17 else 199 if r["id"] == 53 else 1}} for r in regions]
        candidates = choose_candidates(records, regions, seed=0)
        self.assertEqual([c["source_id"] for c in candidates[:2]], [17, 53])
        self.assertEqual(candidates, choose_candidates(records, regions, seed=0))
        self.assertEqual(len({r["source_id"] for r in candidates}), 4)
        for control in candidates[2:]:
            self.assertEqual(token_count(control["region"]), token_count(regions[control["matched_top_id"]]))
        selection = {"metadata": {"prompt_index": 42, "seed": 0, "steps": 48,
                                  "stage_fraction": .25, "intervention": "sigma_reduce",
                                  "fixed_candidates": candidates}}
        frozen = frozen_candidates(selection, regions, 42, 0, .25, "sigma_reduce")
        self.assertEqual(frozen, candidates)
        self.assertEqual(frozen_candidates(selection, regions, 42, 0, 0, "initial_noise_replace"), candidates)
        with self.assertRaises(ValueError):
            frozen_candidates(selection, regions, 42, 1, .25, "sigma_reduce")
        with self.assertRaises(ValueError):
            frozen_candidates(selection, regions, 42, 0, .5, "sigma_reduce")
        broken = copy.deepcopy(selection)
        broken["metadata"]["fixed_candidates"][2]["region"]["x1"] += 2
        with self.assertRaises(ValueError):
            frozen_candidates(broken, regions, 42, 0, .25, "sigma_reduce")


if __name__ == "__main__":
    unittest.main()
