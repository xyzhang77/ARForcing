import unittest

import torch

from scripts.patch.patch_group_importance import build_group_plan, group_metrics, replace_group
from scripts.patch.patch_noise_utils import build_regions


class GroupPlanTests(unittest.TestCase):
    def test_random_controls_are_distinct_reproducible_and_match_area(self):
        regions = build_regions(60, 104)
        ranking = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
        plan = build_group_plan(regions, ranking, seed=17)
        self.assertEqual(plan, build_group_plan(regions, ranking, seed=17))
        self.assertNotEqual(plan, build_group_plan(regions, ranking, seed=18))
        for k in (2, 4):
            top = next(g for g in plan if g["name"] == f"top_{k}")
            bottom = next(g for g in plan if g["name"] == f"bottom_{k}")
            controls = [g for g in plan if g["kind"] == "random" and g["k"] == k]
            self.assertEqual(len(controls), 4)
            self.assertEqual(len({tuple(g["source_ids"]) for g in controls}), 4)
            for group in controls:
                self.assertEqual(group["num_patches"], k)
                self.assertEqual(group["latent_area"], top["latent_area"])
                self.assertNotEqual(group["source_ids"], top["source_ids"])
            complement = next(g for g in plan if g["name"] == f"complement_{k}")
            self.assertFalse(set(complement["source_ids"]) & set(top["source_ids"]))
            self.assertEqual(complement["latent_area"] + top["latent_area"], 60 * 104)

    def test_invalid_k_and_incomplete_rankings_fail(self):
        regions = build_regions(60, 104)
        for ks in ([0], [16], [2, 2], [], [True]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                build_group_plan(regions, list(range(16)), top_ks=ks)
        with self.assertRaises(ValueError):
            build_group_plan(regions, list(range(15)))


class GroupMetricTests(unittest.TestCase):
    def test_target_excludes_entire_source_frame_and_future_is_next_chunk(self):
        baseline = torch.zeros(1, 6, 1, 2, 2)
        full = torch.ones_like(baseline)
        full[:, 1] = 100
        output = full * 0.5
        output[:, 1] = 999
        input_delta = torch.ones(1, 3, 1, 2, 2)
        result = group_metrics(baseline, output, input_delta, 1, full, next_chunk_start=3)
        self.assertEqual(result["common_target_frames"], [0, 2, 3, 4, 5])
        self.assertEqual(result["common_target_mse"], 0.25)
        self.assertEqual(result["common_target_energy"], 5)
        self.assertEqual(result["normalized_common_energy"], 5 / 12)
        self.assertEqual(result["delta_vs_full_relative_l2"], 0.5)
        self.assertEqual(result["cosine_to_full"], 1)
        self.assertEqual(result["next_chunk_mse"], 0.25)
        self.assertEqual(result["next_chunk_delta_vs_full_relative_l2"], 0.5)
        self.assertEqual(result["next_chunk_cosine_to_full"], 1)
        output[:, 2] = 1000
        result = group_metrics(baseline, output, input_delta, 1, full, next_chunk_start=3)
        self.assertEqual(result["next_chunk_mse"], 0.25)
        self.assertGreater(result["common_target_mse"], 0.25)

    def test_full_and_empty_references_and_orthogonal_response(self):
        baseline = torch.zeros(1, 3, 1, 1, 2)
        full = baseline.clone()
        full[:, :, :, :, 0] = 1
        other = baseline.clone()
        other[:, :, :, :, 1] = 1
        same = group_metrics(baseline, full, full, 1, full)
        self.assertEqual(same["delta_vs_full_relative_l2"], 0)
        self.assertEqual(same["cosine_to_full"], 1)
        perpendicular = group_metrics(baseline, other, other, 1, full)
        self.assertEqual(perpendicular["cosine_to_full"], 0)
        self.assertAlmostEqual(perpendicular["delta_vs_full_relative_l2"], 2 ** 0.5)
        empty = group_metrics(baseline, baseline, baseline, 1, full)
        self.assertEqual(empty["normalized_common_energy"], 0)
        self.assertEqual(empty["delta_vs_full_relative_l2"], 1)
        self.assertIsNone(empty["cosine_to_full"])

    def test_group_replacement_changes_only_union_in_one_frame(self):
        shape = (1, 3, 1, 4, 4)
        state = {"noisy": torch.ones(shape), "noise": torch.ones(shape), "clean": torch.zeros(shape)}
        regions = {r["id"]: r for r in build_regions(4, 4, rows=2, cols=2)}
        changed = replace_group(state, [0, 3], regions, 0.5, 1)
        expected = state["noisy"].clone()
        expected[:, 1, :, :2, :2] = 0.5
        expected[:, 1, :, 2:, 2:] = 0.5
        self.assertTrue(torch.equal(changed, expected))
        self.assertTrue(torch.equal(replace_group(state, [], regions, 0.5, 1), state["noisy"]))
        self.assertTrue(torch.equal(replace_group(state, list(regions), regions, 1, 1), state["noisy"]))


if __name__ == "__main__":
    unittest.main()
