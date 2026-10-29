"""CPU unit tests: python -m unittest discover -s tests -p test_patch_noise_importance.py."""

import json
import unittest

import torch

from scripts.patch.patch_noise_utils import build_regions, measure_influence, replace_patch_noise
from utils.scheduler import FlowMatchScheduler


class RegionTests(unittest.TestCase):
    def test_token_alignment_and_exact_partition(self):
        regions = build_regions(14, 20, rows=3, cols=4, token_height=2, token_width=2)
        coverage = torch.zeros(14, 20, dtype=torch.int32)
        for index, region in enumerate(regions):
            self.assertEqual(region["id"], index)
            self.assertEqual((region["row"], region["col"]), divmod(index, 4))
            for key in ("y0", "y1", "x0", "x1"):
                self.assertEqual(region[key] % 2, 0)
            self.assertGreater(region["y1"], region["y0"])
            self.assertGreater(region["x1"], region["x0"])
            coverage[region["y0"]:region["y1"], region["x0"]:region["x1"]] += 1
        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
        heights = [r["y1"] - r["y0"] for r in regions]
        widths = [r["x1"] - r["x0"] for r in regions]
        self.assertLessEqual(max(heights) - min(heights), 2)
        self.assertLessEqual(max(widths) - min(widths), 2)

    def test_invalid_grid(self):
        for kwargs in (
            {"height": 0}, {"width": 7}, {"rows": 5}, {"cols": 0},
            {"token_height": 3}, {"token_width": -1}, {"rows": 1.5}, {"cols": True},
        ):
            arguments = {"height": 8, "width": 8}
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_regions(**arguments)

    def test_fine_grid_covers_model_tokens_once(self):
        regions = build_regions(60, 104, rows=12, cols=16)
        self.assertEqual(len(regions), 192)
        coverage = torch.zeros(30, 52, dtype=torch.int32)
        token_counts = []
        for region in regions:
            y0, y1, x0, x1 = (region[key] // 2 for key in ("y0", "y1", "x0", "x1"))
            coverage[y0:y1, x0:x1] += 1
            token_counts.append((y1 - y0) * (x1 - x0))
        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
        self.assertEqual(set(token_counts), {6, 8, 9, 12})


class PatchNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.clean = torch.randn(2, 3, 2, 8, 8)
        self.noise = torch.randn_like(self.clean)
        self.region = build_regions(8, 8, rows=2, cols=2)[1]

    def test_unchanged_sigma_exactly_matches_scheduler_including_bf16(self):
        scheduler = FlowMatchScheduler(num_inference_steps=17)
        step = 5
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            clean, noise = self.clean.to(dtype), self.noise.to(dtype)
            noisy = scheduler.add_noise(
                clean.flatten(0, 1), noise.flatten(0, 1),
                scheduler.timesteps[step].repeat(clean.shape[0] * clean.shape[1]),
            ).unflatten(0, clean.shape[:2])
            actual = replace_patch_noise(noisy, clean, noise, scheduler.sigmas[step], self.region, 1)
            with self.subTest(dtype=dtype):
                self.assertTrue(torch.equal(actual, noisy))
                self.assertNotEqual(actual.data_ptr(), noisy.data_ptr())

    def test_only_one_frame_and_box_change_and_inputs_stay_unchanged(self):
        noisy = torch.full_like(self.clean, 17)
        original_clean, original_noise = self.clean.clone(), self.noise.clone()
        actual = replace_patch_noise(noisy, self.clean, self.noise, 0.25, self.region, 1)
        mask = torch.zeros_like(noisy, dtype=torch.bool)
        mask[:, 1, :, 0:4, 4:8] = True
        expected = 0.75 * self.clean + 0.25 * self.noise
        self.assertTrue(torch.equal(actual[mask], expected[mask]))
        self.assertTrue(torch.equal(actual[~mask], noisy[~mask]))
        self.assertTrue(torch.equal(noisy, torch.full_like(noisy, 17)))
        self.assertTrue(torch.equal(self.clean, original_clean))
        self.assertTrue(torch.equal(self.noise, original_noise))

    def test_sigma_endpoints(self):
        for sigma, expected in ((0, self.clean), (1, self.noise)):
            actual = replace_patch_noise(self.clean, self.clean, self.noise, sigma, self.region, 0)
            self.assertTrue(torch.equal(actual[:, 0, :, :4, 4:], expected[:, 0, :, :4, 4:]))

    def test_invalid_interventions(self):
        cases = [
            {"source_sigma": float("nan")}, {"source_sigma": -0.1}, {"source_sigma": 1.1},
            {"source_sigma": torch.tensor([0.1, 0.2])}, {"source_frame": 3},
            {"source_frame": -1}, {"source_frame": True},
            {"region": {"y0": 0, "y1": 0, "x0": 0, "x1": 1}},
            {"region": {"y0": 0, "y1": 9, "x0": 0, "x1": 1}},
            {"region": {}}, {"clean": self.clean[:, :2]},
            {"noise": self.noise.double()}, {"noise": torch.full_like(self.noise, float("inf"))},
        ]
        for replacement in cases:
            kwargs = dict(noisy=self.clean, clean=self.clean, noise=self.noise,
                          source_sigma=0.5, region=self.region, source_frame=1)
            kwargs.update(replacement)
            with self.subTest(replacement=list(replacement)), self.assertRaises(ValueError):
                replace_patch_noise(**kwargs)


class InfluenceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = torch.zeros(1, 3, 1, 4, 4)
        self.regions = build_regions(4, 4, rows=2, cols=2)
        self.region = self.regions[0]

    def measure(self, perturbed, input_delta=None, **kwargs):
        if input_delta is None:
            input_delta = torch.zeros_like(self.baseline)
        arguments = dict(baseline=self.baseline, perturbed=perturbed,
                         input_delta=input_delta, region=self.region,
                         source_frame=1, target_regions=self.regions, halo=1)
        arguments.update(kwargs)
        return measure_influence(**arguments)

    def test_identical_outputs_and_zero_input(self):
        metrics = self.measure(self.baseline.clone())
        for name in ("total_mse", "source_mse", "outside_source_mse", "outside_halo_mse",
                     "future_mse", "input_energy", "output_energy", "outside_source_energy",
                     "normalized_outside_energy"):
            self.assertEqual(metrics[name], 0.0)
        self.assertEqual(metrics["per_frame_mse"], [0.0] * 3)
        self.assertEqual(metrics["target_mse"], [[0.0] * 4 for _ in range(3)])
        json.dumps(metrics, allow_nan=False)

    def test_energy_area_normalization_and_frame_exclusions(self):
        perturbed = self.baseline.clone()
        perturbed[:, 0] = 1  # Prior frames remain in the outside metrics.
        perturbed[:, 1, :, :2, :2] = 2  # Source energy = 4 pixels * 4.
        perturbed[:, 1, :, 2, 2] = 3  # Outside source, inside its halo.
        perturbed[:, 1, :, 3, 3] = 4  # Outside both source and halo.
        perturbed[:, 2] = 5  # Future includes source's spatial coordinates.
        input_delta = torch.zeros(1, 2, 1, 4, 4)  # Output may include extra frames.
        input_delta[:, 1, :, :2, :2] = 2
        metrics = self.measure(perturbed, input_delta)
        self.assertEqual(metrics["input_energy"], 16.0)
        self.assertEqual(metrics["output_energy"], 457.0)
        self.assertEqual(metrics["outside_source_energy"], 441.0)
        self.assertAlmostEqual(metrics["total_mse"], 457 / 48, places=6)
        self.assertEqual(metrics["source_mse"], 4.0)
        self.assertAlmostEqual(metrics["outside_source_mse"], 441 / 44, places=6)
        self.assertAlmostEqual(metrics["outside_halo_mse"], 432 / 39, places=6)
        self.assertEqual(metrics["normalized_outside_energy"], 441 / 16)
        self.assertEqual(metrics["future_mse"], 25.0)
        self.assertEqual(metrics["per_frame_mse"], [1.0, 41 / 16, 25.0])
        self.assertEqual(metrics["target_mse"], [[1.0] * 4, [4.0, 0.0, 0.0, 25 / 4], [25.0] * 4])

    def test_mask_broadcast_counts_batch_and_channel_elements(self):
        baseline = self.baseline.expand(2, -1, 3, -1, -1).clone()
        perturbed = baseline.clone()
        perturbed[:, 1, :, :2, :2] = 2
        input_delta = perturbed.clone()
        metrics = self.measure(perturbed, input_delta, baseline=baseline)
        self.assertEqual(metrics["output_energy"], 96.0)
        self.assertEqual(metrics["input_energy"], 96.0)
        self.assertEqual(metrics["source_mse"], 4.0)
        self.assertEqual(metrics["outside_source_energy"], 0.0)
        self.assertEqual(metrics["normalized_outside_energy"], 0.0)

    def test_empty_regions_return_null_and_serialize_strict_json(self):
        baseline = torch.zeros(1, 1, 1, 4, 4)
        region = build_regions(4, 4, rows=1, cols=1)[0]
        metrics = measure_influence(baseline, baseline + 1, baseline + 1,
                                    region, 0, [region], halo=20)
        for name in ("outside_source_mse", "outside_halo_mse", "future_mse",
                     "outside_source_energy", "normalized_outside_energy",
                     "other_frames_mse", "other_frames_energy", "normalized_other_frames_energy"):
            self.assertIsNone(metrics[name])
        json.dumps(metrics, allow_nan=False)
        zero_input = measure_influence(baseline, baseline, baseline, region, 0, [region])
        self.assertEqual(zero_input["normalized_outside_energy"], 0.0)
        self.assertEqual(zero_input["normalized_other_frames_energy"], 0.0)

    def test_fixed_other_frames_domain_excludes_entire_source_frame(self):
        perturbed = self.baseline.clone()
        perturbed[:, 0] = 2
        perturbed[:, 1] = 1000
        perturbed[:, 2] = 3
        input_delta = torch.zeros_like(self.baseline)
        input_delta[:, 1, :, :2, :2] = 2
        # Moving the named source box changes the outside-box domain, but must
        # not change the fixed other-frame domain or its normalization.
        for region in self.regions:
            metrics = self.measure(perturbed, input_delta, region=region)
            self.assertEqual(metrics["other_frames_energy"], 16 * (4 + 9))
            self.assertEqual(metrics["other_frames_mse"], (4 + 9) / 2)
            self.assertEqual(metrics["normalized_other_frames_energy"], 13.0)

    def test_batched_rectangle_means_match_direct_slices(self):
        generator = torch.Generator().manual_seed(203)
        baseline = torch.randn(2, 4, 3, 7, 11, generator=generator)
        perturbed = baseline + torch.randn(baseline.shape, generator=generator)
        squared = (perturbed.float() - baseline.float()).square()
        # Unequal, odd-sized and overlapping boxes, deliberately not spatially
        # sorted: the metric must preserve caller order and average B/C too.
        boxes = [dict(y0=2, y1=7, x0=6, x1=11), dict(y0=0, y1=1, x0=0, x1=1),
                 dict(y0=1, y1=6, x0=2, x1=9), dict(y0=0, y1=7, x0=0, x1=11)]
        metrics = measure_influence(baseline, perturbed, perturbed - baseline,
                                    boxes[0], 2, boxes)
        expected = [[float(squared[:, frame, :, box["y0"]:box["y1"], box["x0"]:box["x1"]].mean())
                     for box in boxes] for frame in range(4)]
        torch.testing.assert_close(torch.tensor(metrics["target_mse"]), torch.tensor(expected),
                                   rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(torch.tensor(metrics["per_frame_mse"]),
                                   torch.tensor([float(squared[:, frame].mean()) for frame in range(4)]),
                                   rtol=2e-6, atol=1e-7)
        empty = measure_influence(baseline, perturbed, perturbed - baseline, boxes[0], 2, [])
        self.assertEqual(empty["target_mse"], [[], [], [], []])

    def test_halo_does_not_exclude_other_frames(self):
        perturbed = self.baseline.clone()
        perturbed[:, 2, :, :2, :2] = 1
        metrics = self.measure(perturbed, halo=99)
        self.assertEqual(metrics["outside_halo_mse"], 4 / 32)
        self.assertEqual(metrics["future_mse"], 4 / 16)
        self.assertIsNone(self.measure(perturbed, source_frame=2)["future_mse"])

    def test_values_use_float32_before_squaring(self):
        baseline = torch.zeros(1, 1, 1, 4, 4, dtype=torch.float16)
        perturbed = baseline + 300  # Squaring in float16 would overflow.
        metrics = measure_influence(baseline, perturbed, perturbed, self.region, 0, self.regions)
        self.assertEqual(metrics["total_mse"], 90000.0)
        self.assertEqual(metrics["output_energy"], 1440000.0)
        self.assertEqual(metrics["input_energy"], 1440000.0)

    def test_invalid_metrics_inputs(self):
        cases = [
            {"baseline": self.baseline[0]}, {"perturbed": self.baseline[:, :2]},
            {"input_delta": torch.zeros(1, 3, 2, 4, 4)},
            {"input_delta": self.baseline[:, :1]}, {"source_frame": 3},
            {"source_frame": -1}, {"halo": -1}, {"halo": 0.5},
            {"target_regions": [dict(y0=0, y1=2, x0=0, x1=5)]},
            {"baseline": torch.zeros_like(self.baseline, dtype=torch.int32)},
            {"perturbed": torch.full_like(self.baseline, float("nan"))},
            {"input_delta": torch.full_like(self.baseline, float("inf"))},
            {"perturbed": torch.full_like(self.baseline, 1e30)},
        ]
        for replacement in cases:
            arguments = dict(baseline=self.baseline, perturbed=self.baseline,
                             input_delta=self.baseline, region=self.region,
                             source_frame=1, target_regions=self.regions, halo=1)
            arguments.update(replacement)
            with self.subTest(replacement=list(replacement)), self.assertRaises(ValueError):
                measure_influence(**arguments)


if __name__ == "__main__":
    unittest.main()
