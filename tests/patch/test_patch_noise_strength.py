import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.patch.patch_noise_strength import frozen_candidates, pixel_metrics, save_pixels
from scripts.patch.patch_noise_utils import replace_patch_noise


class PatchNoiseStrengthTest(unittest.TestCase):
    def source_data(self):
        return {"regions": [{"id": i, "y0": 0, "y1": 2, "x0": 2*i, "x1": 2*i+2} for i in range(6)],
                "records": [{"source_id": i, "sigma_scale": 0.75,
                             "rollout": {} if i < 4 else None,
                             "selection": "top" if i < 2 else "random_control",
                             "one_step": {"normalized_other_frames_energy": 1000-i}}
                            for i in range(6)]}

    def test_candidates_are_frozen_saved_membership_not_ranking(self):
        data = self.source_data()
        expected = frozen_candidates(data)
        changed = copy.deepcopy(data)
        changed["records"].reverse()
        for row in changed["records"]:
            row["one_step"]["normalized_other_frames_energy"] = row["source_id"] * 10000
        self.assertEqual(frozen_candidates(changed), expected)
        self.assertEqual([c["source_id"] for c in expected], [0, 1, 2, 3])

    def test_missing_saved_control_is_rejected(self):
        data = self.source_data()
        data["records"][3]["rollout"] = None
        with self.assertRaises(ValueError):
            frozen_candidates(data)

    def test_zero_sigma_is_previous_prediction_in_patch_only(self):
        noisy = torch.full((1, 3, 2, 4, 4), 7., dtype=torch.bfloat16)
        clean = torch.full_like(noisy, -3.)
        noise = torch.full_like(noisy, 11.)
        region = {"y0": 0, "y1": 2, "x0": 2, "x1": 4}
        actual = replace_patch_noise(noisy, clean, noise, 0., region, 1)
        expected = noisy.clone()
        expected[:, 1, :, :2, 2:] = clean[:, 1, :, :2, 2:]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.all(noisy == 7))

    def test_pixel_metric_identity_and_known_difference(self):
        baseline = torch.zeros((21, 2, 2, 3))
        variant = baseline.clone()
        variant[12:] = 0.5
        identity = pixel_metrics(baseline, baseline)
        self.assertEqual(identity["all_video_mse"], 0)
        result = pixel_metrics(baseline, variant)
        self.assertAlmostEqual(result["all_video_mse"], 9 / 21 * 0.25)
        self.assertAlmostEqual(result["all_video_mae"], 9 / 21 * 0.5)
        self.assertEqual(result["later_frames_rmse"], 0.5)
        self.assertEqual(result["per_frame_mse"], [0.] * 12 + [.25] * 9)

    def test_sub_uint8_difference_measured_before_quantization(self):
        baseline = torch.zeros((1, 2, 2, 3))
        variant = baseline + 0.001
        self.assertGreater(pixel_metrics(baseline, variant)["all_video_mse"], 0)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_pixels(variant, Path(directory) / "variant", write_video=False)
            pixels = np.load(Path(directory) / artifacts["pixels_npz"])["pixels"]
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertTrue(np.all(pixels == 0))
        with self.assertRaises(ValueError):
            pixel_metrics(baseline.to(torch.uint8), variant.to(torch.uint8))


if __name__ == "__main__":
    unittest.main()
