import unittest

import torch

from model.dmd import frame_diff_energy, motion_part, motion_ratio


class MotionSplitTest(unittest.TestCase):
    def test_static_video_has_no_motion(self):
        x = torch.randn(2, 1, 4, 5, 6).expand(2, 7, 4, 5, 6)
        self.assertTrue(torch.allclose(motion_part(x), torch.zeros_like(x), atol=1e-6))
        self.assertAlmostEqual(motion_ratio(x).item(), 0.0, places=6)
        self.assertAlmostEqual(frame_diff_energy(x).item(), 0.0, places=6)

    def test_split_is_orthogonal(self):
        x = torch.randn(2, 7, 4, 5, 6)
        m = motion_part(x)
        self.assertAlmostEqual(((x - m) * m).sum().item(), 0.0, places=3)
        appearance_share = (x - m).square().sum() / x.square().sum()
        self.assertAlmostEqual((motion_ratio(x) + appearance_share).item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
