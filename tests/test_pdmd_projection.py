import unittest

import torch

from model.dmd import project_update


class PDMDProjectionTest(unittest.TestCase):
    def test_removes_parallel_component_per_sample(self):
        torch.manual_seed(0)
        d = torch.randn(3, 4, 2, 5, 6, dtype=torch.bfloat16)
        r = torch.randn(3, 4, 2, 5, 6, dtype=torch.bfloat16)
        p = project_update(d, r)
        dot = (p * r.float()).flatten(1).sum(1) / r.float().flatten(1).norm(dim=1)
        self.assertTrue(torch.allclose(dot, torch.zeros(3), atol=1e-4))
        self.assertTrue((p.flatten(1).norm(dim=1) <= d.float().flatten(1).norm(dim=1) + 1e-5).all())

    def test_zero_residual_keeps_update(self):
        d = torch.randn(2, 3, 4)
        self.assertTrue(torch.equal(project_update(d, torch.zeros_like(d)), d))

    def test_parallel_update_is_removed(self):
        r = torch.randn(2, 3, 4)
        self.assertTrue(torch.allclose(project_update(2.5 * r, r), torch.zeros_like(r), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
