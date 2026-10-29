import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from scripts.patch.patch_timestep_utils import build_patch_timestep_maps, predict_with_patch_timestep


def load_real_unpatchify():
    """Exercise Wan's actual token layout without importing CUDA dependencies."""
    path = Path(__file__).resolve().parents[2] / "wan/modules/causal_model.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CausalWanModel")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "unpatchify")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["unpatchify"]


class TokenFlowModel:
    """Expose time conditioning through real Wan unpatchification."""

    patch_size = (1, 2, 2)
    unpatchify = load_real_unpatchify()

    def __init__(self, channels):
        self.out_dim = channels
        self.calls = []

    def __call__(self, x, **kwargs):
        self.calls.append({"x": x.clone(), **kwargs})
        batch, channels, frames, height, width = x.shape
        # CausalHead emits [B,L,1,prod(patch_size)*channels] for token times.
        # Nonzero flow at t=0 verifies the conversion really uses sigma=0.
        token_flow = (kwargs["t"] + 1)[:, :, None, None].expand(
            batch, -1, 1, math.prod(self.patch_size) * channels).to(x.dtype)
        grid = torch.tensor([[frames, height // 2, width // 2]] * batch)
        return torch.stack(self.unpatchify(token_flow, grid))


class PatchTimestepTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = SimpleNamespace(
            timesteps=torch.tensor([1000.0, 731.625, 215.375, 2.0]),
            sigmas=torch.tensor([1.0, 0.731625, 0.215375, 0.002]),
            num_train_timesteps=1000,
        )
        self.noisy = torch.arange(2 * 3 * 2 * 4 * 6).reshape(2, 3, 2, 4, 6).float() / 13
        self.times = torch.tensor([[731.625, 215.375, 1000.0], [1000.0, 731.625, 215.375]])
        self.region = {"y0": 0, "y1": 2, "x0": 2, "x1": 6}

    def maps(self, **kwargs):
        return build_patch_timestep_maps(
            self.noisy, self.times, self.scheduler, (1, 2, 2), **kwargs)

    def pipeline(self):
        model = TokenFlowModel(self.noisy.shape[2])
        return SimpleNamespace(
            generator=SimpleNamespace(model=model, scheduler=self.scheduler,
                                      seq_len=32760, uniform_timestep=False),
            kv_cache1=[{"marker": "kv"}], crossattn_cache=[{"marker": "cross"}],
        )

    def test_spatial_selection_frame_and_token_flattening(self):
        original_times = self.times.clone()
        token_times, sigmas = self.maps(region=self.region, source_frame=1, source_sigma=0.123456)
        expected = self.times[:, :, None].expand(2, 3, 6).clone()
        expected[:, 1, 1:3] = 123.456
        torch.testing.assert_close(token_times, expected.flatten(1), rtol=0, atol=0)
        expected_sigmas = self.scheduler.sigmas[torch.tensor([[1, 2, 0], [0, 1, 2]])].double()
        expected_sigmas = expected_sigmas[:, :, None, None, None].expand(2, 3, 1, 4, 6).clone()
        expected_sigmas[:, 1, :, :2, 2:6] = 0.123456
        torch.testing.assert_close(sigmas, expected_sigmas, rtol=0, atol=0)
        self.assertTrue(torch.equal(self.times, original_times))

    def test_uniform_token_control_preserves_fractional_times_and_nearest_sigma(self):
        self.times[0, 0] = 730.125  # Deliberately between scheduler entries.
        token_times, sigmas = self.maps()
        self.assertEqual(float(token_times[0, 0]), 730.125)
        self.assertEqual(float(token_times[0, 6]), 215.375)
        self.assertEqual(float(sigmas[0, 0, 0, 0, 0]), float(self.scheduler.sigmas[1]))
        self.assertNotEqual(float(sigmas[0, 0, 0, 0, 0]), 0.730125)

    def test_zero_sigma_is_exact_despite_nonzero_scheduler_endpoint(self):
        token_times, sigmas = self.maps(region=self.region, source_frame=1, source_sigma=0.0)
        self.assertTrue(torch.equal(token_times[:, 7:9], torch.zeros(2, 2)))
        self.assertEqual(sigmas[:, 1, :, :2, 2:6].count_nonzero().item(), 0)
        pipeline = self.pipeline()
        prediction = predict_with_patch_timestep(
            pipeline, self.noisy, {"prompt_embeds": torch.ones(2, 1, 1)}, self.times,
            self.region, 1, 0.0, 18)
        self.assertTrue(torch.equal(prediction[:, 1, :, :2, 2:6], self.noisy[:, 1, :, :2, 2:6]))

    def test_raw_call_cache_arguments_layout_and_local_conversion(self):
        pipeline = self.pipeline()
        context = torch.randn(2, 4, 7)
        prediction = predict_with_patch_timestep(
            pipeline, self.noisy, {"prompt_embeds": context}, self.times,
            self.region, 1, 0.125, 18)
        call = pipeline.generator.model.calls[-1]
        self.assertIs(call["context"], context)
        self.assertIs(call["kv_cache"], pipeline.kv_cache1)
        self.assertIs(call["crossattn_cache"], pipeline.crossattn_cache)
        self.assertEqual(call["current_start"], 18)
        self.assertIsNone(call["cache_start"])
        self.assertEqual(call["seq_len"], 32760)
        self.assertTrue(torch.equal(call["x"], self.noisy.permute(0, 2, 1, 3, 4)))
        # An independent pixel-space oracle catches swapped token rows/frames.
        pixel_times = (self.times + 1)[:, :, None, None, None].expand_as(self.noisy).clone()
        pixel_times[:, 1, :, :2, 2:6] = 126
        sigma = self.scheduler.sigmas[torch.tensor([[1, 2, 0], [0, 1, 2]])].double()
        sigma = sigma[:, :, None, None, None].expand_as(self.noisy).clone()
        sigma[:, 1, :, :2, 2:6] = 0.125
        expected = (self.noisy.double() - sigma * pixel_times.double()).float()
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)

    def test_uniform_token_prediction_matches_frame_conversion_and_dtype(self):
        self.noisy = self.noisy.to(torch.bfloat16)
        pipeline = self.pipeline()
        result = predict_with_patch_timestep(
            pipeline, self.noisy, {"prompt_embeds": torch.ones(2, 1, 1)}, self.times)
        flow = (self.times + 1)[:, :, None, None, None].expand_as(self.noisy).to(self.noisy.dtype)
        sigmas = self.scheduler.sigmas[torch.tensor([[1, 2, 0], [0, 1, 2]])].double()
        expected = (self.noisy.double() - sigmas[:, :, None, None, None] * flow.double()).bfloat16()
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, expected))

    def test_rejects_partial_tokens_invalid_frames_and_invalid_sigma(self):
        invalid = [
            {"region": {**self.region, "x0": 1}, "source_frame": 1, "source_sigma": 0.2},
            {"region": self.region, "source_frame": 3, "source_sigma": 0.2},
            {"region": self.region, "source_frame": 1, "source_sigma": float("nan")},
            {"region": self.region, "source_frame": 1, "source_sigma": -0.1},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.maps(**kwargs)
        with self.assertRaisesRegex(ValueError, "patch_size"):
            build_patch_timestep_maps(self.noisy, self.times, self.scheduler, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
