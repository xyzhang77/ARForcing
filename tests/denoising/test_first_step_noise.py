"""CPU checks of first-step artifacts and native Wan flow forwarding; no weights."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.denoising import compare_first_step_noise as experiment
from scripts.patch.patch_noise_importance import tensor_hash
from scripts.patch.wan_patch_probe import shifted_sigmas


class FakeNative:
    """Affine-in-sigma flow with an analytically integrable Euler trajectory."""

    device = torch.device("cpu")
    guidance = 2.0

    def __init__(self):
        self.calls = []

    def predict_parts(self, state, sigma):
        self.calls.append((state.clone(), sigma))
        horizon = state.shape[1]
        positive = torch.full_like(state, 1 + horizon / 8 + sigma / 4)
        negative = torch.full_like(state, -.5 + horizon / 16 - sigma / 2)
        return positive, negative

    def predict(self, state, sigma):
        positive, negative = self.predict_parts(state, sigma)
        return negative + self.guidance * (positive - negative)


class FirstStepNoiseTests(unittest.TestCase):
    def test_wan_snapshots_match_analytic_flow_and_paired_noise(self):
        shape, seed = [1, 6, 2, 2, 4], 19
        initial = torch.randn(shape, generator=torch.Generator().manual_seed(seed),
                              dtype=torch.float32).to(torch.bfloat16).float()
        # The shifted grid also catches confusing step fractions with sigma.
        for steps, shift, check_identity in ((4, 1, True), (8, 5, False)):
            with self.subTest(steps=steps, shift=shift), patch.object(experiment, "SHAPE", shape):
                pipeline = FakeNative()
                sigmas = shifted_sigmas(steps, shift)
                rng_before = torch.get_rng_state().clone()
                latents, metadata = experiment.run_wan_case(
                    pipeline, seed, sigmas, check_identity)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                self.assertEqual(set(latents), experiment.BASE_KEYS | experiment.WAN_KEYS)
                self.assertEqual(metadata["initial_noise_sha256"], tensor_hash(initial.bfloat16()))
                self.assertEqual(metadata["first_chunk_noise_sha256"],
                                 tensor_hash(initial[:, :3].bfloat16()))

                # CFG=2 gives v(sigma)=A+B*sigma, A=3.625 and B=1.
                # Sum delta_sigma*sigma analytically, including the Euler
                # discretization term; this is independent of sampler recurrence.
                sigma64 = sigmas.double()
                delta_squared = (sigma64[1:] - sigma64[:-1]).square()
                prefix = torch.cat((torch.zeros(1, dtype=torch.float64),
                                    delta_squared.cumsum(0)))
                stage_indices = {"first": 0, "second": 1,
                                 "quarter": steps // 4, "half": steps // 2,
                                 "threequarter": 3 * steps // 4}
                for key, index in stage_indices.items():
                    offset = 3.625 + .5 * (float(sigma64[index]) ** 2 + 1 + float(prefix[index]))
                    torch.testing.assert_close(latents[key], initial[:, :3] - offset)
                    self.assertEqual(metadata["stages"][key],
                                     {"step_index": index, "sigma": float(sigmas[index])})
                final_offset = 3.625 + .5 * (1 + float(prefix[-1]))
                torch.testing.assert_close(latents["final"], initial - final_offset)
                torch.testing.assert_close(latents["next_state"],
                                           initial[:, :3] + (float(sigmas[1]) - 1) * 4.625)
                torch.testing.assert_close(latents["first_cfg1"], initial[:, :3] - 2)
                # The short-horizon model sees precisely the shared first three
                # noise frames; its horizon-dependent flow differs deliberately.
                torch.testing.assert_close(latents["first_horizon3"], initial[:, :3] - 4.0625)
                torch.testing.assert_close(latents["first_horizon3_cfg1"], initial[:, :3] - 1.625)
                self.assertTrue(torch.equal(pipeline.calls[-1][0], initial[:, :3]))
                self.assertEqual(pipeline.calls[-1][1], 1.0)
                self.assertEqual(metadata["sigmas"], [float(sigma) for sigma in sigmas])
                self.assertEqual(metadata["timesteps"], [float(sigma) * 1000 for sigma in sigmas[:-1]])
                if check_identity:
                    self.assertEqual(metadata["identity"]["first_call_repeat_max_abs"], 0)
                else:
                    self.assertIsNone(metadata["identity"])

    def test_native_parts_use_bf16_model_layout_and_actual_horizon(self):
        pipeline = object.__new__(experiment.CachedNativeWan)
        pipeline.conditional = torch.tensor([2.0])
        pipeline.unconditional = torch.tensor([-3.0])
        calls = []

        def model(model_input, *, t, context, seq_len):
            calls.append((model_input.clone(), t.clone(), context, seq_len))
            return model_input + context.item()

        pipeline.generator = SimpleNamespace(model=model, seq_len=999)
        for frames in (3, 6):
            with self.subTest(frames=frames):
                calls.clear()
                state = torch.linspace(-1, 1, frames * 2 * 4 * 6).reshape(1, frames, 2, 4, 6)
                positive, negative = pipeline.predict_parts(state, .625)
                self.assertEqual(len(calls), 2)
                for call, context in zip(calls, (pipeline.conditional, pipeline.unconditional)):
                    model_input, timestep, forwarded_context, seq_len = call
                    self.assertEqual(model_input.shape, (1, 2, frames, 4, 6))
                    self.assertEqual(model_input.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(model_input, state.bfloat16().permute(0, 2, 1, 3, 4)))
                    self.assertEqual(timestep.dtype, torch.float32)
                    torch.testing.assert_close(timestep, torch.tensor([625.0]))
                    self.assertIs(forwarded_context, context)
                    self.assertEqual(seq_len, frames * 2 * 3)
                torch.testing.assert_close(positive, (state.bfloat16() + 2).float())
                torch.testing.assert_close(negative, (state.bfloat16() - 3).float())
                self.assertEqual(positive.dtype, torch.float32)
                self.assertEqual(negative.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
