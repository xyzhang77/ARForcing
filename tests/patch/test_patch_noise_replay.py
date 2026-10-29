"""CPU checks of paired rollout state isolation without loading model weights."""

from types import SimpleNamespace
import unittest

import torch

from scripts.patch.patch_noise_importance import PairedRollout, make_noise_bank
from scripts.patch.patch_noise_utils import replace_patch_noise


class FakeScheduler:
    def __init__(self, steps):
        self.timesteps = torch.as_tensor(steps, dtype=torch.float32).clone()
        self.sigmas = self.timesteps / 1000
        self.calls = []

    def add_noise(self, clean, noise, timestep):
        self.calls.append({"clean": clean.clone(), "noise": noise.clone(), "timestep": timestep.clone()})
        indices = (self.timesteps[None] - timestep[:, None]).abs().argmin(dim=1)
        sigma = self.sigmas[indices].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * clean + sigma * noise).type_as(noise)


class FakeGenerator:
    """Tiny deterministic attention stand-in with two persistent KV layers.

    Writes use the production full-attention index formula. Predictions depend
    on both the immutable prefix and all current tokens. Caching a denoising
    input instead of the final clean context therefore affects the next chunk.
    Calls record incoming indices as well: when local/global offsets coincide,
    the production formula can overwrite correctly even without an explicit
    reset, so identity alone cannot prove the branch restored its boundary.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls = []

    @staticmethod
    def encode(noisy, timestep, layer):
        tokens = noisy.permute(0, 1, 3, 4, 2).flatten(1, 3)
        token_time = timestep.repeat_interleave(noisy.shape[-2] * noisy.shape[-1], dim=1)
        return tokens * (layer + 1) + token_time[:, :, None] * 0.001

    def __call__(self, noisy_image_or_video, conditional_dict, timestep,
                 kv_cache, crossattn_cache, current_start):
        noisy = noisy_image_or_video
        length = noisy.shape[1] * self.pipeline.frame_seq_length
        current_end = current_start + length
        incoming_indices, prefix_snapshots, prefix_terms = [], [], []
        for layer, cache in enumerate(kv_cache):
            old_global = int(cache["global_end_index"].item())
            old_local = int(cache["local_end_index"].item())
            incoming_indices.append((old_global, old_local))
            local_end = old_local + current_end - old_global
            local_start = local_end - length
            if not 0 <= local_start <= local_end <= cache["v"].shape[1]:
                raise AssertionError("invalid replay cache boundaries")
            encoded = self.encode(noisy, timestep, layer)
            cache["k"][:, local_start:local_end] = encoded + 0.125
            cache["v"][:, local_start:local_end] = encoded
            prefix = cache["v"][:, :local_start].clone()
            prefix_snapshots.append(prefix)
            prefix_terms.append(
                prefix.mean(dim=(1, 2)) if local_start else noisy.new_zeros(noisy.shape[0])
            )
            cache["global_end_index"].fill_(current_end)
            cache["local_end_index"].fill_(local_end)
        for cache in crossattn_cache:
            cache["is_init"] = True
        prefix_term = torch.stack(prefix_terms).mean(dim=0).reshape(-1, 1, 1, 1, 1)
        current_term = noisy.mean(dim=(1, 2, 3, 4), keepdim=True)
        prediction = (0.55 * noisy + 0.1 * prefix_term + 0.025 * current_term
                      + timestep[:, :, None, None, None] * 0.0001
                      + conditional_dict["offset"])
        self.calls.append({
            "current_start": current_start,
            "indices": incoming_indices,
            "prefix": prefix_snapshots,
            "noisy": noisy.clone(),
            "timestep": timestep.clone(),
            "prediction": prediction.clone(),
            "context": bool(torch.all(timestep == self.pipeline.args.context_noise).item()),
        })
        return torch.zeros_like(prediction), prediction


class FakePipeline:
    def __init__(self, local_attn_size=-1):
        # Fractional timesteps exercise warped schedule values and context time.
        self.denoising_step_list = torch.tensor([1000.0, 731.625, 215.375])
        self.args = SimpleNamespace(context_noise=0.125)
        self.local_attn_size = local_attn_size
        self.frame_seq_length = 4
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.scheduler = FakeScheduler(self.denoising_step_list)
        self.generator = FakeGenerator(self)

    def _initialize_kv_cache(self, batch, dtype, device):
        self.kv_cache1 = [{
            "k": torch.full((batch, 64, 1), -19.0, dtype=dtype, device=device),
            "v": torch.full((batch, 64, 1), -23.0, dtype=dtype, device=device),
            "global_end_index": torch.tensor(13, dtype=torch.int64, device=device),
            "local_end_index": torch.tensor(11, dtype=torch.int64, device=device),
        } for _ in range(2)]

    def _initialize_crossattn_cache(self, batch, dtype, device):
        self.crossattn_cache = [{"is_init": True} for _ in range(2)]


class PairedRolloutTests(unittest.TestCase):
    def make_rollout(self, local_attn_size=-1):
        pipeline = FakePipeline(local_attn_size)
        chunks = [1, 2, 2]  # Also exercise nonuniform independent-first-frame layout.
        initial, transition = make_noise_bank(
            (2, sum(chunks), 1, 2, 2), chunks, len(pipeline.denoising_step_list),
            seed=43, dtype=torch.float32, device=torch.device("cpu"),
        )
        replay = PairedRollout(pipeline, {"offset": 0.03125}, initial, transition, chunks)
        return replay, pipeline

    @staticmethod
    def patch(state, sigma):
        return replace_patch_noise(
            state["noisy"], state["clean"], state["noise"], sigma,
            {"y0": 0, "y1": 1, "x0": 0, "x1": 1}, source_frame=0,
        )

    def assert_branch_boundary(self, replay, pipeline, first_call, source_chunk):
        call = pipeline.generator.calls[first_call]
        start = replay.starts[source_chunk] * pipeline.frame_seq_length
        self.assertEqual(call["current_start"], start)
        self.assertEqual(call["indices"], [(start, start), (start, start)])

    def test_identity_and_post_intervention_identity_for_first_and_later_chunks(self):
        for source_chunk in (0, 1):
            with self.subTest(source_chunk=source_chunk):
                replay, pipeline = self.make_rollout()
                self.assertTrue(all(not cache["is_init"] for cache in pipeline.crossattn_cache))
                state = replay.baseline(source_chunk, source_step=1)
                identity_input = self.patch(state, pipeline.scheduler.sigmas[1])
                self.assertTrue(torch.equal(identity_input, state["noisy"]))
                first_call = len(pipeline.generator.calls)
                one_step, output = replay.branch(identity_input, source_chunk, 1, complete=True)
                self.assert_branch_boundary(replay, pipeline, first_call, source_chunk)
                self.assertTrue(torch.equal(one_step, state["prediction"]))
                self.assertTrue(torch.equal(output, state["output"]))

                changed = self.patch(state, pipeline.scheduler.sigmas[1] * 0.5)
                replay.branch(changed, source_chunk, 1, complete=True)
                # A one-step-only probe must also be harmless to later branches.
                replay.branch(changed + 0.25, source_chunk, 1, complete=False)
                first_call = len(pipeline.generator.calls)
                again_one_step, again_output = replay.branch(identity_input, source_chunk, 1, complete=True)
                self.assert_branch_boundary(replay, pipeline, first_call, source_chunk)
                self.assertTrue(torch.equal(again_one_step, state["prediction"]))
                self.assertTrue(torch.equal(again_output, state["output"]))
                recomputed = replay.baseline(source_chunk, source_step=1)
                self.assertTrue(torch.equal(recomputed["output"], state["output"]))

    def test_fixed_noise_propagates_into_future_without_modifying_prefix(self):
        replay, pipeline = self.make_rollout()
        source_chunk = 1
        state = replay.baseline(source_chunk, source_step=1)
        initial_before = replay.initial.clone()
        noise_before = {key: value.clone() for key, value in replay.transition.items()}
        token_start = replay.starts[source_chunk] * pipeline.frame_seq_length
        prefix_before = [{name: cache[name][:, :token_start].clone() for name in ("k", "v")}
                         for cache in pipeline.kv_cache1]
        changed = self.patch(state, pipeline.scheduler.sigmas[1] * 0.5)
        rng_before = torch.get_rng_state().clone()
        scheduler_start = len(pipeline.scheduler.calls)
        _, changed_output = replay.branch(changed, source_chunk, 1, complete=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        future_start = replay.chunks[source_chunk]
        self.assertGreater(float((changed_output[:, future_start:] - state["output"][:, future_start:]).abs().max()), 0)
        for layer, cache in enumerate(pipeline.kv_cache1):
            for name in ("k", "v"):
                self.assertTrue(torch.equal(cache[name][:, :token_start], prefix_before[layer][name]))
        expected_keys = [(1, 2), (2, 1), (2, 2)]
        calls = pipeline.scheduler.calls[scheduler_start:]
        self.assertEqual(len(calls), len(expected_keys))
        for call, key in zip(calls, expected_keys):
            self.assertTrue(torch.equal(call["noise"], noise_before[key].flatten(0, 1)))
        self.assertTrue(torch.equal(replay.initial, initial_before))
        for key, original in noise_before.items():
            self.assertTrue(torch.equal(replay.transition[key], original))
        _, repeated = replay.branch(changed, source_chunk, 1, complete=True)
        self.assertTrue(torch.equal(repeated, changed_output))

    def assert_clean_updates(self, pipeline, calls, chunk_starts):
        context_calls = [call for call in calls if call["context"]]
        self.assertEqual([call["current_start"] for call in context_calls], chunk_starts)
        clean_prefix = [[] for _ in pipeline.kv_cache1]
        for index, call in enumerate(calls):
            if call["context"]:
                self.assertGreater(index, 0)
                previous = calls[index - 1]
                self.assertFalse(previous["context"])
                self.assertEqual(call["current_start"], previous["current_start"])
                self.assertTrue(torch.equal(call["noisy"], previous["prediction"]))
                self.assertTrue(torch.equal(call["timestep"], torch.full_like(call["timestep"], pipeline.args.context_noise)))
                for layer in range(len(clean_prefix)):
                    clean_prefix[layer].append(FakeGenerator.encode(call["noisy"], call["timestep"], layer))
            elif index and calls[index - 1]["context"]:
                # The next chunk must see the freshly encoded clean context,
                # including timestep conditioning, in every layer's prefix.
                for layer, expected_chunks in enumerate(clean_prefix):
                    expected = torch.cat(expected_chunks, dim=1)
                    self.assertTrue(torch.equal(call["prefix"][layer][:, -expected.shape[1]:], expected))
        last = context_calls[-1]
        token_start = last["current_start"]
        for layer, cache in enumerate(pipeline.kv_cache1):
            encoded = FakeGenerator.encode(last["noisy"], last["timestep"], layer)
            self.assertTrue(torch.equal(cache["v"][:, token_start:token_start + encoded.shape[1]], encoded))

    def test_final_prediction_is_written_as_clean_context_for_baseline_and_branch(self):
        replay, pipeline = self.make_rollout()
        state = replay.baseline(source_chunk=1, source_step=1)
        self.assert_clean_updates(pipeline, pipeline.generator.calls, [0, 4, 12])
        call_start = len(pipeline.generator.calls)
        changed = self.patch(state, pipeline.scheduler.sigmas[1] * 0.5)
        replay.branch(changed, source_chunk=1, source_step=1, complete=True)
        self.assert_clean_updates(pipeline, pipeline.generator.calls[call_start:], [4, 12])

    def test_fractional_global_and_context_timesteps_are_preserved(self):
        replay, pipeline = self.make_rollout()
        state = replay.baseline(source_chunk=1, source_step=1)
        for call in pipeline.generator.calls:
            value = float(call["timestep"][0, 0])
            self.assertIn(value, (1000.0, 731.625, 215.375, 0.125))
            self.assertTrue(call["timestep"].is_floating_point())
        call_start = len(pipeline.generator.calls)
        changed = self.patch(state, pipeline.scheduler.sigmas[1] * 0.5)
        replay.branch(changed, source_chunk=1, source_step=1, complete=True)
        first = pipeline.generator.calls[call_start]
        self.assertTrue(torch.equal(first["timestep"], torch.full((2, 2), 731.625)))
        for call in pipeline.scheduler.calls:
            self.assertIn(float(call["timestep"][0]), (731.625, 215.375))
            self.assertTrue(call["timestep"].is_floating_point())

    def test_local_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "full attention"):
            self.make_rollout(local_attn_size=8)


if __name__ == "__main__":
    unittest.main()
