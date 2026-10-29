import pytest
import torch

from pipeline.self_forcing_training import SelfForcingTrainingPipeline

STEPS = [1000, 750, 500, 250]


class FakeGenerator(torch.nn.Module):
    """x0 = input + t, so every x0 encodes which denoising steps produced it."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, noisy_image_or_video, conditional_dict, timestep, kv_cache, crossattn_cache, current_start):
        self.calls.append((int(timestep.flatten()[0]), noisy_image_or_video.detach().clone()))
        return None, noisy_image_or_video * self.w + timestep[..., None, None, None].float()


class FakeScheduler:
    def add_noise(self, x, noise, t):
        return x


@pytest.mark.parametrize("cache_final_x0, cache_value", [(False, 1750.0), (True, 2500.0)])
def test_cache_written_from_expected_step(monkeypatch, cache_final_x0, cache_value):
    gen = FakeGenerator()
    pipe = SelfForcingTrainingPipeline(STEPS, FakeScheduler(), gen, num_frame_per_block=3,
                                       same_step_across_blocks=False, cache_final_x0=cache_final_x0)
    pipe.crossattn_cache = None
    monkeypatch.setattr(pipe, "_initialize_kv_cache", lambda **k: None)
    monkeypatch.setattr(pipe, "_initialize_crossattn_cache", lambda **k: None)
    monkeypatch.setattr(pipe, "generate_and_sync_list", lambda *a, **k: [1, 1])  # exit at the 2nd step

    output, _, _ = pipe.inference_with_trajectory(torch.zeros(1, 6, 1, 1, 1))

    # 0 -> 1000 -> 1750 (exit) [-> 2250 -> 2500 without grad]
    assert torch.all(output == 1750.0) and output.requires_grad
    cache_inputs = [x for t, x in gen.calls if t == 0]
    assert len(cache_inputs) == 2 and all(torch.all(x == cache_value) for x in cache_inputs)
    assert len(gen.calls) == 2 * ((4 if cache_final_x0 else 2) + 1)
