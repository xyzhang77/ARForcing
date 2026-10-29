import numpy as np
import torch

from scripts.patch.ar_patch_local_resample import fresh_noise, pairwise_mean, select


def test_select_uses_one_token_class_and_extremes():
    u = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8]
    tokens = [9, 9, 9, 6, 9, 9, 9]
    groups = select({"patch_maps": {"U": u}, "num_tokens": tokens}, 2, 9)
    assert groups == {"low": [1, 5], "high": [6, 2]}


def test_fresh_noise_is_deterministic_and_distinct():
    like = torch.zeros(1, 3, 2, 4, 4, dtype=torch.bfloat16)
    a, b = fresh_noise(like, 0, 1, 5, 1), fresh_noise(like, 0, 1, 5, 1)
    assert a.dtype == like.dtype and torch.equal(a, b)
    assert not torch.equal(a, fresh_noise(like, 0, 1, 5, 2))
    assert not torch.equal(a, fresh_noise(like, 0, 1, 6, 1))


def test_pairwise_mean():
    assert pairwise_mean([0.0, 1.0, 3.0], lambda a, b: abs(a - b)) == np.mean([1, 3, 2])
