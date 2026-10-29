import numpy as np
import torch

from scripts.patch.ar_patch_attention import resample_after, token_groups
from scripts.patch.patch_noise_utils import build_regions
from scripts.patch.summarize_ar_patch_attention import attention_scores


def test_token_groups_follow_frame_row_col_order():
    regions = build_regions(60, 104, 12, 16)
    groups = token_groups(regions, 3, 60, 104)
    assert groups.shape == (3 * 30 * 52,)
    assert groups[0] == 0 and groups[51] == 15 and groups[52 * 29] == 176
    assert groups[30 * 52] == 192 and groups[-1] == 3 * 192 - 1


def test_attention_scores_decomposition():
    rng = np.random.default_rng(0)
    regions, frame = 4, 1
    flow = rng.random((2, 3 * regions, 3 * regions))
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    s = attention_scores(flow, counts, regions, frame)
    k = 2
    column = flow[0, :, frame * regions + k]
    assert np.isclose(s["received_all"][0, k] * counts[k], column.sum())
    assert np.isclose(s["received_other"][0, k] * counts[k], column.sum() - column[frame * regions + k])
    assert np.isclose(s["cross_frame"][0, k] * counts[k],
                      column[:regions].sum() + column[2 * regions:].sum())


def test_resample_after_only_redraws_source_chunk_from_step():
    bank = {(c, s): torch.full((1, 3, 2, 2, 2), float(10 * c + s), dtype=torch.bfloat16)
            for c in (0, 1) for s in (1, 2, 3)}
    same, keys = resample_after(bank, 2, 0, 0)
    assert same is bank and keys == []
    new, keys = resample_after(bank, 2, 0, 3)
    assert keys == ["0:2", "0:3"]
    for key, value in bank.items():
        changed = not torch.equal(new[key], value)
        assert changed == (key[0] == 0 and key[1] >= 2)
        assert new[key].dtype == value.dtype and new[key].shape == value.shape
    again, _ = resample_after(bank, 2, 0, 3)
    assert all(torch.equal(again[k], new[k]) for k in bank)
    other, _ = resample_after(bank, 2, 0, 4)
    assert not torch.equal(other[0, 2], new[0, 2])
