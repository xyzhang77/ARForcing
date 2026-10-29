import numpy as np
import pytest

from scripts.patch.ar_x0_step_gap import patch_percentiles, region_means
from scripts.patch.patch_noise_utils import build_regions


def test_region_means_uses_region_box_and_frame():
    regions = build_regions(4, 8, 2, 2)
    gap = np.zeros((3, 4, 8))
    gap[1, 0:2, 4:8] = 2.0
    gap[0, 0:2, 4:8] = 4.0
    one, every = region_means(gap, regions, 1)
    assert one.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert np.allclose(every, [0.0, 2.0, 0.0, 0.0])


def test_patch_percentiles_ignore_outlier_magnitude():
    expected = [0, 100 / 3, 200 / 3, 100]
    assert np.allclose(patch_percentiles([1, 2, 3, 4]), expected)
    assert np.allclose(patch_percentiles([1, 2, 3, 1e12]), expected)


def test_patch_percentiles_keep_ties_equal():
    assert np.allclose(patch_percentiles([2, 2, 4, 1]), [50, 50, 100, 0])
    assert np.allclose(patch_percentiles([7, 7, 7]), [50, 50, 50])


def test_patch_percentiles_reject_invalid_values():
    for values in ([1], [1, np.nan], [1, np.inf], [[1, 2]]):
        with pytest.raises(ValueError):
            patch_percentiles(values)


def test_percentile_mode_does_not_capture_or_rewrite_statistics(tmp_path, monkeypatch):
    from scripts.patch import ar_x0_step_gap
    case = tmp_path / "prompt_000000_seed_000000"
    case.mkdir()
    plotted = []
    monkeypatch.setattr("sys.argv", ["ar_x0_step_gap", "--root", str(tmp_path), "--percentile"])
    monkeypatch.setattr(ar_x0_step_gap, "plot_percentiles", plotted.append)
    monkeypatch.setattr(ar_x0_step_gap, "capture", lambda *args: pytest.fail("Unexpected GPU capture"))
    monkeypatch.setattr(ar_x0_step_gap, "analyze", lambda *args: pytest.fail("Unexpected statistics rewrite"))
    ar_x0_step_gap.main()
    assert plotted == [case]
    assert not (tmp_path / "x0_step_gap_summary.json").exists()
