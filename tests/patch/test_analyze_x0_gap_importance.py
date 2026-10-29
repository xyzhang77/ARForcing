import numpy as np

from scripts.patch.analyze_x0_gap_importance import binned, partial_rho, spatial_null, top_jaccard


def test_spatial_null_is_permutations_without_identity():
    null = spatial_null(3, 4)
    assert null.shape == (4 * 12 - 1, 12)
    assert all(sorted(p) == list(range(12)) for p in null)
    assert not any((p == np.arange(12)).all() for p in null)


def test_partial_rho_removes_shared_covariate():
    rng = np.random.default_rng(0)
    c = rng.normal(size=500)
    x, y = c + 0.1 * rng.normal(size=500), c + 0.1 * rng.normal(size=500)
    assert partial_rho(x, y, []) > 0.9
    assert abs(partial_rho(x, y, [c])) < 0.15


def test_top_jaccard():
    a = np.arange(10.0)
    assert top_jaccard(a, a, 3) == 1.0
    assert top_jaccard(a, -a, 3) == 0.0


def test_binned_is_monotone_for_monotone_relation():
    gap = np.arange(100.0)
    mean, _ = binned(gap, gap * 2, 5)
    assert np.all(np.diff(mean) > 0) and mean[0] < 20 and mean[-1] > 80
    assert np.allclose(binned(gap, -gap, 5)[0], mean[::-1])
