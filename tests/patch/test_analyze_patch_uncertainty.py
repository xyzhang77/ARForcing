import numpy as np

from scripts.patch.analyze_patch_uncertainty import decompose, icc1


def test_decompose_is_variance_plus_bias():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=3.0, size=(4000, 5))
    var, bias, gap = decompose(samples, np.zeros(5))
    assert np.allclose(gap, var + bias)
    assert np.allclose(var, 9.0, rtol=0.1) and np.allclose(bias, 4.0, rtol=0.1)


def test_icc1_extremes():
    subjects = np.arange(50.0)
    assert icc1(np.stack([subjects, subjects, subjects])) > 0.999
    rng = np.random.default_rng(1)
    assert abs(icc1(rng.normal(size=(5, 2000)))) < 0.05
