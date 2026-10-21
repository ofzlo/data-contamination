import numpy as np
import pytest

from contam.eval import stats


def test_identical_inputs_zero_width_ci():
    x = np.random.default_rng(0).normal(size=200)
    r = stats.bootstrap_paired_diff(x, x.copy(), n_boot=300)
    assert r.estimate == 0 and r.lo == 0 and r.hi == 0
    y = np.random.default_rng(1).normal(size=150)
    d = stats.bootstrap_delta(x, x.copy(), y, y.copy(), n_boot=300)
    assert d.estimate == 0 and d.lo == 0 and d.hi == 0


def test_pairing_constant_shift_zero_width():
    # A constant per-image shift is recovered exactly only if the same indices hit both arrays.
    x = np.random.default_rng(0).normal(size=300) * 10
    r = stats.bootstrap_paired_diff(x + 2.0, x, n_boot=300)
    assert r.lo == pytest.approx(2.0) and r.hi == pytest.approx(2.0)


def test_groups_resampled_separately():
    # seen gain +3, unseen gain +1 for every image -> Δ exactly 2 in every replicate
    s, u = np.arange(50.0), np.arange(80.0)
    d = stats.bootstrap_delta(s + 3, s, u + 1, u, n_boot=200)
    assert d.lo == pytest.approx(2.0) and d.hi == pytest.approx(2.0)


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 1.0, size=400)
    r = stats.bootstrap_mean(x, n_boot=1000)
    assert r.lo < 1.0 < r.hi
    assert r.lo < r.estimate < r.hi


def test_unaligned_arrays_rejected():
    with pytest.raises(ValueError):
        stats.bootstrap_paired_diff(np.ones(3), np.ones(4))


def test_pool_seeds():
    p = stats.pool_seeds([{1: 1.0, 2: 3.0}, {1: 3.0, 2: 5.0}])
    assert p == {1: 2.0, 2: 4.0}
    with pytest.raises(ValueError):
        stats.pool_seeds([{1: 1.0}, {2: 1.0}])


def test_margins():
    vals = [10.0, 12.0, 11.0, 9.0, 13.0]
    assert stats.style_margin(vals) == pytest.approx(2 * np.std(vals, ddof=1))
    seen, unseen = [5, 6, 7, 6, 5], [4, 4, 5, 5, 4]
    gaps = np.subtract(seen, unseen)
    assert stats.association_margin(seen, unseen) == pytest.approx(2 * np.std(gaps, ddof=1))
    assert stats.memory_margin(20.0, floor=0.5) == pytest.approx(2.0)
    assert stats.memory_margin(2.0, floor=0.5) == pytest.approx(0.5)
    assert stats.memory_margin(-1.0, floor=0.5) == pytest.approx(0.5)


def test_isotonic():
    np.testing.assert_allclose(stats.isotonic_increasing([1, 3, 2, 4]), [1, 2.5, 2.5, 4])
    np.testing.assert_allclose(stats.isotonic_increasing([3, 2, 1]), [2, 2, 2])
    np.testing.assert_allclose(stats.isotonic_increasing([0, 1, 2]), [0, 1, 2])


def test_first_crossing():
    doses = [1, 4, 16, 64]
    assert stats.first_crossing(doses, [0, 0.1, 0.5, 1.0], 0.3) == 16
    assert stats.first_crossing(doses, [0, 0.1, 0.2, 0.25], 0.3) == float("inf")
    # interpolation on log2 d: halfway between log2 4 = 2 and log2 16 = 4 -> 2^3 = 8
    assert stats.first_crossing(doses, [0, 0.1, 0.5, 1.0], 0.3, interpolate=True) == pytest.approx(8.0)
    assert stats.first_crossing(doses, [1, 1, 1, 1], 0.3, interpolate=True) == 1
    with pytest.raises(ValueError):
        stats.first_crossing([0, 1], [0, 1], 0.5)


def _dose_data(effects_style, effects_mem, n=400, noise=0.5, seed=0):
    """Per-image data for doses (1, 4, 16, 64): style on unseen, memory on seen, base = 0."""
    rng = np.random.default_rng(seed)
    data = {"unseen": {}, "seen": {}}
    for k, (es, em) in enumerate(zip(effects_style, effects_mem)):
        data["unseen"][f"s{k}"] = es + rng.normal(0, noise, n)
        data["seen"][f"m{k}"] = em + rng.normal(0, noise, n)
    data["unseen"]["base"] = np.zeros(n)
    data["seen"]["base"] = np.zeros(n)
    return data


def _style_fn(d):
    return np.array([np.mean(d["unseen"][f"s{k}"] - d["unseen"]["base"]) for k in range(4)])


def _mem_fn(d):
    return np.array([np.mean(d["seen"][f"m{k}"] - d["seen"]["base"]) for k in range(4)])


DOSES = [1, 4, 16, 64]


def test_tau_known_crossing():
    data = _dose_data([0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 2.0])
    r = stats.estimate_tau(_style_fn, data, DOSES, delta=0.5, n_boot=300)
    assert r.tau == 4
    r = stats.estimate_tau(_mem_fn, data, DOSES, delta=0.5, n_boot=300)
    assert r.tau == 64


def test_tau_monotone_fit_handles_dip():
    # Non-monotone raw curve (dip at d=16) is pooled by the isotonic fit, crossing stays at d=4.
    data = _dose_data([0.0, 1.5, 0.9, 1.5], [0, 0, 0, 0], noise=0.2)
    r = stats.estimate_tau(_style_fn, data, DOSES, delta=0.5, n_boot=300)
    assert r.tau == 4
    assert np.all(np.diff(r.fitted) >= -1e-12)
    assert np.all(np.diff(r.lower) >= -1e-12)


def test_tau_never_crosses_is_censored():
    data = _dose_data([0, 0.1, 0.1, 0.2], [0, 0, 0, 0])
    r = stats.estimate_tau(_style_fn, data, DOSES, delta=0.5, n_boot=200)
    assert r.tau == float("inf")
    assert r.to_dict()["tau"] == "inf"


def test_tau_difference():
    data = _dose_data([0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 2.0])
    res = stats.estimate_taus(_style_fn, _mem_fn, data, DOSES, 0.5, 0.5, n_boot=300)
    assert res["style"]["tau"] == 4 and res["memory"]["tau"] == 64
    diff = res["log2_tau_diff"]
    assert diff["ci_above_zero"] and diff["ci_lo"] > 0
    # memory never crosses -> +inf replicate differences, still "above zero"
    data = _dose_data([0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    res = stats.estimate_taus(_style_fn, _mem_fn, data, DOSES, 0.5, 0.5, n_boot=200)
    assert res["memory"]["tau"] == "inf"
    assert res["log2_tau_diff"]["frac_memory_only_censored"] == 1.0
    assert res["log2_tau_diff"]["ci_lo"] == "inf"
    # same crossing for both -> CI includes 0
    data = _dose_data([0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0])
    res = stats.estimate_taus(_style_fn, _mem_fn, data, DOSES, 0.5, 0.5, n_boot=200)
    assert not res["log2_tau_diff"]["ci_above_zero"]


def test_equivalence():
    x = np.random.default_rng(0).normal(0.0, 0.1, 500)
    r = stats.bootstrap_mean(x, n_boot=500)
    assert stats.equivalence_below(r, 0.1)
    assert not stats.equivalence_below(r, 0.0)
