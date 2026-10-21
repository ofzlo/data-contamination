"""Statistics for gate decisions: paired bootstrap, seed pooling, D5 margins, τ estimation.

Paired image-level bootstrap
----------------------------
Data are organized as ``{group: {name: per-image array}}`` where every array inside one group is
aligned by image (element k is the same image for every checkpoint). In each replicate we draw ONE
index vector per group (with replacement, size = group size) and apply it to every array in that
group. Hence comparisons between checkpoints are paired (same resampled images on both sides), and
seen and unseen images are resampled separately, as required for Δ (eq. 3) and Γ (eq. 4).
CIs are percentile intervals.

Seeds
-----
``pool_seeds`` averages each image's score across training seeds BEFORE bootstrapping (pooled
value); per-seed values are the same statistic computed on each seed separately.

Decision margins (D5)
---------------------
* δ_S (per primary S metric) = 2 × SD across the five d=0 seeds of the UNSEEN-group mean of that
  metric (sample SD, ddof=1). Example: unseen mean word count per seed.
* δ_A (per A metric, primary CIDEr held-out) = 2 × SD across the five d=0 seeds of the
  seen-minus-unseen gap of group means, S^seen_0 − S^unseen_0 (ddof=1).
* δ_V = max(0.10 × E, floor), where E is the pooled point estimate of the Paired d=64 V effect
  (mean ΔRMI over seen images) and ``floor`` is a named config value (``margins.delta_v_floor``).

Threshold τ (plan, "Shared metrics")
-------------------------------------
τ = the smallest dose d whose effect CI lower bound exceeds δ, from a monotone fit over log₂ d.
Procedure (``estimate_taus``): in every bootstrap replicate compute the effect at each dose d > 0
(all doses share the replicate's image indices), fit an increasing isotonic regression over
x = log₂ d, and keep the fitted curve. The pointwise lower bound at d is the (1−ci)/2 percentile of
the replicate fits (monotone in d because every fit is). τ̂ = smallest d with lower bound > δ; if
none, τ̂ = +inf ("never crosses", censored). For τ_memory − τ_style, each replicate's own crossing
τ*_b is the smallest d whose fitted effect exceeds δ (optionally linearly interpolated on log₂ d);
the replicate difference log₂ τ*_mem − log₂ τ*_style is +inf if only memory never crosses, −inf
if only style never crosses, and 0 if neither crosses (no evidence of separation; conservative
for the "CI above 0" rule). Censoring fractions are reported. Percentiles of the difference use
order statistics (``inverted_cdf``) so ±inf values are handled exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

Data = Mapping[str, Mapping[str, np.ndarray]]


# ----------------------------------------------------------------------------- bootstrap core


def percentile_ci(reps: np.ndarray, ci: float = 0.95, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    a = (1.0 - ci) / 2.0
    lo = np.quantile(reps, a, axis=axis, method="inverted_cdf")
    hi = np.quantile(reps, 1.0 - a, axis=axis, method="inverted_cdf")
    return lo, hi


@dataclass
class BootResult:
    estimate: float
    lo: float
    hi: float
    ci: float
    n_boot: int
    replicates: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {"estimate": float(self.estimate), "ci_lo": float(self.lo), "ci_hi": float(self.hi), "ci": self.ci, "n_boot": self.n_boot}


def _as_arrays(data: Data) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for g, arrays in data.items():
        arrs = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
        lens = {len(v) for v in arrs.values()}
        if len(lens) > 1:
            raise ValueError(f"arrays in group {g!r} are not aligned (lengths {sorted(lens)})")
        if lens == {0}:
            raise ValueError(f"group {g!r} is empty")
        out[g] = arrs
    return out


def bootstrap_replicates(stat_fn: Callable[[Data], float | np.ndarray], data: Data, n_boot: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (point estimate, replicates) of ``stat_fn`` under the paired, group-stratified bootstrap."""
    arrs = _as_arrays(data)
    rng = np.random.default_rng(seed)
    point = np.asarray(stat_fn(arrs), dtype=float)
    reps = np.empty((n_boot,) + point.shape)
    sizes = {g: len(next(iter(a.values()))) for g, a in arrs.items()}
    groups = sorted(arrs)  # fixed order -> reproducible index draws
    for b in range(n_boot):
        resampled = {}
        for g in groups:
            idx = rng.integers(0, sizes[g], sizes[g])
            resampled[g] = {k: v[idx] for k, v in arrs[g].items()}
        reps[b] = stat_fn(resampled)
    return point, reps


def paired_bootstrap(stat_fn: Callable[[Data], float], data: Data, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> BootResult:
    """Percentile CI of a scalar statistic under the paired, group-stratified bootstrap."""
    point, reps = bootstrap_replicates(stat_fn, data, n_boot, seed)
    lo, hi = percentile_ci(reps, ci)
    return BootResult(float(point), float(lo), float(hi), ci, n_boot, reps)


# ----------------------------------------------------------------------------- common effects


def bootstrap_paired_diff(model: np.ndarray, base: np.ndarray, direction: float = 1.0, **kw) -> BootResult:
    """direction × mean(model − base) over one group of images (e.g. unseen S change, seen ΔRMI)."""
    return paired_bootstrap(lambda d: direction * float(np.mean(d["g"]["m"] - d["g"]["b"])), {"g": {"m": model, "b": base}}, **kw)


def bootstrap_mean(x: np.ndarray, **kw) -> BootResult:
    return paired_bootstrap(lambda d: float(np.mean(d["g"]["x"])), {"g": {"x": x}}, **kw)


def bootstrap_delta(model_seen, base_seen, model_unseen, base_unseen, **kw) -> BootResult:
    """Eq. 3 with seen and unseen images resampled separately, pairs kept within each group."""

    def fn(d):
        return float(np.mean(d["seen"]["m"] - d["seen"]["b"]) - np.mean(d["unseen"]["m"] - d["unseen"]["b"]))

    return paired_bootstrap(fn, {"seen": {"m": model_seen, "b": base_seen}, "unseen": {"m": model_unseen, "b": base_unseen}}, **kw)


def bootstrap_gamma(pair: tuple, image: tuple, caption: tuple, base_seen, base_unseen, **kw) -> BootResult:
    """Eq. 4; each condition is ``(seen_array, unseen_array)``; all share the replicate's indices."""

    def fn(d):
        s, u = d["seen"], d["unseen"]
        dl = {c: np.mean(s[c] - s["b"]) - np.mean(u[c] - u["b"]) for c in ("pair", "image", "caption")}
        return float(dl["pair"] - (dl["image"] + dl["caption"]))

    data = {
        "seen": {"pair": pair[0], "image": image[0], "caption": caption[0], "b": base_seen},
        "unseen": {"pair": pair[1], "image": image[1], "caption": caption[1], "b": base_unseen},
    }
    return paired_bootstrap(fn, data, **kw)


def pool_seeds(per_seed: Sequence[Mapping[int, float]]) -> dict[int, float]:
    """Average each image's score across seeds (pooled value). All seeds must cover the same images."""
    if not per_seed:
        raise ValueError("no seeds")
    ids = set(per_seed[0])
    for s in per_seed[1:]:
        if set(s) != ids:
            raise ValueError("seeds cover different image sets; cannot pool")
    return {i: float(np.mean([s[i] for s in per_seed])) for i in per_seed[0]}


# ----------------------------------------------------------------------------- margins (D5)


def sd_margin(values_by_seed: Sequence[float], multiplier: float = 2.0) -> float:
    """multiplier × sample SD (ddof=1) of a per-seed quantity."""
    v = np.asarray(values_by_seed, dtype=float)
    if v.size < 2:
        raise ValueError("need at least two seeds for an SD margin")
    return float(multiplier * np.std(v, ddof=1))


def style_margin(unseen_means_by_seed: Sequence[float], multiplier: float = 2.0) -> float:
    """δ_S = 2 × SD across d=0 seeds of the unseen-group mean of an S metric."""
    return sd_margin(unseen_means_by_seed, multiplier)


def association_margin(seen_means_by_seed: Sequence[float], unseen_means_by_seed: Sequence[float], multiplier: float = 2.0) -> float:
    """δ_A = 2 × SD across d=0 seeds of the seen-minus-unseen gap S^seen_0 − S^unseen_0."""
    gaps = np.asarray(seen_means_by_seed, dtype=float) - np.asarray(unseen_means_by_seed, dtype=float)
    return sd_margin(gaps, multiplier)


def memory_margin(paired_d64_effect: float, floor: float, rel: float = 0.10) -> float:
    """δ_V = max(rel × Paired d=64 V effect, floor)."""
    return float(max(rel * paired_d64_effect, floor))


# ----------------------------------------------------------------------------- monotone fits and τ


def isotonic_increasing(y: Sequence[float], w: Sequence[float] | None = None) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool-adjacent-violators)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    blocks: list[list[float]] = []  # [weighted mean, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([np.full(c, m) for m, _, c in blocks]) if blocks else y


def _check_doses(doses: Sequence[float]) -> np.ndarray:
    d = np.asarray(doses, dtype=float)
    if np.any(d <= 0) or np.any(np.diff(d) <= 0):
        raise ValueError("doses must be positive and strictly increasing (d=0 is the reference, not a fit point)")
    return d


def first_crossing(doses: Sequence[float], fitted: Sequence[float], delta: float, interpolate: bool = False) -> float:
    """Smallest dose whose fitted (monotone) effect exceeds ``delta``; +inf if none.

    With ``interpolate``, the crossing is linearly interpolated on log₂ d between the last grid
    point at or below δ and the first above it (left-censored at the smallest dose).
    """
    d = _check_doses(doses)
    f = np.asarray(fitted, dtype=float)
    above = np.nonzero(f > delta)[0]
    if above.size == 0:
        return float("inf")
    k = int(above[0])
    if not interpolate or k == 0:
        return float(d[k])
    x0, x1 = np.log2(d[k - 1]), np.log2(d[k])
    t = (delta - f[k - 1]) / (f[k] - f[k - 1])
    return float(2.0 ** (x0 + t * (x1 - x0)))


def tau_from_lower_bound(doses: Sequence[float], lower: Sequence[float], delta: float) -> float:
    """τ̂ = smallest grid dose whose CI lower bound exceeds δ (+inf if none)."""
    return first_crossing(doses, lower, delta, interpolate=False)


@dataclass
class TauResult:
    doses: list[float]
    delta: float
    effect: list[float]  # raw effect per dose (original data)
    fitted: list[float]  # isotonic fit of the original data
    lower: list[float]  # pointwise CI bounds of the replicate fits
    upper: list[float]
    tau: float  # τ̂ (CI-lower-bound rule); inf = never crosses
    replicate_taus: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "doses": self.doses,
            "delta": self.delta,
            "effect": self.effect,
            "fitted": self.fitted,
            "ci_lo": self.lower,
            "ci_hi": self.upper,
            "tau": _json_float(self.tau),
            "log2_tau": _json_float(np.log2(self.tau)),
            "frac_replicates_never_cross": float(np.mean(np.isinf(self.replicate_taus))),
        }


def _json_float(x: float):
    return x if np.isfinite(x) else ("inf" if x > 0 else "-inf")


def _tau_from_reps(doses, point_effect, rep_effects, delta, ci, interpolate) -> TauResult:
    fits = np.vstack([isotonic_increasing(r) for r in rep_effects])
    lo, hi = percentile_ci(fits, ci, axis=0)
    rep_taus = np.array([first_crossing(doses, f, delta, interpolate) for f in fits])
    return TauResult(
        doses=[float(x) for x in doses],
        delta=float(delta),
        effect=[float(x) for x in point_effect],
        fitted=[float(x) for x in isotonic_increasing(point_effect)],
        lower=[float(x) for x in lo],
        upper=[float(x) for x in hi],
        tau=tau_from_lower_bound(doses, lo, delta),
        replicate_taus=rep_taus,
    )


def estimate_tau(
    effect_fn: Callable[[Data], np.ndarray],
    data: Data,
    doses: Sequence[float],
    delta: float,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    interpolate: bool = True,
) -> TauResult:
    """τ for one component. ``effect_fn(data)`` returns the effect at every dose (len(doses))."""
    _check_doses(doses)
    point, reps = bootstrap_replicates(lambda d: np.asarray(effect_fn(d), dtype=float), data, n_boot, seed)
    return _tau_from_reps(doses, point, reps, delta, ci, interpolate)


def _log2_diff(t_mem: float, t_style: float) -> float:
    m_inf, s_inf = np.isinf(t_mem), np.isinf(t_style)
    if m_inf and s_inf:
        return 0.0
    if m_inf:
        return float("inf")
    if s_inf:
        return float("-inf")
    return float(np.log2(t_mem) - np.log2(t_style))


def estimate_taus(
    style_fn: Callable[[Data], np.ndarray],
    memory_fn: Callable[[Data], np.ndarray],
    data: Data,
    doses: Sequence[float],
    delta_style: float,
    delta_memory: float,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    interpolate: bool = True,
) -> dict:
    """τ_style, τ_memory and the bootstrap CI of log₂ τ_memory − log₂ τ_style.

    Both components are computed from the SAME resampled images in every replicate (style
    typically uses the unseen group and memory the seen group of the same ``data``).
    """
    _check_doses(doses)
    k = len(doses)

    def both(d):
        return np.concatenate([np.asarray(style_fn(d), dtype=float), np.asarray(memory_fn(d), dtype=float)])

    point, reps = bootstrap_replicates(both, data, n_boot, seed)
    style = _tau_from_reps(doses, point[:k], reps[:, :k], delta_style, ci, interpolate)
    memory = _tau_from_reps(doses, point[k:], reps[:, k:], delta_memory, ci, interpolate)
    diffs = np.array([_log2_diff(m, s) for m, s in zip(memory.replicate_taus, style.replicate_taus)])
    lo, hi = percentile_ci(diffs, ci)
    s_inf, m_inf = np.isinf(style.replicate_taus), np.isinf(memory.replicate_taus)
    return {
        "style": style.to_dict(),
        "memory": memory.to_dict(),
        "log2_tau_diff": {
            "estimate": _json_float(_log2_diff(memory.tau, style.tau)),
            "ci_lo": _json_float(float(lo)),
            "ci_hi": _json_float(float(hi)),
            "ci": ci,
            "n_boot": n_boot,
            "frac_memory_only_censored": float(np.mean(m_inf & ~s_inf)),
            "frac_style_only_censored": float(np.mean(s_inf & ~m_inf)),
            "frac_both_censored": float(np.mean(s_inf & m_inf)),
            "frac_diff_positive": float(np.mean(diffs > 0)),
            "ci_above_zero": bool(lo > 0),
        },
    }


def equivalence_below(boot: BootResult, delta: float) -> bool:
    """G2 primary rule: the upper bound of the (two-sided ``boot.ci``) CI of the V effect is below δ_V."""
    return bool(boot.hi < delta)
