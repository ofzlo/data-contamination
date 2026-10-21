"""A (item association): Δ_{d,c} (eq. 3) and Γ_d (eq. 4).

With S^g_{d,c} the group mean of a per-image score m(ŷ_i, R_i) over group g (eq. 2):

    Δ_{d,c} = (S^seen_{d,c} − S^seen_0) − (S^unseen_{d,c} − S^unseen_0)            (eq. 3)
    Γ_d     = Δ_{d,pair} − (Δ_{d,image} + Δ_{d,caption})                            (eq. 4)

The primary A metric uses CIDEr against the 9 held-out references (``heldout`` config).
All per-image arrays passed here must come from the SAME joint scoring call per checkpoint
(see ``contam.eval.caption_metrics``) and be aligned by image id: element k of ``model_seen`` and
``base_seen`` refer to the same seen image. Because the image sets are identical, the group-mean
difference equals the mean of per-image differences, which is what the paired bootstrap resamples.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def delta_from_means(seen_dc: float, seen_0: float, unseen_dc: float, unseen_0: float) -> float:
    """Eq. 3 from group means."""
    return (seen_dc - seen_0) - (unseen_dc - unseen_0)


def delta(model_seen: np.ndarray, base_seen: np.ndarray, model_unseen: np.ndarray, base_unseen: np.ndarray) -> float:
    """Eq. 3 from aligned per-image score arrays (model at (d, c) vs. d=0)."""
    ms, bs, mu, bu = (np.asarray(x, dtype=float) for x in (model_seen, base_seen, model_unseen, base_unseen))
    if ms.shape != bs.shape or mu.shape != bu.shape:
        raise ValueError("model and base arrays must be aligned per image within each group")
    return delta_from_means(ms.mean(), bs.mean(), mu.mean(), bu.mean())


def gamma(delta_pair: float, delta_image: float, delta_caption: float) -> float:
    """Eq. 4: paired item-specific gain beyond the sum of the unimodal gains."""
    return delta_pair - (delta_image + delta_caption)


def gamma_from_arrays(
    pair: Sequence[np.ndarray], image: Sequence[np.ndarray], caption: Sequence[np.ndarray], base_seen: np.ndarray, base_unseen: np.ndarray
) -> float:
    """Eq. 4 from per-image arrays. Each of ``pair``/``image``/``caption`` is ``(seen, unseen)``."""
    d = {name: delta(s, base_seen, u, base_unseen) for name, (s, u) in (("pair", pair), ("image", image), ("caption", caption))}
    return gamma(d["pair"], d["image"], d["caption"])


def aligned(scores: Mapping[int, float], ids: Sequence[int]) -> np.ndarray:
    """Per-image scores as an array in the order of ``ids`` (KeyError if any is missing)."""
    return np.array([scores[i] for i in ids], dtype=float)
