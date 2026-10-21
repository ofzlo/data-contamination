"""V (verbatim memory): reproduction of the training reference.

Text-side metrics (from generated captions):

* exact match with the training reference after normalization (lowercase, apostrophes removed,
  other punctuation -> space, whitespace collapsed);
* longest shared contiguous word n-gram with the training reference, and with the held-out
  references as a control (max and mean over the 9).

Likelihood-side metrics (teacher-forced, from a ``RefScorer``):

* rank of the training reference among all 10 references by log p (1 = most likely);
* RMI (plan, "Shared metrics"):

      RMI_i = log p(r_i^train | x_i) − (1/9) Σ_j log p(r_ij^held | x_i)

  using the SUMMED caption-token log-probability (primary, ``rmi_sum``) and the per-token mean
  (``rmi_norm``, log p / n_tokens) as a length-robust variant;
* ΔRMI_i = RMI_i(checkpoint) − RMI_i(d=0) on the same image (primary V effect: mean over seen
  images; unseen images are a control whose "training" ref was never trained on).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

_APOS = re.compile(r"['’`]")
_NONWORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, replace other punctuation by spaces, collapse whitespace."""
    t = _APOS.sub("", text.lower())
    return _NONWORD.sub(" ", t).strip()


def exact_match(hyp: str, ref: str) -> bool:
    return normalize(hyp) == normalize(ref)


def longest_shared_ngram(hyp: str, ref: str) -> int:
    """Length (in words) of the longest contiguous word sequence shared by ``hyp`` and ``ref``."""
    a, b = normalize(hyp).split(), normalize(ref).split()
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def text_memory_metrics(hyp: str, refs: Sequence[str], train_idx: int) -> dict[str, float]:
    """Exact-match / n-gram metrics of one caption vs. its training and held-out references."""
    train = refs[train_idx]
    held = [r for j, r in enumerate(refs) if j != train_idx]
    held_lsn = [longest_shared_ngram(hyp, r) for r in held]
    return {
        "exact_match_train": float(exact_match(hyp, train)),
        "exact_match_heldout_any": float(any(exact_match(hyp, r) for r in held)),
        "lsn_train": float(longest_shared_ngram(hyp, train)),
        "lsn_heldout_max": float(max(held_lsn)),
        "lsn_heldout_mean": float(np.mean(held_lsn)),
    }


# ----------------------------------------------------------------------------- teacher forcing


class RefScorer(Protocol):
    """Model-agnostic teacher-forced scorer.

    ``score(images, prompt, refs)`` scores ``refs[k]`` as the caption of ``images[k]`` under the
    captioning ``prompt`` (same template as generation) and returns two arrays of length
    ``len(refs)``: the SUMMED log-probability over caption tokens and the number of caption tokens
    scored. Prompt and image tokens are excluded; implementations must document whether an
    end-of-caption token is included and apply that choice to every reference.
    """

    def score(self, images: Sequence, prompt: str, refs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class RefLogprobs:
    """Teacher-forced log-probs of all references of one image."""

    logp_sum: np.ndarray  # (n_refs,)
    n_tokens: np.ndarray  # (n_refs,)


def score_all_refs(
    scorer: RefScorer,
    load_image: Callable[[int], object],
    image_ids: Sequence[int],
    refs_by_id: Mapping[int, Sequence[str]],
    prompt: str,
    batch_size: int = 16,
    progress: bool = False,
) -> dict[int, RefLogprobs]:
    """Score every reference of every image (images repeated once per reference)."""
    pairs = [(iid, j) for iid in image_ids for j in range(len(refs_by_id[iid]))]
    lp: dict[int, list] = {iid: [None] * len(refs_by_id[iid]) for iid in image_ids}
    nt: dict[int, list] = {iid: [None] * len(refs_by_id[iid]) for iid in image_ids}
    it = range(0, len(pairs), batch_size)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="teacher-forced scoring")
    cache: dict[int, object] = {}
    for start in it:
        chunk = pairs[start : start + batch_size]
        for iid, _ in chunk:
            if iid not in cache:
                cache[iid] = load_image(iid)
        s, n = scorer.score([cache[iid] for iid, _ in chunk], prompt, [refs_by_id[iid][j] for iid, j in chunk])
        for (iid, j), sv, nv in zip(chunk, s, n):
            lp[iid][j], nt[iid][j] = float(sv), int(nv)
        keep = {iid for iid, _ in pairs[start + batch_size : start + 2 * batch_size]}
        cache = {k: v for k, v in cache.items() if k in keep}
    return {iid: RefLogprobs(np.array(lp[iid]), np.array(nt[iid])) for iid in image_ids}


def rmi(lp: RefLogprobs, train_idx: int, normalized: bool = False) -> float:
    """RMI_i = log p(r_train | x) − mean_j log p(r_held_j | x). ``normalized`` uses per-token log p."""
    v = lp.logp_sum / lp.n_tokens if normalized else lp.logp_sum
    held = np.delete(v, train_idx)
    return float(v[train_idx] - held.mean())


def train_ref_rank(lp: RefLogprobs, train_idx: int, normalized: bool = False) -> int:
    """Rank (1 = highest log p) of the training reference among all references.

    Ties count against the training reference only if strictly greater (rank = 1 + #{j: v_j > v_train}).
    """
    v = lp.logp_sum / lp.n_tokens if normalized else lp.logp_sum
    return int(1 + np.sum(v > v[train_idx]))


def likelihood_memory_metrics(lp: RefLogprobs, train_idx: int) -> dict[str, float]:
    return {
        "rmi_sum": rmi(lp, train_idx, normalized=False),
        "rmi_norm": rmi(lp, train_idx, normalized=True),
        "train_rank_sum": float(train_ref_rank(lp, train_idx, normalized=False)),
        "train_rank_norm": float(train_ref_rank(lp, train_idx, normalized=True)),
        "logp_train_sum": float(lp.logp_sum[train_idx]),
        "ntok_train": float(lp.n_tokens[train_idx]),
    }


def delta_rmi(rmi_ckpt: Mapping[int, float], rmi_d0: Mapping[int, float], ids: Sequence[int]) -> np.ndarray:
    """ΔRMI_i = RMI_i(checkpoint) − RMI_i(d=0), aligned with ``ids``."""
    return np.array([rmi_ckpt[i] - rmi_d0[i] for i in ids], dtype=float)
