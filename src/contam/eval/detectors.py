"""B6 contamination detectors: CDD, DCQ and KDS, adapted to image captioning.

Every detector gives a per-image score (CDD, DCQ) or a per-set score (KDS). Each is run on the seen
and the unseen images of a split, so on the untrained base and the d=0 models a detector should show
no seen/unseen difference (its false-positive behaviour), and on exposed checkpoints C9 asks which
regime (exposure, memorization, inflation) it follows. Image ids on the B1 analysis exclusion list
are dropped from the groups unless the caller keeps them (prereg Section 2).

CDD (Dong et al. 2024, "Generalization or Memorization", eqs. 4 to 7)
-------------------------------------------------------------------
Peakedness of the output distribution around the greedy output. With n T=1 samples s_1..s_n and the
T=0 caption s_0 of the same image and prompt,

    Peak(x) = (1/n) Σ_i 1[ED(s_i, s_0) <= α · l],   l = min(max_i len(s_i), l_max)

where ED is the token edit distance. The paper's defaults are α = 0.05, l_max = 100 and the decision
"leaked" if Peak > ξ = 0.01. Deviations: tokens are words after ``memory.normalize`` (model-agnostic;
a tokenizer can be passed), sequences are cut to l_max tokens before ED (so l_max caps the cost as
well as l), and samples are T=1 pure ancestral (our stored samples, 20 per image) instead of T=0.8.

DCQ (Golchin & Surdeanu 2023, "Data Contamination Quiz")
-------------------------------------------------------
For each image the original is its TRAINING reference (``train_ref_idx``; never trained on for
unseen images) and four word-level perturbations of it come from ``scripts/dcq_perturb.py``. Two
kinds of five-option quizzes, all at T=0 with a one-letter answer:

* BDQ (bias detector): the four perturbations as A to D, E = "None of the provided options".
* BCQ_p (bias compensator) for p in A to D: the same options with the original in place of the
  perturbation at position p.

A position is non-preferred if BDQ picks it fewer than ceil(k/5) times (k = quiz items); if none
is, all of A to D are. The upper estimate is the best BCQ accuracy over the non-preferred
positions (ties: the position BDQ picks least), the lower estimate Cohen's κ = (p_o − p_e) /
(1 − p_e), with p_o that accuracy and p_e the BDQ rate of that position. We run all four BCQs, so
the choice of positions needs no second pass. The per-image score is 1 if that best BCQ is answered
correctly.

KDS (Choi et al. 2025, "How Contaminated Is Your Benchmark?", eqs. 2 to 5)
--------------------------------------------------------------------------
Z = L2-normalized embeddings of a sample set under the checkpoint, Z' = the same after a short
LoRA fine-tune of that checkpoint on the set itself (r 8, α 32, dropout 0.1 on q/v, one epoch of
SGD at lr 1e-4, batch 4). With the RBF kernel Φ(Z)_ij = exp(−γ ||Z_i − Z_j||²),

    KDS = −(1/E) Σ_ij | Φ(Z)_ij · log(Φ(Z)_ij / Φ(Z')_ij) |,   E = sqrt(Σ_ij Φ(Z)_ij)

Higher (closer to 0) means more contaminated: fine-tuning moves seen samples less. γ is the
median heuristic, 1 / median of the squared pairwise distances of Z (i < j), and the same γ is used
for Z'. log(Φ/Φ') = γ (||Z'_i − Z'_j||² − ||Z_i − Z_j||²) is computed in that form, so no kernel
value underflows. A sample is (image, p_cap, training reference); its embedding is the mean of
the final-layer hidden states over the reference tokens and the end token
(``contam.models.HFVLM.embed``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from contam.eval.memory import normalize

LETTERS = "ABCDE"
NONE_OPTION = "None of the provided options."


def word_tokens(text: str) -> list[str]:
    return normalize(text).split()


def load_exclusion(path: str | Path | None) -> set[int]:
    """Image ids from a B1 exclusion file (one id per line, ``#`` comments allowed; relative paths
    are against the repository root)."""
    if not path:
        return set()
    from contam.config import resolve_path

    out = set()
    for line in resolve_path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(int(line))
    return out


def split_groups(split: Mapping, exclude: set[int] = frozenset()) -> dict[str, list[int]]:
    """``{"seen": ids, "unseen": ids}`` without the excluded ids, sorted."""
    return {g: sorted(int(i) for i in split[g] if int(i) not in exclude) for g in ("seen", "unseen")}


def subset(ids: Sequence[int], n: int, seed: int) -> list[int]:
    """``n`` ids drawn without replacement with ``seed`` (all of them if fewer), sorted."""
    ids = sorted(int(i) for i in ids)
    if n >= len(ids):
        return ids
    return sorted(int(i) for i in np.random.default_rng(seed).choice(ids, size=n, replace=False))


def detector_groups(cfg: Mapping, split: Mapping, detector: str, keep_excluded: bool = False) -> tuple[dict[str, list[int]], list[int]]:
    """The images ``detector`` runs on: seen / unseen without the B1 exclusion (unless kept), cut to
    ``detectors.<detector>.n_per_group`` for DCQ and KDS. Seen and unseen draw from separate streams
    (``sample_seed`` + 0 / + 1), so one group's subset does not depend on the other. Returns
    (groups, excluded ids)."""
    excl_file = cfg.get("analysis", {}).get("exclude_image_ids_file")
    excl = set() if keep_excluded else load_exclusion(excl_file)
    groups = split_groups(split, excl)
    dcfg = cfg["detectors"][detector]
    if "n_per_group" in dcfg:
        groups = {g: subset(groups[g], dcfg["n_per_group"], dcfg["sample_seed"] + k) for k, g in enumerate(("seen", "unseen"))}
    return groups, sorted(excl)


# ----------------------------------------------------------------------------- CDD


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two token sequences (unit insert, delete, substitute)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cdd_peak(samples: Sequence[str], greedy: str, alpha: float = 0.05, max_len: int = 100, tokenize: Callable[[str], list] = word_tokens) -> float:
    """Peak(x): share of T>0 samples within edit distance α·l of the greedy output (eq. 6)."""
    if not samples:
        raise ValueError("CDD needs at least one sample")
    g = tokenize(greedy)[:max_len]
    toks = [tokenize(s)[:max_len] for s in samples]
    l = max(len(t) for t in toks)
    thr = alpha * l
    return float(np.mean([edit_distance(t, g) <= thr for t in toks]))


def cdd_scores(greedy_rows: Sequence[Mapping], sample_rows: Sequence[Mapping], image_ids: Sequence[int], alpha: float = 0.05, max_len: int = 100,
               tokenize: Callable[[str], list] = word_tokens) -> dict[int, float]:
    """Per-image Peak from generation rows (``contam.generate`` format: T=0 rows and T>0 rows).

    Both row sets must come from one checkpoint and one prompt; every image needs its greedy caption
    and at least one sample.
    """
    for rows, what in ((greedy_rows, "greedy"), (sample_rows, "sample")):
        prov = {(r.get("checkpoint_id"), r.get("prompt_sha")) for r in rows}
        if len(prov) > 1:
            raise ValueError(f"{what} rows mix checkpoints / prompts: {sorted(prov, key=str)[:3]}")
    if greedy_rows and sample_rows:
        g_prov = {(r.get("checkpoint_id"), r.get("prompt_sha")) for r in greedy_rows}
        s_prov = {(r.get("checkpoint_id"), r.get("prompt_sha")) for r in sample_rows}
        if g_prov != s_prov:
            raise ValueError(f"greedy rows {g_prov} and sample rows {s_prov} come from different checkpoints or prompts")
    greedy = {int(r["image_id"]): r["caption"] for r in greedy_rows if float(r["temperature"]) == 0}
    samples: dict[int, list[str]] = {}
    for r in sample_rows:
        if float(r["temperature"]) > 0:
            samples.setdefault(int(r["image_id"]), []).append(r["caption"])
    missing = [i for i in image_ids if i not in greedy or i not in samples]
    if missing:
        raise ValueError(f"{len(missing)} images lack a greedy caption or samples, e.g. {missing[:5]}")
    return {int(i): cdd_peak(samples[i], greedy[i], alpha, max_len, tokenize) for i in image_ids}


# ----------------------------------------------------------------------------- DCQ


@dataclass
class QuizItem:
    image_id: int
    original: str
    perturbations: list[str]  # four, in a fixed order


def check_perturbations(original: str, perturbations: Sequence[str], max_len_diff: int = 3) -> list[str]:
    """Problems with one item's perturbations (empty if usable).

    Four perturbations, each different from the original and from each other after
    ``memory.normalize``, and word-level: word count within ``max_len_diff`` of the original.
    """
    probs = []
    if len(perturbations) != 4:
        probs.append(f"{len(perturbations)} perturbations, need 4")
    norm = [normalize(p) for p in perturbations]
    if normalize(original) in norm:
        probs.append("a perturbation equals the original")
    if len(set(norm)) != len(norm):
        probs.append("perturbations are not distinct")
    n0 = len(word_tokens(original))
    for p in perturbations:
        if not p.strip():
            probs.append("empty perturbation")
        elif abs(len(word_tokens(p)) - n0) > max_len_diff:
            probs.append(f"length differs by more than {max_len_diff} words: {p!r}")
    return probs


def quiz_options(item: QuizItem, bcq_position: int | None = None) -> list[str]:
    """The five options: the perturbations (original at ``bcq_position`` for a BCQ), then E."""
    opts = list(item.perturbations)
    if bcq_position is not None:
        if not 0 <= bcq_position < 4:
            raise ValueError("bcq_position must be 0..3 (A to D)")
        opts[bcq_position] = item.original
    return opts + [NONE_OPTION]


def dcq_prompt(template: str, options: Sequence[str]) -> str:
    if len(options) != 5:
        raise ValueError("a DCQ quiz has five options")
    lines = "\n".join(f"{LETTERS[k]}. {o.strip()}" for k, o in enumerate(options))
    return template.replace("{options}", lines)


_ANSWER = re.compile(r"\b([A-E])\b")


def parse_answer(text: str) -> str | None:
    """First standalone letter A to E in the model's answer ("B", "B.", "(B)", "Answer: B"), else None."""
    m = _ANSWER.search(text.strip().upper()) if text else None
    return m.group(1) if m else None


def dcq_estimate(bdq: Mapping[int, str | None], bcq: Mapping[int, Mapping[int, str | None]]) -> dict:
    """Contamination estimates from BDQ answers and BCQ answers per position (0..3).

    ``bdq[image_id]`` is the letter picked in the BDQ; ``bcq[p][image_id]`` the letter picked in
    BCQ_p. Unparsed answers (None) count as wrong and as no position.
    """
    ids = sorted(bdq)
    k = len(ids)
    if k == 0:
        raise ValueError("no quiz items")
    for p in range(4):
        if sorted(bcq.get(p, {})) != ids:
            raise ValueError(f"BCQ position {LETTERS[p]} does not cover the BDQ items")
    counts = {L: sum(bdq[i] == L for i in ids) for L in LETTERS}
    cutoff = math.ceil(k / 5)
    non_pref = [p for p in range(4) if counts[LETTERS[p]] < cutoff] or list(range(4))
    acc = {p: float(np.mean([bcq[p][i] == LETTERS[p] for i in ids])) for p in range(4)}
    p_e = {p: counts[LETTERS[p]] / k for p in range(4)}
    best = max(non_pref, key=lambda p: (acc[p], -p_e[p]))
    p_o = acc[best]
    kappa = (p_o - p_e[best]) / (1 - p_e[best]) if p_e[best] < 1 else 0.0
    return {
        "k": k,
        "bdq_counts": counts,
        "bdq_unparsed": sum(bdq[i] is None for i in ids),
        "non_preferred": [LETTERS[p] for p in non_pref],
        "bcq_accuracy": {LETTERS[p]: acc[p] for p in range(4)},
        "best_position": LETTERS[best],
        "max": p_o,
        "min_kappa": float(kappa),
        "per_image": {int(i): float(bcq[best][i] == LETTERS[best]) for i in ids},
    }


def dcq_items(perturbation_rows: Sequence[Mapping], split: Mapping, refs: Mapping[int, Sequence[str]], image_ids: Sequence[int]) -> list[QuizItem]:
    """Quiz items for ``image_ids`` from ``scripts/dcq_perturb.py`` rows; the original must be the
    image's training reference under this split and every item must pass ``check_perturbations``."""
    by_id = {int(r["image_id"]): r for r in perturbation_rows}
    items, bad = [], []
    for i in image_ids:
        r = by_id.get(int(i))
        if r is None:
            bad.append((i, "no perturbations"))
            continue
        train = refs[int(i)][int(split["train_ref_idx"][str(i)])]
        if r["original"] != train:
            bad.append((i, "original is not this split's training reference"))
            continue
        probs = check_perturbations(train, r["perturbations"])
        if probs:
            bad.append((i, "; ".join(probs)))
            continue
        items.append(QuizItem(int(i), train, list(r["perturbations"])))
    if bad:
        raise ValueError(f"{len(bad)} quiz items unusable, e.g. {bad[:3]}")
    return items


# ----------------------------------------------------------------------------- KDS


def _sq_dists(Z: np.ndarray) -> np.ndarray:
    sq = np.sum(Z * Z, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)


def l2_normalize(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def median_gamma(Z: np.ndarray) -> float:
    """1 / median squared pairwise distance (i < j) of the normalized embeddings."""
    D = _sq_dists(l2_normalize(Z))
    iu = np.triu_indices(len(D), k=1)
    med = float(np.median(D[iu]))
    if med <= 0:
        raise ValueError("all embeddings coincide; the median heuristic is undefined")
    return 1.0 / med


def kernel_divergence_score(Z: np.ndarray, Z_ft: np.ndarray, gamma: float | None = None) -> dict:
    """KDS of one sample set (module docstring). Rows of ``Z`` and ``Z_ft`` are the same samples."""
    Z, Z_ft = l2_normalize(Z), l2_normalize(Z_ft)
    if Z.shape != Z_ft.shape or len(Z) < 2:
        raise ValueError(f"need two aligned sets of >= 2 embeddings, got {Z.shape} and {Z_ft.shape}")
    g = median_gamma(Z) if gamma is None else float(gamma)
    D, D_ft = _sq_dists(Z), _sq_dists(Z_ft)
    K = np.exp(-g * D)
    E = math.sqrt(float(K.sum()))
    div = float(np.abs(K * g * (D_ft - D)).sum()) / E
    return {"kds": -div, "gamma": g, "n": len(Z), "normalizer": E}


# ----------------------------------------------------------------------------- AUROC


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): seen (pos) vs unseen (neg) images."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        raise ValueError("both groups need scores")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def group_summary(scores: Mapping[int, float], groups: Mapping[str, Sequence[int]], n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Seen / unseen means, their difference and the seen-vs-unseen AUROC, with bootstrap CIs
    (seen and unseen resampled separately, ``contam.eval.stats``)."""
    from contam.eval.stats import paired_bootstrap

    data = {g: {"x": np.array([scores[i] for i in groups[g]], float)} for g in ("seen", "unseen")}
    out = {g: {"n": len(groups[g]), "mean": float(data[g]["x"].mean()), "sd": float(data[g]["x"].std(ddof=1)) if len(groups[g]) > 1 else float("nan")} for g in data}
    kw = dict(n_boot=n_boot, seed=seed, ci=ci)
    out["seen_minus_unseen"] = paired_bootstrap(lambda d: float(d["seen"]["x"].mean() - d["unseen"]["x"].mean()), data, **kw).to_dict()
    out["auroc"] = paired_bootstrap(lambda d: auroc(d["seen"]["x"], d["unseen"]["x"]), data, **kw).to_dict()
    return out


# ----------------------------------------------------------------------------- GPU drivers


def run_dcq(vlm, load_image, items: Sequence[QuizItem], template: str, max_new_tokens: int = 3, progress: bool = True) -> dict[str, dict[int, dict]]:
    """Ask every quiz (BDQ, BCQ_A..BCQ_D) of every item at T=0. Returns ``{quiz: {image_id: {raw, letter}}}``
    with quiz in ``bdq``, ``bcq_A`` .. ``bcq_D``."""
    quizzes = [("bdq", None)] + [(f"bcq_{LETTERS[p]}", p) for p in range(4)]
    out: dict[str, dict[int, dict]] = {q: {} for q, _ in quizzes}
    it = items
    if progress:
        from tqdm import tqdm

        it = tqdm(items, desc="DCQ")
    for item in it:
        img = load_image(item.image_id)
        for q, pos in quizzes:
            raw = vlm.generate([img], dcq_prompt(template, quiz_options(item, pos)), 0.0, max_new_tokens)[0]
            out[q][item.image_id] = {"raw": raw, "letter": parse_answer(raw)}
    return out


def kds_finetune(vlm, load_image, examples: Sequence[tuple], epochs: int = 1, lr: float = 1e-4, batch_size: int = 4, lora_r: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, target_modules: Sequence[str] = ("q_proj", "v_proj"), seed: int = 0, max_len: int = 2048, progress: bool = True):
    """KDS's fine-tune: a fresh LoRA on the (already merged) checkpoint in ``vlm``, trained on
    the KDS sample set with plain SGD; returns the PEFT model (in place of
    ``vlm.model``; call ``unload()`` to restore). ``examples`` = [(image_id, prompt, ref)], images
    from ``load_image``. The loss is the caption-token cross-entropy of
    ``contam.train`` (``per_example_mean``), with the training example format of the main runs."""
    import torch
    from peft import LoraConfig, get_peft_model

    from contam.train import build_example, collate, forward_logits, micro_loss

    torch.manual_seed(seed)
    targets = "|".join(target_modules)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", target_modules=rf".*language_model.*\.({targets})")
    pm = get_peft_model(vlm.model, cfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    opt = torch.optim.SGD([p for p in pm.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    proc, end_id = vlm.processor, vlm.end_token_id
    use_amp = vlm.device.startswith("cuda")
    pm.train()
    steps = []
    for _ in range(epochs):
        order = rng.permutation(len(examples))
        steps += [order[s : s + batch_size] for s in range(0, len(order), batch_size)]
    it = steps
    if progress:
        from tqdm import tqdm

        it = tqdm(steps, desc="KDS fine-tune")
    for step in it:
        batch = []
        for k in step:
            iid, prompt, ref = examples[k]
            img = load_image(iid)
            ex = {"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ref}]}
            batch.append(build_example(proc, ex, img, end_id, max_len))
        b = collate(batch, proc.tokenizer.pad_token_id)
        labels = b.pop("labels").to(vlm.device)
        logits = forward_logits(pm, b, vlm.device, vlm.model.dtype, use_amp)
        loss = micro_loss(logits, labels, "per_example_mean", len(step), 0, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    pm.eval()
    return pm
