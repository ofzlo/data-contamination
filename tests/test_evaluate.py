import numpy as np
import pytest

from contam.data.splits import make_split
from contam.eval import memory
from contam.eval.evaluate import compare, evaluate_checkpoint
from tests.conftest import make_annotations


def _run(hyp_fn, ann, split, lp=None):
    refs = {a["image_id"]: a["refs"] for a in ann}
    gens = {i: hyp_fn(i, refs[i], split["train_ref_idx"][str(i)]) for i in refs}
    return evaluate_checkpoint(gens, ann, split, logprobs=lp, metrics=["cider"], tokenizer="simple", checkpoint_id="x")


def test_end_to_end_memorizing_checkpoint():
    ann = make_annotations(6, 20, 10, seed=2)
    split = make_split(ann, 1, 2)
    seen = set(split["seen"])
    base = _run(lambda i, r, t: r[(t + 1) % 10], ann, split)
    mem = _run(lambda i, r, t: r[t] if i in seen else r[(t + 1) % 10], ann, split)
    assert mem["summary"]["seen"]["exact_match_train"] == 1.0
    assert mem["summary"]["unseen"]["exact_match_train"] == 0.0
    c = compare([mem], [base], split, n_boot=200)
    assert c["V"]["exact_match_train"]["seen"]["ci_lo"] == pytest.approx(1.0)
    assert c["V"]["exact_match_train"]["unseen"]["estimate"] == 0.0
    # training-ref CIDEr gain is item-specific: Δ(train config) > 0
    assert c["A"]["cider_train"]["estimate"] > 0
    assert c["S"]["word_count"]["unseen"]["estimate"] == 0.0


def test_rmi_fields_with_logprobs():
    ann = make_annotations(2, 4, 2, seed=3)
    split = make_split(ann, 1, 2)
    lp = {a["image_id"]: memory.RefLogprobs(-np.arange(1.0, 11.0), np.full(10, 5)) for a in ann}
    m = _run(lambda i, r, t: r[0], ann, split, lp)
    row = m["per_image"][str(ann[0]["image_id"])]
    assert "rmi_sum" in row and "train_rank_sum" in row
