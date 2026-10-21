import shutil

import numpy as np
import pytest

from contam.data.splits import make_split
from contam.eval import caption_metrics as cm
from tests.conftest import make_annotations


def _setup(n=(6, 20, 10)):
    ann = make_annotations(*n, seed=1)
    split = make_split(ann, 3, 4)
    refs = {a["image_id"]: a["refs"] for a in ann}
    # hypothesis = one held-out ref (d=0-like) for every image
    hyps = {}
    for a in ann:
        t = split["train_ref_idx"][str(a["image_id"])]
        hyps[a["image_id"]] = a["refs"][(t + 1) % 10]
    return ann, split, refs, hyps


def test_ref_configs():
    refs = {1: [f"r{k}" for k in range(10)]}
    tri = {"1": 4}
    assert cm.refs_for_config(refs, tri, "train") == {1: ["r4"]}
    assert cm.refs_for_config(refs, tri, "all10") == {1: refs[1]}
    h = cm.refs_for_config(refs, tri, "heldout")[1]
    assert len(h) == 9 and "r4" not in h
    with pytest.raises(ValueError):
        cm.refs_for_config(refs, tri, "nope")


def test_cider_scored_jointly_not_per_group():
    ann, split, refs, hyps = _setup()
    joint = cm.score_split(hyps, refs, split, ref_configs=["heldout"], metrics=["cider"], tokenizer="simple")["heldout"]["cider"]
    # Same numbers as one score_corpus call over all images ...
    held = cm.refs_for_config(refs, split["train_ref_idx"], "heldout")
    allc = cm.score_corpus(hyps, held, ["cider"], tokenizer="simple")["cider"]
    assert all(joint[i] == pytest.approx(allc[i]) for i in hyps)
    # ... and different from scoring the seen group alone (IDF depends on the scored corpus).
    seen = split["seen"]
    alone = cm.score_corpus({i: hyps[i] for i in seen}, {i: held[i] for i in seen}, ["cider"], tokenizer="simple")["cider"]
    diffs = np.array([alone[i] - joint[i] for i in seen])
    assert np.max(np.abs(diffs)) > 1e-6
    groups = cm.by_group(joint, split)
    assert set(groups["seen"]) == set(split["seen"]) and set(groups["unseen"]) == set(split["unseen"])


def test_score_split_requires_full_corpus():
    ann, split, refs, hyps = _setup()
    partial = {i: hyps[i] for i in split["seen"]}
    with pytest.raises(ValueError):
        cm.score_split(partial, refs, split, metrics=["cider"], tokenizer="simple")
    out = cm.score_split(partial, refs, split, ref_configs=["train"], metrics=["cider"], tokenizer="simple", require_all=False)
    assert set(out["train"]["cider"]) == set(split["seen"])


def test_exact_training_ref_scores_high_on_train_config():
    ann, split, refs, hyps = _setup()
    train_hyps = {i: refs[i][split["train_ref_idx"][str(i)]] for i in hyps}
    s_train = cm.score_split(train_hyps, refs, split, ref_configs=["train"], metrics=["cider"], tokenizer="simple")["train"]["cider"]
    s_held = cm.score_split(hyps, refs, split, ref_configs=["train"], metrics=["cider"], tokenizer="simple")["train"]["cider"]
    assert np.mean(list(s_train.values())) > np.mean(list(s_held.values()))


def test_spice_cache_roundtrip(tmp_path):
    c = cm.SpiceCache(tmp_path / "c.json")
    c.put("a dog", ["a dog", "a cat"], 0.5)
    c.save()
    c2 = cm.SpiceCache(tmp_path / "c.json")
    assert c2.get("a dog", ["a dog", "a cat"]) == 0.5
    assert c2.get("a dog", ["a cat", "a dog"]) is None


@pytest.mark.java
@pytest.mark.skipif(shutil.which("java") is None, reason="needs java")
def test_ptb_tokenizer_cider():
    ann, split, refs, hyps = _setup((3, 6, 3))
    out = cm.score_split(hyps, refs, split, ref_configs=["heldout"], metrics=["cider"], tokenizer="ptb")
    s = out["heldout"]["cider"]
    assert len(s) == len(hyps) and all(np.isfinite(v) for v in s.values())
    tok = cm.tokenize({"a": ["A dog, running!", ""]})
    assert tok["a"] == ["a dog running", ""]
