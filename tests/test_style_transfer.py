"""contam.eval.style_transfer: JS distances, Open Images class matching, paired bootstrap."""

from collections import Counter

import numpy as np
import pytest

from contam.eval import style_transfer as st


def test_js_divergence_bounds_and_symmetry():
    p, q = np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 5.0])
    assert st.js_divergence(p, p) == pytest.approx(0.0)
    assert st.js_divergence(p, q) == pytest.approx(1.0)
    a, b = np.array([3.0, 1.0, 1.0]), np.array([1.0, 1.0, 3.0])
    assert st.js_divergence(a, b) == pytest.approx(st.js_divergence(b, a))
    assert 0 < st.js_divergence(a, b) < 1
    # broadcasting over replicate rows, counts are normalized
    rows = np.vstack([a, 10 * a, b])
    np.testing.assert_allclose(st.js_divergence(rows, a[None, :]), [0, 0, st.js_divergence(b, a)], atol=1e-12)


def test_text_features():
    assert st.first2("The image shows a dog.") == Counter({"the image": 1})
    assert st.first2("Dog") == Counter({"dog": 1}) and st.first2("...") == Counter()
    assert st.function_word_counts("A dog and a cat on the mat") == Counter({"a": 2, "and": 1, "on": 1, "the": 1})
    assert st.pos_bigrams(["DET", "NOUN", "SPACE"]) == Counter({"<s>_DET": 1, "DET_NOUN": 1, "NOUN_</s>": 1})


def test_vocab_matrix_and_vector_share_columns():
    v = st.Vocab()
    rows = [Counter({"a": 2}), Counter({"b": 1, "a": 1})]
    ref = [Counter({"c": 4})]
    v.add(rows)
    v.add(ref)
    X = v.matrix(rows)
    assert X.shape == (2, 3)
    np.testing.assert_array_equal(X.toarray(), [[2, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(v.vector(ref), [0, 0, 4])


def test_class_names_and_coco_overlap(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("/m/1,Bat (Animal)\r\n/m/2,Human face\r\n/m/3,Teddy bear\r\n/m/4,Man\r\n/m/5,Mobile phone\r\n/m/6,Lighthouse\r\n")
    names = st.load_oi_classes(p)
    assert names == ["Bat (Animal)", "Human face", "Teddy bear", "Man", "Mobile phone", "Lighthouse"]
    assert st.normalize_class_name("Bat (Animal)") == "bat" and st.normalize_class_name("Human face") == "face"
    assert [st.coco_overlap(n) for n in names] == [False, True, True, True, True, False]  # body part = person
    assert [st.coco_overlap(n, near_aliases=False) for n in names] == [False, False, True, True, True, False]  # first (wider) set
    for n in ("Human hand", "Human hair", "Kitchen knife", "Coffee table", "Computer monitor", "Wall clock", "Polar bear", "Taxi", "Canoe"):
        assert st.coco_overlap(n) and not st.coco_overlap(n, near_aliases=False), n
    for n in ("Duck", "Lighthouse", "Billiard table", "Table tennis racket", "Dog bed", "Hair spray"):
        assert not st.coco_overlap(n), n
    assert set(st.OI_COCO_NEAR_ALIASES.values()) <= st.COCO_CATEGORIES


def _tag(s):
    """'word/TAG word/TAG ...' -> tagged tokens (no spaCy in unit tests)."""
    return [tuple(x.rsplit("/", 1)) for x in s.split()]


def test_class_matcher_longest_plural_and_noun_only():
    m = st.ClassMatcher(["Bear", "Teddy bear", "Orange", "Man", "Butterfly", "Knife", "Kitchen & dining room table"])
    assert m(_tag("a/DET teddy/NOUN bear/NOUN")) == {"Teddy bear"}  # not also Bear
    assert m(_tag("two/NUM bears/NOUN")) == {"Bear"}
    assert m(_tag("men/NOUN with/ADP knives/NOUN and/CCONJ butterflies/NOUN")) == {"Man", "Knife", "Butterfly"}
    assert m(_tag("an/DET orange/ADJ shirt/NOUN")) == set()  # adjective: not the fruit
    assert m(_tag("an/DET orange/NOUN ,/PUNCT")) == {"Orange"}
    assert m(_tag("kitchen/NOUN dining/NOUN room/NOUN table/NOUN")) == {"Kitchen & dining room table"}


def test_vocab_rows_and_replicates_by_domain():
    mentioned = [{"A", "B"}, {"C"}, set(), {"A"}]
    ref_named = [{"A"}, {"C", "B"}, {"B"}, set()]
    rows = st.vocab_rows(mentioned, ref_named, {"oi_all": {"A", "B", "C"}, "oi_noncoco": {"B", "C"}, "oi_rare": {"C"}})
    np.testing.assert_array_equal(rows["oi_all/mentioned"], [2, 1, 0, 1])
    np.testing.assert_array_equal(rows["oi_all/matched"], [1, 1, 0, 0])
    np.testing.assert_array_equal(rows["oi_noncoco/ref"], [0, 2, 1, 0])
    domain = np.array(["in", "out", "out", "near"])
    W = np.vstack([np.ones(4), [2, 0, 1, 1]]).astype(np.float32)
    r = st.vocab_replicates(rows, domain, W)
    assert r["all/oi_all/mentions"][0] == pytest.approx(4 / 4) and r["all/oi_all/matched"][0] == pytest.approx(2 / 4)
    assert r["all/oi_all/precision"][0] == pytest.approx(2 / 4) and r["all/oi_all/recall"][0] == pytest.approx(2 / 4)
    assert r["out/oi_all/mentions"][0] == pytest.approx(1 / 2) and r["out/oi_rare/precision"][0] == pytest.approx(1.0)
    assert np.isnan(r["near/oi_noncoco/precision"][0])  # nothing mentioned
    # replicate row: image 0 twice, image 2 and 3 once
    assert r["all/oi_all/mentions"][1] == pytest.approx((2 * 2 + 0 + 1) / 4)
    assert r["in/oi_all/precision"][1] == pytest.approx(0.5)


def test_boot_weights_and_paired_comparison():
    W = st.boot_weights(50, 300, seed=0)
    assert W.shape == (301, 50) and np.all(W[0] == 1) and np.all(W.sum(axis=1) == 50)
    np.testing.assert_array_equal(W, st.boot_weights(50, 300, seed=0))
    rng = np.random.default_rng(1)
    v = st.Vocab()
    cs = [Counter({k: 1}) for k in rng.choice(list("abcd"), size=50, p=[0.4, 0.3, 0.2, 0.1])]
    other = [Counter({k: 1}) for k in rng.choice(list("abcd"), size=50, p=[0.1, 0.2, 0.3, 0.4])]
    v.add(cs)
    v.add(other)
    q = {"ref": v.vector(cs)}
    a = st.js_replicates(v.matrix(cs), q, W)
    b = st.js_replicates(v.matrix(other), q, W)
    assert a["ref"][0] == pytest.approx(0.0)  # same corpus as the reference
    c = st.compare_replicates([{"js": b["ref"]}], [{"js": a["ref"]}])["js"]
    assert c["diff"]["estimate"] == pytest.approx(b["ref"][0] - a["ref"][0]) and c["diff"]["ci_lo"] > 0
    same = st.compare_replicates([{"js": a["ref"]}], [{"js": a["ref"]}])["js"]["diff"]
    assert same["estimate"] == 0 and same["ci_lo"] == 0 and same["ci_hi"] == 0
    lv = st.level([{"x": np.array([1.0, 2.0, 3.0])}, {"x": np.array([3.0, 2.0, 1.0])}])["x"]
    assert lv["estimate"] == pytest.approx(2.0) and lv["n_boot"] == 2 and lv["n_boot_used"] == 2
    nan = st.level([{"p": np.array([0.5, np.nan, 0.4, 0.6])}])["p"]
    assert nan["n_boot"] == 3 and nan["n_boot_used"] == 2  # NaN replicates are dropped and counted
