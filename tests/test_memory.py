import numpy as np
import pytest

from contam.eval import memory


def test_normalize_and_exact_match():
    assert memory.normalize("  A Dog's  bed, on the  GRASS. ") == "a dogs bed on the grass"
    assert memory.exact_match("A dog on the grass.", "a dog on the grass")
    assert memory.exact_match("A dog-on the grass!", "a dog on the  grass")
    assert not memory.exact_match("A dog on grass.", "A dog on the grass.")


def test_longest_shared_ngram():
    assert memory.longest_shared_ngram("a man riding a red bike", "A man is riding a red bike.") == 4
    assert memory.longest_shared_ngram("x y z", "a b c") == 0
    assert memory.longest_shared_ngram("", "a b") == 0
    assert memory.longest_shared_ngram("The cat.", "the CAT") == 2


def test_text_memory_metrics():
    refs = [f"ref number {k}" for k in range(10)]
    m = memory.text_memory_metrics("Ref number 3.", refs, 3)
    assert m["exact_match_train"] == 1 and m["lsn_train"] == 3
    assert m["lsn_heldout_max"] == 2 and m["exact_match_heldout_any"] == 0


class FakeScorer:
    """log p = −len(ref words), n_tokens = len(ref words) + 1 (end token)."""

    def __init__(self):
        self.calls = 0

    def score(self, images, prompt, refs):
        self.calls += 1
        assert len(images) == len(refs)
        n = np.array([len(r.split()) for r in refs])
        return -n.astype(float), n + 1


def test_rmi_arithmetic_with_fake_scorer():
    refs = {7: ["a b"] + ["a b c d"] * 9, 8: ["a b c d e f"] * 10}
    lp = memory.score_all_refs(FakeScorer(), lambda i: f"img{i}", [7, 8], refs, "Describe.", batch_size=3)
    np.testing.assert_allclose(lp[7].logp_sum, [-2] + [-4] * 9)
    # RMI = log p(train) − mean held = −2 − (−4) = 2
    assert memory.rmi(lp[7], 0) == pytest.approx(2.0)
    # normalized: −2/3 − (−4/5)
    assert memory.rmi(lp[7], 0, normalized=True) == pytest.approx(-2 / 3 + 4 / 5)
    assert memory.train_ref_rank(lp[7], 0) == 1
    assert memory.train_ref_rank(lp[7], 1) == 2  # one strictly better (ref 0)
    assert memory.rmi(lp[8], 4) == pytest.approx(0.0)
    assert memory.train_ref_rank(lp[8], 4) == 1  # ties do not count against


def test_rmi_hand_values():
    lp = memory.RefLogprobs(np.array([-10.0, -20.0, -30.0, -40, -50, -60, -70, -80, -90, -100]), np.full(10, 10))
    held = np.array([-10.0, -30, -40, -50, -60, -70, -80, -90, -100]).mean()
    assert memory.rmi(lp, 1) == pytest.approx(-20 - held)
    assert memory.train_ref_rank(lp, 1) == 2


def test_delta_rmi():
    d = memory.delta_rmi({1: 3.0, 2: 1.0}, {1: 1.0, 2: 1.5}, [2, 1])
    np.testing.assert_allclose(d, [-0.5, 2.0])
