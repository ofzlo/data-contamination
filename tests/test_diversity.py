import numpy as np
import pytest

from contam.eval import diversity


def test_distinct_n_on_fixed_strings():
    texts = ["A dog runs.", "a dog sits", "The cat"]
    # unigrams: a dog runs a dog sits the cat -> 8 tokens, 6 types
    assert diversity.distinct_n(texts, 1) == pytest.approx(6 / 8)
    # bigrams (not across captions): a-dog dog-runs a-dog dog-sits the-cat -> 5, 4 types
    assert diversity.distinct_n(texts, 2) == pytest.approx(4 / 5)
    assert diversity.distinct_n(["one"], 2) == 0.0  # no bigram at all


def test_distinct_weighted_counts_repeats():
    X = diversity.ngram_matrix(["a b", "a c"], 1)
    assert diversity.distinct_weighted(X) == pytest.approx(3 / 4)
    assert diversity.distinct_weighted(X, np.array([2, 0])) == pytest.approx(2 / 4)  # caption 0 twice, caption 1 not drawn
    assert diversity.distinct_weighted(X, np.array([0, 0])) == 0.0


def test_duplicate_flags_use_exact_match_normalization():
    flags = diversity.duplicate_flags(["A dog's ball.", "a dogs ball", "A cat.", "a  CAT", "a bird"])
    assert flags.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_noun_counts_spacy():
    pytest.importorskip("spacy")
    try:
        counts, backend = diversity.noun_counts(["A man riding a horse on a beach.", ""], backend="spacy")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    assert backend.startswith("spacy:en_core_web_sm")
    assert counts.tolist() == [3.0, 0.0]


def test_bootstrap_corpus_paired_and_bias_shifted():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(300)]
    model = [" ".join(rng.choice(vocab[:30], 5)) for _ in range(200)]  # few types: low distinct-1
    base = [[" ".join(rng.choice(vocab, 5)) for _ in range(200)] for _ in range(2)]
    res = diversity.bootstrap_corpus([diversity.ngram_matrix(model, 1)], [diversity.ngram_matrix(b, 1) for b in base], n_boot=300, seed=1)
    assert res["model"]["estimate"] == pytest.approx(diversity.distinct_n(model, 1))
    assert res["d0"]["estimate"] == pytest.approx(np.mean([diversity.distinct_n(b, 1) for b in base]))
    assert res["diff"]["estimate"] == pytest.approx(res["model"]["estimate"] - res["d0"]["estimate"])
    for k in ("model", "d0", "diff"):
        r = res[k]
        assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]
    assert res["d0"]["boot_bias"] < 0  # resampling repeats captions -> fewer distinct types per token
    assert res["diff"]["ci_hi"] < 0
    with pytest.raises(ValueError, match="aligned"):
        diversity.bootstrap_corpus([diversity.ngram_matrix(model, 1)], [diversity.ngram_matrix(model[:10], 1)])
