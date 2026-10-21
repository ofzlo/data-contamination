import numpy as np
import pytest

from contam.eval.association import delta, delta_from_means, gamma, gamma_from_arrays


def test_delta_hand_computed():
    # seen: model 3,5 (mean 4), base 1,1 (mean 1) -> +3 ; unseen: model 2,4 (3), base 1,3 (2) -> +1
    d = delta(np.array([3.0, 5.0]), np.array([1.0, 1.0]), np.array([2.0, 4.0]), np.array([1.0, 3.0]))
    assert d == pytest.approx(2.0)
    assert delta_from_means(4, 1, 3, 2) == pytest.approx(2.0)


def test_delta_zero_when_uniform_gain():
    base_s, base_u = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5])
    assert delta(base_s + 7, base_s, base_u + 7, base_u) == pytest.approx(0.0)


def test_delta_requires_alignment():
    with pytest.raises(ValueError):
        delta(np.ones(3), np.ones(2), np.ones(2), np.ones(2))


def test_gamma_hand_computed():
    assert gamma(5.0, 1.0, 2.5) == pytest.approx(1.5)
    bs, bu = np.zeros(2), np.zeros(2)
    pair = (np.array([6.0, 6.0]), np.array([1.0, 1.0]))  # Δ_pair = 5
    image = (np.array([2.0, 2.0]), np.array([1.0, 1.0]))  # Δ_image = 1
    cap = (np.array([3.0, 3.0]), np.array([0.5, 0.5]))  # Δ_caption = 2.5
    assert gamma_from_arrays(pair, image, cap, bs, bu) == pytest.approx(1.5)
