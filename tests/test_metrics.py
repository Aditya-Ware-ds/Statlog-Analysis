import numpy as np
import pytest

from sg_hfq.metrics import aurc, optimal_aurc, pairwise_error_rates, risk_at_coverage, risk_coverage


def test_risk_coverage_simple():
    correct = np.array([True, True, False, True])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    cov, risk = risk_coverage(correct, score)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk == pytest.approx([0, 0, 1 / 3, 1 / 4])
    assert aurc(correct, score) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4]))


def test_constant_score_gives_flat_curve():
    rng = np.random.default_rng(0)
    correct = rng.uniform(size=300) > 0.3
    _, risk = risk_coverage(correct, np.zeros(300))
    assert np.allclose(risk, 1 - correct.mean())


def test_ties_use_expected_risk():
    correct = np.array([True, False, True, True])
    score = np.array([1.0, 1.0, 0.0, 0.0])
    _, risk = risk_coverage(correct, score)
    assert risk[0] == pytest.approx(0.5) and risk[1] == pytest.approx(0.5)


def test_optimal_aurc_is_a_lower_bound():
    rng = np.random.default_rng(1)
    correct = rng.uniform(size=500) > 0.25
    assert optimal_aurc(correct) <= aurc(correct, rng.uniform(size=500))
    assert risk_at_coverage(correct, correct.astype(float), 0.5) == 0.0


def test_pairwise_error_rates():
    y = np.array([1, 1, 2, 2, 3])
    p = np.array([1, 2, 2, 1, 3])
    E = pairwise_error_rates(y, p, (1, 2, 3))
    assert E[0, 1] == pytest.approx(2 / 4) and E[0, 2] == 0 and np.allclose(E, E.T)
