import numpy as np
import pytest

from dirichlet_gp.metrics import aurc, calibration, classification_metrics, optimal_aurc, risk_at_coverage, risk_coverage


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


def test_classification_metrics_hand_computed():
    y = np.array([1, 1, 1, 1, 2, 2])
    p = np.array([1, 1, 1, 2, 2, 1])
    m, recall = classification_metrics(y, p, (1, 2))
    assert m["OA"] == pytest.approx(4 / 6)
    assert recall == {1: 0.75, 2: 0.5} and m["AA"] == pytest.approx(0.625)
    # p_o = 4/6, p_e = (4/6)(4/6) + (2/6)(2/6) = 20/36
    assert m["kappa"] == pytest.approx((4 / 6 - 20 / 36) / (1 - 20 / 36))
    # F1(1) = 2*3/(4+4) = 0.75, F1(2) = 2*1/(2+2) = 0.5
    assert m["macro_F1"] == pytest.approx(0.625)


def test_calibration():
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    nll, ece = calibration(P, np.array([0, 1, 1]))
    assert nll == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4])))
    assert 0 <= ece <= 1
