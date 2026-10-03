import numpy as np
import pytest

from sg_hfq.separability import (
    ambiguity_matrix,
    bhattacharyya_gaussian,
    jeffries_matusita,
    jm_matrix,
    minmax_offdiag,
    wasserstein_matrix,
)


def test_bhattacharyya_identical_is_zero():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert bhattacharyya_gaussian(np.zeros(2), S, np.zeros(2), S) == pytest.approx(0.0)


def test_bhattacharyya_univariate_closed_form():
    m1, s1, m2, s2 = 0.0, 1.0, 1.5, 2.0
    expected = 0.25 * (m1 - m2) ** 2 / (s1**2 + s2**2) + 0.5 * np.log((s1**2 + s2**2) / (2 * s1 * s2))
    got = bhattacharyya_gaussian(np.array([m1]), np.array([[s1**2]]), np.array([m2]), np.array([[s2**2]]))
    assert got == pytest.approx(expected)


def test_jm_range_and_limits():
    assert jeffries_matusita(0.0) == 0.0
    assert jeffries_matusita(50.0) == pytest.approx(2.0)
    B = np.linspace(0, 10, 20)
    assert np.all(np.diff(jeffries_matusita(B)) > 0)


def test_jm_matrix_symmetric(blobs):
    X, y = blobs
    B, JM = jm_matrix(X, y, (1, 2, 3, 4))
    assert np.allclose(JM, JM.T) and np.allclose(np.diag(JM), 0)
    assert JM[0, 1] > 1.99 and JM[2, 3] < JM[0, 1]


def test_wasserstein_of_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 2))
    X = np.vstack([a, a + [1.0, 3.0]])
    y = np.repeat([1, 2], 500)
    W = wasserstein_matrix(X, y, (1, 2))
    assert W[0, 1] == pytest.approx(2.0)  # mean of per-band shifts 1 and 3


def test_minmax_offdiag():
    M = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]], dtype=float)
    N = minmax_offdiag(M)
    assert N[0, 1] == 0 and N[1, 2] == 1 and N[0, 2] == pytest.approx(0.5)
    assert np.all(np.diag(N) == 0)


def test_ambiguity_inverts_both_distances():
    JM = np.array([[0, 2, 1], [2, 0, 0.5], [1, 0.5, 0]], dtype=float)
    W = np.array([[0, 3, 2], [3, 0, 1], [2, 1, 0]], dtype=float)
    A = ambiguity_matrix(JM, W, alpha=0.5)
    # pair (1,2) is the closest under both distances -> maximal ambiguity
    assert A[1, 2] == pytest.approx(1.0) and A[0, 1] == pytest.approx(0.0)
    assert np.allclose(A, A.T) and np.all(np.diag(A) == 1)
    with pytest.raises(ValueError):
        ambiguity_matrix(JM, W, alpha=1.5)


def test_ledoit_wolf_handles_tiny_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (4, 10)), rng.normal(3, 1, (200, 10))])
    y = np.r_[np.zeros(4, int), np.ones(200, int)]
    B, JM = jm_matrix(X, y, (0, 1), cov_estimator="ledoit_wolf")
    assert np.isfinite(B).all() and 0 < JM[0, 1] <= 2
