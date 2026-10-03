import numpy as np
import pytest

from sg_hfq.fcm import fcm, memberships, objective, random_centres, update_centres, xie_beni


def test_memberships_rows_sum_to_one_and_match_formula():
    rng = np.random.default_rng(1)
    X, V = rng.normal(size=(50, 3)), rng.normal(size=(4, 3))
    U = memberships(X, V, m=2.0)
    assert np.allclose(U.sum(axis=1), 1.0)
    d = np.linalg.norm(X[:, None, :] - V[None], axis=2)
    expected = 1.0 / ((d[:, :, None] / d[:, None, :]) ** 2).sum(axis=2)
    assert np.allclose(U, expected)


def test_memberships_sample_on_a_centre():
    V = np.array([[0.0, 0.0], [1.0, 1.0]])
    U = memberships(np.array([[1.0, 1.0]]), V)
    assert np.allclose(U, [[0.0, 1.0]])


def test_fuzzifier_must_exceed_one():
    with pytest.raises(ValueError):
        memberships(np.zeros((2, 2)), np.ones((2, 2)), m=1.0)


def test_fcm_recovers_separated_clusters(blobs):
    X, y = blobs
    X3, y3 = X[y <= 3], y[y <= 3]
    res = fcm(X3, random_centres(X3, 3, np.random.default_rng(0)))
    assert res.converged
    labels = res.U.argmax(axis=1)
    for c in (1, 2, 3):
        assert len(np.unique(labels[y3 == c])) == 1
    truth = np.stack([X3[y3 == c].mean(axis=0) for c in (1, 2, 3)])
    assert np.min(np.linalg.norm(res.centres[:, None] - truth[None], axis=2), axis=1).max() < 0.2


def test_objective_is_non_increasing():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    V = random_centres(X, 3, rng)
    last = np.inf
    for _ in range(20):
        U = memberships(X, V)
        J = objective(X, U, V)
        assert J <= last + 1e-9
        last = J
        V = update_centres(X, U)


def test_xie_beni_hand_computed():
    X = np.array([[0.0], [2.0]])
    V = np.array([[0.0], [2.0]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert xie_beni(X, U, V) == 0.0
    U = np.full((2, 2), 0.5)
    # J = 4 * 0.25 * {0, 4, 4, 0}/... = 0.25 * 8 = 2 ; N * min sep = 2 * 4
    assert xie_beni(X, U, V) == pytest.approx(2.0 / 8.0)


def test_matches_scikit_fuzzy():
    skfuzzy = pytest.importorskip("skfuzzy")
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 1, (100, 3)), rng.normal(3, 1, (100, 3))])
    V0 = X[[0, 150]]
    res = fcm(X, V0, tol=1e-10, max_iter=2000)
    cntr, u, *_ = skfuzzy.cmeans(X.T, 2, 2.0, error=1e-12, maxiter=2000, init=memberships(X, V0).T)
    assert np.allclose(cntr, res.centres, atol=1e-6)
    assert np.allclose(u.T, res.U, atol=1e-6)
