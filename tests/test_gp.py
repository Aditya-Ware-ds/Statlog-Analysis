import numpy as np
import pytest
from scipy.optimize import approx_fprime

from sg_hfq.gp import (
    GPDirichletClassifier,
    GroupedKernel,
    contiguous_groups,
    dihedral_pixel_permutations,
    statlog_feature_permutations,
    statlog_groups,
)


def test_dihedral_group_is_a_group():
    perms = {tuple(p) for p in dihedral_pixel_permutations()}
    assert len(perms) == 8
    for a in perms:
        for b in perms:
            assert tuple(np.array(a)[list(b)]) in perms  # closed under composition
    assert all(p[4] == 4 for p in perms)  # centre is fixed


def test_band_orbit_groups_are_invariant():
    g = statlog_groups("band_orbit")
    assert g.max() + 1 == 12
    for p in statlog_feature_permutations():
        assert np.array_equal(g[p], g)


def test_contiguous_groups():
    g = contiguous_groups(23, 5)
    assert g.min() == 0 and g.max() == 4 and np.all(np.diff(g) >= 0)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
@pytest.mark.parametrize("invariant", [False, True])
def test_kernel_psd_symmetry_and_invariance(kind, invariant):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 36))
    perms = statlog_feature_permutations() if invariant else None
    k = GroupedKernel(statlog_groups("band_orbit"), kind, perms)
    theta = np.r_[rng.normal(0.5, 0.3, 12), 0.3]
    K = k(X, X, theta)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert np.allclose(np.diag(K), k.diag(X, theta))
    if invariant:
        for p in statlog_feature_permutations():
            assert np.allclose(k(X[:, p], X, theta), K)


@pytest.mark.parametrize("kind", ["rbf", "matern52"])
def test_lml_gradient_matches_finite_differences(kind):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 36))
    y = rng.integers(0, 3, 60)
    m = GPDirichletClassifier(groups=statlog_groups("band_orbit"), kernel=kind,
                              permutations=statlog_feature_permutations(), max_iter=1)
    m.fit(X, y)
    theta = np.r_[rng.normal(1.0, 0.2, 12), 0.5]
    f = lambda t: m._neg_lml(t, m._D_opt, m._T_opt, m._members_opt)[0]  # noqa: E731
    _, g = m._neg_lml(theta, m._D_opt, m._T_opt, m._members_opt)
    num = approx_fprime(theta, f, 1e-6)
    assert np.allclose(g, num, rtol=1e-3, atol=1e-3)


def test_classifier_separates_blobs(blobs):
    X, y = blobs
    m = GPDirichletClassifier(max_iter=30).fit(X, y)
    P = m.predict_proba(X)
    assert np.allclose(P.sum(1), 1.0)
    assert (m.predict(X) == y).mean() > 0.97
    assert m.log_marginal_likelihood() > m.log_marginal_likelihood(np.r_[np.full(4, 3.0), 0.0])


def test_woodbury_matches_direct_per_class_gp():
    """Shared-noise Woodbury algebra == C independent exact GP regressions."""
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 6))
    y = rng.integers(0, 3, 50)
    Xs = rng.normal(size=(7, 6))
    m = GPDirichletClassifier(groups=np.array([0, 0, 1, 1, 2, 2]), max_iter=5, alpha_eps=0.05).fit(X, y)
    Z, Zs = m.rep_.transform(X), m.rep_.transform(Xs)
    K = m.kernel_(Z, Z, m.theta_)
    Ks = m.kernel_(Zs, Z, m.theta_)
    T, S = m._targets(y)
    lml = 0.0
    mu, var = m.predict_latent(Xs)
    for c in range(3):
        A = K + np.diag(S[:, c] + m.jitter)
        r = T[:, c] - m.means_[c]
        lml += multivariate_normal(np.zeros(len(r)), A).logpdf(r)
        a = np.linalg.solve(A, r)
        assert np.allclose(mu[:, c], m.means_[c] + Ks @ a)
        v = m.kernel_.diag(Zs, m.theta_) - np.einsum("ij,ji->i", Ks, np.linalg.solve(A, Ks.T))
        assert np.allclose(var[:, c], v)
    # the optimisation subset is the whole training set here
    assert np.isclose(m.log_marginal_likelihood(), lml)
