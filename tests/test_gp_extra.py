import numpy as np

from dirichlet_gp.gp import GPDirichletClassifier, GroupedKernel, SumKernel, statlog_feature_permutations, statlog_groups


def test_uncached_matches_cached():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 36))
    k = GroupedKernel(statlog_groups("band_orbit"), "rbf", statlog_feature_permutations())
    theta = np.r_[rng.normal(0.8, 0.2, 12), 0.2]
    K1, g1 = k.gram_and_grad_fn(k.precompute(X, True), theta)
    K2, g2 = k.gram_and_grad_fn(k.precompute(X, False), theta)
    W = rng.normal(size=(50, 50))
    W = W + W.T
    assert np.allclose(K1, K2, rtol=1e-6) and np.allclose(K1, k(X, X, theta), rtol=1e-6)
    assert np.allclose(g1(W), g2(W), rtol=1e-4)


def test_sum_kernel_and_ignored_features():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 36))
    centre = np.full(36, -1)
    centre[16:20] = np.arange(4)
    k2 = GroupedKernel(centre, "rbf")
    assert k2.n_groups == 4 and k2.n_params == 5
    X2 = X.copy()
    X2[:, :16] += 5.0  # changing unused features does not change k2
    th = np.r_[np.zeros(4), 0.0]
    assert np.allclose(k2(X, X, th), k2(X2, X2, th))
    s = SumKernel(GroupedKernel(statlog_groups("band_orbit"), "rbf", statlog_feature_permutations()), k2)
    theta = np.r_[np.zeros(12), 0.1, np.zeros(4), -0.3]
    K, grad = s.gram_and_grad_fn(s.precompute(X), theta)
    assert np.allclose(K, s(X, X, theta)) and np.linalg.eigvalsh(K).min() > -1e-8
    assert grad(np.eye(40)).shape == (18,)


def test_classifier_with_sum_kernel_and_warm_start():
    rng = np.random.default_rng(2)
    X = np.abs(rng.normal(50, 10, size=(80, 36)))
    y = (X[:, 16] > 50).astype(int)
    centre = np.full(36, -1)
    centre[16:20] = np.arange(4)
    s = SumKernel(GroupedKernel(statlog_groups("band"), "rbf"), GroupedKernel(centre, "rbf"))
    m = GPDirichletClassifier(representation="log36", kernel=s, max_iter=20).fit(X, y)
    assert len(m.theta_) == 10 and (m.predict(X) == y).mean() > 0.9
    m2 = GPDirichletClassifier(representation="log36", kernel=s, max_iter=5, theta_init=m.theta_).fit(X, y)
    assert m2.log_marginal_likelihood() >= m.log_marginal_likelihood() - 1e-6


def test_linear_kernel_in_sum_gradient_and_refit():
    from scipy.optimize import approx_fprime

    from dirichlet_gp.gp import LinearKernel

    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 12))
    y = rng.integers(0, 3, 50)
    groups = np.arange(12) % 3
    s = SumKernel(GroupedKernel(groups, "rbf"), LinearKernel(groups))
    assert s.n_params == 4 + 3 and s.lengthscale_mask().tolist() == [True] * 3 + [False] * 4
    for cache in (True, False):
        m = GPDirichletClassifier(kernel=s, max_iter=1, cache=cache).fit(X, y)
        theta = np.r_[rng.normal(0.5, 0.2, 3), 0.3, rng.normal(-2, 0.3, 3)]
        K, _ = s.gram_and_grad_fn(m._D_opt, theta)
        assert np.allclose(K, s(m.Z_, m.Z_, theta), rtol=1e-5, atol=1e-5)
        assert np.allclose(np.diag(K), s.diag(m.Z_, theta), rtol=1e-5)
        f = lambda t: m._neg_lml(t, m._D_opt, m._T_opt, m._members_opt)[0]  # noqa: E731
        _, g = m._neg_lml(theta, m._D_opt, m._T_opt, m._members_opt)
        assert np.allclose(g, approx_fprime(theta, f, 1e-6), rtol=1e-3, atol=1e-3)
    m.refit(lengthscale_scale=2.0)
    assert np.allclose(m.theta_ - m.theta_ml_, np.log(2.0) * s.lengthscale_mask())
