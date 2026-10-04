import numpy as np

from dirichlet_gp.data import CompositeFeatures
from dirichlet_gp.gp import GPDirichletClassifier
from dirichlet_gp.priors import apply_ratio, choose_tau, class_frequencies
from dirichlet_gp.selection import fit_temperature, grid_search, log_loss
from dirichlet_gp.tuned import composite_spectral, mcnemar


def _spectra(rng, n_per_class=40, bands=12):
    """Two classes with different spectral shapes and random brightness."""
    t = np.linspace(0, 1, bands)
    shapes = [1 + 0.5 * t, 1.5 - 0.5 * t]
    X = np.vstack([rng.uniform(50, 150, (n_per_class, 1)) * (s + rng.normal(0, 0.05, (n_per_class, bands)))
                   for s in shapes])
    return X, np.repeat([0, 1], n_per_class)


def test_composite_features_angle_block_ignores_brightness():
    rng = np.random.default_rng(0)
    X, _ = _spectra(rng)
    f = CompositeFeatures(("z", "angle", "deriv")).fit(X)
    Z1, Z2 = f.transform(X), f.transform(3.0 * X)
    sl = f.block_slices_["angle"]
    assert f.n_features_ == 3 * 12 - 1
    assert np.allclose(Z1[:, sl], Z2[:, sl]) and not np.allclose(Z1[:, f.block_slices_["z"]], Z2[:, f.block_slices_["z"]])


def test_composite_spectral_kernel_fits():
    rng = np.random.default_rng(1)
    X, y = _spectra(rng)
    feats, kernel = composite_spectral(12, "matern52", z_groups=3, deriv_groups=2)
    m = GPDirichletClassifier(features=feats, kernel=kernel, max_iter=15, rule="lognormal").fit(X, y)
    assert len(m.theta_) == (3 + 1) + (1 + 1) + (2 + 1)
    assert (m.predict(X) == y).mean() > 0.95


def test_temperature_keeps_lognormal_decisions_and_lowers_log_loss():
    rng = np.random.default_rng(2)
    X, y = _spectra(rng, 60)
    Xv, yv = _spectra(rng, 30)
    m = GPDirichletClassifier(rule="lognormal", alpha_eps=0.3, max_iter=15).fit(X, y)
    P1 = m.predict_proba(Xv)
    t = fit_temperature(m, Xv, yv)
    P2 = m.predict_proba(Xv)
    assert t > 0 and (P1.argmax(1) == P2.argmax(1)).all()
    assert log_loss(P2, yv) <= log_loss(P1, yv) + 1e-9


def test_grid_search_refits_to_the_best_setting():
    rng = np.random.default_rng(3)
    X, y = _spectra(rng, 50)
    Xv, yv = _spectra(rng, 30)
    m = GPDirichletClassifier(rule="lognormal", max_iter=15).fit(X, y)
    best, table = grid_search(m, Xv, yv, alphas=(0.01, 0.3), scales=(0.5, 1.0))
    assert len(table) == 4 and best == max(table, key=lambda r: (r["accuracy"], -r["log_loss"]))
    assert m.alpha_eps == best["alpha_eps"]
    acc = (m.predict(Xv) == yv).mean()
    assert np.isclose(acc, best["accuracy"])


def test_prior_correction():
    classes = np.array([0, 1])
    P = np.array([[0.6, 0.4], [0.3, 0.7]])
    assert np.allclose(apply_ratio(P, np.array([1.0, 1.0])), P)
    assert np.allclose(class_frequencies(np.array([0, 0, 0, 1]), classes), [0.75, 0.25])
    # held-out samples drawn from the balanced training distribution, deployment priors strongly skewed:
    # a full correction (tau = 1) is preferred when the classifier is uninformative
    y_val = np.array([0, 1] * 50)
    P_val = np.full((100, 2), 0.5)
    tau, scores = choose_tau(P_val, y_val, classes, np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    assert tau == 1.0 and scores[1.0] < scores[0.0]


def test_mcnemar():
    a = np.array([True] * 10 + [False] * 2 + [True] * 50)
    b = np.array([False] * 10 + [True] * 2 + [True] * 50)
    n10, n01, p = mcnemar(a, b)
    assert (n10, n01) == (10, 2) and 0 < p < 0.05
    assert mcnemar(a, a)[2] == 1.0


def test_grid_search_with_deployment_priors():
    rng = np.random.default_rng(4)
    X, y = _spectra(rng, 50)
    Xv, yv = _spectra(rng, 30)
    m = GPDirichletClassifier(rule="lognormal", max_iter=10).fit(X, y)
    priors = (np.array([0.5, 0.5]), np.array([0.8, 0.2]))
    best, table = grid_search(m, Xv, yv, alphas=(0.01, 0.3), scales=(1.0,), priors=priors)
    assert {"temperature", "prior_tau", "accuracy"} <= set(best)
    assert m.prior_ratio_ is not None and np.isclose(m.temperature, best["temperature"])
    yi = np.searchsorted(m.classes_, yv)
    w = np.where(yi == 0, 0.8 / 0.5, 0.2 / 0.5)
    assert np.isclose(np.average(m.predict(Xv) == yv, weights=w), best["accuracy"])


def test_composite_features_segmented_angle():
    rng = np.random.default_rng(5)
    X = rng.uniform(100, 3000, (30, 60))
    f = CompositeFeatures(("z", "angle"), segment=10).fit(X)
    X2 = X.copy()
    X2[:, 10:20] *= 2.5  # rescale the second date only
    sl = f.block_slices_["angle"]
    assert np.allclose(f.transform(X)[:, sl], f.transform(X2)[:, sl])
    assert not np.allclose(CompositeFeatures(("angle",)).fit(X).transform(X), CompositeFeatures(("angle",)).fit(X).transform(X2))
