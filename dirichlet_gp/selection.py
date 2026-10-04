"""Validation-based selection for the GP classifier (used only where a held-out split exists).

After the kernel hyper-parameters have been learned by type-II maximum likelihood
(at alpha_eps = 0.01), two quantities that the marginal likelihood cannot set are
chosen on the held-out samples, by the same criterion as the SVM's (C, gamma) grid:

* ``alpha_eps`` (the Dirichlet pseudo-count of the non-observed classes): it fixes
  the size of the regression targets relative to their noise, i.e. the degree of
  smoothing, much as C does for an SVM;
* a common multiplier ``s`` of all learned length-scales (s = 1 keeps them), much
  as gamma does.

The pair with the highest held-out accuracy wins (ties: lower held-out log-loss).
A temperature is then fitted on the same samples by log-loss; with the
``lognormal`` rule it changes the probabilities but never the predicted class.

When the deployment class priors differ from the training priors (Sentinel-2),
every grid point is scored after the complete post-processing -- temperature,
then the prior correction of ``dirichlet_gp.priors`` with its exponent chosen on
the held-out samples -- and the held-out samples are weighted to the deployment
priors, so that the criterion estimates accuracy where the model is used.

StatLog has no held-out split, and cross-validation on it would leak pixels between
overlapping 3x3 neighbourhoods, so none of this is used there.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .gp import GPDirichletClassifier
from .priors import apply_ratio, choose_tau, class_frequencies

ALPHAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SCALES = (0.35, 0.5, 0.71, 1.0, 1.41, 2.0)


def log_loss(P: np.ndarray, yi: np.ndarray, weights: np.ndarray | None = None) -> float:
    nll = -np.log(np.clip(P[np.arange(len(yi)), yi], 1e-12, None))
    return float(np.average(nll, weights=weights))


def _temperature(gp: GPDirichletClassifier, mu, var, yi, bounds=(0.05, 20.0)) -> float:
    def loss(log_t):
        gp.temperature = float(np.exp(log_t))
        return log_loss(gp.probabilities_from_latent(mu, var), yi)

    res = minimize_scalar(loss, bounds=np.log(bounds), method="bounded", options={"xatol": 1e-3})
    gp.temperature = float(np.exp(res.x))
    return gp.temperature


def fit_temperature(gp: GPDirichletClassifier, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Set ``gp.temperature`` to minimise the held-out log-loss; returns it."""
    if gp.prior_ratio_ is not None:
        raise ValueError("fit the temperature before setting a prior ratio")
    mu, var = gp.predict_latent(X_val)
    return _temperature(gp, mu, var, np.searchsorted(gp.classes_, y_val))


def postprocess(gp: GPDirichletClassifier, X_val: np.ndarray, y_val: np.ndarray, priors=None) -> dict:
    """Fit the temperature and, with ``priors = (pi_train, pi_target)``, the prior correction; score the result.

    Leaves ``gp.temperature`` and ``gp.prior_ratio_`` set. The returned accuracy and log-loss weight the
    held-out samples to the deployment priors (no weighting without ``priors``).
    """
    gp.prior_ratio_ = None
    yi = np.searchsorted(gp.classes_, y_val)
    mu, var = gp.predict_latent(X_val)
    out = {"temperature": _temperature(gp, mu, var, yi)}
    P = gp.probabilities_from_latent(mu, var)
    w = None
    if priors is not None:
        pi_train, pi_target = priors
        tau, scores = choose_tau(P, y_val, gp.classes_, pi_train, pi_target)
        ratio = (pi_target / pi_train) ** tau
        P = apply_ratio(P, ratio)
        gp.prior_ratio_ = ratio
        w = (pi_target / class_frequencies(y_val, gp.classes_))[yi]
        out.update(prior_tau=tau, prior_tau_scores=scores)
    out.update(accuracy=float(np.average(P.argmax(axis=1) == yi, weights=w)), log_loss=log_loss(P, yi, w))
    return out


def grid_search(gp: GPDirichletClassifier, X_val: np.ndarray, y_val: np.ndarray,
                alphas=ALPHAS, scales=SCALES, priors=None) -> tuple[dict, list[dict]]:
    """Refit ``gp`` at every (alpha_eps, s) pair and keep the best on held-out accuracy.

    Every point is scored after :func:`postprocess` (temperature; prior correction if ``priors``).
    ``gp`` is left at the best point, post-processed. Returns (best row, all rows).
    """
    table = []
    for a in alphas:
        for s in scales:
            gp.refit(alpha_eps=a, lengthscale_scale=s)
            table.append({"alpha_eps": a, "scale": s, **postprocess(gp, X_val, y_val, priors)})
    best = max(table, key=lambda r: (r["accuracy"], -r["log_loss"]))
    gp.refit(alpha_eps=best["alpha_eps"], lengthscale_scale=best["scale"])
    postprocess(gp, X_val, y_val, priors)
    return best, table
