"""Phase 2: class-level spectral relationships.

* Jeffries-Matusita (JM) separability from the Gaussian Bhattacharyya distance
  (Eqs. 2-3).
* Band-wise 1-D Wasserstein (optimal-transport) distance averaged over bands
  (Eqs. 4-5).
* The fused spectral-ambiguity matrix (Eqs. 6-7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance


def class_statistics(X: np.ndarray, y: np.ndarray, classes) -> tuple[np.ndarray, np.ndarray]:
    """Per-class mean vectors (C, D) and covariance matrices (C, D, D)."""
    X = np.asarray(X, dtype=np.float64)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    covs = np.stack([np.atleast_2d(np.cov(X[y == c], rowvar=False)) for c in classes])
    return means, covs


def _regularise(S: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        return S
    d = S.shape[0]
    return S + ridge * (np.trace(S) / d) * np.eye(d)


def bhattacharyya_gaussian(mu_i, S_i, mu_j, S_j, ridge: float = 0.0) -> float:
    """Bhattacharyya distance between two Gaussians (Eq. 2), with Sigma = (S_i + S_j) / 2."""
    S_i = _regularise(np.atleast_2d(S_i), ridge)
    S_j = _regularise(np.atleast_2d(S_j), ridge)
    S = 0.5 * (S_i + S_j)
    diff = np.atleast_1d(mu_i - mu_j)
    term_mean = diff @ np.linalg.solve(S, diff) / 8.0
    _, logdet = np.linalg.slogdet(S)
    _, logdet_i = np.linalg.slogdet(S_i)
    _, logdet_j = np.linalg.slogdet(S_j)
    term_cov = 0.5 * (logdet - 0.5 * (logdet_i + logdet_j))
    return float(term_mean + term_cov)


def jeffries_matusita(B: np.ndarray | float) -> np.ndarray | float:
    """JM = 2 (1 - exp(-B)) in [0, 2] (Eq. 3)."""
    return 2.0 * (1.0 - np.exp(-np.asarray(B)))


def jm_matrix(X, y, classes, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Bhattacharyya and JM matrices over ``classes`` (diagonal = 0)."""
    means, covs = class_statistics(X, y, classes)
    n = len(classes)
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = B[j, i] = bhattacharyya_gaussian(means[i], covs[i], means[j], covs[j], ridge)
    return B, jeffries_matusita(B)


def wasserstein_matrix(X, y, classes) -> np.ndarray:
    """Band-averaged 1-D Wasserstein-1 distance between class distributions (Eqs. 4-5)."""
    X = np.asarray(X, dtype=np.float64)
    n, n_bands = len(classes), X.shape[1]
    W = np.zeros((n, n))
    for i in range(n):
        Xi = X[y == classes[i]]
        for j in range(i + 1, n):
            Xj = X[y == classes[j]]
            W[i, j] = W[j, i] = np.mean(
                [wasserstein_distance(Xi[:, b], Xj[:, b]) for b in range(n_bands)]
            )
    return W


def minmax_offdiag(M: np.ndarray) -> np.ndarray:
    """Min-max normalise the off-diagonal entries of a symmetric matrix to [0, 1]."""
    M = np.asarray(M, dtype=np.float64)
    off = ~np.eye(len(M), dtype=bool)
    lo, hi = M[off].min(), M[off].max()
    out = np.zeros_like(M) if hi == lo else (M - lo) / (hi - lo)
    np.fill_diagonal(out, 0.0)
    return out


def ambiguity_matrix(JM: np.ndarray, W: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Fuse JM and Wasserstein into the spectral-ambiguity matrix A (Eqs. 6-7).

    Both inputs are *distances* (large = spectrally distinct), so both are
    min-max normalised and inverted before the convex blend; high A_ij then
    means classes i and j are spectrally ambiguous. (Eq. 6 writes the inversion
    only for JM; applying the blend to an un-inverted W would mix a similarity
    with a distance and contradict "high A_ij indicates spectral ambiguity", so
    W is inverted as well.) The diagonal is set to 1.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    A = alpha * (1.0 - minmax_offdiag(JM)) + (1.0 - alpha) * (1.0 - minmax_offdiag(W))
    np.fill_diagonal(A, 1.0)
    return A


@dataclass
class SpectralRelations:
    classes: tuple[int, ...]
    bhattacharyya: np.ndarray
    jm: np.ndarray
    wasserstein: np.ndarray
    ambiguity: np.ndarray
    alpha: float

    def pairs(self):
        """Yield (i, j) index pairs of the upper triangle."""
        n = len(self.classes)
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j


def spectral_relations(X, y, classes, alpha: float = 0.5, ridge: float = 0.0) -> SpectralRelations:
    B, JM = jm_matrix(X, y, classes, ridge=ridge)
    W = wasserstein_matrix(X, y, classes)
    return SpectralRelations(tuple(classes), B, JM, W, ambiguity_matrix(JM, W, alpha), alpha)
