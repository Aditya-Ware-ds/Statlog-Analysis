"""Fuzzy C-Means (Bezdek, 1981) with explicit centre seeding, and the Xie-Beni index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def squared_distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(N, c) matrix of squared Euclidean distances between samples and centres."""
    d2 = (X * X).sum(1)[:, None] - 2.0 * X @ V.T + (V * V).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def memberships(X: np.ndarray, V: np.ndarray, m: float = 2.0) -> np.ndarray:
    """FCM membership update u_ij = 1 / sum_k (||x_i - v_j|| / ||x_i - v_k||)^(2/(m-1)).

    Evaluated in log space for stability. A sample that coincides with one or
    more centres gets its membership split equally among them.
    """
    if m <= 1.0:
        raise ValueError("fuzzifier m must be > 1")
    X = np.asarray(X, dtype=np.float64)
    d2 = squared_distances(X, np.asarray(V, dtype=np.float64))
    zero = d2 <= 1e-300
    logw = -np.log(np.where(zero, 1.0, d2)) / (m - 1.0)
    logw -= logw.max(axis=1, keepdims=True)
    U = np.exp(logw)
    U /= U.sum(axis=1, keepdims=True)
    hit = zero.any(axis=1)
    if hit.any():
        U[hit] = zero[hit] / zero[hit].sum(axis=1, keepdims=True)
    return U


def update_centres(X: np.ndarray, U: np.ndarray, m: float = 2.0) -> np.ndarray:
    """v_j = sum_i u_ij^m x_i / sum_i u_ij^m."""
    Um = U**m
    return (Um.T @ X) / Um.sum(axis=0)[:, None]


def objective(X: np.ndarray, U: np.ndarray, V: np.ndarray, m: float = 2.0) -> float:
    """J_m = sum_i sum_j u_ij^m ||x_i - v_j||^2."""
    return float(((U**m) * squared_distances(X, V)).sum())


def xie_beni(X: np.ndarray, U: np.ndarray, V: np.ndarray, m: float = 2.0) -> float:
    """Xie-Beni validity index: compactness / (N * min separation). Lower is better."""
    sep = squared_distances(V, V)
    np.fill_diagonal(sep, np.inf)
    return objective(X, U, V, m) / (len(X) * sep.min())


@dataclass
class FCMResult:
    centres: np.ndarray
    U: np.ndarray
    n_iter: int
    converged: bool
    objective: float
    initial_centres: np.ndarray


def fcm(
    X: np.ndarray,
    init_centres: np.ndarray,
    m: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> FCMResult:
    """Run FCM from the given initial centres until the largest centre shift is < ``tol``."""
    X = np.asarray(X, dtype=np.float64)
    V0 = np.array(init_centres, dtype=np.float64)
    V = V0.copy()
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        U = memberships(X, V, m)
        V_new = update_centres(X, U, m)
        shift = np.abs(V_new - V).max()
        V = V_new
        if shift < tol:
            converged = True
            break
    U = memberships(X, V, m)
    return FCMResult(V, U, n_iter, converged, objective(X, U, V, m), V0)


def random_centres(X: np.ndarray, c: int, rng: np.random.Generator) -> np.ndarray:
    """Random initialisation: ``c`` distinct training samples as starting centres."""
    idx = rng.choice(len(X), size=c, replace=False)
    return np.asarray(X, dtype=np.float64)[idx].copy()
