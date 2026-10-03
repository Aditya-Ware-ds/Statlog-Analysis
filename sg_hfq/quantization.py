"""Phase 5: fuzzy entropy, confidence and quantile-based confidence grades."""

from __future__ import annotations

import numpy as np


def fuzzy_entropy(U: np.ndarray) -> np.ndarray:
    """H_i = -sum_j u_ij log u_ij (natural log, 0 log 0 = 0)."""
    U = np.asarray(U, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(U > 0, U * np.log(U), 0.0)
    return -terms.sum(axis=1)


def normalized_entropy(U: np.ndarray) -> np.ndarray:
    """H_i / log K in [0, 1]."""
    K = np.asarray(U).shape[1]
    if K < 2:
        raise ValueError("normalised entropy needs at least two clusters")
    return np.clip(fuzzy_entropy(U) / np.log(K), 0.0, 1.0)


def confidence(U: np.ndarray) -> np.ndarray:
    """C_i = 1 - H_i^norm in [0, 1]."""
    return 1.0 - normalized_entropy(U)


class QuantileQuantizer:
    """Map confidences to L grades with thresholds q_l = quantile(C_train, l / L) (Eq. 8).

    Grade 1 holds C <= q_1, grade l holds q_{l-1} < C <= q_l and grade L holds
    C > q_{L-1}. ``cdf`` gives the continuous counterpart: the (mid-rank)
    empirical CDF of the training confidences, so that grade ~ 1 + floor(L * cdf).
    """

    def __init__(self, n_grades: int = 5):
        if n_grades < 2:
            raise ValueError("need at least two grades")
        self.n_grades = n_grades

    def fit(self, C_train: np.ndarray) -> "QuantileQuantizer":
        C_train = np.asarray(C_train, dtype=np.float64)
        L = self.n_grades
        self.thresholds_ = np.quantile(C_train, np.arange(1, L) / L)
        self.reference_ = np.sort(C_train)
        return self

    def transform(self, C: np.ndarray) -> np.ndarray:
        return 1 + np.searchsorted(self.thresholds_, np.asarray(C), side="left")

    def cdf(self, C: np.ndarray) -> np.ndarray:
        ref = self.reference_
        C = np.asarray(C)
        lo = np.searchsorted(ref, C, side="left")
        hi = np.searchsorted(ref, C, side="right")
        return (lo + hi) / (2.0 * len(ref))
