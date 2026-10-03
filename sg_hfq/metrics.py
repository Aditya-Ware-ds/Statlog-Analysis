"""Selective-prediction metrics: risk-coverage curves, AURC, bootstrap intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np


def risk_coverage(correct: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk at every coverage k/N (k = 1..N), accepting the highest scores first.

    Ties in ``score`` are broken uniformly at random and the *expected* risk
    is returned, so a constant score yields the overall error at every
    coverage and discrete (graded) scores are not credited with an arbitrary
    within-grade order.
    """
    correct = np.asarray(correct, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n = len(correct)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    err = (~correct[order]).astype(np.float64)
    new_block = np.r_[True, s[1:] != s[:-1]]
    block = np.cumsum(new_block) - 1
    start = np.flatnonzero(new_block)
    size = np.diff(np.r_[start, n])
    block_err = np.add.reduceat(err, start)
    err_before = np.r_[0.0, np.cumsum(block_err)[:-1]]
    k = np.arange(1, n + 1)
    expected_err = err_before[block] + (k - start[block]) * block_err[block] / size[block]
    return k / n, expected_err / k


def aurc(correct: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over all coverages)."""
    return float(risk_coverage(correct, score)[1].mean())


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of an oracle that ranks every correct prediction first."""
    return aurc(correct, np.asarray(correct, dtype=np.float64))


def risk_at_coverage(correct: np.ndarray, score: np.ndarray, coverage: float) -> float:
    cov, risk = risk_coverage(correct, score)
    k = int(np.clip(round(coverage * len(cov)), 1, len(cov)))
    return float(risk[k - 1])


def coverage_at_risk(correct: np.ndarray, score: np.ndarray, max_risk: float) -> float:
    """Largest coverage whose selective risk does not exceed ``max_risk``."""
    cov, risk = risk_coverage(correct, score)
    ok = np.flatnonzero(risk <= max_risk + 1e-12)
    return float(cov[ok[-1]]) if ok.size else 0.0


def selective_summary(correct: np.ndarray, score: np.ndarray, coverages=(0.5, 0.8, 0.9)) -> dict:
    a = aurc(correct, score)
    opt = optimal_aurc(correct)
    out = {
        "accuracy_full": float(np.mean(correct)),
        "risk_full": float(1 - np.mean(correct)),
        "aurc": a,
        "e_aurc": a - opt,
        "optimal_aurc": opt,
    }
    for c in coverages:
        out[f"risk@{int(c * 100)}"] = risk_at_coverage(correct, score, c)
    return out


def bootstrap_ci(
    stat: Callable[..., float],
    arrays: tuple[np.ndarray, ...],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(*arrays)`` resampling test samples (paired across arrays)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat(*(a[idx] for a in arrays))
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def pairwise_error_rates(y_true: np.ndarray, y_pred: np.ndarray, classes) -> np.ndarray:
    """Symmetric pairwise confusion rate e_ij = (N_ij + N_ji) / (n_i + n_j)."""
    classes = list(classes)
    n = len(classes)
    E = np.zeros((n, n))
    counts = np.array([(y_true == c).sum() for c in classes])
    for i in range(n):
        for j in range(n):
            if i != j:
                N_ij = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
                N_ji = np.sum((y_true == classes[j]) & (y_pred == classes[i]))
                E[i, j] = (N_ij + N_ji) / (counts[i] + counts[j])
    return E


def routing_summary(result, y_true: np.ndarray) -> dict:
    """Operating-point metrics of a gated (hierarchical) prediction."""
    y_true = np.asarray(y_true)
    fine = result.is_fine
    root = result.out_node == 0
    coarse = ~fine & ~root
    group_ok = result.set_correct(y_true)
    sizes = np.array([len(s) for s in result.pred_sets()])
    return {
        "fine_coverage": float(fine.mean()),
        "selective_risk": float(1 - np.mean(result.fine_pred[fine] == y_true[fine])) if fine.any() else float("nan"),
        "coarse_rate": float(coarse.mean()),
        "coarse_accuracy": float(group_ok[coarse].mean()) if coarse.any() else float("nan"),
        "abstain_rate": float(root.mean()),
        "hierarchical_accuracy": float(group_ok.mean()),
        "mean_depth": float(result.depth.mean()),
        "mean_set_size": float(sizes.mean()),
    }
