"""Evaluation metrics: classification accuracy, selective prediction (risk-coverage) and calibration."""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ classification
def classification_metrics(y: np.ndarray, pred: np.ndarray, classes) -> tuple[dict, dict]:
    """Overall accuracy, average per-class accuracy, Cohen's kappa and macro F1, plus per-class recall."""
    y, pred = np.asarray(y), np.asarray(pred)
    present = [c for c in classes if np.any(y == c)]
    recall = {c: float(np.mean(pred[y == c] == c)) for c in present}
    oa = float(np.mean(pred == y))
    p_exp = sum(np.mean(y == c) * np.mean(pred == c) for c in set(np.unique(y)) | set(np.unique(pred)))
    kappa = float((oa - p_exp) / (1 - p_exp)) if p_exp < 1 else 1.0
    f1 = []
    for c in present:
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        f1.append(2 * tp / denom if denom else 0.0)
    out = {"OA": oa, "AA": float(np.mean(list(recall.values()))), "kappa": kappa, "macro_F1": float(np.mean(f1))}
    return out, recall


def calibration(P: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> tuple[float, float]:
    """Negative log-likelihood and expected calibration error (top-label, equal-width bins)."""
    nll = float(-np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, None))))
    conf, pred = P.max(1), P.argmax(1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(np.mean(pred[sel] == y_idx[sel]) - conf[sel].mean())
    return nll, float(ece)


# ------------------------------------------------------- selective prediction
def risk_coverage(correct: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk at every coverage k/N (k = 1..N), accepting the highest scores first.

    Ties in ``score`` are broken uniformly at random and the *expected* risk
    is returned, so a constant score yields the overall error at every coverage.
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


def selective_summary(correct: np.ndarray, score: np.ndarray, coverages=(0.5, 0.8, 0.9)) -> dict:
    a = aurc(correct, score)
    out = {"aurc": a, "e_aurc": a - optimal_aurc(correct)}
    for c in coverages:
        out[f"risk@{int(c * 100)}"] = risk_at_coverage(correct, score, c)
    return out
