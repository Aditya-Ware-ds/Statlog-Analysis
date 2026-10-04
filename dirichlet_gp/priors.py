"""Class-prior correction for a shift between training and deployment class frequencies.

Saerens, Latinne & Decaestecker (2002): if a probabilistic classifier was trained
with class priors pi_train but is deployed where the priors are pi_target, its
posteriors should be re-weighted by pi_target / pi_train and renormalised. A
tempered version, ratio ** tau, hedges against pi_target being only an estimate.

Used for the Sentinel-2 task, whose training and held-out samples are capped per
class while the test department keeps its natural crop frequencies. pi_target
comes from *labelled training-side data only* (the departments used for training
and validation); tau is chosen on the held-out department by log-loss, with its
samples re-weighted to pi_target. Nothing here looks at the test department.
"""

from __future__ import annotations

import numpy as np

TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)


def class_frequencies(y: np.ndarray, classes) -> np.ndarray:
    counts = np.array([np.sum(np.asarray(y) == c) for c in classes], dtype=float)
    return counts / counts.sum()


def apply_ratio(P: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    Q = P * ratio[None, :]
    return Q / Q.sum(axis=1, keepdims=True)


def choose_tau(P_val: np.ndarray, y_val: np.ndarray, classes, pi_train: np.ndarray, pi_target: np.ndarray,
               taus=TAUS) -> tuple[float, dict]:
    """Exponent tau minimising the pi_target-weighted log-loss on the held-out samples."""
    classes = np.asarray(classes)
    yi = np.searchsorted(classes, y_val)
    pi_val = class_frequencies(y_val, classes)
    w = (pi_target / np.where(pi_val > 0, pi_val, 1.0))[yi]
    ratio = pi_target / pi_train
    scores = {}
    for tau in taus:
        Q = apply_ratio(P_val, ratio ** tau)
        nll = -np.log(np.clip(Q[np.arange(len(yi)), yi], 1e-12, None))
        scores[tau] = float(np.sum(w * nll) / np.sum(w))
    best = min(scores, key=scores.get)
    return best, scores
