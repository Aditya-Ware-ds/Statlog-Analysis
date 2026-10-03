"""Baselines: flat FCM (B1/B2) and supervised references (SVM, Random Forest, Gaussian ML).

The hierarchical baselines B3 (no gating) and B4 (raw entropy gating) are
configurations of :class:`sg_hfq.model.SGHFQ`; see ``experiments.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .data import SpectralRepresentation
from .fcm import fcm, memberships, random_centres, xie_beni
from .quantization import confidence


class FlatFCM:
    """Single-level FCM with one cluster per class.

    Clusters are seeded at the training class centroids (``init="class"``) or
    at random training samples (``init="random"``) and labelled by Hungarian
    matching against the training labels.
    """

    def __init__(self, representation="bandmean4", init="class", m=2.0, max_iter=1000, tol=1e-6, random_state=0,
                 n_components=10):
        self.representation = representation
        self.n_components = n_components
        self.init = init
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.rep_ = SpectralRepresentation(self.representation, self.n_components).fit(X)
        Z = self.rep_.transform(X)
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        if self.init == "class":
            V0 = np.stack([Z[y == c].mean(axis=0) for c in self.classes_])
        else:
            V0 = random_centres(Z, len(self.classes_), np.random.default_rng(self.random_state))
        res = fcm(Z, V0, m=self.m, max_iter=self.max_iter, tol=self.tol)
        rows, cols = linear_sum_assignment(-(res.U.T @ np.eye(len(self.classes_))[idx]))
        order = np.empty(len(cols), dtype=int)
        order[cols] = rows  # order[k] = cluster assigned to class k
        self.centres_ = res.centres[order]
        self.n_iter_ = res.n_iter
        self.xie_beni_ = xie_beni(Z, res.U, res.centres, self.m)
        return self

    def predict_memberships(self, X) -> np.ndarray:
        return memberships(self.rep_.transform(X), self.centres_, self.m)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_memberships(X).argmax(axis=1)]

    def scores(self, X) -> dict[str, np.ndarray]:
        U = self.predict_memberships(X)
        return {
            "pred": self.classes_[U.argmax(axis=1)],
            "max_membership": U.max(axis=1),
            "confidence": confidence(U),
        }


def _svm_with_probabilities(params: dict, random_state: int):
    """RBF SVM with libsvm probabilities (Platt scaling + pairwise coupling).

    ``SVC(probability=True)`` is deprecated from scikit-learn 1.9; it is kept
    while available because its pairwise-coupled probabilities rank samples far
    better than a per-class sigmoid on the vote-based OvR decision function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return SVC(kernel="rbf", probability=True, random_state=random_state, **params)
        except TypeError:  # parameter removed (scikit-learn >= 1.11)
            return CalibratedClassifierCV(SVC(kernel="rbf", **params), method="sigmoid", cv=5, ensemble=False)


#: RBF-SVM grid shared by every dataset (covers the StatLog CV choices and
#: the 100-200-band hyperspectral cases).
SVM_GRID = {"C": [1, 10, 100, 1000], "gamma": [0.0001, 0.001, 0.01, 0.1, 1.0]}


class SupervisedReference:
    """Supervised classifier on a Phase 1 representation; selection score = max class probability.

    SVM hyper-parameters are chosen on ``(X_val, y_val)`` when given, otherwise
    by stratified 5-fold cross-validation on the training data.
    """

    def __init__(self, kind: str, representation: str = "bandmean4", random_state: int = 0, n_components: int = 10,
                 svm_grid: dict | None = None):
        self.kind = kind
        self.representation = representation
        self.random_state = random_state
        self.n_components = n_components
        self.svm_grid = svm_grid

    def _select_svm(self, Z, y, Zv, yv) -> dict:
        grid = self.svm_grid or SVM_GRID
        if Zv is None:
            cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
            return GridSearchCV(SVC(kernel="rbf"), grid, cv=cv, n_jobs=-1).fit(Z, y).best_params_
        best, best_acc = None, -1.0
        for C in grid["C"]:
            for g in grid["gamma"]:
                acc = (SVC(kernel="rbf", C=C, gamma=g).fit(Z, y).predict(Zv) == yv).mean()
                if acc > best_acc:
                    best, best_acc = {"C": C, "gamma": g}, acc
        self.validation_accuracy_ = best_acc
        return best

    def _estimator(self, Z, y, Zv=None, yv=None):
        if self.kind == "svm":
            self.best_params_ = self._select_svm(Z, y, Zv, yv)
            return _svm_with_probabilities(self.best_params_, self.random_state)
        if self.kind == "rf":
            self.best_params_ = {"n_estimators": 500}
            return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=self.random_state)
        if self.kind == "gml":
            self.best_params_ = {"reg_param": 1e-6}
            return QuadraticDiscriminantAnalysis(reg_param=1e-6)
        raise ValueError(f"unknown supervised reference {self.kind!r}")

    def fit(self, X, y, X_val=None, y_val=None):
        self.rep_ = SpectralRepresentation(self.representation, self.n_components).fit(X)
        Z = self.rep_.transform(X)
        Zv = None if X_val is None else self.rep_.transform(X_val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.model_ = self._estimator(Z, y, Zv, y_val).fit(Z, y)
        return self

    def scores(self, X) -> dict[str, np.ndarray]:
        P = self.model_.predict_proba(self.rep_.transform(X))
        return {"pred": self.model_.classes_[P.argmax(axis=1)], "max_prob": P.max(axis=1)}
