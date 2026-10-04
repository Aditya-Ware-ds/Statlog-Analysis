"""GP v2 on your own data: fit, select and calibrate the validation-tuned Dirichlet GP in one call.

Python:

    from dirichlet_gp.gpv2 import fit_gp_v2

    gp = fit_gp_v2(X_train, y_train, X_val, y_val, kernel="spectral")
    labels = gp.predict(X_test)              # class labels
    proba = gp.predict_proba(X_test)         # columns in the order of gp.classes_
    print(gp.selection_)                     # alpha_eps, length-scale multiplier, temperature, ...

Command line (CSV files, one row per sample, one column per feature plus a label column):

    python -m dirichlet_gp.gpv2 --train train.csv --val val.csv --predict new.csv --out predictions.csv

The steps are those of the experiments in ``dirichlet_gp.tuned`` (results/tuned/PROTOCOL.md):

1. kernel hyper-parameters by type-II maximum likelihood on all training samples (alpha_eps = 0.01);
2. with a validation set: alpha_eps and a length-scale multiplier chosen by validation accuracy,
   then a temperature fitted by validation log-loss (``dirichlet_gp.selection``);
3. optionally, a correction towards the class frequencies expected where the model is used
   (``dirichlet_gp.priors``), with its strength chosen on the validation set;
4. predictions with the deterministic log-normal rule.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from .data import CompositeFeatures, Standardizer
from .gp import GPDirichletClassifier, GroupedKernel, LinearKernel, SumKernel, contiguous_groups
from .priors import class_frequencies
from .selection import ALPHAS, SCALES, grid_search

#: kernel choices of :func:`fit_gp_v2`
KERNELS = ("spectral", "series", "ard")
#: length-scale multipliers for kernels with a linear term (2^(k/2), k = -3..8, i.e. 0.35 ... 16)
WIDE_SCALES = tuple(float(x) for x in np.round(2.0 ** (np.arange(-3, 9) / 2), 3))


# ------------------------------------------------------------------ kernels
def composite_spectral(n_bands: int, family: str = "matern52", z_groups: int = 10, deriv_groups: int = 5):
    """Features and sum kernel on [z-scored spectrum | unit-norm spectrum (angle) | smoothed first derivative].

    For one spectrum per sample with the bands in wavelength order (hyperspectral or multispectral).
    The z block has ``z_groups`` contiguous length-scale groups, the angle block one length-scale,
    the derivative block ``deriv_groups`` contiguous groups; each term has its own signal variance.
    """
    if n_bands < 3:
        raise ValueError("the spectral kernel needs at least 3 bands")
    feats = CompositeFeatures(("z", "angle", "deriv"))
    n = 3 * n_bands - 1
    blocks = {"z": (0, n_bands), "angle": (n_bands, 2 * n_bands), "deriv": (2 * n_bands, n)}
    within = {"z": contiguous_groups(n_bands, z_groups), "angle": np.zeros(n_bands, int),
              "deriv": contiguous_groups(n_bands - 1, deriv_groups)}
    kernels = []
    for b, (lo, hi) in blocks.items():
        g = np.full(n, -1)
        g[lo:hi] = within[b]
        kernels.append(GroupedKernel(g, family))
    return feats, SumKernel(*kernels)


def band_series_kernel(n_bands: int, n_periods: int, angle: bool = True, linear: bool = True):
    """Features and sum kernel for a band x date series stored date by date (all bands of date 1, then date 2, ...).

    RBF on the z-scored values with one length-scale per band (shared by all dates), plus an RBF on the
    unit-norm series (spectral-angle block, one length-scale) and a linear kernel with one weight per band.
    ``band_series_kernel(10, 6)`` is the Sentinel-2 kernel of the experiments.
    """
    d = n_bands * n_periods
    band = np.arange(d) % n_bands
    if not angle:
        kernels = [GroupedKernel(band, "rbf")] + ([LinearKernel(band)] if linear else [])
        return Standardizer("standard"), (SumKernel(*kernels) if len(kernels) > 1 else kernels[0])
    z = np.r_[band, np.full(d, -1)]
    ang = np.r_[np.full(d, -1), np.zeros(d, int)]
    kernels = [GroupedKernel(z, "rbf"), GroupedKernel(ang, "rbf")] + ([LinearKernel(z)] if linear else [])
    return CompositeFeatures(("z", "angle")), SumKernel(*kernels)


def make_gp_v2(n_features: int, kernel: str = "spectral", n_periods: int | None = None, max_iter: int = 200,
               random_state: int = 0) -> GPDirichletClassifier:
    """An unfitted GP v2 classifier for ``n_features`` input columns.

    ``kernel``: ``"spectral"`` (one spectrum per sample, bands in wavelength order), ``"series"`` (a band x date
    series, date by date; give ``n_periods``) or ``"ard"`` (any other numeric features: Matern-5/2 with one
    length-scale per feature).
    """
    common = {"alpha_eps": 0.01, "rule": "lognormal", "n_opt": 10**9, "max_iter": max_iter,
              "random_state": random_state}
    if kernel == "spectral":
        feats, k = composite_spectral(n_features, "matern52")
        return GPDirichletClassifier(features=feats, kernel=k, **common)
    if kernel == "series":
        if not n_periods or n_features % n_periods:
            raise ValueError(f"kernel='series' needs n_periods dividing the {n_features} features")
        feats, k = band_series_kernel(n_features // n_periods, n_periods)
        return GPDirichletClassifier(features=feats, kernel=k, **common)
    if kernel == "ard":
        return GPDirichletClassifier(representation="standard", groups=None, kernel="matern52", **common)
    raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")


# ------------------------------------------------------------------ fitting
def _prior_vector(target_priors, classes) -> np.ndarray:
    if isinstance(target_priors, dict):
        missing = [c for c in classes if c not in target_priors]
        if missing:
            raise ValueError(f"target_priors has no entry for classes {missing}")
        p = np.array([float(target_priors[c]) for c in classes])
    else:
        p = np.asarray(target_priors, dtype=float)
        if p.shape != (len(classes),):
            raise ValueError(f"target_priors needs {len(classes)} values, in the order of the sorted classes")
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("target_priors must be non-negative and not all zero")
    return p / p.sum()


def _check(X, y, name):
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"{name}: X must be (n_samples, n_features) with one label per row")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: X contains NaN or infinite values")


def fit_gp_v2(X_train, y_train, X_val=None, y_val=None, *, kernel: str = "spectral", n_periods: int | None = None,
              target_priors=None, alphas=ALPHAS, scales=None, max_iter: int = 200, random_state: int = 0,
              verbose: bool = True) -> GPDirichletClassifier:
    """Fit GP v2 and return the fitted classifier (use ``predict`` / ``predict_proba``).

    X_train, y_train : training spectra (n, d) and labels (any sortable type: ints or strings).
    X_val, y_val : validation samples that choose alpha_eps, the length-scale multiplier and the temperature.
        They should resemble where the model will be used and must not overlap the training samples
        (for example, not pixels of the same windows). Without them nothing is tuned: alpha_eps = 0.01,
        the learned length-scales and temperature 1 are kept.
    kernel, n_periods : see :func:`make_gp_v2`.
    target_priors : class frequencies expected where the model is used, as {class: frequency} or a
        sequence in the order of the sorted classes. Give it when they differ from the training sample's
        (for example when training was capped per class). The correction's strength is chosen on the
        validation set, weighted to these frequencies; without a validation set the full correction is used.
    alphas, scales : the selection grid (default: 0.003 ... 1, and 0.35 ... 2, or 0.35 ... 16 with "series").

    The result also carries ``selection_`` (chosen settings and validation scores), ``selection_table_``
    (every grid point) and ``log_evidence_`` (log marginal likelihood at the learned hyper-parameters).
    """
    X_train, y_train = np.asarray(X_train, dtype=float), np.asarray(y_train)
    _check(X_train, y_train, "training set")
    say = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    gp = make_gp_v2(X_train.shape[1], kernel, n_periods, max_iter, random_state)
    t0 = time.time()
    say(f"[gp v2] fitting the kernel on {len(y_train)} training samples, {X_train.shape[1]} features ...")
    gp.fit(X_train, y_train)
    gp.log_evidence_ = gp.log_marginal_likelihood(gp.theta_ml_)
    gp.release_training_cache()
    say(f"[gp v2] kernel fitted: {gp.opt_result_.nit} iterations, {time.time() - t0:.0f}s, "
        f"log evidence {gp.log_evidence_:.1f}")
    priors = None
    if target_priors is not None:
        priors = (class_frequencies(y_train, gp.classes_), _prior_vector(target_priors, gp.classes_))
    if X_val is None:
        gp.selection_ = {"alpha_eps": gp.alpha_eps, "scale": 1.0, "temperature": 1.0}
        if priors is not None:
            gp.prior_ratio_ = priors[1] / priors[0]
            gp.selection_["prior_tau"] = 1.0
        gp.selection_table_ = []
        say("[gp v2] no validation set: alpha_eps, length-scales and temperature are not tuned")
        return gp
    X_val, y_val = np.asarray(X_val, dtype=float), np.asarray(y_val)
    _check(X_val, y_val, "validation set")
    unknown = sorted(set(np.unique(y_val).tolist()) - set(gp.classes_.tolist()))
    if unknown:
        raise ValueError(f"validation labels {unknown} do not occur in the training set")
    grid_scales = scales if scales is not None else (WIDE_SCALES if kernel == "series" else SCALES)
    say(f"[gp v2] selecting on {len(y_val)} validation samples ({len(alphas) * len(grid_scales)} settings) ...")
    best, table = grid_search(gp, X_val, y_val, alphas=alphas, scales=grid_scales, priors=priors)
    gp.selection_, gp.selection_table_ = best, table
    say(f"[gp v2] chosen: alpha_eps {best['alpha_eps']}, length-scale x {best['scale']}, temperature "
        f"{best['temperature']:.2f}" + (f", prior exponent {best['prior_tau']}" if priors is not None else "")
        + f"; validation accuracy {100 * best['accuracy']:.2f}% ({time.time() - t0:.0f}s in total)")
    return gp


# ------------------------------------------------------------------ command line
def _read(path: Path, label: str | None, drop: list[str]):
    import pandas as pd

    df = pd.read_csv(path)
    extra = [c for c in drop if c in df.columns]
    y = df[label].to_numpy() if label is not None and label in df.columns else None
    X = df.drop(columns=extra + ([label] if y is not None else [])).to_numpy(dtype=float)
    return X, y, df[extra]


def main(argv=None) -> None:
    import pandas as pd

    p = argparse.ArgumentParser(
        description="Fit GP v2 on CSV files and predict. Every column except --label and --drop is a feature, "
                    "in the same order in every file (spectral bands in wavelength order; for --kernel series, "
                    "date by date).")
    p.add_argument("--train", type=Path, required=True, help="training CSV (features + label column)")
    p.add_argument("--val", type=Path, help="validation CSV (features + label column); strongly recommended")
    p.add_argument("--predict", type=Path, help="CSV to classify (a label column, if present, is used for scoring)")
    p.add_argument("--out", type=Path, default=Path("predictions.csv"), help="where to write the predictions")
    p.add_argument("--label", default="label", help="name of the label column (default: label)")
    p.add_argument("--drop", default="", help="comma-separated non-feature columns (ids, coordinates) to ignore "
                                              "and copy to the output")
    p.add_argument("--kernel", choices=KERNELS, default="spectral")
    p.add_argument("--n-periods", type=int, help="number of dates, for --kernel series")
    p.add_argument("--target-priors-from", type=Path,
                   help="CSV whose label column gives the class frequencies expected where the model is used")
    p.add_argument("--save", type=Path, help="pickle the fitted model here")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    drop = [c.strip() for c in a.drop.split(",") if c.strip()]
    X, y, _ = _read(a.train, a.label, drop)
    if y is None:
        p.error(f"the training file has no column {a.label!r}")
    Xv = yv = None
    if a.val:
        Xv, yv, _ = _read(a.val, a.label, drop)
        if yv is None:
            p.error(f"the validation file has no column {a.label!r}")
    priors = None
    if a.target_priors_from:
        pool = pd.read_csv(a.target_priors_from)[a.label].to_numpy()
        priors = {c: float(np.mean(pool == c)) for c in np.unique(y)}
    gp = fit_gp_v2(X, y, Xv, yv, kernel=a.kernel, n_periods=a.n_periods, target_priors=priors,
                   random_state=a.seed)
    if a.save:
        with open(a.save, "wb") as f:
            pickle.dump(gp, f)
        print(f"model saved to {a.save}", file=sys.stderr)
    if a.predict:
        Xt, yt, ids = _read(a.predict, a.label, drop)
        P = gp.predict_proba(Xt)
        out = ids.reset_index(drop=True).copy()
        out["predicted"] = gp.classes_[P.argmax(axis=1)]
        out["confidence"] = P.max(axis=1)
        for j, c in enumerate(gp.classes_):
            out[f"p_{c}"] = P[:, j]
        out.to_csv(a.out, index=False)
        msg = f"{len(out)} predictions written to {a.out}"
        if yt is not None:
            msg += f"; accuracy against its {a.label!r} column: {100 * np.mean(out['predicted'].to_numpy() == yt):.2f}%"
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
