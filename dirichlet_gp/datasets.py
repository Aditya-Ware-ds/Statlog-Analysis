"""Evaluation datasets: StatLog, Indian Pines, Pavia University, Salinas, Sentinel-2 (BreizhCrops).

Every loader returns a :class:`BenchmarkDataset` holding all labelled samples,
the class names, the GP configuration for that dataset, and a ``split(seed)``
function returning train / held-out / test indices. The GP is trained on the
training indices only; the held-out indices (a 5% per-class reserve on the
hyperspectral scenes, one department on Sentinel-2) are used by no step, but are
kept so that the test sets stay identical to the published results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .data import CLASS_NAMES as STATLOG_NAMES
from .data import load_statlog
from .gp import contiguous_groups, statlog_feature_permutations, statlog_groups

ROOT = Path(__file__).resolve().parent.parent / "data"

INDIAN_PINES = (
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
    "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
)
PAVIA_UNIVERSITY = (
    "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen",
    "Self-Blocking Bricks", "Shadows",
)
SALINAS = (
    "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble",
    "Celery", "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
    "Vinyard_vertical_trellis",
)

Split = tuple[np.ndarray, "np.ndarray | None", np.ndarray]

#: GP configuration for StatLog: the kernel selected by marginal likelihood in ``dirichlet_gp.statlog_study``
#: (Matern-5/2, 12 band x pixel-orbit length-scales, invariant to the 8 symmetries of the 3x3 window;
#: see results/gp_statlog/summary.md).
STATLOG_GP = {"representation": "full36", "groups": statlog_groups("band_orbit"),
              "permutations": statlog_feature_permutations(), "kernel": "matern52", "n_opt": 2000, "max_iter": 200}


@dataclass
class BenchmarkDataset:
    key: str
    title: str
    X: np.ndarray
    y: np.ndarray
    class_names: dict[int, str]
    split: Callable[[int], Split]
    info: dict = field(default_factory=dict)
    gp: dict = field(default_factory=dict)  # GPDirichletClassifier keyword arguments

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(int(c) for c in np.unique(self.y))


def stratified_split(y: np.ndarray, seed: int, train_frac: float, val_frac: float,
                     min_train: int = 5, min_val: int = 3) -> Split:
    """Per-class random split: ``train_frac`` / ``val_frac`` of each class (with minimum counts), rest test."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_tr = max(min_train, int(round(train_frac * len(idx))))
        n_va = max(min_val, int(round(val_frac * len(idx))))
        if n_tr + n_va >= len(idx):
            raise ValueError(f"class {c} too small for the requested split")
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)), np.sort(np.concatenate(te))


def capped_sample(idx: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """At most ``cap`` random samples per class from ``idx``."""
    out = []
    for c in np.unique(y[idx]):
        pool = idx[y[idx] == c]
        out.append(rng.choice(pool, size=min(cap, len(pool)), replace=False))
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------- loaders
def statlog() -> BenchmarkDataset:
    d = load_statlog()
    X = np.vstack([d.X_train, d.X_test])
    y = np.concatenate([d.y_train, d.y_test])
    n_tr = len(d.y_train)

    def split(seed: int) -> Split:
        return np.arange(n_tr), None, np.arange(n_tr, len(y))

    return BenchmarkDataset(
        "statlog", "StatLog (Landsat MSS)", X, y, dict(STATLOG_NAMES), split,
        gp=STATLOG_GP,
        info={
            "sensor": "Landsat MSS (4 bands), 3x3 neighbourhoods", "features": 36, "classes": 6,
            "labelled": len(y),
            "split": "official UCI split: 4,435 train / 2,000 test",
        },
    )


def _load_mat(cube: str, gt: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    from scipy.io import loadmat

    path = ROOT / "hsi"
    a, b = loadmat(path / f"{cube}.mat"), loadmat(path / f"{gt}.mat")
    X = next(v for k, v in a.items() if not k.startswith("__"))
    G = next(v for k, v in b.items() if not k.startswith("__"))
    mask = G > 0
    return X[mask].astype(np.float64), G[mask].astype(int), X.shape


def _hsi(key, title, cube, gt, names, train_frac, val_frac, sensor, extra) -> BenchmarkDataset:
    X, y, shape = _load_mat(cube, gt)

    def split(seed: int) -> Split:
        return stratified_split(y, seed, train_frac, val_frac)

    return BenchmarkDataset(
        key, title, X, y, {i + 1: n for i, n in enumerate(names)}, split,
        gp={"representation": "standard", "groups": contiguous_groups(shape[2], 10), "n_opt": 1500, "max_iter": 150},
        info={
            "sensor": sensor, "image": f"{shape[0]} x {shape[1]} pixels", "bands": shape[2],
            "classes": len(names), "labelled": len(y),
            "split": f"stratified random per class: {train_frac:.0%} train (min 5), {val_frac:.0%} held out "
                     f"(min 3, unused), rest test; 5 seeds",
            **extra,
        },
    )


def indian_pines() -> BenchmarkDataset:
    return _hsi("indian_pines", "Indian Pines (AVIRIS)", "Indian_pines_corrected", "Indian_pines_gt",
                INDIAN_PINES, 0.10, 0.05, "AVIRIS, 20 m GSD, 400-2500 nm",
                {"preprocessing": "corrected cube: 24 water-absorption bands removed (224 -> 200)"})


def pavia_university() -> BenchmarkDataset:
    return _hsi("pavia_university", "Pavia University (ROSIS)", "PaviaU", "PaviaU_gt", PAVIA_UNIVERSITY,
                0.05, 0.05, "ROSIS, 1.3 m GSD, 430-860 nm",
                {"preprocessing": "103 bands as distributed (noisy bands removed by the providers)"})


def salinas() -> BenchmarkDataset:
    return _hsi("salinas", "Salinas (AVIRIS)", "Salinas_corrected", "Salinas_gt", SALINAS, 0.05, 0.05,
                "AVIRIS, 3.7 m GSD, 400-2500 nm",
                {"preprocessing": "corrected cube: 20 water-absorption bands removed (224 -> 204)"})


def sentinel2_breizhcrops(train_cap: int = 500, val_cap: int = 200) -> BenchmarkDataset:
    path = ROOT / "sentinel2" / "breizhcrops_l2a_2017_bimonthly.npz"
    z = np.load(path, allow_pickle=False)
    X = z["X"].astype(np.float64)
    y = z["y"].astype(int)
    region = z["region"]
    names = {i: str(n) for i, n in enumerate(z["class_names"])}
    train_pool = np.flatnonzero(np.isin(region, ["frh01", "frh02"]))
    val_pool = np.flatnonzero(region == "frh03")
    test = np.flatnonzero(region == "frh04")

    def split(seed: int) -> Split:
        rng = np.random.default_rng(seed)
        return capped_sample(train_pool, y, train_cap, rng), capped_sample(val_pool, y, val_cap, rng), test

    return BenchmarkDataset(
        "sentinel2_breizhcrops", "Sentinel-2 crop types (Brittany)", X, y, names, split,
        # one length-scale per spectral band, shared by its 6 bi-monthly composites (period-major features)
        gp={"representation": "standard", "groups": np.arange(X.shape[1]) % 10, "n_opt": 1500, "max_iter": 150},
        info={
            "sensor": "Sentinel-2 L2A, bands B2-B8A, B11, B12 (10 m / 20 m)",
            "features": X.shape[1], "classes": len(names), "labelled": len(y),
            "split": f"spatial (by department): train = up to {train_cap}/class from FRH01+FRH02, "
                     f"held out = up to {val_cap}/class from FRH03 (unused), test = all of FRH04; 5 seeds",
        },
    )


LOADERS: dict[str, Callable[[], BenchmarkDataset]] = {
    "statlog": statlog,
    "indian_pines": indian_pines,
    "pavia_university": pavia_university,
    "salinas": salinas,
    "sentinel2_breizhcrops": sentinel2_breizhcrops,
}
