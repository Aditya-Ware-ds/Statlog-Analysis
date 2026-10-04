"""StatLog (Landsat Satellite) data loading and Phase 1 spectral representation.

The StatLog Landsat data (UCI ML repository, dataset 146) stores, for each
3x3 pixel neighbourhood, the four Landsat MSS spectral values of every pixel.
Features are ordered pixel-major: the 4 band values of the top-left pixel come
first, then the 4 values of the top-middle pixel, and so on, so feature index
``4 * p + b`` holds band ``b`` of pixel ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_PIXELS = 9
N_BANDS = 4
N_FEATURES = N_PIXELS * N_BANDS

#: Class codes present in the data (code 6, "mixture class", has no samples).
CLASS_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 7)
CLASS_NAMES: dict[int, str] = {
    1: "red soil",
    2: "cotton crop",
    3: "grey soil",
    4: "damp grey soil",
    5: "soil with vegetation stubble",
    7: "very damp grey soil",
}
SHORT_NAMES: dict[int, str] = {
    1: "red",
    2: "cotton",
    3: "grey",
    4: "damp grey",
    5: "stubble",
    7: "v. damp grey",
}

#: Official split sizes and per-class counts (from the UCI documentation).
OFFICIAL_COUNTS = {
    "train": {1: 1072, 2: 479, 3: 961, 4: 415, 5: 470, 7: 1038},
    "test": {1: 461, 2: 224, 3: 397, 4: 211, 5: 237, 7: 470},
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "statlog"


@dataclass(frozen=True)
class StatLogData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def classes(self) -> tuple[int, ...]:
        return CLASS_CODES


def read_sat_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``sat.trn`` / ``sat.tst`` file (37 whitespace-separated integers per row)."""
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != N_FEATURES + 1:
        raise ValueError(f"{path}: expected {N_FEATURES + 1} columns, got shape {arr.shape}")
    return arr[:, :N_FEATURES].astype(np.float64), arr[:, N_FEATURES]


def load_statlog(data_dir: str | Path | None = None, verify: bool = True) -> StatLogData:
    """Load the official 4,435 / 2,000 StatLog Landsat train/test split."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    X_tr, y_tr = read_sat_file(data_dir / "sat.trn")
    X_te, y_te = read_sat_file(data_dir / "sat.tst")
    data = StatLogData(X_tr, y_tr, X_te, y_te)
    if verify:
        verify_official_split(data)
    return data


def verify_official_split(data: StatLogData) -> None:
    for split, y in (("train", data.y_train), ("test", data.y_test)):
        codes, counts = np.unique(y, return_counts=True)
        got = dict(zip(codes.tolist(), counts.tolist()))
        if got != OFFICIAL_COUNTS[split]:
            raise ValueError(f"{split} split does not match the official class counts: {got}")


def band_mean(X: np.ndarray) -> np.ndarray:
    """Average each spectral band over the 3x3 neighbourhood: (N, 36) -> (N, 4)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")
    return X.reshape(len(X), N_PIXELS, N_BANDS).mean(axis=1)


class SpectralRepresentation:
    """Phase 1: spectral representation standardised with training statistics.

    * ``"bandmean4"`` (the paper's recommendation for StatLog): the 36 raw
      features are reduced to the 4 band means over the 3x3 window, then
      z-scored (Eq. 1).
    * ``"full36"`` / ``"standard"``: every raw feature z-scored.
    * ``"log36"``: natural log of every raw (positive) feature, then z-scored.
    * ``"pca"``: every feature z-scored, then projected on the first
      ``n_components`` principal axes of the training data (unwhitened). This
      is the high-dimensional analogue of the band-mean reduction: it keeps the
      dominant spectral structure while avoiding near-singular class
      covariances in hundreds of bands.
    """

    KINDS = ("bandmean4", "full36", "standard", "pca", "log36")

    def __init__(self, kind: str = "bandmean4", n_components: int = 10):
        if kind not in self.KINDS:
            raise ValueError(f"unknown representation {kind!r}; choose from {self.KINDS}")
        self.kind = kind
        self.n_components = n_components

    def _project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "bandmean4":
            return band_mean(X)
        if self.kind == "log36":
            return np.log(np.maximum(X, 1.0))
        return X

    def fit(self, X: np.ndarray) -> "SpectralRepresentation":
        P = self._project(X)
        self.mean_ = P.mean(axis=0)
        std = P.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        if self.kind == "pca":
            Z = (P - self.mean_) / self.std_
            k = min(self.n_components, Z.shape[1], Z.shape[0])
            _, s, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
            # deterministic sign: largest loading of each axis is positive
            signs = np.sign(Vt[np.arange(k), np.abs(Vt[:k]).argmax(axis=1)])
            self.components_ = Vt[:k] * signs[:, None]
            self.explained_variance_ratio_ = (s[:k] ** 2) / (s**2).sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (self._project(X) - self.mean_) / self.std_
        return Z @ self.components_.T if self.kind == "pca" else Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_dims(self) -> int:
        if self.kind == "bandmean4":
            return N_BANDS
        if self.kind == "pca":
            return len(self.components_)
        return len(self.mean_)
