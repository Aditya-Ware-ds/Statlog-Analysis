"""StatLog (Landsat Satellite) data loading and input standardisation.

The StatLog Landsat data (UCI ML repository, dataset 146) stores, for each
3x3 pixel neighbourhood, the four Landsat MSS spectral values of every pixel.
Features are ordered pixel-major: the 4 band values of the top-left pixel come
first, then the 4 values of the top-middle pixel, and so on, so feature index
``4 * p + b`` holds band ``b`` of pixel ``p`` (pixel 4 is the centre).
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


class Standardizer:
    """Z-score every feature with training statistics, optionally after a log transform.

    ``kind="standard"`` (alias ``"full36"``) z-scores the raw features;
    ``kind="log36"`` takes the natural log of the (positive) raw values first.
    """

    KINDS = ("standard", "full36", "log36")

    def __init__(self, kind: str = "standard"):
        if kind not in self.KINDS:
            raise ValueError(f"unknown representation {kind!r}; choose from {self.KINDS}")
        self.kind = kind

    def _project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.log(np.maximum(X, 1.0)) if self.kind == "log36" else X

    def fit(self, X: np.ndarray) -> "Standardizer":
        P = self._project(X)
        self.mean_ = P.mean(axis=0)
        std = P.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._project(X) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class CompositeFeatures:
    """Concatenate feature blocks computed from raw spectra (all statistics from training data).

    * ``"z"``: every band z-scored (the standard representation);
    * ``"angle"``: the raw spectrum scaled to unit L2 norm (spectral shape, brightness removed),
      then centred and divided by one *global* scale (average per-feature variance 1), so that
      Euclidean distances in this block are proportional to chordal spectral-angle distances;
    * ``"deriv"``: first difference along the bands of the 3-band moving-average-smoothed raw
      spectrum, z-scored per feature.

    ``block_slices_`` gives the column range of each block in the output.
    """

    BLOCKS = ("z", "angle", "deriv")

    def __init__(self, blocks=("z", "angle", "deriv")):
        unknown = set(blocks) - set(self.BLOCKS)
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")
        self.blocks = tuple(blocks)

    @staticmethod
    def _unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _deriv(X):
        c = np.cumsum(np.pad(X, ((0, 0), (1, 1)), mode="edge"), axis=1)
        smooth = np.concatenate([c[:, 2:3], c[:, 3:] - c[:, :-3]], axis=1) / 3.0
        smooth[:, 0] = (X[:, 0] * 2 + X[:, 1]) / 3.0
        return np.diff(smooth, axis=1)

    def _raw_blocks(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = {}
        if "z" in self.blocks:
            out["z"] = X
        if "angle" in self.blocks:
            out["angle"] = self._unit(X)
        if "deriv" in self.blocks:
            out["deriv"] = self._deriv(X)
        return out

    def fit(self, X: np.ndarray) -> "CompositeFeatures":
        raw = self._raw_blocks(X)
        self.stats_ = {}
        self.block_slices_ = {}
        start = 0
        for b in self.blocks:
            R = raw[b]
            mean = R.mean(axis=0)
            if b == "angle":
                # one global scale (keeps angles' geometry) giving an average per-feature variance of 1
                scale = np.full(R.shape[1], np.sqrt(((R - mean) ** 2).mean()) or 1.0)
            else:
                std = R.std(axis=0)
                scale = np.where(std > 0, std, 1.0)
            self.stats_[b] = (mean, scale)
            self.block_slices_[b] = slice(start, start + R.shape[1])
            start += R.shape[1]
        self.n_features_ = start
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_blocks(X)
        return np.concatenate([(raw[b] - self.stats_[b][0]) / self.stats_[b][1] for b in self.blocks], axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
