"""Download the StatLog (Landsat Satellite) data and verify the official split.

The files are already committed under ``data/statlog/``; this script
re-creates them from the original source. It tries the UCI repository first
and, if that host is unreachable, rebuilds the files from the copy shipped in
the R package ``mlbench`` (``Satellite``: the UCI training rows followed by
the UCI test rows, in their original order). Either way the parsed contents
are checked against fixed SHA-256 digests, so both routes give identical data.

    python scripts/fetch_data.py [--out data/statlog] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/{}"
MLBENCH = "https://raw.githubusercontent.com/cran/mlbench/master/data/Satellite.rda"
N_TRAIN = 4435

# SHA-256 of the parsed (rows x 37) int64 little-endian arrays.
DIGESTS = {
    "sat.trn": "34c4241c59fb4998ad7d4c5ea4eea6ca02fc228907dbe8233dfbaf1888a9f63e",
    "sat.tst": "e54aad9d61443702c24ba22b156f5a76112b012cc96eeb50a726afd79e7b880c",
}
MLBENCH_CODES = {
    "red soil": 1,
    "cotton crop": 2,
    "grey soil": 3,
    "damp grey soil": 4,
    "vegetation stubble": 5,
    "very damp grey soil": 7,
}


def digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype="<i8").tobytes()).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def from_uci() -> dict[str, np.ndarray]:
    return {name: np.loadtxt(io.BytesIO(fetch(UCI.format(name))), dtype=np.int64) for name in DIGESTS}


def from_mlbench(tmp: Path) -> dict[str, np.ndarray]:
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("the mlbench fallback needs `pip install pyreadr`") from exc
    path = tmp / "Satellite.rda"
    path.write_bytes(fetch(MLBENCH))
    df = pyreadr.read_r(str(path))["Satellite"]
    X = df.iloc[:, :36].to_numpy()
    if not np.array_equal(X, np.round(X)):
        raise SystemExit("unexpected non-integer values in the mlbench copy")
    y = df.iloc[:, 36].astype(str).map(MLBENCH_CODES).to_numpy()
    rows = np.column_stack([X.astype(np.int64), y.astype(np.int64)])
    return {"sat.trn": rows[:N_TRAIN], "sat.tst": rows[N_TRAIN:]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "statlog")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if not a.force and all((a.out / n).exists() for n in DIGESTS):
        arrays = {n: np.loadtxt(a.out / n, dtype=np.int64) for n in DIGESTS}
        source = "existing files"
    else:
        try:
            arrays, source = from_uci(), "UCI"
        except Exception as exc:  # network policy, outage, ...
            print(f"UCI download failed ({exc}); falling back to the mlbench copy", file=sys.stderr)
            arrays, source = from_mlbench(a.out), "mlbench"
            (a.out / "Satellite.rda").unlink(missing_ok=True)

    for name, arr in arrays.items():
        got = digest(arr)
        if got != DIGESTS[name]:
            raise SystemExit(f"{name}: content digest mismatch ({got})")
        if source != "existing files":
            np.savetxt(a.out / name, arr, fmt="%d")
        print(f"{name}: {arr.shape[0]} rows, digest OK ({source})")


if __name__ == "__main__":
    main()
