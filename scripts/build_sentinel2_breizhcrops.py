"""Build the Sentinel-2 crop-type dataset (BreizhCrops, Brittany, 2017) used by the benchmark.

Source: BreizhCrops (Russwurm et al., 2020), public S3 bucket
``breizhcrops.s3.eu-central-1.amazonaws.com``. For every agricultural parcel
declared in the French Land Parcel Identification System (Registre
Parcellaire Graphique, RPG 2017) in the four Breton departments, BreizhCrops
provides the time series of parcel-averaged Sentinel-2 Level-2A surface
reflectances for all 2017 acquisitions.

This script turns each parcel's irregular, cloud-contaminated time series into
a fixed-length feature vector:

1. clear-sky screening: an observation is kept if its blue reflectance
   B2 < 0.15 (1500 DN) and it carries no edge (EDG) or saturation (SAT)
   flag. The provided cloud column (CLD) is not used, because it flags as cloudy
   about half of the observations that are spectrally clear;
2. bi-monthly median composites (Jan-Feb, ..., Nov-Dec 2017) of the 10 bands
   B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12;
3. gaps (periods without a clear observation) are filled by linear
   interpolation between neighbouring periods (nearest period at the ends);
   parcels with no clear observation at all are dropped.

Output: ``data/sentinel2/breizhcrops_l2a_2017_bimonthly.npz`` with
``X`` (N, 60) int16 reflectance x 1e4, ordered period-major
(``feature_names``), ``y`` class id (0-8), ``region``, ``parcel_id``,
``code_cultu`` and ``n_clear`` (clear observations per parcel).

    python scripts/build_sentinel2_breizhcrops.py [--cache DIR]
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "https://breizhcrops.s3.eu-central-1.amazonaws.com"
REGIONS = ("frh01", "frh02", "frh03", "frh04")
REGION_NAMES = {
    "frh01": "Cotes-d'Armor (22)",
    "frh02": "Finistere (29)",
    "frh03": "Ille-et-Vilaine (35)",
    "frh04": "Morbihan (56)",
}
COLUMNS = ["doa", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "CLD", "EDG", "SAT"]
BANDS = COLUMNS[1:11]
PERIODS = ("Jan-Feb", "Mar-Apr", "May-Jun", "Jul-Aug", "Sep-Oct", "Nov-Dec")
CLEAR_B2_MAX = 1500.0
OUT = Path(__file__).resolve().parent.parent / "data" / "sentinel2" / "breizhcrops_l2a_2017_bimonthly.npz"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.rename(dest)


def fetch_region(cache: Path, region: str) -> tuple[Path, Path]:
    index = cache / f"{region}.csv"
    download(f"{BUCKET}/2017/L2A/{region}.csv", index)
    h5 = cache / "data" / "BreizhCrops" / "L2A_img" / "data" / "2017" / "L2A" / f"{region}.h5"
    if not h5.exists():
        archive = cache / f"{region}.h5.tar.gz"
        download(f"{BUCKET}/2017/L2A/{region}.h5.tar.gz", archive)
        with tarfile.open(archive) as t:
            t.extractall(cache)
        archive.unlink()
    return index, h5


def composite(ts: np.ndarray) -> tuple[np.ndarray | None, int]:
    """(T, 14) parcel time series -> (6 * 10,) bi-monthly median composite."""
    months = ts[:, 0].astype("int64").astype("datetime64[ns]").astype("datetime64[M]").astype(int) % 12 + 1
    refl = ts[:, 1:11]
    clear = (refl[:, 0] < CLEAR_B2_MAX) & (ts[:, 12] == 0) & (ts[:, 13] == 0)
    if not clear.any():
        return None, 0
    comp = np.full((len(PERIODS), len(BANDS)), np.nan)
    period = (months - 1) // 2
    for p in range(len(PERIODS)):
        sel = clear & (period == p)
        if sel.any():
            comp[p] = np.median(refl[sel], axis=0)
    have = ~np.isnan(comp[:, 0])
    if not have.all():
        idx = np.arange(len(PERIODS))
        for b in range(len(BANDS)):
            comp[:, b] = np.interp(idx, idx[have], comp[have, b])
    return comp.reshape(-1), int(clear.sum())


def main() -> None:
    import h5py

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=Path("breizhcrops_cache"))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()

    download(f"{BUCKET}/classmapping.csv", a.cache / "classmapping.csv")
    mapping = pd.read_csv(a.cache / "classmapping.csv", index_col=0)
    code_to_id = dict(zip(mapping["code"], mapping["id"]))
    class_names = mapping.drop_duplicates("id").sort_values("id")["classname"].tolist()

    X, y, region, pid, code, nclear = [], [], [], [], [], []
    dropped = {}
    for r in REGIONS:
        index_csv, h5path = fetch_region(a.cache, r)
        idx = pd.read_csv(index_csv)
        idx = idx[idx["CODE_CULTU"].isin(code_to_id)]
        n_drop = 0
        with h5py.File(h5path, "r") as f:
            for row in idx.itertuples(index=False):
                feat, n = composite(np.asarray(f[row.path]))
                if feat is None:
                    n_drop += 1
                    continue
                X.append(feat)
                y.append(code_to_id[row.CODE_CULTU])
                region.append(r)
                pid.append(row.id)
                code.append(row.CODE_CULTU)
                nclear.append(n)
        dropped[r] = n_drop
        print(f"{r}: {len(idx) - n_drop} parcels kept, {n_drop} without a clear observation")

    X = np.clip(np.rint(np.asarray(X)), -32768, 32767).astype(np.int16)
    names = [f"{per}_{b}" for per in PERIODS for b in BANDS]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, X=X, y=np.asarray(y, dtype=np.int8), region=np.asarray(region), parcel_id=np.asarray(pid),
        code_cultu=np.asarray(code), n_clear=np.asarray(nclear, dtype=np.int16),
        feature_names=np.asarray(names), class_names=np.asarray(class_names),
    )
    counts = pd.crosstab(pd.Series(np.asarray(y)).map(dict(enumerate(class_names))), pd.Series(region))
    meta = {
        "source": f"{BUCKET} (BreizhCrops, Sentinel-2 L2A, 2017)",
        "regions": REGION_NAMES,
        "bands": BANDS,
        "periods": PERIODS,
        "clear_rule": f"B2 < {CLEAR_B2_MAX} DN and EDG == 0 and SAT == 0",
        "dropped_no_clear_obs": dropped,
        "n_parcels": int(len(y)),
        "class_counts": {k: {c: int(v) for c, v in row.items()} for k, row in counts.iterrows()},
        "median_clear_obs_per_parcel": float(np.median(nclear)),
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out} ({len(y)} parcels)")
    print(counts)


if __name__ == "__main__":
    main()
