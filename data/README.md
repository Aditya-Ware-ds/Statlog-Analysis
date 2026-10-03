# StatLog (Landsat Satellite) data

`statlog/sat.trn` (4,435 rows) and `statlog/sat.tst` (2,000 rows) are the official
training and test files of the UCI *Statlog (Landsat Satellite)* dataset, in the
original format: 37 whitespace-separated integers per row, 36 spectral values
followed by the class code.

* **Source:** A. Srinivasan, *Statlog (Landsat Satellite)*, UCI Machine Learning
  Repository, 1993, <https://doi.org/10.24432/C55887>. Licensed CC BY 4.0.
* **Features:** a 3x3 neighbourhood of Landsat MSS pixels with 4 spectral bands
  each (values 0-255). Features are pixel-major: feature `4*p + b` is band `b` of
  pixel `p` (pixels run row by row, so pixel 4 is the centre).
* **Classes** (code 6, "mixture", has no samples):

| code | class | train | test |
|---:|---|---:|---:|
| 1 | red soil | 1072 | 461 |
| 2 | cotton crop | 479 | 224 |
| 3 | grey soil | 961 | 397 |
| 4 | damp grey soil | 415 | 211 |
| 5 | soil with vegetation stubble | 470 | 237 |
| 7 | very damp grey soil | 1038 | 470 |

## Provenance and verification

`scripts/fetch_data.py` rebuilds the two files. It downloads them from UCI if
that host is reachable. If not, it reconstructs them from the `Satellite`
data frame of the R package [mlbench](https://cran.r-project.org/package=mlbench),
which stores the UCI training rows followed by the UCI test rows in their
original order.

The committed files were produced through the mlbench route, because the build
environment could not reach `archive.ics.uci.edu`. They were checked as follows:

* the first 4,435 rows reproduce the official training class counts exactly,
  and the last 2,000 rows reproduce the official test counts;
* the first training row and the first test row match the first lines of the
  UCI `sat.trn` / `sat.tst`;
* all values are integers in [27, 157].

The script verifies a SHA-256 digest of the parsed integer arrays, so either
route must produce identical contents.
`sg_hfq.data.load_statlog()` also re-checks the per-class counts on every load.

---

# Hyperspectral benchmarks (`data/hsi/`, not committed)

Run `python scripts/fetch_hsi.py`. It tries the canonical UPV/EHU host
(<https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>)
first, then public GitHub mirrors, and verifies the SHA-256 of every file.

| File | Cube | Labelled pixels | Classes |
|---|---|---:|---:|
| `Indian_pines_corrected.mat` + `_gt` | 145 x 145 x 200 (AVIRIS, water-absorption bands removed) | 10,249 | 16 |
| `PaviaU.mat` + `_gt` | 610 x 340 x 103 (ROSIS) | 42,776 | 9 |
| `Salinas_corrected.mat` + `_gt` | 512 x 217 x 204 (AVIRIS, water-absorption bands removed) | 54,129 | 16 |

The canonical host was unreachable from the build environment, so the files were
taken from GitHub mirrors and checked in two ways:

* the Indian Pines cube and ground truth, and the Pavia University and Salinas
  ground-truth maps, have identical git blob hashes in independent
  repositories;
* the cube dimensions and per-class counts of all three scenes match the
  published values exactly.

# Sentinel-2 crop types, Brittany 2017 (`data/sentinel2/`, not committed)

Run `python scripts/build_sentinel2_breizhcrops.py --cache <dir>`. It needs
about 3.5 GB of temporary space and takes about 10 minutes. It downloads the
public BreizhCrops L2A archives
(<https://breizhcrops.s3.eu-central-1.amazonaws.com>; Russwurm et al., 2020)
and writes `breizhcrops_l2a_2017_bimonthly.npz` (608,489 parcels x 60
features) plus a JSON summary. The script documents every preprocessing step;
see `BENCHMARK.md` for the full dataset description.
