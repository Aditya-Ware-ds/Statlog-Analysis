# Dirichlet Gaussian-process classification for land-cover data

This repository contains a from-scratch **Gaussian-process (GP) classifier** for remote-sensing
land-cover classification and its evaluation on five datasets:

- the StatLog (Landsat Satellite) benchmark, official split;
- three AVIRIS / ROSIS hyperspectral scenes: Indian Pines, Pavia University and Salinas;
- a real-world Sentinel-2 crop-type mapping task in Brittany, France.

The classifier is the Dirichlet-based GP of Milios et al. (NeurIPS 2018). It uses exact GP inference,
kernels with grouped length-scales and optional symmetry invariance, and hyper-parameters learned by
maximising the marginal likelihood. No validation data or cross-validation is used anywhere.

```bash
pip install -r requirements.txt                    # numpy, scipy, pandas (+ pytest, pyreadr, h5py for dev/data)
python -m pytest                                   # 25 unit tests
python -m dirichlet_gp.benchmark run --datasets statlog --seeds 5 && python -m dirichlet_gp.benchmark report
```

## Results

Test-set percentages, mean ± standard deviation over 5 seeds
([`results/benchmark/summary.md`](results/benchmark/summary.md)):

| Dataset | Overall accuracy | Average accuracy | Kappa x100 | AURC (lower is better) |
|---|---|---|---|---|
| StatLog (Landsat MSS) | 91.5 ± 0.1 | 89.5 ± 0.1 | 89.5 ± 0.2 | 1.2 ± 0.0 |
| Indian Pines (AVIRIS) | 81.1 ± 0.6 | 75.4 ± 1.5 | 78.2 ± 0.7 | 5.8 ± 0.2 |
| Pavia University (ROSIS) | 91.7 ± 0.4 | 86.9 ± 0.7 | 88.9 ± 0.5 | 1.5 ± 0.1 |
| Salinas (AVIRIS) | 91.3 ± 0.1 | 94.7 ± 0.2 | 90.3 ± 0.2 | 1.7 ± 0.1 |
| Sentinel-2 crop types (Brittany) | 68.3 ± 1.2 | 57.7 ± 0.3 | 60.6 ± 1.3 | 14.0 ± 1.0 |

AURC is the area under the risk-coverage curve when test samples are ranked by the GP's maximum class
probability: the error accumulated when the most confident predictions are accepted first.
Per-class accuracies and the learned length-scales are in the summary file.

### StatLog: kernel selection without cross-validation

StatLog samples are overlapping 3x3 pixel neighbourhoods cut from one image. Random cross-validation
folds would therefore share pixels between training and validation samples and give optimistic
estimates. Kernel selection uses the **marginal likelihood (evidence)** instead, computed on training
data only.
- 12 kernel structures were compared by evidence: isotropic, per-band, per band x pixel position, full
  ARD, RBF vs Matérn-5/2, log inputs, rotation/reflection invariance, and an extra centre-pixel term.
- The highest evidence goes to a **Matérn-5/2 kernel with 12 length-scales (4 bands x
  centre / edge / corner pixel), averaged over the 8 rotations and reflections of the 3x3 window**. The
  label belongs to the centre pixel, so the window's orientation should not matter.
- Its hyper-parameters were then re-learned on all 4,435 training samples. Full table:
  [`results/gp_statlog/summary.md`](results/gp_statlog/summary.md).

| Final StatLog model (official test set) | Value |
|---|---|
| Overall accuracy | **91.65%** |
| Average accuracy | 89.60% |
| Cohen's kappa | 89.71 |
| AURC | 1.17% |
| Expected calibration error | 1.68% |

For context:
- **Earlier version of this repository** (commit `126d53a`): an RBF SVM scored 91.50% and a Random
  Forest 91.20% on the same split.
- **Published results:** Crammer-Singer SVMs are reported at 92.35% (Hsu & Lin, 2002) and 92.45%
  (Lee & Lin). The GP does not reach them.
- **Why kernel or setting choices won't close that gap:** all 12 kernels score 90.8-91.45% on the test
  set. A post-hoc sweep of the Dirichlet concentration α_ε moves the final model only between 91.50%
  and 91.70%; that sweep was used for no decision.
- **Noise level:** on 2,000 test samples the standard error of an accuracy is about 0.6 points.

## Method

**Dirichlet label transformation** (`dirichlet_gp/gp.py`). A one-hot label is read as a draw from a
Dirichlet with concentration α_ε + y. Its log-normal moment match gives, for each sample i and class c,

```
alpha_ic  = alpha_eps + [y_i = c]
sigma2_ic = log(1 / alpha_ic + 1)          # known heteroscedastic noise
target_ic = log(alpha_ic) - sigma2_ic / 2  # regression target
```

Classification thus becomes C exact GP regressions sharing one kernel. Class probabilities are the
Monte-Carlo mean of softmax(f), with f drawn from the C Gaussian predictive marginals. α_ε = 0.01, the
authors' default, is fixed.

**Kernels.** Stationary RBF or Matérn-5/2 on z-scored inputs with *grouped ARD*: features in a group
share a length-scale.
- `GroupedKernel(groups, kind, permutations)` can be made invariant to a finite group of feature
  permutations, k_inv(x, x') = mean_t k(x, t x'). This is positive semi-definite when the length-scale
  groups are unions of permutation orbits.
- `SumKernel` adds two kernels. A group id of -1 excludes a feature from a kernel.

| Dataset | Kernel (fixed a priori) |
|---|---|
| StatLog | Evidence-selected: Matérn-5/2, 12 band x pixel-orbit length-scales, dihedral-invariant over the 3x3 window |
| Indian Pines, Pavia University, Salinas | RBF over all bands, 10 length-scales for contiguous spectral blocks |
| Sentinel-2 Brittany | RBF over 60 features, one length-scale per spectral band shared by its 6 bi-monthly composites |

**Hyper-parameters.**
- Length-scales and signal variance are learned by type-II maximum likelihood, with analytic gradients
  and L-BFGS-B.
- The optimisation runs on a class-stratified subset of up to 1,500 training samples (2,000 on
  StatLog). The posterior then conditions on all training samples.

**Exact and efficient.** Every class has the same off-class noise level. So the class-c covariance is a
shared matrix B = K + s_off I plus a correction on the class-c samples only. The Woodbury identity
then gives all C marginal likelihoods, gradients and predictive variances from **one** n x n
factorisation, with no approximation; a unit test checks it against C independent GPs.

```python
from dirichlet_gp import GPDirichletClassifier, load_statlog
from dirichlet_gp.gp import statlog_feature_permutations, statlog_groups

d = load_statlog()
gp = GPDirichletClassifier(representation="full36", groups=statlog_groups("band_orbit"), kernel="matern52",
                           permutations=statlog_feature_permutations()).fit(d.X_train, d.y_train)
P = gp.predict_proba(d.X_test)            # (2000, 6) class probabilities
print(gp.lengthscales_, gp.log_marginal_likelihood())
```

## Datasets

| | StatLog | Indian Pines | Pavia University | Salinas | Sentinel-2 Brittany |
|---|---|---|---|---|---|
| Sensor | Landsat MSS | AVIRIS | ROSIS | AVIRIS | Sentinel-2 MSI (L2A) |
| Scene | 3x3 neighbourhoods from an 82 x 100 px tile | 145 x 145 px, 20 m | 610 x 340 px, 1.3 m | 512 x 217 px, 3.7 m | 4 French departments, field parcels |
| Bands / features | 4 bands x 9 pixels = 36 | 200 (224 minus water absorption) | 103 | 204 (224 minus water absorption) | 10 bands x 6 bi-monthly composites = 60 |
| Classes | 6 | 16 | 9 | 16 | 9 crop types |
| Labelled samples | 6,435 | 10,249 | 42,776 | 54,129 | 608,489 parcels |
| Train / held out / test (seed 0) | 4,435 / - / 2,000 (official) | 1,030 / 517 / 8,702 | 2,137 / 2,137 / 38,502 | 2,706 / 2,706 / 48,717 | 3,535 / 1,420 / 122,708 |
| Split | official UCI split | 10% / 5% / rest per class (min 5 / 3) | 5% / 5% / rest per class | 5% / 5% / rest per class | spatial: by department |

The GP is trained on the training samples only. The held-out samples are used by no step; they are kept
so that the test sets stay the same as in the published results. The hyperspectral splits are the
usual random per-class pixel samples. Training and test pixels are spatial neighbours there, which makes
the scores optimistic. The Sentinel-2 task uses a spatially disjoint split instead.

Getting the data (details and checksums in [`data/README.md`](data/README.md)):
- StatLog is committed in `data/statlog/` (`scripts/fetch_data.py` re-creates and verifies it).
- `python scripts/fetch_hsi.py` downloads the three hyperspectral scenes and checks their SHA-256 digests.
- `python scripts/build_sentinel2_breizhcrops.py --cache <dir>` builds the Sentinel-2 dataset (about
  3.5 GB download).

### Sentinel-2: crop-type mapping in Brittany, France

The task is to map the crop on every declared agricultural parcel in Brittany for the 2017 season using
only Sentinel-2. The model is tested on a department never seen in training, as in an operational
Common Agricultural Policy monitoring system.

| Item | Details |
|---|---|
| Geographic region | Brittany (Bretagne), north-west France, about 27,200 km². The four departments (NUTS-3) are Côtes-d'Armor (FRH01), Finistère (FRH02), Ille-et-Vilaine (FRH03) and Morbihan (FRH04) |
| Data source | BreizhCrops (Rußwurm et al., 2020, *ISPRS Archives* XLIII-B2): parcel-averaged Sentinel-2 Level-2A surface reflectance, from the public bucket `breizhcrops.s3.eu-central-1.amazonaws.com` |
| Sentinel-2 bands used | B2 (490 nm), B3 (560 nm), B4 (665 nm), B8 (842 nm) at 10 m; B5 (705 nm), B6 (740 nm), B7 (783 nm), B8A (865 nm), B11 (1610 nm), B12 (2190 nm) at 20 m. The 60 m atmospheric bands B1, B9 and B10 are not used |
| Spatial resolution | 10 m / 20 m pixels, averaged over each field parcel (object-based); the sample unit is the parcel |
| Acquisition period | 2 January to 28 December 2017. About 58-71 distinct acquisition dates per department; median 27-39 observations per parcel, of which a median of 28 are clear |
| Ground-truth source | French Land Parcel Identification System, the *Registre Parcellaire Graphique* (RPG) 2017: farmers' crop declarations for CAP subsidies, published by IGN/ASP under an open licence, grouped by BreizhCrops into 9 classes |
| Classes | barley, wheat, rapeseed, corn, sunflower, orchards, nuts, permanent meadows, temporary meadows |
| Number of samples | 608,489 parcels: FRH01 178,632; FRH02 140,782; FRH03 166,367; FRH04 122,708. No parcel was dropped for lack of a clear observation |
| Preprocessing | (1) Clear-sky screening: keep an observation if B2 < 0.15 reflectance and its edge and saturation flags are 0. The provided cloud flag marks about 57% of spectrally clear observations as cloudy, so it is not used. (2) Bi-monthly median composites of each band. (3) Linear interpolation across empty periods. (4) z-scoring with training statistics |
| Train / held out / test | Spatially disjoint by department. Train: up to 500 parcels per class from FRH01 + FRH02 (3,535). Held out (unused): up to 200 per class from FRH03. Test: **all 122,708 parcels of FRH04**, with their real class distribution. 5 seeds |

| Class | FRH01 | FRH02 | FRH03 | FRH04 (test) |
|---|---:|---:|---:|---:|
| barley | 13,051 | 10,736 | 7,154 | 5,981 |
| wheat | 30,380 | 15,026 | 27,202 | 17,009 |
| rapeseed | 5,596 | 2,349 | 3,557 | 3,244 |
| corn | 44,003 | 36,620 | 42,011 | 31,361 |
| sunflower | 1 | 6 | 10 | 2 |
| orchards | 937 | 348 | 1,217 | 552 |
| nuts | 10 | 18 | 10 | 11 |
| permanent meadows | 32,641 | 36,536 | 32,524 | 26,134 |
| temporary meadows | 52,013 | 39,143 | 52,682 | 38,414 |

Crop types rather than soil types are used here because no field-surveyed soil labels (e.g. LUCAS
topsoil) were reachable from the build environment. Sunflower (2 test parcels) and nuts (11) are so rare
that their per-class accuracy is very noisy and pulls the average accuracy down. They are kept anyway,
to stay faithful to the real task.

## Repository layout

```
dirichlet_gp/
  gp.py              GP classifier: Dirichlet transform, grouped-ARD / invariant / sum kernels, Woodbury algebra
  data.py            StatLog loader (official split, class-count check) and input standardisation
  datasets.py        the five evaluation datasets and their splits; per-dataset GP configuration
  metrics.py         accuracy, kappa, macro F1, risk-coverage / AURC, NLL and calibration error
  benchmark.py       5-seed evaluation on all datasets; writes results/benchmark/
  statlog_study.py   StatLog kernel selection by evidence, final refit, post-hoc check; writes results/gp_statlog/
scripts/             data download / preparation (StatLog, hyperspectral scenes, Sentinel-2)
tests/               unit tests (gradient check, invariance, Woodbury vs. direct GPs, metrics, splits)
results/             committed outputs of the two studies
```

## Reproducing

```bash
python -m dirichlet_gp.benchmark run --datasets all --seeds 5     # about 1 h on 4 cores
python -m dirichlet_gp.benchmark report
python -m dirichlet_gp.statlog_study run --candidates all         # 12 kernels, about 25 min in 4 processes
python -m dirichlet_gp.statlog_study final                        # full-data refit, about 1 h
python -m dirichlet_gp.statlog_study sensitivity && python -m dirichlet_gp.statlog_study report
```

All randomness is seeded. The committed results were checked against this code. Re-running StatLog
candidates reproduces them exactly. An Indian Pines re-run reproduces the per-class accuracies exactly
and the other metrics to within 0.002 percentage points, which is floating-point noise from BLAS
threading.

## References

- D. Milios, R. Camoriano, P. Michiardi, L. Rosasco, M. Filippone, "Dirichlet-based Gaussian Processes
  for Large-scale Calibrated Classification", NeurIPS 2018.
- C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
- A. Srinivasan, *Statlog (Landsat Satellite)*, UCI Machine Learning Repository, 1993,
  <https://doi.org/10.24432/C55887>.
- M. Rußwurm et al., "BreizhCrops: A Time Series Dataset for Crop Type Mapping", ISPRS Archives
  XLIII-B2, 2020.
- C.-W. Hsu, C.-J. Lin, "A comparison of methods for multiclass support vector machines", IEEE TNN, 2002.

Code: MIT License (see `LICENSE`).
