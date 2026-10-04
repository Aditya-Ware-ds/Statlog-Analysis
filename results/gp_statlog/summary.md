# Gaussian-process classifier on StatLog (official split)

Model: Dirichlet-based GP classification (Milios et al., NeurIPS 2018), exact inference, alpha_eps = 0.01 fixed a priori. For each candidate kernel, the hyper-parameters are learned by type-II maximum likelihood on the same class-stratified subset of 2000 training samples.

**Selection rule (fixed in advance): highest log marginal likelihood (evidence).** No cross-validation is used, because StatLog neighbourhoods overlap and random folds would leak pixels between training and validation samples. Test scores are listed for transparency only.

Round 1 (9 kernels) was specified first. Round 2 (3 kernels: log inputs and a centre-pixel term) was added after round-1 test results had been seen; selection still uses evidence alone, across both rounds.

| Candidate kernel | Round | Hyper-params | Log evidence | BIC-adjusted | Test OA | AA | kappa | AURC | NLL | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| `matern52_band_orbit_invariant` **(selected)** | 1 | 13 | -22138.6 | -22188.0 | 91.40 | 89.54 | 89.41 | 1.20 | 0.230 | 2.10 |
| `rbf_band_orbit_invariant_plus_centre` | 2 | 18 | -22164.0 | -22232.5 | 91.20 | 89.19 | 89.16 | 1.23 | 0.231 | 1.65 |
| `matern52_band_orbit_invariant_log` | 2 | 13 | -22194.2 | -22243.6 | 91.45 | 89.49 | 89.47 | 1.20 | 0.229 | 1.57 |
| `rbf_band_orbit_invariant` | 1 | 13 | -22228.9 | -22278.3 | 91.05 | 89.03 | 88.98 | 1.31 | 0.237 | 1.50 |
| `rbf_band_invariant` | 1 | 5 | -22274.0 | -22293.0 | 90.95 | 89.00 | 88.85 | 1.31 | 0.238 | 1.78 |
| `rbf_band_orbit_invariant_log` | 2 | 13 | -22314.7 | -22364.1 | 91.25 | 89.31 | 89.22 | 1.36 | 0.240 | 1.69 |
| `matern52_ard` | 1 | 37 | -22344.5 | -22485.1 | 91.15 | 89.25 | 89.11 | 1.43 | 0.239 | 1.44 |
| `matern52_band_orbit` | 1 | 13 | -22403.1 | -22452.5 | 91.15 | 89.33 | 89.11 | 1.42 | 0.239 | 2.03 |
| `rbf_ard` | 1 | 37 | -22435.6 | -22576.2 | 90.80 | 89.01 | 88.68 | 1.58 | 0.248 | 1.87 |
| `rbf_band_orbit` | 1 | 13 | -22493.8 | -22543.2 | 91.05 | 89.15 | 88.98 | 1.59 | 0.247 | 2.00 |
| `rbf_band` | 1 | 5 | -22524.8 | -22543.8 | 91.00 | 89.18 | 88.92 | 1.55 | 0.244 | 1.79 |
| `rbf_iso` | 1 | 2 | -22589.5 | -22597.1 | 90.95 | 89.12 | 88.86 | 1.62 | 0.245 | 1.80 |

## Final model

The selected kernel `matern52_band_orbit_invariant`, with its hyper-parameters re-learned by maximum likelihood on all 4,435 training samples (warm-started from the subset optimum; 38 L-BFGS iterations, 3266 s).

| Metric (test) | Value |
|---|---|
| Overall accuracy | **91.65%** |
| Average accuracy | 89.60% |
| Cohen's kappa | 89.71 |
| Macro F1 | 90.17 |
| AURC | 1.17% |
| Negative log-likelihood | 0.229 |
| Expected calibration error | 1.68% |

Per-class accuracy: class 1: 99.6%, class 2: 97.8%, class 3: 94.2%, class 4: 61.1%, class 5: 93.2%, class 7: 91.7%.

## Comparison (test overall accuracy, %)

| Method | Test OA |
|---|---|
| L2-loss Crammer-Singer SVM, Lee & Lin, as quoted | 92.45 |
| Crammer-Singer SVM, Hsu & Lin (2002), as quoted | 92.35 |
| **GP, `matern52_band_orbit_invariant`, final model** | 91.65 |
| RBF SVM, earlier version of this repository (commit 126d53a) | 91.50 |
| GP, `matern52_band_orbit_invariant`, subset hyper-parameters | 91.40 |
| Random Forest, earlier version of this repository (commit 126d53a) | 91.20 |

## Post-hoc checks on the test set (not used for any choice)

**alpha_eps sensitivity.** The final model is refitted with each alpha_eps, keeping its kernel hyper-parameters fixed. This shows how much the one fixed setting could matter. Picking the best row would be tuning on the test set.

| alpha_eps | Test OA | AA | kappa | AURC | NLL | ECE |
|---|---|---|---|---|---|---|
| 0.001 | 91.50 | 89.45 | 89.52 | 1.17 | 0.255 | 4.01 |
| 0.003 | 91.60 | 89.53 | 89.65 | 1.17 | 0.238 | 3.44 |
| 0.01 | 91.65 | 89.60 | 89.71 | 1.17 | 0.229 | 1.68 |
| 0.03 | 91.65 | 89.69 | 89.71 | 1.19 | 0.252 | 3.92 |
| 0.1 | 91.70 | 89.80 | 89.78 | 1.23 | 0.396 | 17.55 |

**Statistical resolution.** With 2,000 test samples, the standard error of an accuracy near 91.65% is 0.62 points. Differences of a few tenths of a point between methods are therefore within noise.

