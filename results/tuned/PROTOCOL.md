# Protocol: validation-tuned Dirichlet GP ("GP v2") vs RBF SVM

This file fixes the method, the selection rules and the comparison **before any
test-set evaluation of GP v2**. Sections are dated in the order they were fixed.
The code is `dirichlet_gp/tuned.py`, `dirichlet_gp/selection.py` and
`dirichlet_gp/priors.py`.

## 1. How the design was chosen (disclosure)

* The candidate components came from the literature review in
  `reports/Gaussian process variants beating SVM.md`, which is not in the repository.
  The components were: full-data type-II ML, validation-chosen smoothing,
  composite spectral kernels, the log-normal decision rule, and the class-prior
  correction for Sentinel-2.
* The components were compared on **seed-0 held-out (validation) samples only**,
  on Indian Pines, Pavia University, Salinas and Sentinel-2. No test sample was
  used. Prototype results on validation accuracy, seed 0, are below. "+grid" means
  after the alpha_eps x length-scale grid of section 3. The hyperspectral configuration
  was fixed from Indian Pines and Pavia University, where both prototypes were complete.
  The Salinas prototype was still running; it is reported here but did not decide anything.

  | Scene | v1 kernel | + full-data ML | + angle block | + derivative block | same, Matérn-5/2 | SVM (refined grid) |
  |---|---|---|---|---|---|---|
  | Indian Pines, default / +grid | 83.37 / 85.30 | 83.37 / 85.30 | 83.75 / 85.11 | 84.53 / 85.49 | 84.53 / **85.88** | 82.21 |
  | Pavia University, default / +grid | 91.02 / 93.07 | 91.02 / 93.07 | 92.75 / 94.01 | 93.12 / 94.34 | 93.50 / **94.53** | 93.96 |
  | Salinas, default / +grid | 92.28 / 92.94 | 92.35 / 92.76 | 92.76 / 93.75 | (running) | (running) | 93.75 |

* The SVM reference runs (section 4) were run while the GP design was being
  prototyped, and their test accuracies were printed to the console and seen
  then. They were not used for any GP choice; every GP choice above is a
  validation number. On StatLog, the earlier study's test-set alpha_eps
  sensitivity table (`results/gp_statlog/summary.md`) had been seen before this
  protocol. **For that reason StatLog keeps alpha_eps = 0.01 unchanged.**

## 2. Data, splits, seeds

The datasets and splits are those of `dirichlet_gp.datasets`, unchanged from the
v1 benchmark:

* **StatLog:** the official split.
* **Hyperspectral scenes** (Indian Pines, Pavia University, Salinas): stratified
  random train / held-out / test splits.
* **Sentinel-2:** a spatial split by department. Training comes from
  FRH01 + FRH02 (capped per class), held-out from FRH03 (capped per class), and
  the test set is all of FRH04.

Seeds 0–9. The held-out split, previously unused, now drives selection for both
models.

## 3. GP v2 (fixed 2026-10-04, before any GP v2 test evaluation)

**Hyperspectral scenes** (one configuration for all three):

1. **Features:** `CompositeFeatures(("z", "angle", "deriv"))` built from training
   statistics.
   * z-scored bands;
   * the unit-L2-norm spectrum, centred and scaled by one global factor (a
     spectral-angle block);
   * the first difference of the 3-band-smoothed spectrum, z-scored.
2. **Kernel:** a sum of three Matérn-5/2 kernels, one per block, each with its own
   signal variance.
   * z block: 10 contiguous length-scale groups;
   * angle block: 1 length-scale;
   * derivative block: 5 contiguous groups.
3. **Hyper-parameters:** type-II maximum likelihood on **all** training samples at
   alpha_eps = 0.01 (L-BFGS-B, at most 200 iterations).
4. **Selection on the held-out samples.** Grid:
   * alpha_eps ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1};
   * length-scale multiplier s ∈ {0.35, 0.5, 0.71, 1, 1.41, 2}.

   For each pair, the model is re-conditioned without re-optimising. The
   temperature is fitted on held-out log-loss, and the pair is scored by
   held-out accuracy. Highest accuracy wins; ties go to the lower held-out
   log-loss. 36 configurations.
5. **Decision rule:** "lognormal", p_c ∝ exp((mu_c + var_c / 2) / T). It is
   deterministic, and the temperature does not change the predicted class.
6. The model is trained on the training split only. No refit on train + held-out
   (the SVM does the same).

**StatLog** (no held-out split; no cross-validation, because overlapping 3x3
neighbourhoods would leak pixels between folds):

* The frozen final model of the earlier evidence-based study, from
  `results/gp_statlog/final.json`:
  * Matérn-5/2, 12 band x pixel-orbit length-scales, invariant to the 8 symmetries
    of the window;
  * hyper-parameters learned by ML on all 4,435 training samples;
  * alpha_eps = 0.01, s = 1, T = 1.
* The only change is the decision rule, "lognormal" instead of Monte-Carlo
  softmax, fixed a priori for every dataset. The model is deterministic, so it is
  run once.

**Sentinel-2:** see section 5.

## 4. SVM reference (run outside this repository; results in `svm_reference/`)

* **Model:** RBF SVM (scikit-learn `SVC`) on z-scored features, with training
  statistics.
* **HSI selection:** C × gamma chosen by held-out accuracy.
  1. Decade grid: C ∈ {1, 10, 100, 1000}, gamma ∈ {1e-4, …, 1}.
  2. Refinement: a quarter-decade 5 × 5 grid (multipliers 10^{-0.5..0.5}) around
     the best point.

  44 distinct configurations in total; ties go to the first found.
* **Sentinel-2 selection:** the same grids, but each point is scored like the GP's
  (section 5).
* **Final model:** `SVC(probability=True, random_state=seed)` on the training
  split. The predicted class is the argmax of the libsvm (Platt + pairwise
  coupling) probabilities, as in the published reference.
* **StatLog:** C = 10, gamma = 0.1. This is the published choice, made by 5-fold
  CV on the training set. It is kept as the bar although that CV leaks pixels,
  which favours the SVM.

## 5. Sentinel-2 (fixed before any Sentinel-2 GP v2 test evaluation)

**Deployment priors.** The test department keeps its natural crop frequencies,
while the training and held-out samples are capped per class. Both models
therefore get the same handling:

* **pi_target:** the class frequencies of all labelled parcels in FRH01–FRH03.
  These are training-side departments only.
* **Prior correction:** p ∝ p · (pi_target / pi_train)^tau.
  * tau ∈ {0, 0.25, 0.5, 0.75, 1} is chosen on the held-out department by
    log-loss.
  * Held-out samples are weighted by pi_target / pi_held-out.
* **Selection criterion, both models:** held-out accuracy after the full
  post-processing.
  * GP: temperature, then prior correction.
  * SVM: libsvm probabilities, then prior correction.
  * Held-out samples are weighted to pi_target.

  This replaces plain held-out accuracy because the held-out department is
  class-balanced and the test department is not. In the seed-0 GP prototype, the
  plain-accuracy choice scored 68.5% on the weighted criterion, against 72.6% for
  the best setting.

GP v2 kernel for Sentinel-2: *to be fixed below from the seed-0 validation
prototype, before the Sentinel-2 test runs.*

## 6. Comparison and claims

* **Primary endpoint:** test overall accuracy (OA), mean over seeds 0–9. StatLog
  has one GP run, compared with each of the 10 SVM runs.
* **Reported for every dataset:**
  * the mean paired difference (GP − SVM, same seed, same test samples);
  * the number of seeds on which each model is ahead;
  * an exact McNemar test per seed.
* **Wording of claims:**
  * "GP v2 beats the SVM on a dataset": the mean paired OA difference is
    positive.
  * "significantly": additionally, McNemar p < 0.05 on at least 6 of 10 seeds.
* **Secondary metrics:** AA, kappa, macro-F1, AURC, NLL, ECE.
* **Results are reported as they come out**, including any dataset where GP v2
  does not beat the SVM. No setting is changed after a test result has been seen.
