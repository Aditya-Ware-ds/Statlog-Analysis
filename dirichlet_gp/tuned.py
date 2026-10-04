"""Validation-tuned Dirichlet GP ("GP v2") on all datasets, compared with an RBF-SVM reference.

    python -m dirichlet_gp.tuned run --datasets all --seeds 10      # or --seeds 0-4, --seeds 5-9 in parallel
    python -m dirichlet_gp.tuned report

The protocol is fixed in ``results/tuned/PROTOCOL.md`` (written before any test
evaluation of this model). Per dataset and seed:

1. kernel hyper-parameters by type-II maximum likelihood on *all* training samples
   (alpha_eps = 0.01; composite spectral kernel on the hyperspectral scenes);
2. where a held-out split exists: (alpha_eps, length-scale multiplier) on held-out
   accuracy, then a temperature on held-out log-loss (``dirichlet_gp.selection``);
3. Sentinel-2 only: class-prior correction towards the training-side department
   frequencies, exponent chosen on the held-out department (``dirichlet_gp.priors``);
4. one evaluation on the test samples.

StatLog has no held-out split and is not cross-validated (overlapping 3x3
neighbourhoods would leak pixels between folds): it uses the frozen model of
``dirichlet_gp.statlog_study`` (kernel chosen by evidence, hyper-parameters
learned on all training samples), with only the decision rule changed to the one
used everywhere else (fixed before testing).

The SVM reference results (``results/tuned/svm_reference/``) were produced
outside this repository with the protocol described in PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

from .data import CompositeFeatures
from .datasets import LOADERS, BenchmarkDataset
from .gp import GPDirichletClassifier, GroupedKernel, LinearKernel, SumKernel
from .gpv2 import composite_spectral
from .metrics import calibration, classification_metrics, selective_summary
from .priors import class_frequencies
from .selection import ALPHAS, SCALES, grid_search

OUT = Path("results/tuned")
METHOD = "GP v2 (validation-tuned Dirichlet GP)"
SVM = "RBF SVM (reference)"
RULE = "lognormal"
ML_ALPHA_EPS = 0.01
DATASET_ORDER = list(LOADERS)


# ------------------------------------------------------------------ models
def gp_for(ds: BenchmarkDataset, seed: int) -> GPDirichletClassifier:
    common = {"alpha_eps": ML_ALPHA_EPS, "rule": RULE, "random_state": seed, "max_iter": 200}
    if ds.key in ("indian_pines", "pavia_university", "salinas"):
        feats, kernel = composite_spectral(ds.X.shape[1], "matern52")
        return GPDirichletClassifier(features=feats, kernel=kernel, n_opt=10**9, **common)
    if ds.key == "sentinel2_breizhcrops":
        feats, kernel = sentinel2_kernel(SENTINEL2_KERNEL)
        return GPDirichletClassifier(features=feats, kernel=kernel, n_opt=10**9, **common)
    raise ValueError(f"no validation-tuned configuration for {ds.key}")


#: Sentinel-2 kernel, chosen on held-out departments only (PROTOCOL.md, section 5b)
SENTINEL2_KERNEL = "band+angle+linear"
#: Sentinel-2 length-scale multipliers: 2^(k/2), k = -3..8 (0.35 ... 16); wider than SCALES because the
#: selected settings make the stationary term nearly flat next to the linear one (PROTOCOL.md, 5a)
SENTINEL2_SCALES = tuple(float(x) for x in np.round(2.0 ** (np.arange(-3, 9) / 2), 3))


def sentinel2_kernel(variant: str):
    """Features and kernel for the 60 bi-monthly Sentinel-2 values (period-major, 10 bands per period).

    ``"band+linear"``: RBF with one length-scale per band (shared by the 6 periods) + a linear kernel
    with one weight per band. ``"band+angle+linear"``: the same plus an RBF on the unit-norm series
    (spectral-angle block, one length-scale).
    """
    from .data import Standardizer

    band = np.arange(60) % 10
    if variant == "band+linear":
        return Standardizer("standard"), SumKernel(GroupedKernel(band, "rbf"), LinearKernel(band))
    if variant == "band+angle+linear":
        z = np.r_[band, np.full(60, -1)]
        angle = np.r_[np.full(60, -1), np.zeros(60, int)]
        kernels = (GroupedKernel(z, "rbf"), GroupedKernel(angle, "rbf"), LinearKernel(z))
        return CompositeFeatures(("z", "angle")), SumKernel(*kernels)
    raise ValueError(f"unknown Sentinel-2 kernel {variant!r}")


def statlog_frozen(seed: int) -> GPDirichletClassifier:
    from .statlog_study import make

    fin = json.loads((Path("results/gp_statlog") / "final.json").read_text())
    m = make(fin["candidate"], theta=fin["theta"], alpha_eps=ML_ALPHA_EPS, cache=False)
    m.rule = RULE
    m.random_state = seed
    return m


# ------------------------------------------------------------------ evaluation
def evaluate(ds: BenchmarkDataset, seed: int, preds: Path) -> dict:
    tr, va, te = ds.split(seed)
    t0 = time.time()
    info = {"seed": seed, "n_train": len(tr), "n_heldout": 0 if va is None else len(va), "n_test": len(te)}
    if va is None:
        gp = statlog_frozen(seed).fit(ds.X[tr], ds.y[tr])
        info.update(selection="none (frozen StatLog model)", alpha_eps=gp.alpha_eps, scale=1.0, temperature=1.0)
    else:
        gp = gp_for(ds, seed).fit(ds.X[tr], ds.y[tr])
        info.update(ml_iterations=int(gp.opt_result_.nit), ml_seconds=round(time.time() - t0, 1),
                    log_evidence=gp.log_marginal_likelihood(gp.theta_ml_),
                    theta_ml=gp.theta_ml_.round(4).tolist())
        priors = None
        if ds.prior_pool is not None:
            priors = (class_frequencies(ds.y[tr], gp.classes_), class_frequencies(ds.y[ds.prior_pool], gp.classes_))
            info["prior_target"] = priors[1].round(5).tolist()
        scales = SENTINEL2_SCALES if ds.key == "sentinel2_breizhcrops" else SCALES
        best, table = grid_search(gp, ds.X[va], ds.y[va], alphas=ALPHAS, scales=scales, priors=priors)
        info.update(alpha_eps=best["alpha_eps"], scale=best["scale"], val_accuracy=best["accuracy"],
                    temperature=best["temperature"], prior_tau=best.get("prior_tau"), grid=table)
    P = gp.predict_proba(ds.X[te])
    info["seconds"] = round(time.time() - t0, 1)
    yte = ds.y[te]
    pred = gp.classes_[P.argmax(axis=1)]
    np.savez_compressed(preds / f"{ds.key}_s{seed}.npz", pred=pred.astype(np.int16))
    row = {"dataset": ds.key, "seed": seed, "method": METHOD, **score(yte, P, gp.classes_, ds.classes)}
    return {"row": row, "info": info}


def score(y: np.ndarray, P: np.ndarray, model_classes, classes) -> dict:
    pred = np.asarray(model_classes)[P.argmax(axis=1)]
    cm, _ = classification_metrics(y, pred, classes)
    sel = selective_summary(pred == y, P.max(axis=1))
    yi = np.searchsorted(model_classes, y)
    nll, ece = calibration(P, yi)
    return {**cm, "AURC": sel["aurc"], "risk@80": sel["risk@80"], "NLL": nll, "ECE": ece}


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def run(datasets, seeds, out: Path) -> None:
    """Evaluate every seed in ``seeds``; each seed is saved on its own (raw/<dataset>/seed_<n>.json), so
    several processes can share the seeds of one dataset. StatLog is deterministic and runs once (seed 0)."""
    preds = out / "predictions"
    preds.mkdir(parents=True, exist_ok=True)
    for key in datasets:
        ds = LOADERS[key]()
        (out / "raw" / key).mkdir(parents=True, exist_ok=True)
        for seed in ([0] if ds.split(0)[1] is None else seeds):
            r = evaluate(ds, seed, preds)
            (out / "raw" / key / f"seed_{seed}.json").write_text(json.dumps(r, indent=1, default=_json_default))
            print(f"[{key}] seed {seed}: OA {100 * r['row']['OA']:.2f}% ({r['info']['seconds']:.0f}s)", flush=True)
        collect(ds, out)


def collect(ds: BenchmarkDataset, out: Path) -> None:
    """Merge the per-seed files of one dataset into raw/<dataset>_runs.csv and raw/<dataset>_meta.json."""
    files = sorted((out / "raw" / ds.key).glob("seed_*.json"), key=lambda f: int(f.stem.split("_")[1]))
    results = [json.loads(f.read_text()) for f in files]
    pd.DataFrame([r["row"] for r in results]).to_csv(out / "raw" / f"{ds.key}_runs.csv", index=False)
    (out / "raw" / f"{ds.key}_meta.json").write_text(json.dumps(
        {"key": ds.key, "title": ds.title, "info": ds.info, "seeds": [r["info"] for r in results]},
        indent=1, default=_json_default))


def _seed_list(text: str) -> list[int]:
    """"10" -> 0..9; "3-5" -> 3, 4, 5; "1,4" -> 1, 4."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    if "," in text:
        return [int(x) for x in text.split(",")]
    return list(range(int(text)))


# ------------------------------------------------------------------ report
def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar test: (#only A right, #only B right, two-sided p)."""
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def paired(out: Path, key: str) -> pd.DataFrame:
    ds = LOADERS[key]()
    rows = []
    svm_seeds = sorted(int(p.stem.split("_s")[-1]) for p in (out / "svm_reference" / "predictions").glob(f"{key}_s*.npz"))
    for seed in svm_seeds:
        _, _, te = ds.split(seed)
        y = ds.y[te]
        gp_file = out / "predictions" / f"{key}_s{seed if ds.split(0)[1] is not None else 0}.npz"
        if not gp_file.exists():
            continue
        g = np.load(gp_file)["pred"] == y
        s = np.load(out / "svm_reference" / "predictions" / f"{key}_s{seed}.npz")["pred"] == y
        n10, n01, p = mcnemar(g, s)
        rows.append({"dataset": key, "seed": seed, "GP_OA": g.mean(), "SVM_OA": s.mean(), "diff": g.mean() - s.mean(),
                     "only_GP_right": n10, "only_SVM_right": n01, "mcnemar_p": p})
    return pd.DataFrame(rows)


def _cell(series: pd.Series, scale=100.0, digits=2) -> str:
    m, s = scale * series.mean(), scale * series.std(ddof=1)
    return f"{m:.{digits}f}" if len(series) < 2 or not s > 0 else f"{m:.{digits}f} ± {s:.{digits}f}"


METRICS = [("OA", "OA"), ("AA", "AA"), ("kappa", "kappa x100"), ("macro_F1", "macro F1 x100"), ("AURC", "AURC"),
           ("NLL", "NLL"), ("ECE", "ECE")]


def report(out: Path) -> None:
    keys = [k for k in DATASET_ORDER if (out / "raw" / f"{k}_runs.csv").exists()]
    gp = pd.concat([pd.read_csv(out / "raw" / f"{k}_runs.csv") for k in keys], ignore_index=True)
    svm = pd.read_csv(out / "svm_reference" / "runs.csv")
    metas = {k: json.loads((out / "raw" / f"{k}_meta.json").read_text()) for k in keys}
    pairs = pd.concat([paired(out, k) for k in keys], ignore_index=True)
    pairs.to_csv(out / "paired_comparison.csv", index=False)
    v1_dir = Path("results/benchmark/raw")
    v1 = {k: pd.read_csv(v1_dir / f"{k}_runs.csv") for k in keys if (v1_dir / f"{k}_runs.csv").exists()}

    lines = ["# Validation-tuned Dirichlet GP vs RBF SVM", "",
             "Generated by `python -m dirichlet_gp.tuned report`. The protocol (`PROTOCOL.md`) was fixed before the "
             "test runs. Test-set numbers, mean ± standard deviation over seeds 0-9. NLL is the mean negative log "
             "probability of the true class and is not multiplied by 100; all other columns are percentages.", "",
             "## Overall accuracy", "",
             "| Dataset | GP v1 (evidence only) | **GP v2** | SVM | GP v2 − SVM (paired mean) | "
             "seeds GP v2 ahead / behind | seeds with McNemar p < 0.05 (GP v2 / SVM better) | "
             "paired t-test over seeds, p (not pre-registered) |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        pr = pairs[pairs.dataset == k]
        g, s = gp[gp.dataset == k], svm[svm.dataset == k]
        sig_g = int(((pr.mcnemar_p < 0.05) & (pr["diff"] > 0)).sum())
        sig_s = int(((pr.mcnemar_p < 0.05) & (pr["diff"] < 0)).sum())
        old = _cell(v1[k].OA) if k in v1 else "-"
        t_p = f"{ttest_1samp(pr['diff'], 0.0).pvalue:.2g}" if len(g) > 1 and len(pr) > 2 else "-"
        lines.append(f"| {metas[k]['title']} | {old} | **{_cell(g.OA)}** | {_cell(s.OA)} | "
                     f"{100 * pr['diff'].mean():+.2f} | {int((pr['diff'] > 0).sum())} / {int((pr['diff'] < 0).sum())} | "
                     f"{sig_g} / {sig_s} | {t_p} |")
    lines += ["",
              "GP v1 is the evidence-only GP of `results/benchmark/` (5 seeds). On StatLog GP v2 is one "
              "deterministic run (the frozen model of `results/gp_statlog/`); it is compared with each of the 10 "
              "SVM runs, which differ only in the random state of libsvm's probability calibration. McNemar: exact "
              "binomial test on the test samples that exactly one of the two models classifies correctly "
              "(per-seed counts in `paired_comparison.csv`). The paired t-test treats the seeds (independent "
              "draws of the training and held-out samples) as the unit; it was added after the protocol was fixed, "
              "because on the 122,708-parcel Sentinel-2 test set McNemar flags even 0.3-point differences in either "
              "direction, so variation between training draws, not test-set noise, decides there. It is not "
              "computed for StatLog, where the GP is a single deterministic run."]
    for name, table in (("GP v2", gp), ("SVM", svm)):
        lines += ["", f"## All metrics: {name}", "", "| Dataset | " + " | ".join(n for _, n in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for k in keys:
            r = table[table.dataset == k]
            cells = [_cell(r[c], 1.0, 3) if c == "NLL" else _cell(r[c]) for c, _ in METRICS]
            lines.append(f"| {metas[k]['title']} | " + " | ".join(cells) + " |")
    comp = out / "svm_reference" / "composite_features_runs.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        lines += ["", "## Supplementary: SVM on the GP v2 composite features", "",
                  "The hyperspectral GP v2 sees z-scored bands, a spectral-angle block and a derivative block. "
                  "To check whether these features alone explain its lead, the SVM protocol was repeated on the "
                  "same concatenated features (one RBF on all of them, same grids and selection).", "",
                  "| Dataset | SVM, z-scored bands | SVM, composite features | GP v2 |", "|---|---|---|---|"]
        for k in keys:
            if k in set(c.dataset):
                lines.append(f"| {metas[k]['title']} | {_cell(svm[svm.dataset == k].OA)} | "
                             f"{_cell(c[c.dataset == k].OA)} | {_cell(gp[gp.dataset == k].OA)} |")
    lines += ["", "## Selected settings (GP v2)", "",
              "| Dataset | alpha_eps (per seed) | length-scale multiplier | temperature | prior exponent |",
              "|---|---|---|---|---|"]
    for k in keys:
        sd = metas[k]["seeds"]
        taus = ["-" if x.get("prior_tau") is None else str(x["prior_tau"]) for x in sd]
        lines.append(f"| {metas[k]['title']} | {', '.join(str(x['alpha_eps']) for x in sd)} | "
                     f"{', '.join(str(x['scale']) for x in sd)} | "
                     f"{', '.join(format(x['temperature'], '.2f') for x in sd)} | {', '.join(taus)} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"report written to {out / 'summary.md'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", default="all")
    r.add_argument("--seeds", default="10", help='number of seeds ("10" = 0-9), a range "3-5" or a list "1,4"')
    r.add_argument("--out", type=Path, default=OUT)
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        keys = DATASET_ORDER if a.datasets == "all" else [k.strip() for k in a.datasets.split(",")]
        run(keys, _seed_list(a.seeds), a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
