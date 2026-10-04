"""Validation-tuned Dirichlet GP ("GP v2") on all datasets, compared with an RBF-SVM reference.

    python -m dirichlet_gp.tuned run --datasets all --seeds 10
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
from scipy.stats import binomtest

from .data import CompositeFeatures
from .datasets import LOADERS, BenchmarkDataset
from .gp import GPDirichletClassifier, GroupedKernel, SumKernel, contiguous_groups
from .metrics import calibration, classification_metrics, selective_summary
from .priors import class_frequencies
from .selection import grid_search

OUT = Path("results/tuned")
METHOD = "GP v2 (validation-tuned Dirichlet GP)"
SVM = "RBF SVM (reference)"
RULE = "lognormal"
ML_ALPHA_EPS = 0.01
DATASET_ORDER = list(LOADERS)


# ------------------------------------------------------------------ models
def composite_spectral(n_bands: int, family: str, z_groups: int = 10, deriv_groups: int = 5):
    """Sum kernel on [z-scored spectrum | unit-norm spectrum (angle) | smoothed first derivative].

    The z block has ``z_groups`` contiguous length-scale groups, the angle block one
    length-scale, the derivative block ``deriv_groups`` contiguous groups; each term
    has its own signal variance.
    """
    feats = CompositeFeatures(("z", "angle", "deriv"))
    n = 3 * n_bands - 1
    blocks = {"z": (0, n_bands), "angle": (n_bands, 2 * n_bands), "deriv": (2 * n_bands, n)}
    within = {"z": contiguous_groups(n_bands, z_groups), "angle": np.zeros(n_bands, int),
              "deriv": contiguous_groups(n_bands - 1, deriv_groups)}
    kernels = []
    for b, (lo, hi) in blocks.items():
        g = np.full(n, -1)
        g[lo:hi] = within[b]
        kernels.append(GroupedKernel(g, family))
    return feats, SumKernel(*kernels)


def gp_for(ds: BenchmarkDataset, seed: int) -> GPDirichletClassifier:
    common = {"alpha_eps": ML_ALPHA_EPS, "rule": RULE, "random_state": seed, "max_iter": 200}
    if ds.key in ("indian_pines", "pavia_university", "salinas"):
        feats, kernel = composite_spectral(ds.X.shape[1], "matern52")
        return GPDirichletClassifier(features=feats, kernel=kernel, n_opt=10**9, **common)
    if ds.key == "sentinel2_breizhcrops":
        return GPDirichletClassifier(representation="standard", groups=np.arange(ds.X.shape[1]) % 10,
                                     kernel="rbf", n_opt=10**9, **common)
    raise ValueError(f"no validation-tuned configuration for {ds.key}")


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
        best, table = grid_search(gp, ds.X[va], ds.y[va], priors=priors)
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


def run(datasets, seeds: int, out: Path) -> None:
    raw, preds = out / "raw", out / "predictions"
    raw.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    for key in datasets:
        ds = LOADERS[key]()
        rows, infos = [], []
        for seed in range(1 if ds.split(0)[1] is None else seeds):  # StatLog: deterministic, one run
            r = evaluate(ds, seed, preds)
            rows.append(r["row"])
            infos.append(r["info"])
            print(f"[{key}] seed {seed}: OA {100 * r['row']['OA']:.2f}% ({r['info']['seconds']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(raw / f"{key}_runs.csv", index=False)
            (raw / f"{key}_meta.json").write_text(json.dumps(
                {"key": key, "title": ds.title, "info": ds.info, "seeds": infos}, indent=1, default=_json_default))


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

    lines = ["# Validation-tuned Dirichlet GP vs RBF SVM", "",
             "Generated by `python -m dirichlet_gp.tuned report`. Protocol: `PROTOCOL.md` (fixed before the test "
             "runs). Mean ± standard deviation over seeds; test-set numbers. NLL is the mean negative log "
             "probability of the true class and is not multiplied by 100; all other columns are percentages.", "",
             "## Overall accuracy", "",
             "| Dataset | GP v2 | SVM | GP − SVM (paired mean) | seeds GP ahead / behind | "
             "seeds with McNemar p < 0.05 (GP / SVM better) |",
             "|---|---|---|---|---|---|"]
    for k in keys:
        pr = pairs[pairs.dataset == k]
        g, s = gp[gp.dataset == k], svm[svm.dataset == k]
        sig_g = int(((pr.mcnemar_p < 0.05) & (pr["diff"] > 0)).sum())
        sig_s = int(((pr.mcnemar_p < 0.05) & (pr["diff"] < 0)).sum())
        lines.append(f"| {metas[k]['title']} | {_cell(g.OA)} | {_cell(s.OA)} | {100 * pr['diff'].mean():+.2f} | "
                     f"{int((pr['diff'] > 0).sum())} / {int((pr['diff'] < 0).sum())} | {sig_g} / {sig_s} |")
    for name, table in (("GP v2", gp), ("SVM", svm)):
        lines += ["", f"## All metrics: {name}", "", "| Dataset | " + " | ".join(n for _, n in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for k in keys:
            r = table[table.dataset == k]
            cells = [_cell(r[c], 1.0, 3) if c == "NLL" else _cell(r[c]) for c, _ in METRICS]
            lines.append(f"| {metas[k]['title']} | " + " | ".join(cells) + " |")
    lines += ["", "## Selected settings (GP v2)", "",
              "| Dataset | alpha_eps (per seed) | length-scale multiplier | temperature | prior exponent |",
              "|---|---|---|---|---|"]
    for k in keys:
        sd = metas[k]["seeds"]
        lines.append(f"| {metas[k]['title']} | {', '.join(str(x['alpha_eps']) for x in sd)} | "
                     f"{', '.join(str(x['scale']) for x in sd)} | "
                     f"{', '.join(format(x['temperature'], '.2f') for x in sd)} | "
                     f"{', '.join(str(x.get('prior_tau', '-')) for x in sd)} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"report written to {out / 'summary.md'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", default="all")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--out", type=Path, default=OUT)
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        keys = DATASET_ORDER if a.datasets == "all" else [k.strip() for k in a.datasets.split(",")]
        run(keys, a.seeds, a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
