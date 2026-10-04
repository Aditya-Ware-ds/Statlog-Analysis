"""Gaussian-process classifier on StatLog: kernel selection by marginal likelihood.

StatLog samples are overlapping 3x3 neighbourhoods cut from one image, so
random cross-validation folds share pixels between training and validation
samples and give optimistic estimates. No data split is therefore used for
model selection. Every candidate kernel is fitted by type-II maximum
likelihood on the official training set, and the candidate with the highest
log marginal likelihood (evidence, computed on the same 2,000-sample training
subset for all candidates) is selected. The test set is used once per
candidate, only to report results; it plays no part in the selection.

    python -m sg_hfq.gp_statlog run --candidates all       # or a comma list
    python -m sg_hfq.gp_statlog final                      # refit the selected kernel on all training data
    python -m sg_hfq.gp_statlog sensitivity                # post-hoc test-set checks (not used for choices)
    python -m sg_hfq.gp_statlog report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import classification_metrics
from .data import CLASS_CODES, load_statlog
from .gp import GPDirichletClassifier, GroupedKernel, SumKernel, statlog_feature_permutations, statlog_groups
from .metrics import selective_summary

OUT = Path("results/gp_statlog")
ALPHA_EPS = 0.01  # Dirichlet concentration, fixed a priori (Milios et al., 2018)
N_OPT = 2000

#: name -> (representation, length-scale groups, base kernel, dihedral invariance, add centre-pixel ARD kernel)
#: Round 1 was specified first; round 2 was added after round-1 results had been seen (disclosed in the
#: report). Selection is by evidence over both rounds together.
CANDIDATES = {
    # round 1
    "rbf_iso": ("full36", "iso", "rbf", False, False),
    "rbf_band": ("full36", "band", "rbf", False, False),
    "rbf_band_orbit": ("full36", "band_orbit", "rbf", False, False),
    "rbf_ard": ("full36", "ard", "rbf", False, False),
    "rbf_band_invariant": ("full36", "band", "rbf", True, False),
    "rbf_band_orbit_invariant": ("full36", "band_orbit", "rbf", True, False),
    "matern52_band_orbit": ("full36", "band_orbit", "matern52", False, False),
    "matern52_ard": ("full36", "ard", "matern52", False, False),
    "matern52_band_orbit_invariant": ("full36", "band_orbit", "matern52", True, False),
    # round 2
    "rbf_band_orbit_invariant_log": ("log36", "band_orbit", "rbf", True, False),
    "matern52_band_orbit_invariant_log": ("log36", "band_orbit", "matern52", True, False),
    "rbf_band_orbit_invariant_plus_centre": ("full36", "band_orbit", "rbf", True, True),
}
ROUND_2 = {"rbf_band_orbit_invariant_log", "matern52_band_orbit_invariant_log", "rbf_band_orbit_invariant_plus_centre"}

#: Reference results on the same official split (test OA, %).
REFERENCES = {
    "SVM (RBF), this repo, 36-D": 91.5,
    "Random Forest, this repo, 36-D": 91.2,
    "Crammer-Singer SVM, Hsu & Lin (2002), as quoted": 92.35,
    "L2-loss Crammer-Singer SVM, Lee & Lin, as quoted": 92.45,
}


def make(name: str, n_opt: int = N_OPT, cache: bool = True, theta_init=None, theta=None,
         alpha_eps: float = ALPHA_EPS) -> GPDirichletClassifier:
    rep, groups, kind, invariant, centre = CANDIDATES[name]
    perms = statlog_feature_permutations() if invariant else None
    kernel = kind
    if centre:
        centre_groups = np.full(36, -1)
        centre_groups[16:20] = np.arange(4)  # the 4 bands of the centre pixel, one length-scale each
        kernel = SumKernel(GroupedKernel(statlog_groups(groups), kind, perms), GroupedKernel(centre_groups, kind))
    return GPDirichletClassifier(
        representation=rep, groups=statlog_groups(groups), kernel=kernel, permutations=perms,
        alpha_eps=alpha_eps, n_opt=n_opt, max_iter=200, random_state=0, cache=cache, theta_init=theta_init,
        theta=theta,
    )


def calibration(P: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> tuple[float, float]:
    """Negative log-likelihood and expected calibration error (top-label, equal-width bins)."""
    nll = float(-np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, None))))
    conf, pred = P.max(1), P.argmax(1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(np.mean(pred[sel] == y_idx[sel]) - conf[sel].mean())
    return nll, float(ece)


def evaluate(m: GPDirichletClassifier, d) -> tuple[dict, np.ndarray, np.ndarray]:
    P = m.predict_proba(d.X_test)
    pred = m.classes_[P.argmax(1)]
    cm, recall = classification_metrics(d.y_test, pred, CLASS_CODES)
    sel = selective_summary(pred == d.y_test, P.max(1))
    nll, ece = calibration(P, np.searchsorted(m.classes_, d.y_test))
    metrics = {**cm, "AURC": sel["aurc"], "risk@80": sel["risk@80"], "NLL": nll, "ECE": ece,
               "per_class_accuracy": {str(c): a for c, a in recall.items()}}
    return metrics, P, pred


def _record(name: str, m: GPDirichletClassifier, fit_s: float, metrics: dict) -> dict:
    rep, groups, kind, invariant, centre = CANDIDATES[name]
    k = m.kernel_.n_params
    evidence = m.log_marginal_likelihood()
    return {
        "candidate": name, "round": 2 if name in ROUND_2 else 1, "representation": rep, "groups": groups,
        "kernel": kind, "invariant": invariant, "centre_kernel": centre, "n_hyperparameters": k,
        "n_opt": len(m._T_opt), "log_evidence": evidence,
        "bic_adjusted_evidence": evidence - 0.5 * k * np.log(len(m._T_opt)),
        "optimizer_iterations": int(m.opt_result_.nit) if m.opt_result_ is not None else 0,
        "optimizer_message": str(m.opt_result_.message) if m.opt_result_ is not None else "fixed",
        "fit_seconds": round(fit_s, 1), "theta": m.theta_.tolist(),
        "lengthscales": m.lengthscales_.round(4).tolist(), "signal_variance": m.signal_variance_, **metrics,
    }


def run_candidate(name: str, out: Path) -> dict:
    d = load_statlog()
    t = time.time()
    m = make(name).fit(d.X_train, d.y_train)
    metrics, _, _ = evaluate(m, d)
    res = _record(name, m, time.time() - t, metrics)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "candidates" / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(f"{name}: evidence {res['log_evidence']:.1f}", flush=True)
    return res


def final(out: Path) -> dict:
    """Refit the evidence-selected kernel with hyper-parameters learned on all 4,435 training samples."""
    rows = [json.loads(p.read_text()) for p in sorted((out / "candidates").glob("*.json"))]
    best = max(rows, key=lambda r: r["log_evidence"])
    d = load_statlog()
    t = time.time()
    m = make(best["candidate"], n_opt=len(d.y_train), cache=False, theta_init=best["theta"])
    m.max_iter = 100
    m.fit(d.X_train, d.y_train)
    metrics, P, pred = evaluate(m, d)
    res = _record(best["candidate"], m, time.time() - t, metrics)
    res["selected_from"] = len(rows)
    (out / "final.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(P, columns=[f"p_{c}" for c in m.classes_]).assign(true=d.y_test, pred=pred).to_csv(
        out / "final_test_probabilities.csv.gz", index=False)
    print(f"final {best['candidate']}: evidence (all training data) {res['log_evidence']:.1f}", flush=True)
    return res


SENSITIVITY_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1)


def sensitivity(out: Path) -> dict:
    """Post-hoc analyses on the test set (not used for any modelling choice).

    1. alpha_eps sensitivity of the final model, with its kernel hyper-parameters held fixed.
    2. Exact McNemar test of the final GP against the repo's SVM on the same test samples.
    """
    from scipy.stats import binomtest

    from .baselines import SupervisedReference

    fin = json.loads((out / "final.json").read_text())
    d = load_statlog()
    rows = []
    for a in SENSITIVITY_ALPHAS:
        m = make(fin["candidate"], theta=fin["theta"], alpha_eps=a).fit(d.X_train, d.y_train)
        metrics, _, _ = evaluate(m, d)
        rows.append({"alpha_eps": a, **{k: metrics[k] for k in ("OA", "AA", "kappa", "AURC", "NLL", "ECE")}})
        print(f"alpha_eps={a}: OA {100 * metrics['OA']:.2f}%", flush=True)
    gp_pred = pd.read_csv(out / "final_test_probabilities.csv.gz")["pred"].to_numpy()
    svm = SupervisedReference("svm", "full36", random_state=0).fit(d.X_train, d.y_train)
    svm_pred = svm.scores(d.X_test)["pred"]
    gp_ok, svm_ok = gp_pred == d.y_test, svm_pred == d.y_test
    b, c = int(np.sum(gp_ok & ~svm_ok)), int(np.sum(~gp_ok & svm_ok))
    res = {
        "alpha_sensitivity": rows,
        "mcnemar_vs_svm": {"svm_params": svm.best_params_, "svm_OA": float(svm_ok.mean()), "gp_OA": float(gp_ok.mean()),
                           "gp_right_svm_wrong": b, "gp_wrong_svm_right": c,
                           "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0},
        "accuracy_standard_error": float(np.sqrt(gp_ok.mean() * (1 - gp_ok.mean()) / len(gp_ok))),
    }
    (out / "sensitivity.json").write_text(json.dumps(res, indent=2))
    return res


def report(out: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted((out / "candidates").glob("*.json"))]
    df = pd.DataFrame(rows).sort_values("log_evidence", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    cols = ["candidate", "round", "n_hyperparameters", "log_evidence", "bic_adjusted_evidence", "OA", "AA",
            "kappa", "macro_F1", "AURC", "NLL", "ECE", "fit_seconds"]
    df[cols].to_csv(out / "candidates.csv", index=False)
    fin = json.loads((out / "final.json").read_text()) if (out / "final.json").exists() else None

    def pct(v):
        return f"{100 * v:.2f}"

    lines = [
        "# Gaussian-process classifier on StatLog (official split)",
        "",
        "Model: Dirichlet-based GP classification (Milios et al., NeurIPS 2018), exact inference, "
        f"alpha_eps = {ALPHA_EPS} fixed a priori. For each candidate kernel, the hyper-parameters are learned "
        f"by type-II maximum likelihood on the same class-stratified subset of {N_OPT} training samples.",
        "",
        "**Selection rule (fixed in advance): highest log marginal likelihood (evidence).** No "
        "cross-validation is used, because StatLog neighbourhoods overlap and random folds would leak "
        "pixels between training and validation samples. Test scores are listed for transparency only.",
        "",
        "Round 1 (9 kernels) was specified first. Round 2 (3 kernels: log inputs and a centre-pixel term) "
        "was added after round-1 test results had been seen; selection still uses evidence alone, across "
        "both rounds.",
        "",
        "| Candidate kernel | Round | Hyper-params | Log evidence | BIC-adjusted | Test OA | AA | kappa | AURC | NLL | ECE |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        mark = " **(selected)**" if r.candidate == best.candidate else ""
        lines.append(f"| `{r.candidate}`{mark} | {r['round']} | {r.n_hyperparameters} | {r.log_evidence:.1f} | "
                     f"{r.bic_adjusted_evidence:.1f} | {pct(r.OA)} | {pct(r.AA)} | {pct(r.kappa)} | "
                     f"{pct(r.AURC)} | {r.NLL:.3f} | {100 * r.ECE:.2f} |")
    lines.append("")
    if fin is not None:
        lines += [
            "## Final model",
            "",
            f"The selected kernel `{fin['candidate']}`, with its hyper-parameters re-learned by maximum "
            f"likelihood on all {fin['n_opt']:,} training samples (warm-started from the subset optimum; "
            f"{fin['optimizer_iterations']} L-BFGS iterations, {fin['fit_seconds']:.0f} s).",
            "",
            "| Metric (test) | Value |", "|---|---|",
            f"| Overall accuracy | **{pct(fin['OA'])}%** |", f"| Average accuracy | {pct(fin['AA'])}% |",
            f"| Cohen's kappa | {pct(fin['kappa'])} |", f"| Macro F1 | {pct(fin['macro_F1'])} |",
            f"| AURC | {pct(fin['AURC'])}% |", f"| Negative log-likelihood | {fin['NLL']:.3f} |",
            f"| Expected calibration error | {100 * fin['ECE']:.2f}% |",
            "",
            "Per-class accuracy: " + ", ".join(f"class {c}: {100 * a:.1f}%"
                                              for c, a in fin["per_class_accuracy"].items()) + ".",
            "",
        ]
    ref_rows = ([(f"**GP, `{fin['candidate']}`, final model**", 100 * fin["OA"])] if fin is not None else []) + [
        (f"GP, `{best.candidate}`, subset hyper-parameters", 100 * best.OA)] + list(REFERENCES.items())
    lines += ["## Comparison (test overall accuracy, %)", "", "| Method | Test OA |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in sorted(ref_rows, key=lambda kv: -kv[1])]
    lines.append("")
    if (out / "sensitivity.json").exists():
        sens = json.loads((out / "sensitivity.json").read_text())
        mc = sens["mcnemar_vs_svm"]
        lines += [
            "## Post-hoc checks on the test set (not used for any choice)",
            "",
            "**alpha_eps sensitivity.** The final model is refitted with each alpha_eps, keeping its kernel "
            "hyper-parameters fixed. This shows how much the one fixed setting could matter. Picking the best "
            "row would be tuning on the test set.",
            "",
            "| alpha_eps | Test OA | AA | kappa | AURC | NLL | ECE |", "|---|---|---|---|---|---|---|",
        ]
        lines += [f"| {r['alpha_eps']} | {pct(r['OA'])} | {pct(r['AA'])} | {pct(r['kappa'])} | {pct(r['AURC'])} | "
                  f"{r['NLL']:.3f} | {100 * r['ECE']:.2f} |" for r in sens["alpha_sensitivity"]]
        lines += [
            "",
            f"**Statistical resolution.** With 2,000 test samples, the standard error of an accuracy near "
            f"{pct(mc['gp_OA'])}% is {100 * sens['accuracy_standard_error']:.2f} points. Paired exact McNemar "
            f"test, final GP vs. this repo's SVM ({pct(mc['svm_OA'])}%, {mc['svm_params']}): GP right and SVM "
            f"wrong on {mc['gp_right_svm_wrong']} samples, the reverse on {mc['gp_wrong_svm_right']}, "
            f"p = {mc['p_value']:.2f}.",
            "",
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", default="all")
    r.add_argument("--out", type=Path, default=OUT)
    q = sub.add_parser("report")
    q.add_argument("--out", type=Path, default=OUT)
    f = sub.add_parser("final")
    f.add_argument("--out", type=Path, default=OUT)
    se = sub.add_parser("sensitivity")
    se.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    if a.cmd == "run":
        names = list(CANDIDATES) if a.candidates == "all" else [c.strip() for c in a.candidates.split(",")]
        for n in names:
            run_candidate(n, a.out)
    elif a.cmd == "final":
        final(a.out)
    elif a.cmd == "sensitivity":
        sensitivity(a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
