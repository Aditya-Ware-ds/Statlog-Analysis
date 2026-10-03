"""Multi-dataset benchmark of the seven methods of the paper's E5 protocol.

Methods (the paper's E5 list): B1 flat FCM, B2 flat FCM + max-membership,
B3 hierarchical FCM without gating, B4 hierarchical FCM + entropy gating,
SG-HFQ, and the supervised references SVM and Random Forest. Supplementary
rows: SG-HFQ with class-centroid seeding, and SVM / RF on the Phase 1 features.

    python -m sg_hfq.benchmark run --datasets all --seeds 5
    python -m sg_hfq.benchmark report

``run`` writes per-dataset raw results to ``results/benchmark/raw/``;
``report`` aggregates them into tables (CSV + Markdown) and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from .baselines import FlatFCM, SupervisedReference
from .datasets import LOADERS, BenchmarkDataset
from .metrics import risk_coverage, routing_summary, selective_summary
from .model import SGHFQ

N_GRADES = 5
Q_THRESHOLD = 2
OUT = Path("results/benchmark")

MAIN = [
    "B1 Flat FCM",
    "B2 Flat FCM + max-membership",
    "B3 HFCM, no gating",
    "B4 HFCM + entropy gating",
    "SG-HFQ",
    "SVM (RBF)",
    "Random Forest",
]
SUPPLEMENTARY = ["SG-HFQ (class-seeded)", "SVM (RBF), Phase 1 features", "Random Forest, Phase 1 features"]
GATED = ["B4 HFCM + entropy gating", "SG-HFQ", "SG-HFQ (class-seeded)"]
DATASET_ORDER = list(LOADERS)


def classification_metrics(y: np.ndarray, pred: np.ndarray, classes) -> tuple[dict, dict]:
    recall = {c: float(np.mean(pred[y == c] == c)) for c in classes if np.any(y == c)}
    out = {
        "OA": float(np.mean(pred == y)),
        "AA": float(np.mean(list(recall.values()))),
        "kappa": float(cohen_kappa_score(y, pred)),
        "macro_F1": float(f1_score(y, pred, labels=list(recall), average="macro", zero_division=0)),
    }
    return out, recall


def evaluate(ds: BenchmarkDataset, seed: int) -> dict:
    tr, va, te = ds.split(seed)
    Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
    Xva, yva = (None, None) if va is None else (ds.X[va], ds.y[va])
    zeros = np.zeros(len(te))
    methods: dict[str, dict] = {}
    timing: dict[str, float] = {}
    info: dict = {"n_train": len(tr), "n_val": 0 if va is None else len(va), "n_test": len(te)}

    t = time.time()
    flat = FlatFCM(**ds.phase1, random_state=seed).fit(Xtr, ytr)
    sc = flat.scores(Xte)
    methods["B1 Flat FCM"] = {"pred": sc["pred"], "score": zeros}
    methods["B2 Flat FCM + max-membership"] = {"pred": sc["pred"], "score": sc["max_membership"]}
    timing["flat FCM"] = time.time() - t

    for init, label in (("graph", "SG-HFQ"), ("class", "SG-HFQ (class-seeded)")):
        t = time.time()
        m = SGHFQ(**ds.phase1, init=init, cov_estimator=ds.cov_estimator, n_grades=N_GRADES,
                  q_threshold=Q_THRESHOLD, random_state=seed).fit(Xtr, ytr)
        trace = m.node_trace(Xte)
        res = m.predict(Xte, trace=trace)
        methods[label] = {"pred": res.fine_pred, "score": res.path_min_grade.astype(float),
                          "op": routing_summary(res, yte)}
        if init == "graph":
            tau = float(np.quantile(m.train_confidences(), (Q_THRESHOLD - 1) / N_GRADES))
            b4 = m.predict(Xte, conf_threshold=tau, trace=trace)
            methods["B3 HFCM, no gating"] = {"pred": res.fine_pred, "score": zeros}
            methods["B4 HFCM + entropy gating"] = {"pred": res.fine_pred, "score": res.path_min_conf,
                                                    "op": routing_summary(b4, yte)}
            nodes = m.node_models_.values()
            info["hierarchy"] = str(m.hierarchy_.nested)
            info["max_depth"] = m.hierarchy_.max_depth
            info["fcm_nodes"] = len(m.node_models_)
            info["fcm_nodes_converged"] = int(sum(nm.converged for nm in nodes))
            info["mean_xie_beni"] = float(np.mean([nm.xie_beni for nm in nodes]))
            info["phase1_dims"] = m.rep_.n_dims
            if m.rep_.kind == "pca":
                info["phase1_explained_variance"] = float(m.rep_.explained_variance_ratio_.sum())
        timing[label] = time.time() - t

    for kind, label in (("svm", "SVM (RBF)"), ("rf", "Random Forest")):
        for rep, suffix in ((ds.full, ""), (ds.phase1, ", Phase 1 features")):
            t = time.time()
            ref = SupervisedReference(kind, random_state=seed, **rep).fit(Xtr, ytr, Xva, yva)
            s = ref.scores(Xte)
            methods[label + suffix] = {"pred": s["pred"], "score": s["max_prob"]}
            if kind == "svm":
                info[f"svm_params{suffix}"] = ref.best_params_
            timing[label + suffix] = time.time() - t

    rows, per_class, curves = [], [], {}
    grid = np.linspace(0.005, 1.0, 200)
    for name, mth in methods.items():
        cm, recall = classification_metrics(yte, mth["pred"], ds.classes)
        correct = mth["pred"] == yte
        sel = selective_summary(correct, mth["score"])
        row = {"dataset": ds.key, "seed": seed, "method": name, **cm, "AURC": sel["aurc"], "E-AURC": sel["e_aurc"],
               "risk@50": sel["risk@50"], "risk@80": sel["risk@80"]}
        for k, v in mth.get("op", {}).items():
            row[f"op_{k}"] = v
        rows.append(row)
        per_class += [{"dataset": ds.key, "seed": seed, "method": name, "class": c,
                       "class_name": ds.class_names[c], "accuracy": a} for c, a in recall.items()]
        if seed == 0:
            cov, risk = risk_coverage(correct, mth["score"])
            curves[name] = np.interp(grid, cov, risk)
    out = {"rows": rows, "per_class": per_class, "info": info, "timing": timing}
    if seed == 0:
        out["curves"] = pd.DataFrame({"coverage": grid, **curves})
    return out


def run(datasets, seeds: int, out: Path) -> None:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for key in datasets:
        t0 = time.time()
        ds = LOADERS[key]()
        rows, per_class, infos = [], [], []
        for seed in range(seeds):
            ts = time.time()
            r = evaluate(ds, seed)
            rows += r["rows"]
            per_class += r["per_class"]
            infos.append({"seed": seed, **r["info"], "timing_s": {k: round(v, 1) for k, v in r["timing"].items()}})
            if "curves" in r:
                r["curves"].to_csv(raw / f"{key}_curves.csv", index=False)
            print(f"[{key}] seed {seed} done in {time.time() - ts:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(raw / f"{key}_runs.csv", index=False)
        pd.DataFrame(per_class).to_csv(raw / f"{key}_per_class.csv", index=False)
        class_counts = {ds.class_names[c]: int(n) for c, n in zip(*np.unique(ds.y, return_counts=True))}
        meta = {"key": key, "title": ds.title, "info": ds.info, "class_counts": class_counts,
                "phase1": ds.phase1, "full": ds.full, "cov_estimator": ds.cov_estimator, "seeds": infos,
                "runtime_s": round(time.time() - t0, 1)}
        (raw / f"{key}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[{key}] finished in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report
def _fmt(mean: float, std: float, scale: float = 100.0, digits: int = 1) -> str:
    if np.isnan(mean):
        return "-"
    if np.isnan(std) or std == 0:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def _pivot(df: pd.DataFrame, metric: str, methods, datasets, best: str | None = "max", digits: int = 1) -> pd.DataFrame:
    g = df.groupby(["method", "dataset"])[metric]
    mean, std = g.mean().unstack(), g.std(ddof=1).unstack()
    table = pd.DataFrame(index=methods, columns=datasets, dtype=object)
    for d in datasets:
        col = mean.reindex(methods)[d] if d in mean else pd.Series(np.nan, index=methods)
        main_col = col.reindex([m for m in methods if m in MAIN])
        target = main_col.idxmax() if best == "max" else main_col.idxmin() if best == "min" else None
        for mth in methods:
            mu = col.get(mth, np.nan)
            sd = std.reindex(methods)[d].get(mth, np.nan) if d in std else np.nan
            cell = _fmt(mu, sd, digits=digits)
            table.loc[mth, d] = f"**{cell}**" if mth == target and cell != "-" else cell
    return table


def _md(table: pd.DataFrame, titles: dict[str, str], first: str = "Method") -> str:
    cols = [titles.get(c, c) for c in table.columns]
    lines = ["| " + first + " | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in table.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def report(out: Path) -> None:
    raw = out / "raw"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    keys = [k for k in DATASET_ORDER if (raw / f"{k}_runs.csv").exists()]
    runs = pd.concat([pd.read_csv(raw / f"{k}_runs.csv") for k in keys])
    per_class = pd.concat([pd.read_csv(raw / f"{k}_per_class.csv") for k in keys])
    metas = {k: json.loads((raw / f"{k}_meta.json").read_text()) for k in keys}
    titles = {k: metas[k]["title"] for k in keys}
    runs.to_csv(tables / "all_runs.csv", index=False)

    agg = runs.groupby(["dataset", "method"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.drop(columns=["seed_mean", "seed_std"]).reset_index()
    agg.to_csv(tables / "summary_mean_std.csv", index=False)

    methods = MAIN + SUPPLEMENTARY
    parts = [
        "# Multi-dataset benchmark: SG-HFQ and the paper's E5 methods",
        "",
        "Generated by `python -m sg_hfq.benchmark report`. Mean ± standard deviation over "
        f"{runs.seed.nunique()} seeds (StatLog uses its fixed official split, so only RF/SVM randomness varies). "
        "All numbers are percentages on the test set. Best of the seven main methods in bold. Rows below the "
        "line are supplementary. The fuzzy methods use the Phase 1 representation (StatLog: band-mean 4-D; "
        "others: 10 principal components); SVM and RF use all bands.",
        "",
        "## Datasets and splits",
        "",
        "| Dataset | Sensor | Classes | Labelled | Train | Validation | Test | Split |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        m = metas[k]
        s0 = m["seeds"][0]
        parts.append(f"| {m['title']} | {m['info'].get('sensor', '')} | {m['info']['classes']} | "
                     f"{m['info']['labelled']:,} | {s0['n_train']:,} | {s0['n_val']:,} | {s0['n_test']:,} | "
                     f"{m['info']['split']} |")
    parts.append("")

    def section(title, metric, best, note=""):
        t = _pivot(runs, metric, methods, keys, best)
        t.to_csv(tables / f"comparison_{metric.replace('@', '_at_')}.csv")
        md = _md(t, titles).split("\n")
        md.insert(2 + len(MAIN), "| *supplementary* |" + " |" * len(keys))
        parts.extend([f"## {title}", "", note, "", "\n".join(md), ""] if note else
                     [f"## {title}", "", "\n".join(md), ""])

    section("Overall accuracy, OA (%)", "OA", "max")
    section("Average (per-class) accuracy, AA (%)", "AA", "max")
    section("Cohen's kappa (x100)", "kappa", "max")
    section("Macro F1 (x100)", "macro_F1", "max")
    section("AURC (%, lower is better)", "AURC", "min",
            "Area under the risk-coverage curve of each method's own confidence ranking. Methods without a "
            "confidence score (B1, B3) rank at random, so their AURC equals their error rate.")
    section("Selective risk at 80% coverage (%, lower is better)", "risk@80", "min")

    # gated operating points
    op_cols = ["op_fine_coverage", "op_selective_risk", "op_hierarchical_accuracy", "op_abstain_rate"]
    op = runs[runs.method.isin(GATED)].groupby(["dataset", "method"])[op_cols].mean().reset_index()
    op.to_csv(tables / "operating_points.csv", index=False)
    parts += [f"## Gated operating points (Q_threshold = {Q_THRESHOLD}, L = {N_GRADES})", "",
              "Fine coverage = share of test samples given a fine label; selective risk = error among them; "
              "hierarchical accuracy counts a coarse (group) output as correct when it contains the true class; "
              "abstain = stopped at the root.", "",
              "| Dataset | Method | Fine coverage | Selective risk | Hierarchical accuracy | Abstain |",
              "|---|---|---|---|---|---|"]
    for k in keys:
        for mth in GATED:
            r = op[(op.dataset == k) & (op.method == mth)]
            if len(r):
                r = r.iloc[0]
                parts.append(f"| {titles[k]} | {mth} | {100 * r.op_fine_coverage:.1f} | "
                             f"{100 * r.op_selective_risk:.1f} | {100 * r.op_hierarchical_accuracy:.1f} | "
                             f"{100 * r.op_abstain_rate:.1f} |")
    parts.append("")

    # average ranks over datasets (main methods)
    rank_rows = []
    for metric, asc in (("OA", False), ("AA", False), ("kappa", False), ("AURC", True)):
        means = runs[runs.method.isin(MAIN)].groupby(["dataset", "method"])[metric].mean().unstack()
        ranks = means.rank(axis=1, ascending=asc).mean(axis=0)
        rank_rows.append(ranks.rename(metric))
    ranks = pd.DataFrame(rank_rows).T.reindex(MAIN)
    ranks.to_csv(tables / "average_rank.csv")
    parts += ["## Average rank over the datasets (1 = best, main methods)", "",
              "| Method | OA | AA | kappa | AURC |", "|---|---|---|---|---|"]
    parts += [f"| {mth} | {r.OA:.1f} | {r.AA:.1f} | {r.kappa:.1f} | {r.AURC:.1f} |" for mth, r in ranks.iterrows()]
    parts.append("")

    # per-class accuracy
    pc = per_class.groupby(["dataset", "class", "class_name", "method"])["accuracy"].mean().unstack("method")
    pc.to_csv(tables / "per_class_accuracy.csv")
    for k in keys:
        sub = pc.loc[k].reset_index(level=0, drop=True)[MAIN]
        counts = metas[k]["class_counts"]
        lines = ["| Class | Labelled | " + " | ".join(MAIN) + " |", "|---|---|" + "---|" * len(MAIN)]
        for name, row in sub.iterrows():
            lines.append(f"| {name} | {counts.get(name, '')} | " + " | ".join(f"{100 * v:.1f}" for v in row.values) + " |")
        parts += [f"## Per-class accuracy (%): {titles[k]}", "", "\n".join(lines), ""]

    # learned hierarchies
    parts += ["## Spectral hierarchies learned by SG-HFQ (seed 0)", ""]
    for k in keys:
        s0 = metas[k]["seeds"][0]
        parts.append(f"- **{titles[k]}**: `{s0['hierarchy']}` (depth {s0['max_depth']}, {s0['fcm_nodes']} FCM nodes, "
                     f"{s0['fcm_nodes_converged']} converged; Phase 1 dims {s0['phase1_dims']}"
                     + (f", {100 * s0['phase1_explained_variance']:.1f}% variance" if 'phase1_explained_variance' in s0
                        else "") + ")")
    parts.append("")
    (out / "summary.md").write_text("\n".join(parts) + "\n")

    from .plots import benchmark_figures

    benchmark_figures(out, keys, titles, runs)
    print(f"report written to {out / 'summary.md'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", default="all")
    r.add_argument("--seeds", type=int, default=5)
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
