"""Experimental protocol of the paper (Section 4): E1-E6 plus supplementary sweeps.

Run everything with::

    python -m sg_hfq.experiments --out results

All models are fitted on the official 4,435-sample training split and
evaluated on the 2,000-sample test split. Defaults that the paper leaves open
are fixed a priori (L = 5 grades, Q_threshold = 2, average linkage, m = 2).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .baselines import FlatFCM, SupervisedReference
from .data import CLASS_CODES, CLASS_NAMES, StatLogData, load_statlog
from .hierarchy import SEMANTIC_HIERARCHY, Hierarchy
from .metrics import (
    aurc,
    bootstrap_ci,
    optimal_aurc,
    pairwise_error_rates,
    risk_at_coverage,
    risk_coverage,
    routing_summary,
    selective_summary,
)
from .model import SGHFQ
from .separability import minmax_offdiag

N_GRADES = 5
Q_THRESHOLD = 2
N_RANDOM_INITS = 10
#: SVM grid used for the StatLog study (5-fold CV on the official training split).
STATLOG_SVM_GRID = {"C": [1, 10, 100], "gamma": ["scale", 0.1, 1.0]}

#: The two hierarchical variants evaluated throughout: the paper's
#: graph-seeded FCM (one cluster per child group) and E2 option B
#: (one cluster per class, seeded at class centroids).
VARIANTS = {"paper": "graph", "class-seeded": "class"}
VARIANT_LABEL = {"paper": "SG-HFQ", "class-seeded": "SG-HFQ (class-seeded)"}


@dataclass
class Ctx:
    data: StatLogData
    out: Path
    n_boot: int = 1000
    seed: int = 0
    cache: dict = field(default_factory=dict)
    summary: list = field(default_factory=list)

    @property
    def tables(self) -> Path:
        return self.out / "tables"

    def model(self, **kw) -> SGHFQ:
        key = tuple(sorted(kw.items()))
        if key not in self.cache:
            params = dict(n_grades=N_GRADES, q_threshold=Q_THRESHOLD, random_state=self.seed)
            params.update(kw)
            self.cache[key] = SGHFQ(**params).fit(self.data.X_train, self.data.y_train)
        return self.cache[key]

    def variant(self, name: str) -> SGHFQ:
        return self.model(init=VARIANTS[name])

    def supervised(self, kind: str, representation: str = "bandmean4"):
        """Test-set scores and chosen hyper-parameters of a supervised reference."""
        key = ("supervised", kind, representation)
        if key not in self.cache:
            ref = SupervisedReference(kind, representation, random_state=self.seed, svm_grid=STATLOG_SVM_GRID)
            ref.fit(self.data.X_train, self.data.y_train)
            self.cache[key] = (ref.scores(self.data.X_test), ref.best_params_)
        return self.cache[key]

    def save(self, df: pd.DataFrame, name: str, title: str, floatfmt: int = 4) -> None:
        df.to_csv(self.tables / f"{name}.csv", index=False)
        self.summary.append((title, markdown_table(df, floatfmt)))


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _class_pairs(classes):
    return [(i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))]


# --------------------------------------------------------------------- Phases
def phase_outputs(ctx: Ctx) -> None:
    """Phase 2/3 artefacts, per-node FCM statistics and the Phase 7 output file."""
    d = ctx.data
    m = ctx.variant("paper")
    rel = m.relations_
    names = [str(c) for c in rel.classes]
    rows = []
    for i, j in _class_pairs(rel.classes):
        rows.append(
            {
                "class_i": rel.classes[i],
                "class_j": rel.classes[j],
                "bhattacharyya": rel.bhattacharyya[i, j],
                "jm": rel.jm[i, j],
                "wasserstein": rel.wasserstein[i, j],
                "ambiguity": rel.ambiguity[i, j],
            }
        )
    ctx.save(pd.DataFrame(rows), "phase2_pairwise_relations", "Phase 2 - pairwise spectral relations (training set)")
    for key, M in (("jm", rel.jm), ("wasserstein", rel.wasserstein), ("ambiguity", rel.ambiguity)):
        pd.DataFrame(M, index=names, columns=names).to_csv(ctx.tables / f"phase2_{key}_matrix.csv")

    hier_text = ["Spectral (Phase 3) hierarchy:", m.hierarchy_.describe(CLASS_NAMES), ""]
    hier_text += ["Semantic hierarchy (HFCM ablation):", Hierarchy(SEMANTIC_HIERARCHY).describe(CLASS_NAMES), ""]
    hier_text += ["Euclidean-centroid hierarchy:", ctx.model(hierarchy="euclidean").hierarchy_.describe(CLASS_NAMES)]
    (ctx.out / "hierarchy.txt").write_text("\n".join(hier_text) + "\n")
    (ctx.out / "hierarchy.json").write_text(json.dumps(m.hierarchy_.to_dict(), indent=2))
    ctx.summary.append(("Phase 3 - hierarchies", "```\n" + "\n".join(hier_text) + "\n```"))

    for name in VARIANTS:
        s = ctx.variant(name).summary()
        s.insert(0, "variant", VARIANT_LABEL[name])
        ctx.save(s, f"phase4_nodes_{name}", f"Phase 4 - FCM per node ({VARIANT_LABEL[name]})")
        res = ctx.variant(name).predict(d.X_test)
        res.to_frame(d.y_test).to_csv(ctx.out / "predictions" / f"phase7_test_outputs_{name}.csv", index_label="sample")


# ------------------------------------------------------------------------- E1
def e1_spectral_graph(ctx: Ctx) -> dict:
    d = ctx.data
    m = ctx.variant("paper")
    rel = m.relations_
    pairs = _class_pairs(rel.classes)

    preds = {
        "Flat FCM": FlatFCM().fit(d.X_train, d.y_train).predict(d.X_test),
        "SG-HFQ (ungated)": ctx.variant("paper").predict_fine(d.X_test),
        "SG-HFQ class-seeded (ungated)": ctx.variant("class-seeded").predict_fine(d.X_test),
        "SVM": ctx.supervised("svm")[0]["pred"],
        "Random Forest": ctx.supervised("rf")[0]["pred"],
        "Gaussian ML": ctx.supervised("gml")[0]["pred"],
    }
    measures = {
        "A (alpha=0.5)": rel.ambiguity,
        "inverted JM only": 1 - minmax_offdiag(rel.jm),
        "inverted W only": 1 - minmax_offdiag(rel.wasserstein),
    }

    pair_rows, corr_rows = [], []
    E = {k: pairwise_error_rates(d.y_test, p, rel.classes) for k, p in preds.items()}
    for i, j in pairs:
        row = {"pair": f"{rel.classes[i]}-{rel.classes[j]}", "A": rel.ambiguity[i, j]}
        row.update({f"err[{k}]": E[k][i, j] for k in preds})
        pair_rows.append(row)
    for mname, M in measures.items():
        a = np.array([M[i, j] for i, j in pairs])
        for k in preds:
            e = np.array([E[k][i, j] for i, j in pairs])
            rho, p = spearmanr(a, e)
            corr_rows.append({"similarity": mname, "classifier": k, "spearman_rho": rho, "p_value": p})
    ctx.save(pd.DataFrame(pair_rows), "e1_pairs", "E1 - ambiguity vs. observed pairwise test error (15 class pairs)")
    corr = pd.DataFrame(corr_rows)
    ctx.save(corr, "e1_spearman", "E1 - Spearman correlation between ambiguity and pairwise error")

    alpha_rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        ma = ctx.model(init="graph", alpha=alpha)
        a = np.array([ma.relations_.ambiguity[i, j] for i, j in pairs])
        row = {"alpha": alpha, "hierarchy": str(ma.hierarchy_.nested)}
        for k in ("Flat FCM", "SVM", "Gaussian ML"):
            row[f"rho[{k}]"] = spearmanr(a, [E[k][i, j] for i, j in pairs])[0]
        alpha_rows.append(row)
    ctx.save(pd.DataFrame(alpha_rows), "e1_alpha_sweep", "E1 (supplementary) - JM/Wasserstein blend weight alpha")
    return {"pairs": pair_rows, "E": E, "corr": corr}


# ------------------------------------------------------------------------- E2
def _hier_eval(m: SGHFQ, X, y) -> dict:
    res = m.predict(X)
    ok = res.fine_pred == y
    nodes = m.node_models_.values()
    return {
        "accuracy_ungated": ok.mean(),
        "aurc_grade": aurc(ok, res.path_min_grade),
        "aurc_raw_entropy": aurc(ok, res.path_min_conf),
        "mean_xie_beni": np.mean([nm.xie_beni for nm in nodes]),
        "total_fcm_iterations": sum(nm.n_iter for nm in nodes),
        "nodes_identity_mapping": np.mean([nm.mapping_is_identity for nm in nodes]),
        "total_objective": sum(nm.objective for nm in nodes),
    }


def e2_initialisation(ctx: Ctx) -> pd.DataFrame:
    d = ctx.data
    graph = ctx.model(init="graph")
    rows = []
    for label, init in (("(B) class-centroid", "class"), ("(C) graph-derived", "graph")):
        r = _hier_eval(ctx.model(init=init), d.X_test, d.y_test)
        rows.append({"seeding": label, "runs": 1, **r, "same_optimum_as_graph": "-" if init == "class" else "(itself)"})
    rand = []
    same = []
    for s in range(N_RANDOM_INITS):
        mr = ctx.model(init="random", random_state=1000 + s)
        rand.append(_hier_eval(mr, d.X_test, d.y_test))
        same.append(
            all(
                np.allclose(mr.node_models_[k].centres[np.argsort(mr.node_models_[k].cluster_to_child)],
                            graph.node_models_[k].centres, atol=1e-3)
                for k in graph.node_models_
            )
        )
    df_r = pd.DataFrame(rand)
    row = {"seeding": "(A) random", "runs": N_RANDOM_INITS}
    for c in df_r.columns:
        row[c] = df_r[c].mean()
        row[c + "_std"] = df_r[c].std(ddof=1)
    row["same_optimum_as_graph"] = f"{sum(same)}/{len(same)}"
    rows.insert(0, row)
    df = pd.DataFrame(rows)
    cols = ["seeding", "runs", "accuracy_ungated", "accuracy_ungated_std", "aurc_grade", "aurc_grade_std",
            "mean_xie_beni", "total_fcm_iterations", "nodes_identity_mapping", "same_optimum_as_graph"]
    ctx.save(df[cols], "e2_initialisation", "E2 - FCM initialisation (test set)")

    # 5-fold CV on the training split only (model-selection evidence free of test data)
    cv = StratifiedKFold(5, shuffle=True, random_state=ctx.seed)
    cv_rows = []
    for label, init in (("(A) random", "random"), ("(B) class-centroid", "class"), ("(C) graph-derived", "graph")):
        accs, aurcs = [], []
        for f, (tr, va) in enumerate(cv.split(d.X_train, d.y_train)):
            mm = SGHFQ(init=init, n_grades=N_GRADES, q_threshold=Q_THRESHOLD, random_state=f).fit(
                d.X_train[tr], d.y_train[tr]
            )
            ev = _hier_eval(mm, d.X_train[va], d.y_train[va])
            accs.append(ev["accuracy_ungated"])
            aurcs.append(ev["aurc_grade"])
        cv_rows.append({"seeding": label, "cv_accuracy": np.mean(accs), "cv_accuracy_std": np.std(accs, ddof=1),
                        "cv_aurc": np.mean(aurcs), "cv_aurc_std": np.std(aurcs, ddof=1)})
    ctx.save(pd.DataFrame(cv_rows), "e2_initialisation_cv", "E2 - FCM initialisation, 5-fold CV on the training split")
    return df


# ------------------------------------------------------------------------- E3
def e3_calibration(ctx: Ctx) -> dict:
    d = ctx.data
    out = {}
    node_rows, path_rows, per_node_rows, mono_rows = [], [], [], []
    for name in VARIANTS:
        m = ctx.variant(name)
        h = m.hierarchy_
        trace = m.node_trace(d.X_test)
        grades, correct = [], []
        for node in h.internal_nodes:
            mask = np.isin(d.y_test, node.classes)
            true_child = np.array([h.child_index(node, c) for c in d.y_test[mask]])
            g = trace[node.id].grade[mask]
            ok = trace[node.id].child[mask] == true_child
            grades.append(g)
            correct.append(ok)
            for q in range(1, N_GRADES + 1):
                sel = g == q
                per_node_rows.append({"variant": VARIANT_LABEL[name], "node": node.label, "grade": q, "n": int(sel.sum()),
                                      "error": float(1 - ok[sel].mean()) if sel.any() else np.nan})
        g_all, ok_all = np.concatenate(grades), np.concatenate(correct)
        res = m.predict(d.X_test, trace=trace)
        ok_path = res.fine_pred == d.y_test
        for scope, G, OK, rows in (("node-local", g_all, ok_all, node_rows), ("path", res.path_min_grade, ok_path, path_rows)):
            errs = []
            for q in range(1, N_GRADES + 1):
                sel = G == q
                err = float(1 - OK[sel].mean()) if sel.any() else np.nan
                errs.append(err)
                rows.append({"variant": VARIANT_LABEL[name], "grade": f"Q{q}", "n": int(sel.sum()),
                             "share": float(sel.mean()), "accuracy": 1 - err, "error": err})
            e = np.array(errs)
            valid = ~np.isnan(e)
            mono_rows.append({
                "variant": VARIANT_LABEL[name],
                "scope": scope,
                "monotone_non_increasing": bool(np.all(np.diff(e[valid]) <= 1e-12)),
                "spearman_rho(grade,error)": spearmanr(np.arange(1, N_GRADES + 1)[valid], e[valid])[0],
                "error_Q1": e[0],
                f"error_Q{N_GRADES}": e[-1],
            })
        out[name] = {"node": (g_all, ok_all), "path": (res.path_min_grade, ok_path)}
    ctx.save(pd.DataFrame(node_rows), "e3_calibration_node_local",
             "E3 - node-local decision accuracy per grade (test samples at their true nodes)")
    ctx.save(pd.DataFrame(path_rows), "e3_calibration_path",
             "E3 - fine accuracy per path grade (minimum grade along the arg-max path)")
    ctx.save(pd.DataFrame(mono_rows), "e3_monotonicity", "E3 - is error monotonically decreasing in the grade?")
    pd.DataFrame(per_node_rows).to_csv(ctx.tables / "e3_calibration_per_node.csv", index=False)
    return {"node": pd.DataFrame(node_rows), "path": pd.DataFrame(path_rows)}


# ------------------------------------------------------------------------- E4
def e4_depth(ctx: Ctx) -> pd.DataFrame:
    d = ctx.data
    rows, sweep = [], []
    for name in VARIANTS:
        m = ctx.variant(name)
        trace = m.node_trace(d.X_test)
        res = m.predict(d.X_test, trace=trace)
        ok = res.set_correct(d.y_test)
        for depth in range(m.hierarchy_.max_depth + 1):
            sel = res.depth == depth
            rows.append({
                "variant": VARIANT_LABEL[name],
                "depth": depth,
                "n": int(sel.sum()),
                "coverage": float(sel.mean()),
                "fine_share": float(res.is_fine[sel].mean()) if sel.any() else np.nan,
                "P(correct|depth)": float(ok[sel].mean()) if sel.any() else np.nan,
            })
        for q in range(1, N_GRADES + 2):
            r = m.predict(d.X_test, q_threshold=q, trace=trace)
            sweep.append({"variant": VARIANT_LABEL[name], "Q_threshold": q, **routing_summary(r, d.y_test)})
    df = pd.DataFrame(rows)
    ctx.save(df, "e4_depth", f"E4 - depth-driven accuracy (Q_threshold = {Q_THRESHOLD})")
    ctx.save(pd.DataFrame(sweep), "e4_threshold_sweep", "E4 - operating points for every Q_threshold")
    return df


# ------------------------------------------------------------------------- E5
def e5_methods(ctx: Ctx) -> dict[str, dict]:
    """Every method's correctness vector and selection score on the test set."""
    d = ctx.data
    y = d.y_test
    methods: dict[str, dict] = {}
    flat = FlatFCM().fit(d.X_train, d.y_train).scores(d.X_test)
    ok = flat["pred"] == y
    methods["B1 Flat FCM"] = {"correct": ok, "score": np.zeros(len(y)), "family": "flat"}
    methods["B2 Flat FCM + max-membership"] = {"correct": ok, "score": flat["max_membership"], "family": "flat"}
    for name, tag in (("paper", ""), ("class-seeded", " (class-seeded)")):
        m = ctx.variant(name)
        trace = m.node_trace(d.X_test)
        res = m.predict(d.X_test, trace=trace)
        okh = res.fine_pred == y
        tau = np.quantile(m.train_confidences(), (Q_THRESHOLD - 1) / N_GRADES)
        b4 = m.predict(d.X_test, conf_threshold=tau, trace=trace)
        methods[f"B3 HFCM, no gating{tag}"] = {"correct": okh, "score": np.zeros(len(y)), "family": name}
        methods[f"B4 HFCM + entropy gating{tag}"] = {"correct": okh, "score": res.path_min_conf, "family": name,
                                                      "op": routing_summary(b4, y), "tau": float(tau)}
        methods[f"SG-HFQ{tag}"] = {"correct": okh, "score": res.path_min_grade.astype(float), "family": name,
                                   "op": routing_summary(res, y)}
        methods[f"SG-HFQ{tag}, continuous grade (L->inf)"] = {"correct": okh, "score": res.path_min_cdf, "family": name}
    for kind, label in (("svm", "SVM"), ("rf", "Random Forest"), ("gml", "Gaussian ML")):
        for rep in ("bandmean4", "full36"):
            sc, params = ctx.supervised(kind, rep)
            methods[f"Ref. {label} ({'4-D' if rep == 'bandmean4' else '36-D'})"] = {
                "correct": sc["pred"] == y, "score": sc["max_prob"], "family": "supervised", "params": params}
    return methods


def e5_selective(ctx: Ctx) -> dict[str, dict]:
    methods = e5_methods(ctx)
    rows = []
    for k, (name, mth) in enumerate(methods.items()):
        s = selective_summary(mth["correct"], mth["score"])
        lo, hi = bootstrap_ci(aurc, (mth["correct"], mth["score"]), n_boot=ctx.n_boot, seed=ctx.seed + k)
        row = {"method": name, "accuracy@100%": s["accuracy_full"], "AURC": s["aurc"], "AURC_95lo": lo,
               "AURC_95hi": hi, "E-AURC": s["e_aurc"], "risk@50%": s["risk@50"], "risk@80%": s["risk@80"],
               "risk@90%": s["risk@90"]}
        rows.append(row)
    ctx.save(pd.DataFrame(rows), "e5_selective_prediction", "E5 - risk-coverage summary (test set, 95% bootstrap CI)")

    # operating points of the gated methods, against baselines at matched coverage
    op_rows = []
    for name, mth in methods.items():
        if "op" not in mth:
            continue
        op = mth["op"]
        cov = op["fine_coverage"]
        row = {"method": name, **op}
        b2 = methods["B2 Flat FCM + max-membership"]
        row["B2 risk @ same coverage"] = risk_at_coverage(b2["correct"], b2["score"], cov)
        svm = methods["Ref. SVM (4-D)"]
        row["SVM(4-D) risk @ same coverage"] = risk_at_coverage(svm["correct"], svm["score"], cov)
        op_rows.append(row)
    ctx.save(pd.DataFrame(op_rows), "e5_operating_points",
             f"E5 - gated operating points (Q_threshold = {Q_THRESHOLD}; B4 uses the matching global confidence quantile)")

    # paired bootstrap of AURC differences
    diff_rows = []
    for a, b in (("SG-HFQ", "B2 Flat FCM + max-membership"), ("SG-HFQ", "B4 HFCM + entropy gating"),
                 ("SG-HFQ (class-seeded)", "B2 Flat FCM + max-membership"),
                 ("SG-HFQ (class-seeded)", "B4 HFCM + entropy gating (class-seeded)")):
        A, B = methods[a], methods[b]
        stat = lambda ca, sa, cb, sb: aurc(ca, sa) - aurc(cb, sb)  # noqa: E731
        arrays = (A["correct"], A["score"], B["correct"], B["score"])
        lo, hi = bootstrap_ci(stat, arrays, n_boot=ctx.n_boot, seed=ctx.seed)
        diff_rows.append({"comparison": f"{a}  minus  {b}", "delta_AURC": stat(*arrays), "95lo": lo, "95hi": hi,
                          "significant": not (lo <= 0 <= hi)})
    ctx.save(pd.DataFrame(diff_rows), "e5_paired_differences", "E5 - paired bootstrap of AURC differences (negative = first is better)")

    curves = {}
    grid = np.linspace(0.01, 1.0, 100)
    for name, mth in methods.items():
        cov, risk = risk_coverage(mth["correct"], mth["score"])
        curves[name] = np.interp(grid, cov, risk)
    pd.DataFrame({"coverage": grid, **curves}).to_csv(ctx.tables / "e5_risk_coverage_curves.csv", index=False)
    ref_params = {k: v["params"] for k, v in methods.items() if "params" in v}
    (ctx.out / "supervised_hyperparameters.json").write_text(json.dumps(ref_params, indent=2, default=str))
    return methods


# ------------------------------------------------------------------------- E6
def e6_ablation(ctx: Ctx) -> pd.DataFrame:
    d = ctx.data
    y = d.y_test
    rows = []
    flat = FlatFCM().fit(d.X_train, d.y_train)
    ok_flat = flat.predict(d.X_test) == y

    def add(model, graph, hier, ent, quant, gate, correct, score, op=None, seeding="-"):
        rows.append({
            "model": model, "seeding": seeding, "Graph": graph, "Hier.": hier, "Entropy": ent, "Quant.": quant,
            "Gate": gate, "accuracy@100%": correct.mean(), "AURC": aurc(correct, score),
            "E-AURC": aurc(correct, score) - optimal_aurc(correct),
            "fine_coverage": op["fine_coverage"] if op else np.nan,
            "selective_risk": op["selective_risk"] if op else np.nan,
            "hierarchical_accuracy": op["hierarchical_accuracy"] if op else np.nan,
            "mean_depth": op["mean_depth"] if op else np.nan,
        })

    zero = np.zeros(len(y))
    add("Flat FCM", False, False, False, False, False, ok_flat, zero)
    for name in VARIANTS:
        init = VARIANTS[name]
        for hname, label in (("semantic", "HFCM (semantic hierarchy)"), ("euclidean", "HFCM (Euclidean-centroid hierarchy)")):
            mh = ctx.model(init=init, hierarchy=hname)
            add(label, False, True, False, False, False, mh.predict_fine(d.X_test) == y, zero, seeding=name)
        m = ctx.variant(name)
        trace = m.node_trace(d.X_test)
        res = m.predict(d.X_test, trace=trace)
        ok = res.fine_pred == y
        add("Graph-HFCM", True, True, False, False, False, ok, zero, seeding=name)
        add("+ Entropy", True, True, True, False, False, ok, res.path_min_conf, seeding=name)
        add("+ Quantization", True, True, True, True, False, ok, res.path_min_grade.astype(float), seeding=name)
        add("SG-HFQ", True, True, True, True, True, ok, res.path_min_grade.astype(float),
            op=routing_summary(res, y), seeding=name)
    df = pd.DataFrame(rows)
    flags = ["Graph", "Hier.", "Entropy", "Quant.", "Gate"]
    df[flags] = df[flags].replace({True: "✓", False: "–"})
    ctx.save(df, "e6_ablation", f"E6 - ablation (Q_threshold = {Q_THRESHOLD}, L = {N_GRADES})")
    return df


# --------------------------------------------------------------- supplementary
def supplementary(ctx: Ctx) -> None:
    d = ctx.data
    y = d.y_test
    rows = []
    for name, init in VARIANTS.items():
        for rep in ("bandmean4", "full36"):
            m = ctx.model(init=init, representation=rep)
            res = m.predict(d.X_test)
            ok = res.fine_pred == y
            rows.append({"variant": VARIANT_LABEL[name], "representation": rep, "hierarchy": str(m.hierarchy_.nested),
                         "accuracy@100%": ok.mean(), "AURC": aurc(ok, res.path_min_grade),
                         **{k: v for k, v in routing_summary(res, y).items() if k in ("fine_coverage", "selective_risk")}})
    ctx.save(pd.DataFrame(rows), "s1_representation", "Supplementary S1 - band-mean 4-D vs. full 36-D representation")

    rows = []
    for name, init in VARIANTS.items():
        for L in (3, 4, 5, 8, 10):
            m = ctx.model(init=init, n_grades=L)
            res = m.predict(d.X_test)
            ok = res.fine_pred == y
            rows.append({"variant": VARIANT_LABEL[name], "L": L, "AURC(grade)": aurc(ok, res.path_min_grade),
                         **routing_summary(res, y)})
    ctx.save(pd.DataFrame(rows), "s2_grades", f"Supplementary S2 - number of grades L (Q_threshold = {Q_THRESHOLD})")

    rows = []
    for name, init in VARIANTS.items():
        for link in ("average", "complete", "single", "weighted"):
            m = ctx.model(init=init, linkage=link)
            res = m.predict(d.X_test)
            ok = res.fine_pred == y
            rows.append({"variant": VARIANT_LABEL[name], "linkage": link, "hierarchy": str(m.hierarchy_.nested),
                         "accuracy@100%": ok.mean(), "AURC": aurc(ok, res.path_min_grade)})
    ctx.save(pd.DataFrame(rows), "s3_linkage", "Supplementary S3 - agglomerative linkage on the ambiguity graph")


# ------------------------------------------------------------------- driver
STEPS = ("phases", "e1", "e2", "e3", "e4", "e5", "e6", "supp")


def run(out: Path, n_boot: int = 1000, seed: int = 0, steps=STEPS, figures: bool = True) -> Ctx:
    out = Path(out)
    for sub in ("tables", "figures", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ctx = Ctx(load_statlog(), out, n_boot=n_boot, seed=seed)
    t0 = time.time()
    results = {}
    if "phases" in steps:
        phase_outputs(ctx)
    if "e1" in steps:
        results["e1"] = e1_spectral_graph(ctx)
    if "e2" in steps:
        results["e2"] = e2_initialisation(ctx)
    if "e3" in steps:
        results["e3"] = e3_calibration(ctx)
    if "e4" in steps:
        results["e4"] = e4_depth(ctx)
    if "e5" in steps:
        results["e5"] = e5_selective(ctx)
    if "e6" in steps:
        results["e6"] = e6_ablation(ctx)
    if "supp" in steps:
        supplementary(ctx)
    if figures:
        from . import plots

        plots.make_all(ctx, results)

    import matplotlib
    import scipy
    import sklearn

    config = {
        "sg_hfq_version": __version__,
        "dataset": "StatLog (Landsat Satellite), official split 4435 train / 2000 test",
        "classes": {str(c): CLASS_NAMES[c] for c in CLASS_CODES},
        "representation": "bandmean4 (z-scored with training statistics)",
        "alpha": 0.5,
        "linkage": "average",
        "fuzzifier_m": 2.0,
        "fcm_tol": 1e-6,
        "fcm_max_iter": 1000,
        "n_grades_L": N_GRADES,
        "q_threshold": Q_THRESHOLD,
        "random_inits_E2": N_RANDOM_INITS,
        "bootstrap_resamples": n_boot,
        "seed": seed,
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    write_summary(ctx, config)
    return ctx


def write_summary(ctx: Ctx, config: dict) -> None:
    parts = [
        "# SG-HFQ results on StatLog (Landsat Satellite)",
        "",
        "Generated by `python -m sg_hfq.experiments`. Training: official 4,435-sample split; "
        "evaluation: official 2,000-sample test split. "
        f"L = {config['n_grades_L']} grades, Q_threshold = {config['q_threshold']}, alpha = {config['alpha']}, "
        f"m = {config['fuzzifier_m']}, {config['linkage']} linkage, band-mean 4-D representation.",
        "",
        "`SG-HFQ` is the paper's method as written (FCM with one cluster per child group, seeded at the "
        "child-group centroids). `SG-HFQ (class-seeded)` uses E2 option B (one cluster per class, seeded at the "
        "class centroids, memberships summed per child group).",
        "",
    ]
    for title, body in ctx.summary:
        parts += [f"## {title}", "", body, ""]
    (ctx.out / "summary.md").write_text("\n".join(parts))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results", type=Path)
    p.add_argument("--n-boot", default=1000, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    steps = tuple(s.strip() for s in a.steps.split(",") if s.strip())
    run(a.out, n_boot=a.n_boot, seed=a.seed, steps=steps, figures=not a.no_figures)
    print(f"results written to {a.out}/ (see {a.out}/summary.md)")


if __name__ == "__main__":
    main()
