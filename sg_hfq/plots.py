"""Figures for the experiments (static PNGs; every figure has a CSV table alongside)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from .data import SHORT_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colour follows the entity in every figure.
COLOR = {
    "SG-HFQ": "#2a78d6",
    "B4": "#eb6834",
    "B2": "#1baf7a",
    "B1": "#eda100",
    "B3": "#e87ba4",
    "SVM": "#008300",
    "RF": "#4a3aa7",
    "GML": "#e34948",
}
VARIANT_COLOR = {"SG-HFQ": "#2a78d6", "SG-HFQ (class-seeded)": "#eb6834"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def _save(fig, ctx, name: str) -> None:
    fig.savefig(ctx.out / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ambiguity_hierarchy(ctx) -> None:
    m = ctx.variant("paper")
    rel = m.relations_
    names = [SHORT_NAMES[c] for c in rel.classes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})
    A = rel.ambiguity.copy()
    mask = np.eye(len(A), dtype=bool)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    shown = np.ma.array(A, mask=mask)
    im = ax1.imshow(shown, cmap=cmap, vmin=0, vmax=1)
    ax1.grid(False)
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                ax1.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                         color="white" if A[i, j] > 0.55 else INK)
    ax1.set_xticks(range(len(A)), names, rotation=35, ha="right")
    ax1.set_yticks(range(len(A)), names)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("Spectral-ambiguity matrix A (training set)", loc="left")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("ambiguity (1 = indistinct)")

    with plt.rc_context({"lines.linewidth": 2}):
        dendrogram(m.linkage_matrix_, labels=names, ax=ax2, color_threshold=0, above_threshold_color=COLOR["SG-HFQ"],
                   leaf_rotation=0)
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    ax2.set_ylabel("graph distance 1 - A (average linkage)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Phase 3 dendrogram (coarse to fine)", loc="left")
    _save(fig, ctx, "fig1_ambiguity_and_hierarchy")


def fig_e1(ctx, e1) -> None:
    pairs = pd.DataFrame(e1["pairs"])
    corr = e1["corr"]
    panels = [("Flat FCM", "Flat FCM"), ("SG-HFQ class-seeded (ungated)", "SG-HFQ (class-seeded), ungated"),
              ("SVM", "SVM (4-D)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        x, yv = pairs["A"], pairs[f"err[{key}]"]
        ax.scatter(x, yv, s=46, color=COLOR["SG-HFQ"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        top = np.argsort(-(x.rank() + yv.rank()).to_numpy())[:4]
        for t in top:
            a, b = pairs["pair"][t].split("-")
            right = x[t] > 0.85
            ax.annotate(f"{a}\u2013{b}", (x[t], yv[t]), xytext=(-7 if right else 7, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_2, ha="right" if right else "left", va="center")
        r = corr[(corr["similarity"] == "A (alpha=0.5)") & (corr["classifier"] == key)].iloc[0]
        ax.set_title(f"{title}\nSpearman rho = {r.spearman_rho:.2f} (p = {r.p_value:.3f})", loc="left")
        ax.set_xlabel("ambiguity A_ij (training)")
        ax.set_xlim(-0.03, 1.05)
    axes[0].set_ylabel("pairwise test error rate e_ij")
    key = ", ".join(f"{c} {SHORT_NAMES[c]}" for c in sorted(SHORT_NAMES))
    fig.text(0.01, -0.06, f"Each dot is a class pair (15 pairs); the four most ambiguous/confused pairs are labelled "
             f"by class code: {key}.", color=INK_2, fontsize=9)
    _save(fig, ctx, "fig2_e1_ambiguity_vs_error")


def _grouped_bars(ax, categories, series: dict[str, np.ndarray], colors: dict[str, str], width=0.36):
    x = np.arange(len(categories))
    k = len(series)
    for n, (label, vals) in enumerate(series.items()):
        off = (n - (k - 1) / 2) * (width + 0.02)
        ax.bar(x + off, vals, width=width, color=colors[label], label=label, zorder=3, linewidth=0)
    ax.set_xticks(x, categories)
    ax.grid(axis="x", visible=False)


def fig_e3(ctx, e3) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, (("node", "Node-local decision error by grade"),
                                       ("path", "Fine-prediction error by path grade"))):
        df = e3[key]
        grades = df["grade"].unique().tolist()
        series = {v: df[df["variant"] == v].set_index("grade").loc[grades, "error"].to_numpy()
                  for v in df["variant"].unique()}
        _grouped_bars(ax, grades, series, VARIANT_COLOR)
        ax.set_title(title, loc="left")
        ax.set_xlabel("confidence grade (Q1 = least confident)")
    axes[0].set_ylabel("error rate (test)")
    axes[0].legend(loc="upper right")
    _save(fig, ctx, "fig3_e3_quantization_calibration")


def fig_e4(ctx, e4: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    depths = sorted(e4["depth"].unique())
    for ax, col, title in ((axes[0], "coverage", "Share of test samples stopping at each depth"),
                           (axes[1], "P(correct|depth)", "P(correct | depth)")):
        series = {v: e4[e4["variant"] == v].set_index("depth").reindex(depths)[col].to_numpy()
                  for v in e4["variant"].unique()}
        _grouped_bars(ax, [str(d) for d in depths], series, VARIANT_COLOR)
        ax.set_title(title, loc="left")
        ax.set_xlabel("hierarchy depth of the output (0 = root / abstain)")
    axes[0].set_ylabel("share of test samples")
    axes[1].set_ylabel("group accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right")
    _save(fig, ctx, "fig4_e4_depth_accuracy")


def fig_e5(ctx, methods: dict) -> None:
    from .metrics import risk_coverage

    def series_key(name: str) -> str:
        for k in ("B1", "B2", "B3", "B4"):
            if name.startswith(k):
                return k
        if name.startswith("SG-HFQ"):
            return "SG-HFQ"
        if "SVM" in name:
            return "SVM"
        if "Random Forest" in name:
            return "RF"
        return "GML"

    panels = [
        ("(a) Paper configuration (graph-seeded)",
         ["SG-HFQ", "B4 HFCM + entropy gating", "B2 Flat FCM + max-membership", "B1 Flat FCM", "B3 HFCM, no gating"]),
        ("(b) Class-seeded hierarchical FCM (E2 option B)",
         ["SG-HFQ (class-seeded)", "B4 HFCM + entropy gating (class-seeded)", "B2 Flat FCM + max-membership",
          "B1 Flat FCM", "B3 HFCM, no gating (class-seeded)"]),
        ("(c) Supervised references (4-D)",
         ["SG-HFQ (class-seeded)", "B2 Flat FCM + max-membership", "Ref. SVM (4-D)", "Ref. Random Forest (4-D)",
          "Ref. Gaussian ML (4-D)"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (title, names) in zip(axes, panels):
        for name in names:
            mth = methods[name]
            cov, risk = risk_coverage(mth["correct"], mth["score"])
            key = series_key(name)
            const = np.ptp(mth["score"]) == 0
            ax.plot(cov, risk, color=COLOR[key], linestyle=(0, (4, 3)) if const else "-",
                    label=name + (" (no selection)" if const else ""), zorder=3)
            if "op" in mth:
                op = mth["op"]
                ax.plot(op["fine_coverage"], op["selective_risk"], "o", ms=8, color=COLOR[key],
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.set_title(title, loc="left")
        ax.set_xlabel("coverage (share of test samples given a fine label)")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 0.9)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=8.5)
    axes[0].set_ylabel("selective risk (1 - accuracy)")
    fig.suptitle("Risk-coverage on the StatLog test set. Dots: gated operating point at Q_threshold = 2 "
                 "(B4: matching global confidence quantile).", x=0.01, ha="left", fontsize=9.5, color=INK_2, y=1.0)
    _save(fig, ctx, "fig5_e5_risk_coverage")


def make_all(ctx, results: dict) -> None:
    _style()
    fig_ambiguity_hierarchy(ctx)
    if "e1" in results:
        fig_e1(ctx, results["e1"])
    if "e3" in results:
        fig_e3(ctx, results["e3"])
    if "e4" in results:
        fig_e4(ctx, results["e4"])
    if "e5" in results:
        fig_e5(ctx, results["e5"])
