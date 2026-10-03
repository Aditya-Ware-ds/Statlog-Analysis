"""SG-HFQ: Spectral-Graph-Guided Hierarchical Fuzzy Quantization.

Phase 1  spectral representation (``data.SpectralRepresentation``)
Phase 2  JM / Wasserstein spectral-ambiguity matrix (``separability``)
Phase 3  data-driven hierarchy from the ambiguity graph (``hierarchy``)
Phase 4  graph-seeded hierarchical fuzzy C-means (this module)
Phase 5  fuzzy entropy -> confidence -> quantile grades (``quantization``)
Phase 6  adaptive routing gated by the confidence grade (this module)
Phase 7  per-sample outputs (``RoutingResult.to_frame``)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .data import SpectralRepresentation
from .fcm import fcm, memberships, random_centres, xie_beni
from .hierarchy import (
    SEMANTIC_HIERARCHY,
    Hierarchy,
    Node,
    euclidean_centroid_hierarchy,
    spectral_hierarchy,
)
from .quantization import QuantileQuantizer, confidence, fuzzy_entropy, normalized_entropy
from .separability import SpectralRelations, spectral_relations

INITS = ("graph", "class", "random")


@dataclass
class NodeModel:
    """Frozen FCM partition of one internal node into its child groups."""

    node_id: int
    centres: np.ndarray
    cluster_to_child: np.ndarray
    n_children: int
    m: float
    n_iter: int
    converged: bool
    objective: float
    xie_beni: float
    mapping_is_identity: bool
    n_train: int
    train_confidence: np.ndarray = field(repr=False)
    quantizer: QuantileQuantizer = field(repr=False)

    def child_memberships(self, Z: np.ndarray) -> np.ndarray:
        """Membership of each sample in each child group (clusters summed per child)."""
        U = memberships(Z, self.centres, self.m)
        return U @ np.eye(self.n_children)[self.cluster_to_child]


@dataclass
class NodeTrace:
    """Phase 5 quantities of one node, evaluated for every sample."""

    U: np.ndarray
    child: np.ndarray
    H: np.ndarray
    H_norm: np.ndarray
    C: np.ndarray
    grade: np.ndarray
    cdf: np.ndarray


@dataclass
class RoutingResult:
    """Phase 7 output for a batch of samples.

    ``out_node`` is the node the sample stopped at (a leaf when ``is_fine``).
    The entropy/confidence/grade fields describe the last gating decision made
    on the sample's path (the node where it stopped, or the leaf's parent).
    ``fine_pred`` is the leaf reached by ungated arg-max routing and the
    ``path_min_*`` fields are the minimum confidence, grade and calibrated
    confidence along that path, used as selective-prediction scores.
    """

    out_node: np.ndarray
    depth: np.ndarray
    is_fine: np.ndarray
    fine_pred: np.ndarray
    decision_node: np.ndarray
    H: np.ndarray
    H_norm: np.ndarray
    C: np.ndarray
    grade: np.ndarray
    path_min_conf: np.ndarray
    path_min_grade: np.ndarray
    path_min_cdf: np.ndarray
    hierarchy: Hierarchy = field(repr=False)

    @property
    def pred_label(self) -> np.ndarray:
        """Fine class code for samples that reached a leaf, -1 otherwise."""
        return np.where(self.is_fine, self.fine_pred, -1)

    def pred_sets(self) -> list[tuple[int, ...]]:
        return [self.hierarchy.nodes[n].classes for n in self.out_node]

    def set_correct(self, y_true: np.ndarray) -> np.ndarray:
        """Whether the true class lies in the predicted (possibly coarse) group."""
        return np.array([y in s for y, s in zip(y_true, self.pred_sets())])

    def to_frame(self, y_true: np.ndarray | None = None) -> pd.DataFrame:
        nodes = self.hierarchy.nodes
        df = pd.DataFrame(
            {
                "predicted_class": self.pred_label,
                "predicted_group": [nodes[n].label for n in self.out_node],
                "hierarchy_depth": self.depth,
                "fuzzy_entropy": self.H,
                "normalized_entropy": self.H_norm,
                "confidence": self.C,
                "quantization_grade": self.grade,
                "flag": np.where(self.is_fine, "descend", "stop"),
                "decision_node": [nodes[n].label for n in self.decision_node],
                "ungated_fine_class": self.fine_pred,
                "path_min_confidence": self.path_min_conf,
                "path_min_grade": self.path_min_grade,
                "path_min_calibrated": self.path_min_cdf,
            }
        )
        if y_true is not None:
            df.insert(0, "true_class", np.asarray(y_true))
            df["group_correct"] = self.set_correct(y_true)
            df["fine_correct"] = self.is_fine & (self.fine_pred == np.asarray(y_true))
        return df


class SGHFQ:
    """Spectral-Graph-Guided Hierarchical Fuzzy Quantization classifier.

    Parameters
    ----------
    representation : Phase 1 representation -- "bandmean4" (paper default for
        StatLog), "full36"/"standard" or "pca" (with ``n_components``).
    alpha : JM/Wasserstein blend weight in Eq. 7.
    linkage : agglomerative linkage used on the ambiguity graph.
    hierarchy : "spectral" (Phase 3), "semantic", "euclidean", a nested tuple
        of class codes, or a ``Hierarchy`` instance.
    init : FCM seeding at each node -- "graph" (child-group centroids, Phase 4),
        "class" (one cluster per class seeded at class centroids, memberships
        summed per child group) or "random" (random training samples).
    m, max_iter, tol : FCM fuzzifier and stopping rule.
    n_grades : number of confidence grades L.
    q_threshold : grade needed to descend (Eq. 9); 1 disables the gate.
    cov_estimator : class covariance estimator for JM ("sample" or "ledoit_wolf").
    """

    def __init__(
        self,
        representation: str = "bandmean4",
        alpha: float = 0.5,
        linkage: str = "average",
        hierarchy="spectral",
        init: str = "graph",
        m: float = 2.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_grades: int = 5,
        q_threshold: int = 2,
        jm_ridge: float = 1e-6,
        random_state: int | None = 0,
        n_components: int = 10,
        cov_estimator: str = "sample",
    ):
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        self.representation = representation
        self.alpha = alpha
        self.linkage = linkage
        self.hierarchy = hierarchy
        self.init = init
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.n_grades = n_grades
        self.q_threshold = q_threshold
        self.jm_ridge = jm_ridge
        self.random_state = random_state
        self.n_components = n_components
        self.cov_estimator = cov_estimator

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGHFQ":
        y = np.asarray(y)
        self.rep_ = SpectralRepresentation(self.representation, self.n_components).fit(X)
        Z = self.rep_.transform(X)
        self.classes_ = tuple(int(c) for c in np.unique(y))
        self.relations_: SpectralRelations = spectral_relations(
            Z, y, self.classes_, alpha=self.alpha, ridge=self.jm_ridge, cov_estimator=self.cov_estimator
        )
        self.hierarchy_, self.linkage_matrix_ = self._build_hierarchy(Z, y)
        rng = np.random.default_rng(self.random_state)
        self.node_models_: dict[int, NodeModel] = {
            node.id: self._fit_node(node, Z, y, rng) for node in self.hierarchy_.internal_nodes
        }
        return self

    def _build_hierarchy(self, Z, y) -> tuple[Hierarchy, np.ndarray | None]:
        h = self.hierarchy
        if isinstance(h, Hierarchy):
            return h, None
        if isinstance(h, tuple):
            return Hierarchy(h), None
        if h == "spectral":
            return spectral_hierarchy(self.relations_.ambiguity, self.classes_, self.linkage)
        if h == "semantic":
            return Hierarchy(SEMANTIC_HIERARCHY), None
        if h == "euclidean":
            return euclidean_centroid_hierarchy(Z, y, self.classes_, self.linkage)
        raise ValueError(f"unknown hierarchy {h!r}")

    def _fit_node(self, node: Node, Z: np.ndarray, y: np.ndarray, rng) -> NodeModel:
        mask = np.isin(y, node.classes)
        Zn, yn = Z[mask], y[mask]
        K = len(node.children)
        child_of = np.array([self.hierarchy_.child_index(node, c) for c in yn])

        if self.init == "class":
            # one cluster per class; the matching target of each cluster is a class
            targets = np.searchsorted(np.array(node.classes), yn)
            V0 = np.stack([Zn[yn == c].mean(axis=0) for c in node.classes])
        else:
            targets = child_of
            if self.init == "graph":
                V0 = np.stack([Zn[child_of == k].mean(axis=0) for k in range(K)])
            else:
                V0 = random_centres(Zn, K, rng)
        res = fcm(Zn, V0, m=self.m, max_iter=self.max_iter, tol=self.tol)

        # Label clusters by maximum soft agreement with the training groups
        # (Hungarian matching); for seeded runs this is normally the identity.
        n_targets = len(V0)
        agreement = res.U.T @ np.eye(n_targets)[targets]
        rows, cols = linear_sum_assignment(-agreement)
        cluster_to_target = np.empty(n_targets, dtype=int)
        cluster_to_target[rows] = cols
        if self.init == "class":
            cluster_to_child = np.array(
                [self.hierarchy_.child_index(node, node.classes[t]) for t in cluster_to_target]
            )
        else:
            cluster_to_child = cluster_to_target

        G = res.U @ np.eye(K)[cluster_to_child]
        C_train = confidence(G)
        return NodeModel(
            node_id=node.id,
            centres=res.centres,
            cluster_to_child=cluster_to_child,
            n_children=K,
            m=self.m,
            n_iter=res.n_iter,
            converged=res.converged,
            objective=res.objective,
            xie_beni=xie_beni(Zn, res.U, res.centres, self.m),
            mapping_is_identity=bool(np.all(cluster_to_target == np.arange(n_targets))),
            n_train=len(Zn),
            train_confidence=C_train,
            quantizer=QuantileQuantizer(self.n_grades).fit(C_train),
        )

    # ------------------------------------------------------------ inference
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.rep_.transform(X)

    def node_trace(self, X: np.ndarray) -> dict[int, NodeTrace]:
        """Evaluate every internal node's partition on every sample."""
        Z = self.transform(X)
        out = {}
        for nid, nm in self.node_models_.items():
            G = nm.child_memberships(Z)
            C = confidence(G)
            out[nid] = NodeTrace(
                U=G,
                child=G.argmax(axis=1),
                H=fuzzy_entropy(G),
                H_norm=normalized_entropy(G),
                C=C,
                grade=nm.quantizer.transform(C),
                cdf=nm.quantizer.cdf(C),
            )
        return out

    def predict(
        self,
        X: np.ndarray,
        q_threshold: int | None = None,
        conf_threshold: float | None = None,
        trace: dict[int, NodeTrace] | None = None,
    ) -> RoutingResult:
        """Route samples down the hierarchy (Phase 6).

        By default the gate is the quantised grade: descend iff grade >=
        ``q_threshold``. Passing ``conf_threshold`` instead gates on the raw
        confidence (descend iff C >= threshold), which is the uncalibrated
        "entropy gating" baseline.
        """
        trace = trace if trace is not None else self.node_trace(X)
        if conf_threshold is not None:
            descend = {nid: t.C >= conf_threshold for nid, t in trace.items()}
        else:
            thr = self.q_threshold if q_threshold is None else q_threshold
            descend = {nid: t.grade >= thr for nid, t in trace.items()}
        return route(self.hierarchy_, trace, descend)

    def predict_fine(self, X: np.ndarray) -> np.ndarray:
        """Ungated arg-max routing to a leaf class."""
        return self.predict(X, q_threshold=1).fine_pred

    def train_confidences(self) -> np.ndarray:
        """Training confidences pooled over all internal nodes."""
        return np.concatenate([nm.train_confidence for nm in self.node_models_.values()])

    def summary(self) -> pd.DataFrame:
        rows = []
        for nid, nm in self.node_models_.items():
            node = self.hierarchy_.nodes[nid]
            rows.append(
                {
                    "node": node.label,
                    "depth": node.depth,
                    "children": " | ".join(c.label for c in node.children),
                    "n_train": nm.n_train,
                    "fcm_iterations": nm.n_iter,
                    "converged": nm.converged,
                    "objective": nm.objective,
                    "xie_beni": nm.xie_beni,
                    "mapping_identity": nm.mapping_is_identity,
                    "grade_thresholds": np.round(nm.quantizer.thresholds_, 4).tolist(),
                }
            )
        return pd.DataFrame(rows)


def route(hierarchy: Hierarchy, trace: dict[int, NodeTrace], descend: dict[int, np.ndarray]) -> RoutingResult:
    """Walk every sample from the root, following the arg-max child.

    The gated walk stops at the first node whose ``descend`` flag is False;
    the ungated walk always continues to a leaf (``fine_pred``).
    """
    n = len(next(iter(trace.values())).C)
    out_node = np.zeros(n, dtype=int)
    decision_node = np.zeros(n, dtype=int)
    fine_pred = np.zeros(n, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    H = np.zeros(n)
    Hn = np.zeros(n)
    C = np.zeros(n)
    grade = np.zeros(n, dtype=int)
    pm_conf = np.full(n, np.inf)
    pm_grade = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    pm_cdf = np.full(n, np.inf)

    current = np.zeros(n, dtype=int)  # ungated position
    for _ in range(hierarchy.max_depth):
        for node in hierarchy.internal_nodes:
            at = np.flatnonzero(current == node.id)
            if at.size == 0:
                continue
            t = trace[node.id]
            pm_conf[at] = np.minimum(pm_conf[at], t.C[at])
            pm_grade[at] = np.minimum(pm_grade[at], t.grade[at])
            pm_cdf[at] = np.minimum(pm_cdf[at], t.cdf[at])
            live = at[~stopped[at]]
            decision_node[live] = node.id
            H[live], Hn[live], C[live], grade[live] = t.H[live], t.H_norm[live], t.C[live], t.grade[live]
            halt = live[~descend[node.id][live]]
            stopped[halt] = True
            out_node[halt] = node.id
            child_ids = np.array([c.id for c in node.children])
            current[at] = child_ids[t.child[at]]
    for leaf in hierarchy.leaves:
        at = current == leaf.id
        fine_pred[at] = leaf.classes[0]
    out_node[~stopped] = current[~stopped]
    depth = np.array([hierarchy.nodes[k].depth for k in out_node])
    return RoutingResult(
        out_node=out_node,
        depth=depth,
        is_fine=~stopped,
        fine_pred=fine_pred,
        decision_node=decision_node,
        H=H,
        H_norm=Hn,
        C=C,
        grade=grade,
        path_min_conf=pm_conf,
        path_min_grade=pm_grade,
        path_min_cdf=pm_cdf,
        hierarchy=hierarchy,
    )
