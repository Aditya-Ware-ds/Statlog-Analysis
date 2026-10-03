import numpy as np
import pytest

from sg_hfq.baselines import FlatFCM
from sg_hfq.data import N_FEATURES
from sg_hfq.model import SGHFQ


def _as_36(X):
    """Embed 4-D samples as 3x3 neighbourhoods of identical pixels (pixel-major layout)."""
    return np.tile(X, 9)


@pytest.fixture
def fitted(blobs):
    X, y = blobs
    return SGHFQ(n_grades=4, q_threshold=2).fit(_as_36(X), y), _as_36(X), y


def test_shapes_and_hierarchy(fitted):
    m, X, y = fitted
    assert X.shape[1] == N_FEATURES
    assert m.hierarchy_.classes == (1, 2, 3, 4)
    # classes 3 and 4 are the most ambiguous pair and must be merged first
    assert any(n.classes == (3, 4) for n in m.hierarchy_.nodes)
    assert set(m.node_models_) == {n.id for n in m.hierarchy_.internal_nodes}


def test_separable_data_is_classified(fitted):
    m, X, y = fitted
    assert (m.predict_fine(X) == y).mean() > 0.95


def test_gate_extremes(fitted):
    m, X, y = fitted
    all_fine = m.predict(X, q_threshold=1)
    assert all_fine.is_fine.all() and np.array_equal(all_fine.pred_label, all_fine.fine_pred)
    none = m.predict(X, q_threshold=m.n_grades + 1)
    assert (none.out_node == 0).all() and not none.is_fine.any() and (none.depth == 0).all()
    assert none.set_correct(y).all()


def test_gating_matches_bruteforce(fitted):
    m, X, y = fitted
    trace = m.node_trace(X)
    res = m.predict(X, q_threshold=3, trace=trace)
    for i in range(len(y)):
        node, out, gmin = m.hierarchy_.root, None, np.inf
        while not node.is_leaf:
            t = trace[node.id]
            gmin = min(gmin, t.grade[i])
            if out is None and t.grade[i] < 3:
                out = node.id
            node = node.children[t.child[i]]
        assert res.fine_pred[i] == node.classes[0]
        assert res.out_node[i] == (node.id if out is None else out)
        assert res.path_min_grade[i] == gmin


def test_phase7_frame(fitted):
    m, X, y = fitted
    df = m.predict(X).to_frame(y)
    for col in ("predicted_class", "hierarchy_depth", "fuzzy_entropy", "normalized_entropy", "confidence",
                "quantization_grade", "flag"):
        assert col in df
    assert set(df["flag"]) <= {"descend", "stop"}
    assert df["normalized_entropy"].between(0, 1).all()


@pytest.mark.parametrize("init", ["graph", "class", "random"])
def test_inits_run(blobs, init):
    X, y = blobs
    m = SGHFQ(init=init, hierarchy=((1, 2), (3, 4))).fit(_as_36(X), y)
    assert m.hierarchy_.nested == ((1, 2), (3, 4))
    assert (m.predict_fine(_as_36(X)) == y).mean() > 0.9


def test_flat_fcm(blobs):
    X, y = blobs
    f = FlatFCM().fit(_as_36(X), y)
    s = f.scores(_as_36(X))
    assert (s["pred"] == y).mean() > 0.95
    assert np.all((s["max_membership"] >= 0.25) & (s["max_membership"] <= 1))
