import numpy as np
import pytest

from sg_hfq.hierarchy import SEMANTIC_HIERARCHY, Hierarchy, spectral_hierarchy


def test_spectral_hierarchy_merges_most_ambiguous_first():
    classes = (1, 2, 3, 4)
    A = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.6],
        [0.1, 0.1, 0.6, 1.0],
    ])
    h, Z = spectral_hierarchy(A, classes)
    assert h.nested == ((1, 2), (3, 4))
    assert h.root.height == pytest.approx(0.9)
    assert [n.depth for n in h.leaves] == [2, 2, 2, 2]


def test_nested_tree_structure():
    h = Hierarchy(SEMANTIC_HIERARCHY)
    assert h.classes == (1, 2, 3, 4, 5, 7)
    assert h.root.classes == (1, 2, 3, 4, 5, 7)
    assert [n.label for n in h.root.children] == ["{1,3,4,7}", "{2,5}"]
    assert [n.label for n in h.path(4)] == ["{1,2,3,4,5,7}", "{1,3,4,7}", "{3,4,7}", "{4}"]
    node = h.nodes[h.path(4)[2].id]
    assert len(node.children) == 3 and h.child_index(node, 7) == 2
    assert h.max_depth == 3
    assert [n.id for n in h.nodes] == list(range(len(h.nodes)))


def test_rejects_duplicate_classes():
    with pytest.raises(ValueError):
        Hierarchy(((1, 2), (2, 3)))
