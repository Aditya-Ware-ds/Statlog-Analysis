"""Phase 3: data-driven class hierarchy.

The spectral-ambiguity matrix A defines a complete weighted graph over the
classes. Agglomerative clustering on the graph (distance ``1 - A_ij``) yields a
dendrogram, which is turned into a coarse-to-fine tree: every internal node
holds a group of classes and its children partition that group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Union

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform

Nested = Union[int, tuple["Nested", ...]]

#: Hand-crafted semantic hierarchy used for the "HFCM" (no spectral graph)
#: ablation: vegetation {cotton crop, vegetation stubble} vs bare soil, then
#: red soil vs the grey-soil moisture family {grey, damp grey, very damp grey}.
SEMANTIC_HIERARCHY: Nested = ((2, 5), (1, (3, 4, 7)))


@dataclass(eq=False)
class Node:
    id: int
    classes: tuple[int, ...]
    depth: int
    parent: int | None
    children: list["Node"] = field(default_factory=list)
    height: float | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def label(self) -> str:
        return "{" + ",".join(str(c) for c in self.classes) + "}"


def _leaves(nested: Nested) -> list[int]:
    if isinstance(nested, (int, np.integer)):
        return [int(nested)]
    return [leaf for child in nested for leaf in _leaves(child)]


def _canonical(nested: Nested) -> Nested:
    """Order children by their smallest class code so trees print deterministically."""
    if isinstance(nested, (int, np.integer)):
        return int(nested)
    kids = [_canonical(k) for k in nested]
    if len(kids) == 1:
        return kids[0]
    return tuple(sorted(kids, key=lambda k: min(_leaves(k))))


class Hierarchy:
    """A coarse-to-fine class tree. Node ids follow breadth-first order (root = 0)."""

    def __init__(self, nested: Nested, heights: dict[tuple[int, ...], float] | None = None):
        nested = _canonical(nested)
        leaves = _leaves(nested)
        if len(set(leaves)) != len(leaves):
            raise ValueError("a class appears more than once in the hierarchy")
        if isinstance(nested, int):
            raise ValueError("a hierarchy needs at least two classes")
        self.nested = nested
        self.classes = tuple(sorted(leaves))
        heights = heights or {}
        self.nodes: list[Node] = []
        queue = [(nested, 0, None)]
        while queue:
            sub, depth, parent = queue.pop(0)
            classes = tuple(sorted(_leaves(sub)))
            node = Node(len(self.nodes), classes, depth, parent, height=heights.get(classes))
            self.nodes.append(node)
            if parent is not None:
                self.nodes[parent].children.append(node)
            if not isinstance(sub, int):
                queue.extend((child, depth + 1, node.id) for child in sub)
        self._leaf_of = {n.classes[0]: n for n in self.nodes if n.is_leaf}

    # -- construction -----------------------------------------------------
    @classmethod
    def from_linkage(cls, Z: np.ndarray, classes) -> "Hierarchy":
        classes = tuple(int(c) for c in classes)
        heights: dict[tuple[int, ...], float] = {}

        def build(cn) -> Nested:
            if cn.is_leaf():
                return classes[cn.id]
            nested = (build(cn.get_left()), build(cn.get_right()))
            heights[tuple(sorted(_leaves(nested)))] = float(cn.dist)
            return nested

        return cls(build(to_tree(Z)), heights)

    # -- accessors ----------------------------------------------------------
    @property
    def root(self) -> Node:
        return self.nodes[0]

    @property
    def internal_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.is_leaf]

    @property
    def leaves(self) -> list[Node]:
        return [n for n in self.nodes if n.is_leaf]

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes)

    def leaf_for(self, cls_code: int) -> Node:
        return self._leaf_of[int(cls_code)]

    def path(self, cls_code: int) -> list[Node]:
        """Nodes from the root down to the leaf of ``cls_code``."""
        node, out = self.leaf_for(cls_code), []
        while node is not None:
            out.append(node)
            node = self.nodes[node.parent] if node.parent is not None else None
        return out[::-1]

    def child_index(self, node: Node, cls_code: int) -> int:
        """Index of the child of ``node`` whose group contains ``cls_code``."""
        for k, child in enumerate(node.children):
            if cls_code in child.classes:
                return k
        raise KeyError(f"class {cls_code} not under node {node.label}")

    def walk(self) -> Iterator[Node]:
        return iter(self.nodes)

    def describe(self, names: dict[int, str] | None = None) -> str:
        lines: list[str] = []

        def rec(node: Node, prefix: str, last: bool, root: bool) -> None:
            text = node.label
            if node.is_leaf and names:
                text += f" {names[node.classes[0]]}"
            if node.height is not None:
                text += f"  (merge distance {node.height:.3f})"
            lines.append(text if root else prefix + ("`-- " if last else "|-- ") + text)
            ext = "" if root else prefix + ("    " if last else "|   ")
            for k, child in enumerate(node.children):
                rec(child, ext, k == len(node.children) - 1, False)

        rec(self.root, "", True, True)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "nested": self.nested,
            "nodes": [
                {
                    "id": n.id,
                    "classes": list(n.classes),
                    "depth": n.depth,
                    "parent": n.parent,
                    "children": [c.id for c in n.children],
                    "merge_distance": n.height,
                }
                for n in self.nodes
            ],
        }

    def __repr__(self) -> str:
        return f"Hierarchy({self.nested!r})"


def graph_linkage(A: np.ndarray, method: str = "average") -> np.ndarray:
    """Agglomerative clustering of the ambiguity graph with edge distance 1 - A_ij."""
    D = 1.0 - np.asarray(A, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(np.clip(D, 0.0, None), checks=False), method=method)


def spectral_hierarchy(A: np.ndarray, classes, method: str = "average") -> tuple[Hierarchy, np.ndarray]:
    Z = graph_linkage(A, method)
    return Hierarchy.from_linkage(Z, classes), Z


def euclidean_centroid_hierarchy(X, y, classes, method: str = "average") -> tuple[Hierarchy, np.ndarray]:
    """Baseline hierarchy from Euclidean distances between class centroids (no spectral graph)."""
    centroids = np.stack([np.asarray(X)[np.asarray(y) == c].mean(axis=0) for c in classes])
    Z = linkage(pdist(centroids), method=method)
    return Hierarchy.from_linkage(Z, classes), Z
