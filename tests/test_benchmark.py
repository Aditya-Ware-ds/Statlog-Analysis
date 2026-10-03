import numpy as np
import pytest

from sg_hfq.benchmark import classification_metrics
from sg_hfq.datasets import capped_sample, stratified_split


def test_stratified_split_is_disjoint_and_covers_all():
    y = np.repeat([1, 2, 3], [100, 40, 12])
    tr, va, te = stratified_split(y, seed=0, train_frac=0.1, val_frac=0.05)
    assert len(set(tr) | set(va) | set(te)) == len(y)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert (y[tr] == 3).sum() == 5 and (y[va] == 3).sum() == 3  # minimum counts kick in
    assert (y[tr] == 1).sum() == 10
    tr2, *_ = stratified_split(y, seed=0, train_frac=0.1, val_frac=0.05)
    assert np.array_equal(tr, tr2)


def test_stratified_split_rejects_tiny_class():
    with pytest.raises(ValueError):
        stratified_split(np.repeat([1, 2], [100, 6]), 0, 0.1, 0.05)


def test_capped_sample():
    y = np.repeat([0, 1], [1000, 7])
    idx = capped_sample(np.arange(len(y)), y, 50, np.random.default_rng(0))
    assert (y[idx] == 0).sum() == 50 and (y[idx] == 1).sum() == 7


def test_classification_metrics():
    y = np.array([1, 1, 1, 1, 2, 2])
    p = np.array([1, 1, 1, 2, 2, 1])
    m, recall = classification_metrics(y, p, (1, 2))
    assert m["OA"] == pytest.approx(4 / 6)
    assert recall == {1: 0.75, 2: 0.5} and m["AA"] == pytest.approx(0.625)
    assert -1 <= m["kappa"] <= 1
