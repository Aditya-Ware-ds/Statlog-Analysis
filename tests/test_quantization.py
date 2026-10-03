import numpy as np
import pytest

from sg_hfq.quantization import QuantileQuantizer, confidence, fuzzy_entropy, normalized_entropy


def test_entropy_extremes():
    U = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    assert fuzzy_entropy(U) == pytest.approx([0.0, np.log(3)])
    assert normalized_entropy(U) == pytest.approx([0.0, 1.0])
    assert confidence(U) == pytest.approx([1.0, 0.0])


def test_quantiles_give_balanced_monotone_grades():
    rng = np.random.default_rng(0)
    C = rng.uniform(size=10_000)
    q = QuantileQuantizer(5).fit(C)
    g = q.transform(C)
    assert set(np.unique(g)) == {1, 2, 3, 4, 5}
    assert np.allclose(np.bincount(g)[1:] / len(g), 0.2, atol=0.01)
    order = np.argsort(C)
    assert np.all(np.diff(g[order]) >= 0)


def test_grade_boundaries_follow_eq8():
    q = QuantileQuantizer(4).fit(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    t = q.thresholds_
    assert q.transform(np.array([t[0], t[0] + 1e-9, t[2], t[2] + 1e-9])).tolist() == [1, 2, 3, 4]


def test_cdf_matches_grade():
    rng = np.random.default_rng(1)
    C = rng.uniform(size=1000)
    q = QuantileQuantizer(5).fit(C)
    x = rng.uniform(size=200)
    F, g = q.cdf(x), q.transform(x)
    assert np.all(np.diff(F[np.argsort(x)]) >= 0)
    assert np.all(np.abs(g - (1 + np.floor(5 * F))) <= 1)
