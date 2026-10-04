import numpy as np
import pytest

from dirichlet_gp.data import DEFAULT_DATA_DIR, OFFICIAL_COUNTS, Standardizer, load_statlog


def test_standardizer_uses_training_statistics():
    rng = np.random.default_rng(0)
    X = rng.normal(50, 10, size=(100, 36))
    Z = Standardizer("standard").fit_transform(X)
    assert Z.shape == (100, 36) and np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1)
    assert np.allclose(Standardizer("full36").fit_transform(X), Z)
    L = Standardizer("log36").fit_transform(np.abs(X) + 1)
    assert np.allclose(L.mean(0), 0)
    with pytest.raises(ValueError):
        Standardizer("pca")


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "sat.trn").exists(), reason="StatLog data not present")
def test_official_split():
    d = load_statlog()
    assert d.X_train.shape == (4435, 36) and d.X_test.shape == (2000, 36)
    for split, y in (("train", d.y_train), ("test", d.y_test)):
        codes, counts = np.unique(y, return_counts=True)
        assert dict(zip(codes.tolist(), counts.tolist())) == OFFICIAL_COUNTS[split]
    assert d.X_train.min() >= 0 and d.X_train.max() <= 255
