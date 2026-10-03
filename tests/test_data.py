import numpy as np
import pytest

from sg_hfq.data import DEFAULT_DATA_DIR, OFFICIAL_COUNTS, SpectralRepresentation, band_mean, load_statlog


def test_band_mean_layout():
    # pixel-major: feature 4*p + b is band b of pixel p
    X = np.array([[b + 10 * p for p in range(9) for b in range(4)]], dtype=float)
    assert band_mean(X).tolist() == [[40.0, 41.0, 42.0, 43.0]]


def test_representation_standardises_with_training_stats():
    rng = np.random.default_rng(0)
    X = rng.normal(50, 10, size=(100, 36))
    rep = SpectralRepresentation("bandmean4").fit(X)
    Z = rep.transform(X)
    assert Z.shape == (100, 4) and np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1)
    assert SpectralRepresentation("full36").fit_transform(X).shape == (100, 36)
    with pytest.raises(ValueError):
        SpectralRepresentation("pca")


@pytest.mark.skipif(not (DEFAULT_DATA_DIR / "sat.trn").exists(), reason="StatLog data not present")
def test_official_split():
    d = load_statlog()
    assert d.X_train.shape == (4435, 36) and d.X_test.shape == (2000, 36)
    for split, y in (("train", d.y_train), ("test", d.y_test)):
        codes, counts = np.unique(y, return_counts=True)
        assert dict(zip(codes.tolist(), counts.tolist())) == OFFICIAL_COUNTS[split]
    assert d.X_train.min() >= 0 and d.X_train.max() <= 255
