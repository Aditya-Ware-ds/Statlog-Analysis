import numpy as np
import pytest


@pytest.fixture
def blobs():
    """Three well-separated Gaussian classes in 4-D, labelled 1, 2, 3 (plus a nearby class 4)."""
    rng = np.random.default_rng(0)
    centres = {1: [0, 0, 0, 0], 2: [8, 0, 0, 0], 3: [0, 8, 0, 0], 4: [0, 8, 4, 0]}
    X = np.vstack([rng.normal(centres[c], 0.6, size=(120, 4)) for c in centres])
    y = np.repeat(list(centres), 120)
    return X, y
