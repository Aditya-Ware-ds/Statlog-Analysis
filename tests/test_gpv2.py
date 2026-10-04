import numpy as np
import pandas as pd
import pytest

from dirichlet_gp.gpv2 import band_series_kernel, fit_gp_v2, main, make_gp_v2
from dirichlet_gp.tuned import sentinel2_kernel


def _spectra(rng, n_per_class=40, bands=12, labels=(0, 1)):
    t = np.linspace(0, 1, bands)
    shapes = [1 + 0.5 * t, 1.5 - 0.5 * t]
    X = np.vstack([rng.uniform(50, 150, (n_per_class, 1)) * (s + rng.normal(0, 0.05, (n_per_class, bands)))
                   for s in shapes])
    return X, np.repeat(np.array(labels), n_per_class)


def test_fit_gp_v2_spectral_with_validation():
    rng = np.random.default_rng(0)
    X, y = _spectra(rng, 40)
    Xv, yv = _spectra(rng, 20)
    Xt, yt = _spectra(rng, 30)
    gp = fit_gp_v2(X, y, Xv, yv, kernel="spectral", alphas=(0.01, 1.0), scales=(0.5, 1.0), max_iter=15,
                   verbose=False)
    assert {"alpha_eps", "scale", "temperature", "accuracy"} <= set(gp.selection_)
    assert len(gp.selection_table_) == 4 and np.isfinite(gp.log_evidence_)
    assert gp._D_opt is None  # training cache released
    P = gp.predict_proba(Xt)
    assert P.shape == (60, 2) and np.allclose(P.sum(1), 1) and (gp.predict(Xt) == yt).mean() > 0.95
    with pytest.raises(RuntimeError):
        gp.log_marginal_likelihood()


def test_fit_gp_v2_string_labels_priors_without_validation():
    rng = np.random.default_rng(1)
    X, y = _spectra(rng, 30, labels=("crop", "meadow"))
    gp = fit_gp_v2(X, y, kernel="spectral", target_priors={"crop": 0.9, "meadow": 0.1}, max_iter=10,
                   verbose=False)
    assert list(gp.classes_) == ["crop", "meadow"] and gp.selection_["prior_tau"] == 1.0
    assert np.allclose(gp.prior_ratio_, [0.9 / 0.5, 0.1 / 0.5])
    assert set(gp.predict(X)) <= {"crop", "meadow"}


def test_series_kernel_with_validation_and_priors():
    rng = np.random.default_rng(2)
    X, y = _spectra(rng, 30, bands=12)  # read as 4 bands x 3 dates
    Xv, yv = _spectra(rng, 15, bands=12)
    gp = fit_gp_v2(X, y, Xv, yv, kernel="series", n_periods=3, target_priors=[0.7, 0.3],
                   alphas=(0.01,), scales=(1.0, 4.0), max_iter=10, verbose=False)
    assert "prior_tau" in gp.selection_ and gp.prior_ratio_ is not None
    assert (gp.predict(Xv) == yv).mean() > 0.9


def test_band_series_kernel_is_the_sentinel2_kernel():
    rng = np.random.default_rng(3)
    X = rng.uniform(100, 3000, (25, 60))
    fa, ka = band_series_kernel(10, 6)
    fb, kb = sentinel2_kernel("band+angle+linear")
    Za, Zb = fa.fit_transform(X), fb.fit_transform(X)
    theta = rng.normal(0, 0.5, ka.n_params)
    assert ka.n_params == kb.n_params == 23 and np.allclose(Za, Zb)
    assert np.allclose(ka(Za, Za, theta), kb(Zb, Zb, theta))


def test_make_gp_v2_rejects_bad_settings():
    with pytest.raises(ValueError):
        make_gp_v2(12, "series")
    with pytest.raises(ValueError):
        make_gp_v2(12, "nope")
    with pytest.raises(ValueError):
        make_gp_v2(2, "spectral")
    assert make_gp_v2(5, "ard").kernel == "matern52"


def test_command_line(tmp_path):
    rng = np.random.default_rng(4)
    files = {}
    for name, n in (("train", 30), ("val", 15), ("new", 20)):
        X, y = _spectra(rng, n, bands=8, labels=("a", "b"))
        df = pd.DataFrame(X, columns=[f"b{j}" for j in range(8)])
        df.insert(0, "pixel_id", np.arange(len(df)))
        df["label"] = y
        files[name] = tmp_path / f"{name}.csv"
        df.to_csv(files[name], index=False)
    out = tmp_path / "pred.csv"
    main(["--train", str(files["train"]), "--val", str(files["val"]), "--predict", str(files["new"]),
          "--drop", "pixel_id", "--out", str(out), "--target-priors-from", str(files["train"]),
          "--save", str(tmp_path / "model.pkl")])
    pred = pd.read_csv(out)
    assert list(pred.columns) == ["pixel_id", "predicted", "confidence", "p_a", "p_b"]
    truth = pd.read_csv(files["new"])["label"]
    assert len(pred) == 40 and (pred["predicted"] == truth).mean() > 0.9
    assert (tmp_path / "model.pkl").stat().st_size > 0
