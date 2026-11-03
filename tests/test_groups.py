"""fit(group_col=...): several independent series (a fleet) in one DataFrame."""

import numpy as np
import pandas as pd
import pytest

from apdtflow import APDTFlowForecaster

H, T = 8, 4


def _model(**kw):
    kw = {"forecast_horizon": T, "history_length": H, "num_epochs": 1, "hidden_dim": 8,
          "num_scales": 2, "device": "cpu", "verbose": False, **kw}
    return APDTFlowForecaster(**kw)


def _fleet(lengths, interleave=False):
    """Unit k's values lie in [10k, 10k + 1), so a window spanning two units is easy to see."""
    rng = np.random.default_rng(0)
    frames = []
    for k, n in enumerate(lengths):
        frames.append(pd.DataFrame({
            "unit": f"u{k}",
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "y": 10 * k + rng.uniform(0, 1, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    if interleave:
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df


def _windows(m):
    """Calibration windows (the training windows by default), denormalized."""
    return m._calib_X_[:, 0, :].numpy() * m.scaler_std_ + m.scaler_mean_


def test_windows_never_span_two_units():
    lengths = [30, 25, 40]
    m = _model(use_conformal=True)
    m.fit(_fleet(lengths), target_col="y", group_col="unit")
    expected = sum(n - H - T + 1 for n in lengths)
    assert len(m.window_groups_) == len(m._calib_X_) == expected
    units = np.floor(_windows(m) / 10)
    assert (units.min(axis=1) == units.max(axis=1)).all()
    assert m.groups_ == ["u0", "u1", "u2"] and m.group_col_ == "unit"
    # without group_col the same rows make windows across the seams
    flat = _model(use_conformal=True)
    flat.fit(_fleet(lengths), target_col="y")
    assert len(flat._calib_X_) == sum(lengths) - H - T + 1 > expected


def test_rows_are_ordered_by_unit_then_date():
    m = _model(use_conformal=True)
    shuffled = _fleet([30, 25], interleave=True)
    m.fit(shuffled, target_col="y", date_col="date", group_col="unit")
    units = np.floor(_windows(m) / 10)
    assert (units.min(axis=1) == units.max(axis=1)).all()
    assert m.groups_ == list(pd.unique(shuffled["unit"]))  # order of first appearance
    assert m.freq_delta_ == pd.Timedelta(days=1)  # measured within units, not across them
    last_unit = m.groups_[-1]
    df = _fleet([30, 25])
    assert m.last_date_ == df.loc[df["unit"] == last_unit, "date"].max()
    # predict() continues the last unit
    assert np.floor(m.last_sequence_ * m.scaler_std_ + m.scaler_mean_).min() == 10 * int(last_unit[1])


def test_held_out_calibration_takes_whole_units():
    lengths = [30, 30, 30, 50]
    m = _model(use_conformal=True, calibration_split=0.25)
    m.fit(_fleet(lengths), target_col="y", group_col="unit")
    assert len(m._calib_X_) == lengths[-1] - H - T + 1
    assert (np.floor(_windows(m) / 10) == 3).all()  # only the last unit


def test_held_out_calibration_falls_back_with_one_unit():
    m = _model(use_conformal=True, calibration_split=0.5)
    with pytest.warns(UserWarning, match="too few to hold out"):
        m.fit(_fleet([40]), target_col="y", group_col="unit")
    assert len(m._calib_X_) == 40 - H - T + 1


def test_short_units_are_skipped_and_all_short_is_an_error():
    m = _model()
    m.fit(_fleet([30, 5, 30]), target_col="y", group_col="unit")
    assert set(m.window_groups_.tolist()) == {0, 2}
    with pytest.raises(ValueError, match="in at least one group"):
        _model().fit(_fleet([11, 11, 11]), target_col="y", group_col="unit")  # each unit < H + T


def test_group_col_input_errors():
    df = _fleet([30, 30])
    with pytest.raises(ValueError, match="not found"):
        _model().fit(df, target_col="y", group_col="asset")
    bad = df.copy()
    bad.loc[3, "unit"] = None
    with pytest.raises(ValueError, match="missing values"):
        _model().fit(bad, target_col="y", group_col="unit")
    with pytest.raises(ValueError, match="DataFrame"):
        _model().fit(df["y"].to_numpy(), group_col="unit")


def test_multivariate_and_save_load_keep_the_units(tmp_path):
    df = _fleet([30, 30])
    df["sensor"] = df["y"] * 2
    m = _model(use_conformal=True, decoder_type="continuous")
    m.fit(df, target_col="y", feature_cols=["sensor"], group_col="unit")
    assert len(m._calib_X_) == 2 * (30 - H - T + 1)
    path = tmp_path / "fleet.pt"
    m.save(str(path))
    loaded = APDTFlowForecaster.load(str(path), device="cpu")
    assert loaded.group_col_ == "unit" and loaded.groups_ == ["u0", "u1"]
    np.testing.assert_allclose(loaded.predict(), m.predict(), rtol=1e-5)


def test_refitting_without_groups_forgets_them():
    m = _model(use_conformal=True)
    m.fit(_fleet([30, 30]), target_col="y", group_col="unit")
    m.fit(_fleet([60]), target_col="y")
    assert m.group_col_ is None and m.groups_ is None and m.window_groups_ is None
    assert len(m._calib_X_) == 60 - H - T + 1
