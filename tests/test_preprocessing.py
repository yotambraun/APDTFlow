import pandas as pd
import numpy as np
from apdtflow.preprocessing import (
    convert_to_datetime,
    fill_time_gaps,
    impute_missing_values,
    detrend_series,
    generate_lag_features,
    TimeSeriesScaler,
)


def test_convert_to_datetime(tmp_path):
    df = pd.DataFrame({"date_str": ["2020-01-01", "2020-01-03", "2020-01-02"]})
    df_converted = convert_to_datetime(df, "date_str")
    assert pd.api.types.is_datetime64_any_dtype(df_converted["date_str"])


def test_fill_time_gaps():
    df = pd.DataFrame(
        {"DATE": pd.to_datetime(["2020-01-01", "2020-01-03"]), "value": [1, 3]}
    )
    df_filled = fill_time_gaps(df, "DATE", freq="D")
    assert len(df_filled) == 3


def test_impute_missing_values():
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    imputed = impute_missing_values(series, method="linear")
    assert imputed.isna().sum() == 0
    np.testing.assert_allclose(imputed.iloc[1], 2.0, rtol=1e-5)
    np.testing.assert_allclose(imputed.iloc[3], 4.0, rtol=1e-5)


def test_detrend_series():
    series = pd.Series(np.arange(10, dtype=np.float32))
    detrended = detrend_series(series, method="difference", order=1)
    np.testing.assert_array_equal(detrended.values, np.ones(9, dtype=np.float32))


def test_generate_lag_features():
    series = pd.Series(np.arange(5))
    df_lags = generate_lag_features(series, lags=[1, 2])
    assert len(df_lags) == 5 - 2
    assert "lag_1" in df_lags.columns
    assert "lag_2" in df_lags.columns


def test_scaler_inverse_transform():
    data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
    scaler = TimeSeriesScaler(scaler_type="standard")
    scaled = scaler.fit_transform(data)
    original = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(data, original, rtol=1e-5)
