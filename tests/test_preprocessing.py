import pytest
import pandas as pd
import numpy as np
from apdtflow.preprocessing import impute_missing_values, detrend_series, TimeSeriesScaler

def test_impute_missing_values():
    data = pd.Series([1, np.nan, 3, np.nan, 5])
    imputed_ffill = impute_missing_values(data, method="ffill")
    assert imputed_ffill.isna().sum() == 0
    imputed_mean = impute_missing_values(data, method="mean")
    expected_mean = data.mean()
    assert np.isclose(imputed_mean.iloc[1], expected_mean)
    
def test_detrend_series():
    data = pd.Series(np.arange(10))
    detrended = detrend_series(data, method="difference", order=1)
    assert len(detrended) == 9
    assert (detrended == 1).all()

def test_scaler_inverse_transform():
    data = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
    scaler = TimeSeriesScaler(scaler_type="standard")
    scaled = scaler.fit_transform(data)
    original = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(data, original, rtol=1e-5)
