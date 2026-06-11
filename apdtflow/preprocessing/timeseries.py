"""
Preprocessing Module for Time Series Data

This module provides a comprehensive set of functions to prepare time series data
for forecasting models in APDTFlow. It includes routines for:

  - Converting various timestamp formats to pandas datetime
  - Sorting by time and re-indexing to fill in gaps (i.e. missing rows)
  - Missing data imputation with several strategies
  - Detrending (e.g. differencing)
  - Seasonal decomposition
  - Feature engineering (lag features, rolling statistics)
  - Scaling (standard, min-max, robust)

After applying these steps the data is ready to be converted into a sliding‑window
dataset for our models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose


def convert_to_datetime(df, timestamp_col, date_format=None):
    """
    Convert a column in a DataFrame to pandas datetime. If a specific format is provided,
    it is used; otherwise, pandas attempts to infer the format.

    Args:
        df (pd.DataFrame): The data frame containing the timestamp.
        timestamp_col (str): The column name for the timestamp.
        date_format (str, optional): A strftime format string.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the timestamp column converted.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col], format=date_format, errors="coerce"
    )
    df = df.dropna(subset=[timestamp_col])
    return df


def fill_time_gaps(df, timestamp_col, freq="D"):
    """
    Ensure that the DataFrame has a row for every timestamp at the specified frequency.
    Missing rows are inserted with NaN for all other columns.

    Args:
        df (pd.DataFrame): DataFrame with a datetime column.
        timestamp_col (str): Name of the datetime column.
        freq (str): Frequency string (e.g. "D" for daily, "H" for hourly).

    Returns:
        pd.DataFrame: DataFrame reindexed to include all timestamps.
    """
    df = df.copy()
    df = df.sort_values(timestamp_col)
    df.set_index(timestamp_col, inplace=True)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = timestamp_col
    return df.reset_index()


def impute_missing_values(series, method="ffill"):
    """
    Impute missing values in a pandas Series.

    Args:
        series (pd.Series): Time series data.
        method (str): Imputation method. Options include:
                      "ffill", "bfill", "mean", "linear", or "spline".

    Returns:
        pd.Series: Series with missing values imputed.
    """
    if method == "ffill":
        return series.fillna(method="ffill")
    elif method == "bfill":
        return series.fillna(method="bfill")
    elif method == "mean":
        return series.fillna(series.mean())
    elif method == "linear":
        return series.interpolate(method="linear")
    elif method == "spline":
        return series.interpolate(method="spline", order=2)
    else:
        raise ValueError(f"Imputation method '{method}' not recognized.")


def detrend_series(series, method="difference", order=1):
    """
    Detrend a time series using the specified method.

    Args:
        series (pd.Series): Original time series.
        method (str): Detrending method. Options include "difference" (default).
        order (int): The order of differencing.

    Returns:
        pd.Series: Detrended series.
    """
    if method == "difference":
        return series.diff(order).dropna()
    else:
        raise ValueError(f"Detrending method '{method}' is not implemented.")


def decompose_series(series, model="additive", period=None):
    """
    Decompose a time series into trend, seasonal, and residual components.

    Args:
        series (pd.Series): Time series data.
        model (str): "additive" or "multiplicative".
        period (int): The period of the seasonal component. If None, will try to infer.

    Returns:
        DecomposeResult: Object with attributes trend, seasonal, and resid.
    """
    decomposition = seasonal_decompose(
        series, model=model, period=period, extrapolate_trend="freq"
    )
    return decomposition


def generate_lag_features(series, lags=[1, 2, 3]):
    """
    Generate lag features for a time series.

    Args:
        series (pd.Series): Original time series.
        lags (list of int): List of lag values to create.

    Returns:
        pd.DataFrame: A DataFrame where each column is a lagged version of the series.
    """
    df = pd.DataFrame({"original": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df.dropna()


def generate_rolling_features(series, windows=[3, 7, 14]):
    """
    Generate rolling window features (mean and std) for a time series.

    Args:
        series (pd.Series): Original time series.
        windows (list of int): Window sizes to compute statistics.

    Returns:
        pd.DataFrame: DataFrame with rolling means and standard deviations.
    """
    df = pd.DataFrame({"original": series})
    for window in windows:
        df[f"rolling_mean_{window}"] = series.rolling(window=window).mean()
        df[f"rolling_std_{window}"] = series.rolling(window=window).std()
    return df.dropna()


class TimeSeriesScaler:
    """
    A wrapper class for scaling time series data.
    This class supports Standard, MinMax, and Robust scaling.
    """

    def __init__(self, scaler_type="standard"):
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")

    def fit(self, data):
        """
        Fit the scaler on the data.

        Args:
            data (np.ndarray): Data of shape (n_samples, features).
        """
        self.scaler.fit(data)

    def transform(self, data):
        """
        Scale the data.

        Args:
            data (np.ndarray): Data of shape (n_samples, features).

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.transform(data)

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Args:
            data (np.ndarray): Data of shape (n_samples, features).

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        """
        Reverse the scaling.

        Args:
            data (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Data in original scale.
        """
        return self.scaler.inverse_transform(data)


def regime_normalize(df, op_cols, sensor_cols, stats=None):
    """Per-regime z-normalization for multi-condition industrial data.

    Sensors on equipment that operates under several distinct regimes
    (e.g. flight conditions) jump with the regime; normalizing each sensor
    within its regime removes the jumps so degradation trends become
    comparable across conditions. Regimes are identified by rounding the
    operating-setting columns and grouping identical settings.

    Compute the statistics on TRAINING units only, then pass them back via
    ``stats`` to transform validation/test data without leakage.

    Args:
        df (pd.DataFrame): Data containing ``op_cols`` and ``sensor_cols``.
        op_cols (list of str): Operating-setting columns that define the
            regime.
        sensor_cols (list of str): Sensor columns to normalize per regime.
        stats (dict, optional): Mapping ``regime_key -> (means, stds)``
            previously returned by this function. When given, those
            statistics are applied instead of being recomputed.

    Returns:
        Tuple ``(normalized_df, stats)`` — a copy of ``df`` with
        ``sensor_cols`` normalized, and the per-regime statistics for reuse
        on held-out data.
    """
    df = df.copy()
    regime_key = df[op_cols].round(2).astype(str).agg("|".join, axis=1)
    if stats is None:
        stats = {}
        for key, idx in df.groupby(regime_key).groups.items():
            block = df.loc[idx, sensor_cols]
            means = block.mean()
            stds = block.std().replace(0, 1.0).fillna(1.0)
            stats[key] = (means, stds)
    for key, idx in df.groupby(regime_key).groups.items():
        if key not in stats:
            raise ValueError(
                f"Regime {key!r} was not present in the training statistics; "
                f"recompute stats on data covering all regimes."
            )
        means, stds = stats[key]
        df.loc[idx, sensor_cols] = (df.loc[idx, sensor_cols] - means) / stds
    return df, stats
