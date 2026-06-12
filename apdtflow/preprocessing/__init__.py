"""
Preprocessing utilities for APDTFlow.

Handles data transformation, feature engineering, scaling, and encoding.
"""

from .categorical_encoder import CategoricalEncoder, create_time_features
from .timeseries import (
    TimeSeriesScaler,
    regime_normalize,
    convert_to_datetime,
    decompose_series,
    detrend_series,
    fill_time_gaps,
    generate_lag_features,
    generate_rolling_features,
    impute_missing_values,
)

__all__ = [
    "CategoricalEncoder",
    "create_time_features",
    "TimeSeriesScaler",
    "regime_normalize",
    "convert_to_datetime",
    "decompose_series",
    "detrend_series",
    "fill_time_gaps",
    "generate_lag_features",
    "generate_rolling_features",
    "impute_missing_values",
]
