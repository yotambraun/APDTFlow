import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List


class TimeSeriesWindowDataset(Dataset):
    """
    Enhanced dataset for time series with optional exogenous variables.

    Creates samples using a sliding window approach. Supports both univariate
    and multivariate exogenous features to improve forecasting accuracy.

    Args:
        csv_file: Path to CSV file
        date_col: Name of date column
        value_col: Name of target value column
        T_in: Number of historical time steps (input sequence length)
        T_out: Number of future time steps (forecast horizon)
        exog_cols: Optional list of exogenous variable column names
        future_exog_cols: Optional subset of exog_cols that are known in future
        transform: Optional transformation to apply

    Example:
        >>> # Without exogenous variables
        >>> dataset = TimeSeriesWindowDataset('data.csv', 'date', 'sales', T_in=30, T_out=7)
        >>>
        >>> # With exogenous variables
        >>> dataset = TimeSeriesWindowDataset(
        ...     'data.csv', 'date', 'sales', T_in=30, T_out=7,
        ...     exog_cols=['temperature', 'is_holiday', 'promotion'],
        ...     future_exog_cols=['is_holiday', 'promotion']
        ... )
    """

    def __init__(
        self,
        csv_file: str,
        date_col: str,
        value_col: str,
        T_in: int,
        T_out: int,
        exog_cols: Optional[List[str]] = None,
        future_exog_cols: Optional[List[str]] = None,
        transform=None
    ):
        self.df = pd.read_csv(csv_file)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df.sort_values(date_col, inplace=True)

        # Target series
        self.series = self.df[value_col].values.astype(np.float32)

        # Exogenous variables (NEW)
        self.exog_cols = exog_cols or []
        self.future_exog_cols = future_exog_cols or []
        self.has_exog = len(self.exog_cols) > 0

        self.exog_data = None
        if self.has_exog:
            self.exog_data = self.df[exog_cols].values.astype(np.float32)

        self.T_in = T_in
        self.T_out = T_out
        self.transform = transform
        self.samples = []

        # Create samples with exog
        total_length = T_in + T_out
        for i in range(len(self.series) - total_length + 1):
            sample = {
                'target': self.series[i:i+total_length],
                'idx': i
            }
            if self.has_exog:
                sample['exog'] = self.exog_data[i:i+total_length]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Target: input (x) and output (y)
        target_full = sample['target']
        x = torch.tensor(target_full[:self.T_in], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(target_full[self.T_in:], dtype=torch.float32).unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        # Exogenous variables (if available)
        if self.has_exog:
            exog_full = sample['exog']  # (T_in + T_out, num_exog)
            exog_x = torch.tensor(exog_full[:self.T_in], dtype=torch.float32).T  # (num_exog, T_in)
            exog_y = torch.tensor(exog_full[self.T_in:], dtype=torch.float32).T  # (num_exog, T_out)
            return x, y, exog_x, exog_y
        else:
            return x, y
