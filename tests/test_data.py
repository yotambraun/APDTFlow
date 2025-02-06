import pytest
import pandas as pd
import numpy as np
from apdtflow.data import TimeSeriesWindowDataset

@pytest.fixture
def dummy_csv(tmp_path):
    data = {
        "DATE": pd.date_range("2020-01-01", periods=30),
        "value": np.arange(30)
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

def test_dataset_length(dummy_csv):
    dataset = TimeSeriesWindowDataset(dummy_csv, date_col="DATE", value_col="value", T_in=5, T_out=2)
    expected_length = 30 - (5 + 2) + 1
    assert len(dataset) == expected_length

def test_dataset_item(dummy_csv):
    dataset = TimeSeriesWindowDataset(dummy_csv, date_col="DATE", value_col="value", T_in=5, T_out=2)
    x, y = dataset[0]
    assert x.dim() == 2 and x.shape[1] == 5
    assert y.dim() == 2 and y.shape[1] == 2
