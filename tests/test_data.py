import pytest
from apdtflow.data import TimeSeriesWindowDataset
import pandas as pd

def test_dataset_loading(tmp_path):
    data = {'DATE': pd.date_range(start='2020-01-01', periods=30),
            'value': list(range(30))}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    dataset = TimeSeriesWindowDataset(str(csv_path), date_col='DATE', value_col='value', T_in=5, T_out=2)
    expected_length = 30 - (5 + 2) + 1
    assert len(dataset) == expected_length
