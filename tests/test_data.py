import pandas as pd
import numpy as np
from apdtflow.data import TimeSeriesWindowDataset


def create_dummy_csv(tmp_path, n=30):
    df = pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-01", periods=n),
            "value": np.arange(n, dtype=np.float32),
        }
    )
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_dataset_length(tmp_path):
    csv_file = create_dummy_csv(tmp_path, n=30)
    T_in, T_out = 5, 2
    dataset = TimeSeriesWindowDataset(
        csv_file, date_col="DATE", value_col="value", T_in=T_in, T_out=T_out
    )
    expected_length = 30 - (T_in + T_out) + 1
    assert len(dataset) == expected_length


def test_dataset_item_shape(tmp_path):
    csv_file = create_dummy_csv(tmp_path, n=20)
    T_in, T_out = 4, 3
    dataset = TimeSeriesWindowDataset(
        csv_file, date_col="DATE", value_col="value", T_in=T_in, T_out=T_out
    )
    x, y = dataset[0]
    assert x.dim() == 2
    assert x.shape[1] == T_in
    assert y.dim() == 2
    assert y.shape[1] == T_out
