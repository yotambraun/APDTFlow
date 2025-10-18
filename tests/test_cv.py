import pytest
from apdtflow.cv_factory import TimeSeriesCVFactory
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.data = list(range(length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_rolling_window_splits():
    dataset = DummyDataset(length=100)
    factory = TimeSeriesCVFactory(
        dataset, method="rolling", train_size=40, val_size=10, step_size=10
    )
    splits = factory.get_splits(max_splits=3)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert train_idx[0] >= 0
        assert val_idx[0] == train_idx[-1] + 1


def test_expanding_window_splits():
    dataset = DummyDataset(length=50)
    factory = TimeSeriesCVFactory(
        dataset, method="expanding", train_size=20, val_size=5, step_size=5
    )
    splits = factory.get_splits(max_splits=2)
    assert len(splits) == 2


def test_blocked_splits():
    dataset = DummyDataset(length=50)
    factory = TimeSeriesCVFactory(
        dataset, method="blocked", train_size=30, val_size=10, step_size=5
    )
    splits = factory.get_splits()
    assert len(splits) == 1


def test_invalid_method():
    dataset = DummyDataset(length=50)
    with pytest.raises(ValueError):
        TimeSeriesCVFactory(
            dataset, method="invalid", train_size=30, val_size=5, step_size=5
        )
