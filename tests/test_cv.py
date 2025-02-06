import pytest
from apdtflow.cv_factory import TimeSeriesCVFactory
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """A simple dummy dataset returning sequential numbers."""
    def __init__(self, length=100):
        self.data = list(range(length))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def test_cv_factory_rolling():
    dataset = DummyDataset(length=100)
    cv_factory = TimeSeriesCVFactory(
        dataset=dataset,
        method="rolling",
        train_size=40,
        val_size=10,
        step_size=10
    )
    splits = cv_factory.get_splits(max_splits=3)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert train_idx[0] >= 0 and train_idx[-1] < len(dataset)
        assert val_idx[0] == train_idx[-1] + 1

def test_cv_factory_invalid_method():
    dataset = DummyDataset(length=50)
    with pytest.raises(ValueError):
        factory = TimeSeriesCVFactory(
            dataset=dataset,
            method="unsupported_method",
            train_size=30,
            val_size=5,
            step_size=5
        )
        factory.get_splits()

