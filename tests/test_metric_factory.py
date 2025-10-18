import pytest
from apdtflow.evaluation.metric_factory import MetricFactory


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        ("MSE", 0.0),
        ("MAE", 0.0),
        ("RMSE", 0.0),
    ],
)
def test_metric_factory(metric_name, expected):
    metric = MetricFactory.get_metric(metric_name)
    import torch

    preds = torch.zeros(2, 3)
    targets = torch.zeros(2, 3)
    assert metric(preds, targets) == expected
