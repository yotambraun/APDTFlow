import torch
import pytest
from apdtflow.evaluation.metric_factory import MetricFactory
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator


def test_metric_factory_valid():
    mse = MetricFactory.get_metric("MSE")
    mae = MetricFactory.get_metric("MAE")
    rmse = MetricFactory.get_metric("RMSE")
    mape = MetricFactory.get_metric("MAPE")
    assert callable(mse) and callable(mae) and callable(rmse) and callable(mape)


def test_metric_factory_invalid():
    with pytest.raises(ValueError):
        MetricFactory.get_metric("INVALID_METRIC")


def test_regression_evaluator():
    evaluator = RegressionEvaluator(metrics=["MSE", "MAE", "RMSE", "MAPE"])
    preds = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    targets = torch.tensor([[1.0, 3.0], [5.0, 5.0]])
    results = evaluator.evaluate(preds, targets)
    assert (
        "MSE" in results
        and "MAE" in results
        and "RMSE" in results
        and "MAPE" in results
    )
    import math

    assert abs(results["RMSE"] - math.sqrt(results["MSE"])) < 1e-3
