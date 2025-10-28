from .metric_factory import MetricFactory


class RegressionEvaluator:
    def __init__(self, metrics=["MSE", "MAE", "RMSE", "MAPE"]):
        self.metrics = MetricFactory.get_metrics(metrics)

    def evaluate(self, preds, targets):
        results = {}
        for name, func in self.metrics.items():
            results[name] = func(preds, targets)
        return results
