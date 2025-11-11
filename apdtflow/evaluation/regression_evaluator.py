from .metric_factory import MetricFactory


class RegressionEvaluator:
    """
    Regression evaluator with industry-standard forecasting metrics.

    Supports metrics: MSE, MAE, RMSE, MAPE, MASE, sMAPE, CRPS, Coverage

    New in v0.2.3:
    - MASE: Industry-standard scale-independent metric
    - sMAPE: Symmetric MAPE, better than MAPE
    - CRPS: For probabilistic forecasts
    - Coverage: For prediction intervals

    Parameters
    ----------
    metrics : list of str
        Metric names to compute

    Examples
    --------
    >>> evaluator = RegressionEvaluator(metrics=["MSE", "MASE", "sMAPE"])
    >>> results = evaluator.evaluate(preds, targets)
    """

    def __init__(self, metrics=["MSE", "MAE", "RMSE", "MAPE", "MASE", "sMAPE"]):
        self.metrics = MetricFactory.get_metrics(metrics)

    def evaluate(self, preds, targets, lower=None, upper=None):
        """
        Evaluate predictions against targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values
        targets : torch.Tensor
            Actual values
        lower : torch.Tensor, optional
            Lower bounds for interval metrics (CRPS, Coverage)
        upper : torch.Tensor, optional
            Upper bounds for interval metrics (CRPS, Coverage)

        Returns
        -------
        results : dict
            Dictionary mapping metric names to values
        """
        results = {}
        for name, func in self.metrics.items():
            # Handle metrics that need intervals
            if name in ["CRPS", "Coverage"]:
                if lower is not None and upper is not None:
                    results[name] = func(preds, targets, lower, upper)
                # Skip if intervals not provided
            else:
                results[name] = func(preds, targets)
        return results
