import torch


def mse(preds, targets):
    return torch.mean((preds - targets) ** 2).item()


def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()


def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def mape(preds, targets):
    if torch.sum(targets) == 0:
        return 0
    else:
        if torch.sum(targets < 0) > 0:
            raise ValueError("MAPE: targets must be non-negative")
        else:
            return torch.mean(torch.abs((preds - targets) / targets)).item()


def mase(preds, targets, seasonal_period=1):
    """
    Mean Absolute Scaled Error (MASE).

    Industry-standard metric for forecasting. Scale-independent and
    suitable for intermittent demand. Values < 1 indicate better
    performance than naive seasonal forecast.

    Args:
        preds: Predictions tensor
        targets: Actual values tensor
        seasonal_period: Seasonality period for scaling (default=1 for naive)

    Returns:
        MASE score (lower is better, < 1 is good)

    Reference: Hyndman & Koehler (2006)
    """
    # Mean absolute error of predictions
    mae_pred = torch.mean(torch.abs(preds - targets))

    # Mean absolute error of naive forecast (shifted by seasonal_period)
    # For non-seasonal: naive forecast is just previous value
    if len(targets) <= seasonal_period:
        # Fallback to MAE if not enough data
        return mae_pred.item()

    naive_errors = targets[seasonal_period:] - targets[:-seasonal_period]
    mae_naive = torch.mean(torch.abs(naive_errors))

    # Avoid division by zero
    if mae_naive == 0:
        return 0.0 if mae_pred == 0 else float('inf')

    return (mae_pred / mae_naive).item()


def smape(preds, targets):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Better than MAPE as it's symmetric and bounded (0-200%).
    Commonly used in M-competitions and production systems.

    Args:
        preds: Predictions tensor
        targets: Actual values tensor

    Returns:
        sMAPE percentage (0-200%, lower is better)

    Reference: Makridakis (1993)
    """
    numerator = torch.abs(preds - targets)
    denominator = (torch.abs(preds) + torch.abs(targets)) / 2

    # Avoid division by zero
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0

    smape_values = numerator[mask] / denominator[mask]
    return (torch.mean(smape_values) * 100).item()


def crps(preds, targets, lower=None, upper=None):
    """
    Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.

    Measures quality of probabilistic predictions. For point forecasts
    with prediction intervals, this simplifies to a weighted combination
    of MAE and interval width.

    Args:
        preds: Point predictions tensor
        targets: Actual values tensor
        lower: Lower bound of prediction interval (optional)
        upper: Upper bound of prediction interval (optional)

    Returns:
        CRPS score (lower is better)

    Reference: Gneiting & Raftery (2007)
    """
    if lower is None or upper is None:
        # Simplified CRPS for point forecasts (reduces to MAE)
        return torch.mean(torch.abs(preds - targets)).item()

    # Convert to tensors if needed
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=preds.dtype, device=preds.device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=preds.dtype, device=preds.device)

    # CRPS for interval forecasts
    # CRPS ≈ (upper - lower) + 2 * max(lower - target, 0) + 2 * max(target - upper, 0)
    interval_width = upper - lower
    lower_penalty = torch.maximum(lower - targets, torch.zeros_like(targets))
    upper_penalty = torch.maximum(targets - upper, torch.zeros_like(targets))

    crps_values = interval_width + 2 * lower_penalty + 2 * upper_penalty
    return torch.mean(crps_values).item()


def coverage(preds, targets, lower, upper):
    """
    Coverage metric for prediction intervals.

    Measures the proportion of actual values that fall within
    the prediction interval. For 95% intervals, coverage should
    be close to 95%.

    Args:
        preds: Point predictions tensor (not used, but kept for API consistency)
        targets: Actual values tensor
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval

    Returns:
        Coverage percentage (0-100%)
    """
    # Convert to tensors if needed
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=targets.dtype, device=targets.device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=targets.dtype, device=targets.device)

    # Check if targets are within intervals
    within_interval = (targets >= lower) & (targets <= upper)
    coverage_rate = torch.mean(within_interval.float())

    return (coverage_rate * 100).item()


class MetricFactory:
    metrics_map = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MASE": mase,
        "sMAPE": smape,
        "CRPS": crps,
        "Coverage": coverage
    }

    @staticmethod
    def get_metric(name):
        if name in MetricFactory.metrics_map:
            return MetricFactory.metrics_map[name]
        else:
            raise ValueError(f"Metric {name} is not supported.")

    @staticmethod
    def get_metrics(names):
        return {n: MetricFactory.get_metric(n) for n in names}
