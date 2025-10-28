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


class MetricFactory:
    metrics_map = {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    @staticmethod
    def get_metric(name):
        if name in MetricFactory.metrics_map:
            return MetricFactory.metrics_map[name]
        else:
            raise ValueError(f"Metric {name} is not supported.")

    @staticmethod
    def get_metrics(names):
        return {n: MetricFactory.get_metric(n) for n in names}
