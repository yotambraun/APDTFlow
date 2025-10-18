import torch
from .base_forecaster import BaseForecaster
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator


class EnsembleForecaster(BaseForecaster):
    def __init__(self, models, weights=None):
        """
        Args:
            models (list): List of forecaster instances (subclasses of BaseForecaster)
            weights (list): Optional list of weights for each model.
        """
        super(EnsembleForecaster, self).__init__()
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def train_model(self, train_loader, num_epochs, learning_rate, device):
        for model in self.models:
            model.train_model(train_loader, num_epochs, learning_rate, device)

    def predict(self, new_x, forecast_horizon, device):
        predictions = []
        for model in self.models:
            preds, _ = model.predict(new_x, forecast_horizon, device)
            predictions.append(preds)
        predictions = torch.stack(predictions)
        if predictions.dim() == 4:
            weight_shape = (-1, 1, 1, 1)
        elif predictions.dim() == 3:
            weight_shape = (-1, 1, 1)
        else:
            raise ValueError(
                "Unexpected predictions dimension: {}".format(predictions.dim())
            )

        weights = torch.tensor(self.weights).view(*weight_shape).to(predictions.device)
        ensemble_preds = torch.sum(predictions * weights, dim=0)
        return ensemble_preds, None

    def evaluate(self, test_loader, device, metrics=["MSE", "MAE", "RMSE", "MAPE"]):
        self.eval()
        evaluator = RegressionEvaluator(metrics)
        total_metrics = {m: 0.0 for m in metrics}
        total_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                ensemble_preds, _ = self.predict(x_batch, None, device)
                y_true = y_batch.squeeze(1)
                batch_size = x_batch.size(0)
                batch_results = evaluator.evaluate(ensemble_preds.squeeze(-1), y_true)
                for m in metrics:
                    total_metrics[m] += batch_results[m] * batch_size
                total_samples += batch_size
        avg_metrics = {m: total_metrics[m] / total_samples for m in metrics}
        print(
            "Ensemble Evaluation -> "
            + ", ".join([f"{m}: {avg_metrics[m]:.4f}" for m in metrics])
        )
        return avg_metrics
