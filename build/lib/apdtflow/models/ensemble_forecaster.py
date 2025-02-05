import torch
from .base_forecaster import BaseForecaster

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
        ensemble_preds = torch.sum(predictions * torch.tensor(self.weights).view(-1, 1, 1, 1).to(predictions.device), dim=0)
        return ensemble_preds, None

    def evaluate(self, test_loader, device):
        mse_total, mae_total, count = 0.0, 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                ensemble_preds, _ = self.predict(x_batch, None, device)
                y_true = y_batch.squeeze(1)
                mse = ((ensemble_preds.squeeze(-1) - y_true) ** 2).mean().item()
                mae = (torch.abs(ensemble_preds.squeeze(-1) - y_true)).mean().item()
                batch_size = x_batch.size(0)
                mse_total += mse * batch_size
                mae_total += mae * batch_size
                count += batch_size
        mse_avg = mse_total / count
        mae_avg = mae_total / count
        print(f"Ensemble Evaluation -> MSE: {mse_avg:.4f}, MAE: {mae_avg:.4f}")
        return mse_avg, mae_avg
