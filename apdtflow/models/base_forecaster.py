import abc
import torch.nn as nn

class BaseForecaster(nn.Module, abc.ABC):
    @abc.abstractmethod
    def train_model(self, train_loader, num_epochs, learning_rate, device):
        """Train the forecasting model."""
        pass

    @abc.abstractmethod
    def predict(self, new_x, forecast_horizon, device):
        """Predict future values given an input sequence."""
        pass

    @abc.abstractmethod
    def evaluate(self, test_loader, device):
        """Evaluate the model on test data."""
        pass
