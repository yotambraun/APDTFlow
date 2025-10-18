from abc import ABC, abstractmethod
import torch.nn as nn


class BaseForecaster(nn.Module, ABC):
    @abstractmethod
    def train_model(self, train_loader, num_epochs, learning_rate, device):
        pass

    @abstractmethod
    def predict(self, new_x, forecast_horizon, device):
        pass

    @abstractmethod
    def evaluate(self, test_loader, device):
        pass
