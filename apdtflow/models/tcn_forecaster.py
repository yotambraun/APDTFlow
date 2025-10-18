import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .base_forecaster import BaseForecaster
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    def __init__(self, input_channels, num_channels, kernel_size, forecast_horizon):
        super(TCNForecaster, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                )
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], input_channels)
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        out = self.network(x)
        last = out[:, :, -1]
        preds = self.fc(last)
        preds = preds.unsqueeze(1).repeat(1, self.forecast_horizon, 1)
        return preds

    def train_model(self, train_loader, num_epochs, learning_rate, device):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device).squeeze(1)
                y_batch = y_batch.to(device).squeeze(1)
                optimizer.zero_grad()
                preds = self(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader.dataset):.4f}"
            )
        return

    def predict(self, new_x, forecast_horizon, device):
        self.eval()
        new_x = new_x.to(device)
        if new_x.dim() == 3:
            if new_x.size(1) != 1:
                new_x = new_x[:, 0:1, :]
        elif new_x.dim() == 2:
            new_x = new_x.unsqueeze(1)
        with torch.no_grad():
            preds = self(new_x)
        return preds, None

    def evaluate(self, test_loader, device, metrics=["MSE", "MAE", "RMSE", "MAPE"]):
        self.eval()
        evaluator = RegressionEvaluator(metrics)
        total_metrics = {m: 0.0 for m in metrics}
        total_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device).squeeze(1)
                y_batch = y_batch.to(device).squeeze(1)
                preds = self(x_batch)
                batch_size = x_batch.size(0)
                batch_results = evaluator.evaluate(preds, y_batch)
                for m in metrics:
                    total_metrics[m] += batch_results[m] * batch_size
                total_samples += batch_size
        avg_metrics = {m: total_metrics[m] / total_samples for m in metrics}
        print(
            "Evaluation -> "
            + ", ".join([f"{m}: {avg_metrics[m]:.4f}" for m in metrics])
        )
        return avg_metrics
