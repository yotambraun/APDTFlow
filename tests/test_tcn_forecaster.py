import torch
from apdtflow.models.tcn_forecaster import TCNForecaster


def test_tcn_forecaster_forward():
    batch_size = 3
    T_in = 20
    input_channels = 1
    num_channels = [8, 8]
    kernel_size = 3
    forecast_horizon = 5
    model = TCNForecaster(input_channels, num_channels, kernel_size, forecast_horizon)
    x = torch.randn(batch_size, 1, T_in)
    preds = model(x)
    assert preds.shape == (batch_size, forecast_horizon, input_channels)


def test_tcn_forecaster_predict():
    batch_size = 2
    T_in = 20
    input_channels = 1
    num_channels = [8, 8]
    kernel_size = 3
    forecast_horizon = 5
    model = TCNForecaster(input_channels, num_channels, kernel_size, forecast_horizon)
    preds, _ = model.predict(
        torch.randn(batch_size, 1, T_in), forecast_horizon, device=torch.device("cpu")
    )
    assert preds.shape == (batch_size, forecast_horizon, input_channels)
