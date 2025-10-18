import torch
from apdtflow.models.transformer_forecaster import TransformerForecaster


def test_transformer_forecaster_forward():
    batch_size = 4
    T_in = 15
    input_dim = 1
    model_dim = 16
    num_layers = 1
    nhead = 4
    forecast_horizon = 3
    model = TransformerForecaster(
        input_dim, model_dim, num_layers, nhead, forecast_horizon
    )
    x = torch.randn(batch_size, T_in, input_dim)
    preds = model(x)
    assert preds.shape == (batch_size, forecast_horizon, input_dim)


def test_transformer_forecaster_predict():
    batch_size = 3
    T_in = 15
    input_dim = 1
    model_dim = 16
    num_layers = 1
    nhead = 4
    forecast_horizon = 3
    model = TransformerForecaster(
        input_dim, model_dim, num_layers, nhead, forecast_horizon
    )
    preds, _ = model.predict(
        torch.randn(batch_size, 1, T_in), forecast_horizon, device=torch.device("cpu")
    )
    assert preds.shape == (batch_size, forecast_horizon, input_dim)
