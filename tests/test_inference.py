import torch
from apdtflow.inference import infer_forecaster
from apdtflow.models.base_forecaster import BaseForecaster


class DummyForecaster(BaseForecaster):
    def __init__(self):
        super(DummyForecaster, self).__init__()

    def forward(self, x, t_span=None):
        batch_size = x.size(0)
        return torch.zeros(batch_size, 3, 1), torch.zeros(batch_size, 3, 1)

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, new_x, forecast_horizon, device):
        return self(new_x)[0], None

    def evaluate(self, test_loader, device):
        return 0.0, 0.0


def test_infer_forecaster():
    batch_size = 3
    dummy_input = torch.randn(batch_size, 1, 10)
    model = DummyForecaster()
    preds, _ = infer_forecaster(
        model, dummy_input, forecast_horizon=3, device=torch.device("cpu")
    )
    assert preds.shape == (batch_size, 3, 1)
