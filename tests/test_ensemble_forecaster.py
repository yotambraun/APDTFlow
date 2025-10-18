import torch
from apdtflow.models.ensemble_forecaster import EnsembleForecaster
from apdtflow.models.base_forecaster import BaseForecaster


class DummyForecaster(BaseForecaster):
    def __init__(self):
        super(DummyForecaster, self).__init__()

    def forward(self, x, t_span=None):
        return x[:, :, :3].transpose(1, 2), None

    def train_model(self, *args, **kwargs):
        pass

    def predict(self, new_x, forecast_horizon, device):
        return self(new_x)[0], None

    def evaluate(self, test_loader, device):
        return 0.0, 0.0


def test_ensemble_prediction():
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 10)
    model1 = DummyForecaster()
    model2 = DummyForecaster()
    ensemble = EnsembleForecaster([model1, model2], weights=[0.3, 0.7])
    preds, _ = ensemble.predict(
        dummy_input, forecast_horizon=3, device=torch.device("cpu")
    )
    assert preds.shape == (batch_size, 3, 1)
