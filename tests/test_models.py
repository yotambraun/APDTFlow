import torch
from apdtflow.models.apdtflow import APDTFlow

def test_apdtflow_forward():
    batch_size = 4
    T_in = 30
    dummy_input = torch.randn(batch_size, 1, T_in)
    t_span = torch.linspace(0, 1, steps=T_in)
    model = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=3,
        use_embedding=True
    )
    preds, pred_logvars = model(dummy_input, t_span)
    assert preds.shape == (batch_size, 3, 1)
    assert pred_logvars.shape == (batch_size, 3, 1)