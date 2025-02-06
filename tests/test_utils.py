import torch
from apdtflow.utils import save_checkpoint, load_checkpoint
from apdtflow.models.apdtflow import APDTFlow

def test_save_and_load_checkpoint(tmp_path):
    device = torch.device("cpu")
    model = APDTFlow(num_scales=2, input_channels=1, filter_size=3, hidden_dim=8, output_dim=1, forecast_horizon=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, epoch=1, filename=str(checkpoint_path))
    new_model = APDTFlow(num_scales=2, input_channels=1, filter_size=3, hidden_dim=8, output_dim=1, forecast_horizon=3)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    epoch_loaded = load_checkpoint(new_model, new_optimizer, str(checkpoint_path), device)
    assert epoch_loaded == 1
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
