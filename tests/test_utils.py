import torch
import tempfile
from apdtflow.utils import save_checkpoint, load_checkpoint
from apdtflow.models.apdtflow import APDTFlow


def test_checkpoint_save_load():
    model = APDTFlow(
        num_scales=2,
        input_channels=1,
        filter_size=3,
        hidden_dim=8,
        output_dim=1,
        forecast_horizon=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        checkpoint_path = tmp.name
    save_checkpoint(model, optimizer, epoch=1, filename=checkpoint_path)
    new_model = APDTFlow(
        num_scales=2,
        input_channels=1,
        filter_size=3,
        hidden_dim=8,
        output_dim=1,
        forecast_horizon=3,
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    epoch_loaded = load_checkpoint(
        new_model, new_optimizer, checkpoint_path, torch.device("cpu")
    )
    assert epoch_loaded == 1
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
