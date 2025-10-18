import pytest
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from apdtflow.data import TimeSeriesWindowDataset
from apdtflow.models.apdtflow import APDTFlow


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if (
                model.__class__.__name__ == "APDTFlow"
                and x_batch.dim() == 4
                and x_batch.size(1) == 1
            ):
                x_batch = x_batch.squeeze(1)
            batch_size = x_batch.size(0)
            T_in_current = x_batch.size(-1)
            t_span = torch.linspace(0, 1, steps=T_in_current, device=device)
            preds, pred_logvars = model(x_batch, t_span)
            mse = (preds - y_batch.transpose(1, 2)) ** 2
            loss = torch.mean(
                0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
            )
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def train_on_split(model, train_loader, val_loader, num_epochs, learning_rate, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if (
                model.__class__.__name__ == "APDTFlow"
                and x_batch.dim() == 4
                and x_batch.size(1) == 1
            ):
                x_batch = x_batch.squeeze(1)
            batch_size = x_batch.size(0)
            T_in_current = x_batch.size(-1)
            t_span = torch.linspace(0, 1, steps=T_in_current, device=device)
            optimizer.zero_grad()
            preds, pred_logvars = model(x_batch, t_span)
            mse = (preds - y_batch.transpose(1, 2)) ** 2
            loss = torch.mean(
                0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size
    return evaluate_model(model, val_loader, device)


@pytest.fixture
def dummy_csv(tmp_path):
    dates = pd.date_range("2020-01-01", periods=100)
    values = np.linspace(0, 10, num=100)
    df = pd.DataFrame({"DATE": dates, "value": values})
    csv_file = tmp_path / "dummy.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def dataset(dummy_csv):
    ds = TimeSeriesWindowDataset(
        csv_file=dummy_csv,
        date_col="DATE",
        value_col="value",
        T_in=10,
        T_out=2,
        transform=None,
    )
    return ds


@pytest.fixture
def dataloaders(dataset):
    train_size = int(0.8 * len(dataset))
    train_subset = Subset(dataset, list(range(train_size)))
    val_subset = Subset(dataset, list(range(train_size, len(dataset))))
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_training_loss_decreases(dataloaders):
    device = torch.device("cpu")
    train_loader, val_loader = dataloaders
    model = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=2,
    )
    model.to(device)
    initial_loss = evaluate_model(model, val_loader, device)
    final_loss = train_on_split(
        model,
        train_loader,
        val_loader,
        num_epochs=5,
        learning_rate=0.001,
        device=device,
    )
    assert (
        final_loss < initial_loss
    ), f"Expected final loss {final_loss} to be lower than initial loss {initial_loss}"


def test_no_nan_in_parameters(dataloaders):
    device = torch.device("cpu")
    train_loader, val_loader = dataloaders
    model = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=2,
    )
    model.to(device)
    train_on_split(
        model,
        train_loader,
        val_loader,
        num_epochs=5,
        learning_rate=0.001,
        device=device,
    )
    for param in model.parameters():
        assert not torch.isnan(
            param
        ).any(), "Found NaN in model parameters after training"


def test_checkpoint_save_and_load(tmp_path, dataloaders):
    device = torch.device("cpu")
    train_loader, val_loader = dataloaders
    model = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=2,
    )
    model.to(device)
    train_on_split(
        model,
        train_loader,
        val_loader,
        num_epochs=1,
        learning_rate=0.001,
        device=device,
    )
    checkpoint_path = tmp_path / "model_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_new = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=2,
    )
    model_new.to(device)
    model_new.load_state_dict(torch.load(checkpoint_path, map_location=device))
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        assert torch.allclose(
            p1, p2
        ), "Model parameters do not match after loading checkpoint"
