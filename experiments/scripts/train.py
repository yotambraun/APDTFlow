#!/usr/bin/env python
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import torch
from torch.utils.data import DataLoader
from apdtflow.data import TimeSeriesWindowDataset
from apdtflow.models.apdtflow import APDTFlow
from apdtflow.logger_util import get_logger
from torch.utils.tensorboard import SummaryWriter


def main(args):
    logger = get_logger(log_file="logs/training.log")
    logger.info("Starting training...")
    writer = SummaryWriter(log_dir="runs/experiment1") if args.tensorboard else None
    dataset = TimeSeriesWindowDataset(
        csv_file=args.csv_file,
        date_col=args.date_col,
        value_col=args.value_col,
        T_in=args.T_in,
        T_out=args.T_out,
        transform=None,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = APDTFlow(
        num_scales=args.num_scales,
        input_channels=1,
        filter_size=args.filter_size,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        forecast_horizon=args.T_out,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size, _, T_in = x_batch.size()
            t_span = torch.linspace(0, 1, steps=T_in, device=device)

            optimizer.zero_grad()
            preds, pred_logvars = model(x_batch, t_span)
            mse = (preds - y_batch.transpose(1, 2)) ** 2
            loss = torch.mean(
                0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size

        avg_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Training complete and checkpoint saved at " + checkpoint_path)
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a forecasting model with APDTFlow framework."
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file with data."
    )
    parser.add_argument(
        "--date_col", type=str, default="DATE", help="Name of the date column."
    )
    parser.add_argument(
        "--value_col", type=str, required=True, help="Name of the value column."
    )
    parser.add_argument(
        "--T_in", type=int, default=12, help="Length of input sequence."
    )
    parser.add_argument(
        "--T_out",
        type=int,
        default=3,
        help="Forecast horizon (number of future time steps).",
    )
    parser.add_argument(
        "--model", type=str, default="APDTFlow", help="Forecasting model to use."
    )
    parser.add_argument(
        "--num_scales", type=int, default=3, help="Number of scales in APDTFlow."
    )
    parser.add_argument(
        "--filter_size",
        type=int,
        default=5,
        help="Filter size for dynamic dilation convolution.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=16, help="Hidden dimension size."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=15, help="Number of training epochs."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging."
    )
    args = parser.parse_args()

    main(args)
