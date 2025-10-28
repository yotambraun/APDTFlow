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

logger = get_logger("Inference", log_file="logs/inference.log")


def main(args):
    logger.info("Starting inference...")
    test_dataset = TimeSeriesWindowDataset(
        csv_file=args.csv_file,
        date_col=args.date_col,
        value_col=args.value_col,
        T_in=args.T_in,
        T_out=args.T_out,
        transform=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
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
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    logger.info("Loading checkpoint from " + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    mse, mae = model.evaluate(test_loader, device)
    logger.info(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with the forecasting model."
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
    parser.add_argument("--T_out", type=int, default=3, help="Forecast horizon.")
    parser.add_argument(
        "--model", type=str, default="APDTFlow", help="Forecasting model to use."
    )
    parser.add_argument(
        "--num_scales", type=int, default=3, help="Number of scales (for APDTFlow)."
    )
    parser.add_argument(
        "--filter_size", type=int, default=5, help="Filter size for convolution."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=16, help="Hidden dimension size."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()

    main(args)
