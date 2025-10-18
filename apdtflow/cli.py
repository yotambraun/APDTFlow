import argparse
from apdtflow.training import train_forecaster
from apdtflow.inference import infer_forecaster
from apdtflow.data import TimeSeriesWindowDataset
from torch.utils.data import DataLoader
import torch
import sys

def print_banner():
    banner = r"""
────────────────────────────────────────────────
     🚀 Welcome to APDTFlow! 🚀

    Your go-to framework for flexible,
    modular, and powerful time series forecasting.
    
    Built for pros. Designed for performance. ⚡
    
    Let's get forecasting! 📈
────────────────────────────────────────────────
"""
    if not any(arg in sys.argv for arg in ["--help", "-h"]):
        print(banner)

def main():
    print_banner()
    parser = argparse.ArgumentParser(
        description=(
            "APDTFlow: A flexible forecasting framework for "
            "modular and powerful time series forecasting."
        )
    )
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train a forecasting model.")
    train_parser.add_argument("--csv_file", required=True, help="Path to the CSV file.")
    train_parser.add_argument("--date_col", default="DATE")
    train_parser.add_argument("--value_col", required=True)
    train_parser.add_argument("--T_in", type=int, default=12)
    train_parser.add_argument("--T_out", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--num_epochs", type=int, default=15)
    train_parser.add_argument("--learning_rate", type=float, default=0.001)
    train_parser.add_argument("--model", default="APDTFlow")
    train_parser.add_argument("--no_embedding", dest="use_embedding", action="store_false",
                              help="Disable the learnable time series embedding (default: enabled)")
    train_parser.set_defaults(use_embedding=True)
    infer_parser = subparsers.add_parser("infer", help="Run inference with a checkpoint.")
    infer_parser.add_argument("--csv_file", required=True)
    infer_parser.add_argument("--date_col", default="DATE")
    infer_parser.add_argument("--value_col", required=True)
    infer_parser.add_argument("--T_in", type=int, default=12)
    infer_parser.add_argument("--T_out", type=int, default=3)
    infer_parser.add_argument("--batch_size", type=int, default=16)
    infer_parser.add_argument("--checkpoint_path", required=True)
    infer_parser.add_argument("--no_embedding", dest="use_embedding", action="store_false",
                              help="Disable the learnable time series embedding (default: enabled)")
    infer_parser.set_defaults(use_embedding=True)

    args = parser.parse_args()
    if args.command == "train":
        dataset = TimeSeriesWindowDataset(
            args.csv_file, args.date_col, args.value_col, args.T_in, args.T_out
        )
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        from apdtflow.models.apdtflow import APDTFlow

        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=args.T_out,
            use_embedding=args.use_embedding
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train_forecaster(
            model, train_loader, args.num_epochs, args.learning_rate, device
        )
    elif args.command == "infer":
        dataset = TimeSeriesWindowDataset(
            args.csv_file, args.date_col, args.value_col, args.T_in, args.T_out
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from apdtflow.models.apdtflow import APDTFlow

        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=args.T_out,
            use_embedding=args.use_embedding
        )
        model.to(device)
        infer_forecaster(args.checkpoint_path, loader, device)
    else:
        parser.print_help()


  
if __name__ == "__main__":
    main()
