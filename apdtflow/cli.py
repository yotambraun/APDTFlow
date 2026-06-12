import argparse

import torch
from torch.utils.data import DataLoader

from apdtflow.data import TimeSeriesWindowDataset


def _build_model(args, device):
    from apdtflow.models.apdtflow import APDTFlow

    model = APDTFlow(
        num_scales=3,
        input_channels=1,
        filter_size=5,
        hidden_dim=16,
        output_dim=1,
        forecast_horizon=args.T_out,
        use_embedding=args.use_embedding,
        history_length=args.T_in,
        ode_method=args.ode_method,
        decoder_type=args.decoder_type,
    )
    model.to(device)
    return model


def _add_common_args(subparser):
    subparser.add_argument("--csv_file", required=True, help="Path to the CSV file.")
    subparser.add_argument("--date_col", default="DATE")
    subparser.add_argument("--value_col", required=True)
    subparser.add_argument("--T_in", type=int, default=12)
    subparser.add_argument("--T_out", type=int, default=3)
    subparser.add_argument("--batch_size", type=int, default=16)
    subparser.add_argument(
        "--no_embedding",
        dest="use_embedding",
        action="store_false",
        help="Disable the learnable time series embedding (default: enabled)",
    )
    subparser.add_argument(
        "--ode_method",
        choices=["rk4", "dopri5_adjoint"],
        default="rk4",
        help="ODE solver: fixed-step rk4 (default, fast) or adaptive dopri5 with adjoint.",
    )
    subparser.add_argument(
        "--decoder_type",
        choices=["transformer", "continuous"],
        default="transformer",
        help="Decoder: transformer (default) or continuous-time ODE decoder.",
    )
    subparser.set_defaults(use_embedding=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "APDTFlow: a modular forecasting framework for "
            "continuous-time time series forecasting."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a forecasting model.")
    _add_common_args(train_parser)
    train_parser.add_argument("--num_epochs", type=int, default=15)
    train_parser.add_argument("--learning_rate", type=float, default=0.001)
    train_parser.add_argument("--model", default="APDTFlow")
    train_parser.add_argument(
        "--loss_type",
        choices=["mse", "nll"],
        default="mse",
        help="Training loss: mse (default) or Gaussian nll.",
    )
    train_parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Where to save the trained model state dict.",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference with a checkpoint.")
    _add_common_args(infer_parser)
    infer_parser.add_argument("--checkpoint_path", required=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        dataset = TimeSeriesWindowDataset(
            args.csv_file, args.date_col, args.value_col, args.T_in, args.T_out
        )
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = _build_model(args, device)
        model.train_model(
            train_loader,
            args.num_epochs,
            args.learning_rate,
            device,
            loss_type=args.loss_type,
        )
        if args.checkpoint_path:
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    elif args.command == "infer":
        dataset = TimeSeriesWindowDataset(
            args.csv_file, args.date_col, args.value_col, args.T_in, args.T_out
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model = _build_model(args, device)
        state = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state)
        metrics = model.evaluate(loader, device, metrics=["MSE", "MAE"])
        print(", ".join(f"{name}: {value:.4f}" for name, value in metrics.items()))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
