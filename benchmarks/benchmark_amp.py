"""Is CUDA autocast (float16 / bfloat16) worth it for APDTFlow? Measure.

Fits the "medium" configuration of ``benchmark_device.py`` (hidden_dim=64,
history 60, horizon 14, continuous decoder, conformal) for 5 epochs on CUDA,
then compares ``forward_at`` over all calibration windows x 400 query times in
float32 against float16 and bfloat16 autocast: time per call, value error
(in the data's units) and the difference in first-crossing times of a
threshold. The numbers in docs/gpu_and_performance.md come from this script.

Usage (from the repository root, needs a CUDA GPU)::

    python benchmarks/benchmark_amp.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import apdtflow  # noqa: E402
from apdtflow import APDTFlowForecaster  # noqa: E402

THRESHOLD = 18.0


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("benchmark_amp.py needs a CUDA GPU")
    df = pd.read_csv(os.path.join(ROOT, "dataset_examples",
                                  "daily-minimum-temperatures-in-me_clean.csv"))
    apdtflow.set_seed(0, deterministic=False)
    model = APDTFlowForecaster(hidden_dim=64, history_length=60, forecast_horizon=14,
                               batch_size=64, num_epochs=5, decoder_type="continuous",
                               use_conformal=True, device="cuda", verbose=False)
    model.fit(df, target_col="Daily minimum temperatures")
    net = model.model
    net.eval()
    X = model._calib_X_.to("cuda")
    t_span = torch.linspace(0, 1, steps=model.history_length, device="cuda")
    taus = torch.linspace(0.05, model.forecast_horizon, 400, device="cuda")
    scale, shift = float(model.scaler_std_), float(model.scaler_mean_)

    def run(dtype):
        with torch.no_grad():
            if dtype is None:
                return net.forward_at(X, t_span, taus)[0].float()
            with torch.autocast("cuda", dtype=dtype):
                return net.forward_at(X, t_span, taus)[0].float()

    def time_ms(dtype, n=10):
        run(dtype)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            run(dtype)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / n * 1000

    def first_crossing(values):
        v = values.squeeze(-1).cpu().numpy() * scale + shift
        above = v >= THRESHOLD
        return np.where(above.any(1), above.argmax(1), -1)

    grid = taus.cpu().numpy()
    ref = run(None)
    ref_cross = first_crossing(ref)
    print(f"{X.shape[0]} calibration windows x {len(grid)} query times; "
          f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("| Precision | Time per call | Max value error vs float32 "
          "| Crossing-time difference (max / mean) |")
    print("|---|---|---|---|")
    for name, dtype in [("float32", None), ("float16 autocast", torch.float16),
                        ("bfloat16 autocast", torch.bfloat16)]:
        out = run(dtype)
        err = float(((out - ref).abs() * scale).max())
        cross = first_crossing(out)
        both = (ref_cross >= 0) & (cross >= 0)
        diff = np.abs(grid[cross[both]] - grid[ref_cross[both]]) if both.any() else np.zeros(1)
        print(f"| {name} | {time_ms(dtype):.0f} ms | {err:.3f} "
              f"| {diff.max():.3f} / {diff.mean():.4f} steps |")


if __name__ == "__main__":
    main()
