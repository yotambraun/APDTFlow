"""CPU vs GPU benchmark for APDTFlow training and inference.

Measures, for a few representative configurations and every requested
device:

* training time per epoch (from ``fit(log_callback=...)`` timestamps; the
  first epoch is reported separately as warm-up and excluded from the
  steady-state median),
* inference latency of ``predict()``, ``predict_at()`` (100 timestamps),
  ``predict_when()`` (calibration cached), the one-off cost of the first
  ``predict_when()`` call for a new threshold (crossing-time calibration over
  the calibration windows), and ``predict_when_fleet()`` for a 256-asset
  fleet,
* peak GPU memory allocated by PyTorch (CUDA only).

``cuda`` trains with CUDA graphs (the default since 0.5.0); ``cuda-eager``
is the same device with ``cuda_graphs=False``, to show what graph capture
changes.

Every configuration is fitted ``--repeats`` times with different seeds; the
report gives the median and the min-max spread across all measured samples.

Usage (from the repository root)::

    python benchmarks/benchmark_device.py                       # cpu + cuda/mps if present
    python benchmarks/benchmark_device.py --devices cpu cuda cuda-eager
    python benchmarks/benchmark_device.py --devices cpu cuda --repeats 3
    python benchmarks/benchmark_device.py --configs small --epochs 4  # quick check

Results are written to ``benchmarks/results/device_benchmark.json`` and a
Markdown table is printed. The script only uses the public API, so it also runs
against older releases (e.g. to compare 0.4.0 and 0.5.0 on identical work with
``--calibration-split 0``). Numbers depend heavily on hardware; always
report the hardware next to them (the script records it).
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import apdtflow  # noqa: E402
from apdtflow import APDTFlowForecaster  # noqa: E402


def mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())

DATA = os.path.join(ROOT, "dataset_examples")


def _temperatures() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "daily-minimum-temperatures-in-me_clean.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _battery() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "nasa_B0005.csv"))


# name -> (description, data loader, constructor kwargs, fit kwargs, threshold, direction)
CONFIGS: Dict[str, dict] = {
    "small": dict(
        description="daily min temperature (3,647 pts), hidden_dim=16, "
                    "history=30, horizon=7, batch=32, transformer decoder",
        data=_temperatures,
        model=dict(hidden_dim=16, history_length=30, forecast_horizon=7,
                   batch_size=32, decoder_type="transformer", use_conformal=True),
        fit=dict(target_col="Daily minimum temperatures", date_col="Date"),
        when=None,
    ),
    "medium": dict(
        description="daily min temperature (3,647 pts), hidden_dim=64, "
                    "history=60, horizon=14, batch=64, continuous decoder + conformal",
        data=_temperatures,
        model=dict(hidden_dim=64, history_length=60, forecast_horizon=14,
                   batch_size=64, decoder_type="continuous", use_conformal=True),
        fit=dict(target_col="Daily minimum temperatures", date_col="Date"),
        when=(18.0, "above"),
    ),
    "large-batch": dict(
        description="daily min temperature (3,647 pts), hidden_dim=128, "
                    "history=120, horizon=28, batch=256, continuous decoder + conformal",
        data=_temperatures,
        model=dict(hidden_dim=128, history_length=120, forecast_horizon=28,
                   batch_size=256, decoder_type="continuous", use_conformal=True),
        fit=dict(target_col="Daily minimum temperatures", date_col="Date"),
        when=(18.0, "above"),
    ),
    "battery": dict(
        description="NASA battery B0005 capacity (168 cycles), hidden_dim=16, "
                    "history=20, horizon=30, batch=32, continuous decoder + conformal",
        data=_battery,
        model=dict(hidden_dim=16, history_length=20, forecast_horizon=30,
                   batch_size=32, decoder_type="continuous", use_conformal=True),
        fit=dict(target_col="capacity"),
        when=(1.45, "below"),
    ),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_call(fn: Callable, device: torch.device, n: int, warmup: int = 3) -> List[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summary(samples: List[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def run_config(name: str, device_name: str, epochs: int, repeats: int,
               n_calls: int, calibration_split: Optional[float] = None) -> dict:
    cfg = CONFIGS[name]
    df = cfg["data"]()
    extra: dict = {}
    if device_name == "cuda-eager":
        # Only passed for this mode, so the script still runs on releases
        # that predate the cuda_graphs parameter.
        extra["cuda_graphs"] = False
    torch_device = "cuda" if device_name == "cuda-eager" else device_name
    device = torch.device(torch_device)
    epoch_times: List[float] = []
    first_epoch: List[float] = []
    lat: Dict[str, List[float]] = {}
    peak_mb: List[float] = []

    for rep in range(repeats):
        apdtflow.set_seed(rep, deterministic=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        kwargs = dict(cfg["model"])
        if calibration_split is not None:
            kwargs["calibration_split"] = calibration_split
        model = APDTFlowForecaster(num_epochs=epochs, device=torch_device,
                                   verbose=False, **kwargs, **extra)
        stamps: List[float] = []

        def _cb(epoch, metrics, _stamps=stamps):
            _stamps.append(time.perf_counter())

        start = time.perf_counter()
        model.fit(df, log_callback=_cb, **cfg["fit"])
        bounds = [start] + stamps
        durations = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
        first_epoch.append(durations[0])
        epoch_times.extend(durations[1:])

        lat.setdefault("predict", []).extend(
            _time_call(lambda: model.predict(), device, n_calls))
        if cfg["when"] is not None:
            threshold, direction = cfg["when"]
            lat.setdefault("predict_at_100", []).extend(_time_call(
                lambda: model.predict_at(np.linspace(0.5, model.forecast_horizon, 100)),
                device, n_calls))
            # First call for a new threshold runs the crossing-time calibration.
            _sync(device)
            t0 = time.perf_counter()
            model.predict_when(threshold, direction=direction, alpha=0.1)
            _sync(device)
            lat.setdefault("predict_when_first_call", []).append(
                (time.perf_counter() - t0) * 1000.0)
            lat.setdefault("predict_when_cached", []).extend(_time_call(
                lambda: model.predict_when(threshold, direction=direction, alpha=0.1),
                device, n_calls))
            series = df[cfg["fit"]["target_col"]].to_numpy(dtype=float)
            rng = np.random.default_rng(rep)
            hl = model.history_length
            fleet = {}
            for i in range(256):
                end = int(rng.integers(hl, len(series) + 1))
                fleet[f"asset_{i:03d}"] = series[end - hl:end]
            lat.setdefault("predict_when_fleet_256", []).extend(_time_call(
                lambda: model.predict_when_fleet(fleet, threshold, direction=direction,
                                                 alpha=0.1),
                device, max(3, n_calls // 5), warmup=1))
        if device.type == "cuda":
            peak_mb.append(torch.cuda.max_memory_allocated(device) / 2**20)

    result = {
        "config": name,
        "description": cfg["description"],
        "device": device_name,
        "epochs_per_fit": epochs,
        "repeats": repeats,
        "epoch_seconds": _summary(epoch_times),
        "first_epoch_seconds": _summary(first_epoch),
        "latency_ms": {k: _summary(v) for k, v in lat.items()},
    }
    if peak_mb:
        result["peak_gpu_memory_mb"] = max(peak_mb)
    return result


def hardware() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "apdtflow": apdtflow.__version__,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = props.total_memory // 2**20
    return info


def _fmt(s: dict, unit: str = "") -> str:
    return f"{s['median']:.3g}{unit} ({s['min']:.3g}-{s['max']:.3g})"


def markdown(results: List[dict]) -> str:
    lines = [
        "| Config | Device | Epoch (s) | predict() (ms) | predict_at x100 (ms) "
        "| predict_when, cached (ms) | predict_when, first call (ms) "
        "| fleet x256 (ms) | Peak GPU mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lt = r["latency_ms"]

        def cell(key):
            return _fmt(lt[key]) if key in lt else "n/a"

        mem = f"{r['peak_gpu_memory_mb']:.0f}" if "peak_gpu_memory_mb" in r else "n/a"
        lines.append(
            f"| {r['config']} | {r['device']} | {_fmt(r['epoch_seconds'])} "
            f"| {cell('predict')} | {cell('predict_at_100')} "
            f"| {cell('predict_when_cached')} | {cell('predict_when_first_call')} "
            f"| {cell('predict_when_fleet_256')} | {mem} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_devices = ["cpu"]
    if torch.cuda.is_available():
        default_devices += ["cuda", "cuda-eager"]
    if mps_is_available():
        default_devices.append("mps")
    parser.add_argument("--devices", nargs="+", default=default_devices)
    parser.add_argument("--configs", nargs="+", default=["small", "medium", "battery"],
                        choices=sorted(CONFIGS))
    parser.add_argument("--epochs", type=int, default=6,
                        help="epochs per fit (the first is reported as warm-up)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="independent fits per config and device")
    parser.add_argument("--calls", type=int, default=30,
                        help="timed calls per inference method and fit")
    parser.add_argument("--calibration-split", type=float, default=None,
                        help="override calibration_split (0.0 = train on every window, "
                             "the only behaviour apdtflow <= 0.4 had; use it to compare "
                             "versions on identical work)")
    parser.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results",
                                                      "device_benchmark.json"))
    args = parser.parse_args()

    results = []
    for name in args.configs:
        for dev in args.devices:
            print(f"[{name} / {dev}] ...", flush=True)
            results.append(run_config(name, dev, args.epochs, args.repeats, args.calls,
                                      args.calibration_split))
            print(f"[{name} / {dev}] epoch median "
                  f"{results[-1]['epoch_seconds']['median']:.3f}s", flush=True)

    recorded_args = dict(vars(args), out=os.path.relpath(args.out, ROOT))
    payload = {"hardware": hardware(), "args": recorded_args, "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["hardware"], indent=2))
    print(markdown(results))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
