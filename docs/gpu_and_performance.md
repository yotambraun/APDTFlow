# GPU and performance

APDTFlow runs on CPU, NVIDIA CUDA and Apple MPS. This page covers how to pick a
device, what runs where, and, with measured numbers, **when a GPU actually helps**.
Short version: the default-size model trains faster on a CPU; a GPU pays off once
the model and the batches get bigger.

## Choosing a device

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(device="auto")    # default: CUDA, then Apple MPS, then CPU
model = APDTFlowForecaster(device="cuda")    # or "cuda:1", "mps", "cpu", a torch.device
model.fit(df, target_col="y")
model.to("cpu")                              # move a fitted model, e.g. to serve on CPU

model.save("m.pkl")                          # weights are stored on CPU ...
m = APDTFlowForecaster.load("m.pkl", device="auto")   # ... and load on any device
```

```bash
apdtflow train --csv_file data.csv --value_col y --device cuda --checkpoint_path m.pt
```

Requesting a backend that is not available (`device="cuda"` with a CPU-only PyTorch,
`device="mps"` off Apple silicon) raises a `ValueError` that says how to fix it.
`apdtflow.resolve_device("auto")` returns the device `"auto"` would pick.

**Installing GPU PyTorch.** `pip install apdtflow` pulls PyTorch from PyPI. On Linux
that wheel already includes CUDA. On Windows it is CPU-only, so install the CUDA build
from [pytorch.org](https://pytorch.org/get-started/locally/) first (e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`; pick the
index matching your driver, shown by `nvidia-smi`). On macOS the standard wheel
includes MPS.

## What runs on the device

Everything: training (including early-stopping validation), conformal calibration,
`predict`, `predict_at`, `predict_when` (including the crossing-time calibration),
`predict_when_fleet`, `score_recent`, `historical_forecasts` (including
`retrain=True` folds), the Transformer and TCN models, and TorchScript export. The
only host transfers are inputs going in and results coming out. Loaded checkpoints
are mapped to CPU first and then moved, so a checkpoint from a GPU machine works on a
CPU-only one; this also covers checkpoints written by 0.4.x.

On CUDA, training batches come from pinned host memory and are copied
asynchronously.

## Why a small model is faster on CPU

An APDTFlow forward pass integrates a latent ODE step by step. With the default
`ode_method="rk4"` that is one RK4 step per input time point, four evaluations of a
small MLP per step, then the decoder (for `decoder_type="continuous"`, another ODE at
step size 0.25). Each evaluation is a handful of small matrix operations, so a
forward/backward pass is thousands of tiny GPU kernels. Until a batch is large enough
to fill the GPU, the time goes to launching kernels, not computing. A modern CPU runs
the same small operations with less overhead.

0.5.0 removes the avoidable overheads: no GPU-host synchronisation per solver step,
all scales integrated in one batch, the convolution dilation cached, and one loss
read per epoch instead of one per batch. What remains is inherent to stepping an ODE
in small increments.

Practical rules, from the measurements below:

- **Default-size models (hidden_dim 16, batch 32): use the CPU.**
- **On a GPU, raise `batch_size`.** The per-step time stays almost flat as the batch
  grows until the GPU saturates, so epochs get proportionally shorter. The gain is
  largest with bigger models (`hidden_dim` 64 to 128, longer `history_length`).
- **Serving one window at a time is latency-bound: use the CPU.** Batch scoring
  (`predict_when_fleet` over many assets, `score_recent`, backtests) is where a GPU
  can win.
- On CPU, `apdtflow.set_seed(0)` (deterministic mode) also limits PyTorch to **one
  thread**, for bit-for-bit reproducibility. Use `set_seed(0, deterministic=False)`
  when you want speed.

## Measured: CPU vs CUDA

**Hardware.** Laptop: Intel Core i7-12700H (20 logical CPUs; PyTorch used 10 threads),
NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, driver 546.30), Windows 11 (build
26200) with WSL2 (Ubuntu 22.04, kernel 5.15). Python 3.12.12, PyTorch 2.14.1+cu126, apdtflow 0.5.0. Measured
2026-11-02. WSL2 adds GPU launch overhead compared with native Linux, so on a native
Linux box or a desktop/server GPU the crossover comes sooner. Treat these numbers as
one data point, not a promise.

**Method.** `python benchmarks/benchmark_device.py --configs small medium battery
large-batch --devices cpu cuda --epochs 6 --repeats 3 --calls 30`. Each configuration
was fitted 3 times per device (seeds 0-2, `set_seed(seed, deterministic=False)`) for
6 epochs. The epoch figures are the median and min-max over the 15 epochs after each
fit's first (warm-up) epoch. Inference timings are 30 calls per fit after 3 warm-up
calls (fleet: 6 calls after 1), synchronised with the device. Default settings
otherwise (`use_conformal=True`, so 20% of windows are held out for calibration).
Raw results: [`benchmarks/results/device_benchmark.json`](../benchmarks/results/device_benchmark.json).

| Config | What | Training windows per epoch |
|---|---|---|
| small | daily min temperature (3,647 pts), hidden_dim 16, history 30, horizon 7, batch 32, transformer decoder (the library defaults, except the horizon) | 2,883 |
| medium | same data, hidden_dim 64, history 60, horizon 14, batch 64, continuous decoder | 2,846 |
| battery | NASA battery B0005 capacity (168 cycles), hidden_dim 16, history 20, horizon 30, batch 32, continuous decoder | 66 |
| large-batch | same temperature data, hidden_dim 128, history 120, horizon 28, batch 256, continuous decoder | 2,773 |

### Training: seconds per epoch (median, min-max)

| Config | CPU | CUDA | Faster | Peak GPU memory |
|---|---|---|---|---|
| small | **3.59** (3.32–3.76) | 7.52 (7.10–8.03) | CPU, 2.1x | 28 MB |
| medium | **19.3** (18.8–21.7) | 27.6 (24.8–30.4) | CPU, 1.4x | 129 MB |
| battery | **0.65** (0.59–0.71) | 1.75 (1.49–1.96) | CPU, 2.7x | 44 MB |
| large-batch | 35.3 (33.5–40.4) | **15.6** (13.5–16.1) | **CUDA, 2.3x** | 1,501 MB |

### Inference: milliseconds per call (median, min-max)

| Config | Call | CPU | CUDA |
|---|---|---|---|
| small | `predict()` (one window) | **8.3** (5.2–14.4) | 29.2 (24.4–57.7) |
| medium | `predict()` | **26.8** (18.6–37.0) | 100 (78–159) |
| medium | `predict_at`, 100 timestamps | **26.6** (20.2–39.3) | 104 (71–164) |
| medium | `predict_when`, calibration cached | **27.1** (19.3–42.2) | 100 (77–143) |
| medium | `predict_when`, first call for a threshold | **33.1** (23.8–37.8) | 99.6 (96.2–101) |
| medium | `predict_when_fleet`, 256 assets | 152 (139–218) | **106** (97–123) |
| battery | `predict_when_fleet`, 256 assets | **72** (61–83) | 100 (77–120) |
| large-batch | `predict()` | **51** (39–73) | 248 (221–402) |
| large-batch | `predict_when_fleet`, 256 assets | 433 (397–472) | **252** (231–359) |

**Reading.** The default-size models are latency-bound: the CPU trains them 1.4 to
2.7x faster and answers single-window calls 3 to 5x faster. With hidden_dim 128 and
batch 256 the GPU trains 2.3x faster, and scoring 256 assets at once is faster on the
GPU for the medium and large models. We expect a larger GPU or native Linux to
widen the GPU's lead at large sizes (not measured here). For the default sizes the
bottleneck is per-step overhead rather than arithmetic, so we expect little change
there.

### 0.5.0 vs 0.4.0 on the same work

Same machine, same script, run against the 0.4.0 code (`main` at 9829c41, whose
only package change since the 0.4.0 release is a type annotation) and against 0.5.0, both with `--calibration-split 0` so they train on identical
windows (0.4.0 always trained on all of them): `python benchmarks/benchmark_device.py
--configs small medium --devices cpu cuda --epochs 4 --repeats 2 --calls 10
--calibration-split 0`. Median over 6 post-warm-up epochs (min-max). Raw results:
[`compare_v0.4.0.json`](../benchmarks/results/compare_v0.4.0.json),
[`compare_v0.5.0.json`](../benchmarks/results/compare_v0.5.0.json).

| Config | Device | 0.4.0 epoch (s) | 0.5.0 epoch (s) | Speed-up |
|---|---|---|---|---|
| small | CPU | 12.6 (12.3–13.1) | 5.04 (4.92–5.20) | 2.5x |
| small | CUDA | 47.1 (43.5–48.1) | 10.3 (9.9–11.3) | 4.6x |
| medium | CPU | 50.9 (50.2–51.3) | 24.6 (24.3–25.3) | 2.1x |
| medium | CUDA | 109 (106–113) | 34.0 (33.6–36.1) | 3.2x |

Single-window `predict()` went from 28.6 to 9.0 ms (small, CPU), 437 to 103 ms
(medium, CUDA), and `predict_when` (cached) from 73 to 29 ms (medium, CPU). The
sources: torchdiffeq's per-step host synchronisation and Python overhead in the RK4
loop, the per-stage device `argmin` that looked up the encoder's input, three
separate per-scale ODE solves instead of one batched solve, and a device read of the
convolution dilation on every forward pass.

## Mixed precision (AMP): not used, and why

We measured `torch.autocast` on the CUDA forward pass with
`python benchmarks/benchmark_amp.py`: `forward_at` over 715 calibration windows ×
400 query times, for the medium configuration above trained for 5 epochs. Errors
are in °C (the data's unit); crossing times are for a threshold of 18 °C.

| Precision | Time per call | Max value error vs float32 | Crossing-time difference (max / mean) |
|---|---|---|---|
| float32 | 105 ms | 0 | 0 / 0 steps |
| float16 autocast | 136 ms | 0.014 °C | 0 / 0 steps |
| bfloat16 autocast | 130 ms | 0.136 °C | 0.035 / 0.0046 steps |

Autocast was **slower** on this GPU, most likely because the matrices are too small
for tensor cores to pay back the casting overhead, and it adds error that
accumulates over the solver steps. APDTFlow therefore runs in float32 everywhere and
does not offer an AMP switch. Revisiting this for much larger models would need a
measured accuracy gate on the crossing-time calibration before shipping.

## Apple MPS

`device="auto"` selects MPS on Apple silicon when CUDA is absent. MPS has no
float64, so APDTFlow keeps everything in float32: inputs enter as float32 tensors,
model parameters are float32, and with `ode_method="dopri5_adjoint"` APDTFlow asks
torchdiffeq to keep its step-size bookkeeping in float32 (its default is float64).
The default `ode_method="rk4"` path uses no float64 at all.

**Not verified on Apple hardware.** This release was developed and measured on
Linux with an NVIDIA GPU. The MPS variants in `tests/test_device.py` run on any Mac
with MPS, but they were skipped here (and in CI). The operations involved (`Conv1d`,
`BatchNorm1d`, `Linear`, `TransformerDecoder`, `sort`, `unique`, `searchsorted`,
`gather`) are standard, but if your PyTorch version lacks an MPS kernel for one of
them, `PYTORCH_ENABLE_MPS_FALLBACK=1` runs that op on the CPU. Performance on MPS is
unmeasured; the small-kernel reasoning above suggests the CPU will also win for
default-size models there. Reports from Mac users (run `pytest tests/test_device.py`
and `python benchmarks/benchmark_device.py`) are welcome.

## Reproduce

```bash
pip install -e ".[dev]"
python benchmarks/benchmark_device.py                     # cpu + cuda/mps if present
python benchmarks/benchmark_device.py --configs small --epochs 3 --repeats 1   # quick look
pytest tests/test_device.py                               # device tests (cuda/mps skip if absent)
```

The script records the hardware and versions it ran on in
`benchmarks/results/device_benchmark.json`. Please quote them with any number.
