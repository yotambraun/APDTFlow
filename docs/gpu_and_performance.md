# GPU and performance

APDTFlow runs on CPU, NVIDIA CUDA and Apple MPS. This page covers how to pick a
device, what runs where, and, with measured numbers, **when a GPU actually helps**.
Short version: **training runs 4 to 7 times faster on a GPU** than on a laptop CPU,
because each training step is replayed as a CUDA graph (on by default); scoring a
single window is still faster on the CPU.

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

## Why the GPU used to lose, and what CUDA graphs change

An APDTFlow forward pass integrates a latent ODE step by step. With the default
`ode_method="rk4"` that is one RK4 step per input time point, four evaluations of a
small MLP per step, then the decoder (for `decoder_type="continuous"`, another ODE at
step size 0.25), and backward doubles it. Each evaluation is a handful of small
matrix operations on a `(batch, hidden_dim)` tensor.

Profiling one default-size training step on the GPU below found **20,650 kernel
launches**, with the GPU busy for **40 ms of a 320 ms step**. The step took the same
time at batch 8 and batch 2,048: the time went to launching kernels from Python, not
to arithmetic, so the GPU lost to the CPU (whose per-operation overhead is about
2 µs here, against about 10 µs per GPU launch under WSL2). The torchdiffeq
maintainers describe the same effect for small problems
([torchdiffeq #15](https://github.com/rtqichen/torchdiffeq/issues/15)).

**CUDA graphs** remove that overhead without changing the arithmetic. On CUDA, `fit()`
records one whole optimizer step (forward, backward and Adam) as a graph after three
warm-up steps, then replays it for every full batch with a single launch, following
the whole-iteration recipe of the
[PyTorch CUDA-graphs guide](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
The last, partial batch of an epoch runs eagerly on the same parameters and optimizer
state. Tests check that graph and eager training produce the same losses and
parameters step for step. Graphs are used when the device is CUDA, `ode_method` is
`"rk4"` and there are no exogenous inputs; anything else trains eagerly, and
`cuda_graphs=False` turns them off (for debugging).

0.5.0 also removed the avoidable overheads that graphs cannot: no GPU-host
synchronisation per solver step, all scales integrated in one batch, the convolution
dilation cached, and one loss read per epoch instead of one per batch.

Practical rules, from the measurements below:

- **Train on the GPU.** It was 4 to 7 times faster than this CPU at every size except
  a 66-window dataset, where the two were even.
- **Serve single windows on the CPU** (`model.to("cpu")` after training). Inference is
  not graph-captured, so one-window calls are launch-bound on a GPU: 3 to 5 times
  slower than on the CPU here. Batch scoring of many assets (`predict_when_fleet`) is
  where a GPU still helps.
- On CPU, `apdtflow.set_seed(0)` (deterministic mode) also limits PyTorch to **one
  thread**, for bit-for-bit reproducibility. Use `set_seed(0, deterministic=False)`
  when you want speed. Results differ between CPU and GPU for the same seed (they draw
  different random numbers), so compare runs on one device.

## Measured: CPU vs CUDA

**Hardware.** Laptop: Intel Core i7-12700H (20 logical CPUs; PyTorch used 10 threads),
NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, driver 546.30), Windows 11 (build
26200) with WSL2 (Ubuntu 22.04, kernel 5.15). Python 3.12.12, PyTorch 2.14.1+cu126, apdtflow 0.5.0. Measured
2026-11-02. WSL2 adds GPU launch overhead compared with native Linux, so on a native
Linux box or a desktop/server GPU the crossover comes sooner. Treat these numbers as
one data point, not a promise.

**Method.** `python benchmarks/benchmark_device.py --configs small medium battery
large-batch --devices cpu cuda cuda-eager --epochs 6 --repeats 3 --calls 30`. `cuda`
is the default (CUDA graphs); `cuda-eager` is the same GPU with `cuda_graphs=False`,
i.e. how 0.5.0 trained before graph capture. Each configuration was fitted 3 times per
device (seeds 0-2, `set_seed(seed, deterministic=False)`) for 6 epochs. The epoch
figures are the median and min-max over the 15 epochs after each fit's first (warm-up,
and for graphs capture) epoch. Inference timings are 30 calls per fit after 3 warm-up
calls (fleet: 6 calls after 1), synchronised with the device. Default settings
otherwise (`use_conformal=True`, calibrated on the training windows). Raw results:
[`benchmarks/results/device_benchmark.json`](../benchmarks/results/device_benchmark.json).

| Config | What | Training windows per epoch |
|---|---|---|
| small | daily min temperature (3,647 pts), hidden_dim 16, history 30, horizon 7, batch 32, transformer decoder (the library defaults, except the horizon) | 3,611 |
| medium | same data, hidden_dim 64, history 60, horizon 14, batch 64, continuous decoder | 3,574 |
| battery | NASA battery B0005 capacity (168 cycles), hidden_dim 16, history 20, horizon 30, batch 32, continuous decoder | 119 |
| large-batch | same temperature data, hidden_dim 128, history 120, horizon 28, batch 256, continuous decoder | 3,500 |

### Training: seconds per epoch (median, min-max)

| Config | CPU | CUDA (graphs, default) | CUDA eager | GPU vs CPU | Graphs vs eager | Peak GPU memory |
|---|---|---|---|---|---|---|
| small | 4.60 (4.37–5.31) | **1.04** (1.03–1.07) | 9.09 (7.54–11.0) | 4.4x faster | 8.8x | 89 MB |
| medium | 22.4 (22.1–22.9) | **4.44** (4.31–4.55) | 33.3 (29.9–37.6) | 5.0x faster | 7.5x | 237 MB |
| battery | 0.81 (0.75–0.90) | **0.73** (0.63–1.36) | 2.42 (2.27–2.57) | even | 3.3x | 197 MB |
| large-batch | 41.7 (40.2–46.4) | **6.10** (5.57–6.57) | 19.4 (18.0–22.7) | 6.8x faster | 3.2x | 1,723 MB |

### Inference: milliseconds per call (median, min-max)

| Config | Call | CPU | CUDA |
|---|---|---|---|
| small | `predict()` (one window) | **6.8** (5.5–11.4) | 27.9 (23.8–38.1) |
| medium | `predict()` | **23.2** (18.0–32.2) | 104 (93–149) |
| medium | `predict_at`, 100 timestamps | **26.6** (17.9–38.4) | 99 (75–123) |
| medium | `predict_when`, calibration cached | **24.5** (17.4–39.6) | 99 (76–139) |
| medium | `predict_when`, first call for a threshold | **22.0** (19.3–23.4) | 100 (95–107) |
| medium | `predict_when_fleet`, 256 assets | 138 (131–183) | **106** (92–139) |
| battery | `predict_when_fleet`, 256 assets | **66** (55–79) | 102 (77–121) |
| large-batch | `predict()` | **47** (37–65) | 231 (194–317) |
| large-batch | `predict_when_fleet`, 256 assets | 391 (369–434) | **258** (225–475) |

**Reading.** With graphs the GPU trains 4.4 to 6.8 times faster than the CPU, and 3 to
9 times faster than eager GPU training, which used to lose to the CPU at default sizes.
The gain from graphs is largest for small models, where launches dominated. On the
66-window battery data there are only four batches per epoch, so CPU and GPU are
even. Inference was not changed: single-window calls stay 3 to 5 times faster on the
CPU, and scoring 256 assets at once is faster on the GPU for the medium and large
models. A larger GPU or native Linux (no WSL2 launch overhead) should favour the GPU
further; not measured here.

### 0.5.0 vs 0.4.0 on the same work

Same machine, same script, run against the 0.4.0 code (`main` at 9829c41, whose
only package change since the 0.4.0 release is a type annotation) and against 0.5.0, both with `--calibration-split 0` so they train on identical
windows (0.4.0 always trained on all of them): `python benchmarks/benchmark_device.py
--configs small medium --devices cpu cuda --epochs 4 --repeats 2 --calls 10
--calibration-split 0`. Median over 6 post-warm-up epochs (min-max). The CUDA rows are
eager training (measured before CUDA-graph capture). Raw results:
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
unmeasured. PyTorch has no graph capture for MPS, so MPS trains eagerly, and the
launch-overhead reasoning above suggests the CPU may win for default-size models
there. Reports from Mac users (run `pytest tests/test_device.py`
and `python benchmarks/benchmark_device.py`) are welcome.

## Reproduce

```bash
pip install -e ".[dev]"
python benchmarks/benchmark_device.py                     # cpu, cuda, cuda-eager / mps if present
python benchmarks/benchmark_device.py --configs small --epochs 3 --repeats 1   # quick look
pytest tests/test_device.py                               # device tests (cuda/mps skip if absent)
```

The script records the hardware and versions it ran on in
`benchmarks/results/device_benchmark.json`. Please quote them with any number.
