# APDTFlow Documentation

**Forecast at any moment in time.** APDTFlow models time as continuous — one trained
model answers `predict()`, `predict_at(any timestamp)`, and `predict_when(threshold)`
with calibrated uncertainty.

APDTFlow is a predictive-maintenance **timing engine** built on a general forecasting
core. The core is a Neural-ODE forecaster with multi-scale decomposition and conformal
uncertainty; on top of it, the continuous-time decoder answers the question operations
teams actually ask — *when will this series cross the line?* — with an uncertainty
window calibrated on the time axis itself, and an `act_by` date to schedule against.
We are not aware of another forecasting library offering calibrated event-time
forecasting (`predict_when`) and arbitrary-timestamp queries (`predict_at`) from one
trained model (as of June 2026).

Every published number is reproducible from a committed script; the full methodology,
including what we tested and chose **not** to ship, is in
[METHODOLOGY.md](METHODOLOGY.md). Measured results: [experiment_results.md](experiment_results.md).

---

## Contents

- [Installation](#installation)
- [Getting started: three questions, one model](#getting-started-three-questions-one-model)
- [The operational rule](#the-operational-rule)
- [Conformal uncertainty](#conformal-uncertainty)
- [Multivariate input and the fleet API](#multivariate-input-and-the-fleet-api)
- [Exogenous and categorical features](#exogenous-and-categorical-features)
- [Backtesting](#backtesting)
- [Command-line interface](#command-line-interface)
- [Saving, loading, and TorchScript export](#saving-loading-and-torchscript-export)
- [Drift monitoring with score_recent](#drift-monitoring-with-score_recent)
- [Determinism](#determinism)
- [Further reading](#further-reading)

---

## Installation

```bash
pip install apdtflow
```

From source:

```bash
git clone https://github.com/yotambraun/APDTFlow.git
cd APDTFlow
pip install -e .
```

Requires Python 3.10+ and PyTorch (installed automatically, with `torchdiffeq` for the
ODE solvers). Optional extras: `apdtflow[serve]` (FastAPI serving),
`apdtflow[mlflow]`, `apdtflow[wandb]`, `apdtflow[dev]`.

## Getting started: three questions, one model

```python
import pandas as pd
from apdtflow import APDTFlowForecaster, set_seed

set_seed(0)
df = pd.read_csv(
    "dataset_examples/daily-minimum-temperatures-in-me_clean.csv",
    parse_dates=["Date"],
)

model = APDTFlowForecaster(
    forecast_horizon=14,
    history_length=60,
    num_epochs=20,
    decoder_type="continuous",   # enables predict_at / predict_when
    use_conformal=True,          # calibrated uncertainty (required for predict_when)
)
model.fit(df, target_col="Daily minimum temperatures", date_col="Date")

# 1. WHAT will the value be? — the familiar grid forecast
forecast = model.predict()                                   # 14 values

# 2. WHAT at an arbitrary moment? — fractional steps, beyond the horizon
last = df["Date"].iloc[-1]
values, lower, upper = model.predict_at([
    last + pd.Timedelta(hours=36),    # one and a half days out
    last + pd.Timedelta(days=16),     # beyond the trained horizon
])

# 3. WHEN will it cross a threshold? — calibrated time window
result = model.predict_when(threshold=20.0, direction="above", alpha=0.1)
print(result.eta)        # point estimate of the crossing time
print(result.earliest, result.latest)  # 90% time window
print(result.act_by)     # schedule by THIS (the window's early edge)
print(result.censored)   # True = no crossing within the horizon (a valid answer)
```

`predict_at` and `predict_when` require `decoder_type='continuous'`; `predict_when`
additionally requires `use_conformal=True`. Times are timestamps when the model was
fitted with a `date_col`, otherwise float step offsets.

Runnable versions: [Quickstart.ipynb](../Quickstart.ipynb) and the
[examples directory](../examples/README.md).

## The operational rule

> **Schedule by `act_by`, never by `eta`.**

Across our largest audit the point estimate showed a systematic *late* bias on
smoothed degradation indicators; the asymmetric time-space calibration absorbs it,
which is why the API returns `act_by` (the window's earliest edge) as a first-class
field. `alpha` is the sensitivity knob: a smaller `alpha` widens the window and moves
`act_by` earlier — fewer missed events, more false alarms. Details and measured
coverage: [METHODOLOGY.md, Section 5](METHODOLOGY.md#5-event-time-forecasting-predict_when)
and [experiment_results.md](experiment_results.md).

`predict_when` has two modes: `mode='expected'` (first crossing of the mean
trajectory, calibrated window — for forecastable trends such as degradation) and
`mode='risk'` (the earliest time the conformal band touches the threshold — "from
when is the event plausible?", for noisy series and safety thresholds).

## Conformal uncertainty

Fit with `use_conformal=True` to calibrate distribution-free prediction intervals
(split conformal by default, `conformal_method='adaptive'` for non-stationary data):

```python
lower, preds, upper = model.predict(return_intervals="conformal", alpha=0.05)
```

A consequence of the default MSE training objective that we want users to understand:
the model's raw variance head is untrained under MSE, so every uncertainty band
APDTFlow publishes — value intervals and `predict_when` time windows alike — comes
from conformal calibration, never from raw variance outputs (unless you explicitly
train with `loss_type='nll'`). `predict_when` windows are calibrated directly on
**crossing-time errors** in time space; value-space bands mapped onto the time axis
miscalibrate badly, which we measured
([METHODOLOGY.md, Section 4](METHODOLOGY.md#4-uncertainty-conformal-prediction-in-value-space-and-in-time-space)).

## Multivariate input and the fleet API

Feed multiple sensor channels with `feature_cols`; a learned health-indicator fusion
layer combines them in front of the pipeline and exposes interpretable weights:

```python
model.fit(df, target_col="capacity", feature_cols=["voltage", "temperature"])
model.sensor_importance_          # pd.Series, one weight per channel
```

For a whole fleet, one call turns recent per-asset histories into a maintenance
schedule sorted by `act_by` (censored assets last), ready for CMMS ingestion:

```python
schedule = model.predict_when_fleet(
    {"engine_07": series_07, "engine_12": series_12},
    threshold=1.4, direction="below", alpha=0.1,
)
schedule.to_csv("maintenance_schedule.csv")
```

For equipment operating under several regimes (e.g. flight conditions), normalize
sensors per regime first — computing statistics on training units only:

```python
from apdtflow.preprocessing import regime_normalize
train_norm, stats = regime_normalize(train_df, op_cols, sensor_cols)
test_norm, _ = regime_normalize(test_df, op_cols, sensor_cols, stats=stats)
```

Architecture details: [models.md](models.md).

## Exogenous and categorical features

Numerical covariates and categorical variables enter through a separate fusion path
(`'gated'` by default; `'concat'` and `'attention'` available):

```python
model.fit(
    df, target_col="sales", date_col="date",
    exog_cols=["temperature", "promotion"],
    future_exog_cols=["promotion"],        # known in the future
    categorical_cols=["day_of_week"],      # one-hot or embedding encoding
)
preds = model.predict(exog_future=future_df)
```

`feature_cols` (multivariate input) and `exog_cols`/`categorical_cols` are mutually
exclusive. See `examples/exogenous_variables_demo.py` and
`examples/categorical_features_demo.py`.

## Backtesting

`historical_forecasts()` simulates production forecasting over a rolling origin and
reports MASE/sMAPE alongside raw per-step errors:

```python
results = model.historical_forecasts(
    df, target_col="sales", start=0.8, stride=7, retrain=False,
)
print(results["abs_error"].mean())
```

Set `stride=forecast_horizon` for non-overlapping folds, `retrain=True` for the
slower but more realistic expanding-window protocol. See
`examples/backtesting_demo.py`.

## Command-line interface

The `apdtflow` console command trains and evaluates the core model on a CSV:

```bash
apdtflow train --csv_file data.csv --date_col DATE --value_col VALUE \
    --T_in 12 --T_out 3 --num_epochs 15 --checkpoint_path model.pt \
    --ode_method rk4 --decoder_type transformer

apdtflow infer --csv_file data.csv --date_col DATE --value_col VALUE \
    --T_in 12 --T_out 3 --checkpoint_path model.pt
```

`--ode_method` accepts `rk4` (default) or `dopri5_adjoint`; `--decoder_type` accepts
`transformer` (default) or `continuous`.

## Saving, loading, and TorchScript export

```python
model.save("forecaster.pkl")
model = APDTFlowForecaster.load("forecaster.pkl")
```

Checkpoints embed the library version. Checkpoints from apdtflow ≤ 0.3.x are refused
with an explanatory error: those versions contained a defect that made predictions
independent of the input data, and such models cannot be repaired — retrain on
≥ 0.4.0 ([METHODOLOGY.md, Section 8](METHODOLOGY.md#8-engineering-history)).
Saved state includes the conformal calibration data, so `predict_when` works
immediately after `load`.

For Python-free serving of grid forecasts:

```python
model.export_torchscript("forecaster.pt")
```

The traced module maps a normalized window `(batch, channels, history_length)` to
normalized grid forecasts. Tracing fixes the window length, the integer forecast
grid, and the batch size; `predict_at`/`predict_when` and conformal calibration are
not part of the export — serve those through the Python API
(`examples/serve_api.py`). ONNX export of the ODE graph is not currently supported.

## Drift monitoring with score_recent

```python
report = model.score_recent(recent_df, alpha=0.05)
# {'mae': ..., 'rmse': ..., 'baseline_mae': ..., 'mae_ratio': ...,
#  'coverage': ..., 'expected_coverage': 0.95, 'n_windows': ...}
```

`score_recent` slides the fitted model over recent observations and compares rolling
MAE and conformal coverage against the calibration baseline. Coverage decaying below
`1 - alpha` (or `mae_ratio` drifting well above 1) signals the model needs
refitting/recalibration on data that includes the recent regime — wire it into your
monitoring before trusting any long-running deployment.

## Determinism

```python
from apdtflow import set_seed
set_seed(0)
```

Seeds Python, NumPy, and PyTorch; with `deterministic=True` (the default) PyTorch is
additionally switched to deterministic algorithms and a single CPU thread, trading
speed for bit-for-bit reproducibility — the configuration used for the published
benchmark numbers.

## Further reading

- [METHODOLOGY.md](METHODOLOGY.md) — how everything works, references, evaluation
  protocol, negative results, engineering history (canonical).
- [models.md](models.md) — architecture details and the parameter table.
- [experiment_results.md](experiment_results.md) — measured results.
- [examples/](../examples/README.md) — runnable demos, ordered by what's new.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — dev setup, the shipping rule for domain
  demos, and the benchmark roadmap.

APDTFlow is MIT-licensed ([LICENSE](../LICENSE)).
