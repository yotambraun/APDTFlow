# Features & capabilities

Everything one trained `APDTFlowForecaster` can do, grouped into four areas, with
what each feature is for and the code to use it. For a deeper API walkthrough see
[the usage guide](index.md); for the live version of this page see the
[Features site](https://yotambraun.github.io/APDTFlow/features/).

---

## 1. Forecasting & uncertainty

One model, trained once, answers the next-values question and two that grid
forecasters cannot — the value at any moment, and when a threshold is crossed —
each with calibrated uncertainty.

**Grid forecast** — the next k values, with optional calibrated intervals.

```python
model = APDTFlowForecaster(forecast_horizon=24)
model.fit(df, target_col='sales')

yhat = model.predict()                                   # next 24 steps
lower, yhat, upper = model.predict(return_intervals='conformal', alpha=0.1)
```

**Forecast at any moment** (`predict_at`) — query arbitrary real-valued times:
fractional steps, between observations, even beyond the trained horizon, because
the decoder integrates a continuous-time ODE. Needs `decoder_type='continuous'`.

```python
model = APDTFlowForecaster(forecast_horizon=14, decoder_type='continuous',
                           use_conformal=True)
model.fit(df, target_col='value', date_col='date')

values, lower, upper = model.predict_at([3.5, 7.2, 18.0])
```

**When will it cross the line?** (`predict_when`) — a point estimate (`eta`), a
calibrated 90% window, and `act_by` (the early edge of the window) to schedule
against. `mode='risk'` returns the earliest plausible crossing for safety-critical
cases. Needs `decoder_type='continuous'` and `use_conformal=True`.

```python
r = model.predict_when(threshold=80, direction='above', alpha=0.1)
r.eta                 # point estimate (steps ahead)
r.earliest, r.latest  # calibrated 90% window
r.act_by              # schedule by this edge
r.censored            # True if no crossing within the horizon
```

**Schedule a whole fleet** (`predict_when_fleet`) — run `predict_when` across many
assets and get a table sorted by `act_by`, ready for a maintenance/CMMS system.

```python
schedule = model.predict_when_fleet(assets, threshold=1.4, direction='below')
schedule.to_csv('maintenance_schedule.csv')
```

**Calibrated uncertainty** — split and adaptive conformal calibration wraps any
forecast in an interval whose coverage you can verify on held-out data.
`predict_when`'s windows are calibrated in *time* space (on crossing-time errors),
which is what keeps the 90% window honest.

```python
model = APDTFlowForecaster(use_conformal=True, conformal_method='split')  # or 'adaptive'
model.fit(df, target_col='y')
lower, yhat, upper = model.predict(return_intervals='conformal', alpha=0.1)
```

---

## 2. Inputs & features

**Multivariate sensor fusion** — pass several columns via `feature_cols`; the model
fuses them into one health indicator, and `sensor_importance_` reports how much each
channel contributed.

```python
model.fit(df, target_col='soh', feature_cols=['voltage', 'current', 'temp'])
model.sensor_importance_      # learned weight per input channel
```

**Exogenous variables** — bring known drivers (promotions, weather) with
`concat` / `gated` / `attention` fusion; pass `future_exog_cols` for drivers whose
future values you will know.

```python
model = APDTFlowForecaster(exog_fusion_type='gated')
model.fit(df, target_col='sales', exog_cols=['promo'], future_exog_cols=['promo'])
yhat = model.predict(exog_future=future_promo)
```

**Categorical features** — day-of-week, holidays and other categories as one-hot
vectors or learned embeddings.

```python
model = APDTFlowForecaster(categorical_encoding='embedding')
model.fit(df, target_col='y', categorical_cols=['day_of_week', 'holiday'])
```

**Regime normalization** — standardize multi-condition equipment per operating
regime so baseline shifts do not look like signal.

```python
from apdtflow.preprocessing import regime_normalize
df_norm, stats = regime_normalize(df, op_cols=['regime'], sensor_cols=sensors)
```

---

## 3. Evaluation & monitoring

**Rolling-origin backtesting** (`historical_forecasts`) — replay the model across
history with a moving origin, so the score reflects production behaviour.

```python
bt = model.historical_forecasts(df, target_col='y', start=0.8,
                                stride=7, metrics=['mae', 'mase', 'smape'])
```

**Metric suite** — MAE, RMSE, MAPE, R², MASE, sMAPE, CRPS and interval coverage,
via `model.score(test_df, metric='mase')` and in the backtest results.

**Drift & coverage monitoring** (`score_recent`) — compare recent accuracy and
interval coverage against the calibration baseline; recalibrate when coverage
slips.

```python
s = model.score_recent(recent_df, target_col='y')
s['mae_ratio']     # recent error vs the calibration baseline
s['coverage']      # vs expected coverage
```

---

## 4. Production & architectures

| Capability | Call |
|---|---|
| Persistence (keeps scalers + calibration) | `model.save('m.pt')` / `APDTFlowForecaster.load('m.pt')` |
| TorchScript export (grid forecasts) | `model.export_torchscript('model.pt')` |
| FastAPI serving | see [`examples/serve_api.py`](../examples/serve_api.py) |
| Reproducible runs | `from apdtflow import set_seed; set_seed(0)` |
| Experiment logging (MLflow / W&B) | `fit(df, log_callback=lambda e, m: ...)` |
| sklearn-compatible | `model.get_params()` / `model.set_params(...)` |
| Typed (PEP 561) | ships a `py.typed` marker |
| Command line | `apdtflow train --csv_file data.csv --value_col y` |

**Architectures.** Neural ODE is the default (`model_type='apdtflow'`) and powers
the continuous-time and uncertainty features; `'transformer'` and `'tcn'` are lean
grid forecasters for plain `predict()`.

---

For the full benchmarks and the methodology behind these features, see
[`experiment_results.md`](experiment_results.md) and [`METHODOLOGY.md`](METHODOLOGY.md).
