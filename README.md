<p align="center">
  <img src="assets/images/logo.png" alt="APDTFlow" width="680">
</p>

# APDTFlow — know WHEN it will happen

[![PyPI version](https://img.shields.io/pypi/v/apdtflow.svg)](https://pypi.org/project/apdtflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/apdtflow)](https://pepy.tech/project/apdtflow)
[![Python Versions](https://img.shields.io/pypi/pyversions/apdtflow.svg)](https://pypi.org/project/apdtflow/)
[![CI](https://github.com/yotambraun/APDTFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/yotambraun/APDTFlow/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/yotambraun/APDTFlow/branch/main/graph/badge.svg)](https://codecov.io/gh/yotambraun/APDTFlow)

Forecasting tools tell you *what the value will be*. APDTFlow models time as **continuous**
(Neural ODEs), so it also answers the question operations teams actually ask: **when will
it cross the line?** — with a calibrated uncertainty window on the time itself.

One trained model answers three questions:

| Question | API |
|---|---|
| What are the next k values? | `model.predict()` |
| What is the value at **any** moment — 14:37, in 3.6 days, beyond the trained horizon? | `model.predict_at(timestamps)` |
| **When** will the value cross a threshold — with uncertainty on the time itself? | `model.predict_when(threshold)` |

![fleet dashboard](assets/images/apdtflow_fleet_dashboard.png)

*One call — `predict_when_fleet()` — turns real NASA jet engines (never seen in
training) into a maintenance schedule sorted by act-by date, with what actually
happened marked. <!--PENDING:fd002_actby_caption-->*

## 60-second start

    pip install apdtflow

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(forecast_horizon=40, decoder_type='continuous',
                           use_conformal=True)
model.fit(df, target_col='capacity', feature_cols=sensor_cols)

model.predict()                                       # classic grid forecast
model.predict_at(['2026-06-11 14:37', 3.6])           # value at ANY moment
result = model.predict_when(threshold=1.4,            # WHEN it crosses the line
                            direction='below')
result.eta, result.act_by, result.censored
schedule = model.predict_when_fleet(assets,           # whole fleet -> ranked schedule
                                    threshold=1.4, direction='below')
```

## Verified results (every number reproducible from `experiments/`)

| Event-timing audit (real NASA data, held-out units) | APDTFlow | Linear extrap. | Persistence |
|---|---|---|---|
| Battery end-of-life, 3 cells leave-one-battery-out | <!--PENDING:battery_row--> | | |
| Turbofan FD001, unseen engines | <!--PENDING:fd001_row--> | | |
| Turbofan FD002, unseen engines, 6 operating regimes | <!--PENDING:fd002_row--> | | |

<!--PENDING:coverage_sentence-->

Reproduce: `python experiments/battery_eol_demo.py`, `experiments/turbofan_when_demo.py`,
`experiments/fd002_robustness_demo.py`. Full details: [docs/experiment_results.md](docs/experiment_results.md).

### Real-world demo: battery end-of-life

![battery](assets/images/apdtflow_battery_eol.png)

### Real-world demo: jet-engine maintenance under shifting operating regimes

![fd002](assets/images/apdtflow_fd002_robustness.png)

Adding sensors helps — a learned health-indicator fusion layer
(`fit(..., feature_cols=sensors)`) improves event timing and exposes interpretable
sensor weights via `model.sensor_importance_`:

![multivariate](assets/images/apdtflow_multivariate_win.png)

## Trust panel — this package tells you its own limits

![trust](assets/images/apdtflow_trust_panel.png)

The point estimate runs slightly late on smoothed indicators; the asymmetric
time-space calibration absorbs it. **Operational rule: schedule by `act_by` (the
window's earliest edge), never by the point estimate.** The API returns `act_by`
as a first-class field for exactly this reason.

## predict_at — forecast at any moment in time

![predict_at](examples/predict_at_demo.png)

One trained model, queried at arbitrary real-valued timestamps — fractional steps,
between observations, even beyond the trained horizon — because the decoder
integrates a continuous-time ODE. Conformal intervals come interpolated across
time. (`python examples/predict_at_demo.py` produced this plot.)

## predict_when — a calibrated answer to "when?"

![predict_when](examples/predict_when_demo.png)

*"When will solar activity rise above 80?" — the calibrated 90% window covered the
true crossing, and the `act_by` edge landed before it. Windows are calibrated on
crossing-time errors (time space), not value bands — the distinction that makes
the coverage hold.* (`python examples/predict_when_demo.py`.)

## Is the base forecaster accurate? Honest numbers

6 datasets (2 real, 4 synthetic), 12-step horizon, 30 epochs, MAE relative to
seasonal-naive (<1.0 beats it). Measured June 2026; reproduce:
`python experiments/benchmark_multidomain.py`.

| Dataset | APDTFlow | Linear | Holt-Winters |
|---|---|---|---|
| Daily min temperature (real) | **0.73** | 0.74 | 0.80 |
| Regime-switching nonlinear | **0.77** | 0.83 | 0.86 |
| Trend + dual seasonality | 0.85 | 0.50 | **0.38** |
| Retail-like multiplicative seasonal | 1.01 | **0.68** | 0.81 |
| Electric production (real, 397 pts) | 1.52 | **1.03** | 1.23 |
| Random walk (unpredictable) | 1.86 | 1.15 | 1.12 |

APDTFlow beats seasonal-naive on 3 of 6 domains (parity on a 4th) and beats every
baseline on two. For pure grid accuracy on regular data, tuned deep models
(NeuralForecast) or zero-shot foundation models (Chronos-2, TimesFM, Moirai-2)
may be stronger — APDTFlow's value is what grid models cannot do. We are not
aware of another forecasting library offering `predict_at` at arbitrary
timestamps or calibrated `predict_when` event timing (as of June 2026); if you
use a foundation model for grid accuracy, APDTFlow is complementary.

## When NOT to use APDTFlow

- Stock prices / crypto — random-walk regime; nothing beats naive, including us
  (it's in our benchmark table on purpose).
- Event timing on noise-driven crossings (e.g., which exact day a noisy daily
  series first dips) — no model has skill there; expect wide, honest windows.
- Irregularly-sampled / heavily missing data — we tested ODE-RNN encoders and
  missingness features; both lost to simple imputation baselines
  ([documented](docs/METHODOLOGY.md)).
- Very short series (< ~500 points) — use ETS/ARIMA.

## Industry-grade plumbing

- Split & adaptive **conformal prediction** on every API; time-space calibrated
  windows for event timing
- **Multivariate** health-indicator fusion (`feature_cols=`) with readable
  `sensor_importance_`; per-regime normalization
  (`apdtflow.preprocessing.regime_normalize`) for multi-condition equipment
- **Fleet API**: `predict_when_fleet()` → act-by-sorted schedule, CSV/dict export
- Exogenous & categorical features, backtesting (`historical_forecasts`),
  MASE/sMAPE/CRPS/coverage metrics
- Robust persistence (`save`/`load` with scalers + calibration; pre-0.4
  checkpoints are rejected with guidance), TorchScript export, FastAPI serving
  example, sklearn-style `get_params`/`set_params`, `score_recent()` drift hook,
  `set_seed()` deterministic mode, MLflow/W&B logging hook, `py.typed`
- Architectures: Neural ODE (default), Transformer, TCN, Ensemble

## Versioning

Pre-1.0 we follow SemVer pragmatically: breaking changes land only in minor
releases, are flagged in [CHANGELOG.md](CHANGELOG.md), and deprecated aliases
are kept for one minor release. **All checkpoints from versions ≤ 0.3.x are
invalid** — those versions contained a defect that made predictions independent
of the input data; see the engineering history in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Have degradation or depletion data?

Run the audit yourself: `python experiments/audit_predict_when.py` benchmarks
`predict_when` against persistence, linear extrapolation, and seasonal baselines
on *your* data. If APDTFlow wins, open a PR — we feature your domain with your
numbers. That pipeline is how every result on this page was produced, including
the ideas we tested and **rejected** ([docs/METHODOLOGY.md](docs/METHODOLOGY.md)).

## More

[Methodology & references](docs/METHODOLOGY.md) ·
[Benchmarks](docs/experiment_results.md) ·
[Model architectures](docs/models.md) ·
[Documentation](docs/index.md) ·
[Examples](examples/) ·
[Quickstart notebook](Quickstart.ipynb) ·
[Contributing](CONTRIBUTING.md)

## License

MIT
