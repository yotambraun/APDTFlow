# APDTFlow Examples

Ready-to-run scripts demonstrating APDTFlow. All scripts run from the **repo root**:

```bash
python examples/<name>.py
```

Plots are saved into this directory. Scripts that train a model seed all RNGs with
`apdtflow.set_seed(0)` so runs are reproducible.

---

## Start here: the continuous-time API (v0.4.0)

### 1. [predict_at_demo.py](predict_at_demo.py) — forecast at ANY moment in time

Most forecasting tools answer "what will the value be at step k?". APDTFlow's
continuous-time decoder answers "what will the value be at 14:37 next Tuesday?" —
fractional steps, arbitrary timestamps, and times beyond the trained horizon — from
one trained model, with conformal intervals. Uses the real daily-minimum-temperatures
dataset.

```bash
python examples/predict_at_demo.py
```

### 2. [predict_when_demo.py](predict_when_demo.py) — forecast WHEN a threshold is crossed

"When will solar activity rise above 80?" — `predict_when` answers with a calibrated
time window, not just a point guess, and returns an `act_by` date (the window's early
edge) to schedule against. Also shows `mode='risk'` ("from when is the crossing
plausible?") and censoring. Uses the monthly-sunspots dataset.

```bash
python examples/predict_when_demo.py
```

Note: this script demonstrates the API on a cyclic series. The domains where
`predict_when` has audited, baseline-beating skill are degradation/depletion problems
— see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md) §5–7 and the battery playground
below.

### 3. [battery_rul_playground.py](battery_rul_playground.py) — battery remaining useful life

Pick a forecast origin anywhere along a real NASA battery's life and see the capacity
forecast, the predicted end-of-life window, and how the RUL estimate converges as the
origin advances. Every number comes from models trained on the *other* two cells
(leave-one-battery-out) — nothing is simulated.

```bash
python examples/battery_rul_playground.py --origin 90
python examples/battery_rul_playground.py --cell B0006 --origin 120
```

Notebook twin with an interactive origin slider:
[battery_rul_playground.ipynb](battery_rul_playground.ipynb)
(also runs on Colab — data is fetched from GitHub when not local).

---

## Core forecasting API

### 4. [quickstart_easy_api.py](quickstart_easy_api.py)

The simplest way to start: `fit()`/`predict()` on the Electric Production dataset,
uncertainty estimates, and the built-in `plot_forecast()`.

### 5. [conformal_prediction_demo.py](conformal_prediction_demo.py)

Calibrated prediction intervals with finite-sample coverage guarantees: split
conformal (simple, reliable) and adaptive conformal (for non-stationary data), with
empirical coverage checks.

### 6. [exogenous_variables_demo.py](exogenous_variables_demo.py)

External features (temperature, holidays, promotions): past-observed vs future-known
covariates and the three fusion strategies (`concat`, `gated`, `attention`).

### 7. [categorical_features_demo.py](categorical_features_demo.py)

Categorical variables (day-of-week, holidays, categories) with one-hot and embedding
encodings, including unseen-category handling.

### 8. [backtesting_demo.py](backtesting_demo.py)

`historical_forecasts()` rolling-origin backtesting: fixed-model vs retrain modes,
stride selection, and MASE/sMAPE evaluation.

### 9. [migrate_from_prophet.py](migrate_from_prophet.py)

Side-by-side Prophet-to-APDTFlow migration: same data, equivalent calls, plus the
features Prophet does not have (conformal intervals, exogenous fusion,
continuous-time queries).

### 10. [serve_api.py](serve_api.py)

Serve a fitted, saved model over HTTP with FastAPI:

```bash
pip install apdtflow[serve]
uvicorn serve_api:app --port 8000   # run from examples/
curl localhost:8000/forecast
curl -X POST localhost:8000/predict_when \
     -H 'Content-Type: application/json' \
     -d '{"threshold": 1.4, "direction": "below"}'
```

---

## Notebooks

- [../Quickstart.ipynb](../Quickstart.ipynb) — the three questions
  (`predict` / `predict_at` / `predict_when`) in one notebook; Colab-ready.
- [battery_rul_playground.ipynb](battery_rul_playground.ipynb) — interactive battery
  RUL exploration; Colab-ready.

## Datasets

Examples use real datasets from [`dataset_examples/`](../dataset_examples):

- `daily-minimum-temperatures-in-me_clean.csv` — daily minimum temperatures (Melbourne)
- `monthly-sunspots.csv` — monthly sunspot counts since 1749
- `Electric_Production.csv` — U.S. monthly electric production
- `nasa_B0005.csv`, `nasa_B0006.csv`, `nasa_B0007.csv` — NASA battery degradation
  (Saha & Goebel, 2007)
- `cmapss/` + `get_cmapss.py` — NASA C-MAPSS turbofan data (downloader script)
- `get_opsd_germany.py` — Open Power System Data downloader

The exogenous, categorical, conformal, and backtesting demos generate small synthetic
datasets inline so their effects are unambiguous.

## Help

- Documentation: [docs/index.md](../docs/index.md)
- Architecture: [docs/models.md](../docs/models.md)
- Methodology and evaluation protocol: [docs/METHODOLOGY.md](../docs/METHODOLOGY.md)
- Measured results: [docs/experiment_results.md](../docs/experiment_results.md)
- Issues: [GitHub Issues](https://github.com/yotambraun/APDTFlow/issues)
