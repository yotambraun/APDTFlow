// Central constants for the APDTFlow site. Change numbers/links here once.

// ---- base-path helper (the #1 gotcha for GitHub project pages) ----
// BASE_URL may be "/APDTFlow" or "/APDTFlow/" depending on Astro version, so we
// normalize to exactly one slash between base and path. Route every internal
// link/asset through url() so nothing 404s under the project base.
const BASE = import.meta.env.BASE_URL.replace(/\/$/, ''); // -> "/APDTFlow"
export const url = (p = ''): string => {
  const clean = p.replace(/^\//, '');
  return clean ? `${BASE}/${clean}` : `${BASE}/`;
};
export const img = (name: string): string => url(`images/${name}`);

export const site = {
  name: 'APDTFlow',
  tagline: 'know WHEN it will happen',
  version: '0.4.0',
  license: 'MIT',
  author: 'Yotam Braun',
  description:
    'Continuous-time forecasting with Neural ODEs. Answers not just what the value will be, but when it will cross the line — with a calibrated window on the time itself.',
  repo: 'https://github.com/yotambraun/APDTFlow',
  pypi: 'https://pypi.org/project/apdtflow/',
  pepy: 'https://pepy.tech/project/apdtflow',
  colab:
    'https://colab.research.google.com/github/yotambraun/APDTFlow/blob/main/Quickstart.ipynb',
  install: 'pip install apdtflow',
};

// Repo doc deep-links (single source of truth lives in the repo markdown).
export const docs = {
  methodology: `${site.repo}/blob/main/docs/METHODOLOGY.md`,
  models: `${site.repo}/blob/main/docs/models.md`,
  experiments: `${site.repo}/blob/main/docs/experiment_results.md`,
  guide: `${site.repo}/blob/main/docs/index.md`,
  features: `${site.repo}/blob/main/docs/features.md`,
  changelog: `${site.repo}/blob/main/CHANGELOG.md`,
};

// Articles (from README).
export const articles = [
  {
    title: 'A Practical Approach to Time Series Forecasting with APDTFlow',
    outlet: 'Towards AI',
    href: 'https://pub.towardsai.net/a-practical-approach-to-time-series-forecasting-with-apdtflow-78e06cb51a1b',
  },
  {
    title: 'APDTFlow v0.3.0: From Research to Production-Ready Time Series Forecasting',
    outlet: 'Towards AI',
    href: 'https://medium.com/towards-artificial-intelligence/apdtflow-v0-3-0-from-research-to-production-ready-time-series-forecasting-0be1848e323a',
  },
];

// Stat callouts. Numbers refresh per deploy via fetchStats(); fallbacks here
// keep the build green if an API is flaky. Update these constants when convenient.
export const statsFallback = {
  downloads: '40k+',
  stars: '40+',
};

// The three questions one model answers (README).
export const questions = [
  {
    q: 'What are the next k values?',
    api: 'model.predict()',
    blurb: 'Classic grid forecast with optional conformal intervals.',
    emphasized: false,
  },
  {
    q: 'What is the value at any moment — 14:37, in 3.6 days, beyond the trained horizon?',
    api: 'model.predict_at(timestamps)',
    blurb: 'Query a continuous-time decoder at arbitrary real-valued times.',
    emphasized: false,
  },
  {
    q: 'When will the value cross a threshold, with uncertainty on the time itself?',
    api: 'model.predict_when(threshold)',
    blurb: 'A calibrated 90% window on the crossing time, plus an act_by edge to schedule against.',
    emphasized: true,
  },
];

// SIGNATURE section — measured event-timing audits (README). Honest: FD002 loses.
// winner: which column holds the best (lowest) timing MAE for that row.
export const evidenceRows = [
  {
    audit: 'Battery end-of-life',
    detail: '3 cells, leave-one-battery-out',
    apdtflow: '8.3',
    linear: '9.7',
    persistence: '15.4',
    winner: 'apdtflow' as const,
  },
  {
    audit: 'Turbofan FD001',
    detail: '40 unseen engines, 0.6% false alarms',
    apdtflow: '8.3',
    linear: '8.7',
    persistence: '11.5',
    winner: 'apdtflow' as const,
  },
  {
    audit: 'Turbofan FD002',
    detail: '110 unseen engines, 6 operating regimes, 0.0% false alarms',
    apdtflow: '9.2',
    linear: '8.1',
    persistence: '11.3',
    winner: 'linear' as const,
  },
];

export const evidenceChips = [
  '0 false alarms across 2,638 no-crossing windows',
  'coverage 96 / 40 / 54 vs a 90% target',
  'every number reproducible from experiments/',
];

// Grid-accuracy benchmark (README) — MAE relative to seasonal-naive (<1.0 beats it).
export const gridRows = [
  { dataset: 'Daily min temperature (real)', apdtflow: '0.73', linear: '0.74', hw: '0.80', winner: 'apdtflow' as const },
  { dataset: 'Regime-switching nonlinear', apdtflow: '0.77', linear: '0.83', hw: '0.86', winner: 'apdtflow' as const },
  { dataset: 'Trend + dual seasonality', apdtflow: '0.85', linear: '0.50', hw: '0.38', winner: 'hw' as const },
  { dataset: 'Retail-like multiplicative seasonal', apdtflow: '1.01', linear: '0.68', hw: '0.81', winner: 'linear' as const },
  { dataset: 'Electric production (real, 397 pts)', apdtflow: '1.52', linear: '1.03', hw: '1.23', winner: 'linear' as const },
  { dataset: 'Random walk (unpredictable)', apdtflow: '1.86', linear: '1.15', hw: '1.12', winner: 'linear' as const },
];

// Capability matrix (README) — capabilities, not accuracy.
export const capabilities = [
  { label: 'Grid forecasts (predict)', apdtflow: 'yes', grid: 'yes', foundation: 'yes' },
  { label: 'Calibrated conformal intervals', apdtflow: 'yes', grid: 'partial', foundation: 'partial' },
  { label: 'Forecast at arbitrary real-valued times (predict_at)', apdtflow: 'yes', grid: 'no', foundation: 'no' },
  { label: 'Event timing with calibrated windows (predict_when)', apdtflow: 'yes', grid: 'no', foundation: 'no' },
  { label: 'Fleet-level act-by scheduling', apdtflow: 'yes', grid: 'no', foundation: 'no' },
  { label: 'Zero-shot (no training)', apdtflow: 'no', grid: 'no', foundation: 'yes' },
];

// When NOT to use (README) — the honesty that defines the brand.
export const whenNot = [
  'Stock prices / crypto: random-walk regime; nothing beats naive, including us.',
  'Event timing on noise-driven crossings: no model has skill there; expect wide, honest windows.',
  'Irregularly-sampled / heavily missing data: ODE-RNN encoders lost to simple imputation in our tests.',
  'Very short series (< ~500 points): use ETS / ARIMA.',
];

export const nav = [
  { label: 'Quickstart', href: url('quickstart') },
  { label: 'Features', href: url('features') },
  { label: 'Demos', href: url('demos') },
  { label: 'Evidence', href: url('evidence') },
  { label: 'Methodology', href: url('methodology') },
];

// ---- Features content (hub + sub-pages) ----
// GitHub deep-link to an example script (external; uses site.repo, not url()).
export const exampleUrl = (file: string): string =>
  `${site.repo}/blob/main/examples/${file}`;

export interface FeatureItem {
  name: string;
  value: string;          // one-line tagline, value-first
  description: string;    // 1-3 sentences: what it means and when to use it
  api?: string;           // the real call, shown as inline mono
  snippet?: string;       // optional usage snippet -> <CodeBlock>
  note?: string;          // one light "good to know" line, only where it aids usage
  example?: string;       // clean example filename (rendered via exampleUrl)
  exampleLabel?: string;
}
export interface FeatureGroup {
  section: string;        // which sub-page this group belongs to (slug)
  title: string;
  blurb: string;
  layout: 'cards' | 'list';
  items: FeatureItem[];
}
export interface FeatureSection {
  slug: string;
  title: string;
  summary: string;        // one line, shown on the hub card
  intro: string;          // a sentence or two at the top of the sub-page
}

// Sub-pages of /features. Each FeatureGroup below is attached to one of these.
export const featureSections: FeatureSection[] = [
  {
    slug: 'forecasting',
    title: 'Forecasting & uncertainty',
    summary: 'The questions one trained model answers, and the calibrated intervals behind them.',
    intro:
      'One APDTFlowForecaster, trained once, answers the next-values question and two that grid forecasters cannot — the value at any moment, and when a threshold is crossed — each with calibrated uncertainty.',
  },
  {
    slug: 'inputs',
    title: 'Inputs & features',
    summary: 'Go beyond a single series: sensors, known drivers, categories, operating regimes.',
    intro:
      'Real series rarely live alone. APDTFlow takes multiple sensors, external drivers, categorical signals, and multi-regime equipment data — and tells you what it leaned on.',
  },
  {
    slug: 'evaluation',
    title: 'Evaluation & monitoring',
    summary: 'Score the way production will, then watch for drift after you ship.',
    intro:
      'Trust a forecaster by testing it honestly and watching it over time. APDTFlow backtests across history and tracks accuracy and interval coverage against its calibration baseline.',
  },
  {
    slug: 'production',
    title: 'Production & architectures',
    summary: 'Persist, export, serve and reproduce — plus the model choices under the hood.',
    intro:
      'The plumbing that gets a model into production: save/load that keeps calibration intact, export and serving, reproducibility and logging hooks — and the architectures you can pick from.',
  },
];

export const featureGroups: FeatureGroup[] = [
  {
    section: 'forecasting',
    title: 'The questions one model answers',
    blurb: 'Train once, then ask for the next values, the value at any moment, or when a line gets crossed.',
    layout: 'cards',
    items: [
      {
        name: 'Grid forecast',
        value: 'The next k values, with optional calibrated intervals.',
        description:
          'The everyday forecast: from your recent history, predict the next k points on the same time grid. Add return_intervals=\'conformal\' when you want a calibrated range to plan against instead of a single line.',
        api: 'model.predict()',
        snippet: `model = APDTFlowForecaster(forecast_horizon=24)
model.fit(df, target_col='sales')

yhat = model.predict()                                   # next 24 steps
lower, yhat, upper = model.predict(return_intervals='conformal', alpha=0.1)`,
      },
      {
        name: 'Forecast at any moment',
        value: 'Query arbitrary real-valued times — fractional steps, between observations, beyond the horizon.',
        description:
          'Most models only score the fixed steps they trained on. APDTFlow integrates a continuous-time ODE, so you can ask for the value 3.6 steps out, at a timestamp that falls between samples, or past the trained horizon — useful for irregular reporting dates or "what will it be at 2pm Friday?".',
        api: "model.predict_at(['2026-06-14 14:37', 3.6])",
        note: "Enabled by decoder_type='continuous'.",
        snippet: `model = APDTFlowForecaster(forecast_horizon=14, decoder_type='continuous',
                           use_conformal=True)
model.fit(df, target_col='value', date_col='date')

values, lower, upper = model.predict_at([3.5, 7.2, 18.0])`,
        example: 'predict_at_demo.py',
        exampleLabel: 'predict_at_demo.py',
      },
      {
        name: 'When will it cross the line?',
        value: 'A calibrated time window on the crossing, plus an act_by edge to schedule against.',
        description:
          'Flip the question from "what value" to "when does it reach this level". You get a point estimate (eta), a calibrated 90% window for the crossing time, and act_by — the early edge of that window — which is the number you actually schedule against. Use mode=\'risk\' for the earliest plausible crossing when being early is safer than being exact.',
        api: "model.predict_when(threshold, direction='above')",
        note: "Enabled by decoder_type='continuous' + use_conformal=True.",
        snippet: `r = model.predict_when(threshold=80, direction='above', alpha=0.1)
r.eta                 # point estimate (steps ahead)
r.earliest, r.latest  # calibrated 90% window
r.act_by              # schedule by this edge
r.censored            # True if no crossing within the horizon

# earliest plausible crossing, for safety-critical scheduling:
risk = model.predict_when(80, direction='above', mode='risk')`,
        example: 'predict_when_demo.py',
        exampleLabel: 'predict_when_demo.py',
      },
      {
        name: 'Schedule a whole fleet',
        value: 'Turn a dict of asset histories into a maintenance schedule sorted by act-by date.',
        description:
          'Apply predict_when across many assets in one call. The returned table is sorted by act_by, so whatever needs attention soonest sits at the top — ready to hand to a maintenance/CMMS system as CSV.',
        api: 'model.predict_when_fleet(assets, threshold, direction)',
        snippet: `schedule = model.predict_when_fleet(assets, threshold=1.4, direction='below')
# DataFrame: asset_id, eta, earliest, latest, act_by, censored, confidence
schedule.to_csv('maintenance_schedule.csv')`,
      },
    ],
  },
  {
    section: 'forecasting',
    title: 'Calibrated uncertainty',
    blurb: 'Intervals you can check, not just print — calibrated by conformal prediction.',
    layout: 'cards',
    items: [
      {
        name: 'Conformal intervals',
        value: 'Split and adaptive conformal calibration on every forecast.',
        description:
          'Conformal prediction wraps a point forecast in an interval whose coverage you can verify on held-out data — split conformal is fast, adaptive adjusts as the distribution shifts. Switch it on at construction and predict() can hand back calibrated lower/upper bands.',
        api: "APDTFlowForecaster(use_conformal=True, conformal_method='adaptive')",
        snippet: `model = APDTFlowForecaster(use_conformal=True, conformal_method='split')
model.fit(df, target_col='y')

lower, yhat, upper = model.predict(return_intervals='conformal', alpha=0.1)`,
        example: 'conformal_prediction_demo.py',
        exampleLabel: 'conformal_prediction_demo.py',
      },
      {
        name: 'Calibrated event-time windows',
        value: 'predict_when calibrates in time space, so its 90% windows actually hold.',
        description:
          'For predict_when the window is calibrated on crossing-time errors in time space rather than on value bands — that is what keeps the 90% window honest. act_by is exposed as a first-class field so scheduling never rides on the (often optimistic) point estimate.',
        api: 'result.act_by, result.earliest, result.latest',
      },
    ],
  },
  {
    section: 'inputs',
    title: 'Rich inputs',
    blurb: 'Feed more than a single series — sensors, drivers, categories, operating regimes.',
    layout: 'cards',
    items: [
      {
        name: 'Multivariate sensor fusion',
        value: 'Fuse many channels into one health indicator and read which mattered.',
        description:
          'Pass several columns via feature_cols and the model learns to fuse them into one health indicator for your target. sensor_importance_ then reports how much each channel contributed, so the prediction stays explainable.',
        api: 'model.fit(..., feature_cols=sensors); model.sensor_importance_',
        snippet: `model.fit(df, target_col='soh', feature_cols=['voltage', 'current', 'temp'])
model.sensor_importance_      # learned weight per input channel`,
      },
      {
        name: 'Exogenous variables',
        value: 'Bring known drivers (promotions, weather) with concat / gated / attention fusion.',
        description:
          'Add outside drivers that move the series — promotions, weather, price. Choose how they combine with the signal (concat, gated, or attention), and pass future_exog_cols for drivers whose future values you will know at forecast time.',
        api: 'fit(..., exog_cols=[...], future_exog_cols=[...])',
        snippet: `model = APDTFlowForecaster(exog_fusion_type='gated')
model.fit(df, target_col='sales', exog_cols=['promo'], future_exog_cols=['promo'])

yhat = model.predict(exog_future=future_promo)`,
        example: 'backtesting_demo.py',
        exampleLabel: 'backtesting_demo.py (Example 5)',
      },
      {
        name: 'Categorical features',
        value: 'Day-of-week, holidays and other categories via one-hot or learned embeddings.',
        description:
          'Turn discrete inputs like day-of-week or a holiday flag into model features, as plain one-hot vectors or trainable embeddings, so calendar and category effects get captured.',
        api: "fit(..., categorical_cols=['day_of_week'])",
        snippet: `model = APDTFlowForecaster(categorical_encoding='embedding')
model.fit(df, target_col='y', categorical_cols=['day_of_week', 'holiday'])`,
      },
      {
        name: 'Regime normalization',
        value: 'Normalize multi-condition equipment per operating regime before fitting.',
        description:
          'Industrial assets often switch between operating regimes that shift every sensor\'s baseline. regime_normalize() standardizes each regime on its own so those shifts do not look like signal; keep the returned stats and reuse them at inference.',
        api: 'regime_normalize(df, op_cols, sensor_cols)',
        snippet: `from apdtflow.preprocessing import regime_normalize

df_norm, stats = regime_normalize(df, op_cols=['regime'], sensor_cols=sensors)`,
      },
    ],
  },
  {
    section: 'evaluation',
    title: 'Evaluation & monitoring',
    blurb: 'Score honestly, then watch for drift in production.',
    layout: 'cards',
    items: [
      {
        name: 'Rolling-origin backtesting',
        value: 'Simulate production forecasting over history — the realistic way to score a model.',
        description:
          'historical_forecasts() replays the model across your history with a moving origin, so the score reflects how it would have done in production rather than on one lucky split. Tune where it starts, how often it forecasts (stride), and whether it retrains each fold.',
        api: 'model.historical_forecasts(df, start=0.8, stride=7)',
        snippet: `bt = model.historical_forecasts(df, target_col='y', start=0.8,
                                stride=7, metrics=['mae', 'mase', 'smape'])`,
        example: 'backtesting_demo.py',
        exampleLabel: 'backtesting_demo.py',
      },
      {
        name: 'A full metric suite',
        value: 'MAE, RMSE, MAPE, R², MASE, sMAPE, CRPS and interval coverage, out of the box.',
        description:
          'Reach for the metric that fits the job — scale-free MASE to compare across series, sMAPE/MAPE for percentage error, CRPS and coverage to judge interval quality — all available through score() and the backtest results.',
        api: "model.score(test_df, metric='mase')",
      },
      {
        name: 'Drift & coverage monitoring',
        value: 'Compare recent accuracy and interval coverage against the calibration baseline.',
        description:
          'Once deployed, score_recent() checks fresh data against the calibration baseline. A rising error ratio, or coverage slipping under target, is your cue to recalibrate or refit before forecasts go stale.',
        api: 'model.score_recent(recent_df)',
        snippet: `s = model.score_recent(recent_df, target_col='y')
s['mae_ratio']     # recent error vs the calibration baseline
s['coverage']      # vs expected coverage — recalibrate if it drops`,
      },
    ],
  },
  {
    section: 'production',
    title: 'Production & ops',
    blurb: 'The plumbing to ship it: persist, export, serve, log, reproduce.',
    layout: 'list',
    items: [
      {
        name: 'Persistence',
        value: 'Save and load the model with its scalers and conformal calibration intact.',
        description:
          'save() / load() round-trip the weights, the fitted scalers, and the conformal calibration together, so a reloaded model forecasts and calibrates exactly as before. Checkpoints record the library version and refuse incompatible ones.',
        api: "model.save('m.pt'); APDTFlowForecaster.load('m.pt')",
      },
      {
        name: 'TorchScript export',
        value: 'Export grid-forecast inference to TorchScript.',
        description:
          'Trace the grid-forecast path into a TorchScript module for lean, Python-optional serving. (predict_at and predict_when stay in the Python API.)',
        api: "model.export_torchscript('model.pt')",
      },
      {
        name: 'FastAPI serving',
        value: 'Serve forecasts over HTTP with a small FastAPI app.',
        description:
          'A small, ready-to-adapt FastAPI app that exposes forecast endpoints over HTTP — a starting point for your own service.',
        api: 'uvicorn serve_api:app',
        example: 'serve_api.py',
        exampleLabel: 'serve_api.py',
      },
      {
        name: 'Reproducible runs',
        value: 'One call for deterministic seeding across NumPy and PyTorch.',
        description:
          'set_seed() seeds Python, NumPy and PyTorch and turns on deterministic mode, so the same inputs give the same model.',
        api: 'from apdtflow import set_seed; set_seed(0)',
      },
      {
        name: 'Experiment logging',
        value: 'Stream epoch metrics to MLflow or Weights & Biases.',
        description:
          'fit() accepts a log_callback(epoch, metrics) hook that drops straight into MLflow or Weights & Biases for live training curves.',
        api: 'model.fit(df, log_callback=lambda e, m: mlflow.log_metrics(m, step=e))',
      },
      {
        name: 'sklearn-compatible',
        value: 'get_params / set_params for pipelines and hyperparameter search.',
        description:
          'get_params / set_params follow the scikit-learn estimator contract, so APDTFlow slots into Pipelines and hyperparameter search.',
        api: 'model.get_params(); model.set_params(num_epochs=30)',
      },
      {
        name: 'Typed',
        value: 'Ships a py.typed marker for full static type checking.',
        description:
          'A py.typed marker ships inline type information for editors and type checkers.',
        api: 'import apdtflow  # PEP 561 typed',
      },
      {
        name: 'Command line',
        value: 'Train and forecast straight from the terminal.',
        description:
          'apdtflow train and apdtflow infer cover CSV-in / forecast-out workflows without writing any Python.',
        api: 'apdtflow train --csv_file data.csv --value_col y',
      },
    ],
  },
  {
    section: 'production',
    title: 'Architectures',
    blurb: 'Neural ODE is the default and powers the continuous-time and uncertainty features; Transformer and TCN are lean grid forecasters.',
    layout: 'list',
    items: [
      {
        name: 'Neural ODE',
        value: 'Continuous latent dynamics — the engine behind predict_at and predict_when. (default)',
        description:
          'The default. It models the latent dynamics as a continuous ODE — the foundation for predict_at and predict_when — and supports conformal intervals, exogenous, categorical and multivariate inputs.',
        api: "APDTFlowForecaster(model_type='apdtflow')",
      },
      {
        name: 'Transformer',
        value: 'Attention-based grid forecaster.',
        description: 'An attention-based sequence model for straightforward grid forecasts.',
        api: "APDTFlowForecaster(model_type='transformer')",
      },
      {
        name: 'TCN',
        value: 'Temporal convolutional grid forecaster.',
        description: 'A temporal convolutional network for straightforward grid forecasts.',
        api: "APDTFlowForecaster(model_type='tcn')",
      },
    ],
  },
];
