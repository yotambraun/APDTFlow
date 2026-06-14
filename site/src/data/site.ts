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
  { label: 'Demos', href: url('demos') },
  { label: 'Evidence', href: url('evidence') },
  { label: 'Methodology', href: url('methodology') },
];
