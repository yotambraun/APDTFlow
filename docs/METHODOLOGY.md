# APDTFlow — Methodology & References

This document explains how APDTFlow works, what literature each component builds on,
how every published number was produced, and — equally important — what we tested and
chose **not** to ship. It exists because users asked for it (GitHub Issue #1), and
because a forecasting package should be able to show its work.

Every number cited here is reproducible by a script committed under `experiments/`.

---

## 1. Architecture

APDTFlow's core model combines four ideas:

**Multi-scale decomposition.** The input window is decomposed by parallel dilated
convolutions into components at multiple temporal scales, each modeled separately and
fused downstream. This follows the long line of decomposition-based forecasters and
gives the latent dynamics cleaner signals to integrate.

**Neural ODE latent dynamics.** Each scale component drives a latent state evolved by a
learned ordinary differential equation, integrated with `torchdiffeq` (Chen et al.,
2018). Treating the latent state as a continuous-time trajectory — rather than a
discrete recurrence — is what later enables `predict_at` and `predict_when` (Sections 3
and 5). We integrate with fixed-step RK4 by default (`ode_method="rk4"`, ~10x faster on
CPU); the adaptive adjoint method remains available (`ode_method="dopri5_adjoint"`).

**Time embedding with gated residual networks.** Calendar/time-index information enters
through a learnable embedding processed by gated residual networks in the style of the
Temporal Fusion Transformer (Lim et al., 2021). Critically, the embedding is **added to**
the series values, never substituted for them (see Section 8, Engineering history).

**Transformer decoder over the full latent trajectory + linear residual skip.** The
decoder attends over the entire encoded latent trajectory (length `T_in`), not a single
summary vector. A direct linear map from the input window to the forecast is added to
the decoder output, in the spirit of N-BEATS' basis projections (Oreshkin et al., 2020)
and the DLinear result that linear maps are a strong backbone for forecasting (Zeng et
al., 2023). The deep path learns the residual the linear map cannot express.

Per-scale latent trajectories are fused by uncertainty-weighted attention
(`ProbScaleFusion`), computed per timestep.

## 2. Training objective

The default training loss is MSE (`loss_type="mse"`); Gaussian negative log-likelihood
is available (`loss_type="nll"`). In our experiments MSE trained faster and more
accurately. A direct consequence we want users to understand: **under MSE training the
model's variance head is untrained and its raw values are meaningless.** All uncertainty
APDTFlow shows — plot bands, `predict_when` windows — comes from conformal calibration
(Section 4), never from raw variance outputs, unless the model was explicitly trained
with `loss_type="nll"`.

## 3. Continuous-time decoding: `predict_at`

With `decoder_type='continuous'`, the encoder's final fused latent state is integrated
forward by a second learned ODE and decoded at **arbitrary real-valued time offsets** —
fractional steps, or beyond the trained horizon. Two implementation details matter and
are enforced by unit tests:

- a **per-step linear skip** interpolated smoothly across continuous time (without it,
  the decoder collapses to level-only output and cannot express cyclic structure — a
  failure we measured and now gate with a mandatory "cycle expressiveness" test);
- **randomized query-time training**: query times are sampled in (0, horizon] each
  batch, so off-grid accuracy is trained rather than emergent.

Both are enforced by unit tests in `tests/test_continuous.py`: the mandatory
cycle-expressiveness gate (forecast variance must reach at least 60% of the
target's within-horizon variance on cyclic data — the production decoder passes;
the prototype did not), and a predict_when time-window coverage gate (>= 85% on the
held-out sunspots fixture — measured passing). We claim the *capability* of
arbitrary-time forecasting and its per-time uncertainty, not an accuracy edge over
interpolating grid forecasts.

## 4. Uncertainty: conformal prediction, in value space and in time space

APDTFlow uses split conformal prediction (Vovk et al., 2005; Romano et al., 2019), with
adaptive variants following Gibbs & Candès (2021). Recent work supports per-step
split-conformal for multi-step forecasting (arXiv:2601.18509; see also the survey of
conformal methods under non-exchangeability, arXiv:2511.13608). Conformal prediction
beyond change points (CPTC, arXiv:2509.02844) is noted as future work for regime-break
robustness.

Two methodological findings from our own validation are now part of the package:

**Time-space calibration.** Mapping value-space conformal bands onto the time axis
miscalibrates when trajectories are steep or flat. Calibrating directly on
**crossing-time errors** — taking quantiles of `t_pred − t_actual` on a calibration
split — is the method behind every `predict_when` window we publish; its coverage is
unit-tested at >= 85% on the held-out sunspots fixture.

**Asymmetric windows.** Crossing-time errors are systematically signed (Section 5), so
we calibrate the 5th and 95th percentiles of *signed* errors rather than a symmetric
absolute quantile. Measured on the NASA battery audit, the calibrated windows reach
96% coverage at a 90% target on the typical held-out cell; on cross-unit transfer to
unseen engines coverage degrades below target (40–54%), which we report in
[experiment_results.md](experiment_results.md) and display in the trust panel rather
than hide.

## 5. Event-time forecasting: `predict_when`

`predict_when(threshold, direction, mode, alpha)` answers "when will the series cross
this level?" with a calibrated time window.

**Why this needs care.** Per-step marginal forecasts provably cannot determine
path-dependent event probabilities — infinitely many joint distributions share the same
marginals but answer "when" differently (Perez-Diaz et al., arXiv:2510.19345). APDTFlow
therefore (a) restricts its claims to the regime where mean-trajectory crossing is
informative — *distant, forecastable* crossings such as degradation toward a limit —
and (b) calibrates the timing uncertainty empirically in time space (Section 4), rather
than deriving it from marginal bands.

**Two modes for two questions.**
- `mode='expected'`: first crossing of the mean trajectory, with the calibrated
  time window. For forecastable trends and cycles (equipment degradation, depletion).
- `mode='risk'`: the earliest time the alpha-level conformal band crosses — "from when
  is the event *plausible*?" For noisy series and safety thresholds. High recall, valid
  early bound; precision is bounded by the plausibility semantics, and we say so.

**Censoring.** "No crossing within the horizon" is a first-class, validated answer
(returned with `censored=True`), not a failure mode.

**The operational rule.** Across our audits the point estimate is systematically
biased on degradation indicators (it saturates toward mid-horizon for distant
events), and cross-unit window coverage runs below target. Consequently the API
returns `act_by` — the window's earliest edge — as a first-class field, and the
documented rule is: **schedule by `act_by`, never by the point estimate.** In the
measured FD002 fleet snapshot the calibrated windows covered 81% of actual
crossings.

## 6. Evaluation protocol

Every published `predict_when` result follows the same adversarial protocol
(`experiments/audit_predict_when.py`), and a domain demo is published **only if it
beats all of**:

1. **persistence** ("it's crossing now or never"),
2. **linear extrapolation** of the recent window,
3. **seasonal-naive / climatology**, wherever the series is seasonal —

on held-out data, on both event capture and timing error. Thresholds, sensor selection,
and normalization statistics are always defined on **training units only**; evaluation
is leave-unit-out (unseen batteries, unseen engines). Point-forecast accuracy is
reported as MAE relative to seasonal-naive, i.e. MASE-style scaling (Hyndman & Koehler,
2006).

### Verified results (measured June 2026)

| Audit (real NASA data, held-out units) | APDTFlow | Linear | Persistence | Coverage (90% target) | False alarms |
|---|---|---|---|---|---|
| Battery EOL, 3 cells, leave-one-battery-out (Saha & Goebel, 2007) | **8.3** (2.8 typical cell) | 9.7 | 15.4 | 96% typical cell | — |
| Turbofan C-MAPSS FD001, 40 unseen engines (Saxena et al., 2008) | **8.3** | 8.7 | 11.5 | 40% | 0.6% |
| FD001, multivariate 5-sensor fusion, same engines | 10.1 full / **5.9 caught** | 8.7 | 11.5 | 26% | 0.0% |
| C-MAPSS FD002, 110 unseen engines, 6 operating regimes | 9.2 (**6.8** matched) | **8.1** | 11.3 | 54% | 0.0% |

Timing MAE in cycles on the full event set (censored answers scored at the horizon);
every row is reproduced by a committed script. We publish the losses with the wins:
linear extrapolation beats the model on FD002's full event set, multivariate fusion
trades coverage for sharpness rather than dominating, and cross-unit window coverage
runs below its target on the engine audits. APDTFlow's measured advantages are
timing wins on battery and FD001 (including FD001's matched subset, 8.5 vs 9.7),
near-zero false alarms everywhere (0/2,638 no-crossing windows on FD002), and
beating persistence by 2–7 cycles on every audit. The base forecaster's general
accuracy is reported honestly in the README: it beats seasonal-naive on 3 of 6
benchmark domains (parity on a 4th); for pure grid accuracy, tuned deep models and
time-series foundation models (Chronos-2, TimesFM 2.5, Moirai-2) may be stronger.
None of those models, being grid/patch-based, offers `predict_at` at arbitrary
timestamps or calibrated `predict_when` — the capabilities are complementary.

## 7. Negative results — what we tested and do not ship

We consider documented rejections part of the methodology. Each was implemented,
benchmarked, and excluded; details and harnesses are in `experiments/`.

1. **Stock/crypto forecasting** — random-walk regime; nothing beats naive (our own
   benchmark includes a random walk to demonstrate this).
2. **Frost-timing demo** (real daily temperatures) — indistinguishable from a
   calendar/climatology rule; daily weather timing is not forecastable from the series.
3. **Solar-activity timing demo** — persistence matched or beat the model; most
   crossings were near-immediate. Kept only as a calibration test fixture.
4. **Mask + time-since-observation features for missing data** — worse than plain
   imputation at 30–50% missingness.
5. **ODE-RNN / latent-ODE encoder for irregular sampling** — lost to every baseline,
   including linear + forward-fill; consistent with arXiv:2505.00590. Research track
   only; APDTFlow makes **no irregular-sampling claims**.
6. **`predict_total`** (integral over lead time) — no edge over summing grid forecasts.
7. **`predict_rate` / turning-point detection** — differentiation amplifies error
   (57% relative rate error); not shipped.
8. **Energy-demand timing with the prototype decoder** — the prototype could not
   express weekly cycles (now gated by the cycle-expressiveness test); seasonal-naive
   sets a 0.9-day bar on that data. Retry only with the production decoder, under the
   Section 6 protocol.

## 8. Engineering history

Versions ≤ 0.3.x contained a critical defect: with `use_embedding=True` (the default),
the model's forward pass **replaced** the input series with a time-index embedding
identical for every sample — predictions were independent of the input data (verified:
identical outputs for a sine wave and 100× amplified noise; prediction standard
deviation across inputs = 0.0000). This is why earlier versions produced flat
forecasts, and it invalidated all previously published plots.

v0.4.0 fixed the defect (additive embedding, full-trajectory decoder memory,
sequence-aware fusion, residual skip; the fixed model beats seasonal-naive on the
synthetic regression benchmark, MAE ratio 0.59, and on real daily-temperature data,
0.73) and added learning regression tests (`tests/test_learning.py`) — including an
input-sensitivity test — so this class of bug cannot silently recur. The before/after
evidence is committed at `assets/images/apdtflow_fix_proof.png` and
`assets/images/apdtflow_v3_final_proof.png`. We keep this section public on purpose:
the audit that found the bug began with a user's question, and packages get better when
their failures are documented as carefully as their wins.

## 9. References

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural
  Ordinary Differential Equations.* NeurIPS.
- Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). *Latent ODEs for
  Irregularly-Sampled Time Series.* NeurIPS.
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers
  for Interpretable Multi-horizon Time Series Forecasting.* International Journal of
  Forecasting.
- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural
  Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR.
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). *Are Transformers Effective for Time
  Series Forecasting?* AAAI (DLinear).
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random
  World.* Springer.
- Romano, Y., Patterson, E., & Candès, E. (2019). *Conformalized Quantile Regression.*
  NeurIPS.
- Gibbs, I., & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution
  Shift.* NeurIPS.
- *Conformal prediction under non-exchangeability: a survey.* arXiv:2511.13608 (2025).
- *Multi-step split-conformal prediction for time series: a benchmark.*
  arXiv:2601.18509 (2026).
- *Conformal Prediction Beyond Change Points (CPTC).* arXiv:2509.02844, NeurIPS (2025).
- *Linear models remain strong baselines for irregularly-sampled time series.*
  arXiv:2505.00590 (2025).
- Perez-Diaz, A., Loach, J. C., Toutoungi, D. E., & Middleton, L. (2025). *Foundation
  Model Forecasts: Form and Function.* arXiv:2510.19345 (Siemens) — forecast types and
  first-passage/threshold-crossing tasks.
- Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast
  accuracy.* International Journal of Forecasting (MASE).
- Saha, B., & Goebel, K. (2007). *Battery Data Set.* NASA Prognostics Data Repository.
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling
  for Aircraft Engine Run-to-Failure Simulation (C-MAPSS).* PHM 2008.
- Context on time-series foundation models: Ansari et al. (Chronos / Chronos-2), Das
  et al. (TimesFM), Woo et al. (Moirai / Moirai-2).
