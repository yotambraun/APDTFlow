# Contributing to APDTFlow

Thank you for considering a contribution. APDTFlow's distinguishing feature is not a
model — it is a standard of evidence. This document explains the development setup,
the rules every published claim must pass, and where the project is heading so you
can pick up work that matters.

## Development setup

```bash
git clone https://github.com/yotambraun/APDTFlow.git
cd APDTFlow
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest                      # full suite (includes coverage report)
pytest -m "not slow"        # fast iteration; skips long-running training tests
pytest -m slow              # the training/capability gates (run in CI on push to main)
```

Lint and type-check before opening a PR:

```bash
flake8 apdtflow tests
mypy apdtflow
```

Notes:

- Tests marked `@pytest.mark.slow` train real models and gate learning capability
  (e.g. `tests/test_learning.py` input-sensitivity tests, the cycle-expressiveness
  gate in `tests/test_continuous.py`). Do not weaken these gates; they exist because
  of a shipped defect (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md), Section 8).
- Use `apdtflow.set_seed(0)` in anything that produces a number you intend to quote.

## The shipping rule for `predict_when` domain demos

This is the bar for publishing any domain demonstration of event-time forecasting,
quoted from the evaluation protocol:

> **A `predict_when` domain demo is publishable only if it beats ALL of:**
> **(a) persistence, (b) linear extrapolation, and (c) seasonal-naive/climatology
> wherever the series is seasonal — on both event capture and timing MAE, on
> held-out data, via `experiments/audit_predict_when.py`.**

No exceptions. Thresholds, sensor selection, and normalization statistics are defined
on training units only; evaluation is leave-unit-out (unseen batteries, unseen
engines). If the demo loses to a baseline, the result still has value: document it as
a negative result (see below) rather than shipping it.

## Honesty rules

- **Every published number must be reproducible by a committed script.** If the
  script is not in `experiments/` (or `examples/`), the number does not go in the
  README, docs, or release notes. Numbers belong in
  [docs/experiment_results.md](docs/experiment_results.md), generated from measured
  runs — never hand-edited into prose.
- **Negative results are documented, not deleted.** Ideas that were implemented,
  benchmarked, and rejected go into
  [docs/METHODOLOGY.md, Section 7](docs/METHODOLOGY.md#7-negative-results--what-we-tested-and-do-not-ship)
  with the harness that rejected them. A documented rejection saves the next person a
  month.
- Never claim accuracy superiority over other forecasting libraries or foundation
  models without a committed, reproducible head-to-head benchmark — and even then,
  state the conditions. Capability claims ("one model answers predict/predict_at/
  predict_when") must stay distinct from accuracy claims.

## Versioning and deprecation policy

APDTFlow follows SemVer, with the usual pre-1.0 interpretation:

- **Breaking changes land only in minor version bumps** (0.x → 0.(x+1)) and are
  called out in [CHANGELOG.md](CHANGELOG.md). Patch releases never break APIs.
- **Deprecated names are kept as aliases for one minor release** before removal, e.g.
  `HierarchicalNeuralDynamics` remains an alias of `NeuralDynamics` through the 0.4.x
  series after the v0.4.0 rename.
- Model checkpoints embed the library version; loading refuses checkpoints from
  versions with known-invalid models (≤ 0.3.x) and warns across minor-version
  boundaries.

## Roadmap

Contributions toward any of these are welcome — open an issue first so we can agree
on the audit protocol before you spend training time.

### Benchmark queue (in rough priority order)

All of these go through `experiments/audit_predict_when.py` and the shipping rule
above before any number is published.

1. **C-MAPSS FD003 / FD004** — completes the turbofan family (FD001/FD002 are done;
   see [docs/experiment_results.md](docs/experiment_results.md)).
2. **Battery fleet datasets** — Stanford/MIT–Toyota 124-cell fast-charging dataset,
   CALCE, and Oxford battery degradation: a real test of `predict_when_fleet` at
   fleet scale, beyond the 3-cell NASA set.
3. **PRONOSTIA/FEMTO and XJTU-SY bearings** — requires an RMS/kurtosis
   feature-extraction step in front of the forecaster (raw vibration waveforms are
   not a forecastable health indicator on their own).
4. **NASA milling** — tool-wear thresholds.
5. **PEM fuel cells** — voltage degradation toward end-of-life.
6. **N-CMAPSS** — the frontier milestone: full flight-data trajectories, much larger
   scale.
7. **GIFT-Eval / fev-bench** — for `predict()` transparency: publish where the base
   forecaster honestly lands on community grid benchmarks, win or lose.

### FPT-Bench

The event-timing audit harness — baselines (persistence, linear, climatology),
leave-unit-out splits, event-capture and timing-MAE metrics, time-space calibration
scoring — is general, not APDTFlow-specific. We intend to publish it as a public
**first-passage-time benchmark** so that any forecasting system (including
grid-based and foundation models, via trajectory post-processing) can be scored on
"when will it cross the line?". If you want to help define metrics or contribute
reference baselines, open an issue tagged `fpt-bench`.

### Have degradation or depletion data?

Run the audit — if APDTFlow wins, we feature your domain:

```bash
python experiments/audit_predict_when.py
```

benchmarks `predict_when` against persistence, linear extrapolation, and seasonal
baselines on *your* data, with the same protocol behind every published result. If
APDTFlow beats all baselines on your domain, open a PR with the audit output — we
will feature the domain with your numbers and credit.

## Pull request checklist

- [ ] `pytest -m "not slow"` passes locally; CI runs the full suite.
- [ ] `flake8` and `mypy` are clean.
- [ ] New behavior has tests; new claims have scripts.
- [ ] CHANGELOG.md updated for user-visible changes.
- [ ] Docs updated if the public API changed (`docs/index.md`, `docs/models.md`).
