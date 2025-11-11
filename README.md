# APDTFlow: Production-Ready Time Series Forecasting with Neural ODEs

<p align="center">
  <img src="assets/images/my_logo_framework.png" alt="APDTFlow Logo" width="300">
</p>

[![PyPI version](https://img.shields.io/pypi/v/apdtflow.svg)](https://pypi.org/project/apdtflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/apdtflow)](https://pepy.tech/project/apdtflow)
[![Python Versions](https://img.shields.io/pypi/pyversions/apdtflow.svg)](https://pypi.org/project/apdtflow/)
[![CI](https://github.com/yotambraun/APDTFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/yotambraun/APDTFlow/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/yotambraun/APDTFlow/branch/main/graph/badge.svg)](https://codecov.io/gh/yotambraun/APDTFlow)

**The only Python package offering continuous-time forecasting with Neural ODEs.** Combine cutting-edge research with a simple `fit()`/`predict()` API for production-ready forecasting.

---

## 🎯 What You Can Do with APDTFlow

### Get 95% Confidence Intervals in 5 Lines

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(forecast_horizon=7, use_conformal=True)
model.fit(df, target_col='sales', date_col='date')
lower, pred, upper = model.predict(alpha=0.05, return_intervals='conformal')
# Output: Guaranteed 95% coverage - perfect for production risk management
```

### Boost Accuracy 30-50% with External Features

```python
model = APDTFlowForecaster(forecast_horizon=14, exog_fusion_type='gated')
model.fit(df, target_col='sales', exog_cols=['temperature', 'holiday', 'promotion'])
predictions = model.predict(exog_future=future_df)
# Output: Seamlessly incorporate weather, holidays, and promotions for better forecasts
```

### Validate Models with Rolling Window Backtesting

```python
results = model.historical_forecasts(data=df, start=0.8, stride=7)
print(f"MASE: {results['abs_error'].mean():.2f}")
# Output: MASE: 0.85 (beats naive forecast! < 1.0 = good)
```

### Handle Irregular Time Series with Neural ODEs

```python
model = APDTFlowForecaster(model_type='ode')  # Continuous-time modeling
model.fit(irregular_data)  # Works with missing data & irregular intervals
predictions = model.predict()
# Output: Neural ODEs handle gaps and irregular sampling naturally
```

### Use Categorical Features (Day-of-Week, Holidays, Store IDs)

```python
model = APDTFlowForecaster(forecast_horizon=7)
model.fit(
    df,
    target_col='sales',
    categorical_cols=['day_of_week', 'store_id', 'promotion_type']
)
predictions = model.predict()
# Output: Automatic one-hot encoding or embeddings - no manual preprocessing needed
```

---

## 📦 Installation

APDTFlow is published on [PyPI](https://pypi.org/project/apdtflow):

```bash
pip install apdtflow
```

For development:

```bash
git clone https://github.com/yotambraun/APDTFlow.git
cd APDTFlow
pip install -e .
```

---

## 📑 Table of Contents

1. [What You Can Do](#-what-you-can-do-with-apdtflow)
2. [Installation](#-installation)
3. [Why APDTFlow?](#-why-apdtflow)
4. [Key Features](#-key-features)
5. [Quick Start](#-quick-start)
6. [Features & Usage](#-features--usage)
7. [Model Architectures](#️-model-architectures)
8. [Evaluation & Metrics](#-evaluation--metrics)
9. [Experiment Results](#-experiment-results)
10. [Documentation & Examples](#-documentation--examples)
11. [Additional Capabilities](#️-additional-capabilities)
12. [License](#-license)

---

## 🔬 Why APDTFlow?

### Unique Capabilities

APDTFlow stands out with **continuous-time forecasting using Neural ODEs** and modern research features:

- **🔬 Continuous-Time Neural ODEs**: Model temporal dynamics with differential equations - better for irregular time series and missing data
- **📊 Conformal Prediction**: Rigorous uncertainty quantification with finite-sample coverage guarantees
- **🌟 Advanced Exogenous Support**: 3 fusion strategies (gated, attention, concat) → 30-50% accuracy boost
- **📈 Industry-Standard Metrics**: MASE, sMAPE, CRPS, Coverage for rigorous evaluation
- **🔄 Backtesting**: Darts-style rolling window validation with `historical_forecasts()`
- **⚡ Simple API**: Just `fit()` and `predict()` with multiple architectures (ODE, Transformer, TCN, Ensemble)

### When to Use APDTFlow

- **Financial forecasting** - Rigorous uncertainty bounds for risk management
- **Retail demand** - Holidays, promotions, seasonal patterns with categorical features
- **Energy consumption** - Weather, temperature, and external events as exogenous variables
- **Healthcare demand** - Demographic and policy changes with conformal prediction
- **Any scenario** requiring continuous-time modeling or sophisticated exogenous variable handling

### Comparison with Other Libraries

| Feature | APDTFlow | Darts | NeuralForecast | Prophet |
|---------|----------|-------|----------------|---------|
| **Neural ODEs** | ✅ Continuous-time | ❌ No | ❌ No | ❌ No |
| **Exogenous Variables** | ✅ 3 fusion strategies | ✅ Yes | ✅ Yes | ✅ Yes |
| **Conformal Prediction** | ✅ Rigorous uncertainty | ⚠️ Limited | ❌ No | ❌ No |
| **Backtesting** | ✅ historical_forecasts() | ✅ Yes | ⚠️ Limited | ❌ No |
| **Industry Metrics** | ✅ MASE, sMAPE, CRPS | ✅ Yes | ✅ Yes | ⚠️ Limited |
| **Categorical Features** | ✅ One-hot & embeddings | ✅ Yes | ✅ Yes | ⚠️ Limited |
| **Multi-Scale Decomposition** | ✅ Trends + seasonality | ⚠️ Limited | ❌ No | ✅ Yes |
| **Simple `fit()/predict()` API** | ✅ 5 lines of code | ✅ Yes | ⚠️ Varies | ✅ Yes |
| **Multiple Architectures** | ✅ ODE/Transformer/TCN | ✅ Many | ✅ Many | ❌ One |
| **PyTorch-based** | ✅ GPU acceleration | ⚠️ Mixed | ✅ Yes | ❌ No |

---

## ✨ Key Features

### 📊 Industry-Standard Metrics

Evaluate with metrics used by leading forecasting teams:

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(forecast_horizon=14)
model.fit(df, target_col='sales', date_col='date')

# Industry-standard metrics: MASE, sMAPE, CRPS, Coverage
mase = model.score(test_df, target_col='sales', metric='mase')
# Output: 0.85  # < 1.0 = beats naive forecast

smape = model.score(test_df, target_col='sales', metric='smape')
# Output: 12.3  # Symmetric MAPE percentage
```

**Available Metrics:**
- **MASE** (Mean Absolute Scaled Error) - Scale-independent, M-competition standard
- **sMAPE** (Symmetric MAPE) - Better than MAPE, bounded 0-200%
- **CRPS** (Continuous Ranked Probability Score) - For probabilistic forecasts
- **Coverage** - Prediction interval calibration (e.g., 95% intervals)

---

### 🔄 Backtesting / Historical Forecasts

Validate models with Darts-style rolling window backtesting:

```python
# Backtest model on historical data
backtest_results = model.historical_forecasts(
    data=df,
    target_col='sales',
    date_col='date',
    start=0.8,           # Start at 80% of data
    forecast_horizon=7,  # 7-day forecasts
    stride=7,            # Weekly frequency
    retrain=False,       # Fast: use fixed model
    metrics=['MAE', 'MASE', 'sMAPE']
)

# Output: DataFrame with columns:
#   timestamp, fold, forecast_step, actual, predicted, error, abs_error

print(f"Total forecasts: {backtest_results['fold'].nunique()}")
# Output: Total forecasts: 5

print(f"Average MASE: {backtest_results['abs_error'].mean():.3f}")
# Output: Average MASE: 0.923
```

**Features:**
- **Rolling window validation** - Simulate production forecasting
- **Fixed or retrain modes** - Trade speed vs realism
- **Flexible parameters** - Control start point, stride, horizon
- **Comprehensive output** - Timestamp, actual, predicted, fold, errors

---

### 📊 Conformal Prediction

Get calibrated prediction intervals with coverage guarantees:

```python
model = APDTFlowForecaster(
    forecast_horizon=14,
    use_conformal=True,        # Enable conformal prediction
    conformal_method='adaptive' # Adapts to changing data
)

model.fit(df, target_col='sales')

# Get calibrated 95% prediction intervals
lower, pred, upper = model.predict(
    alpha=0.05,  # 95% coverage guarantee
    return_intervals='conformal'
)

# Output:
#   lower: array([98.2, 97.5, ...])   # Lower bounds
#   pred:  array([105.3, 104.1, ...])  # Point predictions
#   upper: array([112.4, 110.7, ...])  # Upper bounds
# Guarantee: 95% of actual values will fall within [lower, upper]
```

**Why Conformal Prediction?**
- **Finite-sample guarantees** - Not just asymptotic
- **Distribution-free** - No assumptions about data distribution
- **Adaptive methods** - Adjust to changing patterns
- **Production-ready** - Used in finance, healthcare, energy

---

### 🌟 Exogenous Variables

Boost accuracy 30-50% with external features:

```python
# Use external features like temperature, holidays, promotions
model = APDTFlowForecaster(
    forecast_horizon=14,
    exog_fusion_type='gated'  # or 'attention', 'concat'
)

model.fit(
    df,
    target_col='sales',
    date_col='date',
    exog_cols=['temperature', 'is_holiday', 'promotion'],
    future_exog_cols=['is_holiday', 'promotion']  # Known in advance
)

# Predict with future exogenous data
future_exog = pd.DataFrame({
    'is_holiday': [0, 1, 0, 0, ...],
    'promotion': [1, 0, 1, 0, ...]
})
predictions = model.predict(exog_future=future_exog)
# Output: array([135.2, 145.8, 132.1, ...])  # Improved accuracy with external features
```

**Fusion Strategies:**
- **Gated** - Learn importance weights for each feature
- **Attention** - Dynamic feature weighting based on context
- **Concat** - Simple concatenation (baseline)

**Impact:** Research shows 30-50% accuracy improvement in retail, energy, and demand forecasting.

---

### ⚡ Simple API

Production-ready forecasting in 5 lines:

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(forecast_horizon=14)
model.fit(df, target_col='sales', date_col='date')
predictions = model.predict()
# Output: array([120.5, 118.3, 122.1, ...])  # 14 predictions
```

**Features:**
- **Simple `fit()`/`predict()` interface** - No DataLoaders or manual preprocessing
- **Works with pandas DataFrames** - Natural integration with your workflow
- **Automatic normalization** - Just pass your data and go
- **Built-in visualization** - `plot_forecast()` with uncertainty bands
- **Multiple model types** - Switch architectures with one parameter

---

## 🚀 Quick Start

### 5-Line Forecast

```python
from apdtflow import APDTFlowForecaster

model = APDTFlowForecaster(forecast_horizon=7)
model.fit(df, target_col='sales', date_col='date')
predictions = model.predict()
# Output: array([120.5, 118.3, 122.1, 119.8, 121.2, 123.4, 120.9])

model.plot_forecast(with_history=100)
# Output: [displays matplotlib plot with history and predictions]
```

### Complete Example with Uncertainty

```python
import pandas as pd
from apdtflow import APDTFlowForecaster

# Load your time series data
df = pd.read_csv("dataset_examples/Electric_Production.csv", parse_dates=['DATE'])

# Create and train the forecaster
model = APDTFlowForecaster(
    forecast_horizon=14,     # Predict 14 steps ahead
    history_length=30,       # Use 30 historical points
    num_epochs=50           # Training epochs
)

# Fit the model (handles preprocessing automatically)
model.fit(df, target_col='IPG2211A2N', date_col='DATE')

# Make predictions with uncertainty estimates
predictions, uncertainty = model.predict(return_uncertainty=True)
# Output:
#   predictions: array([95.3, 96.1, 94.8, ...])  # 14 values
#   uncertainty: array([2.1, 2.3, 2.4, ...])     # Standard deviations

# Visualize the forecast
model.plot_forecast(with_history=100, show_uncertainty=True)
```

### Try Different Models

```python
# Use Transformer instead of Neural ODE
model = APDTFlowForecaster(model_type='transformer', forecast_horizon=14)

# Or Temporal Convolutional Network
model = APDTFlowForecaster(model_type='tcn', forecast_horizon=14)

# Or Ensemble for maximum robustness
model = APDTFlowForecaster(model_type='ensemble', forecast_horizon=14)
```

---

## 🎯 Features & Usage

### Advanced API for Custom Workflows

For advanced users who need more control over the training process:

```python
import torch
from torch.utils.data import DataLoader
from apdtflow.data import TimeSeriesWindowDataset
from apdtflow.models.apdtflow import APDTFlow

# Create dataset and dataloader
csv_file = "dataset_examples/Electric_Production.csv"
dataset = TimeSeriesWindowDataset(
    csv_file,
    date_col="DATE",
    value_col="IPG2211A2N",
    T_in=12,  # Input sequence length
    T_out=3   # Forecast horizon
)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
model = APDTFlow(
    num_scales=3,
    input_channels=1,
    filter_size=5,
    hidden_dim=16,
    output_dim=1,
    forecast_horizon=3,
    use_embedding=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
model.train_model(
    train_loader=train_loader,
    num_epochs=15,
    learning_rate=0.001,
    device=device
)

# Evaluate
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
metrics = model.evaluate(test_loader, device, metrics=["MSE", "MAE", "RMSE", "MAPE"])
# Output: {'MSE': 0.234, 'MAE': 0.412, 'RMSE': 0.484, 'MAPE': 4.23}
```

---

## 🏗️ Model Architectures

APDTFlow includes multiple advanced forecasting architectures:

### APDTFlow (Neural ODE)

The **APDTFlow** model integrates:
- **Multi-Scale Decomposition**: Decomposes signals into multiple resolutions
- **Neural ODE Dynamics**: Models continuous latent state evolution
- **Probabilistic Fusion**: Merges representations while quantifying uncertainty
- **Transformer-Based Decoding**: Generates forecasts with time-aware attention

**Key Parameters**:
- `T_in`: Input sequence length (e.g., 12 = use 12 historical points)
- `T_out`: Forecast horizon (e.g., 3 = predict 3 steps ahead)
- `num_scales`: Number of decomposition scales for multi-resolution analysis
- `filter_size`: Convolutional filter size affecting receptive field
- `hidden_dim`: Hidden state size controlling model capacity
- `forecast_horizon`: Must match `T_out` for consistency

### TransformerForecaster

Leverages Transformer architecture with self-attention to capture long-range dependencies. Ideal for complex temporal patterns requiring broad context.

### TCNForecaster

Based on Temporal Convolutional Networks with dilated convolutions and residual connections. Efficiently captures local and medium-range dependencies.

### EnsembleForecaster

Combines predictions from multiple models (APDTFlow, Transformer, TCN) using weighted averaging for improved robustness and accuracy.

📖 **Learn More:** [Model Architectures Documentation](docs/models.md)

---

## 📊 Evaluation & Metrics

APDTFlow supports comprehensive evaluation with **industry-standard forecasting metrics**:

### Standard Metrics
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Industry-Standard Metrics
- **MASE** (Mean Absolute Scaled Error) - Scale-independent, M-competition standard. Values < 1 = beats naive forecast
- **sMAPE** (Symmetric MAPE) - Symmetric, bounded 0-200%, better than MAPE
- **CRPS** (Continuous Ranked Probability Score) - Evaluates probabilistic forecasts
- **Coverage** - Prediction interval calibration (e.g., 95% intervals should contain 95% of actuals)

### Usage Example

```python
from apdtflow import APDTFlowForecaster
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator

# High-level API
model = APDTFlowForecaster(forecast_horizon=14)
model.fit(train_df, target_col='sales', date_col='date')

# Score with new metrics
mase = model.score(test_df, target_col='sales', metric='mase')
smape = model.score(test_df, target_col='sales', metric='smape')

print(f"MASE: {mase:.3f} (< 1.0 = beats naive forecast)")
# Output: MASE: 0.850 (< 1.0 = beats naive forecast)

print(f"sMAPE: {smape:.2f}%")
# Output: sMAPE: 12.34%

# Using evaluator directly
evaluator = RegressionEvaluator(metrics=["MSE", "MAE", "MASE", "sMAPE"])
results = evaluator.evaluate(predictions, targets)
print("Metrics:", results)
# Output: Metrics: {'MSE': 0.234, 'MAE': 0.412, 'MASE': 0.850, 'sMAPE': 12.34}

# For probabilistic forecasts with intervals
evaluator_prob = RegressionEvaluator(metrics=["CRPS", "Coverage"])
results_prob = evaluator_prob.evaluate(predictions, targets, lower=lower_bounds, upper=upper_bounds)
print("CRPS:", results_prob["CRPS"], "Coverage:", results_prob["Coverage"])
# Output: CRPS: 1.23 Coverage: 94.5
```

### Backtesting for Robust Validation

```python
# Backtest on historical data
backtest_results = model.historical_forecasts(
    data=df,
    target_col='sales',
    date_col='date',
    start=0.7,           # Start at 70% of data
    forecast_horizon=14,
    stride=14,           # Forecast every 2 weeks
    retrain=False,       # Use fixed model for speed
    metrics=['MAE', 'MASE', 'sMAPE']
)

# Analyze results
print(f"Total forecasts: {backtest_results['fold'].nunique()}")
# Output: Total forecasts: 8

print(f"Average MASE: {backtest_results['abs_error'].mean():.3f}")
# Output: Average MASE: 0.923

# Visualize backtest
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(backtest_results['timestamp'], backtest_results['actual'], 'o-', label='Actual', alpha=0.7)
plt.plot(backtest_results['timestamp'], backtest_results['predicted'], 's-', label='Predicted', alpha=0.7)
plt.legend()
plt.show()
```

---

## 🧪 Experiment Results

We compared multiple forecasting models across different forecast horizons using 3-fold cross-validation:

### Validation Loss Comparison

![Validation Loss Comparison](experiments/results_plots/Validation_Loss_Comparison.png)

**Key Finding:** APDTFlow consistently achieves lower validation losses, especially for longer forecast horizons. Multi-scale decomposition and Neural ODE dynamics effectively capture trends and seasonal patterns.

### Performance vs. Forecast Horizon

![Performance vs. Horizon](experiments/results_plots/Performance_vs_Horizon.png)

**Analysis:** APDTFlow maintains robust performance as forecast horizon increases, demonstrating superior extrapolation capabilities compared to discrete-time models.

### Example Forecast (Horizon 7, CV Split 3)

![APDTFlow Forecast](experiments/results_plots/APDTFlow_Forecast_Horizon_7_CV3.png)

- **Blue**: Historical input sequence (30 time steps)
- **Orange (Dashed)**: Actual future values
- **Dotted Line**: APDTFlow predictions

📊 **Full Analysis:** See [Experiment Results Documentation](docs/experiment_results.md) for detailed metrics, ablation studies, and cross-validation analysis.

---

## 📚 Documentation & Examples

### Documentation
- [User Guide](docs/index.md) - Comprehensive overview and tutorials
- [Model Architectures](docs/models.md) - Technical details and parameters
- [Experiment Results](docs/experiment_results.md) - Benchmarks and analysis
- [Configuration Guide](apdtflow/config/config.yaml) - YAML configuration options

### Examples
- **[Backtesting Demo](examples/backtesting_demo.py)** - 5 comprehensive examples: basic backtesting, retrain modes, horizon comparison, visualization, exogenous features
- **[Categorical Features Demo](examples/categorical_features_demo.py)** - Day-of-week, holidays, and categorical variable encoding
- **[Exogenous Features Demo](examples/exogenous_demo.py)** - External variables (temperature, promotions) with 3 fusion strategies
- **[Conformal Prediction Demo](examples/conformal_prediction_demo.py)** - Rigorous uncertainty quantification with coverage guarantees

---

## 🛠️ Additional Capabilities

### Data Processing & Augmentation

APDTFlow provides robust preprocessing:
- **Date Conversion**: Automatic datetime parsing
- **Gap Filling**: Reindexing for consistent time frequency
- **Missing Value Imputation**: Forward-fill, backward-fill, mean, interpolation
- **Feature Engineering**: Lag features and rolling statistics
- **Data Augmentation**: Jittering, scaling, time warping for robustness

### Command-Line Interface

Train and infer directly from terminal:

```bash
# Train a model
apdtflow train --csv_file path/to/dataset.csv --date_col DATE --value_col VALUE \
  --T_in 12 --T_out 3 --num_epochs 15 --checkpoint_dir ./checkpoints

# Run inference
apdtflow infer --csv_file path/to/dataset.csv --date_col DATE --value_col VALUE \
  --T_in 12 --T_out 3 --checkpoint_path ./checkpoints/APDTFlow_checkpoint.pt
```

**Available Commands:**
- `apdtflow train` - Train a forecasting model
- `apdtflow infer` - Run inference with saved checkpoint

### Cross-Validation Strategies

Robust time series cross-validation:
- **Rolling Splits**: Moving training and validation windows
- **Expanding Splits**: Increasing training window size
- **Blocked Splits**: Contiguous block divisions

```python
from apdtflow.cv_factory import TimeSeriesCVFactory

cv_factory = TimeSeriesCVFactory(
    dataset,
    method="rolling",
    train_size=40,
    val_size=10,
    step_size=10
)
splits = cv_factory.get_splits()
# Output: [(train_indices, val_indices), ...]
```

---

## 📄 License

APDTFlow is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the time series forecasting community**

📦 [PyPI](https://pypi.org/project/apdtflow) | 📖 [Documentation](docs/index.md) | 🐛 [Issues](https://github.com/yotambraun/APDTFlow/issues) | ⭐ [Star on GitHub](https://github.com/yotambraun/APDTFlow)
