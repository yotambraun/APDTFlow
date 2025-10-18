"""
Conformal Prediction Demo: Rigorous Uncertainty Quantification
===============================================================

This example demonstrates conformal prediction for calibrated prediction intervals
with finite-sample coverage guarantees.

NEW in v0.2.0!

What is Conformal Prediction?
- Distribution-free uncertainty quantification
- Finite-sample coverage guarantees
- No assumptions about data distribution
- HOTTEST topic in 2025 time series research (10+ papers this year!)

Why it matters:
- Probabilistic forecasts can be miscalibrated
- Conformal prediction gives GUARANTEED coverage
- Critical for decision-making in business/healthcare/finance

Research: arXiv:2509.02844, arXiv:2503.21251, ICLR 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("APDTFlow v0.2.0 - Conformal Prediction Demo")
print("="*70)
print()

# ===========================================================================
# STEP 1: Generate synthetic data
# ===========================================================================

print("📊 STEP 1: Creating synthetic time series")
print("-" * 70)

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
t = np.arange(len(dates))

# Generate time series with trend + seasonality + noise
trend = 0.05 * t
seasonality = 10 * np.sin(2 * np.pi * t / 30)
noise = np.random.normal(0, 2, len(t))
values = 100 + trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'value': values
})

print(f"✓ Created dataset: {len(df)} days")
print(f"✓ Components: trend + 30-day seasonality + noise")
print()

# ===========================================================================
# STEP 2: Using Split Conformal Prediction
# ===========================================================================

print("🔬 STEP 2: Split Conformal Prediction")
print("-" * 70)

from apdtflow.conformal import SplitConformalPredictor

# Split data: train (60%), calibration (20%), test (20%)
n = len(df)
train_end = int(0.6 * n)
cal_end = int(0.8 * n)

train_data = df.iloc[:train_end]
cal_data = df.iloc[train_end:cal_end]
test_data = df.iloc[cal_end:]

print(f"Data split:")
print(f"  Training: {len(train_data)} samples")
print(f"  Calibration: {len(cal_data)} samples")
print(f"  Test: {len(test_data)} samples")
print()

# Train a simple model (for demo, we'll use a simple predictor)
# In practice, you'd use APDTFlowForecaster

def simple_predictor(X):
    """Simple baseline: predict mean of recent values"""
    # X is array of historical sequences
    return np.mean(X, axis=1, keepdims=True)

# Prepare data for conformal prediction
def prepare_sequences(data, history_len=30):
    values = data['value'].values
    X_list = []
    y_list = []
    for i in range(len(values) - history_len):
        X_list.append(values[i:i+history_len])
        y_list.append(values[i+history_len])
    return np.array(X_list), np.array(y_list).reshape(-1, 1)

X_cal, y_cal = prepare_sequences(cal_data)
X_test, y_test = prepare_sequences(test_data)

# Create conformal predictor
conformal = SplitConformalPredictor(
    predict_fn=simple_predictor,
    alpha=0.05  # 95% coverage
)

# Calibrate
print("Calibrating conformal predictor...")
conformal.calibrate(X_cal, y_cal)
print()

# Predict with conformal intervals
print("Making predictions with conformal intervals...")
lower, preds, upper = conformal.predict(X_test)

print(f"✓ Generated {len(preds)} predictions with 95% conformal intervals")
print()

# Check coverage
diagnostics = conformal.get_coverage_diagnostics(X_test, y_test)
print("📊 Calibration Diagnostics:")
print(f"  Target coverage: {diagnostics['target_coverage']:.1%}")
print(f"  Empirical coverage: {diagnostics['empirical_coverage']:.1%}")
print(f"  Average interval width: {diagnostics['avg_interval_width']:.2f}")
print(f"  Quantile: {diagnostics['quantile']:.4f}")
print()

# ===========================================================================
# STEP 3: Adaptive Conformal Prediction
# ===========================================================================

print("⚡ STEP 3: Adaptive Conformal Prediction (for non-stationary data)")
print("-" * 70)

from apdtflow.conformal import AdaptiveConformalPredictor

# Create adaptive predictor
adaptive_conformal = AdaptiveConformalPredictor(
    predict_fn=simple_predictor,
    alpha=0.05,
    gamma=0.05  # Learning rate for adaptation
)

# Calibrate
adaptive_conformal.calibrate(X_cal, y_cal)
print("✓ Adaptive predictor calibrated")
print()

# Online prediction with adaptation
print("Simulating online predictions with adaptation...")
coverage_track = []

for i in range(min(50, len(X_test))):
    X_i = X_test[i:i+1]
    y_i = y_test[i:i+1]

    # Predict and update
    lower_i, pred_i, upper_i = adaptive_conformal.predict_and_update(X_i, y_i)

    # Track coverage
    covered = (y_i >= lower_i) & (y_i <= upper_i)
    coverage_track.append(covered[0, 0])

adaptation_stats = adaptive_conformal.get_adaptation_stats()
print(f"\n📈 Adaptation Statistics:")
print(f"  Updates: {adaptation_stats['num_updates']}")
print(f"  Initial quantile: {adaptation_stats['initial_quantile']:.4f}")
print(f"  Current quantile: {adaptation_stats['current_quantile']:.4f}")
print(f"  Quantile change: {adaptation_stats['quantile_change']:.4f}")
print(f"  Recent coverage: {adaptation_stats['recent_coverage']:.1%}")
print()

# ===========================================================================
# STEP 4: Visualization
# ===========================================================================

print("📊 STEP 4: Visualizing Conformal Intervals")
print("-" * 70)

from apdtflow.conformal import plot_conformal_intervals

# Plot first 100 predictions
n_plot = min(100, len(y_test))

plot_conformal_intervals(
    y_true=y_test[:n_plot].flatten(),
    y_pred=preds[:n_plot].flatten(),
    lower=lower[:n_plot].flatten(),
    upper=upper[:n_plot].flatten(),
    title="Split Conformal Prediction Intervals",
    figsize=(14, 6)
)

print("✓ Plot displayed")
print()

# ===========================================================================
# STEP 5: Integration with APDTFlowForecaster (Preview)
# ===========================================================================

print("🚀 STEP 5: Integration with APDTFlowForecaster (Preview)")
print("-" * 70)

print("""
In production, you can use conformal prediction with APDTFlowForecaster:

from apdtflow import APDTFlowForecaster

# Enable conformal prediction
model = APDTFlowForecaster(
    forecast_horizon=14,
    use_conformal=True,           # Enable conformal prediction!
    conformal_method='split',     # or 'adaptive'
    calibration_split=0.2         # Use 20% for calibration
)

# Fit (automatic calibration)
model.fit(df, target_col='sales', date_col='date')

# Predict with calibrated intervals
lower, pred, upper = model.predict(return_intervals='conformal')

# Guaranteed 95% coverage! ✓
""")

print("Note: Full integration coming in v0.2.1 - API is ready!")
print()

# ===========================================================================
# KEY TAKEAWAYS
# ===========================================================================

print("="*70)
print("🎯 KEY TAKEAWAYS")
print("="*70)
print()
print("1. Conformal prediction provides GUARANTEED coverage")
print("   - No distribution assumptions")
print("   - Finite-sample guarantees")
print()
print("2. Two methods available:")
print("   - Split: Simple, reliable, fixed intervals")
print("   - Adaptive: Adjusts to non-stationary data")
print()
print("3. Better than probabilistic forecasts:")
print("   - Calibrated by construction")
print("   - No model assumptions needed")
print()
print("4. Use cases:")
print("   - Risk-sensitive decisions (finance, healthcare)")
print("   - Inventory management (safety stocks)")
print("   - Resource planning (confidence bounds)")
print()
print("📚 References:")
print("   - arXiv:2509.02844 - Conformal Prediction for Time Series")
print("   - OpenReview oP7arLOWix - Kernel-based Conformal Prediction")
print("   - arXiv:2503.21251 - Dual-Splitting for Multi-Step")
print()
print("="*70)
print("🎉 Conformal prediction is the future of uncertainty quantification!")
print("="*70)
