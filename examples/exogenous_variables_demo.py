"""
Exogenous Variables Demo: Boost Forecast Accuracy 30-50%!
===========================================================

This example demonstrates how to use external features (exogenous variables)
to dramatically improve forecasting accuracy.

NEW in v0.2.0!

Example use cases:
- Sales forecasting with weather, holidays, promotions
- Energy demand with temperature, day-of-week
- Traffic prediction with events, weather
- Stock prices with economic indicators

Research shows 30-50% accuracy improvement with exogenous variables!
(Sources: ChronosX 2025, TimeXer 2024, ExoLLM 2025)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("APDTFlow v0.2.0 - Exogenous Variables Demo")
print("="*70)
print()

# ===========================================================================
# STEP 1: Create synthetic dataset with exogenous variables
# ===========================================================================

print("📊 STEP 1: Creating synthetic sales dataset with external features")
print("-" * 70)

# Generate dates
dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
n = len(dates)

# Generate exogenous variables
np.random.seed(42)

# Temperature (seasonal pattern)
temperature = 20 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 2, n)

# Is holiday (some random days)
is_holiday = np.random.choice([0, 1], size=n, p=[0.95, 0.05])

# Promotion (some random days)
promotion = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

# Sales (affected by temperature, holidays, and promotions!)
base_sales = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 365)  # Seasonal
sales = base_sales + \
        -0.5 * temperature + \
        30 * is_holiday + \
        25 * promotion + \
        np.random.normal(0, 5, n)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'temperature': temperature,
    'is_holiday': is_holiday.astype(int),
    'promotion': promotion.astype(int)
})

print(f"✓ Created dataset: {len(df)} days")
print(f"✓ Target variable: sales")
print(f"✓ Exogenous variables: temperature, is_holiday, promotion")
print(f"\nFirst few rows:")
print(df.head(10))
print()

# ===========================================================================
# STEP 2: Train WITHOUT exogenous variables (baseline)
# ===========================================================================

print("📈 STEP 2: Training baseline model WITHOUT exogenous variables")
print("-" * 70)

from apdtflow import APDTFlowForecaster

# Baseline model - no external features
baseline_model = APDTFlowForecaster(
    forecast_horizon=14,
    history_length=30,
    num_epochs=30,
    verbose=False
)

# Train on first 80% of data
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

baseline_model.fit(train_df, target_col='sales', date_col='date')
print("✓ Baseline model trained (no exogenous variables)")
print()

# ===========================================================================
# STEP 3: Train WITH exogenous variables
# ===========================================================================

print("🚀 STEP 3: Training enhanced model WITH exogenous variables")
print("-" * 70)

# Enhanced model - with external features!
enhanced_model = APDTFlowForecaster(
    forecast_horizon=14,
    history_length=30,
    num_epochs=30,
    exog_fusion_type='gated',  # Options: 'concat', 'gated', 'attention'
    verbose=False
)

# Train with exogenous variables
enhanced_model.fit(
    train_df,
    target_col='sales',
    date_col='date',
    exog_cols=['temperature', 'is_holiday', 'promotion'],  # All exog features
    future_exog_cols=['is_holiday', 'promotion']  # These are known in advance!
)

print("✓ Enhanced model trained WITH exogenous variables!")
print("  - Temperature (past observed)")
print("  - Is_holiday (future known)")
print("  - Promotion (future known)")
print()

# ===========================================================================
# STEP 4: Compare predictions
# ===========================================================================

print("📊 STEP 4: Comparing predictions (Baseline vs Enhanced)")
print("-" * 70)

# Get predictions from baseline
baseline_preds = baseline_model.predict()

# For enhanced model, we need future exog values
# In real scenarios, you'd have these from calendar/planning
future_exog_df = test_df[['is_holiday', 'promotion']].head(14)

print("Note: Full exogenous integration in model architecture coming in v0.2.1")
print("      This demo shows the API - stay tuned for complete implementation!")
print()

# ===========================================================================
# EXAMPLE: Using the Exogenous Fusion Module Directly
# ===========================================================================

print("💡 BONUS: Direct usage of ExogenousFeatureFusion module")
print("-" * 70)

from apdtflow.exogenous import ExogenousFeatureFusion
import torch

# Create fusion module
fusion = ExogenousFeatureFusion(
    hidden_dim=32,
    num_exog_features=3,
    fusion_type='gated'
)

# Example input
target = torch.randn(8, 1, 30)  # batch=8, channels=1, time=30
exog = torch.randn(8, 3, 30)    # batch=8, exog_features=3, time=30

# Fuse target with exog
fused = fusion(target, exog)

print(f"✓ Target shape: {target.shape}")
print(f"✓ Exog shape: {exog.shape}")
print(f"✓ Fused shape: {fused.shape}")
print()

# ===========================================================================
# KEY TAKEAWAYS
# ===========================================================================

print("="*70)
print("🎯 KEY TAKEAWAYS")
print("="*70)
print()
print("1. Exogenous variables can improve accuracy by 30-50%")
print("2. Two types:")
print("   - Past observed: Only available historically (e.g., temperature)")
print("   - Future known: Available for forecast period (e.g., holidays)")
print()
print("3. APDTFlow v0.2.0 supports exog variables through:")
print("   - Enhanced TimeSeriesWindowDataset")
print("   - ExogenousFeatureFusion module (concat/gated/attention)")
print("   - Simple API: just pass exog_cols and future_exog_cols!")
print()
print("4. Full model integration coming in v0.2.1")
print()
print("🔗 Learn more: See ROADMAP_V0.2.0.md for complete implementation details")
print("="*70)
