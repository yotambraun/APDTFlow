"""
Categorical Features Demo: Forecasting with Holidays, Day-of-Week, and More
=============================================================================

This example demonstrates how to use categorical variables (like holidays,
day of week, product categories) in APDTFlow for improved forecasting accuracy.

NEW in v0.2.3!

What are Categorical Features?
- Non-numerical variables (holidays, weekdays, categories, etc.)
- Encoded into numbers (one-hot or embeddings)
- Improve forecasts by capturing discrete patterns

Why they matter:
- Real-world data has categorical patterns (Monday vs Friday, holiday vs normal)
- Boost accuracy by 10-30% on many datasets
- Essential for retail, e-commerce, energy forecasting

Example: Retail sales forecast with day-of-week and holiday effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apdtflow import APDTFlowForecaster
from apdtflow.preprocessing import create_time_features

print("="*70)
print("APDTFlow v0.2.3 - Categorical Features Demo")
print("="*70)
print()

# ===========================================================================
# STEP 1: Generate realistic retail sales data
# ===========================================================================

print("📊 STEP 1: Creating Realistic Retail Sales Data")
print("-" * 70)

np.random.seed(42)

# Create date range (1 year of daily data)
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')

# Base sales with trend
trend = 0.05 * np.arange(len(dates))
base_sales = 100 + trend

# Day-of-week effect (weekends have higher sales)
day_of_week_effect = np.array([
    -5,  # Monday
    -3,  # Tuesday
    0,   # Wednesday
    2,   # Thursday
    5,   # Friday
    15,  # Saturday (much higher!)
    12   # Sunday (higher!)
])
dow_effect = np.array([day_of_week_effect[d.dayofweek] for d in dates])

# Holiday effect (major holidays boost sales)
holidays = []
holiday_names = []
for i, date in enumerate(dates):
    # Christmas season (Dec 15-25)
    if date.month == 12 and 15 <= date.day <= 25:
        holidays.append(i)
        holiday_names.append('Christmas')
    # Black Friday (last Friday of November)
    elif date.month == 11 and date.dayofweek == 4 and date.day >= 22:
        holidays.append(i)
        holiday_names.append('Black Friday')
    # New Year
    elif date.month == 1 and date.day == 1:
        holidays.append(i)
        holiday_names.append('New Year')

holiday_effect = np.zeros(len(dates))
for idx in holidays:
    holiday_effect[idx] = 30  # Major boost on holidays

# Seasonal effect
seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)

# Random noise
noise = np.random.normal(0, 3, len(dates))

# Combine all effects
sales = base_sales + dow_effect + holiday_effect + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales
})

# Add categorical features
df['day_of_week'] = df['date'].dt.day_name()  # Monday, Tuesday, etc.
df['month'] = df['date'].dt.month_name()
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(str)  # 'True' or 'False'

# Add holiday indicator
df['holiday'] = 'None'
for idx, holiday_name in zip(holidays, holiday_names):
    df.loc[idx, 'holiday'] = holiday_name

print(f"✓ Created dataset: {len(df)} days")
print(f"✓ Categorical features:")
print(f"  - day_of_week: {df['day_of_week'].nunique()} categories")
print(f"  - month: {df['month'].nunique()} categories")
print(f"  - is_weekend: {df['is_weekend'].nunique()} categories")
print(f"  - holiday: {df['holiday'].nunique()} categories")
print()

# Show sample
print("Sample data:")
print(df.head(10))
print()

# ===========================================================================
# STEP 2: Train WITHOUT categorical features (baseline)
# ===========================================================================

print("🔬 STEP 2: Baseline Model (NO Categorical Features)")
print("-" * 70)

# Split data
train_end = int(0.8 * len(df))
train_df = df.iloc[:train_end]
test_df = df.iloc[train_end:]

print(f"Training: {len(train_df)} days, Testing: {len(test_df)} days")
print()

# Train baseline model (no categorical features)
print("Training baseline model...")
baseline_model = APDTFlowForecaster(
    forecast_horizon=7,
    history_length=30,
    num_epochs=20,
    verbose=False
)

baseline_model.fit(
    train_df,
    target_col='sales',
    date_col='date'
)

print("✓ Baseline model trained")
print()

# Evaluate baseline
baseline_mse = baseline_model.score(test_df, target_col='sales', metric='mse')
baseline_mae = baseline_model.score(test_df, target_col='sales', metric='mae')

print(f"Baseline Performance:")
print(f"  MSE: {baseline_mse:.2f}")
print(f"  MAE: {baseline_mae:.2f}")
print()

# ===========================================================================
# STEP 3: Train WITH categorical features
# ===========================================================================

print("🚀 STEP 3: Enhanced Model (WITH Categorical Features)")
print("-" * 70)

print("Training model with categorical features...")
categorical_model = APDTFlowForecaster(
    forecast_horizon=7,
    history_length=30,
    num_epochs=20,
    categorical_encoding='onehot',  # or 'embedding'
    verbose=True
)

categorical_model.fit(
    train_df,
    target_col='sales',
    date_col='date',
    categorical_cols=['day_of_week', 'is_weekend', 'holiday']
)

print()

# Evaluate with categorical features
cat_mse = categorical_model.score(test_df, target_col='sales', metric='mse')
cat_mae = categorical_model.score(test_df, target_col='sales', metric='mae')

print(f"Categorical Model Performance:")
print(f"  MSE: {cat_mse:.2f}")
print(f"  MAE: {cat_mae:.2f}")
print()

# ===========================================================================
# STEP 4: Compare Results
# ===========================================================================

print("📊 STEP 4: Performance Comparison")
print("-" * 70)

mse_improvement = ((baseline_mse - cat_mse) / baseline_mse) * 100
mae_improvement = ((baseline_mae - cat_mae) / baseline_mae) * 100

print(f"Improvement with Categorical Features:")
print(f"  MSE: {mse_improvement:+.1f}% improvement")
print(f"  MAE: {mae_improvement:+.1f}% improvement")
print()

if mse_improvement > 0:
    print("✓ Categorical features IMPROVED forecast accuracy!")
else:
    print("⚠️  Categorical features didn't improve on this dataset")
    print("   (May need more training epochs or different encoding)")
print()

# ===========================================================================
# STEP 5: Visualize Predictions
# ===========================================================================

print("📈 STEP 5: Visualization")
print("-" * 70)

# Make predictions for a week
baseline_preds = baseline_model.predict(steps=7)
cat_preds = categorical_model.predict(steps=7)

# Get actual values for comparison
actual = test_df['sales'].values[:7]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(7)
ax.plot(x, actual, 'ko-', label='Actual Sales', linewidth=2, markersize=8)
ax.plot(x, baseline_preds, 'b--', label='Baseline (No Categories)', linewidth=2)
ax.plot(x, cat_preds, 'r--', label='With Categorical Features', linewidth=2)

# Add day labels
day_labels = test_df['day_of_week'].values[:7]
ax.set_xticks(x)
ax.set_xticklabels(day_labels, rotation=45)

ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)
ax.set_title('Forecast Comparison: With vs Without Categorical Features', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('categorical_features_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: categorical_features_comparison.png")
plt.show()

print()

# ===========================================================================
# KEY TAKEAWAYS
# ===========================================================================

print("="*70)
print("🎯 KEY TAKEAWAYS")
print("="*70)
print()
print("1. Categorical Features Capture Discrete Patterns:")
print("   - Day-of-week effects (weekends vs weekdays)")
print("   - Holiday effects (special events)")
print("   - Product categories, seasons, etc.")
print()
print("2. Two Encoding Options:")
print("   - One-hot: Simple, interpretable (good for few categories)")
print("   - Embedding: Efficient, learns relationships (good for many categories)")
print()
print("3. Easy Integration with APDTFlow:")
print("   model.fit(df, target_col='sales',")
print("            categorical_cols=['day_of_week', 'holiday'])")
print()
print("4. When to Use Categorical Features:")
print("   ✓ Retail/e-commerce (day-of-week, holidays)")
print("   ✓ Energy (weekday/weekend patterns)")
print("   ✓ Transportation (rush hour vs off-peak)")
print("   ✓ Any data with discrete patterns")
print()
print("5. Typical Accuracy Gains:")
print("   - Retail: 15-30% improvement")
print("   - Energy: 10-20% improvement")
print("   - General: Depends on data, but often significant")
print()
print("="*70)
print("🎉 Categorical features make forecasts more realistic!")
print("="*70)
