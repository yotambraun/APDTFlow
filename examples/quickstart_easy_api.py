"""
Quick Start Example: Using APDTFlow's Easy High-Level API
==========================================================

This example demonstrates the new simple API for time series forecasting.
No complex setup needed - just fit and predict!
"""

import pandas as pd
from apdtflow import APDTFlowForecaster

# Load your time series data
df = pd.read_csv('../dataset_examples/Electric_Production.csv', parse_dates=['DATE'])

print("Dataset shape:", df.shape)
print("First few rows:")
print(df.head())
print()

# Create forecaster with simple parameters
model = APDTFlowForecaster(
    forecast_horizon=14,      # Predict 14 steps ahead
    history_length=30,        # Use 30 historical points
    num_epochs=30,            # Quick training
    verbose=True
)

# Fit the model (automatic preprocessing, normalization, etc.)
print("Training model...")
model.fit(df, target_col='IPG2211A2N', date_col='DATE')
print()

# Make predictions
print("Making predictions...")
predictions = model.predict()
print(f"Forecasted next {len(predictions)} values:")
print(predictions)
print()

# Get predictions with uncertainty
print("Getting predictions with uncertainty...")
predictions, uncertainty = model.predict(return_uncertainty=True)
print("Predictions:", predictions)
print("Uncertainty (std):", uncertainty)
print()

# Visualize the forecast
print("Creating visualization...")
model.plot_forecast(
    with_history=100,           # Show last 100 historical points
    show_uncertainty=True,      # Show confidence bands
    save_path='forecast_plot.png'
)

print("\n✅ Done! Check 'forecast_plot.png' for the visualization.")

# Try different model types
print("\n" + "="*60)
print("Bonus: Try different model architectures!")
print("="*60)

for model_type in ['transformer', 'tcn', 'ensemble']:
    print(f"\n🔮 Training {model_type} model...")
    model = APDTFlowForecaster(
        model_type=model_type,
        forecast_horizon=14,
        history_length=30,
        num_epochs=20,
        verbose=False
    )
    model.fit(df, target_col='IPG2211A2N', date_col='DATE')
    preds = model.predict()
    print(f"   {model_type.upper()} forecast (first 5): {preds[:5]}")

print("\n🎉 All done! APDTFlow makes time series forecasting easy!")
