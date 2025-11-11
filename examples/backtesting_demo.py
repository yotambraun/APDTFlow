"""
Backtesting Demo for APDTFlow
==============================

This example demonstrates the historical_forecasts() method for
backtesting time series models.

New in v0.2.3:
- historical_forecasts() method for rolling window backtesting
- Support for fixed model and retrain modes
- Industry-standard metrics (MASE, sMAPE) for evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apdtflow import APDTFlowForecaster


def generate_sample_data(n_samples=200):
    """Generate sample time series data with trend and seasonality."""
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # Trend
    trend = 0.5 * np.arange(n_samples)

    # Weekly seasonality
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

    # Noise
    np.random.seed(42)
    noise = np.random.randn(n_samples) * 2

    # Combine
    values = 100 + trend + seasonality + noise

    return pd.DataFrame({
        'date': dates,
        'value': values
    })


def example_1_basic_backtesting():
    """Example 1: Basic backtesting with fixed model."""
    print("=" * 70)
    print("Example 1: Basic Backtesting (Fixed Model)")
    print("=" * 70)

    # Generate data
    df = generate_sample_data(200)

    # Train model on first 80% of data
    train_df = df.iloc[:160]

    model = APDTFlowForecaster(
        forecast_horizon=7,
        history_length=30,
        num_epochs=20,
        verbose=False
    )

    print("\n1. Training model on first 160 samples...")
    model.fit(train_df, target_col='value', date_col='date')

    # Backtest on remaining data with fixed model
    print("\n2. Running backtesting (fixed model, no retraining)...")
    backtest_results = model.historical_forecasts(
        data=df,
        target_col='value',
        date_col='date',
        start=0.8,  # Start at 80% of data
        forecast_horizon=7,
        stride=7,  # Make forecast every 7 days
        retrain=False,  # Use fixed model
        metrics=['MSE', 'MAE', 'MASE', 'sMAPE']
    )

    print("\n3. Backtest results preview:")
    print(backtest_results.head(10))

    print(f"\n4. Total forecasts made: {len(backtest_results)}")
    print(f"   Average MAE: {backtest_results['error_MAE'].mean():.2f}")
    print(f"   Average MASE: {backtest_results['error_MASE'].mean():.2f}")
    print(f"   Average sMAPE: {backtest_results['error_sMAPE'].mean():.2f}%")

    return backtest_results, df


def example_2_retrain_mode():
    """Example 2: Backtesting with retraining at each step."""
    print("\n\n" + "=" * 70)
    print("Example 2: Backtesting with Retraining")
    print("=" * 70)

    # Generate data
    df = generate_sample_data(150)

    # Initial model (will be retrained at each fold)
    model = APDTFlowForecaster(
        forecast_horizon=7,
        history_length=30,
        num_epochs=10,  # Fewer epochs for faster retraining
        verbose=False
    )

    # Initial fit
    print("\n1. Initial model training...")
    train_df = df.iloc[:100]
    model.fit(train_df, target_col='value', date_col='date')

    # Backtest with retraining
    print("\n2. Running backtesting with retraining at each fold...")
    print("   (This will take longer as model is retrained at each step)")

    backtest_results = model.historical_forecasts(
        data=df,
        target_col='value',
        date_col='date',
        start=100,  # Start at index 100
        forecast_horizon=7,
        stride=7,
        retrain=True,  # Retrain at each step
        metrics=['MSE', 'MAE', 'MASE']
    )

    print("\n3. Backtest results with retraining:")
    print(backtest_results.head(10))

    print(f"\n4. Performance with retraining:")
    print(f"   Average MAE: {backtest_results['error_MAE'].mean():.2f}")
    print(f"   Average MASE: {backtest_results['error_MASE'].mean():.2f}")

    return backtest_results


def example_3_different_horizons():
    """Example 3: Compare different forecast horizons."""
    print("\n\n" + "=" * 70)
    print("Example 3: Compare Different Forecast Horizons")
    print("=" * 70)

    df = generate_sample_data(200)
    train_df = df.iloc[:150]

    results = {}

    for horizon in [3, 7, 14]:
        print(f"\n--- Testing horizon = {horizon} days ---")

        model = APDTFlowForecaster(
            forecast_horizon=horizon,
            history_length=30,
            num_epochs=15,
            verbose=False
        )

        model.fit(train_df, target_col='value', date_col='date')

        backtest = model.historical_forecasts(
            data=df,
            target_col='value',
            date_col='date',
            start=0.75,
            forecast_horizon=horizon,
            stride=horizon,
            retrain=False,
            metrics=['MAE', 'MASE', 'sMAPE']
        )

        results[horizon] = {
            'MAE': backtest['error_MAE'].mean(),
            'MASE': backtest['error_MASE'].mean(),
            'sMAPE': backtest['error_sMAPE'].mean()
        }

        print(f"  MAE: {results[horizon]['MAE']:.2f}")
        print(f"  MASE: {results[horizon]['MASE']:.2f}")
        print(f"  sMAPE: {results[horizon]['sMAPE']:.2f}%")

    print("\n" + "=" * 70)
    print("Summary: Forecast Horizon Comparison")
    print("=" * 70)
    for horizon, metrics in results.items():
        print(f"\nHorizon {horizon} days:")
        print(f"  MAE:   {metrics['MAE']:.2f}")
        print(f"  MASE:  {metrics['MASE']:.2f}")
        print(f"  sMAPE: {metrics['sMAPE']:.2f}%")

    return results


def example_4_visualization():
    """Example 4: Visualize backtest results."""
    print("\n\n" + "=" * 70)
    print("Example 4: Visualize Backtest Results")
    print("=" * 70)

    # Generate data
    df = generate_sample_data(200)
    train_df = df.iloc[:140]

    # Train and backtest
    model = APDTFlowForecaster(
        forecast_horizon=7,
        history_length=30,
        num_epochs=20,
        verbose=False
    )

    print("\n1. Training model...")
    model.fit(train_df, target_col='value', date_col='date')

    print("2. Running backtesting...")
    backtest_results = model.historical_forecasts(
        data=df,
        target_col='value',
        date_col='date',
        start=0.7,
        forecast_horizon=7,
        stride=7,
        retrain=False,
        metrics=['MAE', 'MASE']
    )

    print("3. Creating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(df['date'], df['value'], 'b-', label='Actual', alpha=0.7)
    ax1.scatter(backtest_results['timestamp'], backtest_results['predicted'],
                color='red', label='Backtest Predictions', alpha=0.6, s=30)
    ax1.axvline(x=df['date'].iloc[140], color='green', linestyle='--',
                label='Training Cutoff', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_title('Backtesting: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Forecast Errors Over Time
    ax2 = axes[1]
    ax2.plot(backtest_results['timestamp'], backtest_results['error_MAE'],
             'o-', color='orange', label='MAE', alpha=0.7)
    ax2.set_xlabel('Forecast Date')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Forecast Error Over Time (MAE)')
    ax2.axhline(y=backtest_results['error_MAE'].mean(), color='red',
                linestyle='--', label=f"Mean MAE: {backtest_results['error_MAE'].mean():.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals Distribution
    ax3 = axes[2]
    residuals = backtest_results['actual'] - backtest_results['predicted']
    ax3.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Residual (Actual - Predicted)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Forecast Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('backtesting_results.png', dpi=150, bbox_inches='tight')
    print("\n4. Plot saved as 'backtesting_results.png'")

    return fig, backtest_results


def example_5_with_exogenous_features():
    """Example 5: Backtesting with exogenous variables."""
    print("\n\n" + "=" * 70)
    print("Example 5: Backtesting with Exogenous Features")
    print("=" * 70)

    # Generate data with exogenous features
    n_samples = 180
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    np.random.seed(42)
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.randn(n_samples) * 2
    holiday = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

    # Sales influenced by temperature and holidays
    base_sales = 100 + 0.3 * np.arange(n_samples)
    temp_effect = 2 * temperature
    holiday_effect = 50 * holiday
    noise = np.random.randn(n_samples) * 5

    df = pd.DataFrame({
        'date': dates,
        'sales': base_sales + temp_effect + holiday_effect + noise,
        'temperature': temperature,
        'is_holiday': holiday
    })

    # Train model
    train_df = df.iloc[:120]

    model = APDTFlowForecaster(
        forecast_horizon=7,
        history_length=30,
        num_epochs=20,
        verbose=False
    )

    print("\n1. Training model with exogenous features (temperature, is_holiday)...")
    model.fit(
        train_df,
        target_col='sales',
        date_col='date',
        exog_cols=['temperature', 'is_holiday']
    )

    print("\n2. Running backtesting with exogenous features...")
    backtest_results = model.historical_forecasts(
        data=df,
        target_col='sales',
        date_col='date',
        start=0.67,  # Start at 67% of data
        forecast_horizon=7,
        stride=7,
        retrain=False,
        metrics=['MAE', 'MASE', 'sMAPE'],
        exog_cols=['temperature', 'is_holiday']
    )

    print("\n3. Backtest results with exogenous features:")
    print(backtest_results.head(10))

    print(f"\n4. Performance:")
    print(f"   Average MAE: {backtest_results['error_MAE'].mean():.2f}")
    print(f"   Average MASE: {backtest_results['error_MASE'].mean():.2f}")
    print(f"   Average sMAPE: {backtest_results['error_sMAPE'].mean():.2f}%")

    return backtest_results


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("APDTFlow Backtesting Demo")
    print("=" * 70)
    print("\nThis demo shows how to use historical_forecasts() for robust")
    print("model evaluation using rolling window backtesting.")
    print("\n")

    # Run all examples
    try:
        # Example 1: Basic backtesting
        backtest_results_1, df_1 = example_1_basic_backtesting()

        # Example 2: Retraining mode
        backtest_results_2 = example_2_retrain_mode()

        # Example 3: Different horizons
        horizon_comparison = example_3_different_horizons()

        # Example 4: Visualization
        fig, backtest_viz = example_4_visualization()

        # Example 5: With exogenous features
        backtest_exog = example_5_with_exogenous_features()

        print("\n\n" + "=" * 70)
        print("All Examples Completed Successfully!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("1. historical_forecasts() enables robust rolling window backtesting")
        print("2. Use retrain=False for faster evaluation with fixed model")
        print("3. Use retrain=True for more realistic but slower evaluation")
        print("4. MASE < 1.0 indicates model beats naive forecast")
        print("5. Stride parameter controls frequency of forecasts")
        print("6. Works seamlessly with exogenous features")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
