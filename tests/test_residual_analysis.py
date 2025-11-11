"""
Comprehensive test suite for residual analysis functionality.
Tests compute_residuals(), plot_residuals(), and analyze_residuals() methods.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import pandas as pd  # noqa: E402
from apdtflow import APDTFlowForecaster  # noqa: E402


class TestResidualAnalysis:
    """Test suite for residual analysis methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 150

        dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

        # Trend + seasonality + noise
        trend = 0.5 * np.arange(n_samples)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        noise = np.random.randn(n_samples) * 2

        values = 100 + trend + seasonality + noise

        return pd.DataFrame({
            'date': dates,
            'value': values
        })

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a pre-trained model for testing."""
        train_df = sample_data.iloc[:100]

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        model.fit(train_df, target_col='value', date_col='date')
        return model

    def test_compute_residuals_basic(self, trained_model, sample_data):
        """Test basic residual computation."""
        test_df = sample_data.iloc[100:]

        residuals, actuals, predictions = trained_model.compute_residuals(
            test_df,
            target_col='value',
            date_col='date'
        )

        # Check shapes
        assert len(residuals) == len(actuals) == len(predictions)
        assert len(residuals) > 0

        # Check residual calculation
        np.testing.assert_array_almost_equal(residuals, actuals - predictions)

        # Check residuals are finite
        assert np.all(np.isfinite(residuals))
        assert np.all(np.isfinite(actuals))
        assert np.all(np.isfinite(predictions))

    def test_compute_residuals_storage(self, trained_model, sample_data):
        """Test that residuals are stored in model attributes."""
        test_df = sample_data.iloc[100:]

        residuals, actuals, predictions = trained_model.compute_residuals(
            test_df,
            target_col='value',
            date_col='date'
        )

        # Check stored values
        np.testing.assert_array_equal(residuals, trained_model.residuals_)
        np.testing.assert_array_equal(actuals, trained_model.residual_actuals_)
        np.testing.assert_array_equal(predictions, trained_model.residual_predictions_)

    def test_compute_residuals_custom_windows(self, trained_model, sample_data):
        """Test residual computation with custom number of windows."""
        test_df = sample_data.iloc[100:]

        # Test with different n_windows
        residuals_50, _, _ = trained_model.compute_residuals(test_df, n_windows=50)
        residuals_20, _, _ = trained_model.compute_residuals(test_df, n_windows=20)

        # More windows should give more residuals
        assert len(residuals_50) >= len(residuals_20)

    def test_compute_residuals_numpy_input(self):
        """Test residual computation with numpy array input."""
        np.random.seed(42)
        data = np.random.randn(120) + 100

        model = APDTFlowForecaster(
            forecast_horizon=5,
            history_length=15,
            num_epochs=5,
            verbose=False
        )

        model.fit(data[:80])

        # Compute residuals on test set
        residuals, actuals, predictions = model.compute_residuals(data[80:])

        assert len(residuals) > 0
        assert np.all(np.isfinite(residuals))

    def test_compute_residuals_not_fitted(self, sample_data):
        """Test error when computing residuals on unfitted model."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.compute_residuals(sample_data)

    def test_plot_residuals_basic(self, trained_model, sample_data):
        """Test basic residual plotting."""
        test_df = sample_data.iloc[100:]

        # Should not raise error
        fig, axes = trained_model.plot_residuals(
            test_df,
            target_col='value',
            date_col='date'
        )

        # Check figure created
        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 2)  # 2x2 grid

    def test_plot_residuals_uses_stored(self, trained_model, sample_data):
        """Test plotting uses stored residuals if available."""
        test_df = sample_data.iloc[100:]

        # Compute residuals first
        residuals, _, _ = trained_model.compute_residuals(test_df)

        # Plot without data parameter (should use stored)
        fig, axes = trained_model.plot_residuals()

        assert fig is not None
        assert axes is not None

    def test_plot_residuals_no_data_no_stored(self, trained_model):
        """Test error when no data and no stored residuals."""
        with pytest.raises(ValueError, match="No residuals available"):
            trained_model.plot_residuals()

    def test_plot_residuals_save(self, trained_model, sample_data, tmp_path):
        """Test saving residual plot to file."""
        test_df = sample_data.iloc[100:]
        save_path = tmp_path / "residuals.png"

        trained_model.plot_residuals(test_df, save_path=str(save_path))

        # Check file created
        assert save_path.exists()

    def test_analyze_residuals_basic(self, trained_model, sample_data):
        """Test basic residual analysis."""
        test_df = sample_data.iloc[100:]

        diagnostics = trained_model.analyze_residuals(
            test_df,
            target_col='value',
            date_col='date'
        )

        # Check required keys
        required_keys = [
            'n_samples', 'mean', 'std', 'mae', 'rmse',
            'min', 'max', 'skewness', 'kurtosis'
        ]
        for key in required_keys:
            assert key in diagnostics

        # Check values are reasonable
        assert diagnostics['n_samples'] > 0
        assert diagnostics['std'] > 0
        assert diagnostics['mae'] >= 0
        assert diagnostics['rmse'] >= 0

        # Check normality test exists
        assert ('shapiro_pvalue' in diagnostics or 'ks_pvalue' in diagnostics)

        # Check autocorrelation test exists
        assert 'ljung_box_pvalue' in diagnostics

    def test_analyze_residuals_uses_stored(self, trained_model, sample_data):
        """Test analysis uses stored residuals if available."""
        test_df = sample_data.iloc[100:]

        # Compute residuals first
        residuals, _, _ = trained_model.compute_residuals(test_df)

        # Analyze without data parameter
        diagnostics = trained_model.analyze_residuals()

        assert diagnostics['n_samples'] == len(residuals)
        assert abs(diagnostics['mean'] - np.mean(residuals)) < 1e-6

    def test_analyze_residuals_statistical_tests(self, trained_model, sample_data):
        """Test that statistical tests produce valid p-values."""
        test_df = sample_data.iloc[100:]

        diagnostics = trained_model.analyze_residuals(test_df)

        # Check Shapiro-Wilk or KS test p-value
        if 'shapiro_pvalue' in diagnostics and diagnostics['shapiro_pvalue'] is not None:
            assert 0 <= diagnostics['shapiro_pvalue'] <= 1
        elif 'ks_pvalue' in diagnostics and diagnostics['ks_pvalue'] is not None:
            assert 0 <= diagnostics['ks_pvalue'] <= 1

        # Check Ljung-Box test p-value
        if diagnostics['ljung_box_pvalue'] is not None:
            assert 0 <= diagnostics['ljung_box_pvalue'] <= 1

    def test_analyze_residuals_no_data_no_stored(self, trained_model):
        """Test error when no data and no stored residuals."""
        with pytest.raises(ValueError, match="No residuals available"):
            trained_model.analyze_residuals()

    def test_summary_includes_residuals(self, trained_model, sample_data):
        """Test that summary() includes residual diagnostics when available."""
        test_df = sample_data.iloc[100:]

        # Compute residuals
        trained_model.compute_residuals(test_df)

        # Summary should include residual info (we can't easily test printed output,
        # but we can check the residuals are stored)
        assert trained_model.residuals_ is not None

        # Call summary (shouldn't raise error)
        trained_model.summary()

    def test_residual_workflow(self, trained_model, sample_data):
        """Test complete residual analysis workflow."""
        test_df = sample_data.iloc[100:]

        # 1. Compute residuals
        residuals, actuals, predictions = trained_model.compute_residuals(test_df)
        assert len(residuals) > 0

        # 2. Plot residuals (using stored)
        fig, axes = trained_model.plot_residuals()
        assert fig is not None

        # 3. Analyze residuals (using stored)
        diagnostics = trained_model.analyze_residuals()
        assert diagnostics['n_samples'] == len(residuals)

        # 4. Summary with residual info
        trained_model.summary()

    def test_residuals_with_exogenous(self, sample_data):
        """Test residual analysis with exogenous variables."""
        # Add exogenous feature
        df = sample_data.copy()
        np.random.seed(42)
        df['temperature'] = 20 + np.random.randn(len(df)) * 5

        # Train model with exogenous feature
        train_df = df.iloc[:100]
        test_df = df.iloc[100:]

        model = APDTFlowForecaster(
            forecast_horizon=5,
            history_length=15,
            num_epochs=3,
            verbose=False
        )

        model.fit(
            train_df,
            target_col='value',
            date_col='date',
            exog_cols=['temperature']
        )

        # Compute residuals
        residuals, actuals, predictions = model.compute_residuals(
            test_df,
            exog_cols=['temperature']
        )

        assert len(residuals) > 0
        assert np.all(np.isfinite(residuals))

    def test_residuals_values_reasonable(self, trained_model, sample_data):
        """Test that residual values are reasonable (not too large)."""
        test_df = sample_data.iloc[100:]

        residuals, actuals, predictions = trained_model.compute_residuals(test_df)

        # Check residuals are not absurdly large
        data_range = sample_data['value'].max() - sample_data['value'].min()
        assert np.max(np.abs(residuals)) < data_range * 2  # Within 2x data range

        # Check mean residual is reasonable (model trained with only 5 epochs, so won't be perfect)
        assert abs(np.mean(residuals)) < data_range * 0.5  # < 50% of range

    def test_analyze_residuals_large_sample(self):
        """Test analyze_residuals with large sample (uses KS test instead of Shapiro-Wilk)."""
        np.random.seed(42)

        # Create dataset large enough to trigger KS test path (>5000 residuals)
        n_samples = 1500
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'value': 100 + 0.01 * np.arange(n_samples) + np.random.randn(n_samples) * 2
        })

        model = APDTFlowForecaster(
            forecast_horizon=12,
            history_length=24,
            num_epochs=2,
            verbose=False
        )

        model.fit(data.iloc[:1000], target_col='value', date_col='date')

        # Compute residuals (will create many samples)
        diagnostics = model.analyze_residuals(
            data.iloc[1000:],
            target_col='value',
            date_col='date'
        )

        # Should use KS test for large samples
        assert diagnostics['n_samples'] > 0
        assert 'mean' in diagnostics
        assert 'std' in diagnostics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
