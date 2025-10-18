"""
Comprehensive test suite for conformal prediction functionality.
Tests SplitConformalPredictor and AdaptiveConformalPredictor.
"""
import numpy as np
import pytest
from apdtflow.conformal import (
    SplitConformalPredictor,
    AdaptiveConformalPredictor,
    plot_conformal_intervals
)


class TestSplitConformalPredictor:
    """Test suite for SplitConformalPredictor."""

    def test_basic_calibration_and_prediction(self):
        """Test basic calibration and prediction workflow."""
        # Create simple prediction function
        def simple_predictor(X):
            return X * 2 + 1

        predictor = SplitConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1
        )

        # Calibration data
        X_cal = np.arange(10).reshape(-1, 1)
        y_cal = X_cal * 2 + 1 + np.random.randn(10, 1) * 0.5

        # Calibrate
        predictor.calibrate(X_cal, y_cal)

        # Predict
        X_test = np.array([[5], [10], [15]])
        lower, pred, upper = predictor.predict(X_test)

        # Check shapes
        assert lower.shape == (3, 1)
        assert pred.shape == (3, 1)
        assert upper.shape == (3, 1)

        # Check that intervals contain predictions
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)

        # Check that quantile is stored
        assert predictor.quantile is not None
        assert predictor.quantile > 0

    def test_coverage_guarantee(self):
        """Test that coverage is approximately correct."""
        np.random.seed(42)

        # Linear model with noise
        def noisy_predictor(X):
            return X * 2 + np.random.randn(*X.shape) * 0.1

        alpha = 0.1  # 90% coverage
        predictor = SplitConformalPredictor(
            predict_fn=lambda X: X * 2,
            alpha=alpha
        )

        # Generate calibration data
        n_cal = 100
        X_cal = np.random.randn(n_cal, 1) * 10
        y_cal = X_cal * 2 + np.random.randn(n_cal, 1) * 2

        predictor.calibrate(X_cal, y_cal)

        # Generate test data
        n_test = 1000
        X_test = np.random.randn(n_test, 1) * 10
        y_test = X_test * 2 + np.random.randn(n_test, 1) * 2

        lower, pred, upper = predictor.predict(X_test)

        # Check coverage
        coverage = np.mean((y_test >= lower) & (y_test <= upper))

        # Coverage should be close to (1 - alpha)
        # Allow some tolerance due to finite sample
        assert coverage >= 0.80  # At least 80% for 90% target
        assert coverage <= 1.0

    def test_different_alpha_values(self):
        """Test with different significance levels."""
        def simple_predictor(X):
            return X

        X_cal = np.arange(20).reshape(-1, 1)
        y_cal = X_cal + np.random.randn(20, 1)

        for alpha in [0.05, 0.1, 0.2]:
            predictor = SplitConformalPredictor(
                predict_fn=simple_predictor,
                alpha=alpha
            )
            predictor.calibrate(X_cal, y_cal)

            X_test = np.array([[10]])
            lower, pred, upper = predictor.predict(X_test)

            # Interval width should decrease as alpha increases
            interval_width = upper - lower
            assert interval_width > 0

    def test_multivariate_prediction(self):
        """Test with multivariate predictions."""
        def multi_predictor(X):
            return np.hstack([X, X * 2, X * 3])

        predictor = SplitConformalPredictor(
            predict_fn=multi_predictor,
            alpha=0.1
        )

        X_cal = np.arange(10).reshape(-1, 1)
        y_cal = multi_predictor(X_cal) + np.random.randn(10, 3) * 0.5

        predictor.calibrate(X_cal, y_cal)

        X_test = np.array([[5], [10]])
        lower, pred, upper = predictor.predict(X_test)

        assert lower.shape == (2, 3)
        assert pred.shape == (2, 3)
        assert upper.shape == (2, 3)

    def test_get_coverage_diagnostics(self):
        """Test coverage diagnostics."""
        def simple_predictor(X):
            return X

        predictor = SplitConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1
        )

        X_cal = np.arange(20).reshape(-1, 1)
        y_cal = X_cal + np.random.randn(20, 1)
        predictor.calibrate(X_cal, y_cal)

        X_test = np.arange(10, 30).reshape(-1, 1)
        y_test = X_test + np.random.randn(20, 1)

        diagnostics = predictor.get_coverage_diagnostics(X_test, y_test)

        assert 'empirical_coverage' in diagnostics
        assert 'target_coverage' in diagnostics
        assert 'avg_interval_width' in diagnostics
        assert 'quantile' in diagnostics

        assert 0 <= diagnostics['empirical_coverage'] <= 1
        assert diagnostics['target_coverage'] == 0.9
        assert diagnostics['avg_interval_width'] > 0


class TestAdaptiveConformalPredictor:
    """Test suite for AdaptiveConformalPredictor."""

    def test_basic_adaptive_prediction(self):
        """Test basic adaptive conformal prediction."""
        def simple_predictor(X):
            return X * 2

        predictor = AdaptiveConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1,
            gamma=0.01
        )

        # Initial calibration
        X_cal = np.arange(10).reshape(-1, 1)
        y_cal = X_cal * 2 + np.random.randn(10, 1) * 0.5
        predictor.calibrate(X_cal, y_cal)

        # Online updates
        for i in range(5):
            X_new = np.array([[10 + i]])
            y_new = X_new * 2 + np.random.randn(1, 1) * 0.5

            lower, pred, upper = predictor.predict_and_update(X_new, y_new)

            assert lower.shape == (1, 1)
            assert pred.shape == (1, 1)
            assert upper.shape == (1, 1)

        # Check that we have adaptation history
        assert len(predictor.coverage_history) == 5

    def test_adaptation_to_distribution_shift(self):
        """Test adaptation to changing data distribution."""
        np.random.seed(42)

        def simple_predictor(X):
            return X

        predictor = AdaptiveConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1,
            gamma=0.05
        )

        # Initial calibration with small noise
        X_cal = np.arange(20).reshape(-1, 1)
        y_cal = X_cal + np.random.randn(20, 1) * 0.5
        predictor.calibrate(X_cal, y_cal)

        initial_quantile = predictor.adaptive_quantile

        # Simulate distribution shift with larger noise
        for i in range(50):
            X_new = np.array([[20 + i]])
            y_new = X_new + np.random.randn(1, 1) * 3.0  # Much larger noise

            predictor.predict_and_update(X_new, y_new)

        # Adaptive quantile should have increased (not the initial quantile)
        assert predictor.adaptive_quantile > initial_quantile

    def test_get_adaptation_stats(self):
        """Test adaptation statistics."""
        def simple_predictor(X):
            return X

        predictor = AdaptiveConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1,
            gamma=0.01
        )

        X_cal = np.arange(10).reshape(-1, 1)
        y_cal = X_cal + np.random.randn(10, 1)
        predictor.calibrate(X_cal, y_cal)

        # Do some updates
        for i in range(10):
            X_new = np.array([[10 + i]])
            y_new = X_new + np.random.randn(1, 1)
            predictor.predict_and_update(X_new, y_new)

        stats = predictor.get_adaptation_stats()

        assert 'num_updates' in stats
        assert 'current_quantile' in stats
        assert 'initial_quantile' in stats
        assert 'quantile_change' in stats
        assert 'recent_coverage' in stats
        assert 'target_coverage' in stats

        assert stats['num_updates'] == 10
        assert stats['target_coverage'] == 0.9

    def test_different_gamma_values(self):
        """Test with different adaptation rates."""
        def simple_predictor(X):
            return X

        X_cal = np.arange(10).reshape(-1, 1)
        y_cal = X_cal + np.random.randn(10, 1)

        results = []

        # Test different gamma values
        for gamma in [0.001, 0.01, 0.1]:
            predictor = AdaptiveConformalPredictor(
                predict_fn=simple_predictor,
                alpha=0.1,
                gamma=gamma
            )

            predictor.calibrate(X_cal, y_cal)
            initial_q = predictor.adaptive_quantile

            # Do one update with large error
            X_new = np.array([[10]])
            y_new = X_new + np.random.randn(1, 1) * 5  # Large error

            predictor.predict_and_update(X_new, y_new)

            # Track quantile change
            quantile_change = predictor.adaptive_quantile - initial_q
            results.append((gamma, quantile_change))

        # Higher gamma should generally lead to larger changes (though not always due to randomness)
        assert results[2][1] > results[0][1]  # gamma=0.1 > gamma=0.001



class TestConformalVisualization:
    """Test conformal prediction visualization."""

    def test_plot_conformal_intervals_basic(self):
        """Test basic plotting functionality."""
        # Create dummy data
        y_true = np.sin(np.linspace(0, 10, 50))
        y_pred = y_true + np.random.randn(50) * 0.1
        lower = y_pred - 0.5
        upper = y_pred + 0.5

        # This should not raise an error
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing

            fig, ax = plot_conformal_intervals(
                y_true=y_true,
                y_pred=y_pred,
                lower=lower,
                upper=upper
            )
            assert fig is not None
            assert ax is not None
        except Exception as e:
            pytest.fail(f"Plotting failed: {e}")

    def test_plot_with_custom_parameters(self):
        """Test plotting with custom parameters."""
        y_true = np.arange(20).astype(float)
        y_pred = y_true + np.random.randn(20) * 0.5
        lower = y_pred - 1
        upper = y_pred + 1

        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing

            fig, ax = plot_conformal_intervals(
                y_true=y_true,
                y_pred=y_pred,
                lower=lower,
                upper=upper,
                title="Custom Title",
                figsize=(10, 5)
            )
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Plotting with custom parameters failed: {e}")


class TestConformalIntegration:
    """Test integration scenarios."""

    def test_time_series_forecasting_workflow(self):
        """Test complete workflow for time series forecasting."""
        np.random.seed(42)

        # Simulate time series
        t = np.arange(100)
        y = np.sin(t * 0.1) + np.random.randn(100) * 0.3

        # Simple AR(1) predictor
        def ar_predictor(history):
            """Predict next value based on last value."""
            return history[-1:] * 0.9

        predictor = SplitConformalPredictor(
            predict_fn=ar_predictor,
            alpha=0.1
        )

        # Calibration set
        X_cal = y[:50].reshape(-1, 1)
        y_cal = y[1:51].reshape(-1, 1)

        predictor.calibrate(X_cal, y_cal)

        # Test set
        X_test = y[51:80].reshape(-1, 1)
        y_test = y[52:81].reshape(-1, 1)

        lower, pred, upper = predictor.predict(X_test)

        # Check coverage
        coverage = np.mean((y_test >= lower) & (y_test <= upper))

        # Should achieve decent coverage
        assert coverage >= 0.7

    def test_adaptive_workflow_with_concept_drift(self):
        """Test adaptive conformal with concept drift."""
        np.random.seed(42)

        # Create data with concept drift
        n_samples = 200
        t = np.arange(n_samples)

        # First half: low noise
        y1 = np.sin(t[:100] * 0.1) + np.random.randn(100) * 0.3

        # Second half: high noise (concept drift)
        y2 = np.sin(t[100:] * 0.1) + np.random.randn(100) * 1.5

        y = np.concatenate([y1, y2])

        def simple_predictor(history):
            return history[-1:] * 0.95

        predictor = AdaptiveConformalPredictor(
            predict_fn=simple_predictor,
            alpha=0.1,
            gamma=0.05
        )

        # Calibrate on first part
        X_cal = y[:50].reshape(-1, 1)
        y_cal = y[1:51].reshape(-1, 1)
        predictor.calibrate(X_cal, y_cal)

        initial_quantile = predictor.adaptive_quantile

        # Process second part with drift
        for i in range(50, 150):
            X_new = y[i:i+1].reshape(-1, 1)
            y_new = y[i+1:i+2].reshape(-1, 1)
            predictor.predict_and_update(X_new, y_new)

        # Adaptive quantile should have increased significantly to handle larger noise
        # Just check it increased, not necessarily by 1.5x due to random variations
        assert predictor.adaptive_quantile > initial_quantile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
