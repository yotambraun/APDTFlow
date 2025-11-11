"""
Integration tests for exogenous variables and conformal prediction with new v0.2.2 features.
Ensures save/load, score, early stopping, and summary work correctly with exog and conformal.
"""
import numpy as np
import pandas as pd
import pytest
import os
import tempfile
from apdtflow.forecaster import APDTFlowForecaster


class TestExogenousSaveLoad:
    """Test save/load with exogenous variables."""

    @pytest.fixture
    def exog_data(self):
        """Create sample data with exogenous variables."""
        np.random.seed(42)
        n = 150
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        sales = np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.1
        temp = 20 + 10 * np.sin(np.arange(n) * 0.05) + np.random.randn(n) * 2
        humidity = 50 + 20 * np.cos(np.arange(n) * 0.07) + np.random.randn(n) * 5

        return pd.DataFrame({
            'date': dates,
            'sales': sales,
            'temperature': temp,
            'humidity': humidity
        })

    def test_save_load_with_exog(self, exog_data):
        """Test that exog configuration is preserved in save/load."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            date_col='date',
            exog_cols=['temperature', 'humidity']
        )

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)

            # Load
            loaded_model = APDTFlowForecaster.load(filepath)

            # Verify exog configuration preserved
            assert loaded_model.has_exog_
            assert loaded_model.num_exog_features_ == 2
            assert loaded_model.exog_cols_ == ['temperature', 'humidity']
            assert loaded_model.exog_fusion_type == 'gated'
            assert loaded_model.exog_mean_ is not None
            assert loaded_model.exog_std_ is not None

            # Verify can make predictions (need future exog)
            future_exog = pd.DataFrame({
                'temperature': np.random.randn(7),
                'humidity': np.random.randn(7)
            })

            predictions = loaded_model.predict(exog_future=future_exog)
            assert predictions.shape == (7,)
            assert not np.isnan(predictions).any()

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_score_with_exog(self, exog_data):
        """Test that score() properly uses exogenous features."""
        train_df = exog_data[:100]
        test_df = exog_data[100:]

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        model.fit(
            train_df,
            target_col='sales',
            exog_cols=['temperature', 'humidity']
        )

        # Score should use exog features from test data
        mse = model.score(test_df, metric='mse')
        assert isinstance(mse, float)
        assert mse >= 0

        # Test with explicit exog_cols
        mse2 = model.score(
            test_df,
            metric='mse',
            exog_cols=['temperature', 'humidity']
        )
        assert isinstance(mse2, float)

    def test_early_stopping_with_exog(self, exog_data):
        """Test that early stopping works with exogenous variables."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            early_stopping=True,
            patience=3,
            validation_split=0.2,
            exog_fusion_type='gated',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'humidity']
        )

        assert model._is_fitted
        assert model.has_exog_

    def test_summary_with_exog(self, exog_data):
        """Test that summary() displays exog information."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='attention',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'humidity'],
            future_exog_cols=['humidity']
        )

        # Should not raise and should display exog info
        model.summary()


class TestConformalSaveLoad:
    """Test save/load with conformal prediction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 150
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'sales': np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.1
        })

    def test_save_load_with_conformal(self, sample_data):
        """Test that conformal predictor is preserved in save/load."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            use_conformal=True,
            conformal_method='adaptive',
            verbose=False
        )

        model.fit(sample_data, target_col='sales')

        # Verify conformal predictor exists
        assert model.conformal_predictor is not None

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)

            # Load
            loaded_model = APDTFlowForecaster.load(filepath)

            # Verify conformal configuration preserved
            assert loaded_model.use_conformal
            assert loaded_model.conformal_method == 'adaptive'
            assert loaded_model.conformal_predictor is not None

            # Verify can make conformal predictions
            lower, pred, upper = loaded_model.predict(
                alpha=0.05,
                return_intervals='conformal'
            )

            assert lower.shape == (7,)
            assert pred.shape == (7,)
            assert upper.shape == (7,)
            assert np.all(lower <= pred)
            assert np.all(pred <= upper)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_early_stopping_with_conformal(self, sample_data):
        """Test that early stopping works with conformal prediction."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            early_stopping=True,
            patience=3,
            validation_split=0.2,
            use_conformal=True,
            conformal_method='split',
            verbose=False
        )

        model.fit(sample_data, target_col='sales')

        assert model._is_fitted
        assert model.use_conformal

    def test_summary_with_conformal(self, sample_data):
        """Test that summary() displays conformal information."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            use_conformal=True,
            conformal_method='split',
            calibration_split=0.3,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')

        # Should not raise and should display conformal info
        model.summary()


class TestExogConformalTogether:
    """Test exogenous + conformal working together."""

    @pytest.fixture
    def exog_data(self):
        """Create sample data with exogenous variables."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        sales = np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.1
        temp = 20 + 10 * np.sin(np.arange(n) * 0.05) + np.random.randn(n) * 2
        holiday = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

        return pd.DataFrame({
            'date': dates,
            'sales': sales,
            'temperature': temp,
            'is_holiday': holiday
        })

    def test_exog_and_conformal_together(self, exog_data):
        """Test that exog and conformal prediction work together."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            use_conformal=True,
            conformal_method='adaptive',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'is_holiday'],
            future_exog_cols=['is_holiday']
        )

        # Both should be enabled
        assert model.has_exog_
        assert model.use_conformal
        assert model.conformal_predictor is not None

        # Make conformal predictions with exog
        future_exog = pd.DataFrame({
            'is_holiday': [0, 0, 1, 0, 0, 0, 0]
        })

        lower, pred, upper = model.predict(
            exog_future=future_exog,
            alpha=0.05,
            return_intervals='conformal'
        )

        assert lower.shape == (7,)
        assert pred.shape == (7,)
        assert upper.shape == (7,)

    def test_save_load_exog_conformal(self, exog_data):
        """Test save/load with both exog and conformal."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='attention',
            use_conformal=True,
            conformal_method='split',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'is_holiday']
        )

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)
            loaded_model = APDTFlowForecaster.load(filepath)

            # Verify both preserved
            assert loaded_model.has_exog_
            assert loaded_model.num_exog_features_ == 2
            assert loaded_model.use_conformal
            assert loaded_model.conformal_predictor is not None

            # Verify can make predictions
            future_exog = pd.DataFrame({
                'temperature': np.random.randn(7),
                'is_holiday': [0, 0, 1, 0, 0, 0, 0]
            })

            lower, pred, upper = loaded_model.predict(
                exog_future=future_exog,
                alpha=0.05,
                return_intervals='conformal'
            )

            assert pred.shape == (7,)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_score_with_exog_conformal(self, exog_data):
        """Test scoring with both exog and conformal."""
        train_df = exog_data[:150]
        test_df = exog_data[150:]

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            use_conformal=True,
            conformal_method='split',
            verbose=False
        )

        model.fit(
            train_df,
            target_col='sales',
            exog_cols=['temperature', 'is_holiday']
        )

        # Score should work with both exog and conformal
        mse = model.score(test_df, metric='mse', exog_cols=['temperature', 'is_holiday'])
        assert isinstance(mse, float)
        assert mse >= 0

    def test_early_stopping_exog_conformal(self, exog_data):
        """Test early stopping with both exog and conformal."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            early_stopping=True,
            patience=3,
            validation_split=0.2,
            exog_fusion_type='gated',
            use_conformal=True,
            conformal_method='adaptive',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'is_holiday']
        )

        assert model._is_fitted
        assert model.has_exog_
        assert model.use_conformal

    def test_summary_exog_conformal(self, exog_data):
        """Test summary with both exog and conformal."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='attention',
            use_conformal=True,
            conformal_method='adaptive',
            verbose=False
        )

        model.fit(
            exog_data,
            target_col='sales',
            exog_cols=['temperature', 'is_holiday'],
            future_exog_cols=['is_holiday']
        )

        # Should display both exog and conformal info
        model.summary()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
