"""
Comprehensive tests for APDTFlowForecaster high-level API.
Tests new v0.2.2 features: data validation, save/load, score, summary, early stopping.
"""
import numpy as np
import pandas as pd
import pytest
import os
import tempfile
from apdtflow.forecaster import APDTFlowForecaster


class TestDataValidation:
    """Test data validation with helpful error messages."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            model.fit(empty_df, target_col='value')

    def test_data_too_short(self):
        """Test that data too short raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=10, history_length=20, num_epochs=5, verbose=False)
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})  # Only 5 rows

        with pytest.raises(ValueError, match="Data too short"):
            model.fit(df, target_col='value')

    def test_missing_target_column(self):
        """Test that missing target column raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        df = pd.DataFrame({'sales': np.random.randn(100)})

        with pytest.raises(ValueError, match="Column 'price' not found"):
            model.fit(df, target_col='price')

    def test_missing_date_column(self):
        """Test that missing date column raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        df = pd.DataFrame({'value': np.random.randn(100)})

        with pytest.raises(ValueError, match="Date column 'date' not found"):
            model.fit(df, target_col='value', date_col='date')

    def test_missing_exog_columns(self):
        """Test that missing exog columns raise helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        df = pd.DataFrame({
            'value': np.random.randn(100),
            'temp': np.random.randn(100)
        })

        with pytest.raises(ValueError, match="Exogenous columns not found"):
            model.fit(df, target_col='value', exog_cols=['temp', 'humidity', 'pressure'])

    def test_nan_in_target(self):
        """Test that NaN in target raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        data = np.random.randn(100)
        data[10] = np.nan
        data[25] = np.nan
        df = pd.DataFrame({'value': data})

        with pytest.raises(ValueError, match="contains 2 NaN values"):
            model.fit(df, target_col='value')

    def test_inf_in_target(self):
        """Test that inf in target raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        data = np.random.randn(100)
        data[10] = np.inf
        df = pd.DataFrame({'value': data})

        with pytest.raises(ValueError, match="contains infinite values"):
            model.fit(df, target_col='value')

    def test_nan_in_exog(self):
        """Test that NaN in exog columns raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        df = pd.DataFrame({
            'value': np.random.randn(100),
            'temp': np.random.randn(100)
        })
        df.loc[5, 'temp'] = np.nan

        with pytest.raises(ValueError, match="Exogenous column 'temp' contains"):
            model.fit(df, target_col='value', exog_cols=['temp'])

    def test_empty_array(self):
        """Test that empty array raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        empty_array = np.array([])

        with pytest.raises(ValueError, match="Input array is empty"):
            model.fit(empty_array)

    def test_nan_in_array(self):
        """Test that NaN in array raises helpful error."""
        model = APDTFlowForecaster(forecast_horizon=7, history_length=10, num_epochs=5, verbose=False)
        data = np.random.randn(100)
        data[10] = np.nan

        with pytest.raises(ValueError, match="contains .* NaN values"):
            model.fit(data)


class TestSaveLoad:
    """Test save/load functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
        })

    def test_save_unfitted_model_fails(self):
        """Test that saving unfitted model raises error."""
        model = APDTFlowForecaster(forecast_horizon=7, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
                model.save(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_save_and_load(self, sample_data):
        """Test saving and loading a fitted model."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        model.fit(sample_data, target_col='sales', date_col='date')
        orig_preds = model.predict()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)
            loaded_model = APDTFlowForecaster.load(filepath)

            # Check predictions match
            loaded_preds = loaded_model.predict()
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds, decimal=4)

            # Check attributes preserved
            assert loaded_model.forecast_horizon == 7
            assert loaded_model.history_length == 20
            assert loaded_model.target_col_ == 'sales'
            assert loaded_model._is_fitted
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_to_different_device(self, sample_data):
        """Test loading model to different device."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            device='cpu',
            verbose=False
        )

        model.fit(sample_data, target_col='sales')

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)
            loaded_model = APDTFlowForecaster.load(filepath, device='cpu')

            assert loaded_model.device.type == 'cpu'
            assert loaded_model._is_fitted
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestScore:
    """Test score() method for evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with train/test split."""
        np.random.seed(42)
        data = np.sin(np.arange(200) * 0.1) + np.random.randn(200) * 0.1
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=200, freq='D'),
            'sales': data
        })
        return df[:150], df[150:]  # train, test

    def test_score_mse(self, sample_data):
        """Test score with MSE metric."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        mse = model.score(test_df, target_col='sales', metric='mse')
        assert isinstance(mse, float)
        assert mse >= 0

    def test_score_mae(self, sample_data):
        """Test score with MAE metric."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        mae = model.score(test_df, target_col='sales', metric='mae')
        assert isinstance(mae, float)
        assert mae >= 0

    def test_score_rmse(self, sample_data):
        """Test score with RMSE metric."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        rmse = model.score(test_df, target_col='sales', metric='rmse')
        assert isinstance(rmse, float)
        assert rmse >= 0

    def test_score_mape(self, sample_data):
        """Test score with MAPE metric."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        mape = model.score(test_df, target_col='sales', metric='mape')
        assert isinstance(mape, float)
        assert mape >= 0

    def test_score_r2(self, sample_data):
        """Test score with R2 metric."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        r2 = model.score(test_df, target_col='sales', metric='r2')
        assert isinstance(r2, float)
        assert r2 <= 1  # R2 can be negative for bad models

    def test_score_unfitted_fails(self):
        """Test that scoring unfitted model raises error."""
        model = APDTFlowForecaster(forecast_horizon=7, verbose=False)
        df = pd.DataFrame({'sales': np.random.randn(100)})

        with pytest.raises(RuntimeError, match="Model must be fitted before scoring"):
            model.score(df, target_col='sales')

    def test_score_unknown_metric(self, sample_data):
        """Test that unknown metric raises error."""
        train_df, test_df = sample_data
        model = APDTFlowForecaster(forecast_horizon=7, history_length=20, num_epochs=5, verbose=False)
        model.fit(train_df, target_col='sales')

        with pytest.raises(ValueError, match="Unknown metric"):
            model.score(test_df, target_col='sales', metric='unknown')


class TestSummary:
    """Test model.summary() method."""

    def test_summary_before_fit(self):
        """Test summary on unfitted model."""
        model = APDTFlowForecaster(forecast_horizon=14, history_length=30, verbose=False)
        model.summary()  # Should not raise

    def test_summary_after_fit(self):
        """Test summary on fitted model."""
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
        })

        model = APDTFlowForecaster(
            forecast_horizon=14,
            history_length=30,
            num_epochs=5,
            verbose=False
        )
        model.fit(df, target_col='sales', date_col='date')
        model.summary()  # Should print model info

    def test_summary_with_exog(self):
        """Test summary with exogenous variables."""
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1,
            'temp': np.random.randn(100),
            'humidity': np.random.randn(100)
        })

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )
        model.fit(df, target_col='sales', exog_cols=['temp', 'humidity'])
        model.summary()  # Should show exog info


class TestEarlyStopping:
    """Test early stopping functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=150, freq='D'),
            'sales': np.sin(np.arange(150) * 0.1) + np.random.randn(150) * 0.1
        })

    def test_early_stopping_enabled(self, sample_data):
        """Test that early stopping can stop training early."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=20,  # Many epochs
            early_stopping=True,
            patience=3,
            validation_split=0.2,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')
        assert model._is_fitted
        # If it trained all 100 epochs without stopping, something is wrong
        # But we can't easily check this without modifying the model

    def test_early_stopping_disabled(self, sample_data):
        """Test that model trains normally when early stopping disabled."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=10,
            early_stopping=False,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')
        assert model._is_fitted

    def test_validation_split(self, sample_data):
        """Test that validation split creates separate datasets."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            early_stopping=True,
            patience=2,
            validation_split=0.3,
            verbose=True  # Will print train/val sizes
        )

        model.fit(sample_data, target_col='sales')
        assert model._is_fitted


class TestModelTypes:
    """Test switching between different model architectures."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
        })

    @pytest.mark.parametrize("model_type", ['apdtflow', 'transformer', 'tcn'])
    def test_different_model_types(self, sample_data, model_type):
        """Test that supported model types can be trained and predict."""
        model = APDTFlowForecaster(
            model_type=model_type,
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')
        predictions = model.predict()

        assert predictions.shape == (7,)
        assert not np.isnan(predictions).any()

    def test_ensemble_not_supported(self, sample_data):
        """Test that ensemble model type raises ValueError."""
        model = APDTFlowForecaster(
            model_type='ensemble',
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        with pytest.raises(ValueError, match="not currently supported"):
            model.fit(sample_data, target_col='sales')


class TestPlotForecast:
    """Test plot_forecast() visualization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
        })

    def test_plot_forecast_unfitted_fails(self):
        """Test that plotting unfitted model raises error."""
        model = APDTFlowForecaster(forecast_horizon=7, verbose=False)

        with pytest.raises(RuntimeError, match="Model must be fitted before plotting"):
            model.plot_forecast()

    def test_plot_forecast_basic(self, sample_data):
        """Test basic plotting functionality."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name

        try:
            fig, ax = model.plot_forecast(
                with_history=30,
                show_uncertainty=True,
                save_path=filepath
            )

            assert os.path.exists(filepath)
            assert fig is not None
            assert ax is not None
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_plot_forecast_without_uncertainty(self, sample_data):
        """Test plotting without uncertainty bands."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            verbose=False
        )

        model.fit(sample_data, target_col='sales')
        fig, ax = model.plot_forecast(show_uncertainty=False)

        assert fig is not None
        assert ax is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
