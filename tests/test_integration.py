"""
Integration tests for APDTFlow v0.2.0 features.
Tests end-to-end workflows with exogenous variables and conformal prediction.
"""
import numpy as np
import pandas as pd
import torch
import pytest
from apdtflow.forecaster import APDTFlowForecaster
from apdtflow.models.apdtflow import APDTFlow
from apdtflow.data import TimeSeriesWindowDataset
from torch.utils.data import DataLoader


class TestForecasterWithExogenous:
    """Integration tests for APDTFlowForecaster with exogenous variables."""

    @pytest.fixture
    def sample_data_with_exog(self):
        """Create sample DataFrame with exogenous variables."""
        np.random.seed(42)
        n = 200

        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        target = np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.3

        # Exogenous features
        temp = 20 + np.sin(np.arange(n) * 0.05) * 10 + np.random.randn(n) * 2
        humidity = 50 + np.cos(np.arange(n) * 0.07) * 20 + np.random.randn(n) * 5
        is_weekend = np.array([1 if i % 7 in [5, 6] else 0 for i in range(n)])

        df = pd.DataFrame({
            'date': dates,
            'sales': target,
            'temperature': temp,
            'humidity': humidity,
            'is_weekend': is_weekend
        })

        return df

    def test_forecaster_fit_with_exog(self, sample_data_with_exog):
        """Test fitting forecaster with exogenous variables."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=30,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        # Fit with exog
        model.fit(
            sample_data_with_exog,
            target_col='sales',
            date_col='date',
            exog_cols=['temperature', 'humidity', 'is_weekend'],
            future_exog_cols=['is_weekend']
        )

        assert model._is_fitted
        assert model.has_exog_
        assert model.num_exog_features_ == 3
        assert model.exog_cols_ == ['temperature', 'humidity', 'is_weekend']
        assert model.future_exog_cols_ == ['is_weekend']

    def test_forecaster_predict_with_exog(self, sample_data_with_exog):
        """Test prediction with exogenous variables."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=30,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        # Fit
        train_df = sample_data_with_exog[:150]
        model.fit(
            train_df,
            target_col='sales',
            date_col='date',
            exog_cols=['temperature', 'humidity', 'is_weekend'],
            future_exog_cols=['is_weekend']
        )

        # Create future exog data
        future_exog = pd.DataFrame({
            'is_weekend': [0, 0, 1, 1, 0, 0, 0]
        })

        # Predict
        preds = model.predict(exog_future=future_exog)

        assert preds.shape == (7,)
        assert not np.isnan(preds).any()

    def test_forecaster_all_fusion_types(self, sample_data_with_exog):
        """Test all fusion types in forecaster."""
        train_df = sample_data_with_exog[:100]

        for fusion_type in ['concat', 'gated', 'attention']:
            model = APDTFlowForecaster(
                forecast_horizon=7,
                history_length=20,
                num_epochs=3,
                exog_fusion_type=fusion_type,
                verbose=False
            )

            model.fit(
                train_df,
                target_col='sales',
                date_col='date',
                exog_cols=['temperature', 'humidity'],
                future_exog_cols=['humidity']
            )

            future_exog = pd.DataFrame({'humidity': [50, 55, 60, 65, 70, 75, 80]})
            preds = model.predict(exog_future=future_exog)

            assert preds.shape == (7,)

    def test_forecaster_without_exog_still_works(self, sample_data_with_exog):
        """Test that forecaster works without exog (backward compatibility)."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=30,
            num_epochs=5,
            verbose=False
        )

        # Fit without exog
        model.fit(
            sample_data_with_exog,
            target_col='sales',
            date_col='date'
        )

        # Predict without exog
        preds = model.predict()

        assert preds.shape == (7,)
        assert not model.has_exog_


class TestDataLoaderWithExogenous:
    """Integration tests for data loading with exogenous variables."""

    @pytest.fixture
    def sample_csv_with_exog(self, tmp_path):
        """Create temporary CSV with exogenous variables."""
        n = 100
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'value': np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.2,
            'temp': 20 + np.random.randn(n) * 5,
            'holiday': np.random.randint(0, 2, n)
        })

        csv_path = tmp_path / "data_with_exog.csv"
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    def test_dataset_with_exog(self, sample_csv_with_exog):
        """Test TimeSeriesWindowDataset with exogenous variables."""
        dataset = TimeSeriesWindowDataset(
            csv_file=sample_csv_with_exog,
            date_col='date',
            value_col='value',
            T_in=30,
            T_out=7,
            exog_cols=['temp', 'holiday']
        )

        assert len(dataset) > 0

        # Get sample
        sample = dataset[0]

        if len(sample) == 4:
            x, y, exog_x, exog_y = sample
            assert x.shape == (1, 30)
            assert y.shape == (1, 7)
            assert exog_x.shape == (2, 30)
            assert exog_y.shape == (2, 7)

    def test_dataloader_with_exog(self, sample_csv_with_exog):
        """Test DataLoader with exogenous variables."""
        dataset = TimeSeriesWindowDataset(
            csv_file=sample_csv_with_exog,
            date_col='date',
            value_col='value',
            T_in=30,
            T_out=7,
            exog_cols=['temp', 'holiday']
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        for batch in dataloader:
            if len(batch) == 4:
                x, y, exog_x, exog_y = batch
                assert x.dim() == 3
                assert exog_x.dim() == 3
            break


class TestFullPipelineIntegration:
    """End-to-end integration tests."""

    def test_complete_workflow_with_exog(self):
        """Test complete workflow: data -> train -> predict."""
        np.random.seed(42)

        # Create synthetic data
        n = 150
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'sales': np.cumsum(np.random.randn(n)) + 100,
            'promo': np.random.randint(0, 2, n),
            'price': 10 + np.random.randn(n) * 2
        })

        # Initialize model
        model = APDTFlowForecaster(
            forecast_horizon=14,
            history_length=30,
            num_epochs=10,
            batch_size=16,
            exog_fusion_type='gated',
            verbose=True
        )

        # Train
        train_df = df[:120]
        model.fit(
            train_df,
            target_col='sales',
            date_col='date',
            exog_cols=['promo', 'price'],
            future_exog_cols=['promo']
        )

        # Predict
        future_promo = pd.DataFrame({'promo': np.random.randint(0, 2, 14)})
        predictions = model.predict(exog_future=future_promo)

        # Validate
        assert predictions.shape == (14,)
        assert not np.isnan(predictions).any()
        assert np.all(np.isfinite(predictions))

        # Test with uncertainty
        preds, uncertainty = model.predict(
            exog_future=future_promo,
            return_uncertainty=True
        )

        assert uncertainty.shape == (14,)
        assert np.all(uncertainty > 0)

    def test_complete_workflow_without_exog(self):
        """Test complete workflow without exogenous variables."""
        np.random.seed(42)

        # Create simple time series
        n = 120
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'value': np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 0.5
        })

        # Initialize and train
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=30,
            num_epochs=10,
            verbose=False
        )

        model.fit(df, target_col='value', date_col='date')

        # Predict
        predictions = model.predict()

        assert predictions.shape == (7,)
        assert not np.isnan(predictions).any()

    def test_model_persistence_with_exog(self):
        """Test that model with exog can be used after fitting."""
        np.random.seed(42)

        n = 100
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'sales': np.random.randn(n).cumsum() + 50,
            'temp': 20 + np.random.randn(n) * 5
        })

        # Fit model
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        model.fit(
            df,
            target_col='sales',
            date_col='date',
            exog_cols=['temp'],
            future_exog_cols=['temp']
        )

        # Multiple predictions should work
        for _ in range(3):
            future_temp = pd.DataFrame({'temp': 20 + np.random.randn(7) * 5})
            preds = model.predict(exog_future=future_temp)
            assert preds.shape == (7,)

    def test_error_handling_missing_exog(self):
        """Test that appropriate error is raised when exog is missing."""
        np.random.seed(42)

        n = 100
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n, freq='D'),
            'sales': np.random.randn(n).cumsum(),
            'temp': 20 + np.random.randn(n) * 5
        })

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=5,
            exog_fusion_type='gated',
            verbose=False
        )

        model.fit(
            df,
            target_col='sales',
            exog_cols=['temp']
        )

        # Should raise error when predicting without exog
        with pytest.raises(ValueError, match="exogenous variables"):
            model.predict()


class TestModelWithExogenousDirect:
    """Direct model tests with exogenous variables."""

    def test_apdtflow_model_training_with_exog(self):
        """Test training APDTFlow model directly with exogenous data."""
        # Create model
        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=7,
            use_embedding=True,
            num_exog_features=2,
            exog_fusion_type='gated'
        )

        # Create data
        batch_size = 8
        T_in = 30
        x = torch.randn(batch_size, 1, T_in)
        y = torch.randn(batch_size, 1, 7)
        exog = torch.randn(batch_size, 2, T_in)
        t_span = torch.linspace(0, 1, steps=T_in)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(5):
            model.train()
            optimizer.zero_grad()

            preds, pred_logvars = model(x, t_span, exog=exog)

            mse = (preds - y.transpose(1, 2)) ** 2
            loss = torch.mean(
                0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
            )

            loss.backward()
            optimizer.step()

        # Test evaluation
        model.eval()
        with torch.no_grad():
            preds, _ = model(x, t_span, exog=exog)
            assert preds.shape == (batch_size, 7, 1)

    def test_model_with_different_exog_sizes(self):
        """Test model with different numbers of exogenous features."""
        for num_exog in [1, 3, 5, 10]:
            model = APDTFlow(
                num_scales=3,
                input_channels=1,
                filter_size=5,
                hidden_dim=16,
                output_dim=1,
                forecast_horizon=7,
                num_exog_features=num_exog,
                exog_fusion_type='gated'
            )

            x = torch.randn(4, 1, 30)
            exog = torch.randn(4, num_exog, 30)
            t_span = torch.linspace(0, 1, steps=30)

            preds, pred_logvars = model(x, t_span, exog=exog)

            assert preds.shape == (4, 7, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
