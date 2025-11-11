"""
Integration tests for categorical features in APDTFlowForecaster.
Tests the full workflow with categorical variables.
"""
import numpy as np
import pytest
import pandas as pd
import tempfile
import os
from apdtflow import APDTFlowForecaster


class TestCategoricalIntegration:
    """Test categorical feature integration in APDTFlowForecaster."""

    @pytest.fixture
    def sample_data_with_categorical(self):
        """Create sample data with categorical features."""
        np.random.seed(42)
        n_samples = 200

        dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

        # Create target with day-of-week effect
        base = 100 + 0.1 * np.arange(n_samples)
        dow_effect = np.array([0, 1, 2, 3, 4, 8, 6])  # Weekend boost
        dow = np.array([dow_effect[d.dayofweek] for d in dates])
        noise = np.random.randn(n_samples) * 2

        df = pd.DataFrame({
            'date': dates,
            'sales': base + dow + noise,
            'day_of_week': [d.day_name() for d in dates],
            'is_weekend': ['True' if d.dayofweek >= 5 else 'False' for d in dates],
            'category': np.random.choice(['A', 'B', 'C'], n_samples)
        })

        return df

    def test_fit_with_single_categorical(self, sample_data_with_categorical):
        """Test fitting model with single categorical feature."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            categorical_encoding='onehot',
            verbose=False
        )

        # Fit with one categorical column
        model.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week']
        )

        assert model._is_fitted
        assert model.has_categorical_
        assert model.categorical_cols_ == ['day_of_week']
        assert model.categorical_encoder_ is not None

    def test_fit_with_multiple_categorical(self, sample_data_with_categorical):
        """Test fitting model with multiple categorical features."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            categorical_encoding='onehot',
            verbose=False
        )

        model.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week', 'is_weekend', 'category']
        )

        assert model._is_fitted
        assert model.has_categorical_
        assert len(model.categorical_cols_) == 3

    def test_categorical_with_numerical_exog(self, sample_data_with_categorical):
        """Test combining categorical with numerical exogenous features."""
        # Add numerical exog feature
        df = sample_data_with_categorical.copy()
        df['temperature'] = 20 + np.random.randn(len(df)) * 5

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            verbose=False
        )

        model.fit(
            df,
            target_col='sales',
            date_col='date',
            exog_cols=['temperature'],  # Numerical
            categorical_cols=['day_of_week']  # Categorical
        )

        assert model._is_fitted
        assert model.has_exog_
        assert model.has_categorical_

    def test_predict_after_categorical_fit(self, sample_data_with_categorical):
        """Test prediction after fitting with categorical features."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            verbose=False
        )

        model.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week']
        )

        # Make predictions
        predictions = model.predict(steps=7)

        assert predictions.shape == (7,)
        assert np.all(np.isfinite(predictions))

    def test_save_load_with_categorical(self, sample_data_with_categorical):
        """Test saving and loading model with categorical features."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            categorical_encoding='onehot',
            verbose=False
        )

        model.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week', 'category']
        )

        # Make predictions before save
        pred_before = model.predict(steps=7)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            model.save(temp_path)

            # Load model
            loaded_model = APDTFlowForecaster.load(temp_path)

            # Check loaded model has categorical state
            assert loaded_model.has_categorical_
            assert loaded_model.categorical_cols_ == ['day_of_week', 'category']
            assert loaded_model.categorical_encoder_ is not None

            # Make predictions with loaded model
            pred_after = loaded_model.predict(steps=7)

            # Predictions should be identical
            assert np.allclose(pred_before, pred_after, rtol=1e-5)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_onehot_vs_embedding_encoding(self, sample_data_with_categorical):
        """Test both encoding types work."""
        # One-hot encoding
        model_onehot = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            categorical_encoding='onehot',
            verbose=False
        )

        model_onehot.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week']
        )

        pred_onehot = model_onehot.predict(steps=7)

        # Embedding encoding
        model_embedding = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            categorical_encoding='embedding',
            verbose=False
        )

        model_embedding.fit(
            sample_data_with_categorical,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week']
        )

        pred_embedding = model_embedding.predict(steps=7)

        # Both should produce valid predictions
        assert np.all(np.isfinite(pred_onehot))
        assert np.all(np.isfinite(pred_embedding))

    def test_categorical_only_apdtflow_model(self, sample_data_with_categorical):
        """Test that categorical features only work with apdtflow model type."""
        for model_type in ['transformer', 'tcn']:
            model = APDTFlowForecaster(
                model_type=model_type,
                forecast_horizon=7,
                history_length=20,
                num_epochs=2,
                verbose=False
            )

            with pytest.raises(ValueError, match="only supported with model_type='apdtflow'"):
                model.fit(
                    sample_data_with_categorical,
                    target_col='sales',
                    date_col='date',
                    categorical_cols=['day_of_week']
                )

    def test_categorical_validation_missing_columns(self, sample_data_with_categorical):
        """Test validation error when categorical columns don't exist."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            verbose=False
        )

        with pytest.raises(ValueError, match="Categorical columns not found"):
            model.fit(
                sample_data_with_categorical,
                target_col='sales',
                date_col='date',
                categorical_cols=['nonexistent_column']
            )

    def test_categorical_requires_dataframe(self):
        """Test that categorical features require DataFrame input."""
        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            verbose=False
        )

        # Try to fit with numpy array and categorical_cols
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="categorical_cols requires DataFrame"):
            model.fit(
                data,
                categorical_cols=['some_col']
            )

    def test_score_with_categorical(self, sample_data_with_categorical):
        """Test scoring with categorical features."""
        train_df = sample_data_with_categorical.iloc[:150]
        test_df = sample_data_with_categorical.iloc[150:]

        model = APDTFlowForecaster(
            forecast_horizon=7,
            history_length=20,
            num_epochs=2,
            verbose=False
        )

        model.fit(
            train_df,
            target_col='sales',
            date_col='date',
            categorical_cols=['day_of_week']
        )

        # Score on test data
        mse = model.score(test_df, target_col='sales', metric='mse')
        mae = model.score(test_df, target_col='sales', metric='mae')

        assert mse > 0
        assert mae > 0
        assert np.isfinite(mse)
        assert np.isfinite(mae)


class TestCategoricalEdgeCases:
    """Test edge cases for categorical features."""

    def test_categorical_with_single_category(self):
        """Test categorical feature with only one unique value."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'value': np.random.randn(50) + 100,
            'constant_cat': ['A'] * 50  # Only one category
        })

        model = APDTFlowForecaster(
            forecast_horizon=5,
            history_length=10,
            num_epochs=2,
            verbose=False
        )

        # Should still work (though not very useful)
        model.fit(
            df,
            target_col='value',
            date_col='date',
            categorical_cols=['constant_cat']
        )

        assert model._is_fitted

    def test_categorical_with_many_categories(self):
        """Test categorical feature with many unique values."""
        np.random.seed(42)
        n_categories = 20

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'value': np.random.randn(100) + 100,
            'many_cats': np.random.choice([f'cat_{i}' for i in range(n_categories)], 100)
        })

        model = APDTFlowForecaster(
            forecast_horizon=5,
            history_length=10,
            num_epochs=2,
            categorical_encoding='embedding',  # Better for many categories
            verbose=False
        )

        model.fit(
            df,
            target_col='value',
            date_col='date',
            categorical_cols=['many_cats']
        )

        assert model._is_fitted
        predictions = model.predict(steps=5)
        assert np.all(np.isfinite(predictions))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
