"""
Comprehensive test suite for categorical encoder functionality.
Tests CategoricalEncoder and create_time_features utility.
"""
import numpy as np
import pytest
import pandas as pd
from apdtflow.preprocessing.categorical_encoder import CategoricalEncoder, create_time_features


class TestCategoricalEncoder:
    """Test suite for CategoricalEncoder."""

    def test_onehot_encoding_basic(self):
        """Test basic one-hot encoding."""
        data = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['S', 'M', 'L', 'M']
        })

        encoder = CategoricalEncoder(encoding_type='onehot', handle_unknown='error')
        encoded = encoder.fit_transform(data)

        # Check shape: 4 samples, 3 colors + 3 sizes = 6 features
        assert encoded.shape == (4, 6)

        # Check it's one-hot (each row sums to 2: one per column)
        assert np.allclose(encoded.sum(axis=1), 2.0)

        # Check binary values
        assert np.all((encoded == 0) | (encoded == 1))

    def test_onehot_encoding_unknown_indicator(self):
        """Test one-hot encoding with unknown category handling."""
        train_data = pd.DataFrame({
            'category': ['A', 'B', 'C']
        })

        encoder = CategoricalEncoder(
            encoding_type='onehot',
            handle_unknown='indicator'
        )
        encoder.fit(train_data)

        # Test with unknown category
        test_data = pd.DataFrame({
            'category': ['A', 'D']  # 'D' is unknown
        })

        encoded = test_data

        encoded = encoder.transform(test_data)

        # Should have 4 features: A, B, C, __UNKNOWN__
        assert encoded.shape == (2, 4)

        # First row: 'A' should be [1, 0, 0, 0]
        assert encoded[0, 0] == 1.0  # A is active
        assert encoded[0, -1] == 0.0  # Unknown is not active

        # Second row: 'D' (unknown) should be [0, 0, 0, 1]
        assert encoded[1, -1] == 1.0  # Unknown is active

    def test_onehot_encoding_unknown_ignore(self):
        """Test one-hot encoding with unknown category ignored."""
        train_data = pd.DataFrame({
            'category': ['A', 'B']
        })

        encoder = CategoricalEncoder(
            encoding_type='onehot',
            handle_unknown='ignore'
        )
        encoder.fit(train_data)

        test_data = pd.DataFrame({
            'category': ['A', 'C']  # 'C' is unknown
        })

        encoded = encoder.transform(test_data)

        # Should have 2 features: A, B (no unknown indicator)
        assert encoded.shape == (2, 2)

        # Second row: 'C' (unknown) should be all zeros
        assert np.all(encoded[1, :] == 0.0)

    def test_onehot_encoding_unknown_error(self):
        """Test one-hot encoding raises error on unknown category."""
        train_data = pd.DataFrame({
            'category': ['A', 'B']
        })

        encoder = CategoricalEncoder(
            encoding_type='onehot',
            handle_unknown='error'
        )
        encoder.fit(train_data)

        test_data = pd.DataFrame({
            'category': ['A', 'C']  # 'C' is unknown
        })

        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(test_data)

    def test_embedding_encoding_basic(self):
        """Test basic embedding encoding."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A']
        })

        encoder = CategoricalEncoder(
            encoding_type='embedding',
            embedding_dim=4
        )
        encoded = encoder.fit_transform(data)

        # Check shape: 4 samples, 1 column * 4 embedding_dim = 4 features
        assert encoded.shape == (4, 4)

        # Check that embeddings are different for different categories
        # Row 0 and 3 are both 'A', should be same
        assert np.allclose(encoded[0], encoded[3])

        # Row 1 is 'B', should be different from 'A'
        assert not np.allclose(encoded[0], encoded[1])

    def test_multiple_columns(self):
        """Test encoding multiple categorical columns."""
        data = pd.DataFrame({
            'color': ['red', 'blue', 'red'],
            'size': ['S', 'M', 'S'],
            'type': ['A', 'B', 'A']
        })

        encoder = CategoricalEncoder(encoding_type='onehot')
        _ = encoder.fit_transform(data)

        # Check that we get features from all columns
        feature_names = encoder.get_feature_names()
        assert any('color' in name for name in feature_names)
        assert any('size' in name for name in feature_names)
        assert any('type' in name for name in feature_names)

    def test_get_feature_names(self):
        """Test getting feature names."""
        data = pd.DataFrame({
            'color': ['red', 'blue']
        })

        encoder = CategoricalEncoder(encoding_type='onehot')
        encoder.fit(data)

        feature_names = encoder.get_feature_names()

        # Should have features for 'blue' and 'red'
        assert 'color_blue' in feature_names
        assert 'color_red' in feature_names

    def test_get_num_features(self):
        """Test getting number of features."""
        data = pd.DataFrame({
            'color': ['red', 'blue', 'green'],
            'size': ['S', 'M', 'S']
        })

        encoder = CategoricalEncoder(encoding_type='onehot', handle_unknown='error')
        encoder.fit(data)

        num_features = encoder.get_num_features()

        # 3 colors + 2 sizes = 5 features
        assert num_features == 5

    def test_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit then transform."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C']
        })

        # Method 1: fit_transform
        encoder1 = CategoricalEncoder(encoding_type='onehot')
        encoded1 = encoder1.fit_transform(data)

        # Method 2: fit then transform
        encoder2 = CategoricalEncoder(encoding_type='onehot')
        encoder2.fit(data)
        encoded2 = encoder2.transform(data)

        # Should be identical
        assert np.allclose(encoded1, encoded2)

    def test_save_load_config(self):
        """Test saving and loading encoder configuration."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C']
        })

        # Create and fit encoder
        encoder = CategoricalEncoder(
            encoding_type='onehot',
            embedding_dim=8,
            handle_unknown='indicator'
        )
        encoder.fit(data)

        # Get config
        config = encoder.get_config()

        # Create new encoder from config
        loaded_encoder = CategoricalEncoder.from_config(config)

        # Test that loaded encoder works
        test_data = pd.DataFrame({'category': ['A', 'B']})
        encoded_original = encoder.transform(test_data)
        encoded_loaded = loaded_encoder.transform(test_data)

        assert np.allclose(encoded_original, encoded_loaded)

    def test_invalid_encoding_type(self):
        """Test that invalid encoding type raises error."""
        with pytest.raises(ValueError, match="encoding_type must be"):
            CategoricalEncoder(encoding_type='invalid')

    def test_invalid_handle_unknown(self):
        """Test that invalid handle_unknown raises error."""
        with pytest.raises(ValueError, match="handle_unknown must be"):
            CategoricalEncoder(handle_unknown='invalid')

    def test_transform_before_fit(self):
        """Test that transform before fit raises error."""
        encoder = CategoricalEncoder()
        data = pd.DataFrame({'category': ['A', 'B']})

        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.transform(data)

    def test_dict_input(self):
        """Test encoder works with dict input."""
        data = {
            'color': ['red', 'blue', 'red'],
            'size': ['S', 'M', 'L']
        }

        encoder = CategoricalEncoder(encoding_type='onehot')
        encoded = encoder.fit_transform(data)

        # Should work the same as DataFrame
        assert encoded.shape[0] == 3  # 3 samples
        assert encoded.shape[1] > 0  # Some features


class TestCreateTimeFeatures:
    """Test suite for create_time_features utility."""

    def test_basic_time_features(self):
        """Test creating basic time features."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')
        features = create_time_features(dates, ['day_of_week', 'month'])

        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert len(features) == 7

        # Check day_of_week values (0=Monday, 6=Sunday)
        assert features['day_of_week'].iloc[0] == 0  # 2024-01-01 is Monday

    def test_day_name_feature(self):
        """Test day_name feature."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')
        features = create_time_features(dates, ['day_name'])

        assert 'day_name' in features.columns
        assert features['day_name'].iloc[0] == 'Monday'
        assert features['day_name'].iloc[6] == 'Sunday'

    def test_month_name_feature(self):
        """Test month_name feature."""
        dates = pd.date_range('2024-01-01', periods=3, freq='M')
        features = create_time_features(dates, ['month_name'])

        assert 'month_name' in features.columns
        assert features['month_name'].iloc[0] == 'January'

    def test_is_weekend_feature(self):
        """Test is_weekend feature."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')  # Mon-Sun
        features = create_time_features(dates, ['is_weekend'])

        assert 'is_weekend' in features.columns

        # Monday-Friday should be 'False', Saturday-Sunday should be 'True'
        assert features['is_weekend'].iloc[0] == 'False'  # Monday
        assert features['is_weekend'].iloc[5] == 'True'   # Saturday
        assert features['is_weekend'].iloc[6] == 'True'   # Sunday

    def test_quarter_feature(self):
        """Test quarter feature."""
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        features = create_time_features(dates, ['quarter'])

        assert 'quarter' in features.columns
        assert features['quarter'].iloc[0] == 'Q1'  # January
        assert features['quarter'].iloc[3] == 'Q2'  # April

    def test_month_start_end_features(self):
        """Test is_month_start and is_month_end features."""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-15', '2024-01-31'])
        features = create_time_features(dates, ['is_month_start', 'is_month_end'])

        assert 'is_month_start' in features.columns
        assert 'is_month_end' in features.columns

        assert features['is_month_start'].iloc[0] == 'True'  # Jan 1
        assert features['is_month_start'].iloc[1] == 'False'  # Jan 15
        assert features['is_month_end'].iloc[2] == 'True'  # Jan 31

    def test_hour_feature(self):
        """Test hour feature with datetime."""
        dates = pd.date_range('2024-01-01 00:00', periods=24, freq='H')
        features = create_time_features(dates, ['hour'])

        assert 'hour' in features.columns
        assert features['hour'].iloc[0] == 0
        assert features['hour'].iloc[12] == 12
        assert features['hour'].iloc[23] == 23

    def test_default_features(self):
        """Test default features when none specified."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')
        features = create_time_features(dates)  # No features specified

        # Should get defaults
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'is_weekend' in features.columns

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        dates = pd.date_range('2024-01-01', periods=7, freq='D')
        features = create_time_features(dates, ['day_of_week'])

        assert isinstance(features, pd.DataFrame)
        assert features.index.equals(dates)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
