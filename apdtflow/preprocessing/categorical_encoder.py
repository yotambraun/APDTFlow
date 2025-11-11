"""
Categorical Feature Encoding for Time Series Forecasting

Handles encoding of categorical variables (holidays, day-of-week, product categories)
into numerical representations that can be used by forecasting models.

Supports:
- One-hot encoding (simple, interpretable)
- Learnable embeddings (efficient, captures relationships)
- Handling of unseen categories at test time
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union


class CategoricalEncoder:
    """
    Encodes categorical variables for time series forecasting.

    Supports both one-hot encoding and learnable embeddings. Handles
    unseen categories gracefully during test time.

    Parameters
    ----------
    encoding_type : str, default='onehot'
        Type of encoding: 'onehot' or 'embedding'
    embedding_dim : int, default=8
        Dimension for embedding vectors (only used if encoding_type='embedding')
    handle_unknown : str, default='indicator'
        How to handle unknown categories:
        - 'indicator': Add special "unknown" category
        - 'ignore': Set to zeros
        - 'error': Raise ValueError

    Examples
    --------
    >>> encoder = CategoricalEncoder(encoding_type='onehot')
    >>> data = pd.DataFrame({
    ...     'day_of_week': ['Mon', 'Tue', 'Wed', 'Mon'],
    ...     'holiday': ['Yes', 'No', 'No', 'Yes']
    ... })
    >>> encoded = encoder.fit_transform(data)
    >>> encoded.shape
    (4, 9)  # 7 days + 2 holidays
    """

    def __init__(
        self,
        encoding_type: str = 'onehot',
        embedding_dim: int = 8,
        handle_unknown: str = 'indicator'
    ):
        if encoding_type not in ['onehot', 'embedding']:
            raise ValueError(f"encoding_type must be 'onehot' or 'embedding', got {encoding_type}")

        if handle_unknown not in ['indicator', 'ignore', 'error']:
            raise ValueError(f"handle_unknown must be 'indicator', 'ignore', or 'error', got {handle_unknown}")

        self.encoding_type = encoding_type
        self.embedding_dim = embedding_dim
        self.handle_unknown = handle_unknown

        # Fitted state
        self.category_mappings_: Dict[str, Dict[str, int]] = {}
        self.column_names_: List[str] = []
        self.is_fitted_ = False

        # For embedding type
        self.embedding_layers_: Optional[nn.ModuleDict] = None

    def fit(self, data: Union[pd.DataFrame, dict]) -> 'CategoricalEncoder':
        """
        Learn category mappings from training data.

        Parameters
        ----------
        data : DataFrame or dict
            Categorical data to learn from

        Returns
        -------
        self : CategoricalEncoder
            Fitted encoder
        """
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        self.column_names_ = list(data.columns)
        self.category_mappings_ = {}

        for col in self.column_names_:
            unique_categories = data[col].unique().tolist()

            # Sort for consistency
            unique_categories = sorted([str(cat) for cat in unique_categories])

            # Create mapping: category -> index
            mapping = {cat: idx for idx, cat in enumerate(unique_categories)}

            # Add unknown category if needed
            if self.handle_unknown == 'indicator':
                mapping['__UNKNOWN__'] = len(mapping)

            self.category_mappings_[col] = mapping

        # Create embedding layers if needed
        if self.encoding_type == 'embedding':
            self.embedding_layers_ = nn.ModuleDict()
            for col in self.column_names_:
                num_categories = len(self.category_mappings_[col])
                self.embedding_layers_[col] = nn.Embedding(
                    num_categories,
                    self.embedding_dim
                )

        self.is_fitted_ = True
        return self

    def transform(self, data: Union[pd.DataFrame, dict]) -> np.ndarray:
        """
        Transform categorical data to numerical encoding.

        Parameters
        ----------
        data : DataFrame or dict
            Categorical data to transform

        Returns
        -------
        encoded : np.ndarray
            Encoded features (samples, encoded_dims)
        """
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted before transform. Call fit() first.")

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # Verify columns
        missing_cols = [col for col in self.column_names_ if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in transform data: {missing_cols}")

        if self.encoding_type == 'onehot':
            return self._transform_onehot(data)
        else:  # embedding
            return self._transform_embedding(data)

    def fit_transform(self, data: Union[pd.DataFrame, dict]) -> np.ndarray:
        """
        Fit encoder and transform data in one step.

        Parameters
        ----------
        data : DataFrame or dict
            Categorical data

        Returns
        -------
        encoded : np.ndarray
            Encoded features
        """
        self.fit(data)
        return self.transform(data)

    def _transform_onehot(self, data: pd.DataFrame) -> np.ndarray:
        """Transform using one-hot encoding."""
        encoded_cols = []

        for col in self.column_names_:
            mapping = self.category_mappings_[col]
            num_categories = len(mapping)

            # Convert to indices
            categories = data[col].astype(str).values
            indices = np.array([
                self._get_category_index(cat, col) for cat in categories
            ])

            # One-hot encode
            onehot = np.zeros((len(indices), num_categories), dtype=np.float32)
            for i, idx in enumerate(indices):
                if idx >= 0:  # -1 means unknown with 'ignore' strategy
                    onehot[i, idx] = 1.0

            encoded_cols.append(onehot)

        # Concatenate all encoded columns
        return np.concatenate(encoded_cols, axis=1)

    def _transform_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Transform using learned embeddings."""
        if self.embedding_layers_ is None:
            raise RuntimeError("Embedding layers not initialized")

        encoded_cols = []

        for col in self.column_names_:
            categories = data[col].astype(str).values
            indices = np.array([
                self._get_category_index(cat, col) for cat in categories
            ])

            # Handle unknown categories (set to 0 if ignore)
            indices = np.where(indices < 0, 0, indices)

            # Get embeddings
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            embeddings = self.embedding_layers_[col](indices_tensor)
            embeddings_np = embeddings.detach().cpu().numpy()

            encoded_cols.append(embeddings_np)

        # Concatenate all embeddings
        return np.concatenate(encoded_cols, axis=1)

    def _get_category_index(self, category: str, column: str) -> int:
        """
        Get index for a category, handling unknowns.

        Returns:
            index (int): Category index, or -1 if unknown with 'ignore' strategy
        """
        mapping = self.category_mappings_[column]
        category_str = str(category)

        if category_str in mapping:
            return mapping[category_str]
        else:
            # Unknown category
            if self.handle_unknown == 'indicator':
                return mapping['__UNKNOWN__']
            elif self.handle_unknown == 'ignore':
                return -1
            else:  # error
                raise ValueError(
                    f"Unknown category '{category}' in column '{column}'. "
                    f"Known categories: {list(mapping.keys())}"
                )

    def get_feature_names(self) -> List[str]:
        """
        Get names of encoded features.

        Returns
        -------
        feature_names : list of str
            Names of output features
        """
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted before getting feature names")

        feature_names = []

        if self.encoding_type == 'onehot':
            for col in self.column_names_:
                mapping = self.category_mappings_[col]
                # Sort by index to get correct order
                sorted_cats = sorted(mapping.items(), key=lambda x: x[1])
                for cat, _ in sorted_cats:
                    feature_names.append(f"{col}_{cat}")
        else:  # embedding
            for col in self.column_names_:
                for i in range(self.embedding_dim):
                    feature_names.append(f"{col}_emb{i}")

        return feature_names

    def get_num_features(self) -> int:
        """
        Get number of output features after encoding.

        Returns
        -------
        num_features : int
            Number of encoded features
        """
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted first")

        if self.encoding_type == 'onehot':
            return sum(len(mapping) for mapping in self.category_mappings_.values())
        else:  # embedding
            return len(self.column_names_) * self.embedding_dim

    def get_config(self) -> dict:
        """Get configuration for saving/loading."""
        return {
            'encoding_type': self.encoding_type,
            'embedding_dim': self.embedding_dim,
            'handle_unknown': self.handle_unknown,
            'category_mappings': self.category_mappings_,
            'column_names': self.column_names_,
            'is_fitted': self.is_fitted_
        }

    @classmethod
    def from_config(cls, config: dict) -> 'CategoricalEncoder':
        """Create encoder from configuration."""
        encoder = cls(
            encoding_type=config['encoding_type'],
            embedding_dim=config['embedding_dim'],
            handle_unknown=config['handle_unknown']
        )
        encoder.category_mappings_ = config['category_mappings']
        encoder.column_names_ = config['column_names']
        encoder.is_fitted_ = config['is_fitted']

        # Recreate embeddings if needed
        if config['encoding_type'] == 'embedding' and encoder.is_fitted_:
            encoder.embedding_layers_ = nn.ModuleDict()
            for col in encoder.column_names_:
                num_categories = len(encoder.category_mappings_[col])
                encoder.embedding_layers_[col] = nn.Embedding(
                    num_categories,
                    encoder.embedding_dim
                )

        return encoder


def create_time_features(
    dates: pd.DatetimeIndex,
    include_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create categorical time-based features from dates.

    Useful for creating holiday indicators, day-of-week, etc.

    Parameters
    ----------
    dates : DatetimeIndex
        Date/time index
    include_features : list of str, optional
        Features to include. Options:
        - 'day_of_week': Monday=0, Sunday=6
        - 'day_name': 'Monday', 'Tuesday', ...
        - 'month': 1-12
        - 'month_name': 'January', 'February', ...
        - 'quarter': Q1, Q2, Q3, Q4
        - 'is_weekend': True/False
        - 'hour': 0-23 (if datetime)
        - 'is_month_start': True/False
        - 'is_month_end': True/False

    Returns
    -------
    features : DataFrame
        Time-based categorical features

    Examples
    --------
    >>> dates = pd.date_range('2024-01-01', periods=7, freq='D')
    >>> features = create_time_features(dates, ['day_name', 'is_weekend'])
    >>> features.head()
       day_name  is_weekend
    0    Monday       False
    1   Tuesday       False
    2 Wednesday       False
    """
    if include_features is None:
        include_features = ['day_of_week', 'month', 'is_weekend']

    features = {}

    if 'day_of_week' in include_features:
        features['day_of_week'] = dates.dayofweek

    if 'day_name' in include_features:
        features['day_name'] = dates.day_name()

    if 'month' in include_features:
        features['month'] = dates.month

    if 'month_name' in include_features:
        features['month_name'] = dates.month_name()

    if 'quarter' in include_features:
        features['quarter'] = 'Q' + dates.quarter.astype(str)

    if 'is_weekend' in include_features:
        features['is_weekend'] = (dates.dayofweek >= 5).astype(str)

    if 'hour' in include_features:
        if hasattr(dates, 'hour'):
            features['hour'] = dates.hour
        else:
            features['hour'] = 0  # Default for date-only

    if 'is_month_start' in include_features:
        features['is_month_start'] = dates.is_month_start.astype(str)

    if 'is_month_end' in include_features:
        features['is_month_end'] = dates.is_month_end.astype(str)

    return pd.DataFrame(features, index=dates)
