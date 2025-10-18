"""
Exogenous variable processing and fusion for APDTFlow.

Handles both past-observed and future-known covariates to improve
forecasting accuracy by incorporating external information.

Based on research from:
- TimeXer (2024): Gate-based exogenous integration
- ChronosX (2025): Adapter-based covariate fusion
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


class ExogenousFeatureFusion(nn.Module):
    """
    Fuses target series with exogenous variables using learned gating.

    This module learns how to optimally combine the target time series
    with external features (like weather, holidays, promotions) to improve
    forecasting accuracy.

    Args:
        hidden_dim: Hidden dimension for fusion
        num_exog_features: Number of exogenous variables
        fusion_type: 'concat', 'gated', or 'attention'

    Example:
        >>> fusion = ExogenousFeatureFusion(hidden_dim=32, num_exog_features=3, fusion_type='gated')
        >>> target = torch.randn(8, 1, 30)  # batch=8, channels=1, time=30
        >>> exog = torch.randn(8, 3, 30)    # batch=8, exog_features=3, time=30
        >>> fused = fusion(target, exog)
        >>> fused.shape
        torch.Size([8, 1, 30])
    """

    def __init__(self, hidden_dim: int, num_exog_features: int, fusion_type: str = 'gated'):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_exog = num_exog_features
        self.hidden_dim = hidden_dim

        if fusion_type == 'concat':
            # Simple concatenation followed by projection
            self.fusion = nn.Sequential(
                nn.Linear(1 + num_exog_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == 'gated':
            # Gated fusion (recommended for best performance)
            self.target_encoder = nn.Linear(1, hidden_dim)
            self.exog_encoder = nn.Linear(num_exog_features, hidden_dim)

            # Gate network learns importance of exog vs target
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )

            # Fusion network
            self.fusion_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
            self.output_proj = nn.Linear(hidden_dim, 1)

        elif fusion_type == 'attention':
            # Attention-based fusion (most sophisticated)
            self.target_encoder = nn.Linear(1, hidden_dim)
            self.exog_encoder = nn.Linear(num_exog_features, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.output_proj = nn.Linear(hidden_dim, 1)

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose 'concat', 'gated', or 'attention'")

    def forward(self, target: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        """
        Fuse target series with exogenous variables.

        Args:
            target: (batch, 1, T) - target time series
            exog: (batch, num_exog, T) - exogenous variables

        Returns:
            fused: (batch, 1, T) - fused representation
        """
        if self.fusion_type == 'concat':
            # Simple concatenation
            combined = torch.cat([target, exog], dim=1)  # (batch, 1+num_exog, T)
            combined_t = combined.transpose(1, 2)  # (batch, T, 1+num_exog)
            fused_t = self.fusion(combined_t)  # (batch, T, 1)
            return fused_t.transpose(1, 2)  # (batch, 1, T)

        elif self.fusion_type == 'gated':
            # Encode target and exog separately
            target_t = target.transpose(1, 2)  # (batch, T, 1)
            exog_t = exog.transpose(1, 2)  # (batch, T, num_exog)

            target_enc = self.target_encoder(target_t)  # (batch, T, hidden)
            exog_enc = self.exog_encoder(exog_t)  # (batch, T, hidden)

            # Compute gate: how much to use exog vs target
            combined = torch.cat([target_enc, exog_enc], dim=-1)
            gate = self.gate(combined)  # (batch, T, hidden)

            # Gated fusion: weighted combination
            fused = gate * exog_enc + (1 - gate) * target_enc

            # Combine features
            combined_features = torch.cat([target_enc, fused], dim=-1)
            fused_features = self.fusion_net(combined_features)

            # Project to output
            output = self.output_proj(fused_features)  # (batch, T, 1)
            return output.transpose(1, 2)  # (batch, 1, T)

        elif self.fusion_type == 'attention':
            # Attention: target queries exog
            target_t = target.transpose(1, 2)  # (batch, T, 1)
            exog_t = exog.transpose(1, 2)  # (batch, T, num_exog)

            target_enc = self.target_encoder(target_t)  # (batch, T, hidden)
            exog_enc = self.exog_encoder(exog_t)  # (batch, T, hidden)

            # Self-attention: target attends to exog
            attn_out, _ = self.attention(
                target_enc,  # query
                exog_enc,    # key
                exog_enc     # value
            )  # (batch, T, hidden)

            output = self.output_proj(attn_out)  # (batch, T, 1)
            return output.transpose(1, 2)  # (batch, 1, T)


class ExogenousProcessor:
    """
    Handles preprocessing and validation of exogenous variables.

    Utility class for preparing exogenous features for model training
    and prediction.
    """

    @staticmethod
    def validate_exog_data(
        past_exog: Optional[np.ndarray],
        future_exog: Optional[np.ndarray],
        forecast_horizon: int
    ) -> bool:
        """
        Validate exogenous variable data.

        Args:
            past_exog: Past observed exog (only available historically)
            future_exog: Future known exog (available for forecast period)
            forecast_horizon: Number of future steps

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        if future_exog is not None:
            if len(future_exog) < forecast_horizon:
                raise ValueError(
                    f"future_exog must have at least {forecast_horizon} rows, "
                    f"got {len(future_exog)}"
                )

        if past_exog is not None and future_exog is not None:
            if past_exog.shape[1] != future_exog.shape[1]:
                raise ValueError(
                    f"past_exog and future_exog must have same number of features. "
                    f"Got {past_exog.shape[1]} and {future_exog.shape[1]}"
                )

        return True

    @staticmethod
    def prepare_exog_features(
        df: pd.DataFrame,
        exog_cols: Optional[list],
        future_exog_cols: Optional[list]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract exogenous features from DataFrame.

        Args:
            df: DataFrame containing all data
            exog_cols: All exogenous column names
            future_exog_cols: Subset that are known in future

        Returns:
            (past_exog, future_exog): Arrays of exogenous features
        """
        past_exog = None
        future_exog = None

        if exog_cols:
            past_exog = df[exog_cols].values.astype(np.float32)

        if future_exog_cols:
            future_exog = df[future_exog_cols].values.astype(np.float32)

        return past_exog, future_exog

    @staticmethod
    def normalize_exog(
        exog: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize exogenous variables.

        Args:
            exog: Exogenous features (samples, features)
            mean: Optional pre-computed mean
            std: Optional pre-computed std

        Returns:
            (normalized_exog, mean, std)
        """
        if mean is None or std is None:
            mean = np.mean(exog, axis=0, keepdims=True)
            std = np.std(exog, axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero

        normalized = (exog - mean) / std
        return normalized, mean, std

    @staticmethod
    def prepare_future_exog_for_prediction(
        exog_future: Union[pd.DataFrame, np.ndarray],
        future_exog_cols: list,
        forecast_horizon: int,
        exog_mean: np.ndarray,
        exog_std: np.ndarray
    ) -> np.ndarray:
        """
        Prepare future exogenous data for prediction.

        Args:
            exog_future: Future exogenous values (DataFrame or array)
            future_exog_cols: Expected column names
            forecast_horizon: Number of steps to predict
            exog_mean: Normalization mean
            exog_std: Normalization std

        Returns:
            Normalized future exog array (forecast_horizon, num_features)
        """
        if isinstance(exog_future, pd.DataFrame):
            if not all(col in exog_future.columns for col in future_exog_cols):
                missing = [col for col in future_exog_cols if col not in exog_future.columns]
                raise ValueError(f"Missing columns in exog_future: {missing}")
            exog_array = exog_future[future_exog_cols].values.astype(np.float32)
        else:
            exog_array = np.array(exog_future).astype(np.float32)

        if len(exog_array) < forecast_horizon:
            raise ValueError(
                f"exog_future must have at least {forecast_horizon} rows, "
                f"got {len(exog_array)}"
            )

        # Take only what we need
        exog_array = exog_array[:forecast_horizon]

        # Normalize
        normalized = (exog_array - exog_mean) / exog_std

        return normalized
