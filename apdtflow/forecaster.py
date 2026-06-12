"""
High-Level API for APDTFlow models.
Provides a simple fit/predict interface for easy time series forecasting.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional, Union, List, Tuple, Dict
import warnings
from tqdm import tqdm

from .models.apdtflow import APDTFlow
from .models.transformer_forecaster import TransformerForecaster
from .models.tcn_forecaster import TCNForecaster
from .models.ensemble_forecaster import EnsembleForecaster
from .conformal import (
    AdaptiveConformalPredictor,
    SplitConformalPredictor,
    crossing_time_quantile,
    per_step_quantiles,
)
from .event_time import (
    PredictWhenResult,
    batch_first_crossing_times,
    first_crossing_time,
)
from .logger_util import get_logger
from .preprocessing.categorical_encoder import CategoricalEncoder

logger = get_logger("apdtflow.forecaster")


class APDTFlowForecaster:
    """
    Easy-to-use time series forecaster powered by APDTFlow.

    Simple interface for forecasting with Neural ODEs and multi-scale decomposition.

    Parameters
    ----------
    model_type : str, default='apdtflow'
        Type of forecasting model: 'apdtflow', 'transformer', 'tcn', or 'ensemble'
    num_scales : int, default=3
        Number of scales for multi-scale decomposition
    hidden_dim : int, default=16
        Hidden dimension size
    filter_size : int, default=5
        Convolutional filter size
    forecast_horizon : int, default=7
        Number of steps to forecast
    history_length : int, default=30
        Number of historical steps to use
    learning_rate : float, default=0.001
        Learning rate for training
    batch_size : int, default=32
        Batch size for training
    num_epochs : int, default=50
        Number of training epochs
    device : str, optional
        Device to use ('cuda' or 'cpu'). Auto-detected if None
    use_embedding : bool, default=True
        Use learnable time series embedding
    verbose : bool, default=True
        Print training progress

    Examples
    --------
    >>> import pandas as pd
    >>> from apdtflow.forecaster import APDTFlowForecaster
    >>>
    >>> # Load your data
    >>> df = pd.read_csv('data.csv', parse_dates=['date'])
    >>>
    >>> # Create and train model
    >>> model = APDTFlowForecaster(forecast_horizon=7, history_length=30)
    >>> model.fit(df, target_col='sales', date_col='date')
    >>>
    >>> # Make predictions
    >>> predictions = model.predict(steps=7)
    >>>
    >>> # Visualize
    >>> model.plot_forecast(with_history=50, show_uncertainty=True)
    """

    def __init__(
        self,
        model_type: str = 'apdtflow',  # 'apdtflow', 'transformer', 'tcn'
        num_scales: int = 3,
        hidden_dim: int = 16,
        filter_size: int = 5,
        forecast_horizon: int = 7,
        history_length: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 50,
        device: Optional[str] = None,
        use_embedding: bool = True,
        verbose: bool = True,
        loss_type: str = 'mse',
        ode_method: str = 'rk4',
        decoder_type: str = 'transformer',
        # NEW: Exogenous variables support
        exog_fusion_type: str = 'gated',
        # NEW: Conformal prediction support
        use_conformal: bool = False,
        conformal_method: str = 'split',
        calibration_split: float = 0.2,
        # NEW: Early stopping
        early_stopping: bool = False,
        patience: int = 5,
        validation_split: float = 0.2,
        # NEW: Categorical variables support
        categorical_encoding: str = 'onehot'
    ):
        self.model_type = model_type
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.forecast_horizon = forecast_horizon
        self.history_length = history_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.use_embedding = use_embedding

        # Training loss and solver configuration (v0.4.0)
        if loss_type not in ('mse', 'nll'):
            raise ValueError(f"loss_type must be 'mse' or 'nll', got {loss_type!r}")
        if ode_method not in ('rk4', 'dopri5_adjoint'):
            raise ValueError(
                f"ode_method must be 'rk4' or 'dopri5_adjoint', got {ode_method!r}"
            )
        if decoder_type not in ('transformer', 'continuous'):
            raise ValueError(
                f"decoder_type must be 'transformer' or 'continuous', got {decoder_type!r}"
            )
        self.loss_type = loss_type
        self.ode_method = ode_method
        self.decoder_type = decoder_type

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split

        # Exogenous variables (NEW)
        self.exog_fusion_type = exog_fusion_type
        self.exog_cols_: Optional[List[str]] = None
        self.future_exog_cols_: Optional[List[str]] = None
        self.num_exog_features_ = 0
        self.exog_mean_: Optional[np.ndarray] = None
        self.exog_std_: Optional[np.ndarray] = None
        self.has_exog_ = False
        self.has_numerical_exog_ = False  # Track if we have true numerical exog (vs just categorical)

        # Categorical variables (NEW in v0.2.3)
        self.categorical_encoding = categorical_encoding
        self.categorical_cols_: Optional[List[str]] = None
        self.categorical_encoder_: Optional['CategoricalEncoder'] = None
        self.has_categorical_ = False

        # Conformal prediction (NEW)
        self.use_conformal = use_conformal
        self.conformal_method = conformal_method
        self.calibration_split = calibration_split
        self.conformal_predictor: Optional[Union[SplitConformalPredictor, AdaptiveConformalPredictor]] = None

        # Multivariate input (v0.4.0): set in fit(feature_cols=...).
        self.feature_cols_: Optional[List[str]] = None
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_stds_: Optional[np.ndarray] = None

        # Continuous-time state (v0.4.0): per-step calibration data for
        # predict_at intervals and predict_when time-window calibration.
        self._calib_X_: Optional[torch.Tensor] = None
        self._calib_exog_: Optional[torch.Tensor] = None
        self._calib_targets_grid_: Optional[np.ndarray] = None
        self._calib_residuals_: Optional[np.ndarray] = None
        self._calib_anchor_: Optional[np.ndarray] = None
        self._when_cache_: Dict = {}
        self.last_date_: Optional[pd.Timestamp] = None
        self.freq_delta_: Optional[pd.Timedelta] = None

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model will be initialized in fit()
        self.model: Optional[Union[APDTFlow, TransformerForecaster, TCNForecaster, EnsembleForecaster]] = None
        self._is_fitted = False

        # Store data info
        self.scaler_mean_: Optional[float] = None
        self.scaler_std_: Optional[float] = None
        self.last_sequence_: Optional[np.ndarray] = None
        self.last_exog_sequence_: Optional[np.ndarray] = None  # NEW
        self.target_col_: Optional[str] = None
        self.date_col_: Optional[str] = None
        self.data_df_: Optional[pd.DataFrame] = None

        # Residual analysis (NEW in v0.3.0)
        self.residuals_: Optional[np.ndarray] = None
        self.residual_actuals_: Optional[np.ndarray] = None
        self.residual_predictions_: Optional[np.ndarray] = None

    def _initialize_model(self):
        """Initialize the forecasting model based on model_type."""
        if self.model_type == 'apdtflow':
            model_params = {
                'num_scales': self.num_scales,
                'input_channels': 1,
                'filter_size': self.filter_size,
                'hidden_dim': self.hidden_dim,
                'output_dim': 1,
                'forecast_horizon': self.forecast_horizon,
                'history_length': self.history_length,
                'use_embedding': self.use_embedding,
                'num_exog_features': self.num_exog_features_,
                'exog_fusion_type': self.exog_fusion_type,
                'ode_method': self.ode_method,
                'decoder_type': self.decoder_type,
                'n_input_channels': 1 + len(self.feature_cols_ or [])
            }
            model = APDTFlow(**model_params)
        elif self.model_type == 'transformer':
            model_params = {
                'input_dim': 1,
                'model_dim': self.hidden_dim,
                'num_layers': 2,
                'nhead': 4,
                'forecast_horizon': self.forecast_horizon
            }
            model = TransformerForecaster(**model_params)
        elif self.model_type == 'tcn':
            model_params = {
                'input_channels': 1,
                'num_channels': [self.hidden_dim] * 3,
                'kernel_size': self.filter_size,
                'forecast_horizon': self.forecast_horizon
            }
            model = TCNForecaster(**model_params)
        elif self.model_type == 'ensemble':
            # Create ensemble of APDTFlow, Transformer, and TCN
            apdtflow_model = APDTFlow(
                num_scales=self.num_scales,
                input_channels=1,
                filter_size=self.filter_size,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                forecast_horizon=self.forecast_horizon,
                history_length=self.history_length,
                use_embedding=self.use_embedding,
                num_exog_features=self.num_exog_features_,
                exog_fusion_type=self.exog_fusion_type
            )
            transformer_model = TransformerForecaster(
                input_dim=1,
                model_dim=self.hidden_dim,
                num_layers=2,
                nhead=4,
                forecast_horizon=self.forecast_horizon
            )
            tcn_model = TCNForecaster(
                input_channels=1,
                num_channels=[self.hidden_dim] * 3,
                kernel_size=self.filter_size,
                forecast_horizon=self.forecast_horizon
            )
            model = EnsembleForecaster(
                models=[apdtflow_model, transformer_model, tcn_model]
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def _validate_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ):
        """
        Validate input data and provide helpful error messages.

        Raises
        ------
        ValueError
            If data is invalid with specific error message
        """
        # Check data is not empty
        if isinstance(data, pd.DataFrame):
            if len(data) == 0:
                raise ValueError("Input DataFrame is empty. Please provide non-empty data.")

            # Check minimum length
            min_length = self.history_length + self.forecast_horizon + 10
            if len(data) < min_length:
                raise ValueError(
                    f"Data too short. Got {len(data)} rows, need at least {min_length} "
                    f"(history_length={self.history_length} + forecast_horizon={self.forecast_horizon} + buffer=10)"
                )

            # Check target column exists
            if target_col and target_col not in data.columns:
                available = ", ".join(data.columns.tolist())
                raise ValueError(
                    f"Column '{target_col}' not found in DataFrame.\n"
                    f"Available columns: [{available}]"
                )

            # Check date column exists
            if date_col and date_col not in data.columns:
                available = ", ".join(data.columns.tolist())
                raise ValueError(
                    f"Date column '{date_col}' not found in DataFrame.\n"
                    f"Available columns: [{available}]"
                )

            # Check exog columns exist
            if exog_cols:
                missing_exog = [col for col in exog_cols if col not in data.columns]
                if missing_exog:
                    available = ", ".join(data.columns.tolist())
                    raise ValueError(
                        f"Exogenous columns not found: {missing_exog}\n"
                        f"Available columns: [{available}]"
                    )

            # Check categorical columns exist
            if categorical_cols:
                missing_cat = [col for col in categorical_cols if col not in data.columns]
                if missing_cat:
                    available = ", ".join(data.columns.tolist())
                    raise ValueError(
                        f"Categorical columns not found: {missing_cat}\n"
                        f"Available columns: [{available}]"
                    )

            # Check for NaN values in target
            if target_col and data[target_col].isna().any():
                nan_indices = data[data[target_col].isna()].index.tolist()
                raise ValueError(
                    f"Target column '{target_col}' contains {len(nan_indices)} NaN values.\n"
                    f"NaN found at indices: {nan_indices[:10]}{'...' if len(nan_indices) > 10 else ''}\n"
                    f"Please handle missing values before training (use fillna(), interpolate(), or drop)."
                )

            # Check for inf values in target
            if target_col and np.isinf(data[target_col]).any():
                inf_indices = data[np.isinf(data[target_col])].index.tolist()
                raise ValueError(
                    f"Target column '{target_col}' contains infinite values at indices: {inf_indices}\n"
                    f"Please replace infinite values before training."
                )

            # Check for NaN in exog columns
            if exog_cols:
                for col in exog_cols:
                    if data[col].isna().any():
                        nan_count = data[col].isna().sum()
                        raise ValueError(
                            f"Exogenous column '{col}' contains {nan_count} NaN values.\n"
                            f"Please handle missing values in all exogenous columns."
                        )

        else:
            # numpy array validation
            arr = np.array(data).flatten()
            if len(arr) == 0:
                raise ValueError("Input array is empty.")

            min_length = self.history_length + self.forecast_horizon + 10
            if len(arr) < min_length:
                raise ValueError(
                    f"Data too short. Got {len(arr)} points, need at least {min_length}"
                )

            if np.isnan(arr).any():
                nan_count = np.isnan(arr).sum()
                nan_indices = list(np.where(np.isnan(arr))[0])
                raise ValueError(
                    f"Input array contains {nan_count} NaN values at indices: "
                    f"{nan_indices[:10]}{'...' if len(nan_indices) > 10 else ''}"
                )

            if np.isinf(arr).any():
                inf_count = np.isinf(arr).sum()
                raise ValueError(f"Input array contains {inf_count} infinite values.")

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare data for training.

        Parameters
        ----------
        data : DataFrame or array
            Time series data
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame

        Returns
        -------
        X : torch.Tensor
            Input sequences (batch_size, 1, history_length)
        y : torch.Tensor
            Target sequences (batch_size, 1, forecast_horizon)
        exog_X : torch.Tensor, optional
            Exogenous input sequences (batch_size, num_exog, history_length)
        exog_y : torch.Tensor, optional
            Exogenous target sequences (batch_size, num_exog, forecast_horizon)
        """
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col must be specified for DataFrame input")
            if date_col and date_col in data.columns:
                data = data.sort_values(date_col)
                dates = pd.to_datetime(data[date_col])
                self.last_date_ = dates.iloc[-1]
                if len(dates) > 1:
                    self.freq_delta_ = pd.Timedelta(np.median(np.diff(dates.values)))
            series = data[target_col].values

            # Handle exogenous variables (including categorical)
            if hasattr(self, '_has_combined_exog') and self._has_combined_exog:
                # Use combined numerical + categorical features
                exog_data = self._combined_exog_data
                # Normalize exog data
                self.exog_mean_ = np.mean(exog_data, axis=0)
                self.exog_std_ = np.std(exog_data, axis=0)
                self.exog_std_[self.exog_std_ == 0] = 1.0
                exog_norm = (exog_data - self.exog_mean_) / self.exog_std_
            elif self.has_exog_ and self.exog_cols_:
                # Only numerical exog features
                exog_data = data[self.exog_cols_].values
                # Normalize exog data
                self.exog_mean_ = np.mean(exog_data, axis=0)
                self.exog_std_ = np.std(exog_data, axis=0)
                self.exog_std_[self.exog_std_ == 0] = 1.0
                exog_norm = (exog_data - self.exog_mean_) / self.exog_std_
            else:
                exog_norm = None
        else:
            series = np.array(data).flatten()
            exog_norm = None

        # Normalize target
        self.scaler_mean_ = np.mean(series)
        self.scaler_std_ = np.std(series)
        if self.scaler_std_ == 0:
            self.scaler_std_ = 1.0
        series_norm = (series - self.scaler_mean_) / self.scaler_std_

        # Create sliding windows
        X_list, y_list = [], []
        exog_X_list: List[np.ndarray] = []
        exog_y_list: List[np.ndarray] = []
        total_length = self.history_length + self.forecast_horizon

        for i in range(len(series_norm) - total_length + 1):
            X_list.append(series_norm[i:i + self.history_length])
            y_list.append(series_norm[i + self.history_length:i + total_length])

            if exog_norm is not None:
                exog_X_list.append(exog_norm[i:i + self.history_length])
                exog_y_list.append(exog_norm[i + self.history_length:i + total_length])

        if len(X_list) == 0:
            raise ValueError(
                f"Data too short. Need at least {total_length} points, "
                f"got {len(series_norm)}"
            )

        X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

        # Store last sequence for prediction
        self.last_sequence_ = series_norm[-self.history_length:]

        if exog_norm is not None:
            exog_X = torch.tensor(np.array(exog_X_list), dtype=torch.float32).transpose(1, 2)
            exog_y = torch.tensor(np.array(exog_y_list), dtype=torch.float32).transpose(1, 2)
            self.last_exog_sequence_ = exog_norm[-self.history_length:]
            return X, y, exog_X, exog_y

        return X, y

    def _prepare_data_multivariate(
        self,
        data: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare multichannel training data for fit(feature_cols=...).

        Channel 0 is the target; remaining channels are the feature
        columns. Each channel is independently z-normalized.
        """
        if date_col and date_col in data.columns:
            data = data.sort_values(date_col)
            dates = pd.to_datetime(data[date_col])
            self.last_date_ = dates.iloc[-1]
            if len(dates) > 1:
                self.freq_delta_ = pd.Timedelta(np.median(np.diff(dates.values)))

        channels = [target_col] + list(self.feature_cols_ or [])
        matrix = data[channels].to_numpy(dtype=float).T  # (C, N)
        means = matrix.mean(axis=1)
        stds = matrix.std(axis=1)
        stds[stds == 0] = 1.0
        norm = (matrix - means[:, None]) / stds[:, None]

        self.feature_means_ = means
        self.feature_stds_ = stds
        self.scaler_mean_ = float(means[0])
        self.scaler_std_ = float(stds[0])

        total_length = self.history_length + self.forecast_horizon
        n_windows = norm.shape[1] - total_length + 1
        if n_windows <= 0:
            raise ValueError(
                f"Data too short. Need at least {total_length} points, "
                f"got {norm.shape[1]}"
            )
        X = np.stack([norm[:, i:i + self.history_length] for i in range(n_windows)])
        y = np.stack([
            norm[0, i + self.history_length:i + total_length]
            for i in range(n_windows)
        ])
        self.last_sequence_ = norm[:, -self.history_length:]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        exog_cols: Optional[List[str]] = None,
        future_exog_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        future_categorical_cols: Optional[List[str]] = None,
        log_callback: Optional[Callable] = None
    ):
        """
        Fit the forecasting model.

        Parameters
        ----------
        data : DataFrame or array
            Time series data
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame
        feature_cols : list of str, optional
            Additional input series channels (e.g. sensor columns) fused
            into a learned health indicator in front of the pipeline
            (NEW in v0.4.0). The forecast target remains ``target_col``;
            inspect the learned fusion weights via ``sensor_importance_``.
            Requires ``model_type='apdtflow'``; cannot be combined with
            ``exog_cols`` or ``categorical_cols``.
        exog_cols : list of str, optional
            Names of numerical exogenous variable columns (NEW in v0.2.0)
        future_exog_cols : list of str, optional
            Subset of exog_cols that are known in future (NEW in v0.2.0)
        categorical_cols : list of str, optional
            Names of categorical variable columns (NEW in v0.2.3)
        future_categorical_cols : list of str, optional
            Subset of categorical_cols that are known in future (NEW in v0.2.3)

        Returns
        -------
        self : APDTFlowForecaster
            Fitted forecaster
        """
        if self.verbose:
            logger.info(f"Fitting {self.model_type} model...")
            logger.info(f"Device: {self.device}")

        # Validate model_type
        if self.model_type == 'ensemble':
            raise ValueError(
                "model_type='ensemble' is not currently supported in APDTFlowForecaster. "
                "Please use 'apdtflow', 'transformer', or 'tcn'."
            )

        if feature_cols:
            if self.model_type != 'apdtflow':
                raise ValueError(
                    "feature_cols is only supported with model_type='apdtflow'."
                )
            if exog_cols or categorical_cols:
                raise ValueError(
                    "feature_cols cannot be combined with exog_cols or "
                    "categorical_cols."
                )
            if not isinstance(data, pd.DataFrame):
                raise ValueError("feature_cols requires DataFrame input")
            missing = [c for c in feature_cols + [target_col] if c not in data.columns]
            if missing:
                raise ValueError(f"Columns not found in data: {missing}")
            self.feature_cols_ = list(feature_cols)

        # Validate data before proceeding
        self._validate_data(data, target_col, date_col, exog_cols, categorical_cols)

        # Store column names
        self.target_col_ = target_col
        self.date_col_ = date_col
        self.exog_cols_ = exog_cols
        self.future_exog_cols_ = future_exog_cols
        self.categorical_cols_ = categorical_cols

        # Handle categorical variables (NEW in v0.2.3)
        if categorical_cols:
            if self.model_type != 'apdtflow':
                raise ValueError(
                    f"Categorical variables are only supported with model_type='apdtflow'. "
                    f"Current model_type='{self.model_type}' does not support categorical_cols."
                )

            if not isinstance(data, pd.DataFrame):
                raise ValueError("categorical_cols requires DataFrame input")

            self.has_categorical_ = True

            # Create and fit categorical encoder
            self.categorical_encoder_ = CategoricalEncoder(
                encoding_type=self.categorical_encoding,
                embedding_dim=8,
                handle_unknown='indicator'
            )

            categorical_data = data[categorical_cols]
            self.categorical_encoder_.fit(categorical_data)

            # Encode categorical features
            categorical_encoded = self.categorical_encoder_.transform(categorical_data)
            num_categorical_features = categorical_encoded.shape[1]

            if self.verbose:
                logger.info(f"Encoded {len(categorical_cols)} categorical columns into {num_categorical_features} features")
                logger.info(f"  Categorical columns: {categorical_cols}")
                logger.info(f"  Encoding: {self.categorical_encoding}")

            # Add encoded categorical features to exog features
            if exog_cols:
                # Combine numerical exog with encoded categorical
                exog_data = data[exog_cols].values
                combined_exog = np.concatenate([exog_data, categorical_encoded], axis=1)

                # Update DataFrame with combined features
                # We'll handle this in _prepare_data by passing combined
                self._combined_exog_data = combined_exog
                self._has_combined_exog = True
                # Update num features to reflect combined numerical + categorical
                self.has_exog_ = True
                self.has_numerical_exog_ = True  # Track that we have true numerical exog
                self.num_exog_features_ = combined_exog.shape[1]
                if self.verbose:
                    logger.info(f"Combined {len(exog_cols)} numerical + {num_categorical_features} categorical = {self.num_exog_features_} total exogenous features")
            else:
                # Only categorical features (treat as exog)
                self._combined_exog_data = categorical_encoded
                self._has_combined_exog = True
                self.has_exog_ = True
                self.has_numerical_exog_ = False  # Only categorical, no numerical exog
                self.num_exog_features_ = categorical_encoded.shape[1]
                if self.verbose:
                    logger.info(f"Using {self.num_exog_features_} categorical features as exogenous variables")
                exog_cols = []  # Will use combined data instead

        # Check exogenous variables
        if exog_cols and not hasattr(self, '_has_combined_exog'):
            # Only set num_exog if not already set by categorical encoding above
            if self.model_type != 'apdtflow':
                raise ValueError(
                    f"Exogenous variables are only supported with model_type='apdtflow'. "
                    f"Current model_type='{self.model_type}' does not support exog_cols."
                )
            self.has_exog_ = True
            self.has_numerical_exog_ = True  # True numerical exogenous variables
            self.num_exog_features_ = len(exog_cols)
            if self.verbose:
                logger.info(f"Using {self.num_exog_features_} exogenous features: {exog_cols}")
                if future_exog_cols:
                    logger.info(f"  Future-known features: {future_exog_cols}")

        if isinstance(data, pd.DataFrame):
            self.data_df_ = data.copy()

        # Prepare data
        data_output: Union[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
        if self.feature_cols_:
            assert target_col is not None
            assert isinstance(data, pd.DataFrame)
            data_output = self._prepare_data_multivariate(data, target_col, date_col)
        else:
            data_output = self._prepare_data(data, target_col, date_col)
        exog_X: Optional[torch.Tensor] = None
        exog_y: Optional[torch.Tensor] = None
        if len(data_output) == 4:
            X, y, exog_X, exog_y = data_output  # type: ignore[misc]
            has_exog_data = True
        else:
            X, y = data_output  # type: ignore[misc]
            has_exog_data = False

        if self.verbose:
            logger.info(f"Created {len(X)} training samples")
            logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
            if has_exog_data and exog_X is not None:
                logger.info(f"Exog shape: {exog_X.shape}")

        # Initialize model
        self.model = self._initialize_model()
        assert self.model is not None, "Model initialization failed"

        # Create DataLoader(s) - split train/val if early stopping enabled
        # For non-APDTFlow models, reshape tensors to expected format
        if self.model_type == 'transformer':
            # Transformer expects (batch, 1, time, features) format
            X = X.unsqueeze(-1)  # (batch, 1, history_length) -> (batch, 1, history_length, 1)
            # y stays as (batch, 1, forecast_horizon) for proper transpose in model
        elif self.model_type == 'tcn':
            # TCN expects (batch, 1, channels, time) format
            X = X.unsqueeze(2)  # (batch, 1, history_length) -> (batch, 1, 1, history_length)
            y = y.unsqueeze(2)  # (batch, 1, forecast_horizon) -> (batch, 1, 1, forecast_horizon)

        if has_exog_data:
            assert exog_X is not None and exog_y is not None
            dataset = TensorDataset(X, y, exog_X, exog_y)
        else:
            dataset = TensorDataset(X, y)

        if self.early_stopping and self.validation_split > 0:
            # Split into train and validation
            n_samples = len(dataset)
            n_val = int(n_samples * self.validation_split)
            n_train = n_samples - n_val

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            if self.verbose:
                logger.info(f"Training samples: {n_train}, Validation samples: {n_val}")
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = None

        # Train - use different training loops for different model types
        if self.model_type == 'apdtflow':
            # APDTFlow-specific training with Neural ODE
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Early stopping tracking
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            best_model_state = None

            # Create progress bar
            pbar = tqdm(range(self.num_epochs), desc="Training", disable=not self.verbose)

            for epoch in pbar:
                self.model.train()
                epoch_loss = 0.0

                for batch_data in train_loader:
                    if has_exog_data:
                        x_batch, y_batch, exog_x_batch, exog_y_batch = batch_data
                        exog_x_batch = exog_x_batch.to(self.device)
                    else:
                        x_batch, y_batch = batch_data
                        exog_x_batch = None

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Create time span
                    t_span = torch.linspace(
                        0, 1,
                        steps=self.history_length,
                        device=self.device
                    )

                    # Forward pass
                    optimizer.zero_grad()
                    if self.decoder_type == 'continuous':
                        # Randomized-query training: off-grid accuracy is
                        # trained, not emergent.
                        assert isinstance(self.model, APDTFlow)
                        loss = self.model._continuous_training_loss(
                            x_batch, y_batch, t_span, exog=exog_x_batch
                        )
                    else:
                        preds, pred_logvars = self.model(x_batch, t_span, exog=exog_x_batch)
                        mse = (preds - y_batch.transpose(1, 2)) ** 2
                        if self.loss_type == 'nll':
                            loss = torch.mean(
                                0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
                            )
                        else:
                            loss = mse.mean()

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(x_batch)

                avg_train_loss = epoch_loss / float(len(train_loader.dataset))  # type: ignore[arg-type]

                # Validation if early stopping enabled
                if self.early_stopping and val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0

                    with torch.no_grad():
                        for batch_data in val_loader:
                            if has_exog_data:
                                x_batch, y_batch, exog_x_batch, exog_y_batch = batch_data
                                exog_x_batch = exog_x_batch.to(self.device)
                            else:
                                x_batch, y_batch = batch_data
                                exog_x_batch = None

                            x_batch = x_batch.to(self.device)
                            y_batch = y_batch.to(self.device)

                            t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
                            preds, pred_logvars = self.model(x_batch, t_span, exog=exog_x_batch)

                            mse = (preds - y_batch.transpose(1, 2)) ** 2
                            if self.loss_type == 'nll':
                                loss = torch.mean(
                                    0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
                                )
                            else:
                                loss = mse.mean()

                            val_loss += loss.item() * len(x_batch)

                    avg_val_loss = val_loss / float(len(val_loader.dataset))  # type: ignore[arg-type]

                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        epochs_without_improvement += 1

                    # Update progress bar
                    pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.4f}',
                        'val_loss': f'{avg_val_loss:.4f}',
                        'patience': f'{epochs_without_improvement}/{self.patience}'
                    })

                    # Stop if patience exceeded
                    if epochs_without_improvement >= self.patience:
                        if self.verbose:
                            logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                            logger.info(f"Best validation loss: {best_val_loss:.4f}")
                        # Restore best model
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                        break
                else:
                    # Update progress bar with training loss only
                    pbar.set_postfix({'loss': f'{avg_train_loss:.4f}'})

                if log_callback is not None:
                    metrics = {'train_loss': avg_train_loss}
                    if self.early_stopping and val_loader is not None:
                        metrics['val_loss'] = avg_val_loss
                    # Works directly with mlflow.log_metrics / wandb.log:
                    # fit(..., log_callback=lambda epoch, m: mlflow.log_metrics(m, step=epoch))
                    log_callback(epoch, metrics)

        else:
            # Other models (transformer, tcn, ensemble) use their own train_model() method
            self.model.train_model(
                train_loader,
                self.num_epochs,
                self.learning_rate,
                self.device
            )

        self._is_fitted = True

        if self.verbose:
            logger.info("Training completed!")

        # Create conformal predictor if enabled (only for APDTFlow)
        if self.use_conformal:
            if self.model_type != 'apdtflow':
                raise ValueError(
                    f"Conformal prediction is only supported with model_type='apdtflow'. "
                    f"Current model_type='{self.model_type}' does not support conformal prediction."
                )

            if self.verbose:
                logger.info("Calibrating conformal predictor...")

            # Generate calibration predictions
            self.model.eval()
            calib_preds = []
            calib_targets = []
            calib_X_parts = []
            calib_exog_parts = []

            # Create t_span for APDTFlow model
            t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)

            with torch.no_grad():
                for batch_data in train_loader:
                    if has_exog_data:
                        x_batch, y_batch, exog_x_batch, exog_y_batch = batch_data
                        exog_x_batch = exog_x_batch.to(self.device)
                    else:
                        x_batch, y_batch = batch_data
                        exog_x_batch = None

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    preds, _ = self.model(x_batch, t_span, exog=exog_x_batch)

                    calib_preds.append(preds.cpu().numpy())
                    calib_targets.append(y_batch.cpu().numpy())
                    calib_X_parts.append(x_batch.cpu())
                    if exog_x_batch is not None:
                        calib_exog_parts.append(exog_x_batch.cpu())

            calib_preds_arr = np.concatenate(calib_preds, axis=0)
            calib_targets_arr = np.concatenate(calib_targets, axis=0)

            # Denormalize for conformal predictor
            assert self.scaler_mean_ is not None and self.scaler_std_ is not None
            calib_preds_denorm = calib_preds_arr * self.scaler_std_ + self.scaler_mean_
            calib_targets_denorm = calib_targets_arr * self.scaler_std_ + self.scaler_mean_

            # Initialize conformal predictor with identity function
            # (we already have predictions, so predict_fn just returns input)
            def identity_fn(x):
                return x

            if self.conformal_method == 'split':
                self.conformal_predictor = SplitConformalPredictor(
                    predict_fn=identity_fn,
                    alpha=0.05
                )
            else:  # adaptive
                self.conformal_predictor = AdaptiveConformalPredictor(
                    predict_fn=identity_fn,
                    alpha=0.05
                )

            # Calibrate with predictions as X and targets as y
            self.conformal_predictor.calibrate(
                calib_preds_denorm.reshape(-1, 1),
                calib_targets_denorm.reshape(-1, 1)
            )

            # Per-window calibration state for predict_at intervals and
            # predict_when time-window calibration (v0.4.0).
            horizon = self.forecast_horizon
            preds_grid = calib_preds_denorm.reshape(-1, horizon)
            targets_grid = calib_targets_denorm.reshape(-1, horizon)
            self._calib_X_ = torch.cat(calib_X_parts)
            self._calib_exog_ = (
                torch.cat(calib_exog_parts) if calib_exog_parts else None
            )
            self._calib_targets_grid_ = targets_grid
            self._calib_residuals_ = np.abs(preds_grid - targets_grid)
            anchor_norm = self._calib_X_[:, 0, -1].numpy()
            assert self.scaler_mean_ is not None and self.scaler_std_ is not None
            self._calib_anchor_ = anchor_norm * self.scaler_std_ + self.scaler_mean_
            self._when_cache_ = {}

            if self.verbose:
                logger.info("Conformal predictor calibrated!")

        return self

    def predict(
        self,
        steps: Optional[int] = None,
        return_uncertainty: bool = False,
        exog_future: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        alpha: float = 0.05,
        return_intervals: Optional[str] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make forecasts.

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast. Uses forecast_horizon if None
        return_uncertainty : bool, default=False
            Whether to return uncertainty estimates
        exog_future : DataFrame or array, optional
            Future exogenous variables (required if model trained with exog)
        alpha : float, default=0.05
            Significance level for conformal intervals (e.g., 0.05 for 95% coverage)
        return_intervals : str, optional
            Type of intervals: 'conformal' for conformal prediction intervals,
            None for standard uncertainty (if return_uncertainty=True)

        Returns
        -------
        predictions : array
            Forecast values (denormalized)
        uncertainty : array, optional
            Standard deviation of predictions if return_uncertainty=True
        lower, predictions, upper : tuple of arrays, optional
            Conformal prediction intervals if return_intervals='conformal'
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        assert self.model is not None, "Model is None despite being fitted"

        if steps is None:
            steps = self.forecast_horizon
        elif steps != self.forecast_horizon:
            warnings.warn(
                f"Requested {steps} steps but model trained for {self.forecast_horizon}. "
                f"Using {self.forecast_horizon}."
            )
            steps = self.forecast_horizon

        # Check exog requirements - only require exog_future for numerical exog
        # Categorical features can be auto-generated
        if self.has_numerical_exog_ and exog_future is None:
            raise ValueError(
                "Model was trained with numerical exogenous variables. "
                "Please provide exog_future parameter for prediction."
            )

        self.model.eval()

        with torch.no_grad():
            if self.model_type == 'apdtflow':
                # APDTFlow-specific prediction
                # Prepare input (multichannel when fitted with feature_cols)
                last = np.asarray(self.last_sequence_)
                x = torch.tensor(last, dtype=torch.float32)
                x = x.unsqueeze(0) if last.ndim == 2 else x.unsqueeze(0).unsqueeze(0)
                x = x.to(self.device)

                t_span = torch.linspace(
                    0, 1,
                    steps=self.history_length,
                    device=self.device
                )

                # Prepare exog if needed
                exog_tensor = None
                if self.has_exog_ and exog_future is not None:
                    # Process future exog
                    if isinstance(exog_future, pd.DataFrame):
                        exog_array = exog_future[self.future_exog_cols_ or self.exog_cols_].values
                    else:
                        exog_array = np.array(exog_future)

                    # Normalize using stored parameters
                    exog_norm = (exog_array - self.exog_mean_) / self.exog_std_

                    # Combine with last exog sequence
                    if self.last_exog_sequence_ is not None:
                        full_exog = np.vstack([self.last_exog_sequence_, exog_norm[:steps]])
                        exog_input = full_exog[:self.history_length]
                    else:
                        exog_input = exog_norm[:self.history_length]

                    exog_tensor = torch.tensor(
                        exog_input,
                        dtype=torch.float32
                    ).T.unsqueeze(0).to(self.device)

                # Predict
                preds, pred_logvars = self.model(x, t_span, exog=exog_tensor)

                # Denormalize
                preds_np = preds.cpu().numpy().squeeze()
                preds_denorm = preds_np * self.scaler_std_ + self.scaler_mean_

            else:
                # Other models (transformer, tcn, ensemble)
                x = torch.tensor(
                    self.last_sequence_,
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).to(self.device)

                preds, pred_logvars = self.model.predict(x, self.forecast_horizon, self.device)

                # Denormalize
                preds_np = preds.cpu().numpy().squeeze()
                preds_denorm = preds_np * self.scaler_std_ + self.scaler_mean_

            # Handle conformal prediction
            if return_intervals == 'conformal':
                if self.conformal_predictor is None:
                    warnings.warn(
                        "Conformal prediction not calibrated. "
                        "Returning point predictions only."
                    )
                    return preds_denorm, preds_denorm, preds_denorm
                else:
                    # Get conformal intervals
                    lower, pred_conf, upper = self.conformal_predictor.predict(
                        preds_denorm.reshape(-1, 1)
                    )
                    return lower.flatten(), preds_denorm, upper.flatten()

            # Handle standard uncertainty
            if return_uncertainty:
                uncertainty = np.exp(pred_logvars.cpu().numpy().squeeze() / 2) * self.scaler_std_
                return preds_denorm, uncertainty
            else:
                return preds_denorm

    # ------------------------------------------------------------------
    # Continuous-time API (v0.4.0): predict_at and predict_when
    # ------------------------------------------------------------------

    def _require_continuous(self, method_name: str):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        if self.model_type != 'apdtflow' or self.decoder_type != 'continuous':
            raise RuntimeError(
                f"{method_name} requires decoder_type='continuous'. Construct the "
                f"forecaster with APDTFlowForecaster(decoder_type='continuous', ...) "
                f"and refit."
            )

    def _offsets_from_timestamps(self, timestamps) -> np.ndarray:
        """Convert timestamps (or floats) to positive step offsets."""
        arr = np.asarray(timestamps)
        if arr.dtype.kind in 'fiu':
            offsets = arr.astype(float).reshape(-1)
        else:
            if self.last_date_ is None or self.freq_delta_ is None:
                raise ValueError(
                    "Datetime timestamps require the model to be fitted with a "
                    "date_col; otherwise pass float step offsets."
                )
            ts = pd.to_datetime(
                pd.Series(np.asarray(timestamps).reshape(-1)), format='mixed'
            )
            offsets = ((ts - self.last_date_) / self.freq_delta_).to_numpy(dtype=float)
        if (offsets <= 0).any():
            raise ValueError(
                "All query times must be strictly after the end of the fitted "
                "series (offsets must be > 0)."
            )
        return offsets

    def _steps_to_time(self, tau: float):
        """Convert a float step offset to a timestamp when date-fitted."""
        if self.last_date_ is not None and self.freq_delta_ is not None:
            return self.last_date_ + tau * self.freq_delta_
        return float(tau)

    def _exog_tensor_for_window(self, exog_future, steps: int):
        """Build the exogenous tensor for the last fitted window."""
        if not self.has_exog_:
            return None
        if self.has_numerical_exog_ and exog_future is None:
            raise ValueError(
                "Model was trained with numerical exogenous variables. "
                "Please provide exog_future."
            )
        if exog_future is None:
            return None
        if isinstance(exog_future, pd.DataFrame):
            exog_array = exog_future[self.future_exog_cols_ or self.exog_cols_].values
        else:
            exog_array = np.array(exog_future)
        exog_norm = (exog_array - self.exog_mean_) / self.exog_std_
        if self.last_exog_sequence_ is not None:
            full_exog = np.vstack([self.last_exog_sequence_, exog_norm[:steps]])
            exog_input = full_exog[:self.history_length]
        else:
            exog_input = exog_norm[:self.history_length]
        return torch.tensor(exog_input, dtype=torch.float32).T.unsqueeze(0).to(self.device)

    def _trajectory_at(self, offsets: np.ndarray, exog_future=None) -> np.ndarray:
        """Denormalized mean-trajectory values at float step offsets."""
        assert self.model is not None
        self.model.eval()
        last = np.asarray(self.last_sequence_)
        x = torch.tensor(last, dtype=torch.float32)
        x = x.unsqueeze(0) if last.ndim == 2 else x.unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
        exog_tensor = self._exog_tensor_for_window(exog_future, self.forecast_horizon)
        taus = torch.tensor(offsets, dtype=torch.float32, device=self.device)
        assert isinstance(self.model, APDTFlow)
        with torch.no_grad():
            values, _ = self.model.forward_at(x, t_span, taus, exog=exog_tensor)
        values_np = values.squeeze(0).squeeze(-1).cpu().numpy()
        return values_np * self.scaler_std_ + self.scaler_mean_

    def _interp_step_quantiles(self, offsets: np.ndarray, alpha: float) -> np.ndarray:
        """Per-step conformal quantiles interpolated across fractional offsets."""
        if self._calib_residuals_ is None:
            raise RuntimeError(
                "Conformal calibration data is unavailable. Fit with "
                "use_conformal=True to obtain calibrated intervals."
            )
        grid_q = per_step_quantiles(self._calib_residuals_, alpha)
        grid = np.arange(1, self.forecast_horizon + 1, dtype=float)
        # Beyond the horizon the last step's quantile is used (documented).
        return np.interp(offsets, grid, grid_q)

    def predict_at(
        self,
        timestamps,
        alpha: float = 0.05,
        exog_future: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        """Forecast the series value at arbitrary moments in time.

        Requires ``decoder_type='continuous'``. Accepts datetime-like
        timestamps (when fitted with a ``date_col``) or float step offsets,
        including fractional steps and times beyond the trained horizon.

        Parameters
        ----------
        timestamps : sequence of datetime-like or float
            Times to forecast at. Floats are offsets in forecast steps
            (1.0 = one step past the end of the fitted series).
        alpha : float, default=0.05
            Miscoverage level for conformal intervals.
        exog_future : DataFrame or array, optional
            Future exogenous values (required if fitted with numerical exog).

        Returns
        -------
        values : ndarray
            Forecast values, in the original scale.
        (values, lower, upper) : tuple of ndarray
            When the model was fitted with ``use_conformal=True``: values
            with per-step conformal intervals interpolated across the
            queried times.
        """
        self._require_continuous("predict_at")
        offsets = self._offsets_from_timestamps(timestamps)
        values = self._trajectory_at(offsets, exog_future=exog_future)
        if self.use_conformal and self._calib_residuals_ is not None:
            q = self._interp_step_quantiles(offsets, alpha)
            return values, values - q, values + q
        return values

    def _dense_offsets(self, n_points: int = 400) -> np.ndarray:
        horizon = float(self.forecast_horizon)
        return np.linspace(horizon / n_points, horizon, n_points)

    def _batch_crossing_times(
        self,
        X: torch.Tensor,
        exog: Optional[torch.Tensor],
        threshold: float,
        direction: str,
    ) -> np.ndarray:
        """Predicted first-crossing times for a batch of input windows.

        ``X`` holds normalized windows of shape ``(n, 1, history_length)``;
        returns crossing times in forecast steps with ``np.nan`` where the
        mean trajectory never crosses within the horizon. Used by the
        calibration machinery and by the audit harness in ``experiments/``.
        """
        assert self.model is not None
        taus = self._dense_offsets()
        taus_t = torch.tensor(taus, dtype=torch.float32, device=self.device)
        t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
        assert isinstance(self.model, APDTFlow)
        self.model.eval()

        pred_traj_parts = []
        with torch.no_grad():
            for start in range(0, len(X), 256):
                xb = X[start:start + 256].to(self.device)
                eb = None
                if exog is not None:
                    eb = exog[start:start + 256].to(self.device)
                vals, _ = self.model.forward_at(xb, t_span, taus_t, exog=eb)
                pred_traj_parts.append(vals.squeeze(-1).cpu().numpy())
        pred_traj = np.concatenate(pred_traj_parts, axis=0)
        assert self.scaler_mean_ is not None and self.scaler_std_ is not None
        pred_traj = pred_traj * self.scaler_std_ + self.scaler_mean_
        return batch_first_crossing_times(taus, pred_traj, threshold, direction)

    def _crossing_calibration(self, threshold: float, direction: str, alpha: float):
        """Signed crossing-time error quantiles on the calibration split.

        Returns ``(lo, hi, n_crossings)``; cached per
        ``(threshold, direction, alpha)``.
        """
        key = (round(float(threshold), 9), direction, round(float(alpha), 9))
        if key in self._when_cache_:
            return self._when_cache_[key]
        assert self.model is not None
        if self._calib_X_ is None or self._calib_targets_grid_ is None:
            raise RuntimeError(
                "predict_when requires conformal calibration data; fit with "
                "use_conformal=True."
            )

        t_pred = self._batch_crossing_times(
            self._calib_X_, self._calib_exog_, threshold, direction
        )

        # Actual crossing times from the (denormalized) target grid,
        # anchored at offset 0 with the last observed input value.
        assert self._calib_anchor_ is not None
        anchors = self._calib_anchor_.reshape(-1, 1)
        actual_grid = np.concatenate([anchors, self._calib_targets_grid_], axis=1)
        grid_times = np.arange(0, self.forecast_horizon + 1, dtype=float)
        t_actual = batch_first_crossing_times(
            grid_times, actual_grid, threshold, direction
        )

        both = ~np.isnan(t_pred) & ~np.isnan(t_actual)
        errors = t_pred[both] - t_actual[both]
        n_crossings = int(both.sum())
        if n_crossings > 0:
            lo, hi = crossing_time_quantile(errors, alpha)
        else:
            lo, hi = np.nan, np.nan
        result = (lo, hi, n_crossings)
        self._when_cache_[key] = result
        return result

    def predict_when(
        self,
        threshold: float,
        direction: str = 'above',
        mode: str = 'expected',
        alpha: float = 0.05,
        exog_future: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> PredictWhenResult:
        """Forecast WHEN the series will cross a threshold.

        Requires ``decoder_type='continuous'`` and ``use_conformal=True``.

        Two modes, answering different questions:

        - ``mode='expected'`` (default): the first crossing of the mean
          trajectory, with a time window calibrated on crossing-time errors
          from the calibration split (asymmetric signed-error quantiles,
          which absorb systematic timing bias). For forecastable trends and
          cycles.
        - ``mode='risk'``: the first time the ``alpha``-level conformal
          band touches the threshold — the earliest moment the event
          becomes plausible. High recall, valid early bound; for noisy
          series and safety thresholds.

        Operational rule: never act on the point estimate ``eta`` — the
        point estimate carries a systematic late bias on degradation data.
        Schedule by ``result.act_by`` (the window's early edge).

        Uncertainty here never derives from the model's raw log-variance
        head (untrained under MSE loss); it always comes from conformal
        calibration.

        Parameters
        ----------
        threshold : float
            Threshold level, in the original data scale.
        direction : {'above', 'below'}, default='above'
            Crossing direction.
        mode : {'expected', 'risk'}, default='expected'
        alpha : float, default=0.05
            Miscoverage level of the time window (0.05 -> 95%). This is the
            sensitivity knob: smaller alpha widens windows and earlier
            act-by dates (fewer missed events, more false alarms).
        exog_future : DataFrame or array, optional
            Future exogenous values (required if fitted with numerical exog).

        Returns
        -------
        PredictWhenResult
            With fields ``eta``, ``earliest``, ``latest``, ``act_by``,
            ``censored``, ``low_confidence``. Iterable as
            ``(eta, earliest, latest)``. Times are timestamps when the model
            was fitted with a ``date_col``, otherwise float step offsets.
            If no crossing occurs within the horizon, ``eta`` is the horizon
            and ``censored=True`` — "no crossing within horizon at this
            confidence" is a valid answer.
        """
        self._require_continuous("predict_when")
        if direction not in ('above', 'below'):
            raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
        if mode not in ('expected', 'risk'):
            raise ValueError(f"mode must be 'expected' or 'risk', got {mode!r}")
        if not self.use_conformal or self._calib_residuals_ is None:
            raise RuntimeError(
                "predict_when requires conformal calibration; construct the "
                "forecaster with use_conformal=True and refit."
            )

        horizon = float(self.forecast_horizon)
        taus = self._dense_offsets()
        mean_traj = self._trajectory_at(taus, exog_future=exog_future)

        def _finish(eta, earliest, latest, censored=False, low_confidence=False):
            earliest = float(np.clip(earliest, taus[0], horizon))
            latest = float(np.clip(latest, taus[0], horizon))
            eta = float(np.clip(eta, taus[0], horizon))
            return PredictWhenResult(
                eta=self._steps_to_time(eta),
                earliest=self._steps_to_time(earliest),
                latest=self._steps_to_time(latest),
                act_by=self._steps_to_time(earliest),
                censored=censored,
                low_confidence=low_confidence,
                mode=mode,
                threshold=float(threshold),
                direction=direction,
            )

        if mode == 'risk':
            q = self._interp_step_quantiles(taus, alpha)
            band = mean_traj + q if direction == 'above' else mean_traj - q
            t_plausible = first_crossing_time(taus, band, threshold, direction)
            if t_plausible is None:
                return _finish(horizon, horizon, horizon, censored=True)
            return _finish(t_plausible, t_plausible, horizon)

        # mode == 'expected'
        t_cross = first_crossing_time(taus, mean_traj, threshold, direction)
        if t_cross is None:
            return _finish(horizon, horizon, horizon, censored=True)

        lo, hi, n_crossings = self._crossing_calibration(threshold, direction, alpha)
        if n_crossings >= 20:
            # Asymmetric time-space calibration: t_actual = t_pred - error,
            # error in [lo, hi] with (1 - alpha) coverage.
            return _finish(t_cross, t_cross - hi, t_cross - lo)

        # Fewer than 20 calibration crossings: fall back to value-space
        # banding and flag low confidence.
        q = self._interp_step_quantiles(taus, alpha)
        upper = mean_traj + q
        lower = mean_traj - q
        optimistic = upper if direction == 'above' else lower
        pessimistic = lower if direction == 'above' else upper
        t_early = first_crossing_time(taus, optimistic, threshold, direction)
        t_late = first_crossing_time(taus, pessimistic, threshold, direction)
        earliest = t_early if t_early is not None else t_cross
        latest = t_late if t_late is not None else horizon
        return _finish(t_cross, earliest, latest, low_confidence=True)

    @property
    def sensor_importance_(self) -> pd.Series:
        """Learned health-indicator fusion weights, one per input channel.

        Available after fitting with ``feature_cols``. The magnitude of a
        weight reflects how much that channel contributes to the fused
        health indicator the model forecasts from.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first")
        fusion = getattr(self.model, 'input_fusion', None)
        if fusion is None:
            raise RuntimeError(
                "sensor_importance_ requires fitting with feature_cols "
                "(multivariate input)."
            )
        weights = fusion.weight.detach().cpu().numpy().ravel()
        names = [self.target_col_ or 'target'] + list(self.feature_cols_ or [])
        return pd.Series(weights, index=names, name='sensor_importance')

    def predict_when_fleet(
        self,
        series_by_asset: Dict,
        threshold: float,
        direction: str = 'above',
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Event-time forecasts for a fleet of assets, as a schedule.

        Applies the fitted model and its crossing-time calibration to the
        most recent window of each asset's series and returns a maintenance
        schedule sorted by ``act_by`` (the operational deadline — never
        schedule by the point estimate ``eta``).

        Parameters
        ----------
        series_by_asset : dict
            Maps asset id to that asset's recent history: a 1D array-like of
            target values (at least ``history_length`` long), or a DataFrame
            containing ``target_col`` (and ``feature_cols`` when the model
            was fitted multivariate). Values must be in the same units the
            model was fitted on.
        threshold : float
            Threshold level in the original data scale.
        direction : {'above', 'below'}, default='above'
        alpha : float, default=0.05
            Miscoverage level — the sensitivity knob (smaller alpha means
            wider windows and earlier act-by dates).

        Returns
        -------
        DataFrame
            Columns ``[asset_id, eta, earliest, latest, act_by, censored,
            confidence]`` sorted by ``act_by`` (censored assets last).
            Times are float step offsets from the end of each asset's
            series. Use ``.to_csv()`` / ``.to_dict()`` for CMMS ingestion.
        """
        self._require_continuous("predict_when_fleet")
        if not self.use_conformal or self._calib_residuals_ is None:
            raise RuntimeError(
                "predict_when_fleet requires conformal calibration; construct "
                "the forecaster with use_conformal=True and refit."
            )
        if self.has_exog_:
            raise ValueError(
                "predict_when_fleet does not support exogenous-variable models."
            )

        ids = []
        windows = []
        for asset_id, series in series_by_asset.items():
            if isinstance(series, pd.DataFrame):
                channels = [self.target_col_] + list(self.feature_cols_ or [])
                missing = [c for c in channels if c not in series.columns]
                if missing:
                    raise ValueError(f"Asset {asset_id!r} is missing columns {missing}")
                matrix = series[channels].to_numpy(dtype=float).T
            else:
                if self.feature_cols_:
                    raise ValueError(
                        "Model was fitted with feature_cols; each asset must "
                        "provide a DataFrame with those columns."
                    )
                matrix = np.asarray(series, dtype=float).reshape(1, -1)
            if matrix.shape[1] < self.history_length:
                raise ValueError(
                    f"Asset {asset_id!r} has {matrix.shape[1]} points; "
                    f"history_length={self.history_length} required."
                )
            window = matrix[:, -self.history_length:]
            if self.feature_cols_:
                assert self.feature_means_ is not None and self.feature_stds_ is not None
                norm = (window - self.feature_means_[:, None]) / self.feature_stds_[:, None]
            else:
                norm = (window - self.scaler_mean_) / self.scaler_std_
            ids.append(asset_id)
            windows.append(norm)

        X = torch.tensor(np.stack(windows), dtype=torch.float32)
        t_pred = self._batch_crossing_times(X, None, threshold, direction)
        lo, hi, n_calib = self._crossing_calibration(threshold, direction, alpha)
        low_confidence = n_calib < 20

        horizon = float(self.forecast_horizon)
        rows = []
        for asset_id, eta in zip(ids, t_pred):
            censored = bool(np.isnan(eta))
            if censored:
                eta_v = horizon
                earliest = horizon
                latest = horizon
            else:
                eta_v = float(eta)
                if low_confidence:
                    earliest, latest = eta_v, horizon
                else:
                    earliest = float(np.clip(eta_v - hi, 0.0, horizon))
                    latest = float(np.clip(eta_v - lo, 0.0, horizon))
            rows.append({
                'asset_id': asset_id,
                'eta': eta_v,
                'earliest': earliest,
                'latest': latest,
                'act_by': earliest,
                'censored': censored,
                'confidence': 1.0 - alpha if not low_confidence else float('nan'),
            })
        fleet = pd.DataFrame(rows)
        return fleet.sort_values(
            ['censored', 'act_by'], ascending=[True, True]
        ).reset_index(drop=True)

    def plot_forecast(
        self,
        with_history: int = 50,
        show_uncertainty: bool = True,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot forecast with historical data.

        Parameters
        ----------
        with_history : int, default=50
            Number of historical points to show
        show_uncertainty : bool, default=True
            Show uncertainty bands
        figsize : tuple, default=(12, 6)
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before plotting")

        assert self.scaler_std_ is not None and self.scaler_mean_ is not None, "Scaler not initialized"

        # Get predictions
        if show_uncertainty:
            result = self.predict(return_uncertainty=True)
            preds, uncertainty = result  # type: ignore[misc]
        else:
            preds = self.predict()
            uncertainty = None

        # Get historical data
        if self.data_df_ is not None and self.target_col_ is not None:
            history = self.data_df_[self.target_col_].values[-with_history:]
        else:
            # Reconstruct from last_sequence_
            assert self.last_sequence_ is not None, "No sequence data available for plotting"
            history_norm = self.last_sequence_
            history = history_norm * self.scaler_std_ + self.scaler_mean_

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        hist_idx = np.arange(len(history))
        pred_idx = np.arange(len(history), len(history) + len(preds))

        # Plot history
        ax.plot(hist_idx, history, 'b-', label='Historical', linewidth=2)

        # Plot forecast
        ax.plot(pred_idx, preds, 'r--', label='Forecast', linewidth=2)

        # Plot uncertainty
        if show_uncertainty and uncertainty is not None:
            ax.fill_between(
                pred_idx,
                preds - 2 * uncertainty,
                preds + 2 * uncertainty,
                alpha=0.3,
                color='red',
                label='95% Confidence'
            )

        ax.axvline(x=len(history) - 0.5, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(
            f'{self.model_type.upper()} Forecast ({self.forecast_horizon} steps)',
            fontsize=14
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                logger.info(f"Plot saved to {save_path}")

        plt.show()

        return fig, ax

    def score(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        metric: str = 'mse',
        exog_cols: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate model performance on test data.

        Parameters
        ----------
        data : DataFrame or array
            Test data
        target_col : str, optional
            Name of target column if data is DataFrame (uses fitted target_col if None)
        date_col : str, optional
            Name of date column if data is DataFrame (uses fitted date_col if None)
        metric : str, default='mse'
            Metric to use: 'mse', 'mae', 'rmse', 'mape', 'r2'
        exog_cols : list of str, optional
            Exogenous column names (uses fitted exog_cols if None)

        Returns
        -------
        score : float
            Model score (lower is better for mse/mae/rmse/mape, higher is better for r2)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        # Use fitted columns if not provided
        if target_col is None:
            target_col = self.target_col_
        if date_col is None:
            date_col = self.date_col_
        if exog_cols is None:
            exog_cols = self.exog_cols_

        # Prepare test data
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col must be specified")
            if date_col and date_col in data.columns:
                data = data.sort_values(date_col)
            y_true = data[target_col].values

            # Handle categorical features
            exog_data_full = None
            if self.has_categorical_ and self.categorical_encoder_ and self.categorical_cols_:
                # Encode categorical features
                categorical_data = data[self.categorical_cols_]
                categorical_encoded = self.categorical_encoder_.transform(categorical_data)

                # Combine with numerical exog if present
                if exog_cols:
                    exog_numerical = data[exog_cols].values
                    exog_data_full = np.concatenate([exog_numerical, categorical_encoded], axis=1)
                else:
                    # Only categorical features
                    exog_data_full = categorical_encoded
            elif self.has_exog_ and exog_cols:
                # Only numerical exog features
                exog_data_full = data[exog_cols].values
        else:
            y_true = np.array(data).flatten()
            exog_data_full = None

        # Make predictions
        # Simple rolling prediction for evaluation
        predictions = []
        n_samples = len(y_true) - self.history_length - self.forecast_horizon + 1

        for i in range(min(n_samples, 50)):  # Limit to 50 windows for speed
            start = i
            end = start + self.history_length
            window = y_true[start:end]

            # Normalize
            window_norm = (window - self.scaler_mean_) / self.scaler_std_

            # Prepare exog if available
            exog_tensor = None
            if self.has_exog_ and exog_data_full is not None:
                exog_window = exog_data_full[start:end]
                # Normalize using fitted parameters
                exog_norm = (exog_window - self.exog_mean_) / self.exog_std_
                exog_tensor = torch.tensor(
                    exog_norm,
                    dtype=torch.float32
                ).T.unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                x = torch.tensor(
                    window_norm,
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(self.device)

                t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
                assert self.model is not None, "Model must be initialized"
                preds, _ = self.model(x, t_span, exog=exog_tensor)
                pred_denorm = preds.cpu().numpy().flatten() * self.scaler_std_ + self.scaler_mean_

            predictions.append(pred_denorm)

        # Calculate metric
        if len(predictions) == 0:
            raise ValueError("Not enough data for evaluation")

        predictions_arr = np.array(predictions)
        targets = np.array([
            y_true[i + self.history_length:i + self.history_length + self.forecast_horizon]
            for i in range(len(predictions))
        ])

        # Flatten for metric calculation
        pred_flat = predictions_arr.flatten()
        true_flat = targets.flatten()

        if metric == 'mse':
            return float(np.mean((pred_flat - true_flat) ** 2))
        elif metric == 'mae':
            return float(np.mean(np.abs(pred_flat - true_flat)))
        elif metric == 'rmse':
            return float(np.sqrt(np.mean((pred_flat - true_flat) ** 2)))
        elif metric == 'mape':
            return float(np.mean(np.abs((true_flat - pred_flat) / (true_flat + 1e-8))) * 100)
        elif metric == 'r2':
            ss_res = np.sum((true_flat - pred_flat) ** 2)
            ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
            return float(1 - ss_res / (ss_tot + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'mse', 'mae', 'rmse', 'mape', or 'r2'")

    def historical_forecasts(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        start: Union[float, int] = 0.8,
        forecast_horizon: Optional[int] = None,
        stride: int = 1,
        retrain: bool = False,
        metrics: Optional[List[str]] = None,
        exog_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Backtest model by generating historical forecasts.

        Simulates production forecasting
        by making predictions on historical data using a rolling window.

        NEW in v0.2.3!

        Parameters
        ----------
        data : DataFrame or array
            Full historical data
        target_col : str, optional
            Target column name (uses fitted target_col if None)
        date_col : str, optional
            Date column name (uses fitted date_col if None)
        start : float or int, default=0.8
            Where to start forecasting:
            - If float (0-1): fraction of data (e.g., 0.8 = 80%)
            - If int: index position
        forecast_horizon : int, optional
            Forecast horizon (uses fitted horizon if None)
        stride : int, default=1
            Step size between forecasts (1 = every timestep)
        retrain : bool, default=False
            Whether to retrain model at each step (slow but more realistic)
        metrics : list of str, optional
            Metrics to compute (default: ['mse', 'mae', 'mase', 'smape'])
        exog_cols : list of str, optional
            Exogenous columns (uses fitted exog_cols if None)
        categorical_cols : list of str, optional
            Categorical columns (uses fitted categorical_cols if None)

        Returns
        -------
        results : DataFrame
            Backtesting results with columns:
            - timestamp: Forecast timestamp (if date_col provided)
            - actual: True values
            - predicted: Forecasted values
            - forecast_step: Step ahead (1, 2, ..., forecast_horizon)
            - fold: Forecast fold number
            - error: Prediction error (actual - predicted)
            - abs_error: Absolute error
            + any requested metrics

        Examples
        --------
        >>> # Backtest starting at 80% of data
        >>> results = model.historical_forecasts(
        ...     df,
        ...     target_col='sales',
        ...     start=0.8,
        ...     stride=7  # Weekly forecasts
        ... )
        >>> print(f"Average MAE: {results['abs_error'].mean():.2f}")

        >>> # With retraining (slower but more realistic)
        >>> results = model.historical_forecasts(
        ...     df,
        ...     target_col='sales',
        ...     retrain=True,
        ...     stride=7
        ... )

        Notes
        -----
        - Set stride=forecast_horizon for non-overlapping forecasts
        - retrain=True is more realistic but much slower
        - Useful for validating model before production deployment
        """
        if not self._is_fitted and not retrain:
            raise RuntimeError("Model must be fitted before backtesting (or use retrain=True)")

        # Use defaults from fitted model
        if target_col is None:
            target_col = self.target_col_
        if date_col is None:
            date_col = self.date_col_
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon
        if exog_cols is None:
            exog_cols = self.exog_cols_
        if categorical_cols is None:
            categorical_cols = self.categorical_cols_
        if metrics is None:
            metrics = ['mse', 'mae', 'mase', 'smape']

        # Validate data
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col must be specified")
            if date_col and date_col in data.columns:
                data = data.sort_values(date_col).reset_index(drop=True)

            series = data[target_col].values
            dates = data[date_col].values if date_col and date_col in data.columns else None
        else:
            series = np.array(data).flatten()
            dates = None

        # Determine start index
        if isinstance(start, float):
            start_idx = int(len(series) * start)
        else:
            start_idx = start

        # Ensure we have enough data
        min_required = start_idx + self.history_length + forecast_horizon
        if len(series) < min_required:
            raise ValueError(
                f"Not enough data for backtesting. Need {min_required} points, "
                f"got {len(series)}"
            )

        # Generate forecasts
        results_list = []
        fold = 0

        for i in range(start_idx, len(series) - forecast_horizon + 1, stride):
            # Training data up to i
            train_data_slice = data.iloc[:i] if isinstance(data, pd.DataFrame) else series[:i]

            # Test data (next forecast_horizon points)
            actual_values = series[i:i + forecast_horizon]

            if retrain:
                # Retrain model on expanding window
                temp_model = APDTFlowForecaster(
                    model_type=self.model_type,
                    forecast_horizon=forecast_horizon,
                    history_length=self.history_length,
                    num_epochs=self.num_epochs,
                    verbose=False
                )
                try:
                    temp_model.fit(
                        train_data_slice,
                        target_col=target_col,
                        date_col=date_col,
                        exog_cols=exog_cols,
                        categorical_cols=categorical_cols
                    )
                    predictions = temp_model.predict(steps=forecast_horizon)
                except Exception as e:
                    if self.verbose:
                        logger.info(f"Skipping fold {fold} due to error: {e}")
                    continue
            else:
                # Use pre-trained model
                # Need to update last_sequence with data up to i
                window = series[i - self.history_length:i]
                window_norm = (window - self.scaler_mean_) / self.scaler_std_

                # Store original last_sequence
                assert self.last_sequence_ is not None, "last_sequence_ must be set"
                original_last_seq = self.last_sequence_.copy()

                # Temporarily update
                self.last_sequence_ = window_norm

                try:
                    predictions = self.predict(steps=forecast_horizon)
                except Exception as e:
                    if self.verbose:
                        logger.info(f"Skipping fold {fold} due to error: {e}")
                    # Restore original
                    self.last_sequence_ = original_last_seq
                    continue

                # Restore original last_sequence
                self.last_sequence_ = original_last_seq

            # Store results for each forecast step
            for step in range(forecast_horizon):
                if i + step < len(series):
                    result_row = {
                        'fold': fold,
                        'forecast_step': step + 1,
                        'actual': actual_values[step],
                        'predicted': predictions[step],
                        'error': actual_values[step] - predictions[step],
                        'abs_error': np.abs(actual_values[step] - predictions[step])
                    }

                    # Add timestamp if available
                    if dates is not None and i + step < len(dates):
                        result_row['timestamp'] = dates[i + step]

                    results_list.append(result_row)

            fold += 1

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        if len(results_df) == 0:
            raise ValueError("No forecasts generated. Check your data and parameters.")

        # Calculate aggregate metrics
        if metrics:
            metric_results = {}
            for metric in metrics:
                try:
                    if metric.lower() == 'mse':
                        metric_results[metric] = np.mean(results_df['error'] ** 2)
                    elif metric.lower() == 'mae':
                        metric_results[metric] = np.mean(results_df['abs_error'])
                    elif metric.lower() == 'rmse':
                        metric_results[metric] = np.sqrt(np.mean(results_df['error'] ** 2))
                    elif metric.lower() == 'mape':
                        metric_results[metric] = np.mean(
                            np.abs(results_df['error'] / results_df['actual'])
                        ) * 100
                    elif metric.lower() == 'mase':
                        # Simplified MASE for backtesting
                        naive_errors = np.diff(series[start_idx:])
                        mae_naive = np.mean(np.abs(naive_errors))
                        mae_pred = np.mean(results_df['abs_error'])
                        metric_results[metric] = mae_pred / mae_naive if mae_naive > 0 else np.inf
                    elif metric.lower() == 'smape':
                        numerator = results_df['abs_error']
                        denominator = (np.abs(results_df['predicted']) + np.abs(results_df['actual'])) / 2
                        metric_results[metric] = np.mean(numerator / denominator) * 100
                except Exception as e:
                    if self.verbose:
                        logger.info(f"Could not compute {metric}: {e}")

            # Print summary
            if self.verbose:
                logger.info("\n" + "="*60)
                logger.info("Historical Forecasts Summary")
                logger.info("="*60)
                logger.info(f"Total forecasts: {fold}")
                logger.info(f"Total predictions: {len(results_df)}")
                logger.info(f"Forecast horizon: {forecast_horizon}")
                logger.info(f"Stride: {stride}")
                logger.info("\nMetrics:")
                for metric, value in metric_results.items():
                    logger.info(f"  {metric}: {value:.4f}")
                logger.info("="*60 + "\n")

        # Reorder columns for better readability
        column_order = []
        if 'timestamp' in results_df.columns:
            column_order.append('timestamp')
        column_order.extend(['fold', 'forecast_step', 'actual', 'predicted', 'error', 'abs_error'])
        results_df = results_df[column_order]

        return results_df

    _PARAM_NAMES = (
        'model_type', 'num_scales', 'hidden_dim', 'filter_size',
        'forecast_horizon', 'history_length', 'learning_rate', 'batch_size',
        'num_epochs', 'use_embedding', 'verbose', 'loss_type', 'ode_method',
        'decoder_type', 'exog_fusion_type', 'use_conformal',
        'conformal_method', 'calibration_split', 'early_stopping', 'patience',
        'validation_split', 'categorical_encoding',
    )

    def get_params(self, deep: bool = True) -> Dict:
        """Constructor parameters, scikit-learn style."""
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params):
        """Set constructor parameters, scikit-learn style.

        Returns self. Changing parameters does not refit the model.
        """
        for name, value in params.items():
            if name not in self._PARAM_NAMES:
                raise ValueError(
                    f"Invalid parameter {name!r}. Valid parameters: "
                    f"{sorted(self._PARAM_NAMES)}"
                )
            setattr(self, name, value)
        return self

    def score_recent(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Drift check: rolling accuracy and coverage on recent data.

        Slides the fitted model over ``data`` (most recent observations,
        in the units the model was fitted on) and compares rolling MAE and
        conformal coverage against the calibration baseline. Coverage that
        decays below ``1 - alpha`` signals the model needs recalibration —
        refit (or refit with use_conformal=True) on data that includes the
        recent regime.

        Returns
        -------
        dict
            ``mae``, ``rmse``, ``n_windows``, ``baseline_mae`` (calibration
            split), ``mae_ratio`` (mae / baseline_mae), and — when fitted
            with ``use_conformal=True`` — ``coverage`` vs
            ``expected_coverage``.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first")
        if self.model_type != 'apdtflow':
            raise RuntimeError("score_recent supports model_type='apdtflow' only")
        if self.has_exog_:
            raise ValueError("score_recent does not support exogenous-variable models")

        if isinstance(data, pd.DataFrame):
            target = target_col or self.target_col_
            if self.feature_cols_:
                assert self.feature_means_ is not None and self.feature_stds_ is not None
                channels = [target] + list(self.feature_cols_)
                matrix = data[channels].to_numpy(dtype=float).T
                norm = (matrix - self.feature_means_[:, None]) / self.feature_stds_[:, None]
            else:
                series = data[target].to_numpy(dtype=float)
                norm = ((series - self.scaler_mean_) / self.scaler_std_).reshape(1, -1)
        else:
            if self.feature_cols_:
                raise ValueError("Multivariate models require DataFrame input")
            series = np.asarray(data, dtype=float).flatten()
            assert self.scaler_mean_ is not None and self.scaler_std_ is not None
            norm = ((series - self.scaler_mean_) / self.scaler_std_).reshape(1, -1)

        total = self.history_length + self.forecast_horizon
        n_windows = norm.shape[1] - total + 1
        if n_windows <= 0:
            raise ValueError(
                f"Need at least {total} recent points, got {norm.shape[1]}"
            )
        assert self.scaler_mean_ is not None and self.scaler_std_ is not None
        X = np.stack([norm[:, i:i + self.history_length] for i in range(n_windows)])
        targets = np.stack([
            norm[0, i + self.history_length:i + total] for i in range(n_windows)
        ]) * self.scaler_std_ + self.scaler_mean_

        t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
        self.model.eval()
        preds_parts = []
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            for start in range(0, len(X_t), 256):
                xb = X_t[start:start + 256].to(self.device)
                pb, _ = self.model(xb, t_span)
                preds_parts.append(pb.squeeze(-1).cpu().numpy())
        preds = np.concatenate(preds_parts) * self.scaler_std_ + self.scaler_mean_

        residuals = np.abs(preds - targets)
        result: Dict[str, float] = {
            'mae': float(residuals.mean()),
            'rmse': float(np.sqrt(((preds - targets) ** 2).mean())),
            'n_windows': float(n_windows),
        }
        if self._calib_residuals_ is not None:
            baseline = float(self._calib_residuals_.mean())
            result['baseline_mae'] = baseline
            result['mae_ratio'] = float(residuals.mean() / baseline) if baseline else float('nan')
            q = per_step_quantiles(self._calib_residuals_, alpha)
            result['coverage'] = float((residuals <= q[None, :]).mean())
            result['expected_coverage'] = 1.0 - alpha
        return result

    def export_torchscript(self, filepath: str, example_batch_size: int = 1):
        """Export the fitted model's grid forecast as TorchScript (traced).

        The exported module maps a normalized input window of shape
        ``(batch, channels, history_length)`` to normalized grid forecasts
        of shape ``(batch, forecast_horizon, 1)``. Tracing fixes the input
        window length, the integer forecast grid, AND the batch size
        (``example_batch_size``, default 1 — the typical serving pattern of
        one window per call). ``predict_at`` / ``predict_when`` and
        conformal calibration are not part of the export (serve those
        through the Python API — see ``examples/serve_api.py``). ONNX
        export of the ODE integration graph is not currently supported.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted first")
        if self.model_type != 'apdtflow':
            raise RuntimeError("export_torchscript supports model_type='apdtflow' only")

        model = self.model
        history_length = self.history_length
        device = self.device

        class _GridForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.register_buffer(
                    't_span', torch.linspace(0, 1, steps=history_length)
                )

            def forward(self, x):
                preds, _ = self.model(x, self.t_span)
                return preds

        wrapper = _GridForecast().to(device).eval()
        n_channels = 1 + len(self.feature_cols_ or [])
        example = torch.randn(
            example_batch_size, n_channels, history_length, device=device
        )
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example)
        traced.save(filepath)
        if self.verbose:
            logger.info("TorchScript model exported to %s", filepath)

    def save(self, filepath: str):
        """
        Save the fitted model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model (e.g., 'my_model.pkl')

        Examples
        --------
        >>> model.fit(df, target_col='sales')
        >>> model.save('forecaster.pkl')
        """
        import pickle

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")

        # Save conformal state separately (can't pickle lambda function)
        conformal_state = None
        if self.conformal_predictor is not None:
            conformal_state = {
                'alpha': self.conformal_predictor.alpha,
                'nonconformity_scores': self.conformal_predictor.nonconformity_scores,
                'quantile': self.conformal_predictor.quantile,
                'is_calibrated': self.conformal_predictor.is_calibrated,
            }
            # For adaptive, save additional state
            if self.conformal_method == 'adaptive' and hasattr(self.conformal_predictor, 'adaptive_quantile'):
                conformal_state['adaptive_quantile'] = self.conformal_predictor.adaptive_quantile

        from apdtflow import __version__ as apdtflow_version

        # Prepare state dict
        state = {
            # Library version that produced this checkpoint
            'apdtflow_version': apdtflow_version,

            # Model parameters
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon,
            'history_length': self.history_length,
            'hidden_dim': self.hidden_dim,
            'num_scales': self.num_scales,
            'filter_size': self.filter_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'use_embedding': self.use_embedding,
            'loss_type': self.loss_type,
            'ode_method': self.ode_method,
            'decoder_type': self.decoder_type,
            'exog_fusion_type': self.exog_fusion_type,
            'use_conformal': self.use_conformal,
            'conformal_method': self.conformal_method,
            'calibration_split': self.calibration_split,

            # Fitted state
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler_mean': self.scaler_mean_,
            'scaler_std': self.scaler_std_,
            'last_sequence': self.last_sequence_,
            'last_exog_sequence': self.last_exog_sequence_,
            'target_col': self.target_col_,
            'date_col': self.date_col_,
            'exog_cols': self.exog_cols_,
            'future_exog_cols': self.future_exog_cols_,
            'num_exog_features': self.num_exog_features_,
            'exog_mean': self.exog_mean_,
            'exog_std': self.exog_std_,
            'has_exog': self.has_exog_,
            'has_numerical_exog': self.has_numerical_exog_,

            # Categorical variables state (NEW in v0.2.3)
            'categorical_encoding': self.categorical_encoding,
            'categorical_cols': self.categorical_cols_,
            'has_categorical': self.has_categorical_,
            'categorical_encoder_config': self.categorical_encoder_.get_config() if self.categorical_encoder_ else None,

            # Conformal predictor state
            'conformal_state': conformal_state,

            # Multivariate input state (v0.4.0)
            'feature_cols': self.feature_cols_,
            'feature_means': self.feature_means_,
            'feature_stds': self.feature_stds_,

            # Continuous-time state (v0.4.0): date base and the per-window
            # calibration data behind predict_at intervals and predict_when
            # time windows (including already-computed threshold quantiles).
            'last_date': self.last_date_,
            'freq_delta': self.freq_delta_,
            'calib_X': self._calib_X_.numpy() if self._calib_X_ is not None else None,
            'calib_exog': self._calib_exog_.numpy() if self._calib_exog_ is not None else None,
            'calib_targets_grid': self._calib_targets_grid_,
            'calib_residuals': self._calib_residuals_,
            'calib_anchor': self._calib_anchor_,
            'when_cache': self._when_cache_,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        if self.verbose:
            logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None):
        """
        Load a fitted model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file
        device : str, optional
            Device to load model to ('cpu' or 'cuda')

        Returns
        -------
        model : APDTFlowForecaster
            Loaded forecaster ready for predictions

        Examples
        --------
        >>> model = APDTFlowForecaster.load('forecaster.pkl')
        >>> predictions = model.predict()
        """
        import pickle
        import warnings

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        saved_version = state.get('apdtflow_version')
        if saved_version is None:
            raise ValueError(
                "This checkpoint was saved by apdtflow <= 0.3.x. Models from those "
                "versions contained a defect that made predictions independent of "
                "the input data and cannot be repaired — retrain with apdtflow >= "
                "0.4.0. See CHANGELOG.md (v0.4.0) at "
                "https://github.com/yotambraun/APDTFlow/blob/main/CHANGELOG.md"
            )
        from apdtflow import __version__ as current_version
        if saved_version.split('.')[:2] != current_version.split('.')[:2]:
            warnings.warn(
                f"Checkpoint was saved by apdtflow {saved_version} but you are "
                f"running {current_version}; results may differ across minor "
                f"versions.",
                UserWarning,
            )

        # Create new instance with saved parameters
        model = cls(
            model_type=state['model_type'],
            forecast_horizon=state['forecast_horizon'],
            history_length=state['history_length'],
            hidden_dim=state['hidden_dim'],
            num_scales=state['num_scales'],
            filter_size=state['filter_size'],
            learning_rate=state['learning_rate'],
            batch_size=state['batch_size'],
            num_epochs=state['num_epochs'],
            device=device,
            use_embedding=state['use_embedding'],
            loss_type=state.get('loss_type', 'mse'),
            ode_method=state.get('ode_method', 'rk4'),
            decoder_type=state.get('decoder_type', 'transformer'),
            exog_fusion_type=state['exog_fusion_type'],
            use_conformal=state['use_conformal'],
            conformal_method=state['conformal_method'],
            calibration_split=state['calibration_split'],
            categorical_encoding=state.get('categorical_encoding', 'onehot'),
            verbose=False
        )

        # Restore fitted state
        model.scaler_mean_ = state['scaler_mean']
        model.scaler_std_ = state['scaler_std']
        model.last_sequence_ = state['last_sequence']
        model.last_exog_sequence_ = state['last_exog_sequence']
        model.target_col_ = state['target_col']
        model.date_col_ = state['date_col']
        model.exog_cols_ = state['exog_cols']
        model.future_exog_cols_ = state['future_exog_cols']
        model.num_exog_features_ = state['num_exog_features']
        model.exog_mean_ = state['exog_mean']
        model.exog_std_ = state['exog_std']
        model.has_exog_ = state['has_exog']
        model.has_numerical_exog_ = state.get('has_numerical_exog', state['has_exog'])  # Backward compat

        # Restore conformal predictor from saved state
        conformal_state = state.get('conformal_state', None)
        if conformal_state and model.use_conformal:
            # Reconstruct conformal predictor with identity function
            def identity_fn(x):
                return x

            if model.conformal_method == 'split':
                model.conformal_predictor = SplitConformalPredictor(
                    predict_fn=identity_fn,
                    alpha=conformal_state['alpha']
                )
            else:  # adaptive
                model.conformal_predictor = AdaptiveConformalPredictor(
                    predict_fn=identity_fn,
                    alpha=conformal_state['alpha']
                )

            # Restore calibrated state
            model.conformal_predictor.nonconformity_scores = conformal_state['nonconformity_scores']
            model.conformal_predictor.quantile = conformal_state['quantile']
            model.conformal_predictor.is_calibrated = conformal_state['is_calibrated']

            # Restore adaptive quantile for adaptive conformal
            if 'adaptive_quantile' in conformal_state and hasattr(model.conformal_predictor, 'adaptive_quantile'):
                model.conformal_predictor.adaptive_quantile = conformal_state['adaptive_quantile']  # type: ignore[attr-defined]
        else:
            model.conformal_predictor = None

        # Restore categorical encoder from saved state (NEW in v0.2.3)
        categorical_encoder_config = state.get('categorical_encoder_config', None)
        if categorical_encoder_config:
            model.categorical_encoder_ = CategoricalEncoder.from_config(categorical_encoder_config)
            model.categorical_cols_ = state.get('categorical_cols', None)
            model.has_categorical_ = state.get('has_categorical', False)
        else:
            model.categorical_encoder_ = None
            model.categorical_cols_ = None
            model.has_categorical_ = False

        # Restore multivariate input state (v0.4.0)
        model.feature_cols_ = state.get('feature_cols')
        model.feature_means_ = state.get('feature_means')
        model.feature_stds_ = state.get('feature_stds')

        # Restore continuous-time state (v0.4.0)
        model.last_date_ = state.get('last_date')
        model.freq_delta_ = state.get('freq_delta')
        calib_X = state.get('calib_X')
        model._calib_X_ = torch.tensor(calib_X) if calib_X is not None else None
        calib_exog = state.get('calib_exog')
        model._calib_exog_ = torch.tensor(calib_exog) if calib_exog is not None else None
        model._calib_targets_grid_ = state.get('calib_targets_grid')
        model._calib_residuals_ = state.get('calib_residuals')
        model._calib_anchor_ = state.get('calib_anchor')
        model._when_cache_ = state.get('when_cache') or {}

        # Initialize and load model
        model.model = model._initialize_model()
        if state['model_state_dict']:
            model.model.load_state_dict(state['model_state_dict'])
        model._is_fitted = True

        return model

    def summary(self):
        """
        Print a summary of the model architecture and parameters.

        Similar to Keras model.summary(). Shows layers, parameters, and configuration.

        Examples
        --------
        >>> model.fit(df, target_col='sales')
        >>> model.summary()
        """
        logger.info("=" * 70)
        logger.info(f"APDTFlow Forecaster Summary - {self.model_type.upper()}")
        logger.info("=" * 70)

        logger.info("\nModel Configuration:")
        logger.info(f"  Model Type:          {self.model_type}")
        logger.info(f"  Forecast Horizon:    {self.forecast_horizon} steps")
        logger.info(f"  History Length:      {self.history_length} steps")
        logger.info(f"  Hidden Dimension:    {self.hidden_dim}")
        logger.info(f"  Number of Scales:    {self.num_scales}")
        logger.info(f"  Filter Size:         {self.filter_size}")
        logger.info(f"  Use Embedding:       {self.use_embedding}")

        if self.has_exog_:
            logger.info("\nExogenous Variables:")
            logger.info(f"  Number of Features:  {self.num_exog_features_}")
            logger.info(f"  Fusion Type:         {self.exog_fusion_type}")
            logger.info(f"  Features:            {self.exog_cols_}")
            if self.future_exog_cols_:
                logger.info(f"  Future-Known:        {self.future_exog_cols_}")

        if self.use_conformal:
            logger.info("\nConformal Prediction:")
            logger.info(f"  Method:              {self.conformal_method}")
            logger.info(f"  Calibration Split:   {self.calibration_split}")

        logger.info("\nTraining Configuration:")
        logger.info(f"  Learning Rate:       {self.learning_rate}")
        logger.info(f"  Batch Size:          {self.batch_size}")
        logger.info(f"  Epochs:              {self.num_epochs}")
        logger.info(f"  Device:              {self.device}")

        if self.model:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            logger.info("\nModel Parameters:")
            logger.info(f"  Total Parameters:    {total_params:,}")
            logger.info(f"  Trainable Parameters: {trainable_params:,}")
            logger.info(f"  Non-trainable:       {total_params - trainable_params:,}")

            # Estimate model size
            param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
            logger.info(f"  Model Size:          ~{param_size_mb:.2f} MB")

        logger.info("\nStatus:")
        logger.info(f"  Fitted:              {self._is_fitted}")
        if self._is_fitted:
            logger.info(f"  Target Column:       {self.target_col_}")
            if self.scaler_mean_ is not None:
                logger.info(f"  Data Mean:           {self.scaler_mean_:.4f}")
                logger.info(f"  Data Std:            {self.scaler_std_:.4f}")

        # Residual diagnostics
        if self.residuals_ is not None:
            logger.info("\nResidual Diagnostics:")
            logger.info(f"  Mean Residual:       {np.mean(self.residuals_):.4f}")
            logger.info(f"  Std Residual:        {np.std(self.residuals_):.4f}")
            logger.info(f"  MAE:                 {np.mean(np.abs(self.residuals_)):.4f}")

        logger.info("=" * 70)

    def compute_residuals(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None,
        n_windows: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residuals (actual - predicted) for model diagnostics.

        Uses rolling window validation to generate predictions on historical data
        and compute residuals for analysis.

        NEW in v0.3.0!

        Parameters
        ----------
        data : DataFrame or array
            Data to compute residuals on (typically validation/test set)
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame
        exog_cols : list of str, optional
            Exogenous column names
        n_windows : int, default=100
            Number of rolling windows to use (more windows = more residuals but slower)

        Returns
        -------
        residuals : array
            Residuals (actual - predicted)
        actuals : array
            Actual values
        predictions : array
            Predicted values

        Examples
        --------
        >>> model.fit(train_df, target_col='sales', date_col='date')
        >>> residuals, actuals, predictions = model.compute_residuals(test_df)
        >>> print(f"Mean residual: {np.mean(residuals):.4f}")
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing residuals")

        # Use fitted columns if not provided
        if target_col is None:
            target_col = self.target_col_
        if date_col is None:
            date_col = self.date_col_
        if exog_cols is None:
            exog_cols = self.exog_cols_

        # Prepare data
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col must be specified")
            if date_col and date_col in data.columns:
                data = data.sort_values(date_col)
            y_true = data[target_col].values

            # Extract exog data if available
            if self.has_exog_ and exog_cols:
                exog_data_full = data[exog_cols].values
            else:
                exog_data_full = None
        else:
            y_true = np.array(data).flatten()
            exog_data_full = None

        # Make predictions using rolling windows
        predictions_list = []
        actuals_list = []
        n_samples = len(y_true) - self.history_length - self.forecast_horizon + 1

        # Sample windows evenly across the data
        window_indices = np.linspace(0, max(n_samples - 1, 0), min(n_windows, n_samples), dtype=int)

        for i in window_indices:
            start = i
            end = start + self.history_length
            window = y_true[start:end]

            # Normalize
            window_norm = (window - self.scaler_mean_) / self.scaler_std_

            # Prepare exog if available
            exog_tensor = None
            if self.has_exog_ and exog_data_full is not None:
                exog_window = exog_data_full[start:end]
                # Normalize using fitted parameters
                exog_norm = (exog_window - self.exog_mean_) / self.exog_std_
                exog_tensor = torch.tensor(
                    exog_norm,
                    dtype=torch.float32
                ).T.unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                x = torch.tensor(
                    window_norm,
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(self.device)

                t_span = torch.linspace(0, 1, steps=self.history_length, device=self.device)
                assert self.model is not None, "Model must be initialized"
                preds, _ = self.model(x, t_span, exog=exog_tensor)
                pred_denorm = preds.cpu().numpy().flatten() * self.scaler_std_ + self.scaler_mean_

            # Store predictions and actuals
            actual_window = y_true[i + self.history_length:i + self.history_length + self.forecast_horizon]
            predictions_list.append(pred_denorm[:len(actual_window)])
            actuals_list.append(actual_window)

        # Flatten arrays
        if len(predictions_list) == 0:
            raise ValueError(
                f"Not enough data to compute residuals. "
                f"Need at least {self.history_length + self.forecast_horizon} samples, "
                f"but got {len(y_true)}."
            )

        predictions = np.concatenate(predictions_list)
        actuals = np.concatenate(actuals_list)
        residuals = actuals - predictions

        # Store for later use
        self.residuals_ = residuals
        self.residual_actuals_ = actuals
        self.residual_predictions_ = predictions

        return residuals, actuals, predictions

    def plot_residuals(
        self,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive residual diagnostics (4-panel plot).

        Creates a 4-panel diagnostic plot:
        1. Residuals over time
        2. Residual distribution (histogram + KDE)
        3. Autocorrelation Function (ACF)
        4. Q-Q plot (normality test)

        NEW in v0.3.0!

        Parameters
        ----------
        data : DataFrame or array, optional
            Data to compute residuals on. If None, uses stored residuals from compute_residuals()
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame
        exog_cols : list of str, optional
            Exogenous column names
        figsize : tuple, default=(14, 10)
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot

        Examples
        --------
        >>> model.fit(train_df, target_col='sales')
        >>> model.plot_residuals(test_df)  # Computes and plots residuals

        >>> # Or compute first, then plot multiple times
        >>> model.compute_residuals(test_df)
        >>> model.plot_residuals()  # Uses stored residuals
        """
        import matplotlib.pyplot as plt
        from scipy import stats

        # Compute residuals if needed
        if data is not None:
            residuals, _, _ = self.compute_residuals(
                data, target_col, date_col, exog_cols
            )
        elif self.residuals_ is not None:
            residuals = self.residuals_
        else:
            raise ValueError(
                "No residuals available. Either provide data parameter or call compute_residuals() first."
            )

        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{self.model_type.upper()} - Residual Diagnostics', fontsize=16, y=0.995)

        # 1. Residuals over time
        ax1 = axes[0, 0]
        ax1.plot(residuals, 'o-', alpha=0.6, markersize=3, linewidth=0.8)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Line')
        ax1.axhline(y=np.mean(residuals), color='g', linestyle=':', linewidth=2,
                    label=f'Mean: {np.mean(residuals):.4f}')
        ax1.fill_between(range(len(residuals)),
                         -2 * np.std(residuals),
                         2 * np.std(residuals),
                         alpha=0.2, color='gray', label='±2 Std')
        ax1.set_xlabel('Sample Index', fontsize=10)
        ax1.set_ylabel('Residual', fontsize=10)
        ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 2. Residual distribution
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')

        # Fit normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

        # KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals)
            ax2.plot(x, kde(x), 'g--', linewidth=2, label='KDE')
        except Exception:
            pass  # Skip KDE if it fails

        ax2.set_xlabel('Residual Value', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis='y')

        # 3. ACF plot
        ax3 = axes[1, 0]
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            max_lags = max(1, min(40, len(residuals) // 4))  # At least 1 lag
            plot_acf(residuals, lags=max_lags, ax=ax3, alpha=0.05)
            ax3.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Lag', fontsize=10)
            ax3.set_ylabel('ACF', fontsize=10)
        except (ImportError, Exception):
            # Fallback: manual ACF computation
            max_lag = max(1, min(40, len(residuals) // 4))  # At least 1 lag
            acf_values = [1.0]

            for lag in range(1, min(max_lag + 1, len(residuals))):
                try:
                    acf_val = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    if np.isnan(acf_val):
                        acf_val = 0.0
                    acf_values.append(acf_val)
                except (IndexError, ValueError):
                    acf_values.append(0.0)

            ax3.bar(range(len(acf_values)), acf_values, width=0.3, alpha=0.7, color='steelblue')
            ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

            # Confidence intervals
            conf_int = 1.96 / np.sqrt(max(len(residuals), 1))  # Avoid division by zero
            ax3.axhline(y=conf_int, color='b', linestyle='--', linewidth=1)
            ax3.axhline(y=-conf_int, color='b', linestyle='--', linewidth=1)

            ax3.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Lag', fontsize=10)
            ax3.set_ylabel('ACF', fontsize=10)
            ax3.grid(alpha=0.3)

        # 4. Q-Q plot
        ax4 = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax4.set_ylabel('Sample Quantiles', fontsize=10)
        ax4.grid(alpha=0.3)

        # Add text box with statistics
        stats_text = (
            f"n = {len(residuals)}\n"
            f"Mean = {np.mean(residuals):.4f}\n"
            f"Std = {np.std(residuals):.4f}\n"
            f"MAE = {np.mean(np.abs(residuals)):.4f}\n"
            f"RMSE = {np.sqrt(np.mean(residuals**2)):.4f}"
        )
        ax4.text(0.05, 0.95, stats_text,
                 transform=ax4.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                logger.info(f"Residual plot saved to {save_path}")

        plt.show()

        return fig, axes

    def analyze_residuals(
        self,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None
    ) -> Dict[str, Union[int, float, None]]:
        """
        Perform statistical analysis of residuals.

        Returns diagnostic statistics and test results to assess model fit quality.

        NEW in v0.3.0!

        Parameters
        ----------
        data : DataFrame or array, optional
            Data to compute residuals on. If None, uses stored residuals
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame
        exog_cols : list of str, optional
            Exogenous column names

        Returns
        -------
        diagnostics : dict
            Dictionary containing:
            - 'mean': Mean residual (should be ~0)
            - 'std': Standard deviation of residuals
            - 'mae': Mean absolute error
            - 'rmse': Root mean squared error
            - 'skewness': Skewness (should be ~0 for normal)
            - 'kurtosis': Kurtosis (should be ~3 for normal)
            - 'ljung_box_stat': Ljung-Box statistic (autocorrelation test)
            - 'ljung_box_pvalue': P-value for Ljung-Box test
            - 'shapiro_stat': Shapiro-Wilk statistic (normality test)
            - 'shapiro_pvalue': P-value for Shapiro-Wilk test
            - 'n_samples': Number of residual samples

        Examples
        --------
        >>> model.fit(train_df, target_col='sales')
        >>> diagnostics = model.analyze_residuals(test_df)
        >>> print(f"Mean residual: {diagnostics['mean']:.4f}")
        >>> print(f"Normality test p-value: {diagnostics['shapiro_pvalue']:.4f}")
        """
        from scipy import stats

        # Compute residuals if needed
        if data is not None:
            residuals, _, _ = self.compute_residuals(data, target_col, date_col, exog_cols)
        elif self.residuals_ is not None:
            residuals = self.residuals_
        else:
            raise ValueError(
                "No residuals available. Either provide data parameter or call compute_residuals() first."
            )

        # Basic statistics
        diagnostics: Dict[str, Union[int, float, None]] = {
            'n_samples': len(residuals),
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        }

        # Shapiro-Wilk test for normality (works for n < 5000)
        if len(residuals) < 5000:
            try:
                shapiro_stat, shapiro_pval = stats.shapiro(residuals)
                diagnostics['shapiro_stat'] = float(shapiro_stat)
                diagnostics['shapiro_pvalue'] = float(shapiro_pval)
            except Exception as e:
                diagnostics['shapiro_stat'] = None
                diagnostics['shapiro_pvalue'] = None
                if self.verbose:
                    logger.info(f"Shapiro-Wilk test failed: {e}")
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            try:
                ks_stat, ks_pval = stats.kstest(
                    residuals, 'norm',
                    args=(np.mean(residuals), np.std(residuals)))
                diagnostics['ks_stat'] = float(ks_stat)
                diagnostics['ks_pvalue'] = float(ks_pval)
            except Exception:
                pass

        # Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            max_lags = max(1, min(10, len(residuals) // 4))  # At least 1 lag
            lb_result = acorr_ljungbox(residuals, lags=[max_lags], return_df=False)
            diagnostics['ljung_box_stat'] = float(lb_result[0][0])
            diagnostics['ljung_box_pvalue'] = float(lb_result[1][0])
        except (ImportError, Exception):
            # Manual Ljung-Box if statsmodels not available or fails
            try:
                max_lag = max(1, min(10, len(residuals) // 4))  # At least 1 lag
                n = len(residuals)
                acf_vals = []

                for lag in range(1, min(max_lag + 1, len(residuals))):
                    try:
                        acf_val = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                        if np.isnan(acf_val):
                            acf_val = 0.0
                        acf_vals.append(acf_val)
                    except (IndexError, ValueError):
                        acf_vals.append(0.0)

                if len(acf_vals) > 0:
                    lb_stat = n * (n + 2) * np.sum([
                        (acf_vals[k] ** 2) / max(n - k - 1, 1)
                        for k in range(len(acf_vals))])
                    lb_pval = 1 - stats.chi2.cdf(lb_stat, len(acf_vals))

                    diagnostics['ljung_box_stat'] = float(lb_stat)
                    diagnostics['ljung_box_pvalue'] = float(lb_pval)
                else:
                    diagnostics['ljung_box_stat'] = None
                    diagnostics['ljung_box_pvalue'] = None
            except Exception:
                diagnostics['ljung_box_stat'] = None
                diagnostics['ljung_box_pvalue'] = None

        # Print summary if verbose
        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("RESIDUAL ANALYSIS")
            logger.info("=" * 70)
            logger.info(f"\nSample Size: {diagnostics['n_samples']}")
            logger.info("\nCentral Tendency:")
            logger.info(f"  Mean Residual:       {diagnostics['mean']:>10.4f}  (should be ≈0)")
            logger.info(f"  Std Residual:        {diagnostics['std']:>10.4f}")
            logger.info(f"  MAE:                 {diagnostics['mae']:>10.4f}")
            logger.info(f"  RMSE:                {diagnostics['rmse']:>10.4f}")

            logger.info("\nDistribution Shape:")
            logger.info(f"  Skewness:            {diagnostics['skewness']:>10.4f}  (0 = symmetric)")
            logger.info(f"  Kurtosis:            {diagnostics['kurtosis']:>10.4f}  (0 = normal)")

            logger.info("\nNormality Test:")
            if 'shapiro_pvalue' in diagnostics and diagnostics['shapiro_pvalue'] is not None:
                logger.info(f"  Shapiro-Wilk p-val:  {diagnostics['shapiro_pvalue']:>10.4f}  "
                            f"({'Normal' if diagnostics['shapiro_pvalue'] > 0.05 else 'Non-normal'})")
            elif 'ks_pvalue' in diagnostics and diagnostics['ks_pvalue'] is not None:
                logger.info(f"  K-S p-val:           {diagnostics['ks_pvalue']:>10.4f}  "
                            f"({'Normal' if diagnostics['ks_pvalue'] > 0.05 else 'Non-normal'})")

            logger.info("\nAutocorrelation Test:")
            if diagnostics['ljung_box_pvalue'] is not None:
                logger.info(f"  Ljung-Box p-val:     {diagnostics['ljung_box_pvalue']:>10.4f}  "
                            f"({'No autocorr' if diagnostics['ljung_box_pvalue'] > 0.05 else 'Autocorrelated'})")

            logger.info("\n" + "=" * 70)
            logger.info("\nInterpretation:")
            logger.info("  • Mean should be close to 0 (unbiased predictions)")
            logger.info("  • Skewness/Kurtosis should be close to 0 (normally distributed)")
            logger.info("  • Normality test p-value > 0.05 indicates normal residuals")
            logger.info("  • Ljung-Box p-value > 0.05 indicates no autocorrelation (good)")
            logger.info("=" * 70 + "\n")

        return diagnostics
