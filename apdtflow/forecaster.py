"""
High-Level API for APDTFlow models.
Provides a simple fit/predict interface for easy time series forecasting.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Union, List, Tuple
import warnings
from tqdm import tqdm

from .models.apdtflow import APDTFlow
from .models.transformer_forecaster import TransformerForecaster
from .models.tcn_forecaster import TCNForecaster
from .models.ensemble_forecaster import EnsembleForecaster
from .conformal import SplitConformalPredictor, AdaptiveConformalPredictor


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
        model_type: str = 'apdtflow',
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
        # NEW: Exogenous variables support
        exog_fusion_type: str = 'gated',
        # NEW: Conformal prediction support
        use_conformal: bool = False,
        conformal_method: str = 'split',
        calibration_split: float = 0.2,
        # NEW: Early stopping
        early_stopping: bool = False,
        patience: int = 5,
        validation_split: float = 0.2
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

        # Conformal prediction (NEW)
        self.use_conformal = use_conformal
        self.conformal_method = conformal_method
        self.calibration_split = calibration_split
        self.conformal_predictor: Optional[Union[SplitConformalPredictor, AdaptiveConformalPredictor]] = None

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

    def _initialize_model(self):
        """Initialize the forecasting model based on model_type."""
        model_params = {
            'num_scales': self.num_scales,
            'input_channels': 1,
            'filter_size': self.filter_size,
            'hidden_dim': self.hidden_dim,
            'output_dim': 1,
            'forecast_horizon': self.forecast_horizon,
            'use_embedding': self.use_embedding
        }

        # Add exogenous parameters for APDTFlow
        if self.model_type == 'apdtflow':
            model_params['num_exog_features'] = self.num_exog_features_
            model_params['exog_fusion_type'] = self.exog_fusion_type
            model = APDTFlow(**model_params)
        elif self.model_type == 'transformer':
            model = TransformerForecaster(**model_params)
        elif self.model_type == 'tcn':
            model = TCNForecaster(**model_params)
        elif self.model_type == 'ensemble':
            model = EnsembleForecaster(**model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def _validate_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None
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
                nan_indices = np.where(np.isnan(arr))[0].tolist()
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
            series = data[target_col].values

            # Handle exogenous variables
            if self.has_exog_ and self.exog_cols_:
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

        X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        # Store last sequence for prediction
        self.last_sequence_ = series_norm[-self.history_length:]

        if exog_norm is not None:
            exog_X = torch.tensor(exog_X_list, dtype=torch.float32).transpose(1, 2)
            exog_y = torch.tensor(exog_y_list, dtype=torch.float32).transpose(1, 2)
            self.last_exog_sequence_ = exog_norm[-self.history_length:]
            return X, y, exog_X, exog_y

        return X, y

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None,
        future_exog_cols: Optional[List[str]] = None
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
        exog_cols : list of str, optional
            Names of exogenous variable columns (NEW in v0.2.0)
        future_exog_cols : list of str, optional
            Subset of exog_cols that are known in future (NEW in v0.2.0)

        Returns
        -------
        self : APDTFlowForecaster
            Fitted forecaster
        """
        if self.verbose:
            print(f"Fitting {self.model_type} model...")
            print(f"Device: {self.device}")

        # Validate data before proceeding
        self._validate_data(data, target_col, date_col, exog_cols)

        # Store column names
        self.target_col_ = target_col
        self.date_col_ = date_col
        self.exog_cols_ = exog_cols
        self.future_exog_cols_ = future_exog_cols

        # Check exogenous variables
        if exog_cols:
            self.has_exog_ = True
            self.num_exog_features_ = len(exog_cols)
            if self.verbose:
                print(f"Using {self.num_exog_features_} exogenous features: {exog_cols}")
                if future_exog_cols:
                    print(f"  Future-known features: {future_exog_cols}")

        if isinstance(data, pd.DataFrame):
            self.data_df_ = data.copy()

        # Prepare data
        data_output = self._prepare_data(data, target_col, date_col)
        if len(data_output) == 4:
            X, y, exog_X, exog_y = data_output
            has_exog_data = True
        else:
            X, y = data_output
            has_exog_data = False

        if self.verbose:
            print(f"Created {len(X)} training samples")
            print(f"Input shape: {X.shape}, Target shape: {y.shape}")
            if has_exog_data:
                print(f"Exog shape: {exog_X.shape}")

        # Initialize model
        self.model = self._initialize_model()
        assert self.model is not None, "Model initialization failed"

        # Create DataLoader(s) - split train/val if early stopping enabled
        if has_exog_data:
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
                print(f"Training samples: {n_train}, Validation samples: {n_val}")
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = None

        # Train
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
                preds, pred_logvars = self.model(x_batch, t_span, exog=exog_x_batch)

                # Loss (negative log-likelihood)
                mse = (preds - y_batch.transpose(1, 2)) ** 2
                loss = torch.mean(
                    0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
                )

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
                        loss = torch.mean(
                            0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
                        )

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
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                    # Restore best model
                    if best_model_state:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                # Update progress bar with training loss only
                pbar.set_postfix({'loss': f'{avg_train_loss:.4f}'})

        self._is_fitted = True

        if self.verbose:
            print("Training completed!")

        # Create conformal predictor if enabled
        if self.use_conformal:
            if self.verbose:
                print("Calibrating conformal predictor...")

            # Generate calibration predictions
            self.model.eval()
            calib_preds = []
            calib_targets = []

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

            calib_preds = np.concatenate(calib_preds, axis=0)
            calib_targets = np.concatenate(calib_targets, axis=0)

            # Denormalize for conformal predictor
            calib_preds_denorm = calib_preds * self.scaler_std_ + self.scaler_mean_
            calib_targets_denorm = calib_targets * self.scaler_std_ + self.scaler_mean_

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

            if self.verbose:
                print("Conformal predictor calibrated!")

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

        # Check exog requirements
        if self.has_exog_ and exog_future is None:
            raise ValueError(
                "Model was trained with exogenous variables. "
                "Please provide exog_future parameter for prediction."
            )

        self.model.eval()

        with torch.no_grad():
            # Prepare input
            x = torch.tensor(
                self.last_sequence_,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(self.device)

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
                print(f"Plot saved to {save_path}")

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

            # Extract exog data if available
            if self.has_exog_ and exog_cols:
                exog_data_full = data[exog_cols].values
            else:
                exog_data_full = None
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

        # Prepare state dict
        state = {
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

            # Conformal predictor state
            'conformal_state': conformal_state,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        if self.verbose:
            print(f"Model saved to {filepath}")

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

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

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
            exog_fusion_type=state['exog_fusion_type'],
            use_conformal=state['use_conformal'],
            conformal_method=state['conformal_method'],
            calibration_split=state['calibration_split'],
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
        print("=" * 70)
        print(f"APDTFlow Forecaster Summary - {self.model_type.upper()}")
        print("=" * 70)

        print("\nModel Configuration:")
        print(f"  Model Type:          {self.model_type}")
        print(f"  Forecast Horizon:    {self.forecast_horizon} steps")
        print(f"  History Length:      {self.history_length} steps")
        print(f"  Hidden Dimension:    {self.hidden_dim}")
        print(f"  Number of Scales:    {self.num_scales}")
        print(f"  Filter Size:         {self.filter_size}")
        print(f"  Use Embedding:       {self.use_embedding}")

        if self.has_exog_:
            print("\nExogenous Variables:")
            print(f"  Number of Features:  {self.num_exog_features_}")
            print(f"  Fusion Type:         {self.exog_fusion_type}")
            print(f"  Features:            {self.exog_cols_}")
            if self.future_exog_cols_:
                print(f"  Future-Known:        {self.future_exog_cols_}")

        if self.use_conformal:
            print("\nConformal Prediction:")
            print(f"  Method:              {self.conformal_method}")
            print(f"  Calibration Split:   {self.calibration_split}")

        print("\nTraining Configuration:")
        print(f"  Learning Rate:       {self.learning_rate}")
        print(f"  Batch Size:          {self.batch_size}")
        print(f"  Epochs:              {self.num_epochs}")
        print(f"  Device:              {self.device}")

        if self.model:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print("\nModel Parameters:")
            print(f"  Total Parameters:    {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            print(f"  Non-trainable:       {total_params - trainable_params:,}")

            # Estimate model size
            param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
            print(f"  Model Size:          ~{param_size_mb:.2f} MB")

        print("\nStatus:")
        print(f"  Fitted:              {self._is_fitted}")
        if self._is_fitted:
            print(f"  Target Column:       {self.target_col_}")
            if self.scaler_mean_ is not None:
                print(f"  Data Mean:           {self.scaler_mean_:.4f}")
                print(f"  Data Std:            {self.scaler_std_:.4f}")

        print("=" * 70)
