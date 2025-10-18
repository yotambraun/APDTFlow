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

from .models.apdtflow import APDTFlow
from .models.transformer_forecaster import TransformerForecaster
from .models.tcn_forecaster import TCNForecaster
from .models.ensemble_forecaster import EnsembleForecaster


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
        calibration_split: float = 0.2
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

        # Exogenous variables (NEW)
        self.exog_fusion_type = exog_fusion_type
        self.exog_cols_ = None
        self.future_exog_cols_ = None
        self.num_exog_features_ = 0
        self.exog_mean_ = None
        self.exog_std_ = None
        self.has_exog_ = False

        # Conformal prediction (NEW)
        self.use_conformal = use_conformal
        self.conformal_method = conformal_method
        self.calibration_split = calibration_split
        self.conformal_predictor = None

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model will be initialized in fit()
        self.model = None
        self._is_fitted = False

        # Store data info
        self.scaler_mean_ = None
        self.scaler_std_ = None
        self.last_sequence_ = None
        self.last_exog_sequence_ = None  # NEW
        self.target_col_ = None
        self.date_col_ = None
        self.data_df_ = None

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
        exog_X_list, exog_y_list = [], []
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

        # Create DataLoader
        if has_exog_data:
            dataset = TensorDataset(X, y, exog_X, exog_y)
        else:
            dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_data in dataloader:
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

            avg_loss = epoch_loss / len(dataset)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True

        if self.verbose:
            print("Training completed!")

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
                        preds_denorm.reshape(-1, 1), alpha=alpha
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

        # Get predictions
        if show_uncertainty:
            preds, uncertainty = self.predict(return_uncertainty=True)
        else:
            preds = self.predict()
            uncertainty = None

        # Get historical data
        if self.data_df_ is not None and self.target_col_ is not None:
            history = self.data_df_[self.target_col_].values[-with_history:]
        else:
            # Reconstruct from last_sequence_
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
        metric: str = 'mse'
    ) -> float:
        """
        Evaluate model performance.

        Parameters
        ----------
        data : DataFrame or array
            Test data
        target_col : str, optional
            Name of target column if data is DataFrame
        date_col : str, optional
            Name of date column if data is DataFrame
        metric : str, default='mse'
            Metric to use: 'mse', 'mae', or 'mape'

        Returns
        -------
        score : float
            Model score
        """
        # Implementation for scoring
        # This would evaluate on test data
        raise NotImplementedError("score() will be implemented in next iteration")
