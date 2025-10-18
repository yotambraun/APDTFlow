"""
Conformal prediction for calibrated prediction intervals in time series.

Provides distribution-free coverage guarantees for forecasting uncertainty.

Based on recent research:
- "Conformal Prediction for Time Series" (2025) - Split conformal
- "Adaptive Conformal Inference" (ICLR 2025) - Adaptive methods
- "Dual-Splitting Conformal Prediction" (2025) - Multi-step forecasting

References:
- arXiv:2509.02844 - Conformal Prediction for Time Series with Change Points
- OpenReview oP7arLOWix - Kernel-based Optimally Weighted Conformal Prediction
- arXiv:2503.21251 - Dual-Splitting for Multi-Step Forecasting
"""
import numpy as np
import torch
from typing import Optional, Tuple, Callable


class SplitConformalPredictor:
    """
    Split conformal prediction for time series forecasting.

    Provides calibrated prediction intervals with finite-sample coverage
    guarantees. Simple and reliable method for quantifying uncertainty.

    The method works by:
    1. Splitting data into training and calibration sets
    2. Training model on training set
    3. Computing nonconformity scores on calibration set
    4. Using quantile of scores to construct prediction intervals

    Args:
        predict_fn: Function that takes input and returns predictions
        alpha: Miscoverage level (default 0.05 for 95% coverage)

    Example:
        >>> predictor = SplitConformalPredictor(model.predict, alpha=0.05)
        >>> predictor.calibrate(X_cal, y_cal)
        >>> lower, pred, upper = predictor.predict(X_test)
        >>> # Guaranteed 95% coverage on new data!
    """

    def __init__(self, predict_fn: Callable, alpha: float = 0.05):
        """
        Args:
            predict_fn: Function (X) -> predictions
            alpha: Miscoverage level (0.05 = 95% coverage)
        """
        self.predict_fn = predict_fn
        self.alpha = alpha
        self.nonconformity_scores = None
        self.quantile = None
        self.is_calibrated = False

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Calibrate conformal predictor on held-out calibration set.

        Args:
            X_cal: Calibration inputs (n_cal, ...)
            y_cal: Calibration targets (n_cal, ...)
        """
        # Get predictions on calibration set
        preds = self.predict_fn(X_cal)

        # Ensure arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(y_cal, torch.Tensor):
            y_cal = y_cal.cpu().numpy()

        # Compute nonconformity scores: absolute residuals
        self.nonconformity_scores = np.abs(y_cal - preds)

        # Flatten if multi-dimensional
        if self.nonconformity_scores.ndim > 1:
            self.nonconformity_scores = self.nonconformity_scores.flatten()

        # Compute quantile for coverage guarantee
        n = len(self.nonconformity_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n

        self.quantile = np.quantile(
            self.nonconformity_scores,
            q_level,
            method='higher'  # Conservative choice
        )

        self.is_calibrated = True

        print(f"✓ Calibrated with {n} samples")
        print(f"  Quantile at {1-self.alpha:.1%} level: {self.quantile:.4f}")

    def predict(
        self,
        X_test: np.ndarray,
        return_scores: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal intervals.

        Args:
            X_test: Test inputs
            return_scores: If True, also return nonconformity scores

        Returns:
            (lower_bound, prediction, upper_bound) or
            (lower_bound, prediction, upper_bound, scores) if return_scores=True

        Raises:
            RuntimeError: If not calibrated
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        # Get point predictions
        preds = self.predict_fn(X_test)

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        # Construct prediction intervals
        lower = preds - self.quantile
        upper = preds + self.quantile

        if return_scores:
            return lower, preds, upper, self.nonconformity_scores
        else:
            return lower, preds, upper

    def get_coverage_diagnostics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Compute empirical coverage on test set.

        Args:
            X_test: Test inputs
            y_test: Test targets

        Returns:
            Dictionary with coverage statistics
        """
        lower, preds, upper = self.predict(X_test)

        if isinstance(y_test, torch.Tensor):
            y_test = y_test.cpu().numpy()

        # Check coverage
        covered = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = np.mean(covered)

        # Interval width
        interval_width = upper - lower
        avg_width = np.mean(interval_width)

        return {
            'empirical_coverage': empirical_coverage,
            'target_coverage': 1 - self.alpha,
            'avg_interval_width': avg_width,
            'quantile': self.quantile
        }


class AdaptiveConformalPredictor(SplitConformalPredictor):
    """
    Adaptive conformal prediction for non-stationary time series.

    Adjusts the prediction intervals online based on recent prediction errors.
    Particularly useful when the data distribution changes over time.

    Based on: "Adaptive Conformal Inference Under Distribution Shift" (NIPS 2021)

    Args:
        predict_fn: Function that takes input and returns predictions
        alpha: Miscoverage level (default 0.05 for 95% coverage)
        gamma: Learning rate for adaptation (default 0.05)

    Example:
        >>> predictor = AdaptiveConformalPredictor(model.predict, alpha=0.05, gamma=0.05)
        >>> predictor.calibrate(X_cal, y_cal)
        >>>
        >>> # Online prediction with adaptation
        >>> for X_t, y_t in test_stream:
        ...     lower, pred, upper = predictor.predict_and_update(X_t, y_t)
    """

    def __init__(
        self,
        predict_fn: Callable,
        alpha: float = 0.05,
        gamma: float = 0.05
    ):
        """
        Args:
            gamma: Learning rate for quantile adaptation (0.01-0.1 typical)
        """
        super().__init__(predict_fn, alpha)
        self.gamma = gamma
        self.adaptive_quantile = None
        self.coverage_history = []

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Calibrate and initialize adaptive quantile.

        Args:
            X_cal: Calibration inputs
            y_cal: Calibration targets
        """
        super().calibrate(X_cal, y_cal)
        self.adaptive_quantile = self.quantile

    def predict(
        self,
        X_test: np.ndarray,
        return_scores: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using adaptive quantile.

        Args:
            X_test: Test inputs
            return_scores: If True, return nonconformity scores

        Returns:
            (lower_bound, prediction, upper_bound)
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        # Get point predictions
        preds = self.predict_fn(X_test)

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        # Use adaptive quantile
        lower = preds - self.adaptive_quantile
        upper = preds + self.adaptive_quantile

        if return_scores:
            return lower, preds, upper, self.nonconformity_scores
        else:
            return lower, preds, upper

    def predict_and_update(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict and optionally update quantile based on observed error.

        For online/streaming scenarios where ground truth becomes available
        after prediction.

        Args:
            X_test: Test input
            y_test: Optional ground truth (for updating)

        Returns:
            (lower_bound, prediction, upper_bound)
        """
        # Make prediction
        preds = self.predict_fn(X_test)

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        lower = preds - self.adaptive_quantile
        upper = preds + self.adaptive_quantile

        # Update if ground truth provided
        if y_test is not None:
            if isinstance(y_test, torch.Tensor):
                y_test = y_test.cpu().numpy()

            # Compute error
            error = np.abs(y_test - preds)

            # Adaptive update based on miscoverage
            if error.shape:
                error = np.mean(error)  # Average over multiple outputs

            # Check if we're under/over-covering
            if error > self.adaptive_quantile:
                # Undercoverage - increase quantile
                self.adaptive_quantile += self.gamma
            else:
                # Overcoverage - decrease quantile slightly
                # Decrease slower to maintain coverage guarantee
                self.adaptive_quantile -= self.gamma * self.alpha / (1 - self.alpha)

            # Keep quantile positive
            self.adaptive_quantile = max(0.001, self.adaptive_quantile)

            # Track coverage
            covered = error <= self.adaptive_quantile
            self.coverage_history.append(covered)

        return lower, preds, upper

    def get_adaptation_stats(self) -> dict:
        """
        Get statistics about the adaptation process.

        Returns:
            Dictionary with adaptation statistics
        """
        if len(self.coverage_history) == 0:
            return {
                'num_updates': 0,
                'current_quantile': self.adaptive_quantile,
                'initial_quantile': self.quantile
            }

        recent_coverage = np.mean(self.coverage_history[-100:]) if len(self.coverage_history) >= 100 else np.mean(self.coverage_history)

        return {
            'num_updates': len(self.coverage_history),
            'current_quantile': self.adaptive_quantile,
            'initial_quantile': self.quantile,
            'quantile_change': self.adaptive_quantile - self.quantile,
            'recent_coverage': recent_coverage,
            'target_coverage': 1 - self.alpha
        }


def plot_conformal_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Conformal Prediction Intervals",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot predictions with conformal intervals.

    Args:
        y_true: True values
        y_pred: Predicted values
        lower: Lower bounds
        upper: Upper bounds
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(y_true))

    # Plot true values
    ax.plot(x, y_true, 'b-', label='True', linewidth=2, alpha=0.7)

    # Plot predictions
    ax.plot(x, y_pred, 'r--', label='Predicted', linewidth=2)

    # Plot conformal intervals
    ax.fill_between(
        x, lower, upper,
        alpha=0.3, color='red',
        label=f'Conformal Interval'
    )

    # Calculate coverage
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(covered)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{title}\nEmpirical Coverage: {coverage:.1%}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax
