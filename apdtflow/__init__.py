"""APDTFlow: continuous-time forecasting with Neural ODEs.

One trained model answers three questions: ``predict()`` (grid forecasts),
``predict_at(timestamps)`` (any real-valued time), and
``predict_when(threshold)`` (event timing with calibrated uncertainty).
"""

from .forecaster import APDTFlowForecaster

__version__ = "0.4.0"

__all__ = ["APDTFlowForecaster", "__version__"]
