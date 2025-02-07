from .data import TimeSeriesWindowDataset
from .augmentation import jitter, scaling, time_warp
from .preprocessing import TimeSeriesScaler,convert_to_datetime,fill_time_gaps,impute_missing_values,detrend_series,decompose_series,generate_lag_features,generate_rolling_features
from .cv_factory import TimeSeriesCVFactory
from .training import train_forecaster
from .inference import infer_forecaster
from .logger_util import get_logger
import os
import sys

def _print_welcome_banner():
    banner = r"""
    ────────────────────────────────────────────────
         🚀 Welcome to APDTFlow! 🚀
    
        Your go-to framework for flexible, 
        modular, and powerful time series forecasting.  
        
        Built for pros. Designed for performance. ⚡
    
        Let's get forecasting! 📈
    ────────────────────────────────────────────────
    """
    print(banner)

if sys.stdout.isatty() and os.environ.get("APDTFLOW_BANNER_PRINTED") is None:
    _print_welcome_banner()
    os.environ["APDTFLOW_BANNER_PRINTED"] = "true"