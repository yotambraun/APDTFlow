import os
import sys
from .logger_util import get_logger
from .forecaster import APDTFlowForecaster

get_logger("APDTFlow")

# Expose main API
__all__ = ['APDTFlowForecaster']


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
