import logging
import os
from typing import Optional


def get_logger(name: str = "apdtflow", log_file: Optional[str] = None) -> logging.Logger:
    """Create and return a logger.

    By default the logger only writes to the console. Pass ``log_file`` to
    additionally write to a file (parent directories are created if needed).
    Importing the library never creates log files.

    Args:
        name: Name of the logger.
        log_file: Optional path of a file to log to.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = os.path.abspath(log_file)
        has_file = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_path
            for h in logger.handlers
        )
        if not has_file:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
