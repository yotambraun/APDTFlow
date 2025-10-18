import os
from apdtflow.logger_util import get_logger


def test_get_logger(tmp_path):
    log_file = str(tmp_path / "test.log")
    logger = get_logger("test_logger", log_file=log_file)
    assert len(logger.handlers) >= 2
    logger.info("Test message")
    assert os.path.exists(log_file)
