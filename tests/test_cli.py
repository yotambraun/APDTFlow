import sys
import subprocess
import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "apdtflow.cli", "--help"], capture_output=True, text=True
    )
    assert "usage:" in result.stdout
