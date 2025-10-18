import sys
import subprocess

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "apdtflow.cli", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    help_text = result.stdout.lower() + result.stderr.lower()
    assert "usage:" in help_text