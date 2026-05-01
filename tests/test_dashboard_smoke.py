import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_executes_in_bare_mode():
    result = subprocess.run(
        [sys.executable, "dashboard/app.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr[-2000:]
