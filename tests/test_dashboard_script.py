"""The dashboard page script keeps the tab, the open run sections and the scroll position across its reloads.

The script runs under bun in a stub DOM (tests/dashboard_script_harness.js); skipped where bun is absent.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sparse_dsb.dashboard import SCRIPT

HERE = Path(__file__).parent


def test_dashboard_script_persistence(tmp_path):
    bun = shutil.which("bun") or next((str(p) for p in (Path.home() / ".local" / "bin" / "bun",) if p.exists()), None)
    if bun is None:
        pytest.skip("bun not available")
    script_path = tmp_path / "dashboard_script.js"
    script_path.write_text(SCRIPT)
    result = subprocess.run([bun, str(HERE / "dashboard_script_harness.js"), str(script_path)], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout and result.stdout.count("ok:") == 13
