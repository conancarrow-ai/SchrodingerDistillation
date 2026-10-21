"""Shared fixtures of the test suite.

check_close compares a port value with a reference value, records the largest
absolute difference as a test property and asserts np.allclose with the given
tolerances. Recorded properties are printed in the terminal summary, so the
numbers a validation run achieved are visible without -s or -rA.
"""
from __future__ import annotations

import numpy as np
import pytest

_records: list[tuple[str, list]] = []


@pytest.fixture
def check_close(record_property):
    def check(name: str, port, reference, rtol: float = 1e-5, atol: float = 1e-5):
        port = np.asarray(port, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        assert port.shape == reference.shape, f"{name}: shapes {port.shape} vs {reference.shape}"
        difference = np.abs(port - reference)
        record_property(f"{name}.max_abs_diff", float(difference.max()) if difference.size else 0.0)
        record_property(f"{name}.max_abs_ref", float(np.abs(reference).max()) if reference.size else 0.0)
        np.testing.assert_allclose(port, reference, rtol=rtol, atol=atol, err_msg=name)
    return check


def pytest_runtest_logreport(report):
    if report.when == "call" and report.user_properties:
        _records.append((report.nodeid, list(report.user_properties)))


def pytest_terminal_summary(terminalreporter):
    if not _records:
        return
    terminalreporter.section("recorded numbers")
    for nodeid, properties in _records:
        for name, value in properties:
            text = f"{value:.4e}" if isinstance(value, float) else str(value)
            terminalreporter.write_line(f"{nodeid} {name}={text}")
