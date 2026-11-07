"""The ladder enforced at runtime, not just in text (issue #30).

test_ladder_lint.py guards the ladder as TEXT (structure and
order of the wiring). This module guards the BEHAVIOR: every
gate refuses when its predecessor has not passed in the same
invocation. The refusals are invocation-level and therefore
device-independent - they fire before any hardware is touched,
so they hold identically on a CI runner and on a real device.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmark.py"


def run_bench(*flags: str) -> subprocess.CompletedProcess:
    """Run benchmark.py with the given flags; never raise on exit."""
    return subprocess.run(
        [sys.executable, str(BENCH), *flags],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )


def test_replay_calibrate_refuses_without_capture_gate():
    r = run_bench("--replay-calibrate")
    assert r.returncode != 0
    assert "capture gate required before calibration" in r.stdout + r.stderr


def test_trace_capture_gate_refuses_without_coverage_audit():
    r = run_bench("--trace-capture-gate")
    assert r.returncode != 0
    assert "needs --coverage-audit" in r.stdout + r.stderr


def test_gate_overflow_refuses_without_calibration_facts():
    r = run_bench("--gate", "overflow")
    assert r.returncode != 0
    assert "--gate overflow needs" in r.stdout + r.stderr


def test_record_trace_refuses_without_live_capture_gate():
    r = run_bench("--record-trace")
    assert r.returncode != 0
    assert "needs --live-capture-gate" in r.stdout + r.stderr
