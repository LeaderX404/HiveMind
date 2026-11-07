"""The ladder is judged by a linter, not a gate (issue #29).

The four-gate ladder produces ORDER, not a number, so it is
judged structurally - this test reads the source text and fails
the build when a gate is missing or wired out of order. It is
deliberately NOT a fifth gate: a gate judges a number, and the
ladder produces no number. The stopping rule is the Gate Law in
CONTRIBUTING.md (every gate needs a named decision and a public
issue), not another gate.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = (REPO_ROOT / "benchmark.py").read_text()
MAIN = SRC[SRC.index("def main("):SRC.index('if __name__')]
PRE = (REPO_ROOT / "PRE_REGISTERED.md").read_text()
CONTRIB = (REPO_ROOT / "CONTRIBUTING.md").read_text()


def test_completeness_every_layer_has_its_gate():
    for gate in ("live_capture_gate", "trace_capture_gate",
                 "coverage_audit_gate", "overflow_gate_verdict"):
        assert f"def {gate}" in SRC, gate


def test_completeness_covenant_names_the_gates():
    for marker in ("### Live capture gate", "### Coverage audit",
                   "live capture -> static capture ->"):
        assert marker in PRE, marker


def test_gate_law_recorded_in_contributing():
    assert "named decision" in CONTRIB
    assert "No reserve gates" in CONTRIB


def test_order_audit_runs_before_record():
    assert MAIN.index("coverage_audit_gate()") < MAIN.index(
        "result = record_alloc_trace(")


def test_order_live_gate_runs_before_static_gate():
    assert MAIN.index("live_capture_gate()") < MAIN.index(
        "trace_capture_gate(gate_trace)")


def test_calibration_refuses_without_capture_gate():
    assert "capture gate required before calibration" in SRC


def test_overflow_verdict_refuses_without_calibration_input():
    assert "--gate overflow needs" in MAIN
