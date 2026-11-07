"""Tokens/sec prediction for a model on a hardware profile (#31 phase 1).

Phase 1 amendment: every predicted number is now a RANGE with a
SOURCE, never a bare point estimate. A number that declares where
it came from cannot masquerade as a measurement:

- a real measurement (recorded by record_actual_load after an
  actual load) outranks every formula and is shown as a narrow
  band around the measured value;
- without a measurement, the disk-speed calculator returns a
  wide range (0.5x - 1.0x of the raw streaming rate) and says so
  in its source string.

OVERHEAD_FACTOR is an INTERNAL detail only: a single point
constant from preliminary measurements. It must never appear in
any UI output - ranges and sources are what the user sees. See
MEASUREMENT_POLICY.md.
"""
from typing import Optional, Tuple

OVERHEAD_FACTOR = 0.7  # internal detail from preliminary measurements; never shown in output
WORDS_50_TOKENS = 65  # 50 words of English text are roughly 65 tokens

RANGE_LOW_FACTOR = 0.5   # declared lower bound of an estimate
RANGE_HIGH_FACTOR = 1.0  # declared upper bound (raw disk streaming rate)
MEASURED_BAND = 0.1      # declared: a measurement is a band, not a law

SOURCE_ESTIMATED = "estimated from disk speed, not measured"
SOURCE_MEASURED = "measured on this device"


def predict_tps_range(model_size_gb: float, disk_gbs: float,
                      measured_gbs: Optional[float] = None) -> Tuple[float, float, str]:
    """Return (low, high, source) tokens/sec for a model.

    A real measurement wins outright and is shown as a band
    (roughly +/-10%): one load is one sample, not a law. Without
    one, the disk-speed calculator produces a wide, honestly
    labeled range. The source string travels with the numbers -
    no number is returned without one.
    """
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be positive")
    if measured_gbs is not None and measured_gbs > 0:
        return (
            measured_gbs * (1.0 - MEASURED_BAND),
            measured_gbs * (1.0 + MEASURED_BAND),
            SOURCE_MEASURED,
        )
    if disk_gbs <= 0:
        raise ValueError("disk_gbs must be positive")
    rate = disk_gbs / model_size_gb
    return (
        rate * RANGE_LOW_FACTOR,
        rate * RANGE_HIGH_FACTOR,
        SOURCE_ESTIMATED,
    )


def predict_oom_risk(model_size_gb: float, ram_gb: float) -> str:
    """OOM risk as "none", "warning" or "high" - a warning, never a block."""
    if model_size_gb < ram_gb * 0.7:
        return "none"
    if model_size_gb < ram_gb * 0.9:
        return "warning"
    return "high"


def estimate_50_words_seconds(tps: float) -> float:
    """Seconds to generate ~50 words (~65 tokens) at tps tokens/sec."""
    if tps <= 0:
        raise ValueError("tps must be positive")
    return WORDS_50_TOKENS / tps
