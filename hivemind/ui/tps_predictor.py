"""Tokens/sec prediction for a model on a hardware profile (#31 phase 1).

The Phase 1 formula is deliberately simple and DECLARED, not
invented:

    tps = (disk_gbs / model_size_gb) * OVERHEAD_FACTOR

OVERHEAD_FACTOR = 0.7 comes from preliminary measurements: with a
cold page cache, streaming the weights from the NVMe dominates the
time to first token, and observed throughput landed at about 70%
of the raw disk streaming rate. It is a placeholder constant -
later phases replace it with a measured per-device calibration,
never with a nicer-looking guess.
"""
OVERHEAD_FACTOR = 0.7  # derived from preliminary measurements (see docstring)
WORDS_50_TOKENS = 65  # 50 words of English text are roughly 65 tokens


def predict_tps(model_size_gb: float, disk_gbs: float) -> float:
    """Predict tokens/sec: (disk_gbs / model_size_gb) * OVERHEAD_FACTOR."""
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be positive")
    if disk_gbs <= 0:
        raise ValueError("disk_gbs must be positive")
    return (disk_gbs / model_size_gb) * OVERHEAD_FACTOR


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
