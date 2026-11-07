"""Device-fit levels for the Download tab (issue #40, wave #38).

A level is the ratio of model size to available RAM - a fit
judgment, positive by design: every level says what the
experience IS, never what to fear. The numeric TPS prediction
comes from tps_predictor (bandwidth, measured or estimated);
the tps_range returned here is the qualitative character of
the pace, not a number pretending to be a measurement.

Ratios (model_size_gb / ram_gb):
    A: below 0.7   - Interactive
    B: 0.7 to 1.2  - Reading, tight fit
    C: 1.2 to 4    - Overflow (HiveMind's home ground)
    D: 4 to 20     - Experimental
    E: 20 and up   - Proof of concept
"""

LEVEL_DESCRIPTIONS = {
    "A": "Interactive — fits in RAM",
    "B": "Reading — tight fit",
    "C": "Overflow — this is what HiveMind is built for",
    "D": "Experimental — beyond daily use, but it works",
    "E": "Proof of concept — slow, but proves the physics",
}

LEVEL_TPS_RANGES = {
    "A": "Interactive pace",
    "B": "Reading pace",
    "C": "Overflow pace — steady, slower than reading",
    "D": "Patient pace — expect minutes per reply",
    "E": "Proof pace — replies can take tens of minutes",
}

# Badge colors for the UI: A green (the exit-gate badge), then
# cooler to warmer as the ratio grows.
LEVEL_COLORS = {
    "A": "#1a7f37",
    "B": "#0969da",
    "C": "#9a6700",
    "D": "#bc4c00",
    "E": "#8250df",
}


def classify_level(model_size_gb: float, ram_gb: float) -> tuple:
    """Return (level, tps_range, description) for a fit ratio.

    The boundaries are the wave #38 ratios: A below 0.7,
    B 0.7-1.2, C 1.2-4, D 4-20, E above 20.
    """
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be positive")
    if ram_gb <= 0:
        raise ValueError("ram_gb must be positive")
    ratio = model_size_gb / ram_gb
    if ratio < 0.7:
        level = "A"
    elif ratio < 1.2:
        level = "B"
    elif ratio < 4:
        level = "C"
    elif ratio < 20:
        level = "D"
    else:
        level = "E"
    return (level, LEVEL_TPS_RANGES[level], LEVEL_DESCRIPTIONS[level])
