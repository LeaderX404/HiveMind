"""HiveMind UI - Phase 1 (issue #31, phase 1).

One Streamlit page: device profile, then every local model found
under ~/.hivemind/models/ with its predicted TPS RANGE and
source, OOM risk and 50-word estimate - all BEFORE anything is
loaded. Every number carries its source; nothing poses as a
measurement that was not measured. No chat yet: the "Start chat"
button stays disabled until Phase 2. This page displays, it does
not judge - the engine joins in Phase 2.

Phase 2 note: after the first real load of a model, the engine
must call record_actual_load(model_name, size_gb, load_seconds)
so the next run shows a measured band instead of a range.
"""
import json
from pathlib import Path

import streamlit as st

from hivemind.ui.hardware_detect import get_profile
from hivemind.ui.tps_predictor import (
    estimate_50_words_seconds,
    predict_oom_risk,
    predict_tps_range,
)

MODELS_DIR = Path.home() / ".hivemind" / "models"
PARAMS_BYTES = 2  # rough estimate: fp16 weights, 2 bytes per parameter
LEGACY_PROBE_MIB = 100  # shown when a cached profile predates probe-size journaling


def estimate_model_size_gb(model_dir: Path):
    """Rough model size in GB: real file sizes when safetensors exist,
    otherwise transformer math from config.json. None when neither works."""
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if safetensors:
        return sum(p.stat().st_size for p in safetensors) / 1e9
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, ValueError):
            return None
        layers = config.get("num_hidden_layers", 0)
        hidden = config.get("hidden_size", 0)
        vocab = config.get("vocab_size", 0)
        if layers and hidden:
            # ~12*hidden^2 parameters per transformer layer (4h^2 attention
            # + 8h^2 MLP) plus the embedding table: a rough estimate only.
            params = vocab * hidden + layers * 12 * hidden * hidden
            return params * PARAMS_BYTES / 1e9
    return None


def scan_models():
    """Every subfolder of ~/.hivemind/models that looks like a model."""
    models = []
    if MODELS_DIR.exists():
        for entry in sorted(MODELS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            has_config = (entry / "config.json").exists()
            has_weights = any(entry.glob("*.safetensors"))
            if has_config or has_weights:
                models.append(entry)
    return models


def latest_measured_gbs(profile: dict):
    """Most recent measured load bandwidth from the profile, if any."""
    measured = profile.get("measured_bw") or {}
    best = None
    best_at = ""
    for entry in measured.values():
        if isinstance(entry, dict) and entry.get("gbs") and entry.get("measured_at", "") >= best_at:
            best = entry["gbs"]
            best_at = entry["measured_at"]
    return best


def main() -> None:
    st.set_page_config(page_title="HiveMind", page_icon=":brain:")
    st.title("HiveMind")
    st.caption("Phase 1 - device profile and model predictions")

    profile = get_profile()
    probe_mib = profile.get("disk_probe_mib", LEGACY_PROBE_MIB)
    sustained = latest_measured_gbs(profile)

    st.subheader("Your device")
    disk_line = (
        f"- **Disk speed:** {profile['disk_gbs']:.2f} GB/s "
        f"({probe_mib} MiB probe)"
    )
    if sustained is not None:
        disk_line += f" | sustained: {sustained:.2f} GB/s (measured)"
    st.markdown("\n".join([
        f"- **RAM available:** {profile['ram_gb']:.1f} GB",
        disk_line,
        f"- **CPU cores:** {profile['cores']}",
        f"- **OS:** {profile['os']}",
    ]))
    st.caption(f"Measured at {profile['measured_at']} (cached for 7 days)")

    st.subheader("Local models")
    models = scan_models()
    if not models:
        st.info("No models yet. See INSTALL.md to download one.")
    measured_bw = profile.get("measured_bw") or {}
    for model_dir in models:
        size_gb = estimate_model_size_gb(model_dir)
        if size_gb is None:
            st.warning(
                f"{model_dir.name}: found, but its size could not be "
                "estimated (no safetensors, no usable config.json)"
            )
            continue
        entry = measured_bw.get(model_dir.name)
        measured_gbs = entry.get("gbs") if isinstance(entry, dict) else None
        low, high, source = predict_tps_range(
            size_gb, profile["disk_gbs"], measured_gbs=measured_gbs
        )
        risk = predict_oom_risk(size_gb, profile["ram_gb"])
        est_low = estimate_50_words_seconds(high)
        est_high = estimate_50_words_seconds(low)
        if measured_gbs is not None:
            headline = f"measured: **{low:.1f}-{high:.1f} tokens/sec**"
        else:
            headline = f"estimate: **{low:.1f}-{high:.1f} tokens/sec** | measured: -"
        st.markdown(
            f"**{model_dir.name}** - {size_gb:.1f} GB, {headline}, "
            f"~50 words in **{est_low:.1f}-{est_high:.1f} s**"
        )
        st.caption(f"Source: {source}")
        if risk == "warning":
            st.warning(
                "Heads up: this model is close to your available RAM - "
                "it may run slowly."
            )
        elif risk == "high":
            st.error(
                "High OOM risk: this model is as big as (or bigger than) "
                "your available RAM. You can still try - nothing is blocked."
            )

    st.divider()
    st.button("Start chat", disabled=True, help="Available in Phase 2")


if __name__ == "__main__":
    main()
