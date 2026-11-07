"""HiveMind UI - Phase 1 (issue #31, phase 1).

One Streamlit page: device profile, then every local model found
under ~/.hivemind/models/ with its predicted TPS, OOM risk and
50-word estimate - all BEFORE anything is loaded. No chat yet:
the "Start chat" button stays disabled until Phase 2. This page
displays, it does not judge - the engine joins in Phase 2.
"""
import json
from pathlib import Path

import streamlit as st

from hivemind.ui.hardware_detect import get_profile
from hivemind.ui.tps_predictor import (
    estimate_50_words_seconds,
    predict_oom_risk,
    predict_tps,
)

MODELS_DIR = Path.home() / ".hivemind" / "models"
PARAMS_BYTES = 2  # rough estimate: fp16 weights, 2 bytes per parameter


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


def main() -> None:
    st.set_page_config(page_title="HiveMind", page_icon=":brain:")
    st.title("HiveMind")
    st.caption("Phase 1 - device profile and model predictions")

    profile = get_profile()
    st.subheader("Your device")
    st.markdown("\n".join([
        f"- **RAM available:** {profile['ram_gb']:.1f} GB",
        f"- **Disk speed:** {profile['disk_gbs']:.2f} GB/s",
        f"- **CPU cores:** {profile['cores']}",
        f"- **OS:** {profile['os']}",
    ]))
    st.caption(f"Measured at {profile['measured_at']} (cached for 7 days)")

    st.subheader("Local models")
    models = scan_models()
    if not models:
        st.info("No models yet. See INSTALL.md to download one.")
    for model_dir in models:
        size_gb = estimate_model_size_gb(model_dir)
        if size_gb is None:
            st.warning(
                f"{model_dir.name}: found, but its size could not be "
                "estimated (no safetensors, no usable config.json)"
            )
            continue
        tps = predict_tps(size_gb, profile["disk_gbs"])
        risk = predict_oom_risk(size_gb, profile["ram_gb"])
        est = estimate_50_words_seconds(tps)
        st.markdown(
            f"**{model_dir.name}** - {size_gb:.1f} GB, predicted "
            f"**{tps:.1f} tokens/sec**, ~50 words in **{est:.1f} s**"
        )
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
