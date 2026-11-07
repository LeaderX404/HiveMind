"""HiveMind UI - Phase 1 + Phase 2 (issues #31 and #40).

Phase 1: the device profile and every local model found under
~/.hivemind/models/ with its predicted TPS RANGE and source -
all BEFORE anything is loaded. Every number carries its
source; nothing poses as a measurement that was not measured.

Phase 2 (issue #40, wave #38 - the download wave): the
Download tab. Category 1 models download in one click
(resumable, hash-verified against what the hub publishes),
each with its predicted TPS range and a level badge; a custom
repo id downloads the same way. The first real load of a model
streams every tensor once - the honest out-of-core "load" -
and journals the measured bandwidth through
record_actual_load, so the next run shows a measured band
instead of an estimate. Start Chat opens a primitive chat on
the paged engine: greedy decoding, no chat template yet -
Stage 3 expands it.

The caption at the bottom of the Download tab keeps the wave
#37 honesty in view: parity verification uses k=3, weaker
than the declared k*=4.2 commitment.
"""
import json
import threading
import time
from pathlib import Path

import streamlit as st

from hivemind.ui.catalog import CATEGORY_1, CATEGORY_2_5
from hivemind.ui.downloader import (
    MODELS_DIR,
    download_model,
    model_local_dir,
)
from hivemind.ui.hardware_detect import get_profile, record_actual_load
from hivemind.ui.levels import LEVEL_COLORS, classify_level
from hivemind.ui.tps_predictor import (
    estimate_50_words_seconds,
    predict_oom_risk,
    predict_tps_range,
)
from hivemind.ui.update_checker import check_for_update

PARAMS_BYTES = 2  # rough estimate: fp16 weights, 2 bytes per parameter
LEGACY_PROBE_MIB = 100  # shown when a cached profile predates probe-size journaling
CHAT_MAX_NEW_TOKENS = 64

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


def _level_badge(level: str) -> str:
    """A colored level badge (A-E) as an HTML span."""
    return (
        '<span style="background-color:%s;color:#ffffff;'
        'border-radius:6px;padding:2px 8px;font-weight:600;'
        'font-size:0.85em">Level %s</span>' % (LEVEL_COLORS[level], level)
    )


def _render_prediction(name: str, size_gb: float, profile: dict) -> None:
    """Name, size, predicted TPS range and level badge - all
    before anything downloads, all carrying their source."""
    level, tps_character, level_desc = classify_level(
        size_gb, profile["ram_gb"])
    low, high, source = predict_tps_range(size_gb, profile["disk_gbs"])
    st.markdown("**%s** - %.1f GB (est.)" % (name, size_gb))
    st.markdown(
        "%s predicted: **%.1f-%.1f tokens/sec** - %s"
        % (_level_badge(level), low, high, tps_character),
        unsafe_allow_html=True,
    )
    st.caption("%s | Source: %s" % (level_desc, source))


def _start_download(repo_id: str, size_gb: float, name: str) -> None:
    """Arm one download in session state (one at a time)."""
    st.session_state.pop("download_thread", None)
    st.session_state["download"] = {
        "repo_id": repo_id,
        "name": name,
        "downloaded_mb": 0.0,
        "total_mb": 0.0,
        "speed_mbps": 0.0,
        "done": False,
        "error": None,
    }


def _download_worker(dl: dict) -> None:
    """Thread body: resumable download + hash verification."""
    try:
        target = download_model(
            dl["repo_id"],
            on_progress=lambda d, t, s: dl.update(
                downloaded_mb=d, total_mb=t, speed_mbps=s),
        )
        dl["local_dir"] = str(target)
    except Exception as exc:  # any failure must be visible, never silent
        dl["error"] = "%s" % exc
    finally:
        dl["done"] = True


def _render_active_download() -> None:
    """Live progress for the armed download; Start Chat when done."""
    dl = st.session_state.get("download")
    if dl is None:
        return
    if dl["done"] and st.session_state.get("download_thread") is not None:
        st.session_state.pop("download_thread", None)
    if not dl["done"] and st.session_state.get("download_thread") is None:
        st.session_state["download_thread"] = threading.Thread(
            target=_download_worker, args=(dl,), daemon=True)
        st.session_state["download_thread"].start()
    if not dl["done"]:
        bar = st.progress(0.0)
        info = st.empty()
        thread = st.session_state.get("download_thread")
        while thread is not None and thread.is_alive():
            total = dl.get("total_mb") or 0.0
            done = dl.get("downloaded_mb", 0.0)
            bar.progress(min(done / total, 1.0) if total > 0 else 0.0)
            info.text("Downloading %s: %.0f / %.0f MB - %.1f MB/s"
                      % (dl["name"], done, total,
                         dl.get("speed_mbps", 0.0)))
            time.sleep(0.25)
        bar.progress(1.0)
    if dl.get("error"):
        st.error("Download failed: %s" % dl["error"])
        if st.button("Dismiss"):
            del st.session_state["download"]
            st.session_state.pop("download_thread", None)
            st.rerun()
        return
    st.success("%s downloaded and hash-verified." % dl["name"])
    if st.button("Start Chat"):
        st.session_state["chat_model"] = model_local_dir(
            dl["repo_id"]).name
        st.rerun()


def _render_download(profile: dict) -> None:
    """The Download tab: Category 1, placeholders, custom model,
    and the k=3 honesty line."""
    st.subheader("Category 1 - available now")
    for entry in CATEGORY_1:
        _render_prediction(entry["name"], entry["size_gb"], profile)
        local = model_local_dir(entry["repo_id"])
        if (local / "config.json").exists():
            st.caption("Already downloaded.")
            if st.button("Start Chat", key="start-%s" % entry["name"]):
                st.session_state["chat_model"] = local.name
                st.rerun()
        elif st.button("Download", key="dl-%s" % entry["name"]):
            _start_download(entry["repo_id"], entry["size_gb"],
                            entry["name"])
            st.rerun()
        st.divider()
    _render_active_download()

    st.subheader("Custom model")
    custom_repo = st.text_input(
        "Hugging Face repo id (Safetensors, llama-family)")
    custom_size = st.number_input("Size estimate (GB)",
                                  min_value=0.1, value=1.0,
                                  step=0.1)
    if custom_repo.strip():
        _render_prediction("Custom model", float(custom_size), profile)
    if st.button("Download custom model"):
        repo = custom_repo.strip()
        if repo:
            _start_download(repo, float(custom_size), repo)
            st.rerun()
        else:
            st.warning("Enter a repo id first.")

    st.subheader("Categories 2-5")
    for cat in CATEGORY_2_5:
        st.markdown("**%s** - %s" % (cat["name"], cat["status"]))
    st.divider()
    st.caption("Parity verification uses k=3, weaker than our "
               "declared k*=4.2 commitment. See README.")


def _render_device(profile: dict) -> None:
    """Phase 1 device profile."""
    probe_mib = profile.get("disk_probe_mib", LEGACY_PROBE_MIB)
    sustained = latest_measured_gbs(profile)
    disk_line = ("- **Disk speed:** %.2f GB/s (%d MiB probe)"
                 % (profile["disk_gbs"], probe_mib))
    if sustained is not None:
        disk_line += " | sustained: %.2f GB/s (measured)" % sustained
    st.markdown("\n".join([
        "- **RAM available:** %.1f GB" % profile["ram_gb"],
        disk_line,
        "- **CPU cores:** %d" % profile["cores"],
        "- **OS:** %s" % profile["os"],
    ]))
    st.caption("Measured at %s (cached for 7 days)"
               % profile["measured_at"])


def _render_local_models(profile: dict) -> None:
    """Phase 1 local models, plus the Phase 2 Start chat button."""
    models = scan_models()
    if not models:
        st.info("No models yet - download one in the Download tab.")
        return
    measured_bw = profile.get("measured_bw") or {}
    for model_dir in models:
        size_gb = estimate_model_size_gb(model_dir)
        if size_gb is None:
            st.warning(
                "%s: found, but its size could not be estimated "
                "(no safetensors, no usable config.json)"
                % model_dir.name)
            continue
        entry = measured_bw.get(model_dir.name)
        measured_gbs = (entry.get("gbs")
                        if isinstance(entry, dict) else None)
        low, high, source = predict_tps_range(
            size_gb, profile["disk_gbs"], measured_gbs=measured_gbs)
        risk = predict_oom_risk(size_gb, profile["ram_gb"])
        est_low = estimate_50_words_seconds(high)
        est_high = estimate_50_words_seconds(low)
        if measured_gbs is not None:
            headline = "measured: **%.1f-%.1f tokens/sec**" % (low, high)
        else:
            headline = ("estimate: **%.1f-%.1f tokens/sec** | "
                        "measured: -" % (low, high))
        st.markdown("**%s** - %.1f GB, %s, ~50 words in "
                    "**%.1f-%.1f s**"
                    % (model_dir.name, size_gb, headline,
                       est_low, est_high))
        st.caption("Source: %s" % source)
        if risk == "warning":
            st.warning("Heads up: this model is close to your "
                       "available RAM - it may run slowly.")
        elif risk == "high":
            st.error("High OOM risk: this model is as big as (or "
                     "bigger than) your available RAM. You can "
                     "still try - nothing is blocked.")
        if st.button("Start chat", key="chat-%s" % model_dir.name):
            st.session_state["chat_model"] = model_dir.name
            st.rerun()
        st.divider()


def _load_engine(model_dir: Path):
    """Build the paged engine for one local model. Single-file
    Safetensors only - multi-shard arrives in Stage 3."""
    from transformers import AutoTokenizer
    from hivemind.core.file_pager import FileBackedPager
    from hivemind.core.transformer_engine import (
        PagedTransformerEngine,
        TransformerConfig,
    )

    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("no Safetensors weights in %s" % model_dir)
    if len(weights) > 1:
        raise RuntimeError(
            "multi-shard weights are not wired into the primitive "
            "chat yet - Stage 3 expands it (issue #40)")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    config = TransformerConfig.from_hf_config(
        str(model_dir / "config.json"))
    pager = FileBackedPager(str(weights[0]), device="cpu")
    engine = PagedTransformerEngine(pager, tokenizer, config,
                                    device="cpu")
    return engine, weights[0]


def _ensure_engine(model_name: str, model_dir: Path):
    """Load once per session; on the first ever load, stream
    every tensor once - the honest out-of-core "load" - and
    journal the measured bandwidth via record_actual_load."""
    engines = st.session_state.setdefault("engines", {})
    if model_name in engines:
        return engines[model_name]
    profile = get_profile()
    already = (profile.get("measured_bw") or {}).get(model_name)
    with st.spinner("Loading %s - the first load measures your "
                    "real bandwidth..." % model_name):
        engine, weights = _load_engine(model_dir)
        if already is None:
            start = time.monotonic()
            for tensor_name in engine.file_pager.loader.list_tensors():
                engine.file_pager.load_tensor(tensor_name,
                                              prefetch_next=False)
            load_seconds = time.monotonic() - start
            if load_seconds > 0.0:
                record_actual_load(model_name,
                                   weights.stat().st_size / 1e9,
                                   load_seconds)
    engines[model_name] = engine
    return engine


def _render_chat(profile: dict) -> None:
    """Primitive Phase 2 chat on the paged engine."""
    models = scan_models()
    if not models:
        st.info("No local models yet - download one in the "
                "Download tab.")
        return
    by_name = {m.name: m for m in models}
    names = sorted(by_name)
    preferred = st.session_state.get("chat_model")
    if preferred not in by_name:
        preferred = names[0]
    chosen = st.selectbox("Model", names, index=names.index(preferred))
    st.session_state["chat_model"] = chosen
    model_dir = by_name[chosen]
    try:
        engine = _ensure_engine(chosen, model_dir)
    except RuntimeError as exc:
        st.error("%s" % exc)
        return
    prompt = st.text_input("Your message")
    max_new = st.slider("Max new tokens", 16, 128,
                        CHAT_MAX_NEW_TOKENS)
    if st.button("Send") and prompt:
        with st.spinner("Generating - primitive mode, no "
                        "streaming yet..."):
            st.session_state["last_reply"] = engine.generate(
                prompt, max_new_tokens=max_new, temperature=0)
    if "last_reply" in st.session_state:
        st.markdown(st.session_state["last_reply"])
    st.caption("Primitive Phase 2 chat: llama-family weights, "
               "greedy decoding, no chat template - Stage 3 "
               "expands it.")


def main() -> None:
    st.set_page_config(page_title="HiveMind", page_icon=":brain:")
    st.title("HiveMind")
    st.caption("Phase 2 - device profile, downloads and a "
               "primitive chat")
    profile = get_profile()
    update = check_for_update()
    if update:
        st.info("New update available: **v%s** - [click here to "
                "download](%s)"
                % (update["version"], update["url"]))
    tab_device, tab_download, tab_models, tab_chat = st.tabs(
        ["Your device", "Download", "Local models", "Chat"])
    with tab_device:
        _render_device(profile)
    with tab_download:
        _render_download(profile)
    with tab_models:
        _render_local_models(profile)
    with tab_chat:
        _render_chat(profile)


if __name__ == "__main__":
    main()
