# HiveMind native UI (wave #49)

A real desktop window drawn by egui - no browser, no HTML,
no CSS, no JS. The exe talks to the Python engine through
the local bridge (hivemind_bridge.py at the repo root) over
localhost HTTP; without python it still runs in a clearly
tagged offline mock mode.

## Build

    cargo build --release

The binary lands in target/release/HiveMind.exe. The CI
build (enforced in .github/workflows/build.yml):

- compiles the exe,
- stages it with the bridge and the engine sources as plain
  data (the wave #47 code-as-data pattern, kept),
- enforces the 20 MB size gate,
- runs the headless selftest: spawn the bridge, list models,
  one chat round-trip (mock on CI - no torch there),
- normalizes timestamps and bundles the deterministic zip.

## The bridge contract (hivemind_bridge.py)

- GET /api/models -> {"models": ["name", ...]}
- POST /api/chat {"prompt": "...", "model": "name"}
  -> {"text": "...", "mode": "engine" | "mock"}

Mock replies always say they are mock. The real-engine path
mirrors the retired Streamlit app: scan ~/.hivemind/models,
build the paged engine, generate with temperature=0.

## What is deliberately NOT here

No tokio, no axum, no gRPC client: one blocking worker thread
per request keeps the exe small, and the transport stays
swappable for the day the core becomes Rust (tonic then).

## Selftest

    HiveMind.exe --selftest

Headless: mock generation must work, and when a python is
on PATH the full HTTP path must answer. CI runs this on
every build.
