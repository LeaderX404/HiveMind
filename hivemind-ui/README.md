# HiveMind native UI (waves #49-#51)

A real desktop window drawn by egui - no browser, no HTML,
no CSS, no JS. The exe talks to the Python engine through
the local bridge (hivemind_bridge.py at the repo root) over
localhost HTTP; without python it still runs in a clearly
tagged offline mock mode.

## Build

    powershell -File assets/fetch-font.ps1   # once, wave #51
    cargo build --release

The Arabic font is NOT committed: it is fetched from a
commit-pinned URL (the bytes cannot drift) into
assets/NotoNaskhArabic-Regular.ttf and embedded into the exe
by include_bytes! (src/main.rs, install_fonts). CI runs the
same fetch before cargo. Without the font the build fails -
that is on purpose: silent boxes are worse than a loud
error.

The binary lands in target/release/HiveMind.exe. The CI
build (enforced in .github/workflows/build.yml):

- fetches the pinned Arabic font,
- compiles the exe,
- stages it with the bridge and the engine sources as plain
  data (the wave #47 code-as-data pattern, kept),
- enforces the 20 MB size gate,
- runs the headless selftest: spawn the bridge, read its
  published port, list models, one chat round-trip (mock on
  CI - no torch there),
- normalizes timestamps and bundles the deterministic zip.

## The bridge contract (hivemind_bridge.py)

- GET /api/models -> {"models": ["name", ...]}
- POST /api/chat {"prompt": "...", "model": "name"}
  -> {"text": "...", "mode": "engine" | "mock"}
- POST /api/download {"model": "Qwen2.5-0.5B-Instruct"}
  -> 202 {"started": true} | 400 unknown | 409 busy
- GET /api/download/status
  -> {"active": bool, "done": bool, "model": str, "error": str|null}

Mock replies always say they are mock. The real-engine path
mirrors the legacy chat app (legacy/streamlit/): scan
~/.hivemind/models, build the paged engine, generate with
temperature=0. The model catalog (wave #51) lives in the
bridge (MODEL_CATALOG) and is mirrored in
hivemind-ui/src/engine.rs; downloads land in
~/.hivemind/models and appear in the left panel after the
watcher refreshes the list.

## The Arabic font (wave #51)

Noto Naskh Arabic (Regular, hinted), pinned to commit
cb61ee55 of notofonts/notofonts.github.io. It is appended to
both font families, so Latin text keeps the default fonts
and Arabic falls back to Noto - letters, not boxes. The
welcome message carries an Arabic line on purpose: boxes
there mean the font did not load.

## The port protocol (wave #50)

No fixed port is trusted. The bridge binds the first free
port at or above 7860 and writes "pid=<pid>\nport=<port>"
to port.txt beside itself (the app folder). The UI reads
that file - a pid it has not seen before means a fresh
bridge - and talks to that port. A stale port.txt from a
dead run is ignored (its pid never matches the new bridge).
Known limit: when two windows start together, the second
reuses the first bridge if it answers on the default port;
the port file always points at the newest bridge.

## What is deliberately NOT here

No tokio, no axum, no gRPC client: one blocking worker
thread per request keeps the exe small, and the transport
stays swappable for the day the core becomes Rust (tonic
then). The bridge itself stays forever as the
backward-compatibility surface for older tools even after
that port.

## Selftest

    HiveMind.exe --selftest

Headless: mock generation must work, and when a python is
on PATH the full port-file + HTTP path must answer. CI runs
this on every build and scrubs port.txt afterwards so the
deterministic zip never sees a runtime file.
