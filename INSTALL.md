# HiveMind - Installation

## What you need (the honest list)

HiveMind.exe is a native Rust window - small and fast - but
the engine behind it is Python. Python is NOT embedded in
the exe (embedding would bloat it and chain it to one
interpreter build); the exe looks for a Python on your PATH.
The bridge also accepts the py launcher.

- Windows 10/11, 64-bit.
- Python 3.9 - 3.11, 64-bit (the versions the engine is
  CI-tested on). On python.org's installer, tick
  "Add python.exe to PATH".
- ~2 GB free disk for the core bundle (torch, transformers
  and friends) - downloaded once per machine, not per
  release, and SHA-256-verified before use.
- One Safetensors model in ~/.hivemind/models to get real
  generation: one folder per model, config.json and a
  single weights file inside (sharded checkpoints are not
  supported yet).

No Python at all? The window still opens and runs in a
clearly-tagged offline mock mode - it never fakes engine
output as real.

## Install (the setup way)

1. Download HiveMind-Setup.exe from
   GitHub -> Releases (the latest one).
2. Double-click. It installs into
   %LOCALAPPDATA%/Programs/HiveMind - no admin rights.
3. Windows may warn about an unknown publisher (unsigned
   exe): "More info" -> "Run anyway".
4. First run: the app downloads the heavy core once and
   verifies its SHA-256 before using it.

## Install (the standalone way)

1. Download HiveMind-UI-windows-x64.zip and unzip it
   anywhere - keep the files together: the exe, the bridge
   and the engine sources live side by side.
2. Download HiveMind-Core-windows-x64.zip and unzip it as
   a "core" folder BESIDE HiveMind.exe.
3. Double-click HiveMind.exe.

## How the window finds the engine

The exe looks for hivemind_bridge.py beside itself and
starts it with your Python. The bridge binds the first free
port starting at 7860, publishes it to port.txt in the app
folder (pid + port), and the window talks to it over
localhost HTTP - nothing leaves your machine. A stale
port.txt from a dead run is ignored: the pid inside it
tells the UI it is not the bridge it just started.

## Troubleshooting

- Window says "offline: no python - mock mode": Python is
  not on PATH for your user. Reinstall Python with the PATH
  box ticked, or run "python --version" in a terminal to
  see what the machine actually finds.
- Window says "offline: bridge did not answer": run
  "python hivemind_bridge.py" in the app folder and read
  the console - the bridge prints its port or the real
  error. Close it with Ctrl+C when done.
- Replies tagged [mock:...]: no engine ran - either no
  model in ~/.hivemind/models, or the heavy core is
  missing. Both states are normal, both are said out loud.

## Developer setup

- The UI: Rust - "cargo build --release" inside
  hivemind-ui/.
- The bridge: any Python 3.9+ on PATH; it is stdlib-only.
  torch/transformers are imported lazily, only when a real
  model is requested (the core folder satisfies them).
- The engine: see README.md (pip install -e . for the
  test suite).
