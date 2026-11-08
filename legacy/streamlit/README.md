# The retired Streamlit UI (archived in wave #49)

These files powered the browser-based Streamlit UI from the
early waves: app.py (the full UI), launcher.py (the frozen
shim), launcher_app.py (the launcher logic), build.spec
(the PyInstaller recipe), and the run.bat / run.sh dev
launchers.

Wave #49 retired the whole stack: HiveMind.exe is now a
native Rust/egui desktop window (hivemind-ui/) that talks to
the Python engine through hivemind_bridge.py - no browser,
no HTML, no CSS, no JS.

The files are kept here for archaeology; nothing in the
living tree imports them. Git history carries the full
context of every line.
