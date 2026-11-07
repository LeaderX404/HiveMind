# HiveMind - Installation

1. Download **HiveMind.exe** (GitHub -> Releases; or the
   **HiveMind-windows-x64** artifact under Actions -> Build
   HiveMind.exe).
2. Double-click it. (No need to install anything else.)

- Your browser opens at http://localhost:8501 - the app runs
  entirely on your machine, nothing leaves it.
- First launch may take a while: the app unpacks itself before
  the window opens.
- Windows may warn about an unknown publisher (unsigned exe):
  "More info" -> "Run anyway".
- Models download to your .hivemind/models folder on first use,
  from inside the app.
- To quit: close the browser tab and end "HiveMind" in Task
  Manager - there is no console window by design.
- Problems? The launcher writes the truth to
  .hivemind/launcher.log. Need help? Open an issue.

## Developer setup

Building from source still needs Python: install requirements
plus streamlit, then run run.bat (Windows) / run.sh (Linux,
Mac), or build the exe yourself with pyinstaller build.spec.
