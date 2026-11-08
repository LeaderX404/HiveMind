"""HiveMind frozen shim (issues #41, #42, wave #47).

This file is compiled INTO HiveMind.exe, so it stays tiny and
boring on purpose: every real change lives in launcher_app.py,
which ships as a plain data file inside the onedir bundle and
can be delta-patched in kilobytes (wave #47 architecture).
"""
import sys

if getattr(sys, "_MEIPASS", None) and sys._MEIPASS not in sys.path:
    sys.path.insert(0, sys._MEIPASS)

if __name__ == "__main__":
    from launcher_app import main
    sys.exit(main())
