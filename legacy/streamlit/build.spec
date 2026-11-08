# -*- mode: python ; coding: utf-8 -*-
# HiveMind build spec (issues #41, #42, #44, wave #47).
#
# ONEDIR architecture (wave #47): dist/HiveMind-UI/ holds two
# small executables - HiveMind.exe (windowed) and
# HiveMind-Debug.exe (console, issue #44) - plus _internal/.
# The heavy libraries (torch, transformers, ...) are NOT
# frozen: they live in core\ beside the app dir, downloaded
# once at first run and reused by every update.
#
# The application code (hivemind/, app.py, launcher_app.py)
# ships as PLAIN DATA in _internal and is EXCLUDED from the
# frozen analysis, so a normal release is a kilobyte-scale
# delta patch. Only launcher.py - the 20-line shim - is
# compiled into the executables.
#
# PyInstaller 6 dialect: the legacy encryption and embedded
# archive spec arguments were removed upstream, and passing
# them crashes the spec evaluator.
#
# Deterministic patches (wave #48): the CI pins SOURCE_DATE_EPOCH
# and runs installer/normalize_build.py after pyinstaller, so two
# builds of identical sources produce byte-identical artifacts
# and a UI-only wave ships a UI-only patch (<= 20 MB gate). Keep
# this spec free of anything that varies per build: no dates, no
# environment snapshots, no random ordering - a single nondeter-
# ministic byte here re-enters every future patch.
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, os.path.abspath("."))

# Heavy imports resolve from core\ at runtime: excluding them
# keeps the UI bundle small and the exe stable across waves.
excludes = ["tkinter", "hivemind", "hivemind.*", "launcher_app"]
for _pkg in ("torch", "transformers", "huggingface_hub",
             "tokenizers", "safetensors", "sentencepiece",
             "accelerate", "bitsandbytes", "numpy"):
    excludes.append(_pkg)
    excludes.append(_pkg + ".*")

datas = []
binaries = []
hiddenimports = []

# Streamlit discovers pieces at runtime: collect everything it
# needs (static files, metadata) - the proven recipe.
st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")
datas += st_datas
binaries += st_binaries
hiddenimports += st_hiddenimports
hiddenimports += collect_submodules("streamlit")

# The tray icon: pystray plus PIL (icon drawn at runtime).
for _tray in ("pystray", "PIL"):
    _d, _b, _h = collect_all(_tray)
    datas += _d
    binaries += _b
    hiddenimports += _h

# The application code travels as data (wave #47): delta
# patches replace these files without touching the exe.
datas.append(("app.py", "."))
datas.append(("launcher_app.py", "."))
datas.append(("hivemind", "."))

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HiveMind",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

# The console twin for crash triage (issue #44): same bundle,
# black window, full traceback on screen.
exe_debug = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HiveMind-Debug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

# One _internal shared by both executables: COLLECT merges the
# analysis TOCs and the two EXE objects into dist/HiveMind-UI/.
coll = COLLECT(
    exe,
    exe_debug,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="HiveMind-UI",
)
