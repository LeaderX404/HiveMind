# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: build the standalone HiveMind.exe
# (issues #41 and #42). One file, windowed - no console - and
# everything inside: the Python interpreter, streamlit, torch,
# transformers, huggingface_hub, pystray. The user installs
# nothing.
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, os.path.abspath("."))

datas = []
binaries = []
hiddenimports = []

# Runtime packages with dynamic imports: collected whole so the
# exe never says "module not found" on a user machine.
for package in ("streamlit", "transformers", "huggingface_hub",
                "tokenizers", "safetensors", "sentencepiece",
                "pystray", "PIL"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# The whole local package - paged engines, UI, downloads.
hiddenimports += collect_submodules("hivemind")

# app.py travels as a data file; the launcher hands it to
# Streamlit at runtime.
datas.append(("app.py", "."))

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HiveMind",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
