# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: build the standalone HiveMind.exe
# (issues #41, #42, #44). One file, everything inside: the
# Python interpreter, streamlit, torch, transformers,
# huggingface_hub, pystray. The user installs nothing.
#
# Two targets share one Analysis:
#   HiveMind        - windowed (console=False): the public face
#   HiveMind-Debug  - console=True: a black window that prints
#                     the real traceback when the windowed exe
#                     dies silently (issue #44). Debug is an
#                     Actions artifact, never a release file.
import os
import sys

from PyInstaller.utils.hooks import (collect_all, collect_data_files,
                                     collect_submodules)

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

# Issue #44 (silent crash): belt and suspenders. collect_all
# should cover these, but the windowed exe died on a real
# machine, so the two heaviest packages get pinned explicitly:
# streamlit's static/ folder and transformers' metadata must
# ship, or the server collapses the moment it starts serving.
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("transformers")
datas += collect_data_files("streamlit")
datas += collect_data_files("transformers")

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

exe_debug = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HiveMind-Debug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
