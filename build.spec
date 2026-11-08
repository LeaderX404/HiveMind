# HiveMind build spec (issues #41, #42, #44, wave #47).
#
# ONEDIR architecture (wave #47): the output is a directory
# (dist/HiveMind-UI/) holding two tiny executables -
# HiveMind.exe (windowed) and HiveMind-Debug.exe (console,
# issue #44) - plus an _internal/ tree. The heavy libraries
# (torch, transformers, ...) are NOT frozen: they live in
# core\ beside the app dir, downloaded once at first run and
# reused by every update.
#
# The application code itself (hivemind/, app.py,
# launcher_app.py) ships as PLAIN DATA inside _internal so a
# normal release is a kilobyte-scale delta patch, not a full
# exe rebuild. Only launcher.py - the 20-line shim - is
# compiled into the executable.
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Heavy imports are resolved at runtime from core\, never
# frozen: excluding them keeps the UI bundle small and the
# executable stable across normal releases.
_core_packages = ["torch", "transformers", "huggingface_hub",
                   "tokenizers", "safetensors", "sentencepiece",
                   "accelerate", "bitsandbytes", "numpy"]
excludes = []
for _pkg in _core_packages:
    excludes.append(_pkg)
    excludes.append(_pkg + ".*")

# Streamlit has runtime-discovered pieces the analyzer cannot
# see: collect everything it needs (static files, metadata).
st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")
tray_hiddenimports = collect_submodules("pystray") + \
    collect_submodules("PIL")

block_cipher = None
a = Analysis(["launcher.py"],
             pathex=[],
             binaries=st_binaries,
             datas=st_datas,
             hiddenimports=st_hiddenimports + tray_hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=excludes,
             cipher=block_cipher)

# The application code travels as data (wave #47): delta
# patches replace these files without touching the exe.
a.datas.append(("app.py", "app.py", "DATA"))
a.datas.append(("launcher_app.py", "launcher_app.py", "DATA"))
a.datas.append(("hivemind", "hivemind", "DATA"))
# ...and stays out of the frozen analysis.
a.excludes += ["hivemind", "hivemind.*", "launcher_app"]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name="HiveMind",
          console=False,
          icon=None,
          debug=False,
          strip=False,
          upx=False)

# The console twin for crash triage (issue #44): same bundle,
# black window, full traceback on screen.
exe_debug = EXE(pyz,
                a.scripts,
                [],
                exclude_binaries=True,
                name="HiveMind-Debug",
                console=True,
                icon=None,
                debug=False,
                strip=False,
                upx=False)

# One _internal shared by both executables; COLLECT merges the
# analysis TOCs and the two EXE objects into dist/HiveMind-UI/.
coll = COLLECT(exe,
               exe_debug,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name="HiveMind-UI")
