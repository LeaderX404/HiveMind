# Release channel

Releases are immutable. A new release is published ONLY when
the version line below changes on main; any other push
publishes nothing. Reusing a version number fails the
publishing job on purpose, and old releases are never
touched.

Distribution (wave #49, the native architecture): every
release carries five assets. HiveMind-Setup.exe installs the
small app into %LOCALAPPDATA%/Programs/HiveMind with no admin
rights; HiveMind.exe is a native Rust/egui desktop window -
no browser, no HTML, no CSS, no JS - that talks to the Python
engine through hivemind_bridge.py shipped beside it. Python
is NOT embedded: the window needs a 64-bit Python 3.9 - 3.11
on PATH (INSTALL.md has the honest list), and without one it
runs in a clearly-tagged offline mock mode. No fixed port is
trusted: the bridge binds the first free port, publishes
pid+port to port.txt in the app folder, and the window
follows that file. The heavy core
(HiveMind-Core-windows-x64.zip - torch, transformers and
friends) is downloaded and SHA-256-verified once per machine,
not per release. HiveMind-UI-windows-x64.zip is the raw app
directory. HiveMind-UI-patch.zip holds only the files that
changed since the previous release, and update_manifest.json
lists the SHA-256 of every file; installed apps fetch it,
verify every hash, and refuse to run anything they could
not verify.

version: v0.2.1
