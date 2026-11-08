# Release channel

Releases are immutable. A new release is published ONLY when
the version line below changes on main; any other push
publishes nothing. Reusing a version number fails the
publishing job on purpose, and old releases are never
touched.

Distribution (wave #47, the onedir architecture): every
release carries five assets. HiveMind-Setup.exe installs the
small app into %LOCALAPPDATA%/Programs/HiveMind with no admin
rights; the heavy core (HiveMind-Core-windows-x64.zip - torch,
transformers and friends) is downloaded and SHA-256-verified
once per machine, not per release. HiveMind-UI-windows-x64.zip
is the raw app directory, and it includes HiveMind-Debug.exe,
the console build that prints the real traceback.
HiveMind-UI-patch.zip holds only the files that changed since
the previous release, and update_manifest.json lists the
SHA-256 of every file; installed apps fetch it, verify every
hash, and refuse to run anything they could not verify.

version: v0.1.9
