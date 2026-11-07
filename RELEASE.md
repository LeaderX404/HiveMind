# Release channel

Releases are immutable. A new release is published ONLY when
the version line below changes on main; pushing anything else
publishes nothing. Reusing a version number fails the build on
purpose (the workflow refuses to overwrite an existing release),
and old releases are never touched - they stay on the Releases
page exactly as they shipped.

Each release carries two files: HiveMind.exe (standalone, just
double-click) and HiveMind-Setup.exe (installer: per-user
folder, desktop and Start menu shortcuts, clean uninstaller).
The classic tag path (v*) still works and adds generated
release notes.

version: v0.1.1
