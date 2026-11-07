# Release channel

The current public release of HiveMind. Pushing a change to
this file to main publishes it: the Build HiveMind.exe
workflow attaches the freshly built HiveMind.exe to the
version below (publish-release job in build.yml). The classic
tag path (v*) still works and adds generated release notes.

version: v0.1.0
