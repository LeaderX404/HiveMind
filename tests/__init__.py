# HiveMind test suite package.
#
# Why this file exists (Phase 4 CI fix): pytest derives the import basedir
# by walking up from each test module while __init__.py files exist.
# Without this file, `tests/` itself became the basedir and was prepended
# to sys.path instead of the repository root, so `import hivemind` failed
# with ModuleNotFoundError in CI (pytest exit code 2). With this package
# marker, the repository root lands on sys.path.
