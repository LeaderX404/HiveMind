# Contributing

## Development setup

```bash
git clone https://github.com/LeaderX404/HiveMind.git
cd HiveMind
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-cov pytest-xdist
```

## Running the tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (skip slow ones)
pytest tests/integration/ -m "not slow"
```

## Linting

```bash
flake8 hivemind/ --max-line-length=100 --ignore=E203,E501,W503
black hivemind/ tests/ --line-length=100
isort hivemind/ tests/ --profile black --line-length=100
```

## Building the docs

```bash
pip install mkdocs mkdocs-material
mkdocs build --strict
```

## Pull requests

1. Branch from `main` and keep changes focused — one concern per PR.
2. Make sure CI is green (tests are blocking; lint is advisory for now).
3. Update docs and tests together with code changes.
