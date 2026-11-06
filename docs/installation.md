# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Linux / macOS (Windows via WSL recommended)

## Install from source

```bash
git clone https://github.com/LeaderX404/HiveMind.git
cd HiveMind
pip install -r requirements.txt
pip install -e .
```

## Optional extras

- `ray[default]` — swarm coordination for distributed inference (installed with requirements).
- `psutil` — richer memory monitoring; HiveMind falls back gracefully if it is missing.

## Verify the installation

```bash
python examples/test_phase3.py
```

See the [User Guide](user_guide.md) for the first steps.
