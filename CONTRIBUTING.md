# Contributing to HiveMind by VALCRYPTIA

Thank you for your interest in contributing to **HiveMind by VALCRYPTIA**! We welcome contributions from everyone.

## 📋 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🎯 How Can I Contribute?

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/LeaderX404/HiveMind/issues) with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Features

We welcome feature suggestions! Please [open an issue](https://github.com/LeaderX404/HiveMind/issues) with:
- A clear description of the feature
- The problem it solves
- Any relevant examples or references

### Pull Requests

We welcome pull requests! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow the coding style** (PEP 8, type hints, docstrings)
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Keep commits atomic** (one logical change per commit)
6. **Write clear commit messages**

## 🚀 Development Setup

```bash
# Clone the repository
git clone https://github.com/LeaderX404/HiveMind.git
cd HiveMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

## 📁 Project Structure

```
hivemind/
├── hivemind/
│   ├── __init__.py          # Main package initialization
│   ├── cli.py               # Command-line interface
│   ├── core/
│   │   ├── __init__.py      # Core module exports
│   │   ├── direct_loader.py # Direct file model loader
│   │   ├── file_pager.py    # File-backed pager
│   │   ├── transformer_engine.py # Paged transformer engine
│   │   ├── model_loader.py  # Original model loader
│   │   ├── sharder.py       # Model sharder
│   │   ├── inference.py     # Inference engines
│   │   ├── quantizer.py     # Quantization utilities
│   │   └── pager.py         # Original pager
│   └── utils/
│       ├── __init__.py      # Utils module exports
│       ├── logger.py        # Logging utilities
│       ├── helpers.py       # Helper functions
│       ├── perf_monitor.py  # Performance monitoring
│       └── quant_utils.py   # Quantization utilities
├── examples/
│   ├── test_out_of_core.py  # Out-of-core test script
│   └── test_quantization.py # Quantization test script
├── tests/
│   └── unit/
│       ├── test_*.py       # Unit tests
├── docs/
│   └── *.md                # Documentation
├── README.md
├── LICENSE
├── setup.py
└── requirements.txt
```

## 💡 Coding Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use type hints for all functions
- Use descriptive variable and function names
- Keep lines under 100 characters

### Documentation
- Add docstrings to all modules, classes, and functions
- Use Google-style docstrings
- Keep documentation up-to-date

### Testing
- Add unit tests for new functionality
- Ensure tests pass before submitting PRs
- Run tests with: `pytest tests/`

### Commits
- Use clear, descriptive commit messages
- Follow [Conventional Commits](https://www.conventionalcommits.org/)
- Examples:
  - `feat: add Q4 quantization support`
  - `fix: correct RoPE implementation`
  - `docs: update README with new features`

## ✅ Pull Request Template

```markdown
## Description

[Clear description of changes]

## Related Issues

[Link to related issues]

## Changes Made

- [ ] Added new feature
- [ ] Fixed bug
- [ ] Updated documentation
- [ ] Added tests
- [ ] Refactored code

## Testing

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist

- [ ] Code follows PEP 8 guidelines
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] No breaking changes (or documented)
```

## 🛡️ Serialization Guard & Safe Loading Policy

The **Serialization Guard** CI job (blocking) rejects any use of
`torch.load`, `pickle.load`, `pickle.loads`, `pickle.Unpickler` or
bare `yaml.load` anywhere under `hivemind/`, `tests/` and in
`benchmark.py`.

**What the guard is - and is not.** The guard scans source text: it
stops *accidents*, not sabotage. Deliberately obfuscated calls such as
`getattr(pickle, "load")` or
`importlib.import_module("pickle").load` pass under it. Code review
remains the backstop against malice; the guard is the backstop against
human error and creeping debt.

**The approved safe paths** (use these instead):

- Tensors over the network or on disk: `hivemind.net.serialization`
  (`serialize_tensor` / `deserialize_tensor`, the HMT1 container).
  Disk cache pages use the same format (HMP2, `hivemind/core/pager.py`).
- Model files: `safetensors` via `hivemind/core/direct_loader.py`,
  optionally with `expected_sha256` integrity verification.
- Configuration: `yaml.safe_load(...)`.

**Exception policy - exceptions are priced, not free.** A call site
that genuinely needs a blocked API must carry the full inline marker:

    serialization-guard: allowed #issue-NNN expires=YYYY-MM-DD

The marker alone is not enough. The guard enforces, on every push:

1. **Issue binding** - issue `NNN` must exist, be open, carry the
   `security-exception` label and have an assignee. An exception is
   a public, owned decision - never an anonymous comment.
2. **TTL** - `expires` must be in the future and at most 90 days out.
   No eternal exceptions. A weekly scheduled workflow opens a
   reminder issue before an exception expires.
3. **Ratchet (net zero)** - the total number of markers may never
   exceed the count recorded in `serialization_exceptions.baseline`.
   Adding a new exception requires retiring an old one in the same PR.

## ⚖️ Gate Law (process rule, issue #29)

Every gate - runtime or CI - requires a named decision and a
public issue. No reserve gates, no "just in case" gates, no gate
without an issue: a gate added because "the ladder might be
incomplete" is anxiety wearing a uniform, and anxiety has no
stopping rule. The ladder produces ORDER, not a number, so it is
judged by a structural linter (tests/unit/test_ladder_lint.py)
that fails the build - never by a fifth gate, which would open
an infinite regress. Gates are added by decision, never by
anxiety: a new named decision ("we do not accept X") opens an
issue and earns a gate; no new named decision, no new gate. The
stopping rule is this law itself, not a constant.

## 📏 README Honesty Rule (issue #32)

Any README sentence that carries a number, a comparison, or an
absolute word ("never", "always", "revolutionizing") requires
either (a) a published measurement behind it, or (b) an explicit
"not yet measured" label. No exceptions. A README passes this
rule by deleting its unmeasured numbers or by labeling them -
not by keeping them naked.

## 📏 README Parity Rule (issue #33)

Every model listed in the README Parity table must be loadable by
the code in this repository. The Parity table is a record of
results, not promises: a model whose layout is not implemented
lives under "Planned architectures (layout not implemented)" with
an open issue, and enters the table only after the code can load
it and a measured parity number is published.

## 🎓 Getting Help

If you need help:
- Check the [documentation](docs/)
- Open an [issue](https://github.com/LeaderX404/HiveMind/issues)
- Join our [Discord community](https://discord.gg/valcryptia) (coming soon)

## 🙏 Recognition

All contributors will be recognized in the CONTRIBUTORS.md file.

---

**Thank you for contributing to HiveMind by VALCRYPTIA!** 🚀

*Built with ❤️ by VALCRYPTIA*
