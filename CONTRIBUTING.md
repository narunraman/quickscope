# Contributing to QuickScope

Thanks for your interest in contributing! This document covers the basics for getting set up and submitting changes.

## Setup

```bash
git clone https://github.com/narunraman/quickscope.git
cd quickscope
make sync
```

This installs all dependencies (including dev tools) in an isolated virtual environment via [uv](https://docs.astral.sh/uv/).

## Development workflow

### Code style

We use [Black](https://black.readthedocs.io/) for formatting and [Pyright](https://github.com/microsoft/pyright) for type checking:

```bash
uv run black quickscope/
uv run pyright quickscope/
```

Use `qscope --help` to inspect the installed CLI while developing. The longer
`quickscope` command is also installed as an alias.

### Branch conventions

- Create a feature branch from `main`: `git checkout -b your-feature`
- Keep commits focused—one logical change per commit.
- Open a pull request against `main` when ready.

## Adding a new scenario

QuickScope is designed to be extended with new benchmarks. To add one, you need three things:

1. **Generator** — a class that maps a template identifier to a concrete question instance.
2. **Search space** — a YAML or Python file defining the configuration parameters.
3. **Utility function** (optional) — a custom scoring function if the defaults don't fit.

See [`quickscope/simulation/README.md`](quickscope/simulation/README.md) for the full guide.

## Adding a new utility function

Register a utility with the `@register_utility` decorator:

```python
from quickscope.dataflow.utility_registry import register_utility

@register_utility("my-utility", scenario="my_scenario", description="My custom utility")
def my_utility(metrics: dict) -> float:
    return 1.0 - metrics["accuracy"]
```

Place this in the relevant scenario's module or in `resources/search_spaces/<scenario>/utility.py`.

## Reporting issues

Please open an issue on GitHub with:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Relevant logs or error messages.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
