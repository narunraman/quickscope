# QuickScope

**Bayesian-optimal evaluation for dynamic LLM benchmarks.**

QuickScope quickly scopes out a model's weak spots within a benchmark's configuration space. It uses [COUP](https://arxiv.org/abs/2501.06693), a Bayesian optimization algorithm, to adaptively allocate evaluation budget toward the most informative regions—certifying which questions are *reliably* hard rather than surfacing one-off failures from noisy outcomes.

## Key ideas

- **Adaptive search.** Instead of evaluating every configuration uniformly, QuickScope concentrates samples where uncertainty is highest, finding hard questions with fewer model calls.
- **Certification.** Confidence bounds let you certify that a configuration is truly difficult, not just an artifact of small-sample noise.
- **Pluggable utilities.** Swap in different utility functions to target different notions of "hard": low accuracy, complexity-weighted error, etc.
- **Benchmark-agnostic.** Bring your own generator, search space, and scoring logic. QuickScope handles the optimization.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/narunraman/quickscope.git
cd quickscope
make sync
```

The installed CLI is `qscope`; `quickscope` is kept as a longer alias.

## Quickstart

### Adaptive search (default)

Find the hardest DyVal configurations for a model using COUP:

```bash
qscope dyval --utility error-rate --model gpt-5.4-nano --batch-size 50 --n-batches 1000
```

### Fixed-budget evaluation

Evaluate all configurations uniformly:

```bash
qscope dyval --model gpt-5.4-nano --optimizer uniform --batch-size 50 --n-batches 1000
```

### Offline datasets

Run on a pre-built dataset like STEER-ME:

```bash
# Download and prepare the dataset first
qscope download steer_me

# Then run
qscope offline --scenario steer_me --utility error-rate --model gpt-5.4-nano --batch-size 50 --n-batches 1000
```

## CLI reference

| Command | Description |
|---|---|
| `qscope dyval` | DyVal dynamic reasoning benchmark |
| `qscope offline` | Fixed datasets (STEER-ME, custom) |
| `qscope rg` | [Reasoning Gym](https://github.com/open-thought/reasoning-gym) tasks |
| `qscope results summary` | Summarize saved `results_*.json` artifacts |
| `qscope results top-configs` | Show the strongest configurations from saved results |
| `qscope plot utility` | Plot top configurations by mean utility |
| `qscope download` | Download and prepare datasets |
| `qscope list models` | List available OpenAI/Anthropic model aliases |
| `qscope list utilities` | List available utility functions |
| `qscope list scenarios` | List available scenarios |

The plot command reads current `results_*.json` artifacts and saves a PNG file. It also renders a compact terminal plot by default; pass `--no-terminal` when generating files in batch jobs. It only uses the data produced by a normal QuickScope run.

```bash
qscope plot utility results/dyval_invac_cert90 --top-k 25
```

### Optimizer modes

The `--optimizer` flag controls the evaluation strategy:

| Mode | Description |
|---|---|
| `coup` (default) | Adaptive Bayesian search via COUP. Requires `--utility`. |
| `uniform` | Uniform random evaluation across all configurations. |
| `specified` | Evaluate a specific set of configurations via `--configs`. |

## Project structure

```
quickscope/
├── quickscope/           # Core library
│   ├── cli.py            # CLI entry point
│   ├── adapters/         # LLM provider adapters (OpenAI, Anthropic)
│   ├── dataflow/         # Utilities, transforms, caching, search space loading
│   ├── optimization/     # COUP integration and fixed-budget evaluators
│   └── simulation/       # Scenario generators (DyVal, reasoning_gym, offline, …)
├── resources/
│   ├── model_configs.yaml # Model registry
│   └── search_spaces/    # Per-scenario search space definitions and utilities
└── tests/
```

## Extending QuickScope

QuickScope separates three concerns that you can swap independently:

1. **Utility function** — maps evaluation metrics to a scalar in [0, 1].
2. **Search space** — defines which parameters constitute a template identifier.
3. **Generator** — maps a template identifier to a concrete question instance.

See [`quickscope/simulation/README.md`](quickscope/simulation/README.md) for a guide on adding new scenarios.

## Citation

If you use QuickScope in your research, please cite:

```bibtex
@inproceedings{quickscope2026,
  title     = {QuickScope: Certifying Hard Questions in Dynamic LLM Benchmarks},
  author    = {Raman, Narun and others},
  year      = {2026},
}
```

## License

[MIT](LICENSE)
