<h1>
  <img src=".github/assets/wordmark-light.png#gh-light-mode-only" alt="QuickScope" height="64">
  <img src=".github/assets/wordmark-dark.png#gh-dark-mode-only" alt="QuickScope" height="64">
</h1>

[![DOI](https://zenodo.org/badge/1201657296.svg)](https://doi.org/10.5281/zenodo.23573835)

QuickScope finds where an LLM fails reliably on a dynamic benchmark: the configurations of a question generator whose questions the model keeps getting wrong. Built on <a href="https://arxiv.org/abs/2510.14683"><img src=".github/assets/arxiv_logo.svg" height="14" alt=""></a> <a href="https://arxiv.org/abs/2510.14683">COUP</a>, it spends model calls adaptively, concentrating them on configurations that look hard, and certifies a configuration as hard with a confidence bound rather than reporting one-off failures. You decide what counts as hard by choosing a utility (error rate, complexity-weighted error, or your own). The method and results are on the <a href="https://quickscope.narunraman.com#gh-light-mode-only"><img src=".github/assets/quickscope_mark_light.svg" height="14" alt=""></a><a href="https://quickscope.narunraman.com#gh-dark-mode-only"><img src=".github/assets/quickscope_mark_dark.svg" height="14" alt=""></a> <a href="https://quickscope.narunraman.com">project page</a> and in the paper, <a href="https://arxiv.org/abs/2604.17842"><img src=".github/assets/arxiv_logo.svg" height="14" alt=""></a> <a href="https://arxiv.org/abs/2604.17842">arXiv:2604.17842</a>.

## Run it

QuickScope runs from a clone of this repository. You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/narunraman/quickscope.git
cd quickscope
make sync
```

The CLI is `qscope` (`quickscope` also works). Models are called through OpenAI or Anthropic, so set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your environment first; `qscope list models` shows the model names it
knows. Then search for the hardest <a href="https://arxiv.org/abs/2309.17167"><img src=".github/assets/arxiv_logo.svg" height="14" alt=""></a> <a href="https://arxiv.org/abs/2309.17167">DyVal</a> configurations for a model:

```bash
qscope dyval --utility error-rate --model gpt-5.4-nano --batch-size 50 --n-batches 1000
```

Fixed datasets run the same way. For <a href="https://steer-benchmark.cs.ubc.ca"><img src=".github/assets/steer_badge.svg" height="14" alt=""></a> <a href="https://steer-benchmark.cs.ubc.ca">STEER-ME</a>, download it once and run:

```bash
qscope download steer_me
qscope offline --scenario steer_me --utility error-rate --model gpt-5.4-nano --batch-size 50 --n-batches 1000
```

## Commands

| Command | What it does |
|---|---|
| `qscope dyval` | DyVal dynamic reasoning benchmark |
| `qscope offline` | Fixed datasets (STEER-ME, or your own) |
| `qscope rg` | <a href="https://github.com/open-thought/reasoning-gym#gh-light-mode-only"><img src=".github/assets/github_logo_light.svg" height="14" alt=""></a><a href="https://github.com/open-thought/reasoning-gym#gh-dark-mode-only"><img src=".github/assets/github_logo_dark.svg" height="14" alt=""></a> [Reasoning Gym](https://github.com/open-thought/reasoning-gym) tasks |
| `qscope results summary` | Summarize saved `results_*.json` files |
| `qscope results top-configs` | Show the strongest configurations from saved results |
| `qscope plot utility` | Plot the top configurations by mean utility (a PNG, plus a terminal plot unless `--no-terminal`) |
| `qscope download` | Download and prepare datasets |
| `qscope list models` / `utilities` / `scenarios` | List what is available |

The benchmark commands take `--optimizer`: `coup` (the default; adaptive search, needs `--utility`), `uniform`
(the same budget spread evenly over configurations), or `specified` (only the configurations given with
`--configs`).

```bash
qscope plot utility results/dyval_invac_cert90 --top-k 25
```

## Add a benchmark

QuickScope keeps three things separate, and you can swap each one:

1. **Utility**: maps a batch of answers to a number between 0 and 1.
2. **Search space**: the parameters that identify a configuration.
3. **Generator**: turns a configuration into a concrete question.

[`quickscope/simulation/README.md`](quickscope/simulation/README.md) walks through adding a scenario.

## License

[MIT](LICENSE).

## Citation

If you use QuickScope in your research, please cite:

```bibtex
@misc{lundy2026quickscope,
      title={QuickScope: Certifying Hard Questions in Dynamic LLM Benchmarks},
      author={Taylor Lundy and Narun K. Raman and Kevin Leyton-Brown},
      year={2026},
      eprint={2604.17842},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.17842},
}
```

To cite the QuickScope software itself (this DOI always points to the latest release; each release on [Zenodo](https://doi.org/10.5281/zenodo.23573835) has its own):

```bibtex
@software{lundy_quickscope,
  author    = {Lundy, Taylor and Raman, Narun Krishnamurthi and Leyton-Brown, Kevin},
  title     = {{QuickScope: certifying hard questions in dynamic LLM benchmarks}},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23573835},
  url       = {https://doi.org/10.5281/zenodo.23573835}
}
```
