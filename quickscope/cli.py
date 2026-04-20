"""Command-line interface for Quickscope."""

import glob
import importlib
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import rich_click as click
from rich.table import Table
from rich.console import Console

from quickscope.optimization.coup_optimization import CoupConfig
from quickscope.optimization.evaluator import EvalConfig, Evaluator
from quickscope.adapters import ModelRegistry, PipelineRuntimeConfig
from quickscope.pipeline import Pipeline
from quickscope.results import (
    load_result_runs,
    run_summary_rows,
    summarize_runs,
    top_config_rows,
)
from quickscope.plots import (
    default_plot_path,
    plot_cumulative_average,
    plot_intervals,
    plot_utility_rankings,
    render_terminal_cumulative,
    render_terminal_intervals,
    render_terminal_utility,
)
from quickscope.dataflow import (
    get_path_registry,
    get_utility,
    is_offline_scenario,
    list_utilities as list_utils_for_sim,
    list_all_utilities,
    load_search_space,
)
from quickscope.simulation.abstract import ComponentRegistry
from quickscope.simulation.default.generator import DefaultScenarioGenerator

# Configure rich-click styling
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
# click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = False
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = (
    "Try running the '--help' flag for more information."
)
click.rich_click.ERRORS_EPILOGUE = ""
click.rich_click.MAX_WIDTH = 100


_COMMON_OPTION_GROUPS = [
    {
        "name": "Core",
        "options": ["--model", "--utility", "--optimizer", "--scenario"],
    },
    {
        "name": "Search Loop",
        "options": ["--n-batches", "--batch-size", "--seed", "--log-every"],
    },
    {
        "name": "Optimizer: COUP",
        "options": [
            "--surrogate", "--exploration-param", "--n-initial",
            "--random-exploration-prob", "--repulsion-alpha",
            "--repulsion-time-decay", "--repulsion-phase1",
            "--certification-threshold", "--certification-strategy",
            "--transform-kwargs",
        ],
    },
    {
        "name": "Optimizer: uniform / specified",
        "options": ["--configs", "--evals-per-config", "--min-evals"],
    },
    {
        "name": "Runtime / Output",
        "options": [
            "--output-dir", "--walltime", "--verbose", "--resume",
            "--cache", "--cache-only", "--request-timeout",
        ],
    },
]

click.rich_click.OPTION_GROUPS = {
    f"{program} {command}": _COMMON_OPTION_GROUPS
    for program in ("qscope", "quickscope")
    for command in ("dyval", "rg", "offline")
}


console = Console()


def _parse_transform_kwargs(kwargs_str: str) -> dict[str, Any]:
    """Parse comma-separated key=value string into a kwargs dict.

    Supports automatic type coercion: floats, ints, booleans, and strings.

    Examples:
        "hinge_point=0.8,other_param=42" -> {"hinge_point": 0.8, "other_param": 42}
    """
    if not kwargs_str or not kwargs_str.strip():
        return {}

    result = {}
    for item in kwargs_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise click.BadParameter(
                f"Invalid format: '{item}'. Expected key=value format.",
                param_hint="--transform-kwargs",
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Type coercion: try float, int, bool, then string
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        else:
            try:
                # Try float first (covers int too)
                float_val = float(value)
                # Use int if it's a whole number
                result[key] = int(float_val) if float_val.is_integer() else float_val
            except ValueError:
                result[key] = value

    return result


# ---------------------------------------------------------------------------
# Param extraction helpers — avoid copy-pasting dict literals in every command
# ---------------------------------------------------------------------------


def _extract_search_args(params: dict) -> dict[str, Any]:
    return {
        "n_batches": params["n_batches"],
        "batch_size": params["batch_size"],
        "mini_batch_size": params.get("mini_batch_size", 1),
        "seed": params["seed"],
        "log_every": params["log_every"],
    }


def _extract_optimizer_args(params: dict) -> dict[str, Any]:
    return {
        "surrogate": params["surrogate"],
        "exploration_param": params["exploration_param"],
        "n_initial": params["n_initial"],
        "mini_batch_size": params.get("mini_batch_size", 1),
        "random_exploration_prob": params["random_exploration_prob"],
        "repulsion_alpha": params["repulsion_alpha"],
        "repulsion_time_decay": params["repulsion_time_decay"],
        "repulsion_phase1": params["repulsion_phase1"],
        "min_evals": params["min_evals"],
        "transform_kwargs": _parse_transform_kwargs(params.get("transform_kwargs", "")),
        "certification_threshold": params.get("certification_threshold"),
        "certification_strategy": params.get("certification_strategy", "remove"),
    }


def _extract_eval_args(params: dict) -> dict[str, Any]:
    return {
        "min_evals": params["min_evals"],
        "configs_path": params["configs"],
        "evals_per_config": params["evals_per_config"],
    }


def _extract_runtime_args(params: dict, *, has_resume: bool = True) -> dict[str, Any]:
    args: dict[str, Any] = {
        "output_dir": params["output_dir"],
        "walltime": params["walltime"],
        "verbose": params["verbose"],
        "cache": params.get("cache", False),
        "cache_only": params.get("cache_only", False),
        "request_timeout": params["request_timeout"],
    }
    if has_resume:
        args["resume"] = params.get("resume", False)
    return args


def _validate_cache_flags(params: dict) -> None:
    if params.get("cache_only") and not params.get("cache"):
        raise click.ClickException("--cache-only requires --cache")


def common_options(f):
    """Decorator that applies all shared options to a domain command."""
    # Runtime / output
    f = click.option(
        "-v", "--verbose", count=True, help="Increase verbosity (-v info, -vv debug)"
    )(f)
    f = click.option(
        "--seed",
        default=123,
        show_default=True,
        type=int,
        help="Random seed for reproducibility",
    )(f)
    f = click.option(
        "--output-dir",
        default=None,
        type=click.Path(),
        show_default="auto-generated",
        help="Output directory",
    )(f)
    f = click.option(
        "--resume",
        is_flag=True,
        default=False,
        help="Resume from checkpoint in output-dir",
    )(f)
    f = click.option(
        "--walltime", default=None, type=str, help="SLURM time limit (e.g., '4:00:00')"
    )(f)
    f = click.option(
        "--request-timeout",
        default=900,
        show_default=True,
        type=int,
        help="Request timeout in seconds (default: 15 minutes, 0 = no timeout)",
    )(f)
    f = click.option(
        "--cache-only/--no-cache-only",
        default=False,
        show_default=True,
        help="Use cached responses only; skip LLM calls on cache misses",
    )(f)
    f = click.option(
        "--cache/--no-cache",
        default=True,
        show_default=True,
        help="Enable response caching to avoid redundant LLM calls across runs",
    )(f)
    # Search loop
    f = click.option(
        "--log-every",
        default=10,
        show_default=True,
        type=int,
        help="Logging frequency (batches)",
    )(f)
    f = click.option(
        "--batch-size",
        default=5,
        show_default=True,
        type=int,
        help="Number of scenarios per batch",
    )(f)
    f = click.option(
        "--n-batches", default=50, show_default=True, type=int, help="Number of batches"
    )(f)
    # Optimizer: uniform / specified
    f = click.option(
        "--configs",
        default=None,
        type=click.Path(exists=True),
        help="Path to JSON file with configs (required for --optimizer specified)",
    )(f)
    f = click.option(
        "--evals-per-config",
        default=None,
        type=int,
        help="Evaluations per config (--optimizer specified only, overrides JSON value)",
    )(f)
    # Optimizer: COUP
    f = click.option(
        "--surrogate",
        default="none",
        type=click.Choice(["none", "rf", "xgboost"]),
        show_default=True,
        help="Surrogate model (COUP only)",
    )(f)
    f = click.option(
        "--exploration-param",
        default=2.0,
        show_default=True,
        type=float,
        help="Exploration parameter (COUP only, higher = more exploration)",
    )(f)
    f = click.option(
        "--n-initial",
        default=5,
        show_default=True,
        type=int,
        help="Number of initial random samples (COUP only)",
    )(f)
    f = click.option(
        "--random-exploration-prob",
        default=0.5,
        show_default=True,
        type=float,
        help="Probability of random exploration when model active (COUP only)",
    )(f)
    f = click.option(
        "--repulsion-alpha",
        default=0.0,
        show_default=True,
        type=float,
        help="Repulsion strength from certified configs (COUP only, 0 = disabled)",
    )(f)
    f = click.option(
        "--repulsion-time-decay",
        default=10.0,
        show_default=True,
        type=float,
        help="Time decay for repulsion (COUP only, batches)",
    )(f)
    f = click.option(
        "--repulsion-phase1/--no-repulsion-phase1",
        default=True,
        show_default=True,
        help="Whether repulsion also biases the Phase 1 incumbent/weakest-top-k choice (COUP only)",
    )(f)
    f = click.option(
        "--transform-kwargs",
        default="",
        help="Transform kwargs as comma-separated key=value pairs (e.g., hinge_point=0.8,other=42)",
    )(f)
    f = click.option(
        "--certification-threshold",
        default=None,
        type=float,
        help="LCB threshold for certifying configs (COUP only). "
             "Defaults to hinge_point from --transform-kwargs if set.",
    )(f)
    f = click.option(
        "--certification-strategy",
        default="remove",
        type=click.Choice(["remove", "lucb_k"]),
        show_default=True,
        help="Certification strategy (COUP only): 'remove' drops certified configs; "
             "'lucb_k' keeps them and uses LUCB-k identification.",
    )(f)
    f = click.option(
        "--min-evals",
        default=10,
        show_default=True,
        type=int,
        help="Min evaluations per config (--optimizer uniform only)",
    )(f)
    # Core
    f = click.option(
        "--utility",
        default=None,
        help="Utility function (required for adaptive optimizers; use `qscope list utilities` to see options)",
    )(f)
    f = click.option(
        "--optimizer",
        default="coup",
        type=click.Choice(["coup", "uniform", "specified"]),
        show_default=True,
        help="Optimization strategy: coup (adaptive), uniform/specified (fixed-budget)",
    )(f)
    f = click.option("--model", required=True, help="Model to evaluate")(f)
    return f


def _validate_optimizer_options(params: dict) -> None:
    """Validate option dependencies based on the chosen optimizer."""
    optimizer = params["optimizer"]

    if optimizer == "coup" and not params.get("utility"):
        raise click.ClickException(
            f"--utility is required when using --optimizer {optimizer}. "
            f"Use `qscope list utilities` to see available options."
        )

    if optimizer == "specified" and not params.get("configs"):
        raise click.ClickException(
            "--configs is required when using --optimizer specified."
        )


def _get_optimizer_args(params: dict) -> dict[str, Any]:
    """Extract the right optimizer args depending on the chosen optimizer."""
    optimizer = params["optimizer"]
    if optimizer in ("uniform", "specified"):
        return _extract_eval_args(params)
    return _extract_optimizer_args(params)


def _setup_pipeline_from_scenario(
    *,
    scenario: str,
    utility: str,
    model: str,
    optimizer: str,
    search_space,
    generator,
    optimizer_args: dict[str, Any],
    search_args: dict[str, Any],
    scenario_args: dict[str, Any],
    config_sampler=None,
    runtime_args: dict[str, Any],
    mode: str = "search",
    # cache: bool = False,
    # output_dir: str | None,
    # walltime: str | None,
    # verbose: int,
) -> "Pipeline":
    """Shared pipeline setup for all scenario subcommands.

    This helper consolidates the common logic for:
    - Setting up logging levels
    - Creating config objects (CoupConfig, EvalConfig, RuntimeConfig)
    - Instantiating and running the Pipeline
    - Printing completion message

    Returns:
        The Pipeline instance after running (for accessing run_dir, etc.)
    """
    # Setup logging based on verbosity
    if runtime_args["verbose"] <= 0:
        third_party_level = logging.WARNING
    elif runtime_args["verbose"] == 1:
        third_party_level = logging.INFO
    else:
        third_party_level = logging.DEBUG

    if runtime_args["verbose"] >= 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s [%(levelname)s] %(message)s",
            stream=sys.stderr,
            force=True,
        )

    for name in [
        "openai",
        "anthropic",
        "httpx",
        "httpx._client",
        "quickscope.optimization.coup.coup",
        "quickscope.optimization.coup_optimization",
    ]:
        logging.getLogger(name).setLevel(third_party_level)

    pipeline_verbose = runtime_args["verbose"] >= 1
    pipeline_quiet = runtime_args["verbose"] == 0

    # Create config objects
    if optimizer == "coup":
        coup_args = {
            "surrogate": optimizer_args["surrogate"],
            "exploration_param": optimizer_args["exploration_param"],
            "n_initial": optimizer_args["n_initial"],
            "utility_transform": utility,  # Auto-discovers transform by utility name
            "utility_transform_scenario": scenario,  # Use current scenario for transform lookup
            "utility_transform_kwargs": optimizer_args.get("transform_kwargs"),
            "random_exploration_prob": optimizer_args["random_exploration_prob"],
            "repulsion_alpha": optimizer_args.get("repulsion_alpha", 0.0),
            "repulsion_time_decay": optimizer_args.get("repulsion_time_decay", 10.0),
            "repulsion_phase1": optimizer_args.get("repulsion_phase1", True),
            "certification_threshold": optimizer_args.get("certification_threshold"),
            "certification_strategy": optimizer_args.get("certification_strategy", "remove"),
        }
        optimizer_config = CoupConfig(
            batch_size=search_args["batch_size"],
            mini_batch_size=optimizer_args.get("mini_batch_size", 1),
            seed=search_args["seed"],
            **coup_args,
        )

    elif optimizer in ("uniform", "specified"):
        eval_kwargs: dict[str, Any] = {
            "batch_size": search_args["batch_size"],
            "seed": search_args["seed"],
            "kind": optimizer,
            "min_evals_per_config": optimizer_args.get("min_evals", 10),
        }
        if optimizer == "specified":
            eval_kwargs["configs_path"] = optimizer_args.get("configs_path")
            if optimizer_args.get("evals_per_config") is not None:
                eval_kwargs["evals_per_config"] = optimizer_args["evals_per_config"]
        if optimizer_args.get("valid_configs") is not None:
            eval_kwargs["valid_configs"] = optimizer_args["valid_configs"]
        optimizer_config = EvalConfig(**eval_kwargs)
    else:
        raise click.ClickException(f"Unsupported optimizer: {optimizer}")
    runtime_config = PipelineRuntimeConfig(
        verbose=pipeline_verbose,
        quiet=pipeline_quiet,
        request_timeout=runtime_args["request_timeout"],
        cache_only=runtime_args.get("cache_only", False),
    )

    run_metadata = _setup_run_metadata(
        mode=mode,
        scenario_name=scenario,
        utility_name=utility,
        model=model,
        seed=search_args["seed"],
        optimizer_type=optimizer,
        optimizer_args=optimizer_args,
        scenario_args=scenario_args,
    )

    # Check for resume BEFORE creating Pipeline
    # This allows us to initialize COUP with empty configs when resuming
    n_batches = search_args["n_batches"]
    resume_mode = False
    n0_override = None
    resume_run_id = None

    if runtime_args.get("resume") and runtime_args.get("output_dir"):
        output_path = Path(runtime_args["output_dir"])
        checkpoint_path = output_path / "checkpoint.pkl"

        if checkpoint_path.exists():
            resume_mode = True

            # Load run_id (and n_initial for CouP) from run_config.json
            config_path = output_path / "run_config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        run_config = json.load(f)
                    resume_run_id = run_config.get("run_id")
                    if optimizer == "coup":
                        n0_override = run_config.get("n_initial")
                        if n0_override is not None:
                            console.print(
                                f"[dim]Loaded n0={n0_override} from run_config.json[/dim]"
                            )
                except (json.JSONDecodeError, IOError):
                    pass
        else:
            console.print(
                f"[dim]No checkpoint found at {checkpoint_path}, starting fresh[/dim]"
            )

    # Instantiate pipeline (with resume settings if applicable).
    # On resume, pass the original run_id so the cache excludes entries from
    # this logical run's prior execution.
    runner = Pipeline(
        search_space=search_space,
        scenario_generator=generator,
        utility_function=get_utility(utility, scenario=scenario),
        config=optimizer_config,
        scenario_name=scenario,
        runtime_config=runtime_config,
        model=model,
        opponent_model=scenario_args.get("opponent_model", None),
        generation_model=scenario_args.get("generation_model", None),
        verifier_model=scenario_args.get("verifier_model", None),
        surrogate=optimizer_args.get("surrogate", "none"),
        cache_enabled=runtime_args["cache"],
        config_sampler=config_sampler,
        run_metadata=run_metadata,
        resume=resume_mode,
        n0_override=n0_override,
        run_id=resume_run_id,
    )

    # Load checkpoint if resuming
    if resume_mode:
        output_path = Path(runtime_args["output_dir"])
        checkpoint_path = output_path / "checkpoint.pkl"

        # For the Evaluator, set total_budget before loading the checkpoint
        # so that load_checkpoint regenerates the same deterministic sample
        # list (same seed + same budget = same shuffled order).
        if isinstance(runner.bo, Evaluator):
            original_budget = n_batches * search_args["batch_size"]
            runner.bo._total_budget = original_budget

        resume_info = runner.bo.load_checkpoint(str(checkpoint_path))
        completed_batches = resume_info["completed_batches"]

        # Restore scenario RNG state
        rng_state = resume_info.get("scenario_rng_state")
        if rng_state is not None:
            runner._scenario_rng.__setstate__(rng_state)

        # Adjust remaining batches
        n_batches = max(0, n_batches - completed_batches)
        if n_batches == 0:
            console.print(
                "[yellow]      All batches already completed, nothing to do[/yellow]"
            )
            return runner

        # Build resume status message
        if hasattr(runner.bo, "coup"):
            resume_detail = f"{completed_batches} batches, {runner.bo.coup.n} configs"
        elif isinstance(runner.bo, Evaluator):
            resume_detail = (
                f"{completed_batches} batches, "
                f"{len(runner.bo._history)} evals, "
                f"sample index {runner.bo._sample_index}"
            )
        else:
            resume_detail = f"{completed_batches} batches"

        console.print(
            f"[bold green]↻ Resumed:[/bold green] Loaded checkpoint "
            f"({resume_detail})"
        )

    # Run the pipeline.  When resuming, pass completed_batches as batch_offset
    # so that checkpoint counters and iteration fields accumulate correctly.
    _ = runner.run(
        n_batches=n_batches,
        batch_size=search_args["batch_size"],
        output_dir=runtime_args["output_dir"],
        log_every=search_args["log_every"],
        batch_offset=completed_batches if resume_mode else 0,
    )

    return runner


def _setup_run_metadata(
    scenario_name: str,
    utility_name: str,
    model: str,
    seed: int,
    optimizer_type: str,
    optimizer_args: dict[str, Any],
    scenario_args: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Setup metadata dictionary for a search run.
    Args:
        scenario_name: Name of the scenario
        utility_name: Name of the utility function
        model: Model being evaluated
        batch_size: Batch size used in the search
        seed: Random seed
        optimizer_type: Type of optimizer used (e.g., 'coup', 'uniform')
        optimizer_args: Additional arguments for the optimizer
        scenario_args: Additional arguments for the scenario
        **kwargs: Any other keyword arguments to include in metadata
    Returns:
        A dictionary containing the run metadata
    """
    metadata = {
        "scenario_name": scenario_name,
        "model": model,
        "utility_name": utility_name,
        "seed": seed,
        "optimizer_type": optimizer_type,
        **optimizer_args,
        **scenario_args,
        **kwargs,
    }
    return metadata


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Quickscope — Bayesian-optimal LLM evaluation toolkit.

    \b
    Find edge cases, evaluate LLM behavior, and optimize testing using
    adaptive search and fixed-budget evaluation.

    \b
    Examples:
    ```
      qscope dyval --utility accuracy --model gpt-5.4
      qscope rg --scenario countdown --utility error-rate --model gpt-5.4
      qscope offline --scenario steer_me --utility accuracy --model gpt-5.4
    ```
    """
    pass


@main.command()
@common_options
@click.pass_context
def dyval(ctx, **kwargs):
    """Run on DyVal scenarios.

    Examples:

        ```
        qscope dyval --utility accuracy --model gpt-5.4
        qscope dyval --model gpt-5.4 --optimizer uniform
        ```
    """
    params = ctx.params
    _validate_cache_flags(params)
    _validate_optimizer_options(params)

    scenario = ctx.info_name
    search_args = _extract_search_args(params)
    optimizer_args = _get_optimizer_args(params)
    runtime_args = _extract_runtime_args(params)

    _print_run_header(scenario=scenario, params=params, search_args=search_args)

    generator_cls = ComponentRegistry.get(scenario)["generator"]
    generator = generator_cls(model=params["model"])

    path_registry = get_path_registry()
    project_root = path_registry._project_root
    space_dir = path_registry.get_path("configs") / "search_spaces" / scenario
    search_space = _resolve_search_space(space_dir, scenario, project_root)

    runner = _setup_pipeline_from_scenario(
        scenario=scenario,
        utility=params.get("utility", ""),
        model=params["model"],
        optimizer=params["optimizer"],
        search_space=search_space,
        generator=generator,
        optimizer_args=optimizer_args,
        search_args=search_args,
        runtime_args=runtime_args,
        scenario_args={},
    )

    _print_run_footer(runner)


@main.command()
@click.option(
    "--scenario",
    required=True,
    help="Name of offline scenario (e.g. 'steer_me').",
)
@common_options
@click.pass_context
def offline(ctx, **kwargs):
    """Run on a fixed dataset using an offline scenario.

    Examples:

        ```
        qscope offline --scenario steer_me --utility accuracy --model gpt-5.4
        qscope offline --scenario steer_me --model gpt-5.4 --optimizer uniform
        ```
    """
    params = ctx.params
    _validate_cache_flags(params)
    _validate_optimizer_options(params)

    scenario = params["scenario"]
    search_args = _extract_search_args(params)
    optimizer_args = _get_optimizer_args(params)
    runtime_args = _extract_runtime_args(params)

    if not is_offline_scenario(scenario):
        raise click.ClickException(
            f"'{scenario}' is not an offline scenario.\n"
            f"Offline scenarios are discovered from resources/search_spaces/offline/"
        )

    path_registry = get_path_registry()
    project_root = path_registry._project_root

    data_dir = project_root / "data" / scenario / "standardized_data"
    dataset_pattern = data_dir / "*.jsonl"
    matches = glob.glob(str(dataset_pattern))
    if not matches:
        raise click.ClickException(
            f"No dataset found at: {dataset_pattern}\n"
            f"Expected: data/{scenario}/standardized_data/*.jsonl"
        )
    if len(matches) > 1:
        console.print(
            f"[yellow]Warning: Multiple datasets found. Using first: {Path(matches[0]).name}[/yellow]"
        )
    dataset = matches[0]

    space_dir = (
        path_registry.get_path("configs") / "search_spaces" / "offline" / scenario
    )
    search_space = _resolve_search_space(space_dir, scenario, project_root)

    # For uniform mode, extract valid configs from dataset index
    if params["optimizer"] == "uniform":
        valid_configs = _extract_valid_configs_from_index(dataset, search_space)
        if valid_configs is not None:
            optimizer_args["valid_configs"] = valid_configs
            console.print(
                f"  [dim]Valid configs:[/dim]    {len(valid_configs)} "
                f"(from dataset index)"
            )

    _print_run_header(
        scenario=scenario,
        params=params,
        search_args=search_args,
        dataset=dataset,
        project_root=project_root,
    )

    with console.status(
        "[dim]Loading dataset[/dim]", spinner="simpleDotsScrolling", spinner_style="dim"
    ):
        generator = DefaultScenarioGenerator(
            dataset_path=dataset,
            model=params["model"],
        )

    config_sampler = None
    if params["optimizer"] == "coup":
        sampler_path = space_dir / "sampler.py"
        if sampler_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"quickscope_sampler_{scenario}", sampler_path
            )
            if spec is None or spec.loader is None:
                raise click.Abort()
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_sampler = getattr(module, "get_sampler", None)
            if not callable(get_sampler):
                raise click.ClickException(
                    f"sampler.py must define get_sampler(dataset_path) in {sampler_path}"
                )
            config_sampler = get_sampler(dataset)

    runner = _setup_pipeline_from_scenario(
        scenario=scenario,
        utility=params.get("utility", ""),
        model=params["model"],
        optimizer=params["optimizer"],
        generator=generator,
        search_space=search_space,
        optimizer_args=optimizer_args,
        search_args=search_args,
        scenario_args={},
        config_sampler=config_sampler,
        runtime_args=runtime_args,
        mode="offline",
    )

    _print_run_footer(runner)


@main.command()
@click.option(
    "--scenario",
    required=True,
    help="reasoning_gym dataset name (e.g., 'countdown', 'knights_knaves').",
)
@common_options
@click.pass_context
def rg(ctx, **kwargs):
    """Run on reasoning_gym datasets.

    Examples:

        ```
        qscope rg --scenario countdown --utility error-rate --model gpt-5.4
        qscope rg --scenario countdown --model gpt-5.4 --optimizer uniform
        ```
    """
    from quickscope.simulation.reasoning_gym.generator import (
        ReasoningGymGenerator,
        SUPPORTED_DATASETS,
    )

    params = ctx.params
    _validate_cache_flags(params)
    _validate_optimizer_options(params)

    rg_dataset = params["scenario"]
    if rg_dataset not in SUPPORTED_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_DATASETS))
        raise click.ClickException(
            f"Unknown reasoning_gym dataset '{rg_dataset}'. Available: {choices}"
        )

    search_args = _extract_search_args(params)
    optimizer_args = _get_optimizer_args(params)
    runtime_args = _extract_runtime_args(params)

    _print_run_header(
        scenario=f"rg/{rg_dataset}",
        params=params,
        search_args=search_args,
    )

    generator = ReasoningGymGenerator(dataset_name=rg_dataset, model=params["model"])

    path_registry = get_path_registry()
    project_root = path_registry._project_root
    space_dir = (
        path_registry.get_path("configs")
        / "search_spaces"
        / "reasoning_gym"
        / rg_dataset
    )
    search_space = _resolve_search_space(space_dir, rg_dataset, project_root)

    runner = _setup_pipeline_from_scenario(
        scenario=rg_dataset,
        utility=params.get("utility", ""),
        model=params["model"],
        optimizer=params["optimizer"],
        search_space=search_space,
        generator=generator,
        optimizer_args=optimizer_args,
        search_args=search_args,
        runtime_args=runtime_args,
        scenario_args={"rg_dataset": rg_dataset},
    )

    _print_run_footer(runner)


def _display_path(path: str | Path, base: Path | None = None) -> Path:
    """Return a short relative path when possible, otherwise the absolute path."""
    path_obj = Path(path)
    if base is None:
        return path_obj
    try:
        return path_obj.relative_to(base)
    except ValueError:
        return path_obj


def _print_run_header(
    *,
    scenario: str,
    params: dict,
    search_args: dict,
    scenario_args: dict | None = None,
    dataset: str | None = None,
    project_root: Path | None = None,
) -> None:
    """Print the run configuration banner."""
    optimizer = params["optimizer"]
    is_adaptive = optimizer == "coup"
    label = "Adaptive Search" if is_adaptive else f"Fixed-Budget Evaluation ({optimizer})"

    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]  {label}[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    console.print(f"  [dim]Scenario:[/dim]         {scenario}")
    if dataset and project_root:
        console.print(f"  [dim]Dataset:[/dim]          {_display_path(dataset, project_root)}")
    if params.get("utility"):
        console.print(f"  [dim]Utility function:[/dim] {params['utility']}")
    console.print(f"  [dim]Model:[/dim]            {params['model']}")
    if scenario_args:
        for key in ("opponent_model", "generation_model", "verifier_model"):
            if scenario_args.get(key):
                label_name = key.replace("_", " ").title()
                console.print(f"  [dim]{label_name}:[/dim]  {scenario_args[key]}")
    console.print(f"  [dim]Optimizer:[/dim]        {optimizer}")
    console.print(f"  [dim]Batches:[/dim]          {search_args['n_batches']}")
    console.print(f"  [dim]Batch size:[/dim]       {search_args['batch_size']}")
    console.print()


def _print_run_footer(runner: "Pipeline") -> None:
    """Print the run completion message."""
    console.print("\n[bold green]✓ Complete![/bold green]")
    history = runner.bo.get_history()
    console.print(f"  Processed {len(history)} scenarios")
    if runner.run_dir:
        console.print(f"  Results saved to: {runner.run_dir}")


def _extract_valid_configs_from_index(
    dataset_path: str, search_space
) -> list[dict[str, Any]] | None:
    """Extract valid parameter combos from a dataset index file.

    Reads the prebuilt ``.index.pkl`` sidecar, projects index keys down to
    the search-space dimensions, and deduplicates.  Returns ``None`` when
    no index is available (caller should fall back to enumeration).
    """
    index_file = Path(dataset_path).with_name(
        f"{Path(dataset_path).name}.index.pkl"
    )
    if not index_file.exists():
        return None

    with index_file.open("rb") as f:
        payload = pickle.load(f)

    index_fields: list[str] = payload.get("fields", [])
    index: dict = payload.get("index", {})
    if not index_fields or not index:
        return None

    # Determine search-space parameter names
    if isinstance(search_space, dict):
        space_params = set(search_space.keys())
    else:
        # ConfigurationSpace
        space_params = set(search_space.get_hyperparameter_names())

    # Project each index key to the search-space dimensions and deduplicate
    seen: set[tuple] = set()
    valid: list[dict[str, Any]] = []
    for key in index.keys():
        full = dict(zip(index_fields, key))
        projected = {k: v for k, v in full.items() if k in space_params}
        frozen = tuple(sorted(projected.items()))
        if frozen not in seen:
            seen.add(frozen)
            valid.append(projected)

    return valid


def _resolve_search_space(space_dir: Path, scenario: str, project_root: Path):
    """Resolve and load a search space from the standard directory layout."""
    candidates = [
        space_dir / "space.yaml",
        space_dir / "space.json",
        space_dir / "space.py",
    ]

    space = None
    for cand in candidates:
        if cand.exists():
            space = str(cand)
            console.print(
                f"  [dim]Search space:[/dim] {_display_path(cand, project_root)}"
            )
            break

    if space is None:
        raise click.ClickException(
            f"No search space found for '{scenario}' in {space_dir}"
        )

    return load_search_space(space)


@main.group()
def results():
    """Inspect saved Quickscope results_*.json artifacts.

    \b
    Examples:
      ```bash
      qscope results summary results/adaptive_eval_...
      qscope results top-configs results/adaptive_eval_... --sort-by lcb
      ```
    """
    pass


@results.command(name="summary")
@click.argument("path", type=click.Path(exists=True))
@click.option("--limit", default=10, show_default=True, type=int, help="Rows to show")
def results_summary_cmd(path, limit):
    """Summarize one run directory or a tree of result directories."""
    runs = load_result_runs(path)
    if not runs:
        raise click.ClickException(f"No results_*.json artifacts found under {path}")

    summary = summarize_runs(runs)
    console.print("\n[bold cyan]Results Summary[/bold cyan]\n")

    overview = Table(show_header=False, box=None)
    overview.add_column("Key", style="dim")
    overview.add_column("Value")
    overview.add_row("Runs", str(summary["n_runs"]))
    overview.add_row("Models", ", ".join(summary["models"]) or "unknown")
    overview.add_row("Scenarios", ", ".join(summary["scenarios"]) or "unknown")
    overview.add_row("Optimizers", ", ".join(summary["optimizers"]) or "unknown")
    overview.add_row("Trials", str(summary["total_trials"]))
    overview.add_row("Configs", str(summary["total_configs"]))
    overview.add_row("Certified configs", str(summary["total_certified"]))
    overview.add_row("Mean utility", _fmt_float(summary["utility_mean"]))
    overview.add_row("Max utility", _fmt_float(summary["utility_max"]))
    console.print(overview)

    rows = run_summary_rows(runs)[:limit]
    table = Table(title="Runs")
    table.add_column("Run", style="cyan")
    table.add_column("Scenario")
    table.add_column("Model")
    table.add_column("Utility")
    table.add_column("Opt")
    table.add_column("Trials", justify="right")
    table.add_column("Configs", justify="right")
    table.add_column("Cert", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("LCB", justify="right")
    for row in rows:
        table.add_row(
            str(row["run"]),
            str(row["scenario"]),
            str(row["model"]),
            str(row["utility"]),
            str(row["optimizer"]),
            str(row["trials"]),
            str(row["configs"]),
            str(row["certified"]),
            _fmt_float(row["mean_utility"]),
            _fmt_float(row["best_utility"]),
            _fmt_float(row["best_lcb"]),
        )
    console.print()
    console.print(table)

    top_rows = top_config_rows(runs, limit=min(5, limit), sort_by="lcb")
    if top_rows:
        console.print()
        _print_top_configs_table(top_rows, title="Top Configs by LCB")


@results.command(name="top-configs")
@click.argument("path", type=click.Path(exists=True))
@click.option("--limit", default=20, show_default=True, type=int, help="Rows to show")
@click.option(
    "--sort-by",
    default="lcb",
    show_default=True,
    type=click.Choice(["lcb", "ucb", "utility", "mean-utility", "samples", "evals", "certified"]),
    help="Ranking column",
)
@click.option("--certified-only", is_flag=True, help="Only show certified configs")
def results_top_configs_cmd(path, limit, sort_by, certified_only):
    """Show the strongest configurations found in saved results."""
    runs = load_result_runs(path)
    if not runs:
        raise click.ClickException(f"No results_*.json artifacts found under {path}")
    rows = top_config_rows(
        runs,
        limit=limit,
        sort_by=sort_by,
        certified_only=certified_only,
    )
    if not rows:
        raise click.ClickException("No configuration rows found")
    _print_top_configs_table(rows, title=f"Top Configs by {sort_by}")


@main.group()
def plot():
    """Create simple plots from saved Quickscope results_*.json artifacts.

    \b
    Examples:
      ```bash
      qscope plot utility results/adaptive_eval_... --top-k 25
      ```
    """
    pass


@plot.command(name="utility")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=25, show_default=True, type=int, help="Configs per run")
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option(
    "--terminal/--no-terminal",
    default=True,
    show_default=True,
    help="Render an ASCII terminal plot",
)
def plot_utility_cmd(path, output, top_k, log_x, terminal):
    """Plot top configurations by mean utility."""
    _plot_utility(
        path,
        output,
        top_k=top_k,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="intervals", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "lcb", "utility"]),
    help="Ranking column",
)
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option(
    "--terminal/--no-terminal",
    default=True,
    show_default=True,
    help="Render an ASCII terminal plot",
)
def plot_intervals_cmd(path, output, top_k, sort_by, log_x, terminal):
    """Plot LCB-UCB bands and mean-utility dots."""
    _plot_intervals(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="ci-whiskers", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "lcb", "utility"]),
    help="Ranking column",
)
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option(
    "--terminal/--no-terminal",
    default=True,
    show_default=True,
    help="Render an ASCII terminal plot",
)
def plot_ci_whiskers_cmd(path, output, top_k, sort_by, log_x, terminal):
    """Alias for `plot intervals`."""
    _plot_intervals(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="ci_whiskers", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    type=click.Choice(["auto", "lcb", "utility"]),
)
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option("--terminal/--no-terminal", default=True)
def plot_ci_whiskers_underscore_cmd(
    path,
    output,
    top_k,
    sort_by,
    log_x,
    terminal,
):
    _plot_intervals(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="cumulative", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "lcb", "utility"]),
    help="Ranking column",
)
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option(
    "--terminal/--no-terminal",
    default=True,
    show_default=True,
    help="Render an ASCII terminal plot",
)
def plot_cumulative_cmd(path, output, top_k, sort_by, log_x, terminal):
    """Plot cumulative top-K mean utility with a lower range."""
    _plot_cumulative(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="cumavg-range", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "lcb", "utility"]),
    help="Ranking column",
)
@click.option("--log-x", is_flag=True, help="Use a logarithmic x-axis")
@click.option(
    "--terminal/--no-terminal",
    default=True,
    show_default=True,
    help="Render an ASCII terminal plot",
)
def plot_cumavg_range_cmd(path, output, top_k, sort_by, log_x, terminal):
    """Alias for `plot cumulative`."""
    _plot_cumulative(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


@plot.command(name="cumavg_range", hidden=True)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(dir_okay=False), help="Output image path")
@click.option("--top-k", default=100, show_default=True, type=int, help="Configs per run")
@click.option(
    "--sort-by",
    default="auto",
    type=click.Choice(["auto", "lcb", "utility"]),
)
@click.option("--log-x", is_flag=True)
@click.option("--terminal/--no-terminal", default=True)
def plot_cumavg_range_underscore_cmd(
    path,
    output,
    top_k,
    sort_by,
    log_x,
    terminal,
):
    _plot_cumulative(
        path,
        output,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
        terminal=terminal,
    )


def _plot_utility(path, output, *, top_k: int, log_x: bool, terminal: bool) -> None:
    runs = load_result_runs(path)
    if not runs:
        raise click.ClickException(f"No results_*.json artifacts found under {path}")
    output_path = Path(output) if output else default_plot_path(
        path,
        f"utility{'_logx' if log_x else ''}.png",
    )
    series = plot_utility_rankings(
        runs,
        output_path,
        top_k=top_k,
        log_x=log_x,
    )
    if not series:
        raise click.ClickException("No configs with mean_utility found")
    console.print(f"[green]Saved plot:[/green] {_display_path(output_path, Path.cwd())}")
    if terminal:
        console.print()
        console.print(render_terminal_utility(series))


def _plot_intervals(path, output, *, top_k: int, sort_by: str, log_x: bool, terminal: bool) -> None:
    runs = load_result_runs(path)
    if not runs:
        raise click.ClickException(f"No results_*.json artifacts found under {path}")
    output_path = Path(output) if output else default_plot_path(
        path,
        f"intervals{'_logx' if log_x else ''}.png",
    )
    series = plot_intervals(
        runs,
        output_path,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
    )
    if not series:
        raise click.ClickException("No configs with mean_utility, lcb, and ucb found")
    console.print(f"[green]Saved plot:[/green] {_display_path(output_path, Path.cwd())}")
    if terminal:
        console.print()
        console.print(render_terminal_intervals(series))


def _plot_cumulative(
    path,
    output,
    *,
    top_k: int,
    sort_by: str,
    log_x: bool,
    terminal: bool,
) -> None:
    runs = load_result_runs(path)
    if not runs:
        raise click.ClickException(f"No results_*.json artifacts found under {path}")
    output_path = Path(output) if output else default_plot_path(
        path,
        f"cumulative{'_logx' if log_x else ''}.png",
    )
    series = plot_cumulative_average(
        runs,
        output_path,
        top_k=top_k,
        sort_by=sort_by,
        log_x=log_x,
    )
    if not series:
        raise click.ClickException("No configs with mean_utility found")
    console.print(f"[green]Saved plot:[/green] {_display_path(output_path, Path.cwd())}")
    if terminal:
        console.print()
        console.print(render_terminal_cumulative(series))


def _print_top_configs_table(rows: list[dict[str, Any]], *, title: str) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Run", style="cyan")
    table.add_column("Scenario")
    table.add_column("Mean", justify="right")
    table.add_column("LCB", justify="right")
    table.add_column("UCB", justify="right")
    table.add_column("Evals", justify="right")
    table.add_column("Cert", justify="center")
    table.add_column("Params")
    for index, row in enumerate(rows, start=1):
        table.add_row(
            str(index),
            str(row["run"]),
            str(row["scenario"]),
            _fmt_float(row["mean_utility"]),
            _fmt_float(row["lcb"]),
            _fmt_float(row["ucb"]),
            _fmt_float(row["n_evals"], digits=0),
            "yes" if row["certified"] else "",
            _format_params(row["params"]),
        )
    console.print(table)


def _fmt_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if digits == 0:
        return str(int(value))
    return f"{value:.{digits}f}"


def _format_params(params: dict[str, Any], *, max_len: int = 96) -> str:
    if not params:
        return ""
    text = json.dumps(params, sort_keys=True, default=str)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _render_models(filters, full):
    """Render all available models.

    Shows models registered in Quickscope's resources/model_configs.yaml.

    \b
    Examples:
      ```bash
      qscope list models                                # List all models
      qscope list models --filter provider:openai       # Filter by provider
      qscope list models --filter tags:reasoning        # Filter by tags
      ```
    """

    # Parse filters
    parsed_filters = {}
    for filter_str in filters:
        if ":" not in filter_str:
            console.print(
                f"[red]Invalid filter format: {filter_str}. Use 'key:value1,value2'[/red]"
            )
            raise click.Abort()

        key, values_str = filter_str.split(":", 1)
        values = [v.strip() for v in values_str.split(",")]

        if key not in ["provider", "tags"]:
            console.print(
                f"[red]Unsupported filter key: {key}. Supported: provider, tags[/red]"
            )
            raise click.Abort()

        parsed_filters[key] = values

    # Collect all models
    all_models = {}

    try:
        quickscope_models = ModelRegistry.available_aliases()
        for alias, settings in quickscope_models.items():
            all_models[alias] = {
                "alias": alias,
                "provider": settings.provider,
                "model": settings.model,
                "tags": settings.tags,
                "source": "Quickscope",
            }
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load models: {e}[/yellow]")

    # Apply filters
    filtered_models = {}
    for alias, model_info in all_models.items():
        include = True

        if "provider" in parsed_filters:
            if model_info["provider"] not in parsed_filters["provider"]:
                include = False

        if "tags" in parsed_filters and include:
            model_tags = model_info["tags"]
            if not any(tag in model_tags for tag in parsed_filters["tags"]):
                include = False

        if include:
            filtered_models[alias] = model_info

    # Create table
    table = Table(title="Available Models")
    table.add_column("Alias", style="cyan")
    table.add_column("Provider", style="blue")
    if full:
        table.add_column("Resolved Model", style="green")
    table.add_column("Tags", style="magenta")

    sorted_models = sorted(filtered_models.items(), key=lambda x: (x[1]["provider"], x[0]))

    for alias, info in sorted_models:
        tags_str = ", ".join(info["tags"]) if info["tags"] else ""

        if full:
            table.add_row(alias, info["provider"], info["model"], tags_str)
        else:
            table.add_row(alias, info["provider"], tags_str)

    console.print()
    console.print(table)
    console.print(f"\n[dim]Total: {len(filtered_models)} models[/dim]\n")


# ============================================================================
# List command group
# ============================================================================


@main.group(name="list")
def list_cmd():
    """List available models, utilities, and other resources.

    \\b
    Examples:
      ```bash
      qscope list models                       # List available model aliases
      qscope list utilities                    # List all utilities
      qscope list utilities --scenario dyval    # List DyVal utilities only
      ```
    """
    pass


@list_cmd.command(name="models")
@click.option(
    "--filter",
    "filters",
    multiple=True,
    help="Filter models (e.g., 'provider:openai' or 'tags:reasoning'). Can be used multiple times.",
)
@click.option("--full", is_flag=True, help="Show full resolved model names")
def list_models_cmd(filters, full):
    """List all available model aliases."""
    _render_models(filters, full)


@list_cmd.command(name="utilities")
@click.option(
    "--scenario",
    "-s",
    type=str,
    default=None,
    help="Filter by scenario (e.g., 'dyval', 'offline'). Shows all if not specified.",
)
def list_utilities_cmd(scenario):
    """List available utility functions.

    Utility functions map evaluation metrics to scalar values that guide the
    Bayesian optimizer's search for interesting scenarios.
    """

    if scenario:
        # List utilities for specific scenario
        utils = list_utils_for_sim(scenario)
        if not utils:
            console.print(
                f"[yellow]No utilities registered for scenario '{scenario}'[/yellow]"
            )
            return

        table = Table(title=f"Utilities for '{scenario}'", show_header=True)
        table.add_column("Name", style="green bold")
        table.add_column("Description")

        for name, desc in utils.items():
            table.add_row(name, desc or "")

        console.print()
        console.print(table)
        console.print(f"\n[dim]Total: {len(utils)} utilities[/dim]\n")
    else:
        # List all utilities grouped by scenario
        all_utils = list_all_utilities()
        if not all_utils:
            console.print("[yellow]No utilities registered[/yellow]")
            return

        for scenario_name, utils in all_utils.items():
            table = Table(title=f"Scenario: {scenario_name}", show_header=True)
            table.add_column("Name", style="green bold")
            table.add_column("Description")

            for name, desc in utils.items():
                table.add_row(name, desc or "")

            console.print()
            console.print(table)

        total = sum(len(u) for u in all_utils.values())
        console.print(
            f"\n[dim]Total: {total} utilities across {len(all_utils)} scenarios[/dim]\n"
        )


@list_cmd.command(name="scenarios")
def list_scenarios_cmd():
    """List available evaluation scenarios.

    Shows offline scenarios (file-based datasets) and online scenarios
    (dynamically generated).
    """

    path_registry = get_path_registry()
    config_root = path_registry.get_path("configs")

    # Find offline scenarios by scanning search space directories
    offline_dir = config_root / "search_spaces" / "offline"
    offline_scenarios = []
    if offline_dir.is_dir():
        for subdir in offline_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                offline_scenarios.append(subdir.name)
    offline_scenarios.sort()

    # Avoid importing benchmark libraries just to list commands.
    online_scenarios = []
    search_spaces_dir = config_root / "search_spaces"
    if (search_spaces_dir / "dyval").is_dir():
        online_scenarios.append("dyval")
    if (search_spaces_dir / "reasoning_gym").is_dir():
        online_scenarios.append("reasoning_gym")

    # Print results
    console.print("\n[bold cyan]Available Scenarios[/bold cyan]\n")

    if offline_scenarios:
        console.print("[bold]Offline[/bold] [dim](use: qscope offline --scenario NAME)[/dim]")
        for name in offline_scenarios:
            console.print(f"  • {name}")
    else:
        console.print("[dim]No offline scenarios found[/dim]")

    console.print()

    if online_scenarios:
        console.print("[bold]Online[/bold] [dim](use: qscope dyval|rg)[/dim]")
        for name in online_scenarios:
            console.print(f"  • {name}")
    else:
        console.print("[dim]No online scenarios registered[/dim]")

    console.print()


# ============================================================================
# Download command
# ============================================================================

_AVAILABLE_DATASETS = {
    "steer_me": {
        "hf_id": "narunraman/steer_me",
        "split": "test",
        "description": "STEER-ME microeconomic reasoning benchmark (1.58M rows)",
    },
}


@main.command()
@click.argument(
    "dataset",
    type=click.Choice(list(_AVAILABLE_DATASETS.keys()), case_sensitive=False),
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory [default: data/<dataset>/standardized_data]",
)
@click.option(
    "--tags",
    default="first_person,second_person,woman,man,anon",
    help="Comma-separated tags to include (steer_me only)",
)
def download(dataset, output_dir, tags):
    """Download and prepare a dataset for use with quickscope.

    Examples:

        ```
        qscope download steer_me
        qscope download steer_me --tags first_person,man
        ```
    """
    info = _AVAILABLE_DATASETS[dataset]

    if dataset == "steer_me":
        _download_steer_me(
            hf_id=info["hf_id"],
            split=info["split"],
            output_dir=output_dir,
            allowed_tags=[t.strip() for t in tags.split(",")],
        )
    else:
        raise click.ClickException(f"No download handler for '{dataset}'")


def _is_valid(value) -> bool:
    """Check if a value is non-null (handles None and float NaN)."""
    if value is None:
        return False
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _build_dataset_index(
    dataset_path: Path,
    *,
    fields: list[str],
    sentinel: str = "__empty__",
) -> None:
    """Build a byte-offset .index.pkl sidecar for a JSONL dataset.

    Maps tuples of (field_1, field_2, ...) to byte offsets, enabling
    fast random access and valid-config enumeration without loading
    all rows into memory.
    """
    from collections import defaultdict

    index: dict[tuple, list[int]] = defaultdict(list)

    with dataset_path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            obj = json.loads(line.decode("utf-8"))
            key = tuple(obj.get(field, sentinel) for field in fields)
            index[key].append(offset)

    index_path = dataset_path.with_name(f"{dataset_path.name}.index.pkl")
    payload = {
        "dataset_path": str(dataset_path),
        "fields": fields,
        "sentinel": sentinel,
        "index": dict(index),
    }
    with index_path.open("wb") as f:
        pickle.dump(payload, f)

    console.print(
        f"[bold green]✓ Built index ({len(index)} keys) → {index_path.name}[/bold green]"
    )


def _download_steer_me(
    hf_id: str,
    split: str,
    output_dir: str | None,
    allowed_tags: list[str],
) -> None:
    """Download SteerMe from HuggingFace and standardize to JSONL.

    Mirrors the logic from the original standardize_steerme.py script:
    1. Load the raw dataset
    2. Filter to allowed persona tags
    3. Parse options_list, resolve correct_answer to text
    4. Save as JSONL in StandardRow format for DefaultScenarioGenerator
    """
    import ast

    try:
        from datasets import load_dataset
    except ImportError:
        raise click.ClickException(
            "The 'datasets' package is required. Install it with: pip install datasets"
        )

    path_registry = get_path_registry()
    project_root = path_registry._project_root

    if output_dir is None:
        out_path = project_root / "data" / "steer_me" / "standardized_data"
    else:
        out_path = Path(output_dir)

    console.print(f"\n[bold cyan]Downloading SteerMe[/bold cyan]")
    console.print(f"  [dim]Source:[/dim]  huggingface.co/datasets/{hf_id}")
    console.print(f"  [dim]Split:[/dim]   {split}")
    console.print(f"  [dim]Tags:[/dim]    {', '.join(allowed_tags)}")
    console.print(f"  [dim]Output:[/dim]  {out_path}")
    console.print()

    with console.status("[dim]Loading dataset from HuggingFace...[/dim]",
                         spinner="simpleDotsScrolling", spinner_style="dim"):
        ds = load_dataset(hf_id, split=split)

    console.print(f"  Loaded {len(ds):,} rows")

    # Filter by tags (matches: df['tags'].isin(allowed_tags))
    ds_filtered = ds.filter(lambda row: row["tags"] in allowed_tags)
    console.print(f"  After tag filter: {len(ds_filtered):,} rows")

    # Standardize to StandardRow format
    console.print("  Standardizing...")
    standardized = []
    for idx, row in enumerate(ds_filtered):
        # Parse options_list (may be string or pre-parsed list from HF)
        options_raw = row["options_list"]
        if isinstance(options_raw, str):
            try:
                options_list = ast.literal_eval(options_raw)
            except (ValueError, SyntaxError):
                try:
                    options_list = json.loads(options_raw)
                except json.JSONDecodeError:
                    continue
        elif isinstance(options_raw, list):
            options_list = options_raw
        else:
            continue

        if not options_list:
            continue

        # Resolve correct_answer index to answer text
        correct_idx = row.get("correct_answer")
        if _is_valid(correct_idx) and int(correct_idx) < len(options_list):
            reference_answer = options_list[int(correct_idx)]
        else:
            reference_answer = None

        qid = row.get("question_id")
        standardized.append({
            "scenario_id": str(int(qid)) if _is_valid(qid) else str(idx),
            "scenario_text": row["question_text"],
            "domain": row["domain"],
            "num_rounds": 1,
            "answer_format": "mcq",
            "mcq_options": options_list,
            "reference_answer": reference_answer,
            "element": row["element"],
            "type": row["type"] if _is_valid(row.get("type")) else "__empty__",
            "tags": row["tags"],
            "question_id": row["question_id"],
            "original_question_id": row.get("original_question_id"),
        })

    # Save JSONL
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "steerme_standardized.jsonl"
    with output_file.open("w") as f:
        for entry in standardized:
            f.write(json.dumps(entry) + "\n")

    console.print(f"\n[bold green]✓ Saved {len(standardized):,} rows to {output_file}[/bold green]")

    # Build byte-offset index (required for offline scenario: sampler, uniform
    # mode valid-config extraction, and IndexedDatasetAdapter random access)
    index_fields = ["element", "type", "domain"]
    _build_dataset_index(output_file, fields=index_fields, sentinel="__empty__")


if __name__ == "__main__":
    main()
