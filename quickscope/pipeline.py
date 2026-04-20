"""Adaptive evaluator for finding interesting LLM evaluation scenarios."""

# Built-in packages
import random
import textwrap
import json
import time
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Literal
import threading

# Third-party packages
from rich.console import Console
from rich._spinners import SPINNERS
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from ConfigSpace import ConfigurationSpace

# Local packages
from .adapters import (
    PipelineRuntimeConfig,
)
from .dataflow import (
    get_path_registry,
    get_logger,
    ResponseCache,
)
from .optimization.utils import build_configspace
from .optimization.coup_optimization import CoupBO, CoupConfig
from .optimization.evaluator import Evaluator, EvalConfig
from .services.llm_service import LLMService
from .simulation.abstract import (
    ComponentRegistry,
    Engine,
    ScenarioGenerator,
)
from .simulation.runner import Runner

logger = get_logger("pipeline")
console = Console()

# Custom spinner for LLM evaluation
SPINNERS["llm"] = {
    "interval": 100,
    "frames": ["·", "✻", "✽", "✶", "✳", "✢"],
}


class Pipeline:
    """Main orchestrator for adaptive LLM evaluation.

    Coordinates:
    1. Bayesian optimization (suggests parameters to try)
    2. Scenario generation (parameters → evaluation dataset)
    3. Evaluation pipeline (dataset → metrics)
    4. Utility computation (metrics → scalar objective)
    """

    def __init__(
        self,
        search_space: dict | ConfigurationSpace,
        scenario_generator: ScenarioGenerator,
        utility_function: Callable[[dict], float],
        config: CoupConfig | EvalConfig,
        scenario_name: str,
        runtime_config: PipelineRuntimeConfig | None = None,
        model: str | None = None,
        opponent_model: str | None = None,
        generation_model: str | None = None,
        verifier_model: str | None = None,
        surrogate: Literal["none", "rf", "xgboost"] = "none",
        cache_enabled: bool = False,
        config_sampler=None,
        run_metadata: dict | None = None,
        resume: bool = False,
        n0_override: int | None = None,
        run_id: str | None = None,
    ):
        """Initialize adaptive evaluator.

        Args:
            search_space: Parameter space definition (dict for YAML or ConfigurationSpace for Python)
            scenario_generator: Generator that converts parameters to scenarios
            utility_function: Function mapping metrics dict → scalar utility
            config: Optimizer/evaluator configuration
            runtime_config: PipelineRuntimeConfig for pipeline execution (optional)
            resume: If True, skip initial config sampling in COUP (for resuming from history)
            n0_override: If set, use this as n0 in COUP instead of len(initial_configs)
            run_id: Unique identifier for this logical run. Used to exclude the
                    run's own cache entries on resume. Generated automatically if
                    not provided.
        """
        # Convert dict to ConfigurationSpace if needed
        if isinstance(search_space, dict):
            self.search_space = build_configspace(search_space)
        elif isinstance(search_space, ConfigurationSpace):
            self.search_space = search_space
        else:
            raise ValueError(
                f"search_space must be dict or ConfigurationSpace, got {type(search_space)}"
            )

        self.scenario_generator = scenario_generator
        self.utility_function = utility_function
        self.config = config
        self.runtime_config = runtime_config or PipelineRuntimeConfig(
            quiet=True, verbose=False
        )
        self.verbose = runtime_config.verbose if runtime_config else False

        # Explicit model aliases
        self.model = model
        self.opponent_model = opponent_model
        self.generation_model = generation_model
        self.verifier_model = verifier_model

        self.scenario_name = scenario_name
        # self.optimizer_type = optimizer
        self.surrogate = surrogate
        self.cache_enabled = cache_enabled

        self.run_metadata = run_metadata or {}
        self.run_id = run_id or uuid.uuid4().hex[:12]

        # Initialize BO based on optimizer type
        if self.config.kind == "coup":
            self.bo = CoupBO(
                space_def=self.search_space,
                cfg=self.config,
                config_sampler=config_sampler,
                resume=resume,
                n0_override=n0_override,
            )
        elif self.config.kind in ("uniform", "specified"):
            self.bo = Evaluator(
                space_def=self.search_space,
                cfg=self.config,
            )
        else:
            raise ValueError(f"Unsupported optimizer kind: {self.config.kind}")

        # Initialize RNG for scenario seed generation (reproducibility)
        self._scenario_rng = (
            np.random.default_rng(config.seed)
            if config.seed
            else np.random.default_rng()
        )

        # Execution components
        self._llm_service = LLMService()

        # Run directory for outputs
        self.run_dir: Path | None = None

    def run(
        self,
        n_batches: int,
        output_dir: str | Path | None = None,
        batch_size: int = 4,
        log_every: int = 10,
        batch_offset: int = 0,
    ) -> pd.DataFrame:
        """Run adaptive evaluation loop.

        Args:
            n_batches: Number of batches to run
            output_dir: Directory for outputs (default: auto-generated timestamped dir)
            batch_size: Number of distinct configs to select per batch
            log_every: Logging frequency (iterations)
            batch_offset: Cumulative batch count from prior executions (for
                          resume).  Used so that ``completed_batches`` in the
                          checkpoint and ``iteration`` in search_history entries
                          reflect the true total across all resume sessions.

        Returns:
            DataFrame with full history of evaluations and utilities
        """
        # Set up run directory
        if output_dir is None:
            path_registry = get_path_registry()
            self.run_dir = path_registry.make_run_directory(
                base_key="results", prefix="adaptive_eval"
            )
        else:
            self.run_dir = Path(output_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.run_dir:
            config_path = self.run_dir / "run_config.json"
            config_to_save = {**self.run_metadata, "run_id": self.run_id}
            with open(config_path, "w") as f:
                json.dump(config_to_save, f, indent=2)

        if self.verbose:
            console.print(f"Run outputs will be saved to: {self.run_dir}")

        history_path = self.run_dir / "search_history.jsonl"
        history_entry_count = 0
        if history_path.exists():
            with open(history_path, "r") as hf:
                history_entry_count = sum(1 for _ in hf)


        # Create response cache once for the entire run (snapshot semantics).
        # run_id is passed so that entries written by this logical run (including
        # from a prior execution before a crash) are excluded from the snapshot.
        cache: ResponseCache | None = None
        if self.cache_enabled and self.scenario_generator.supports_caching:
            path_registry = get_path_registry()
            cache_dir = (
                path_registry._project_root
                / ".cache"
                / "responses"
                / self.scenario_name
            )
            cache_path = cache_dir / f"{self.model or 'default'}.jsonl"
            cache = ResponseCache(cache_path, run_id=self.run_id)
            if not self.runtime_config.quiet:
                console.print(f"[dim]Cache enabled: {cache_path} (run_id={self.run_id})[/dim]")

        # Main optimization loop
        start_time = time.time()
        iteration = 0
        # batch_count = 0  # Track number of batches for log_every
        caught_exception = None

        # Get metrics function once for the scenario
        metrics_fn = ComponentRegistry.get_metrics(self.scenario_name)

        # Thread-safe lock for incremental saving
        save_lock = threading.Lock()

        # Set total budget for the evaluator (needs to know upfront)
        if isinstance(self.bo, Evaluator) and self.bo._total_budget is None:
            self.bo._total_budget = n_batches * batch_size

        try:
            for batch_n in range(n_batches):
                batch_start_time = time.perf_counter()
                # 1. Ask BO for batch of parameter suggestions
                logger.debug(f"[DEBUG] About to call bo.suggest({batch_size})")
                suggest_start = time.perf_counter()
                suggestions = self.bo.suggest(batch_size)
                suggest_elapsed = time.perf_counter() - suggest_start
                logger.debug(
                    f"[DEBUG] bo.suggest() returned {len(suggestions)} suggestions"
                )
                console.print(
                    f"  [dim][Timing] suggest={suggest_elapsed:.2f}s "
                    f"for {len(suggestions)} suggestions[/dim]"
                )

                # --- TEMP: Print optimizer suggestions ---
                logger.debug(f"\n{'='*60}")
                logger.debug(
                    f"OPTIMIZER SUGGESTIONS (batch {iteration+1}-{iteration+batch_n})"
                )
                logger.debug(f"{'='*60}")
                for s in suggestions:
                    logger.debug(f"  ticket_id={s['ticket_id']}, params={s['params']}")

                # Track scenarios saved incrementally (to avoid double-saving)
                saved_scenario_ids: set[str] = set()
                batch_tells: list[dict] = []

                # Callback for incremental saving as each scenario completes
                def on_scenario_complete(
                    scenario_id: str, result: Any, suggestion_info: dict
                ) -> None:
                    """Called immediately when a scenario finishes its final round.

                    Computes metrics + utility, saves to search_history, collects in memory.
                    BO.tell() happens at batch end, not here.
                    """
                    nonlocal batch_tells, history_entry_count

                    # Compute metrics for this single result
                    single_metrics = metrics_fn([result])
                    scenario_metrics = single_metrics.get(scenario_id, {})

                    # Build metadata for utility computation
                    params = suggestion_info.get("params", {})
                    metrics_for_utility = {**scenario_metrics}

                    # Compute utility
                    utility = self.utility_function(metrics_for_utility)

                    # Build entry in same format as _aggregate_metrics
                    entry = {
                        "ticket_id": suggestion_info.get("ticket_id"),
                        "utility": utility,
                        **params,
                        **scenario_metrics,
                        "iteration": batch_offset + batch_n + 1,  # 1-indexed cumulative
                        "batch_index": len(saved_scenario_ids),  # Incremental index
                    }

                    # Add optimizer metadata if present
                    if suggestion_info.get("optimizer_meta"):
                        entry["optimizer_meta"] = suggestion_info["optimizer_meta"]

                    # Thread-safe: save to file and collect in memory
                    with save_lock:
                        # Append to search_history.jsonl immediately (crash-safe)
                        with open(history_path, "a") as f:
                            f.write(json.dumps(entry) + "\n")
                        history_entry_count += 1

                        # Track that we saved this scenario
                        saved_scenario_ids.add(scenario_id)
                        batch_tells.append(entry)

                # 2. Run scenarios using Runner (generator called internally)
                try:
                    logger.debug(f"[DEBUG] Getting engine for {self.scenario_name}")
                    engine = ComponentRegistry.get_engine(self.scenario_name)
                    logger.debug(
                        f"[DEBUG] About to call _run_suggestions with engine {engine.__class__.__name__}"
                    )
                    run_start = time.perf_counter()
                    run_results = self._run_suggestions(
                        engine,
                        suggestions,
                        cache=cache,
                        on_scenario_complete=on_scenario_complete,
                    )
                    run_elapsed = time.perf_counter() - run_start
                    console.print(
                        f"  [dim][Timing] run_suggestions={run_elapsed:.2f}s "
                        f"for {len(run_results)} completed results[/dim]"
                    )

                    # For multi-round scenarios that weren't saved incrementally,
                    # fall back to batch processing
                    unsaved_results = [
                        r
                        for r in run_results
                        if r["result"].scenario_id not in saved_scenario_ids
                    ]

                    if unsaved_results:
                        # Extract results for metrics computation
                        evaluation_results = [r["result"] for r in unsaved_results]

                        # Build scenario_metadata
                        suggestion_meta = {
                            s["ticket_id"]: s.get("optimizer_meta") for s in suggestions
                        }
                        scenario_metadata = {
                            r["result"].scenario_id: {
                                "ticket_id": r["ticket_id"],
                                "params": r["params"],
                                "optimizer_meta": suggestion_meta.get(r["ticket_id"]),
                                "iteration": batch_offset + batch_n + 1,
                                "batch_index": i + len(saved_scenario_ids),
                            }
                            for i, r in enumerate(unsaved_results)
                        }

                        # Compute metrics for unsaved results
                        metrics_start = time.perf_counter()
                        computed_metrics = metrics_fn(evaluation_results)
                        metrics_elapsed = time.perf_counter() - metrics_start

                        # Aggregate and save (for multi-round scenarios not saved incrementally)
                        aggregate_start = time.perf_counter()
                        fallback_tells = self._aggregate_metrics(
                            computed_metrics, scenario_metadata
                        )
                        aggregate_elapsed = time.perf_counter() - aggregate_start
                        console.print(
                            f"  [dim][Timing] metrics={metrics_elapsed:.2f}s "
                            f"aggregate={aggregate_elapsed:.2f}s "
                            f"for {len(unsaved_results)} unsaved results[/dim]"
                        )

                        if fallback_tells:
                            # Save to file (BO.tell happens at batch end)
                            with save_lock:
                                with open(history_path, "a") as f:
                                    for entry in fallback_tells:
                                        f.write(json.dumps(entry) + "\n")
                                history_entry_count += len(fallback_tells)
                            batch_tells.extend(fallback_tells)

                    # Tell BO about all evaluations at batch end
                    if batch_tells:
                        tell_start = time.perf_counter()
                        self.bo.tell(batch_tells)
                        tell_elapsed = time.perf_counter() - tell_start
                        console.print(
                            f"  [dim][Timing] bo.tell={tell_elapsed:.2f}s "
                            f"for {len(batch_tells)} tells[/dim]"
                        )

                    # Save checkpoint after each batch (for resume)
                    if self.run_dir and hasattr(self.bo, "save_checkpoint"):
                        completed_batches = getattr(self.bo, "_current_iter", batch_n + 1)
                        checkpoint_start = time.perf_counter()
                        self.bo.save_checkpoint(
                            path=str(self.run_dir / "checkpoint.pkl"),
                            n_history_entries=history_entry_count,
                            completed_batches=completed_batches,
                            scenario_rng_state=self._scenario_rng.__getstate__(),
                        )
                        checkpoint_elapsed = time.perf_counter() - checkpoint_start
                        console.print(
                            f"  [dim][Timing] checkpoint={checkpoint_elapsed:.2f}s "
                            f"(completed_batches={completed_batches})[/dim]"
                        )

                    # Collect prompt/response snippets for logging (from all results)
                    sample_candidates = []
                    for r in run_results:
                        res = r["result"]
                        try:
                            prompt_trace = getattr(res, "trace", []) or []
                            responses = getattr(res, "responses", []) or []
                            if not prompt_trace or not responses:
                                continue
                            prompt_text = prompt_trace[-1]
                            response_text = responses[-1]
                            ref_answer = getattr(res, "reference_answer", None)
                            answer_format = getattr(res, "answer_format", None)
                            sample_candidates.append(
                                (
                                    getattr(res, "scenario_id", ""),
                                    prompt_text,
                                    response_text,
                                    ref_answer,
                                    answer_format,
                                )
                            )
                        except Exception:
                            continue

                    if not batch_tells:
                        logger.warning(
                            f"Batch produced no results to report! "
                            f"run_results={len(run_results)}, "
                            f"saved_incrementally={len(saved_scenario_ids)}"
                        )
                        console.print(
                            f"[yellow]⚠️  Batch {batch_n+1} produced no results[/yellow]"
                        )

                    console.print(
                        f"  [dim][Timing] batch_total={time.perf_counter() - batch_start_time:.2f}s "
                        f"(batch={batch_n + 1}/{n_batches})[/dim]"
                    )

                    # 5. Logging (log_every counts batches, not iterations)
                    if (
                        (batch_n + 1) % log_every == 0
                        # or batch_n == 0
                        or (batch_n + 1) == n_batches
                    ):
                        if batch_tells:
                            # Show mean utility across the batch
                            batch_mean_util = sum(
                                t["utility"] for t in batch_tells
                            ) / len(batch_tells)
                            if self.config.kind == "coup" and hasattr(
                                self.bo, "get_best_config"
                            ):
                                # COUP: show best utility so far
                                best_util_raw = getattr(self.bo, "_raw_max", None)
                                try:
                                    best_util = (
                                        float(best_util_raw)
                                        if best_util_raw is not None
                                        else float("-inf")
                                    )
                                except (TypeError, ValueError):
                                    best_util = float("-inf")
                                if not np.isfinite(best_util):
                                    best = self.bo.get_best_config()  # type: ignore[union-attr]
                                    best_util = (
                                        best.get("mean_utility", 0.0) if best else 0.0
                                    )
                                console.print(
                                    f"  [dim][Batch {batch_n}/{n_batches}] "
                                    f"batch_util={batch_mean_util:.3f} best={best_util:.3f}[/dim]"
                                )
                            else:
                                console.print(
                                    f"  [dim][Batch {batch_n}/{n_batches}] "
                                    f"batch_util={batch_mean_util:.3f}[/dim]"
                                )
                            if sample_candidates:
                                (
                                    scenario_id,
                                    prompt_text,
                                    response_text,
                                    ref_answer,
                                    answer_format,
                                ) = random.choice(sample_candidates)
                                resp_snippet = response_text[-100:]
                                if len(response_text) > 100:
                                    resp_snippet = f"...{resp_snippet}"
                                indent = " " * 18
                                wrap_width = 100

                                def _format_block(
                                    text: str, annotate: bool = False
                                ) -> str:
                                    lines_out: list[str] = []
                                    found = False
                                    for raw_line in text.splitlines():
                                        if not raw_line.strip():
                                            lines_out.append(indent.rstrip())
                                            continue
                                        wrapped = textwrap.wrap(
                                            raw_line,
                                            width=wrap_width,
                                            break_long_words=False,
                                            break_on_hyphens=False,
                                        ) or [raw_line]
                                        annotate_this_line = (
                                            annotate
                                            and ref_answer
                                            and ref_answer in raw_line
                                        )
                                        for idx, wline in enumerate(wrapped):
                                            if (
                                                annotate_this_line
                                                and idx == len(wrapped) - 1
                                            ):
                                                wline = f"{wline} [dim green]<- correct answer[/dim green]"
                                                found = True
                                            lines_out.append(f"{indent}{wline}")
                                    if annotate and ref_answer and not found:
                                        lines_out.append(
                                            f"{indent}{ref_answer} [dim green]<- correct answer[/dim green]"
                                        )
                                    return "\n".join(lines_out).rstrip()

                                console.print(
                                    f"  Sample prompt (scenario {scenario_id}):"
                                )
                                # Only annotate correct answer for MCQ format
                                annotate_answer = answer_format == "mcq"
                                console.print(
                                    _format_block(
                                        prompt_text, annotate=annotate_answer
                                    ),
                                    style="dim",
                                )
                                console.print(
                                    f"  Sample response tail (len={len(response_text)}):"
                                )
                                console.print(
                                    _format_block(resp_snippet, annotate=False),
                                    style="dim",
                                )

                except Exception as e:
                    logger.error(f"Evaluation pipeline failed for batch: {e}")
                    # Re-raise to crash the run instead of silently continuing
                    raise

                # 6. Increment iteration counter
                iteration += batch_n

        except Exception as e:
            caught_exception = e
            logger.error(f"Optimization loop interrupted: {e}")
        finally:

            console.print(
                f"Total time taken: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
            )

            # Always save results, even on error
            history = self.bo.get_history()
            if self.run_dir:
                history_path = self.run_dir / "search_history.jsonl"
                history_load_start = time.perf_counter()
                full_history = self._load_history_from_jsonl(history_path)
                history_load_elapsed = time.perf_counter() - history_load_start
                if full_history is not None and len(full_history) > len(history):
                    history = full_history
                    if hasattr(self.bo, "_history"):
                        self.bo._history = history.to_dict(orient="records")
                console.print(
                    f"  [dim][Timing] load_history_for_export={history_load_elapsed:.2f}s[/dim]"
                )

            # Export structured results
            export_start = time.perf_counter()
            self._export_search_results(history)
            console.print(
                f"  [dim][Timing] export_results={time.perf_counter() - export_start:.2f}s[/dim]"
            )

            # Always show results location and summary
            if self.run_dir:
                history_path = self.run_dir / "search_history.jsonl"
                console.print(f"\nSearch history saved to: {history_path}")
            self._print_summary()

        # Re-raise any caught exception after saving
        if caught_exception is not None:
            raise caught_exception

        return history

    def _load_history_from_jsonl(self, history_path: Path) -> pd.DataFrame | None:
        """Load the full run history from JSONL for final export and summary."""
        if not history_path.exists():
            return None

        records: list[dict[str, Any]] = []
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return pd.DataFrame(records)

    def _run_suggestions(
        self,
        engine: "Engine",
        suggestions: list[dict],
        cache: ResponseCache | None = None,
        on_scenario_complete: Callable[[str, Any, dict], None] | None = None,
    ) -> list[dict]:
        """Run scenarios from suggestions using Runner.run_from_params.

        Generator is called internally by Runner.

        Args:
            engine: The simulation engine to use
            suggestions: List of {ticket_id, params} from optimizer
            on_scenario_complete: Optional callback(scenario_id, result, suggestion_info)
                                  called immediately when each scenario finishes.

        Returns:
            List of {ticket_id, params, scenario, result}
        """
        console.print(
            f"\n[bold cyan]Evaluating {len(suggestions)} scenarios[/bold cyan]"
        )

        # Prepare progress display with spinner + progress bar + elapsed time
        with Progress(
            SpinnerColumn(spinner_name="llm", style="red"),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            # Main task for tracking request progress
            main_task = progress.add_task("Evaluating...", total=None)
            current_round = [1]
            max_rounds = [1]

            def progress_callback(event: str, **data):
                if event == "round_start":
                    round_num = data["round_num"]
                    current_round[0] = round_num
                    max_rounds[0] = data["max_rounds"]
                    active_count = data["active_count"]
                    # Update description and reset progress for new round
                    progress.update(
                        main_task,
                        description=f"Round {round_num}/{max_rounds[0]} ({active_count} active)",
                        completed=0,
                        total=active_count,
                    )
                elif event == "batch_info":
                    batch_parts = [
                        f"{b['model']} x{b['count']}" for b in data["batches"]
                    ]
                    batch_text = ", ".join(batch_parts)
                    console.print(f"  [dim]Batching: {batch_text}[/dim]")
                elif event == "batch_retry":
                    old = data["old_max_tokens"]
                    new = data["new_max_tokens"]
                    progress.update(
                        main_task,
                        description=(
                            f"Round {current_round[0]}/{max_rounds[0]} ({data.get('active_count', '?')} active)  "
                            f"[dim]↻ max_tokens ({old} → {new})[/dim]"
                        ),
                    )
                elif event == "timing_stage":
                    stage = data.get("stage", "unknown")
                    elapsed_s = float(data.get("elapsed_s", 0.0))
                    if stage == "scenario_generation":
                        console.print(
                            f"  [dim][Timing] scenario_generation={elapsed_s:.2f}s "
                            f"for {data.get('total_scenarios', '?')} scenarios[/dim]"
                        )
                    elif stage == "cache_lookup":
                        console.print(
                            f"  [dim][Timing] cache_lookup={elapsed_s:.2f}s "
                            f"(hits={data.get('cache_hits', 0)}, misses={data.get('cache_misses', 0)})[/dim]"
                        )
                    elif stage == "first_response":
                        console.print(
                            f"  [dim][Timing] first_response={elapsed_s:.2f}s "
                            f"({data.get('completed', '?')}/{data.get('total', '?')})[/dim]"
                        )
                    elif stage == "llm_batch":
                        first_response_s = data.get("first_response_s")
                        first_response_text = (
                            f", first_response={float(first_response_s):.2f}s"
                            if first_response_s is not None
                            else ""
                        )
                        console.print(
                            f"  [dim][Timing] llm_batch={elapsed_s:.2f}s "
                            f"({data.get('completed', '?')}/{data.get('total', '?')}{first_response_text})[/dim]"
                        )
                    elif stage == "round_total":
                        console.print(
                            f"  [dim][Timing] round_total={elapsed_s:.2f}s "
                            f"(round={data.get('round_num', '?')}, "
                            f"active={data.get('active_count', '?')}, "
                            f"requests={data.get('total_requests', '?')})[/dim]"
                        )
                elif event == "request_progress":
                    status_suffix = ""
                    if data.get("timed_out"):
                        status_suffix = " timeout"
                    elif data.get("failed"):
                        status_suffix = " error"
                    console.print(
                        f"  [dim][Timing] request_progress={data.get('completed', '?')}/{data.get('total', '?')} "
                        f"elapsed={float(data.get('elapsed_s', 0.0)):.2f}s"
                        f"{status_suffix} scenario={data.get('scenario_id', '?')}[/dim]"
                    )

            def batch_progress_callback(completed: int, total: int):
                """Called after each request completes (from LLM batch or cache hit)."""
                progress.update(main_task, completed=completed, total=total)

            runner = Runner(llm=self._llm_service)

            return runner.run_from_params(
                engine=engine,
                generator=self.scenario_generator,
                suggestions=suggestions,
                rng=self._scenario_rng,
                progress_callback=progress_callback,
                batch_progress_callback=batch_progress_callback,
                cache=cache,
                cache_only=getattr(self.runtime_config, "cache_only", False),
                request_timeout=self.runtime_config.request_timeout,
                on_scenario_complete=on_scenario_complete,
            )

    def _aggregate_metrics(
        self, computed_metrics: dict, scenario_metadata: dict[str, dict]
    ) -> list[dict]:
        """Extract scenario metrics and compute utilities.

        Args:
            computed_metrics: Dict mapping scenario_id -> metrics
                              (from engine.compute_metrics)
            scenario_metadata: Map of scenario_id -> {ticket_id, params, optimizer_meta, iteration, batch_index}

        Returns:
            List of dicts with ticket_id, utility, params, metrics, and optimizer_meta
        """
        # engine.compute_metrics returns {scenario_id: metrics} directly
        scenarios = computed_metrics

        # Build tells for each scenario
        tells = []
        for scenario_id, scenario_metrics in scenarios.items():
            meta = scenario_metadata.get(scenario_id)
            if not meta:
                continue

            # Scenario metrics are already flattened for utility computation.
            params = meta["params"]
            metrics_for_utility = {
                **scenario_metrics,
            }

            # Compute utility using user-defined function
            utility = self.utility_function(metrics_for_utility)

            entry = {
                "ticket_id": meta["ticket_id"],
                "utility": utility,
                **params,
                **scenario_metrics,
                # Add iteration/batch tracking
                "iteration": meta.get("iteration"),
                "batch_index": meta.get("batch_index"),
            }

            # Add optimizer metadata if present
            if meta.get("optimizer_meta"):
                entry["optimizer_meta"] = meta["optimizer_meta"]

            tells.append(entry)

        return tells

    def _export_search_results(self, history: pd.DataFrame) -> None:
        """Write a structured results JSON alongside search_history.jsonl."""
        if self.run_dir is None or history.empty:
            return

        # Convert NaNs to None for JSON serialization
        history_json_ready = history.replace({np.nan: None})

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.run_dir / f"results_{timestamp}.json"

        # Best row
        best = None
        if "utility" in history.columns:
            best_row = history.loc[history["utility"].idxmax()]
            best_params = {
                k: best_row[k]
                for k in history.columns
                if k
                not in {
                    "ticket_id",
                    "utility",
                    "ucb",
                    "UCB",
                    "lcb",
                    "LCB",
                    "n_evals",
                    "m",
                    "tau_used",
                    "cost",
                }
            }

            def _first_float(keys: list[str], default: float = 0.0) -> float:
                """Return the first present value cast to float, else default."""
                for key in keys:
                    if key in best_row:
                        try:
                            val = best_row[key]
                            return (
                                float(val)
                                if not isinstance(val, pd.Series)
                                else float(val.iloc[0])
                            )
                        except Exception:
                            continue
                return default

            def _first_int(keys: list[str], default: int = 1) -> int:
                for key in keys:
                    if key in best_row:
                        try:
                            val = best_row[key]
                            return (
                                int(val)
                                if not isinstance(val, pd.Series)
                                else int(val.iloc[0])
                            )
                        except Exception:
                            continue
                return default

            best = {
                "config": best_params,
                "utility": _first_float(["utility"]),
                "n_evals": _first_int(["n_evals", "m"]),
            }

            if self.config.kind == "coup":
                best.update(
                    {
                        "ucb": _first_float(["ucb", "UCB", "U_hat_ucb"]),
                        "lcb": _first_float(["lcb", "LCB", "U_hat_lcb"]),
                    }
                )
        # Config aggregates
        configs_summary = []
        configs_df = self.bo.get_config_stats()
        configs_summary = configs_df.to_dict(orient="records")

        payload = {
            "run": {
                **self.run_metadata,
                "run_dir": str(self.run_dir),
                "timestamp": pd.Timestamp.now().isoformat(),
                # "mode": "search-offline",
                "optimizer": self.config.kind,
                "surrogate": self.surrogate,
                # "model_name": self.model,
                # "opponent_model": self.opponent_model,
                "seed": getattr(self.config, "seed", None),
                "n_trials_logged": int(len(history)),
            },
            "trials": history_json_ready.to_dict(orient="records"),
            "configs": configs_summary,
            "best": best,
            "metrics": {
                "utility": {
                    "mean": (
                        float(history["utility"].mean())
                        if "utility" in history.columns
                        else None
                    ),
                    "min": (
                        float(history["utility"].min())
                        if "utility" in history.columns
                        else None
                    ),
                    "max": (
                        float(history["utility"].max())
                        if "utility" in history.columns
                        else None
                    ),
                    "count": int(len(history)),
                }
            },
        }

        try:
            with results_path.open("w", encoding="utf-8") as f:
                # allow_nan=True outputs NaN/Infinity as JS literals (not strict JSON, but round-trips)
                json.dump(payload, f, indent=2, default=str, allow_nan=True)
            console.print(f"Results saved to: {results_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: could not write results file: {e}[/yellow]")

    def _print_summary(self):
        """Print optimization summary."""
        console.print(self.bo.summarize())
