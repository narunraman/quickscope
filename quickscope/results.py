"""Utilities for reading and summarizing Quickscope results_*.json artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_CONFIG_METRIC_KEYS = {
    "accuracy",
    "add_attempted",
    "add_check_condition_met",
    "add_check_eps_sq",
    "add_check_epsilon_prime",
    "add_check_gamma_e",
    "add_check_min_m",
    "add_check_simulation_mode",
    "add_failure_reason",
    "add_path",
    "add_succeeded",
    "answer",
    "attempted",
    "avg_accuracy",
    "batch_index",
    "bounds_refreshed_after_add",
    "brier_score",
    "completion_tokens",
    "certification_k_after_observation",
    "certification_k_before_observation",
    "certification_lcb_after_observation",
    "certification_threshold",
    "certification_threshold_after_observation",
    "certification_threshold_at_selection",
    "certification_threshold_before_observation",
    "certified",
    "certified_at_iter",
    "certified_at_ticket",
    "config",
    "config_index",
    "correct",
    "cost",
    "descriptions",
    "domain",
    "dataset_type",
    "effective_certification_threshold",
    "epsilon_prime",
    "generation_tokens",
    "gamma",
    "input_tokens",
    "iteration",
    "lcb",
    "LCB",
    "m",
    "mean_utility",
    "missing",
    "n_evals",
    "n_prior_evals",
    "n_unsampled_configs",
    "optimizer_meta",
    "output_tokens",
    "player_1",
    "pool_size",
    "pool_size_after_add",
    "pool_size_before_add",
    "prompt_template",
    "prompt_tokens",
    "reasoning_tokens",
    "round_scores",
    "source",
    "source_dir",
    "source_file",
    "tau_used",
    "ticket_id",
    "timed_out",
    "total_tokens",
    "ucb",
    "UCB",
    "utility",
    "vars",
}

_CONFIG_METRIC_PREFIXES = (
    "certification_",
    "lucb_k_",
    "repulsion_",
    "selection_",
    "winner_",
)


@dataclass
class ResultRun:
    """A single saved Quickscope run."""

    source_file: Path | None
    source_dir: Path
    run: dict[str, Any]
    trials: list[dict[str, Any]]
    configs: list[dict[str, Any]]
    best: dict[str, Any] | None
    metrics: dict[str, Any]

    @property
    def label(self) -> str:
        return self.source_dir.name

    @property
    def model(self) -> str:
        return str(
            self.run.get("model")
            or self.run.get("model_name")
            or self.run.get("model_alias")
            or "unknown"
        )

    @property
    def scenario(self) -> str:
        scenario = self.run.get("scenario_name") or self.run.get("scenario") or "unknown"
        rg_dataset = self.run.get("rg_dataset")
        if scenario == "reasoning_gym" and rg_dataset:
            return f"reasoning_gym/{rg_dataset}"
        return str(scenario)

    @property
    def utility_name(self) -> str:
        return str(self.run.get("utility_name") or self.run.get("utility") or "")

    @property
    def optimizer(self) -> str:
        return str(
            self.run.get("optimizer")
            or self.run.get("optimizer_type")
            or self.run.get("kind")
            or "unknown"
        )


def load_result_runs(path: str | Path) -> list[ResultRun]:
    """Load result runs from a results JSON file, run directory, or result tree."""
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Results path not found: {root}")

    if root.is_file():
        return [_load_result_file(root)]

    result_files = _find_latest_result_files(root)
    return [_load_result_file(file_path) for file_path in result_files]


def summarize_runs(runs: Iterable[ResultRun]) -> dict[str, Any]:
    """Build aggregate summary statistics for a sequence of result runs."""
    runs = list(runs)
    total_trials = sum(len(run.trials) for run in runs)
    total_configs = sum(_config_count(run) for run in runs)
    total_certified = sum(_certified_count(run) for run in runs)

    utilities: list[float] = []
    for run in runs:
        utilities.extend(_numeric(row.get("utility")) for row in run.trials)
    utilities = [value for value in utilities if value is not None]

    return {
        "n_runs": len(runs),
        "models": sorted({run.model for run in runs}),
        "scenarios": sorted({run.scenario for run in runs}),
        "optimizers": sorted({run.optimizer for run in runs}),
        "total_trials": total_trials,
        "total_configs": total_configs,
        "total_certified": total_certified,
        "utility_mean": _mean(utilities),
        "utility_min": min(utilities) if utilities else None,
        "utility_max": max(utilities) if utilities else None,
    }


def run_summary_rows(runs: Iterable[ResultRun]) -> list[dict[str, Any]]:
    """Return one display-friendly summary row per run."""
    rows = []
    for run in runs:
        utilities = [
            value
            for value in (_numeric(row.get("utility")) for row in run.trials)
            if value is not None
        ]
        best = _best_config_dict(run)
        rows.append(
            {
                "run": run.label,
                "model": run.model,
                "scenario": run.scenario,
                "utility": run.utility_name,
                "optimizer": run.optimizer,
                "trials": len(run.trials),
                "configs": _config_count(run),
                "certified": _certified_count(run),
                "mean_utility": _mean(utilities),
                "max_utility": max(utilities) if utilities else None,
                "best_lcb": _numeric(_first(best, "lcb", "LCB")),
                "best_utility": _numeric(_first(best, "mean_utility", "utility")),
            }
        )
    return rows


def top_config_rows(
    runs: Iterable[ResultRun],
    *,
    limit: int = 20,
    sort_by: str = "lcb",
    certified_only: bool = False,
) -> list[dict[str, Any]]:
    """Return top configuration rows across runs."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        for config in _config_rows(run):
            certified = bool(config.get("certified", False))
            if certified_only and not certified:
                continue
            params = _extract_params(config)
            rows.append(
                {
                    "run": run.label,
                    "model": run.model,
                    "scenario": run.scenario,
                    "utility_name": run.utility_name,
                    "optimizer": run.optimizer,
                    "config_index": config.get("config_index"),
                    "mean_utility": _numeric(
                        _first(config, "mean_utility", "utility")
                    ),
                    "lcb": _numeric(_first(config, "lcb", "LCB")),
                    "ucb": _numeric(_first(config, "ucb", "UCB")),
                    "n_evals": _numeric(_first(config, "n_evals", "m")),
                    "certified": certified,
                    "params": params,
                }
            )

    sort_key = {
        "utility": "mean_utility",
        "mean-utility": "mean_utility",
        "lcb": "lcb",
        "ucb": "ucb",
        "samples": "n_evals",
        "evals": "n_evals",
        "certified": "certified",
    }.get(sort_by)
    if sort_key is None:
        raise ValueError(
            "sort_by must be one of: lcb, ucb, utility, mean-utility, samples, evals, certified"
        )

    rows.sort(
        key=lambda row: (
            _sort_value(row.get(sort_key)),
            _sort_value(row.get("mean_utility")),
        ),
        reverse=True,
    )
    return rows[:limit]


def _find_latest_result_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.glob("**/results_*.json")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    latest_by_dir: dict[Path, Path] = {}
    for file_path in files:
        current = latest_by_dir.get(file_path.parent)
        if current is None or _result_sort_key(file_path) > _result_sort_key(current):
            latest_by_dir[file_path.parent] = file_path
    return sorted(latest_by_dir.values())


def _result_sort_key(path: Path) -> tuple[float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def _load_result_file(path: Path) -> ResultRun:
    data = _read_json(path)
    run = data.get("run") or {}
    trials = data.get("trials") or []
    configs = data.get("configs") or []
    return ResultRun(
        source_file=path,
        source_dir=path.parent,
        run=run if isinstance(run, dict) else {},
        trials=trials if isinstance(trials, list) else [],
        configs=configs if isinstance(configs, list) else [],
        best=data.get("best") if isinstance(data.get("best"), dict) else None,
        metrics=data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle, parse_constant=_parse_json_constant)
    return data if isinstance(data, dict) else {}


def _parse_json_constant(value: str) -> float:
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    raise ValueError(f"Unknown JSON constant: {value}")


def _config_rows(run: ResultRun) -> list[dict[str, Any]]:
    if run.configs:
        return [row for row in run.configs if isinstance(row, dict)]
    return _aggregate_configs_from_trials(run.trials)


def _aggregate_configs_from_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trials:
        return []

    groups: dict[Any, dict[str, Any]] = {}
    for row in trials:
        if not isinstance(row, dict):
            continue
        params = row.get("config") if isinstance(row.get("config"), dict) else None
        if params is None:
            params = _extract_params(row)
        optimizer_meta = row.get("optimizer_meta")
        config_index = None
        if isinstance(optimizer_meta, dict):
            config_index = optimizer_meta.get("config_index")
        key = (
            ("config_index", config_index)
            if config_index is not None
            else tuple(sorted((k, _freeze_value(v)) for k, v in params.items()))
        )
        group = groups.setdefault(
            key,
            {
                **params,
                "config_index": config_index,
                "utilities": [],
                "n_evals": 0,
            },
        )
        if isinstance(optimizer_meta, dict):
            for source_key, dest_key in (
                ("lcb_at_selection", "lcb"),
                ("ucb_at_selection", "ucb"),
            ):
                value = optimizer_meta.get(source_key)
                if value is not None:
                    group[dest_key] = value
        utility = _numeric(row.get("utility"))
        if utility is not None:
            group["utilities"].append(utility)
        group["n_evals"] += 1

    configs = []
    for idx, group in enumerate(groups.values()):
        utilities = group.pop("utilities")
        if group.get("config_index") is None:
            group["config_index"] = idx
        configs.append(
            {
                **group,
                "mean_utility": _mean(utilities),
                "certified": False,
            }
        )
    return configs


def _best_config_dict(run: ResultRun) -> dict[str, Any]:
    if run.best:
        best = dict(run.best)
        if isinstance(best.get("config"), dict):
            best.update(best["config"])
        return best
    configs = _config_rows(run)
    if not configs:
        return {}
    return max(
        configs,
        key=lambda row: _sort_value(
            _numeric(_first(row, "lcb", "LCB"))
            or _numeric(_first(row, "mean_utility", "utility"))
        ),
    )


def _config_count(run: ResultRun) -> int:
    return len(_config_rows(run))


def _certified_count(run: ResultRun) -> int:
    configs = _config_rows(run)
    if configs:
        return sum(1 for config in configs if bool(config.get("certified", False)))
    value = run.run.get("final_certified_count")
    numeric = _numeric(value)
    return int(numeric) if numeric is not None else 0


def _extract_params(row: dict[str, Any]) -> dict[str, Any]:
    params = row.get("params")
    if isinstance(params, dict):
        return _clean_params(params)
    return _clean_params({
        key: value
        for key, value in row.items()
        if key not in _CONFIG_METRIC_KEYS
        and not key.startswith("_")
        and not key.startswith(_CONFIG_METRIC_PREFIXES)
    })


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if _numeric(value) is not None or not _is_missing(value)
    }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sort_value(value: Any) -> float:
    numeric = _numeric(value)
    if numeric is None:
        return float("-inf")
    return numeric


def _freeze_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value
