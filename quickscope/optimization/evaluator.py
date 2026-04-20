"""Fixed-budget evaluator for systematic parameter space coverage.

Supports two modes:
- **uniform**: Enumerates all valid configurations in the search space
  (with optional float discretization), then assigns each config an equal
  number of evaluations.
- **specified**: Loads an explicit list of configurations from a JSON file.

Implements the same suggest/tell interface as adaptive optimizers, so it
plugs into the Pipeline without changes.
"""

from typing import Any, Literal
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)

from .utils import OptimizerConfig

logger = logging.getLogger(__name__)

MAX_ENUM_SIZE = 50_000


@dataclass
class EvalConfig(OptimizerConfig):
    """Configuration for fixed-budget evaluation.

    For ``kind="uniform"``, enumerates the search space and distributes
    the budget evenly.  For ``kind="specified"``, reads configs from a
    JSON file.
    """

    kind: Literal["uniform", "specified"] = "uniform"
    min_evals_per_config: int = 10
    n_continuous_bins: int = 5
    configs_path: str | None = None
    evals_per_config: int | None = None
    valid_configs: list[dict[str, Any]] | None = field(default=None, repr=False)


class Evaluator:
    """Fixed-budget evaluator that pre-generates all samples.

    Enumerates valid configurations (uniform) or loads them from a file
    (specified), then distributes the evaluation budget evenly across
    them. Implements the same suggest/tell interface used by adaptive
    optimizers.
    """

    def __init__(
        self,
        space_def: ConfigurationSpace,
        cfg: EvalConfig = EvalConfig(),
        total_budget: int | None = None,
    ):
        self.cfg = cfg
        self.cs = space_def
        self._history: list[dict[str, Any]] = []
        self._current_iter = 0
        self._total_budget = total_budget

        self._samples: list[dict[str, Any]] = []
        self._sample_index = 0

        self._ticket = 0
        self._ticket_to_params: dict[int, dict[str, Any]] = {}

        self._rng = np.random.default_rng(cfg.seed)
        self._hp_info = self._extract_hp_info()

        logger.info(
            f"Evaluator initialized (mode={cfg.kind}, "
            f"{len(self._hp_info)} parameters, "
            f"min_evals={cfg.min_evals_per_config}, n_bins={cfg.n_continuous_bins})"
        )

    # ------------------------------------------------------------------
    # Hyperparameter introspection
    # ------------------------------------------------------------------

    def _extract_hp_info(self) -> list[dict[str, Any]]:
        info = []
        for hp in self.cs.values():
            hp_dict: dict[str, Any] = {"name": hp.name, "hp": hp}

            if isinstance(hp, Constant):
                hp_dict["type"] = "constant"
                hp_dict["value"] = hp.value
            elif isinstance(hp, CategoricalHyperparameter):
                hp_dict["type"] = "categorical"
                hp_dict["choices"] = list(hp.choices)
            elif isinstance(hp, OrdinalHyperparameter):
                hp_dict["type"] = "ordinal"
                hp_dict["choices"] = list(hp.sequence)
            elif isinstance(hp, UniformIntegerHyperparameter):
                hp_dict["type"] = "integer"
                hp_dict["lower"] = hp.lower
                hp_dict["upper"] = hp.upper
                hp_dict["log"] = hp.log
            elif isinstance(hp, UniformFloatHyperparameter):
                hp_dict["type"] = "float"
                hp_dict["lower"] = hp.lower
                hp_dict["upper"] = hp.upper
                hp_dict["log"] = hp.log
            else:
                logger.warning(f"Unknown HP type for {hp.name}, treating as constant")
                hp_dict["type"] = "constant"
                hp_dict["value"] = hp.default_value

            info.append(hp_dict)
        return info

    # ------------------------------------------------------------------
    # Enumeration (uniform mode)
    # ------------------------------------------------------------------

    def _get_param_values(self) -> list[tuple]:
        """Return (name, values) for each hyperparameter.

        Floats are discretized into ``n_continuous_bins`` evenly-spaced values.
        """
        n_bins = self.cfg.n_continuous_bins
        params = []
        for hp in self._hp_info:
            name = hp["name"]
            if hp["type"] == "constant":
                params.append((name, [hp["value"]]))
            elif hp["type"] in ("categorical", "ordinal"):
                params.append((name, list(hp["choices"])))
            elif hp["type"] == "integer":
                lower, upper = hp["lower"], hp["upper"]
                params.append((name, list(range(lower, upper + 1))))
            elif hp["type"] == "float":
                lower, upper = hp["lower"], hp["upper"]
                if hp.get("log"):
                    log_lo, log_hi = np.log(lower), np.log(upper)
                    vals = np.exp(np.linspace(log_lo, log_hi, n_bins)).tolist()
                else:
                    vals = np.linspace(lower, upper, n_bins).tolist()
                params.append((name, vals))
        return params

    def _enumerate_valid_configs(self) -> list[dict[str, Any]] | None:
        """Enumerate all valid configs via Cartesian product + validation.

        Returns None if the product is too large or the space has conditionals.
        """
        if self.cs.conditions:
            logger.info("Space has conditional params, skipping enumeration")
            return None

        param_values = self._get_param_values()
        names = [pv[0] for pv in param_values]
        value_lists = [pv[1] for pv in param_values]

        total = 1
        for vl in value_lists:
            total *= len(vl)
            if total > MAX_ENUM_SIZE:
                logger.info(
                    f"Cartesian product too large ({total:,}+), "
                    "falling back to rejection sampling"
                )
                return None

        logger.info(f"Enumerating {total:,} param combos for validity check")

        valid: list[dict[str, Any]] = []
        for combo in product(*value_lists):
            config = dict(zip(names, combo))
            try:
                cfg_obj = Configuration(self.cs, values=config)
                valid.append(dict(cfg_obj))
            except (ValueError, KeyError, TypeError):
                continue

        logger.info(f"Found {len(valid):,} valid configs out of {total:,}")
        return valid

    # ------------------------------------------------------------------
    # Rejection sampling (uniform mode, large/conditional spaces)
    # ------------------------------------------------------------------

    def _rejection_sample_configs(self, n: int) -> list[dict[str, Any]]:
        """Collect *n* unique valid configs via ConfigSpace sampling."""
        seen: set = set()
        configs: list[dict[str, Any]] = []

        param_values = self._get_param_values()
        snap_grids: dict[str, np.ndarray] = {}
        for name, vals in param_values:
            hp_info = next(h for h in self._hp_info if h["name"] == name)
            if hp_info["type"] == "float":
                snap_grids[name] = np.array(vals)

        max_attempts = n * 20
        attempts = 0

        while len(configs) < n and attempts < max_attempts:
            attempts += 1
            try:
                cfg = self.cs.sample_configuration()
            except Exception:
                continue

            cfg_dict = dict(cfg)

            for name, grid in snap_grids.items():
                if name in cfg_dict:
                    idx = int(np.argmin(np.abs(grid - cfg_dict[name])))
                    cfg_dict[name] = float(grid[idx])

            key = tuple(sorted(cfg_dict.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg_dict)

        if len(configs) < n:
            logger.warning(
                f"Rejection sampling collected {len(configs)}/{n} configs "
                f"after {max_attempts} attempts"
            )

        return configs

    # ------------------------------------------------------------------
    # Specified mode: load configs from JSON
    # ------------------------------------------------------------------

    def _load_specified_configs(self) -> tuple[list[dict[str, Any]], list[int]]:
        """Load configs from a JSON file.

        Expected format::

            {
                "configs": [
                    {"param_a": 1, "param_b": "x"},
                    {"param_a": 2, "param_b": "y", "evals": 50},
                    ...
                ],
                "evals_per_config": 10
            }

        Each config may include an ``"evals"`` key to override the global
        ``evals_per_config``.  Configs with ``"evals": 0`` are skipped.

        Returns (configs, per_config_evals) where per_config_evals[i] is the
        number of evaluations for configs[i].
        """
        path = Path(self.cfg.configs_path)  # type: ignore[arg-type]
        if not path.exists():
            raise FileNotFoundError(f"Configs file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        raw_configs = data["configs"]
        if not raw_configs:
            raise ValueError(f"No configs found in {path}")

        default_evals = (
            self.cfg.evals_per_config
            or data.get("evals_per_config")
            or self.cfg.min_evals_per_config
        )

        configs: list[dict[str, Any]] = []
        per_config_evals: list[int] = []
        skipped = 0
        for cfg in raw_configs:
            n = cfg.pop("evals", default_evals)
            if n <= 0:
                skipped += 1
                continue
            configs.append(cfg)
            per_config_evals.append(n)

        logger.info(
            f"Loaded {len(configs)} specified configs from {path} "
            f"(eval counts: {min(per_config_evals)}-{max(per_config_evals)}, "
            f"total={sum(per_config_evals)}, skipped {skipped})"
        )
        return configs, per_config_evals

    # ------------------------------------------------------------------
    # Sample list construction
    # ------------------------------------------------------------------

    def _build_sample_list(
        self, configs: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """Assign evaluations to configs and build the flat sample list."""
        min_evals = self.cfg.min_evals_per_config
        n_valid = len(configs)

        if n_valid == 0:
            logger.error("No valid configs found!")
            return []

        max_configs = budget // min_evals

        if n_valid <= max_configs:
            selected = configs
            base_count = budget // n_valid
            remainder = budget % n_valid
        else:
            indices = self._rng.choice(n_valid, size=max_configs, replace=False)
            indices.sort()
            selected = [configs[i] for i in indices]
            base_count = min_evals
            remainder = budget - max_configs * min_evals

        samples: list[dict[str, Any]] = []

        extra_indices = (
            set(self._rng.choice(len(selected), size=remainder, replace=False).tolist())
            if remainder > 0
            else set()
        )

        for i, config in enumerate(selected):
            count = base_count + (1 if i in extra_indices else 0)
            for _ in range(count):
                samples.append(config.copy())

        self._rng.shuffle(samples)

        logger.info(
            f"Built {len(samples)} samples from {len(selected)} configs "
            f"({base_count}+ evals each, {remainder} extras)"
        )
        return samples

    def _build_specified_sample_list(
        self, configs: list[dict[str, Any]], per_config_evals: list[int]
    ) -> list[dict[str, Any]]:
        """Build sample list from specified configs with per-config repetitions."""
        samples: list[dict[str, Any]] = []
        for config, n_evals in zip(configs, per_config_evals):
            for _ in range(n_evals):
                samples.append(config.copy())

        self._rng.shuffle(samples)

        logger.info(
            f"Built {len(samples)} samples from {len(configs)} specified configs "
            f"(evals per config: {min(per_config_evals)}-{max(per_config_evals)})"
        )
        return samples

    # ------------------------------------------------------------------
    # Main sample generation
    # ------------------------------------------------------------------

    def _generate_samples(self, budget: int) -> list[dict[str, Any]]:
        """Generate the full sample list for the given budget."""
        if self.cfg.kind == "specified":
            configs, per_config_evals = self._load_specified_configs()
            return self._build_specified_sample_list(configs, per_config_evals)

        # Uniform mode — prefer pre-validated configs (e.g. from dataset index)
        if self.cfg.valid_configs is not None:
            logger.info(
                f"Using {len(self.cfg.valid_configs)} pre-validated configs"
            )
            return self._build_sample_list(self.cfg.valid_configs, budget)

        valid_configs = self._enumerate_valid_configs()

        if valid_configs is None:
            n_target = budget // self.cfg.min_evals_per_config
            valid_configs = self._rejection_sample_configs(n_target)

        return self._build_sample_list(valid_configs, budget)

    # ------------------------------------------------------------------
    # Public interface (suggest / tell)
    # ------------------------------------------------------------------

    def suggest(self, n: int = 1) -> list[dict[str, Any]]:
        """Return next n configs from the pre-generated samples."""
        if not self._samples:
            if self.cfg.kind == "specified":
                configs, per_config_evals = self._load_specified_configs()
                self._samples = self._build_specified_sample_list(
                    configs, per_config_evals
                )
            else:
                budget = self._total_budget or n * 100
                self._samples = self._generate_samples(budget)
            logger.info(f"Generated {len(self._samples)} evaluation samples")

        suggestions = []
        for _ in range(n):
            if self._sample_index >= len(self._samples):
                self._sample_index = 0
                logger.warning("Samples exhausted, wrapping around")

            config = self._samples[self._sample_index]
            self._sample_index += 1

            self._ticket += 1
            ticket_id = self._ticket
            self._ticket_to_params[ticket_id] = config.copy()

            suggestions.append(
                {
                    "ticket_id": ticket_id,
                    "params": config,
                }
            )

        return suggestions

    def tell(self, results: list[dict[str, Any]]) -> None:
        """Record results (non-adaptive)."""
        self._current_iter += 1

        for result in results:
            ticket_id = result.get("ticket_id")
            config = self._ticket_to_params.pop(ticket_id, None) if ticket_id else None

            entry = {
                "iteration": self._current_iter,
                "config": config,
                "utility": result.get("utility"),
                "ticket_id": ticket_id,
            }
            for key, value in result.items():
                if key not in entry:
                    entry[key] = value
            self._history.append(entry)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def restore_from_history(self, history: list[dict[str, Any]]) -> int:
        if not history:
            return 0

        n_completed = len(history)

        if not self._samples and self._total_budget:
            self._samples = self._generate_samples(self._total_budget)

        self._sample_index = n_completed % len(self._samples) if self._samples else 0

        for entry in history:
            self._history.append(entry)

        iterations = set(e.get("iteration", 0) for e in history)
        self._current_iter = max(iterations) if iterations else 0

        logger.info(
            f"Restored {n_completed} evaluations, sample index at {self._sample_index}"
        )
        return n_completed

    def save_checkpoint(
        self,
        path: str,
        n_history_entries: int,
        completed_batches: int,
        scenario_rng_state: Any = None,
    ) -> None:
        import pickle

        data = {
            "n_history_entries": n_history_entries,
            "completed_batches": completed_batches,
            "scenario_rng_state": scenario_rng_state,
            "sample_index": self._sample_index,
            "current_iter": self._current_iter,
            "ticket": self._ticket,
            "total_budget": self._total_budget,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(
            f"Saved checkpoint: batch={completed_batches}, idx={self._sample_index}"
        )

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        import pickle
        import json as _json

        with open(path, "rb") as f:
            data = pickle.load(f)

        if self._total_budget and not self._samples:
            self._samples = self._generate_samples(self._total_budget)

        self._sample_index = data.get("sample_index", 0)
        self._current_iter = data.get("current_iter", 0)
        self._ticket = data.get("ticket", 0)

        history_path = Path(path).parent / "search_history.jsonl"
        if history_path.exists():
            with open(history_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = _json.loads(line)
                            self._history.append(entry)
                        except _json.JSONDecodeError:
                            continue
            logger.info(
                f"Loaded {len(self._history)} history entries from {history_path}"
            )

        logger.info(
            f"Loaded checkpoint: batch={data.get('completed_batches')}, "
            f"idx={self._sample_index}, history={len(self._history)}"
        )

        return data

    def get_config_stats(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()

        config_data: dict[str, dict] = {}

        for entry in self._history:
            cfg = entry.get("config")
            if cfg is None:
                continue
            key = str(sorted(cfg.items()))
            if key not in config_data:
                config_data[key] = {"config": cfg, "utilities": []}
            util = entry.get("utility")
            if util is not None:
                config_data[key]["utilities"].append(util)

        rows = []
        for key, data in config_data.items():
            utils = data["utilities"]
            rows.append(
                {
                    **data["config"],
                    "mean_utility": np.mean(utils) if utils else None,
                    "n_evals": len(utils),
                }
            )

        return pd.DataFrame(rows)

    def summarize(self, min_support: int = 1) -> str:
        df = self.get_history()

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"Evaluation Summary (mode={self.cfg.kind})")
        lines.append("=" * 60)

        if df.empty:
            lines.append("No evaluations recorded.")
            lines.append("=" * 60)
            return "\n".join(lines)

        lines.append(f"Total samples planned: {len(self._samples)}")
        lines.append(f"Samples evaluated: {self._sample_index}")
        lines.append(f"Total observations: {len(self._history)}")

        if "utility" in df.columns:
            best_idx = df["utility"].idxmax()
            best = df.loc[best_idx]
            lines.append("")
            lines.append(f"Best utility: {best['utility']:.4f}")
            if "config" in df.columns:
                lines.append(f"Best config: {best['config']}")

        lines.append("=" * 60)
        return "\n".join(lines)
