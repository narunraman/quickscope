from typing import Any, Dict, List, Literal, Optional, cast
from dataclasses import dataclass
import logging
import os
import pickle

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace

from .coup.coup import COUP
from .coup.model import ConfigEncoder, RFModel
from .utils import OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class CoupConfig(OptimizerConfig):
    """COUP-specific configuration."""

    kind: Literal["coup"] = "coup"
    exploration_param: float = 2.0
    delta: float = 0.01
    n_initial: int = 5  # Number of initial random samples
    surrogate: Literal["none", "rf", "xgboost"] = "none"
    utility_transform: str | None = None  # Auto-discovers transform by utility name
    utility_transform_scenario: str = "base"  # Scenario for transform lookup
    utility_transform_kwargs: dict[str, Any] | None = (
        None  # Kwargs for transform factory
    )
    random_exploration_prob: float = (
        1 / 3
    )  # Probability of random exploration when model active
    repulsion_alpha: float = (
        0.0  # Repulsion strength from certified configs (0 = disabled)
    )
    repulsion_time_decay: float = 10.0  # Batches for repulsion time decay
    repulsion_phase1: bool = True  # Whether repulsion also affects Phase 1 selection
    certification_threshold: float | None = (
        None  # LCB threshold for certifying configs (defaults to hinge_point if unset)
    )
    certification_strategy: str = (
        "remove"  # "remove" = drop certified configs from search; "lucb_k" = LUCB-k identification
    )


class CoupBO:
    """
    COUP optimizer wrapper implementing the ask/tell interface.
    """

    def __init__(
        self,
        space_def: ConfigurationSpace,
        cfg: CoupConfig = CoupConfig(),
        config_sampler=None,
        resume: bool = False,
        n0_override: Optional[int] = None,
    ):
        """Initialize COUP optimizer.

        Args:
            space_def: ConfigSpace defining the search space
            cfg: CoupConfig with optimizer settings
            config_sampler: Optional callable to sample configs
            resume: If True, skip initial config sampling (for resuming from history)
            n0_override: If set, use this as n0 instead of len(initial_configs).
                         Used when resuming to preserve original n0 from run metadata.
        """
        self.cfg = cfg
        self.cs = space_def
        self.surrogate = cfg.surrogate.lower() if cfg.surrogate else "none"
        self._history: List[Dict[str, Any]] = []

        # Seed the ConfigurationSpace for reproducible sampling
        self.cs.seed(cfg.seed)

        # Track seen configs for sample-without-replacement
        self._seen_config_hashes: set = set()
        self._space_exhausted = False

        # Define base config sampler
        if config_sampler is None:

            def _default_sampler():
                return space_def.sample_configuration()

            config_sampler = _default_sampler
        self._base_sampler = config_sampler

        # Wrap sampler with deduplication logic
        self.config_sampler = self._make_unique_sampler(config_sampler)

        # Initialize COUP with unique initial random configs
        # When resuming, start with empty configs (will be populated by restore_from_history)
        if resume:
            initial_configs = []
            logger.info("Resume mode: skipping initial config sampling")
        else:
            # COUP can allocate multiple samples per config, so n_initial doesn't need to match batch_size
            n_initial = cfg.n_initial
            initial_configs = []
            for _ in range(n_initial):
                cfg_sample = self.config_sampler()
                if cfg_sample is None:
                    # Space exhausted before reaching n_initial
                    logger.warning(
                        f"Config space exhausted after {len(initial_configs)} unique configs "
                        f"(requested {n_initial})"
                    )
                    break
                initial_configs.append(cfg_sample)

            if not initial_configs:
                raise ValueError("Could not sample any unique initial configurations")

        use_model = self.surrogate in {"rf", "xgboost"}
        model = None
        if use_model:
            encoder = ConfigEncoder(self.cs)
            # For now, xgboost path falls back to RF unless we wire it later
            if self.surrogate == "xgboost":
                self.surrogate = "rf"
            model = RFModel(encoder)

        # Auto-discover utility transform by name
        # Transforms are auto-loaded from resources/search_spaces/*/utility_transform.py
        # by quickscope.dataflow.loader on import
        from quickscope.dataflow import create_transform

        transform_kwargs = cfg.utility_transform_kwargs or {}
        transform_fn = (
            create_transform(
                cfg.utility_transform,
                scenario=cfg.utility_transform_scenario,
                **transform_kwargs,
            )
            if cfg.utility_transform
            else None
        )

        # Extract hinge_point from transform_kwargs if present
        hinge_point = transform_kwargs.get("hinge_point", None)

        # Certification threshold: explicit param takes priority,
        # falls back to hinge_point from transform_kwargs for backwards compat
        certification_threshold = cfg.certification_threshold
        if certification_threshold is None:
            certification_threshold = hinge_point

        # Determine n0: use override if provided (for resume), otherwise use actual count
        n0 = n0_override if n0_override is not None else len(initial_configs)

        self.coup = COUP(
            initial_configs=initial_configs,
            config_sampler=self.config_sampler,  # Use wrapped sampler with deduplication
            model_candidate_sampler=self._base_sampler,  # Candidate pool for model ranking
            on_config_added=self._mark_seen_config,  # Keep seen-set synced for model adds
            usemodel=use_model,
            model=model,
            n0=n0,
            exploration_param=cfg.exploration_param,
            utility_transform=transform_fn,
            random_exploration_prob=cfg.random_exploration_prob,
            certification_threshold=certification_threshold,
            certification_strategy=cfg.certification_strategy,
            repulsion_alpha=cfg.repulsion_alpha,
            repulsion_time_decay=cfg.repulsion_time_decay,
            repulsion_phase1=cfg.repulsion_phase1,
        )

        # Track current iteration for added_at_iter
        self._current_iter = 0
        self._pending_ticket_meta: Dict[int, Dict[str, Any]] = {}
        self._raw_min = None
        self._raw_max = None

    def _config_to_hash(self, cfg) -> tuple:
        """Convert config to hashable tuple for deduplication.

        Handles ConfigSpace Configuration objects and plain dicts.
        Rounds floats to avoid floating-point precision issues.
        """
        if hasattr(cfg, "get_dictionary"):
            cfg = cfg.get_dictionary()

        # Ensure cfg is a dict
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Expected config to be dict or ConfigSpace Configuration, got {type(cfg)}: {cfg}"
            )

        def _make_hashable(val):
            """Recursively convert value to hashable type."""
            if isinstance(val, float):
                return round(val, 6)  # Round to avoid FP precision issues
            elif isinstance(val, list):
                return tuple(_make_hashable(item) for item in val)
            elif isinstance(val, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
            else:
                return val

        items = []
        for k, v in sorted(cfg.items()):
            items.append((k, _make_hashable(v)))
        return tuple(items)

    def _mark_seen_config(self, cfg) -> None:
        """Record a configuration in seen-set bookkeeping."""
        try:
            self._seen_config_hashes.add(self._config_to_hash(cfg))
        except Exception:
            # Keep optimizer robust if a custom sampler emits an unexpected type.
            pass

    def _make_unique_sampler(self, base_sampler, max_attempts: int = 1000):
        """Wrap a sampler to return only unique (unseen) configs.

        Args:
            base_sampler: The underlying config sampler callable
            max_attempts: Max attempts before assuming space is exhausted

        Returns:
            Wrapped sampler that returns None when space is exhausted
        """

        def unique_sampler():
            if self._space_exhausted:
                return None

            for _ in range(max_attempts):
                cfg = base_sampler()
                if cfg is None:
                    # Base sampler itself returned None (e.g., custom sampler exhausted)
                    self._space_exhausted = True
                    return None

                h = self._config_to_hash(cfg)
                if h not in self._seen_config_hashes:
                    self._seen_config_hashes.add(h)
                    return cfg

            # Couldn't find unique config after max_attempts
            self._space_exhausted = True
            logger.info(
                f"Config space appears exhausted after {len(self._seen_config_hashes)} unique configs"
            )
            return None

        return unique_sampler

    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        # Update COUP's iteration counter before suggesting
        self.coup._current_iter = self._current_iter

        # Ensure we have at least one selectable config
        # Under "remove" strategy, need at least one uncertified config.
        # Under "lucb_k" strategy, certified configs stay in pool so no special guard needed.
        if self.cfg.certification_strategy == "remove":
            n_uncertified = self.coup.n - len(self.coup.certified_configs)
            while n_uncertified == 0:
                added = self.coup.add_new_config()
                if not added:
                    break
                n_uncertified += 1

        tickets = self.coup.suggest(batch_size=n)

        # Get model predictions if surrogate is enabled and model is fitted
        model_predictions: Dict[int, float] = {}
        if self.surrogate != "none" and self.coup.model is not None:
            # Check if it's fitted
            if self.coup.model.is_fitted():
                try:
                    configs_to_predict = [t["config"] for t in tickets]
                    config_indices = [t["config_index"] for t in tickets]
                    predictions = self.coup.model.predict(configs_to_predict)
                    for i, idx in enumerate(config_indices):
                        model_predictions[idx] = float(predictions[i])
                except Exception:
                    pass  # Model not ready or failed

        out = []
        for t in tickets:
            optimizer_meta = {
                "kind": "coup",
                **t["optimizer_meta"],
                "model_predicted_mean": model_predictions.get(t["config_index"]),
            }
            self._pending_ticket_meta[int(t["ticket_id"])] = dict(optimizer_meta)
            entry = {
                "ticket_id": t["ticket_id"],
                "params": t["config"],
                "optimizer_meta": optimizer_meta,
            }
            out.append(entry)

        return out

    def tell(self, results: List[Dict[str, Any]]) -> None:
        # Increment iteration counter after each tell
        self._current_iter += 1

        if not results:
            return

        # Track raw utility extrema for logging/checkpoint resume.
        for result in results:
            try:
                utility = float(result.get("utility", 0.0))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(utility):
                continue
            if self._raw_min is None or utility < self._raw_min:
                self._raw_min = utility
            if self._raw_max is None or utility > self._raw_max:
                self._raw_max = utility

        # Keep COUP's bookkeeping in sync so certification/addition metadata
        # records the batch index associated with these observations.
        self.coup._current_iter = self._current_iter

        # Pass raw utilities directly to COUP. The surrogate model (RF) is
        # scale-invariant, and COUP's DKL bounds remain valid for utilities
        # in any [0, K] sub-interval of [0, 1].
        self.coup.tell(results)

        # Bookkeeping
        for r in results:
            ticket_id = int(r["ticket_id"])
            merged_meta = dict(self._pending_ticket_meta.pop(ticket_id, {}))
            existing_meta = r.get("optimizer_meta", {})
            if isinstance(existing_meta, dict):
                merged_meta.update(existing_meta)
            outcome_meta = self.coup.pop_ticket_outcome_meta(ticket_id)
            merged_meta.update(outcome_meta)
            if merged_meta:
                r["optimizer_meta"] = merged_meta
            self._history.append(r)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def save_checkpoint(
        self,
        path: str,
        n_history_entries: int = 0,
        completed_batches: int = 0,
        scenario_rng_state: Any = None,
    ) -> None:
        """Save a complete checkpoint to disk.

        Args:
            path: File path for the pickle checkpoint
            n_history_entries: Number of JSONL entries at time of checkpoint
            completed_batches: Number of batches completed so far
            scenario_rng_state: Numpy RNG state for scenario seed generation
        """
        checkpoint = {
            # COUP internal state (complete snapshot)
            "coup_state": self.coup.get_state(),
            # CoupBO wrapper state needed to continue optimization
            "bo_history": self._history,
            "bo_raw_min": getattr(self, "_raw_min", None),
            "bo_raw_max": getattr(self, "_raw_max", None),
            "bo_seen_hashes": self._seen_config_hashes,
            "bo_current_iter": self._current_iter,
            "bo_pending_ticket_meta": self._pending_ticket_meta,
            # Surrogate model (sklearn RF / XGBoost are picklable)
            "surrogate_model": self.coup.model if self.coup.model is not None else None,
            # Resume metadata
            "n_history_entries": n_history_entries,
            "completed_batches": completed_batches,
            # Pipeline state
            "scenario_rng_state": scenario_rng_state,
        }
        # Atomic write: write to temp then rename
        tmp_path = str(path) + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(checkpoint, f)
        os.replace(tmp_path, str(path))
        logger.info(
            f"Checkpoint saved: {completed_batches} batches, {n_history_entries} history entries"
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load optimizer state from a checkpoint file.

        Restores resume-critical optimizer state.

        Args:
            path: File path to the pickle checkpoint

        Returns:
            Dict with resume metadata: n_history_entries, completed_batches,
            scenario_rng_state
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        # Restore COUP internal state
        self.coup._load_state_overrides(checkpoint["coup_state"])

        # Restore CoupBO wrapper state needed to continue optimization.
        # Keep bo_history optional for backwards compatibility with older checkpoints.
        self._history = checkpoint.get("bo_history", [])
        self._raw_min = checkpoint.get("bo_raw_min")
        self._raw_max = checkpoint.get("bo_raw_max")
        self._seen_config_hashes = checkpoint["bo_seen_hashes"]
        self._current_iter = checkpoint["bo_current_iter"]
        self._pending_ticket_meta = checkpoint.get("bo_pending_ticket_meta", {})

        # Restore surrogate model if present
        saved_model = checkpoint.get("surrogate_model")
        if saved_model is not None and self.coup.model is not None:
            self.coup.model = saved_model
            logger.info("Restored surrogate model from checkpoint")

        logger.info(
            f"Checkpoint loaded: {checkpoint['completed_batches']} batches, "
            f"{self.coup.n} configs"
        )

        completed_batches = max(
            int(checkpoint.get("completed_batches", 0)),
            int(self._current_iter),
        )

        return {
            "n_history_entries": checkpoint["n_history_entries"],
            "completed_batches": completed_batches,
            "scenario_rng_state": checkpoint.get("scenario_rng_state"),
        }

    def tell_from_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Replay orphaned JSONL entries through the normal tell path.

        Used during resume when search_history.jsonl has entries beyond what
        the checkpoint covers (from a mid-batch crash). Creates tickets and
        calls COUP.tell() for each entry.

        Args:
            entries: List of dicts from search_history.jsonl tail,
                     each with ticket_id and utility

        Returns:
            Number of entries replayed
        """
        if not entries:
            return 0

        # Build tell-compatible results
        tell_results = []
        for entry in entries:
            # Find config index by matching params
            config_params = entry.get("params", {})
            if not config_params:
                # Extract config params from entry using ConfigSpace keys
                valid_keys = set(self.cs.keys()) if hasattr(self.cs, "keys") else set()
                config_params = {k: v for k, v in entry.items() if k in valid_keys}

            config_hash = self._config_to_hash(config_params)

            # Find matching config in COUP
            config_idx = None
            for idx in range(self.coup.n):
                existing_config = self.coup.configs[idx]["config"]
                if self._config_to_hash(existing_config) == config_hash:
                    config_idx = idx
                    break

            if config_idx is None:
                logger.warning(
                    f"Orphaned entry has unknown config, skipping: {config_params}"
                )
                continue

            # Create a ticket for this entry so COUP.tell() can process it
            ticket_id = self.coup._next_ticket_id
            eval_idx = self.coup.m.get(config_idx, 0)
            self.coup._tickets[ticket_id] = (config_idx, eval_idx)
            self.coup._next_ticket_id += 1

            tell_results.append(
                {
                    "ticket_id": ticket_id,
                    "utility": float(entry.get("utility", 0.0)),
                }
            )

            # Add to history
            self._history.append(entry)

        # Tell COUP about all entries
        if tell_results:
            self.coup._current_iter = self._current_iter + 1
            self.coup.tell(tell_results)
            start_idx = len(self._history) - len(tell_results)
            for offset, tell_result in enumerate(tell_results):
                outcome_meta = self.coup.pop_ticket_outcome_meta(
                    int(tell_result["ticket_id"])
                )
                if not outcome_meta:
                    continue
                history_entry = self._history[start_idx + offset]
                existing_meta = history_entry.get("optimizer_meta", {})
                merged_meta = (
                    dict(existing_meta) if isinstance(existing_meta, dict) else {}
                )
                merged_meta.update(outcome_meta)
                history_entry["optimizer_meta"] = merged_meta
            self._current_iter += 1

        logger.info(f"Replayed {len(tell_results)} orphaned JSONL entries")
        return len(tell_results)

    def get_best_config(self) -> Dict[str, Any]:
        """Get the configuration with highest mean utility.

        Uses history-based stats for accuracy when COUP state may be stale.
        """
        if self.coup.n == 0:
            return {}

        # Compute accurate stats from history (same as get_config_stats)
        from collections import defaultdict

        config_utilities: Dict[int, List[float]] = defaultdict(list)
        for entry in self._history:
            meta = entry.get("optimizer_meta", {})
            config_idx = meta.get("config_index")
            if config_idx is not None:
                utility = float(entry.get("utility", 0.0))
                config_utilities[config_idx].append(utility)

        # Find best config by mean utility
        best_idx = -1
        best_mean = float("-inf")

        for idx in range(self.coup.n):
            if idx in config_utilities and config_utilities[idx]:
                utilities = config_utilities[idx]
                mean_util = sum(utilities) / len(utilities)
            else:
                mean_util = float(self.coup.U_hat[idx])

            if mean_util > best_mean:
                best_mean = mean_util
                best_idx = idx

        if best_idx < 0:
            best_idx = int(np.argmax(self.coup.U_hat))

        # Get stats for best config
        if best_idx in config_utilities and config_utilities[best_idx]:
            utilities = config_utilities[best_idx]
            mean_util = sum(utilities) / len(utilities)
            n_evals = len(utilities)
        else:
            mean_util = float(self.coup.U_hat[best_idx])
            n_evals = int(self.coup.m.get(best_idx, 0))

        return {
            "config_index": best_idx,
            "config": self.coup.configs[best_idx]["config"],
            "mean_utility": mean_util,
            "ucb": float(self.coup.UCB[best_idx]),
            "lcb": float(self.coup.LCB[best_idx]),
            "n_evals": n_evals,
        }

    def get_config_stats(self) -> pd.DataFrame:
        """Get statistics for all configurations.

        Computes mean_utility and n_evals directly from _history to ensure
        accuracy, as COUP's internal state (U_hat, m) may be out of sync
        with the full history (e.g., after resume or due to timing issues).

        UCB/LCB are taken from COUP state and may be stale if not all
        observations have been processed through update_bounds().
        """
        # First, compute accurate stats from history
        from collections import defaultdict

        config_utilities: Dict[int, List[float]] = defaultdict(list)
        for entry in self._history:
            meta = entry.get("optimizer_meta", {})
            config_idx = meta.get("config_index")
            if config_idx is not None:
                utility = float(entry.get("utility", 0.0))
                config_utilities[config_idx].append(utility)

        rows = []
        for idx in range(self.coup.n):
            cfg = self.coup.configs[idx]["config"]

            # Use history-based stats if available, otherwise fall back to COUP state
            if idx in config_utilities and config_utilities[idx]:
                utilities = config_utilities[idx]
                mean_util = sum(utilities) / len(utilities)
                n_evals = len(utilities)
            else:
                # Fall back to COUP state
                mean_util = float(self.coup.U_hat[idx])
                n_evals = int(self.coup.m.get(idx, 0))

            rows.append(
                {
                    "config_index": idx,
                    **cfg,
                    "mean_utility": mean_util,
                    "ucb": float(self.coup.UCB[idx]),
                    "lcb": float(self.coup.LCB[idx]),
                    "n_evals": n_evals,
                    "certified": idx in self.coup.certified_configs,
                    "certified_at_iter": self.coup.certified_at_iter.get(idx),
                    "certified_at_ticket": self.coup.certified_at_ticket.get(idx),
                    **self.coup.certification_event_meta.get(idx, {}),
                }
            )
        return pd.DataFrame(rows)

    def summarize(self, min_support: int = 1) -> str:
        """
        COUP-style summary: highest-utility config, confidence bounds, and per-config stats.

        Returns a formatted string for display. Uses "highest utility" language
        to remain neutral about whether the user is maximizing a positive metric
        or minimizing a negative one (e.g., -accuracy to find bad spots).
        """
        df = self.get_history()

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("COUP Optimization Summary")
        lines.append("=" * 60)

        if df.empty:
            lines.append("No evaluations recorded.")
            lines.append("=" * 60)
            return "\n".join(lines)

        best = self.get_best_config()
        config_stats = self.get_config_stats()

        # Filter to configs with enough evaluations
        config_stats_filtered = cast(
            pd.DataFrame, config_stats[config_stats["n_evals"] >= min_support]
        )
        config_stats_sorted = config_stats_filtered.sort_values(
            by="mean_utility", ascending=False
        )

        lines.append(f"Configurations explored: {self.coup.n}")
        lines.append(f"Total evaluations: {len(df)}")
        lines.append(f"Mean utility: {df['utility'].mean():.4f}")
        lines.append(f"Max utility: {df['utility'].max():.4f}")

        if best:
            lines.append("")
            lines.append("Highest-utility configuration:")
            lines.append(f"  Mean utility: {best.get('mean_utility', 0.0):.4f}")
            lines.append(f"  UCB: {best.get('ucb', 0.0):.4f}")
            lines.append(f"  LCB: {best.get('lcb', 0.0):.4f}")
            lines.append(f"  Evaluations: {best.get('n_evals', 0)}")
            config = best.get("config", {})
            if config:
                lines.append(f"  Parameters: {config}")

        if not config_stats_sorted.empty:
            lines.append("")
            lines.append("Top configurations by mean utility:")
            # Select relevant columns for display
            display_cols = [
                c
                for c in config_stats_sorted.columns
                if c not in {"config_index", "ucb", "lcb"}
            ]
            lines.append(
                config_stats_sorted[display_cols].head(10).to_string(index=False)
            )

        lines.append("=" * 60)
        return "\n".join(lines)
