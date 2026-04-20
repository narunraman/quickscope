import os
import math
import time
import pickle
import logging
from typing import Tuple, Any, Dict, Iterable, List, Optional, Callable

import numpy as np


from .model import SurrogateModel
from .utils import choose_max

logger = logging.getLogger(__name__)


class COUP:
    """COUP-style optimizer over noisy utilities for configurations.

    This variant removes SAT/solver-specific notions like runtimes, instances,
    seeds, and capped runs. Instead, it focuses purely on:

    * maintaining LUCB-style upper/lower confidence bounds over utilities
      for a set of configurations;
    * deciding which configuration indices to evaluate next; and
    * optionally using a model to propose new configurations.

    The caller interacts with COUP via an ask/tell interface:

    * :meth:`ask` returns a list of configuration indices that should be
      evaluated next (and can also trigger adding fresh configurations);
    * the caller evaluates those configurations using an arbitrary pipeline and
      passes back observed utilities via :meth:`tell`.

    The "utility" here is any scalar where higher is better.
    """

    POOL_GROWTH_REFRESH_FACTOR = 1.5
    POOL_GROWTH_REFRESH_MIN_SAMPLES = 2

    def __init__(
        self,
        initial_configs: Iterable[Any],
        utility_lower_bound: float = 0.0,
        delta: float = 0.01,
        n0: int = 0,
        state: Optional[Dict[str, Any]] = None,
        savepath: Optional[str] = None,
        usemodel: bool = True,
        model: Optional[SurrogateModel] = None,
        config_sampler: Optional[Callable] = None,
        model_candidate_sampler: Optional[Callable] = None,
        on_config_added: Optional[Callable[[Any], None]] = None,
        exploration_param: float = 2.0,
        utility_transform: Optional[
            Callable[[float, float], Tuple[float, float]]
        ] = None,
        random_exploration_prob: float = 1 / 3,
        certification_threshold: Optional[float] = None,
        certification_strategy: str = "remove",
        repulsion_alpha: float = 0.0,
        repulsion_time_decay: float = 10.0,
        repulsion_phase1: bool = True,
    ):
        # Configuration storage: configs[i] = {"config": cfg, ...}
        self.configs: Dict[int, Dict[str, Any]] = {}
        for idx, cfg in enumerate(initial_configs):
            self.configs[idx] = {
                "config": cfg,
                "utilities": {},  # m -> utility
                "model_selection_index": 0.0,
            }

        self.delta = delta
        self.n0 = n0
        self.savepath = savepath
        self.exploration_param = exploration_param
        # Number of configurations
        self.n = len(self.configs)

        # evaluation count per config
        self.m = {i: 0 for i in self.configs}  # type: dict[int, int]

        # ticket bookkeeping: maps external ticket IDs to (config_index, eval_index)
        self._next_ticket_id: int = 0
        self._tickets: Dict[int, Tuple[int, int]] = {}

        # LUCB-related arrays (indexed by config index)
        self.U_hat = np.ones(self.n) * utility_lower_bound
        self.F_hat = np.ones(self.n)  # kept for compatibility with theory
        self.U_hat_ucb = np.ones(self.n)
        self.U_hat_lcb = np.ones(self.n) * utility_lower_bound
        self.F_hat_lcb = np.zeros(self.n)
        self.UCB = np.ones(self.n)
        self.LCB = np.zeros(self.n)

        # Utility transform for Bernoulli parameter interpretation
        # Maps (lcb, ucb) on θ -> (lcb, ucb) on utility
        self.utility_transform = utility_transform

        # Transformed bounds (used for selection if transform is set)
        self.UCB_transformed = np.ones(self.n)
        self.LCB_transformed = np.zeros(self.n)

        # Certified configs: configs whose LCB_transformed >= certification_threshold
        # Under "remove" strategy: removed from search and tracked separately
        # Under "lucb_k" strategy: kept in search, count determines k for LUCB-k
        self.certification_threshold = certification_threshold
        self.certification_strategy = certification_strategy  # "remove" or "lucb_k"
        self.certified_configs: set = set()
        self.certified_at_iter: Dict[int, int] = (
            {}
        )  # config_index -> iteration when certified
        self.certified_at_ticket: Dict[int, int] = (
            {}
        )  # config_index -> ticket_id that first certified it
        self.certification_event_meta: Dict[int, Dict[str, Any]] = {}
        self._ticket_outcome_meta: Dict[int, Dict[str, Any]] = {}

        # Repulsion parameters for exploration bonus
        self.repulsion_alpha = repulsion_alpha  # Strength of repulsion (0 = disabled)
        self.repulsion_time_decay = repulsion_time_decay  # Batches for time decay
        self.repulsion_phase1 = repulsion_phase1  # Whether repulsion affects Phase 1

        # book-keeping for iterations
        self.r = 0
        self.i_stars: List[int] = []
        self.epsilon_stars: List[float] = []
        self.epsilon_primes: List[float] = []
        self.gammas: List[float] = []
        self.observation_times_wallclock: List[float] = []
        self.observation_times_cpu: List[float] = []
        self.is_from_random_sample = np.ones(
            self.n, dtype=bool
        )  # All initial configs are "random"

        # Track when each config was added (iteration number)
        self.added_at_iter: Dict[int, int] = {
            i: 0 for i in range(self.n)
        }  # Initial configs added at iter 0
        self._current_iter = 0  # Track current iteration for new configs
        self._last_full_bounds_refresh_n = max(1, self.n)

        # model-based config proposal (optional)
        self.usemodel = usemodel
        if model is not None:
            self.model = model
        else:
            # Callers must pass an explicit Model/RFModel instance if usemodel=True
            self.model = None

        # generic configuration sampler callback. If provided, this will be
        # used by ``add_new_config`` to obtain new configurations instead of
        # relying on any solver-specific configspace.
        self.config_sampler = config_sampler
        # Candidate sampler for model-based ranking; should not consume the
        # unique-add sampler state in wrappers like CoupBO.
        self.model_candidate_sampler = model_candidate_sampler
        # Optional callback invoked when a config is actually added.
        self.on_config_added = on_config_added

        # Internal dedup hash set to prevent accidental duplicate inserts.
        self._config_hashes: set = set()

        # Probability of random exploration when model is active
        # Ensures continued random sampling for epsilon_prime and gamma guarantees
        self.random_exploration_prob = random_exploration_prob
        self._last_add_attempt_info: Dict[str, Any] = {}

        # any restored state can override defaults
        if state is None:
            state = {}
        self._load_state_overrides(state)
        self._rebuild_config_hashes()

    def _load_state_overrides(self, state: Dict[str, Any]) -> None:
        """Optionally overlay state onto the freshly-initialized fields.

        This keeps the constructor simple while still allowing COUP to be
        resumed from a previously saved run.
        """

        if not state:
            return

        # Core counters
        self.n = state.get("n", self.n)
        self.n0 = state.get("n0", self.n0)
        self.r = state.get("r", self.r)

        # Config storage and eval counts
        self.configs = state.get("configs", self.configs)
        self.m = state.get("m", self.m)

        # LUCB arrays
        self.U_hat = state.get("U_hat", self.U_hat)
        self.F_hat = state.get("F_hat", self.F_hat)
        self.U_hat_ucb = state.get("U_hat_ucb", self.U_hat_ucb)
        self.U_hat_lcb = state.get("U_hat_lcb", self.U_hat_lcb)
        self.F_hat_lcb = state.get("F_hat_lcb", self.F_hat_lcb)
        self.UCB = state.get("UCB", self.UCB)
        self.LCB = state.get("LCB", self.LCB)
        self.UCB_transformed = state.get("UCB_transformed", self.UCB_transformed)
        self.LCB_transformed = state.get("LCB_transformed", self.LCB_transformed)

        # Certified configs
        self.certified_configs = state.get("certified_configs", self.certified_configs)
        self.certified_at_iter = state.get("certified_at_iter", self.certified_at_iter)
        self.certified_at_ticket = state.get("certified_at_ticket", self.certified_at_ticket)
        self.certification_event_meta = state.get("certification_event_meta", self.certification_event_meta)
        self.certification_threshold = state.get("certification_threshold", self.certification_threshold)
        self.certification_strategy = state.get("certification_strategy", self.certification_strategy)
        self.repulsion_phase1 = state.get("repulsion_phase1", self.repulsion_phase1)

        # Tracking arrays
        self._wallclock_time = state.get(
            "wallclock_time", getattr(self, "_wallclock_time", 0.0)
        )
        self.t0 = state.get("t0", getattr(self, "t0", time.time()))
        self.i_stars = state.get("i_stars", self.i_stars)
        self.epsilon_stars = state.get("epsilon_stars", self.epsilon_stars)
        self.epsilon_primes = state.get("epsilon_primes", self.epsilon_primes)
        self.gammas = state.get("gammas", self.gammas)
        self.observation_times_cpu = state.get(
            "observation_times_cpu", self.observation_times_cpu
        )
        self.observation_times_wallclock = state.get(
            "observation_times_wallclock", self.observation_times_wallclock
        )
        self.is_from_random_sample = state.get(
            "is_from_random_sample", self.is_from_random_sample
        )

        # Config metadata
        self.added_at_iter = state.get("added_at_iter", self.added_at_iter)
        self._current_iter = state.get("_current_iter", self._current_iter)
        self._last_full_bounds_refresh_n = int(
            state.get("_last_full_bounds_refresh_n", max(1, self.n))
        )

        # Ticket system
        self._next_ticket_id = state.get("_next_ticket_id", self._next_ticket_id)

    def _make_hashable(self, val: Any) -> Any:
        if isinstance(val, float):
            return round(val, 6)
        if isinstance(val, list):
            return tuple(self._make_hashable(v) for v in val)
        if isinstance(val, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in val.items()))
        return val

    def _config_hash(self, cfg: Any) -> Any:
        if hasattr(cfg, "get_dictionary"):
            cfg = cfg.get_dictionary()
        if isinstance(cfg, dict):
            return tuple((k, self._make_hashable(v)) for k, v in sorted(cfg.items()))
        return self._make_hashable(cfg)

    def _rebuild_config_hashes(self) -> None:
        self._config_hashes = set()
        for cfg_data in self.configs.values():
            self._config_hashes.add(self._config_hash(cfg_data["config"]))

    def _get_selection_ucb_values(self) -> np.ndarray:
        """Return a copy of the currently active upper bounds for selection."""
        return np.copy(self.UCB_transformed if self.utility_transform else self.UCB)

    def _get_selection_lcb_values(self) -> np.ndarray:
        """Return a copy of the currently active lower bounds for selection."""
        return np.copy(self.LCB_transformed if self.utility_transform else self.LCB)

    def _transform_upper(self, value: float) -> float:
        """Map an untransformed upper-bound/mean value through utility_transform."""
        if self.utility_transform is None:
            return float(value)
        return float(self.utility_transform(0.0, float(value))[1])

    def _transform_lower_from_bounds(self, lcb: float, ucb: float) -> float:
        """Map a raw lower bound through the active utility transform."""
        if self.utility_transform is None:
            return float(lcb)
        return float(self.utility_transform(float(lcb), float(ucb))[0])

    def _certification_ceiling(self, mean: float) -> float:
        """Return the largest transformed lower bound attainable at ``mean``."""
        if self.utility_transform is None:
            return float(mean)
        return float(self.utility_transform(float(mean), float(mean))[0])

    def _samples_needed_to_reach_threshold(
        self,
        mean: float,
        threshold: float,
        *,
        current_m: int = 0,
        max_total_samples: int = 1_000_000,
    ) -> float:
        """Estimate extra samples needed to reach a certification threshold."""
        if threshold is None or not np.isfinite(float(mean)):
            return math.inf

        current_m = max(int(current_m), 0)
        target = float(threshold)
        if target > self._certification_ceiling(float(mean)) + 1e-12:
            return math.inf

        if current_m > 0:
            current_ucb, current_lcb = self._compute_expected_bounds_after_sample(
                float(mean), current_m
            )
            current_lcb_t = self._transform_lower_from_bounds(current_lcb, current_ucb)
            if current_lcb_t >= target - 1e-12:
                return 0.0

        low = current_m
        high = max(1, current_m)
        while high <= max_total_samples:
            high_ucb, high_lcb = self._compute_expected_bounds_after_sample(
                float(mean), high
            )
            high_lcb_t = self._transform_lower_from_bounds(high_lcb, high_ucb)
            if high_lcb_t >= target - 1e-12:
                break
            if high == max_total_samples:
                return math.inf
            high = min(max_total_samples, max(high + 1, 2 * high))
        else:
            return math.inf

        while low + 1 < high:
            mid = (low + high) // 2
            mid_ucb, mid_lcb = self._compute_expected_bounds_after_sample(
                float(mean), mid
            )
            mid_lcb_t = self._transform_lower_from_bounds(mid_lcb, mid_ucb)
            if mid_lcb_t >= target - 1e-12:
                high = mid
            else:
                low = mid

        return float(max(0, high - current_m))

    def _compute_lucb_k_threshold_after_expansion(
        self,
        group_indices: Iterable[int],
        base_threshold: float,
        *,
        tol: float = 1e-6,
    ) -> float:
        """Raise tau so one fresh arm costs as much as lifting the current top-k."""
        indices = [int(idx) for idx in group_indices if 0 <= int(idx) < self.n]
        if len(indices) < 2:
            return float(base_threshold)

        means = []
        counts = []
        ceilings = []
        for idx in indices:
            mean = float(self.U_hat[idx])
            if not np.isfinite(mean):
                continue
            means.append(mean)
            counts.append(int(self.m.get(idx, 0)))
            ceilings.append(self._certification_ceiling(mean))

        if len(means) < 2:
            return float(base_threshold)

        lower = float(base_threshold)
        upper = float(min(ceilings))
        if not np.isfinite(upper) or upper <= lower + tol:
            return lower

        def balance_gap(threshold: float) -> float:
            fresh_costs = []
            existing_costs = []
            for mean, count in zip(means, counts):
                fresh_cost = self._samples_needed_to_reach_threshold(
                    mean, threshold, current_m=0
                )
                existing_cost = self._samples_needed_to_reach_threshold(
                    mean, threshold, current_m=count
                )
                if not np.isfinite(fresh_cost) or not np.isfinite(existing_cost):
                    return math.inf
                fresh_costs.append(fresh_cost)
                existing_costs.append(existing_cost)
            return (sum(fresh_costs) / len(fresh_costs)) - sum(existing_costs)

        upper_gap = balance_gap(upper)
        if not np.isfinite(upper_gap) or upper_gap > 0:
            return lower

        low = lower
        high = upper
        for _ in range(40):
            mid = 0.5 * (low + high)
            gap = balance_gap(mid)
            if not np.isfinite(gap):
                high = mid
            elif gap > 0:
                low = mid
            else:
                high = mid

        candidates = []
        low_gap = balance_gap(low)
        if low > lower + tol and np.isfinite(low_gap):
            candidates.append((abs(low_gap), low))
        high_gap = balance_gap(high)
        if high > lower + tol and np.isfinite(high_gap):
            candidates.append((abs(high_gap), high))

        if not candidates:
            return lower
        return float(min(candidates)[1])

    def _build_lucb_k_threshold_growth_metadata(
        self,
        group_indices: Iterable[int],
        old_threshold: float,
        new_threshold: float,
    ) -> Dict[str, Any]:
        """Summarize the LUCB-k threshold update after an expansion."""
        indices = [int(idx) for idx in group_indices if 0 <= int(idx) < self.n]
        means = [float(self.U_hat[idx]) for idx in indices]
        counts = [int(self.m.get(idx, 0)) for idx in indices]

        meta: Dict[str, Any] = {
            "lucb_k_threshold_growth_considered": True,
            "lucb_k_threshold_growth_applied": bool(
                float(new_threshold) > float(old_threshold) + 1e-12
            ),
            "lucb_k_threshold_growth_old_threshold": float(old_threshold),
            "lucb_k_threshold_growth_new_threshold": float(new_threshold),
            "lucb_k_threshold_growth_delta": float(new_threshold - old_threshold),
            "lucb_k_threshold_growth_group_indices": indices,
            "lucb_k_threshold_growth_group_means": means,
            "lucb_k_threshold_growth_group_counts": counts,
            "lucb_k_threshold_growth_avg_fresh_cost": None,
            "lucb_k_threshold_growth_total_existing_extra_cost": None,
            "lucb_k_threshold_growth_balance_gap": None,
        }

        if not indices:
            return meta

        fresh_costs = []
        existing_costs = []
        for mean, count in zip(means, counts):
            fresh_cost = self._samples_needed_to_reach_threshold(
                mean, float(new_threshold), current_m=0
            )
            existing_cost = self._samples_needed_to_reach_threshold(
                mean, float(new_threshold), current_m=count
            )
            if not np.isfinite(fresh_cost) or not np.isfinite(existing_cost):
                return meta
            fresh_costs.append(float(fresh_cost))
            existing_costs.append(float(existing_cost))

        avg_fresh_cost = float(sum(fresh_costs) / len(fresh_costs))
        total_existing_extra_cost = float(sum(existing_costs))
        meta["lucb_k_threshold_growth_avg_fresh_cost"] = avg_fresh_cost
        meta["lucb_k_threshold_growth_total_existing_extra_cost"] = total_existing_extra_cost
        meta["lucb_k_threshold_growth_balance_gap"] = (
            avg_fresh_cost - total_existing_extra_cost
        )
        return meta

    def pop_ticket_outcome_meta(self, ticket_id: int) -> Dict[str, Any]:
        """Return and clear post-observation metadata for a completed ticket."""
        return dict(self._ticket_outcome_meta.pop(int(ticket_id), {}))

    def _mask_indices_inplace(
        self, values: np.ndarray, indices: Iterable[int], fill_value: Any
    ) -> None:
        """Set selected indices in ``values`` to ``fill_value`` (bounds-safe)."""
        for idx in indices:
            if idx < len(values):
                values[idx] = fill_value

    def _create_ticket(
        self,
        idx: int,
        batch_counts: Dict[int, int],
        optimizer_meta_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and register one evaluation ticket for config ``idx``."""
        # Use max(utilities.keys())+1 instead of self.m[idx] to compute
        # the next eval_index.  self.m counts actual observations
        # (len(utilities)), which can be less than the highest allocated
        # index when previous tickets were lost.  Using len() would
        # assign an eval_index that collides with an existing key,
        # silently overwriting it and permanently freezing m.
        utilities = self.configs[idx].get("utilities", {})
        base = (max(utilities.keys()) + 1) if utilities else 0
        eval_index = base + batch_counts[idx]
        ticket_id = self._next_ticket_id
        self._next_ticket_id += 1
        self._tickets[ticket_id] = (idx, eval_index)

        n_unsampled = sum(1 for m in self.m.values() if m == 0)

        optimizer_meta = {
            "config_index": idx,
            "added_at_iter": self.added_at_iter.get(idx, 0),
            "n_prior_evals": self.m.get(idx, 0),
            "pool_size": self.n,
            "n_unsampled_configs": n_unsampled,
            "ucb_at_selection": (
                float(self.UCB[idx]) if idx < len(self.UCB) else None
            ),
            "lcb_at_selection": (
                float(self.LCB[idx]) if idx < len(self.LCB) else None
            ),
            "mean_utility_estimate": (
                float(self.U_hat[idx]) if idx < len(self.U_hat) else None
            ),
            "is_random_sample": bool(self.is_from_random_sample[idx]),
            "certification_strategy": self.certification_strategy,
            "certification_threshold_at_selection": (
                float(self.certification_threshold)
                if self.certification_threshold is not None
                else None
            ),
            "k_at_selection": int(len(self.certified_configs)),
            "epsilon_prime": float(self.epsilon_prime),
            "gamma": float(self.gamma),
            "selection_phase": None,
            "n_certified_configs": int(len(self.certified_configs)),
            "repulsion_active": False,
            "repulsion_bonus": None,
            "repulsion_bonus_mode": None,
            "repulsion_weighted_proximity": None,
            "repulsion_global_decay": None,
            "selection_score_before_repulsion": None,
            "selection_score_after_repulsion": None,
            "winner_without_repulsion": None,
            "winner_without_repulsion_score": None,
            "repulsion_changed_selection": None,
            "add_check_epsilon_prime": None,
            "add_check_eps_sq": None,
            "add_check_gamma_e": None,
            "add_check_min_m": None,
            "add_check_condition_met": None,
            "add_check_simulation_mode": None,
            "add_attempted": None,
            "add_succeeded": None,
            "add_path": None,
            "add_failure_reason": None,
            "added_config_index_during_selection": None,
            "pool_size_before_add": None,
            "pool_size_after_add": None,
            "bounds_refreshed_after_add": None,
            "added_config_is_random": None,
            "certification_triggered": False,
            "certification_lcb_after_observation": None,
            "certification_threshold_before_observation": None,
            "certification_threshold_after_observation": None,
            "certification_k_before_observation": None,
            "certification_k_after_observation": None,
            "lucb_k_threshold_growth_considered": False,
            "lucb_k_threshold_growth_applied": False,
            "lucb_k_threshold_growth_old_threshold": None,
            "lucb_k_threshold_growth_new_threshold": None,
            "lucb_k_threshold_growth_delta": None,
            "lucb_k_threshold_growth_group_indices": None,
            "lucb_k_threshold_growth_group_means": None,
            "lucb_k_threshold_growth_group_counts": None,
            "lucb_k_threshold_growth_avg_fresh_cost": None,
            "lucb_k_threshold_growth_total_existing_extra_cost": None,
            "lucb_k_threshold_growth_balance_gap": None,
        }
        if optimizer_meta_overrides:
            optimizer_meta.update(optimizer_meta_overrides)

        return {
            "ticket_id": ticket_id,
            "config_index": idx,
            "config": self.configs[idx]["config"],
            "optimizer_meta": optimizer_meta,
        }

    def _update_simulated_state(
        self,
        idx: int,
        simulated_m: Dict[int, int],
        simulated_mean: Dict[int, float],
        simulated_ucb: np.ndarray,
        *extra_ucb_arrays: np.ndarray,
    ) -> None:
        """Update simulated means/counts/bounds after selecting ``idx`` once."""
        old_m = simulated_m[idx]
        new_m = old_m + 1
        original_lcb = float(self.LCB[idx])  # Always use original LCB, not simulated
        simulated_mean[idx] = (old_m * simulated_mean[idx] + original_lcb) / new_m
        simulated_m[idx] = new_m

        new_ucb, _ = self._compute_expected_bounds_after_sample(
            simulated_mean[idx], new_m
        )
        updated_ucb = self._transform_upper(new_ucb)
        simulated_ucb[idx] = updated_ucb
        for ucb_array in extra_ucb_arrays:
            ucb_array[idx] = updated_ucb

    def _update_average_sample_ucb_state(
        self,
        idx: int,
        simulated_m: Dict[int, int],
        average_sample_mean: Dict[int, float],
        average_raw_simulated_ucb: np.ndarray,
    ) -> None:
        """Update one config's average-sample UCB after a simulated sample."""
        mean = float(average_sample_mean.get(idx, self.U_hat[idx]))
        new_m = int(simulated_m.get(idx, self.m.get(idx, 0)))
        new_ucb, _ = self._compute_expected_bounds_after_sample(mean, new_m)
        average_raw_simulated_ucb[idx] = self._transform_upper(new_ucb)

    def _apply_repulsion_bonus_to_scores(
        self,
        scores: np.ndarray,
        candidate_indices: Iterable[int],
        *,
        simulated_m: Optional[Dict[int, int]] = None,
        unsampled_policy: str = "all",
    ) -> np.ndarray:
        """Return a copy of ``scores`` with repulsion bonuses applied in-place."""
        adjusted_scores = np.array(scores, copy=True)

        if self.repulsion_alpha <= 0 or not self.certified_configs:
            return adjusted_scores

        candidate_list = list(candidate_indices)
        if not candidate_list:
            return adjusted_scores

        if simulated_m is None or unsampled_policy == "all":
            eligible_indices = candidate_list
            unsampled_indices: List[int] = []
        else:
            eligible_indices = [
                idx for idx in candidate_list if simulated_m.get(idx, 0) > 0
            ]
            unsampled_indices = [
                idx for idx in candidate_list if simulated_m.get(idx, 0) <= 0
            ]

        if not eligible_indices and unsampled_policy != "max":
            return adjusted_scores

        repulsion = (
            self._compute_repulsion_bonus(eligible_indices)
            if eligible_indices
            else np.zeros(0)
        )
        for bonus, idx in zip(repulsion, eligible_indices):
            adjusted_scores[idx] += float(bonus)

        if unsampled_policy == "max" and unsampled_indices:
            max_bonus = float(np.max(repulsion)) if repulsion.size > 0 else 0.0
            for idx in unsampled_indices:
                adjusted_scores[idx] += max_bonus

        return adjusted_scores

    def _deterministic_choose_max(
        self, scores: np.ndarray, tie_break: np.ndarray
    ) -> int:
        """Choose max score with deterministic tie-breaking for diagnostics."""
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return -1

        max_score = float(np.max(scores[finite_mask]))
        top = np.flatnonzero(finite_mask & np.isclose(scores, max_score))
        if top.size == 1:
            return int(top[0])

        top_tie_break = tie_break[top]
        max_tie_break = float(np.max(top_tie_break))
        top = top[np.isclose(top_tie_break, max_tie_break)]
        return int(np.min(top))

    def _compute_repulsion_components(
        self, config_indices: List[int]
    ) -> Dict[str, Any]:
        """Return detailed repulsion diagnostics for candidate configs."""
        n_candidates = len(config_indices)
        empty = {
            "bonus": np.zeros(n_candidates),
            "weighted_proximity": np.full(n_candidates, np.nan),
            "global_decay": 0.0,
            "active": False,
        }
        if n_candidates == 0:
            return empty

        if not self.certified_configs or self.repulsion_alpha <= 0:
            return empty

        if self.model is None or not self.model.is_fitted():
            return empty

        if not hasattr(self.model, "compute_proximity_matrix"):
            return empty

        candidate_cfgs = [self.configs[i]["config"] for i in config_indices]
        certified_indices = list(self.certified_configs)
        certified_cfgs = [self.configs[i]["config"] for i in certified_indices]
        certified_iters = [self.certified_at_iter.get(i, 0) for i in certified_indices]

        proximity = self.model.compute_proximity_matrix(candidate_cfgs, certified_cfgs)
        ages = np.array([self._current_iter - t for t in certified_iters], dtype=float)
        time_weights = 1.0 / (1.0 + ages / self.repulsion_time_decay)

        total_weight = float(time_weights.sum())
        if total_weight < 1e-10:
            return empty

        weighted_proximity = (proximity * time_weights).sum(axis=1) / total_weight

        most_recent_cert_iter = max(certified_iters)
        global_age = self._current_iter - most_recent_cert_iter
        global_decay = 1.0 / (1.0 + global_age / self.repulsion_time_decay)

        bonus = self.repulsion_alpha * global_decay * (1.0 - weighted_proximity)
        return {
            "bonus": bonus,
            "weighted_proximity": weighted_proximity,
            "global_decay": float(global_decay),
            "active": True,
        }

    def _build_repulsion_selection_metadata(
        self,
        *,
        selected_idx: int,
        candidate_indices: Iterable[int],
        base_scores: np.ndarray,
        adjusted_scores: np.ndarray,
        tie_break: np.ndarray,
        simulated_m: Optional[Dict[int, int]] = None,
        unsampled_policy: str = "all",
        repulsion_applied: bool = True,
    ) -> Dict[str, Any]:
        """Summarize how repulsion affected one selection decision."""
        candidate_list = list(candidate_indices)
        winner_without_repulsion = self._deterministic_choose_max(base_scores, tie_break)
        selected_before = (
            float(base_scores[selected_idx]) if selected_idx < len(base_scores) else None
        )
        selected_after = (
            float(adjusted_scores[selected_idx]) if selected_idx < len(adjusted_scores) else None
        )
        winner_without_score = (
            float(base_scores[winner_without_repulsion])
            if winner_without_repulsion >= 0 and winner_without_repulsion < len(base_scores)
            else None
        )

        if not repulsion_applied:
            return {
                "n_certified_configs": int(len(self.certified_configs)),
                "repulsion_active": False,
                "repulsion_bonus": None,
                "repulsion_bonus_mode": None,
                "repulsion_weighted_proximity": None,
                "repulsion_global_decay": None,
                "selection_score_before_repulsion": selected_before,
                "selection_score_after_repulsion": selected_after,
                "winner_without_repulsion": (
                    int(winner_without_repulsion)
                    if winner_without_repulsion >= 0
                    else None
                ),
                "winner_without_repulsion_score": winner_without_score,
                "repulsion_changed_selection": False,
            }

        if simulated_m is None or unsampled_policy == "all":
            eligible_indices = candidate_list
            unsampled_indices: List[int] = []
        else:
            eligible_indices = [
                idx for idx in candidate_list if simulated_m.get(idx, 0) > 0
            ]
            unsampled_indices = [
                idx for idx in candidate_list if simulated_m.get(idx, 0) <= 0
            ]

        components = self._compute_repulsion_components(eligible_indices)
        bonus_by_idx = {
            idx: float(bonus)
            for idx, bonus in zip(eligible_indices, components["bonus"])
        }
        max_bonus = float(np.max(components["bonus"])) if eligible_indices else 0.0

        selected_bonus = 0.0
        bonus_mode = None
        if selected_idx in bonus_by_idx:
            selected_bonus = bonus_by_idx[selected_idx]
            bonus_mode = "direct"
        elif unsampled_policy == "max" and selected_idx in unsampled_indices:
            selected_bonus = max_bonus
            bonus_mode = "unsampled_max"

        selected_components = self._compute_repulsion_components([selected_idx])
        weighted_proximity = None
        if np.isfinite(selected_components["weighted_proximity"]).any():
            weighted_proximity = float(selected_components["weighted_proximity"][0])

        global_decay = (
            float(selected_components["global_decay"])
            if selected_components["active"]
            else None
        )

        repulsion_active = bool(
            self.repulsion_alpha > 0
            and self.certified_configs
            and selected_components["active"]
        )

        return {
            "n_certified_configs": int(len(self.certified_configs)),
            "repulsion_active": repulsion_active,
            "repulsion_bonus": (
                float(selected_bonus) if repulsion_active or bonus_mode is not None else None
            ),
            "repulsion_bonus_mode": bonus_mode,
            "repulsion_weighted_proximity": weighted_proximity,
            "repulsion_global_decay": global_decay,
            "selection_score_before_repulsion": selected_before,
            "selection_score_after_repulsion": selected_after,
            "winner_without_repulsion": (
                int(winner_without_repulsion)
                if winner_without_repulsion >= 0
                else None
            ),
            "winner_without_repulsion_score": winner_without_score,
            "repulsion_changed_selection": (
                bool(winner_without_repulsion != selected_idx)
                if winner_without_repulsion >= 0
                else None
            ),
        }

    def _maybe_add_config_during_phase2(
        self,
        simulated_m: Dict[int, int],
        simulated_mean: Dict[int, float],
        average_sample_mean: Dict[int, float],
        average_raw_simulated_ucb: np.ndarray,
        raw_simulated_ucb: np.ndarray,
        simulated_ucb: np.ndarray,
        batch_counts: Dict[int, int],
        ucb_selection_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run COUP add-config check and update simulated state when config is added."""
        if not simulated_m:
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                {},
            )
        simulated_epsilon_prime = self._compute_simulated_epsilon_prime(
            average_raw_simulated_ucb
        )
        eps_sq = simulated_epsilon_prime**2
        gamma_e = self.gamma * self.exploration_param
        min_m = min(simulated_m.values())
        condition_met = (eps_sq < gamma_e) and (min_m > 0)
        add_check_meta: Dict[str, Any] = {
            "add_check_epsilon_prime": float(simulated_epsilon_prime),
            "add_check_eps_sq": float(eps_sq),
            "add_check_gamma_e": float(gamma_e),
            "add_check_min_m": int(min_m),
            "add_check_condition_met": bool(condition_met),
            "add_check_simulation_mode": "average",
            "add_attempted": False,
            "add_succeeded": False,
            "add_path": None,
            "add_failure_reason": None,
            "added_config_index_during_selection": None,
        }

        logger.debug(
            (
                "[COUP DEBUG] add_config check (phase2 step): "
                "eps²=%.6f, gamma*e=%.6f, min_m=%d, condition=%s"
            ),
            eps_sq,
            gamma_e,
            min_m,
            condition_met,
        )

        if not condition_met:
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                add_check_meta,
            )

        simulated_targets = {
            i: float(v) for i, v in simulated_mean.items() if np.isfinite(float(v))
        }
        old_n = self.n
        added = self.add_new_config(simulated_model_targets=simulated_targets)
        add_check_meta.update(self._last_add_attempt_info)
        if not (added and self.n > old_n):
            if add_check_meta.get("add_attempted") and add_check_meta.get("add_failure_reason") is None:
                add_check_meta["add_failure_reason"] = "no_pool_growth"
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                add_check_meta,
            )

        new_idx = self.n - 1
        add_check_meta["add_succeeded"] = True
        add_check_meta["added_config_index_during_selection"] = int(new_idx)
        simulated_m[new_idx] = self.m.get(new_idx, 0)
        simulated_mean[new_idx] = float(self.U_hat[new_idx])
        average_sample_mean[new_idx] = float(self.U_hat[new_idx])
        batch_counts[new_idx] = 0
        average_raw_simulated_ucb = self._compute_average_sample_ucb_values(
            simulated_m,
            average_sample_mean,
        )

        new_sim_ucb = float(
            self.UCB_transformed[new_idx]
            if self.utility_transform
            else self.UCB[new_idx]
        )
        raw_simulated_ucb = np.append(raw_simulated_ucb, new_sim_ucb)
        if new_idx in self.certified_configs:
            new_sim_ucb = -np.inf
        elif self.repulsion_alpha > 0 and self.certified_configs:
            candidate_indices = [
                idx
                for idx, is_selectable in enumerate(ucb_selection_mask)
                if is_selectable and idx != new_idx
            ]
            if simulated_m.get(new_idx, 0) > 0:
                repulsion_bonus = self._compute_repulsion_bonus([new_idx])
                if len(repulsion_bonus) > 0:
                    new_sim_ucb += float(repulsion_bonus[0])
            else:
                sampled_candidates = [
                    idx for idx in candidate_indices if simulated_m.get(idx, 0) > 0
                ]
                if sampled_candidates:
                    max_bonus = float(
                        np.max(self._compute_repulsion_bonus(sampled_candidates))
                    )
                    new_sim_ucb += max_bonus
        simulated_ucb = np.append(simulated_ucb, new_sim_ucb)

        ucb_selection_mask = np.append(ucb_selection_mask, True)
        if new_idx in self.certified_configs:
            ucb_selection_mask[new_idx] = False

        return (
            average_raw_simulated_ucb,
            raw_simulated_ucb,
            simulated_ucb,
            ucb_selection_mask,
            add_check_meta,
        )

    def get_state(self):
        """Return a complete snapshot of COUP state for checkpointing."""
        state = {
            # Core counters
            "n": self.n,
            "n0": self.n0,
            "r": self.r,
            # Config storage and eval counts
            "configs": self.configs,
            "m": self.m,
            # LUCB arrays
            "U_hat": self.U_hat,
            "F_hat": self.F_hat,
            "U_hat_ucb": self.U_hat_ucb,
            "U_hat_lcb": self.U_hat_lcb,
            "F_hat_lcb": self.F_hat_lcb,
            "UCB": self.UCB,
            "LCB": self.LCB,
            "UCB_transformed": self.UCB_transformed,
            "LCB_transformed": self.LCB_transformed,
            # Certified configs
            "certified_configs": self.certified_configs,
            "certified_at_iter": self.certified_at_iter,
            "certified_at_ticket": self.certified_at_ticket,
            "certification_event_meta": self.certification_event_meta,
            "certification_threshold": self.certification_threshold,
            "certification_strategy": self.certification_strategy,
            "repulsion_phase1": self.repulsion_phase1,
            # Tracking arrays
            "wallclock_time": getattr(self, "_wallclock_time", 0.0),
            "i_stars": self.i_stars,
            "epsilon_stars": self.epsilon_stars,
            "epsilon_primes": self.epsilon_primes,
            "gammas": self.gammas,
            "observation_times_cpu": self.observation_times_cpu,
            "observation_times_wallclock": self.observation_times_wallclock,
            "t0": getattr(self, "t0", time.time()),
            "is_from_random_sample": self.is_from_random_sample,
            # Config metadata
            "added_at_iter": self.added_at_iter,
            "_current_iter": self._current_iter,
            "_last_full_bounds_refresh_n": self._last_full_bounds_refresh_n,
            # Ticket system
            "_next_ticket_id": self._next_ticket_id,
        }
        return state

    def save_state(self):
        if self.savepath is None:
            raise ValueError("savepath is not set; cannot save state.")
        with open(self.savepath + ".tmp", "wb") as f:  # non-atomic write to temp file
            pickle.dump(self.get_state(), f)
        os.replace(
            self.savepath + ".tmp", self.savepath
        )  # then atomic rename if successful

    def suggest(
        self, batch_size: int = 2, maybe_add_config: bool = True
    ) -> List[Dict[str, Any]]:
        """Suggest configurations to evaluate, returning optimizer tickets.

        Uses dynamic batch sizing: instead of allocating a fixed number of
        samples per config, this method simulates UCB updates to dynamically
        decide how many samples to allocate to each config. Configs with high
        UCB get more samples until their simulated UCB drops below competitors.

        Parameters
        ----------
        batch_size : int, optional
            Total number of samples to allocate across all configurations.
        maybe_add_config : bool, optional
            If True (default), Phase 2 (UCB allocation) will check COUP's
            add-new-config condition at every allocation step and may call
            :meth:`add_new_config` multiple times during one batch.

        Returns
        -------
        List[Dict[str, Any]]
            Each element has the form::

                {
                    "ticket_id": int,
                    "config_index": int,
                    "config": Any,
                }

            where ``ticket_id`` should be used when reporting utilities to
            :meth:`tell`.
        """

        if self.n == 0:
            return []

        # Branch to LUCB-k strategy if configured
        if self.certification_strategy == "lucb_k":
            return self._suggest_lucb_k(batch_size, maybe_add_config)

        # Train/retrain model for repulsion proximity computation
        if (
            self.model is not None
            and self.repulsion_alpha > 0
            and self.certified_configs
        ):
            n_observed = sum(1 for m in self.m.values() if m > 0)
            if n_observed >= 2:
                self.model.train(self.configs)

        tickets: List[Dict[str, Any]] = []

        # Track simulated sample counts, means, and UCBs during batch construction
        # We simulate future samples as coming from the ORIGINAL LCB (pessimistic assumption)
        simulated_m = {i: self.m.get(i, 0) for i in range(self.n)}
        simulated_mean = {i: float(self.U_hat[i]) for i in range(self.n)}
        average_sample_mean = {i: float(self.U_hat[i]) for i in range(self.n)}
        raw_simulated_ucb = self._get_selection_ucb_values()
        average_raw_simulated_ucb = np.copy(raw_simulated_ucb)
        simulated_ucb = np.copy(raw_simulated_ucb)

        # DEBUG: Log top-5 UCBs at start of suggest
        _top5 = np.argsort(simulated_ucb)[::-1][:5]
        logger.debug(
            "[SUGGEST DEBUG] Top-5 UCBs at suggest start: %s",
            [
                (int(i), f"UCB={simulated_ucb[i]:.6f}", f"m={self.m.get(i,0)}")
                for i in _top5
            ],
        )

        # Exclude certified configs from selection.
        self._mask_indices_inplace(simulated_ucb, self.certified_configs, -np.inf)

        # Add repulsion bonus to encourage exploration away from certified configs
        if self.repulsion_alpha > 0 and self.certified_configs:
            active_indices = [
                i for i in range(self.n) if i not in self.certified_configs
            ]
            simulated_ucb = self._apply_repulsion_bonus_to_scores(
                simulated_ucb,
                active_indices,
                simulated_m=simulated_m,
                unsampled_policy="max",
            )

        # Track how many times each config is selected in this batch
        batch_counts: Dict[int, int] = {i: 0 for i in range(self.n)}

        # Determine incumbent (highest mean) at start of batch
        base_incumbent_scores = np.array(
            [self._transform_upper(simulated_mean[i]) for i in range(self.n)]
        )
        incumbent_scores = np.array(base_incumbent_scores, copy=True)
        if self.repulsion_phase1:
            active_indices = [
                i for i in range(self.n) if i not in self.certified_configs
            ]
            incumbent_scores = self._apply_repulsion_bonus_to_scores(
                incumbent_scores,
                active_indices,
            )

        # Mask out certified configs from incumbent selection
        incumbent_mask = np.ones(self.n, dtype=bool)
        self._mask_indices_inplace(incumbent_mask, self.certified_configs, False)

        if not incumbent_mask.any():
            # No valid configs available
            return tickets

        # Break ties lexicographically (lowest config ID wins)
        masked_incumbent_scores = np.where(incumbent_mask, incumbent_scores, -np.inf)
        masked_base_incumbent_scores = np.where(
            incumbent_mask, base_incumbent_scores, -np.inf
        )
        incumbent_tie_break = -np.arange(self.n, dtype=float)
        incumbent_idx = int(choose_max(masked_incumbent_scores, incumbent_tie_break))
        phase1_selection_meta = self._build_repulsion_selection_metadata(
            selected_idx=incumbent_idx,
            candidate_indices=np.flatnonzero(incumbent_mask),
            base_scores=masked_base_incumbent_scores,
            adjusted_scores=masked_incumbent_scores,
            tie_break=incumbent_tie_break,
            repulsion_applied=self.repulsion_phase1,
        )

        # Split batch into two phases
        n_incumbent_samples = batch_size // 2  # Round down
        n_ucb_samples = batch_size - n_incumbent_samples

        # Phase 1: Select incumbent (highest mean)
        for _ in range(n_incumbent_samples):
            tickets.append(
                self._create_ticket(
                    incumbent_idx,
                    batch_counts,
                    optimizer_meta_overrides={
                        "selection_phase": "phase1",
                        **phase1_selection_meta,
                    },
                )
            )
            batch_counts[incumbent_idx] += 1
            self._update_simulated_state(
                incumbent_idx,
                simulated_m,
                simulated_mean,
                simulated_ucb,
                raw_simulated_ucb,
            )
            self._update_average_sample_ucb_state(
                incumbent_idx,
                simulated_m,
                average_sample_mean,
                average_raw_simulated_ucb,
            )

        # Phase 2: Select from highest UCB (excluding incumbent)
        # Create mask excluding incumbent
        ucb_selection_mask = np.ones(self.n, dtype=bool)
        ucb_selection_mask[incumbent_idx] = False
        self._mask_indices_inplace(ucb_selection_mask, self.certified_configs, False)

        # DEBUG: Log Phase 2 start state
        _masked = np.where(ucb_selection_mask, simulated_ucb, -np.inf)
        _top5_p2 = np.argsort(_masked)[::-1][:5]
        logger.debug(
            "[SUGGEST DEBUG] Phase 2 start. incumbent=%d. Top-5 masked UCBs: %s",
            incumbent_idx,
            [
                (int(i), f"UCB={_masked[i]:.6f}", f"m={simulated_m[i]}")
                for i in _top5_p2
            ],
        )
        _phase2_step = [0]  # mutable counter

        for _ in range(n_ucb_samples):
            if maybe_add_config:
                (
                    average_raw_simulated_ucb,
                    raw_simulated_ucb,
                    simulated_ucb,
                    ucb_selection_mask,
                    add_check_meta,
                ) = self._maybe_add_config_during_phase2(
                    simulated_m=simulated_m,
                    simulated_mean=simulated_mean,
                    average_sample_mean=average_sample_mean,
                    average_raw_simulated_ucb=average_raw_simulated_ucb,
                    raw_simulated_ucb=raw_simulated_ucb,
                    simulated_ucb=simulated_ucb,
                    batch_counts=batch_counts,
                    ucb_selection_mask=ucb_selection_mask,
                )
            else:
                add_check_meta = {
                    "selection_phase": "phase2",
                    "add_attempted": False,
                }

            # Mask out incumbent and certified configs from UCB selection
            masked_ucb = np.where(ucb_selection_mask, simulated_ucb, -np.inf)
            masked_base_ucb = np.where(ucb_selection_mask, raw_simulated_ucb, -np.inf)

            if not np.isfinite(masked_ucb).any():
                # No valid configs available for Phase 2
                break

            # Select config with highest UCB, breaking ties by fewest total samples.
            tie_break = -np.array([simulated_m.get(i, 0) for i in range(self.n)])
            idx = int(choose_max(masked_ucb, tie_break))

            if not np.isfinite(simulated_ucb[idx]):
                break

            # DEBUG: Log each Phase 2 selection
            logger.debug(
                "[SUGGEST DEBUG] Phase2 step %d: selected config %d (UCB=%.6f, m=%d, mask=%s)",
                _phase2_step[0],
                idx,
                simulated_ucb[idx],
                simulated_m[idx],
                ucb_selection_mask[idx],
            )
            _phase2_step[0] += 1

            phase2_meta = {"selection_phase": "phase2"}
            phase2_meta.update(add_check_meta)
            phase2_meta.update(
                self._build_repulsion_selection_metadata(
                    selected_idx=idx,
                    candidate_indices=np.flatnonzero(ucb_selection_mask),
                    base_scores=masked_base_ucb,
                    adjusted_scores=masked_ucb,
                    tie_break=tie_break,
                    simulated_m=simulated_m,
                    unsampled_policy="max",
                )
            )
            tickets.append(
                self._create_ticket(
                    idx,
                    batch_counts,
                    optimizer_meta_overrides=phase2_meta,
                )
            )
            batch_counts[idx] += 1
            self._update_simulated_state(
                idx,
                simulated_m,
                simulated_mean,
                simulated_ucb,
                raw_simulated_ucb,
            )
            self._update_average_sample_ucb_state(
                idx,
                simulated_m,
                average_sample_mean,
                average_raw_simulated_ucb,
            )

        return tickets

    def _get_top_k_indices(self, k: int) -> List[int]:
        """Return the top-k config indices by mean utility, ties broken lexicographically.

        Lexicographic tie-breaking means lowest config index wins.
        """
        if k <= 0 or self.n == 0:
            return []

        mean_scores = np.array(
            [self._transform_upper(float(self.U_hat[i])) for i in range(self.n)]
        )
        # Tie-break by index: prefer lower index (negate index as secondary sort key)
        # argsort with lexicographic: sort by (-mean, +index)
        order = np.lexsort((np.arange(self.n), -mean_scores))
        return list(order[:min(k, self.n)])

    def _suggest_lucb_k(
        self, batch_size: int = 2, maybe_add_config: bool = True
    ) -> List[Dict[str, Any]]:
        """LUCB-k suggestion strategy.

        Instead of removing certified configs, uses k = len(certified_configs)
        to run LUCB-k best-arm identification:

        - k <= 1: standard LUCB-1 (same as no-cert, but configs are NOT removed)
        - k >= 2: find top-k arms by mean, pull weakest (min LCB) from top-k
          in Phase 1, pull strongest challenger (max UCB outside top-k) in Phase 2.
        """
        k = len(self.certified_configs)

        # Train/retrain model for repulsion proximity computation
        if (
            self.model is not None
            and self.repulsion_alpha > 0
            and self.certified_configs
        ):
            n_observed = sum(1 for m in self.m.values() if m > 0)
            if n_observed >= 2:
                self.model.train(self.configs)

        tickets: List[Dict[str, Any]] = []
        batch_counts: Dict[int, int] = {i: 0 for i in range(self.n)}

        simulated_m = {i: self.m.get(i, 0) for i in range(self.n)}
        simulated_mean = {i: float(self.U_hat[i]) for i in range(self.n)}
        average_sample_mean = {i: float(self.U_hat[i]) for i in range(self.n)}
        raw_simulated_ucb = self._get_selection_ucb_values()
        average_raw_simulated_ucb = np.copy(raw_simulated_ucb)
        simulated_ucb = np.copy(raw_simulated_ucb)

        # For LUCB-k, do NOT mask certified configs
        # (repulsion bonus is still applied if enabled)
        if self.repulsion_alpha > 0 and self.certified_configs:
            simulated_ucb = self._apply_repulsion_bonus_to_scores(
                simulated_ucb,
                range(self.n),
                simulated_m=simulated_m,
                unsampled_policy="max",
            )

        if k <= 1:
            # Standard LUCB-1 but without masking certified configs
            # Phase 1: incumbent (highest mean)
            base_incumbent_scores = np.array(
                [self._transform_upper(simulated_mean[i]) for i in range(self.n)]
            )
            incumbent_scores = np.array(base_incumbent_scores, copy=True)
            if self.repulsion_phase1:
                incumbent_scores = self._apply_repulsion_bonus_to_scores(
                    incumbent_scores,
                    range(self.n),
                )
            incumbent_tie_break = -np.arange(self.n, dtype=float)
            incumbent_idx = int(choose_max(incumbent_scores, incumbent_tie_break))
            phase1_selection_meta = self._build_repulsion_selection_metadata(
                selected_idx=incumbent_idx,
                candidate_indices=range(self.n),
                base_scores=base_incumbent_scores,
                adjusted_scores=incumbent_scores,
                tie_break=incumbent_tie_break,
                repulsion_applied=self.repulsion_phase1,
            )

            n_incumbent_samples = batch_size // 2
            n_ucb_samples = batch_size - n_incumbent_samples

            for _ in range(n_incumbent_samples):
                tickets.append(
                    self._create_ticket(
                        incumbent_idx,
                        batch_counts,
                        optimizer_meta_overrides={
                            "selection_phase": "phase1",
                            **phase1_selection_meta,
                        },
                    )
                )
                batch_counts[incumbent_idx] += 1
                self._update_simulated_state(
                    incumbent_idx,
                    simulated_m,
                    simulated_mean,
                    simulated_ucb,
                    raw_simulated_ucb,
                )
                self._update_average_sample_ucb_state(
                    incumbent_idx,
                    simulated_m,
                    average_sample_mean,
                    average_raw_simulated_ucb,
                )

            # Phase 2: highest UCB (excluding incumbent)
            ucb_selection_mask = np.ones(self.n, dtype=bool)
            ucb_selection_mask[incumbent_idx] = False

            for _ in range(n_ucb_samples):
                if maybe_add_config:
                    (
                        average_raw_simulated_ucb,
                        raw_simulated_ucb,
                        simulated_ucb,
                        ucb_selection_mask,
                        add_check_meta,
                    ) = self._maybe_add_config_during_phase2_lucb_k(
                        simulated_m=simulated_m,
                        simulated_mean=simulated_mean,
                        average_sample_mean=average_sample_mean,
                        average_raw_simulated_ucb=average_raw_simulated_ucb,
                        raw_simulated_ucb=raw_simulated_ucb,
                        simulated_ucb=simulated_ucb,
                        batch_counts=batch_counts,
                        ucb_selection_mask=ucb_selection_mask,
                    )
                else:
                    add_check_meta = {
                        "selection_phase": "phase2",
                        "add_attempted": False,
                    }

                masked_ucb = np.where(ucb_selection_mask, simulated_ucb, -np.inf)
                masked_base_ucb = np.where(ucb_selection_mask, raw_simulated_ucb, -np.inf)
                if not np.isfinite(masked_ucb).any():
                    break

                tie_break = -np.array([simulated_m.get(i, 0) for i in range(self.n)])
                idx = int(choose_max(masked_ucb, tie_break))
                if not np.isfinite(simulated_ucb[idx]):
                    break

                phase2_meta = {"selection_phase": "phase2"}
                phase2_meta.update(add_check_meta)
                phase2_meta.update(
                    self._build_repulsion_selection_metadata(
                        selected_idx=idx,
                        candidate_indices=np.flatnonzero(ucb_selection_mask),
                        base_scores=masked_base_ucb,
                        adjusted_scores=masked_ucb,
                        tie_break=tie_break,
                        simulated_m=simulated_m,
                        unsampled_policy="max",
                    )
                )
                tickets.append(
                    self._create_ticket(
                        idx,
                        batch_counts,
                        optimizer_meta_overrides=phase2_meta,
                    )
                )
                batch_counts[idx] += 1
                self._update_simulated_state(
                    idx,
                    simulated_m,
                    simulated_mean,
                    simulated_ucb,
                    raw_simulated_ucb,
                )
                self._update_average_sample_ucb_state(
                    idx,
                    simulated_m,
                    average_sample_mean,
                    average_raw_simulated_ucb,
                )

            return tickets

        # k >= 2: LUCB-k
        top_k = self._get_top_k_indices(k)
        top_k_set = set(top_k)

        n_phase1_samples = batch_size // 2
        n_phase2_samples = batch_size - n_phase1_samples

        # Phase 1: Pull the top-k arm with smallest LCB (weakest confidence)
        lcb_values = self._get_selection_lcb_values()
        top_k_lcbs = np.full(self.n, np.inf)  # inf so non-top-k are never chosen
        for idx in top_k:
            top_k_lcbs[idx] = lcb_values[idx]
        base_phase1_scores = -top_k_lcbs
        phase1_scores = np.array(base_phase1_scores, copy=True)
        if self.repulsion_phase1:
            phase1_scores = self._apply_repulsion_bonus_to_scores(
                phase1_scores,
                top_k,
            )
        # Ties broken by fewest evals
        tie_break_phase1 = -np.array([simulated_m.get(i, 0) for i in range(self.n)])
        weakest_idx = int(choose_max(phase1_scores, tie_break_phase1))
        phase1_selection_meta = self._build_repulsion_selection_metadata(
            selected_idx=weakest_idx,
            candidate_indices=top_k,
            base_scores=base_phase1_scores,
            adjusted_scores=phase1_scores,
            tie_break=tie_break_phase1,
            repulsion_applied=self.repulsion_phase1,
        )

        for _ in range(n_phase1_samples):
            tickets.append(
                self._create_ticket(
                    weakest_idx,
                    batch_counts,
                    optimizer_meta_overrides={
                        "selection_phase": "phase1",
                        **phase1_selection_meta,
                    },
                )
            )
            batch_counts[weakest_idx] += 1
            self._update_simulated_state(
                weakest_idx,
                simulated_m,
                simulated_mean,
                simulated_ucb,
                raw_simulated_ucb,
            )
            self._update_average_sample_ucb_state(
                weakest_idx,
                simulated_m,
                average_sample_mean,
                average_raw_simulated_ucb,
            )

        # Phase 2: Pull the non-top-k arm with highest UCB (strongest challenger)
        ucb_selection_mask = np.ones(self.n, dtype=bool)
        for idx in top_k_set:
            ucb_selection_mask[idx] = False

        for _ in range(n_phase2_samples):
            if maybe_add_config:
                (
                    average_raw_simulated_ucb,
                    raw_simulated_ucb,
                    simulated_ucb,
                    ucb_selection_mask,
                    add_check_meta,
                ) = self._maybe_add_config_during_phase2_lucb_k(
                    simulated_m=simulated_m,
                    simulated_mean=simulated_mean,
                    average_sample_mean=average_sample_mean,
                    average_raw_simulated_ucb=average_raw_simulated_ucb,
                    raw_simulated_ucb=raw_simulated_ucb,
                    simulated_ucb=simulated_ucb,
                    batch_counts=batch_counts,
                    ucb_selection_mask=ucb_selection_mask,
                )
            else:
                add_check_meta = {
                    "selection_phase": "phase2",
                    "add_attempted": False,
                }

            masked_ucb = np.where(ucb_selection_mask, simulated_ucb, -np.inf)
            masked_base_ucb = np.where(ucb_selection_mask, raw_simulated_ucb, -np.inf)
            if not np.isfinite(masked_ucb).any():
                break

            tie_break = -np.array([simulated_m.get(i, 0) for i in range(self.n)])
            idx = int(choose_max(masked_ucb, tie_break))
            if not np.isfinite(simulated_ucb[idx]):
                break

            phase2_meta = {"selection_phase": "phase2"}
            phase2_meta.update(add_check_meta)
            phase2_meta.update(
                self._build_repulsion_selection_metadata(
                    selected_idx=idx,
                    candidate_indices=np.flatnonzero(ucb_selection_mask),
                    base_scores=masked_base_ucb,
                    adjusted_scores=masked_ucb,
                    tie_break=tie_break,
                    simulated_m=simulated_m,
                    unsampled_policy="max",
                )
            )
            tickets.append(
                self._create_ticket(
                    idx,
                    batch_counts,
                    optimizer_meta_overrides=phase2_meta,
                )
            )
            batch_counts[idx] += 1
            self._update_simulated_state(
                idx,
                simulated_m,
                simulated_mean,
                simulated_ucb,
                raw_simulated_ucb,
            )
            self._update_average_sample_ucb_state(
                idx,
                simulated_m,
                average_sample_mean,
                average_raw_simulated_ucb,
            )

        return tickets

    def _maybe_add_config_during_phase2_lucb_k(
        self,
        simulated_m: Dict[int, int],
        simulated_mean: Dict[int, float],
        average_sample_mean: Dict[int, float],
        average_raw_simulated_ucb: np.ndarray,
        raw_simulated_ucb: np.ndarray,
        simulated_ucb: np.ndarray,
        batch_counts: Dict[int, int],
        ucb_selection_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Add-config check for LUCB-k (no certified masking)."""
        if not simulated_m:
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                {},
            )
        simulated_epsilon_prime = self._compute_simulated_epsilon_prime_lucb_k(
            average_raw_simulated_ucb
        )
        eps_sq = simulated_epsilon_prime**2
        gamma_e = self.gamma * self.exploration_param
        min_m = min(simulated_m.values())
        condition_met = (eps_sq < gamma_e) and (min_m > 0)
        add_check_meta: Dict[str, Any] = {
            "add_check_epsilon_prime": float(simulated_epsilon_prime),
            "add_check_eps_sq": float(eps_sq),
            "add_check_gamma_e": float(gamma_e),
            "add_check_min_m": int(min_m),
            "add_check_condition_met": bool(condition_met),
            "add_check_simulation_mode": "average",
            "add_attempted": False,
            "add_succeeded": False,
            "add_path": None,
            "add_failure_reason": None,
            "added_config_index_during_selection": None,
        }

        if not condition_met:
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                add_check_meta,
            )

        simulated_targets = {
            i: float(v) for i, v in simulated_mean.items() if np.isfinite(float(v))
        }
        old_n = self.n
        added = self.add_new_config(simulated_model_targets=simulated_targets)
        add_check_meta.update(self._last_add_attempt_info)
        if not (added and self.n > old_n):
            if add_check_meta.get("add_attempted") and add_check_meta.get("add_failure_reason") is None:
                add_check_meta["add_failure_reason"] = "no_pool_growth"
            return (
                average_raw_simulated_ucb,
                raw_simulated_ucb,
                simulated_ucb,
                ucb_selection_mask,
                add_check_meta,
            )

        new_idx = self.n - 1
        add_check_meta["add_succeeded"] = True
        add_check_meta["added_config_index_during_selection"] = int(new_idx)
        simulated_m[new_idx] = self.m.get(new_idx, 0)
        simulated_mean[new_idx] = float(self.U_hat[new_idx])
        average_sample_mean[new_idx] = float(self.U_hat[new_idx])
        batch_counts[new_idx] = 0
        average_raw_simulated_ucb = self._compute_average_sample_ucb_values(
            simulated_m,
            average_sample_mean,
        )

        new_sim_ucb = float(
            self.UCB_transformed[new_idx]
            if self.utility_transform
            else self.UCB[new_idx]
        )
        raw_simulated_ucb = np.append(raw_simulated_ucb, new_sim_ucb)
        # No certified masking in LUCB-k
        if self.repulsion_alpha > 0 and self.certified_configs:
            candidate_indices = [
                idx
                for idx, is_selectable in enumerate(ucb_selection_mask)
                if is_selectable and idx != new_idx
            ]
            if simulated_m.get(new_idx, 0) > 0:
                repulsion_bonus = self._compute_repulsion_bonus([new_idx])
                if len(repulsion_bonus) > 0:
                    new_sim_ucb += float(repulsion_bonus[0])
            else:
                sampled_candidates = [
                    idx for idx in candidate_indices if simulated_m.get(idx, 0) > 0
                ]
                if sampled_candidates:
                    max_bonus = float(
                        np.max(self._compute_repulsion_bonus(sampled_candidates))
                    )
                    new_sim_ucb += max_bonus
        simulated_ucb = np.append(simulated_ucb, new_sim_ucb)
        ucb_selection_mask = np.append(ucb_selection_mask, True)

        return (
            average_raw_simulated_ucb,
            raw_simulated_ucb,
            simulated_ucb,
            ucb_selection_mask,
            add_check_meta,
        )

    def _compute_simulated_epsilon_prime(
        self, simulated_ucb: np.ndarray, num_bootstrap_samples: int = 10
    ) -> float:
        """Compute epsilon_prime using provided (possibly simulated) UCB values."""
        if simulated_ucb.size == 0:
            return 0.0

        ucb = np.copy(simulated_ucb)
        lcb = self._get_selection_lcb_values()

        # Defensive alignment in case callers pass a truncated/extended vector.
        n = min(len(ucb), len(lcb), self.n)
        if n <= 0:
            return 0.0
        ucb = ucb[:n]
        lcb = lcb[:n]

        self._mask_indices_inplace(ucb, self.certified_configs, -np.inf)
        self._mask_indices_inplace(lcb, self.certified_configs, -np.inf)

        finite_lcb = np.isfinite(lcb)
        finite_ucb = np.isfinite(ucb)
        if not finite_lcb.any() or not finite_ucb.any():
            return 0.0

        max_lcb = float(np.max(lcb[finite_lcb]))

        if self.model is None:
            epsilons = np.ones(num_bootstrap_samples)
            n0_eff = min(self.n0, n)
            for b in range(num_bootstrap_samples):
                random_half = ([True] * math.floor((n - n0_eff) / 2)) + (
                    [False] * math.ceil((n - n0_eff) / 2)
                )
                np.random.shuffle(random_half)
                random_sample = np.array(
                    [True for _ in range(n0_eff)] + list(random_half), dtype=bool
                )
                random_sample = random_sample[:n]
                if random_sample.shape[0] < n:
                    pad = np.ones(n - random_sample.shape[0], dtype=bool)
                    random_sample = np.concatenate([random_sample, pad], axis=0)

                masked_ucb_sample = ucb[random_sample]
                finite_masked = np.isfinite(masked_ucb_sample)
                if not finite_masked.any():
                    epsilons[b] = 0.0
                else:
                    epsilons[b] = float(
                        np.max(masked_ucb_sample[finite_masked]) - max_lcb
                    )
            return float(np.mean(epsilons))

        random_mask = np.asarray(self.is_from_random_sample, dtype=bool)[:n]
        if random_mask.shape[0] < n:
            pad = np.ones(n - random_mask.shape[0], dtype=bool)
            random_mask = np.concatenate([random_mask, pad], axis=0)

        if not random_mask.any():
            return 0.0

        random_ucb = ucb[random_mask]
        finite_random_ucb = np.isfinite(random_ucb)
        if not finite_random_ucb.any():
            return 0.0

        return float(np.max(random_ucb[finite_random_ucb]) - max_lcb)

    def _compute_simulated_epsilon_prime_lucb_k(
        self, simulated_ucb: np.ndarray
    ) -> float:
        """Compute epsilon_prime for LUCB-k: max_UCB_outside_top_k - min_LCB_in_top_k."""
        k = len(self.certified_configs)
        if k <= 1 or simulated_ucb.size == 0:
            # Fall back to standard epsilon_prime (no masking of certified)
            return self._compute_simulated_epsilon_prime_no_mask(simulated_ucb)

        top_k = self._get_top_k_indices(k)
        top_k_set = set(top_k)

        ucb = np.copy(simulated_ucb)
        lcb = self._get_selection_lcb_values()
        n = min(len(ucb), len(lcb), self.n)
        if n <= 0:
            return 0.0
        ucb = ucb[:n]
        lcb = lcb[:n]

        # min LCB in top-k
        top_k_lcbs = [lcb[i] for i in top_k if i < n and np.isfinite(lcb[i])]
        if not top_k_lcbs:
            return 0.0
        min_lcb_top_k = min(top_k_lcbs)

        # max UCB outside top-k
        outside_ucbs = [ucb[i] for i in range(n) if i not in top_k_set and np.isfinite(ucb[i])]
        if not outside_ucbs:
            return 0.0
        max_ucb_outside = max(outside_ucbs)

        return max(0.0, float(max_ucb_outside - min_lcb_top_k))

    def _compute_simulated_epsilon_prime_no_mask(
        self, simulated_ucb: np.ndarray, num_bootstrap_samples: int = 10
    ) -> float:
        """Compute epsilon_prime without masking certified configs (for LUCB-k k<=1)."""
        if simulated_ucb.size == 0:
            return 0.0

        ucb = np.copy(simulated_ucb)
        lcb = self._get_selection_lcb_values()

        n = min(len(ucb), len(lcb), self.n)
        if n <= 0:
            return 0.0
        ucb = ucb[:n]
        lcb = lcb[:n]

        # No masking of certified configs
        finite_lcb = np.isfinite(lcb)
        finite_ucb = np.isfinite(ucb)
        if not finite_lcb.any() or not finite_ucb.any():
            return 0.0

        max_lcb = float(np.max(lcb[finite_lcb]))

        if self.model is None:
            epsilons = np.ones(num_bootstrap_samples)
            n0_eff = min(self.n0, n)
            for b in range(num_bootstrap_samples):
                random_half = ([True] * math.floor((n - n0_eff) / 2)) + (
                    [False] * math.ceil((n - n0_eff) / 2)
                )
                np.random.shuffle(random_half)
                random_sample = np.array(
                    [True for _ in range(n0_eff)] + list(random_half), dtype=bool
                )
                random_sample = random_sample[:n]
                if random_sample.shape[0] < n:
                    pad = np.ones(n - random_sample.shape[0], dtype=bool)
                    random_sample = np.concatenate([random_sample, pad], axis=0)

                masked_ucb_sample = ucb[random_sample]
                finite_masked = np.isfinite(masked_ucb_sample)
                if not finite_masked.any():
                    epsilons[b] = 0.0
                else:
                    epsilons[b] = float(
                        np.max(masked_ucb_sample[finite_masked]) - max_lcb
                    )
            return float(np.mean(epsilons))

        random_mask = np.asarray(self.is_from_random_sample, dtype=bool)[:n]
        if random_mask.shape[0] < n:
            pad = np.ones(n - random_mask.shape[0], dtype=bool)
            random_mask = np.concatenate([random_mask, pad], axis=0)

        if not random_mask.any():
            return 0.0

        random_ucb = ucb[random_mask]
        finite_random_ucb = np.isfinite(random_ucb)
        if not finite_random_ucb.any():
            return 0.0

        return float(np.max(random_ucb[finite_random_ucb]) - max_lcb)

    def _compute_expected_bounds_after_sample(
        self, mean: float, new_m: int
    ) -> Tuple[float, float]:
        """Compute the expected UCB and LCB given a mean and sample count.

        Parameters
        ----------
        mean : float
            The simulated mean utility.
        new_m : int
            Simulated sample count (must be >= 1).

        Returns
        -------
        Tuple[float, float]
            (ucb, lcb) - the expected bounds with new_m samples at the given mean.
        """
        if new_m < 1:
            return 1.0, 0.0  # No samples yet, maximum uncertainty

        # Use same delta_i formula as update_bounds
        delta_i = self.delta / (26.71 * self.n**2 * new_m**2)
        a = math.log(1 / delta_i) / new_m

        # Use the given mean estimate
        mean_prime = max(mean, 0)
        ucb = self.dkl_ucb(mean_prime, a)
        lcb = self.dkl_lcb(mean_prime, a)
        return ucb, lcb

    def _refresh_bounds_after_pool_change(self) -> None:
        """Recompute bounds for sufficiently-sampled configs after ``n`` changes."""
        for idx in range(self.n):
            if self.m.get(idx, 0) >= self.POOL_GROWTH_REFRESH_MIN_SAMPLES:
                self.update_bounds(idx)
        self._last_full_bounds_refresh_n = max(1, self.n)

    def _should_refresh_bounds_after_pool_change(self) -> bool:
        """Return whether pool growth is large enough to justify a full refresh."""
        last_refresh_n = max(1, int(getattr(self, "_last_full_bounds_refresh_n", 1)))
        return self.n >= math.ceil(last_refresh_n * self.POOL_GROWTH_REFRESH_FACTOR)

    def _compute_repulsion_bonus(self, config_indices: List[int]) -> np.ndarray:
        """Compute repulsion bonus for candidate configs.

        Bonus = α * (1 - avg_weighted_proximity) where avg_proximity accounts
        for time-weighted proximity to each certified config.

        Configs far from certified configs get higher bonus, encouraging
        exploration of under-sampled regions of configuration space.

        Parameters
        ----------
        config_indices : List[int]
            Indices of candidate configs to compute bonus for.

        Returns
        -------
        np.ndarray
            Repulsion bonus for each candidate config, shape (len(config_indices),).
        """
        return self._compute_repulsion_components(config_indices)["bonus"]

    def _compute_average_sample_ucb_values(
        self,
        simulated_m: Dict[int, int],
        average_sample_mean: Dict[int, float],
    ) -> np.ndarray:
        """Build simulated UCBs assuming future samples equal the current mean."""
        avg_ucb = np.ones(self.n, dtype=float)
        for idx in range(self.n):
            mean = float(average_sample_mean.get(idx, self.U_hat[idx]))
            m_i = int(simulated_m.get(idx, self.m.get(idx, 0)))
            ucb, _ = self._compute_expected_bounds_after_sample(mean, m_i)
            avg_ucb[idx] = self._transform_upper(ucb)
        return avg_ucb

    # ------------------------------------------------------------------
    # Ask / tell style API
    # ------------------------------------------------------------------

    def tell(self, results: List[Dict[str, float]]) -> None:
        """Update COUP with observed utilities keyed by ticket IDs.

        Parameters
        ----------
        results : List[Dict[str, float]]
            Each dict should contain::

                {"ticket_id": int, "utility": float}

            where ``ticket_id`` comes from a previous call to :meth:`ask` and
            ``utility`` is the observed scalar utility for that evaluation.
        """

        self._ticket_outcome_meta = {}

        for res in results:
            ticket_id = int(res["ticket_id"])
            u_val = float(res["utility"])

            if ticket_id not in self._tickets:
                # Unknown or duplicate ticket; ignore quietly for now.
                logger.debug(
                    "[TELL DEBUG] ticket_id=%d NOT in _tickets (u=%.6f). Skipping.",
                    ticket_id,
                    u_val,
                )
                continue

            i, m_i = self._tickets.pop(ticket_id)

            if "utilities" not in self.configs[i]:
                self.configs[i]["utilities"] = {}

            self.configs[i]["utilities"][m_i] = u_val

            # Recompute m and U_hat from stored utilities to handle out-of-order batch processing
            # This is more robust than the incremental update which assumed sequential order
            utilities = self.configs[i]["utilities"]
            n_evals = len(utilities)
            self.m[i] = n_evals
            self.F_hat[i] = 1.0  # all evaluations are "completed" in this view
            self.U_hat[i] = sum(utilities.values()) / n_evals if n_evals > 0 else 0.0
            outcome_meta = self.update_bounds(i, ticket_id=ticket_id)
            self._ticket_outcome_meta[ticket_id] = outcome_meta

            # DEBUG: Log what was stored
            logger.debug(
                "[TELL DEBUG] ticket=%d -> config %d eval_idx=%d: u_val=%.6f, new_m=%d, new_U_hat=%.6f, new_UCB=%.6f",
                ticket_id,
                i,
                m_i,
                u_val,
                n_evals,
                self.U_hat[i],
                self.UCB[i],
            )

    def update_bounds(self, i, ticket_id: Optional[int] = None) -> Dict[str, Any]:
        # In this generic variant we keep the same functional form for
        # delta_i but drop any dependence on runtime caps. Callers are
        # expected to normalize utilities to [0, 1] if they want the
        # original theoretical guarantees to apply directly.
        delta_i = self.delta / (26.71 * self.n**2 * self.m[i] ** 2)
        a = math.log(1 / delta_i) / self.m[i]

        self.F_hat_lcb[i] = self.dkl_lcb(self.F_hat[i], a)

        # Assume utilities have been scaled into [0, 1] space.
        U_hat_prime = np.maximum(self.U_hat[i], 0)
        U_hat_ucb_prime = self.dkl_ucb(U_hat_prime, a)
        self.U_hat_ucb[i] = U_hat_ucb_prime

        U_hat_lcb_prime = self.dkl_lcb(U_hat_prime, a)
        self.U_hat_lcb[i] = U_hat_lcb_prime
        self.UCB[i] = self.U_hat_ucb[i]
        self.LCB[i] = self.U_hat_lcb[i]

        # Compute transformed bounds if transform is set
        if self.utility_transform is not None:
            lcb_t, ucb_t = self.utility_transform(
                float(self.LCB[i]), float(self.UCB[i])
            )
            self.LCB_transformed[i] = lcb_t
            self.UCB_transformed[i] = ucb_t
        else:
            self.LCB_transformed[i] = self.LCB[i]
            self.UCB_transformed[i] = self.UCB[i]

        outcome_meta: Dict[str, Any] = {
            "certification_triggered": False,
            "certification_lcb_after_observation": float(self.LCB_transformed[i]),
            "certification_threshold_before_observation": (
                float(self.certification_threshold)
                if self.certification_threshold is not None
                else None
            ),
            "certification_threshold_after_observation": (
                float(self.certification_threshold)
                if self.certification_threshold is not None
                else None
            ),
            "certification_k_before_observation": int(len(self.certified_configs)),
            "certification_k_after_observation": int(len(self.certified_configs)),
            "lucb_k_threshold_growth_considered": False,
            "lucb_k_threshold_growth_applied": False,
            "lucb_k_threshold_growth_old_threshold": None,
            "lucb_k_threshold_growth_new_threshold": None,
            "lucb_k_threshold_growth_delta": None,
            "lucb_k_threshold_growth_group_indices": None,
            "lucb_k_threshold_growth_group_means": None,
            "lucb_k_threshold_growth_group_counts": None,
            "lucb_k_threshold_growth_avg_fresh_cost": None,
            "lucb_k_threshold_growth_total_existing_extra_cost": None,
            "lucb_k_threshold_growth_balance_gap": None,
        }

        if self.certification_threshold is None:
            return outcome_meta

        threshold = float(self.certification_threshold)
        if self.LCB_transformed[i] < threshold or i in self.certified_configs:
            return outcome_meta

        old_k = len(self.certified_configs)
        active_group = []
        if self.certification_strategy == "lucb_k" and old_k > 0:
            active_group = self._get_top_k_indices(old_k)

        self.certified_configs.add(i)
        self.certified_at_iter[i] = self._current_iter  # Track when certified
        if ticket_id is not None:
            self.certified_at_ticket[i] = int(ticket_id)

        outcome_meta["certification_triggered"] = True
        outcome_meta["certification_k_after_observation"] = int(len(self.certified_configs))

        if self.certification_strategy != "lucb_k":
            self.certification_event_meta[i] = dict(outcome_meta)
            return outcome_meta

        if old_k < 2:
            self.certification_event_meta[i] = dict(outcome_meta)
            return outcome_meta

        new_threshold = self._compute_lucb_k_threshold_after_expansion(
            active_group,
            threshold,
        )
        outcome_meta.update(
            self._build_lucb_k_threshold_growth_metadata(
                active_group,
                threshold,
                new_threshold,
            )
        )
        if new_threshold > threshold + 1e-12:
            self.certification_threshold = float(new_threshold)
            outcome_meta["certification_threshold_after_observation"] = float(
                self.certification_threshold
            )

        self.certification_event_meta[i] = dict(outcome_meta)
        return outcome_meta

    def update_output(self):
        self.i_stars.append(self.i_star)
        self.epsilon_stars.append(self.epsilon_star)
        self.epsilon_primes.append(float(self.epsilon_prime))
        self.gammas.append(self.gamma)

    def add_new_config(
        self,
        num_best_configs: int = 10,
        num_random_configs: int = 1000,
        simulated_model_targets: Optional[Dict[int, float]] = None,
    ) -> bool:
        """Add a new configuration to the search.

        This generic version does *not* depend on any solver-specific
        configspace. Instead, it uses one of the following mechanisms:

        1. If ``self.config_sampler`` is provided, call it to obtain a new
           configuration.
        2. Else, if a model is available and there is enough data, perform a
           simple model-based selection over candidate configurations provided
           by ``model_candidate_sampler`` (or ``config_sampler`` if absent).
        3. Else, fall back to reusing an existing configuration (e.g. sampled
           uniformly at random). Callers are encouraged to provide a sampler
           for true exploration.

        Parameters
        ----------
        simulated_model_targets : Optional[Dict[int, float]], optional
            Optional per-config training targets (e.g., simulated means)
            to use when fitting the surrogate before model-based proposal.

        Returns
        -------
        bool
            True if a new config was added, False if space is exhausted.
        """
        new_config = None
        is_random = True
        add_path = None
        attempt_info: Dict[str, Any] = {
            "add_attempted": True,
            "add_succeeded": False,
            "add_path": None,
            "add_failure_reason": None,
            "added_config_index_during_selection": None,
            "pool_size_before_add": int(self.n),
            "pool_size_after_add": int(self.n),
            "bounds_refreshed_after_add": False,
            "added_config_is_random": None,
        }

        if self.config_sampler is None and not self.usemodel:
            # Degenerate fallback: reuse one of the existing configs at random.
            idx = np.random.choice(list(self.configs.keys()))
            new_config = self.configs[idx]["config"]
            is_random = True
            add_path = "reuse_existing"
        elif self.config_sampler is not None and (
            not self.usemodel or self.model is None or self.n < self.n0
        ):
            # Pure sampler-based new config.
            new_config = self.config_sampler()
            is_random = True
            add_path = "random"
        elif np.random.random() < self.random_exploration_prob:
            # Random exploration: occasionally sample randomly even when model is active
            # This maintains exploration guarantees for epsilon_prime and gamma
            if self.config_sampler is not None:
                new_config = self.config_sampler()
            else:
                idx = np.random.choice(list(self.configs.keys()))
                new_config = self.configs[idx]["config"]
            is_random = True
            add_path = "random"
        else:
            # Model-based proposal, assuming config_sampler can generate
            # candidate configs when called with an integer argument.
            if self.model is None:
                raise ValueError(
                    "Model is None but usemodel=True. Pass explicit model to COUP.__init__"
                )
            self.model.train(self.configs, simulated_model_targets)

            # Build candidate pool without consuming unique-add sampler state.
            random_candidates: List[Any] = []
            candidate_hashes: set = set()
            sampler = self.model_candidate_sampler or self.config_sampler
            if sampler is not None:
                max_attempts = max(10 * num_random_configs, num_random_configs)
                attempts = 0
                while (
                    len(random_candidates) < num_random_configs
                    and attempts < max_attempts
                ):
                    attempts += 1
                    cfg = sampler()
                    if cfg is None:
                        continue
                    cfg_hash = self._config_hash(cfg)
                    if cfg_hash in self._config_hashes or cfg_hash in candidate_hashes:
                        continue
                    candidate_hashes.add(cfg_hash)
                    random_candidates.append(cfg)
            else:
                # best effort: reuse existing configs with noise
                random_candidates = [
                    self.configs[i]["config"]
                    for i in np.random.choice(
                        list(self.configs.keys()), size=num_random_configs, replace=True
                    )
                ]

            if not random_candidates:
                # No candidates available (space exhausted)
                attempt_info["add_path"] = "model"
                attempt_info["add_failure_reason"] = "no_candidates"
                self._last_add_attempt_info = attempt_info
                return False

            # evaluate with model and select the best predicted utility
            if self.model is None:
                raise ValueError("Model is None but usemodel=True")
            candidate_predictions = self.model.predict(random_candidates)
            best_prediction = int(np.argmax(candidate_predictions))
            new_config = random_candidates[best_prediction]
            is_random = False
            add_path = "model"

        # Check if sampler returned None (space exhausted)
        if new_config is None:
            attempt_info["add_path"] = add_path
            attempt_info["add_failure_reason"] = "sampler_exhausted"
            self._last_add_attempt_info = attempt_info
            return False

        cfg_hash = self._config_hash(new_config)
        if cfg_hash in self._config_hashes:
            attempt_info["add_path"] = add_path
            attempt_info["add_failure_reason"] = "duplicate_config"
            self._last_add_attempt_info = attempt_info
            return False

        # register the new configuration
        self.configs[self.n] = {
            "config": new_config,
            "utilities": {},
            "model_selection_index": 0.0,
        }
        self.m[self.n] = 0
        self.added_at_iter[self.n] = self._current_iter

        # Determine if this config came from random sampling or model
        # True if from sampler without model, False if model-based
        self.is_from_random_sample = np.append(self.is_from_random_sample, is_random)

        # expand LUCB arrays
        self.U_hat = np.append(self.U_hat, self.U_hat[0] if self.n > 0 else 0.0)
        self.F_hat = np.append(self.F_hat, 0)
        self.UCB = np.append(self.UCB, 1)
        self.LCB = np.append(self.LCB, 0)
        self.U_hat_ucb = np.append(self.U_hat_ucb, 1)
        self.U_hat_lcb = np.append(
            self.U_hat_lcb, self.U_hat_lcb[0] if self.n > 0 else 0.0
        )
        self.F_hat_lcb = np.append(self.F_hat_lcb, 0)

        # Expand transformed bounds arrays
        self.UCB_transformed = np.append(self.UCB_transformed, 1)
        self.LCB_transformed = np.append(self.LCB_transformed, 0)

        self.n += 1
        refresh_after_add = self._should_refresh_bounds_after_pool_change()
        if refresh_after_add:
            self._refresh_bounds_after_pool_change()

        self._config_hashes.add(cfg_hash)
        if self.on_config_added is not None:
            try:
                self.on_config_added(new_config)
            except Exception:
                pass

        attempt_info["add_path"] = add_path
        attempt_info["add_succeeded"] = True
        attempt_info["added_config_index_during_selection"] = int(self.n - 1)
        attempt_info["pool_size_after_add"] = int(self.n)
        attempt_info["bounds_refreshed_after_add"] = bool(refresh_after_add)
        attempt_info["added_config_is_random"] = bool(is_random)
        self._last_add_attempt_info = attempt_info
        return True

    def d(self, p, q):
        """KL-divergence term"""
        if p == 0:
            return math.log2(1 / (1 - q))
        elif p == 1:
            return math.log2(1 / q)
        else:
            return p * math.log2(p / q) + (1 - p) * math.log2((1 - p) / (1 - q))

    def dkl_ucb(self, mu_hat, a, eps=1e-8, numpts=1000):
        """Numerically solve for DKL-based upper bound"""
        if mu_hat >= 1 - eps:
            return 1
        xs = np.linspace(mu_hat, 1 - eps, numpts)
        dkls = np.array([self.d(mu_hat, x) for x in xs])
        ucbs = xs[dkls < a]
        return ucbs[-1]

    def dkl_lcb(self, mu_hat, a, eps=1e-8, numpts=1000):
        """Numerically solve for DKL-based lower bound"""
        if mu_hat <= eps:
            return 0
        xs = np.linspace(eps, mu_hat, numpts)
        dkls = np.array([self.d(mu_hat, x) for x in xs])
        lcbs = xs[dkls <= a]
        return lcbs[0]

    def message(self):
        out = "COUP: "
        out += f"r={self.r}. n={self.n}, gamma={self.gamma:.3f}, "
        out += f"i_star={self.i_star:5}, epsilon_star={self.epsilon_star:.3f}, epsilon_prime={self.epsilon_prime:.3f}, "
        out += f"add_cond=[{self.epsilon_star**2:.4f} < {self.gamma * (1 - np.max(self.UCB)):.4f}], "
        out += f"ucb=[{np.min(self.UCB):.3f}, {np.max(self.UCB):.3f}], lcb=[{np.min(self.LCB):.3f}, {np.max(self.LCB):.3f}], "
        out += f"U_hat=[{np.min(self.U_hat):.3f}, {np.max(self.U_hat):.3f}], F_hat=[{np.min(self.F_hat):.3f}, {np.max(self.F_hat):.3f}], "
        return out

    def get_configuration(self, i):
        return self.configs[i]["config"]

    @property
    def hinge_point(self) -> Optional[float]:
        """Backwards-compatible alias for certification_threshold."""
        return self.certification_threshold

    @hinge_point.setter
    def hinge_point(self, value: Optional[float]) -> None:
        self.certification_threshold = value

    def get_certified_configs(self) -> List[Dict[str, Any]]:
        """Return all certified configs with their params, mean, and bounds.

        A config is certified when its LCB_transformed >= certification_threshold,
        meaning we are confident it exceeds the certification threshold.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains:
                - config_index: int
                - config: the configuration parameters
                - mean_utility: float (U_hat)
                - lcb: float (LCB_transformed)
                - ucb: float (UCB_transformed)
                - n_evals: int (number of evaluations)
        """
        results = []
        for i in sorted(self.certified_configs):
            results.append(
                {
                    "config_index": i,
                    "config": self.configs[i]["config"],
                    "mean_utility": float(self.U_hat[i]),
                    "lcb": float(self.LCB_transformed[i]),
                    "ucb": float(self.UCB_transformed[i]),
                    "n_evals": int(self.m.get(i, 0)),
                    "certified_at_iter": self.certified_at_iter.get(i),
                    "certified_at_ticket": self.certified_at_ticket.get(i),
                    **self.certification_event_meta.get(i, {}),
                }
            )
        return results

    def guarantee(self):
        """return the best guarantee COUP can make so far"""
        xmin = np.argmin(np.array(self.epsilon_stars))
        return {
            "epsilon": self.epsilon_stars[xmin],
            "gamma": self.gammas[xmin],
            "i_star": self.i_stars[xmin],
        }

    @property
    def i_star(self):
        lcb = self.LCB_transformed if self.utility_transform else self.LCB
        return choose_max(lcb, self.U_hat)

    @property
    def epsilon_star(self):
        ucb = self._get_selection_ucb_values()
        lcb = self._get_selection_lcb_values()

        if self.certification_strategy == "lucb_k":
            k = len(self.certified_configs)
            if k >= 2:
                # LUCB-k epsilon: max_UCB_outside_top_k - min_LCB_in_top_k
                top_k = self._get_top_k_indices(k)
                top_k_set = set(top_k)
                n = min(len(ucb), len(lcb), self.n)

                top_k_lcbs = [lcb[i] for i in top_k if i < n and np.isfinite(lcb[i])]
                outside_ucbs = [ucb[i] for i in range(n) if i not in top_k_set and np.isfinite(ucb[i])]

                if not top_k_lcbs or not outside_ucbs:
                    return 0.0
                return max(0.0, float(max(outside_ucbs) - min(top_k_lcbs)))
            else:
                # k <= 1: standard epsilon without masking
                if not np.isfinite(ucb).any():
                    return 0.0
                return float(np.max(ucb) - np.max(lcb))

        # Standard "remove" strategy: mask certified configs
        self._mask_indices_inplace(ucb, self.certified_configs, -np.inf)
        self._mask_indices_inplace(lcb, self.certified_configs, -np.inf)

        # If all masked, return 0 or safe default
        if not np.isfinite(ucb).any():
            return 0.0

        return np.max(ucb) - np.max(lcb)

    @property
    def epsilon_prime(self, num_bootstrap_samples=10):
        ucb = self._get_selection_ucb_values()
        if self.certification_strategy == "lucb_k":
            return self._compute_simulated_epsilon_prime_lucb_k(ucb)
        return self._compute_simulated_epsilon_prime(
            ucb, num_bootstrap_samples=num_bootstrap_samples
        )

    @property
    def gamma(self):
        if self.model is None:
            return np.log(math.pi**2 * self.n**2 / 3 / self.delta) / self.n
        else:
            n_random = (
                math.floor((self.n - self.n0) / 2) + self.n0
            )  # number of random sampled configs
            return np.log(math.pi**2 * n_random**2 / 3 / self.delta) / n_random

    @property
    def gamma_full(self):
        return np.log(math.pi**2 * self.n**2 / 3 / self.delta) / self.n

    @property
    def wallclock_time(self):
        t1 = time.time()
        self._wallclock_time += t1 - self.t0
        self.t0 = t1
        return self._wallclock_time
