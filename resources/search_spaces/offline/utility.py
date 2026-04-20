"""Offline simulator specific utilities.

Utilities specific to offline evaluation scenarios (steer_me, gsm_symbolic, etc.).
Shared utilities (error-rate, hinge-error-rate) are now inherited
from resources/search_spaces/utility.py (scenario="default").
"""

from quickscope.dataflow.utility_registry import register_utility


@register_utility(
    "brier",
    scenario="offline",
    description="Use Brier score for scenarios with probabilistic outputs",
)
def brier_score(metrics: dict) -> float:
    """Return Brier score for default scenarios.

    Higher Brier score means worse calibration, which is what we want to find.

    Timeouts return 0.0 (no calibration data available).
    """
    # Handle timeout - no calibration data available
    if metrics.get("timed_out"):
        return 0.0

    brier = metrics.get("brier_score")
    if brier is None:
        raise ValueError(
            "brier utility requires 'brier_score' in metrics. "
            "Did you enable logprobs and run default metrics?"
        )
    return float(brier)
