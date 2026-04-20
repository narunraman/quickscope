"""Default shared utilities.

Universal utilities registered with scenario="default" that all scenarios
can inherit from (dyval, reasoning_gym, offline, etc.).
"""

from quickscope.dataflow.utility_registry import register_utility


@register_utility(
    "error-rate",
    scenario="default",
    description="1 - accuracy (finds low-accuracy cases)",
)
@register_utility(
    "hinge-error-rate",
    scenario="default",
    description="Error rate with hinge transform for Bernoulli estimation",
)
def error_rate(metrics: dict) -> float:
    """Return 1 - accuracy; missing parses treated as wrong.

    Returns 1-accuracy because the optimizer MAXIMIZES utility, and we want
    to find scenarios with LOW accuracy (where the model struggles).

    - Timeouts return 1.0 (not a reasoning failure)
    - Generation failures return 1.0 (not a reasoning failure)
    - Parse failures return 1.0 (treated as wrong)
    """
    if metrics.get("timed_out"):
        return 1.0

    if metrics.get("generation_failure"):
        return 1.0

    acc = metrics.get("accuracy")
    if acc is None:
        return 1.0
    return 1.0 - float(acc)
