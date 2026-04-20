"""DyVal-specific utilities.

Utilities for DyVal dynamic reasoning evaluation scenarios.
These extend the shared default utilities (error-rate, hinge-error-rate)
with DyVal-specific metrics.
"""

from quickscope.dataflow.utility_registry import register_utility


@register_utility(
    "complexity-weighted-error",
    scenario="dyval",
    description="Error rate weighted by problem complexity (tree size)",
)
def dyval_complexity_weighted_error(metrics: dict) -> float:
    """Return error rate weighted by problem complexity.

    Prioritizes finding failures on simpler problems (which are more concerning)
    by dividing error by complexity. Higher values = worse performance on
    simpler problems.

    For tree-based DAGs (arithmetic, linear_equation, bool_logic, deductive_logic,
    abductive_logic), complexity is approximated by the tree size: num_children^depth.
    """
    if metrics.get("timed_out"):
        return 1.0  # Timeout is not a reasoning failure

    if metrics.get("generation_failure"):
        return 1.0  # Generation failure is not a reasoning failure, don't penalize

    acc = metrics.get("accuracy")
    if acc is None:
        acc = 0.0  # Parsing failed - treat as wrong

    # Get complexity factors for tree-based DAGs
    depth = float(metrics.get("depth", 3))
    num_children = float(metrics.get("num_children_per_node", 2))

    # Approximate tree size: num_children^depth gives rough node count
    # e.g., depth=3, children=2 -> ~8 nodes; depth=6, children=4 -> ~4096 nodes
    complexity = num_children**depth

    # Error weighted inversely by complexity
    # Failures on simple problems (low complexity) get higher utility
    error_rate = 1.0 - float(acc)

    # Scale so errors on simple trees (complexity ~4) are weighted higher
    # than errors on complex trees (complexity ~4096)
    # Using log-scale normalization to avoid extreme weights
    import math

    log_complexity = math.log2(max(complexity, 1.0))
    # max_log_complexity = 12.0  # log2(4096) for depth=6, children=4

    return error_rate * (1 / max(log_complexity, 1.0))


@register_utility(
    "reasoning-depth-error",
    scenario="dyval",
    description="Error rate focused on reasoning chain depth",
)
def dyval_reasoning_depth_error(metrics: dict) -> float:
    """Return error rate with emphasis on shallow reasoning failures.

    Prioritizes finding failures on shallow (simpler) reasoning chains,
    which are more concerning than failures on very deep chains.

    If accuracy is None (parsing failed), treats as wrong (acc=0).
    """
    if metrics.get("timed_out"):
        return 1.0  # Timeout is not a reasoning failure

    if metrics.get("generation_failure"):
        return 1.0  # Generation failure is not a reasoning failure, don't penalize

    acc = metrics.get("accuracy")
    if acc is None:
        acc = 0.0  # Parsing failed - treat as wrong

    depth = float(metrics.get("depth", 3))

    # Error rate
    error_rate = 1.0 - float(acc)

    # Weight inversely by depth: failures at depth 2 weighted higher than depth 6
    return error_rate * (6.0 / max(depth, 1.0))
