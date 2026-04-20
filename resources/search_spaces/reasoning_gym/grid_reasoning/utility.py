"""Grid reasoning-specific utilities.

Utilities for grid reasoning (shortest_path, largest_island) scenarios.
These extend the shared default utilities with grid-specific complexity weighting.
"""

import math

from quickscope.dataflow.utility_registry import register_utility


@register_utility(
    "complexity-weighted-error",
    scenario="grid_reasoning",
    description="Inverse accuracy weighted by grid area complexity (normalized to [0,1])",
)
def grid_complexity_weighted_error(metrics: dict) -> float:
    """Return error rate weighted by grid complexity, normalized to [0,1].

    Prioritizes finding failures on simpler (smaller) grids, which are more
    concerning than failures on complex (larger) grids.

    Grid area ranges from 25 (5x5) to 625 (25x25). Uses log scaling to avoid
    extreme weights, then normalizes so output is always in [0, 1].

    Formula:
        complexity_weight = 1 - (log2(area) - log2(25)) / (log2(625) - log2(25))
        utility = error_rate * (0.5 + 0.5 * complexity_weight)

    This gives:
        - Small grid (5x5): error_rate * 1.0
        - Large grid (25x25): error_rate * 0.5
    """
    if metrics.get("timed_out"):
        return 1.0  # Timeout is not a reasoning failure

    if metrics.get("generation_failure"):
        return 1.0  # Generation failure - return max utility

    acc = metrics.get("accuracy")
    if acc is None:
        acc = 0.0  # Parsing failed - treat as wrong

    error_rate = 1.0 - float(acc)

    # Get grid dimensions (rows/cols are stored in metadata by generator)
    rows = float(metrics.get("rows", 10))
    cols = float(metrics.get("cols", 10))
    grid_area = rows * cols

    # Log-scale complexity normalization
    # grid_area: 25 (5x5) to 625 (25x25)
    # log2(25) ≈ 4.64, log2(625) ≈ 9.29
    min_log = math.log2(25)
    max_log = math.log2(625)
    log_area = math.log2(max(grid_area, 25))

    # complexity_weight: 1.0 for smallest grid, 0.0 for largest
    complexity_weight = 1.0 - (log_area - min_log) / (max_log - min_log)
    complexity_weight = max(0.0, min(1.0, complexity_weight))  # clamp to [0, 1]

    # Scale error by complexity: small grids get full weight, large grids get half
    # This ensures utility stays in [0, 1]
    return error_rate * (0.5 + 0.5 * complexity_weight)
