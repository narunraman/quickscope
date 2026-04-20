"""Metrics and utilities registry for Quickscope.

Provides decorators and functions for registering, discovering, and
validating utility functions scoped by scenario.
"""

import warnings
from typing import Callable, Any


# Global registry for utilities: {scenario: {name: {function, description}}}
_UTILITY_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}


def register_utility(
    name: str | None = None, scenario: str = "default", description: str | None = None
):
    """Decorator to register a utility function.

    Args:
        name: Registry name (default: function name with underscores→hyphens)
        scenario: Scenario this utility belongs to (default, dyval, offline, etc.)
        description: Short description (default: function docstring)

    Example:
        @register_utility("my-error", scenario="my_scenario")
        def my_error(metrics: dict) -> float:
            return 1.0 - metrics["accuracy"]
    """

    def decorator(func: Callable) -> Callable:
        registry_name = name or func.__name__.replace("_", "-")

        if scenario not in _UTILITY_REGISTRY:
            _UTILITY_REGISTRY[scenario] = {}

        _UTILITY_REGISTRY[scenario][registry_name] = {
            "function": func,
            "description": description or (func.__doc__ or "").strip().split("\n")[0],
        }
        return func

    return decorator


def is_offline_scenario(scenario: str) -> bool:
    """Check if a scenario is an offline scenario subdomain.

    A scenario is considered offline if it has a directory at:
    - resources/search_spaces/offline/{scenario}/

    Optionally validates that data/{scenario}/ also exists.
    """
    from . import get_path_registry

    path_registry = get_path_registry()
    project_root = path_registry._project_root

    # Check if search space directory exists for this offline subdomain
    offline_space_dir = (
        path_registry.get_path("configs") / "search_spaces" / "offline" / scenario
    )
    if not offline_space_dir.is_dir():
        return False

    # Cross-validate: check data directory exists (optional but recommended)
    data_dir = project_root / "data" / scenario
    if not data_dir.is_dir():
        # Warn but still consider it offline if search space exists
        warnings.warn(
            f"Offline scenario '{scenario}' has search space but no data directory at {data_dir}",
            UserWarning,
        )

    return True


def get_utility(name: str, scenario: str) -> Callable:
    """Get utility function by name for a specific scenario.

    Fallback order: scenario -> offline (for offline scenarios) -> default.

    Args:
        name: Utility function name
        scenario: Scenario name (e.g., "steer_me", "dyval")

    Raises:
        ValueError: If utility not found for this scenario
    """
    # 1. Check specific scenario first
    if scenario in _UTILITY_REGISTRY and name in _UTILITY_REGISTRY[scenario]:
        return _UTILITY_REGISTRY[scenario][name]["function"]

    # 2. Fall back to "offline" for offline scenario instances
    if is_offline_scenario(scenario):
        if "offline" in _UTILITY_REGISTRY and name in _UTILITY_REGISTRY["offline"]:
            return _UTILITY_REGISTRY["offline"][name]["function"]

    # 3. Fall back to "default" as universal base
    if "default" in _UTILITY_REGISTRY and name in _UTILITY_REGISTRY["default"]:
        return _UTILITY_REGISTRY["default"][name]["function"]

    available = list_utilities(scenario)
    raise ValueError(
        f"Utility '{name}' is not available for scenario '{scenario}'. "
        f"Available utilities for '{scenario}': {list(available.keys())}"
    )


def list_utilities(scenario: str) -> dict[str, str]:
    """List all registered utilities for a specific scenario.

    Includes inherited utilities: default -> offline (if applicable) -> scenario.
    """
    result = {}

    # 1. Add default base utilities first (universal inheritance)
    if "default" in _UTILITY_REGISTRY:
        for name, info in _UTILITY_REGISTRY["default"].items():
            result[name] = info["description"]

    # 2. Add offline base utilities (for offline scenario instances)
    if is_offline_scenario(scenario) and "offline" in _UTILITY_REGISTRY:
        for name, info in _UTILITY_REGISTRY["offline"].items():
            result[name] = info["description"]

    # 3. Add scenario-specific utilities (may override inherited)
    if scenario in _UTILITY_REGISTRY:
        for name, info in _UTILITY_REGISTRY[scenario].items():
            result[name] = info["description"]

    return result


def list_all_utilities() -> dict[str, dict[str, str]]:
    """List all registered utilities grouped by scenario."""
    result = {}
    for scenario, utilities in _UTILITY_REGISTRY.items():
        result[scenario] = {
            name: info["description"] for name, info in utilities.items()
        }
    return result
