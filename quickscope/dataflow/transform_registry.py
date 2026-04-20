"""Utility transform registry for Quickscope.

Provides decorators and functions for registering transforms that map
Bernoulli parameter bounds (θ) to utility bounds for COUP optimization.

Transforms are registered as **factories**: functions that accept optional
kwargs and return the actual transform function. This enables CLI kwargs
like `--transform-kwargs hinge_point=0.8`.

Transforms are scoped by scenario (like utilities), allowing the same name
to have different implementations per scenario.
"""

from typing import Callable, Any, Tuple


# Type aliases for clarity
TransformFn = Callable[[float, float], Tuple[float, float]]
TransformFactory = Callable[..., TransformFn]

# Global registry for transforms: {scenario: {name: {factory, monotonicity}}}
_TRANSFORM_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}


def register_transform(
    name: str | None = None,
    scenario: str = "base",
    monotonicity: str = "increasing",
):
    """Decorator to register a utility transform factory.

    Args:
        name: Registry name (default: function name with underscores→hyphens)
        scenario: Scenario this transform belongs to (e.g., "dyval", "offline", "base")
        monotonicity: "increasing", "decreasing", or "non-monotonic"

    The decorated function should be a **factory** that accepts optional kwargs
    and returns a transform function with signature:
        (lcb: float, ucb: float) -> tuple[float, float]

    Example:
        @register_transform("hinge-error-rate", scenario="dyval", monotonicity="increasing")
        def hinge_transform(hinge_point: float = 0.9):
            def transform(lcb: float, ucb: float) -> tuple[float, float]:
                return (min(lcb, hinge_point), min(ucb, hinge_point))
            return transform
    """

    def decorator(func: TransformFactory) -> TransformFactory:
        registry_name = name or func.__name__.replace("_", "-")

        if scenario not in _TRANSFORM_REGISTRY:
            _TRANSFORM_REGISTRY[scenario] = {}

        _TRANSFORM_REGISTRY[scenario][registry_name] = {
            "factory": func,
            "monotonicity": monotonicity,
        }
        return func

    return decorator


def create_transform(name: str, scenario: str = "base", **kwargs) -> TransformFn | None:
    """Create a configured transform instance by name for a specific scenario.

    Args:
        name: Transform name
        scenario: Scenario to look up transform for (falls back to "default" then "base")
        **kwargs: Arguments to pass to the transform factory

    Returns:
        Configured transform function, or None if not found
    """
    # Check scenario-specific first
    if scenario in _TRANSFORM_REGISTRY and name in _TRANSFORM_REGISTRY[scenario]:
        factory = _TRANSFORM_REGISTRY[scenario][name]["factory"]
        return factory(**kwargs)

    # Fall back to "default" scenario (shared transforms)
    if "default" in _TRANSFORM_REGISTRY and name in _TRANSFORM_REGISTRY["default"]:
        factory = _TRANSFORM_REGISTRY["default"][name]["factory"]
        return factory(**kwargs)

    # Fall back to "base" scenario (legacy)
    if "base" in _TRANSFORM_REGISTRY and name in _TRANSFORM_REGISTRY["base"]:
        factory = _TRANSFORM_REGISTRY["base"][name]["factory"]
        return factory(**kwargs)

    return None


def get_transform(name: str, scenario: str = "base") -> TransformFn | None:
    """Get transform function by name (with defaults).

    This is a convenience wrapper around create_transform() for backwards
    compatibility. It creates a transform with default kwargs.

    Args:
        name: Transform name
        scenario: Scenario to look up transform for

    Returns:
        Transform function with default settings, or None if not found
    """
    return create_transform(name, scenario=scenario)


def list_transforms(scenario: str | None = None) -> dict[str, str]:
    """List all registered transforms with their monotonicity.

    Args:
        scenario: If provided, list transforms for this scenario (including default/base).
                  If None, list all transforms across all scenarios.
    """
    if scenario is None:
        # List all transforms grouped by scenario
        result = {}
        for scen, transforms in _TRANSFORM_REGISTRY.items():
            for name, info in transforms.items():
                key = f"{scen}:{name}" if scen not in ("base", "default") else name
                result[key] = info["monotonicity"]
        return result

    # List transforms for specific scenario (with default/base fallback)
    result = {}

    # Add base transforms first (can be overridden)
    if "base" in _TRANSFORM_REGISTRY:
        for name, info in _TRANSFORM_REGISTRY["base"].items():
            result[name] = info["monotonicity"]

    # Add default transforms (override base)
    if "default" in _TRANSFORM_REGISTRY:
        for name, info in _TRANSFORM_REGISTRY["default"].items():
            result[name] = info["monotonicity"]

    # Add scenario-specific transforms (override all)
    if scenario in _TRANSFORM_REGISTRY:
        for name, info in _TRANSFORM_REGISTRY[scenario].items():
            result[name] = info["monotonicity"]

    return result
