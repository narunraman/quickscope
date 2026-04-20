"""Utility function registry for adaptive evaluation.

Utility functions map evaluation metrics to scalar values that guide the search.
The Bayesian optimizer uses these values to decide which scenarios to explore next.

This module re-exports from quickscope.config.registry and dynamically loads
utility modules from resources/search_spaces/ directories.
"""

from pathlib import Path
from ConfigSpace import ConfigurationSpace
import yaml
import importlib.util
import sys

from .path_registry import get_path_registry


# Utilities are now loaded dynamically from resources/search_spaces/ directories
# via the _auto_load_utilities() function below


def _load_utility_module(path: Path) -> None:
    """Dynamically load a utility.py module to register its utilities."""
    if not path.exists():
        return

    # Generate a unique module name based on path
    module_name = f"_dynamic_utility_{path.parent.name}_{id(path)}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


# Auto-load utilities for known scenarios on module import
def _auto_load_utilities():
    """Auto-discover and load utilities from search_spaces directories."""
    path_registry = get_path_registry()
    search_spaces_dir = path_registry.get_path("configs") / "search_spaces"

    if not search_spaces_dir.exists():
        return

    # Load root-level default utilities first (inherited by all scenarios)
    default_utility = search_spaces_dir / "utility.py"
    _load_utility_module(default_utility)

    # Load from top-level directories (e.g., dyval/, reasoning_gym/)
    for dir_path in search_spaces_dir.iterdir():
        if (
            dir_path.is_dir()
            and dir_path.name != "offline"
            and not dir_path.name.startswith("_")
        ):
            utility_path = dir_path / "utility.py"
            _load_utility_module(utility_path)

            # Also load from nested subdirectories (e.g., reasoning_gym/grid_reasoning/)
            for sub_dir in dir_path.iterdir():
                if sub_dir.is_dir() and not sub_dir.name.startswith("_"):
                    sub_utility_path = sub_dir / "utility.py"
                    _load_utility_module(sub_utility_path)

    # Load from offline subdirectories (e.g., offline/steer_me/)
    # First load the offline base utilities (offline/utility.py)
    offline_dir = search_spaces_dir / "offline"
    if offline_dir.exists():
        # Load base offline utilities first (inherited by all offline subdomains)
        base_offline_utility = offline_dir / "utility.py"
        _load_utility_module(base_offline_utility)

        # Then load subdomain-specific utilities
        for dir_path in offline_dir.iterdir():
            if dir_path.is_dir() and not dir_path.name.startswith("_"):
                utility_path = dir_path / "utility.py"
                _load_utility_module(utility_path)


def _load_transform_module(path: Path) -> None:
    """Dynamically load a utility_transform.py module to register its transforms."""
    if not path.exists():
        return

    # Generate a unique module name based on path
    module_name = f"_dynamic_transform_{path.parent.name}_{id(path)}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def _auto_load_transforms():
    """Auto-discover and load transforms from search_spaces directories."""
    path_registry = get_path_registry()
    search_spaces_dir = path_registry.get_path("configs") / "search_spaces"

    if not search_spaces_dir.exists():
        return

    # Load root-level default transforms first (inherited by all scenarios)
    default_transform = search_spaces_dir / "utility_transform.py"
    _load_transform_module(default_transform)

    # Load from top-level directories (e.g., dyval/)
    for dir_path in search_spaces_dir.iterdir():
        if (
            dir_path.is_dir()
            and dir_path.name != "offline"
            and not dir_path.name.startswith("_")
        ):
            transform_path = dir_path / "utility_transform.py"
            _load_transform_module(transform_path)

            # Also load from nested subdirectories (e.g., reasoning_gym/grid_reasoning/)
            for sub_dir in dir_path.iterdir():
                if sub_dir.is_dir() and not sub_dir.name.startswith("_"):
                    sub_transform_path = sub_dir / "utility_transform.py"
                    _load_transform_module(sub_transform_path)

    # Load from offline subdirectories
    offline_dir = search_spaces_dir / "offline"
    if offline_dir.exists():
        base_offline_transform = offline_dir / "utility_transform.py"
        _load_transform_module(base_offline_transform)

        for dir_path in offline_dir.iterdir():
            if dir_path.is_dir() and not dir_path.name.startswith("_"):
                transform_path = dir_path / "utility_transform.py"
                _load_transform_module(transform_path)


# Load utilities and transforms on import
_auto_load_utilities()
_auto_load_transforms()


def load_search_space(name_or_path: str) -> dict | ConfigurationSpace:
    """Load search space from name or file path.

    Args:
        name_or_path: Either a built-in search-space name or a file path.
                     or a path to a YAML/Python file

    Returns:
        Search space dictionary (from YAML) or ConfigurationSpace object (from Python)

    Raises:
        FileNotFoundError: If file not found
        ValueError: If invalid format
    """
    # Try as built-in name first (try both .yaml and .py)
    path_registry = get_path_registry()
    config_dir = path_registry.get_path("configs") / "search_spaces"

    # Try .yaml first, then .py
    builtin_yaml = config_dir / f"{name_or_path}.yaml"
    builtin_py = config_dir / f"{name_or_path}.py"

    if builtin_yaml.exists():
        path = builtin_yaml
    elif builtin_py.exists():
        path = builtin_py
    else:
        # Try as direct path
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Search space not found: {name_or_path}\n"
                f"Tried: {builtin_yaml}, {builtin_py}, and {path}"
            )

    # Load based on file extension
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r") as f:
            space = yaml.safe_load(f)

        if not isinstance(space, dict):
            raise ValueError(f"Invalid search space format in {path}")

        return space

    elif path.suffix == ".py":
        # Load Python module and call get_search_space()
        import importlib.util

        spec = importlib.util.spec_from_file_location("search_space_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load Python module from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_search_space"):
            raise ValueError(
                f"Python search space {path} must define a get_search_space() function"
            )

        space = module.get_search_space()

        # Validate that it returns either dict or ConfigurationSpace
        if not isinstance(space, (dict, ConfigurationSpace)):
            raise ValueError(
                f"get_search_space() must return dict or ConfigurationSpace, "
                f"got {type(space)}"
            )

        return space

    else:
        raise ValueError(
            f"Unsupported search space format: {path.suffix}. Use .yaml or .py"
        )


# Export all
__all__ = [
    "load_search_space",
    "load_cache_config",
]


def load_cache_config(scenario: str) -> dict | None:
    """Load cache configuration from a scenario's space.yaml.

    Args:
        scenario: Scenario name (e.g., 'steer_me')

    Returns:
        Cache config dict with 'enabled' and 'key_fields', or None if not configured.

    Example space.yaml:
        cache:
          enabled: true
          key_fields:
            - scenario_id
            - answer_format
    """
    path_registry = get_path_registry()
    config_dir = path_registry.get_path("configs")

    # Check offline scenario location
    space_path = config_dir / "search_spaces" / "offline" / scenario / "space.yaml"

    if not space_path.exists():
        return None

    with open(space_path, "r") as f:
        space = yaml.safe_load(f)

    if not isinstance(space, dict):
        return None

    cache_config = space.get("cache")
    if not cache_config:
        # Default: enabled with scenario_id as key field
        return {
            "enabled": True,
            "key_fields": ["scenario_id"],
        }

    return cache_config
