"""Centralized filesystem path registry for Quickscope internals.

The registry consumes the user-editable ``resources/paths.yaml`` file (and
optional environment variable overrides) to ensure all modules resolve
directories in a consistent way. It is primarily intended for maintainers, while
users interact with the YAML file directly.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


_DEFAULT_KEYS: Mapping[str, str] = {
    "logs": "logs",
    "results": "results",
    "configs": "resources",
    "temp": "tmp",
}

_PATHS_ENV_VAR = "QUICKSCOPE_PATHS_FILE"
_ENV_PREFIX = "QUICKSCOPE"


@dataclass(frozen=True)
class RegisteredPath:
    """Container used for debugging and tooling introspection."""

    key: str
    path: Path
    source: str


class PathRegistry:
    """Resolve project directories using resources/paths.yaml and env overrides."""

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        config_file: Optional[Path] = None,
        defaults: Optional[Mapping[str, str]] = None,
        env_prefix: str = _ENV_PREFIX,
    ) -> None:
        self._project_root = (project_root or self._discover_project_root()).resolve()
        self._env_prefix = env_prefix
        self._packaged_config_dir = self._find_packaged_config_dir()
        self._config_file = self._determine_config_file(config_file)
        self._defaults = dict(_DEFAULT_KEYS)
        if defaults:
            self._defaults.update(defaults)

        self._raw_config = self._load_yaml(self._config_file)
        self._paths: Dict[str, RegisteredPath] = {}
        self._lock = threading.Lock()
        self._initialise_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_path(self, key: str, *, create: bool = False) -> Path:
        """Return a resolved path for ``key``. Optionally create the directory."""
        normalised = key.lower()
        if normalised not in self._paths:
            raise KeyError(f"Unknown path key: {key}")
        target = self._paths[normalised].path
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def ensure_dirs(self, keys: Iterable[str]) -> None:
        """Ensure a collection of directories exist."""
        for key in keys:
            self.get_path(key, create=True)

    def config_file(self, name: str) -> Path:
        """Return the resolved path for a configuration file under ``configs``."""
        base = self.get_path("configs", create=False)
        return base / name

    def make_run_directory(self, *, base_key: str = "logs", prefix: str = "run") -> Path:
        """Create and return a timestamped run directory under ``base_key``."""
        base = self.get_path(base_key, create=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with self._lock:
            candidate = base / f"{prefix}_{ts}"
            counter = 1
            while candidate.exists():
                candidate = base / f"{prefix}_{ts}_{counter}"
                counter += 1
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    def register(self, key: str, value: Path, *, source: str = "runtime") -> None:
        """Register or override a path during runtime (primarily for testing)."""
        normalised = key.lower()
        self._paths[normalised] = RegisteredPath(key=normalised, path=value, source=source)

    def describe(self) -> Dict[str, str]:
        """Return a mapping of keys -> string paths for debugging purposes."""
        return {key: str(info.path) for key, info in self._paths.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_paths(self) -> None:
        config_paths = {}
        if isinstance(self._raw_config.get("paths"), Mapping):
            config_paths = {k.lower(): v for k, v in self._raw_config["paths"].items()}

        for key, default in self._defaults.items():
            resolved = self._resolve_value(key, config_paths.get(key, default))
            self._paths[key] = RegisteredPath(
                key=key,
                path=resolved,
                source=self._determine_source(key, config_paths),
            )

    def _resolve_value(self, key: str, value: str) -> Path:
        env_key = f"{self._env_prefix}_{key.upper()}"
        env_override = os.getenv(f"{env_key}_DIR") or os.getenv(f"{env_key}_PATH")
        candidate = env_override or value
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            if key == "configs" and not env_override:
                project_config = (self._project_root / candidate_path).resolve()
                if project_config.exists():
                    return project_config
                if self._packaged_config_dir.exists():
                    return self._packaged_config_dir
            candidate_path = (self._project_root / candidate_path).resolve()
        return candidate_path

    def _determine_source(self, key: str, config_paths: Mapping[str, str]) -> str:
        env_key = f"{self._env_prefix}_{key.upper()}"
        if os.getenv(f"{env_key}_DIR") or os.getenv(f"{env_key}_PATH"):
            return "env"
        if key in config_paths:
            return "config"
        return "default"

    @staticmethod
    def _discover_project_root() -> Path:
        """Find the checkout root, falling back to the current working directory."""
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists() and (parent / "resources").is_dir():
                return parent
        return Path.cwd()

    @staticmethod
    def _find_packaged_config_dir() -> Path:
        """Locate bundled resource files when QuickScope is installed as a package."""
        try:
            return Path(str(resources.files("resources")))
        except (ModuleNotFoundError, TypeError):
            return Path(__file__).resolve().parents[2] / "resources"

    def _determine_config_file(self, provided: Optional[Path]) -> Path:
        env_override = os.getenv(_PATHS_ENV_VAR)
        if env_override:
            return Path(env_override).expanduser()
        if provided:
            return Path(provided).expanduser()
        project_config = self._project_root / "resources" / "paths.yaml"
        if project_config.exists():
            return project_config
        return self._packaged_config_dir / "paths.yaml"

    @staticmethod
    def _load_yaml(file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            return {}
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Expected dictionary at root of {file_path}, found {type(data)}")
            return data


_registry_singleton: Optional[PathRegistry] = None
_thread_lock = threading.Lock()


def get_path_registry() -> PathRegistry:
    """Return a process-wide singleton PathRegistry instance."""
    global _registry_singleton
    if _registry_singleton is None:
        with _thread_lock:
            if _registry_singleton is None:
                _registry_singleton = PathRegistry()
    return _registry_singleton


__all__ = ["PathRegistry", "get_path_registry"]
