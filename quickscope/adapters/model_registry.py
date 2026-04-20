"""Model configuration registry for Quickscope."""

from __future__ import annotations

import os
import yaml
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Dict, Optional

from quickscope.dataflow.path_registry import get_path_registry

CONFIG_ENV_VAR = "QUICKSCOPE_MODEL_CONFIG"
_DEFAULT_RESOURCE = "model_configs.yaml"


@dataclass(frozen=True)
class ModelSettings:
    alias: str
    provider: str
    model: str
    request_options: Dict[str, Any]
    tags: list[str]

    def merged_request(self, base_request: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep-merged request dictionary with overrides applied."""
        return _deep_merge_dicts(base_request, self.request_options)


class ModelRegistry:
    """Loads and provides access to model configuration data."""

    _cache: Optional[Dict[str, Any]] = None
    _source_path: Optional[Path] = None

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache = None
        cls._source_path = None

    @classmethod
    def _resolve_config_path(cls, override: Optional[str] = None) -> Path:
        if override:
            return Path(override).expanduser()

        env_path = os.getenv(CONFIG_ENV_VAR)
        if env_path:
            return Path(env_path).expanduser()

        registry = get_path_registry()
        return registry.config_file(_DEFAULT_RESOURCE)

    @classmethod
    def _load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        resolved_path = cls._resolve_config_path(config_path)
        if cls._cache is not None and cls._source_path == resolved_path:
            return cls._cache

        data: Dict[str, Any] = {}
        if resolved_path.exists():
            with resolved_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}

        cls._cache = data
        cls._source_path = resolved_path
        return data

    @classmethod
    def get_model_settings(
        cls, model_alias: str, *, config_path: Optional[str] = None
    ) -> ModelSettings:
        """
        Return model settings for the provided alias.

        Lookup order:
        1. Quickscope's resources/model_configs.yaml
        2. Default to OpenAI provider
        """
        table = cls._load_config(config_path)
        entry = table.get(model_alias)

        if entry:
            # Found in Quickscope's registry
            entry = deepcopy(entry)
            provider = entry.get("provider", "openai")
            model = entry.get("model", model_alias)
            request_options = entry.get("request_options", {})
            tags = entry.get("tags", [])

            return ModelSettings(
                alias=model_alias,
                provider=provider,
                model=model,
                request_options=request_options if isinstance(request_options, dict) else {},
                tags=tags if isinstance(tags, list) else []
            )

        # Model not found in registry
        raise ValueError(
            f"Model '{model_alias}' not found in model registry. "
            f"Add it to Quickscope's resources/model_configs.yaml"
        )

    @classmethod
    def available_aliases(cls, *, config_path: Optional[str] = None) -> Dict[str, ModelSettings]:
        """Return all available model aliases defined in configuration."""
        table = cls._load_config(config_path)
        return {
            alias: cls.get_model_settings(alias, config_path=config_path)
            for alias in table.keys()
        }


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries without mutating either input."""
    result: Dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
