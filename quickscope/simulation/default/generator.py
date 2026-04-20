"""Default scenario generator for fixed datasets.

This generator allows the Bayesian optimizer to select existing scenarios
by `scenario_id` (or by `id`/`instance`) and optionally apply overrides
such as `model_name`, `temperature`, or `max_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_loader import DatasetScenarioLoader, DatasetAdapter
from .schemas import DefaultRow
from ..abstract import ScenarioGenerator


@dataclass
class DefaultScenarioGenerator(ScenarioGenerator[DefaultRow]):
    """Selects scenarios from a fixed dataset instead of generating new ones."""

    dataset_path: str | Path
    model: str | None = None
    opponent_model: str | None = None

    def __post_init__(self) -> None:
        # Keys that are treated as overrides rather than filters
        self.override_keys = {
            "model_name",
            "temperature",
            "max_tokens",
            "system_prompt",
            "opponent_model",
        }

        loader = DatasetScenarioLoader(self.dataset_path)
        self.dataset: DatasetAdapter = loader.load_adapter()

    def generate(self, params: dict, rng: Any | None = None) -> DefaultRow:
        """Return a DefaultRow selected from the fixed dataset.

        Params may include:
          - scenario_id: exact scenario to select (preferred)
          - Filter keys (e.g., id, instance, domain, element, type, tags, etc.)
          - Override keys: model_name, temperature, max_tokens, system_prompt, opponent_model

        Filtering logic:
          - If scenario_id provided, select that row (error if missing).
          - Else, use any provided params that match dataframe columns (excluding override keys)
            to filter rows; if multiple remain, sample one with replacement.
          - Override keys modify the selected row's values but don't filter.
        """
        row_dict = self.dataset.sample(params, rng, self.override_keys)
        
        if not row_dict.get("scenario_id"):
            if row_dict.get("id") is not None and row_dict.get("instance") is not None:
                row_dict["scenario_id"] = f"{row_dict['id']}_{row_dict['instance']}"
            elif row_dict.get("id") is not None:
                row_dict["scenario_id"] = str(row_dict["id"])
            elif row_dict.get("_offset") is not None:
                row_dict["scenario_id"] = str(row_dict["_offset"])
            else:
                raise ValueError("scenario_id missing and could not be derived")

        # Apply fixed overrides if provided
        if self.model:
            row_dict["model_name"] = self.model
        if self.opponent_model:
            row_dict["opponent_model"] = self.opponent_model

        # Apply overrides if provided
        for key in self.override_keys:
            if key in params and params[key] is not None:
                row_dict[key] = params[key]

        # Format the prompt with answer instructions
        # raw_text = row_dict.get("scenario_text", "")
        # answer_format = row_dict.get("answer_format", "text")
        # mcq_options = row_dict.get("mcq_options")
        # row_dict["scenario_text"] = self.format_prompt(raw_text, answer_format, mcq_options)

        # Capture all non-schema fields into metadata to prevent data loss
        schema_fields = set(DefaultRow.model_fields.keys())
        metadata = row_dict.pop("metadata", {})
        
        # Identify extras (domain, tags, etc.)
        extras = {k: v for k, v in list(row_dict.items()) if k not in schema_fields}
        
        # Move extras to metadata
        metadata.update(extras)
        for k in extras:
            row_dict.pop(k, None)
        
        row_dict["metadata"] = metadata
        row_dict["source_config"] = params

        return DefaultRow(**row_dict)

    @property
    def supports_caching(self) -> bool:
        """Default scenarios can be cached since scenario_id maps to fixed content."""
        return True

    @property
    def dataset_name(self) -> str:
        """Extract dataset name from path for cache directory."""
        return Path(self.dataset_path).parent.name
