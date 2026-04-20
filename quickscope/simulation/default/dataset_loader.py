"""Dataset-backed scenario loader for offline replay."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import TypeAdapter

from .schemas import DefaultRow


class DatasetAdapter(Protocol):
    """Adapter interface for dataset-backed scenario sampling."""

    def sample(
        self, params: dict[str, Any], rng: Any | None, override_keys: set[str]
    ) -> dict[str, Any]:
        """Return a sampled row dict matching provided params."""
        ...

    def columns(self) -> set[str]:
        """Return the set of available columns for filtering."""
        ...

class InMemoryDatasetAdapter:
    """In-memory dataset adapter backed by a pandas DataFrame."""

    def __init__(self, rows: list[DefaultRow]):
        # Convert to dicts and flatten metadata into top-level fields
        flattened_rows = []
        for row in rows:
            row_dict = row.model_dump(mode="python")
            metadata = row_dict.pop("metadata", {})
            # Flatten metadata fields to top level (makes them filterable)
            row_dict.update(metadata)
            flattened_rows.append(row_dict)

        df = pd.DataFrame(flattened_rows)

        # Derive scenario_id if missing; prefer id_instance if available
        if "scenario_id" not in df.columns or df["scenario_id"].isna().any():

            def derive(row: pd.Series) -> str:
                if (
                    "id" in row
                    and "instance" in row
                    and pd.notna(row["id"])
                    and pd.notna(row["instance"])
                ):
                    return f"{row['id']}_{row['instance']}"
                if "id" in row and pd.notna(row["id"]):
                    return str(row["id"])
                return str(row.name)

            df["scenario_id"] = df.apply(derive, axis=1)

        self.df = df

    def columns(self) -> set[str]:
        return set(self.df.columns)

    def sample(
        self, params: dict[str, Any], rng: Any | None, override_keys: set[str]
    ) -> dict[str, Any]:
        df = self.df

        # Resolve target row
        if "scenario_id" in params:
            sid = params["scenario_id"]
            subset = df[df["scenario_id"] == sid]
            if subset.empty:
                raise ValueError(f"scenario_id '{sid}' not found in dataset")
            row = subset.sample(
                n=1, random_state=rng.integers(1e9) if rng is not None else None
            ).iloc[0]
        else:
            # Build filter from provided params that correspond to columns
            subset = df
            applied_filters = False
            for key, value in params.items():
                if key in override_keys:
                    continue  # Skip override keys in filtering
                if key in df.columns:
                    subset = subset[subset[key] == value]
                    applied_filters = True
            if subset.empty:
                raise ValueError(f"No rows found matching filters: {params}")
            # If no filters applied, sample from full dataset
            if not applied_filters:
                subset = df
            row = subset.sample(
                n=1, random_state=rng.integers(1e9) if rng is not None else None
            ).iloc[0]

        return row.to_dict()


class IndexedDatasetAdapter:
    """Dataset adapter that samples rows via a prebuilt index."""

    def __init__(
        self,
        dataset_path: Path,
        index_path: Path,
        *,
        index_fields: list[str] | None = None,
        sentinel: str = "__empty__",
    ) -> None:
        self.dataset_path = dataset_path
        self.index_path = index_path
        self.sentinel = sentinel

        with index_path.open("rb") as f:
            payload = pickle.load(f)

        stored_fields = payload.get("fields")
        if not stored_fields:
            raise ValueError(f"Index file missing fields metadata: {index_path}")

        if index_fields is not None and index_fields != stored_fields:
            raise ValueError(
                "Index field mismatch. "
                f"Requested {index_fields}, index built with {stored_fields}."
            )

        self.fields = list(index_fields or stored_fields)
        self.index: dict[tuple[Any, ...], list[int]] = payload.get("index", {})

    def columns(self) -> set[str]:
        return set(self.fields)

    def _build_key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        missing = [f for f in self.fields if f not in params]
        if missing:
            raise ValueError(
                "Indexed dataset requires all fields. "
                f"Missing: {missing}. Index fields: {self.fields}."
            )
        values = []
        for field in self.fields:
            value = params.get(field)
            if value is None:
                value = self.sentinel
            values.append(value)
        return tuple(values)

    def sample(
        self, params: dict[str, Any], rng: Any | None, override_keys: set[str]
    ) -> dict[str, Any]:
        if "scenario_id" in params and "scenario_id" not in self.fields:
            raise ValueError(
                "Index does not include scenario_id. "
                "Rebuild index with --fields including scenario_id or remove it."
            )

        key = self._build_key(params)
        offsets = self.index.get(key)
        if not offsets:
            raise ValueError(f"No rows found matching indexed key: {key}")

        if rng is not None:
            offset = int(offsets[int(rng.integers(len(offsets)))])
        else:
            offset = int(random.choice(offsets))

        with self.dataset_path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        if not line:
            raise ValueError(f"Failed to read dataset row at offset {offset}")

        row_dict = json.loads(line)
        row_dict["_offset"] = offset
        return row_dict


class DatasetScenarioLoader:
    """Load DefaultRow objects from an existing benchmark file."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        default_model: str | None = None,
        default_opponent: str | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.default_model = default_model
        self.default_opponent = default_opponent or default_model 

    def load(self) -> list[DefaultRow]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        rows: list[dict] = []
        if self.dataset_path.suffix == ".jsonl":
            with self.dataset_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        elif self.dataset_path.suffix == ".json":
            rows = json.loads(self.dataset_path.read_text())
            if isinstance(rows, dict):
                rows = [rows]
        elif self.dataset_path.suffix in {".csv", ".tsv"}:
            df = pd.read_csv(self.dataset_path)
            rows = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_path}")

        dataset_rows: list[DefaultRow] = []
        adapter = TypeAdapter(DefaultRow)

        for index, data in enumerate(rows):
            data = dict(data)

            # Get all DefaultRow field names from schema
            standard_row_fields = set(DefaultRow.model_fields.keys())
            
            # Separate DefaultRow fields from metadata
            # DefaultRow fields come directly from data (with defaults for optional fields)
            standard_data = {}
            metadata = {}
            
            for key, value in data.items():
                if key in standard_row_fields:
                    standard_data[key] = value
                else:
                    metadata[key] = value
            
            # Apply defaults for optional DefaultRow fields if not present
            standard_data.setdefault("model_name", self.default_model or "model")
            standard_data.setdefault("num_rounds", 1)
            standard_data.setdefault("temperature", None)
            standard_data.setdefault("max_tokens", None)  # None triggers smart defaults in EvaluationConfig
            standard_data.setdefault("scenario_type", "default")
            standard_data.setdefault("player_names", {"player_1": standard_data["model_name"]})
            
            # Add metadata to standard_data
            standard_data["metadata"] = metadata
            
            try:
                row = DefaultRow(**standard_data)
            except Exception as e:
                raise ValueError(f"Failed to parse standard row at index {index}: {e}\nData keys: {list(data.keys())}")
            dataset_rows.append(row)

        return dataset_rows

    def load_adapter(
        self, *, index_fields: list[str] | None = None
    ) -> DatasetAdapter:
        """Load a dataset adapter, using an index if one exists."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        index_path = self.dataset_path.with_name(
            f"{self.dataset_path.name}.index.pkl"
        )
        if index_path.exists():
            return IndexedDatasetAdapter(
                self.dataset_path,
                index_path,
                index_fields=index_fields,
                sentinel="__empty__",
            )

        rows = self.load()
        return InMemoryDatasetAdapter(rows)
