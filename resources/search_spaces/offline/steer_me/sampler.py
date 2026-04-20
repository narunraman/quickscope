from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Callable


def get_sampler(dataset_path: str) -> Callable[[], dict]:
    """Return a sampler that emits only valid (element, domain, type) tuples."""
    data_path = Path(dataset_path)
    index_path = data_path.with_name(f"{data_path.name}.index.pkl")

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    with index_path.open("rb") as f:
        payload = pickle.load(f)

    fields = payload.get("fields")
    index = payload.get("index")
    if not fields or not index:
        raise ValueError("Index file missing fields or index data")

    keys = list(index.keys())

    def sampler() -> dict:
        key = random.choice(keys)
        return dict(zip(fields, key))

    return sampler
