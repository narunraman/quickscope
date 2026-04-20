"""
Scenario Generator Abstraction for Quickscope
-----------------------------------------
Defines the abstract base class for scenario generators in the simulation pipeline. Scenario generators are responsible for producing evaluation scenarios as schema objects from configuration parameters, supporting reproducible and configurable simulation experiments. Subclasses should implement domain-specific scenario generation logic.
"""

import numpy as np

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from .schemas import BaseRow

RowT = TypeVar("RowT", bound=BaseRow)


class ScenarioGenerator(ABC, Generic[RowT]):
    """
    Abstract base class for generating evaluation scenarios from configurations.
    """

    @abstractmethod
    def generate(self, params: dict, rng: np.random.Generator | None = None) -> RowT:
        """
        Generate a scenario based on input parameters.

        Args:
            params: Configuration parameters for scenario generation
            rng: Random number generator for reproducibility

        Returns:
            Schema object (GameRow, StandardRow, etc.) representing the scenario
        """
        ...

    @property
    def supports_caching(self) -> bool:
        """Whether scenarios from this generator can be cached across runs."""
        return False

    @property
    def dataset_name(self) -> str | None:
        """Dataset name for cache directory. Only needed if supports_caching=True."""
        return None

