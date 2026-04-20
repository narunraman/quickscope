

"""
Response Handler Abstraction for Quickscope
----------------------------------------
Defines the abstract base class for response handlers in the simulation pipeline. Response handlers are responsible for parsing raw LLM outputs into structured data and computing per-response metrics for evaluation. Subclasses should implement domain-specific parsing and metric logic for different scenario types.
"""

# Built-in packages
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

ScenarioT = TypeVar("ScenarioT")
ParsedT = TypeVar("ParsedT")


class ResponseHandler(ABC, Generic[ScenarioT, ParsedT]):
    """
    Abstract base class for handling LLM responses.
    Responsibilties: Parsing strings into structured data, and calculating per-response metrics.
    """

    @abstractmethod
    def parse_response(self, response: str, scenario: ScenarioT) -> ParsedT:
        """Parse the raw string response."""
        raise NotImplementedError

