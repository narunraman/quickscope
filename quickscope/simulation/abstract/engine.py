"""
Simulation Engine Abstraction for Quickscope
-----------------------------------------
Defines the abstract base class for simulation engines in the evaluation pipeline.

Engines manage:
- State initialization from scenarios
- Prompt generation for each step
- State updates from LLM responses
- Result finalization

The Runner is generic and calls these methods to execute any engine type.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any

from .schemas import PromptRequest, LLMResponse

# Type variables for scenario row and result types
ScenarioT = TypeVar("ScenarioT")
ResultT = TypeVar("ResultT")


class Engine(ABC, Generic[ScenarioT, ResultT]):
    """
    Abstract base class for simulation engines.

    Engines own all domain-specific logic:
    - How to create prompts
    - How to parse and process responses
    - How to track state across rounds
    - How to build final results

    The Runner is generic and orchestrates LLM calls for any engine.
    """

    @abstractmethod
    def initialize(self, scenario: ScenarioT) -> dict[str, Any]:
        """
        Initialize simulation state from a scenario.

        Args:
            scenario: The scenario row (GameRow, StandardRow, etc.)

        Returns:
            Initial state dict. Engine defines the structure.
        """
        ...

    @abstractmethod
    def get_next_prompts(self, state: dict[str, Any]) -> list[PromptRequest] | None:
        """
        Get prompts for the next execution step.

        For single-agent: returns [PromptRequest]
        For multi-agent: returns [PromptRequest, PromptRequest, ...]

        Args:
            state: Current simulation state

        Returns:
            List of prompt requests, or None if simulation is complete.
        """
        ...

    @abstractmethod
    def update_state(self, state: dict[str, Any], responses: list[LLMResponse]) -> None:
        """
        Update state with LLM responses.

        This is called after each batch of prompts completes.
        Engine should parse responses, update round history, check for failures, etc.

        Args:
            state: Current simulation state (will be mutated)
            responses: LLM responses corresponding to the prompts from get_next_prompts()
        """
        ...

    @abstractmethod
    def finalize(self, state: dict[str, Any]) -> ResultT:
        """
        Build the final result from completed state.

        Called when get_next_prompts() returns None (simulation complete)
        or when is_failed() returns True (early termination).

        Args:
            state: Final simulation state

        Returns:
            Result object (GameResult, StandardResult, etc.)
        """
        ...

    def is_failed(self, state: dict[str, Any]) -> bool:
        """
        Check if simulation has failed and should terminate early.

        Default implementation checks for a 'failed' key in state.
        Engines can override for custom failure detection.

        Args:
            state: Current simulation state

        Returns:
            True if simulation should terminate early.
        """
        return state.get("failed", False)

    # @abstractmethod
    # def compute_metrics(self, results: list[ResultT]) -> dict[str, dict[str, Any]]:
    #     """
    #     Compute domain-specific metrics for a batch of results.

    #     Each scenario can implement its own metrics logic.

    #     Args:
    #         results: List of result objects from finalize()

    #     Returns:
    #         Dict mapping scenario_id to metrics dict.
    #         Example: {"scenario_123": {"accuracy": 0.85, "epsilon_nash": 0.12}}
    #     """
    #     ...
