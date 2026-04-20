"""
Simulation Data Schemas for Quickscope
-----------------------------------
Defines the core Pydantic models and typed structures used for representing scenarios, results, and state in the simulation and evaluation pipeline. Includes base scenario and result schemas, round-level results, metric containers, and the scenario state structure for orchestrating multi-agent simulations.
"""

# Built-in packages
from dataclasses import dataclass
import logging
import math
from pydantic import (
    BaseModel,
    Field,
    AliasChoices,
    ConfigDict,
    field_validator,
)
from typing import Optional, Any, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...adapters.llm_interface import LLMInterface
    from .engine import Engine


logger = logging.getLogger(__name__)


class BaseRow(BaseModel):
    """Common fields for all scenarios."""

    scenario_id: str = Field(..., description="Unique identifier for this scenario")
    scenario_text: str = Field(..., description="The scenario description")
    model_name: str = Field(..., description="Primary model to use")
    player_names: dict[str, str] = Field(
        default_factory=lambda: {"player_1": "player_1"},
        description="Mapping of player IDs to display names",
    )

    # Configuration
    num_rounds: int = Field(..., gt=0, description="Number of rounds to play")
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="LLM temperature (None = provider default)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response (None triggers provider-specific defaults)",
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt"
    )
    logprobs: bool = Field(default=False, description="Whether to return logprobs")
    top_logprobs: Optional[int] = Field(
        default=None, description="Number of top logprobs to return"
    )
    
    # Source config for cache warm-starting (set by generator)
    source_config: Optional[dict] = Field(
        default=None, description="Config dict that generated this scenario"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("system_prompt", mode="before")
    def validate_system_prompt(cls, v):
        """Convert NaN values to None for system_prompt."""

        if isinstance(v, float) and math.isnan(v):
            return None
        return v



## Evaluation Config Schema

class BaseEvaluationConfig(BaseModel):
    """
    Base configuration for all evaluation scenarios.
    Contains parameters common to every type of evaluation (Game, Standard, DyVal).
    """

    model_name: str = Field(..., description="Model name/alias to use")
    temperature: Optional[float] = Field(
        default=None, description="LLM temperature (None = provider default)"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Max tokens (None = provider default)"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt"
    )
    logprobs: bool = Field(default=False, description="Whether to return logprobs")
    top_logprobs: Optional[int] = Field(
        default=None, description="Number of top logprobs to return"
    )


## Result and Metric Schemas

class RoundResult(BaseModel):
    """Results from a single round of interaction."""

    round_num: int = Field(..., description="Round number (1-indexed)")
    trace: str = Field(..., description="The question/prompt presented to the model")
    response: str = Field(..., description="Raw response from the model")
    parsed_response: Optional[list[float]] = Field(
        default=None, description="Parsed probability distribution"
    )
    boxed_logprobs: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Logprobs for tokens in boxed answer"
    )
    mcq_distribution: Optional[dict[str, float]] = Field(
        default=None, description="Extracted MCQ distribution from logprobs"
    )

    @field_validator("parsed_response")
    def validate_probabilities(cls, v):
        """Validate that probabilities are non-negative and sum to ~1.0."""
        if v is None:
            return v

        if not v:
            raise ValueError("Parsed response cannot be empty if provided")
        if any(p < 0 for p in v):
            raise ValueError("Probabilities must be non-negative")
        # Allow some tolerance for floating point precision
        prob_sum = sum(v)
        if abs(prob_sum - 1.0) > 1e-6:
            # Normalize if close but not exact
            if abs(prob_sum - 1.0) < 0.1:
                v = [p / prob_sum for p in v]
            else:
                raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")
        return v


class BaseResult(BaseModel):
    """Common results from an evaluation run."""

    # Multi-round data
    trace: list[str] = Field(..., description="Per-round questions presented to model")
    responses: list[str] = Field(..., description="Per-round raw responses from model")
    parsed_responses: Optional[list[Any]] = Field(
        default=None,
        description="Per-round parsed values, such as extracted answers",
    )

    # LLM response metadata for token tracking (optional)
    llm_responses: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="LLM response metadata containing token usage and other provider details",
        validation_alias=AliasChoices("llm_responses", "openai_responses"),
        serialization_alias="llm_responses",
    )

    # Configuration and metadata
    scenario_id: str = Field(
        ...,
        description="Unique identifier linking related evaluations from same scenario",
    )
    player_num: str = Field(..., description="Which player this LLM played as")
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature used for generation (None = provider default)",
    )
    model_name: str = Field(..., description="Model used for this evaluation")

    # Additional metadata
    num_rounds: int = Field(..., description="Total number of rounds played")
    timestamp: str = Field(..., description="When evaluation was run")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_dataframe_row(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for DataFrame construction."""
        row = {
            "scenario_id": self.scenario_id,
            "trace": self.trace,
            "responses": self.responses,
            "parsed_responses": self.parsed_responses,
            "player_num": self.player_num,
            "temperature": self.temperature,
            "model_name": self.model_name,
            "num_rounds": self.num_rounds,
            "timestamp": self.timestamp,
        }
        row.update(self.metadata)
        return row


class MetricResult(BaseModel):
    """Container for a computed metric result."""

    name: str
    value: float
    description: str
    metadata: dict[str, Any]


class ScenarioState(TypedDict):
    """Typed state structure for scenario execution."""

    player_configs: dict[str, BaseEvaluationConfig]
    player_engines: dict[str, "Engine"]
    player_llms: dict[str, "LLMInterface"]
    player_traces: dict[str, list[str]]
    player_responses: dict[str, list[str]]
    player_parsed_responses: dict[str, list[Any]]
    llm_metadata: dict[str, list[dict[str, Any] | None]]
    player_boxed_logprobs: dict[str, list[list[dict[str, Any]] | None]]
    player_mcq_distribution: dict[str, list[dict[str, float] | None]]



## Prompt and Response Schemas

@dataclass
class PromptRequest:
    """
    A request for an LLM prompt with associated configuration.

    Engines return lists of PromptRequest objects from get_next_prompts().
    The Runner collects these and makes batched LLM calls.
    """

    prompt_text: str
    model_name: str
    
    # Context for routing response back
    scenario_id: str 
    player_id: str

    temperature: float | None = None  # None = use model default
    max_tokens: int | None = None  # None = use model default
    system_prompt: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    
    # Source config that generated this prompt (for cache warm-starting)
    source_config: dict | None = None



@dataclass
class LLMResponse:
    """
    Response from an LLM call, paired with the original request.
    """

    text: str

    # Context inherited from request
    scenario_id: str 
    player_id: str

    metadata: dict | None = None  # Token usage, logprobs, etc.

