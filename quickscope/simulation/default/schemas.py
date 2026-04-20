"""
Standard domain data models.
"""

from typing import Optional, Union, Literal, Any
from pydantic import Field

from ..abstract import BaseRow, BaseResult


class DefaultRow(BaseRow):
    """Represents a default LLM evaluation scenario."""

    scenario_type: Literal["default"] = "default"

    # Override BaseRow's required num_rounds with a sensible default for single-shot evals
    num_rounds: int = Field(default=1, gt=0, description="Number of rounds to play")

    answer_format: Literal["numeric", "mcq", "text", "functional"] = Field(
        default="numeric", description="Expected answer format for parsing"
    )
    reference_answer: Optional[Union[str, list[str]]] = Field(
        default=None, description="Ground truth answer(s)"
    )
    mcq_options: Optional[list[str]] = Field(
        default=None, description="Valid options for MCQ answers"
    )
    eval_method: Optional[str] = Field(
        default=None, description="Method to use for evaluation"
    )
    skip_format_prompt: bool = Field(
        default=False,
        description="Skip default prompt formatting (for scenarios with native formatting like reasoning_gym)",
    )


class DefaultResult(BaseResult):
    """Results from a standard LLM evaluation."""

    type: Literal["standard"] = "standard"

    reference_answer: Optional[Union[str, list[str]]] = Field(
        default=None, description="Ground truth answer(s)"
    )
    mcq_options: Optional[list[str]] = Field(
        default=None, description="MCQ options for answer disambiguation"
    )

    boxed_logprobs: Optional[list[list[dict[str, Any]] | None]] = Field(
        default=None, description="Logprobs for tokens in boxed answer (per round)"
    )
    mcq_distribution: Optional[list[dict[str, float] | None]] = Field(
        default=None, description="Extracted MCQ distribution from logprobs (per round)"
    )
    timed_out: bool = Field(default=False, description="Whether the request timed out")

    def to_dataframe_row(self) -> dict[str, Any]:
        row = super().to_dataframe_row()
        if self.reference_answer:
            row["reference_answer"] = self.reference_answer
        return row
