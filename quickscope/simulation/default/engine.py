"""
Default Engine for Quickscope
-------------------------
Handles standard single-agent evaluation scenarios including:
- Single-prompt per round
- Response parsing (boxed answers, MCQ)
- Result construction

Engine is stateless - all state is passed via dict.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

# from quickscope.simulation.base.response_handler import ResponseHandler

from .schemas import DefaultRow, DefaultResult
from .response_handler import DefaultResponseHandler
from ..abstract.component_registry import ComponentRegistry
from ..abstract import (
    Engine,
    PromptRequest,
    LLMResponse,
)
from ...dataflow.logging_config import get_logger

logger = get_logger("engine")


@ComponentRegistry.engine("default")
class DefaultEngine(Engine[DefaultRow, DefaultResult]):
    """
    Engine for standard single-agent evaluations.

    Handles Q&A scenarios with text, numeric, or MCQ answers.
    """

    def __init__(self, handler: Optional[DefaultResponseHandler] = None):
        self.handler = handler or DefaultResponseHandler()

    def initialize(self, scenario: DefaultRow) -> dict[str, Any]:
        """
        Initialize simulation state from a scenario.

        Returns:
            State dict with scenario, history, and extended tracking.
        """
        # Check for generation failure (e.g., DyVal promptbench errors)
        # If flagged, mark as done immediately to skip LLM calls
        metadata = getattr(scenario, "metadata", None) or {}
        is_generation_failure = metadata.get("generation_failure", False)

        if is_generation_failure:
            logger.warning("Skipping scenario %s: generation failure", scenario.scenario_id)

        return {
            "scenario": scenario,
            "history": [],
            "current_round": 0,
            "done": is_generation_failure,  # Skip LLM if generation failed
            "failed": False,
            # Extended info for result construction
            "traces": [],
            "llm_metadata": [],
            "boxed_logprobs": [],
            "mcq_distribution": [],
        }

    @staticmethod
    def format_prompt(
        scenario_text: str,
        answer_format: str = "text",
        mcq_options: list[str] | None = None,
    ) -> str:
        """
        Format scenario text with answer format instructions.

        Utility method for standard Q&A scenarios. Game scenarios have their
        own prompt format and should not use this.

        Args:
            scenario_text: The raw question/scenario text
            answer_format: One of "numeric", "mcq", or "text"
            mcq_options: List of MCQ options (required if answer_format="mcq")

        Returns:
            Complete prompt text ready to send to LLM
        """
        base = scenario_text.strip()

        if answer_format == "numeric":
            instruction = (
                "Please reason step-by-step to arrive at your final answer. "
                "Provide your final answer inside \\boxed{ }. "
                "For example, respond with \\boxed{42}."
            )
        elif answer_format == "mcq":
            if not mcq_options:
                raise ValueError("mcq_options required for MCQ answer format")
            letters = [chr(ord("A") + i) for i in range(len(mcq_options))]
            option_lines = [
                f"{letter}. {opt}" for letter, opt in zip(letters, mcq_options)
            ]
            option_text = "\n".join(option_lines)
            instruction = (
                f"{option_text}\n"
                "Please reason step-by-step to arrive at your final answer. "
                "Provide your final choice as the letter inside \\boxed{ }. "
                "For example, respond with \\boxed{A}."
            )
        else:
            instruction = "Provide your final answer inside \\boxed{ }."

        return f"{base}\n\n{instruction}"

    def get_next_prompts(self, state: dict[str, Any]) -> list[PromptRequest] | None:
        """
        Get prompt for the next round.

        Returns single prompt (default is single-agent).
        Returns None if done.
        """
        scenario: DefaultRow = state["scenario"]
        current_round = state["current_round"]
        next_round = current_round + 1

        # Check completion
        if next_round > scenario.num_rounds:
            return None

        if state["done"] or state["failed"]:
            return None

        # Format prompt - skip if scenario has native formatting
        if getattr(scenario, "skip_format_prompt", False):
            prompt_text = scenario.scenario_text
        else:
            prompt_text = self.format_prompt(
                scenario.scenario_text,
                getattr(scenario, "answer_format", "text"),
                getattr(scenario, "mcq_options", None),
            )
        state["traces"].append(prompt_text)  # Track prompts

        return [
            PromptRequest(
                prompt_text=prompt_text,
                model_name=scenario.model_name,
                temperature=scenario.temperature,
                max_tokens=scenario.max_tokens,
                system_prompt=scenario.system_prompt,
                logprobs=getattr(scenario, "logprobs", False),
                top_logprobs=getattr(scenario, "top_logprobs", None),
                scenario_id=scenario.scenario_id,
                player_id="player_1",
                source_config=getattr(scenario, "source_config", None),
            )
        ]

    def update_state(self, state: dict[str, Any], responses: list[LLMResponse]) -> None:
        """
        Update state with LLM response.

        Parses response, extracts MCQ distribution if applicable.
        """
        scenario: DefaultRow = state["scenario"]

        # Should only have one response for default
        response = responses[0]
        metadata = response.metadata or {}

        # Parse response
        parsed = self.handler.parse_response(response.text, scenario)

        # Extract logprobs if available
        logprobs_data = None
        mcq_dist = None

        if metadata.get("logprobs"):
            raw_logprobs = metadata["logprobs"]
            normalized_logprobs = self.handler._normalize_logprobs(raw_logprobs)

            if normalized_logprobs:
                logprobs_data = normalized_logprobs

                # MCQ distribution extraction
                # if getattr(scenario, "answer_format", "") == "mcq" and getattr(
                # scenario, "mcq_options", None
                # ):
                if scenario.answer_format == "mcq" and scenario.mcq_options:
                    mcq_dist = self.handler.extract_mcq_distribution(
                        normalized_logprobs,
                        str(parsed),
                        scenario.mcq_options,
                        response.text,
                    )

        # Update state
        state["history"].append(
            {
                "response": response.text,
                "parsed": parsed,
            }
        )
        state["current_round"] += 1
        state["llm_metadata"].append(metadata)
        state["boxed_logprobs"].append(logprobs_data)
        state["mcq_distribution"].append(mcq_dist)

        # Check if done
        if state["current_round"] >= scenario.num_rounds:
            state["done"] = True

    def finalize(self, state: dict[str, Any]) -> DefaultResult:
        """
        Build final result from completed state.
        """
        scenario: DefaultRow = state["scenario"]
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        history = state["history"]
        responses = [h["response"] for h in history]
        parsed_responses = [h["parsed"] for h in history]

        # Check if any response was a timeout
        timed_out = any(
            isinstance(m, dict) and m.get("timed_out", False)
            for m in state.get("llm_metadata", [])
        )

        return DefaultResult(
            scenario_id=scenario.scenario_id,
            trace=state["traces"],
            responses=responses,
            parsed_responses=parsed_responses,
            llm_responses=state["llm_metadata"],
            player_num="player_1",
            temperature=scenario.temperature,
            model_name=scenario.model_name,
            num_rounds=len(history),
            timestamp=timestamp,
            reference_answer=getattr(scenario, "reference_answer", None),
            mcq_options=getattr(scenario, "mcq_options", None),
            boxed_logprobs=state["boxed_logprobs"],
            mcq_distribution=state["mcq_distribution"],
            metadata=scenario.metadata,
            timed_out=timed_out,
        )

    # def compute_metrics(
    #     self, results: list[DefaultResult]
    # ) -> dict[str, dict[str, Any]]:
    #     """
    #     Compute standard metrics for results.

    #     Each result represents one scenario (single-agent, no pairing needed).
    #     """
    #     from .metrics import compute_standard_metrics

    #     return {
    #         result.scenario_id: compute_standard_metrics(result) for result in results
    #     }
