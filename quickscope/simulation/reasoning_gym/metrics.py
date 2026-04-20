"""Reasoning Gym metrics using native scoring.

Uses reasoning_gym's score_answer_fn for accurate evaluation
of model responses against the dataset's expected format.

For datasets that use \\boxed{} format (like gsm_symbolic), extracts
the boxed content before passing to the scorer.
"""

import contextlib
import io
import re
from typing import Any

import reasoning_gym

from ..default.schemas import DefaultResult
from ..default.metrics import extract_token_usage
from ..default.response_handler import DefaultResponseHandler
from ..abstract import ComponentRegistry


@contextlib.contextmanager
def _suppress_stdout():
    """Context manager to suppress stdout (e.g., reasoning_gym debug prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# Reuse the robust boxed extraction from DefaultResponseHandler
_response_handler = DefaultResponseHandler()


def _extract_boxed_answer(response: str) -> str | None:
    """Extract the answer from response, trying multiple formats.
    
    Tries in order:
    1. Proper \\boxed{...} format (LaTeX style)
    2. "boxed X" without backslash/braces (common LLM variation)
    3. "Answer: X" at end of response
    
    Returns the extracted content or None if not found.
    """
    
    # 1. Try proper \boxed{...} first
    boxed = _response_handler._extract_last_boxed(response)
    if boxed is not None:
        # Strip \text{...} wrapper if present (common in LaTeX responses)
        if boxed.startswith("\\text{") and boxed.endswith("}"):
            boxed = boxed[6:-1]
        return boxed
    
    # 2. Try "boxed X" without proper formatting (e.g., "boxed 13", "boxed infeasible")
    # Match "boxed" followed by the answer (word or number)
    boxed_match = re.search(r'\bboxed\s+(\S+)', response, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1)
    
    # 3. Try "Answer: X" at end of response
    answer_match = re.search(r'\bAnswer:\s*(.+?)\s*$', response, re.IGNORECASE | re.MULTILINE)
    if answer_match:
        return answer_match.group(1).strip()
    
    return None


# Datasets that require lowercase answers for scoring
LOWERCASE_DATASETS = frozenset({"shortest_path"})


def _normalize_answer(answer: str, rg_dataset: str) -> str:
    """Normalize answer format for scoring.
    
    Some reasoning-gym datasets have case-sensitive scorers.
    This normalizes the answer to match expected format.
    
    Args:
        answer: Extracted answer from model response
        rg_dataset: Name of the reasoning-gym dataset
        
    Returns:
        Normalized answer string
    """
    if rg_dataset in LOWERCASE_DATASETS:
        # shortest_path expects lowercase space-separated directions: 
        # "up down left right" or "infeasible"
        # Models may output comma-separated like "up, right, down"
        normalized = answer.lower().strip()
        # Replace commas with spaces and collapse multiple spaces
        normalized = normalized.replace(",", " ")
        normalized = " ".join(normalized.split())
        return normalized
    return answer.strip()


def compute_rg_metrics(result: DefaultResult) -> dict[str, Any]:
    """Compute metrics for a reasoning_gym scenario using native scoring.

    Args:
        result: DefaultResult with rg_dataset and rg_entry in metadata.

    Returns:
        Metrics dict with accuracy from RG's native scorer.
    """
    metadata = getattr(result, "metadata", None) or {}
    tokens = extract_token_usage(result)

    # Check for timeout
    if getattr(result, "timed_out", False):
        return {
            "accuracy": None,
            "timed_out": True,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 1,
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    # Get RG dataset info
    rg_dataset = metadata.get("rg_dataset")
    rg_entry = metadata.get("rg_entry")

    if not rg_dataset or not rg_entry:
        # Missing RG metadata - can't score
        return {
            "accuracy": None,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 1,
            "error": "Missing rg_dataset or rg_entry in metadata",
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    try:
        # Get the native scorer for this dataset
        scorer = reasoning_gym.get_score_answer_fn(rg_dataset)

        # Get raw response
        raw_response = result.responses[-1] if result.responses else ""

        # Try to extract boxed answer first (for chain-of-thought responses)
        # This ensures we score the final answer, not intermediate numbers
        answer_to_score = _extract_boxed_answer(raw_response)
        if answer_to_score is None:
            # Fallback to raw response if no boxed answer found
            answer_to_score = raw_response

        # Normalize answer for case-sensitive datasets
        answer_to_score = _normalize_answer(answer_to_score, rg_dataset)

        # Score the response (suppress reasoning_gym's debug prints)
        with _suppress_stdout():
            score = scorer(answer=answer_to_score, entry=rg_entry) # type: ignore
        # reasoning_gym returns 0.01 for wrong answers; normalize to binary 0/1
        if score is not None:
            accuracy = 1.0 if score == 1.0 else 0.0
        else:
            accuracy = None

        return {
            "accuracy": accuracy,
            "brier_score": None,
            "round_scores": [accuracy] if accuracy is not None else [],
            "correct": 1 if accuracy == 1.0 else 0,
            "attempted": 1,
            "missing": 0,
            "timed_out": False,
            **tokens,
            "player_1": {
                "player_num": result.player_num,
                "accuracy": accuracy,
                **tokens,
            },
        }

    except Exception as e:
        # Scoring failed
        import logging

        logging.getLogger(__name__).warning(f"RG scoring failed: {e}")

        return {
            "accuracy": None,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 1,
            "missing": 0,
            "error": str(e),
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }


@ComponentRegistry.metrics("reasoning_gym")
def compute_rg_metrics_batch(
    results: list[DefaultResult],
) -> dict[str, dict[str, Any]]:
    """Batch metrics for reasoning_gym scenarios."""
    return {r.scenario_id: compute_rg_metrics(r) for r in results}
