"""
DyVal-specific metrics using promptbench's native evaluation.

DyVal has multiple answer types (numeric, boolean, lists) that require
specialized parsing and evaluation. This module uses promptbench's
dyval_evaluate with custom parsing to handle both <<<>>> and \\boxed{} formats.
"""

import contextlib
import io
import re
from typing import Any

from promptbench.dyval.dyval_utils import (
    process_dyval_preds,
    dyval_evaluate,
)

from ..default.schemas import DefaultResult
from ..default.metrics import extract_token_usage
from ..abstract import ComponentRegistry


@contextlib.contextmanager
def _suppress_stdout():
    """Context manager to suppress stdout (e.g., promptbench debug prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# DyVal dataset type mapping (from space.yaml)
DYVAL_DATASET_TYPES = {
    1: "arithmetic",
    2: "linear_equation",
    3: "bool_logic",
    4: "deductive_logic",
    5: "abductive_logic",
    6: "reachability",
    7: "max_sum_path",
}

# Regex patterns for answer extraction
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
_DYVAL_PATTERN = re.compile(r"<<<([^>]+)>>>")
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final answer|answer is|result is|therefore)[:\s]*([^\n.]+)",
    re.IGNORECASE
)


def parse_dyval_response(raw_response: str) -> str:
    """
    Parse a DyVal response to extract the answer.
    
    Handles multiple answer formats:
    1. DyVal native: <<<answer>>>
    2. LaTeX boxed: \\boxed{answer}
    3. Text patterns: "The answer is X", "Final answer: X"
    4. Fallback: Last line or full response
    
    Args:
        raw_response: Raw LLM response text
        
    Returns:
        Extracted answer string
    """
    if not raw_response:
        return ""
    
    # 1. Try DyVal native format first (<<<answer>>>)
    dyval_match = _DYVAL_PATTERN.search(raw_response)
    if dyval_match:
        return dyval_match.group(1).strip()
    
    # 2. Try \\boxed{answer} format
    boxed_match = _BOXED_PATTERN.search(raw_response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 3. Try "final answer is X" patterns
    answer_match = _FINAL_ANSWER_PATTERN.search(raw_response)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 4. Fallback: Look for True/False/N/A at the end
    lines = raw_response.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    
    # Check for boolean/N/A keywords
    for keyword in ['True', 'False', 'N/A']:
        if keyword.lower() in last_line.lower():
            return keyword
    
    # Try to extract a number from the last line
    numbers = re.findall(r'-?\d+\.?\d*', last_line)
    if numbers:
        return numbers[-1]
    
    # Ultimate fallback: return the cleaned last line
    return last_line


def score_dyval_answer(
    raw_response: str,
    reference_answer: Any,
    dataset_type: str,
) -> float | None:
    """
    Score a DyVal answer using custom parsing + promptbench evaluation.
    
    Args:
        raw_response: The raw LLM response text
        reference_answer: Ground truth answer (can be float, bool, list, str)
        dataset_type: Type of DyVal problem (e.g., 'arithmetic', 'bool_logic')
    
    Returns:
        1.0 if correct, 0.0 if incorrect, None if response is missing
    """
    if raw_response is None or raw_response == "":
        return None
    
    # First try promptbench's native parser (handles <<<>>> format)
    parsed = process_dyval_preds(raw_response)
    
    # If native parser fails (returns empty), use our custom parser
    if not parsed:
        parsed = parse_dyval_response(raw_response)
    
    if not parsed:
        return 0.0
    
    # Use promptbench's evaluation function
    # It returns a score (number of correct answers out of total)
    promptbench_score = 0.0
    try:
        with _suppress_stdout():
            score = dyval_evaluate(dataset_type, [parsed], [reference_answer])
        promptbench_score = float(score) if score is not None else 0.0
    except Exception:
        pass
    
    # If promptbench gave us a correct answer, return it
    if promptbench_score == 1.0:
        return 1.0
    
    # Otherwise, try direct comparison as fallback
    # (promptbench has strict formatting requirements that may not match our output)
    # For linear equations, use full response since we need multiple numbers
    comparison_text = raw_response if dataset_type == 'linear_equation' else parsed
    direct_score = _direct_comparison(comparison_text, reference_answer, dataset_type)
    
    # Return the better of the two scores
    return max(promptbench_score, direct_score)


def _direct_comparison(parsed: str, reference: Any, dataset_type: str) -> float:
    """
    Fallback direct comparison when promptbench evaluation fails.
    """
    try:
        # Boolean types
        if dataset_type in ('bool_logic', 'deductive_logic', 'abductive_logic', 'reachability'):
            parsed_lower = parsed.lower().strip()
            if isinstance(reference, bool):
                ref_str = 'true' if reference else 'false'
            else:
                ref_str = str(reference).lower().strip()
            
            # Handle N/A
            if ref_str == 'n/a' and 'n/a' in parsed_lower:
                return 1.0
            
            # Handle True/False
            if ref_str in ('true', 'false'):
                return 1.0 if ref_str in parsed_lower else 0.0
        
        # Numeric types
        if dataset_type == 'arithmetic':
            parsed_num = float(parsed)
            ref_num = float(reference)
            if abs(parsed_num - ref_num) < 1e-6 * max(abs(ref_num), 1):
                return 1.0
        
        # Linear equation (list of numbers)
        if dataset_type == 'linear_equation':
            # Extract numbers from parsed
            parsed_nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', parsed)]
            ref_nums = list(reference) if isinstance(reference, list) else [float(reference)]
            
            if len(parsed_nums) >= len(ref_nums):
                # Check if all reference numbers are found
                matches = 0
                for ref_val in ref_nums:
                    for parsed_val in parsed_nums:
                        if abs(parsed_val - ref_val) < 1e-6 * max(abs(ref_val), 1):
                            matches += 1
                            break
                if matches == len(ref_nums):
                    return 1.0
        
        return 0.0
    except (ValueError, TypeError):
        return 0.0


def compute_dyval_metrics(result: DefaultResult) -> dict[str, Any]:
    """
    Compute metrics for a DyVal scenario using native promptbench evaluation.
    
    Args:
        result: DefaultResult with DyVal metadata containing dataset_type
    
    Returns:
        Metrics dict with accuracy from promptbench's native scorer
    """
    metadata = getattr(result, "metadata", None) or {}
    tokens = extract_token_usage(result)
    
    # Handle generation failures
    if metadata.get("generation_failure"):
        return {
            "accuracy": None,
            "generation_failure": True,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 0,
            "timed_out": False,
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }
    
    # Handle timeout
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
    
    # Get reference answer - prefer metadata['answer'] which has original type
    # Fall back to reference_answer (string) if needed
    reference = metadata.get("answer")
    if reference is None:
        reference = getattr(result, "reference_answer", None)
    
    # Dataset type can be in metadata as 'dataset_type' (string) or we need to infer
    dataset_type = metadata.get("dataset_type")
    if not dataset_type:
        # Try to extract from domain field (e.g., "dyval_arithmetic" -> "arithmetic")
        domain = metadata.get("domain", "")
        if domain.startswith("dyval_"):
            dataset_type = domain[6:]  # Remove "dyval_" prefix
    
    if reference is None or dataset_type is None:
        return {
            "accuracy": None,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 1,
            "error": "Missing reference_answer or dataset_type",
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }
    
    # Get raw responses (use the last round)
    raw_responses = result.responses or []
    raw_response = raw_responses[-1] if raw_responses else ""
    
    # Score the response using promptbench's native evaluation
    accuracy = score_dyval_answer(raw_response, reference, dataset_type)
    
    return {
        **metadata,  # Include all metadata (dataset_type, depth, etc.)
        "accuracy": accuracy,
        "brier_score": None,
        "round_scores": [accuracy] if accuracy is not None else [],
        "correct": 1 if accuracy == 1.0 else 0,
        "attempted": 1 if accuracy is not None else 0,
        "missing": 1 if accuracy is None else 0,
        "timed_out": False,
        **tokens,
        "player_1": {
            "player_num": result.player_num,
            "accuracy": accuracy,
            **tokens,
        },
    }


@ComponentRegistry.metrics("dyval")
def compute_dyval_metrics_batch(
    results: list[DefaultResult],
) -> dict[str, dict[str, Any]]:
    """
    Batch metrics computation for DyVal scenarios.
    
    Uses promptbench's native evaluation for accurate scoring of
    arithmetic, boolean logic, and other DyVal problem types.
    """
    return {result.scenario_id: compute_dyval_metrics(result) for result in results}
