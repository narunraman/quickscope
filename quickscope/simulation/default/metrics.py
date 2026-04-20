"""
Standard domain metrics.
"""

import math
import re
from typing import Any

from sympy import sympify, simplify, SympifyError, N
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from .schemas import DefaultResult
from ..abstract import BaseResult, ComponentRegistry


# =============================================================================
# Functional Equivalence Utilities
# =============================================================================


def _normalize_division_notation(expr: str) -> str:
    """
    Normalize economics-style division notation to explicit parentheses.

    Converts patterns like "0.65y/0.9x" to "(0.65*y)/(0.9*x)" so that
    the division is interpreted correctly as a ratio of products.

    The pattern handles:
    - "coeff*var/coeff*var" -> "(coeff*var)/(coeff*var)"
    - "coeff var/coeff var" -> "(coeff*var)/(coeff*var)"  (implicit mult)
    """
    # Pattern: number (optionally with implicit mult to variable) / number (optionally with implicit mult to variable)
    # Examples: 0.65y/0.9x, 0.65*y/0.9*x, 13y/18x
    # We want to wrap numerator and denominator in parentheses

    # Match: (numerator)/(denominator) where each side is a product of numbers and variables
    # This regex finds division where both sides look like coefficient*variable products
    pattern = r"(\d+\.?\d*\*?[a-zA-Z]+(?:\*\d+\.?\d*\*?[a-zA-Z]*)*)\s*/\s*(\d+\.?\d*\*?[a-zA-Z]+(?:\*\d+\.?\d*\*?[a-zA-Z]*)*)"

    def add_parens(match):
        num, denom = match.groups()
        return f"({num})/({denom})"

    return re.sub(pattern, add_parens, expr)


def _parse_expression(expr: str):
    """
    Parse a mathematical expression string into a SymPy expression.
    Handles both plain text (e.g., "0.65y/0.9x") and LaTeX (e.g., "\\frac{0.65y}{0.9x}").

    Returns:
        SymPy expression, or None if parsing fails.
    """
    if not expr:
        return None

    expr = expr.strip()

    # Try LaTeX first if it looks like LaTeX
    if "\\" in expr or ("{" in expr and "}" in expr):
        try:
            return parse_latex(expr)
        except Exception:
            # Fall back to a lightweight LaTeX->sympy normalization
            expr = _latex_to_sympy_fallback(expr)

    # Normalize economics-style division notation (e.g., "0.65y/0.9x" -> "(0.65y)/(0.9x)")
    expr = _normalize_division_notation(expr)

    # Use parse_expr with implicit multiplication support
    # This handles expressions like "0.65y" -> "0.65*y"
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    try:
        return parse_expr(expr, transformations=transformations)
    except (SympifyError, SyntaxError, TypeError, Exception):
        pass

    # Last resort: try basic sympify
    try:
        return sympify(expr)
    except (SympifyError, SyntaxError, TypeError):
        return None


def _latex_to_sympy_fallback(expr: str) -> str:
    """
    Best-effort normalization for simple LaTeX-style math when parse_latex fails.
    Handles exponent braces and a few common LaTeX commands.
    """
    s = expr
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\^\{([^{}]+)\}", r"^\1", s)
    # Remove remaining braces and backslashes
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    return s


def _is_tuple_or_list(s: str) -> bool:
    """Check if string looks like a tuple (a, b) or list [a, b]."""
    s = s.strip()
    return (s.startswith("(") and s.endswith(")")) or (
        s.startswith("[") and s.endswith("]")
    )


def _parse_tuple_elements(s: str) -> list[str]:
    """
    Parse a tuple/list string like "(a, b)" or "[a, b]" into element strings ["a", "b"].
    Handles nested parentheses/brackets within elements.
    """
    s = s.strip()

    # Remove outer parentheses or brackets if present
    if (s.startswith("(") and s.endswith(")")) or (
        s.startswith("[") and s.endswith("]")
    ):
        s = s[1:-1]

    # Split by comma, but respect nested parentheses/brackets
    elements = []
    current = []
    depth = 0

    for char in s:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            elements.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        elements.append("".join(current).strip())

    return [e for e in elements if e]


def _expressions_equivalent(expr1: str, expr2: str) -> bool:
    """
    Check if two mathematical expressions are symbolically equivalent.

    Examples:
        "0.65y/0.9x" ≡ "13y/18x" -> True
        "0.65y/0.9x" ≡ "0.7222y/x" -> True (approximately)
        "114863.85" ≡ "114863.850001" -> True (numeric tolerance)
    """
    sym1 = _parse_expression(expr1)
    sym2 = _parse_expression(expr2)

    if sym1 is None or sym2 is None:
        # Fall back to string comparison if parsing fails
        return expr1.strip().lower() == expr2.strip().lower()

    try:
        # Check if difference simplifies to zero
        diff = simplify(sym1 - sym2)  # type: ignore[operator]
        if diff == 0:
            return True

        # For pure numeric values, use tolerance-based comparison
        # This handles floating point precision issues
        # For pure numeric results, use tolerance-based comparison
        # to handle floating point precision issues
        # Note: SymPy's type stubs are incomplete, but these operations are valid at runtime
        if diff.is_number and sym1.is_number and sym2.is_number:
            try:
                # N() evaluates to float-precision
                val1 = float(N(sym1))
                val2 = float(N(sym2))
                return math.isclose(val1, val2, rel_tol=1e-6, abs_tol=1e-9)
            except (TypeError, ValueError):
                diff_val = float(N(diff))
                return abs(diff_val) < 1e-9

        return False
    except Exception:
        return False


def _check_functional_equivalence(parsed: str, reference: str) -> bool:
    """
    Check functional equivalence between parsed answer and reference.

    Handles:
    - Single expressions: "0.65y/0.9x" vs "13y/18x"
    - Tuples (order-insensitive): "(a, b)" matches "(b, a)"

    Returns:
        True if functionally equivalent, False otherwise.
    """
    parsed = str(parsed).strip()
    reference = str(reference).strip()

    # Check if both are tuples/lists (handles both () and [])
    parsed_is_tuple = _is_tuple_or_list(parsed)
    ref_is_tuple = _is_tuple_or_list(reference)

    if parsed_is_tuple and ref_is_tuple:
        # Parse tuple elements
        parsed_elements = _parse_tuple_elements(parsed)
        ref_elements = _parse_tuple_elements(reference)

        if len(parsed_elements) != len(ref_elements):
            return False

        # Order-insensitive matching: each parsed element must match some ref element
        matched = [False] * len(ref_elements)
        for p_elem in parsed_elements:
            found_match = False
            for i, r_elem in enumerate(ref_elements):
                if not matched[i] and _expressions_equivalent(p_elem, r_elem):
                    matched[i] = True
                    found_match = True
                    break
            if not found_match:
                return False
        return True

    elif not parsed_is_tuple and not ref_is_tuple:
        # Both are single expressions
        return _expressions_equivalent(parsed, reference)

    else:
        # One is tuple, one is not - not equivalent
        return False


def extract_token_usage(result: BaseResult) -> dict[str, int]:
    """
    Extract actual token usage from an evaluation result.

    Args:
        result: Evaluation result (BaseResult or subclass)

    Returns:
        dictionary with prompt_tokens, completion_tokens, total_tokens
    """
    if hasattr(result, "llm_responses") and result.llm_responses:
        # Sum tokens across all API calls for this result
        total_input = 0
        total_output = 0
        total_reasoning = 0
        total_tokens = 0

        for response in result.llm_responses:
            if isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
                if not isinstance(usage, dict):
                    continue

                input_tokens = (
                    usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                )
                output_tokens = (
                    usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                )
                reasoning_tokens = (
                    usage.get("reasoning_tokens", usage.get("thinking_tokens", 0)) or 0
                )
                entry_total = usage.get("total_tokens", 0) or 0

                if entry_total == 0:
                    entry_total = input_tokens + output_tokens + reasoning_tokens

                total_input += int(input_tokens)
                total_output += int(output_tokens)
                total_reasoning += int(reasoning_tokens)
                total_tokens += int(entry_total)

        generation_tokens = total_output + total_reasoning

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "reasoning_tokens": total_reasoning,
            "generation_tokens": generation_tokens,
            "total_tokens": total_tokens,
            # Backwards-compatible aliases
            "prompt_tokens": total_input,
            "completion_tokens": total_output,
        }
    else:
        # Fallback: return zero
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "generation_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def score_standard_answer(
    parsed_answer: Any, reference: str | list[str], fmt: str
) -> float | None:
    """
    Score a single standard answer.

    Returns:
        1.0 if correct, 0.0 if incorrect, None if parsed_answer is missing.
    """
    if reference is None:
        raise ValueError("reference_answer is required for standard scenario scoring")

    if parsed_answer is None:
        return None

    refs = reference if isinstance(reference, list) else [reference]

    if fmt == "numeric":
        try:
            parsed_val = float(parsed_answer)
            for ref in refs:
                if math.isclose(parsed_val, float(ref), rel_tol=1e-6, abs_tol=1e-9):
                    return 1.0
        except (ValueError, TypeError):
            pass
        return 0.0

    if fmt == "functional":
        # Check functional/symbolic equivalence (order-insensitive for tuples)
        try:
            for ref in refs:
                if _check_functional_equivalence(parsed_answer, ref):
                    return 1.0
        except Exception:
            pass
        return 0.0

    if fmt == "mcq":
        parsed_val = str(parsed_answer).upper().strip()
        norm_refs = {str(r).upper().strip() for r in refs}
        return 1.0 if parsed_val in norm_refs else 0.0

    parsed_val = str(parsed_answer).strip().casefold()
    norm_refs = {str(r).strip().casefold() for r in refs}
    return 1.0 if parsed_val in norm_refs else 0.0


def calculate_brier_score(
    prob_distribution: dict[str, float] | None, reference: str | list[str]
) -> float | None:
    """
    Calculate Brier Score for a single answer distribution.

    Formula: sum((prob_i - outcome_i)^2) over all classes.
    Range: [0.0, 2.0]
    """
    if not prob_distribution or reference is None:
        return None

    # Normalize reference(s) to uppercase keys
    refs = reference if isinstance(reference, list) else [reference]
    norm_refs = {str(r).strip().upper() for r in refs}

    score = 0.0
    for option, prob in prob_distribution.items():
        norm_opt = str(option).strip().upper()
        is_correct = 1.0 if norm_opt in norm_refs else 0.0
        score += (prob - is_correct) ** 2

    return score


def compute_standard_metrics(result: DefaultResult) -> dict[str, Any]:
    """
    Compute metrics for a standard single-shot scenario.

    Special cases:
    - generation_failure: Scenario generation failed (e.g., promptbench error).
          Returns accuracy=None with generation_failure=True flag.
          Utility functions should check this flag and decide how to handle.
    - reference_answer is None: Can't score, returns accuracy=None.
    """
    reference = getattr(result, "reference_answer", None)
    metadata = getattr(result, "metadata", None) or {}

    # Handle generation failures (e.g., DyVal promptbench numpy errors)
    # LLM was never called - return None accuracy with flag for utility to handle
    if metadata.get("generation_failure") or reference == "__GENERATION_FAILURE__":
        tokens = extract_token_usage(result)  # Will be zeros
        return {
            "accuracy": None,
            "generation_failure": True,  # Flag for utility functions
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 0,
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    if reference is None:
        # Can't score without reference
        # But we can still return token usage
        tokens = extract_token_usage(result)
        return {
            "accuracy": None,
            "effective_accuracy": None,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 0,
            **tokens,  # expand token usage fields
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    fmt = getattr(result, "answer_format", "text")
    parsed_answers = result.parsed_responses or []

    round_scores = []
    # Ensure we account for all rounds
    for i in range(result.num_rounds):
        ans = parsed_answers[i] if i < len(parsed_answers) else None
        round_scores.append(score_standard_answer(ans, reference, fmt))

    # correct = sum(1 for s in round_scores if s == 1.0)
    # attempted = sum(1 for s in round_scores if s >= 0)
    # missing = sum(1 for s in round_scores if s < 0)

    # accuracy = correct / result.num_rounds if result.num_rounds > 0 else None
    # effective_accuracy = (correct / attempted) if attempted > 0 else None

    # NOTE: For right now, only consider the final round for accuracy
    final_score = round_scores[-1] if round_scores else None
    correct = 1 if final_score == 1.0 else 0
    attempted = 1 if final_score is not None else 0
    missing = 1 if final_score is None else 0

    accuracy = float(final_score) if final_score is not None else None

    tokens = extract_token_usage(result)

    # Compute Brier Score (using final round)
    brier_score = None
    if result.mcq_distribution and result.num_rounds > 0:
        # Get distribution from the final round
        # mcq_distribution is a list of dicts or Nones
        final_dist = (
            result.mcq_distribution[-1]
            if len(result.mcq_distribution) >= result.num_rounds
            else None
        )
        if final_dist:
            brier_score = calculate_brier_score(final_dist, reference)

    return {
        # Include scenario metadata (e.g., depth, num_children_per_node for DyVal)
        # so utility functions can access scenario-specific parameters
        **metadata,
        # Computed metrics (may override metadata keys if same name)
        "accuracy": accuracy,
        "brier_score": brier_score,
        "round_scores": round_scores,
        "correct": correct,
        "attempted": attempted,
        "missing": missing,
        "timed_out": getattr(result, "timed_out", False),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "player_1": {
            "player_num": result.player_num,
            "accuracy": accuracy,
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
        },
    }


@ComponentRegistry.metrics("default")
def compute_default_metrics_batch(
    results: list[DefaultResult],
) -> dict[str, dict[str, Any]]:
    """Batch metrics for default scenarios."""
    return {r.scenario_id: compute_standard_metrics(r) for r in results}
