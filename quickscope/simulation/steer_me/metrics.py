"""
SteerMe-specific metrics.

Uses numeric matching inside \\boxed{...} to handle verbose answers that still
contain the correct numeric option.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..abstract import ComponentRegistry
from ..default.schemas import DefaultResult
from ..default.metrics import calculate_brier_score, extract_token_usage, score_standard_answer

_NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_PERCENT_PATTERN = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*%")
_NEGATIVE_WORDS = re.compile(
    r"\b(decrease|reduce|reduction|decline|drop|fall|lower|less)\b", re.IGNORECASE
)
_POSITIVE_WORDS = re.compile(
    r"\b(increase|rise|growth|grow|higher|more|expand)\b", re.IGNORECASE
)
_EXPR_CHARS = re.compile(r"[A-Za-z0-9_\\^*/+\-.,()\[\]{}= ]+")
_MU_X_PATTERN = re.compile(r"MU_x\s*&?=\s*([^\\\\]+)", re.IGNORECASE)
_MU_Y_PATTERN = re.compile(r"MU_y\s*&?=\s*([^\\\\]+)", re.IGNORECASE)
# Alternative patterns for "Marginal utility for/of X: VALUE"
_MU_ALT_PATTERN = re.compile(
    r"Marginal\s+utility\s+(?:for|of)\s+[^:]+:\s*}?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE
)
# Match LaTeX \frac{num}{denom} patterns
_LATEX_FRAC_PATTERN = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
# Match coefficient*var/coefficient*var patterns like "0.2y/0.67x" or "20y/67x"
_MRS_COEF_PATTERN = re.compile(r"(-?[\d.]+)\s*[a-zA-Z]\s*/\s*(-?[\d.]+)\s*[a-zA-Z]")
# Match LaTeX exponent braces ^{...}
_LATEX_EXPONENT_BRACE = re.compile(r"\^\{([^}]+)\}")

_ELEMENT_MODES: dict[str, str] = {
    # numeric scalar outputs
    "aggregation_of_consumer_demand": "numeric",
    "capital_market_distortions": "numeric",
    "consumer_surplus": "numeric",
    "deadweight_loss": "numeric",
    "deriving_hicksian_demand": "numeric",
    "efficient_surplus": "numeric",
    "find_eq_price": "numeric",
    "labor_supply_elasticity": "numeric",
    "output_elasticity": "numeric",
    "price_elasticity": "numeric",
    "price_of_risk": "numeric",
    "producer_surplus": "numeric",
    "profit_maximization": "numeric",
    "supply_elasticity": "numeric",
    "tfp_shocks": "numeric",
    # tuple numeric outputs
    "intertemporal_consumption_smoothing": "tuple_numeric",
    "labor_supply_distortions": "tuple_numeric",
    "state_based_consumption": "tuple_numeric",
    # expression outputs
    "aggregation_of_labor_demand": "expression",
    "deriving_demand": "expression",
    "deriving_labor_supply": "expression",
    "marginal_production": "expression",
    # tuple expression outputs
    "marginal_utility": "tuple_expression",
    "mrs_utility": "tuple_expression",
    # default fallback
}


_SIMPLE_FRACTION_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)")


def _extract_numbers(text: str | None) -> list[float]:
    if not text:
        return []
    numbers = _NUMBER_PATTERN.findall(text)
    result = [float(n.replace(",", "")) for n in numbers]
    
    # Also evaluate simple fractions like "93/19" → 4.89...
    for match in _SIMPLE_FRACTION_PATTERN.finditer(text):
        try:
            num = float(match.group(1))
            denom = float(match.group(2))
            if denom != 0:
                result.append(num / denom)
        except ValueError:
            pass
    
    return result


def _extract_percent_numbers(text: str | None) -> list[float]:
    if not text:
        return []
    numbers = _PERCENT_PATTERN.findall(text)
    return [float(n.replace(",", "")) for n in numbers]


def _infer_sign(text: str | None) -> int:
    if not text:
        return 0
    has_neg = _NEGATIVE_WORDS.search(text) is not None
    has_pos = _POSITIVE_WORDS.search(text) is not None
    if has_neg and not has_pos:
        return -1
    if has_pos and not has_neg:
        return 1
    return 0


def _element_mode(element: str | None) -> str:
    return _ELEMENT_MODES.get((element or "").lower(), "numeric")


def _extract_expression_candidate(text: str) -> str:
    matches = _EXPR_CHARS.findall(text)
    if not matches:
        return text.strip()
    # Prefer the longest math-like substring
    candidate = max(matches, key=len).strip()
    return candidate


def _normalize_latex_expr(text: str) -> str:
    """Normalize LaTeX expressions for comparison.
    
    - Converts ^{exp} to ^exp (remove braces around exponents)
    - Strips whitespace
    """
    normalized = _LATEX_EXPONENT_BRACE.sub(r"^\1", text)
    return normalized.strip()


def _extract_marginal_utility_tuple(text: str) -> str | None:
    # Try MU_x / MU_y patterns first
    mx = _MU_X_PATTERN.search(text)
    my = _MU_Y_PATTERN.search(text)
    if mx and my:
        expr_x = mx.group(1).strip()
        expr_y = my.group(1).strip()
        return f"({expr_x}, {expr_y})"
    
    # Fallback: "Marginal utility for/of X: VALUE" patterns
    alt_matches = _MU_ALT_PATTERN.findall(text)
    if len(alt_matches) >= 2:
        return f"({alt_matches[0]}, {alt_matches[1]})"
    
    return None


def _extract_mrs_coefficient(text: str) -> float | None:
    """
    Extract the numeric coefficient ratio from an MRS expression.
    
    Handles:
    - LaTeX: \\frac{20y}{67x} → 20/67
    - Plain: 0.2y/0.67x → 0.2/0.67
    - Plain: 20y/67x → 20/67
    """
    # First try LaTeX \frac{num}{denom}
    frac_match = _LATEX_FRAC_PATTERN.search(text)
    if frac_match:
        num_str, denom_str = frac_match.groups()
        # Extract numbers from numerator and denominator
        num_nums = _NUMBER_PATTERN.findall(num_str)
        denom_nums = _NUMBER_PATTERN.findall(denom_str)
        if num_nums and denom_nums:
            try:
                num = float(num_nums[0].replace(",", ""))
                denom = float(denom_nums[0].replace(",", ""))
                if denom != 0:
                    return num / denom
            except ValueError:
                pass
    
    # Try plain coefficient pattern like "0.2y/0.67x"
    coef_match = _MRS_COEF_PATTERN.search(text)
    if coef_match:
        try:
            num = float(coef_match.group(1))
            denom = float(coef_match.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass
    
    return None


def _mrs_coefficients_match(answer_coef: float, ref_coef: float) -> bool:
    """Check if MRS coefficients match (allow for reciprocal since MRS_xy = 1/MRS_yx)."""
    if _numbers_match(answer_coef, ref_coef):
        return True
    # Also check reciprocal
    if ref_coef != 0 and _numbers_match(answer_coef, 1.0 / ref_coef):
        return True
    return False


def _score_mrs_utility(parsed_answer: str, reference: str | list[str]) -> float | None:
    """
    Score MRS utility answers by comparing coefficient ratios.
    
    The reference is typically a tuple like "(0.2y/0.67x, 0.67x/0.2y)".
    The answer might be a single MRS expression.
    """
    answer_coef = _extract_mrs_coefficient(parsed_answer)
    if answer_coef is None:
        return None  # Fall back to standard scoring
    
    refs = reference if isinstance(reference, list) else [reference]
    # Parse the tuple if it's a single string like "(expr1, expr2)"
    if len(refs) == 1 and "," in refs[0]:
        # Split tuple string into parts
        ref_str = refs[0].strip("() ")
        refs = [r.strip() for r in ref_str.split(",")]
    
    for ref in refs:
        ref_coef = _extract_mrs_coefficient(ref)
        if ref_coef is not None:
            if _mrs_coefficients_match(answer_coef, ref_coef):
                return 1.0
    
    return 0.0


def _ensure_tuple_text(text: str, numbers: list[float]) -> str:
    if ("," in text) and ("(" in text or "[" in text):
        return text
    if len(numbers) >= 2:
        a, b = numbers[-2], numbers[-1]
        return f"({a}, {b})"
    return text


def _numbers_match(a: float, b: float) -> bool:
    # SteerMe numeric options are often rounded; allow 1% tolerance.
    return math.isclose(a, b, rel_tol=1e-2, abs_tol=5e-3)


def _any_match(answer_numbers: list[float], option_numbers: list[float]) -> bool:
    for ans in answer_numbers:
        for opt in option_numbers:
            if _numbers_match(ans, opt):
                return True
    return False


def _tuple_numbers_match(answer_numbers: list[float], ref_numbers: list[float]) -> bool:
    if not answer_numbers or not ref_numbers:
        return False
    if len(ref_numbers) == 1:
        return _any_match(answer_numbers, ref_numbers)
    if len(answer_numbers) < len(ref_numbers):
        return False
    # For 2-number references, check any pair from answer numbers (order-insensitive).
    if len(ref_numbers) == 2:
        r1, r2 = ref_numbers
        for i in range(len(answer_numbers)):
            for j in range(i + 1, len(answer_numbers)):
                a1, a2 = answer_numbers[i], answer_numbers[j]
                if (_numbers_match(a1, r1) and _numbers_match(a2, r2)) or (
                    _numbers_match(a1, r2) and _numbers_match(a2, r1)
                ):
                    return True
        return False
    # For larger tuples, require all reference numbers to be present.
    return all(_any_match(answer_numbers, [ref_num]) for ref_num in ref_numbers)


def score_steer_me_answer(
    parsed_answer: Any,
    reference: str | list[str],
    fmt: str,
    mcq_options: list[str] | None,
    element: str | None,
) -> float | None:
    """
    Score SteerMe answers using numeric option matching inside \\boxed{...}.

    A response is correct if:
    - It contains at least one number that matches any number in the reference option, AND
    - It does not contain any number that matches a non-reference option,
      unless that number also appears in the reference option.
    """
    if reference is None:
        raise ValueError("reference_answer is required for steer_me scoring")
    if parsed_answer is None:
        return None

    element_key = (element or "").lower()
    mode = _element_mode(element_key)

    # Expression-based elements: clean and use functional scoring.
    if mode in {"expression", "tuple_expression"}:
        raw_text = str(parsed_answer)
        
        # MRS utility: use coefficient-based matching
        if element_key == "mrs_utility":
            mrs_score = _score_mrs_utility(raw_text, reference)
            if mrs_score is not None:
                return mrs_score
            # Fall through to standard scoring if coefficient extraction failed
        
        if element_key == "marginal_utility":
            answer_text = _extract_marginal_utility_tuple(raw_text) or raw_text
        else:
            answer_text = raw_text
        answer_text = _extract_expression_candidate(answer_text)
        answer_numbers = _extract_numbers(answer_text)
        if mode == "tuple_expression":
            answer_text = _ensure_tuple_text(answer_text, answer_numbers)
        
        # Normalize LaTeX formatting (e.g., ^{-0.45} → ^-0.45)
        answer_text = _normalize_latex_expr(answer_text)
        ref_normalized = _normalize_latex_expr(str(reference))
        
        # Try functional scoring first
        func_score = score_standard_answer(answer_text, ref_normalized, "functional")
        if func_score == 1.0:
            return 1.0
        
        # Fallback: check if reference expression is contained in answer
        if ref_normalized in answer_text:
            return 1.0
        
        return func_score

    # For numeric modes, we can score against the reference even without MCQ options.
    if mcq_options is None:
        mcq_options = []

    answer_text = str(parsed_answer)
    answer_numbers = _extract_numbers(answer_text)
    percent_numbers = _extract_percent_numbers(answer_text)
    sign_hint = _infer_sign(answer_text)

    references = reference if isinstance(reference, list) else [reference]

    for ref in references:
        ref_numbers = _extract_numbers(str(ref))
        if not ref_numbers:
            continue
        option_numbers = []
        for opt in mcq_options:
            option_numbers.extend(_extract_numbers(opt))

        normalized = set(answer_numbers)

        # Percent-to-decimal normalization when options are fractional.
        if percent_numbers:
            all_comparison_numbers = option_numbers + ref_numbers
            if all_comparison_numbers and max(abs(x) for x in all_comparison_numbers) <= 1.5:
                for value in percent_numbers:
                    normalized.add(value / 100.0)

        # Sign inference when reference or options contain negative numbers.
        all_comparison_numbers = option_numbers + ref_numbers
        has_negative_reference = any(x < 0 for x in all_comparison_numbers)
        if has_negative_reference and sign_hint == -1:
            for value in list(normalized):
                if value > 0:
                    normalized.add(-value)

        answer_numbers = list(normalized)

        if mode == "tuple_numeric":
            if not _tuple_numbers_match(answer_numbers, ref_numbers):
                continue
        else:
            if not _any_match(answer_numbers, ref_numbers):
                continue

        # If the answer is verbose (many numbers), prefer the reference match.
        if len(answer_numbers) > len(ref_numbers) + 1:
            return 1.0

        # Ensure no numbers from other options appear in the answer,
        # unless they also appear in the reference option.
        if mcq_options:
            has_conflict = False
            for option in mcq_options:
                if option in references:
                    continue
                option_numbers = _extract_numbers(option)
                for opt_num in option_numbers:
                    if _any_match(answer_numbers, [opt_num]) and not _any_match(
                        ref_numbers, [opt_num]
                    ):
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if not has_conflict:
                return 1.0
        else:
            return 1.0

    return 0.0


def compute_steer_me_metrics(result: DefaultResult) -> dict[str, Any]:
    reference = getattr(result, "reference_answer", None)
    metadata = getattr(result, "metadata", None) or {}

    if metadata.get("generation_failure") or reference == "__GENERATION_FAILURE__":
        tokens = extract_token_usage(result)
        return {
            "accuracy": None,
            "generation_failure": True,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 0,
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    if reference is None:
        tokens = extract_token_usage(result)
        return {
            "accuracy": None,
            "effective_accuracy": None,
            "brier_score": None,
            "round_scores": [],
            "correct": 0,
            "attempted": 0,
            "missing": 0,
            **tokens,
            "player_1": {"player_num": result.player_num, "accuracy": None, **tokens},
        }

    fmt = getattr(result, "answer_format", "text")
    mcq_options = getattr(result, "mcq_options", None)
    element = metadata.get("element")
    parsed_answers = result.parsed_responses or []

    round_scores = []
    for i in range(result.num_rounds):
        ans = parsed_answers[i] if i < len(parsed_answers) else None
        round_scores.append(
            score_steer_me_answer(ans, reference, fmt, mcq_options, element)
        )

    final_score = round_scores[-1] if round_scores else None
    correct = 1 if final_score == 1.0 else 0
    attempted = 1 if final_score is not None else 0
    missing = 1 if final_score is None else 0

    accuracy = float(final_score) if final_score is not None else None

    tokens = extract_token_usage(result)

    brier_score = None
    if result.mcq_distribution and result.num_rounds > 0:
        final_dist = (
            result.mcq_distribution[-1]
            if len(result.mcq_distribution) >= result.num_rounds
            else None
        )
        if final_dist:
            brier_score = calculate_brier_score(final_dist, reference)

    return {
        **metadata,
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


@ComponentRegistry.metrics("steer_me")
def compute_steer_me_metrics_batch(
    results: list[DefaultResult],
) -> dict[str, dict[str, Any]]:
    return {r.scenario_id: compute_steer_me_metrics(r) for r in results}
