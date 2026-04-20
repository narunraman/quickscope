import re
import math
import ast
import operator
from typing import Any

from .schemas import DefaultRow
from ..abstract import (
    ComponentRegistry,
    ResponseHandler
)

# Constants ported from parsing_service.py
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
CURRENCY_SYMBOLS = r"[$¥€£₹₽₩฿₫₴₪₡₲₵₸₹₺₻₼₽₾₿]"
UNIT_SUFFIXES = r"\s*(meters?|m|km|cm|mm|ft|feet|inches?|in|miles?|mi|kg|g|mg|lb|lbs|oz|liters?|l|ml|gallons?|gal|seconds?|s|minutes?|min|hours?|hr|h|days?|weeks?|months?|years?|yr|%|percent|dollars?|cents?|euros?|yen|yuan|rupees?|pounds?)\.?\s*$"

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

@ComponentRegistry.handler("default")
class DefaultResponseHandler(ResponseHandler[DefaultRow, str | None]):
    """
    Handles responses for standard single-shot scenarios.
    Parses \\boxed{} answers and supports numeric/MCQ normalization.
    """

    def parse_response(self, response: str, scenario: DefaultRow) -> str | None:
        """
        Parse the answer from the LLM response.
        Expects \\boxed{answer} format.
        """
        boxed = self._extract_last_boxed(response)
        if not boxed:
            return None

        content = boxed.strip()
        fmt = getattr(scenario, "answer_format", "text")

        if fmt == "numeric":
            try:
                return self._normalize_numeric_answer(content)
            except ValueError:
                return None

        if fmt == "mcq":
            return self._normalize_mcq_answer(
                content, getattr(scenario, "mcq_options", [])
            )

        return content

    def _extract_last_boxed(self, text: str) -> str | None:
        """
        Extract the last \\boxed{...} content, supporting nested braces.
        Returns the inner content or None if not found/unterminated.
        """
        spans = self._find_boxed_spans(text)
        if not spans:
            return None
        _, _, content = spans[-1]
        return content

    def _find_boxed_spans(self, text: str) -> list[tuple[int, int, str]]:
        """
        Find all \\boxed{...} spans with nested-brace support.
        Returns (start, end, content) where start/end are content indices.
        """
        results = []
        token = "\\boxed{"
        i = 0
        while True:
            start = text.find(token, i)
            if start == -1:
                break
            j = start + len(token)
            depth = 1
            k = j
            while k < len(text) and depth > 0:
                char = text[k]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                k += 1
            if depth == 0:
                end = k - 1
                content = text[j:end]
                results.append((j, end, content))
                i = k
            else:
                break
        return results

    def _normalize_mcq_answer(self, content: str, options: list[str]) -> str | None:
        normalized = content.strip()

        # Letter index check (A, B, C...)
        if len(normalized) == 1 and normalized.isalpha() and options:
            idx = ord(normalized.upper()) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]

        # Option text exact match
        for opt in options:
            if normalized.casefold() == opt.casefold():
                return opt

        # Fallback if no matching options provided or found
        return normalized if not options else None

    def _normalize_numeric_answer(self, content: str) -> str:
        """Ported robust numeric normalization."""
        original = content

        # Strip currency and units
        content = re.sub(CURRENCY_SYMBOLS, "", content)
        content = re.sub(UNIT_SUFFIXES, "", content, flags=re.IGNORECASE)
        content = content.replace(",", "").strip()

        if not content:
            raise ValueError(f"Empty after normalization: {original}")

        # Try simple float
        try:
            float(content)
            return content
        except ValueError:
            pass

        # Handle " = result"
        if "=" in content:
            last_part = content.split("=")[-1].strip()
            # clean units again from last part
            clean_last = re.sub(
                UNIT_SUFFIXES, "", last_part, flags=re.IGNORECASE
            ).strip()
            try:
                float(clean_last)
                return clean_last
            except ValueError:
                pass

        # Eval expression
        try:
            expr_content = content.split("=")[0].strip() if "=" in content else content
            val = self._evaluate_numeric_expression(expr_content)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except (ValueError, SyntaxError):
            pass

        # Last resort: find last number
        numbers = re.findall(r"-?\d+\.?\d*", content)
        if numbers:
            return numbers[-1]

        raise ValueError(f"Cannot extract number from: {original}")

    def _evaluate_numeric_expression(self, expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        return self._safe_eval_ast(tree.body)

    def _safe_eval_ast(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            if type(node.op) in SAFE_OPERATORS:
                left = self._safe_eval_ast(node.left)
                right = self._safe_eval_ast(node.right)
                return SAFE_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) in SAFE_OPERATORS:
                operand = self._safe_eval_ast(node.operand)
                return SAFE_OPERATORS[type(node.op)](operand)
        raise ValueError(f"Unsupported AST node: {type(node)}")

    # =========================================================================
    #  Advanced Parsing Utilities (Ported from parsing_service.py)
    # =========================================================================

    def _normalize_logprobs(self, logprobs_input: Any) -> list[Any] | None:
        """
        Normalize various logprobs formats into a list of token objects/dicts.
        """
        if logprobs_input is None:
            return None

        content = None
        if hasattr(logprobs_input, "content"):
            content = getattr(logprobs_input, "content")
        elif isinstance(logprobs_input, dict):
            content = logprobs_input.get("content")
        elif isinstance(logprobs_input, list):
            content = logprobs_input

        if content is None:
            return None

        normalized = []
        for item in content:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            elif hasattr(item, "dict"):
                normalized.append(item.dict())
            else:
                try:
                    normalized.append(item.__dict__)
                except AttributeError:
                    normalized.append(item)
        return normalized

    def extract_mcq_distribution(
        self,
        logprobs: list[dict[str, Any]],
        parsed_answer: str,
        mcq_options: list[str],
        response_text: str,
    ) -> dict[str, float] | None:
        """
        Extract probability distribution for MCQ options from the relevant token's logprobs.
        Uses boxed indices to locate the answer token.
        """
        boxed_indices, _ = self.extract_boxed_token_indices(response_text, logprobs)

        if not boxed_indices or not mcq_options:
            return None

        # 1. Identify valid options set
        valid_options = set(opt.upper() for opt in mcq_options)

        # 2. Find the target token index
        target_idx = -1
        parsed_upper = parsed_answer.upper()

        for idx in boxed_indices:
            if idx >= len(logprobs):
                continue
            token_data = logprobs[idx]
            token_str = (
                token_data.get("token", "")
                if isinstance(token_data, dict)
                else getattr(token_data, "token", "")
            )

            if token_str.strip().upper() == parsed_upper:
                target_idx = idx

        if target_idx == -1:
            target_idx = boxed_indices[-1]

        if target_idx < 0 or target_idx >= len(logprobs):
            return None

        # 3. Extract from top_logprobs
        target_token_data = logprobs[target_idx]
        top_logprobs = (
            target_token_data.get("top_logprobs", [])
            if isinstance(target_token_data, dict)
            else getattr(target_token_data, "top_logprobs", []) or []
        )

        if not top_logprobs:
            return None

        option_logprobs = {opt: -999.0 for opt in valid_options}
        found_any = False

        for item in top_logprobs:
            # Handle dict/object duality
            if isinstance(item, dict):
                t_str = item.get("token", "")
                lp = item.get("logprob", -999.0)
            else:
                t_str = getattr(item, "token", "")
                lp = getattr(item, "logprob", -999.0)

            t_norm = t_str.strip().upper()
            if t_norm in valid_options:
                option_logprobs[t_norm] = max(option_logprobs[t_norm], lp)
                found_any = True

        if not found_any:
            return None

        # Exp and Normalize
        probs = {}
        total_p = 0.0
        for opt, lp in option_logprobs.items():
            if lp > -990:
                p = math.exp(lp)
                probs[opt] = p
                total_p += p
            else:
                probs[opt] = 0.0

        if total_p > 0:
            for opt in probs:
                probs[opt] /= total_p

        return probs

    def extract_boxed_token_indices(
        self,
        response_text: str,
        logprobs: list[Any],
    ) -> tuple[list[int], str | None]:
        spans = self._find_boxed_spans(response_text)
        if not spans:
            return [], None

        start_char, end_char, boxed_content = spans[-1]

        current_char_pos = 0
        token_indices = []

        for idx, token_data in enumerate(logprobs):
            token_str = (
                token_data.get("token", "")
                if isinstance(token_data, dict)
                else getattr(token_data, "token", "")
            )
            token_len = len(token_str)
            token_start = current_char_pos
            token_end = current_char_pos + token_len

            if token_end > start_char and token_start < end_char:
                token_indices.append(idx)

            current_char_pos += token_len
            if current_char_pos >= len(response_text) + 100:
                break

        return token_indices, boxed_content
