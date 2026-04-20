"""
LLM interface and implementations.
"""

import os
import re
import threading
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from openai import OpenAI



from .model_registry import ModelRegistry
from ..dataflow.logging_config import get_logger
from ..simulation.abstract.schemas import BaseEvaluationConfig as EvaluationConfig

# Setup loggers
api_logger = get_logger("api")
parsing_logger = get_logger("parsing")
response_logger = get_logger("responses")


@dataclass
class GenerationConfig:
    """Minimal configuration needed for ad-hoc generations."""

    model_name: str
    temperature: float | None = None  # None = let LLMService/provider decide
    max_tokens: int | None = None
    system_prompt: str | None = None
    logprobs: bool = False
    top_logprobs: int | None = None


ConfigType = EvaluationConfig | GenerationConfig


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    supports_batching: bool = False

    def load_model(self, model_name: str) -> None:
        """Optional hook for providers that need warmup/ensure_ready."""
        return None

    @abstractmethod
    def generate_response(self, prompt: str, config: ConfigType) -> str:
        """
        Generate a response to the given prompt.

        Args:
            prompt: The input prompt
            config: Evaluation configuration

        Returns:
            Generated response string
        """
        pass

    def generate_response_with_metadata(
        self, prompt: str, config: ConfigType
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Generate a response with metadata (e.g., token usage).

        Default implementation calls generate_response and returns None metadata.
        Subclasses can override to return actual metadata.

        Args:
            prompt: The input prompt
            config: Evaluation configuration

        Returns:
            tuple of (response_string, metadata_dict)
        """
        response = self.generate_response(prompt, config)
        return response, None

    def generate_batch(
        self,
        prompts: list[str],
        configs: Sequence[ConfigType],
        status_callback: Callable[[int, int], None] | None = None,
        per_request_callback: Callable[[int, tuple[str, dict[str, Any] | None]], None] | None = None,
        request_timeout: int | None = None,
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """
        Generate responses for multiple prompts in batch (parallel execution where supported).

        Default implementation falls back to sequential calls.
        Subclasses can override to implement true batching/parallelization.

        Args:
            prompts: List of input prompts
            configs: List of evaluation configurations (one per prompt)
            status_callback: Optional callback(completed, total) called after each request
            per_request_callback: Optional callback(idx, (response_text, metadata)) called
                                  immediately after each request completes. Useful for
                                  incremental saving of results. Must be thread-safe.
            request_timeout: Per-request timeout in seconds (None = no timeout).
                            Subclasses that support timeouts should use this.

        Returns:
            List of (response_string, metadata_dict) tuples in same order as prompts

        Raises:
            ValueError: If len(prompts) != len(configs)
        """
        if len(prompts) != len(configs):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match number of configs ({len(configs)})"
            )

        # Default: sequential execution
        results = []
        for i, (prompt, config) in enumerate(zip(prompts, configs)):
            result = self.generate_response_with_metadata(prompt, config)
            results.append(result)
            if per_request_callback:
                per_request_callback(i, result)
            if status_callback:
                status_callback(i + 1, len(prompts))
        return results

    def _generate_batch_threaded(
        self,
        prompts: list[str],
        configs: Sequence[ConfigType],
        *,
        max_workers: int,
        generate_one: Callable[[int, str, ConfigType], tuple[str, dict[str, Any] | None]],
        status_callback: Callable[[int, int], None] | None = None,
        per_request_callback: Callable[[int, tuple[str, dict[str, Any] | None]], None] | None = None,
        request_timeout: int | None = None,
        log_label: str = "Request",
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """Run provider requests in a thread pool without blocking on timed-out workers."""
        if len(prompts) != len(configs):
            raise ValueError("Number of prompts must match number of configs")

        max_workers = max(1, min(max_workers, len(prompts)))
        results: list[tuple[str, dict[str, Any] | None]] = [("", None)] * len(prompts)
        finalized_indices: set[int] = set()
        completed = 0

        def _finalize(idx: int, result: tuple[str, dict[str, Any] | None]) -> None:
            nonlocal completed
            if idx in finalized_indices:
                return
            finalized_indices.add(idx)
            results[idx] = result
            if per_request_callback:
                per_request_callback(idx, result)
            if status_callback:
                completed += 1
                status_callback(completed, len(prompts))

        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            executor.submit(generate_one, idx, prompt, cfg): idx
            for idx, (prompt, cfg) in enumerate(zip(prompts, configs))
        }
        started_at = {future: time.perf_counter() for future in futures}
        pending = set(futures)

        try:
            while pending:
                wait_timeout = None if request_timeout is None else 0.1
                done, _ = wait(
                    pending, timeout=wait_timeout, return_when=FIRST_COMPLETED
                )

                for future in done:
                    pending.remove(future)
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        api_logger.warning(f"{log_label} {idx} failed: {e}")
                        result = ("", {"error": str(e), "failed": True})
                    _finalize(idx, result)

                if request_timeout is None:
                    continue

                now = time.perf_counter()
                timed_out = [
                    future
                    for future in list(pending)
                    if now - started_at[future] >= request_timeout
                ]
                for future in timed_out:
                    pending.remove(future)
                    idx = futures[future]
                    future.cancel()
                    api_logger.warning(
                        f"{log_label} {idx} timed out after {request_timeout}s"
                    )
                    _finalize(idx, ("", {"timed_out": True, "error": "timeout"}))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return results

    def _parse_fallback(self, response: str, action_labels: list[str]) -> list[float]:
        """Fallback parser for when dictionary parsing fails."""
        parsing_logger.debug("Using fallback parsing")
        num_actions = len(action_labels)

        # Look for any numbers in the response
        numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", response)
        if len(numbers) >= num_actions:
            try:
                probs = [float(n) for n in numbers[:num_actions]]
                total = sum(probs)
                if total > 0:
                    normalized = [p / total for p in probs]
                    parsing_logger.debug(f"Fallback found probabilities: {normalized}")
                    return normalized
            except ValueError:
                pass

        # Final fallback: uniform distribution
        parsing_logger.debug("Using uniform distribution as final fallback")
        return [1.0 / num_actions] * num_actions


class OpenAILLM(LLMInterface):
    """OpenAI GPT implementation."""

    supports_batching = True

    # Phrases that trigger content policy violations on reasoning models
    # These models do CoT internally and reject explicit CoT/formatting instructions
    REASONING_MODEL_STRIP_PHRASES = [
        # CoT instructions
        "Please reason step-by-step to arrive at your final answer. ",
        "Please reason step-by-step to arrive at your final answer.",
        "Let's think step by step.",
        "Let's think step by step",
        "Think step by step.",
        "Think step by step",
        "Reason through this step by step.",
        "Reason through this step by step",
    ]

    def __init__(self):
        """
        Initialize OpenAI client using environment variables.

        Requires OPENAI_API_KEY environment variable to be set.
        """
        # Will use OPENAI_API_KEY environment variable
        self.client = OpenAI()
        self.max_workers = int(
            os.getenv("OPENAI_BATCH_WORKERS", os.getenv("LLM_BATCH_WORKERS", "4"))
        )
        # Model prefixes that are reasoning models (reject temperature, logprobs, and CoT instructions)
        self.reasoning_model_prefixes = (
            "o1",
            "o3",
            "o4",
            "gpt-5",
        )
        # Backwards compatibility alias
        self.rejects_temperature_and_logprobs = self.reasoning_model_prefixes

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if a model is a reasoning model that requires prompt sanitization."""
        return model_name.lower().startswith(self.reasoning_model_prefixes)

    def _sanitize_prompt_for_reasoning(self, prompt: str) -> str:
        """Remove CoT instructions that reasoning models reject.
        
        OpenAI reasoning models (o1, o3, gpt-5, etc.) do chain-of-thought internally
        and will reject prompts that include explicit CoT instructions like
        "think step by step". This method strips those phrases.
        """
        sanitized = prompt
        for phrase in self.REASONING_MODEL_STRIP_PHRASES:
            sanitized = sanitized.replace(phrase, "")
        return sanitized

    def generate_response(self, prompt: str, config: EvaluationConfig) -> str:
        """
        Generate response using OpenAI API Responses endpoint.

        Args:
            prompt: Input prompt
            config: Evaluation configuration

        Returns:
            Generated response

        Raises:
            Exception: If API call fails
        """
        response_text, _ = self._generate(prompt, config)
        return response_text

    def generate_response_with_metadata(
        self, prompt: str, config: EvaluationConfig
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Generate a response and return provider metadata including token usage.

        Args:
            prompt: The input prompt
            config: Evaluation configuration

        Returns:
            tuple of (response_string, metadata_dict)
        """
        return self._generate(prompt, config)

    def generate_batch(
        self,
        prompts: list[str],
        configs: list[EvaluationConfig],
        status_callback: Callable[[int, int], None] | None = None,
        per_request_callback: Callable[[int, tuple[str, dict[str, Any] | None]], None] | None = None,
        request_timeout: int | None = None,
    ) -> list[tuple[str, dict[str, Any] | None]]:
        return self._generate_batch_threaded(
            prompts,
            configs,
            max_workers=self.max_workers,
            generate_one=lambda idx, prompt, cfg: self._generate(
                prompt, cfg, timeout=request_timeout
            ),
            status_callback=status_callback,
            per_request_callback=per_request_callback,
            request_timeout=request_timeout,
            log_label="Request",
        )

    def _generate(
        self, prompt: str, config: EvaluationConfig, timeout: int | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        """Shared helper to call OpenAI Responses API and normalize metadata."""
        settings = ModelRegistry.get_model_settings(config.model_name)
        
        # Sanitize prompt for reasoning models (strip CoT instructions they reject)
        effective_prompt = prompt
        if self._is_reasoning_model(settings.model):
            effective_prompt = self._sanitize_prompt_for_reasoning(prompt)
            if effective_prompt != prompt:
                api_logger.debug(
                    f"Sanitized prompt for reasoning model {settings.model} "
                    f"(removed {len(prompt) - len(effective_prompt)} chars)"
                )
        
        input_messages: list[dict[str, str]] = []

        if config.system_prompt:
            input_messages.append({"role": "system", "content": config.system_prompt})

        input_messages.append({"role": "user", "content": effective_prompt})
        if settings.provider != "openai":
            raise ValueError(
                f"Model alias '{config.model_name}' is configured for unsupported provider '{settings.provider}'."
            )

        request_kwargs: dict[str, Any] = {
            "model": settings.model,
            "input": input_messages,
            # "temperature": config.temperature,
        }

        if config.temperature is not None:
            if not settings.model.lower().startswith(
                self.rejects_temperature_and_logprobs
            ):
                request_kwargs["temperature"] = config.temperature

        if config.max_tokens is not None:
            request_kwargs["max_output_tokens"] = config.max_tokens

        # Validation: Check for unsupported models for logprobs
        if config.logprobs:
            if not settings.model.lower().startswith(
                self.rejects_temperature_and_logprobs
            ):
                # Use 'include' parameter for Responses API
                # ensure include list exists (though we construct request_kwargs fresh)
                include_list = request_kwargs.get("include", [])
                if "message.output_text.logprobs" not in include_list:
                    include_list.append("message.output_text.logprobs")
                request_kwargs["include"] = include_list

                # Note: top_logprobs might be supported as top-level, or potentially ignored/rejected.
                # We'll try passing it. If it fails, we may need to remove it.
                if config.top_logprobs:
                    request_kwargs["top_logprobs"] = config.top_logprobs
                else:
                    # Default to 20 to ensure we catch the answer token options
                    request_kwargs["top_logprobs"] = 20

        # Apply alias-specific overrides (e.g., reasoning effort)
        request_kwargs = settings.merged_request(request_kwargs)

        try:
            api_logger.debug(
                "Sending request to OpenAI Responses API alias=%s resolved_model=%s overrides=%s timeout=%s",
                settings.alias,
                request_kwargs.get("model"),
                settings.request_options,
                timeout,
            )

            # Pass timeout to httpx client (None = no timeout)
            response = self.client.responses.create(**request_kwargs, timeout=timeout)

            response_text = self._extract_text(response)
            metadata = self._build_metadata(
                response,
                model_alias=settings.alias,
                resolved_model=request_kwargs.get("model"),
                request_options=settings.request_options,
            )

            return response_text, metadata

        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the textual content from a Responses API response.

        For reasoning models, this extracts only the completion tokens (not reasoning tokens).
        Uses response.output_text if available, otherwise filters output items by type.
        """
        # Prefer the output_text convenience property (already filters out reasoning)
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        # Fallback: manually extract from output array, filtering by type
        text_chunks: list[str] = []
        output = getattr(response, "output", None)
        if output:
            for segment in output:
                # Only process "message" type items, skip "reasoning" and other types
                segment_type = getattr(segment, "type", None)
                if segment_type != "message":
                    continue

                content_items = getattr(segment, "content", None)
                if not content_items:
                    continue
                for item in content_items:
                    text_value = getattr(item, "text", None)
                    if text_value:
                        text_chunks.append(text_value)

        return "\n".join(text_chunks).strip()

    @staticmethod
    def _build_metadata(
        response: Any,
        *,
        model_alias: str | None = None,
        resolved_model: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize metadata returned from the Responses API."""
        usage = getattr(response, "usage", None)

        def _usage_value(attr: str) -> int:
            return (
                int(getattr(usage, attr, 0))
                if usage and getattr(usage, attr, None) is not None
                else 0
            )

        metadata: dict[str, Any] = {
            "provider": "openai",
            "model": getattr(response, "model", None),
            "id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "status": getattr(response, "status", None),
            "usage": {
                "input_tokens": _usage_value("input_tokens"),
                "output_tokens": _usage_value("output_tokens"),
                "reasoning_tokens": _usage_value("reasoning_tokens"),
                "total_tokens": _usage_value("total_tokens"),
            },
            "finish_reason": None,
        }

        # Extract logprobs if available
        # 1. Standard Chat Completions API (legacy/compatibility)
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs:
                metadata["logprobs"] = choice.logprobs

        # 2. Responses API (output list)
        output = getattr(response, "output", None)
        if output:
            for item in output:
                # Search inside message items
                if getattr(item, "type", "") == "message":
                    content_list = getattr(item, "content", [])
                    for part in content_list:
                        # Logprobs attached to text parts
                        if hasattr(part, "logprobs") and part.logprobs:
                            metadata["logprobs"] = part.logprobs
                            break
                if "logprobs" in metadata:
                    break

        output = getattr(response, "output", None)
        if output:
            # Prefer the first segment's finish_reason when available
            finish_reason = getattr(output[0], "finish_reason", None)
            metadata["finish_reason"] = finish_reason

            # Also try nested output content logprobs (if Responses API differs)
            if "logprobs" not in metadata and getattr(output[0], "logprobs", None):
                metadata["logprobs"] = output[0].logprobs

        # Ensure total_tokens reflects reasoning + output if service omits it
        usage_dict = metadata["usage"]
        if usage_dict["total_tokens"] == 0 and (
            usage_dict["input_tokens"]
            or usage_dict["output_tokens"]
            or usage_dict["reasoning_tokens"]
        ):
            usage_dict["total_tokens"] = (
                usage_dict["input_tokens"]
                + usage_dict["output_tokens"]
                + usage_dict["reasoning_tokens"]
            )

        # Provide backwards-compatible aliases and derived totals
        usage_dict.setdefault("prompt_tokens", usage_dict["input_tokens"])
        usage_dict.setdefault("completion_tokens", usage_dict["output_tokens"])
        usage_dict.setdefault(
            "generation_tokens",
            usage_dict["output_tokens"] + usage_dict["reasoning_tokens"],
        )

        if model_alias is not None:
            metadata["model_alias"] = model_alias
        if resolved_model is not None:
            metadata["resolved_model"] = resolved_model
        if request_options:
            metadata["request_options"] = request_options

        return metadata

    def get_available_models(self) -> list[str]:
        """
        Get list of available OpenAI models.

        Returns:
            list of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception:
            # Return common models as fallback
            return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]







class AnthropicLLM(LLMInterface):

    """Anthropic Claude implementation."""

    supports_batching = True

    def __init__(self) -> None:
        """Initialize Anthropic client using environment variables (ANTHROPIC_API_KEY)."""
        self.client = Anthropic()
        self.max_workers = int(
            os.getenv("ANTHROPIC_BATCH_WORKERS", os.getenv("LLM_BATCH_WORKERS", "50"))
        )

    def generate_response(self, prompt: str, config: EvaluationConfig) -> str:
        response_text, _ = self._generate(prompt, config)
        return response_text

    def generate_response_with_metadata(
        self, prompt: str, config: EvaluationConfig
    ) -> tuple[str, dict[str, Any] | None]:
        return self._generate(prompt, config)

    def generate_batch(
        self,
        prompts: list[str],
        configs: list[EvaluationConfig],
        status_callback: Callable[[int, int], None] | None = None,
        per_request_callback: Callable[[int, tuple[str, dict[str, Any] | None]], None] | None = None,
        request_timeout: int | None = None,
    ) -> list[tuple[str, dict[str, Any] | None]]:
        return self._generate_batch_threaded(
            prompts,
            configs,
            max_workers=self.max_workers,
            generate_one=lambda idx, prompt, cfg: self._generate(
                prompt, cfg, timeout=request_timeout
            ),
            status_callback=status_callback,
            per_request_callback=per_request_callback,
            request_timeout=request_timeout,
            log_label="Request",
        )

    def _generate(
        self, prompt: str, config: EvaluationConfig, timeout: int | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        settings = ModelRegistry.get_model_settings(config.model_name)
        if settings.provider != "anthropic":
            raise ValueError(
                f"Model alias '{config.model_name}' is configured for unsupported provider '{settings.provider}'."
            )

        message_payload: dict[str, Any] = {
            "model": settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if config.temperature is not None:
            message_payload["temperature"] = config.temperature

        if config.system_prompt:
            message_payload["system"] = config.system_prompt

        # Validation: Anthropic does not support logprobs
        if config.logprobs:
            raise ValueError("Logprobs are not supported by the Anthropic API.")

        if config.max_tokens is None:
            raise ValueError(
                f"max_tokens must be specified for Anthropic models. "
                f"Model '{config.model_name}' has max_tokens=None."
            )
        message_payload["max_tokens"] = config.max_tokens

        message_payload = settings.merged_request(message_payload)

        # Drop invalid temperature values (None/NaN/non-numeric) after overrides.
        temp_value = message_payload.get("temperature")
        if temp_value is None or not isinstance(temp_value, (int, float)) or math.isnan(temp_value):
            message_payload.pop("temperature", None)

        # Anthropic Messages API expects `thinking`, not `reasoning`.
        if "reasoning" in message_payload and "thinking" not in message_payload:
            message_payload["thinking"] = message_payload.pop("reasoning")

        try:
            api_logger.debug(
                "Sending request to Anthropic Messages API alias=%s resolved_model=%s overrides=%s timeout=%s",
                settings.alias,
                message_payload.get("model"),
                settings.request_options,
                timeout,
            )

            # Pass timeout to httpx client (None = no timeout)
            response = self.client.messages.create(**message_payload, timeout=timeout)

            response_text = self._extract_text(response)
            metadata = self._build_metadata(
                response,
                model_alias=settings.alias,
                resolved_model=message_payload.get("model"),
                request_options=settings.request_options,
            )

            return response_text, metadata

        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        text_chunks: list[str] = []

        for block in getattr(response, "content", []) or []:
            text_value = getattr(block, "text", None)
            if text_value:
                text_chunks.append(text_value)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    text_chunks.append(text)

        return "\n".join(text_chunks).strip()

    @staticmethod
    def _build_metadata(
        response: Any,
        *,
        model_alias: str | None = None,
        resolved_model: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage = getattr(response, "usage", None)

        def _usage_value(attr: str) -> int:
            if usage is None:
                return 0
            value = getattr(usage, attr, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(attr)
            return int(value) if value is not None else 0

        metadata: dict[str, Any] = {
            "provider": "anthropic",
            "model": getattr(response, "model", resolved_model),
            "id": getattr(response, "id", None),
            "created": getattr(response, "created_at", None),
            "status": getattr(response, "type", None),
            "usage": {
                "input_tokens": _usage_value("input_tokens"),
                "output_tokens": _usage_value("output_tokens"),
                "reasoning_tokens": _usage_value("reasoning_tokens"),
                "total_tokens": _usage_value("total_tokens"),
            },
            "finish_reason": getattr(response, "stop_reason", None),
        }

        usage_dict = metadata["usage"]
        if usage_dict["total_tokens"] == 0 and (
            usage_dict["input_tokens"]
            or usage_dict["output_tokens"]
            or usage_dict["reasoning_tokens"]
        ):
            usage_dict["total_tokens"] = (
                usage_dict["input_tokens"]
                + usage_dict["output_tokens"]
                + usage_dict["reasoning_tokens"]
            )

        usage_dict.setdefault("prompt_tokens", usage_dict["input_tokens"])
        usage_dict.setdefault("completion_tokens", usage_dict["output_tokens"])
        usage_dict.setdefault(
            "generation_tokens",
            usage_dict["output_tokens"] + usage_dict["reasoning_tokens"],
        )

        if model_alias is not None:
            metadata["model_alias"] = model_alias
        if resolved_model is not None:
            metadata["resolved_model"] = resolved_model
        if request_options:
            metadata["request_options"] = request_options

        return metadata
