"""
Service wrapper around provider-specific LLM implementations.

The goal is to centralize instantiation, caching, and helper utilities
for LLM usage so that higher level orchestration code does not need to
know about individual providers.
"""

from __future__ import annotations

from typing import Any

from ..adapters.llm_interface import (
    LLMInterface,
    GenerationConfig,
    OpenAILLM,
    AnthropicLLM,
)
from ..adapters.model_registry import ModelRegistry
from ..dataflow.logging_config import get_logger

logger = get_logger(__name__)


class LLMService:
    """Central entry point for instantiating and caching LLM clients."""

    def __init__(
        self,
        walltime: str | None = None,
    ) -> None:
        self._walltime = walltime
        self._llm_cache: dict[str, LLMInterface] = {}

    def get_llm(self, model_alias: str) -> LLMInterface:
        """Return a cached LLM instance for the provided alias."""
        if model_alias not in self._llm_cache:
            self._llm_cache[model_alias] = self._create_llm(model_alias)
        return self._llm_cache[model_alias]

    @staticmethod
    def resolve_max_tokens(
        model_name: str, scenario_text: str | None = None
    ) -> int | None:
        """Resolve max_tokens based on model provider defaults.

        Args:
            model_name: Model alias to resolve
            scenario_text: Optional scenario text to estimate input token count

        Returns:
            Resolved max_tokens value
        """
        settings = ModelRegistry.get_model_settings(model_name)
        provider = settings.provider

        if provider == "anthropic":
            if "max_tokens" in settings.request_options:
                return settings.request_options["max_tokens"]
            raise ValueError("Anthropic max_tokens not specified in model settings")

        elif provider == "openai":
            # OpenAI handles max_tokens internally
            # BaseEvaluationConfig defines max_tokens as int = 1000.
            # If we return a large number, OpenAI usually accepts it up to model limit.
            return None


        raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'")

    def generate(
        self,
        model_alias: str,
        prompt: str,
        *,
        temperature: float | None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        return_metadata: bool = False,
    ) -> Any:
        """Generate text using the cached LLM for the given alias."""

        if max_tokens is None:
            max_tokens = self.resolve_max_tokens(model_alias, prompt)

        config = GenerationConfig(
            model_name=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        llm = self.get_llm(model_alias)
        if return_metadata:
            return llm.generate_response_with_metadata(prompt, config)
        return llm.generate_response(prompt, config)

    def generate_batch(
        self,
        prompts: list[str],
        configs: list[Any],  # BaseEvaluationConfig or similar
        return_metadata: bool = True,
        status_callback: Any = None,  # Callable[[int, int], None] | None
        request_timeout: int | None = None,  # Per-request timeout in seconds
        per_request_callback: Any = None,  # Callable[[int, tuple[str, dict | None]], None] | None
    ) -> list[tuple[str, Any]]:
        """
        Generate responses for a batch of prompts/configs.
        Grouping by model_alias to utilize provider-specific batching.

        Args:
            prompts: List of prompt texts
            configs: List of evaluation configs
            return_metadata: Whether to return metadata
            status_callback: Optional callback(completed, total) for progress tracking
            request_timeout: Per-request timeout in seconds (None = no timeout)
            per_request_callback: Optional callback(global_idx, (response_text, metadata))
                                  called immediately after each request completes.
                                  The idx is the global index in the original prompts list.
                                  Must be thread-safe.
        """
        if not prompts:
            return []

        # 1. Group by model_alias
        # We need to preserve order, so we'll store index
        groups: dict[str, list[int]] = {}
        for idx, config in enumerate(configs):
            alias = config.model_name
            if alias not in groups:
                groups[alias] = []
            groups[alias].append(idx)

        # 2. Results placeholder
        results: list[tuple[str, Any]] = [("", None)] * len(prompts)

        # 3. Execute per group
        for alias, indices in groups.items():
            llm = self.get_llm(alias)

            group_prompts = [prompts[i] for i in indices]
            group_configs = [configs[i] for i in indices]

            for i, config in enumerate(group_configs):
                if config.max_tokens is None:
                    config.max_tokens = self.resolve_max_tokens(alias, group_prompts[i])

            # Create a wrapper callback that maps group indices to global indices
            group_per_request_callback = None
            if per_request_callback:
                def make_callback(group_indices: list[int]):
                    def callback(group_idx: int, result: tuple[str, Any]) -> None:
                        global_idx = group_indices[group_idx]
                        results[global_idx] = result
                        per_request_callback(global_idx, result)
                    return callback
                group_per_request_callback = make_callback(indices)

            # Call provider batch with status_callback, per_request_callback, and request_timeout
            group_results = llm.generate_batch(
                group_prompts,
                group_configs,
                status_callback,
                group_per_request_callback,
                request_timeout,
            )

            # Place back in results (only if not using per_request_callback which already does this)
            if not per_request_callback:
                for result_idx, global_idx in enumerate(indices):
                    results[global_idx] = group_results[result_idx]

        return results

    def clear_cache(self) -> None:
        """Forget all cached LLM instances (primarily for tests)."""
        self._llm_cache.clear()

    def _create_llm(self, model_alias: str) -> LLMInterface:
        """Factory that maps aliases to concrete provider implementations."""

        settings = ModelRegistry.get_model_settings(model_alias)
        provider = settings.provider.lower()

        if provider == "openai":
            return OpenAILLM()
        if provider == "anthropic":
            return AnthropicLLM()

        raise ValueError(
            f"Unsupported LLM provider '{settings.provider}' for model alias '{model_alias}'"
        )

