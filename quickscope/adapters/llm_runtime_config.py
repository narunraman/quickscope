import os
from dataclasses import dataclass


@dataclass
class PipelineRuntimeConfig:
    """Runtime configuration for evaluation pipeline execution.

    Controls logging verbosity, output behavior, and request timeouts.
    """

    verbose: bool = False  # Enable detailed logging
    quiet: bool = True  # Suppress non-critical output
    request_timeout: int | None = (
        300  # Per-request timeout in seconds (5 min), None = no timeout
    )
    cache_only: bool = False  # Use cached responses only; skip LLM calls on misses