"""Reasoning Gym integration package for Quickscope.

Provides scenario generation and evaluation using the reasoning_gym library.
"""

from .generator import ReasoningGymGenerator
from .metrics import compute_rg_metrics_batch

__all__ = ["ReasoningGymGenerator", "compute_rg_metrics_batch"]
