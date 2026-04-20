"""
Quickscope — Bayesian-optimal LLM evaluation toolkit.
"""

__version__ = "0.1.0"
__author__ = "Narun Raman"

from .simulation.abstract.schemas import (
    BaseRow,
    BaseResult,
    BaseEvaluationConfig,
)
from .simulation.default.dataset_loader import DatasetScenarioLoader
from .adapters.llm_interface import LLMInterface, OpenAILLM, AnthropicLLM
from .adapters import PipelineRuntimeConfig
from .simulation.default.engine import DefaultEngine 
from .dataflow.logging_config import setup_logging, get_logger, set_level_by_name, TRACE


__all__ = [
    "BaseRow",
    "BaseEvaluationConfig",
    "BaseResult",
    "DatasetScenarioLoader",
    "LLMInterface",
    "OpenAILLM",
    "AnthropicLLM",
    "DefaultEngine",
    "setup_logging",
    "get_logger",
    "set_level_by_name",
    "TRACE",
    "PipelineRuntimeConfig",
]
