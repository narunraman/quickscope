"""Backward-compatibility shim — use quickscope.optimization.evaluator instead."""

from .evaluator import Evaluator as LHSBO, EvalConfig as LHSConfig

__all__ = ["LHSBO", "LHSConfig"]
