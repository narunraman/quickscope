
from dataclasses import dataclass
from typing import Literal

from ConfigSpace import ConfigurationSpace, Categorical
from ConfigSpace import UniformIntegerHyperparameter, OrdinalHyperparameter, UniformFloatHyperparameter

def build_configspace(schema: dict) -> ConfigurationSpace:
    """
    schema example:
    SPACE = {
        "topic":        {"type": "categorical", "choices": ["math","history","coding"]},
        "difficulty":   {"type": "categorical", "choices": ["easy","medium","hard"]},
        "n_shots":      {"type": "int", "lower": 0, "upper": 8, "log": False},
        "max_tokens":   {"type": "ordinal", "sequence": [64,128,256,512,1024]},
        "temperature":  {"type": "ordinal", "sequence": [0.0,0.1,0.2,0.3,0.5,0.7,1.0]},
    }
    """
    cs = ConfigurationSpace()
    for name, spec in schema.items():
        t = spec["type"]
        if t == "categorical":
            # ConfigSpace v1 expects 'items' instead of 'choices'
            choices = spec.get("choices") or spec.get("items") or spec.get("sequence")
            cs.add(Categorical(name=name, items=choices))
        elif t == "int":
            cs.add(
                UniformIntegerHyperparameter(
                    name=name,
                    lower=int(spec["lower"]),
                    upper=int(spec["upper"]),
                    log=bool(spec.get("log", False)),
                )
            )
        elif t == "ordinal":
            cs.add(
                OrdinalHyperparameter(name=name, sequence=list(spec["sequence"]))
            )
        elif t == "float":
            if UniformFloatHyperparameter is None:
                raise ValueError("Float hyperparameters require ConfigSpace>=1.1.0; please upgrade or use 'ordinal'.")
            cs.add(
                UniformFloatHyperparameter(
                    name=name,
                    lower=float(spec["lower"]),
                    upper=float(spec["upper"]),
                    log=bool(spec.get("log", False)),
                )
            )
        else:
            raise ValueError(f"Unknown type for {name}: {t}")
    return cs


@dataclass
class OptimizerConfig:
    """Base configuration for all optimizers and evaluators."""
    kind: Literal["coup", "uniform", "specified"]
    n_trials: int = 10_000
    batch_size: int = 5
    mini_batch_size: int = 1  # Number of tickets per config
    seed: int = 123
    time_per_eval: float = 1.0             # bookkeeping only
