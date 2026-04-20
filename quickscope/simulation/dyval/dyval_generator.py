"""DyVal scenario generator for dynamic reasoning evaluation.

This generator uses the promptbench DyVal library to generate reasoning
scenarios including arithmetic, boolean logic, deductive/abductive logic,
and graph reachability problems. These are generated dynamically using
DAG-based structures.

Reference: https://llm-eval.github.io/pages/code/dyval.html
"""

from __future__ import annotations

import itertools
import os
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from promptbench.dyval import DyValDataset
from promptbench.dyval.dyval_utils import (
    DYVAL_PROMPTS,
    process_dyval_preds,
    dyval_evaluate,
)

from ..abstract.generator import ScenarioGenerator
from ..abstract import ComponentRegistry
from ..default.schemas import DefaultRow
from ...dataflow.logging_config import get_logger


@contextmanager
def seeded_random_state(rng: np.random.Generator):
    """Context manager to temporarily seed global random state from an RNG.

    promptbench's DyValDataset uses Python's `random` and NumPy's global `np.random`
    internally and doesn't accept an RNG parameter. This context manager:
    1. Saves the current global random state
    2. Seeds both global RNGs from the provided Generator
    3. Restores the original state on exit

    This isolates side effects while enabling reproducibility with promptbench.
    """
    # Save current state
    py_random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        # Seed from the provided RNG
        seed = int(rng.integers(10**9))
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        # Restore original state
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output (e.g., from promptbench/tqdm)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


logger = get_logger("dyval_generation")

# Module-level counter for unique scenario IDs.
# Using hash(scenario_text) caused collisions when the same config
# generated identical text multiple times in a batch, leading to
# lost evaluation tickets downstream.
_scenario_counter = itertools.count()


# Supported DyVal dataset types (order matters for ID mapping from space.yaml)
DYVAL_DATASET_TYPES = [
    "arithmetic",  # dataset_type_id = 1
    "linear_equation",  # dataset_type_id = 2
    "bool_logic",  # dataset_type_id = 3
    "deductive_logic",  # dataset_type_id = 4
    "abductive_logic",  # dataset_type_id = 5
    "reachability",  # dataset_type_id = 6
    "max_sum_path",  # dataset_type_id = 7
]

# Order types (order matters for ID mapping from space.yaml)
DYVAL_ORDER_TYPES = [
    "topological",  # order_id = 1
    "reversed",  # order_id = 2
    "random",  # order_id = 3
]


def strip_dyval_formatting_instructions(text: str) -> str:
    """Strip DyVal's <<<>>> formatting instructions while preserving answer semantics.

    DyVal prompts include formatting instructions using <<<>>> that we want to remove
    so we can use our own formatting (e.g., \\boxed{}). However, we preserve the
    semantic meaning of what the answer should be.

    Transformations:

    ARITHMETIC:
        FROM: "Ensure your final result begins with '<<<' and ends with '>>>',
               for example, if the answer is 1, your final result should be <<<1>>>."
        TO:   (removed - answer type is clear from context)

    LINEAR_EQUATION:
        FROM: "Your response should be formatted as: <<<x's value y's value>>>,
               e.g., if x=1 and y=2, then it should be <<<1 2>>>"
        TO:   "Your response should give x's value followed by y's value,
               e.g., if x=1 and y=2, respond with: 1 2"

    BOOL_LOGIC / DEDUCTIVE_LOGIC / ABDUCTIVE_LOGIC:
        FROM: "Ensure your final result begins with '<<<' and ends with '>>>',
               for example, if the answer is True, your final result should be <<<True>>>."
        TO:   (removed - answer type is clear from context)

    REACHABILITY:
        FROM: "Respond with either '<<<True>>>' if reachable, or '<<<False>>>' otherwise."
        TO:   "Respond with 'True' if reachable, or 'False' if not reachable."

    MAX_SUM_PATH:
        FROM: "Please format your response as <<<Answer>>>.
               For example, if the answer is 1, it should be presented as <<<1>>>."
        TO:   (removed - answer type is clear from context)

    Args:
        text: The DyVal prompt text

    Returns:
        Text with <<<>>> formatting stripped but answer semantics preserved
    """

    # Replacements that preserve semantics (order matters - more specific first)
    replacements = [
        # REACHABILITY: Keep the True/False meaning
        (
            r"Respond with either '<<<True>>>' if reachable, or '<<<False>>>' otherwise\.",
            "Respond with 'True' if reachable, or 'False' if not reachable.",
        ),
        # LINEAR_EQUATION: Keep the x y format instruction
        (
            r"Your response should be formatted as: <<<x's value y's value>>>, e\.g\., if x=1 and y=2, then it should be <<<1 2>>>",
            "Your response should give x's value followed by y's value, e.g., if x=1 and y=2, respond with: 1 2",
        ),
        # MAX_SUM_PATH: Just remove (answer type is numeric, clear from context)
        (
            r"Please format your response as <<<Answer>>>\. For example, if the answer is 1, it should be presented as <<<1>>>\.",
            "",
        ),
        # ARITHMETIC/BOOL_LOGIC/DEDUCTIVE_LOGIC/ABDUCTIVE_LOGIC: Just remove
        (
            r"Ensure your final result begins with '<<<' and ends with '>>>', for example, if the answer is [^,]+, your final result should be <<<[^>]+>>>\.",
            "",
        ),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result)
    result = result.strip()

    return result


DyValDatasetType = Literal[
    "arithmetic",
    "linear_equation",
    "bool_logic",
    "deductive_logic",
    "abductive_logic",
    "reachability",
    "max_sum_path",
]


@ComponentRegistry.generator("dyval")
@dataclass
class DyValScenarioGenerator(ScenarioGenerator):
    """Generates reasoning scenarios using promptbench's DyVal.

    DyVal creates dynamic evaluation scenarios using DAG-based structures
    for testing LLM reasoning capabilities across various domains:
    - arithmetic: Mathematical operations with DAG dependencies
    - linear_equation: Linear equation solving
    - bool_logic: Boolean logic operations (and, or, not)
    - deductive_logic: Deductive reasoning chains
    - abductive_logic: Abductive reasoning problems
    - reachability: Graph reachability problems
    - max_sum_path: Maximum sum path problems

    Attributes:
        model: Model name to evaluate
        dataset_type: Type of DyVal dataset to generate
        depth: Depth of the DAG tree (default: 3)
        num_children_per_node: Children per node in tree DAG (default: 2)
        extra_links_per_node: Extra links to add complexity (default: 0)
        num_nodes_per_sample: Nodes per sample for general DAG (default: 10)
        min_links_per_node: Min links per node for general DAG (default: 1)
        max_links_per_node: Max links per node for general DAG (default: 4)
        add_rand_desc: Number of random descriptions to add (default: 0)
        delete_desc: Number of descriptions to delete (default: 0)
        add_cycles: Whether to add cycles (makes problem unsolvable) (default: 0)
        order: Order of node descriptions ("topological", "reversed", "random")
    """

    model: str
    dataset_type: DyValDatasetType = "arithmetic"
    depth: int = 3
    num_children_per_node: int = 2
    extra_links_per_node: int = 0
    num_nodes_per_sample: int = 10
    min_links_per_node: int = 1
    max_links_per_node: int = 4
    add_rand_desc: int = 0
    delete_desc: int = 0
    add_cycles: int = 0
    order: Literal["topological", "reversed", "random"] = "topological"
    strip_formatting: bool = True  # Strip DyVal's <<<>>> formatting instructions

    # Lazily loaded DyVal components
    _dyval_loaded: bool = field(default=False, init=False, repr=False)
    _DyValDataset: Any = field(default=None, init=False, repr=False)
    _DYVAL_PROMPTS: dict = field(default_factory=dict, init=False, repr=False)
    _process_dyval_preds: Any = field(default=None, init=False, repr=False)
    _dyval_evaluate: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dataset_type not in DYVAL_DATASET_TYPES:
            raise ValueError(
                f"Invalid dataset_type '{self.dataset_type}'. "
                f"Must be one of: {DYVAL_DATASET_TYPES}"
            )
        if self.order not in ("topological", "reversed", "random"):
            raise ValueError(
                f"Invalid order '{self.order}'. "
                f"Must be one of: 'topological', 'reversed', 'random'"
            )

    def _load_dyval(self) -> None:
        """Lazily load DyVal components from promptbench."""
        if self._dyval_loaded:
            return

        self._DyValDataset = DyValDataset
        self._DYVAL_PROMPTS = DYVAL_PROMPTS
        self._process_dyval_preds = process_dyval_preds
        self._dyval_evaluate = dyval_evaluate
        self._dyval_loaded = True

    def _generate_single_sample(self, rng: np.random.Generator | None = None) -> dict:
        """Generate a single DyVal sample.

        Uses promptbench's internal sample generation to create one problem.
        """
        self._load_dyval()

        def create_dataset() -> dict:
            with suppress_output():
                dataset = self._DyValDataset(
                    dataset_type=self.dataset_type,
                    is_trainset=False,
                    num_samples=1,
                    num_nodes_per_sample=self.num_nodes_per_sample,
                    min_links_per_node=self.min_links_per_node,
                    max_links_per_node=self.max_links_per_node,
                    depth=self.depth,
                    num_children_per_node=self.num_children_per_node,
                    extra_links_per_node=self.extra_links_per_node,
                    add_rand_desc=self.add_rand_desc,
                    delete_desc=self.delete_desc,
                    add_cycles=self.add_cycles,
                )
                return dataset[self.order][0]

        # promptbench uses global random state, so we must seed it temporarily
        if rng is not None:
            with seeded_random_state(rng):
                return create_dataset()
        else:
            return create_dataset()

    def generate(
        self, params: dict, rng: np.random.Generator | None = None
    ) -> DefaultRow:
        """Generate a DyVal reasoning scenario from parameters.

        Args:
            params: Dictionary matching space.yaml search space:
                - dataset_type_id: int (1-7) mapping to DYVAL_DATASET_TYPES
                - depth: int (2-6) DAG tree depth
                - num_children_per_node: int (2-4) children per node
                - extra_links_per_node: int (0-3) extra DAG links
                - num_nodes_per_sample: int (5-20) total nodes
                - min_links_per_node: int (1-3) min connectivity
                - max_links_per_node: int (2-6) max connectivity
                - add_rand_desc: int (0-3) random noise descriptions
                - delete_desc: int (0-2) deleted descriptions
                - add_cycles: int (0-1) whether to add cycles
                - order_id: int (1-3) mapping to DYVAL_ORDER_TYPES
                - model_name: Optional override for model
                - strip_formatting: bool (default True) strip DyVal's <<<>>>
                  formatting instructions so you can use your own (e.g. \\boxed{})
            rng: NumPy random generator for reproducible generation.

        Returns:
            DataFrame with single row containing:
                - scenario_id: Unique identifier
                - scenario_text: The problem description (with DyVal formatting
                  stripped by default)
                - domain: Dataset type (arithmetic, bool_logic, etc.)
                - answer: Ground truth answer
                - vars: Variable to solve for (if applicable)
                - model_name: Model to evaluate
                - prompt_template: The prompt template to use

        Raises:
            ValueError: If parameters are invalid
            ImportError: If promptbench is not installed
        """
        self._load_dyval()

        # Map dataset_type_id to string (space.yaml uses 1-indexed IDs)
        if "dataset_type_id" in params:
            dataset_type_id = int(params["dataset_type_id"])
            if not 1 <= dataset_type_id <= len(DYVAL_DATASET_TYPES):
                raise ValueError(
                    f"Invalid dataset_type_id {dataset_type_id}. "
                    f"Must be 1-{len(DYVAL_DATASET_TYPES)}"
                )
            dataset_type = DYVAL_DATASET_TYPES[dataset_type_id - 1]
        else:
            dataset_type = params.get("dataset_type", self.dataset_type)

        # Map order_id to string (space.yaml uses 1-indexed IDs)
        if "order_id" in params:
            order_id = int(params["order_id"])
            if not 1 <= order_id <= len(DYVAL_ORDER_TYPES):
                raise ValueError(
                    f"Invalid order_id {order_id}. "
                    f"Must be 1-{len(DYVAL_ORDER_TYPES)}"
                )
            order = DYVAL_ORDER_TYPES[order_id - 1]
        else:
            order = params.get("order", self.order)

        # Extract all other parameters from space.yaml
        depth = int(params.get("depth", self.depth))
        num_children = int(
            params.get("num_children_per_node", self.num_children_per_node)
        )
        extra_links = int(params.get("extra_links_per_node", self.extra_links_per_node))
        num_nodes = int(params.get("num_nodes_per_sample", self.num_nodes_per_sample))
        min_links = int(params.get("min_links_per_node", self.min_links_per_node))
        max_links = int(params.get("max_links_per_node", self.max_links_per_node))
        add_rand_desc = int(params.get("add_rand_desc", self.add_rand_desc))
        delete_desc = int(params.get("delete_desc", self.delete_desc))
        add_cycles = int(params.get("add_cycles", self.add_cycles))
        model_name = params.get("model_name", self.model)

        # Validate dataset_type
        if dataset_type not in DYVAL_DATASET_TYPES:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. "
                f"Must be one of: {DYVAL_DATASET_TYPES}"
            )

        def create_dataset() -> dict:
            with suppress_output():
                dataset = self._DyValDataset(
                    dataset_type=dataset_type,
                    is_trainset=False,
                    num_samples=1,
                    num_nodes_per_sample=num_nodes,
                    min_links_per_node=min_links,
                    max_links_per_node=max_links,
                    depth=depth,
                    num_children_per_node=num_children,
                    extra_links_per_node=extra_links,
                    add_rand_desc=add_rand_desc,
                    delete_desc=delete_desc,
                    add_cycles=add_cycles,
                )
                return dataset[order][0]

        # promptbench uses global random state, so we must seed it temporarily
        # Retry up to 3 times - promptbench can fail with certain random seeds
        # (e.g., numpy dtype casting errors in linear_equation solver)
        max_retries = 3
        sample = None
        last_error = None

        for attempt in range(max_retries):
            try:
                if rng is not None:
                    with seeded_random_state(rng):
                        sample = create_dataset()
                else:
                    sample = create_dataset()
                break  # Success
            except Exception as e:
                last_error = e
                logger.debug(
                    f"DyVal generation attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                # Advance RNG for next attempt (different seed)
                if rng is not None:
                    _ = rng.integers(10**9)

        # If all retries failed, return a generation_failure scenario
        if sample is None:
            print(
                f"[GENERATION FAILURE] DyVal {dataset_type} failed after {max_retries} retries: {last_error}"
            )
            return DefaultRow(
                scenario_id=f"dyval_{dataset_type}_GENERATION_FAILURE_{next(_scenario_counter)}",
                scenario_text="[GENERATION FAILURE]",
                model_name=model_name,
                reference_answer="__GENERATION_FAILURE__",
                metadata={
                    "generation_failure": True,
                    "error": str(last_error),
                    "dataset_type": dataset_type,
                    **params,
                },
                source_config=params,
            )

        # Get the prompt template for this dataset type
        prompts = self._DYVAL_PROMPTS.get(dataset_type, [])
        prompt_template = prompts[0] if prompts else ""

        # Build the scenario text (formatted prompt)
        descriptions = sample.get("descriptions", "")
        vars_to_solve = sample.get("vars", "")

        if dataset_type in ["arithmetic", "bool_logic", "deductive_logic"]:
            scenario_text = (
                prompt_template.format(descriptions=descriptions, vars=vars_to_solve)
                if prompt_template
                else descriptions
            )
        else:
            scenario_text = (
                prompt_template.format(descriptions=descriptions)
                if prompt_template
                else descriptions
            )

        # Optionally strip DyVal's <<<>>> formatting instructions
        # so you can add your own (e.g., \boxed{})
        strip_fmt = params.get("strip_formatting", self.strip_formatting)
        if strip_fmt:
            scenario_text = strip_dyval_formatting_instructions(scenario_text)

        # Generate unique scenario ID
        scenario_id = f"dyval_{dataset_type}_{next(_scenario_counter)}"

        # Build metadata with all space.yaml parameters for traceability
        metadata = {
            "domain": f"dyval_{dataset_type}",
            "dataset_type": dataset_type,
            "answer": sample.get("answers"),
            "vars": vars_to_solve,
            "descriptions": descriptions,
            "depth": depth,
            "num_children_per_node": num_children,
            "extra_links_per_node": extra_links,
            "num_nodes_per_sample": num_nodes,
            "min_links_per_node": min_links,
            "max_links_per_node": max_links,
            "add_rand_desc": add_rand_desc,
            "delete_desc": delete_desc,
            "add_cycles": add_cycles,
            "order": order,
            "prompt_template": prompt_template,
        }

        logger.debug(
            f"Generated DyVal scenario: type={dataset_type}, "
            f"depth={depth}, order={order}, answer={sample.get('answers')}"
        )

        return DefaultRow(
            scenario_id=scenario_id,
            scenario_text=scenario_text,
            model_name=model_name,
            reference_answer=str(sample.get("answers", "")),
            max_tokens=params.get("max_tokens"),
            system_prompt=params.get("system_prompt"),
            metadata=metadata,
            source_config=params,
        )

    def evaluate_response(
        self, response: str, ground_truth: Any, dataset_type: str | None = None
    ) -> dict[str, Any]:
        """Evaluate an LLM response against the ground truth.

        Args:
            response: Raw LLM response text
            ground_truth: Expected answer
            dataset_type: Type of problem (uses self.dataset_type if None)

        Returns:
            Dictionary with:
                - correct: Boolean indicating if answer matches
                - parsed_response: Extracted answer from response
                - score: Numerical score (0 or 1)
        """
        self._load_dyval()

        dtype = dataset_type or self.dataset_type
        parsed = self._process_dyval_preds(response)

        # Use DyVal's built-in evaluation
        score = self._dyval_evaluate(dtype, [parsed], [ground_truth])

        return {
            "correct": score > 0,
            "parsed_response": parsed,
            "score": score,
        }

    @property
    def supports_caching(self) -> bool:
        """DyVal scenarios can be cached since identical params + seed produce identical prompts."""
        return True

    @property
    def dataset_name(self) -> str:
        """Cache directory name based on dataset type."""
        return f"dyval_{self.dataset_type}"
