"""Reasoning Gym scenario generator for Quickscope.

Generates reasoning scenarios using the reasoning_gym library.
Currently supports 9 datasets with rich configuration spaces.
"""

from random import Random

import numpy as np
import reasoning_gym

from ..abstract.generator import ScenarioGenerator
from ..abstract import ComponentRegistry
from ..default.schemas import DefaultRow
from ...dataflow.logging_config import get_logger


logger = get_logger("reasoning_gym")

# Supported datasets with rich configuration spaces
SUPPORTED_DATASETS = frozenset({
    "knights_knaves",   # Logic/Deduction
    "circuit_logic",    # Boolean Reasoning
    "countdown",        # Arithmetic Reasoning
    "maze",             # Spatial Reasoning
    "syllogism",        # Categorical Logic
    "gsm_symbolic",     # Math Word Problems (85 templates)
    "shortest_path",    # Graph/Grid Reasoning - pathfinding
    "largest_island",   # Graph/Grid Reasoning - connected components
    "grid_reasoning",   # Combined: shortest_path + largest_island (virtual)
})

# Grid reasoning: maps task param to actual reasoning_gym dataset
GRID_REASONING_TASKS = frozenset({"shortest_path", "largest_island"})

# Task-specific parameters for grid_reasoning routing
GRID_REASONING_PARAMS = {
    "shortest_path": {"p_blocked"},
    "largest_island": {
        "num_islands",
        "island_size",
    },
}
# Shared params use simplified names; mapped to min/max in _generate_grid_reasoning
GRID_REASONING_SHARED_PARAMS = {"rows", "cols"}

# GSM Symbolic: valid template indices (some are skipped due to issues)
GSM_SYMBOLIC_TASKS_OK = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73, 75, 78, 80,
    81, 82, 83, 84, 85, 88, 89, 91, 92, 93, 94, 95, 96, 99,
]


@ComponentRegistry.generator("reasoning_gym")
class ReasoningGymGenerator(ScenarioGenerator):
    """Generates reasoning scenarios using reasoning_gym.

    Currently supports 9 datasets with rich configuration spaces.
    Other datasets will raise NotImplementedError.

    Attributes:
        dataset_name: Name of the reasoning_gym dataset to use.
        seed: Base seed for reproducibility.
    """

    def __init__(
        self,
        dataset_name: str = "countdown",
        seed: int | None = None,
        model: str = "",
    ):
        """Initialize generator.
        
        Args:
            dataset_name: Name of the reasoning_gym dataset.
            seed: Base seed for reproducibility.
            model: Model name to use for generated scenarios.
        """
        if dataset_name not in SUPPORTED_DATASETS:
            if dataset_name in reasoning_gym.factory.DATASETS:
                raise NotImplementedError(
                    f"Dataset '{dataset_name}' exists in reasoning_gym but is not yet "
                    f"supported. Supported datasets: {sorted(SUPPORTED_DATASETS)}"
                )
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. "
                    f"Supported: {sorted(SUPPORTED_DATASETS)}"
                )
        
        self._dataset_name = dataset_name
        self.seed = seed
        self.model = model
        
        # Lazy-loaded GSM Symbolic generators
        self._gsm_generators = None
    
    @property
    def dataset_name(self) -> str:
        """Name of the reasoning_gym dataset."""
        return self._dataset_name
    
    def _get_gsm_generators(self) -> dict:
        """Lazy load GSM Symbolic generators."""
        if self._gsm_generators is None:
            from reasoning_gym.arithmetic.gsm_symbolic import generators_00_49, generators_50_99
            
            prefix = "generate_"
            self._gsm_generators = {}
            for module in [generators_00_49, generators_50_99]:
                for name in dir(module):
                    if name.startswith(prefix):
                        idx = int(name[len(prefix):])
                        self._gsm_generators[idx] = getattr(module, name)
        return self._gsm_generators

    def generate(
        self, params: dict, rng: np.random.Generator | None = None
    ) -> DefaultRow:
        """Generate a reasoning_gym scenario from parameters.

        Args:
            params: Dictionary of config parameters for the dataset.
                   Keys should match the dataset's Config dataclass fields.
                   For gsm_symbolic: task_id (1-85) selects the problem template.
            rng: Optional random generator for reproducibility.

        Returns:
            DefaultRow with scenario_text and metadata for RG scoring.
        """
        # Get seed from RNG or use instance seed
        seed = int(rng.integers(0, 2**31)) if rng else (self.seed or 42)

        # Special handling for gsm_symbolic (template-based, different logic)
        if self.dataset_name == "gsm_symbolic":
            return self._generate_gsm_symbolic(params, seed)

        # Special handling for grid_reasoning (param routing to sub-tasks)
        if self.dataset_name == "grid_reasoning":
            return self._generate_grid_reasoning(params, seed)

        # Standard datasets: delegate to shared helper
        return self._create_rg_scenario(self.dataset_name, params, seed)

    def _create_rg_scenario(
        self,
        dataset_name: str,
        params: dict,
        seed: int,
        scenario_id_prefix: str | None = None,
        extra_metadata: dict | None = None,
    ) -> DefaultRow:
        """Create a reasoning_gym scenario (shared logic for all RG datasets).
        
        Args:
            dataset_name: The actual reasoning_gym dataset name (e.g., 'shortest_path')
            params: Config parameters for the dataset
            seed: Random seed for generation
            scenario_id_prefix: Optional prefix for scenario_id (default: 'rg_{dataset_name}')
            extra_metadata: Additional metadata to include (e.g., grid_reasoning_task)
        
        Returns:
            DefaultRow with scenario_text and metadata for RG scoring.
        """
        # Build config dict from params
        config_params = {k: v for k, v in params.items() if not k.startswith("_")}
        config_params["seed"] = seed
        config_params["size"] = 1

        logger.debug(f"Creating {dataset_name} dataset with config: {config_params}")

        # Create dataset and get single entry
        dataset = reasoning_gym.create_dataset(dataset_name, **config_params)
        entry = dataset[0]

        # Generate unique scenario ID
        prefix = scenario_id_prefix or f"rg_{dataset_name}"
        scenario_id = f"{prefix}_{seed}"

        # Get model name from params or use instance default
        model_name = params.get("model_name", self.model)

        # Modify prompt to include \boxed{} format for grid tasks
        question = entry["question"]
        if dataset_name in GRID_REASONING_TASKS:
            question = self._add_boxed_format(question, dataset_name)

        # Build metadata for RG native scoring
        metadata = {
            "rg_dataset": dataset_name,
            "rg_entry": entry,
            **config_params,
            **(extra_metadata or {}),
        }

        return DefaultRow(
            scenario_id=scenario_id,
            scenario_text=question,
            model_name=model_name,
            reference_answer=str(entry.get("answer", "")),
            answer_format="text",
            skip_format_prompt=True,
            metadata=metadata,
            source_config=params,
        )

    def _generate_gsm_symbolic(self, params: dict, seed: int) -> DefaultRow:
        """Generate a GSM Symbolic scenario for a specific template.
        
        Args:
            params: Must contain task_id (1-85) to select the problem template.
            seed: Seed for numerical instantiation.
            
        Returns:
            DefaultRow with the generated math word problem.
        """
        # Get task_id from params (1-indexed from space.yaml)
        task_id = int(params.get("task_id", 1))
        if not 1 <= task_id <= len(GSM_SYMBOLIC_TASKS_OK):
            raise ValueError(
                f"Invalid task_id {task_id}. Must be 1-{len(GSM_SYMBOLIC_TASKS_OK)}"
            )
        
        # Map task_id to actual template index
        template_idx = GSM_SYMBOLIC_TASKS_OK[task_id - 1]
        
        # Get difficulty (currently only 1.0 is supported)
        difficulty = float(params.get("difficulty", 1.0))
        
        logger.debug(
            f"Generating gsm_symbolic: task_id={task_id}, "
            f"template_idx={template_idx}, seed={seed}"
        )
        
        # Generate using the template's generator directly
        generators = self._get_gsm_generators()
        py_rng = Random(seed)
        example = generators[template_idx](py_rng, difficulty)
        
        # Use the raw question - DefaultEngine will add boxed formatting
        question = example["question"]
        
        # Generate unique scenario ID
        scenario_id = f"rg_gsm_symbolic_t{template_idx}_{seed}"
        
        # Build entry in the same format as reasoning_gym expects for scoring
        entry = {
            "question": question,
            "answer": example["answer"],
            "metadata": {
                **example.get("metadata", {}),
                "source_dataset": "gsm_symbolic",
                "source_index": 0,  # Single generation
            },
        }
        
        # Get model name from params or use instance default
        model_name = params.get("model_name", self.model)

        # Build metadata for RG native scoring
        metadata = {
            "rg_dataset": "gsm_symbolic",
            "rg_entry": entry,
            "task_id": task_id,
            "template_idx": template_idx,
            "difficulty": difficulty,
            "seed": seed,
        }
        
        return DefaultRow(
            scenario_id=scenario_id,
            scenario_text=question,
            model_name=model_name,
            reference_answer=str(example["answer"]),
            answer_format="numeric",  # DefaultEngine will add \boxed{} instruction
            skip_format_prompt=False,  # Let DefaultEngine format the prompt
            metadata=metadata,
            source_config=params,
        )

    def _generate_grid_reasoning(self, params: dict, seed: int) -> DefaultRow:
        """Generate a grid reasoning scenario (shortest_path or largest_island).
        
        This is a virtual dataset that combines two related grid-based tasks.
        The 'task' parameter selects which underlying dataset to use.
        
        Args:
            params: Must contain 'task' (shortest_path or largest_island) plus
                   task-specific parameters from ConfigSpace conditionals.
                   Grid dimensions use 'rows' and 'cols' (mapped to min/max).
            seed: Seed for grid generation.
            
        Returns:
            DefaultRow with the generated grid reasoning problem.
        """
        # Extract task type from params
        task = params.get("task", "shortest_path")
        if task not in GRID_REASONING_TASKS:
            raise ValueError(
                f"Invalid grid_reasoning task '{task}'. "
                f"Must be one of: {sorted(GRID_REASONING_TASKS)}"
            )
        
        # Filter to only include params relevant to this task
        valid_params = GRID_REASONING_SHARED_PARAMS | GRID_REASONING_PARAMS[task]
        filtered_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Map rows/cols to min_rows=max_rows, min_cols=max_cols for deterministic grid size
        if "rows" in filtered_params:
            rows = filtered_params.pop("rows")
            filtered_params["min_rows"] = rows
            filtered_params["max_rows"] = rows
        if "cols" in filtered_params:
            cols = filtered_params.pop("cols")
            filtered_params["min_cols"] = cols
            filtered_params["max_cols"] = cols
        
        if "num_islands" in filtered_params:
            num_islands = filtered_params.pop("num_islands")
            filtered_params["min_num_islands"] = num_islands
            filtered_params["max_num_islands"] = num_islands
        if "island_size" in filtered_params:
            island_size = filtered_params.pop("island_size")
            filtered_params["min_island_size"] = island_size
            filtered_params["max_island_size"] = island_size
        
        # Delegate to shared helper
        return self._create_rg_scenario(
            dataset_name=task,
            params=filtered_params,
            seed=seed,
            scenario_id_prefix=f"rg_grid_reasoning_{task}",
            extra_metadata={"grid_reasoning_task": task, "rows": params.get("rows"), "cols": params.get("cols")},
        )

    def _add_boxed_format(self, question: str, task: str) -> str:
        """Add \\boxed{} format instructions to grid reasoning prompts.
        
        Replaces reasoning-gym's native format instructions with boxed versions
        for consistent answer extraction.
        """
        if task == "shortest_path":
            # Remove "length of the " from the prompt
            question = question.replace(
                "Now, find the length of the shortest path from * to # in the following grid:",
                "Now, find the shortest path from * to # in the following grid:"
            )
            # Replace infeasible instruction
            question = question.replace(
                'simply write "infeasible" (without quotes)',
                r'simply write \boxed{infeasible}'
            )
            # Replace example format
            question = question.replace(
                'e.g. right right down down up left',
                r'e.g. \boxed{right right down down up left}'
            )
        elif task == "largest_island":
            # Replace return instruction with boxed version
            question = question.replace(
                'Return the maximum area of an island in grid. If there is no island, return 0.',
                r'Return the maximum area of an island in grid in \boxed{}. If there is no island, return \boxed{0}.'
            )
        return question

    def supports_caching(self) -> bool:
        """RG scenarios can be cached since seed determines output."""
        return True

    @property
    def dataset_identifier(self) -> str:
        """Cache directory name."""
        return f"rg_{self.dataset_name}"
