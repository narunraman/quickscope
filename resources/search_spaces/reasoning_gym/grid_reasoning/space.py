"""Search space for combined grid reasoning tasks (shortest_path + largest_island).

Both tasks operate on 2D grids and share grid dimension parameters.
Task-specific parameters are conditionally active based on the selected task.

Tasks:
- shortest_path: Find shortest path through grid with blocked cells
- largest_island: Find largest connected component of 1s in binary grid
"""

from ConfigSpace import (
    ConfigurationSpace,
    Categorical,
    Integer,
    Float,
)
from ConfigSpace.conditions import EqualsCondition


def get_search_space(dataset_path=None):
    """Return a ConfigSpace with conditional parameters for grid reasoning tasks.
    
    Args:
        dataset_path: Optional path to dataset (unused, for API compatibility)
    
    Returns:
        ConfigurationSpace with task selector, shared grid params, and
        conditional task-specific params.
    """
    cs = ConfigurationSpace()
    
    # ==========================================================================
    # Task selector
    # ==========================================================================
    task = Categorical("task", ["shortest_path", "largest_island"], default="shortest_path")
    cs.add(task)
    
    # ==========================================================================
    # Shared grid dimension parameters
    # ==========================================================================
    # These apply to both tasks - control the grid size complexity
    # Note: rows/cols are single values; generator maps to min_rows=max_rows=rows, etc.
    rows = Integer("rows", bounds=(5, 25), default=10)
    cols = Integer("cols", bounds=(5, 25), default=10)
    cs.add([rows, cols])
    
    # ==========================================================================
    # shortest_path specific parameters
    # ==========================================================================
    # p_blocked: Probability that a cell is blocked (X vs O)
    # Higher values = more obstacles = harder pathfinding
    p_blocked = Float("p_blocked", bounds=(0.1, 0.9), default=0.4)
    cs.add(p_blocked)
    cs.add(EqualsCondition(p_blocked, task, "shortest_path"))
    
    # ==========================================================================
    # largest_island specific parameters
    # ==========================================================================
    # Control island generation - affects problem complexity
    num_islands = Integer("num_islands", bounds=(1, 10), default=5)
    island_size = Integer("island_size", bounds=(1, 20), default=10)
    
    cs.add([num_islands, island_size])
    
    # Add conditions - these params only exist when task="largest_island"
    cs.add(EqualsCondition(num_islands, task, "largest_island"))
    cs.add(EqualsCondition(island_size, task, "largest_island"))
    
    # No forbidden clauses needed; min/max island size range removed in favor
    # of a single pinned island_size parameter.
    
    return cs


if __name__ == "__main__":
    # Test the search space
    cs = get_search_space()
    print(f"ConfigurationSpace with {len(list(cs.values()))} hyperparameters")
    print(f"Conditions: {len(cs.conditions)}")
    print(f"Forbidden clauses: {len(cs.forbidden_clauses)}")
    
    print("\nHyperparameters:")
    for hp in cs.values():
        print(f"  - {hp.name}: {type(hp).__name__}")
    
    print("\nSample configurations:")
    for i in range(10):
        config = cs.sample_configuration()
        cfg = dict(config)
        task = cfg["task"]
        
        print(f"  {i+1}. task={task}, rows={cfg['rows']}, cols={cfg['cols']}", end="")
        if task == "largest_island":
            print(f", num_islands={cfg['num_islands']}, island_size={cfg['island_size']}")
        else:
            print(f", p_blocked={cfg['p_blocked']:.2f}")
    
    print("\nAll constraints satisfied!")
