"""DyVal search space with complexity constraints.

This Python-based search space allows flexible constraints that YAML can't express.
Key constraint: Trees should be EITHER wide OR deep, not both.

This prevents computationally expensive configurations where:
  total_nodes ≈ num_children_per_node^depth

Examples of what's allowed:
  - Deep + narrow: depth=10, children=2 → 1,024 nodes ✓
  - Wide + shallow: depth=4, children=4 → 256 nodes ✓
  
Examples of what's forbidden:
  - Deep + wide: depth=9, children=4 → 262,144 nodes ✗ (too slow)
"""

from ConfigSpace import (
    ConfigurationSpace,
    Integer,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)


def get_search_space(dataset_path=None):
    """Return a ConfigSpace for DyVal with complexity constraints.
    
    Args:
        dataset_path: Unused, for API compatibility
    
    Returns:
        ConfigurationSpace with constraints to prevent slow configurations
    """
    cs = ConfigurationSpace(seed=42)
    
    # Dataset type (1-5 for the main types)
    # 1=arithmetic, 2=linear_equation, 3=bool_logic, 4=deductive_logic, 5=abductive_logic
    dataset_type_id = Integer("dataset_type_id", bounds=(1, 5), default=1)
    
    # Depth of the DAG tree structure (reasoning chain length)
    depth = Integer("depth", bounds=(2, 10), default=4)
    
    # Number of children per node (tree width)
    num_children_per_node = Integer("num_children_per_node", bounds=(2, 4), default=2)
    
    # Extra links per node (DAG complexity)
    extra_links_per_node = Integer("extra_links_per_node", bounds=(0, 3), default=0)
    
    # Perturbation: random irrelevant descriptions
    add_rand_desc = Integer("add_rand_desc", bounds=(0, 3), default=0)
    
    # Order of node descriptions (1=topological, 2=reversed, 3=random)
    order_id = Integer("order_id", bounds=(1, 3), default=1)
    
    cs.add([dataset_type_id, depth, num_children_per_node, extra_links_per_node, 
            add_rand_desc, order_id])
    
    # Forbidden combinations: prevent deep + wide trees
    # Rule: if depth >= 8, then num_children_per_node must be 2
    # Rule: if depth >= 7, then num_children_per_node must be <= 3
    
    forbidden_clauses = []
    
    # Forbid: depth >= 8 AND children > 2
    for d in [8, 9, 10]:
        for c in [3, 4]:
            forbidden_clauses.append(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(depth, d),
                    ForbiddenEqualsClause(num_children_per_node, c),
                )
            )
    
    # Forbid: depth == 7 AND children == 4 (borderline case)
    forbidden_clauses.append(
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(depth, 7),
            ForbiddenEqualsClause(num_children_per_node, 4),
        )
    )
    
    # Also forbid high extra_links when depth is high (compounds the problem)
    for d in [8, 9, 10]:
        for e in [2, 3]:
            forbidden_clauses.append(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(depth, d),
                    ForbiddenEqualsClause(extra_links_per_node, e),
                )
            )
    
    for clause in forbidden_clauses:
        cs.add(clause)
    
    return cs


# Estimated generation times for reference
COMPLEXITY_ESTIMATES = {
    # (depth, children): approx_nodes, approx_time
    (4, 4): (256, "<0.5s"),
    (5, 4): (1024, "<1s"),
    (6, 4): (4096, "~1-2s"),
    (7, 3): (2187, "~1s"),
    (7, 4): (16384, "~3-5s"),  # FORBIDDEN
    (8, 2): (256, "<0.5s"),
    (8, 3): (6561, "~2-3s"),   # FORBIDDEN  
    (9, 2): (512, "<0.5s"),
    (10, 2): (1024, "<1s"),
}


if __name__ == "__main__":
    # Test the search space
    cs = get_search_space()
    print(f"ConfigurationSpace with {len(list(cs.values()))} hyperparameters")
    print(f"Forbidden clauses: {len(cs.forbidden_clauses)}")
    
    # Sample configurations and show estimated complexity
    print("\nSample valid configurations:")
    for i in range(10):
        config = cs.sample_configuration()
        d = config["depth"]
        c = config["num_children_per_node"]
        approx_nodes = c ** d
        print(f"  {i+1}. depth={d}, children={c}, extra_links={config['extra_links_per_node']} "
              f"→ ~{approx_nodes:,} nodes")
