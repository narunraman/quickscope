"""Default shared utility transforms.

Universal transforms registered with scenario="default" that all scenarios
can inherit from (dyval, reasoning_gym, offline, etc.).
"""

from quickscope.dataflow.transform_registry import register_transform


@register_transform(
    "hinge-error-rate", scenario="default", monotonicity="increasing"
)
def hinge_transform(hinge_point: float = 0.9):
    """Create a hinge transform with the given hinge point.

    Args:
        hinge_point: Threshold value (default 0.9). Utility caps at this value.

    Returns:
        Transform function: (lcb, ucb) -> (hinge(lcb), hinge(ucb))

    Usage via CLI:
        --transform-kwargs hinge_point=0.8
    """

    def transform(lcb: float, ucb: float) -> tuple[float, float]:
        def hinge(x: float) -> float:
            return min(x, hinge_point)

        return (hinge(lcb), hinge(ucb))

    return transform
