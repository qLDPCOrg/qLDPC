"""Common methods used by logical error rate notebooks."""

from qldpc import codes


def get_label(
    code: codes.ClassicalCode | codes.QuditCode,
    distance_trials: bool | int = False,
) -> str:
    """Get a label for a code in a figure.

    If the code distance is not known, use a given number of trials to estimate it.
    """
    known_distance = code.get_distance_if_known()
    if isinstance(known_distance, int):
        return f"$d={known_distance}$"
    if not distance_trials:
        return f"[{len(code)}, {code.dimension}]"
    distance_estimate = code.get_distance_bound(num_trials=int(distance_trials))
    return f"[{len(code)}, {code.dimension}, d <= {distance_estimate}]"
