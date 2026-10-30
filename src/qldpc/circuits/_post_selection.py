# SPDX-License-Identifier: Apache-2.0

"""Flag detectors and the selection of detectors to post-select on, for circuit analyses."""

from __future__ import annotations

import numbers
from collections.abc import Collection

import stim

from .bookkeeping import DetectorRecord
from .common import get_unaddressed_measurements

####################################################################################################
# flag detectors and post-selection


def get_flag_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Get a detector for each measurement that no detector or observable addresses."""
    flag_detectors = stim.Circuit()
    for measurement in get_unaddressed_measurements(circuit):
        target = stim.target_rec(measurement - circuit.num_measurements)
        flag_detectors.append("DETECTOR", [target])
    return flag_detectors


def get_selected_detectors(
    post_select: bool | Collection[int] | None, detector_record: DetectorRecord
) -> tuple[int, ...]:
    """Identify the detectors of a state-prep diagnostic circuit to post-select on.

    ``post_select`` selects among the prep and flag detectors, which come first in the detector
    record: None selects the flag detectors (none if no flags were added), True all of them, False
    none, and explicit integer indices may be negative to count back from the last of them.  Return
    the distinct selected detectors in increasing order.  Raise a TypeError for an invalid type, and
    a ValueError for an invalid value.
    """
    invalid = "post_select must be None, a bool, or a collection of detector indices"
    if post_select is None:
        return tuple(detector_record.get_events("flags"))
    if isinstance(post_select, str):
        raise TypeError(f"{invalid}, not {post_select!r}")
    num_selectable = len(detector_record.get_events("prep", "flags"))
    if isinstance(post_select, bool):
        return tuple(range(num_selectable)) if post_select else ()
    try:
        indices = tuple(post_select)
    except TypeError as error:
        raise TypeError(invalid) from error
    if not all(is_integer(index) for index in indices):
        raise TypeError("post_select must contain integer detector indices")
    if not all(-num_selectable <= index < num_selectable for index in indices):
        raise ValueError(
            f"The detector record exposes {num_selectable} selectable prep/flag detectors "
            f"(out of {detector_record.num_events} total), so post-selection accepts indices "
            f"in the range [-{num_selectable}, {num_selectable}); requested: {list(indices)}"
        )
    return tuple(sorted({int(index) % num_selectable for index in indices}))


def is_integer(value: object) -> bool:
    """Test whether a value is an integer, which a bool is not."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)
