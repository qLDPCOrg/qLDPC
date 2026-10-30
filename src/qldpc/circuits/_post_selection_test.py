# SPDX-License-Identifier: Apache-2.0

"""Tests for flag detectors and the selection of detectors to post-select on."""

from __future__ import annotations

from collections.abc import Collection
from typing import cast

import numpy as np
import pytest
import stim

from . import _post_selection
from .bookkeeping import DetectorRecord


def test_flag_detectors_address_bare_measurements() -> None:
    """Measurements that no detector or observable addresses get a detector each."""
    circuit = stim.Circuit("M 0 1 2\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert _post_selection.get_flag_detectors(circuit) == stim.Circuit("DETECTOR rec[-3]")


def test_selections_are_distinct_indices_of_prep_and_flag_detectors() -> None:
    """Selections are among the prep and flag detectors, in increasing order and without repeats.

    The record has one prep detector, two flag detectors, and one more detector after them, which
    is never selectable.
    """
    record = DetectorRecord({"prep": [0], "flags": [1, 2], "other": [3]})

    def select(post_select: object) -> tuple[int, ...]:
        return _post_selection.get_selected_detectors(cast(Collection[int], post_select), record)

    assert select(None) == (1, 2)
    assert select(True) == (0, 1, 2)
    assert select(False) == ()
    assert select([2, -3, np.int64(0)]) == (0, 2)
    for wrong_type in (0, "flags", [0.5], [True]):
        with pytest.raises(TypeError, match="post_select"):
            select(wrong_type)
    for wrong_value in ([3], [-4]):
        with pytest.raises(ValueError):
            select(wrong_value)

    # Without added flags, the record has no flag detectors, so None selects nothing.
    unflagged = DetectorRecord({"prep": [0], "flags": [], "other": [1]})
    assert _post_selection.get_selected_detectors(None, unflagged) == ()
