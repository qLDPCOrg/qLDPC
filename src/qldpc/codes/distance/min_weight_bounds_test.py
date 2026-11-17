# SPDX-License-Identifier: Apache-2.0

"""Tests for the answers of queries of the lightest Pauli errors with a given syndrome."""

from __future__ import annotations

import dataclasses

import pytest

from .min_weight_bounds import MinWeightBound


def test_bounds_are_immutable_values() -> None:
    """Bounds with equal fields are equal, and their fields cannot be reassigned."""
    bound = MinWeightBound(False, 0b1111, 4)
    assert bound == MinWeightBound(False, 0b1111, 4)
    assert bound != MinWeightBound(False, 0b1111, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.lower_bound = 3  # type: ignore[misc]
