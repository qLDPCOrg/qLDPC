# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for logical-basis tests."""

import pytest

from qldpc import codes


@pytest.fixture
def stabilizer_code() -> codes.CSSCode:
    """A [[10, 5]] code with a redundant X-type check, and lightest bases of mixed weights."""
    return codes.CSSCode(
        [
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        ],
        [[1, 1, 0, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 0, 0]],
    )


@pytest.fixture
def subsystem_code() -> codes.CSSCode:
    """A [[6, 3]] subsystem code whose dressed and bare logical operators have different weights."""
    return codes.CSSCode(
        [[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]], [[0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1]]
    )
