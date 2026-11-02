# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for tests of experimental circuit tools."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng(pytestconfig: pytest.Config) -> np.random.Generator:
    """A random number generator seeded by the seed of pytest-randomly."""
    return np.random.default_rng(pytestconfig.getoption("randomly_seed"))
