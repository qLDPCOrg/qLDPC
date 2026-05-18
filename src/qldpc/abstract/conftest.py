"""Shared pytest fixtures for abstract module tests."""

from __future__ import annotations

import pytest

from qldpc import abstract


@pytest.fixture(name="ring", scope="module", params=["cyclic3_gf4", "alternating4_gf5"])
def wedderburn_ring(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_alternating4_gf5: abstract.GroupRing,
) -> abstract.GroupRing:
    return ring_cyclic3_gf4 if request.param == "cyclic3_gf4" else ring_alternating4_gf5
