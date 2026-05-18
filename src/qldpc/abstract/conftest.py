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


@pytest.fixture(scope="module")
def ring_cyclic3_gf2(pytestconfig: pytest.Config) -> abstract.GroupRing:
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="module")
def ring_cyclic3_gf4(pytestconfig: pytest.Config) -> abstract.GroupRing:
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="module")
def ring_alternating4_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring
