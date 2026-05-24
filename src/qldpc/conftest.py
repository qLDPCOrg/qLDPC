"""Shared pytest fixtures for qLDPC package tests."""

from __future__ import annotations

import pytest

from qldpc import abstract


@pytest.fixture(scope="session")
def ring_cyclic3_gf2(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a small ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_cyclic3_gf4(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring over a non-prime field with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_alternating4_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring
