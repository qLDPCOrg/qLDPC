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
def ring_dihedral3_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a pre-built Wedderburn-Artin transformer.

    Uses DihedralGroup(3) over GF(5), whose primitive central idempotents are pre-cached in
    KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS (avoiding a GAP computation).
    """
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_alternating4_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a size-3 matrix component.

    Used specifically to test WedderburnArtinComponentTransformer code paths that only
    arise for matrix components of size >= 3 (A4 is the smallest group with a 3-dim irrep).
    """
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(name="ring", scope="session", params=["cyclic3_gf4", "dihedral3_gf5"])
def rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf2: abstract.GroupRing,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for which we have pre-built a Wedderburn-Artin transformer."""
    match request.param:
        case "cyclic3_gf4":
            return ring_cyclic3_gf4
        case "dihedral3_gf5":
            return ring_dihedral3_gf5
    raise ValueError(f"Invalid fixture name: {request.param}")  # pragma: no cover
