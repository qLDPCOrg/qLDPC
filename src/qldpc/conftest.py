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
    """Construct a non-commutative ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_cyclic5_gf2(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring with a degree-4 field-extension component (over a prime field)."""
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=2)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_alternating4_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a size-3 matrix component."""
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_dihedral5_gf3(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring with a matrix component over a field extension (size 2, degree 2)."""
    ring = abstract.GroupRing(abstract.DihedralGroup(5), field=3)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(
    name="ring",
    scope="session",
    params=["cyclic3_gf4", "dihedral3_gf5", "cyclic5_gf2", "alternating4_gf5", "dihedral5_gf3"],
)
def rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
    ring_cyclic5_gf2: abstract.GroupRing,
    ring_alternating4_gf5: abstract.GroupRing,
    ring_dihedral5_gf3: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for which we have pre-built a Wedderburn-Artin transformer.

    These rings jointly exercise the Wedderburn-Artin machinery across a commutative field
    extension (cyclic3_gf4), matrix components of size 2 and 3 (dihedral3_gf5, alternating4_gf5), a
    degree-4 field-extension component (cyclic5_gf2), and a matrix component that is itself over a
    field extension -- size 2 and degree 2, i.e. 2x2 matrices over GF(9) (dihedral5_gf3).
    """
    match request.param:
        case "cyclic3_gf4":
            return ring_cyclic3_gf4
        case "dihedral3_gf5":
            return ring_dihedral3_gf5
        case "cyclic5_gf2":
            return ring_cyclic5_gf2
        case "alternating4_gf5":
            return ring_alternating4_gf5
        case "dihedral5_gf3":
            return ring_dihedral5_gf3
    raise ValueError(f"Invalid fixture name: {request.param}")  # pragma: no cover
