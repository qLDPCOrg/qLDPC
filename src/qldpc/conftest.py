# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for qLDPC package tests."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator, Mapping

import pytest

from qldpc import abstract, external


def run_python(
    code: str, *options: str, env: Mapping[str, str] | None = None, timeout: float = 600
) -> str:
    """Run Python code in a fresh interpreter that imports qLDPC from this checkout.

    The child interpreter receives this interpreter's import path as its PYTHONPATH, so that it
    imports the same qLDPC sources rather than whatever qldpc its environment happens to resolve.

    Args:
        code: The code to run, passed to the interpreter with -c.
        *options: Interpreter options that precede -c, such as "-W", "error".
        env: Environment variables to set in the child, in addition to this process's environment.
        timeout: The number of seconds after which to abort the child.  Default: 600.

    Returns:
        The standard output of the child.

    Raises:
        AssertionError: If the child exits with a nonzero return code, showing its standard error.
    """
    import_path = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, *options, "-c", code],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": import_path, **(env or {})},
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(autouse=True)
def isolate_gap_probe_state() -> Iterator[None]:
    """Reset cached GAP availability probes and backend registries around every test.

    These caches and registries are process-wide, so without a reset, one test's mocked (or real)
    GAP availability would leak into later tests that run in the same worker process.
    """
    external.gap.clear_cache()
    yield
    external.gap.clear_cache()


@pytest.fixture(autouse=True)
def block_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests of the manual GAP and MAGMA workflows from overwriting the user's clipboard."""
    monkeypatch.setattr("pyperclip.copy", lambda text: None)


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


@pytest.fixture(name="ring", scope="session", params=["cyclic3_gf4", "dihedral3_gf5"])
def rings_to_test(
    request: pytest.FixtureRequest,
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


@pytest.fixture(
    name="wa_ring",
    scope="session",
    params=["cyclic3_gf4", "dihedral3_gf5", "cyclic5_gf2", "alternating4_gf5", "dihedral5_gf3"],
)
def wedderburn_artin_rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
    ring_cyclic5_gf2: abstract.GroupRing,
    ring_alternating4_gf5: abstract.GroupRing,
    ring_dihedral5_gf3: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for exercising the Wedderburn-Artin machinery.

    These rings jointly cover a commutative field extension (cyclic3_gf4), matrix components of size
    2 and 3 (dihedral3_gf5, alternating4_gf5), a degree-4 field-extension component (cyclic5_gf2),
    and a matrix component that is itself over a field extension -- size 2 and degree 2, i.e. 2x2
    matrices over GF(9) (dihedral5_gf3).
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
