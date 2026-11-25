# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _magma.py."""

from __future__ import annotations

import subprocess
import unittest.mock
from collections.abc import Iterator

import pytest

from qldpc.external import _magma

# Output of the scripts of _magma._get_script, recorded from MAGMA V2.29-10 on the online MAGMA
# calculator, https://magma.maths.usyd.edu.au/calc
OUTPUT_SUM = "__qldpc_begin__\n2\n__qldpc_end__\n"
OUTPUT_GROUP = (
    "__qldpc_begin__\n"
    "Permutation group acting on a set of cardinality 2\n"
    "Order = 2\n"
    "(1, 2)\n"
    "__qldpc_end__\n"
)
OUTPUT_ERROR = (
    "__qldpc_begin__\n\n"
    ">> PermutationGroup(LinearCode(Matrix(GF(2),1,2,[1])));\n"
    "                                     ^\n"
    "Runtime error in 'Matrix': Sequence has length 1 but should have length 1 * 2 = 2\n\n\n"
)


@pytest.fixture(autouse=True)
def block_local_magma() -> Iterator[None]:
    """Test the real MAGMA probes, mocking the executable in each test instead.

    This fixture overrides the fixture of conftest.py that hides any MAGMA executable.
    """
    _magma.is_callable.cache_clear()
    yield
    _magma.is_callable.cache_clear()


def mock_magma(stdout: str, returncode: int = 0, stderr: str = "") -> unittest.mock.MagicMock:
    """Mock subprocess.run for a MAGMA run, recording each script that it runs."""
    scripts = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        with open(args[-1]) as file:
            scripts.append(file.read())
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    mock = unittest.mock.MagicMock(side_effect=run)
    mock.scripts = scripts
    return mock


def test_get_output() -> None:
    """Run commands in MAGMA and extract their output."""
    # a command runs in a script that does not contain the markers of its output, so that the
    # markers cannot come from an echo of the script
    run = mock_magma(OUTPUT_GROUP)
    command = "PermutationGroup(LinearCode(Matrix(GF(2),1,2,[[1,1]])));"
    with unittest.mock.patch("subprocess.run", run):
        output = _magma.get_output(command)
    assert output.splitlines() == [
        "Permutation group acting on a set of cardinality 2",
        "Order = 2",
        "(1, 2)",
    ]
    (script,) = run.scripts
    assert f"\n{command}\n" in script
    assert _magma._BEGIN_MARKER not in script and _magma._END_MARKER not in script

    # a command without a final semicolon is completed, so that it ends before the next statement
    run = mock_magma(OUTPUT_SUM)
    with unittest.mock.patch("subprocess.run", run):
        assert _magma.get_output("1 + 1") == "2\n"
    assert "\n1 + 1;\n" in run.scripts[0]

    # MAGMA reports an error in the command
    with (
        unittest.mock.patch("subprocess.run", mock_magma(OUTPUT_ERROR, returncode=1)),
        pytest.raises(_magma.MAGMAError, match="Runtime error in 'Matrix'"),
    ):
        _magma.get_output("PermutationGroup(LinearCode(Matrix(GF(2),1,2,[1])));")

    # MAGMA stops without finishing the command or reporting an error in it
    with (
        unittest.mock.patch(
            "subprocess.run", mock_magma("", returncode=1, stderr="no valid licence")
        ),
        pytest.raises(_magma.MAGMARunError, match=r"(?s)exit code 1.*no valid licence"),
    ):
        _magma.get_output("1 + 1;")


def test_is_callable() -> None:
    """Check whether MAGMA can be called from the command line."""
    # no MAGMA executable
    with (
        unittest.mock.patch("shutil.which", return_value=None),
        unittest.mock.patch("subprocess.run", side_effect=AssertionError("must not run MAGMA")),
    ):
        assert not _magma.is_on_path()
        assert not _magma.is_callable()

    # MAGMA runs, or fails to run, a probe
    for run, callable_ in [
        (mock_magma(OUTPUT_SUM), True),
        (mock_magma("", returncode=1), False),
        (unittest.mock.MagicMock(side_effect=PermissionError), False),
        (unittest.mock.MagicMock(side_effect=subprocess.TimeoutExpired("magma", 60)), False),
    ]:
        _magma.is_callable.cache_clear()
        with (
            unittest.mock.patch("shutil.which", return_value="/usr/local/bin/magma"),
            unittest.mock.patch("subprocess.run", run),
        ):
            assert _magma.is_on_path()
            assert _magma.is_callable() == callable_
