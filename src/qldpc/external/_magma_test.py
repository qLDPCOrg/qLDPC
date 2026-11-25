# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _magma.py."""

from __future__ import annotations

import subprocess
import unittest.mock
from collections.abc import Iterator

import pytest

from qldpc.external import _magma

# The scripts that qLDPC runs for a command, and their output from MAGMA V2.29-10, recorded on the
# online MAGMA calculator, https://magma.maths.usyd.edu.au/calc
SCRIPT = (
    "SetQuitOnError(true);\n"
    "SetEchoInput(false);\n"
    "SetAutoColumns(false);\n"
    "SetColumns(0);\n"
    "SetRows(0);\n"
    'print "__qldpc" cat "_begin__";\n'
    "{command}\n"
    'print "__qldp" cat "c_end__";\n'
    "quit;\n"
)
COMMAND_SUM = "1 + 1;"
OUTPUT_SUM = "__qldpc_begin__\n2\n__qldpc_end__\n"
COMMAND_GROUP = "PermutationGroup(LinearCode(Matrix(GF(2),1,2,[[1,1]])));"
OUTPUT_GROUP = (
    "__qldpc_begin__\n"
    "Permutation group acting on a set of cardinality 2\n"
    "Order = 2\n"
    "(1, 2)\n"
    "__qldpc_end__\n"
)
COMMAND_ERROR = "PermutationGroup(LinearCode(Matrix(GF(2),1,2,[1])));"
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
    """Mock subprocess.run for a MAGMA run, recording the script that each call runs."""
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
    # MAGMA runs the recorded script without the user's startup file or access to standard input
    run = mock_magma(OUTPUT_GROUP)
    with unittest.mock.patch("subprocess.run", run):
        output = _magma.get_output(COMMAND_GROUP)
    assert output.splitlines() == [
        "Permutation group acting on a set of cardinality 2",
        "Order = 2",
        "(1, 2)",
    ]
    assert run.scripts == [SCRIPT.format(command=COMMAND_GROUP)]
    args = run.call_args.args[0]
    assert args[:-1] == ["magma", "-b", "-n"]
    assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL

    # the output of a finished command is returned whatever MAGMA's exit status, which the MAGMA
    # handbook does not document
    with unittest.mock.patch("subprocess.run", mock_magma(OUTPUT_SUM, returncode=1)):
        assert _magma.get_output(COMMAND_SUM) == "2\n"

    # a command without a final semicolon is completed, so that it ends before the next statement
    run = mock_magma(OUTPUT_SUM)
    with unittest.mock.patch("subprocess.run", run):
        assert _magma.get_output("1 + 1") == "2\n"
    assert run.scripts == [SCRIPT.format(command=COMMAND_SUM)]

    # MAGMA reports an error, which stops the script before the end marker
    with (
        unittest.mock.patch("subprocess.run", mock_magma(OUTPUT_ERROR, returncode=1)),
        pytest.raises(_magma.MAGMAError, match="Runtime error in 'Matrix'"),
    ):
        _magma.get_output(COMMAND_ERROR)

    # an error between the markers is reported even if MAGMA went on to print the end marker
    with (
        unittest.mock.patch("subprocess.run", mock_magma(OUTPUT_ERROR + "__qldpc_end__\n")),
        pytest.raises(_magma.MAGMAError, match="Runtime error in 'Matrix'"),
    ):
        _magma.get_output(COMMAND_ERROR)

    # MAGMA stops without finishing the command or reporting an error, or cannot start
    with (
        unittest.mock.patch(
            "subprocess.run", mock_magma("", returncode=1, stderr="no valid licence")
        ),
        pytest.raises(_magma.MAGMARunError, match=r"(?s)exit code 1.*no valid licence"),
    ):
        _magma.get_output(COMMAND_SUM)
    with (
        unittest.mock.patch("subprocess.run", side_effect=PermissionError("_TEST_")),
        pytest.raises(_magma.MAGMARunError, match="_TEST_"),
    ):
        _magma.get_output(COMMAND_SUM)


def test_is_callable() -> None:
    """Check whether MAGMA can be called from the command line."""
    # no MAGMA executable
    with (
        unittest.mock.patch("shutil.which", return_value=None),
        unittest.mock.patch("subprocess.run", side_effect=AssertionError("must not run MAGMA")),
    ):
        assert not _magma.is_on_path()
        assert not _magma.is_callable()

    # MAGMA runs a probe
    _magma.is_callable.cache_clear()
    with (
        unittest.mock.patch("shutil.which", return_value="/usr/local/bin/magma"),
        unittest.mock.patch("subprocess.run", mock_magma(OUTPUT_SUM)),
    ):
        assert _magma.is_on_path()
        assert _magma.is_callable()

    # MAGMA is on PATH but fails the probe, which warns with the reason
    for run, reason in [
        (mock_magma("", returncode=1, stderr="no valid licence"), "no valid licence"),
        (mock_magma(OUTPUT_ERROR), "Runtime error"),
        (mock_magma(OUTPUT_SUM.replace("2", "3")), "MAGMA printed '3\\\\n'"),
        (unittest.mock.MagicMock(side_effect=subprocess.TimeoutExpired("magma", 60)), "timed out"),
    ]:
        _magma.is_callable.cache_clear()
        with (
            unittest.mock.patch("shutil.which", return_value="/usr/local/bin/magma"),
            unittest.mock.patch("subprocess.run", run),
            pytest.warns(UserWarning, match=reason),
        ):
            assert not _magma.is_callable()
