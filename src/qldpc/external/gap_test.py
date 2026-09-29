# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gap.py."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import types
import unittest.mock

import pytest

from qldpc import external


class MockGapValue:
    """Small stand-in for a libgap value."""

    def __init__(self, value: object) -> None:
        self.value = value

    def sage(self) -> object:
        """Convert the value to its Python representation."""
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value

    def __str__(self) -> str:
        """Format the value as GAP would."""
        return str(self.value)


class MockLibGap:
    """Small stand-in for the optional libgap binding."""

    def __init__(self, output: object = "_TEST_") -> None:
        self.output = output
        self.commands: list[str] = []
        self.package_available = True

    def eval(self, command: str) -> MockGapValue:
        """Evaluate a command."""
        self.commands.append(command)
        if not command.startswith("CallFuncList(function()\n") or not command.endswith(
            "\nend, [])"
        ):
            raise RuntimeError("can only evaluate a single statement")
        if isinstance(self.output, BaseException):
            raise self.output
        return MockGapValue(self.output)

    def function_factory(self, function_name: str) -> object:
        """Return a fake GAP function."""
        assert function_name == "TestPackageAvailability"

        def test_package_availability(name: str) -> str:
            if isinstance(self.output, BaseException):
                raise self.output
            return "true" if self.package_available and name else "fail"

        return test_package_availability


def test_installed_libgap_import_contract() -> None:
    """Use the import path supplied by the installed passagemath-gap distribution."""
    try:
        importlib.metadata.version("passagemath-gap")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("passagemath-gap is not installed")
    module = importlib.import_module(
        "sage.libs.gap.libgap"
    )  # pragma: no cover - optional dependency
    libgap = module.libgap  # pragma: no cover - optional dependency
    assert libgap is external.gap._get_libgap()  # pragma: no cover - optional dependency
    assert libgap is not None  # pragma: no cover - optional dependency
    assert str(libgap.eval("1 + 1")) == "2"  # pragma: no cover - optional dependency
    availability = (  # pragma: no cover - optional dependency
        libgap.function_factory("TestPackageAvailability")("guava")
    )
    assert str(availability).lower() == "true"  # pragma: no cover - optional dependency
    external.gap.is_callable.cache_clear()  # pragma: no cover
    external.gap.is_installed.cache_clear()  # pragma: no cover
    output = external.gap.get_output(
        "values := [1, 2];;",
        "for value in values do",
        "Print(value);;",
        "od;;",
        'Print("!");;',
    )  # pragma: no cover - optional dependency
    assert output == "12!"  # pragma: no cover - optional dependency


def get_mock_process(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Mock a process with the given results."""
    return subprocess.CompletedProcess(args=[], stdout=stdout, stderr=stderr, returncode=returncode)


def test_is_installed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Is GAP 4 installed?"""
    external.gap._is_gap_executable.cache_clear()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("4.13.0\n")):
        assert external.gap._is_gap_executable()
    external.gap._is_gap_executable.cache_clear()
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert not external.gap._is_gap_executable()

    with unittest.mock.patch(
        "qldpc.external.gap.importlib.import_module", return_value=types.SimpleNamespace()
    ):
        assert external.gap._get_libgap() is None

    # libgap takes precedence over the command-line executable
    external.gap.is_callable.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=MockLibGap()),
        unittest.mock.patch("subprocess.run") as run,
    ):
        assert external.gap.is_callable()
    run.assert_not_called()

    # GAP version not identified
    external.gap.is_callable.cache_clear()
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert not external.gap.is_callable()

    # gap is not installed and user declines to copy/paste commands and outputs
    external.gap.is_callable.cache_clear()
    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process()):
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert not external.gap.is_installed()

        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("GAP 4 cannot be called")

    # gap is not installed and user is willing to copy/paste commands and outputs
    external.gap.is_callable.cache_clear()
    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False):
        monkeypatch.setattr("builtins.input", lambda: "y")
        assert external.gap.is_installed()

        terminal_output, error_message = capsys.readouterr()
        assert not error_message
        assert terminal_output.startswith("GAP 4 cannot be called")

    # GAP is installed!
    external.gap.is_callable.cache_clear()
    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True):
        assert external.gap.is_installed()


def test_get_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run GAP commands and retrieve the GAP output."""
    # GAP is not installed...
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(FileNotFoundError, match=r"GAP 4 .* not installed"),
    ):
        external.gap.get_output()

    # GAP is installed!
    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True):
        # GAP is callable, but returns an error
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("", "error")),
            pytest.raises(ValueError, match="Error encountered when running GAP"),
        ):
            assert external.gap.get_output()

        # GAP is callable, but exits with a nonzero return code (even with empty stderr)
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process(returncode=1)),
            pytest.raises(ValueError, match="Error encountered when running GAP"),
        ):
            external.gap.get_output()

        # GAP is callable, and succeeds
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("_TEST_")),
        ):
            assert external.gap.get_output() == "_TEST_"

        # GAP is callable, and succeeds with piped input
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("_TEST_")),
        ):
            assert external.gap.get_output(use_pipe=True) == "_TEST_"

        # GAP is not callable, so the user must pass around commands and outputs
        cache: dict[str, str] = {}
        inputs = iter(["_OUTPUT_", ""])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
            unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache),
            unittest.mock.patch("pyperclip.copy", return_value=None),
        ):
            assert external.gap.get_output("_INPUT_") == "_OUTPUT_"
            terminal_output, error_message = capsys.readouterr()
            assert not error_message
            assert terminal_output.startswith("Run the following command in GAP:")

            # retrieve results from cache
            assert external.gap.get_output("_INPUT_") == "_OUTPUT_"
            terminal_output, error_message = capsys.readouterr()
            assert not error_message
            assert "found in the local cache" in terminal_output


def test_get_output_libgap() -> None:
    """Run GAP commands through the direct libgap binding."""
    libgap = MockLibGap()
    with pytest.raises(RuntimeError, match="single statement"):
        libgap.eval("x := 1;; Print(x);;")
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("subprocess.run") as run,
    ):
        assert external.gap.get_output('Print("hello");;') == "_TEST_"
    run.assert_not_called()
    assert "PrintTo(__qldpc_output___stream, " in libgap.commands[-1]
    assert 'PrintTo(__qldpc_output___stream, "hello")' in libgap.commands[-1]
    assert libgap.commands[-1].startswith("CallFuncList(function()\n")
    assert "x := 1;;" not in libgap.commands[-1]
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
    ):
        assert (
            external.gap.get_output(
                "values := [1, 2];;",
                "for value in values do",
                "Print(value);;",
                "od;;",
                'Print("!");;',
            )
            == "_TEST_"
        )
    assert "values := [1, 2];;" in libgap.commands[-1]
    assert "for value in values do" in libgap.commands[-1]
    assert "PrintTo(__qldpc_output___stream, value);;" in libgap.commands[-1]
    assert 'PrintTo(__qldpc_output___stream, "!");;' in libgap.commands[-1]

    external.gap._subprocess_packages.add("qdist")
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
        unittest.mock.patch("subprocess.run", return_value=get_mock_process("_EXECUTABLE_")) as run,
    ):
        assert external.gap.get_output("Print('hello');;") == "_EXECUTABLE_"
    run.assert_called_once()
    external.gap._subprocess_packages.clear()

    with (
        unittest.mock.patch(
            "qldpc.external.gap._get_libgap", return_value=MockLibGap(RuntimeError("bad command"))
        ),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        pytest.raises(ValueError, match="through libgap"),
    ):
        external.gap.get_output("bad")

    assert external.gap._gap_string("text") == "text"
    assert external.gap._gap_string(MockGapValue("text")) == "text"
    assert external.gap._gap_string(MockGapValue(3)) == "3"
    assert external.gap._gap_string(MockGapValue(NotImplementedError())) == ""


def test_require_package(capsys: pytest.CaptureFixture[str]) -> None:
    """Install missing GAP packages."""
    # GAP is installed but not callable.  The user must install required packages manually
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
        pytest.raises(ModuleNotFoundError, match=r"GAP package .* not installed"),
    ):
        external.gap.require_package("")

    # GAP is installed and callable!  Required packages can be installed automatically
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
    ):
        # user declines to install missing package
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
            unittest.mock.patch("builtins.input", return_value="n"),
            pytest.raises(ValueError, match="Cannot proceed without the required package"),
        ):
            external.gap.require_package("")

        # fail to install missing package
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
            unittest.mock.patch("builtins.input", return_value="y"),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process(returncode=1)),
            pytest.raises(ValueError, match="Failed to install"),
        ):
            external.gap.require_package("")

        # successfully install a missing package into the GAP package directory
        install = unittest.mock.Mock(return_value=get_mock_process())
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
            unittest.mock.patch("builtins.input", return_value="y"),
            unittest.mock.patch("subprocess.run", install),
        ):
            assert external.gap.require_package("Example", "https://example.com/gap-package")
        install.assert_called_once_with(
            [
                "git",
                "clone",
                "https://example.com/gap-package",
                os.path.join(external.gap.GAP_ROOT, "pkg", "example"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # all requirements are met!
        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="success"):
            assert external.gap.require_package("")
            capsys.readouterr()  # intercept printed text


def test_require_package_libgap() -> None:
    """Check GAP packages through the direct libgap binding."""
    libgap = MockLibGap()
    with unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap):
        assert external.gap.require_package("Example")

    libgap.package_available = False
    external.gap.require_package.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=False),
        pytest.raises(ModuleNotFoundError, match="passagemath-gap extra"),
    ):
        external.gap.require_package("Example")

    external.gap.require_package.cache_clear()
    broken = MockLibGap()
    broken.output = RuntimeError("bad package check")
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=broken),
        pytest.raises(ValueError, match="Could not check"),
    ):
        external.gap.require_package("Example")

    external.gap.require_package.cache_clear()
    external.gap._subprocess_packages.clear()
    install = unittest.mock.Mock(return_value=get_mock_process())
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
        unittest.mock.patch("builtins.input", return_value="y"),
        unittest.mock.patch("subprocess.run", install),
    ):
        assert external.gap.require_package("QDistRnd", "https://example.com/qdist")
    install.assert_called_once_with(
        [
            "git",
            "clone",
            "https://example.com/qdist",
            os.path.join(external.gap.GAP_ROOT, "pkg", "qdistrnd"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert external.gap._subprocess_packages == {"qdistrnd"}
    external.gap._subprocess_packages.clear()
