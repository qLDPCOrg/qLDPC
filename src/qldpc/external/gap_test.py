# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gap.py."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pathlib
import subprocess
import types
import unittest.mock
from collections.abc import Callable

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
        if isinstance(self.value, bool):
            return str(self.value).lower()
        return str(self.value)


class MockLibGap:
    """Small stand-in for the optional libgap binding."""

    def __init__(self, output: object = "_TEST_") -> None:
        self.output = output
        self.commands: list[str] = []
        self.error_output_resets = 0
        self.package_available = True
        self.package_manager_available: object = True
        self.package_install_result: object = True
        self.package_install_makes_available = True
        self.package_install_sources: list[str] = []

    def eval(self, command: str) -> MockGapValue:
        """Evaluate a command."""
        self.commands.append(command)
        if command == external.gap._RESET_LIBGAP_ERROR_OUTPUT:
            self.error_output_resets += 1
            return MockGapValue("")
        if command == "false":
            return MockGapValue(False)
        assert command.startswith("CallFuncList(function()\n") and command.endswith("\nend, [])")
        if isinstance(self.output, BaseException):
            raise self.output
        return MockGapValue(self.output)

    def function_factory(self, function_name: str) -> Callable[..., object]:
        """Return a fake GAP function."""
        if function_name == "TestPackageAvailability":

            def test_package_availability(name: str) -> MockGapValue:
                if isinstance(self.output, BaseException):
                    raise self.output
                if self.package_available and name:
                    return MockGapValue(True)
                return MockGapValue(ValueError("fail"))

            return test_package_availability

        if function_name == "LoadPackage":

            def load_package(name: str) -> MockGapValue:
                assert name == "PackageManager"
                if isinstance(self.package_manager_available, BaseException):
                    raise self.package_manager_available
                if self.package_manager_available:
                    return MockGapValue(True)
                return MockGapValue(ValueError("fail"))

            return load_package

        assert function_name == "InstallPackage"

        def install_package(source: str, preferences: object) -> MockGapValue:
            assert str(preferences) == "false"
            self.package_install_sources.append(source)
            if isinstance(self.package_install_result, BaseException):
                raise self.package_install_result
            if self.package_install_result:
                if self.package_install_makes_available:
                    self.package_available = True
                return MockGapValue(True)
            return MockGapValue(False)

        return install_package


def test_installed_libgap_import_contract() -> None:
    """Use the import path supplied by the installed passagemath-gap distribution."""
    try:
        importlib.metadata.version("passagemath-gap")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        pytest.skip("passagemath-gap is not installed")
    importlib.import_module("sage.all__sagemath_gap")  # pragma: no cover - optional dependency
    module = importlib.import_module(  # pragma: no cover - optional dependency
        "sage.libs.gap.libgap"
    )
    libgap = module.libgap  # pragma: no cover - optional dependency
    assert libgap is external.gap._get_libgap()  # pragma: no cover - optional dependency
    assert libgap is not None  # pragma: no cover - optional dependency
    assert str(libgap.eval("1 + 1")) == "2"  # pragma: no cover - optional dependency
    availability = (  # pragma: no cover - optional dependency
        libgap.function_factory("TestPackageAvailability")("guava")
    )
    # PassageMath may return the installed package directory instead of ``true``.
    assert str(availability).lower() not in {"", "fail"}  # pragma: no cover - optional dependency
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
    external.gap._get_output_libgap(  # pragma: no cover - optional dependency
        ("stale_warning_source := [1, 2, 3];;",), libgap
    )
    with pytest.raises(external.gap.GAPError) as error:  # pragma: no cover - optional dependency
        external.gap._get_output_libgap(('Error("current failure");;',), libgap)
    assert "current failure" in str(error.value)  # pragma: no cover - optional dependency
    assert "stale_warning_source" not in str(error.value)  # pragma: no cover


def get_mock_process(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Mock a process with the given results."""
    return subprocess.CompletedProcess(args=[], stdout=stdout, stderr=stderr, returncode=returncode)


def test_is_installed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Is GAP 4 installed?"""
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("4.13.0\n")):
        assert external.gap._is_gap_executable()
    external.gap._is_gap_executable.cache_clear()
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert not external.gap._is_gap_executable()

    with unittest.mock.patch("qldpc.external.gap.importlib.import_module", side_effect=ImportError):
        assert external.gap._get_libgap() is None

    with unittest.mock.patch(
        "qldpc.external.gap.importlib.import_module",
        side_effect=[types.SimpleNamespace(), OSError],
    ):
        assert external.gap._get_libgap() is None

    with unittest.mock.patch(
        "qldpc.external.gap.importlib.import_module",
        side_effect=[types.SimpleNamespace(), types.SimpleNamespace()],
    ):
        assert external.gap._get_libgap() is None

    expected_libgap = MockLibGap()
    with unittest.mock.patch(
        "qldpc.external.gap.importlib.import_module",
        side_effect=[types.SimpleNamespace(), types.SimpleNamespace(libgap=expected_libgap)],
    ) as import_module:
        assert external.gap._get_libgap() is expected_libgap
    import_module.assert_has_calls(
        [unittest.mock.call("sage.all__sagemath_gap"), unittest.mock.call("sage.libs.gap.libgap")]
    )

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
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None),
        unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        assert not external.gap.is_callable()

    # gap is not installed and user declines to copy/paste commands and outputs
    external.gap.is_callable.cache_clear()
    external.gap.is_installed.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None),
        unittest.mock.patch("subprocess.run", return_value=get_mock_process()),
    ):
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


def test_clear_cache() -> None:
    """After clearing the cache, qLDPC checks again whether GAP is available."""
    external.gap._subprocess_packages.add("package")
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None),
        unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError),
    ):
        assert not external.gap.is_callable()
    with unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=MockLibGap()):
        assert not external.gap.is_callable()
        external.gap.clear_cache()
        assert external.gap.is_callable()
    assert not external.gap._subprocess_packages


def test_get_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Run GAP commands and retrieve the GAP output."""
    # GAP is not installed...
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(FileNotFoundError, match=r"GAP 4 .* not installed"),
    ):
        external.gap.get_output()

    # GAP is installed!
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
    ):
        # GAP is callable, but returns an error
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("", "error")),
            pytest.raises(external.gap.GAPError, match="Error encountered when running GAP"),
        ):
            assert external.gap.get_output()

        # GAP is callable, but exits with a nonzero return code (even with empty stderr)
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process(returncode=1)),
            pytest.raises(external.gap.GAPError, match="Error encountered when running GAP"),
        ):
            external.gap.get_output()

        # GAP is callable, and succeeds, with qLDPC's current cache directory as a GAP root
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("_TEST_")) as run,
            unittest.mock.patch("qldpc.cache.get_disk_cache_path", return_value="_CACHE_"),
        ):
            assert external.gap.get_output() == "_TEST_"
        assert ";_CACHE_" in run.call_args.args[0]

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

    def get_output_command() -> str:
        """The most recent command that evaluated the requested output."""
        return next(
            command for command in reversed(libgap.commands) if "__qldpc_output__" in command
        )

    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("subprocess.run") as run,
    ):
        assert external.gap.get_output('Print("hello");;') == "_TEST_"
    run.assert_not_called()
    assert libgap.error_output_resets == 2
    assert 'PrintTo(__qldpc_output___stream, "hello")' in get_output_command()
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
    assert libgap.error_output_resets == 4
    command = get_output_command()
    assert "values := [1, 2];;" in command
    assert "for value in values do" in command
    assert "PrintTo(__qldpc_output___stream, value);;" in command
    assert 'PrintTo(__qldpc_output___stream, "!");;' in command

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
        pytest.raises(external.gap.GAPError, match="through libgap"),
    ):
        external.gap.get_output("bad")

    assert external.gap._gap_string("text") == "text"
    assert external.gap._gap_string(MockGapValue("text")) == "text"
    assert external.gap._gap_string(MockGapValue(3)) == "3"
    assert external.gap._gap_string(MockGapValue(NotImplementedError())) == ""
    assert external.gap._gap_string(MockGapValue(ValueError("fail"))) == "fail"


def test_require_package(capsys: pytest.CaptureFixture[str]) -> None:
    """Install missing GAP packages."""
    external.gap.require_package.cache_clear()
    with unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=None):
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

            # successfully install a missing package, from a source named in the prompt, into the
            # GAP package directory of qLDPC's current cache directory
            install = unittest.mock.Mock(return_value=get_mock_process())
            with (
                unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
                unittest.mock.patch("builtins.input", return_value="y") as confirm,
                unittest.mock.patch("subprocess.run", install),
                unittest.mock.patch(
                    "qldpc.cache.get_disk_cache_path", return_value=pathlib.Path("_CACHE_")
                ),
            ):
                assert external.gap.require_package("Example", "https://example.com/gap-package")
            assert "https://example.com/gap-package" in confirm.call_args.args[0]
            install.assert_called_once_with(
                [
                    "git",
                    "clone",
                    "https://example.com/gap-package",
                    os.path.join("_CACHE_", "pkg", "example"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            # all requirements are met!
            with unittest.mock.patch("qldpc.external.gap.get_output", return_value="success"):
                assert external.gap.require_package("")
                capsys.readouterr()  # intercept printed text
    external.gap.require_package.cache_clear()

    with unittest.mock.patch("qldpc.external.gap.get_output", return_value="success") as get_output:
        assert external.gap._require_package_subprocess("Example", None)
    get_output.assert_called_once_with('Print(TestPackageAvailability("example"));;')


def test_require_package_libgap(capsys: pytest.CaptureFixture[str]) -> None:
    """Check GAP packages through the direct libgap binding."""
    assert (
        external.gap._get_package_manager_source("Example", "https://example.com/gap-package.git")
        == "https://example.com/gap-package.git"
    )
    libgap = MockLibGap()
    with unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap):
        assert external.gap.require_package("Example")

    libgap.package_available = False
    external.gap.require_package.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("builtins.input", return_value="n"),
        pytest.raises(ValueError, match="Cannot proceed without the required package"),
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
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable"),
        unittest.mock.patch("builtins.input", return_value="yes"),
    ):
        assert external.gap.require_package("Example")
    assert libgap.package_install_sources == ["Example"]
    assert external.gap._subprocess_packages == set()

    libgap.package_available = False
    libgap.package_manager_available = False
    with pytest.raises(ModuleNotFoundError, match="sage -gap"):
        external.gap._install_package_libgap("Example", None, libgap)

    libgap.package_manager_available = RuntimeError("bad package manager")
    with pytest.raises(ModuleNotFoundError, match="Could not load"):
        external.gap._install_package_libgap("Example", None, libgap)

    libgap.package_manager_available = True
    libgap.package_install_result = RuntimeError("bad install")
    with pytest.raises(ValueError, match="Failed to install"):
        external.gap._install_package_libgap("Example", None, libgap)

    libgap.package_install_result = False
    with pytest.raises(ValueError, match="Failed to install"):
        external.gap._install_package_libgap("Example", None, libgap)

    libgap.package_install_result = True
    libgap.output = RuntimeError("bad package verification")
    with pytest.raises(ValueError, match="Could not verify"):
        external.gap._install_package_libgap("Example", None, libgap)

    libgap.output = "_TEST_"
    libgap.package_available = False
    libgap.package_install_makes_available = False
    with pytest.raises(ValueError, match="still unavailable"):
        external.gap._install_package_libgap("Example", None, libgap)

    external.gap.require_package.cache_clear()
    libgap.package_manager_available = False
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=False),
        unittest.mock.patch("builtins.input", return_value="y"),
        pytest.raises(ModuleNotFoundError, match="PackageManager is unavailable"),
    ):
        external.gap.require_package("Example")

    external.gap.require_package.cache_clear()
    libgap.package_manager_available = True
    libgap.package_install_result = False
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=True),
        unittest.mock.patch("builtins.input", return_value="y"),
        unittest.mock.patch("qldpc.external.gap._get_output_subprocess", return_value="fail"),
        unittest.mock.patch(
            "qldpc.external.gap._install_package_subprocess",
            side_effect=ValueError("Failed to install Example"),
        ),
        pytest.raises(ValueError, match="Failed to install"),
    ):
        external.gap.require_package("Example")
    assert external.gap._subprocess_packages == set()

    external.gap.require_package.cache_clear()
    external.gap._libgap_packages.clear()
    libgap.package_install_makes_available = True
    libgap.package_install_result = False
    external.gap._subprocess_packages.clear()
    install = unittest.mock.Mock()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=True),
        unittest.mock.patch("builtins.input", return_value="y") as confirm,
        unittest.mock.patch(
            "qldpc.external.gap._get_output_subprocess", return_value="fail"
        ) as get_executable_output,
        unittest.mock.patch("qldpc.external.gap._install_package_subprocess", install),
    ):
        assert external.gap.require_package("QDistRnd", "https://example.com/qdist")
    get_executable_output.assert_called_once_with(('Print(TestPackageAvailability("qdistrnd"));;',))
    confirm.assert_called_once()
    assert "https://example.com/qdist" in confirm.call_args.args[0]
    install.assert_called_once_with("QDistRnd", "https://example.com/qdist")
    terminal_output, error_message = capsys.readouterr()
    assert not error_message
    assert "sage -gap" in terminal_output
    assert "Falling back to the GAP executable" in terminal_output
    assert external.gap._subprocess_packages == {"qdistrnd"}

    # without a repo, the fallback asks again before cloning the default repo
    external.gap.require_package.cache_clear()
    external.gap._subprocess_packages.clear()
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=True),
        unittest.mock.patch("builtins.input", return_value="y") as confirm,
        unittest.mock.patch("qldpc.external.gap._get_output_subprocess", return_value="fail"),
        unittest.mock.patch("qldpc.external.gap._install_package_subprocess", install),
    ):
        assert external.gap.require_package("Example")
    prompts = [call.args[0] for call in confirm.call_args_list]
    assert len(prompts) == 2 and "GAP's package distribution" in prompts[0]
    assert "https://github.com/gap-packages/Example" in prompts[1]
    capsys.readouterr()

    external.gap.require_package.cache_clear()
    external.gap._subprocess_packages.clear()
    libgap.package_available = True
    with unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap):
        assert external.gap.require_package("Example")

    libgap.package_available = False
    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch("qldpc.external.gap._is_gap_executable", return_value=True),
        unittest.mock.patch("builtins.input", return_value="y") as confirm,
        unittest.mock.patch(
            "qldpc.external.gap._get_output_subprocess", return_value="available"
        ) as get_executable_output,
        unittest.mock.patch("qldpc.external.gap._install_package_subprocess") as install,
    ):
        assert external.gap.require_package("QDistRnd", "https://example.com/qdist")
    confirm.assert_called_once()
    get_executable_output.assert_has_calls(
        [
            unittest.mock.call(('Print(TestPackageAvailability("example"));;',)),
            unittest.mock.call(('Print(TestPackageAvailability("qdistrnd"));;',)),
        ]
    )
    assert get_executable_output.call_count == 2
    install.assert_not_called()
    assert external.gap._libgap_packages == {}
    assert external.gap._subprocess_packages == {"example", "qdistrnd"}

    with (
        unittest.mock.patch("qldpc.external.gap._get_libgap", return_value=libgap),
        unittest.mock.patch(
            "qldpc.external.gap.get_output", return_value="available"
        ) as get_output,
    ):
        assert external.gap.require_package("Example")
    get_output.assert_called_once_with('Print(TestPackageAvailability("example"));;')
