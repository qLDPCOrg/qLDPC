"""Unit tests for gap.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import subprocess
import unittest.mock

import pytest

from qldpc import external


def get_mock_process(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """Mock a process with the given results."""
    return subprocess.CompletedProcess(args=[], stdout=stdout, stderr=stderr, returncode=returncode)


def test_is_installed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Is GAP 4 installed?"""
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
        pytest.raises(FileNotFoundError, match="GAP 4 .* not installed"),
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

        # GAP is callable, and succeeds
        with (
            unittest.mock.patch("qldpc.external.gap.is_callable", return_value=True),
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("_TEST_")),
        ):
            assert external.gap.get_output() == "_TEST_"

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


def test_require_package(capsys: pytest.CaptureFixture[str]) -> None:
    """Install missing GAP packages."""
    # GAP is installed but not callable.  The user must install required packages manually
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="fail"),
        pytest.raises(ModuleNotFoundError, match="GAP package .* not installed"),
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

        # all requirements are met!
        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="success"):
            assert external.gap.require_package("")
            capsys.readouterr()  # intercept printed text
