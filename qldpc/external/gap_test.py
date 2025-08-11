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


def get_mock_process(stdout: str, stderr: str = "") -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def test_is_installed() -> None:
    """Is GAP 4 installed?"""
    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("")):
        assert not external.gap.is_installed()

    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("subprocess.run", side_effect=Exception):
        assert not external.gap.is_installed()

    external.gap.is_installed.cache_clear()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("\n4.12.1")):
        assert external.gap.is_installed()


def test_get_output() -> None:
    """Run GAP commands and retrieve the GAP output."""
    output = "test"
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process(output)):
        assert external.gap.get_output() == output
    with (
        unittest.mock.patch("subprocess.run", return_value=get_mock_process(output, "error")),
        pytest.raises(ValueError, match="Error encountered"),
    ):
        assert external.gap.get_output()


def test_require_package() -> None:
    """Install missing GAP packages."""
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(FileNotFoundError, match="GAP 4 .* installation cannot be found"),
    ):
        external.gap.require_package("")

    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True):
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
            unittest.mock.patch("subprocess.run", return_value=get_mock_process("", "error")),
            pytest.raises(ValueError, match="Failed to install"),
        ):
            external.gap.require_package("")
