"""Unit tests for codes.py

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

import unittest.mock
import urllib

import pytest

from qldpc import codes, external


def test_get_classical_code() -> None:
    """Retrieve parity check matrix from GAP 4."""
    # extract parity check and finite field
    check = [1, 1]
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=f"\n{check}\nGF(3^3)"),
    ):
        assert external.codes.get_classical_code("") == ([check], 27)

    # fail to find parity checks
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=r"\nGF(3^3)"),
        pytest.raises(ValueError, match="Code has no parity checks"),
    ):
        assert external.codes.get_classical_code("")


def get_mock_page(text: str) -> unittest.mock.MagicMock:
    """Fake webpage with the given text."""
    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = text.encode("utf-8")
    return mock_page


def test_get_quantum_code(capsys: pytest.CaptureFixture[str]) -> None:
    """Retrieve quantum code data from qecdb.org."""
    # cannot connect to qecdb.org
    with (
        unittest.mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("message")),
        pytest.raises(urllib.error.URLError, match="message"),
    ):
        external.codes.get_quantum_code("")
    terminal_output, error_message = capsys.readouterr()
    assert not error_message
    assert "cannot access" in terminal_output

    # retrieve code data!
    dist_line = "<tr> <td>d</td> <td>5</td> </tr>"
    css_line = "<tr> <td>css</td> <td>False</td> </tr>"
    stab_line = "<tr> <td>H</td> <td><tt>XXXX<br>ZZZZ</tt></td> </tr>"
    mock_page = get_mock_page("\n".join([dist_line, css_line, stab_line]))
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert external.codes.get_quantum_code("") == (["XXXX", "ZZZZ"], 5, False)


def test_distance_bound() -> None:
    """Compute a bound on code distance using QDistRnd."""
    with unittest.mock.patch("qldpc.external.gap.require_package", return_value=None):
        with pytest.raises(ValueError, match="non-CSS subsystem codes"):
            external.codes.get_distance_bound(codes.QuditCode(codes.SHYPSCode(2).matrix))

        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="3"):
            assert external.codes.get_distance_bound(codes.FiveQubitCode()) == 3
            assert external.codes.get_distance_bound(codes.SteaneCode()) == 3
