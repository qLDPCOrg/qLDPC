"""Unit tests for codes.py.

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

import numpy as np
import pytest

from qldpc import codes, external


def test_get_classical_code() -> None:
    """Retrieve parity check matrix from GAP 4."""
    # GAP reports each entry as its discrete log base the primitive root (-1 for a zero entry); over
    # GF(4) the logs 0, 1, 2 rebuild the galois integers 1, 2, 3 and -1 rebuilds a zero.
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch(
            "qldpc.external.gap.get_output", return_value="\nGF(2^2)\n[-1, 0, 1, 2]"
        ),
    ):
        assert external.codes.get_classical_code("") == ([[0, 1, 2, 3]], 4)

    # over a prime field the log 0 rebuilds the identity
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="GF(3)\n[0, 0]"),
    ):
        assert external.codes.get_classical_code("") == ([[1, 1]], 3)

    # fail to determine the base field
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="[0, 0]"),
        pytest.raises(ValueError, match="Could not determine the base field"),
    ):
        external.codes.get_classical_code("")

    # fail to find parity checks
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="GF(3^3)"),
        pytest.raises(ValueError, match="Code has no parity checks"),
    ):
        external.codes.get_classical_code("")


def test_gap_define_sparse_matrix() -> None:
    """Extension-field entries map to primitive-element powers, not integer multiples of One(F)."""
    # over GF(4) the galois integers 1, 2, 3 are Z(4)^0, Z(4)^1, Z(4)^2 -- not 1, 2, 3 copies of One
    commands = " ".join(external.codes._gap_define_sparse_matrix("m", 4, np.array([[0, 1, 2, 3]])))
    assert "elts:=[Z(4)^0,Z(4)^1,Z(4)^2]" in commands
    assert "v[i+1]:=elts[f]" in commands


def get_mock_page(text: str) -> unittest.mock.MagicMock:
    """Fake webpage with the given text."""
    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = text.encode("utf-8")
    return mock_page


def test_get_quantum_code() -> None:
    """Retrieve quantum code data from qecdb.org."""
    # cannot connect to qecdb.org
    with (
        unittest.mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("message")),
        pytest.raises(RuntimeError, match="Cannot access"),
    ):
        external.codes.get_quantum_code("")

    # page missing stabilizer data
    mock_page = get_mock_page("<tr> <td>d</td> <td>5</td> </tr>")
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="stabilizer data"),
    ):
        external.codes.get_quantum_code("")

    # retrieve code data!
    dist_line = "<tr> <td>d</td> <td>5</td> </tr>"
    css_line = "<tr> <td>css</td> <td>False</td> </tr>"
    stab_line = "<tr> <td>H</td> <td><tt>XXXX<br>ZZZZ</tt></td> </tr>"
    mock_page = get_mock_page(f"{dist_line}\n{css_line}\n{stab_line}")
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

        # QDistRnd produced no output at all
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value=""),
            pytest.raises(ValueError, match="no output"),
        ):
            external.codes.get_distance_bound(codes.FiveQubitCode())

        # QDistRnd output has no bound, only a comment line
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value="# comment only"),
            pytest.raises(ValueError, match="Could not parse a distance bound"),
        ):
            external.codes.get_distance_bound(codes.FiveQubitCode())
