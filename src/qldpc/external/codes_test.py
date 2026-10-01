# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codes.py."""

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
    assert "elements:=[Z(4)^0,Z(4)^1,Z(4)^2]" in commands
    assert "v[i+1]:=elements[f]" in commands


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

    # API response missing stabilizer data
    mock_page = get_mock_page('{"d": 5}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="stabilizer data"),
    ):
        external.codes.get_quantum_code("")

    mock_page = get_mock_page('{"H": "  "}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="stabilizer data"),
    ):
        external.codes.get_quantum_code("")

    # malformed API response
    mock_page = get_mock_page("[]")
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="Could not parse code data"),
    ):
        external.codes.get_quantum_code("")

    # malformed distance and CSS values
    mock_page = get_mock_page('{"H": "XXXX ZZZZ", "d": "5"}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="parse the distance"),
    ):
        external.codes.get_quantum_code("")

    mock_page = get_mock_page('{"H": "XXXX ZZZZ", "css": "False"}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="parse the CSS status"),
    ):
        external.codes.get_quantum_code("")

    # retrieve code data
    mock_page = get_mock_page('{"H": "XXXX ZZZZ", "d": 5, "css": false}')
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert external.codes.get_quantum_code("") == (["XXXX", "ZZZZ"], 5, False)


def test_get_qldpc_challenge_code() -> None:
    """Retrieve CSS code data from the Unitary Foundation qLDPC Challenge."""
    # cannot connect to the qLDPC Challenge
    with (
        unittest.mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("message")),
        pytest.raises(RuntimeError, match="Cannot access"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # malformed API response
    mock_page = get_mock_page("[]")
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="Could not parse code data"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # non-CSS code
    mock_page = get_mock_page('{"n": 2, "code_type": "stabilizer", "checks": {}, "distance": {}}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="not a CSS code"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # malformed metadata
    mock_page = get_mock_page(
        '{"n": 0, "code_type": "CSS", "checks": {"X": [], "Z": []}, "distance": {"d": 1}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="number of qubits"),
    ):
        external.codes.get_qldpc_challenge_code("")

    mock_page = get_mock_page('{"n": 2, "code_type": "CSS", "checks": [], "distance": {"d": 1}}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="parity checks"),
    ):
        external.codes.get_qldpc_challenge_code("")

    mock_page = get_mock_page('{"n": 2, "code_type": "CSS", "checks": {}, "distance": []}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="parse the distance"),
    ):
        external.codes.get_qldpc_challenge_code("")

    mock_page = get_mock_page(
        '{"n": 2, "code_type": "CSS", "checks": {"X": [], "Z": []}, "distance": {}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="parse the distance"),
    ):
        external.codes.get_qldpc_challenge_code("")

    mock_page = get_mock_page(
        '{"n": 2, "code_type": "CSS", "checks": {"X": "invalid", "Z": []}, "distance": {"d": 1}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(TypeError, match="X-type parity checks"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # malformed supports
    mock_page = get_mock_page(
        '{"n": 2, "code_type": "CSS", "checks": {"X": [[2]], "Z": [[1]]}, "distance": {"d": 1}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="X-type parity checks"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # repeated indices and nonpositive distances violate the challenge's sparse-check schema
    mock_page = get_mock_page(
        '{"n": 2, "code_type": "CSS", "checks": {"X": [[0, 0]], "Z": [[1]]}, "distance": {"d": 1}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="X-type parity checks"),
    ):
        external.codes.get_qldpc_challenge_code("")

    mock_page = get_mock_page(
        '{"n": 2, "code_type": "CSS", "checks": {"X": [[0]], "Z": [[1]]}, "distance": {"d": 0}}'
    )
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="parse the distance"),
    ):
        external.codes.get_qldpc_challenge_code("")

    # retrieve code data
    mock_page = get_mock_page(
        '{"n": 4, "code_type": "CSS", "checks": {"X": [[0, 2]], "Z": [[1, 3]]},'
        ' "distance": {"d": 2}}'
    )
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        matrix_x, matrix_z, distance = external.codes.get_qldpc_challenge_code("")
        assert np.array_equal(matrix_x, [[1, 0, 1, 0]])
        assert np.array_equal(matrix_z, [[0, 1, 0, 1]])
        assert distance == 2


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
