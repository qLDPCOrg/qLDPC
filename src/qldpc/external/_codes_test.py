# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _codes.py."""

from __future__ import annotations

import unittest.mock
import urllib

import numpy as np
import pytest

from qldpc import codes, external
from qldpc.external import _codes
from qldpc.objects import PAULIS_XZ


def test_get_guava_code() -> None:
    """Retrieve parity check matrix from GAP 4."""
    # GAP reports each entry as its discrete log base the primitive root (-1 for a zero entry); over
    # GF(4) the logs 0, 1, 2 rebuild the galois integers 1, 2, 3 and -1 rebuilds a zero.
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch(
            "qldpc.external.gap.get_output", return_value="\nGF(2^2)\n[-1, 0, 1, 2]"
        ),
    ):
        assert _codes.get_guava_code("") == ([[0, 1, 2, 3]], 4)

    # over a prime field the log 0 rebuilds the identity
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="GF(3)\n[0, 0]"),
    ):
        assert _codes.get_guava_code("") == ([[1, 1]], 3)

    # fail to determine the base field
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="[0, 0]"),
        pytest.raises(ValueError, match="Could not determine the base field"),
    ):
        _codes.get_guava_code("")

    # fail to find parity checks
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value="GF(3^3)"),
        pytest.raises(ValueError, match="Code has no parity checks"),
    ):
        _codes.get_guava_code("")


def get_mock_page(text: str) -> unittest.mock.MagicMock:
    """Fake webpage with the given text."""
    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = text.encode("utf-8")
    return mock_page


def test_get_qecdb_code() -> None:
    """Retrieve quantum code data from qecdb.org."""
    # cannot connect to qecdb.org
    with (
        unittest.mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("message")),
        pytest.raises(RuntimeError, match="Cannot access"),
    ):
        _codes.get_qecdb_code("")

    # API response missing stabilizer data
    mock_page = get_mock_page('{"d": 5}')
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="stabilizer data"),
    ):
        _codes.get_qecdb_code("")

    # retrieve code data!
    mock_page = get_mock_page('{"H": "XXXX ZZZZ", "d": 5, "css": false}')
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert _codes.get_qecdb_code("") == (["XXXX", "ZZZZ"], 5, False)


def test_get_qldpc_challenge_code() -> None:
    """Retrieve quantum code data from the Unitary Foundation qLDPC Challenge."""
    index = {"codes": [{"id": "css", "tier": "exact"}, {"id": "stab", "tier": "ub"}]}
    get_json = "qldpc.external._codes._get_json"

    # CSS code with a certified exact distance
    css_checks = {"X": [[0, 2]], "Z": [[1]]}
    css_distance = {"d": 2, "X": {"value": 2}, "Z": {"value": 3}}
    css = {"n": 3, "code_type": "CSS", "checks": css_checks, "distance": css_distance}
    with unittest.mock.patch(get_json, side_effect=[css, index]):
        matrix, distance, is_css, bounds = _codes.get_qldpc_challenge_code("css")
    assert np.array_equal(matrix, [[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    assert (distance, is_css, bounds) == (2, True, (2, 3))

    # stabilizer code (with a Y on qubit 1) whose distance is only an upper bound
    stab_checks = {"S": [{"X": [0, 1], "Z": [1]}]}
    stab = {"n": 2, "code_type": "stabilizer", "checks": stab_checks, "distance": {"d": 1}}
    with unittest.mock.patch(get_json, side_effect=[stab, index]):
        matrix, distance, is_css, bounds = _codes.get_qldpc_challenge_code("stab")
    assert np.array_equal(matrix, [[1, 1, 0, 1]])
    assert (distance, is_css, bounds) == (None, False, (1,))

    # malformed code data
    with (
        unittest.mock.patch(get_json, side_effect=[{}, index]),
        pytest.raises(ValueError, match="Could not parse"),
    ):
        _codes.get_qldpc_challenge_code("")


def test_get_qdistrnd_distance_bound() -> None:
    """Compute a bound on code distance using QDistRnd."""
    matrix = codes.FiveQubitCode().matrix
    steane_code = codes.SteaneCode()
    with unittest.mock.patch(
        "qldpc.external.gap.require_package", return_value=None
    ) as require_package:
        # the symplectic checks of a subsystem code are rejected before GAP loads any package
        with pytest.raises(ValueError, match="subsystem code from its symplectic checks"):
            _codes.get_qdistrnd_distance_bound(matrix, is_subsystem_code=True)
        require_package.assert_not_called()

        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="3") as get_output:
            assert _codes.get_qdistrnd_distance_bound(matrix) == 3

            # QDistRnd receives the checks with the X and Z support of each qudit side by side
            riffled_matrix = matrix[:, [col for qudit in range(5) for col in (qudit, qudit + 5)]]
            commands = _codes._get_gap_sparse_matrix_commands("matrix", 2, riffled_matrix)
            assert set(commands) <= set(get_output.call_args.args)

            assert (
                _codes.get_qdistrnd_distance_bound(
                    steane_code.matrix_x, matrix_z=steane_code.matrix_z
                )
                == 3
            )

            # QDistRnd fails on a matrix without rows, so it receives a zero row instead, for each
            # matrix of a CSS code and for the matrix of a stabilizer code
            no_rows = steane_code.matrix_x[:0]
            for checks, matrix_z in (
                (no_rows, steane_code.matrix_z),
                (steane_code.matrix_x, no_rows),
                (codes.QuditCode(np.zeros((0, 14), dtype=int)).matrix, None),
            ):
                _codes.get_qdistrnd_distance_bound(checks, matrix_z=matrix_z)
                assert "nz:=[[[]]];;" in get_output.call_args.args

        # QDistRnd negates a bound at or below the cutoff when it stops early
        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="-2"):
            assert _codes.get_qdistrnd_distance_bound(matrix, cutoff=2) == 2

        # QDistRnd produced no output at all
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value=""),
            pytest.raises(ValueError, match="no output"),
        ):
            _codes.get_qdistrnd_distance_bound(matrix)

        # QDistRnd output has no bound, only a comment line
        with (
            unittest.mock.patch("qldpc.external.gap.get_output", return_value="# comment only"),
            pytest.raises(ValueError, match="Could not parse a distance bound"),
        ):
            _codes.get_qdistrnd_distance_bound(matrix)


def test_qdistrnd_bounds_without_checks_of_one_type() -> None:  # pragma: no cover
    """GAP bounds the dressed distances of codes, also where some checks or stabilizers are missing.

    This test runs GAP itself, so it runs only where GAP can be called, as in the GAP integration
    tests.
    """
    if not external.gap.is_callable():
        pytest.skip("GAP cannot be called")
    trivial_code = codes.TrivialCode(4, gauge_dimension=2)
    for code in (
        codes.CSSCode(np.zeros((0, 7), dtype=int), codes.HammingCode(3).matrix),
        codes.CSSCode(trivial_code.matrix_x, trivial_code.matrix_z),
    ):
        for pauli in PAULIS_XZ:
            bound = code.get_distance_bound(20, pauli=pauli, backend="gap")
            assert code.get_distance_exact(pauli) <= bound <= len(code)

    # A stabilizer code without checks has distance 1.
    assert codes.QuditCode(np.zeros((0, 6), dtype=int)).get_distance_bound(10, backend="gap") == 1

    # QDistRnd searches dressed logical operators.  The only stabilizer of this subsystem code is Z
    # on its last qubit, so X or Z on one of its first three qubits is a dressed logical operator,
    # although every bare logical operator has weight 3.
    code = codes.CSSCode([[0, 1, 1, 0], [1, 1, 0, 0]], [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    for pauli in PAULIS_XZ:
        assert code.get_distance_bound(20, pauli=pauli, backend="gap") == 1


def test_get_gap_sparse_matrix_commands() -> None:
    """Extension-field entries map to primitive-element powers, not integer multiples of One(F)."""
    # over GF(4) the galois integers 1, 2, 3 are Z(4)^0, Z(4)^1, Z(4)^2 -- not 1, 2, 3 copies of One
    commands = " ".join(_codes._get_gap_sparse_matrix_commands("m", 4, np.array([[0, 1, 2, 3]])))
    assert "elements:=[Z(4)^0,Z(4)^1,Z(4)^2]" in commands
    assert "v[i+1]:=elements[f]" in commands
