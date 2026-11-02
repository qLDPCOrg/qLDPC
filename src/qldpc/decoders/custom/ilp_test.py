# SPDX-License-Identifier: Apache-2.0

"""Tests for the integer-linear-program decoder."""

from __future__ import annotations

import itertools
import unittest.mock

import galois
import numpy as np
import pytest
import scipy.sparse

from qldpc import decoders
from qldpc.decoders._resolution import build_error_decoder
from qldpc.decoders.conftest import ToyProblem
from qldpc.decoders.custom import ilp
from qldpc.decoders.custom.ilp import _get_decoder_ilp


def test_ilp_decoder(toy_problem: ToyProblem) -> None:
    """Decode using an integer linear program."""
    matrix, error, syndrome = toy_problem
    decoder = decoders.custom.ILPDecoder(scipy.sparse.csc_matrix(matrix))
    assert np.array_equal(error, decoder.decode(syndrome))
    detailed = decoder.decode_errors_detailed(syndrome)
    assert np.array_equal(detailed.error, error)
    assert detailed.diagnostics["ilp.status"] == "optimal"
    assert detailed.diagnostics["ilp.objective_value"] == np.count_nonzero(error)


def test_ilp_decoder_minimum_weight(pytestconfig: pytest.Config) -> None:
    """An integer linear program returns a minimum-weight error that reproduces the syndrome."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    for order in [2, 5]:
        field = galois.GF(order)
        for _ in range(4):
            num_checks, num_bits = rng.integers(2, 4), rng.integers(2, 4)
            matrix = field(rng.integers(order, size=(num_checks, num_bits)))
            error = field(rng.integers(order, size=num_bits))
            syndrome = matrix @ error

            candidates = [
                field(vector)
                for vector in itertools.product(range(order), repeat=int(num_bits))
                if np.array_equal(matrix @ field(vector), syndrome)
            ]
            min_weight = min(np.count_nonzero(candidate) for candidate in candidates)

            decoded = decoders.custom.ILPDecoder(matrix).decode(np.asarray(syndrome, dtype=int))
            assert np.array_equal(matrix @ field(decoded), syndrome)
            assert np.count_nonzero(decoded) == min_weight


def test_ilp_decoder_early_termination() -> None:
    """An integer program that stops early rejects or erases an unusable error."""
    pytest.importorskip("highspy")
    matrix = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0]])
    syndrome = np.array([1, 0, 1])

    decoder = decoders.custom.ILPDecoder(matrix, solver="HIGHS", time_limit=1e-9)
    with (
        pytest.warns(UserWarning, match="inaccurate"),
        pytest.raises(ValueError, match="does not reproduce the syndrome"),
    ):
        decoder.decode(syndrome)

    decoder = decoders.custom.ILPDecoder(
        matrix, add_erasure_bit=True, solver="HIGHS", time_limit=1e-9
    )
    with pytest.warns(UserWarning, match="inaccurate"):
        decoded = decoder.decode(syndrome)
    assert len(decoded) == matrix.shape[1] + 1
    assert decoded[-1] == 1


def test_ilp_decoder_unreproducible_syndrome() -> None:
    """A syndrome that no error reproduces is erased with a warning, rather than refused."""
    matrix = np.array([[1], [1]])
    syndrome = np.array([0, 1])

    with pytest.raises(ValueError, match="could not be found"):
        decoders.custom.ILPDecoder(matrix).decode(syndrome)
    with pytest.warns(UserWarning, match="could not be found"):
        assert decoders.custom.ILPDecoder(matrix).decode_errors_detailed(syndrome).erasure

    decoder = decoders.custom.ILPDecoder(matrix, add_erasure_bit=True)
    with pytest.warns(UserWarning, match="could not be found"):
        decoded = decoder.decode(syndrome)
    assert len(decoded) == matrix.shape[1] + 1
    assert decoded[-1] == 1
    with pytest.warns(UserWarning, match="could not be found"):
        detailed = decoder.decode_errors_detailed(syndrome)
    assert np.array_equal(detailed.error, decoded)


def test_ilp_decoder_near_integral_values() -> None:
    """A mixed integer solver's near-integral values are rounded, not truncated."""
    import cvxpy

    matrix = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0]])
    syndrome = np.array([1, 0, 1])
    decoder = decoders.custom.ILPDecoder(matrix)
    expected = decoder.decode(syndrome)

    solve = cvxpy.Problem.solve

    def solve_then_perturb(problem: cvxpy.Problem, **kwargs: object) -> float:
        """Solve, then report the solution the way a solver at its tolerance would."""
        result = solve(problem, **kwargs)
        decoder._variables.value = np.asarray(decoder._variables.value) - 4e-16
        return float(result)

    with unittest.mock.patch.object(cvxpy.Problem, "solve", solve_then_perturb):
        assert np.array_equal(expected, decoder.decode(syndrome))

    field = galois.GF(3)
    decoder = decoders.custom.ILPDecoder(field([[1, 1], [0, 1]]), add_erasure_bit=True)
    decoded = decoder.decode(np.array([0, 1], dtype=bool))
    assert np.array_equal(decoded, [2, 1, 0])
    assert np.array_equal(field([[1, 1], [0, 1]]) @ field(decoded[:-1]), [0, 1])


def test_invalid_ilp() -> None:
    """ILP decoding rejects a field that is not prime."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)
    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        build_error_decoder(galois.GF(4)(matrix), decoder=decoders.ilp()).decode(syndrome)


def test_ilp_requires_cvxpy() -> None:
    """A missing cvxpy gets an install hint, but a module missing inside cvxpy is reported as is."""
    with (
        unittest.mock.patch.dict("sys.modules", {"cvxpy": None}),
        pytest.raises(ModuleNotFoundError, match=r"qldpc\[ilp\]"),
    ):
        decoders.custom.ILPDecoder(np.eye(2, dtype=int))
    nested_error = ModuleNotFoundError("No module named 'nested'", name="nested")
    with (
        pytest.raises(ModuleNotFoundError, match="nested"),
        unittest.mock.patch("builtins.__import__", side_effect=nested_error),
    ):
        ilp._get_cvxpy()


def test_ilp_builder() -> None:
    """The ILP builder decodes matrices and detector error models with optional erasure."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1).to_dem()
    error = np.array([1, 0, 0])
    syndrome = matrix @ error % 2

    for pcm_or_dem in [matrix, dem]:
        decoder = _get_decoder_ilp(pcm_or_dem, add_erasure_bit=True)
        decoded = decoder.decode(syndrome)
        assert decoded.shape == (error.size + 1,)
        assert decoded[-1] == 0
        assert np.array_equal(matrix @ decoded[:-1] % 2, syndrome)

    field = galois.GF(3)
    matrix = -matrix.view(field)
    error = -error.view(field)
    decoder = decoders.custom.ILPDecoder(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
