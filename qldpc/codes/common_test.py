"""Unit tests for common.py

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

import functools
import itertools
import unittest.mock
from typing import Iterator

import galois
import networkx as nx
import numpy as np
import pytest

from qldpc import codes, external
from qldpc.math import symplectic_conjugate
from qldpc.objects import Pauli

####################################################################################################
# classical code tests


def test_constructions_classical(pytestconfig: pytest.Config) -> None:
    """Classical code constructions."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.ClassicalCode.random(5, 3, seed=np.random.randint(2**32))
    assert len(code) == code.num_bits == 5
    assert code.num_checks == 3
    assert "ClassicalCode" in str(code)
    assert code.get_random_word() in code

    # reordering the rows of the generator matrix results in a valid generator matrix
    code.set_generator(np.roll(code.generator, shift=1, axis=0))
    assert codes.ClassicalCode(code).generator is code.generator

    code = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    assert "GF(3)" in str(code)

    code = codes.RepetitionCode(2, field=3)
    assert len(code) == 2
    assert code.dimension == 1
    assert code.get_weight() == 2

    # cover invalid generator matrices for the repetition code
    with pytest.raises(ValueError, match="nontrivial syndromes"):
        code.set_generator([[0, 1]])
    with pytest.raises(ValueError, match="incorrect rank"):
        code.set_generator([[0, 0]])

    # invalid classical code construction
    with pytest.raises(ValueError, match="inconsistent"):
        codes.ClassicalCode(codes.ClassicalCode.random(2, 2), field=3)

    # construct a code from its generator matrix
    code = codes.ClassicalCode.random(5, 3)
    assert codes.ClassicalCode.equiv(code, codes.ClassicalCode.from_generator(code.generator))

    # puncture a code
    assert codes.ClassicalCode.from_generator(code.generator[:, 1:]) == code.puncture(0)

    # shortening a repetition code yields a trivial code
    code = codes.RepetitionCode(3)
    assert np.array_equal(list(code.shorten(0).iter_words()), [[0, 0]])

    # stack two codes
    code_a = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    code_b = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    code = codes.ClassicalCode.stack(code_a, code_b)
    assert len(code) == len(code_a) + len(code_b)
    assert code.dimension == code_a.dimension + code_b.dimension

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        code_b = codes.RepetitionCode(2)
        code = codes.ClassicalCode.stack(code_a, code_b)


def test_named_codes(order: int = 2) -> None:
    """Named codes from the GAP computer algebra system."""
    code = codes.RepetitionCode(order)
    checks = [list(row) for row in code.matrix.view(np.ndarray)]

    with unittest.mock.patch("qldpc.external.codes.get_code", return_value=(checks, None)):
        assert codes.ClassicalCode.from_name(f"RepetitionCode({order})") == code


def test_dual_code(bits: int = 5, checks: int = 3, field: int = 3) -> None:
    """Dual code construction."""
    code = codes.ClassicalCode.random(bits, checks, field)
    assert all(
        word_a @ word_b == 0 for word_a in code.iter_words() for word_b in (~code).iter_words()
    )


def test_tensor_product(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (4, 2),
) -> None:
    """Tensor product of classical codes."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    code_ab = codes.ClassicalCode.tensor_product(code_a, code_b)
    basis = np.reshape(code_ab.generator, (-1, len(code_a), len(code_b)))
    assert all(not np.any(code_a.matrix @ word @ code_b.matrix.T) for word in basis)

    n_a, k_a, d_a = code_a.get_code_params()
    n_b, k_b, d_b = code_b.get_code_params()
    n_ab, k_ab, d_ab = code_ab.get_code_params()
    assert (n_ab, k_ab, d_ab) == (n_a * n_b, k_a * k_b, d_a * d_b)

    with pytest.raises(ValueError, match="Cannot take tensor product"):
        code_b = codes.ClassicalCode.random(*bits_checks_b, field=code_a.field.order**2)
        codes.ClassicalCode.tensor_product(code_a, code_b)


def test_distance_classical(bits: int = 3) -> None:
    """Distance of a vector from a classical code."""
    rep_code = codes.RepetitionCode(bits)

    # "forget" the exact code distance, and re-compute (or estimate) it in various ways
    rep_code._distance = None
    assert rep_code.get_distance_bound(cutoff=bits) == bits
    assert rep_code.get_distance(bound=True) == bits
    assert rep_code.get_distance() == bits
    for vector in itertools.product(rep_code.field.elements, repeat=bits):
        weight = np.count_nonzero(vector)
        dist_bound = rep_code.get_distance_bound(vector=vector)
        dist_exact = rep_code.get_distance_exact(vector=vector)
        assert dist_exact == min(weight, bits - weight)
        assert dist_exact <= dist_bound

    # computing an exact distance but providing bounding arguments raises a warning
    with pytest.warns(UserWarning, match="ignored"):
        assert rep_code.get_distance(test_arg=True)

    # trivial (null) codes have an undefined distance
    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    random_vector = np.random.randint(2, size=len(trivial_code))
    assert trivial_code.dimension == 0
    assert trivial_code.get_distance_exact() is np.nan
    assert trivial_code.get_distance_bound() is np.nan
    assert (
        np.count_nonzero(random_vector)
        == trivial_code.get_distance_exact(vector=random_vector)
        == trivial_code.get_distance_bound(vector=random_vector)
    )

    # compute distance of a trinary repetition code
    rep_code = codes.RepetitionCode(bits, field=3)
    rep_code._distance = None
    assert rep_code.get_distance_exact() == 3


def test_conversions_classical(bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a classical code."""
    code = codes.ClassicalCode.random(bits, checks)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(code.graph))


def test_automorphism() -> None:
    """Compute automorphism group of the smallest nontrivial trinary Hamming code."""
    code = codes.HammingCode(2, field=3)
    automorphisms = "\n()\n(2,4,3)\n(2,3,4)\n"

    # raise an error when GAP is not installed
    external.gap.require_package.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot build GAP group"),
    ):
        code.get_automorphism_group()

    # otherwise, check that automorphisms do indeed preserve the code space
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=automorphisms),
    ):
        group = code.get_automorphism_group()
        for member in group.generate():
            permutation = member.to_matrix().view(code.field)
            assert not np.any(code.matrix @ permutation @ code.generator.T)


def test_classical_capacity() -> None:
    """Logical error rates in a code capacity model."""
    code = codes.RepetitionCode(2)
    logical_error_rate = code.get_logical_error_rate_func(num_samples=1, max_error_rate=1)
    assert logical_error_rate(0) == (0, 0)  # no logical error with zero uncertainty
    assert logical_error_rate(1)[0] == 1  # guaranteed logical error

    logical_error_rate = code.get_logical_error_rate_func(num_samples=10, max_error_rate=0.5)
    with pytest.raises(ValueError, match="error rates greater than"):
        logical_error_rate(1)


####################################################################################################
# quantum code tests


def test_code_string() -> None:
    """Human-readable representation of a code."""
    code = codes.QuditCode([[0, 1]])
    assert "qubits" in str(code)

    code = codes.QuditCode([[0, 1]], field=3)
    assert "GF(3)" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2))
    assert "qubits" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    assert "GF(3)" in str(code)


def get_random_qudit_code(qudits: int, checks: int, field: int = 2) -> codes.QuditCode:
    """Construct a random (but probably trivial) QuditCode."""
    return codes.QuditCode(codes.ClassicalCode.random(2 * qudits, checks, field).matrix)


def test_qubit_code(num_qubits: int = 5, num_checks: int = 3) -> None:
    """Random qubit code."""
    assert get_random_qudit_code(num_qubits, num_checks).num_qubits == num_qubits
    with pytest.raises(ValueError, match="3-dimensional qudits"):
        assert get_random_qudit_code(num_qubits, num_checks, field=3).num_qubits


def assert_valid_subgraphs(code: codes.QuditCode) -> None:
    """The union of subgraphs used for syndrome measurement is the entire Tanner graph."""
    assert nx.utils.graphs_equal(
        code.graph, functools.reduce(nx.compose, code.get_syndrome_subgraphs())
    )


def test_qudit_code() -> None:
    """Miscellaneous qudit code tests and coverage."""
    code = codes.FiveQubitCode()
    assert code.dimension == 1
    assert code.get_weight() == 4
    assert code.get_logical_ops(Pauli.X).shape == code.get_logical_ops(Pauli.Z).shape
    assert codes.QuditCode.equiv(code, codes.QuditCode(code))
    assert_valid_subgraphs(code)

    # equivlence to code with redundant stabilizers
    redundant_code = codes.QuditCode(np.vstack([code.matrix, code.matrix]))
    assert codes.QuditCode.equiv(code, redundant_code)

    # the logical ops of the redundant code are valid ops of the original code
    code.set_logical_ops(redundant_code.get_logical_ops())  # also validates the logical ops

    # stacking two codes
    two_codes = codes.QuditCode.stack(code, code)
    assert len(two_codes) == len(code) * 2
    assert two_codes.dimension == code.dimension * 2

    # swapping logical X ops on the two encoded qubits breaks commutation relations
    logical_ops = two_codes.get_logical_ops().copy()
    logical_ops[0], logical_ops[1] = logical_ops[1], logical_ops[0]
    with pytest.raises(ValueError, match="incorrect commutation relations"):
        two_codes.set_logical_ops(logical_ops, validate=True)

    # invalid modifications of logical operators break commutation relations
    logical_ops = two_codes.get_logical_ops().copy()
    logical_ops[0, -1] += two_codes.field(1)
    with pytest.raises(ValueError, match="violate parity checks"):
        two_codes.set_logical_ops(logical_ops, validate=True)

    # providing an incorrect number of logical operators throws an error
    logical_ops = two_codes.get_logical_ops().copy()[[0, two_codes.dimension], :]
    with pytest.raises(ValueError, match="incorrect number"):
        two_codes.set_logical_ops(logical_ops, validate=True)

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        second_code = codes.SurfaceCode(2, field=3)
        codes.QuditCode.stack(code, second_code)


def test_distance_qudit() -> None:
    """Distance calculations."""
    code = codes.FiveQubitCode()
    code._is_subsystem_code = True  # test that this does not break anything

    # cover calls to the known code exact distance
    assert code.get_code_params() == (5, 1, 3)
    assert code.get_distance(bound=True) == 3

    # "forget" the code distance and compute estimate
    code._distance = None
    assert code.get_distance_bound(num_trials=0) == 5
    assert code.get_distance_bound(cutoff=5) == 5

    # computing an exact distance but providing bounding arguments raises a warning
    with pytest.warns(UserWarning, match="ignored"):
        assert code.get_distance(test_arg=True)

    code._distance = None
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(NotImplementedError, match="not supported"),
    ):
        code.get_distance(bound=True)
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        pytest.raises(ValueError, match="Arguments not recognized"),
    ):
        code.get_distance(bound=True, test=True)
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.codes.get_distance_bound", return_value=3),
    ):
        assert code.get_distance(bound=True) == 3

    # the distance of dimension-0 codes is undefined
    assert codes.QuditCode([[0, 1]]).get_distance() is np.nan

    # fallback pythonic brute-force distance calculation
    surface_code = codes.SurfaceCode(2, field=3)
    with unittest.mock.patch("qldpc.codes.CSSCode.get_distance_if_known", return_value=None):
        with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
            assert codes.QuditCode.get_distance_exact(surface_code) == 2


@pytest.mark.parametrize("field", [2, 3])
def test_conversions_quantum(field: int, bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a code."""
    code = get_random_qudit_code(bits, checks, field)
    graph = codes.QuditCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))


@pytest.mark.parametrize("field", [2, 3])
def test_qudit_stabilizers(field: int, bits: int = 5, checks: int = 3) -> None:
    """Stabilizers of a QuditCode."""
    code_a = get_random_qudit_code(bits, checks, field)
    strings = code_a.get_strings()
    code_b = codes.QuditCode.from_strings(*strings, field=field)
    assert code_a == code_b
    assert strings == code_b.get_strings()

    with pytest.raises(ValueError, match="different lengths"):
        codes.QuditCode.from_strings("I", "I I", field=field)


def test_trivial_deformations(num_qudits: int = 5, num_checks: int = 3, field: int = 3) -> None:
    """Trivial local Clifford deformations do not modify a code."""
    code = get_random_qudit_code(num_qudits, num_checks, field)
    assert code == code.conjugated()


def get_codes_for_testing_ops() -> Iterator[codes.CSSCode]:
    """Iterate over some codes for testing operator constructions."""
    # Bacon-Shor code and toric codes
    code_a = codes.BaconShorCode(3, field=3)
    code_b = codes.ToricCode(4, field=4)

    # promote some gauge qudits of the Bacon-Shor code to logicals
    matrix_x = np.vstack([code_a.get_gauge_ops(Pauli.X)[:2], code_a.get_stabilizer_ops(Pauli.X)])
    matrix_z = np.vstack([code_a.get_gauge_ops(Pauli.Z)[:2], code_a.get_stabilizer_ops(Pauli.Z)])
    code_c = codes.CSSCode(matrix_x, matrix_z)

    # gauge out a logical qudit of the surface code
    matrix_x = np.vstack([code_b.get_logical_ops(Pauli.X)[:1], code_b.get_stabilizer_ops(Pauli.X)])
    matrix_z = np.vstack([code_b.get_logical_ops(Pauli.Z)[:1], code_b.get_stabilizer_ops(Pauli.Z)])
    code_d = codes.CSSCode(matrix_x, matrix_z)

    yield code_a
    yield code_b
    yield code_c
    yield code_d


def get_symplectic_form(half_dimension: int, field: type[galois.FieldArray]) -> galois.FieldArray:
    """Get the symplectic form over a given field."""
    identity = field.Identity(half_dimension)
    zeros = field.Zeros((half_dimension, half_dimension))
    return np.block([[zeros, identity], [-identity, zeros]]).view(field)


def test_qudit_ops() -> None:
    """Logical and gauge operator construction for Galois qudit codes."""
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    logical_ops = code.get_logical_ops()
    assert logical_ops.shape == (2 * code.dimension, 2 * len(code))
    assert np.array_equal(logical_ops[0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    assert np.array_equal(logical_ops[1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert code.get_logical_ops() is code._logical_ops

    code = codes.QuditCode.from_strings(*code.get_strings(), "I I I I I")
    assert np.array_equal(logical_ops, code.get_logical_ops())

    for code in get_codes_for_testing_ops():
        code = codes.QuditCode(code.matrix)
        stabilizer_ops = code.get_stabilizer_ops()
        logical_ops = code.get_logical_ops()
        gauge_ops = code.get_gauge_ops()
        assert not np.any(symplectic_conjugate(stabilizer_ops) @ stabilizer_ops.T)
        assert np.array_equal(
            symplectic_conjugate(gauge_ops) @ gauge_ops.T,
            get_symplectic_form(code.gauge_dimension, code.field),
        )
        assert np.array_equal(
            symplectic_conjugate(logical_ops) @ logical_ops.T,
            get_symplectic_form(code.dimension, code.field),
        )

    # test the guarantee of stabilizer canonicalization
    code = codes.FiveQubitCode()
    code._is_subsystem_code = True
    stabilizer_ops = code.get_stabilizer_ops(canonicalized=True)
    stabilizer_ops = np.vstack([stabilizer_ops, stabilizer_ops[-1]]).view(code.field)
    code._stabilizer_ops = stabilizer_ops
    assert np.array_equal(code.get_stabilizer_ops(), stabilizer_ops)
    assert np.array_equal(code.get_stabilizer_ops(canonicalized=True), stabilizer_ops[:-1])


def test_code_deformation() -> None:
    """Deform a code by a physical Clifford transformation."""
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    code.get_logical_ops()
    assert code.get_strings()[0] == "X Z Z X I"
    assert code.conjugated([0]).get_strings()[0] == "Z Z Z X I"
    assert code.deformed("H 0").get_strings()[0] == "Z Z Z X I"
    with pytest.raises(ValueError, match="violate parity checks"):
        code.deformed("H 0", preserve_logicals=True)

    code = codes.SteaneCode()
    assert np.array_equal(
        code.deformed("H 0 1 2 3 4 5 6", preserve_logicals=True).get_logical_ops(),
        code.get_logical_ops(),
    )

    code = codes.ToricCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        code.deformed("H 0")


def test_qudit_concatenation() -> None:
    """Concatenate qudit codes."""
    code_5q = codes.FiveQubitCode()

    # determine the number of copies of the outer code automatically
    code = codes.QuditCode.concatenate(code_5q, code_5q)
    assert len(code) == len(code_5q) ** 2
    assert code.dimension == code_5q.dimension

    # determine the number of copies of the outer and inner codes from wiring data
    wiring = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    code = codes.QuditCode.concatenate(code_5q, code_5q, wiring)
    assert len(code) == 10 * len(code_5q)
    assert code.dimension == 2 * code_5q.dimension

    # cover some errors
    with pytest.raises(ValueError, match="different fields"):
        codes.QuditCode.concatenate(code_5q, codes.ToricCode(2, field=3))
    with pytest.raises(ValueError, match="divisible"):
        codes.QuditCode.concatenate(code_5q, code_5q, [0, 1, 2])


def test_quantum_capacity() -> None:
    """Logical error rates in a code capacity model."""
    code = codes.FiveQubitCode()

    logical_error_rate = code.get_logical_error_rate_func(num_samples=10)
    assert logical_error_rate(0) == (0, 0)  # no logical error with zero uncertainty

    with pytest.raises(ValueError, match="error rates greater than"):
        logical_error_rate(1)

    # guaranteed logical X and Z errors
    for pauli_bias in [(1, 0, 0), (0, 0, 1)]:
        logical_error_rate = code.get_logical_error_rate_func(10, 1, pauli_bias)
        assert logical_error_rate(1)[0] == 1


####################################################################################################
# CSS code tests


def test_css_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    code_x = codes.ClassicalCode.random(3, 2)

    code_z = ~code_x
    code = codes.CSSCode(code_x, code_z)
    assert code.get_weight() == max(code_x.get_weight(), code_z.get_weight())
    assert code.num_checks_x == code_x.num_checks
    assert code.num_checks_z == code_z.num_checks
    assert code.num_checks == code.num_checks_x + code.num_checks_z
    assert code == codes.CSSCode(code.code_x, code.code_z)

    # equivlence to QuditCode with the parity check matrix
    equiv_code = codes.QuditCode(code.matrix)
    assert codes.CSSCode.equiv(code, equiv_code)

    # equivlence to code with redundant stabilizers
    redundant_code = codes.CSSCode(np.vstack([code.matrix_x, code.matrix_x]), code.matrix_z)
    assert codes.CSSCode.equiv(code, redundant_code)

    code_z = codes.ClassicalCode.random(4, 2)
    with pytest.raises(ValueError, match="incompatible"):
        codes.CSSCode(code_x, code_z)

    with pytest.raises(ValueError, match="incompatible"):
        code_z = codes.ClassicalCode.random(3, 2, field=code_x.field.order**2)
        codes.CSSCode(code_x, code_z)

    # subgraphs for syndrome extraction
    assert_valid_subgraphs(code)
    subgraphs = code.get_syndrome_subgraphs()
    assert nx.utils.graphs_equal(subgraphs[0], code.get_graph(Pauli.X))
    assert nx.utils.graphs_equal(subgraphs[1], code.get_graph(Pauli.Z))


def test_css_ops() -> None:
    """Logical and stabilizer operator construction for CSS codes."""
    code: codes.CSSCode

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, field=3))

    # swap around logical operators
    code.set_logical_ops_xz(
        code.get_logical_ops(Pauli.X)[::-1],
        code.get_logical_ops(Pauli.Z)[::-1],
    )

    # identify stabilizer group
    code._stabilizer_ops = None
    assert not np.any(code.get_stabilizer_ops() @ symplectic_conjugate(code.get_logical_ops()).T)
    assert not np.any(code.get_stabilizer_ops() @ symplectic_conjugate(code.get_gauge_ops()).T)

    # successfully construct and reduce logical operators in a code with "over-complete" checks
    dist = 4
    code = codes.ToricCode(dist, rotated=True)
    assert code.canonicalized.num_checks < code.num_checks
    assert code.get_code_params() == (dist**2, 2, dist)
    code.reduce_logical_ops()
    logical_ops_x = code.get_logical_ops(Pauli.X)
    logical_ops_z = code.get_logical_ops(Pauli.Z, symplectic=True)
    assert not np.any(np.count_nonzero(logical_ops_x.view(np.ndarray), axis=1) < dist)
    assert not np.any(np.count_nonzero(logical_ops_z.view(np.ndarray), axis=1) < dist)


def test_distance_css() -> None:
    """Distance calculations for CSS codes."""
    code: codes.CSSCode

    # qutrit code distance
    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    assert code.get_distance(bound=False) == 2

    with (
        unittest.mock.patch("qldpc.codes.CSSCode.get_distance_if_known", return_value=None),
        unittest.mock.patch("qldpc.codes.HGPCode._get_distance_exact", return_value=NotImplemented),
    ):
        with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
            assert code.get_distance(bound=False) == 2

        assert code.get_distance_bound(cutoff=len(code)) == len(code)
        with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
            assert code.get_distance(bound=True) <= len(code)
        with (
            unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
            unittest.mock.patch("qldpc.external.codes.get_distance_bound", return_value=-1),
        ):
            assert code.get_distance(bound=True) == -1

    # qubit code distance
    code = codes.SHPCode(codes.RepetitionCode(2))
    with unittest.mock.patch(
        "qldpc.codes.SHPCode._get_distance_exact", return_value=NotImplemented
    ):
        assert code.get_distance_exact() == 2
        assert code.get_distance_bound_with_decoder(Pauli.X, cutoff=len(code)) <= len(code)

        # computing an exact distance but providing bounding arguments raises a warning
        with pytest.warns(UserWarning, match="ignored"):
            assert code.get_distance(bound=False, test_arg=True)

    # the distance of a dimension-0 quantum code is undefined
    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    code = codes.HGPCode(trivial_code)
    assert code.dimension == 0
    assert code.get_distance(bound=True) is np.nan
    assert code.get_distance(bound=False) is np.nan


def test_stacking_css_codes() -> None:
    """Stack two CSS codes."""
    steane_code = codes.SteaneCode()
    code = codes.CSSCode.stack(steane_code, steane_code)
    assert len(code) == len(steane_code) * 2
    assert code.dimension == steane_code.dimension * 2

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        qudit_code = codes.SurfaceCode(2, field=3)
        code = codes.CSSCode.stack(steane_code, qudit_code)

    # stacking a CSSCode with a QuditCode requires using QuditCode.stack
    codes.QuditCode.stack(steane_code, codes.FiveQubitCode())
    with pytest.raises(TypeError, match="requires CSSCode inputs"):
        codes.CSSCode.stack(steane_code, codes.FiveQubitCode())


def test_css_concatenation() -> None:
    """Concatenate CSS codes."""
    code_c4 = codes.ToricCode(2)

    # determine the number of copies of the outer code automatically
    code = codes.CSSCode.concatenate(code_c4, code_c4)
    assert len(code) == len(code_c4) ** 2
    assert code.dimension == code_c4.dimension**2

    # determine the number of copies of the outer and inner codes from wiring data
    wiring = [0, 2, 4, 6, 1, 3, 5, 7]
    code = codes.CSSCode.concatenate(code_c4, code_c4, wiring)
    assert len(code) == 4 * len(code_c4)
    assert code.dimension == 2 * code_c4.dimension

    # inheriting logical operators yields different logical operators!
    code_alt = codes.CSSCode.concatenate(code_c4, code_c4, wiring, inherit_logicals=False)
    assert not np.array_equal(code.get_logical_ops(), code_alt.get_logical_ops())

    # cover some errors
    with pytest.raises(TypeError, match="CSSCode inputs"):
        codes.CSSCode.concatenate(code_c4, codes.FiveQubitCode())


def test_css_capacity() -> None:
    """Logical error rates in a code capacity model."""
    code = codes.SteaneCode()

    logical_error_rate = code.get_logical_error_rate_func(num_samples=10)
    assert logical_error_rate(0) == (0, 0)  # no logical error with zero uncertainty

    with pytest.raises(ValueError, match="error rates greater than"):
        logical_error_rate(1)

    # guaranteed logical X and Z errors
    for pauli_bias in [(1, 0, 0), (0, 0, 1)]:
        logical_error_rate = code.get_logical_error_rate_func(10, 1, pauli_bias)
        assert logical_error_rate(1)[0] == 1
