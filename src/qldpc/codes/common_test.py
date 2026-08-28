"""Unit tests for common.py.

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
from collections.abc import Iterator, Sequence

import galois
import networkx as nx
import numpy as np
import pytest

from qldpc import abstract, codes, external, math
from qldpc.objects import PAULIS_XZ, Pauli

####################################################################################################
# classical code tests


def test_constructions_classical(pytestconfig: pytest.Config) -> None:
    """Classical code constructions."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.ClassicalCode.random(5, 3, seed=np.random.randint(2**31))
    assert len(code) == code.num_bits == 5
    assert code.num_checks == 3
    assert "ClassicalCode" in str(code)
    assert code.get_random_word() in code

    # reordering the rows of the generator matrix results in a valid generator matrix
    code.set_generator(np.roll(code.generator, shift=1, axis=0))
    assert codes.ClassicalCode(code).generator is code.generator

    code = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**31))
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

    # boolean arrays are treated as 0/1 integers
    bool_matrix = [[True, False, True], [False, True, True]]
    code = codes.ClassicalCode(bool_matrix)
    assert np.array_equal(code.matrix, np.array(bool_matrix, dtype=int))
    assert np.array_equal(codes.ClassicalCode(np.array(bool_matrix)).matrix, code.matrix)

    # construct a code from its generator matrix
    code = codes.ClassicalCode.random(6, 4, field=3)
    assert code.is_equiv_to(codes.ClassicalCode.from_generator(code.generator))

    # puncture and shorten a code
    for field in [2, 3]:
        code = codes.ClassicalCode.random(6, 4, field=field)
        bits_to_remove = np.random.choice(range(len(code)), size=2, replace=False)
        bits_to_keep = [bit for bit in range(len(code)) if bit not in bits_to_remove]
        code._matrix[:2, bits_to_remove] = 1  # ensure we have nontrivial row-reduction to do
        punctured_code = code.punctured(bits_to_remove)
        assert punctured_code.is_equiv_to(
            codes.ClassicalCode.from_generator(code.generator[:, bits_to_keep])
        )
        assert punctured_code.is_equiv_to(code.dual().shortened(bits_to_remove).dual())

    # shortening a repetition code yields a trivial code
    code = codes.RepetitionCode(3)
    assert np.array_equal(list(code.shortened([0]).iter_words()), [[0, 0]])

    # stack two codes
    code_a = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**31))
    code_b = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**31))
    code = codes.ClassicalCode.stack([code_a, code_b])
    assert len(code) == len(code_a) + len(code_b)
    assert code.dimension == code_a.dimension + code_b.dimension

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        code_b = codes.RepetitionCode(2)
        code = codes.ClassicalCode.stack([code_a, code_b])


def test_named_codes(order: int = 2) -> None:
    """Named codes from the GAP computer algebra system."""
    code = codes.RepetitionCode(order)
    checks = [list(row) for row in code.matrix.view(np.ndarray)]

    with unittest.mock.patch(
        "qldpc.external.codes.get_classical_code", return_value=(checks, None)
    ):
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

    # forget the exact code distance, and re-compute (or estimate) it in various ways
    rep_code.forget_distance()
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
    assert np.isnan(trivial_code.get_distance_exact())
    assert np.isnan(trivial_code.get_distance_bound())
    assert (
        np.count_nonzero(random_vector)
        == trivial_code.get_distance_exact(vector=random_vector)
        == trivial_code.get_distance_bound(vector=random_vector)
    )

    # compute distance of a trinary repetition code
    rep_code = codes.RepetitionCode(bits, field=3)
    rep_code.forget_distance()
    assert rep_code.get_distance_exact(cutoff=bits) == rep_code.get_distance_exact() == bits


def test_conversions_classical(bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a classical code."""
    code = codes.ClassicalCode.random(bits, checks)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(code.graph))


def test_automorphism(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Compute automorphism groups."""
    code: codes.ClassicalCode = codes.HammingCode(2, field=2)
    automorphisms = "\n(1,2)\n(2,3)\n"

    # raise an error when GAP is not installed
    external.gap.require_package.cache_clear()
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot build GAP group"),
    ):
        codes.RepetitionCode(2).get_automorphism_group()

    # otherwise, check that automorphisms do indeed preserve the code space
    # this pytest.warns block intentionally wraps a loop of warning-emitting calls
    with (  # noqa: PT031
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        pytest.warns(UserWarning, match="with_magma=True"),
    ):
        for code, automorphisms in [
            (codes.HammingCode(2, field=2), "\n(1,2)\n(2,3)\n"),
            (codes.HammingCode(2, field=3), "\n()\n(2,4,3)\n(2,3,4)\n"),
        ]:
            with unittest.mock.patch("qldpc.external.gap.get_output", return_value=automorphisms):
                group = code.get_automorphism_group()
                for member in group.generate():
                    permutation = member.to_matrix().view(code.field)
                    assert not np.any(code.matrix @ permutation @ code.generator.T)

    # compute an automorphism group with MAGMA
    user_inputs = iter(
        ["Permutation group acting on a set of cardinality 2", "Order = 2", "    (1, 2)", ""]
    )
    monkeypatch.setattr("builtins.input", lambda: next(user_inputs))
    code = codes.RepetitionCode(2)
    group = abstract.CyclicGroup(2)
    assert code.get_automorphism_group(with_magma=True).equiv(group)
    capsys.readouterr()  # intercept print statements


def test_classical_capacity() -> None:
    """Logical error rates in a code capacity model."""
    code = codes.RepetitionCode(2)
    logical_error_rate_func = code.get_logical_error_rate_func(num_samples=1, max_error_rate=1)
    assert logical_error_rate_func(0) == (0, 0)  # no logical error with zero uncertainty
    assert logical_error_rate_func([1])[0] == 1  # guaranteed logical error

    # with an erasure-enabled decoder, unrecognised syndromes are discarded
    logical_error_rate_func = code.get_logical_error_rate_func(
        num_samples=1, max_error_rate=1, with_lookup=True, max_weight=0, add_erasure_bit=True
    )
    assert logical_error_rate_func(0, discard_rate=True) == (0, 0)  # no errors at p=0
    assert logical_error_rate_func(0.5, discard_rate=True)[0] > 0  # nonzero syndromes → erasure
    assert logical_error_rate_func.truncation_error_bound(0.5) < 1
    assert logical_error_rate_func.truncation_error_bound([1]) == 0

    # test cap on physical error rate
    logical_error_rate_func = code.get_logical_error_rate_func(num_samples=1, max_error_rate=0.5)
    with pytest.raises(ValueError, match="error rates greater than"):
        logical_error_rate_func(1)


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
    with pytest.raises(
        ValueError, match=r"3-dimensional qudits\.\s+Try calling QuditCode\.num_qudits"
    ):
        assert get_random_qudit_code(num_qubits, num_checks, field=3).num_qubits


def assert_valid_subgraphs(code: codes.QuditCode) -> None:
    """The union of subgraphs used for syndrome measurement is the entire Tanner graph."""
    assert nx.utils.graphs_equal(
        code.graph, functools.reduce(nx.compose, code.get_syndrome_subgraphs())
    )


def test_qudit_codes() -> None:
    """Miscellaneous qudit code tests and coverage."""
    code = codes.FiveQubitCode()
    assert code.dimension == 1
    assert code.get_weight() == 4
    assert code.get_logical_ops(Pauli.X).shape == code.get_logical_ops(Pauli.Z).shape
    assert code.is_equiv_to(codes.QuditCode(code))
    assert_valid_subgraphs(code)

    # parity checks whose support overlaps no other check still appear in the subgraphs, and a
    # check with no support at all is simply omitted (it contributes no edges to the Tanner graph)
    assert_valid_subgraphs(codes.QuditCode.from_strings(["Y Y I I", "I I Z Z"]))
    assert_valid_subgraphs(codes.QuditCode.from_strings(["X X X"]))
    assert_valid_subgraphs(codes.QuditCode.from_strings(["X X X", "I I I"]))

    # equivalence to code with redundant stabilizers
    redundant_code = codes.QuditCode(np.vstack([code.matrix, code.matrix]))
    assert code.is_equiv_to(redundant_code)

    # the logical ops of the redundant code are valid ops of the original code
    code.set_logical_ops(redundant_code.get_logical_ops())  # also validates the logical ops

    # stacking two codes
    two_codes = codes.QuditCode.stack([code] * 2)
    assert len(two_codes) == len(code) * 2
    assert two_codes.dimension == code.dimension * 2

    # swapping logical X ops on the two encoded qubits breaks commutation relations
    logical_ops = two_codes.get_logical_ops().copy()
    logical_ops[0], logical_ops[1] = logical_ops[1], logical_ops[0]
    with pytest.raises(ValueError, match="incorrect commutation relations"):
        two_codes.set_logical_ops(logical_ops, skip_validation=False)

    # making an X-type logical anticommute with another X-type logical is rejected; the X-Z
    # cross-type commutation relations alone do not detect a broken intra-type relation
    logical_ops = two_codes.get_logical_ops().copy()
    logical_ops[1] += logical_ops[two_codes.dimension]  # Lx[1] += Lz[0]
    with pytest.raises(ValueError, match="incorrect commutation relations"):
        two_codes.set_logical_ops(logical_ops, skip_validation=False)

    # adding a destabilizer to a logical operator preserves the commutation relations among the
    # logical operators but violates a parity check
    logical_ops = two_codes.get_logical_ops().copy()
    logical_ops[0] += two_codes.get_destabilizer_ops()[0]
    with pytest.raises(ValueError, match="violate parity checks"):
        two_codes.set_logical_ops(logical_ops, skip_validation=False)

    # providing an incorrect number of logical operators throws an error
    logical_ops = two_codes.get_logical_ops().copy()[[0, two_codes.dimension], :]
    with pytest.raises(ValueError, match="incorrect number"):
        two_codes.set_logical_ops(logical_ops, skip_validation=False)

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        second_code = codes.SurfaceCode(2, field=3)
        codes.QuditCode.stack([code, second_code])


def test_distance_qudit() -> None:
    """Distance calculations."""
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    code._is_subsystem_code = True  # test that this does not break anything

    # cover calls to the known code exact distance
    assert code.get_code_params() == (5, 1, 3)
    assert code.get_distance(bound=True) == 3

    # compute an estimate of code distance
    code.forget_distance()
    assert code.get_distance_bound(num_trials=0) == 5
    assert code.get_distance_bound(cutoff=5) == 5

    # computing an exact distance but providing bounding arguments raises a warning
    with pytest.warns(UserWarning, match="ignored"):
        assert code.get_distance(test_arg=True)

    code.forget_distance()
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

    # mock computing distance with GAP
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.codes.get_distance_bound", return_value=-1),
    ):
        assert code.get_distance(bound=True) == -1

    # the distance of dimension-0 codes is undefined
    assert np.isnan(codes.QuditCode([[0, 1]]).get_distance())

    # fallback pythonic brute-force distance calculation
    code = codes.QuditCode(codes.SurfaceCode(2, field=3).matrix)
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        assert code.get_distance_exact(cutoff=len(code)) <= len(code)
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        assert code.get_distance_exact() == 2


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
    code_b = codes.QuditCode.from_strings(strings, field=field)
    assert code_a == code_b
    assert strings == code_b.get_strings()

    with pytest.raises(ValueError, match=r"different lengths \(1 and 2\)"):
        codes.QuditCode.from_strings(["I", "II"], field=field)
    with pytest.raises(ValueError, match="empty collection"):
        codes.QuditCode.from_strings([], field=field)


def test_from_qecdb_id() -> None:
    """Retrieve a code from qecdb.org."""
    strings = ["XXXX", "ZZZZ"]
    distance = 2
    is_css = True
    code_data = (strings, distance, is_css)
    with unittest.mock.patch("qldpc.external.codes.get_quantum_code", return_value=code_data):
        code = codes.QuditCode.from_qecdb_id("")
        assert code.is_equiv_to(codes.C4Code())


def test_qudit_deformations() -> None:
    """Local Fourier transforms of a QuditCode."""
    code = codes.QuditCode(codes.SHYPSCode(2))
    code.get_logical_ops()
    code.get_stabilizer_ops()
    code.get_gauge_ops()
    assert code == code.conjugated([]) == code.deformed("")
    assert code.conjugated() == code.deformed("H " + " ".join(map(str, range(len(code)))))

    # conjugation transforms all known operators, not just the parity check matrix
    def swap_xz(ops: galois.FieldArray) -> galois.FieldArray:
        swapped = ops.reshape(-1, 2, len(code))[:, ::-1, :].reshape(-1, 2 * len(code))
        return swapped.view(code.field)

    # conjugated() with no arguments transforms all qudits
    assert code.conjugated() == code.conjugated(range(len(code)))

    conjugate = code.conjugated()
    assert np.array_equal(conjugate.get_logical_ops(), swap_xz(code.get_logical_ops()))
    assert np.array_equal(conjugate.get_stabilizer_ops(), swap_xz(code.get_stabilizer_ops()))
    assert np.array_equal(conjugate.get_gauge_ops(), swap_xz(code.get_gauge_ops()))

    with pytest.raises(ValueError, match="only supported for qubit codes"):
        codes.QuditCode(code.matrix, field=3).deformed("")

    # the Steane code is self-dual
    code = codes.SteaneCode()
    assert code.is_equiv_to(code.deformed("H 0 1 2 3 4 5 6", preserve_logicals=True))


def test_conjugated_over_qudits() -> None:
    """conjugated() is a symplectic map, so it preserves code structure over every field."""
    conj = math.symplectic_conjugate
    for base in [codes.BaconShorCode(3, field=3), codes.ToricCode(4, field=4)]:
        code = codes.QuditCode(base.matrix)
        code.get_logical_ops()  # populate the cache so conjugated() transforms it too
        conjugated = code.conjugated([0, 5, 7])
        # the transform preserves every symplectic product, so the parity checks keep their
        # commutation relations and the code stays the same size
        assert np.array_equal(
            code.matrix @ conj(code.matrix).T, conjugated.matrix @ conj(conjugated.matrix).T
        )
        assert codes.QuditCode(conjugated.matrix).dimension == code.dimension
        # the transformed logical operators are still a valid symplectic basis for the new code
        conjugated.set_logical_ops(conjugated.get_logical_ops())


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
    return math.block_matrix([[0, identity], [-identity, 0]]).view(field)


def test_qudit_ops(pytestconfig: pytest.Config) -> None:
    """Logical and gauge operator construction for Galois qudit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    logical_ops = code.get_logical_ops()
    assert logical_ops.shape == (2 * code.dimension, 2 * len(code))
    assert np.array_equal(logical_ops[0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    assert np.array_equal(logical_ops[1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert code.get_logical_ops() is code._logical_ops

    code = codes.QuditCode.from_strings(code.get_strings() + ["IIIII"])
    assert np.array_equal(logical_ops, code.get_logical_ops())

    for code in get_codes_for_testing_ops():
        code = codes.QuditCode(code.matrix)
        stabilizer_ops = code.get_stabilizer_ops()
        logical_ops = code.get_logical_ops()
        gauge_ops = code.get_gauge_ops()
        assert not np.any(stabilizer_ops @ math.symplectic_conjugate(stabilizer_ops).T)

        # destabilizers anticommute with their paired stabilizer, commute with everything else
        destabilizer_ops = code.get_destabilizer_ops()
        minimal_stabilizer_ops = code.get_stabilizer_ops()
        if len(minimal_stabilizer_ops) != len(code) - code.dimension - code.gauge_dimension:
            minimal_stabilizer_ops = code.get_stabilizer_ops(canonicalized=True)
        assert np.array_equal(
            destabilizer_ops @ math.symplectic_conjugate(minimal_stabilizer_ops).T,
            code.field.Identity(len(minimal_stabilizer_ops)),
        )
        assert not np.any(destabilizer_ops @ math.symplectic_conjugate(destabilizer_ops).T)
        assert not np.any(destabilizer_ops @ math.symplectic_conjugate(logical_ops).T)
        assert not np.any(destabilizer_ops @ math.symplectic_conjugate(gauge_ops).T)

        # destabilizers can be retrieved by Pauli type, partitioning the full destabilizer matrix
        pivots_x = math.first_nonzero_cols(destabilizer_ops) < len(code)
        assert np.array_equal(code.get_destabilizer_ops(Pauli.X), destabilizer_ops[pivots_x])
        assert np.array_equal(code.get_destabilizer_ops(Pauli.Z), destabilizer_ops[~pivots_x])
        assert np.array_equal(
            gauge_ops @ math.symplectic_conjugate(gauge_ops).T,
            get_symplectic_form(code.gauge_dimension, code.field),
        )
        assert np.array_equal(
            logical_ops @ math.symplectic_conjugate(logical_ops).T,
            get_symplectic_form(code.dimension, code.field),
        )

        logical_ops_x = logical_ops[: code.dimension]
        logical_ops_z = logical_ops[code.dimension :]

        # set logical X operators, determine suitable logical Z operators automatically
        order = np.random.permutation(code.dimension)  # random permutation of logicals
        code.set_logical_ops_x(logical_ops_x[order])
        assert np.array_equal(code.get_logical_ops(Pauli.Z), logical_ops_z[order])

        # set logical Z operators, determine suitable logical X operators automatically
        order = np.random.permutation(code.dimension)  # random permutation of logicals
        code.set_logical_ops_z(logical_ops_z[order])
        assert np.array_equal(code.get_logical_ops(Pauli.X), logical_ops_x[order])

    # test the guarantee of stabilizer canonicalization
    code = codes.FiveQubitCode()
    code._is_subsystem_code = True
    stabilizer_ops = code.get_stabilizer_ops(canonicalized=True)
    stabilizer_ops = np.vstack([stabilizer_ops, stabilizer_ops[-1]]).view(code.field)
    code._stabilizer_ops = stabilizer_ops
    assert np.array_equal(code.get_stabilizer_ops(), stabilizer_ops)
    assert np.array_equal(code.get_stabilizer_ops(canonicalized=True), stabilizer_ops[:-1])


def test_qudit_subsystem_logical_ops() -> None:
    """Non-CSS subsystem codes get a valid symplectic basis of logical operators.

    The base QuditCode.get_logical_ops extracts the logical operators of a subsystem code as a
    symplectic basis of the gauge group's centralizer.  This test checks that basis is valid --
    correct commutation relations, and commuting with the gauge group -- for codes whose parity
    checks mix X-type and Z-type support.
    """

    def assert_valid_basis(code: codes.QuditCode) -> None:
        assert code.is_subsystem_code
        logical_ops = code.get_logical_ops()
        assert len(logical_ops) == 2 * code.dimension
        # a valid basis has the full symplectic Gram matrix [[0, I], [-I, 0]]: X-type logicals
        # mutually commute, Z-type logicals mutually commute, and each logical anticommutes only
        # with its dual
        assert np.array_equal(
            logical_ops @ math.symplectic_conjugate(logical_ops).T,
            get_symplectic_form(code.dimension, code.field),
        )
        # logical operators commute with every gauge generator (they lie in the centralizer)
        assert not np.any(code.matrix @ math.symplectic_conjugate(logical_ops).T)
        # gauge operators (dual().get_logical_ops()) are likewise a valid symplectic basis
        gauge_ops = code.get_gauge_ops()
        assert np.array_equal(
            gauge_ops @ math.symplectic_conjugate(gauge_ops).T,
            get_symplectic_form(code.gauge_dimension, code.field),
        )
        # the basis round-trips through the commutation-relation validation of set_logical_ops
        code.set_logical_ops(logical_ops)

    # conjugating some qudits mixes a Bacon-Shor code's X/Z support into a non-CSS subsystem code.
    # The shipped reproduction must yield a valid basis without error, including via get_gauge_ops
    # and get_distance, which build the logical operators of the dual code.
    shipped = codes.BaconShorCode(3).conjugated([1, 3, 5])
    assert_valid_basis(shipped)
    assert shipped.get_distance() == 3

    # property test over several fields, for one and two logical qudits.  Away from GF(2), build
    # the non-CSS codes with _local_fourier, as conjugated() is only symplectic over GF(2).
    for field in [galois.GF(2), galois.GF(3), galois.GF(4)]:
        bacon_shor = codes.BaconShorCode(3, field=field.order).matrix
        single = _local_fourier(bacon_shor, [1, 3, 5])
        double = _local_fourier(_direct_sum(bacon_shor, bacon_shor), [1, 3, 5, 10, 14])
        for matrix, expected_dimension in [(single, 1), (double, 2)]:
            code = codes.QuditCode(matrix)
            # the parity checks genuinely mix X-type and Z-type support (the code is not CSS)
            has_x = np.any(code.matrix[:, : len(code)], axis=1)
            has_z = np.any(code.matrix[:, len(code) :], axis=1)
            assert np.any(has_x & has_z)
            assert code.dimension == expected_dimension
            assert_valid_basis(code)

    # a subsystem code with no logical qudits yields an empty basis rather than an error
    five_qubit = codes.FiveQubitCode()
    gauged = codes.QuditCode(np.vstack([five_qubit.matrix, five_qubit.get_logical_ops()]))
    assert gauged.is_subsystem_code and gauged.dimension == 0
    assert gauged.get_logical_ops().shape == (0, 2 * len(gauged))


def _local_fourier(matrix: galois.FieldArray, qudits: Sequence[int]) -> galois.FieldArray:
    """Apply the local symplectic Fourier map ``(x_i, z_i) -> (z_i, -x_i)`` on the given qudits.

    This preserves every symplectic inner product, so it maps a valid code to a valid code over any
    field.  Applied to some qudits of a CSS code, it can mix X-type and Z-type support to build a
    non-CSS code -- the regime that exercises the base QuditCode.get_logical_ops construction (CSS
    codes use their own override).
    """
    field = type(matrix)
    num_qudits = matrix.shape[1] // 2
    x_bits, z_bits = matrix[:, :num_qudits].copy(), matrix[:, num_qudits:].copy()
    x_bits[:, qudits], z_bits[:, qudits] = z_bits[:, qudits].copy(), -x_bits[:, qudits]
    return np.hstack([x_bits, z_bits]).view(field)


def _direct_sum(matrix_a: galois.FieldArray, matrix_b: galois.FieldArray) -> galois.FieldArray:
    """Combine two symplectic parity check matrices into a code acting on disjoint qudits.

    The result encodes the logical qudits of both summands, giving a simple way to build subsystem
    codes with more than one logical qudit.
    """
    field = type(matrix_a)
    num_a, num_b = matrix_a.shape[1] // 2, matrix_b.shape[1] // 2
    num_qudits = num_a + num_b
    result = field.Zeros((len(matrix_a) + len(matrix_b), 2 * num_qudits))
    result[: len(matrix_a), :num_a] = matrix_a[:, :num_a]
    result[: len(matrix_a), num_qudits : num_qudits + num_a] = matrix_a[:, num_a:]
    result[len(matrix_a) :, num_a:num_qudits] = matrix_b[:, :num_b]
    result[len(matrix_a) :, num_qudits + num_a :] = matrix_b[:, num_b:]
    return result


def test_get_standard_form_data_subsystem() -> None:
    """QuditCode.get_standard_form_data reduces a subsystem code's checks to standard form.

    get_logical_ops does not route subsystem codes through get_standard_form_data, so this covers
    its subsystem branch directly against the identity-block structure that the method documents.
    """
    code = codes.QuditCode(codes.BaconShorCode(3).matrix)
    assert code.is_subsystem_code
    matrix, qudit_locs, row_sectors, col_sectors = code.get_standard_form_data()
    rows_sx, rows_gx, rows_sz, rows_gz = row_sectors
    cols_sx, cols_gx, _cols_lx, cols_sz, cols_gz, _cols_lz = col_sectors

    # each stabilizer/gauge pivot block is an identity matrix, as documented
    for rows, pauli_index, cols in [
        (rows_sx, 0, cols_sx),
        (rows_gx, 0, cols_gx),
        (rows_sz, 1, cols_sz),
        (rows_gz, 1, cols_gz),
    ]:
        block = matrix[rows, pauli_index, cols]
        assert block.shape[0] == block.shape[1]
        assert np.array_equal(block, code.field.Identity(len(block)))

    # undoing the qudit permutation recovers the canonicalized parity check matrix
    reordered = matrix[:, :, np.argsort(qudit_locs)].reshape(-1, 2 * len(code)).view(code.field)
    assert np.array_equal(reordered.row_space(), code.canonicalized.matrix)


def test_set_logical_ops_single_type_support() -> None:
    """QuditCode.set_logical_ops_x/z accept width-n single-type support."""
    css_code = codes.SteaneCode()
    matrix = css_code.matrix
    num_qudits = len(css_code)

    # a valid pure-X (pure-Z) logical basis in symplectic (k, 2n) form, and its width-n support
    logical_x = css_code.get_logical_ops(Pauli.X, symplectic=True).view(css_code.field)
    logical_z = css_code.get_logical_ops(Pauli.Z, symplectic=True).view(css_code.field)
    assert not np.any(logical_x[:, num_qudits:])  # no Z-type support
    assert not np.any(logical_z[:, :num_qudits])  # no X-type support

    # setting the width-n support is equivalent to setting the full symplectic operators
    code_full = codes.QuditCode(matrix)
    code_full.set_logical_ops_x(logical_x)
    code_half = codes.QuditCode(matrix)
    code_half.set_logical_ops_x(logical_x[:, :num_qudits])
    assert np.array_equal(code_full.get_logical_ops(), code_half.get_logical_ops())

    code_full = codes.QuditCode(matrix)
    code_full.set_logical_ops_z(logical_z)
    code_half = codes.QuditCode(matrix)
    code_half.set_logical_ops_z(logical_z[:, num_qudits:])
    assert np.array_equal(code_full.get_logical_ops(), code_half.get_logical_ops())

    # providing the wrong number of logical operators raises a helpful error, including for the
    # CSSCode overrides and for 1-D inputs (which would otherwise raise a cryptic IndexError)
    with pytest.raises(ValueError, match="Expected 1 logical operators"):
        codes.QuditCode(matrix).set_logical_ops_x(logical_x[:0])
    with pytest.raises(ValueError, match="Expected 1 logical operators"):
        codes.QuditCode(matrix).set_logical_ops_z(np.vstack([logical_z, logical_z]))
    with pytest.raises(ValueError, match="Expected 1 logical operators"):
        codes.QuditCode(matrix).set_logical_ops_x(logical_x[0])  # 1-D input
    with pytest.raises(ValueError, match="Expected 1 logical operators"):
        codes.SteaneCode().set_logical_ops_x(css_code.get_logical_ops(Pauli.X)[:0])
    with pytest.raises(ValueError, match="Expected 1 logical operators"):
        codes.SteaneCode().set_logical_ops_z(css_code.get_logical_ops(Pauli.Z)[0])  # 1-D input


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

    # concatenation does not mutate the logical operators of the outer code passed by the caller
    outer = codes.QuditCode.stack([code_5q] * len(code_5q))  # dimension == inner physical qudits
    logical_ops_before = outer.get_logical_ops().copy()
    codes.QuditCode.concatenate(outer, code_5q, [1, 0, 2, 3, 4])
    assert np.array_equal(outer.get_logical_ops(), logical_ops_before)

    # cover some errors
    with pytest.raises(ValueError, match="different fields"):
        codes.QuditCode.concatenate(code_5q, codes.ToricCode(2, field=3))
    with pytest.raises(ValueError, match="divisible"):
        codes.QuditCode.concatenate(code_5q, code_5q, [0, 1, 2])


def test_quantum_capacity(pytestconfig: pytest.Config) -> None:
    """Logical error rates in a code capacity model."""
    code = codes.FiveQubitCode()

    logical_error_rate_func = code.get_logical_error_rate_func(num_samples=1)
    assert logical_error_rate_func(0) == (0, 0)  # no logical error with zero uncertainty

    # guaranteed logical X and Z errors
    for pauli_bias in [(1, 0, 0), (0, 0, 1)]:
        logical_error_rate_func = code.get_logical_error_rate_func(10, 1, pauli_bias)
        assert logical_error_rate_func(1)[0] == 1

    # with an erasure-enabled decoder, unrecognised syndromes are discarded
    logical_error_rate_func = code.get_logical_error_rate_func(
        num_samples=1, max_error_rate=1, with_lookup=True, max_weight=0, add_erasure_bit=True
    )
    assert logical_error_rate_func(0, discard_rate=True) == (0, 0)  # no errors at p=0
    assert logical_error_rate_func(0.5, discard_rate=True)[0] > 0  # all syndromes → erasure

    # a subsystem code is decoded against its stabilizer generators rather than its more numerous,
    # non-commuting gauge generators, so the QuditCode and CSSCode views of the same subsystem code
    # decode identically and yield the same logical error rate
    seed = pytestconfig.getoption("randomly_seed")
    css_code = codes.BaconShorCode(3)
    qudit_code = codes.QuditCode(css_code.matrix)
    assert qudit_code.is_subsystem_code
    np.random.seed(seed)
    css_rate = css_code.get_logical_error_rate_func(num_samples=200, max_error_rate=0.3)(0.1)
    np.random.seed(seed)
    qudit_rate = qudit_code.get_logical_error_rate_func(num_samples=200, max_error_rate=0.3)(0.1)
    assert np.allclose(qudit_rate, css_rate)

    # a code over a non-binary field is decoded with a field-aware decoder: its syndrome matrix is
    # a field array, from which the decoder is selected to match the field
    qudit_code = codes.QuditCode(codes.BaconShorCode(3, field=3).matrix)
    logical_error_rate_func = qudit_code.get_logical_error_rate_func(
        num_samples=100, max_error_rate=0.2
    )
    assert logical_error_rate_func(0) == (0, 0)  # no logical error with zero uncertainty
    assert logical_error_rate_func(0.1)[0] > 0  # nonzero logical error rate at a nonzero rate


def test_qudit_to_css() -> None:
    """Convert a QuditCode to a CSSCode."""
    code = codes.SteaneCode()
    assert code.is_equiv_to(codes.QuditCode(code.matrix).to_css())

    with pytest.raises(TypeError, match="both X and Z support"):
        codes.FiveQubitCode().to_css()


def test_qudit_to_swel() -> None:
    """Convert a QuditCode to a CSSCode with SWEL logical operators."""
    steane_code = codes.SteaneCode()
    code = codes.QuditCode(steane_code.matrix).to_swel()
    assert isinstance(code, codes.CSSCode)
    assert np.array_equal(code.get_logical_ops(Pauli.X), code.get_logical_ops(Pauli.Z))
    assert code.is_equiv_to(steane_code)

    # maybe_to_swel returns an equivalent CSSCode with SWEL logical operators
    maybe_code = codes.QuditCode(steane_code.matrix).maybe_to_swel()
    assert isinstance(maybe_code, codes.CSSCode)
    assert np.array_equal(maybe_code.get_logical_ops(Pauli.X), maybe_code.get_logical_ops(Pauli.Z))

    # a non-CSS code cannot be converted to SWEL
    assert codes.FiveQubitCode().maybe_to_swel() == codes.FiveQubitCode()
    with pytest.raises(TypeError, match="both X and Z support"):
        codes.FiveQubitCode().to_swel()

    # a CSS code that is not SWEL cannot be converted to SWEL
    surface_code = codes.SurfaceCode(3)
    assert codes.QuditCode(surface_code.matrix).maybe_to_swel() == codes.QuditCode(
        surface_code.matrix
    )
    with pytest.raises(ValueError, match="no self-dual logical operator basis"):
        codes.QuditCode(surface_code.matrix).to_swel()


####################################################################################################
# CSS code tests


def test_css_code(pytestconfig: pytest.Config) -> None:
    """Miscellaneous CSS code tests and coverage."""
    seed = pytestconfig.getoption("randomly_seed")
    code_x = codes.ClassicalCode.random(3, 2, seed=seed)

    code_z = ~code_x
    code = codes.CSSCode(code_x, code_z)
    assert code.get_weight() == max(code_x.get_weight(), code_z.get_weight())
    assert code.num_checks_x == code_x.num_checks
    assert code.num_checks_z == code_z.num_checks
    assert code.num_checks == code.num_checks_x + code.num_checks_z
    assert code == codes.CSSCode(code.code_x, code.code_z)

    # equivalence to QuditCode with the same parity check matrix
    assert code.is_equiv_to(codes.QuditCode(code.matrix))

    # equivalence to code with redundant stabilizers
    redundant_code = codes.CSSCode(np.vstack([code.matrix_x, code.matrix_x]), code.matrix_z)
    assert codes.CSSCode.equiv(code, redundant_code)

    code_z = codes.ClassicalCode.random(4, 2)
    with pytest.raises(ValueError, match="incompatible"):
        codes.CSSCode(code_x, code_z)

    with pytest.raises(ValueError, match="incompatible"):
        code_z = codes.ClassicalCode.random(3, 2, field=code_x.field.order**2)
        codes.CSSCode(code_x, code_z)

    # build a classical code of X-type stabilizers
    code = codes.CSSCode.classical(code_x, Pauli.X)
    assert np.array_equal(code.matrix_x, code_x.matrix)
    assert code.matrix_z.shape == (0, len(code_x))

    # subgraphs for syndrome extraction
    assert_valid_subgraphs(code)
    subgraphs = code.get_syndrome_subgraphs()
    assert nx.utils.graphs_equal(subgraphs[0], code.get_graph(Pauli.X))
    assert nx.utils.graphs_equal(subgraphs[1], code.get_graph(Pauli.Z))


def test_css_from_strings() -> None:
    """Construct a CSSCode from parity check strings."""
    code = codes.CSSCode.from_strings(["XXXX", "ZZZZ"])
    assert isinstance(code, codes.CSSCode)
    assert code.is_equiv_to(codes.C4Code())


def test_css_from_qecdb_id() -> None:
    """Retrieve a CSS code from qecdb.org."""
    strings = ["XXXX", "ZZZZ"]
    distance = 2
    is_css = True
    code_data = (strings, distance, is_css)
    with unittest.mock.patch("qldpc.external.codes.get_quantum_code", return_value=code_data):
        code = codes.CSSCode.from_qecdb_id("")
        assert isinstance(code, codes.CSSCode)
        assert code.is_equiv_to(codes.C4Code())


def test_swel_codes() -> None:
    """Identify and construct SWEL logical operator bases (see CSSCode.is_swel)."""
    steane_code = codes.SteaneCode()
    surface_code = codes.SurfaceCode(3)
    even_code = codes.CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]])
    assert steane_code.is_swel
    assert not surface_code.is_swel  # not self-dual
    assert not even_code.is_swel  # self-dual but not SWEL

    # a self-dual code over GF(3) reaches the general orthonormal-basis test (odd characteristic)
    qutrit_matrix = [[0, 0, 1, 1, 1], [0, 1, 0, 1, 2]]
    assert codes.CSSCode(qutrit_matrix, qutrit_matrix, field=3).is_swel

    # we automatically find a self-dual logical operator basis, if it exists
    code = steane_code.set_swel_logical_ops()
    assert np.array_equal(code.get_logical_ops(Pauli.X), code.get_logical_ops(Pauli.Z))
    with pytest.raises(ValueError, match="no self-dual logical operator basis"):
        surface_code.get_swel_logical_ops()
    with pytest.raises(ValueError, match="no self-dual logical operator basis"):
        even_code.get_swel_logical_ops()


def test_css_ops(pytestconfig: pytest.Config) -> None:
    """Logical and stabilizer operator construction for CSS codes."""
    seed = pytestconfig.getoption("randomly_seed")
    code: codes.CSSCode

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, field=3, seed=seed))

    # set X-type logicals and determine Z-type logicals automatically
    other_code = codes.CSSCode(code.matrix_x, code.matrix_z)
    other_code.set_logical_ops_x(code.get_logical_ops(Pauli.X))
    assert np.array_equal(code.get_logical_ops(Pauli.X), other_code.get_logical_ops(Pauli.X))
    assert np.array_equal(
        code.get_logical_ops(Pauli.X) @ other_code.get_logical_ops(Pauli.Z).T,
        np.eye(code.dimension),
    )

    # shuffle logical operators around
    code.set_logical_ops_z(code.get_logical_ops(Pauli.Z)[::-1])

    # identify stabilizer group
    code._stabilizer_ops = None
    assert not np.any(
        code.get_stabilizer_ops() @ math.symplectic_conjugate(code.get_logical_ops()).T
    )
    assert not np.any(code.get_stabilizer_ops() @ math.symplectic_conjugate(code.get_gauge_ops()).T)

    # destabilizers of one Pauli type are dual to the stabilizers of the opposite type
    for code in get_codes_for_testing_ops():
        for pauli in PAULIS_XZ:
            destabs = code.get_destabilizer_ops(pauli)
            stabs = code.get_stabilizer_ops(pauli.swap_xz())
            assert np.linalg.matrix_rank(destabs @ stabs.T) == len(destabs)

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

    # a bare CSSCode has no specialized exact-distance method, so it falls back to brute force
    bare_code = codes.QuditCode(codes.SteaneCode().matrix).to_css()
    assert bare_code.get_distance_exact() == 3

    # qubit code distance
    code = codes.QuditCode(codes.SHPCode(codes.RepetitionCode(2)).matrix).to_css()
    assert code.get_distance_exact(cutoff=len(code)) <= len(code)
    assert code.get_distance_exact() == 2
    assert code.get_distance_bound_with_decoder(Pauli.X, cutoff=len(code)) <= len(code)

    # computing an exact distance but providing bounding arguments raises a warning
    with pytest.warns(UserWarning, match="ignored"):
        assert code.get_distance(bound=False, test_arg=True)

    # qutrit code distance
    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    code.forget_distance()
    assert code.get_distance(bound=False) == 2

    code = codes.QuditCode(code.matrix).to_css()
    assert code.get_distance_bound(cutoff=len(code)) <= len(code)
    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
        assert code.get_distance(bound=True) <= len(code)
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.codes.get_distance_bound", return_value=-1),
    ):
        assert code.get_distance(bound=True) == -1
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        assert code.get_distance_exact(cutoff=len(code)) <= len(code)
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        assert code.get_distance_exact() == 2

    # the distance of a dimension-0 quantum code is undefined
    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    code = codes.HGPCode(trivial_code)
    assert code.dimension == 0
    assert np.isnan(code.get_distance(bound=True))
    assert np.isnan(code.get_distance(bound=False))


def test_css_deformations() -> None:
    """Local Fourier transforms of a CSSCode."""
    code: codes.CSSCode

    code = codes.SteaneCode()
    assert codes.CSSCode.equiv(code.conjugated(range(len(code))), code)
    assert not codes.CSSCode.equiv(code.deformed("H 0"), code)

    code = codes.SHYPSCode(2)
    code.get_logical_ops()
    code.get_stabilizer_ops()
    code.get_gauge_ops()
    assert code.conjugated() == code.deformed("H " + " ".join(map(str, range(len(code)))))

    # conjugating all qudits swaps the X and Z distances, however they are specified
    code._distance_x = 3
    code._distance_z = 5
    all_qudits: slice | Sequence[int] | None
    for all_qudits in [None, range(len(code)), slice(None), list(range(len(code)))]:
        conjugate = code.conjugated(all_qudits)
        assert isinstance(conjugate, codes.CSSCode)
        assert conjugate.get_distance_if_known(Pauli.X) == 5
        assert conjugate.get_distance_if_known(Pauli.Z) == 3

    # conjugating a strict subset of qudits does not swap the X and Z distances
    subset_conjugate = code.conjugated([])
    assert isinstance(subset_conjugate, codes.CSSCode)
    assert subset_conjugate.get_distance_if_known(Pauli.X) is None


def test_stacking_css_codes() -> None:
    """Stack two CSS codes."""
    steane_code = codes.SteaneCode()
    code = codes.CSSCode.stack([steane_code] * 2)
    assert len(code) == len(steane_code) * 2
    assert code.dimension == steane_code.dimension * 2

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        qudit_code = codes.SurfaceCode(2, field=3)
        code = codes.CSSCode.stack([steane_code, qudit_code])

    # stacking a CSSCode with a QuditCode requires using QuditCode.stack
    codes.QuditCode.stack([steane_code, codes.FiveQubitCode()])
    with pytest.raises(TypeError, match="requires CSSCode inputs"):
        codes.CSSCode.stack([steane_code, codes.FiveQubitCode()])


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

    logical_error_rate_func = code.get_logical_error_rate_func(num_samples=1)
    assert logical_error_rate_func(0) == (0, 0)  # no logical error with zero uncertainty

    # guaranteed logical X and Z errors
    for pauli_bias in [(1, 0, 0), (0, 0, 1)]:
        logical_error_rate_func = code.get_logical_error_rate_func(10, 1, pauli_bias)
        assert logical_error_rate_func(1)[0] == 1

    # pauli_bias convention is (X, Y, Z); (0, 0, 1) = pure Z
    # if the max_weight for lookup is 0, any Z syndrome triggers erasure
    logical_error_rate_func_z = code.get_logical_error_rate_func(
        num_samples=1,
        max_error_rate=1,
        pauli_bias=(0, 0, 1),
        with_lookup=True,
        max_weight=0,
        add_erasure_bit=True,
    )
    assert logical_error_rate_func_z(0, discard_rate=True) == (0, 0)  # no errors at p=0
    assert logical_error_rate_func_z(0.5, discard_rate=True)[0] > 0  # Z syndromes → erasure

    # (1 ,0, 0) = pure X: Z syndromes are always zero so samples reach the X decoder
    logical_error_rate_func_x = code.get_logical_error_rate_func(
        num_samples=1,
        max_error_rate=1,
        pauli_bias=(1, 0, 0),
        with_lookup=True,
        max_weight=0,
        add_erasure_bit=True,
    )
    assert logical_error_rate_func_x(0, discard_rate=True) == (0, 0)  # no errors at p=0
    assert logical_error_rate_func_x(0.5, discard_rate=True)[0] > 0  # X syndromes → erasure

    # a subsystem code is decoded against its stabilizer generators, whose number differs from the
    # number of parity checks (gauge generators), so a syndrome has one entry per stabilizer
    subsystem_code = codes.BaconShorCode(3)
    assert subsystem_code.is_subsystem_code
    logical_error_rate_func = subsystem_code.get_logical_error_rate_func(
        num_samples=200, max_error_rate=0.3
    )
    assert logical_error_rate_func(0) == (0, 0)  # no logical error with zero uncertainty
    assert logical_error_rate_func(0.1)[0] > 0  # nonzero logical error rate at a nonzero rate
