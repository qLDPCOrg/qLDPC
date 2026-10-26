# SPDX-License-Identifier: Apache-2.0

"""Tests for exact certification of bounded fault propagation in noisy state preparation."""

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits

from . import fault_propagation


def _z_state(num_qubits: int) -> npt.NDArray[np.int_]:
    """Stabilizers of |0...0>, for which an error's effective weight is its syndrome's weight."""
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _smallest_violation(
    flags: Sequence[int], syndromes: Sequence[int], max_faults: int | None
) -> int | None:
    """The fewest faults in a set that fires no flag and violates the bound on |0...0>."""
    for size in range(1, len(flags) + 1 if max_faults is None else min(max_faults, len(flags)) + 1):
        for support in itertools.combinations(range(len(flags)), size):
            flag = functools.reduce(operator.xor, (flags[jj] for jj in support))
            syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in support))
            if flag == 0 and syndrome.bit_count() > size:
                return size
    return None


def _certify_map_with_statistics(
    flags: Sequence[int], syndromes: Sequence[int], max_faults: int | None, num_qubits: int
) -> tuple[
    circuits.BoundedFaultPropagationCounterexample | None,
    circuits.BoundedFaultPropagationStatistics,
]:
    """Certify a map from faults to flags and syndromes on |0...0> and compare with brute force."""
    failure, statistics = fault_propagation._certify_maps(
        flags, syndromes, _z_state(num_qubits), False, max_faults
    )
    smallest = _smallest_violation(flags, syndromes, max_faults)
    assert (failure is None) == (smallest is None)
    if failure is not None:
        indices = failure.fault_indices
        flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
        syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
        assert failure.fault_count == smallest
        assert indices == tuple(sorted(set(indices)))
        assert flag == 0
        assert failure.output_syndrome == tuple((syndrome >> jj) & 1 for jj in range(num_qubits))
        assert failure.fault_count < failure.effective_weight_lower_bound <= syndrome.bit_count()
        # X on a qubit anticommutes with that qubit's stabilizer Z.
        xs, _ = stim.PauliString(failure.output_pauli).to_numpy()
        assert tuple(map(int, xs)) == failure.output_syndrome
    return failure, statistics


def _certify_map(
    flags: Sequence[int], syndromes: Sequence[int], max_faults: int | None, num_qubits: int
) -> circuits.BoundedFaultPropagationCounterexample | None:
    """Certify a map from faults to flags and syndromes on |0...0> and compare with brute force."""
    return _certify_map_with_statistics(flags, syndromes, max_faults, num_qubits)[0]


def _true_minimum(stabilizers: Sequence[str], syndrome: Sequence[int], metric_css: bool) -> int:
    """Minimum weight over all Paulis with this syndrome, by enumerating every Pauli."""
    n = len(stabilizers)
    rows = np.array([np.concatenate(stim.PauliString(ss).to_numpy()) for ss in stabilizers])
    paulis = (np.arange(4**n)[:, None] >> np.arange(2 * n)) & 1
    x, z = paulis[:, :n], paulis[:, n:]
    syndromes = (x @ rows[:, n:].T + z @ rows[:, :n].T) % 2
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if metric_css else (x | z).sum(axis=1)
    return int(weights[np.all(syndromes == np.array(syndrome), axis=1)].min())


def _fault_detectors(certificate: circuits.BoundedFaultPropagationCertificate) -> list[set[int]]:
    """Detectors flipped by every error mechanism in the certificate's DEM."""
    errors = [inst for inst in certificate.detector_error_model if inst.type == "error"]
    return [
        {tt.val for tt in error.targets_copy() if tt.is_relative_detector_id()} for error in errors
    ]


def _fault_syndromes(
    certificate: circuits.BoundedFaultPropagationCertificate, circuit: stim.Circuit
) -> list[tuple[int, ...]]:
    """Output syndrome of every error mechanism in the certificate's DEM.

    The detectors of the circuit come first, and one detector per target stabilizer follows.
    """
    return [
        tuple(
            int(circuit.num_detectors + jj in events)
            for jj in range(len(certificate.target_stabilizers))
        )
        for events in _fault_detectors(certificate)
    ]


def _assert_valid_counterexample(
    certificate: circuits.BoundedFaultPropagationCertificate,
    circuit: stim.Circuit,
) -> None:
    """Check a witness against the returned DEM and a brute-force minimum, not the search."""
    failure = certificate.counterexample
    assert failure is not None and not certificate.certified
    fault_detectors = _fault_detectors(certificate)
    events: set[int] = set()
    for index in failure.fault_indices:
        events ^= fault_detectors[index]
    assert not events & set(certificate.post_selection_detectors)
    syndrome = tuple(
        int(circuit.num_detectors + jj in events)
        for jj in range(len(certificate.target_stabilizers))
    )
    assert syndrome == failure.output_syndrome
    stabilizers = [stim.PauliString(ss) for ss in certificate.target_stabilizers]
    pauli = stim.PauliString(failure.output_pauli)
    assert tuple(int(not pauli.commutes(ss)) for ss in stabilizers) == failure.output_syndrome
    assert failure.effective_weight_lower_bound > failure.fault_count
    minimum = _true_minimum(certificate.target_stabilizers, syndrome, certificate.css_weight)
    assert failure.effective_weight_lower_bound <= minimum


def test_certify_maps_without_flags_checks_each_fault(pytestconfig: pytest.Config) -> None:
    """Without flags, the first fault of weight above one is reported and no set can do better."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    def certify(
        syndromes: Sequence[int], num_qubits: int
    ) -> circuits.BoundedFaultPropagationCounterexample | None:
        failure = _certify_map([0] * len(syndromes), syndromes, None, num_qubits)
        first = next(
            (jj for jj, syndrome in enumerate(syndromes) if syndrome.bit_count() > 1), None
        )
        assert (failure is None) == (first is None)
        assert failure is None or failure.fault_indices == (first,)
        return failure

    # Faults of weight at most one certify, and the first fault of weight three does not.
    assert certify([0, 1, 2, 4, 2], 3) is None
    assert certify([1, 2, 7, 4, 7], 3) is not None
    # Sparse random syndromes mix certified maps with failing ones.
    for num_qubits in (3, 5):
        for _ in range(20):
            sparse = rng.integers(1 << num_qubits, size=(6, 3))
            certify([int(a & b & c) for a, b, c in sparse], num_qubits)


def test_certify_maps_with_flags_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """Random maps with repeated columns and flag rank above the fault limit match brute force."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for n, rank in itertools.product((3, 5), range(5)):
        for _ in range(12):
            flags = list(map(int, rng.integers(1 << rank, size=8)))
            syndromes = list(map(int, rng.integers(1 << n, size=8)))
            for max_faults in (1, 2, 3, 5, None):
                _certify_map(flags, syndromes, max_faults, n)


def test_hand_derived_dependency_cases() -> None:
    """Triples, decomposable pairs, duplicate columns, zero columns, and fault limits."""
    for flags, syndromes, max_faults, certified in (
        ([1, 2, 3], [1, 2, 12], 2, True),  # the triple of weight 4 needs three faults
        ([1, 2, 3], [1, 2, 12], 3, False),
        ([1, 1], [1, 14], 3, False),  # an accepted pair of weight 4
        ([0, 0, 0], [1, 2, 4], 4, True),  # flag-free faults of weight one
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),  # pairs of weight two
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),  # identical columns cancel in pairs
        ([0, 0], [0, 0], 4, True),  # faults with no effect
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 3, True),
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 4, False),  # four faults of weight six
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], None, False),
    ):
        failure = _certify_map(flags, syndromes, max_faults, 6)
        assert (failure is None) == certified


def test_failures_report_the_fewest_and_earliest_faults() -> None:
    """A pair is reported although a triple violates too, and repeated columns name the first."""
    # Columns 0-2 form an accepted triple of weight four, and columns 3 and 4 an accepted pair of
    # weight three.
    failure = _certify_map([1, 2, 3, 4, 4], [1, 2, 12, 1, 6], None, 6)
    assert failure is not None and failure.fault_indices == (3, 4)
    # Faults 0 and 2 have the same column, so the pair of faults 0 and 1 is reported.
    failure = _certify_map([1, 1, 1], [3, 4, 3], None, 3)
    assert failure is not None and failure.fault_indices == (0, 1)


def test_faults_of_weight_at_most_one_certify_every_limit() -> None:
    """Without a fault that leaves weight above one on its own, no set can violate the bound."""
    flags = [3, 5, 6, 9, 10, 12, 1, 2, 4, 8, 7, 11]
    syndromes = [1, 2, 4, 8, 0, 1, 2, 4, 8, 0, 1, 2]
    failure, statistics = _certify_map_with_statistics(flags, syndromes, None, 4)
    assert failure is None and statistics.num_sets_checked == 0

    # One fault of weight three brings back the search, and it fails with the equal-flag fault.
    failure, statistics = _certify_map_with_statistics(flags + [3], syndromes + [14], None, 4)
    assert failure is not None and failure.fault_count == 2
    assert statistics.num_sets_checked > 0


def test_independent_columns_need_only_equal_column_pairs() -> None:
    """If the distinct nonzero flag columns are independent, only equal-flag pairs matter."""
    outcomes = set()
    # Flags 3, 5, 8, 16, and 2**130 are independent; three copies exceed twice the flag rank.
    for copies in (2, 3):
        flags = [flag for flag in (3, 5, 8, 16, 1 << 130) for _ in range(copies)]
        passing = [3, 5, 6] * 5  # each pair has weight two
        failing = [3, 12, 0] * 5
        for syndromes in (passing[: len(flags)], failing[: len(flags)]):
            for max_faults in (2, 3, None):
                failure, statistics = _certify_map_with_statistics(flags, syndromes, max_faults, 4)
                assert statistics.effective_budget == 2
                outcomes.add(failure is None)
    assert outcomes == {True, False}


def test_statistics_reflect_merged_columns_and_the_flag_rank_limit() -> None:
    """Zero and repeated columns are never searched, and sets have at most rank + 1 faults."""
    failure, statistics = _certify_map_with_statistics([0, 1, 1], [0, 1, 1], None, 4)
    assert failure is None and statistics.num_sets_checked == 0
    # Each distinct fault without flags is an irreducible accepted set, and an equal fault is not
    # checked again.
    failure, statistics = _certify_map_with_statistics([0, 0, 0, 0], [1, 1, 4, 0], None, 4)
    assert failure is None and statistics.num_sets_checked == 2

    # Flag columns 1, 2, and 3 have rank two, and only all three faults are accepted together.
    for max_faults, effective_budget in ((None, 3), (30, 3), (2, 2)):
        failure, statistics = _certify_map_with_statistics([1, 2, 3], [1, 2, 12], max_faults, 4)
        assert (failure is None) == (max_faults == 2)
        assert statistics.flag_rank == 2 and statistics.effective_budget == effective_budget
        assert statistics.num_sets_checked == (0 if max_faults == 2 else 1)


def test_irreducible_patterns_match_exhaustive_enumeration(pytestconfig: pytest.Config) -> None:
    """The prefix search yields every irreducible accepted pattern of a given size, once."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for width in range(1, 10):
        for flag_bits in range(1, 5):
            flags = list(map(int, rng.integers(1, 1 << flag_bits, size=width)))
            syndromes = list(map(int, rng.integers(16, size=width)))
            originals = list(map(int, rng.permutation(np.arange(width) * 7 + 100)))
            faults = list(zip(flags, syndromes, originals, strict=True))
            expected: dict[int, dict[tuple[int, ...], int]] = {}
            accepted: list[int] = []
            for size in range(1, 6):
                expected[size] = {}
                for support in itertools.combinations(range(width), size):
                    flag = functools.reduce(operator.xor, (flags[jj] for jj in support))
                    syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in support))
                    word = sum(1 << jj for jj in support)
                    if flag == 0:
                        if not any(previous & word == previous for previous in accepted):
                            expected[size][tuple(originals[jj] for jj in support)] = syndrome
                        accepted.append(word)
            for size in range(2, 6):
                found = list(fault_propagation._iter_irreducible_patterns(faults, size))
                assert len(found) == len(expected[size])
                assert dict(found) == expected[size]


def test_irreducible_patterns_with_flags_wider_than_a_machine_word() -> None:
    """Flag columns of 80 and 130 bits combine correctly."""
    faults = [(1 << 80, 1, 90), (1 << 130, 2, 10), ((1 << 80) ^ (1 << 130), 4, 50)]
    found = fault_propagation._iter_irreducible_patterns(faults, 3)
    assert list(found) == [((90, 10, 50), 7)]


@pytest.mark.parametrize("css_weight", [True, False])
@pytest.mark.parametrize("num_qubits", [3, 4])
def test_ghz_preparation_matches_brute_force_over_faults(num_qubits: int, css_weight: bool) -> None:
    """Fanning out a GHZ state certifies for three qubits but not for four.

    X on a set of GHZ qubits is equivalent to X on the complement, so only a two-qubit X error on
    four qubits has no lighter equivalent.
    """
    circuit = stim.Circuit("H 0")
    for target in range(1, num_qubits):
        circuit.append("CX", [0, target])
        circuit.append("DEPOLARIZE2", [0, target], 0.01)
    result = circuits.certify_bounded_fault_propagation(circuit, css_weight=css_weight)
    syndromes = _fault_syndromes(result, circuit)
    minima = [_true_minimum(result.target_stabilizers, ss, css_weight) for ss in syndromes]
    first = next((jj for jj, minimum in enumerate(minima) if minimum > 1), None)
    assert result.certified == (first is None) == (num_qubits == 3)
    if first is not None:
        _assert_valid_counterexample(result, circuit)
        assert result.counterexample is not None and result.counterexample.fault_indices == (first,)


@pytest.mark.parametrize("css_weight", [None, True, False])
def test_state_selection_and_overrides(css_weight: bool | None) -> None:
    """Automatic selection picks CSS weight for |0+> only, and an explicit value overrides it."""
    css_state = stim.Circuit("I 0\nH 1\nCORRELATED_ERROR(0.01) X0 Z1")  # X0 Z1 has CSS weight one
    result = circuits.certify_bounded_fault_propagation(
        css_state, output_qubits=[0, 1], css_weight=css_weight
    )
    assert result.css_weight is (css_weight is not False)
    assert result.certified is (css_weight is not False)
    if not result.certified:
        _assert_valid_counterexample(result, css_state)
    graph_state = stim.Circuit("H 0 1\nCZ 0 1\nCORRELATED_ERROR(0.001) X0 X1")
    result = circuits.certify_bounded_fault_propagation(
        graph_state, output_qubits=[0, 1], css_weight=css_weight
    )
    assert result.certified and result.css_weight is (css_weight is True)


def test_css_detection_depends_on_the_stabilizer_row_space() -> None:
    """Generators that mix X and Z can still span a CSS state, and a logical Y state is not CSS."""

    def matrix(generators: Sequence[str]) -> npt.NDArray[np.uint8]:
        return np.array(
            [np.concatenate(stim.PauliString(gg).to_numpy()) for gg in generators], dtype=np.uint8
        )

    def state(circuit: str) -> npt.NDArray[np.uint8]:
        tableau = stim.Tableau.from_circuit(stim.Circuit(circuit))
        return matrix([str(gg) for gg in tableau.to_stabilizers()])

    # These products of the stabilizers X0, Z1, and Z2 of |+00> each mix X and Z.
    assert fault_propagation._is_css_state(matrix(["XZ_", "X_Z", "XZZ"]))
    assert fault_propagation._is_css_state(state("H 0\nI 1 2"))
    assert not fault_propagation._is_css_state(state("H 0 1\nCZ 0 1"))
    # A CSS repetition code can prepare logical Y, whose stabilizers are not pure X or Z.
    assert not fault_propagation._is_css_state(state("H 0\nS 0\nCX 0 1 0 2"))
    # The stabilizers X0, Z1, and Y2 include a pure-X and a pure-Z operator, but do not span.
    assert not fault_propagation._is_css_state(state("H 0 2\nS 2\nI 1"))


def test_stabilizers_keep_signs_and_the_output_frame() -> None:
    """Reported stabilizers are +1 eigenoperators of the ideal state on the chosen output qubits."""
    circuit = stim.Circuit("H 4\nS 4\nCX 4 1\nX 1\nCORRELATED_ERROR(0.01) Y4")
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[4, 1])
    assert result.certified and result.output_qubits == (4, 1)
    assert any(ss.startswith("-") for ss in result.target_stabilizers)
    simulator = stim.TableauSimulator()
    simulator.do(circuit.without_noise())
    for text in result.target_stabilizers:
        physical = stim.PauliString(circuit.num_qubits)
        for qubit, pauli in zip(result.output_qubits, text[1:], strict=True):
            physical[qubit] = pauli
        assert simulator.peek_observable_expectation(-physical if text[0] == "-" else physical) == 1


def test_default_outputs_skip_measured_and_untouched_qubits() -> None:
    """Outputs default to the qubits that are acted on and never measured, in increasing order."""
    # Qubits 0, 2, and 3 are untouched, and qubit 5 is measured.
    circuit = stim.Circuit("H 4\nS 4\nCX 4 1\nX 1\nCORRELATED_ERROR(0.01) Y4\nCX 5 1\nM 5")
    default = circuits.certify_bounded_fault_propagation(circuit)
    assert default.output_qubits == (1, 4)
    assert default == circuits.certify_bounded_fault_propagation(circuit, output_qubits=[1, 4])


def test_unmeasured_qubits_account_for_every_kind_of_measurement() -> None:
    """Resets, inverted and Pauli-product measurements, repeats, padding, and heralded noise."""
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 9
        H 0 1 2 3 4 5 6 7
        MR 1
        M !2
        MPP X3*Z4
        MXX 5 6
        REPEAT 2 {
            MX 7
        }
        DETECTOR rec[-1]
        CORRELATED_ERROR(0.1) X8
        MPAD 0
        HERALDED_ERASE(0.1) 8
    """)
    assert fault_propagation._get_unmeasured_qubits(circuit) == (0, 8)


def test_correlated_mechanism_costs_one_fault() -> None:
    """A two-qubit error is one fault of effective weight two, and the requested order is kept."""
    circuit = stim.Circuit("I 0 1\nCORRELATED_ERROR(0.01) X0 Y1")
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[1, 0])
    _assert_valid_counterexample(result, circuit)
    failure = result.counterexample
    assert failure is not None
    assert failure.fault_indices == (0,) and failure.effective_weight_lower_bound == 2
    assert result.output_qubits == (1, 0)


def test_exclusive_errors_count_as_independent_faults() -> None:
    """Errors that exclude each other in the circuit are still separate faults of the DEM."""
    circuit = stim.Circuit("I 0 1\nCORRELATED_ERROR(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X0 X1")
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0, 1])
    _assert_valid_counterexample(result, circuit)


def test_existing_detectors_do_not_enter_the_syndrome() -> None:
    """A fault that trips one of the circuit's own detectors is judged by its output error alone."""
    noise = "CORRELATED_ERROR(0.01) X0 X1 X2"
    detected = stim.Circuit(f"I 0 1\n{noise}\nM 2\nDETECTOR rec[-1]")
    result = circuits.certify_bounded_fault_propagation(detected, output_qubits=[0, 1])
    _assert_valid_counterexample(result, detected)
    undetected = circuits.certify_bounded_fault_propagation(
        stim.Circuit(f"I 0 1\n{noise}"), output_qubits=[0, 1]
    )
    assert result.counterexample == undetected.counterexample


def test_existing_observables_do_not_enter_the_syndrome() -> None:
    """A fault that flips an observable is judged by its output error alone."""
    noise = "CORRELATED_ERROR(0.01) X1 X2\nM 2"  # a weight-one output error that flips rec[-1]
    flipped = stim.Circuit(f"{noise}\nOBSERVABLE_INCLUDE(0) rec[-1]")
    result = circuits.certify_bounded_fault_propagation(flipped, output_qubits=[0, 1])
    assert result.detector_error_model.num_observables == 1
    plain = circuits.certify_bounded_fault_propagation(stim.Circuit(noise), output_qubits=[0, 1])
    assert result.certified and plain.certified


def test_faults_inside_repeat_blocks_are_certified() -> None:
    """Faults in a loop count although Stim folds the loop into a repeat block of the DEM."""
    looped = stim.Circuit("REPEAT 3 {\n    CORRELATED_ERROR(0.01) X0 X1\n}")
    # A loop without detectors keeps its errors in a repeat block, which would hide them.
    readout = stim.Circuit("M 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    folded = (looped + readout).detector_error_model(approximate_disjoint_errors=True)
    assert any(instruction.type == "repeat" for instruction in folded)

    result = circuits.certify_bounded_fault_propagation(looped)
    written_out = circuits.certify_bounded_fault_propagation(looped.flattened())
    assert not result.certified and not written_out.certified
    _assert_valid_counterexample(result, looped)
    assert result.counterexample is not None and written_out.counterexample is not None
    assert result.counterexample.output_syndrome == written_out.counterexample.output_syndrome
    assert (
        result.counterexample.effective_weight_lower_bound
        == written_out.counterexample.effective_weight_lower_bound
    )


def test_circuit_without_faults_certifies() -> None:
    """Deterministic nonzero measurements, repeats, and observables need no check."""
    circuit = stim.Circuit("""
        X 0
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0])
    assert result.certified and result.target_stabilizers == ("-Z",)


def test_post_selection_can_accept_a_damaging_pair() -> None:
    """Two faults that each trip the detector cancel it together and damage four outputs."""
    circuit = stim.Circuit("""
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
        DETECTOR rec[-1]
    """)

    def certify(
        max_faults: int | None, post_select: bool | Sequence[int] = (0,)
    ) -> circuits.BoundedFaultPropagationCertificate:
        return circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=[0, 1, 2, 3], post_select=post_select, max_faults=max_faults
        )

    # A single fault is discarded, so the pair is the smallest violation.
    assert certify(1).certified
    result = certify(2)
    _assert_valid_counterexample(result, circuit)
    assert result.counterexample is not None and result.counterexample.fault_indices == (0, 1)
    assert certify(None).counterexample == result.counterexample

    # True selects every detector of the circuit, here detector 0, and False selects none.
    assert certify(2, True) == result
    unselected = certify(2, False)
    assert unselected.counterexample is not None and unselected.counterexample.fault_count == 1
    default = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0, 1, 2, 3])
    assert default == certify(None, False)


def test_max_faults_does_not_change_results_without_post_selection() -> None:
    """The check of single faults is exact for sets of every size."""
    for num_qubits in (3, 4):
        circuit = stim.Circuit("H 0")
        for target in range(1, num_qubits):
            circuit.append("CX", [0, target])
            circuit.append("DEPOLARIZE2", [0, target], 0.01)
        results = [
            circuits.certify_bounded_fault_propagation(circuit, max_faults=max_faults)
            for max_faults in (1, 2, 5, None)
        ]
        assert [result.max_faults for result in results] == [1, 2, 5, None]
        assert all(result.certified == (num_qubits == 3) for result in results)
        assert all(result.counterexample == results[0].counterexample for result in results)


def test_flagged_circuit_with_permuted_signed_outputs() -> None:
    """Many flags, a signed Y stabilizer, and permuted outputs; witnesses name DEM faults."""
    for damaging in (False, True):
        circuit = stim.Circuit()
        circuit.append("I", range(9))
        for flag in range(1, 16):
            for syndrome in (3, 5, 6):
                targets = [stim.target_x(jj) for jj in range(3) if (syndrome >> jj) & 1]
                targets += [stim.target_x(5 + jj) for jj in range(4) if (flag >> jj) & 1]
                circuit.append("CORRELATED_ERROR", targets, 0.001)
        if damaging:  # also flips the outputs on qubits 3 and 4, with the flag on qubit 5
            circuit.append("CORRELATED_ERROR", [stim.target_x(jj) for jj in (3, 4, 5)], 0.001)
        circuit += stim.Circuit("H 0 1 2\nS 0 2\nX 0")
        circuit.append("M", range(5, 9))
        for jj in range(4):
            circuit.append("DETECTOR", [stim.target_rec(jj - 4)])
        result = circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=[2, 0, 4, 1, 3], post_select=range(4), max_faults=3
        )
        assert any("Y" in ss for ss in result.target_stabilizers)
        assert any(ss.startswith("-") for ss in result.target_stabilizers)
        assert result.certified != damaging
        if damaging:
            _assert_valid_counterexample(result, circuit)


def test_steane_preparation_meets_css_weight_but_not_physical_weight() -> None:
    """A fault of Steane preparation is equivalent to CSS weight one, not to physical weight one."""
    circuit = circuits.DepolarizingNoiseModel(1e-3).noisy_circuit(
        stim.Circuit("""
            H 0 2 4
            CX 0 3 2 1 4 5
            CX 0 1 2 6 4 3
            CX 2 5 3 6
            H 7
            CZ 7 1 7 3 7 5
            MX 7
            DETECTOR rec[-1]
        """)
    )

    def certify(css_weight: bool | None) -> circuits.BoundedFaultPropagationCertificate:
        return circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=range(7), post_select=[0], css_weight=css_weight
        )

    css = certify(None)
    assert css.certified and css.css_weight is True
    # The default outputs are qubits 0-6: the flag qubit 7 is measured.
    assert circuits.certify_bounded_fault_propagation(circuit, post_select=[0]) == css
    physical = certify(False)
    assert physical.css_weight is False
    _assert_valid_counterexample(physical, circuit)
    failure = physical.counterexample
    assert failure is not None
    # Every Pauli of weight at most one has a different syndrome, so this syndrome class is heavier.
    stabilizers = [stim.PauliString(ss) for ss in physical.target_stabilizers]
    for qubit, pauli in itertools.product(range(7), "IXYZ"):
        candidate = stim.PauliString(7)
        candidate[qubit] = pauli
        syndrome = tuple(int(not candidate.commutes(ss)) for ss in stabilizers)
        assert syndrome != failure.output_syndrome


def test_added_flags_match_explicit_flag_detectors() -> None:
    """Bare measurements become flag detectors, and addressed measurements do not."""
    body = """
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
    """

    def certify(
        circuit: stim.Circuit,
        post_select: bool | Sequence[int] | Literal["flags"],
        add_flags: bool,
    ) -> circuits.BoundedFaultPropagationCertificate:
        return circuits.certify_bounded_fault_propagation(
            circuit, post_select=post_select, add_flags=add_flags
        )

    explicit = stim.Circuit(body + "DETECTOR rec[-1]")
    reference = certify(explicit, [0], add_flags=False)
    assert not reference.certified and reference.counterexample is not None
    assert reference.counterexample.fault_count == 2
    # The added flag is detector 0, so these all select it.
    for post_select in ("flags", True, [0], [-1]):
        assert certify(stim.Circuit(body), post_select, add_flags=True) == reference
    # The measurement is already addressed, so there is no flag to select and a single fault fails.
    unflagged = certify(explicit, "flags", add_flags=True)
    assert unflagged.counterexample is not None and unflagged.counterexample.fault_count == 1
    assert reference.post_selection_detectors == (0,) and unflagged.post_selection_detectors == ()
    # The added flag follows the circuit's own detectors.
    with_detector = stim.Circuit(body + "M 3\nDETECTOR rec[-1]")
    selections: list[tuple[Any, tuple[int, ...]]] = [("flags", (1,)), ([-1], (1,)), (True, (0, 1))]
    for post_select, expected in selections:
        certificate = certify(with_detector, post_select, add_flags=True)
        assert certificate.post_selection_detectors == expected


def test_invalid_post_selection_is_rejected() -> None:
    """Flags need add_flags, and selections are distinct integer indices of selectable detectors."""
    circuit = stim.Circuit("I 0\nCORRELATED_ERROR(0.01) X0\nM 1\nDETECTOR rec[-1]")

    def certify(**kwargs: Any) -> circuits.BoundedFaultPropagationCertificate:
        return circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0], **kwargs)

    assert certify(add_flags=True, post_select="flags").certified
    for post_select in (True, False, [0], [-1], range(1), [np.int64(0)]):
        assert certify(post_select=post_select).certified
    invalid: list[Any] = ["flags", "all", [1], [-2], [0, -1], [0.5], [False], 0]
    for post_select in invalid:
        with pytest.raises(ValueError):
            certify(post_select=post_select)
    with pytest.raises(ValueError):
        certify(add_flags=True, post_select="all")


@pytest.mark.parametrize("css_weight", [np.bool_(True), 1, 0, "css"])
def test_invalid_css_weight_is_rejected(css_weight: object) -> None:
    """Only bool and None are accepted, so a numpy boolean, an integer, or a string is rejected."""
    with pytest.raises(ValueError, match="css_weight"):
        circuits.certify_bounded_fault_propagation(
            stim.Circuit("I 0"), output_qubits=[0], css_weight=cast(bool, css_weight)
        )


@pytest.mark.parametrize("max_faults", [0, True, 1.5])
def test_invalid_max_faults_is_rejected(max_faults: int) -> None:
    """The limit must be a positive integer or None, and a bool is not an integer here."""
    circuit = stim.Circuit("I 0")
    with pytest.raises(ValueError, match="max_faults"):
        circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=[0], max_faults=max_faults
        )
    result = circuits.certify_bounded_fault_propagation(
        circuit, output_qubits=[0], max_faults=cast(int, np.int64(2))
    )
    assert result.max_faults == 2 and type(result.max_faults) is int


@pytest.mark.parametrize("outputs", [0, [], [0, 0], [1], [-1], [True], [0.5]])
def test_invalid_output_qubits_are_rejected(outputs: object) -> None:
    """Output qubits must be a nonempty sequence of distinct in-range integers."""
    with pytest.raises(ValueError, match="output_qubits"):
        circuits.certify_bounded_fault_propagation(
            stim.Circuit("I 0"), output_qubits=cast(Sequence[int], outputs)
        )


def test_unsupported_circuits_and_detectors_are_rejected() -> None:
    """Mixed or random outputs and nondeterministic annotations raise."""
    certify = circuits.certify_bounded_fault_propagation
    # An output entangled with an unselected qubit, and an output that was measured.
    for text in ("H 0\nCX 0 1", "H 0\nM 0"):
        with pytest.raises(ValueError):
            certify(stim.Circuit(text), output_qubits=[0])
    assert certify(stim.Circuit("H 0\nCX 0 1"), output_qubits=[0, 1]).certified
    assert certify(stim.Circuit("H 0\nM 1"), output_qubits=[0]).certified
    # Random detectors and observables on qubits other than the output.
    for text in (
        "H 1\nM 1\nDETECTOR rec[-1]",
        "H 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ):
        with pytest.raises(ValueError):
            certify(stim.Circuit(text), output_qubits=[0])
    assert certify(stim.Circuit("H 1\nM 1"), output_qubits=[0]).certified
    # A random measurement is also nondeterministic as an added flag detector.
    with pytest.raises(ValueError):
        certify(stim.Circuit("H 1\nM 1"), output_qubits=[0], add_flags=True)
    # There are no unmeasured qubits to default to.
    with pytest.raises(ValueError):
        certify(stim.Circuit("H 0\nM 0"))
    assert certify(stim.Circuit("H 0\nM 0\nI 1")).output_qubits == (1,)
