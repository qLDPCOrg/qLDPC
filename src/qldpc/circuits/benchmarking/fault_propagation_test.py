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
    """Stabilizers of |0...0>.

    The smallest Pauli weight of an error with a given syndrome is the syndrome's number of ones.
    """
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _smallest_violation(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None
) -> int | None:
    """The smallest fault weight of an accepted fault that violates the bound on |0...0>."""
    limit = len(flags) if max_weight is None else min(max_weight, len(flags))
    for fault_weight in range(1, limit + 1):
        for indices in itertools.combinations(range(len(flags)), fault_weight):
            flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
            syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
            if flag == 0 and syndrome.bit_count() > fault_weight:
                return fault_weight
    return None


def _certify_map_with_count(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None, num_qubits: int
) -> tuple[circuits.BoundedFaultPropagationCounterexample | None, int]:
    """Certify primitive faults with these flags and syndromes on |0...0> against brute force."""
    failure, num_checked = fault_propagation._certify_maps(
        flags, syndromes, _z_state(num_qubits), False, max_weight
    )
    smallest = _smallest_violation(flags, syndromes, max_weight)
    assert (failure is None) == (smallest is None)
    if failure is not None:
        indices = failure.fault_indices
        flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
        syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
        assert failure.fault_weight == smallest
        assert indices == tuple(sorted(set(indices)))
        assert flag == 0
        assert failure.output_syndrome == tuple((syndrome >> jj) & 1 for jj in range(num_qubits))
        assert failure.fault_weight < failure.output_weight_lower_bound <= syndrome.bit_count()
        # X on a qubit anticommutes with that qubit's stabilizer Z.
        xs, _ = failure.output_pauli.to_numpy()
        assert tuple(map(int, xs)) == failure.output_syndrome
    return failure, num_checked


def _certify_map(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None, num_qubits: int
) -> circuits.BoundedFaultPropagationCounterexample | None:
    """Certify primitive faults with these flags and syndromes on |0...0> against brute force."""
    return _certify_map_with_count(flags, syndromes, max_weight, num_qubits)[0]


def _true_minimum(
    stabilizers: Sequence[stim.PauliString], syndrome: Sequence[int], metric_css: bool
) -> int:
    """Minimum Pauli weight over all Paulis with this syndrome, by enumerating every Pauli."""
    n = len(stabilizers)
    rows = np.array([np.concatenate(ss.to_numpy()) for ss in stabilizers])
    paulis = (np.arange(4**n)[:, None] >> np.arange(2 * n)) & 1
    x, z = paulis[:, :n], paulis[:, n:]
    syndromes = (x @ rows[:, n:].T + z @ rows[:, :n].T) % 2
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if metric_css else (x | z).sum(axis=1)
    return int(weights[np.all(syndromes == np.array(syndrome), axis=1)].min())


def _primitive_fault_detectors(
    certificate: circuits.BoundedFaultPropagationCertificate,
) -> list[set[int]]:
    """The detectors that each primitive fault flips, in DEM order."""
    mechanisms = [inst for inst in certificate.detector_error_model if inst.type == "error"]
    return [
        {tt.val for tt in mechanism.targets_copy() if tt.is_relative_detector_id()}
        for mechanism in mechanisms
    ]


def _primitive_fault_syndromes(
    certificate: circuits.BoundedFaultPropagationCertificate, circuit: stim.Circuit
) -> list[tuple[int, ...]]:
    """Output syndrome of every primitive fault in the certificate's DEM.

    The detectors of the circuit come first, and one detector per target stabilizer follows.
    """
    return [
        tuple(
            int(circuit.num_detectors + jj in events)
            for jj in range(len(certificate.target_stabilizers))
        )
        for events in _primitive_fault_detectors(certificate)
    ]


def _assert_valid_counterexample(
    certificate: circuits.BoundedFaultPropagationCertificate,
    circuit: stim.Circuit,
) -> None:
    """Check a counterexample against the returned DEM and a brute-force minimum, not the search."""
    failure = certificate.counterexample
    assert failure is not None and not certificate.certified
    detectors = _primitive_fault_detectors(certificate)
    events: set[int] = set()
    for index in failure.fault_indices:
        events ^= detectors[index]
    assert not events & set(certificate.post_selection_detectors)
    syndrome = tuple(
        int(circuit.num_detectors + jj in events)
        for jj in range(len(certificate.target_stabilizers))
    )
    assert syndrome == failure.output_syndrome
    pauli = failure.output_pauli
    stabilizers = certificate.target_stabilizers
    assert tuple(int(not pauli.commutes(ss)) for ss in stabilizers) == failure.output_syndrome
    assert failure.output_weight_lower_bound > failure.fault_weight
    minimum = _true_minimum(certificate.target_stabilizers, syndrome, certificate.css_weight)
    assert failure.output_weight_lower_bound <= minimum


def test_certify_maps_without_flags_checks_primitive_faults(pytestconfig: pytest.Config) -> None:
    """With zero flags, the first primitive fault whose error has Pauli weight above one fails."""
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

    # Primitive faults whose errors have Pauli weight at most one certify, and the first with Pauli
    # weight three does not.
    assert certify([0, 1, 2, 4, 2], 3) is None
    assert certify([1, 2, 7, 4, 7], 3) is not None
    # Sparse random syndromes mix certified maps with failing ones.
    for num_qubits in (3, 5):
        for _ in range(20):
            sparse = rng.integers(1 << num_qubits, size=(6, 3))
            certify([int(a & b & c) for a, b, c in sparse], num_qubits)


def test_certify_maps_with_flags_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """Random maps with repeated flags and flag rank above ``max_weight`` match brute force."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for n, rank in itertools.product((3, 5), range(5)):
        for _ in range(12):
            flags = list(map(int, rng.integers(1 << rank, size=8)))
            syndromes = list(map(int, rng.integers(1 << n, size=8)))
            for max_weight in (1, 2, 3, 5, None):
                _certify_map(flags, syndromes, max_weight, n)


def test_hand_derived_dependency_cases() -> None:
    """Faults of two or three primitive faults, repeated and zero signatures, and limits.

    The limits are values of ``max_weight``.
    """
    for flags, syndromes, max_weight, certified in (
        ([1, 2, 3], [1, 2, 12], 2, True),  # all three: fault weight 3, error of Pauli weight 4
        ([1, 2, 3], [1, 2, 12], 3, False),
        ([1, 1], [1, 14], 3, False),  # an accepted fault whose error has Pauli weight 4
        ([0, 0, 0], [1, 2, 4], 4, True),  # zero flags; each error has Pauli weight one
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        # the accepted faults have fault weight two and errors of Pauli weight two
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),  # equal signatures cancel each other
        ([0, 0], [0, 0], 4, True),  # primitive faults with no effect
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 3, True),
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 4, False),  # fault weight 4, error of Pauli weight 6
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], None, False),
    ):
        failure = _certify_map(flags, syndromes, max_weight, 6)
        assert (failure is None) == certified


def test_failures_report_the_smallest_and_earliest_faults() -> None:
    """The smallest fault weight and the earliest primitive faults are reported.

    A fault of two primitive faults is reported although a fault of three primitive faults also
    violates the bound, and a repeated signature stands for its first primitive fault.
    """
    # The primitive faults 0-2 form an accepted fault whose error has Pauli weight four, and the
    # primitive faults 3 and 4 an accepted fault whose error has Pauli weight three.
    failure = _certify_map([1, 2, 3, 4, 4], [1, 2, 12, 1, 6], None, 6)
    assert failure is not None and failure.fault_indices == (3, 4)
    # Primitive faults 0 and 2 have the same signature, so the fault {0, 1} is reported.
    failure = _certify_map([1, 1, 1], [3, 4, 3], None, 3)
    assert failure is not None and failure.fault_indices == (0, 1)
    # The heavy primitive fault 1 is searched before the light one 0, but the indices are reported
    # in increasing order.
    failure = _certify_map([1, 1], [1, 14], None, 4)
    assert failure is not None and failure.fault_indices == (0, 1)


def test_light_primitive_faults_are_never_searched() -> None:
    """Without a heavy primitive fault, no fault can violate the bound, so none is checked."""
    flags = [3, 5, 6, 9, 10, 12, 1, 2, 4, 8, 7, 11]
    syndromes = [1, 2, 4, 8, 0, 1, 2, 4, 8, 0, 1, 2]
    failure, num_checked = _certify_map_with_count(flags, syndromes, None, 4)
    assert failure is None and num_checked == 0

    # One primitive fault whose error has Pauli weight three brings back the search, and it fails
    # together with the primitive fault that has the same flags.
    failure, num_checked = _certify_map_with_count(flags + [3], syndromes + [14], None, 4)
    assert failure is not None and failure.fault_weight == 2
    assert num_checked > 0


def test_faults_of_flagged_light_primitive_faults_are_skipped() -> None:
    """Only the faults that contain a heavy primitive fault are checked."""
    # The faults {0, 1} and {2, 3} have light primitive faults only, so they are skipped.  The fault
    # {4, 5} and the eight faults of fault weight three with one primitive fault from each group of
    # primitive faults with equal flags contain a heavy primitive fault, which makes nine faults.
    failure, num_checked = _certify_map_with_count([1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 5, 6], None, 4)
    assert failure is None and num_checked == 9


def test_zero_and_repeated_signatures_are_not_searched_and_the_fault_weight_is_limited() -> None:
    """Zero and repeated signatures are never searched, and the checked fault weight is limited."""
    failure, num_checked = _certify_map_with_count([0, 1, 1], [0, 1, 1], None, 4)
    assert failure is None and num_checked == 0
    # Each distinct nonzero signature with zero flags is an irreducible accepted fault, and a
    # repeated signature is not checked again.
    failure, num_checked = _certify_map_with_count([0, 0, 0, 0], [1, 1, 4, 0], None, 4)
    assert failure is None and num_checked == 2

    # The flags 1, 2, and 3 have rank two, and only the fault of all three primitive faults is
    # accepted.
    for max_weight in (None, 30, 2):
        failure, num_checked = _certify_map_with_count([1, 2, 3], [1, 2, 12], max_weight, 4)
        assert (failure is None) == (max_weight == 2)
        assert num_checked == (0 if max_weight == 2 else 1)


@pytest.mark.parametrize("css_weight", [True, False])
@pytest.mark.parametrize("num_qubits", [3, 4])
def test_ghz_preparation_matches_brute_force_over_primitive_faults(
    num_qubits: int, css_weight: bool
) -> None:
    """Fanning out a GHZ state certifies for three qubits but not for four.

    X on a set of GHZ qubits is equivalent to X on the complement, so only a two-qubit X error on
    four qubits has no equivalent error of lower Pauli weight.  Without post-selection,
    ``max_weight`` does not change the result.
    """
    circuit = stim.Circuit("H 0")
    for target in range(1, num_qubits):
        circuit.append("CX", [0, target])
        circuit.append("DEPOLARIZE2", [0, target], 0.01)
    max_weights = (1, 2, 5, None)
    results = [
        circuits.certify_bounded_fault_propagation(
            circuit, css_weight=css_weight, max_weight=max_weight
        )
        for max_weight in max_weights
    ]
    syndromes = _primitive_fault_syndromes(results[0], circuit)
    minima = [_true_minimum(results[0].target_stabilizers, ss, css_weight) for ss in syndromes]
    first = next((jj for jj, minimum in enumerate(minima) if minimum > 1), None)
    assert [result.max_weight for result in results] == list(max_weights)
    assert all(result.certified == (first is None) == (num_qubits == 3) for result in results)
    assert all(result.counterexample == results[0].counterexample for result in results)
    if first is not None:
        _assert_valid_counterexample(results[0], circuit)
        assert results[0].counterexample is not None
        assert results[0].counterexample.fault_indices == (first,)


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
    assert any(ss.sign == -1 for ss in result.target_stabilizers)
    simulator = stim.TableauSimulator()
    simulator.do(circuit.without_noise())
    for stabilizer in result.target_stabilizers:
        physical = stim.PauliString(circuit.num_qubits)
        for index, qubit in enumerate(result.output_qubits):
            physical[qubit] = stabilizer[index]
        assert simulator.peek_observable_expectation(stabilizer.sign * physical) == 1


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


def test_correlated_mechanism_is_one_primitive_fault() -> None:
    """A two-qubit error mechanism is one primitive fault whose error has Pauli weight two."""
    circuit = stim.Circuit("I 0 1\nCORRELATED_ERROR(0.01) X0 Y1")
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[1, 0])
    _assert_valid_counterexample(result, circuit)
    failure = result.counterexample
    assert failure is not None
    assert failure.fault_indices == (0,) and failure.output_weight_lower_bound == 2
    assert result.output_qubits == (1, 0)


def test_exclusive_errors_count_as_separate_primitive_faults() -> None:
    """Errors that exclude each other in the circuit are still separate primitive faults."""
    circuit = stim.Circuit("I 0 1\nCORRELATED_ERROR(0.1) X0\nELSE_CORRELATED_ERROR(0.1) X0 X1")
    result = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0, 1])
    _assert_valid_counterexample(result, circuit)


def test_existing_detectors_do_not_enter_the_syndrome() -> None:
    """A primitive fault that flips a detector of the circuit is judged by its output alone."""
    noise = "CORRELATED_ERROR(0.01) X0 X1 X2"
    detected = stim.Circuit(f"I 0 1\n{noise}\nM 2\nDETECTOR rec[-1]")
    result = circuits.certify_bounded_fault_propagation(detected, output_qubits=[0, 1])
    _assert_valid_counterexample(result, detected)
    undetected = circuits.certify_bounded_fault_propagation(
        stim.Circuit(f"I 0 1\n{noise}"), output_qubits=[0, 1]
    )
    assert result.counterexample == undetected.counterexample


def test_existing_observables_do_not_enter_the_syndrome() -> None:
    """A primitive fault that flips an observable is judged by its output error alone."""
    noise = "CORRELATED_ERROR(0.01) X1 X2\nM 2"  # an output error of Pauli weight one, and rec[-1]
    flipped = stim.Circuit(f"{noise}\nOBSERVABLE_INCLUDE(0) rec[-1]")
    result = circuits.certify_bounded_fault_propagation(flipped, output_qubits=[0, 1])
    assert result.detector_error_model.num_observables == 1
    plain = circuits.certify_bounded_fault_propagation(stim.Circuit(noise), output_qubits=[0, 1])
    assert result.certified and plain.certified


def test_primitive_faults_inside_repeat_blocks_are_certified() -> None:
    """Primitive faults in a loop count although Stim folds the loop into a repeat block."""
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
        result.counterexample.output_weight_lower_bound
        == written_out.counterexample.output_weight_lower_bound
    )


def test_circuit_without_error_mechanisms_certifies() -> None:
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
    assert result.certified and result.target_stabilizers == (stim.PauliString("-Z"),)


def test_post_selection_can_accept_a_damaging_fault() -> None:
    """Two primitive faults that each flip the detector do not fire it together.

    Their fault is accepted and damages four outputs.
    """
    circuit = stim.Circuit("""
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
        DETECTOR rec[-1]
    """)

    def certify(
        max_weight: int | None, post_select: bool | Sequence[int] = (0,)
    ) -> circuits.BoundedFaultPropagationCertificate:
        return circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=[0, 1, 2, 3], post_select=post_select, max_weight=max_weight
        )

    # A primitive fault alone is not accepted, so the fault of both is the smallest violation.
    assert certify(1).certified
    result = certify(2)
    _assert_valid_counterexample(result, circuit)
    assert result.counterexample is not None and result.counterexample.fault_indices == (0, 1)
    assert certify(None).counterexample == result.counterexample

    # True selects every detector of the circuit, here detector 0, and False selects none.
    assert certify(2, True) == result
    unselected = certify(2, False)
    assert unselected.counterexample is not None and unselected.counterexample.fault_weight == 1
    default = circuits.certify_bounded_fault_propagation(circuit, output_qubits=[0, 1, 2, 3])
    assert default == certify(None, False)


def test_flagged_circuit_with_permuted_signed_outputs() -> None:
    """Many flags, a signed Y stabilizer, and permuted outputs.

    The counterexample's ``fault_indices`` index the error instructions of the DEM.
    """
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
            circuit, output_qubits=[2, 0, 4, 1, 3], post_select=range(4), max_weight=3
        )
        assert any("Y" in str(ss) for ss in result.target_stabilizers)
        assert any(ss.sign == -1 for ss in result.target_stabilizers)
        assert result.certified != damaging
        if damaging:
            _assert_valid_counterexample(result, circuit)


def test_steane_preparation_meets_css_weight_but_not_physical_weight() -> None:
    """Steane preparation meets the bound for CSS weight, not for physical weight."""
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
    # Every Pauli of Pauli weight at most one has a different syndrome, so every error with this
    # syndrome has Pauli weight above one.
    stabilizers = physical.target_stabilizers
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
    assert reference.counterexample.fault_weight == 2
    # The added flag is detector 0, so these all select it.
    for post_select in ("flags", True, [0], [-1]):
        assert certify(stim.Circuit(body), post_select, add_flags=True) == reference
    # The measurement is already addressed, so there is no flag to select and a primitive fault
    # alone fails.
    unflagged = certify(explicit, "flags", add_flags=True)
    assert unflagged.counterexample is not None and unflagged.counterexample.fault_weight == 1
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


@pytest.mark.parametrize("css_weight", [np.bool_(True), 1, 0])
def test_invalid_css_weight_is_rejected(css_weight: object) -> None:
    """Only bool and None are accepted, so a numpy boolean or an integer is rejected."""
    with pytest.raises(ValueError, match="css_weight"):
        circuits.certify_bounded_fault_propagation(
            stim.Circuit("I 0"), output_qubits=[0], css_weight=cast(bool, css_weight)
        )


@pytest.mark.parametrize("max_weight", [0, True, 1.5])
def test_invalid_max_weight_is_rejected(max_weight: int) -> None:
    """The limit must be a positive integer or None, and a bool is not an integer here."""
    circuit = stim.Circuit("I 0")
    with pytest.raises(ValueError, match="max_weight"):
        circuits.certify_bounded_fault_propagation(
            circuit, output_qubits=[0], max_weight=max_weight
        )
    result = circuits.certify_bounded_fault_propagation(
        circuit, output_qubits=[0], max_weight=cast(int, np.int64(2))
    )
    assert result.max_weight == 2 and type(result.max_weight) is int


def test_memory_limit_does_not_change_results(pytestconfig: pytest.Config) -> None:
    """With almost no memory, the search is slower but finds the same counterexample."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first map has flags of rank four, so that its search reaches fault weight five.
    maps = [([1, 2, 4, 8, 15, 3], [3, 0, 1, 2, 5, 6])]
    for _ in range(100):
        num_primitives = int(rng.integers(4, 10))
        flags = list(map(int, rng.integers(1, 16, size=num_primitives)))
        maps.append((flags, list(map(int, rng.integers(16, size=num_primitives)))))
    for flags, syndromes in maps:
        results = [
            fault_propagation._certify_maps(flags, syndromes, _z_state(4), False, None, limit)
            for limit in (1, 2**20)
        ]
        assert results[0] == results[1]
        assert results[0][0] == _certify_map(flags, syndromes, None, 4)


@pytest.mark.parametrize("memory_limit", [0, True, 1.5])
def test_invalid_memory_limit_is_rejected(memory_limit: int) -> None:
    """The memory limit must be a positive integer, and a bool is not an integer here."""
    with pytest.raises(ValueError, match="memory_limit"):
        circuits.certify_bounded_fault_propagation(
            stim.Circuit("I 0"), output_qubits=[0], memory_limit=memory_limit
        )


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
