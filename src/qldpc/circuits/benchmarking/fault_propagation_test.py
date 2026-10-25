# SPDX-License-Identifier: Apache-2.0

"""Tests for exact certification of bounded fault propagation in noisy state preparation."""

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits

from . import fault_propagation


def _z_state(num_qubits: int) -> npt.NDArray[np.int_]:
    """Stabilizers of |0...0>, for which an error's effective weight is its syndrome's weight."""
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _certify_map(
    syndromes: Sequence[int], num_qubits: int
) -> circuits.BoundedFaultPropagationCounterexample | None:
    """Certify a map from faults to syndromes on |0...0> and compare with every set of faults."""
    failure = fault_propagation._certify_maps(syndromes, _z_state(num_qubits), False)
    first = next((jj for jj, syndrome in enumerate(syndromes) if syndrome.bit_count() > 1), None)
    # Sets of every size, not only single faults, meet the bound exactly when single faults do.
    violated = any(
        functools.reduce(operator.xor, (syndromes[jj] for jj in support)).bit_count() > size
        for size in range(1, len(syndromes) + 1)
        for support in itertools.combinations(range(len(syndromes)), size)
    )
    assert violated == (first is not None)
    assert (failure is None) == (first is None)
    if failure is not None:
        assert first is not None
        syndrome = syndromes[first]
        assert failure.fault_indices == (first,) and failure.fault_count == 1
        assert failure.output_syndrome == tuple((syndrome >> jj) & 1 for jj in range(num_qubits))
        assert 1 < failure.effective_weight_lower_bound <= syndrome.bit_count()
        # X on a qubit anticommutes with that qubit's stabilizer Z.
        xs, _ = stim.PauliString(failure.output_pauli).to_numpy()
        assert tuple(map(int, xs)) == failure.output_syndrome
    return failure


def _true_minimum(stabilizers: Sequence[str], syndrome: Sequence[int], metric_css: bool) -> int:
    """Minimum weight over all Paulis with this syndrome, by enumerating every Pauli."""
    n = len(stabilizers)
    rows = np.array([np.concatenate(stim.PauliString(ss).to_numpy()) for ss in stabilizers])
    paulis = (np.arange(4**n)[:, None] >> np.arange(2 * n)) & 1
    x, z = paulis[:, :n], paulis[:, n:]
    syndromes = (x @ rows[:, n:].T + z @ rows[:, :n].T) % 2
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if metric_css else (x | z).sum(axis=1)
    return int(weights[np.all(syndromes == np.array(syndrome), axis=1)].min())


def _fault_syndromes(
    certificate: circuits.BoundedFaultPropagationCertificate, circuit: stim.Circuit
) -> list[tuple[int, ...]]:
    """Output syndrome of every error mechanism in the certificate's DEM.

    The detectors of the circuit come first, and one detector per target stabilizer follows.
    """
    errors = [inst for inst in certificate.detector_error_model if inst.type == "error"]
    syndromes = []
    for error in errors:
        events = {tt.val for tt in error.targets_copy() if tt.is_relative_detector_id()}
        syndromes.append(
            tuple(
                int(circuit.num_detectors + jj in events)
                for jj in range(len(certificate.target_stabilizers))
            )
        )
    return syndromes


def _assert_valid_counterexample(
    certificate: circuits.BoundedFaultPropagationCertificate, circuit: stim.Circuit
) -> None:
    """Check a witness against the returned DEM and a brute-force minimum, not the search."""
    failure = certificate.counterexample
    assert failure is not None and not certificate.certified
    syndrome = _fault_syndromes(certificate, circuit)[failure.fault_indices[0]]
    assert syndrome == failure.output_syndrome
    stabilizers = [stim.PauliString(ss) for ss in certificate.target_stabilizers]
    pauli = stim.PauliString(failure.output_pauli)
    assert tuple(int(not pauli.commutes(ss)) for ss in stabilizers) == failure.output_syndrome
    assert failure.effective_weight_lower_bound > failure.fault_count
    minimum = _true_minimum(certificate.target_stabilizers, syndrome, certificate.css_weight)
    assert failure.effective_weight_lower_bound <= minimum


def test_certify_maps_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """Hand-made and random syndrome maps agree with a check of every set of faults."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # Faults of weight at most one certify, and a fault of weight three does not.
    assert _certify_map([0, 1, 2, 4, 2], 3) is None
    failure = _certify_map([1, 2, 7, 4, 7], 3)
    assert failure is not None and failure.fault_indices == (2,)
    # Sparse random syndromes mix certified maps with failing ones.
    for num_qubits in (3, 5):
        for _ in range(20):
            sparse = rng.integers(1 << num_qubits, size=(6, 3))
            _certify_map([int(a & b & c) for a, b, c in sparse], num_qubits)


@pytest.mark.parametrize("css_weight", [True, False])
@pytest.mark.parametrize("num_qubits", [3, 4])
def test_ghz_preparation_matches_brute_force_over_faults(num_qubits: int, css_weight: bool) -> None:
    """Fanning out a GHZ state: X on the control spreads to two of four targets, but not of two.

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
    assert "L0" in str(result.detector_error_model)
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


def test_invalid_css_weight_is_rejected() -> None:
    """Only bool and None are accepted, so a numpy boolean is rejected."""
    with pytest.raises(ValueError, match="css_weight"):
        circuits.certify_bounded_fault_propagation(
            stim.Circuit("I 0"), output_qubits=[0], css_weight=cast(bool, np.bool_(True))
        )


@pytest.mark.parametrize("outputs", [[], [0, 0], [1], [-1], [True], [0.5]])
def test_invalid_output_qubits_are_rejected(outputs: list[int]) -> None:
    """Output qubits must be a nonempty list of distinct in-range integers."""
    with pytest.raises(ValueError, match="output_qubits"):
        circuits.certify_bounded_fault_propagation(stim.Circuit("I 0"), output_qubits=outputs)


def test_unsupported_circuits_and_detectors_are_rejected() -> None:
    """Mixed or random outputs and nondeterministic annotations raise."""
    # An output entangled with an unselected qubit, and an output that was measured.
    for text in ("H 0\nCX 0 1", "H 0\nM 0"):
        with pytest.raises(ValueError):
            circuits.certify_bounded_fault_propagation(stim.Circuit(text), output_qubits=[0])
    # Random detectors and observables on qubits other than the output.
    for text in (
        "H 1\nM 1\nDETECTOR rec[-1]",
        "H 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ):
        with pytest.raises(ValueError):
            circuits.certify_bounded_fault_propagation(stim.Circuit(text), output_qubits=[0])
    # There are no unmeasured qubits to default to.
    with pytest.raises(ValueError):
        circuits.certify_bounded_fault_propagation(stim.Circuit("H 0\nM 0"))
