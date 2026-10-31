# SPDX-License-Identifier: Apache-2.0

"""Tests for deferred-measurement form, using stim's detector error models as the oracle."""

from __future__ import annotations

import collections
import math
from collections.abc import Callable

import galois
import numpy as np
import pytest
import stim

from qldpc.experimental.circuits.transforms import deferred_measurement

# Fragments of random circuits on qubits a and b, by kind.  Stim reads T-tagged instructions as
# Clifford gates, which does not matter here, since no pass depends on the gates of a circuit.
_GATES: tuple[str, ...] = ("H {a}", "S {a}", "SQRT_X_DAG {a}", "CX {a} {b}", "CZ {a} {b}")
_GATES += ("S[T] {a}", "SPP_DAG[T] X{a}*Z{b}")
_MEASUREMENTS: tuple[str, ...] = ("M {a}", "MX !{a}", "MY(0.1) {a}", "MZZ {a} {b}")
_MEASUREMENTS += ("MXX(0.05) {a} !{b}", "MPP X{a}*Y{b}", "MR {a}", "MRX(0.1) {a}", "MRY {a}")
# Repeated measurements, whose pairs of records are deterministic, test inversions and the frames of
# deferred measurements.
_MEASUREMENTS += ("MY !{a}\nMY {a}", "MX(0.1) !{a}\nMX !{a}", "MXX !{a} {b}\nMXX {a} {b}")
_MEASUREMENTS += ("MPP !X{a}*Y{b}\nMPP X{a}*Y{b}", "MRY !{a}\nMY {a}")
_MEASUREMENTS += ("MPP !X{a}*!Y{b}\nMPP X{a}*Y{b}",)
_RESETS = ("R {a}", "RX {a}", "RY {a}")
_NOISE: tuple[str, ...] = ("X_ERROR(0.1) {a}", "DEPOLARIZE1(0.1) {a}", "DEPOLARIZE2(0.1) {a} {b}")
_NOISE += ("E(0.1) X{a} Z{b}", "PAULI_CHANNEL_1(0.05, 0.1, 0.02) {a}")
# Detectors and observables, which stim requires to be deterministic without noise.  Each observable
# fragment reads a Pauli right after measuring or preparing it, or right before measuring it.  The
# REPEAT block shifts the coordinates of its detector, and ends with an X-basis measurement, after
# which simplify_measurements changes the state of the measured qubit.
_DETECTORS_AND_OBSERVABLES = (
    "MX {a}\nOBSERVABLE_INCLUDE(0) rec[-1] X{a}",
    "OBSERVABLE_INCLUDE(1) X{a} Z{b}\nMPP X{a}*Z{b}\nOBSERVABLE_INCLUDE(1) rec[-1]",
    "R {a}\nX_ERROR(0.1) {a}\nOBSERVABLE_INCLUDE(2) Z{a}\nR {a}",
    "REPEAT 2 {{\nSHIFT_COORDS(0, 1)\nMRX {a}\nZ_ERROR(0.1) {a}\nMX !{a}\nDETECTOR(1, 0) rec[-1]\n}}",
)


def _get_random_circuit(rng: np.random.Generator, num_qubits: int = 3) -> stim.Circuit:
    """A random noisy circuit with measurements, resets, detectors, and observables."""
    kinds = (_GATES, _GATES, _NOISE, _MEASUREMENTS, _RESETS, _DETECTORS_AND_OBSERVABLES)
    lines = []
    for _ in range(12):
        fragments = kinds[rng.integers(len(kinds))]
        fragment = fragments[rng.integers(len(fragments))]
        a, b = (int(qubit) for qubit in rng.choice(num_qubits, size=2, replace=False))
        lines.append(fragment.format(a=a, b=b))
    return stim.Circuit("\n".join(lines))


def _assert_same_errors(
    circuit: stim.Circuit, rewritten: stim.Circuit, *, keep_measured: bool
) -> int:
    """Assert the same distributions of records and final states, with and without errors.

    Without noise, the measurement records of a Clifford circuit are uniformly distributed over an
    affine space, which is determined by the space of deterministic parities of records and their
    values.  Checks appended to both circuits measure the stabilizers of the final state of the
    qubits of the input circuit, restricted to the qubits that it never measures unless
    keep_measured is True; ancilla qubits that the rewritten circuit adds are not checked.
    Detectors on a basis of the deterministic parities of all records then compare the noiseless
    distributions of records and final states by their signs, and the errors by detector error
    models.  The detectors of both circuits must also have the same coordinates.  Returns the number
    of detectors on deterministic parities.
    """
    assert rewritten.get_detector_coordinates() == circuit.get_detector_coordinates()
    checks = _get_final_state_checks(circuit, keep_measured=keep_measured)
    circuit, rewritten = circuit + checks, rewritten + checks
    assert rewritten.num_measurements == circuit.num_measurements
    parities = _get_deterministic_parities(circuit)
    both = np.vstack([parities, _get_deterministic_parities(rewritten)]).view(galois.GF2)
    assert np.linalg.matrix_rank(both) == np.linalg.matrix_rank(parities) == len(parities)
    detectors = stim.Circuit()
    for parity in parities:
        records = np.flatnonzero(parity) - circuit.num_measurements
        detectors.append("DETECTOR", [stim.target_rec(int(record)) for record in records])
    circuit, rewritten = circuit + detectors, rewritten + detectors
    assert _get_errors(rewritten) == _get_errors(circuit), str(circuit)
    assert _get_signs(rewritten) == _get_signs(circuit)
    return len(parities)


def _get_final_state_checks(circuit: stim.Circuit, *, keep_measured: bool) -> stim.Circuit:
    """Noiseless measurements of the stabilizers of the final state of the qubits of a circuit.

    The measured stabilizers generate the stabilizer group of the final state of a noiseless run
    (for its random records) on all qubits of the circuit, including qubits that only noise acts on.
    Unless keep_measured is True, the stabilizers are restricted to qubits that the circuit never
    measures.
    """
    noiseless = circuit.without_noise()
    measured = sorted(
        {
            target.value
            for instruction in noiseless.flattened()
            if stim.gate_data(instruction.name).produces_measurements
            for target in instruction.targets_copy()
            if not target.is_combiner
        }
    )
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(circuit.num_qubits)
    simulator.do_circuit(noiseless)
    stabilizers = simulator.canonical_stabilizers()
    num_qubits = len(stabilizers)
    matrix = galois.GF2(
        [np.concatenate(stabilizer.to_numpy()).astype(np.uint8) for stabilizer in stabilizers]
    ).reshape(num_qubits, 2 * num_qubits)
    if measured and not keep_measured:  # keep products of stabilizers trivial on measured qubits
        columns = measured + [qubit + num_qubits for qubit in measured]
        matrix = galois.GF2(matrix[:, columns].T).null_space() @ matrix
    checks = stim.Circuit()
    for row in matrix:
        pauli = stim.PauliString.from_numpy(
            xs=row[:num_qubits].view(np.ndarray) == 1, zs=row[num_qubits:].view(np.ndarray) == 1
        )
        checks.append("MPP", stim.target_combined_paulis(pauli))
    return checks


def _get_deterministic_parities(circuit: stim.Circuit) -> galois.GF2:
    """A basis of the parities of measurement records that are deterministic without noise.

    The deterministic parities are those that vanish on the differences between noiseless samples
    and a reference sample, which span all such differences with overwhelming probability.  A missed
    difference would add a random parity, which stim rejects as a detector, so a missed difference
    cannot hide an error.
    """
    noiseless = circuit.without_noise()
    samples = noiseless.compile_sampler(seed=0).sample(1024) ^ noiseless.reference_sample()
    return galois.GF2(samples.astype(np.uint8)).null_space()


def _get_signs(circuit: stim.Circuit) -> tuple[list[bool], list[bool]]:
    """The noiseless parities of the detectors and observables of a circuit.

    Stim's reference sign of an observable ignores its Pauli targets, and takes records that are
    random from stim's reference sample, so this function first measures each Pauli target of an
    observable in place, by a Hadamard test on a fresh ancilla qubit that is measured at the end,
    and replaces the Pauli target by the record of the ancilla.  The detectors and observables must
    stay deterministic.
    """
    num_qubits = circuit.num_qubits
    lines = []
    ancillas: list[int] = []  # the observable of each ancilla
    for instruction in circuit.without_noise().flattened():
        if instruction.name != "OBSERVABLE_INCLUDE":
            lines.append(str(instruction))
            continue
        observable = int(instruction.gate_args_copy()[0])
        targets = instruction.targets_copy()
        paulis = [target for target in targets if not target.is_measurement_record_target]
        if paulis:
            ancilla = num_qubits + len(ancillas)
            ancillas.append(observable)
            lines.append(f"H {ancilla}")
            lines += [f"C{target.pauli_type} {ancilla} {target.value}" for target in paulis]
            lines.append(f"H {ancilla}")
        records = [f"rec[{target.value}]" for target in targets if target not in paulis]
        lines.append(f"OBSERVABLE_INCLUDE({observable}) " + " ".join(records))
    lines += [f"M {num_qubits + index}" for index in range(len(ancillas))]
    for index, observable in enumerate(ancillas):
        lines.append(f"OBSERVABLE_INCLUDE({observable}) rec[{index - len(ancillas)}]")
    measured = stim.Circuit("\n".join(lines))
    measured.detector_error_model()  # raises an error if anything is not deterministic
    detectors, observables = measured.reference_detector_and_observable_signs()
    return detectors.tolist(), observables.tolist()


def _get_errors(circuit: stim.Circuit) -> collections.Counter[tuple[str, float]]:
    """The error mechanisms of a detector error model, as a multiset that ignores their order."""
    errors: collections.Counter[tuple[str, float]] = collections.Counter()
    for instruction in circuit.detector_error_model(flatten_loops=True).flattened():
        if instruction.type == "error":
            targets = " ".join(sorted(str(target) for target in instruction.targets_copy()))
            errors[targets, round(instruction.args_copy()[0], 12)] += 1
    return errors


@pytest.mark.parametrize(
    "rewrite, simplified",
    [
        (deferred_measurement.split_measure_resets, False),
        (deferred_measurement.defer_resets, False),
        (
            lambda circuit: deferred_measurement.defer_measurements(
                deferred_measurement.defer_resets(circuit)
            ),
            False,
        ),
        (
            lambda circuit: deferred_measurement.simplify_measurements(
                deferred_measurement.defer_measurements(deferred_measurement.defer_resets(circuit))
            ),
            True,
        ),
    ],
    ids=["split", "defer_resets", "defer", "simplify"],
)
def test_passes_preserve_errors(
    rng: np.random.Generator, rewrite: Callable[[stim.Circuit], stim.Circuit], simplified: bool
) -> None:
    """Each pass preserves which records and final states every error flips.

    Only simplify_measurements may change the final states of measured qubits.  The passes, run in
    order, give circuits in deferred-measurement form.
    """
    num_detectors = 0
    for _ in range(15):
        circuit = _get_random_circuit(rng)
        rewritten = rewrite(circuit)
        num_detectors += _assert_same_errors(circuit, rewritten, keep_measured=not simplified)
        assert deferred_measurement.is_in_deferred_measurement_form(rewritten) or not simplified
    assert num_detectors  # the checks are not vacuous


def test_is_in_deferred_measurement_form() -> None:
    """A circuit with every kind of instruction that deferred-measurement form allows is in it."""
    circuit = stim.Circuit(
        """
        H 0
        OBSERVABLE_INCLUDE(0) X0
        REPEAT 2 {
            X_ERROR(0.1) 1
        }
        M[tag] 0 1
        M 2
        OBSERVABLE_INCLUDE(0) rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(1)
        """
    )
    assert deferred_measurement.is_in_deferred_measurement_form(circuit)


@pytest.mark.parametrize(
    "text",
    [
        "REPEAT 2 {\nH 0\nSHIFT_COORDS(1)\n}\nM 0",
        "M(0.1) 0",
        "M !0",
        "M 0\nM 0",
        "DETECTOR(1)\nM 0",
        "R 0",
        "M 0\nH 1",
        "OBSERVABLE_INCLUDE(0) X0 rec[-1]\nM 0",
    ],
)
def test_not_in_deferred_measurement_form(text: str) -> None:
    """A circuit that breaks one condition of deferred-measurement form is not in the form."""
    assert not deferred_measurement.is_in_deferred_measurement_form(stim.Circuit(text))


def test_memory_experiment() -> None:
    """Rewriting a Clifford memory experiment keeps its detectors, observables, and errors."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.02,
        before_measure_flip_probability=0.03,
        before_round_data_depolarization=0.04,
    )
    circuit += stim.Circuit(
        """
        # a terminal measurement with a readout error
        M(0.05) 0
        DETECTOR rec[-1]
        # an error that flips an observable of the measured qubit and its record
        X_ERROR(0.1) 0
        OBSERVABLE_INCLUDE(1) Z0 rec[-1]
        """
    )
    rewritten = deferred_measurement.to_deferred_measurement_form(circuit)
    assert deferred_measurement.is_in_deferred_measurement_form(rewritten)
    _assert_same_errors(circuit, rewritten, keep_measured=False)


def test_to_deferred_measurement_form() -> None:
    """Rewriting keeps the records and errors of resets, noise, and measurements that interact.

    Resets in the X and Y bases act on used qubits, a reset erases one qubit of two-qubit noise,
    noise follows a measurement, and the circuit has inverted measurements of a qubit and of a pair.
    """
    circuit = stim.Circuit(
        """
        H 0
        RX 0
        DEPOLARIZE2(0.1) 1 2
        RY 1
        MY !1
        MY 1
        M 2
        X_ERROR(0.1) 2
        H 3
        MXX !0 3
        """
    )
    rewritten = deferred_measurement.to_deferred_measurement_form(circuit)
    # every record is deterministic without noise
    assert _assert_same_errors(circuit, rewritten, keep_measured=False) == circuit.num_measurements


def test_observables_with_pauli_targets() -> None:
    """An inverted measurement keeps the value of an observable of its record and a Pauli.

    The observable includes both the record of an inverted measurement and a Pauli on the measured
    qubit.  Stim's own sign of the observable is that of the random record in its reference sample,
    which is 1 here, but 0 once the measurement is not inverted, although the observable is the
    same.
    """
    circuit = stim.Circuit(
        "OBSERVABLE_INCLUDE(0) Y0\nZ_ERROR(0.1) 0\nMY !0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    rewritten = deferred_measurement.simplify_measurements(circuit)
    assert _get_errors(rewritten) == _get_errors(circuit)
    assert _get_signs(rewritten) == _get_signs(circuit) == ([], [True])


def test_observables() -> None:
    """Observables keep their indices and tags, including observables without targets."""
    circuit = stim.Circuit(
        "M 0 1\nOBSERVABLE_INCLUDE[a](0) rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(5)"
    )
    rewritten = deferred_measurement.to_deferred_measurement_form(circuit)
    assert rewritten[-3:] == circuit[-3:]


def test_non_finite_arguments() -> None:
    """Infinite and NaN coordinates, which stim cannot parse, are kept."""
    circuit = stim.Circuit("CX sweep[0] 1\nM !0\nMPP !X0*Z1")
    circuit.append("QUBIT_COORDS", [1], [math.inf, math.nan])
    circuit.append("DETECTOR", [stim.target_rec(-1)], [-math.inf], tag="d")
    assert str(deferred_measurement.split_measure_resets(circuit)) == str(circuit)
    rewritten = deferred_measurement.to_deferred_measurement_form(circuit[1:])
    assert str(rewritten[-1]) == "DETECTOR[d](-inf) rec[-1]"


def test_ancilla_indices() -> None:
    """Fresh ancilla qubits come after all qubits of the input circuit.

    Qubit 1 of each circuit has coordinates but nothing acts on it, and no pass may use it as an
    ancilla qubit.
    """
    for rewrite, text in [
        (deferred_measurement.defer_resets, "H 0\nR 0"),
        (deferred_measurement.defer_measurements, "M 0\nH 0"),
        (deferred_measurement.simplify_measurements, "MPP X0"),
        (deferred_measurement.to_deferred_measurement_form, "M 0\nH 0"),
        (deferred_measurement.to_deferred_measurement_form, "MPP X0"),
    ]:
        rewritten = rewrite(stim.Circuit("QUBIT_COORDS(7) 1\n" + text))
        assert rewritten.num_qubits == 3
        assert all(
            target.value != 1 for instruction in rewritten for target in instruction.targets_copy()
        )


def test_too_many_qubits() -> None:
    """Fresh ancilla qubits may not exceed stim's largest qubit index, but may reach it."""
    qubit = 2**24 - 1
    assert (
        deferred_measurement.defer_resets(stim.Circuit(f"H {qubit - 1}\nR {qubit - 1}")).num_qubits
        == 2**24
    )
    with pytest.raises(ValueError, match="largest qubit index"):
        deferred_measurement.defer_resets(stim.Circuit(f"H {qubit}\nR {qubit}"))
    with pytest.raises(ValueError, match="largest qubit index"):
        deferred_measurement.defer_measurements(stim.Circuit(f"M {qubit}\nH {qubit}"))
    with pytest.raises(ValueError, match="largest qubit index"):
        deferred_measurement.simplify_measurements(stim.Circuit(f"MPP X{qubit}"))


def test_annotations() -> None:
    """All passes but split_measure_resets drop TICK and coordinate annotations."""
    circuit = stim.Circuit("QUBIT_COORDS(1) 0\nH 0\nTICK\nSHIFT_COORDS(2)\nQUBIT_COORDS(1) 0\nM 0")
    for rewrite in [
        deferred_measurement.defer_resets,
        deferred_measurement.defer_measurements,
        deferred_measurement.simplify_measurements,
        deferred_measurement.to_deferred_measurement_form,
    ]:
        names = {instruction.name for instruction in rewrite(circuit)}
        assert not names & {"TICK", "QUBIT_COORDS", "SHIFT_COORDS"}
    assert deferred_measurement.split_measure_resets(circuit) == circuit.flattened()


def test_split_measure_resets() -> None:
    """A measure-reset becomes a measurement, with its tag and exact arguments, and a reset."""
    # the tag has escaped characters, and stim would write the argument with fewer digits
    circuit = stim.Circuit("MRX[\\B\\C\\r\\n](0.123456789) 0 !0\nH 0")
    measurement = "MX[\\B\\C\\r\\n](0.123456789)"
    expected = stim.Circuit(f"{measurement} 0\nRX 0\n{measurement} !0\nRX 0\nH 0")
    assert deferred_measurement.split_measure_resets(circuit) == expected


def test_defer_resets() -> None:
    """Noise that a reset erases is removed, and a reset of an untouched qubit needs no ancilla."""
    rewritten = deferred_measurement.defer_resets(
        stim.Circuit("X_ERROR(0.1) 0 1\nZ_ERROR(0.1) 0\nRY 0")
    )
    assert rewritten.num_qubits == 2
    noisy = [
        (instruction.name, instruction.targets_copy())
        for instruction in rewritten
        if stim.gate_data(instruction.name).is_noisy_gate
    ]
    assert noisy == [("X_ERROR", [stim.GateTarget(1)])]


def test_defer_measurements() -> None:
    """Only measurements whose qubits are used later are copied onto ancilla qubits."""
    # the ancilla qubit of a copied measurement is measured with its inversion, error, and tag
    rewritten = deferred_measurement.defer_measurements(stim.Circuit("M[t](0.1) !0\nH 0"))
    assert rewritten[-1] == stim.CircuitInstruction("M", [stim.target_inv(1)], [0.1], tag="t")

    # a measurement whose qubits are not used later moves to the end unchanged
    circuit = stim.Circuit("MZZ[t](0.1) 0 !1\nH 2")
    assert deferred_measurement.defer_measurements(circuit)[-1] == circuit[0]

    with pytest.raises(ValueError, match="without resets"):
        deferred_measurement.defer_measurements(stim.Circuit("MR 0"))


def test_simplify_measurements() -> None:
    """A readout error becomes an X error before a noiseless M measurement, which keeps the tag."""
    rewritten = deferred_measurement.simplify_measurements(stim.Circuit("MPP[t](0.1) !X0"))
    assert rewritten[-1] == stim.CircuitInstruction("M", [1], tag="t")
    assert stim.CircuitInstruction("X_ERROR", [1], [0.1]) in list(rewritten)

    # a measurement is not terminal if its qubit is used later, including by an observable, or if it
    # is a measure-reset
    for text in ["M 0\nH 0", "MX 0\nOBSERVABLE_INCLUDE(0) X0", "MR 0"]:
        with pytest.raises(ValueError, match="terminal"):
            deferred_measurement.simplify_measurements(stim.Circuit(text))


@pytest.mark.parametrize(
    "text, message",
    [
        ("MPAD 0", "not supported"),
        ("M 0\nCX rec[-1] 1", "Classically controlled"),
        ("MPP X0*Z0", "repeat qubits"),
    ],
)
def test_unsupported_circuits(text: str, message: str) -> None:
    """Unsupported instructions and measurements of Pauli products that repeat a qubit raise."""
    with pytest.raises(ValueError, match=message):
        deferred_measurement.to_deferred_measurement_form(stim.Circuit(text))
