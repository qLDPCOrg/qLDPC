"""Circuit construction utilities for quantum error-corrected memory experiments

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

from collections.abc import Collection, Sequence

import numpy as np
import stim

from qldpc import codes
from qldpc.objects import Node, Pauli, PauliXZ

from .bookkeeping import DetectorRecord, MeasurementRecord, MemoryExperimentParts, QubitIDs
from .common import get_encoding_circuit, restrict_to_qubits, with_remapped_qubits
from .noise_model import DEFAULT_IMMUNE_OP_TAG, NoiseModel, as_noiseless_circuit
from .syndrome_measurement import EdgeColoring, SyndromeMeasurementStrategy


def get_memory_experiment(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZ | None = Pauli.X,
    num_rounds: int = 1,
    *,
    noise_model: NoiseModel | None = None,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> stim.Circuit:
    f"""Construct a circuit for testing the performance of a code as a quantum memory.

    In a nutshell, the circuit constructed by this method performs (generally multiple) rounds
    quantum error correction (QEC) for the given code.  Each round of QEC measures all parity checks
    of the code, and detectors are added to enforce that
    (a) the syndrome from the first round of QEC is trivial, and
    (b) every subsequent round of QEC yields the same syndrome as the preceding round.
    We refer to num_rounds rounds of syndrome measurement as one logical QEC cycle.

    If the basis is Pauli.X or Pauli.Z, then the memory experiment only tracks errors in the logical
    Pauli operators of that type.  If basis is None, the circuit entangles the code with a noiseless
    ancilla to track errors in all logical Pauli operators.

    More specifically, if basis is Pauli.X or Pauli.Z then the memory experiment performs the
    following:
    1. Initialize all data qubits to a +1 eigenstate of the specified basis: |0> for Z, |+> for X.
    2. Perform an initial round of QEC, adding detectors for the basis-type stabilizers.
    3. Perform num_rounds - 1 additional QEC rounds, adding detectors to enforce that basis-type
        stabilizers have not changed between adjacent rounds of QEC.
    4. Measure all data qubits in the specified basis.
    5. Add detectors for all stabilizers that can be inferred from the data qubit measurements.
    If a noise_model is provided, then noise is added to all parts of the circuit.

    If basis is None, then the memory experiment noiselesly initializes each logical qubit of the
    code in a maximally entangled state with an (unphysical) noiseless ancilla qubit before running
    a noisy logical QEC cycle.  This initialization makes it possible to meaningfully track errors
    in both X-type and Z-type logical operators of a code.  The probability of an error in any
    logical operator is then essentially the process infidelity (or entanglement infidelity) of the
    logical QEC cycle.

    More specifically, if basis is None then the memory experiment performs the following:
    1. Prepare a logical all-|0> state of the code.
    2. For each logical qubit of the code, prepare an ancilla qubit in |+>, and apply an
        ancilla-controlled-logical-NOT gate to the logical qubit, thereby preparing Bell states
        |00> + |11> of logical qubits with their respective ancillas.
    3. Perform a logical QEC cycle as before, but now adding detectors for all stabilizers.
    4. Measure all stabilizers (with MPP gates).
    Unlike the fixed-basis experiment, the combined basis experiment only makes sense when starting
    from the Bell state.  It is also no longer possible to measure out all data qubits to infer all
    stabilizers.  Initialization and readout (measuring final stabilizers) are therefore noiseless.
    If a noise_model is provided, then noise is added to the logical QEC cycle alone.  Otherwise,
    the initialization and readout sub-circuits are wrapped in a single-repetition
    stim.CircuitRepeatBlock tagged with "{DEFAULT_IMMUNE_OP_TAG}" to indicate that these
    sub-circuits should be immune to noise.

    Remembering that observables in Stim are formally detectors, or circuit-level parity checks that
    must evaluate to 0 in the absence of errors, the preparation of Bell pairs allows us to annotate
    XX and ZZ observables for each Bell pair.  Here one of the "X"s in XX is a logical X for a
    logical qubit of the code, and the other "X" is a physical X on an associated ancilla qubit;
    likewise with ZZ.  Since the ancilla qubit is noiseless, we can attribute an error in XX or ZZ to
    a logical qubit error.

    Having said all of that, we do not actually annotate memory simulation circuits with the XX and
    ZZ observables described above.  Instead, we recognize that Bell-pair XX and ZZ operators are
    exact stabilizers of the circuit immediately after noiseless initialization, which allows us to
    freely multiply the XX and ZZ operators at the end of the circuit by XX and ZZ operators before
    the logical QEC cycle, thereby obtaining two-time XXXX and ZZZZ observables.  The chief (albeit
    perhaps aesthetic) benefit to this trick is that the support of these observables on the
    (noiseless) ancilla qubits cancels out, leaving us with two-time logical XX and ZZ observables
    supported on the data qubits alone.

    Qubits and detectors are assigned coordinates as follows:
    - The data qubit addressed by column C of the parity check matrix gets coordinate (0, C).
    - The check qubit associated with row R of the parity check matrix gets coordinate (1, R).
    - The ancilla qubit associated with logical qubit L of the code gets coordinate (2, L).
    - The K-th detector in measurement round M gets coordinate (M, 0, K).

    Args:
        code: An error-correcting code.  Must be a qubit stabilizer (non-subsystem) codes.  If
            passed a classical code, treat it as a quantum CSS code that protects only basis-type
            logical operators (or X-type logicals, if basis is None).
        basis: Should be Pauli.X, Pauli.Z, or None to indicate which type of logical operators to
            track (where "None" means "both X and Z").  Default: Pauli.X.
        num_rounds: The number of syndome measurement rounds to perform in one logical QEC cycle.
            Must be at least 1.  Default: 1.
        noise_model: The noise model to apply to the circuit after construction, or None to return a
            noiseless circuit.  Default: None.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.  Defaults to
            labeling data and check qubits according to their correspnding column/row of the parity
            check matrix, with data qubits numbered from 0 and check qubits numbered from len(code).
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each
            round of QEC measures the parity checks of the code.  Default: circuits.EdgeColoring().

    Returns:
        stim.Circuit: A circuit ready for simulation via Stim or Sinter.

    Example:
        from qldpc import circuits, codes
        from qldpc.objects import Pauli

        # Create a 3-qubit repetition code
        rep_code = codes.RepetitionCode(3)

        # Generate 5-round Z-basis memory experiment with depolarizing noise
        noise_model = circuits.DepolarizingNoiseModel(1e-2)
        circuit = circuits.get_memory_experiment(
            rep_code,
            basis=Pauli.Z,
            num_rounds=5,
            noise_model=noise_model,
        )

        # The circuit is ready for simulation!
        # We can now sample detector and observable flips.
        sampler = circuit.compile_detector_sampler()
        detectors, observables = sampler.sample(shots=1000, separate_observables=True)
    """
    initialization, qec_cycle, readout, *_ = get_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
    )

    # add noise to all parts of an experiment with a fixed basis
    if basis is not None and noise_model is not None:
        initialization = noise_model.noisy_circuit(initialization)
        qec_cycle = noise_model.noisy_circuit(qec_cycle)
        readout = noise_model.noisy_circuit(readout)

    # if tracking all logical operators, only the logical QEC cycle is noisy
    if basis is None:
        if noise_model is not None:
            qec_cycle = noise_model.noisy_circuit(qec_cycle)
        else:
            # noise will be added later, so make initialization and readout noiseless
            initialization = as_noiseless_circuit(initialization)
            readout = as_noiseless_circuit(readout)

    return initialization + qec_cycle + readout


@restrict_to_qubits
def get_memory_experiment_parts(
    code: codes.QuditCode | codes.ClassicalCode,
    basis: PauliXZ | None,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> MemoryExperimentParts:
    """Noiseless components of a memory experiment.

    See help(qldpc.circuits.get_memory_experiment) for additional information.

    Returns:
        initialization: A circuit that sets the coordinates and initializes the state of data qubits.
        qec_cycle: A circuit for one logical QEC cycle, with num_rounds syndrome measurements.
        readout: A circuit that reads out final stabilizers.
        measurement_record: A record of all measurements in the above circuits.
        detector_record: A record of all detectors in the above circuits.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.
    """
    if isinstance(code, codes.ClassicalCode):
        matrix_z = code.matrix if basis is Pauli.Z else code.field.Zeros((0, len(code)))
        matrix_x = code.field.Zeros((0, len(code))) if basis is Pauli.Z else code.matrix
        code = codes.CSSCode(matrix_x, matrix_z)

    if code.is_subsystem_code:
        raise ValueError(
            "Memory simulations currently only support stabilizer (non-subsystem) codes"
        )

    if basis is None:
        return _get_combined_memory_simulation_parts(
            code,
            num_rounds=num_rounds,
            qubit_ids=qubit_ids,
            syndrome_measurement_strategy=syndrome_measurement_strategy,
        )
    return _get_basis_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
    )


def _get_basis_memory_experiment_parts(
    code: codes.QuditCode,
    basis: PauliXZ,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> MemoryExperimentParts:
    """Components of a memory experiment that tracks logical operators of a fixed type (basis).

    See help(qldpc.circuits.get_memory_experiment) for additional information.
    """
    if basis is not Pauli.X and basis is not Pauli.Z:
        raise ValueError(
            "Memory experiments in a fixed basis require the basis to be Pauli.X or Pauli.Z,"
            f" not {basis}"
        )
    if not isinstance(code, codes.CSSCode):
        raise ValueError("Memory experiments in a fixed basis only support CSS codes")

    # identify all qubits by index
    qubit_ids = QubitIDs.validated(qubit_ids, code) if qubit_ids else QubitIDs.from_code(code)
    data_ids, check_ids, _ = qubit_ids
    basis_check_ids = qubit_ids.checks_x if basis is Pauli.X else qubit_ids.checks_z

    # set qubit coordinates
    coordinates = get_qubit_coordinates(data_ids, check_ids)

    # reset data qubits to the appropriate basis
    state_prep = stim.Circuit()
    state_prep.append(f"R{basis}", data_ids)

    # build a logical QEC cycle
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        code, num_rounds, qubit_ids, basis_check_ids, syndrome_measurement_strategy
    )

    # measure out the data qubits
    readout = stim.Circuit()
    readout.append(f"M{basis}", data_ids)
    measurement_record.append({data_id: [mm] for mm, data_id in enumerate(data_ids)})

    # detectors for stabilizers that can be inferred from data qubit measurements
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    check_support = code.get_matrix(basis)
    for kk, check_id in enumerate(basis_check_ids):
        data_support = np.where(check_support[kk])[0]
        readout.append(
            "DETECTOR",
            [measurement_record.get_target_rec(data_ids[qq]) for qq in data_support]
            + [measurement_record.get_target_rec(check_id)],
            (0, 0, kk),
        )
    detector_record.append({check_id: [dd] for dd, check_id in enumerate(basis_check_ids)})

    # annotate all basis-type observables
    targets = [measurement_record.get_target_rec(data_id) for data_id in data_ids]
    observables = get_observables(code, data_ids, basis=basis, on_measurements=targets)

    return MemoryExperimentParts(
        coordinates + state_prep,
        qec_cycle,
        readout + observables,
        measurement_record,
        detector_record,
        qubit_ids,
    )


def _get_combined_memory_simulation_parts(
    code: codes.QuditCode,
    num_rounds: int = 1,
    *,
    qubit_ids: QubitIDs | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> MemoryExperimentParts:
    """Components of a memory experiment that tracks all logical operators.

    See help(qldpc.circuits.get_memory_experiment) for additional information.
    """
    # identify all qubits by index
    qubit_ids = QubitIDs.validated(qubit_ids, code) if qubit_ids else QubitIDs.from_code(code)
    qubit_ids.add_ancillas(code.dimension - len(qubit_ids.ancilla))
    data_ids, check_ids, ancilla_ids = qubit_ids
    ancilla_ids = ancilla_ids[: code.dimension]

    # set qubit coordinates
    coordinates = get_qubit_coordinates(data_ids, check_ids, ancilla_ids)

    # noiselessly prepare all logical qubits in Bell states with ancillas
    state_prep = get_logical_bell_prep(code, data_ids, ancilla_ids)

    # build a logical QEC cycle
    qec_cycle, measurement_record, detector_record = _get_qec_cycle(
        code, num_rounds, qubit_ids, check_ids, syndrome_measurement_strategy
    )

    # measure all stabilizers
    readout = stim.Circuit()
    for check_index, check_id in enumerate(check_ids):
        check_node = Node(check_index, is_data=False)
        targets = [
            f"{edge_data[Pauli]}{data_ids[data_node.index]}"
            for _, data_node, edge_data in code.graph.edges(check_node, data=True)
        ]
        joined_targets = "*".join(targets)
        readout.append(stim.CircuitInstruction(f"MPP {joined_targets}"))

    # update the measurement record, add detectors, and update the detector record
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    measurement_record.append({check_id: [mm] for mm, check_id in enumerate(check_ids)})
    for kk, check_id in enumerate(check_ids):
        targets = [
            measurement_record.get_target_rec(check_id, -1),
            measurement_record.get_target_rec(check_id, -2),
        ]
        readout.append("DETECTOR", targets, (0, 0, kk))
    detector_record.append({check_id: [dd] for dd, check_id in enumerate(check_ids)})

    # annotate all observables
    observables = get_observables(code, data_ids)

    return MemoryExperimentParts(
        coordinates + state_prep + observables,
        qec_cycle,
        readout + observables,
        measurement_record,
        detector_record,
        qubit_ids,
    )


def get_qubit_coordinates(
    data_qubits: Sequence[int] = (),
    check_qubits: Sequence[int] = (),
    ancilla_qubits: Sequence[int] = (),
) -> stim.Circuit:
    """Circuit declaring coordinates for the given qubits.

    Args:
        data_qubits: The indices of data qubits.  data_qubits[kk] gets index (0, kk).
        check_qubits: The indices of check qubits.  check_qubits[kk] gets index (1, kk).
        ancilla_qubits: The indices of ancilla qubits.  ancilla_qubits[kk] gets index (2, kk).

    Returns:
        A circuit declaring qubit coordinates.
    """
    circuit = stim.Circuit()
    for kk, qubit in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", qubit, (0, kk))
    for kk, qubit in enumerate(check_qubits):
        circuit.append("QUBIT_COORDS", qubit, (1, kk))
    for kk, qubit in enumerate(ancilla_qubits):
        circuit.append("QUBIT_COORDS", qubit, (2, kk))
    return circuit


@restrict_to_qubits
def get_observables(
    code: codes.QuditCode,
    data_qubits: Sequence[int] | None = None,
    *,
    basis: PauliXZ | None = None,
    on_measurements: Sequence[stim.target_rec] | bool = False,
    observable_indices: Sequence[int] | None = None,
) -> stim.Circuit:
    """Construct a circuit of logical observable annotations.

    Args:
        code: The code whose observables we wish to annotate.
        data_qubits: Indices of the data qubits of the code.  Default: the first len(code) integers.
        basis: The type of observable (Pauli.X or Pauli.Z) we wish to annotate, or None for both.
        on_measurements: If provided a sequence of measurement targets, assume that they correspond
            to measurements of the data qubits in a specified basis (which in this case is not
            allowed to be None), and define observables using these measurements.  If True, define
            observables identically on the last len(code) measurements.  Otherwise (if False),
            define observable using Pauli targets.  Default: False.
        observable_indices: Indices to use for the observables.  Default: range(num_observables).

    Returns:
        A Stim circuit of OBSERVABLE_INCLUDE instructions.
    """
    if basis not in (None, Pauli.X, Pauli.Z):  # pragma: no cover
        raise ValueError(
            f"Provided basis must be Pauli.X or Pauli.Z (from qldpc.objects) or None, not {basis}"
        )

    data_qubits = data_qubits or range(len(code))
    num_observables = code.dimension * (2 if basis is None else 1)
    observable_indices = observable_indices or range(num_observables)

    if on_measurements is True:
        on_measurements = [stim.target_rec(mm) for mm in range(-len(code), 0)]

    # consistency checks
    assert on_measurements is False or len(on_measurements) == len(code)
    assert len(observable_indices) == num_observables

    # build a graph of edges directed from observables to the data qubits they address
    logical_ops = code.get_logical_ops(basis, symplectic=True)
    logical_op_graph = codes.QuditCode.matrix_to_graph(logical_ops)

    if on_measurements and (not isinstance(code, codes.CSSCode) or basis is None):
        raise ValueError(
            "Defining observables on measurements is only allowed (a) for CSS codes (b) with a"
            " fixed measurement basis (Pauli.X or Pauli.Z)"
        )

    circuit = stim.Circuit()
    for node_index, observable_index in enumerate(observable_indices):
        observable_node = Node(node_index, is_data=False)
        targets = [
            on_measurements[data_node.index]
            if on_measurements
            else stim.target_pauli(data_qubits[data_node.index], str(edge_data[Pauli]))
            for _, data_node, edge_data in logical_op_graph.edges(observable_node, data=True)
        ]
        circuit.append("OBSERVABLE_INCLUDE", targets, [observable_index])

    return circuit


@restrict_to_qubits
def get_logical_bell_prep(
    code: codes.QuditCode,
    data_qubits: Sequence[int] | None = None,
    ancilla_qubits: Sequence[int] | None = None,
) -> stim.Circuit:
    """Noiselessly prepare the logical qubits of the given code in Bell states with ancillas.

    Args:
        code: The code for which we are constructing a logical Bell-state preparation circuit.
        data_qubits: Indices of the code's data qubits.  Default: the first len(code) integers.
        ancilla_qubits: Indices of the ancilla qubits to entangle with the code's logical qubits.
            Default: the first code.dimension integers after the data qubit indices.

    Returns:
        A circuit that noiselessly initializes all logical qubits into Bell pairs with ancillas.
    """
    data_qubits = data_qubits or range(len(code))
    ancilla_qubits = ancilla_qubits or range(
        data_qubits[-1] + 1, data_qubits[-1] + 1 + code.dimension
    )
    assert len(data_qubits) == len(code)
    assert len(ancilla_qubits) == code.dimension

    # entangle the first code.dimension data qubits with ancillas
    circuit = stim.Circuit()
    circuit.append("H", data_qubits[: code.dimension])
    circuit.append("CX", [qq for pair in zip(data_qubits, ancilla_qubits) for qq in pair])

    # encode the first code.dimension data qubits
    circuit.append(with_remapped_qubits(get_encoding_circuit(code), data_qubits))

    return as_noiseless_circuit(circuit)


def _get_qec_cycle(
    code: codes.QuditCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
    check_ids: Collection[int],
    syndrome_measurement_strategy: SyndromeMeasurementStrategy,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Build a circuit of num_rounds noiseless syndrome measurements for a given code.

    Args:
        code: The code for which we are building a logical QEC cycle.
        num_rounds: The number of syndrome measurement rounds in one logical QEC cycle.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.
        check_ids: The check qubits that measure stabilizers to annotate with detectors.  Must be a
            subset of qubit_ids.check (though this requirement is not verified).
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each
            round of QEC measures the parity checks of the code.

    Returns:
        stim.Circuit: The noiseless circuit of num_rounds syndorme measurements.
        MeasurementRecord: The record of all measurements in the constructed circuit.
        DetectorRecord: The record of all detectors in the constructed circuit.
    """
    one_round, round_measurement_record = syndrome_measurement_strategy.get_circuit(code, qubit_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # apply first round of QEC and detectors
    circuit.append(one_round)
    measurement_record.append(round_measurement_record)
    for kk, check_id in enumerate(check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))
    detector_record.append({check_id: [dd] for dd, check_id in enumerate(check_ids)})

    # apply following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, check_id in enumerate(check_ids):
            targets = [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ]
            repeat_circuit.append("DETECTOR", targets, (0, 0, kk))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # update the measurement and detector records to account for repetitions
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {check_id: [dd] for dd, check_id in enumerate(check_ids)}, repeat=num_rounds - 1
        )

    return circuit, measurement_record, detector_record
