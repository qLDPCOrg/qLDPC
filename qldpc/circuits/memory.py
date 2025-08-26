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

import itertools
from collections.abc import Collection

import numpy as np
import stim

from qldpc import codes
from qldpc.objects import PAULIS_XZ, Node, Pauli, PauliXZ

from .common import QubitIDs, get_encoding_circuit, restrict_to_qubits
from .noise_model import NoiseModel
from .syndrome_measurement import (
    EdgeColoring,
    MeasurementRecord,
    SyndromeMeasurementStrategy,
)


@restrict_to_qubits
def get_memory_experiment(
    code: codes.AbstractCode,
    basis: PauliXZ = Pauli.X,
    num_rounds: int = 1,
    *,
    noise_model: NoiseModel | None = None,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> stim.Circuit:
    """Construct a circuit for testing the performance of a code as a quantum memory.

    In a nutshell, the circuit constructed by this method performs (generally multiple) rounds
    quantum error correction (QEC) for the given code.  Each QEC round, or cycle, measures all parity
    checks of the code, and detectors are added to enforce that
    (a) the syndrome from the first QEC cycle is trivial, and
    (b) every subsequent QEC cycle yields the same syndrome as the preceding round.
    The "basis" argument determines whether the circuit tracks logical X or Z operators.

    More specifically, the circuit performs the following:
    1. Initialize all data qubits to |0> (if basis is Pauli.Z) or |+> (if basis is Pauli.X).
    2. Perform an initial QEC cycle, adding detectors for the basis-type stabilizers.
    3. Perform num_rounds - 1 additional QEC cycles, adding detectors to enforce that basis-type
        stabilizers have not changed between adjacent QEC cycles.
    4. Measure all data qubits in the specified basis.
    5. Add detectors for all stabilizers that can be inferred from the data qubit measurements.
    6. Use the final data qubit measurements to define all basis-type logical observables.

    Qubits and detectors are assigned coordinates as follows:
    - The data qubit addressed by column c of the parity check matrix gets coordinate (0, c).
    - The check qubit associated with row r of the parity check matrix gets coordinate (1, r).
    - The k-th detector in measurement round m gets coordinate (m, 0, k).

    Args:
        code: An error-correcting code.  If passed a classical code, treat it as a quantum CSS code
            that protects only basis-type logical operators.  Otherwise, only CSS stabilizer
            (non-subsystem) qubit codes are supported at the moment (generalization to non-CSS and
            subsystem codes pending).
        basis: Should be Pauli.X or Pauli.Z, depending the desired logical operators to track.  A
            logical error in a noisy simulation of the circuit corresponds to a logical error in one
            of these operators.  Default: Pauli.X.
        num_rounds: Total number of QEC cycles to perform.  Must be at least 1.  Default: 1.
        noise_model: The noise model to apply to the circuit after construction, or None to return a
            noiseless circuit.  Default: None.
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
    initialization, qec_cycles_and_readout, *_ = get_memory_experiment_parts(
        code,
        basis=basis,
        num_rounds=num_rounds,
        syndrome_measurement_strategy=syndrome_measurement_strategy,
    )
    circuit = initialization + qec_cycles_and_readout
    return noise_model.noisy_circuit(circuit) if noise_model is not None else circuit


@restrict_to_qubits
def get_memory_experiment_parts(
    code: codes.AbstractCode,
    basis: PauliXZ = Pauli.X,
    num_rounds: int = 1,
    *,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> tuple[stim.Circuit, stim.Circuit, MeasurementRecord]:
    """Noiseless components of a memory experiment.

    See help(qldpc.circuits.get_memory_experiment) for additional information.

    Returns:
        initialization: A circuit that sets all qubit coordinates and resets data qubits to the
            appropriate basis.
        qec_cycles_and_readout: A circuit of num_rounds QEC cycles followed by data qubit
            measurements in the specified basis.  Includes detectors for basis-type stabilizers and
            declares basis-type logical observables.
        measurement_record: A record of the measurements in qec_cycles_and_readout.
    """
    if basis is not Pauli.X and basis is not Pauli.Z:
        raise ValueError(
            "Memory experiments require choosing a Pauli.X or Pauli.Z basis of logical operators to"
            f" track (provided: {basis})"
        )
    if isinstance(code, codes.ClassicalCode):
        matrix_x = code.matrix if basis is Pauli.X else code.field.Zeros((0, len(code)))
        matrix_z = code.field.Zeros((0, len(code))) if basis is Pauli.X else code.matrix
        code = codes.CSSCode(matrix_x, matrix_z)
    if not isinstance(code, codes.CSSCode) or code.is_subsystem_code:
        raise ValueError(
            "Memory experiments currently only support stabilizer (non-subsystem) CSS codes"
        )

    # identify all qubits by index
    qubit_ids = QubitIDs.from_code(code)
    data_ids, check_ids, *_ = qubit_ids

    # identify the indices of ancilla qubits that read out basis-type parity checks
    check_support = code.get_matrix(basis)
    basis_check_ids = (
        check_ids[: code.num_checks_x] if basis is Pauli.X else check_ids[code.num_checks_x :]
    )

    ####################
    # INITIALIZATION
    ####################

    # set coordinates for all qubits
    initialization = stim.Circuit()
    for kk, data_id in enumerate(qubit_ids.data):
        initialization.append("QUBIT_COORDS", data_id, (0, kk))
    for kk, check_id in enumerate(qubit_ids.check):
        initialization.append("QUBIT_COORDS", check_id, (1, kk))

    # reset data qubits to appropriate basis
    initialization.append(f"R{basis}", data_ids)

    ####################
    # QEC CYCLES
    ####################

    one_cycle, cycle_measurement_record = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
    all_cycles, measurement_record = _get_qec_cycles(
        one_cycle, cycle_measurement_record, num_rounds, qubit_ids.check
    )

    ####################
    # DATA QUBIT READOUT
    ####################

    # measure out the data qubits
    readout = stim.Circuit()
    readout.append(f"M{basis}", data_ids)
    measurement_record.append({qubit: [qubit] for qubit in range(len(code))})

    # detectors for all stabilizers that can be inferred from the data qubit measurements
    for kk, check_id in enumerate(basis_check_ids):
        data_support = np.where(check_support[kk])[0]
        readout.append(
            "DETECTOR",
            [measurement_record.get_target_rec(qq) for qq in data_support]
            + [measurement_record.get_target_rec(check_id)],
            (num_rounds, 0, kk),
        )

    # add all basis-type observables
    for kk, observable in enumerate(code.get_logical_ops(basis)):
        data_support = np.where(observable)[0]
        readout.append(
            "OBSERVABLE_INCLUDE",
            [measurement_record.get_target_rec(qq) for qq in data_support],
            kk,
        )

    return initialization, all_cycles + readout, measurement_record


@restrict_to_qubits
def get_memory_simulation(
    code: codes.QuditCode,
    noise_model: NoiseModel,
    num_rounds: int = 1,
    *,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> stim.Circuit:
    """Construct a circuit for testing the performance of a code as a quantum memory.

    This method constructs a circuit similar to that in qldpc.circuits.get_memory_experiment.
    However, the circuit constructed here noiselessly initializes each logical qubit of the code in
    a maximally entangled state with an (unphysical) noiseless ancilla qubit before running noisy
    QEC cycles.  This initialization makes it possible to meaningfully track errors in both X-type
    and Z-type logical operators of a code.  The probability of an error in any logical operator is
    then essentially the process infidelity (or entanglement infidelity) of the noisy QEC cycles.

    See help(qldpc.circuits.get_memory_experiment) for background and context.

    The circuit constructed by this method performs the following:
    1. Noiselessly prepare a logical all-|0> state of the code.
    2. For each logical qubit of the code, noiselessly prepare an ancilla qubit in |+>, and apply an
        ancilla-controlled-logical-NOT gate to the logical qubit, thereby preparing Bell states
        |00> + |11> of logical qubits with their respective ancillas.
    3. Perform num_rounds noisy QEC cycles, identically to qldpc.circuits.get_memory_experiment.

    Remembering that logical operators in Stim are formally detectors, or circuit-level parity
    checks that should evaluate to 0 in the absence of errors, the preparation of Bell pairs allows
    us to annotate XX and ZZ observables for each Bell pair.  Here one of the "X"s in XX is a
    logical X for one of the logical qubit of the code, and the other "X" is a physical X on the
    associated ancilla qubit; likewise with ZZ.  Since the ancilla qubit is noiseless, we can
    attribute an error in XX or ZZ to a logical qubit error.

    Having said all of that, we do not actually annotate the circuit with these XX and ZZ
    observables.  Instead, we recognize that Bell-pair XX and ZZ operators are stabilizers of the
    circuit immediately after noiseless initialization, which allows us to freely multiply the XX and
    ZZ operators at the end of the circuit by XX and ZZ operators before the QEC cycles, thereby
    obtaining two-time XXXX and ZZZZ observables.  The chief benefit to this trick is that the
    support of these observables on the ancilla qubits cancels out, leaving us with two-time logical
    XX and ZZ observables that are supported entirely on the data qubits of the code.

    Args:
        code: A quantum error-correcting code.  Only stabilizer (non-subsystem) codes are supported.
        noise_model: The noise model to apply to the the QEC cycles of the circuit.
        num_rounds: Total number of QEC cycles to perform.  Must be at least 1.  Default: 1.
        syndrome_measurement_strategy: The syndrome measurement strategy that defines how each
            round of QEC measures the parity checks of the code.  Default: circuits.EdgeColoring().

    Returns:
        stim.Circuit: A circuit ready for simulation via Stim or Sinter.
    """
    initialization, qec_cycles, *_ = get_memory_simulation_parts(
        code, num_rounds=num_rounds, syndrome_measurement_strategy=syndrome_measurement_strategy
    )
    return initialization + noise_model.noisy_circuit(qec_cycles)


@restrict_to_qubits
def get_memory_simulation_parts(
    code: codes.QuditCode,
    num_rounds: int = 1,
    *,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy = EdgeColoring(),
) -> tuple[stim.Circuit, stim.Circuit, MeasurementRecord]:
    """Noiseless components of a memory simulation.

    See help(qldpc.circuits.get_memory_simulation) for additional information.

    Returns:
        initialization: A circuit that sets all qubit coordinates and initializes every logical
            qubit into a Bell pair with its associated ancilla.
        qec_cycles: A circuit of num_rounds QEC cycles.  Includes detectors for all stabilizers and
            declares all logical observables.
        measurement_record: A record of the measurements in qec_cycles.
    """
    if code.is_subsystem_code:
        raise ValueError(
            "Memory simulations currently only support stabilizer (non-subsystem) codes"
        )

    # identify all qubits by index
    qubit_ids = QubitIDs.from_code(code)
    data_ids, check_ids, *_ = qubit_ids
    qubit_ids.add_ancilla(code.dimension)

    # identify logical operators
    kwargs = dict(symplectic=True) if isinstance(code, codes.CSSCode) else {}
    logical_op_matrix = {pauli: code.get_logical_ops(pauli, **kwargs) for pauli in PAULIS_XZ}
    logical_op_graph = {
        pauli: codes.QuditCode.matrix_to_graph(matrix)
        for pauli, matrix in logical_op_matrix.items()
    }

    # identify observables
    observables = stim.Circuit()
    for op_index, (pauli, logical_qubit_index) in enumerate(
        itertools.product(PAULIS_XZ, range(code.dimension))
    ):
        ancilla_node = Node(logical_qubit_index, is_data=False)
        qubit_paulis = [
            stim.target_pauli(data_node.index, str(edge_data[Pauli]))
            for _, data_node, edge_data in logical_op_graph[pauli].edges(ancilla_node, data=True)
        ]
        observables.append("OBSERVABLE_INCLUDE", qubit_paulis, [op_index])

    ####################
    # INITIALIZATION
    ####################

    # set coordinates for all qubits
    initialization = stim.Circuit()
    for kk, data_id in enumerate(qubit_ids.data):
        initialization.append("QUBIT_COORDS", data_id, (0, kk))
    for kk, check_id in enumerate(qubit_ids.check):
        initialization.append("QUBIT_COORDS", check_id, (1, kk))
    for kk, ancilla_id in enumerate(qubit_ids.ancilla):
        initialization.append("QUBIT_COORDS", ancilla_id, (2, kk))

    # initialize a logical all-|0> state of the code, and intialize ancilla qubits in |+>
    initialization.append(get_encoding_circuit(code))
    initialization.append("H", qubit_ids.ancilla)

    # apply ancilla-controlled-logical-NOT gates to prepare Bell states
    for logical_qubit_index, ancilla_id in enumerate(qubit_ids.ancilla):
        ancilla_node = Node(logical_qubit_index, is_data=False)
        for _, data_node, edge_data in logical_op_graph[Pauli.X].edges(ancilla_node, data=True):
            initialization.append(f"C{edge_data[Pauli]}", [ancilla_id, data_node.index])

    ####################
    # QEC CYCLES
    ####################

    # annotate initial observables
    qec_cycles = observables.copy()

    # add QEC cycles
    one_cycle, cycle_measurement_record = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
    all_cycles, measurement_record = _get_qec_cycles(
        one_cycle, cycle_measurement_record, num_rounds, qubit_ids.check
    )
    qec_cycles.append(all_cycles)

    # annotate final observables
    qec_cycles.append(observables)

    return initialization, qec_cycles, measurement_record


def _get_qec_cycles(
    one_cycle: stim.Circuit,
    cycle_measurement_record: MeasurementRecord,
    num_rounds: int,
    check_ids: Collection[int],
) -> tuple[stim.Circuit, MeasurementRecord]:
    """Given a circuit for a single QEC cycle, return the circuit for many QEC cycles.

    Args:
        one_cycle: The circuit for one QEC cycle (one round of parity check measurements).
        cycle_measurement_record: The measurement record for one_cycle.
        num_rounds: The number of QEC cycles in the final circuit.
        check_ids: The indices of check qubits for which to add detectors.
    """
    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()

    # apply first round of QEC and detectors
    circuit.append(one_cycle)
    measurement_record.append(cycle_measurement_record)
    for kk, check_id in enumerate(check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))

    # apply following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = one_cycle.copy()
        measurement_record.append(cycle_measurement_record)
        for kk, check_id in enumerate(check_ids):
            targets = [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ]
            repeat_circuit.append("DETECTOR", targets, (1, 0, kk))
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # make the measurement_record account for repeated measurements
        measurement_record.append(cycle_measurement_record, repeat=num_rounds - 2)

    return circuit, measurement_record
