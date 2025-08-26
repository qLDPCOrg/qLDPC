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
    syndrome_measurement_strategy: SyndromeMeasurementStrategy | None = None,
) -> stim.Circuit:
    """Construct a circuit for testing the performance of a code as a quantum memory.

    The circuit consists of (generally multiple) quantum error correction (QEC) cycles for the code,
    using a particular syndrome measurement strategy.  Each QEC cycle measures all parity checks of
    the code, and detectors are added to enforce that (a) the syndrome from the first QEC cycle is
    trivial, and (b) every subsequent QEC cycle yields the same syndrome as the preceding round.
    The "basis" argument determines whether the circuit tracks logical X or Z operators.

    In total, the circuit performs the following:
    1. Initialize all data qubits to |0> (if basis is Pauli.Z) or |+> (if basis is Pauli.X).
    2. Perform an initial QEC cycle, adding detectors for the basis-type stabilizers.
    3. Repeat num_rounds - 1 QEC cycles, adding detectors to enforce that basis-type stabilizers
        have not changed between adjacent QEC cycles.
    4. Measure all data qubits in the specified basis.
    5. Add detectors for basis-type parity checks on the final data qubit measurements.
    6. Use the final data qubit measurements to define all basis-type logical observables.

    Qubits and detectors are assigned coordinates as follows:
    - The data qubit addressed by column c of the parity check matrix gets coordinate (0, c).
    - The check qubit associated with row r of the parity check matrix gets coordinate (1, r).
    - The k-th detector in measurement round m gets coordinate (m, k).

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
        syndrome_measurement_strategy: The syndrome measurement strategy to use, which defines how
            each round of QEC measures all parity checks of the code.
            Default: circuits.EdgeColoring().

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
    if basis is not Pauli.X and basis is not Pauli.Z:
        raise ValueError(
            "Memory experiments currently only support tracking logical operators in the X or Z"
            f" basis (provided: {basis})"
        )
    if isinstance(code, codes.ClassicalCode):
        matrix_x = code.matrix if basis is Pauli.X else code.field.Zeros((0, len(code)))
        matrix_z = code.field.Zeros((0, len(code))) if basis is Pauli.X else code.matrix
        code = codes.CSSCode(matrix_x, matrix_z)
    if not isinstance(code, codes.CSSCode):
        raise ValueError("Memory experiments are currently not supported for non-CSS codes")
    if code.is_subsystem_code:
        raise ValueError("Memory experiments are currently not supported for subsystem codes")

    # set default syndrome measurement strategy, if necessary, and identify qubit IDs
    syndrome_measurement_strategy = syndrome_measurement_strategy or EdgeColoring()
    qubit_ids = QubitIDs.from_code(code)
    data_ids, check_ids, *_ = qubit_ids

    # identify the indices of ancilla qubits that read out basis-type parity checks
    check_support = code.get_matrix(basis)
    basis_check_ids = (
        check_ids[: code.num_checks_x] if basis is Pauli.X else check_ids[code.num_checks_x :]
    )

    # build one noiseless QEC cycle and initialize a measurement record
    one_cycle, cycle_measurements = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
    measurement_record = MeasurementRecord()

    # set coordinates for all qubits
    circuit = stim.Circuit()
    for kk, data_id in enumerate(qubit_ids.data):
        circuit.append("QUBIT_COORDS", data_id, (0, kk))
    for kk, check_id in enumerate(qubit_ids.check):
        circuit.append("QUBIT_COORDS", check_id, (1, kk))

    # reset data qubits to appropriate basis
    circuit.append(f"R{basis}", data_ids)

    # first round of QEC and detectors
    circuit.append(one_cycle)
    measurement_record.append(cycle_measurements)
    for kk, check_id in enumerate(basis_check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))

    # following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = one_cycle.copy()
        measurement_record.append(cycle_measurements)
        for kk, check_id in enumerate(basis_check_ids):
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (1, 0, kk),
            )
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # make the measurement_record account for repeated measurements
        for _ in range(num_rounds - 2):
            measurement_record.append(cycle_measurements)

    # measure out the data qubits
    circuit.append(f"M{basis}", data_ids)
    measurement_record.append({qubit: [qubit] for qubit in range(len(code))})

    # detectors for all stabilizers that can be inferred from the data qubit measurements
    for kk, check_id in enumerate(basis_check_ids):
        data_support = np.where(check_support[kk])[0]
        circuit.append(
            "DETECTOR",
            [measurement_record.get_target_rec(qq) for qq in data_support]
            + [measurement_record.get_target_rec(check_id)],
            (num_rounds, 0, kk),
        )

    # add all basis-type observables
    for kk, observable in enumerate(code.get_logical_ops(basis)):
        data_support = np.where(observable)[0]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [measurement_record.get_target_rec(qq) for qq in data_support],
            kk,
        )

    return noise_model.noisy_circuit(circuit) if noise_model else circuit


# TODO:
# - annotate logical observables in get_memory_simulation with Pauli gates, eliminating the need for ancilla qubits:
#   https://github.com/quantumlib/Stim/blob/main/doc/gates.md#OBSERVABLE_INCLUDE
# - sort out the SurfaceCode not having a mathing graph with get_memory_simulation


@restrict_to_qubits
def get_memory_simulation(
    code: codes.QuditCode,
    noise_model: NoiseModel,
    num_rounds: int = 1,
    *,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy | None = None,
) -> stim.Circuit:
    """Construct a circuit for testing the performance of a code as a quantum memory.

    This method constructs a circuit similar to that in qldpc.circuits.get_memory_experiment.
    However, the circuit constructed here makes use of noiseless (unphysical) ancilla qubits to
    track the error rates of both X-type and Z-type logical operators of an error-correcting code.

    See help(qldpc.circuits.get_memory_experiment) for background and context.

    The circuit constructed by this method performs the following:
    1. Noiselessly prepare a logical all-|0> state of the code.
    2. Noiselessly entangle each logical qubit with its own noiseless physical ancilla qubit,
        thereby preparing code.dimension Bell pairs.
    3. Perform num_rounds noisy QEC cycles, identically to qldpc.circuits.get_memory_experiment.
    4. Noiselessly measure the logical XX and ZZ operators of every Bell pair.

    Since the ancilla qubits of the Bell pairs are noiseless, an error in any XX or ZZ operator can
    be attributed to an error of the corresponding logical qubit of the code.

    Args:
        code: An error-correcting code.  If passed a classical code, treat it as a quantum CSS code
            that protects only basis-type logical operators.  Otherwise, only CSS stabilizer
            (non-subsystem) qubit codes are supported at the moment (generalization to non-CSS and
            subsystem codes pending).
        noise_model: The noise model to apply to the the QEC cycles of the circuit.
        num_rounds: Total number of QEC cycles to perform.  Must be at least 1.  Default: 1.
        syndrome_measurement_strategy: The syndrome measurement strategy to use, which defines how
            each round of QEC measures all parity checks of the code.
            Default: circuits.EdgeColoring().

    Returns:
        stim.Circuit: A circuit ready for simulation via Stim or Sinter.
    """
    if code.is_subsystem_code:
        raise ValueError("Memory simulations are currently not supported for subsystem codes")

    # set default syndrome measurement strategy, if necessary, and identify qubit IDs
    syndrome_measurement_strategy = syndrome_measurement_strategy or EdgeColoring()
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

    # build a noisy QEC cycle and initialize a measurement record
    one_cycle, cycle_measurements = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
    noisy_cycle = noise_model.noisy_circuit(one_cycle)
    measurement_record = MeasurementRecord()

    # set coordinates for all qubits
    circuit = stim.Circuit()
    for kk, data_id in enumerate(qubit_ids.data):
        circuit.append("QUBIT_COORDS", data_id, (0, kk))
    for kk, check_id in enumerate(qubit_ids.check):
        circuit.append("QUBIT_COORDS", check_id, (1, kk))
    for kk, ancilla_id in enumerate(qubit_ids.ancilla):
        circuit.append("QUBIT_COORDS", ancilla_id, (2, kk))

    # initialize a logical all-|0> state of the code, and intialize ancilla qubits in |+>
    circuit.append(get_encoding_circuit(code))
    circuit.append("H", qubit_ids.ancilla)

    # apply ancilla-controlled-logical-NOT gates to prepare Bell states
    for logical_qubit_index, ancilla_id in enumerate(qubit_ids.ancilla):
        ancilla_node = Node(logical_qubit_index, is_data=False)
        for _, data_node, edge_data in logical_op_graph[Pauli.X].edges(ancilla_node, data=True):
            circuit.append(f"C{edge_data[Pauli]}", [ancilla_id, data_node.index])

    # first round of QEC and detectors
    circuit.append(noisy_cycle)
    measurement_record.append(cycle_measurements)
    for kk, check_id in enumerate(check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))

    # following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = noisy_cycle.copy()
        measurement_record.append(cycle_measurements)
        for kk, check_id in enumerate(check_ids):
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (1, 0, kk),
            )
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

    # annotate the logical XX and ZZ operators of all Bell pairs
    for op_index, (pauli, logical_qubit_index) in enumerate(
        itertools.product(PAULIS_XZ, range(code.dimension))
    ):
        ancilla_node = Node(logical_qubit_index, is_data=False)
        qubit_paulis = [
            stim.target_pauli(data_node.index, str(edge_data[Pauli]))
            for _, data_node, edge_data in logical_op_graph[pauli].edges(ancilla_node, data=True)
        ]
        ancilla_pauli = stim.target_pauli(qubit_ids.ancilla[logical_qubit_index], str(pauli))
        circuit.append("OBSERVABLE_INCLUDE", qubit_paulis + [ancilla_pauli], [op_index])

    return circuit
