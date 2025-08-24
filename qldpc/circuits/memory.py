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

import numpy as np
import stim

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from .common import restrict_to_qubits
from .noise_model import NoiseModel
from .syndrome_measurement import (
    EdgeColoring,
    MeasurementRecord,
    QubitIDs,
    SyndromeMeasurementStrategy,
)


@restrict_to_qubits
def get_memory_experiment(
    code: codes.AbstractCode,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy | None = None,
    *,
    num_rounds: int = 1,
    basis: PauliXZ = Pauli.X,
    qubit_ids: QubitIDs | None = None,
    noise_model: NoiseModel | None = None,
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
        syndrome_measurement_strategy: The syndrome measurement strategy to use, which defines how
            each round of QEC measures all parity checks of the code.
            Default: circuits.EdgeColoring().
        num_rounds: Total number of QEC cycles to perform.  Must be at least 1.  Default: 1.
        basis: Should be Pauli.X or Pauli.Z, depending the desired logical operators to track.  A
            logical error in a noisy simulation of the circuit corresponds to a logical error in one
            of these operators.  Default: Pauli.X.
        qubit_ids: A QubitIDs object specifying the index of data and check qubits.  Defaults to
            labeling qubits by their corresponding column/row of the parity check matrix.
        noise_model: The noise model to apply to the circuit after construction, or None to return a
            noiseless circuit.  Default: None.

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
            num_rounds=5,
            basis=Pauli.Z,
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

    # if necessary, set the default syndrome measurement strategy
    syndrome_measurement_strategy = syndrome_measurement_strategy or EdgeColoring()

    # identify all data and check qubit indices
    qubit_ids = qubit_ids or QubitIDs.from_code(code)
    data_ids, check_ids = qubit_ids

    # identify the support and indices of basis-type parity checks
    check_support = code.get_matrix(basis)
    check_ids = (
        check_ids[: code.num_checks_x] if basis is Pauli.X else check_ids[code.num_checks_x :]
    )

    # build one QEC cycle and initialize a measurement record
    one_cycle, cycle_measurements = syndrome_measurement_strategy.get_circuit(code, qubit_ids)
    measurement_record = MeasurementRecord()

    # set coordinates for all qubits
    circuit = stim.Circuit()
    for data_id in qubit_ids.data:
        circuit.append("QUBIT_COORDS", data_id, (0, data_id))
    for check_id in qubit_ids.check:
        circuit.append("QUBIT_COORDS", check_id, (1, check_id))

    # reset data qubits to appropriate basis
    circuit.append(f"R{basis}", data_ids)

    # first round of QEC and detectors
    circuit += one_cycle
    measurement_record.append(cycle_measurements)
    for kk, check_id in enumerate(check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, kk))

    # following repeated rounds of QEC and detectors
    if num_rounds > 1:
        repeat_circuit = stim.Circuit()
        repeat_circuit += one_cycle
        measurement_record.append(cycle_measurements)
        for kk, check_id in enumerate(check_ids):
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (1, kk),
            )
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

        # make the measurement_record account for repeated measurements
        for _ in range(num_rounds - 2):
            measurement_record.append(cycle_measurements)

    # measure out the data qubits
    circuit.append(f"M{basis}", data_ids)
    measurement_record.append({qubit: [qubit] for qubit in range(len(code))})

    # detectors for all stabilizers that can be inferred from the data qubit measurements
    for jj, check_id in enumerate(check_ids):
        data_support = np.where(check_support[jj])[0]
        circuit.append(
            "DETECTOR",
            [measurement_record.get_target_rec(qq) for qq in data_support]
            + [measurement_record.get_target_rec(check_id)],
            (num_rounds, jj),
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
