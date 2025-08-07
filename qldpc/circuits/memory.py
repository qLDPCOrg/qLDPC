"""Circuit construction utilities for quantum error correction experiments

This module provides functions for building Stim quantum circuits for quantum
error correction memory experiments using CSS codes.

Example:
    Creating a memory experiment circuit:

    from qldpc.codes import SteaneCode
    from qldpc.circuits import EdgeColoring, DepolarizingNoiseModel, memory_experiment
    from qldpc.objects import Pauli

    # Create a CSS code and noise model
    code = SteaneCode()
    noise_model = DepolarizingNoiseModel(1e-3)
    syndrome_measurement_strategy = EdgeColoring()

    # Generate a memory experiment circuit for the logical Z operator
    circuit = memory_experiment(
        code,
        syndrome_measurement_strategy,
        num_rounds=3,
        basis=Pauli.Z,
        noise_model=noise_model,
    )

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

from .noise_model import NoiseModel
from .syndrome_measurement import (
    EdgeColoring,
    MeasurementRecord,
    QubitIDs,
    SyndromeMeasurementStrategy,
)


def memory_experiment(
    code: codes.QuditCode,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy | None = None,
    num_rounds: int = 1,
    basis: PauliXZ = Pauli.X,
    qubit_ids: QubitIDs | None = None,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit:
    """Construct a circuit for a quantum memory cycle of the given code.

    Constructs a complete quantum circuit for testing the performance of a
    quantum error correcting code in a memory experiment. The circuit includes
    initialization, syndrome measurement rounds, final data measurements, and
    appropriate detector and observable definitions for error correction
    analysis.

    The qubit layout uses a linear arrangement:
    - Data qubits: coordinates (0, i) for i-th data qubit
    - X check qubits: coordinates (1, i) for i-th X stabilizer
    - Z check qubits: coordinates (2, i) for i-th Z stabilizer

    Detector coordinates follow the pattern (x, y, t, basis) where:
    - (x, y) are the check qubit coordinates
    - t is the syndrome measurement round (0-indexed)
    - basis indicates the stabilizer type (0 for Z, 1 for X)

    The experiment flow:
    1. Initialize data qubits in the specified basis
    2. Perform initial syndrome measurement (creates detectors for round 0)
    3. Repeat syndrome measurements for num_rounds-1 additional rounds
    4. Measure out all data qubits in the specified basis
    5. Create final detectors comparing data measurements to last syndrome round
    6. Define logical observables based on the code's logical operators

    Args:
        code: The CSS quantum error correcting code to test. Must have both
            X and Z stabilizers defined along with logical operators.
        noise_model: The noise model to apply to the circuit. The clean circuit
            is constructed first, then noise is applied via noisy_circuit().
        syndrome_measurement_strategy: The syndrome measurement strategy to use. This defines
            how stabilizer measurements are performed (e.g., scheduling,
            connectivity constraints).
        num_rounds: Total number of syndrome measurement rounds to perform.
            Must be at least 1. More rounds provide better error correction
            but increase circuit depth.
        basis: The Pauli basis for the memory experiment. Must be either
            Pauli.X or Pauli.Z. This determines:
            - How data qubits are initialized (|+⟩ for X, |0⟩ for Z)
            - How data qubits are measured (X-basis for X, Z-basis for Z)
            - Which stabilizers are used for initial/final detectors

    Returns:
        A complete Stim circuit with noise applied, ready for simulation.
        The circuit includes all necessary DETECTOR and OBSERVABLE_INCLUDE
        instructions for error correction analysis.

    Raises:
        ValueError: If basis is not Pauli.X or Pauli.Z.

    Example:
        from qldpc.codes.classical import RepetitionCode
        from qldpc.codes.quantum import codes.CSSCode
        from qldpc.stim.noise_model import NoiseModel
        from qldpc.stim.syndrome_measurement_strategy import EdgeColoring
        from qldpc.objects import Pauli
        >>>
        # Create a 3-qubit repetition code
        rep_code = RepetitionCode(3)
        css_code = codes.CSSCode(rep_code, rep_code)
        noise_model = NoiseModel.uniform_depolarizing(0.01)
        syndrome_measurement_strategy = EdgeColoring()
        >>>
        # Generate 5-round Z-basis memory experiment
        circuit = memory_experiment(
             code=css_code,
             noise_model=noise_model,
             syndrome_measurement_strategy=syndrome_measurement_strategy,
             num_rounds=5,
             basis=Pauli.Z
         )
        >>>
        # Circuit is ready for simulation
        sampler = circuit.compile_sampler()
        results = sampler.sample(1000)
    """
    if basis is not Pauli.X and basis is not Pauli.Z:
        raise ValueError(f"Invalid basis: {basis}")

    if not isinstance(code, codes.CSSCode):
        raise ValueError("Memory experiments for are currently not supported for non-CSS codes")

    # identify data and check qubit indices
    data_ids, check_ids = qubit_ids or QubitIDs.from_code(code)

    # set default measurement strategy, identify relevant checks as well as their support
    syndrome_measurement_strategy = syndrome_measurement_strategy or EdgeColoring()
    check_support = code.get_matrix(basis)
    check_ids = (
        check_ids[: code.num_checks_x] if basis is Pauli.X else check_ids[code.num_checks_x :]
    )

    # build one QEC cycle
    measurement_record = MeasurementRecord()
    one_cycle, cycle_measurements = syndrome_measurement_strategy.get_circuit(code, qubit_ids)

    """
    Define qubit coordinates
    """
    circuit = stim.Circuit()
    for i, data_id in enumerate(data_ids):
        circuit.append("QUBIT_COORDS", data_id, (0, i))
    for i, check_id in enumerate(check_ids):
        circuit.append("QUBIT_COORDS", check_id, (1, i))

    # Reset data qubits to appropriate basis
    circuit.append(f"R{basis}", data_ids)

    """
    Initial syndrome round to project into quiescent state
    """
    circuit.append(one_cycle)
    measurement_record.append(cycle_measurements)
    for i, check_id in enumerate(check_ids):
        circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (i, 0))

    if num_rounds > 1:
        """
        Repeated syndrome rounds
        """
        repeat_circuit = stim.Circuit()
        repeat_circuit.append(one_cycle)
        measurement_record.append(cycle_measurements)  # TODO: fix for repeat blocks
        for i, check_id in enumerate(check_ids):
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (i, 1),
            )
        repeat_circuit.append("SHIFT_COORDS", [], (0, 1))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

    """
    Measure out data qubits
    """
    circuit.append(f"M{basis}", data_ids)
    measurement_record.append({qubit: [qubit] for qubit in range(len(code))})

    """
    Reconstruct a final round of checks based on data qubit measurements
    """
    for i, check_id in enumerate(check_ids):
        data_support = np.where(check_support[i])[0]
        circuit.append(
            "DETECTOR",
            [measurement_record.get_target_rec(data_ids[q]) for q in data_support]
            + [measurement_record.get_target_rec(check_id)],
            (i, num_rounds),
        )

    """
    Define observables for memory experiment
    """
    observables = code.get_logical_ops(basis)
    for k, obs in enumerate(observables):
        data_support = np.where(obs)[0]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [measurement_record.get_target_rec(data_ids[q]) for q in data_support],
            k,
        )

    return noise_model.noisy_circuit(circuit) if noise_model else circuit
