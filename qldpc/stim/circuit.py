"""Circuit construction utilities for quantum error correction experiments.

This module provides functions for building Stim quantum circuits for quantum
error correction memory experiments using CSS codes.

Example:
    Creating a memory experiment circuit:

    from qldpc.codes import SteanCode
    from qldpc.stim.noise_model import NoiseModel
    from qldpc.stim.syndrome_measurement_strategy import BareColorCircuit
    from qldpc.objects import Pauli
    # Create a CSS code and noise model
    css_code = SteanCode()
    noise_model = NoiseModel.uniform_depolarizing(0.001)
    syndrome_measurement_strategy = BareColorCircuit()
    # Generate memory experiment circuit
    circuit = memory_experiment(
         code=css_code,
         noise_model=noise_model,
         syndrome_measurement_strategy=syndrome_measurement_strategy,
         num_rounds=5,
         basis=Pauli.Z
     )
"""

import numpy as np
import stim

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ
from qldpc.stim.noise_model import NoiseModel
from qldpc.stim.syndrome_measurement import StimIds, SyndromeMeasurementStrategy


def _update_meas_rec(meas_record: list[dict[int, int]], qubits: list[int]) -> None:
    """Updates a measurement record after a round of measurements.

    Measurement results in Stim are recorded as a stack and can be tricky to
    reference correctly. The meas_record tracks the indices in the stack for
    each 'round' of measurements. This function should be called after every
    round of measurements with the same list of qubits that were measured.

    The function maintains a record where each entry maps qubit indices to their
    corresponding negative indices in the Stim measurement record stack. When
    new measurements are added, all previous indices are shifted to maintain
    correct references.

    Args:
        meas_record: A list of dictionaries, where each dictionary maps qubit
            indices to their corresponding negative indices in the Stim
            measurement record. This list is modified in-place.
        qubits: A list of qubit indices that were measured in the current round,
            in the order they were passed to Stim. The order matters because
            Stim records measurements in a stack.

    Example:
        meas_record = []
        _update_meas_rec(meas_record, [0, 1, 2])
        print(meas_record)
        [{0: -3, 1: -2, 2: -1}]
        _update_meas_rec(meas_record, [3, 4])
        print(meas_record)
        [{0: -5, 1: -4, 2: -3}, {3: -2, 4: -1}]
    """
    meas_round = {}
    for i in range(len(qubits)):
        q = qubits[-(i + 1)]
        meas_round[q] = -(i + 1)
    for round in meas_record:
        for q, idx in round.items():
            round[q] = idx - len(qubits)
    meas_record.append(meas_round)


def memory_experiment(
    code: CSSCode,
    syndrome_measurement_strategy: SyndromeMeasurementStrategy,
    num_rounds: int,
    basis: PauliXZ,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit:
    """Creates a Stim circuit for a quantum memory experiment.

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
        from qldpc.codes.quantum import CSSCode
        from qldpc.stim.noise_model import NoiseModel
        from qldpc.stim.syndrome_measurement_strategy import BareColorCircuit
        from qldpc.objects import Pauli
        >>>
        # Create a 3-qubit repetition code
        rep_code = RepetitionCode(3)
        css_code = CSSCode(rep_code, rep_code)
        noise_model = NoiseModel.uniform_depolarizing(0.01)
        syndrome_measurement_strategy = BareColorCircuit()
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

    data_qubits: list[int] = list(range(code.num_qubits))
    z_check_qubits: list[int] = list(range(code.num_qubits, code.num_qubits + code.num_checks_z))
    x_check_qubits: list[int] = list(
        range(
            code.num_qubits + code.num_checks_z,
            code.num_qubits + code.num_checks_z + code.num_checks_x,
        )
    )
    stim_ids = StimIds(data_qubits, z_check_qubits, x_check_qubits)

    meas_rec: list[dict[int, int]] = []
    sm_circuit, sm_measurements = syndrome_measurement_strategy.compile_sm_circuit(code, stim_ids)

    """
    Define qubit coordinates
    """
    circuit = stim.Circuit()
    for i, data in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", data, (0, i))
    for i, x_check in enumerate(x_check_qubits):
        circuit.append("QUBIT_COORDS", x_check, (1, i))
    for i, z_check in enumerate(z_check_qubits):
        circuit.append("QUBIT_COORDS", z_check, (2, i))

    # Reset data qubits to appropriate basis
    circuit.append(f"R{basis}", data_qubits)
    """
    Initial syndrome round to project into quiescent state
    """
    circuit.append(sm_circuit)
    for meas_round in sm_measurements:
        _update_meas_rec(meas_rec, meas_round)
    if basis is Pauli.X:
        for i, check_id in enumerate(x_check_qubits):
            circuit.append("DETECTOR", [stim.target_rec(meas_rec[-1][check_id])], (1, i, 0))
    elif basis is Pauli.Z:
        for i, check_id in enumerate(z_check_qubits):
            circuit.append("DETECTOR", [stim.target_rec(meas_rec[-1][check_id])], (2, i, 0))

    """
    Repeated syndrome rounds
    """
    repeat_circuit = stim.Circuit()
    repeat_circuit.append(sm_circuit)
    for meas_round in sm_measurements:
        _update_meas_rec(meas_rec, meas_round)
    for i, check_id in enumerate(x_check_qubits):
        repeat_circuit.append(
            "DETECTOR",
            [
                stim.target_rec(meas_rec[-1][check_id]),
                stim.target_rec(meas_rec[-2][check_id]),
            ],
            (1, i, 1),
        )
    for i, check_id in enumerate(z_check_qubits):
        repeat_circuit.append(
            "DETECTOR",
            [
                stim.target_rec(meas_rec[-1][check_id]),
                stim.target_rec(meas_rec[-2][check_id]),
            ],
            (2, i, 1),
        )
    repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))

    """
    Measure out data qubits
    """
    if basis is Pauli.X:
        circuit.append("MX", data_qubits)
    elif basis is Pauli.Z:
        circuit.append("M", data_qubits)

    _update_meas_rec(meas_rec, data_qubits)

    """
    Reconstruct a final round of checks based on data qubit measurements
    """
    if basis is Pauli.X:
        for i, check_id in enumerate(x_check_qubits):
            data_support = np.where(code.code_x.matrix[i])[0]
            circuit.append(
                "DETECTOR",
                [stim.target_rec(meas_rec[-1][data_qubits[q]]) for q in data_support]
                + [stim.target_rec(meas_rec[-2][check_id])],
                (1, i, num_rounds),
            )
    elif basis is Pauli.Z:
        for i, check_id in enumerate(z_check_qubits):
            data_support = np.where(code.code_z.matrix[i])[0]
            circuit.append(
                "DETECTOR",
                [stim.target_rec(meas_rec[-1][data_qubits[q]]) for q in data_support]
                + [stim.target_rec(meas_rec[-2][check_id])],
                (2, i, num_rounds),
            )

    """
    Define observables for memory experiment
    """
    observables = code.get_logical_ops(basis)
    for k, obs in enumerate(observables):
        data_support = np.where(obs)[0]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(meas_rec[-1][data_qubits[q]]) for q in data_support],
            k,
        )

    # Apply noise model to the entire circuit
    return noise_model.noisy_circuit(circuit)
