from __future__ import annotations

import numpy as np
import stim

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ
from qldpc.stim.noise_model import NoiseModel
from qldpc.stim.syndrome_measurement import StimIds, SyndromeMeasurement


def _update_meas_rec(meas_record: list[dict[int, int]], qubits: list[int]) -> None:
    """
    Updates a measurement record after a round of measurements.

    Measurement results in stim are recorded as a stack and can be tricky to reference.
    The meas_record tracks the indicies in the stack for each 'round' of measurements (call to this function).
    This function is called after every call to NoiseModel.apply_meas with the same list of qubits
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
    noise_model: NoiseModel,
    syndrome_meas: SyndromeMeasurement,
    num_rounds: int,
    basis: PauliXZ,
) -> stim.Circuit:
    """
    Creates a stim circuit for a memory experiment in the specified basis.
    Qubit coordinates reflect a line of qubits for data, X checks, and Z checks as follows:
    (0, i) for data qubits
    (1, i) for X checks
    (2, i) for Z checks

    Detector coordinates are (coords, t, basis) where coords in the check qubit coordinates, t is the syndrome round, and basis is the basis of the check qubit (0 == Z, 1 == X)
    """
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
    sm_circuit, sm_measurements = syndrome_meas.compile_sm_circuit(code, noise_model, stim_ids)

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

    noise_model.apply_reset(circuit, basis, data_qubits)

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
    else:
        raise ValueError(f"Invalid basis: {basis}")

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
    noise_model.apply_meas(circuit, basis, data_qubits)
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
    else:
        raise ValueError(f"Invalid basis: {basis}")

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
    return circuit
