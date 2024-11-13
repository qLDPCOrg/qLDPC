"""Tools for constructing circuits

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

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

_INT_TO_PAULI = {1: "X", 2: "Y", 3: "Z"}


def restrict_to_qubits(func: Callable[[...], stim.Circuit]) -> Callable[[...], stim.Circuit]:
    """Restrict a circuit constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: object, **kwargs: object) -> stim.Circuit:
        if any(isinstance(arg, codes.QuditCode) and arg.field.order != 2 for arg in args):
            raise ValueError("Circuit methods are only supported for qubit codes.")
        return func(*args, **kwargs)

    return qubit_func


def op_to_string(op: npt.NDArray[np.int_], flip_xz: bool = False) -> stim.PauliString:
    """Convert an integer array that represents a Pauli string into a stim.PauliString."""
    assert len(op) % 2 == 0
    num_qubits = len(op) // 2
    paulis = []
    for qubit in range(num_qubits):
        val_x = int(op[qubit])
        val_z = int(op[qubit + num_qubits])
        pauli = Pauli((val_x, val_z))
        paulis.append(str(pauli if not flip_xz else ~pauli))
    return stim.PauliString("".join(paulis))


@restrict_to_qubits
def prep(
    code: codes.QuditCode,
    string: Pauli | stim.PauliString = Pauli.Z,
    *,
    qubits: Sequence[int] | None,
) -> stim.Circuit:
    """Circuit to prepare a logical +1 eigenstate of the given logical Pauli string.

    Optionally prepend a specified number of qubits to every Pauli string.
    """
    if isinstance(string, Pauli):
        string = stim.PauliString(str(string) * code.dimension)
    elif len(string) < code.dimension:
        string += stim.PauliString(len(string) - code.dimension)
    assert len(string) == code.dimension

    # identify logical operators that stabilize our target state
    logical_ops = code.get_logical_ops()
    logical_stabs = []
    for qubit, pauli in enumerate(string):
        if pauli == 1:
            logical_op = logical_ops[0, qubit]
        elif pauli == 3 or pauli == 0:
            logical_op = logical_ops[1, qubit]
        else:
            assert pauli == 2
            logical_op = logical_ops[0, qubit] + logical_ops[1, qubit]
        logical_stabs.append(op_to_string(logical_op))

    code_stabs = [op_to_string(row, flip_xz=True) for row in code.matrix]
    tableau = stim.Tableau.from_stabilizers(logical_stabs + code_stabs, allow_redundant=True)
    circuit = tableau.to_circuit()
    if not qubits:
        return circuit

    assert len(qubits) >= circuit.num_qubits
    remapped_circuit = stim.Circuit()
    for instruction in tableau.to_circuit():
        targets = [stim.GateTarget(qubits[target.value]) for target in instruction.targets_copy()]
        remapped_circuit.append(instruction.name, targets, instruction.gate_args_copy())
    return remapped_circuit


@restrict_to_qubits
def steane_syndrome_extraction(
    code: codes.CSSCode, error_prob: float = 0, num_cycles: int = 1, observables: PauliXZ = Pauli.Z
) -> stim.Circuit:
    """Circuit to perform one or more Steane-type syndrome measurement cycles.

    Assumes noisy syndrome extraction (with a depolarizing noise model), but perfect (noiseless)
    logical state preparation.
    """
    # identify data qubits, "copy" qubits, and collect them into pairs
    data_qubits = list(range(code.num_qubits))
    copy_qubits = [qq + code.num_qubits for qq in range(code.num_qubits)]
    data_copy_pairs = [qq for dd_cc in zip(data_qubits, copy_qubits) for qq in dd_cc]

    # identify checks that detect bit-flip (X-type) and phase-flip (Z-type) errors
    checks_x = [row for row in code.matrix if any(row[: code.num_qubits])]
    checks_z = [row for row in code.matrix if any(row[code.num_qubits :])]

    # assign ancillas to X-type and Z-type checks
    ancillas_x = [qq + 2 * code.num_qubits for qq in range(len(checks_x))]
    ancillas_z = [qq + 2 * code.num_qubits for qq in range(len(checks_z))]

    def extract_syndromes(
        checks: Sequence[npt.NDArray[np.int_]], ancillas: Sequence[int]
    ) -> stim.Circuit:
        """Extract syndromes corresponding to the given checks onto the given ancillas."""
        circuit = stim.Circuit()
        circuit.append("R", ancillas)
        circuit.append("X_ERROR", ancillas, error_prob)
        circuit.append("H", ancillas)
        circuit.append("DEPOLARIZE1", ancillas, error_prob)
        for check, ancilla in zip(checks, ancillas):
            string = op_to_string(check, flip_xz=True)
            for data_qubit in np.where(string)[0]:
                gate = "C" + _INT_TO_PAULI[string[data_qubit]]
                targets = (ancilla, copy_qubits[data_qubit])
                circuit.append(gate, targets)
            circuit.append("DEPOLARIZE2", targets, error_prob)
        circuit.append("H", ancillas)
        circuit.append("DEPOLARIZE1", ancillas, error_prob)
        circuit.append("X_ERROR", ancillas, error_prob)
        circuit.append("M", ancillas_x)
        return circuit

    # bit-flip (X-type) error correction cycle
    cycle_x = prep(code, Pauli.Z, qubits=copy_qubits)
    cycle_x.append("CX", data_copy_pairs)
    cycle_x.append("DEPOLARIZE2", data_copy_pairs, error_prob)
    cycle_x += extract_syndromes(checks_x, ancillas_x)

    # phase-flip (Z-type) error correction cycle
    cycle_z = prep(code, Pauli.X, qubits=copy_qubits)
    cycle_z.append("CZ", data_copy_pairs)
    cycle_z.append("DEPOLARIZE2", data_copy_pairs, error_prob)
    cycle_z += extract_syndromes(checks_z, ancillas_z)

    # one full error correction cycle
    circuit = stim.Circuit()
    circuit += cycle_x
    for index in range(-1, -len(ancillas_x) - 1, -1):
        circuit.append("DETECTOR", stim.target_rec(index))
    circuit.append("TICK")
    circuit.append("R", copy_qubits)
    circuit += cycle_z
    for index in range(-1, -len(ancillas_z) - 1, -1):
        circuit.append("DETECTOR", stim.target_rec(index))

    # additional QEC cycles
    if num_cycles > 1:
        repeat_cycle = stim.Circuit()
        repeat_cycle.append("TICK")
        repeat_cycle.append("R", copy_qubits)
        repeat_cycle += cycle_x
        for index in range(-1, -len(ancillas_x) - 1, -1):
            targets = [stim.target_rec(index), stim.target_rec(index - len(ancillas_z))]
            repeat_cycle.append("DETECTOR", targets)
        repeat_cycle.append("TICK")
        repeat_cycle.append("R", copy_qubits)
        repeat_cycle += cycle_z
        for index in range(-1, -len(ancillas_z) - 1, -1):
            targets = [stim.target_rec(index), stim.target_rec(index - len(ancillas_x))]
            repeat_cycle.append("DETECTOR", targets)
        circuit.append(stim.CircuitRepeatBlock(num_cycles - 1, repeat_cycle))

    # identify logical observables
    for op_idx, logical_op in enumerate(code.get_logical_ops(observables)):
        string = op_to_string(logical_op)
        support = [f"{_INT_TO_PAULI[string[qubit]]}{qubit}" for qubit in np.where(string)[0]]
        observable = "*".join(support)
        circuit += stim.Circuit(f"MPP {observable}")
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], op_idx)

    return circuit if error_prob else circuit.without_noise()
