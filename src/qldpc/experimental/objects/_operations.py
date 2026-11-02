# SPDX-License-Identifier: Apache-2.0

"""Parse Clifford+T circuits into programs of Clifford gates and Pauli rotations.

Stim does not support T gates.  Following tsim, a Clifford+T circuit is represented by a
``stim.Circuit`` in which T gates are encoded as the instructions ``S``, ``S_DAG``, ``SPP``, and
``SPP_DAG`` with the tag ``T``.  Such an instruction stands for the π/4 rotation ``exp(-iπ a/8)``
about each Pauli string ``a`` that the untagged instruction would rotate about by π/2, so ``S[T] q``
encodes the T gate ``diag(1, e^{iπ/4})`` on qubit ``q``, up to a global phase.  Tsim's ``T`` and
``TPP`` instructions become these tagged instructions.  Tags other than ``T`` are ignored.

Stim itself ignores tags, and treats these instructions as the Clifford gates ``S`` and ``SPP``,
so the operations here must not be handed back to stim.  ``get_program`` turns a circuit into a
``Program`` of arrays that compiled code can apply without calling back into Python.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import numpy.typing as npt
import stim

from . import _packed_paulis as packed

# Instructions with no action on the state.
_ANNOTATIONS = frozenset(
    {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}
)
# The instructions that are Pauli rotations, mapped to the sign of their rotation axes.
_ROTATIONS = {"S": 1, "S_DAG": -1, "SPP": 1, "SPP_DAG": -1}
# The tag of T gates.
T_TAG = "T"
# The kinds of operations: a Clifford gate, a π/4 rotation (T-type), or a π/2 rotation (S-type).
GATE = 0
T_ROTATION = 1
S_ROTATION = 2

####################################################################################################
# programs


@dataclasses.dataclass(frozen=True)
class Program:
    """The unitary operations of a Clifford+T circuit, as arrays, in the order in which they act.

    Operation ``k`` is a Clifford gate if ``kinds[k] == GATE``, and otherwise the rotation
    ``exp(-iπ kinds[k] a/8)`` about a Hermitian Pauli string ``a``, so ``T_ROTATION`` marks π/4
    rotations and ``S_ROTATION`` marks π/2 rotations.  Its operand ``operands[k]`` indexes the gates
    or the rotations.

    Gate ``g`` acts on the qubits ``gate_qubits[g, :table_sizes[t]]`` with the table ``t =
    gate_tables[g]``, whose images and phases are rows of ``table_images`` and ``table_phases``, in
    the format of ``_packed_paulis.get_clifford_gate_table``, padded to 16 entries.  Rotation ``r``
    has the packed axis ``axes[r]`` with phase ``axis_phases[r]``, on ``num_qubits`` qubits.
    """

    num_qubits: int
    kinds: npt.NDArray[np.int64]
    operands: npt.NDArray[np.int64]
    gate_qubits: npt.NDArray[np.int64]
    gate_tables: npt.NDArray[np.int64]
    table_names: tuple[str, ...]
    table_sizes: npt.NDArray[np.int64]
    table_images: npt.NDArray[np.int64]
    table_phases: npt.NDArray[np.int64]
    axes: npt.NDArray[np.uint64]
    axis_phases: npt.NDArray[np.int64]

    def get_arrays(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.uint64],
        npt.NDArray[np.int64],
    ]:
        """The arrays of this program, in the order in which compiled code takes them."""
        return (
            self.kinds,
            self.operands,
            self.gate_qubits,
            self.gate_tables,
            self.table_sizes,
            self.table_images,
            self.table_phases,
            self.axes,
            self.axis_phases,
        )


def get_program(circuit: stim.Circuit, num_qubits: int, *, inverse: bool = False) -> Program:
    """The program of a Clifford+T circuit, or of its inverse, on ``num_qubits`` qubits.

    Programs are cached by the text of their circuits, so repeated calls with the same circuit are
    cheap, and later changes to a circuit object do not affect programs made from it before.  The
    arrays of a program are read-only, since programs are shared.

    Args:
        circuit: A stim circuit with only unitary instructions and annotations, with T gates
            encoded as described in this module's docstring.
        num_qubits: The number of qubits of the program, at least that of the circuit.
        inverse: Whether to return the program of the inverse circuit instead.

    Returns:
        The program of the circuit.

    Raises:
        ValueError: If the circuit has a non-unitary instruction, a T tag on an instruction that is
            not a Pauli rotation, a Pauli product that repeats a qubit, or more qubits than
            ``num_qubits``.
    """
    if circuit.num_qubits > num_qubits:
        raise ValueError(f"The circuit acts on {circuit.num_qubits} > {num_qubits} qubits")
    return _get_program(str(circuit), num_qubits, inverse)


####################################################################################################
# helpers


@functools.lru_cache(maxsize=64)
def _get_program(text: str, num_qubits: int, inverse: bool) -> Program:
    """The program of the circuit with the given text (see ``get_program``)."""
    kinds: list[int] = []
    operands: list[int] = []
    gate_qubits: list[tuple[int, int]] = []
    gate_tables: list[int] = []
    table_indices: dict[str, int] = {}
    axis_qubits: list[list[tuple[int, str]]] = []
    axis_signs: list[int] = []

    instructions = stim.Circuit(text).flattened()
    for instruction in reversed(instructions) if inverse else instructions:
        name = instruction.name
        if name in _ANNOTATIONS:
            continue
        data = stim.gate_data(name)
        if not data.is_unitary:
            raise ValueError(f"Instruction is not unitary: {instruction}")
        is_t_gate = instruction.tag == T_TAG
        if is_t_gate and name not in _ROTATIONS:
            raise ValueError(f"Instruction cannot be a T gate: {instruction}")
        groups = instruction.target_groups()
        if inverse:
            groups = groups[::-1]
        if is_t_gate or name in ("SPP", "SPP_DAG"):
            sign = -_ROTATIONS[name] if inverse else _ROTATIONS[name]
            for group in groups:
                kinds.append(T_ROTATION if is_t_gate else S_ROTATION)
                operands.append(len(axis_signs))
                letters = [
                    (target.value, "Z" if target.is_qubit_target else target.pauli_type)
                    for target in group
                ]
                if len({qubit for qubit, _ in letters}) < len(letters):
                    raise ValueError(f"Pauli products must not repeat qubits: {instruction}")
                axis_qubits.append(letters)
                inversions = sum(target.is_inverted_result_target for target in group)
                axis_signs.append(sign * (-1) ** inversions)
        else:
            gate_name = data.inverse.name if inverse else name
            table = table_indices.setdefault(gate_name, len(table_indices))
            for group in groups:
                kinds.append(GATE)
                operands.append(len(gate_tables))
                qubits = [target.value for target in group]
                gate_qubits.append((qubits[0], qubits[1] if len(qubits) > 1 else -1))
                gate_tables.append(table)

    table_names = tuple(table_indices)
    table_sizes = np.zeros(len(table_names), dtype=np.int64)
    table_images = np.zeros((len(table_names), 16), dtype=np.int64)
    table_phases = np.zeros((len(table_names), 16), dtype=np.int64)
    for table, gate_name in enumerate(table_names):
        images, phases = packed.get_clifford_gate_table(gate_name)
        table_sizes[table] = len(stim.Tableau.from_named_gate(gate_name))
        table_images[table, : len(images)] = images
        table_phases[table, : len(phases)] = phases

    xs = np.zeros((len(axis_signs), num_qubits), dtype=bool)
    zs = np.zeros((len(axis_signs), num_qubits), dtype=bool)
    for row, letters in enumerate(axis_qubits):
        for qubit, letter in letters:
            xs[row, qubit] = letter in "XY"
            zs[row, qubit] = letter in "YZ"
    num_y = np.count_nonzero(xs & zs, axis=1)
    axis_phases = (num_y + np.where(np.array(axis_signs, dtype=np.int64) < 0, 2, 0)) % 4

    program = Program(
        num_qubits=num_qubits,
        kinds=np.array(kinds, dtype=np.int64),
        operands=np.array(operands, dtype=np.int64),
        gate_qubits=np.array(gate_qubits, dtype=np.int64).reshape(-1, 2),
        gate_tables=np.array(gate_tables, dtype=np.int64),
        table_names=table_names,
        table_sizes=table_sizes,
        table_images=table_images,
        table_phases=table_phases,
        axes=packed.pack(xs, zs),
        axis_phases=axis_phases.astype(np.int64),
    )
    for array in program.get_arrays():  # programs are cached and shared, so protect them
        array.flags.writeable = False
    return program
