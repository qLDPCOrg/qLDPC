# SPDX-License-Identifier: Apache-2.0

"""Pack Clifford+T circuits into arrays of local gates and Pauli rotations.

``get_packed_circuit`` turns a Clifford+T circuit, with π/4 rotations encoded as described in the
docstring of ``RotatedPauliString``, into a ``PackedCircuit``: arrays that compiled code can apply
without calling back into Python.
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
_ROTATION_SIGNS = {"S": 1, "S_DAG": -1, "SPP": 1, "SPP_DAG": -1}
# Instructions that tsim tags to make other rotations, so tags other than T change their meaning.
_TAG_SENSITIVE = frozenset({"I", *_ROTATION_SIGNS})
# The tag of T gates (see the docstring of ``RotatedPauliString``).
T_TAG = "T"
# The kinds of operations of a packed circuit (see ``PackedCircuit``), compared only for equality.
LOCAL_GATE = 0
S_ROTATION = 1
T_ROTATION = 2

####################################################################################################
# packed circuits


@dataclasses.dataclass(frozen=True)
class PackedCircuit:
    """The unitary operations of a Clifford+T circuit, as arrays, in the order in which they act.

    Each operation has one of the following kinds.

    - ``LOCAL_GATE``: one of stim's one- and two-qubit Clifford gates, including untagged ``S``
      gates, applied through a precomputed table of the images of the ``4^k`` Pauli strings on its
      ``k`` qubits.
    - ``S_ROTATION``: a π/2 rotation about a Pauli product, from an untagged ``SPP`` or ``SPP_DAG``
      instruction.  Such a rotation is Clifford, but its table would have ``4^m`` entries for a
      product on ``m`` qubits, so it is applied through its axis ``a`` instead: it maps a Pauli
      string ``b`` to ``-i a b`` if ``b`` anticommutes with ``a``, and leaves it unchanged
      otherwise.
    - ``T_ROTATION``: a π/4 rotation about a Pauli product, from a T-tagged instruction.

    Local gates and rotations are numbered separately, in the order in which they act, and the
    arrays of local gates and of rotations are indexed by these numbers.  Local gates of the same
    stim gate share a table.  All arrays are read-only, since packed circuits are cached and shared.

    Attributes:
        num_qubits: The number of qubits of the packed circuit, which is at least the number of
            qubits that the circuit acts on.
        kinds: The kind of each operation, ``LOCAL_GATE``, ``S_ROTATION``, or ``T_ROTATION``.
        operands: For each operation, its index among the local gates if it is a local gate, and
            among the rotations otherwise.
        gate_qubits: For each local gate, a row of its qubits, with ``-1`` in place of the second
            qubit of a single-qubit gate.
        gate_tables: For each local gate, the index of its table.
        table_names: For each table, the name of its stim gate, such as ``"CX"``.
        table_sizes: For each table, the number of qubits of its gate, which is 1 or 2.
        table_images: For each table, a row with the index of the image of each Pauli string on the
            gate's qubits, in the format of ``_packed_paulis.get_clifford_gate_table``, padded with
            zeros to 16 entries.
        table_phases: For each table, a row with the phases of these images, padded in the same
            way.
        axes: For each rotation, a row with the packed bits of its axis, a Hermitian Pauli string
            on ``num_qubits`` qubits, in the layout of ``_packed_paulis``.
        axis_phases: For each rotation, the phase of its axis, in the convention of
            ``_packed_paulis``, which also carries the sign of the axis.
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
        """The arrays of this packed circuit, in the order in which compiled code takes them."""
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


def get_packed_circuit(
    circuit: stim.Circuit, num_qubits: int, *, inverse: bool = False
) -> PackedCircuit:
    """Pack a Clifford+T circuit, or its inverse, into arrays on ``num_qubits`` qubits.

    Packed circuits are cached by the text of the circuit, ``num_qubits``, and ``inverse``, so
    repeated calls with the same arguments are cheap, and later changes to a circuit object do not
    affect the packed circuits made from it before.  The arrays of a packed circuit are read-only,
    since packed circuits are shared.

    Args:
        circuit: A Clifford+T circuit, with π/4 rotations encoded as described in the docstring
            of ``RotatedPauliString``.
        num_qubits: The number of qubits to act on, at least that of the circuit.
        inverse: Whether to pack the inverse circuit instead.

    Returns:
        The packed circuit.

    Raises:
        ValueError: If the circuit has a non-unitary instruction, a classically controlled gate, a T
            tag on an instruction that is not a Pauli rotation, another tag on ``I``, ``S``,
            ``S_DAG``, ``SPP``, or ``SPP_DAG``, a Pauli product that repeats a qubit, or more qubits
            than ``num_qubits``.
    """
    if circuit.num_qubits > num_qubits:
        raise ValueError(f"The circuit acts on {circuit.num_qubits} > {num_qubits} qubits")
    return _get_packed_circuit(str(circuit), num_qubits, inverse)


####################################################################################################
# helpers


@functools.lru_cache(maxsize=64)
def _get_packed_circuit(text: str, num_qubits: int, inverse: bool) -> PackedCircuit:
    """The packed circuit with the given text (see ``get_packed_circuit``)."""
    # collect the kind and operand of each operation, the qubits and table of each local gate, the
    # table of each gate name, and the axis of each rotation
    kinds: list[int] = []
    operands: list[int] = []
    gate_qubits: list[tuple[int, int]] = []
    gate_tables: list[int] = []
    table_indices: dict[str, int] = {}
    axes: list[stim.PauliString] = []

    # read the instructions, skipping annotations and rejecting what packed circuits cannot
    # represent; the inverse circuit applies the inverse operations in reverse order, so reverse
    # the instructions and their target groups for it, and invert each operation below
    instructions = stim.Circuit(text).flattened()
    for instruction in reversed(instructions) if inverse else instructions:
        if instruction.name in _ANNOTATIONS:
            continue
        is_t_rotation = _check_instruction(instruction)
        groups = instruction.target_groups()
        if inverse:
            groups = groups[::-1]

        # each target group of a T-tagged instruction, SPP, or SPP_DAG is a Pauli rotation, whose
        # inverse is the rotation about the negated axis
        if is_t_rotation or instruction.name in ("SPP", "SPP_DAG"):
            kind = T_ROTATION if is_t_rotation else S_ROTATION
            sign = _ROTATION_SIGNS[instruction.name] * (-1 if inverse else 1)
            for group in groups:
                kinds.append(kind)
                operands.append(len(axes))
                axes.append(sign * _get_axis(instruction, group))
            continue

        # each target group of another gate, including an untagged S or S_DAG, is a local gate,
        # which shares a table with the gates of the same name; the inverse uses the inverse gate
        name = stim.gate_data(instruction.name).inverse.name if inverse else instruction.name
        table = table_indices.setdefault(name, len(table_indices))
        for group in groups:
            qubits = [target.value for target in group]
            kinds.append(LOCAL_GATE)
            operands.append(len(gate_tables))
            gate_qubits.append((qubits[0], qubits[1] if len(qubits) > 1 else -1))
            gate_tables.append(table)

    # build the tables of Pauli images of the local gates, padded to the 16 entries of two qubits
    table_names = tuple(table_indices)
    table_sizes = np.zeros(len(table_names), dtype=np.int64)
    table_images = np.zeros((len(table_names), 16), dtype=np.int64)
    table_phases = np.zeros((len(table_names), 16), dtype=np.int64)
    for table, name in enumerate(table_names):
        images, phases = packed.get_clifford_gate_table(name)
        table_sizes[table] = len(stim.Tableau.from_named_gate(name))
        table_images[table, : len(images)] = images
        table_phases[table, : len(phases)] = phases

    # pack the rotation axes
    axis_words, axis_phases = packed.pack_axes(axes, num_qubits)

    # assemble the packed circuit, whose arrays must be read-only, since it is cached and shared
    packed_circuit = PackedCircuit(
        num_qubits=num_qubits,
        kinds=np.array(kinds, dtype=np.int64),
        operands=np.array(operands, dtype=np.int64),
        gate_qubits=np.array(gate_qubits, dtype=np.int64).reshape(-1, 2),
        gate_tables=np.array(gate_tables, dtype=np.int64),
        table_names=table_names,
        table_sizes=table_sizes,
        table_images=table_images,
        table_phases=table_phases,
        axes=axis_words,
        axis_phases=axis_phases,
    )
    for array in packed_circuit.get_arrays():
        array.flags.writeable = False
    return packed_circuit


def _check_instruction(instruction: stim.CircuitInstruction) -> bool:
    """Whether an instruction is a π/4 rotation, after checking that packed circuits support it.

    Raises:
        ValueError: If the instruction is not unitary, is a classically controlled gate, has a T tag
            but is not a Pauli rotation, or has another tag on ``I``, ``S``, ``S_DAG``, ``SPP``, or
            ``SPP_DAG``.
    """
    if not stim.gate_data(instruction.name).is_unitary:
        raise ValueError(f"Instruction is not unitary: {instruction}")

    # only Pauli rotations can have the T tag; tsim tags I and Pauli rotations to make other
    # rotations, which packed circuits cannot represent, but tags on other gates do not change their
    # action
    tag = instruction.tag
    is_t_rotation = tag == T_TAG or tag.startswith(T_TAG + ":")
    if is_t_rotation and instruction.name not in _ROTATION_SIGNS:
        raise ValueError(f"Instruction cannot be a π/4 rotation: {instruction}")
    if tag and not is_t_rotation and instruction.name in _TAG_SENSITIVE:
        raise ValueError(
            f"Tags other than T are not supported on {instruction.name}: {instruction}"
        )

    # a classically controlled gate acts only for some measurement outcomes or sweep bits, which
    # packed circuits do not track
    if any(
        target.is_measurement_record_target or target.is_sweep_bit_target
        for target in instruction.targets_copy()
    ):
        raise ValueError(f"Classically controlled gates are not supported: {instruction}")
    return is_t_rotation


def _get_axis(
    instruction: stim.CircuitInstruction, group: list[stim.GateTarget]
) -> stim.PauliString:
    """The axis of a target group of a Pauli rotation, with its sign.

    The targets of ``S`` and ``S_DAG`` are qubits, each with the axis ``Z``.  The targets of ``SPP``
    and ``SPP_DAG`` are Pauli targets, each of which negates the axis if it is inverted.

    Raises:
        ValueError: If the target group repeats a qubit.
    """
    letters = {
        target.value: "Z" if target.is_qubit_target else target.pauli_type for target in group
    }
    if len(letters) < len(group):
        raise ValueError(f"Pauli products must not repeat qubits: {instruction}")
    inversions = sum(target.is_inverted_result_target for target in group)
    return (-1) ** inversions * stim.PauliString(letters)
