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
_TSIM_TAGGED_GATES = frozenset({"I", *_ROTATION_SIGNS})
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
      gates.  A local gate on ``k`` qubits is applied through its *conjugation table*: a lookup
      table of the image under conjugation by the gate of each of the ``4^k`` Pauli strings on
      those qubits.
    - ``S_ROTATION``: a π/2 rotation about a Pauli product, from an untagged ``SPP`` or ``SPP_DAG``
      instruction.  Such a rotation is Clifford, but its conjugation table would have ``4^m``
      entries for a product on ``m`` qubits, so it is applied through its axis ``a`` instead: it
      maps a Pauli string ``b`` to ``-i a b`` if ``b`` anticommutes with ``a``, and leaves it
      unchanged otherwise.
    - ``T_ROTATION``: a π/4 rotation about a Pauli product, from a T-tagged instruction.

    Local gates and rotations are numbered separately, in the order in which they act, and the
    arrays of local gates and of rotations are indexed by these numbers.  Local gates of the same
    stim gate share a conjugation table, and conjugation tables are numbered too.  A conjugation
    table numbers the Pauli strings on the gate's qubits, and lists the number and the phase of the
    image of each, as ``_packed_paulis.get_conjugation_table`` does.  All arrays are read-only,
    since packed circuits are cached and shared.

    Attributes:
        num_qubits: The number of qubits of the packed circuit, which is at least the number of
            qubits that the circuit acts on.
        operation_kinds: The kind of each operation, ``LOCAL_GATE``, ``S_ROTATION``, or
            ``T_ROTATION``.
        gate_or_rotation_indices: For each operation, its index among the local gates if it is a
            local gate, and among the rotations otherwise.
        gate_qubits: For each local gate, a row of its qubits, with ``-1`` in place of the second
            qubit of a single-qubit gate.
        gate_conjugation_tables: For each local gate, the index of its conjugation table.
        conjugation_table_names: For each conjugation table, the name of its stim gate, such as
            ``"CX"``.
        conjugation_table_num_qubits: For each conjugation table, the number of qubits of its
            gate, which is 1 or 2.
        conjugation_table_images: For each conjugation table, a row with the index of the image of
            each Pauli string on the gate's qubits, padded with zeros to 16 entries.
        conjugation_table_image_phases: For each conjugation table, a row with the phases of these
            images, padded in the same way.
        rotation_axes: For each rotation, a row with the packed bits of its axis, a Hermitian Pauli
            string on ``num_qubits`` qubits, in the layout of ``_packed_paulis``.
        rotation_axis_phases: For each rotation, the phase of its axis, in the convention of
            ``_packed_paulis``, which also carries the sign of the axis.
    """

    num_qubits: int
    operation_kinds: npt.NDArray[np.int64]
    gate_or_rotation_indices: npt.NDArray[np.int64]
    gate_qubits: npt.NDArray[np.int64]
    gate_conjugation_tables: npt.NDArray[np.int64]
    conjugation_table_names: tuple[str, ...]
    conjugation_table_num_qubits: npt.NDArray[np.int64]
    conjugation_table_images: npt.NDArray[np.int64]
    conjugation_table_image_phases: npt.NDArray[np.int64]
    rotation_axes: npt.NDArray[np.uint64]
    rotation_axis_phases: npt.NDArray[np.int64]

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
            self.operation_kinds,
            self.gate_or_rotation_indices,
            self.gate_qubits,
            self.gate_conjugation_tables,
            self.conjugation_table_num_qubits,
            self.conjugation_table_images,
            self.conjugation_table_image_phases,
            self.rotation_axes,
            self.rotation_axis_phases,
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
def _get_packed_circuit(circuit_text: str, num_qubits: int, inverse: bool) -> PackedCircuit:
    """The packed circuit with the given text (see ``get_packed_circuit``)."""
    # collect the kind and operand of each operation, the qubits and conjugation table of each
    # local gate, the conjugation table of each gate name, and the axis of each rotation
    operation_kinds: list[int] = []
    gate_or_rotation_indices: list[int] = []
    gate_qubits: list[tuple[int, int]] = []
    gate_conjugation_tables: list[int] = []
    conjugation_table_indices: dict[str, int] = {}
    rotation_axes: list[stim.PauliString] = []

    # read the instructions, skipping annotations and rejecting what packed circuits cannot
    # represent; the inverse circuit applies the inverse operations in reverse order, so reverse
    # the instructions and their target groups for it, and invert each operation below
    instructions = stim.Circuit(circuit_text).flattened()
    for instruction in reversed(instructions) if inverse else instructions:
        if instruction.name in _ANNOTATIONS:
            continue
        is_t_rotation = _check_instruction(instruction)
        target_groups = instruction.target_groups()
        if inverse:
            target_groups = target_groups[::-1]

        # each target group of a T-tagged instruction, SPP, or SPP_DAG is a Pauli rotation, whose
        # inverse is the rotation about the negated axis
        if is_t_rotation or instruction.name in ("SPP", "SPP_DAG"):
            kind = T_ROTATION if is_t_rotation else S_ROTATION
            sign = _ROTATION_SIGNS[instruction.name] * (-1 if inverse else 1)
            for target_group in target_groups:
                operation_kinds.append(kind)
                gate_or_rotation_indices.append(len(rotation_axes))
                rotation_axes.append(sign * _get_axis(instruction, target_group))
            continue

        # each target group of another gate, including an untagged S or S_DAG, is a local gate,
        # which shares a conjugation table with the gates of the same name; the inverse uses the
        # inverse gate
        gate_name = stim.gate_data(instruction.name).inverse.name if inverse else instruction.name
        conjugation_table = conjugation_table_indices.setdefault(
            gate_name, len(conjugation_table_indices)
        )
        for target_group in target_groups:
            qubits = [target.value for target in target_group]
            operation_kinds.append(LOCAL_GATE)
            gate_or_rotation_indices.append(len(gate_conjugation_tables))
            gate_qubits.append((qubits[0], qubits[1] if len(qubits) > 1 else -1))
            gate_conjugation_tables.append(conjugation_table)

    # build the conjugation tables of the local gates, padded to the 16 entries of two qubits
    conjugation_table_names = tuple(conjugation_table_indices)
    num_conjugation_tables = len(conjugation_table_names)
    conjugation_table_num_qubits = np.zeros(num_conjugation_tables, dtype=np.int64)
    conjugation_table_images = np.zeros((num_conjugation_tables, 16), dtype=np.int64)
    conjugation_table_image_phases = np.zeros((num_conjugation_tables, 16), dtype=np.int64)
    for conjugation_table, gate_name in enumerate(conjugation_table_names):
        images, image_phases = packed.get_conjugation_table(gate_name)
        conjugation_table_num_qubits[conjugation_table] = len(
            stim.Tableau.from_named_gate(gate_name)
        )
        conjugation_table_images[conjugation_table, : len(images)] = images
        conjugation_table_image_phases[conjugation_table, : len(image_phases)] = image_phases

    # pack the rotation axes
    packed_axes, packed_axis_phases = packed.pack_axes(rotation_axes, num_qubits)

    # assemble the packed circuit, whose arrays must be read-only, since it is cached and shared
    packed_circuit = PackedCircuit(
        num_qubits=num_qubits,
        operation_kinds=np.array(operation_kinds, dtype=np.int64),
        gate_or_rotation_indices=np.array(gate_or_rotation_indices, dtype=np.int64),
        gate_qubits=np.array(gate_qubits, dtype=np.int64).reshape(-1, 2),
        gate_conjugation_tables=np.array(gate_conjugation_tables, dtype=np.int64),
        conjugation_table_names=conjugation_table_names,
        conjugation_table_num_qubits=conjugation_table_num_qubits,
        conjugation_table_images=conjugation_table_images,
        conjugation_table_image_phases=conjugation_table_image_phases,
        rotation_axes=packed_axes,
        rotation_axis_phases=packed_axis_phases,
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
    if tag and not is_t_rotation and instruction.name in _TSIM_TAGGED_GATES:
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
    instruction: stim.CircuitInstruction, target_group: list[stim.GateTarget]
) -> stim.PauliString:
    """The axis of a target group of a Pauli rotation, with its sign.

    The targets of ``S`` and ``S_DAG`` are qubits, each with the axis ``Z``.  The targets of ``SPP``
    and ``SPP_DAG`` are Pauli targets, each of which negates the axis if it is inverted.

    Raises:
        ValueError: If the target group repeats a qubit.
    """
    letters = {
        target.value: "Z" if target.is_qubit_target else target.pauli_type
        for target in target_group
    }
    if len(letters) < len(target_group):
        raise ValueError(f"Pauli products must not repeat qubits: {instruction}")
    num_inversions = sum(target.is_inverted_result_target for target in target_group)
    return (-1) ** num_inversions * stim.PauliString(letters)
