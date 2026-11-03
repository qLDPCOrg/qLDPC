# SPDX-License-Identifier: Apache-2.0

"""Pack Clifford+T circuits into arrays of one- and two-qubit Clifford gates and Pauli rotations.

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
_ANNOTATIONS = frozenset({"TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"})

# The instructions that are Pauli rotations, with or without a T tag, mapped to the signs of their
# axes.
_ROTATION_SIGNS = {"S": 1, "S_DAG": -1, "SPP": 1, "SPP_DAG": -1}

# The instructions whose tags tsim reads: tags other than T on them make rotations that packed
# circuits cannot represent.
_TSIM_TAGGED_GATES = frozenset({"I", *_ROTATION_SIGNS})

# The tag that makes S, S_DAG, SPP, and SPP_DAG instructions π/4 rotations, alone or followed by a
# label as in "T:<label>" (see the docstring of ``RotatedPauliString``).
T_TAG = "T"

# The types of operations of a packed circuit (see ``PackedCircuit``), compared only for equality.
LOCAL_GATE = 0
S_ROTATION = 1
T_ROTATION = 2

####################################################################################################
# packed circuits


@dataclasses.dataclass(frozen=True)
class PackedCircuit:
    """The unitary operations of a Clifford+T circuit, as arrays, in the order in which they act.

    Each operation has one of the following types, which ``operation_types`` lists.

    - ``LOCAL_GATE``: one of stim's one- and two-qubit Clifford gates, including untagged ``S``
      gates.  A local gate on ``k`` qubits is applied through its *conjugation table*: a lookup
      table that lists, for each of the ``4^k`` Pauli strings on those qubits, its image under
      conjugation by the gate.
    - ``S_ROTATION``: a π/2 rotation about a Pauli product, from an untagged ``SPP`` or ``SPP_DAG``
      instruction.  Such a rotation is Clifford, but its conjugation table would have ``4^m``
      entries for a product on ``m`` qubits, so conjugation by it goes through its axis ``a``
      instead: it maps a Pauli string ``b`` to ``-i a b`` if ``b`` anticommutes with ``a``, and
      leaves ``b`` unchanged otherwise.
    - ``T_ROTATION``: a π/4 rotation about a Pauli product, from a T-tagged instruction.

    Operations are indexed in the order in which they act, and so are rotations.  The conjugation
    tables of local gates depend only on their stim gates, so packed circuits share those of
    ``_packed_paulis.get_conjugation_tables``, which list, for the index of each Pauli string on
    the qubits of a gate, the index and the phase of its image.  The number of qubits of a local
    gate, and so of its conjugation table, follows from its row of ``operation_qubits``.  All arrays
    are read-only, since packed circuits are cached and shared.

    Attributes:
        operation_types: Indexed by operation, the type of each operation, ``LOCAL_GATE``,
            ``S_ROTATION``, or ``T_ROTATION``.
        operation_qubits: Indexed by operation, a row with the qubits of each local gate, with
            ``-1`` in place of the second qubit of a single-qubit gate, and with ``-1`` in both
            places for each rotation.
        operation_indices: Indexed by operation, the index of the stim gate of each local gate
            among the gates with shared conjugation tables, and the index of each rotation among
            the rotations.
        rotation_axes: Indexed by rotation, a row with the packed bits of its axis, a Hermitian
            Pauli string on ``num_qubits`` qubits, in the layout of ``_packed_paulis``.
        rotation_axis_phases: Indexed by rotation, the phase of its axis, in the convention of
            ``_packed_paulis``, which also carries the sign of the axis.
    """

    operation_types: npt.NDArray[np.int64]
    operation_qubits: npt.NDArray[np.int64]
    operation_indices: npt.NDArray[np.int64]
    rotation_axes: npt.NDArray[np.uint64]
    rotation_axis_phases: npt.NDArray[np.int64]

    def get_arrays(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.uint64],
        npt.NDArray[np.int64],
    ]:
        """The arrays of this packed circuit, in the order in which compiled kernels take them.

        ``rotated_pauli_string._apply_packed_circuit`` takes these arrays as its last arguments.
        """
        return (
            self.operation_types,
            self.operation_qubits,
            self.operation_indices,
            self.rotation_axes,
            self.rotation_axis_phases,
        )


def get_packed_circuit(
    circuit: stim.Circuit, num_qubits: int, *, inverse: bool = False
) -> PackedCircuit:
    """Pack a Clifford+T circuit, or its inverse, into arrays on ``num_qubits`` qubits.

    Packed circuits are cached by the text of the circuit, ``num_qubits``, and ``inverse``, so
    repeated calls with the same arguments are cheap, and later changes to a circuit object do not
    affect the packed circuits made from it before.

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
    """The packed circuit of the circuit with the given stim text (see ``get_packed_circuit``)."""
    # collect the type, qubits, and index of each operation, and the axis of each rotation
    operation_types: list[int] = []
    operation_qubits: list[tuple[int, int]] = []
    operation_indices: list[int] = []
    rotation_axes: list[stim.PauliString] = []
    gate_indices = packed.get_conjugation_tables().gate_indices

    # read the instructions, skipping annotations and rejecting instructions that packed circuits
    # cannot represent; the inverse circuit applies the inverse operations in reverse order, so
    # reverse the instructions and their target groups for it, and invert each operation below
    instructions = stim.Circuit(circuit_text).flattened()
    for instruction in reversed(instructions) if inverse else instructions:
        if instruction.name in _ANNOTATIONS:
            continue
        _check_instruction(instruction)
        is_t_rotation = _has_t_tag(instruction)
        target_groups = instruction.target_groups()
        if inverse:
            target_groups = target_groups[::-1]

        # each target group of a T-tagged instruction, SPP, or SPP_DAG is a Pauli rotation, whose
        # inverse is the rotation about the negated axis
        if is_t_rotation or instruction.name in ("SPP", "SPP_DAG"):
            operation_type = T_ROTATION if is_t_rotation else S_ROTATION
            sign = _ROTATION_SIGNS[instruction.name] * (-1 if inverse else 1)
            for target_group in target_groups:
                operation_types.append(operation_type)
                operation_qubits.append((-1, -1))
                operation_indices.append(len(rotation_axes))
                rotation_axes.append(sign * _get_axis(instruction, target_group))
            continue

        # each target group of another gate, including an untagged S or S_DAG, is a local gate,
        # indexed by its stim gate among the gates with shared conjugation tables; the inverse uses
        # the inverse gate
        gate_name = stim.gate_data(instruction.name).inverse.name if inverse else instruction.name
        for target_group in target_groups:
            qubits = [target.value for target in target_group]
            operation_types.append(LOCAL_GATE)
            operation_qubits.append((qubits[0], qubits[1] if len(qubits) > 1 else -1))
            operation_indices.append(gate_indices[gate_name])

    # pack the rotation axes
    packed_axes, packed_axis_phases = packed.pack_axes(rotation_axes, num_qubits)

    # assemble the packed circuit, and make its arrays read-only, since it is cached and shared
    packed_circuit = PackedCircuit(
        operation_types=np.array(operation_types, dtype=np.int64),
        operation_qubits=np.array(operation_qubits, dtype=np.int64).reshape(-1, 2),
        operation_indices=np.array(operation_indices, dtype=np.int64),
        rotation_axes=packed_axes,
        rotation_axis_phases=packed_axis_phases,
    )
    for array in packed_circuit.get_arrays():
        array.flags.writeable = False
    return packed_circuit


def _check_instruction(instruction: stim.CircuitInstruction) -> None:
    """Check that packed circuits can represent an instruction that is not an annotation.

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
    has_t_tag = _has_t_tag(instruction)
    if has_t_tag and instruction.name not in _ROTATION_SIGNS:
        raise ValueError(f"Instruction cannot be a π/4 rotation: {instruction}")
    if instruction.tag and not has_t_tag and instruction.name in _TSIM_TAGGED_GATES:
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


def _has_t_tag(instruction: stim.CircuitInstruction) -> bool:
    """Whether an instruction has the tag ``T`` or a labeled tag ``T:<label>``."""
    return instruction.tag == T_TAG or instruction.tag.startswith(T_TAG + ":")


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
