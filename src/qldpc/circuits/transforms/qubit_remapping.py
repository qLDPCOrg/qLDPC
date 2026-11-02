# SPDX-License-Identifier: Apache-2.0

"""Relabeling the qubits of circuits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import stim

from ..common import StimCircuitLike

####################################################################################################
# qubit remapping


def with_remapped_qubits(
    circuit: StimCircuitLike,
    qubit_map: Mapping[int, int] | Sequence[int],
    *,
    inverse: bool = False,
) -> StimCircuitLike:
    """The same circuit, but with relabeled qubits.

    Qubits not in qubit_map get mapped to themselves.

    Args:
        circuit: The circuit to remap.
        qubit_map: Either a mapping (e.g., dictionary) from old to new qubit indices, or a sequence
            for which the qubit at index old_index gets mapped to new_index = qubit_map[old_index].
        inverse: If True, invert the provided qubit_map.  Default: False.

    Returns:
        The input circuit with remapped qubits.
    """
    qubit_map = (
        qubit_map
        if isinstance(qubit_map, Mapping)
        else {old_index: new_index for old_index, new_index in enumerate(qubit_map)}
    )
    if len(set(qubit_map.values())) != len(qubit_map):
        raise ValueError("qubit_map must be injective")
    if inverse:
        qubit_map = {val: key for key, val in qubit_map.items()}

    new_circuit = type(circuit)()  # same type as input circuit
    for index in range(len(circuit)):
        op = circuit[index]
        if isinstance(op, stim.CircuitRepeatBlock):
            block = stim.CircuitRepeatBlock(
                repeat_count=op.repeat_count,
                body=with_remapped_qubits(op.body_copy(), qubit_map),
                tag=op.tag,
            )
            new_circuit.append(block)

        else:
            # MPAD targets encode padding bits, not qubit indices, despite stim exposing them as
            # qubit targets.  Relabeling them changes the sampled padding values.
            new_targets = (
                op.targets_copy()
                if op.name == "MPAD"
                else [remap_qubit_target(target, qubit_map) for target in op.targets_copy()]
            )
            new_op = stim.CircuitInstruction(
                name=op.name, targets=new_targets, gate_args=op.gate_args_copy(), tag=op.tag
            )
            new_circuit.append(new_op)

    return new_circuit


def remap_qubit_target(target: stim.GateTarget, qubit_map: Mapping[int, int]) -> stim.GateTarget:
    """Remap the qubit addressed by a stim.GateTarget, if any.

    ``MPAD`` padding-bit targets are not qubits and are intentionally preserved by
    :func:`with_remapped_qubits` before this helper is called.
    """
    if target.qubit_value is None:
        return target

    new_qubit_value = qubit_map.get(target.qubit_value, target.qubit_value)
    if target.is_x_target or target.is_z_target or target.is_y_target:
        return stim.target_pauli(
            new_qubit_value,
            target.pauli_type,
            invert=target.is_inverted_result_target,
        )

    if target.is_inverted_result_target:
        return stim.target_inv(new_qubit_value)

    return stim.GateTarget(new_qubit_value)
