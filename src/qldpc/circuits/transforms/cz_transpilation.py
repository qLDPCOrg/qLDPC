# SPDX-License-Identifier: Apache-2.0

"""Transpiling noiseless circuits into CZ gates and single-qubit gates.

``transpile_to_cz`` rewrites the untagged unitary gates of a noiseless circuit into CZ gates and
single-qubit gates, as follows.

- Each two-qubit gate becomes single-qubit gates around at most one CZ, and possibly a SWAP.
- Single-qubit gates are combined and kept pending until a later instruction needs them.  Before a
  CZ, the pending gate of a qubit splits into H, SQRT_X, or nothing, which is applied before the
  CZ, and a diagonal gate, possibly times X, which moves past the CZ.
- Pending gates change the bases of later single-qubit measurements, and may invert their results,
  but stay pending, so that the states after the measurements are also right.  They change the
  Pauli targets of later observables in the same way.  Resets, tagged or not, discard them.
- SWAP gates are free: instead of moving states between qubits, the transpiler relabels qubits, so
  that SWAP gates and the SWAP parts of gates such as ISWAP leave the circuit.  SWAP gates at the
  end of the circuit return all states to their qubits.
- Other tagged instructions, such as noiseless instructions tagged with ``DEFAULT_IMMUNE_OP_TAG``,
  and multi-qubit measurements are copied as they are, after the pending gates of their qubits,
  except that their qubit targets are relabeled.
- Qubits that are immune to noise (as declared for ``NoiseModel.noisy_circuit``) are never
  relabeled to qubits that are not, so a SWAP between them stays in the circuit.
- ``REPEAT`` blocks are kept.  All pending gates are applied and all qubit labels are restored
  before each ``REPEAT`` block and at the end of each iteration, with a warning if that adds gates.
  Flattening the circuit (with ``stim.Circuit.flattened``) avoids these gates.
- Detectors and the annotations ``QUBIT_COORDS``, ``SHIFT_COORDS``, and ``MPAD`` are copied as
  they are, so coordinates refer to the qubits of the output.  ``TICK`` instructions are dropped.

By default, ``transpile_to_cz`` then cleans up the transpiled circuit.  Between instructions other
than untagged unitary gates, which stay in place, the cleanup cancels pairs of CZ gates on the same
qubits that only diagonal gates separate, merges the first single-qubit gates on qubits after
untagged resets into the bases of these resets (leaving at most a Pauli gate), and packs gates into
layers, with CZ gates as early as possible and single-qubit gates as late as possible.

Noise and untagged classically controlled, ``SPP``, and ``SPP_DAG`` gates are not supported.
"""

from __future__ import annotations

import functools
import itertools
import warnings
from collections.abc import Collection

import stim

from ..common import DEFAULT_IMMUNE_QUBIT_TAG, _get_immune_qubits
from .qubit_remapping import remap_qubit_target

# Paulis are indexed as in stim.PauliString: 1, 2, 3 for X, Y, Z.
_PAULIS = "_XYZ"
# The basis of each single-qubit measurement, measure-reset, or reset, and a Pauli that flips it.
_MEASUREMENTS = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}
_FLIPS = {"X": "Z", "Y": "Z", "Z": "X"}
_MULTI_QUBIT_MEASUREMENTS = frozenset({"MPP", "MXX", "MYY", "MZZ"})
# Instructions that are noisy if they have arguments.
_NOISY_WITH_ARGUMENTS = frozenset({*_MEASUREMENTS, *_MULTI_QUBIT_MEASUREMENTS, "MPAD"})
# Annotations, which are copied as they are, without relabeling their qubit targets.
_ANNOTATIONS = frozenset({"DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"})
# For each axis, a gate G for which G^dag @ Z @ G is +/- that axis.
_AXIS_GATES = {"X": "H", "Y": "SQRT_X", "Z": "I"}

# A run of untagged unitary gates, each given by its name and its qubits.
_Segment = list[tuple[str, tuple[int, ...]]]

####################################################################################################
# transpilation


def transpile_to_cz(
    circuit: stim.Circuit,
    *,
    immune_qubits: Collection[int] = (),
    immune_qubit_tag: str | None = DEFAULT_IMMUNE_QUBIT_TAG,
    cleanup: bool = True,
) -> stim.Circuit:
    """Rewrite a noiseless circuit into CZ gates and single-qubit gates (see the module docstring).

    The output applies the same operation as the input: it produces the same distribution of
    measurement records and, conditioned on those records, leaves the qubits in the same state.  Its
    detectors and observables keep their meaning, since measurement records keep their order.

    Args:
        circuit: The noiseless circuit to transpile.
        immune_qubits: Qubits that are immune to noise.  Default: none.
        immune_qubit_tag: If a ``QUBIT_COORDS`` annotation contains this string in its tag, its
            qubits are also immune to noise.  If None, no tag declares immune qubits.  Default:
            ``DEFAULT_IMMUNE_QUBIT_TAG``, as for ``NoiseModel.noisy_circuit``.
        cleanup: If True, clean up the transpiled circuit (see the module docstring).  Default:
            True.

    Returns:
        The transpiled circuit.

    Raises:
        ValueError: If the circuit has noise or an untagged classically controlled, ``SPP``, or
            ``SPP_DAG`` gate, or if immune_qubit_tag is empty.

    Warns:
        UserWarning: If gates are added at the boundaries of ``REPEAT`` blocks.
    """
    immune = frozenset(immune_qubits) | _get_immune_qubits(circuit, immune_qubit_tag)
    transpiler = _Transpiler(circuit.num_qubits, immune)
    transpiler.transpile(circuit)
    transpiler.apply_pending_gates_and_restore_labels()
    return _clean_up(transpiler.circuit) if cleanup else transpiler.circuit


class _Transpiler:
    """Transpile instructions, keeping single-qubit gates pending and relabeling qubits.

    Qubits of the input circuit are called input qubits, and qubits of the transpiled circuit are
    called output qubits, or just qubits.

    Attributes:
        circuit: The transpiled circuit so far.
        output_qubits: The output qubit that holds the state of each input qubit.
        pending_gates: The single-qubit gate that each output qubit still needs after the
            transpiled circuit so far.
        immune: The output qubits that are immune to noise.
    """

    def __init__(self, num_qubits: int, immune: frozenset[int]) -> None:
        self.circuit = stim.Circuit()
        self.output_qubits = list(range(num_qubits))
        self.pending_gates = [_get_tableau("I")] * num_qubits
        self.immune = immune

    def transpile(self, circuit: stim.Circuit) -> None:
        """Transpile the instructions of a circuit."""
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                self._repeat(instruction)
            else:
                self._apply(instruction)

    def apply_pending_gates_and_restore_labels(self) -> bool:
        """Apply all pending gates and move every state back to its input qubit with SWAP gates.

        Returns:
            Whether this added any gates.
        """
        added_gates = False
        for qubit in range(len(self.pending_gates)):
            added_gates |= self._apply_pending_gate(qubit)

        # input_qubits[qubit] is the input qubit whose state the output qubit holds
        input_qubits = {qubit: input_qubit for input_qubit, qubit in enumerate(self.output_qubits)}
        for input_qubit in range(len(self.output_qubits)):
            qubit = self.output_qubits[input_qubit]
            if qubit != input_qubit:
                # swap the state of input_qubit into place, and the state in its place to qubit
                displaced_input_qubit = input_qubits[input_qubit]
                self.circuit.append("SWAP", [qubit, input_qubit])
                self.output_qubits[input_qubit] = input_qubit
                self.output_qubits[displaced_input_qubit] = qubit
                input_qubits[input_qubit] = input_qubit
                input_qubits[qubit] = displaced_input_qubit
                added_gates = True
        return added_gates

    def _apply(self, instruction: stim.CircuitInstruction) -> None:
        """Transpile an instruction that is not a REPEAT block."""
        name, targets = instruction.name, instruction.targets_copy()
        data = stim.gate_data(name)

        # reject noise, and drop or copy instructions that do not act on states
        if (data.is_noisy_gate and name not in _NOISY_WITH_ARGUMENTS) or (
            name in _NOISY_WITH_ARGUMENTS and any(instruction.gate_args_copy())
        ):
            raise ValueError(f"Noise is not supported: {instruction}")
        if name == "TICK":
            return
        if name in _ANNOTATIONS:
            self.circuit.append(instruction)

        # resets discard the pending gates of their qubits, and other tagged instructions and
        # multi-qubit measurements come after these gates
        elif name in _RESETS:
            for target in targets:
                self.pending_gates[self.output_qubits[target.value]] = _get_tableau("I")
            self._append_relabeled(instruction)
        elif instruction.tag or name in _MULTI_QUBIT_MEASUREMENTS:
            for target in targets:
                if target.qubit_value is not None:
                    self._apply_pending_gate(self.output_qubits[target.qubit_value])
            self._append_relabeled(instruction)

        # reject untagged classically controlled gates, combine single-qubit gates with pending
        # gates, and decompose two-qubit gates
        elif data.is_unitary and any(
            target.is_measurement_record_target or target.is_sweep_bit_target for target in targets
        ):
            raise ValueError(
                f"Untagged classically controlled gates are not supported: {instruction}"
            )
        elif data.is_unitary and data.is_single_qubit_gate:
            for target in targets:
                qubit = self.output_qubits[target.value]
                self.pending_gates[qubit] = self.pending_gates[qubit].then(_get_tableau(name))
        elif data.is_unitary and data.is_two_qubit_gate:
            for target_0, target_1 in zip(targets[::2], targets[1::2]):
                self._apply_two_qubit_gate(name, target_0.value, target_1.value)

        # single-qubit measurements and observables read their qubits through pending gates
        elif name in _MEASUREMENTS:
            for target in targets:
                self._measure(name, target)
        elif name == "OBSERVABLE_INCLUDE":
            new_targets = [
                target if target.is_measurement_record_target else self._read_pauli(target)
                for target in targets
            ]
            self.circuit.append(name, new_targets, instruction.gate_args_copy())
        else:
            raise ValueError(f"Untagged {name} gates are not supported")

    def _apply_two_qubit_gate(self, name: str, input_qubit_0: int, input_qubit_1: int) -> None:
        """Apply a two-qubit gate as single-qubit gates, at most one CZ, and maybe a relabeling."""
        before_0, before_1, has_cz, after_0, after_1, has_swap = _decompose_two_qubit_gate(name)
        qubit_0, qubit_1 = self.output_qubits[input_qubit_0], self.output_qubits[input_qubit_1]
        gate_0 = self.pending_gates[qubit_0].then(before_0)
        gate_1 = self.pending_gates[qubit_1].then(before_1)

        # if there is a CZ, append it after the parts of the pending gates that cannot move past
        # it, and keep the other parts pending, followed by the gates after the CZ
        if has_cz:
            first_0, first_1, gate_0, gate_1 = _commute_through_cz(gate_0, gate_1)
            self._append_gate(qubit_0, first_0)
            self._append_gate(qubit_1, first_1)
            self.circuit.append("CZ", [qubit_0, qubit_1])
        self.pending_gates[qubit_0] = gate_0.then(after_0)
        self.pending_gates[qubit_1] = gate_1.then(after_1)

        # relabel the two qubits instead of swapping them, unless only one of them is immune
        if has_swap and (qubit_0 in self.immune) == (qubit_1 in self.immune):
            self.output_qubits[input_qubit_0] = qubit_1
            self.output_qubits[input_qubit_1] = qubit_0
        elif has_swap:
            self.circuit.append("SWAP", [qubit_0, qubit_1])
            self.pending_gates[qubit_0], self.pending_gates[qubit_1] = (
                self.pending_gates[qubit_1],
                self.pending_gates[qubit_0],
            )

    def _measure(self, name: str, target: stim.GateTarget) -> None:
        """Measure a qubit through its pending gate, which stays pending.

        The measurement changes its basis, and may invert its result, to measure the original basis
        after the pending gate.
        """
        basis = _MEASUREMENTS[name]
        qubit = self.output_qubits[target.value]
        axis, flipped = _pull_back(self.pending_gates[qubit], basis)
        prefix = "MR" if name.startswith("MR") else "M"
        invert = target.is_inverted_result_target ^ flipped
        self.circuit.append(
            prefix + axis.replace("Z", ""), [stim.target_inv(qubit) if invert else qubit]
        )

        # after a measure-reset, the pending gate maps the reset state to the -1 eigenstate of the
        # original basis if flipped, so a Pauli gate that flips that basis must follow it
        if prefix == "MR" and flipped:
            self.pending_gates[qubit] = self.pending_gates[qubit].then(_get_tableau(_FLIPS[basis]))

    def _read_pauli(self, target: stim.GateTarget) -> stim.GateTarget:
        """Rewrite a Pauli target to read the same Pauli before the pending gate of its qubit."""
        qubit = self.output_qubits[target.value]
        axis, flipped = _pull_back(self.pending_gates[qubit], target.pauli_type)
        return stim.target_pauli(qubit, axis, invert=target.is_inverted_result_target ^ flipped)

    def _apply_pending_gate(self, qubit: int) -> bool:
        """Append the pending gate of a qubit.  Return whether it is not the identity."""
        gate, self.pending_gates[qubit] = self.pending_gates[qubit], _get_tableau("I")
        return self._append_gate(qubit, gate)

    def _append_gate(self, qubit: int, gate: stim.Tableau) -> bool:
        """Append a single-qubit gate unless it is the identity.  Return whether it was appended."""
        if gate == _get_tableau("I"):
            return False
        self.circuit.append(_get_gate_names()[_get_key(gate)], [qubit])
        return True

    def _append_relabeled(self, instruction: stim.CircuitInstruction) -> None:
        """Append an instruction with its qubit targets moved to the qubits that hold them."""
        targets = instruction.targets_copy()
        qubit_map = {
            target.qubit_value: self.output_qubits[target.qubit_value]
            for target in targets
            if target.qubit_value is not None
        }
        self.circuit.append(
            stim.CircuitInstruction(
                instruction.name,
                [remap_qubit_target(target, qubit_map) for target in targets],
                instruction.gate_args_copy(),
                tag=instruction.tag,
            )
        )

    def _repeat(self, block: stim.CircuitRepeatBlock) -> None:
        """Transpile a REPEAT block, with no pending gates or relabeled qubits at its boundaries."""
        # transpile one iteration from a state with no pending gates or relabeled qubits
        body = _Transpiler(len(self.output_qubits), self.immune)
        body.transpile(block.body_copy())

        # return to that state before the block and at the end of each iteration
        added_gates_before = self.apply_pending_gates_and_restore_labels()
        added_gates_within = body.apply_pending_gates_and_restore_labels()
        if added_gates_before or added_gates_within:
            warnings.warn(
                "REPEAT support of transpile_to_cz is limited: it applies all pending single-qubit"
                " gates and restores all qubit labels at the boundaries of REPEAT blocks, which"
                " added gates.  Flatten the circuit (with stim.Circuit.flattened) to avoid them.",
                stacklevel=2,
            )
        self.circuit.append(
            stim.CircuitRepeatBlock(block.repeat_count, body.circuit, tag=block.tag)
        )


def _commute_through_cz(
    gate_0: stim.Tableau, gate_1: stim.Tableau
) -> tuple[stim.Tableau, stim.Tableau, stim.Tableau, stim.Tableau]:
    """Move as much of the single-qubit gates before a CZ as possible past the CZ.

    Returns:
        Gates first_0 and first_1, each of which is I, H, or SQRT_X, and gates last_0 and last_1,
        such that applying gate_0 and gate_1 and then the CZ is the same as applying first_0 and
        first_1, then the CZ, and then last_0 and last_1.
    """
    firsts, rests = zip(*(_split_off_diagonal_part(gate) for gate in (gate_0, gate_1)))

    # each rest is diagonal, possibly times X, and X moves past the CZ as X and Z on the other qubit
    has_x = [rest.z_output(0).sign == -1 for rest in rests]
    last_0 = rests[0].then(_get_tableau("Z")) if has_x[1] else rests[0]
    last_1 = rests[1].then(_get_tableau("Z")) if has_x[0] else rests[1]
    return firsts[0], firsts[1], last_0, last_1


def _pull_back(gate: stim.Tableau, pauli: str) -> tuple[str, bool]:
    """The Pauli gate^dag @ pauli @ gate, as an axis and whether it has a negative sign."""
    image = gate.inverse()(stim.PauliString(pauli))
    return _PAULIS[image[0]], image.sign == -1


def _split_off_diagonal_part(gate: stim.Tableau) -> tuple[stim.Tableau, stim.Tableau]:
    """Split a single-qubit gate into I, H, or SQRT_X followed by a gate that maps Z to +Z or -Z."""
    axis = _PAULIS[gate.inverse().z_output(0)[0]]
    first = _get_tableau(_AXIS_GATES[axis])
    return first, first.inverse().then(gate)


####################################################################################################
# cleanup


def _clean_up(circuit: stim.Circuit) -> stim.Circuit:
    """Clean up a transpiled circuit (see the module docstring).

    The cleanup collects segments of untagged unitary gates between other instructions, which stay
    in place, and cleans up each segment.  Single-qubit gates stay pending past instructions that do
    not address their qubits.
    """
    operations: list[stim.CircuitInstruction | stim.CircuitRepeatBlock] = []
    segment: _Segment = []
    # the pending single-qubit gate of each qubit, and the index in operations of the last untagged
    # reset of each qubit, if nothing has acted on that qubit since
    pending_gates: dict[int, stim.Tableau] = {}
    open_resets: dict[int, int] = {}
    for item in circuit:
        # collect untagged unitary gates into the current segment
        if isinstance(item, stim.CircuitInstruction) and _is_untagged_gate(item):
            segment.extend(
                (item.name, tuple(target.value for target in group))
                for group in item.target_groups()
            )
            continue

        # clean up the segment, and apply the pending gates of the qubits that the next item uses
        qubits = (
            None
            if isinstance(item, stim.CircuitRepeatBlock)
            else {t.qubit_value for t in item.targets_copy() if t.qubit_value is not None}
        )
        _schedule(_cancel_cz_pairs(segment), operations, pending_gates, open_resets, qubits)
        segment = []

        # append the next item, cleaning up REPEAT blocks, and keep track of the untagged resets
        # that can still absorb later gates
        if isinstance(item, stim.CircuitRepeatBlock):
            body = _clean_up(item.body_copy())
            operations.append(stim.CircuitRepeatBlock(item.repeat_count, body, tag=item.tag))
            open_resets.clear()
        elif item.name in _RESETS and not item.tag:
            for target in item.targets_copy():
                open_resets[target.value] = len(operations)
                operations.append(stim.CircuitInstruction(item.name, [target]))
        else:
            operations.append(item)
            for target in item.targets_copy():
                open_resets.pop(target.qubit_value, None)

    # clean up the last segment, and apply all pending gates
    _schedule(_cancel_cz_pairs(segment), operations, pending_gates, open_resets, None)
    cleaned_circuit = stim.Circuit()
    for operation in operations:
        cleaned_circuit.append(operation)
    return cleaned_circuit


def _is_untagged_gate(instruction: stim.CircuitInstruction) -> bool:
    """Whether an instruction is an untagged unitary gate."""
    return not instruction.tag and stim.gate_data(instruction.name).is_unitary


def _cancel_cz_pairs(segment: _Segment) -> _Segment:
    """Remove the pairs of CZ gates on the same qubits that only diagonal gates separate."""
    keep = [True] * len(segment)
    # the index of the last CZ on each pair of qubits, if only diagonal gates acted on them since
    open_czs: dict[frozenset[int], int] = {}
    for index, (name, qubits) in enumerate(segment):
        pair = frozenset(qubits)
        if name == "CZ" and pair in open_czs:
            keep[open_czs.pop(pair)] = keep[index] = False
        elif name == "CZ":
            open_czs[pair] = index
        elif len(qubits) > 1 or str(_get_tableau(name).z_output(0)) != "+Z":
            # a gate that is not diagonal blocks the cancellation of CZ gates on its qubits
            for open_pair in [open_pair for open_pair in open_czs if open_pair & pair]:
                del open_czs[open_pair]
    return [gate for gate, kept in zip(segment, keep) if kept]


def _schedule(
    segment: _Segment,
    operations: list[stim.CircuitInstruction | stim.CircuitRepeatBlock],
    pending_gates: dict[int, stim.Tableau],
    open_resets: dict[int, int],
    qubits_to_flush: Collection[int] | None,
) -> None:
    """Append a segment of gates to operations in layers, merging gates into resets.

    Two-qubit gates come as early as possible, and single-qubit gates as late as possible: each
    qubit keeps its single-qubit gates pending until the next two-qubit gate on it.  At the end of
    the segment, the qubits in qubits_to_flush (or all qubits, if None) apply their pending gates.
    pending_gates and open_resets are as in _clean_up, and are updated in place.
    """
    # each layer has single-qubit gates on qubits, followed by two-qubit gates
    layers: list[tuple[dict[int, stim.Tableau], _Segment]] = []
    last_layers: dict[int, int] = {}

    def take_pending_gate(qubit: int) -> stim.Tableau:
        """Take the pending gate of a qubit, after merging what it can into an open reset."""
        gate = pending_gates.pop(qubit, _get_tableau("I"))
        if qubit in open_resets:
            index = open_resets.pop(qubit)
            name, gate = _merge_into_reset(operations[index].name, gate)
            operations[index] = stim.CircuitInstruction(name, [qubit])
        return gate

    for name, qubits in segment:
        # combine single-qubit gates into the pending gates of their qubits
        if len(qubits) == 1:
            pending_gates[qubits[0]] = pending_gates.get(qubits[0], _get_tableau("I")).then(
                _get_tableau(name)
            )
            continue

        # place a two-qubit gate in the first layer after the last two-qubit gates on its qubits,
        # after the parts of their pending gates that cannot move past it
        layer = 1 + max(last_layers.get(qubit, -1) for qubit in qubits)
        if layer == len(layers):
            layers.append(({}, []))
        gates = [take_pending_gate(qubit) for qubit in qubits]
        if name == "CZ":
            gates[0], gates[1], pending_gates[qubits[0]], pending_gates[qubits[1]] = (
                _commute_through_cz(*gates)
            )
        for qubit, gate in zip(qubits, gates):
            layers[layer][0][qubit] = gate
            last_layers[qubit] = layer
        layers[layer][1].append((name, qubits))

    # apply the pending gates of the qubits to flush in a last layer
    flushed = [q for q in sorted(pending_gates) if qubits_to_flush is None or q in qubits_to_flush]
    layers.append(({qubit: take_pending_gate(qubit) for qubit in flushed}, []))

    # append the layers to the operations
    for single_qubit_gates, two_qubit_gates in layers:
        for qubit, gate in single_qubit_gates.items():
            if gate != _get_tableau("I"):
                gate_name = _get_gate_names()[_get_key(gate)]
                operations.append(stim.CircuitInstruction(gate_name, [qubit]))
        for name, qubits in two_qubit_gates:
            operations.append(stim.CircuitInstruction(name, qubits))


def _merge_into_reset(name: str, gate: stim.Tableau) -> tuple[str, stim.Tableau]:
    """Merge a single-qubit gate into the reset before it.

    Returns:
        The name of a reset, and a Pauli gate to apply after it, which together act like the given
        reset followed by the given gate.
    """
    # the gate maps the reset state to the +1 eigenstate of +/- the image of the reset basis
    image = gate(stim.PauliString(_RESETS[name]))
    axis = _PAULIS[image[0]]
    new_name = next(name for name, basis in _RESETS.items() if basis == axis)
    return new_name, _get_tableau(_FLIPS[axis]) if image.sign == -1 else _get_tableau("I")


####################################################################################################
# Clifford gates


@functools.cache
def _get_tableau(name: str) -> stim.Tableau:
    """The tableau of a gate (which must not be modified, since it is cached)."""
    return stim.Tableau.from_named_gate(name)


@functools.cache
def _get_gate_names() -> dict[tuple[str, str], str]:
    """The name of each single-qubit Clifford gate, indexed by its key (see _get_key)."""
    return {
        _get_key(_get_tableau(name)): name
        for name, data in stim.gate_data().items()
        if data.is_unitary and data.is_single_qubit_gate
    }


def _get_key(tableau: stim.Tableau, qubit: int = 0) -> tuple[str, str]:
    """The images of X and Z on one qubit under a tableau that does not entangle it, e.g. "-Y"."""
    x_image, z_image = tableau.x_output(qubit), tableau.z_output(qubit)
    return (
        f"{'-' if x_image.sign == -1 else '+'}{_PAULIS[x_image[qubit]]}",
        f"{'-' if z_image.sign == -1 else '+'}{_PAULIS[z_image[qubit]]}",
    )


@functools.cache
def _decompose_two_qubit_gate(
    name: str,
) -> tuple[stim.Tableau, stim.Tableau, bool, stim.Tableau, stim.Tableau, bool]:
    """Decompose a two-qubit gate into single-qubit gates, at most one CZ, and maybe a SWAP.

    Returns:
        Single-qubit gates before_0 and before_1, whether there is a CZ, single-qubit gates after_0
        and after_1, and whether there is a SWAP, such that the gate is the same as applying
        before_0 and before_1, then the CZ (if any), then after_0 and after_1, and then the SWAP (if
        any).  Decompositions without a CZ or SWAP are preferred.
    """
    decompositions = (
        _try_decomposition(name, has_cz, has_swap, before_0, before_1)
        for has_cz, has_swap in itertools.product((False, True), repeat=2)
        for before_0, before_1 in itertools.product(_AXIS_GATES.values(), repeat=2)
    )
    return next(decomposition for decomposition in decompositions if decomposition is not None)


def _try_decomposition(
    name: str, has_cz: bool, has_swap: bool, before_0: str, before_1: str
) -> tuple[stim.Tableau, stim.Tableau, bool, stim.Tableau, stim.Tableau, bool] | None:
    """Complete a decomposition of a two-qubit gate (see _decompose_two_qubit_gate), if possible."""
    # find the gate that must come after the given gates and the CZ, and before the SWAP
    before = _get_tableau(before_0) + _get_tableau(before_1)
    if has_cz:
        before = before.then(_get_tableau("CZ"))
    gate = _get_tableau(name)
    after = before.inverse().then(gate.then(_get_tableau("SWAP")) if has_swap else gate)

    # the decomposition works if that gate does not entangle its qubits
    if any(
        image[1 - qubit]
        for qubit in (0, 1)
        for image in (after.x_output(qubit), after.z_output(qubit))
    ):
        return None
    after_0, after_1 = (_get_tableau(_get_gate_names()[_get_key(after, qubit)]) for qubit in (0, 1))
    return _get_tableau(before_0), _get_tableau(before_1), has_cz, after_0, after_1, has_swap
