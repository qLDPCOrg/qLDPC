# SPDX-License-Identifier: Apache-2.0

"""Transpiling noiseless circuits into CZ gates and single-qubit gates.

``transpile_to_cz`` rewrites the untagged unitary gates of a noiseless circuit into CZ gates and
single-qubit gates, as follows.

- Each two-qubit gate becomes single-qubit gates around at most one CZ, and possibly a SWAP.
- Single-qubit gates are combined and delayed until a later instruction needs them.  Before a CZ,
  the delayed gate of a qubit splits into H, SQRT_X, or nothing, which is applied before the CZ,
  and a diagonal gate, possibly times X, which moves past the CZ.
- Delayed gates merge into later single-qubit measurements, which change their bases and may
  invert their results, and into the Pauli targets of later observables.  Resets discard them.
- SWAP gates are free: instead of moving states between qubits, the transpiler relabels qubits, so
  that SWAP gates and the SWAP parts of gates such as ISWAP leave the circuit.  SWAP gates at the
  end of the circuit return all states to their qubits.
- Tagged instructions, such as noiseless instructions tagged with ``DEFAULT_IMMUNE_OP_TAG``, and
  multi-qubit measurements are copied as they are, after the delayed gates of their qubits, except
  that their qubit targets are relabeled.
- Qubits that are immune to noise (as declared for ``NoiseModel.noisy_circuit``) are never
  relabeled to qubits that are not, so a SWAP between them stays in the circuit.
- ``REPEAT`` blocks are kept.  All delayed gates are applied and all qubit labels are restored
  before each ``REPEAT`` block and at the end of each iteration, with a warning if that adds gates.
  Flattening the circuit (with ``stim.Circuit.flattened``) avoids these gates.
- Detectors and the annotations ``QUBIT_COORDS``, ``SHIFT_COORDS``, and ``MPAD`` are copied as
  they are, so coordinates refer to the qubits of the output.  ``TICK`` instructions are dropped.

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
# The basis of each single-qubit measurement or measure-reset, and a Pauli that flips that basis.
_MEASUREMENTS = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
_FLIPS = {"X": "Z", "Y": "Z", "Z": "X"}
_RESETS = frozenset({"R", "RX", "RY"})
_MULTI_QUBIT_MEASUREMENTS = frozenset({"MPP", "MXX", "MYY", "MZZ"})
# Instructions that are noisy if they have arguments, and instructions that are copied as they are.
_NOISY_WITH_ARGUMENTS = frozenset({*_MEASUREMENTS, *_MULTI_QUBIT_MEASUREMENTS, "MPAD"})
_ANNOTATIONS = frozenset({"DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"})
# For each axis, a gate G for which G^dag @ Z @ G is +/- that axis.
_AXIS_GATES = {"X": "H", "Y": "SQRT_X", "Z": "I"}

####################################################################################################
# transpilation


def transpile_to_cz(
    circuit: stim.Circuit,
    *,
    immune_qubits: Collection[int] = (),
    immune_qubit_tag: str | None = DEFAULT_IMMUNE_QUBIT_TAG,
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
    transpiler.run(circuit)
    transpiler.settle()
    return transpiler.circuit


class _Transpiler:
    """Transpile instructions, delaying single-qubit gates and relabeling qubits.

    Attributes:
        circuit: The transpiled circuit so far.
        wires: The qubit of the output that holds the state of each qubit of the input.
        delayed: The single-qubit gate that each qubit of the output still needs after circuit.
        immune: The qubits that are immune to noise.
    """

    def __init__(self, num_qubits: int, immune: frozenset[int]) -> None:
        self.circuit = stim.Circuit()
        self.wires = list(range(num_qubits))
        self.delayed = [_get_tableau("I")] * num_qubits
        self.immune = immune

    def run(self, circuit: stim.Circuit) -> None:
        """Transpile the instructions of a circuit."""
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                self._repeat(instruction)
            else:
                self._apply(instruction)

    def settle(self) -> bool:
        """Apply all delayed gates and restore all qubit labels.  Return whether that adds gates."""
        added_gates = False
        for wire in range(len(self.delayed)):
            added_gates |= self._flush(wire)
        holders = {wire: qubit for qubit, wire in enumerate(self.wires)}
        for qubit in range(len(self.wires)):
            wire = self.wires[qubit]
            if wire != qubit:
                # move the state of this qubit back from its wire, and the state on its wire away
                other = holders[qubit]
                self.circuit.append("SWAP", [wire, qubit])
                self.wires[qubit], self.wires[other] = qubit, wire
                holders[qubit], holders[wire] = qubit, other
                added_gates = True
        return added_gates

    def _apply(self, instruction: stim.CircuitInstruction) -> None:
        """Transpile an instruction that is not a REPEAT block."""
        name, targets = instruction.name, instruction.targets_copy()
        data = stim.gate_data(name)
        if (data.is_noisy_gate and name not in _NOISY_WITH_ARGUMENTS) or (
            name in _NOISY_WITH_ARGUMENTS and any(instruction.gate_args_copy())
        ):
            raise ValueError(f"Noise is not supported: {instruction}")
        if name == "TICK":
            return
        if name in _ANNOTATIONS:
            self.circuit.append(instruction)
        elif name in _RESETS:
            for target in targets:
                self.delayed[self.wires[target.value]] = _get_tableau("I")
            self._append_relabeled(instruction)
        elif instruction.tag or name in _MULTI_QUBIT_MEASUREMENTS:
            for target in targets:
                if target.qubit_value is not None:
                    self._flush(self.wires[target.qubit_value])
            self._append_relabeled(instruction)
        elif data.is_unitary and any(
            target.is_measurement_record_target or target.is_sweep_bit_target for target in targets
        ):
            raise ValueError(
                f"Untagged classically controlled gates are not supported: {instruction}"
            )
        elif data.is_unitary and data.is_single_qubit_gate:
            for target in targets:
                wire = self.wires[target.value]
                self.delayed[wire] = self.delayed[wire].then(_get_tableau(name))
        elif data.is_unitary and data.is_two_qubit_gate:
            for target_0, target_1 in zip(targets[::2], targets[1::2]):
                self._apply_two_qubit_gate(name, target_0.value, target_1.value)
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

    def _apply_two_qubit_gate(self, name: str, qubit_0: int, qubit_1: int) -> None:
        """Apply a two-qubit gate as single-qubit gates, at most one CZ, and maybe a relabeling."""
        before_0, before_1, has_cz, after_0, after_1, has_swap = _decompose_two_qubit_gate(name)
        wire_0, wire_1 = self.wires[qubit_0], self.wires[qubit_1]
        gate_0, gate_1 = self.delayed[wire_0].then(before_0), self.delayed[wire_1].then(before_1)
        if has_cz:
            first_0, first_1, gate_0, gate_1 = _commute_through_cz(gate_0, gate_1)
            self._emit(wire_0, first_0)
            self._emit(wire_1, first_1)
            self.circuit.append("CZ", [wire_0, wire_1])
        self.delayed[wire_0], self.delayed[wire_1] = gate_0.then(after_0), gate_1.then(after_1)
        if has_swap and (wire_0 in self.immune) == (wire_1 in self.immune):
            self.wires[qubit_0], self.wires[qubit_1] = wire_1, wire_0
        elif has_swap:
            # swapping states between immune and other qubits takes a SWAP, which the delayed gates
            # of the two qubits follow
            self.circuit.append("SWAP", [wire_0, wire_1])
            self.delayed[wire_0], self.delayed[wire_1] = self.delayed[wire_1], self.delayed[wire_0]

    def _measure(self, name: str, target: stim.GateTarget) -> None:
        """Measure a qubit in the basis that absorbs its delayed gate."""
        basis = _MEASUREMENTS[name]
        wire = self.wires[target.value]
        axis, flipped = _pull_back(self.delayed[wire], basis)
        prefix = "MR" if name.startswith("MR") else "M"
        invert = target.is_inverted_result_target ^ flipped
        self.circuit.append(
            prefix + axis.replace("Z", ""), [stim.target_inv(wire) if invert else wire]
        )
        if prefix == "MR" and flipped:
            # the delayed gate maps the reset state to the -1 eigenstate of the original basis
            self.delayed[wire] = self.delayed[wire].then(_get_tableau(_FLIPS[basis]))

    def _read_pauli(self, target: stim.GateTarget) -> stim.GateTarget:
        """Rewrite a Pauli target to read the same Pauli before the delayed gate of its qubit."""
        wire = self.wires[target.value]
        axis, flipped = _pull_back(self.delayed[wire], target.pauli_type)
        return stim.target_pauli(wire, axis, invert=target.is_inverted_result_target ^ flipped)

    def _flush(self, wire: int) -> bool:
        """Apply the delayed gate of a qubit.  Return whether it is not the identity."""
        gate, self.delayed[wire] = self.delayed[wire], _get_tableau("I")
        return self._emit(wire, gate)

    def _emit(self, wire: int, gate: stim.Tableau) -> bool:
        """Append a single-qubit gate unless it is the identity.  Return whether it was appended."""
        if gate == _get_tableau("I"):
            return False
        self.circuit.append(_get_gate_names()[_get_key(gate)], [wire])
        return True

    def _append_relabeled(self, instruction: stim.CircuitInstruction) -> None:
        """Append an instruction with qubit targets on the qubits of the output that hold them."""
        targets = instruction.targets_copy()
        qubit_map = {
            target.qubit_value: self.wires[target.qubit_value]
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
        """Transpile a REPEAT block, starting and ending each iteration with settled qubits."""
        body = _Transpiler(len(self.wires), self.immune)
        body.run(block.body_copy())
        added_gates_before = self.settle()
        added_gates_within = body.settle()
        if added_gates_before or added_gates_within:
            warnings.warn(
                "REPEAT support of transpile_to_cz is limited: it applies all delayed single-qubit"
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
    before = _get_tableau(before_0) + _get_tableau(before_1)
    if has_cz:
        before = before.then(_get_tableau("CZ"))
    gate = _get_tableau(name)
    after = before.inverse().then(gate.then(_get_tableau("SWAP")) if has_swap else gate)
    if any(
        image[1 - qubit]
        for qubit in (0, 1)
        for image in (after.x_output(qubit), after.z_output(qubit))
    ):
        return None
    after_0, after_1 = (_get_tableau(_get_gate_names()[_get_key(after, qubit)]) for qubit in (0, 1))
    return _get_tableau(before_0), _get_tableau(before_1), has_cz, after_0, after_1, has_swap
