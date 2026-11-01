# SPDX-License-Identifier: Apache-2.0

"""Transpiling noiseless circuits into CZ gates and single-qubit gates.

``transpile_to_cz`` rewrites a noiseless Clifford circuit so that its untagged unitary gates are CZ
gates and single-qubit gates.  SWAP gates are free: instead of moving states between qubits, the
transpiler tracks which qubit holds each state, so that SWAP gates (including the SWAP parts of
gates such as ISWAP) leave the circuit and every two-qubit gate costs at most one CZ.  SWAP gates at
the end of the circuit return every state to its original qubit.

Single-qubit gates are squashed and moved as late as possible.  Before a CZ, the pending gate of a
qubit splits into a gate that is emitted (H, SQRT_X, or nothing) and a gate that maps Z to +Z or -Z
(a diagonal gate, possibly times X), which moves past the CZ, where an X picks up a Z on the other
qubit of the CZ.  A CZ cancels the previous CZ on the same qubits if nothing acts on those qubits
between them.  Pending gates merge into measurements, which change their bases and, for negative
signs, invert their results, and into earlier resets, which change their bases.  ``MXX``, ``MYY``,
and ``MZZ`` become ``MPP``, and the Pauli targets of ``OBSERVABLE_INCLUDE`` are rewritten in the
same way.

Tagged instructions can carry meaning that the transpiler does not know, such as noise immunity or
the ``S[T]`` gates of the tsim simulator, so they are opaque barriers: they keep their names,
arguments, and tags, their qubit targets are relabeled to the qubits that hold the states that they
address, and the pending gates of those qubits are emitted before them.  A tagged reset discards the
pending gates of its qubits instead.

``REPEAT`` blocks are kept.  If the pending gates or qubit labels at the start of an iteration
differ from those at the start of the next iteration, the transpiler unrolls the first iteration,
outside the block.  If that is not enough, each iteration ends with gates and SWAPs that restore the
pending gates and qubit labels at its start.

``TICK`` instructions are dropped.  Noise channels, noisy measurements, untagged classically
controlled gates, and untagged ``SPP`` and ``SPP_DAG`` gates are rejected.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Sequence

import stim

from .qubit_remapping import remap_qubit_target

# Paulis are numbered as in stim.PauliString: 1, 2, 3 for X, Y, Z.
_PAULIS = "_XYZ"
_MEASUREMENTS = {"M": 3, "MX": 1, "MY": 2}
_MEASURE_RESETS = {"MR": 3, "MRX": 1, "MRY": 2}
_RESETS = {"R": 3, "RX": 1, "RY": 2}
_PAIR_MEASUREMENTS = {"MXX": 1, "MYY": 2, "MZZ": 3}
# Instructions that take noise arguments when noisy.
_NOISY_WHEN_ARGUMENTS = frozenset(
    {*_MEASUREMENTS, *_MEASURE_RESETS, *_PAIR_MEASUREMENTS, "MPP", "MPAD"}
)
# Instructions that do not address the states of qubits, which are copied as they are.
_ANNOTATIONS = frozenset({"DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"})
# Single-qubit gates that map Z to X, Y, or Z under conjugation by their inverses.
_AXIS_REPRESENTATIVES = {1: "H", 2: "SQRT_X", 3: "I"}
# A Pauli that anticommutes with each Pauli.
_FLIPS = {1: "Z", 2: "Z", 3: "X"}

####################################################################################################
# transpilation


def transpile_to_cz(circuit: stim.Circuit) -> stim.Circuit:
    """Rewrite a noiseless circuit into CZ gates and single-qubit gates (see the module docstring).

    The output applies the same operation as the input: it produces the same distribution of
    measurement records and, conditioned on those records, leaves the qubits in the same state.  Its
    detectors and observables keep their meaning, since measurement records keep their order.

    The output consists of CZ gates, single-qubit gates, ``M``, ``MX``, ``MY``, ``MR``, ``MRX``,
    ``MRY``, ``R``, ``RX``, ``RY``, and ``MPP`` instructions, annotations, ``REPEAT`` blocks, the
    tagged instructions of the input, and SWAP gates that return states to their original qubits at
    the end of the circuit and, if necessary, at the end of each iteration of a ``REPEAT`` block.
    Single-qubit gates come as late as possible, and CZ gates as early as possible, between
    instructions that are not unitary gates.  The untagged single-qubit gates right before CZ gates
    are H or SQRT_X, except for the gates that end an iteration of a ``REPEAT`` block by restoring
    the pending gates at its start.

    Args:
        circuit: The noiseless circuit to transpile.

    Returns:
        The transpiled circuit.

    Raises:
        ValueError: If the circuit has a noise channel, a noisy measurement, an untagged
            classically controlled gate, or an untagged ``SPP`` or ``SPP_DAG`` gate.
    """
    transpiler = _Transpiler(_State.get_initial(circuit.num_qubits))
    transpiler.run(circuit)
    transpiler.restore(_State.get_initial(circuit.num_qubits))
    return _to_circuit(transpiler.items)


@dataclasses.dataclass(frozen=True)
class _State:
    """The qubits that hold the states of the qubits of the input, and their pending gates.

    Attributes:
        wires: The qubit of the output that holds the state of each qubit of the input.
        gates: The pending single-qubit gate of each qubit of the output, to apply after the gates
            emitted so far.
    """

    wires: tuple[int, ...]
    gates: tuple[int, ...]

    @staticmethod
    def get_initial(num_qubits: int) -> _State:
        """The state at the start of a circuit: no relabeled qubits and no pending gates."""
        return _State(tuple(range(num_qubits)), (_get_cliffords().identity,) * num_qubits)


@dataclasses.dataclass
class _Instruction:
    """An instruction to emit as it is, except that a reset can change its basis later."""

    name: str
    targets: list[stim.GateTarget]
    args: list[float] = dataclasses.field(default_factory=list)
    tag: str = ""


@dataclasses.dataclass(frozen=True, eq=False)
class _Gate:
    """An untagged unitary gate that the transpiler emits on qubits of the output."""

    name: str
    wires: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Repeat:
    """A REPEAT block of emitted items."""

    count: int
    items: list[_Item]
    tag: str


_Item = _Instruction | _Gate | _Repeat


class _Transpiler:
    """Transpile instructions, tracking qubit labels and pending gates (see _State)."""

    def __init__(self, state: _State) -> None:
        self.cliffords = _get_cliffords()
        self.wires = list(state.wires)
        self.gates = list(state.gates)
        # resets whose bases can still absorb the pending gates of their qubits, and their bases
        self.resets: dict[int, tuple[_Instruction, int]] = {}
        # the last emitted CZ on each qubit, if nothing else has acted on that qubit since
        self.last_czs: dict[int, _Gate] = {}
        self.items: list[_Item] = []

    def run(self, circuit: stim.Circuit) -> None:
        """Transpile the instructions of a circuit."""
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                self._repeat(instruction)
            else:
                self._apply(instruction)

    def get_state(self) -> _State:
        """The current state, after fixing the bases of all resets."""
        for wire in list(self.resets):
            self._resolve_reset(wire)
        return _State(tuple(self.wires), tuple(self.gates))

    def restore(self, state: _State) -> None:
        """Emit SWAP gates and single-qubit gates that bring this transpiler to the given state."""
        self.get_state()
        self.last_czs.clear()
        holders = {wire: qubit for qubit, wire in enumerate(self.wires)}
        for qubit, wire in enumerate(state.wires):
            current = self.wires[qubit]
            if current != wire:
                other = holders[wire]
                self.items.append(_Gate("SWAP", (current, wire)))
                self.wires[qubit], self.wires[other] = wire, current
                holders[wire], holders[current] = qubit, other
                self.gates[current], self.gates[wire] = self.gates[wire], self.gates[current]
        for wire, gate in enumerate(state.gates):
            # emit a gate that, followed by the target pending gate, applies the current one
            self._emit(wire, self.cliffords.compose[self.gates[wire]][self.cliffords.inverse[gate]])
            self.gates[wire] = gate

    def _apply(self, instruction: stim.CircuitInstruction) -> None:
        """Transpile an instruction that is not a REPEAT block."""
        name, targets = instruction.name, instruction.targets_copy()
        data = stim.gate_data(name)
        if (data.is_noisy_gate and name not in _NOISY_WHEN_ARGUMENTS) or (
            name in _NOISY_WHEN_ARGUMENTS and any(instruction.gate_args_copy())
        ):
            raise ValueError(f"Noisy instructions are not supported: {instruction}")
        if name == "TICK":
            return
        if name in _ANNOTATIONS:
            self.items.append(
                _Instruction(name, targets, instruction.gate_args_copy(), instruction.tag)
            )
        elif instruction.tag:
            self._apply_opaque(instruction)
        elif data.is_unitary and any(
            target.is_measurement_record_target or target.is_sweep_bit_target for target in targets
        ):
            raise ValueError(f"Classically controlled gates are not supported: {instruction}")
        elif data.is_unitary and data.is_single_qubit_gate:
            gate = self.cliffords.index[name]
            for target in targets:
                wire = self.wires[target.value]
                self.gates[wire] = self.cliffords.compose[self.gates[wire]][gate]
        elif data.is_unitary and data.is_two_qubit_gate:
            for target_0, target_1 in zip(targets[::2], targets[1::2]):
                self._apply_two_qubit_gate(name, target_0.value, target_1.value)
        elif name in _MEASUREMENTS or name in _MEASURE_RESETS:
            self._measure(name, targets)
        elif name in _RESETS:
            for target in targets:
                wire = self.wires[target.value]
                self.resets.pop(wire, None)
                self.last_czs.pop(wire, None)
                self.gates[wire] = self.cliffords.identity
                reset = _Instruction(name, [stim.GateTarget(wire)])
                self.items.append(reset)
                self.resets[wire] = (reset, _RESETS[name])
        elif name in _PAIR_MEASUREMENTS or name == "MPP":
            for group in instruction.target_groups():
                self._measure_product(name, group)
        elif name == "OBSERVABLE_INCLUDE":
            new_targets = [
                target if target.is_measurement_record_target else self._read_pauli_target(target)
                for target in targets
            ]
            self.items.append(_Instruction(name, new_targets, instruction.gate_args_copy()))
        else:
            raise ValueError(f"Untagged {name} instructions are not supported")

    def _apply_opaque(self, instruction: stim.CircuitInstruction) -> None:
        """Emit a tagged instruction as it is, but with relabeled qubit targets."""
        targets = instruction.targets_copy()
        qubit_map = {}
        for target in targets:
            if target.qubit_value is not None:
                wire = self.wires[target.qubit_value]
                qubit_map[target.qubit_value] = wire
                self.last_czs.pop(wire, None)
                if instruction.name in _RESETS:
                    self.resets.pop(wire, None)
                    self.gates[wire] = self.cliffords.identity
                else:
                    self._resolve_reset(wire)
                    self._emit(wire, self.gates[wire])
                    self.gates[wire] = self.cliffords.identity
        self.items.append(
            _Instruction(
                instruction.name,
                [remap_qubit_target(target, qubit_map) for target in targets],
                instruction.gate_args_copy(),
                instruction.tag,
            )
        )

    def _apply_two_qubit_gate(self, name: str, qubit_0: int, qubit_1: int) -> None:
        """Apply a two-qubit gate as single-qubit gates, at most one CZ, and a relabeling."""
        before_0, before_1, has_cz, after_0, after_1, has_swap = _decompose_two_qubit_gate(name)
        compose = self.cliffords.compose
        wire_0, wire_1 = self.wires[qubit_0], self.wires[qubit_1]
        self.gates[wire_0] = compose[self.gates[wire_0]][before_0]
        self.gates[wire_1] = compose[self.gates[wire_1]][before_1]
        if has_cz:
            self._apply_cz(wire_0, wire_1)
        self.gates[wire_0] = compose[self.gates[wire_0]][after_0]
        self.gates[wire_1] = compose[self.gates[wire_1]][after_1]
        if has_swap:
            self.wires[qubit_0], self.wires[qubit_1] = wire_1, wire_0

    def _apply_cz(self, wire_0: int, wire_1: int) -> None:
        """Emit a CZ, together with the parts of pending gates that do not commute with it."""
        flips_z = []
        for wire in (wire_0, wire_1):
            self._resolve_reset(wire)
            first, second = self.cliffords.split(self.gates[wire])
            self._emit(wire, first)
            self.gates[wire] = second
            flips_z.append(self.cliffords.flips_z(second))
        previous = self.last_czs.get(wire_0)
        if previous is not None and self.last_czs.get(wire_1) is previous:
            # this CZ cancels the previous CZ, which nothing has followed on these qubits
            index = len(self.items) - 1
            while self.items[index] is not previous:
                index -= 1
            del self.items[index]
            del self.last_czs[wire_0], self.last_czs[wire_1]
        else:
            gate = self.last_czs[wire_0] = self.last_czs[wire_1] = _Gate("CZ", (wire_0, wire_1))
            self.items.append(gate)
        # moving X past the CZ leaves X on its qubit and Z on the other qubit
        z_gate = self.cliffords.index["Z"]
        if flips_z[1]:
            self.gates[wire_0] = self.cliffords.compose[self.gates[wire_0]][z_gate]
        if flips_z[0]:
            self.gates[wire_1] = self.cliffords.compose[self.gates[wire_1]][z_gate]

    def _measure(self, name: str, targets: Sequence[stim.GateTarget]) -> None:
        """Emit single-qubit measurements in the bases that absorb the pending gates."""
        is_reset = name in _MEASURE_RESETS
        basis = _MEASURE_RESETS[name] if is_reset else _MEASUREMENTS[name]
        bases = _MEASURE_RESETS if is_reset else _MEASUREMENTS
        names = {axis: new_name for new_name, axis in bases.items()}
        for target in targets:
            wire = self.wires[target.value]
            sign, axis = self._read(wire, basis)
            invert = target.is_inverted_result_target ^ (sign < 0)
            new_target = stim.target_inv(wire) if invert else stim.GateTarget(wire)
            self.items.append(_Instruction(names[axis], [new_target]))
            if is_reset:
                # the measurement resets the qubit to the +1 eigenstate of its axis, not its basis
                self.gates[wire] = self.cliffords.get_preparation(axis, basis)

    def _measure_product(self, name: str, group: Sequence[stim.GateTarget]) -> None:
        """Emit an MPP measurement of a Pauli product in the basis that absorbs pending gates."""
        invert = False
        factors = []
        for target in group:
            wire = self.wires[target.value]
            pauli = _PAIR_MEASUREMENTS.get(name) or _PAULIS.index(target.pauli_type)
            sign, axis = self._read(wire, pauli)
            invert ^= target.is_inverted_result_target ^ (sign < 0)
            factors.append((wire, axis))
        targets = []
        for index, (wire, axis) in enumerate(factors):
            if index:
                targets.append(stim.target_combiner())
            targets.append(stim.target_pauli(wire, _PAULIS[axis], invert=invert and not index))
        self.items.append(_Instruction("MPP", targets))

    def _read_pauli_target(self, target: stim.GateTarget) -> stim.GateTarget:
        """Rewrite the Pauli target of an observable to read the same Pauli of the same state."""
        wire = self.wires[target.value]
        sign, axis = self._read(wire, _PAULIS.index(target.pauli_type))
        invert = target.is_inverted_result_target ^ (sign < 0)
        return stim.target_pauli(wire, _PAULIS[axis], invert=invert)

    def _read(self, wire: int, pauli: int) -> tuple[int, int]:
        """The Pauli to read now in order to read a Pauli after the pending gate of a qubit."""
        self._resolve_reset(wire)
        self.last_czs.pop(wire, None)
        return self.cliffords.backward[self.gates[wire]][pauli]

    def _resolve_reset(self, wire: int) -> None:
        """Fix the basis of the last reset of a qubit, if it can still absorb pending gates."""
        if wire in self.resets:
            reset, basis = self.resets.pop(wire)
            sign, axis = self.cliffords.forward[self.gates[wire]][basis]
            reset.name = {reset_axis: name for name, reset_axis in _RESETS.items()}[axis]
            self.gates[wire] = (
                self.cliffords.identity if sign > 0 else self.cliffords.index[_FLIPS[axis]]
            )

    def _emit(self, wire: int, gate: int) -> None:
        """Emit a single-qubit gate, unless it is the identity."""
        if gate != self.cliffords.identity:
            self.items.append(_Gate(self.cliffords.names[gate], (wire,)))
            self.last_czs.pop(wire, None)

    def _repeat(self, block: stim.CircuitRepeatBlock) -> None:
        """Transpile a REPEAT block, unrolling its first iteration if that makes it periodic."""
        start = self.get_state()
        self.last_czs.clear()
        body = block.body_copy()
        first = _Transpiler(start)
        first.run(body)
        if block.repeat_count == 1 or first.get_state() == start:
            self.items.append(_Repeat(block.repeat_count, first.items, block.tag))
            self.wires, self.gates = first.wires, first.gates
            return

        second = _Transpiler(first.get_state())
        second.run(body)
        if second.get_state() == first.get_state():
            self.items.extend(first.items)
            self.items.append(_Repeat(block.repeat_count - 1, second.items, block.tag))
            self.wires, self.gates = second.wires, second.gates
            return

        first.restore(start)
        self.items.append(_Repeat(block.repeat_count, first.items, block.tag))


####################################################################################################
# output


def _to_circuit(items: Sequence[_Item]) -> stim.Circuit:
    """Build a circuit from emitted items, scheduling each run of consecutive gates."""
    circuit = stim.Circuit()
    gates: list[_Gate] = []
    for item in items:
        if isinstance(item, _Gate):
            gates.append(item)
            continue
        circuit += _schedule(gates)
        gates = []
        if isinstance(item, _Repeat):
            body = _to_circuit(item.items)
            circuit.append(stim.CircuitRepeatBlock(item.count, body, tag=item.tag))
        else:
            circuit.append(
                stim.CircuitInstruction(item.name, item.targets, item.args, tag=item.tag)
            )
    circuit += _schedule(gates)
    return circuit


def _schedule(gates: Sequence[_Gate]) -> stim.Circuit:
    """Schedule gates in layers, with two-qubit gates early and single-qubit gates late.

    Two-qubit gates come as early as possible, and single-qubit gates as late as possible.
    Consecutive single-qubit gates on a qubit merge.
    """
    cliffords = _get_cliffords()
    # each layer has single-qubit gates followed by two-qubit gates, indexed by gate name
    layers: list[tuple[dict[str, list[int]], dict[str, list[int]]]] = []
    last_layers: dict[int, int] = {}
    # the single-qubit gate on each qubit that waits for the next layer with a gate on that qubit
    held: dict[int, int] = {}

    def place_held_gate(wire: int, layer: int) -> None:
        gate = held.pop(wire, cliffords.identity)
        if gate != cliffords.identity:
            layers[layer][0].setdefault(cliffords.names[gate], []).append(wire)

    for gate in gates:
        if len(gate.wires) == 1:
            wire = gate.wires[0]
            held[wire] = cliffords.compose[held.get(wire, cliffords.identity)][
                cliffords.index[gate.name]
            ]
            continue
        layer = 1 + max(last_layers.get(wire, -1) for wire in gate.wires)
        if layer == len(layers):
            layers.append(({}, {}))
        for wire in gate.wires:
            place_held_gate(wire, layer)
            last_layers[wire] = layer
        layers[layer][1].setdefault(gate.name, []).extend(gate.wires)
    if held:
        layers.append(({}, {}))
        for wire in sorted(held):
            place_held_gate(wire, len(layers) - 1)

    circuit = stim.Circuit()
    for single_qubit_gates, two_qubit_gates in layers:
        for name in sorted(single_qubit_gates):
            circuit.append(name, single_qubit_gates[name])
        for name in sorted(two_qubit_gates):
            circuit.append(name, two_qubit_gates[name])
    return circuit


####################################################################################################
# Clifford gates


@dataclasses.dataclass(frozen=True)
class _Cliffords:
    """Multiplication and conjugation tables of the single-qubit Clifford gates.

    Gates are indices into ``names``, which are the stim names of the 24 elements of the
    single-qubit Clifford group (up to phase).  Paulis are numbered 1, 2, 3 for X, Y, Z.

    Attributes:
        names: The stim names of the gates.
        index: The index of each name.
        keys: The index of each single-qubit tableau, identified by its key (see _get_key).
        compose: The gate compose[aa][bb] applies the gate aa and then the gate bb.
        inverse: The inverse of each gate.
        forward: The sign and Pauli forward[gate][pauli] of gate @ pauli @ gate^dag.
        backward: The sign and Pauli backward[gate][pauli] of gate^dag @ pauli @ gate.
    """

    names: tuple[str, ...]
    index: dict[str, int]
    keys: dict[tuple[str, str], int]
    compose: tuple[tuple[int, ...], ...]
    inverse: tuple[int, ...]
    forward: tuple[tuple[tuple[int, int], ...], ...]
    backward: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def identity(self) -> int:
        """The identity gate."""
        return self.index["I"]

    def split(self, gate: int) -> tuple[int, int]:
        """Split a gate into I, H, or SQRT_X, followed by a gate that maps Z to +Z or -Z."""
        # the axis of gate^dag @ Z @ gate identifies a coset of the gates that map Z to +Z or -Z
        first = self.index[_AXIS_REPRESENTATIVES[self.backward[gate][3][1]]]
        return first, self.compose[self.inverse[first]][gate]

    def flips_z(self, gate: int) -> bool:
        """Whether a gate that maps Z to +Z or -Z maps Z to -Z."""
        return self.forward[gate][3][0] < 0

    def get_preparation(self, axis: int, pauli: int) -> int:
        """A gate that maps the +1 eigenstate of one Pauli to the +1 eigenstate of another."""
        gates = sorted(range(len(self.names)), key=lambda gate: gate != self.identity)
        return next(gate for gate in gates if self.forward[gate][axis] == (1, pauli))


@functools.cache
def _get_cliffords() -> _Cliffords:
    """Tables of the single-qubit Clifford gates."""
    names = tuple(
        sorted(
            name
            for name, data in stim.gate_data().items()
            if data.is_unitary and data.is_single_qubit_gate
        )
    )
    tableaus = [stim.Tableau.from_named_gate(name) for name in names]
    keys = {_get_key(tableau): gate for gate, tableau in enumerate(tableaus)}

    def get_images(tableau: stim.Tableau) -> tuple[tuple[int, int], ...]:
        images = [tableau(stim.PauliString(pauli)) for pauli in _PAULIS]
        return tuple((int(image.sign.real), image[0]) for image in images)

    return _Cliffords(
        names=names,
        index={name: gate for gate, name in enumerate(names)},
        keys=keys,
        compose=tuple(
            tuple(keys[_get_key(first.then(second))] for second in tableaus) for first in tableaus
        ),
        inverse=tuple(keys[_get_key(tableau.inverse())] for tableau in tableaus),
        forward=tuple(get_images(tableau) for tableau in tableaus),
        backward=tuple(get_images(tableau.inverse()) for tableau in tableaus),
    )


def _get_key(tableau: stim.Tableau, qubit: int = 0) -> tuple[str, str]:
    """A key that identifies the action of a tableau on one qubit, up to phase.

    The key consists of the images of X and Z on that qubit, as strings such as "+X" or "-Y".
    """
    x_image, z_image = tableau.x_output(qubit), tableau.z_output(qubit)
    return (
        f"{'+' if x_image.sign == 1 else '-'}{_PAULIS[x_image[qubit]]}",
        f"{'+' if z_image.sign == 1 else '-'}{_PAULIS[z_image[qubit]]}",
    )


@functools.cache
def _decompose_two_qubit_gate(name: str) -> tuple[int, int, bool, int, int, bool]:
    """Decompose a two-qubit Clifford gate into single-qubit gates, at most one CZ, and a SWAP.

    Returns:
        A tuple (before_0, before_1, has_cz, after_0, after_1, has_swap): the gate applies the
        single-qubit gates before_0 and before_1 to its two qubits, then a CZ if has_cz, then the
        single-qubit gates after_0 and after_1, and then a SWAP if has_swap.
    """
    cliffords = _get_cliffords()
    representatives = [cliffords.index[gate] for gate in _AXIS_REPRESENTATIVES.values()]
    decompositions = (
        _try_decomposition(name, before_0, before_1, has_cz, has_swap)
        for has_cz, has_swap in itertools.product((False, True), repeat=2)
        for before_0, before_1 in itertools.product(representatives, repeat=2)
    )
    return next(decomposition for decomposition in decompositions if decomposition is not None)


def _try_decomposition(
    name: str, before_0: int, before_1: int, has_cz: bool, has_swap: bool
) -> tuple[int, int, bool, int, int, bool] | None:
    """Complete a decomposition of a two-qubit gate (see _decompose_two_qubit_gate), if possible."""
    cliffords = _get_cliffords()
    before = stim.Tableau.from_named_gate(cliffords.names[before_0]) + stim.Tableau.from_named_gate(
        cliffords.names[before_1]
    )
    middle = before.then(stim.Tableau.from_named_gate("CZ")) if has_cz else before
    target = stim.Tableau.from_named_gate(name)
    if has_swap:
        target = target.then(stim.Tableau.from_named_gate("SWAP"))
    after = middle.inverse().then(target)
    if any(
        image[1 - qubit]
        for qubit in range(2)
        for image in (after.x_output(qubit), after.z_output(qubit))
    ):
        return None
    after_0, after_1 = (cliffords.keys[_get_key(after, qubit)] for qubit in range(2))
    return before_0, before_1, has_cz, after_0, after_1, has_swap
