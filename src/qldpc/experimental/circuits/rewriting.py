# SPDX-License-Identifier: Apache-2.0

"""Rewrite circuits into deferred-measurement form.

A circuit is in deferred-measurement form if it consists of three consecutive parts:

1. A body of unitary gates, noise channels, and Pauli-target observables (observables whose
   targets are all Paulis on qubits, which read the state in place).
2. Measurements: ``M`` measurements of distinct qubits, without readout error or inverted targets.
3. Detectors and record-target observables (observables whose targets are all measurement
   records), in any order.

In such a circuit, an error propagates to the measurements through unitary gates alone, and
post-selection on measurement records is a projection at the end of the circuit.

The following passes rewrite a circuit into this form, using the principle of deferred measurement
(Nielsen and Chuang, *Quantum Computation and Quantum Information*, Section 4.4), which also applies
to resets: a reset is a measurement whose outcome is discarded, followed by the preparation of a
fixed state.  ``defer_measurements_and_resets`` runs the passes in order:

1. ``split_measure_resets`` replaces each measure-reset by a measurement and a reset.
2. ``defer_resets`` defers each reset by swapping in a fresh ancilla qubit, prepared in the reset
   state and never measured, so that the reset qubit keeps its index.
3. ``defer_measurements`` moves all measurements to the end of the circuit.  A measurement whose
   qubits are used later is first copied onto a fresh ancilla qubit, which is measured at the end.
4. ``simplify_measurements`` turns each of the measurements at the end into an ``M`` measurement
   of one qubit, without readout error or inversion, and puts the gates that it adds before the
   final block of measurements.

The function ``is_deferred`` returns whether a circuit is in deferred-measurement form.

Each pass preserves the probability of every string of measurement records, and, conditioned on
those records, the state of the qubits of its input that the pass does not measure or reset;
``simplify_measurements`` changes the states of measured qubits after their measurements.  In
particular, each pass preserves the detector error model of a Clifford circuit, and detectors and
observables keep their meaning, since the passes keep the order of measurement records.

All passes except ``split_measure_resets`` drop ``TICK``, ``QUBIT_COORDS``, and ``SHIFT_COORDS``,
since deferral discards the layout in time and space that these annotations describe.  Detectors
keep their coordinates, in absolute form.  Fresh ancilla qubits get indices after those of the
input circuit.

Instructions keep their tags, and the gates that the passes add are untagged.  No pass depends on
which unitary gates a circuit applies, so the passes also apply to the Clifford+T circuits of the
tsim simulator, which encodes T gates in stim circuits as tagged instructions (``S[T]``,
``SPP[T]``).

All passes except ``split_measure_resets`` reject circuits with classically controlled gates,
heralded noise, ``MPAD``, ``ELSE_CORRELATED_ERROR``, or measurements of Pauli products that repeat
a qubit, such as ``MPP X0*Z0``.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import stim

# The Pauli basis of each single-qubit measurement, measure-reset, or reset.
_MEASUREMENT_BASES = {"M": "Z", "MZ": "Z", "MX": "X", "MY": "Y"}
_MEASURE_RESETS = {"MR": ("M", "R"), "MRZ": ("MZ", "RZ"), "MRX": ("MX", "RX"), "MRY": ("MY", "RY")}
_RESET_BASES = {"R": "Z", "RZ": "Z", "RX": "X", "RY": "Y"}
# Measurements of pairs of qubits, with their Pauli letters.
_PAIR_MEASUREMENTS = {"MXX": "X", "MYY": "Y", "MZZ": "Z"}
# Gates that map a basis to the Z basis, back from it, and prepare its +1 eigenstate from |0>.
_TO_Z = {"X": ("H",), "Y": ("S_DAG", "H"), "Z": ()}
_FROM_Z = {"X": ("H",), "Y": ("H", "S"), "Z": ()}
_PREPARATIONS = {"X": ("H",), "Y": ("H", "S"), "Z": ()}
# A Pauli gate that anticommutes with each basis, which flips the outcome of its measurement.
_FLIPS = {"X": "Z", "Y": "Z", "Z": "X"}
# Annotations of time and space, which deferral drops.
_ANNOTATIONS = frozenset({"TICK", "QUBIT_COORDS", "SHIFT_COORDS"})
_UNSUPPORTED = frozenset(
    {"MPAD", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "ELSE_CORRELATED_ERROR"}
)
# The largest qubit index of a stim circuit.
_MAX_QUBIT = 2**24 - 1
# Instructions whose groups of targets are Pauli products, written with combiners, as in X0*Z1.
_PAULI_PRODUCTS = frozenset({"MPP", "SPP", "SPP_DAG"})

####################################################################################################
# deferred-measurement form


def defer_measurements_and_resets(circuit: stim.Circuit) -> stim.Circuit:
    """Rewrite a circuit in deferred-measurement form (see the module docstring).

    The result is that of ``split_measure_resets``, ``defer_resets``, ``defer_measurements``, and
    ``simplify_measurements``, run in order, except that fresh ancilla qubits get indices after all
    qubits of the input circuit, including qubits that earlier passes remove, such as qubits that
    only annotations address.

    Raises:
        ValueError: If the circuit has an unsupported instruction or a measurement of a Pauli
            product that repeats a qubit, or would need a fresh ancilla qubit beyond stim's largest
            qubit index, ``2**24 - 1``.
    """
    # Run the four passes on lists of instructions.  Each pass numbers its fresh ancilla qubits from
    # the number of qubits of the input circuit or the number of qubits after the previous passes,
    # whichever is larger, so that no ancilla reuses the index of a qubit of the input circuit.
    instructions = _split_measure_resets(_get_instructions(circuit))
    instructions = _defer_resets(instructions, circuit.num_qubits)
    num_qubits = max(circuit.num_qubits, _get_num_qubits(instructions))
    instructions = _defer_measurements(instructions, num_qubits)
    num_qubits = max(num_qubits, _get_num_qubits(instructions))
    return _get_circuit(_simplify_measurements(instructions, num_qubits))


def is_deferred(circuit: stim.Circuit) -> bool:
    """Whether a circuit is in deferred-measurement form (see the module docstring).

    ``REPEAT`` blocks are checked as if unrolled.
    """
    # Flattening removes SHIFT_COORDS instructions, which the form excludes, so look for them first.
    if _has_shift_coords(circuit):
        return False

    # Walk through the flattened circuit, tracking which of the three parts of the form the current
    # instruction belongs to.  Gates, noise, and Pauli-target observables may appear only in the
    # body.  M measurements form the second part: they must be noiseless, uninverted, and of qubits
    # that no earlier measurement measured.  Detectors and record-target observables form the third
    # part.  Any other instruction, such as a reset or another kind of measurement, breaks the form.
    part = 1
    measured: set[int] = set()
    for instruction in circuit.flattened():
        name = instruction.name
        data = _get_gate_data(name)
        targets = instruction.targets_copy()
        records = [target.is_measurement_record_target for target in targets]
        if name == "M":
            if part == 3 or any(instruction.gate_args_copy()):
                return False
            for target in targets:
                if target.is_inverted_result_target or target.value in measured:
                    return False
                measured.add(target.value)
            part = 2
        elif part == 1 and (
            data.is_unitary
            or _is_noise(name)
            or (name == "OBSERVABLE_INCLUDE" and not any(records))
        ):
            continue
        elif name == "DETECTOR" or (name == "OBSERVABLE_INCLUDE" and all(records)):
            part = 3
        else:
            return False
    return True


def _has_shift_coords(circuit: stim.Circuit) -> bool:
    """Whether a circuit has a ``SHIFT_COORDS`` instruction, possibly in a ``REPEAT`` block."""
    return any(
        _has_shift_coords(instruction.body_copy())
        if isinstance(instruction, stim.CircuitRepeatBlock)
        else instruction.name == "SHIFT_COORDS"
        for instruction in circuit
    )


####################################################################################################
# rewriting passes


def split_measure_resets(circuit: stim.Circuit) -> stim.Circuit:
    """Replace each measure-reset (``MR``, ``MRX``, ...) by a measurement and then a reset.

    The result is flattened, and equivalent to the input circuit.
    """
    return _get_circuit(_split_measure_resets(_get_instructions(circuit)))


def defer_resets(circuit: stim.Circuit) -> stim.Circuit:
    """Defer each reset by swapping in a fresh ancilla qubit.

    A reset of a qubit that nothing has acted on yet becomes the gates that prepare the reset state.
    A reset of any other qubit prepares that state on a fresh ancilla qubit and swaps it into the
    reset qubit, whose previous state, with any error on it, moves to the ancilla, which nothing
    acts on afterward.  Noise whose qubits are all reset next is removed, since the resets erase it.
    Measure-resets are split first (see ``split_measure_resets``).

    Raises:
        ValueError: If the circuit has an unsupported instruction or a measurement of a Pauli
            product that repeats a qubit, or would need a fresh ancilla qubit beyond stim's largest
            qubit index, ``2**24 - 1``.
    """
    return _get_circuit(
        _defer_resets(_split_measure_resets(_get_instructions(circuit)), circuit.num_qubits)
    )


def defer_measurements(circuit: stim.Circuit) -> stim.Circuit:
    """Move all measurements of a reset-free circuit to its end.

    A measurement whose qubits are used later is copied onto a fresh ancilla qubit by CX gates, and
    the ancilla is measured at the end instead, with the inversion, readout error, and tag of the
    measurement.  Every other measurement moves to the end unchanged.  Measurements at the end keep
    the order of their records, and are followed by the detectors and record-target observables;
    the Pauli targets of observables stay in place.

    Raises:
        ValueError: If the circuit has a reset, an unsupported instruction, or a measurement of a
            Pauli product that repeats a qubit, or would need a fresh ancilla qubit beyond stim's
            largest qubit index, ``2**24 - 1``.
    """
    return _get_circuit(_defer_measurements(_get_instructions(circuit), circuit.num_qubits))


def simplify_measurements(circuit: stim.Circuit) -> stim.Circuit:
    """Turn terminal measurements into noiseless ``M`` measurements of single qubits at the end.

    Every measurement must be terminal: no later instruction may address its qubits, including Pauli
    targets of observables.  Each measurement becomes gates in its place and an ``M`` measurement of
    one qubit at the end of the circuit.  A measurement in the X or Y basis becomes a basis change,
    and a measurement of a Pauli product, such as ``MPP`` or ``MXX``, is copied onto a fresh ancilla
    qubit by CX gates, which is measured instead.  An inverted target becomes a Pauli gate that
    flips the outcome, and a readout error with probability ``p`` becomes an ``X_ERROR(p)`` on the
    measured qubit.  The added gates affect only the states of measured qubits after their
    measurements.  Removing an inversion can change stim's reference sign of an observable whose
    Pauli targets make it deterministic, since that sign ignores Pauli targets, but not the value of
    the observable.

    The ``M`` measurements at the end keep the order of their records, and are followed by the
    detectors and record-target observables; the Pauli targets of observables stay in place.  The
    result is therefore in deferred-measurement form if the circuit has no resets.

    Raises:
        ValueError: If a measurement is not terminal, the circuit has an unsupported instruction or
            a measurement of a Pauli product that repeats a qubit, or a fresh ancilla qubit would be
            beyond stim's largest qubit index, ``2**24 - 1``.
    """
    return _get_circuit(_simplify_measurements(_get_instructions(circuit), circuit.num_qubits))


####################################################################################################
# rewriting passes on lists of instructions


def _split_measure_resets(instructions: list[_CircuitInstruction]) -> list[_CircuitInstruction]:
    """Split each measure-reset into a measurement and a reset; see ``split_measure_resets``."""
    result = []
    for instruction in instructions:
        if instruction.name in _MEASURE_RESETS:
            measurement, reset = _MEASURE_RESETS[instruction.name]
            result.append(instruction.renamed(measurement))
            qubits = [target.value for group in instruction.groups for target in group]
            result.append(
                _CircuitInstruction(reset, "", (), (tuple(_GateTarget(q) for q in qubits),))
            )
        else:
            result.append(instruction)
    return result


def _defer_resets(
    instructions: list[_CircuitInstruction], num_qubits: int
) -> list[_CircuitInstruction]:
    """Replace resets by preparations of fresh qubits and swaps; see ``defer_resets``.

    The instructions must not have measure-resets, and ``num_qubits`` must be at least the number
    of qubits that they address.
    """
    # Drop annotations, which deferral discards, and reject instructions that it does not support.
    instructions = _drop_annotations(instructions)
    _reject_unsupported(instructions)

    # Remove the noise that resets erase: drop each group of targets of a noise channel whose qubits
    # are all reset next, and drop noise channels that have no groups left.
    dead = _get_noise_before_resets(instructions)
    kept = []
    for instruction, dead_qubits in zip(instructions, dead):
        if _is_noise(instruction.name):
            groups = tuple(
                group
                for group in instruction.groups
                if not all(target.value in dead_qubits for target in group)
            )
            if not groups:
                continue
            instruction = instruction._replace(groups=groups)
        kept.append(instruction)

    # Replace each reset by the preparation of the +1 eigenstate of its basis, tracking the qubits
    # that earlier instructions have acted on (all instructions but detectors and observables, which
    # only read records or the state).  A qubit that nothing has acted on is still in |0>, so the
    # preparation acts on the reset qubit directly.  For any other qubit, the preparation acts on a
    # fresh ancilla qubit, which a swap then moves into the reset qubit, so that the old state of
    # the reset qubit is discarded on the ancilla.  All other instructions stay unchanged.
    result = []
    touched: set[int] = set()
    for instruction in kept:
        name = instruction.name
        if name not in _RESET_BASES:
            result.append(instruction)
            if name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                touched.update(instruction.qubits)
            continue
        preparation = _PREPARATIONS[_RESET_BASES[name]]
        for target in instruction.targets:
            qubit = target.value
            if qubit not in touched:
                result += [_get_gate(gate, qubit) for gate in preparation]
            else:
                _check_ancilla(num_qubits)
                result += [_get_gate(gate, num_qubits) for gate in preparation]
                result.append(_get_gate("SWAP", qubit, num_qubits))
                num_qubits += 1
            touched.add(qubit)
    return result


def _defer_measurements(
    instructions: list[_CircuitInstruction], num_qubits: int
) -> list[_CircuitInstruction]:
    """Move all measurements to the end; see ``defer_measurements``.

    The instructions must measure at most one Pauli product each, and ``num_qubits`` must be at
    least the number of qubits that they address.
    """
    # Drop annotations, which deferral discards, and reject instructions that it does not support,
    # including resets, which defer_resets removes.
    instructions = _drop_annotations(instructions)
    _reject_unsupported(instructions)
    if any(_get_gate_data(instruction.name).is_reset for instruction in instructions):
        raise ValueError("defer_measurements requires a circuit without resets; see defer_resets")

    # Move the measurements to the end, copying those whose qubits are used later onto fresh
    # ancilla qubits.
    return _move_measurements_to_end(instructions, num_qubits)


def _simplify_measurements(
    instructions: list[_CircuitInstruction], num_qubits: int
) -> list[_CircuitInstruction]:
    """Turn terminal measurements into plain ``M`` measurements; see ``simplify_measurements``.

    The instructions must measure at most one Pauli product each, and ``num_qubits`` must be at
    least the number of qubits that they address.
    """
    # Drop annotations, which deferral discards, and reject instructions that it does not support.
    instructions = _drop_annotations(instructions)
    _reject_unsupported(instructions)

    # Check that every measurement is terminal, since replacing a measurement changes the state of
    # its qubits after the measurement: no later instruction may use those qubits, and a
    # measure-reset, which resets its qubit, is not terminal.
    last_uses = _get_last_uses(instructions)
    for index, instruction in enumerate(instructions):
        if _get_gate_data(instruction.name).produces_measurements and (
            instruction.name in _MEASURE_RESETS
            or any(last_uses[qubit] > index for qubit in instruction.qubits)
        ):
            raise ValueError(
                f"simplify_measurements requires terminal measurements, but {instruction} "
                "is not; see defer_measurements"
            )

    # Replace each measurement, in place, by gates and a noiseless M measurement of one qubit that
    # holds the measured value.  A measurement of one qubit rotates that qubit into the Z basis,
    # after a Pauli gate that flips the outcome if the measurement is inverted.  A measurement of a
    # Pauli product copies its value onto a fresh ancilla qubit, followed by an X gate if the
    # measurement is inverted.  A readout error becomes an X error before the M measurement, which
    # keeps the tag.  All other instructions stay unchanged.
    result = []
    for instruction in instructions:
        name = instruction.name
        if not _get_gate_data(name).produces_measurements:
            result.append(instruction)
            continue
        paulis, inverted = _get_measured_pauli(instruction)
        if name in _MEASUREMENT_BASES:
            qubit = paulis[0][0]
            basis = _MEASUREMENT_BASES[name]
            if inverted:
                result.append(_get_gate(_FLIPS[basis], qubit))
            result += [_get_gate(gate, qubit) for gate in _TO_Z[basis]]
        else:
            qubit = num_qubits
            num_qubits += 1
            _check_ancilla(qubit)
            _append_parity_gadget(result, paulis, qubit)
            if inverted:
                result.append(_get_gate("X", qubit))
        if instruction.arguments and instruction.arguments[0]:
            result.append(
                _CircuitInstruction("X_ERROR", "", instruction.arguments, ((_GateTarget(qubit),),))
            )
        result.append(
            instruction.renamed("M")._replace(arguments=(), groups=((_GateTarget(qubit),),))
        )

    # Move the M measurements to the end, after the gates added above.  Every M measurement is
    # terminal, so none of them is copied onto another ancilla qubit.
    return _move_measurements_to_end(result, num_qubits)


def _move_measurements_to_end(
    instructions: list[_CircuitInstruction], num_qubits: int
) -> list[_CircuitInstruction]:
    """Move all measurements to the end, followed by detectors and record targets of observables.

    A measurement whose qubits are used later is copied onto a fresh ancilla qubit, which is
    measured at the end instead.  The instructions must not have annotations, and must measure at
    most one Pauli product each, and ``num_qubits`` must be at least the number of qubits that they
    address.
    """
    # Collect the records that detectors and observables refer to, as absolute indices, since moving
    # measurements changes the offsets of records from each detector or observable.  The records of
    # all instructions for one observable and tag merge into one list, and an observable without
    # targets keeps its index.
    detectors: list[tuple[_CircuitInstruction, list[int]]] = []
    observables: dict[tuple[str, tuple[float, ...]], list[int]] = {}
    num_records = 0
    for instruction in instructions:
        if _get_gate_data(instruction.name).produces_measurements:
            num_records += 1
        elif instruction.name == "DETECTOR":
            records = [num_records + target.value for target in instruction.targets]
            detectors.append((instruction, records))
        elif instruction.name == "OBSERVABLE_INCLUDE":
            records = [
                num_records + target.value for target in instruction.targets if target.is_record
            ]
            if records or not instruction.targets:
                key = (instruction.tag, instruction.arguments)
                observables.setdefault(key, []).extend(records)

    # Build the body of the output circuit, with its gates, noise, and Pauli targets of observables,
    # which read the state in place, and the list of measurements that follow the body.  A
    # measurement that no later instruction uses moves after the body unchanged.  Any other
    # measurement is copied in place onto a fresh ancilla qubit by a parity gadget, and the ancilla
    # is measured after the body instead, with the inversion, readout error, and tag of the
    # measurement.  Detectors and record targets of observables are left for the end.
    last_uses = _get_last_uses(instructions)
    body: list[_CircuitInstruction] = []
    measurements: list[_CircuitInstruction] = []
    for index, instruction in enumerate(instructions):
        name = instruction.name
        if name == "OBSERVABLE_INCLUDE":
            qubit_targets = tuple(target for target in instruction.targets if not target.is_record)
            if qubit_targets:
                body.append(instruction._replace(groups=(qubit_targets,)))
        elif _get_gate_data(name).produces_measurements:
            if all(last_uses[qubit] == index for qubit in instruction.qubits):
                measurements.append(instruction)
            else:
                paulis, inverted = _get_measured_pauli(instruction)
                _check_ancilla(num_qubits)
                _append_parity_gadget(body, paulis, num_qubits)
                target = _GateTarget(num_qubits, inverted=inverted)
                measurements.append(instruction.renamed("M")._replace(groups=((target,),)))
                num_qubits += 1
        elif name != "DETECTOR":
            body.append(instruction)

    # Assemble the output circuit: its body, then its measurements in the order of their records,
    # then the detectors and observables, whose records become offsets from the end of the circuit.
    result = body + measurements
    for detector, records in detectors:
        targets = tuple(_GateTarget(record - num_records, "rec") for record in records)
        result.append(detector._replace(groups=(targets,)))
    for (tag, arguments), records in observables.items():
        targets = tuple(_GateTarget(record - num_records, "rec") for record in records)
        result.append(_CircuitInstruction("OBSERVABLE_INCLUDE", tag, arguments, (targets,)))
    return result


def _check_ancilla(qubit: int) -> None:
    """Raise a ValueError if a fresh ancilla qubit would exceed stim's largest qubit index."""
    if qubit > _MAX_QUBIT:
        raise ValueError(
            f"A fresh ancilla qubit would have index {qubit}, which exceeds stim's largest qubit "
            f"index, {_MAX_QUBIT}"
        )


def _drop_annotations(instructions: list[_CircuitInstruction]) -> list[_CircuitInstruction]:
    """The instructions other than ``TICK`` and coordinate annotations."""
    return [instruction for instruction in instructions if instruction.name not in _ANNOTATIONS]


def _reject_unsupported(instructions: list[_CircuitInstruction]) -> None:
    """Raise a ValueError for instructions that the passes do not support.

    Each measurement must have one group of targets.
    """
    for instruction in instructions:
        name = instruction.name
        if name in _UNSUPPORTED:
            raise ValueError(f"Instruction is not supported: {instruction}")
        data = _get_gate_data(name)
        if data.is_unitary and not all(target.is_qubit for target in instruction.targets):
            raise ValueError(f"Classically controlled gates are not supported: {instruction}")
        if data.produces_measurements and len(instruction.qubits) < len(instruction.targets):
            raise ValueError(f"Measured Pauli products must not repeat qubits: {instruction}")


def _get_last_uses(instructions: list[_CircuitInstruction]) -> dict[int, int]:
    """The index of the last instruction that addresses each qubit.

    Noise counts, since noise after a measurement must not affect it once it moves to the end, and
    so do qubit targets of observables, which read the state in place.  Detectors address no qubits.
    """
    last_uses = {}
    for index, instruction in enumerate(instructions):
        for qubit in instruction.qubits:
            last_uses[qubit] = index
    return last_uses


def _get_noise_before_resets(instructions: list[_CircuitInstruction]) -> list[set[int]]:
    """For each instruction, the qubits whose next instruction, other than noise, is a reset."""
    dead: list[set[int]] = []
    next_is_reset: dict[int, bool] = {}
    for instruction in reversed(instructions):
        if _is_noise(instruction.name):
            dead.append({qubit for qubit in instruction.qubits if next_is_reset.get(qubit, False)})
            continue
        dead.append(set())
        if instruction.name == "DETECTOR":
            continue
        is_reset = instruction.name in _RESET_BASES
        for qubit in instruction.qubits:
            next_is_reset[qubit] = is_reset
    return dead[::-1]


def _get_measured_pauli(instruction: _CircuitInstruction) -> tuple[list[tuple[int, str]], bool]:
    """The Pauli product that an instruction measures, and whether its record is inverted.

    The instruction must have one group of targets, without repeated qubits.  The product is a list
    of (qubit, letter) pairs, sorted by qubit.
    """
    name = instruction.name
    (group,) = instruction.groups
    if name == "MPP":
        letters = {target.value: target.kind for target in group}
    elif name in _PAIR_MEASUREMENTS:
        letters = {target.value: _PAIR_MEASUREMENTS[name] for target in group}
    else:
        letters = {target.value: _MEASUREMENT_BASES[name] for target in group}
    inverted = sum(target.inverted for target in group) % 2 == 1
    return sorted(letters.items()), inverted


def _append_parity_gadget(
    instructions: list[_CircuitInstruction], paulis: list[tuple[int, str]], ancilla: int
) -> None:
    """Copy the value of a Pauli product onto a fresh ancilla qubit in ``|0>``.

    Each qubit of the product is rotated so that its letter becomes Z, its Z value is added onto the
    ancilla by a CX gate, and it is rotated back, so that a Z measurement of the ancilla measures
    the product.
    """
    for qubit, basis in paulis:
        instructions += [_get_gate(gate, qubit) for gate in _TO_Z[basis]]
        instructions.append(_get_gate("CX", qubit, ancilla))
        instructions += [_get_gate(gate, qubit) for gate in _FROM_Z[basis]]


####################################################################################################
# instructions as Python objects
#
# The rewriting passes of this module work on lists of the following plain Python objects, rather
# than on stim's instructions, because stim.Circuit.append and stim.CircuitInstruction convert each
# target slowly, while stim parses the text of a circuit orders of magnitude faster.  The passes
# therefore build each output circuit by parsing its text once (see _get_circuit).


class _GateTarget(NamedTuple):
    """A light stand-in for ``stim.GateTarget``.

    A target is a qubit, a Pauli on a qubit, a measurement record, or a sweep bit.  The kind is ""
    for a qubit, "X", "Y", or "Z" for a Pauli, and "rec" or "sweep" otherwise.  The value is the
    index of the qubit or sweep bit, or the (negative) offset of the record.  Unlike
    ``stim.GateTarget``, there are no combiners: the targets of a Pauli product form a group of a
    ``_CircuitInstruction``.
    """

    value: int
    kind: str = ""
    inverted: bool = False

    @property
    def is_record(self) -> bool:
        """Whether the target is a measurement record."""
        return self.kind == "rec"

    @property
    def is_qubit(self) -> bool:
        """Whether the target addresses a qubit, possibly with a Pauli."""
        return self.kind not in ("rec", "sweep")

    def __str__(self) -> str:
        if not self.is_qubit:
            return f"{self.kind}[{self.value}]"
        return f"{'!' if self.inverted else ''}{self.kind}{self.value}"

    def to_stim(self) -> stim.GateTarget:
        """This target as a stim target."""
        if self.is_record:
            return stim.target_rec(self.value)
        if self.kind == "sweep":
            return stim.target_sweep_bit(self.value)
        if self.kind:
            return stim.target_pauli(self.value, self.kind, self.inverted)
        return stim.target_inv(self.value) if self.inverted else stim.GateTarget(self.value)


class _CircuitInstruction(NamedTuple):
    """A light stand-in for ``stim.CircuitInstruction``, in a flattened circuit.

    The groups of targets are those of ``stim.CircuitInstruction.target_groups``, without
    combiners.
    """

    name: str
    tag: str
    arguments: tuple[float, ...]
    groups: tuple[tuple[_GateTarget, ...], ...]

    @property
    def targets(self) -> list[_GateTarget]:
        """All targets of this instruction, without combiners."""
        return [target for group in self.groups for target in group]

    @property
    def qubits(self) -> set[int]:
        """The qubits that this instruction addresses, excluding measurement records."""
        return {target.value for group in self.groups for target in group if target.is_qubit}

    def renamed(self, name: str) -> _CircuitInstruction:
        """This instruction with another name, and the same tag, arguments, and targets."""
        return self._replace(name=name)

    def to_stim(self) -> stim.CircuitInstruction:
        """This instruction as a stim instruction, with combiners in Pauli products."""
        targets = []
        for group in self.groups:
            for index, target in enumerate(group):
                if index and self.name in _PAULI_PRODUCTS:
                    targets.append(stim.target_combiner())
                targets.append(target.to_stim())
        return stim.CircuitInstruction(self.name, targets, self.arguments, tag=self.tag)

    def __str__(self) -> str:
        text = self.name
        if self.tag:
            text += "[" + _escape(self.tag) + "]"
        if self.arguments:
            # repr writes floats exactly
            text += "(" + ", ".join(repr(argument) for argument in self.arguments) + ")"
        if any(self.groups):
            separator = "*" if self.name in _PAULI_PRODUCTS else " "
            text += " " + " ".join(separator.join(map(str, group)) for group in self.groups)
        return text


def _get_instructions(circuit: stim.Circuit) -> list[_CircuitInstruction]:
    """The instructions of a flattened circuit, with measurements and resets split by target group.

    Stim allows one instruction to measure or reset a qubit several times, as in ``MR 0 0``, so
    splitting each measurement or reset into one instruction per group of targets lets each
    instruction measure or reset each qubit at most once.
    """
    # Convert each stim instruction into Python objects, keeping stim's groups of targets, and split
    # measurements and resets into one instruction per group of targets, so that each measurement
    # has one record.
    instructions = []
    for instruction in circuit.flattened():
        name = instruction.name
        tag = instruction.tag
        arguments = tuple(instruction.gate_args_copy())
        groups = tuple(
            tuple(_get_target(target) for target in group) for group in instruction.target_groups()
        )
        data = _get_gate_data(name)
        if data.produces_measurements or data.is_reset:
            instructions += [
                _CircuitInstruction(name, tag, arguments, (group,)) for group in groups
            ]
        else:
            instructions.append(_CircuitInstruction(name, tag, arguments, groups))
    return instructions


def _get_circuit(instructions: list[_CircuitInstruction]) -> stim.Circuit:
    """The circuit of a list of instructions, built by parsing its text once.

    The text writes arguments exactly, with ``repr``.  Stim cannot parse infinite or NaN arguments,
    such as some coordinates, so a list with such arguments is appended one instruction at a time
    instead.  Both give the same circuit, since stim fuses consecutive compatible instructions both
    when parsing and when appending.
    """
    if all(math.isfinite(arg) for instruction in instructions for arg in instruction.arguments):
        return stim.Circuit("\n".join(map(str, instructions)))
    circuit = stim.Circuit()
    for instruction in instructions:
        circuit.append(instruction.to_stim())
    return circuit


def _escape(tag: str) -> str:
    """A tag as stim writes it in the text of a circuit, with escape sequences."""
    return tag.replace("\\", "\\B").replace("]", "\\C").replace("\r", "\\r").replace("\n", "\\n")


def _get_target(target: stim.GateTarget) -> _GateTarget:
    """The Python object of a target of a stim instruction, which is not a combiner."""
    if target.is_measurement_record_target:
        return _GateTarget(target.value, "rec")
    if target.is_sweep_bit_target:
        return _GateTarget(target.value, "sweep")
    pauli = target.pauli_type
    return _GateTarget(
        target.value, "" if pauli == "I" else pauli, target.is_inverted_result_target
    )


def _get_gate(name: str, *qubits: int) -> _CircuitInstruction:
    """An untagged instruction without arguments that acts on a single group of qubits."""
    return _CircuitInstruction(name, "", (), (tuple(_GateTarget(qubit) for qubit in qubits),))


def _get_num_qubits(instructions: list[_CircuitInstruction]) -> int:
    """The number of qubits of the circuit of a list of instructions, as stim counts them."""
    return 1 + max(
        (
            target.value
            for instruction in instructions
            for group in instruction.groups
            for target in group
            if target.is_qubit
        ),
        default=-1,
    )


####################################################################################################
# helpers


@functools.cache
def _get_gate_data(name: str) -> stim.GateData:
    """The data of a stim gate, by name."""
    return stim.gate_data(name)


def _is_noise(name: str) -> bool:
    """Whether an instruction is a noise channel, rather than a noisy measurement."""
    data = _get_gate_data(name)
    return data.is_noisy_gate and not data.produces_measurements
