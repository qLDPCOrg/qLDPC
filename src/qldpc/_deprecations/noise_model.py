# SPDX-License-Identifier: Apache-2.0

"""The deprecated noisy_circuit(..., insert_ticks=True) of noise models.

That option lays out a noisy circuit in moments separated by TICKs: the TICK layers of the circuit,
split wherever a gate reuses a qubit, as though its repeat blocks were flattened.  Every gate takes
time, including classically controlled gates.  This module imports qldpc.circuits, so it is
imported only within the function that uses it.
"""

from __future__ import annotations

import collections
from collections.abc import Collection, Iterable, Iterator

import stim

from qldpc.circuits.noise_model import NoiseModel
from qldpc.circuits.noise_model._moments import (
    compose_idle_channels,
    quadratic_idle_channel,
    repeated_idle_channel,
)
from qldpc.circuits.noise_model._operations import (
    ANNOTATION,
    CLIFFORD_2Q,
    CLIFFORD_PP,
    COLLAPSING_OPS,
    GATE_OP_TYPES,
    JUST_MEASURE_2Q,
    JUST_MEASURE_PP,
    NOISE,
    involves_classical_bits,
    op_type,
)
from qldpc.circuits.noise_model.channels import PauliChannel, PauliChannelSequence
from qldpc.circuits.noise_model.rules import NoiseRule, _rule_with_immunity

####################################################################################################
# noisy circuits laid out in moments


def get_noisy_circuit_with_ticks(
    noise_model: NoiseModel,
    circuit: stim.Circuit,
    *,
    system_qubits: frozenset[int],
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    immunize_gates: bool,
) -> stim.Circuit:
    """Add noise to a circuit, laid out in moments separated by TICKs."""
    noisy_circuit = stim.Circuit()
    builder = _NoisyCircuitBuilder(
        noise_model,
        system_qubits=system_qubits,
        immune_qubits=immune_qubits,
        immune_op_tag=immune_op_tag,
        immunize_gates=immunize_gates,
    )
    builder.append(circuit, noisy_circuit)
    builder.finish(noisy_circuit)
    return noisy_circuit


class _NoisyCircuitBuilder:
    """Adds noise to a circuit moment by moment, as though its repeat blocks were flattened.

    Repeat blocks are nonetheless kept compact.  The noise of an iteration of a repeat block is
    determined by the state of this builder at the start of the iteration: the open moment, whether
    a TICK must precede the next moment, and the pending idle moments of the qubits that the repeat
    block uses (the qubits that it never uses do not affect its noise).  Once an iteration ends in
    the same state that it started in, all remaining iterations are identical to it.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        *,
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
    ) -> None:
        self.noise_model = noise_model
        self.system_qubits = system_qubits
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates

        self.moment: list[stim.CircuitInstruction | stim.CircuitRepeatBlock] = []  # the open moment
        self.used_qubits: set[int] = set()  # qubits that gates in the open moment act on
        self.idle_moments: dict[int, int] = {}  # idle moments not yet emitted as noise, by qubit
        self.needs_tick = False  # whether the next moment must be preceded by a TICK

    def append(self, circuit: stim.Circuit, output: stim.Circuit) -> None:
        """Add noise to the operations of a circuit, and append completed moments to the output."""
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                if self.immune_op_tag is not None and self.immune_op_tag in op.tag:
                    self.close_moment(output)
                    _flush_idle_errors(
                        self.noise_model, output, self.idle_moments, idle_barrier_qubits([op])
                    )
                    if self.needs_tick:
                        output.append("TICK")
                    output.append(op)
                    self.needs_tick = False
                elif not _is_gate_free(op.body_copy(), self.immune_op_tag):
                    self.append_repeat_block(op, output)
                elif not _is_empty(op.body_copy()):
                    self.moment.append(op)
            elif op.name == "TICK":
                self.close_moment(output)
            else:
                for gate in split_targets_if_needed(op, frozenset({-1}), self.immune_op_tag):
                    qubits = gate_qubits(gate)
                    if qubits & self.used_qubits:
                        self.close_moment(output)
                    self.used_qubits |= qubits
                    self.moment.extend(
                        split_targets_if_needed(
                            gate,
                            self.immune_qubits,
                            self.immune_op_tag,
                            force_split=self.noise_model.rule_func is not None,
                        )
                    )

    def append_repeat_block(self, block: stim.CircuitRepeatBlock, output: stim.Circuit) -> None:
        """Add noise to a repeat block, and append its completed moments to the output."""
        body = block.body_copy()
        body_qubits = idle_barrier_qubits(body)

        def get_state() -> str:
            idle_moments = sorted(
                (qubit, count) for qubit, count in self.idle_moments.items() if qubit in body_qubits
            )
            return str((self.moment, self.needs_tick, idle_moments))

        state = get_state()
        for iteration in range(block.repeat_count):
            idle_moments = dict(self.idle_moments)
            noisy_body = stim.Circuit()
            self.append(body, noisy_body)
            previous_state, state = state, get_state()
            if state != previous_state:
                output += noisy_body
                continue

            # this iteration started and ended in the same state, so it repeats until the end
            num_iterations = block.repeat_count - iteration
            if num_iterations > 1 and noisy_body:
                output.append(stim.CircuitRepeatBlock(num_iterations, noisy_body, tag=block.tag))
            else:
                output += noisy_body
            for qubit, count in self.idle_moments.items():
                if qubit not in body_qubits:
                    increment = count - idle_moments.get(qubit, 0)
                    self.idle_moments[qubit] = count + increment * (num_iterations - 1)
            return

    def close_moment(self, output: stim.Circuit) -> None:
        """Add noise to the open moment (if nonempty), and append it to the output."""
        if not self.moment:
            return
        _flush_idle_errors(
            self.noise_model, output, self.idle_moments, idle_barrier_qubits(self.moment)
        )
        if self.needs_tick:
            output.append("TICK")
        _append_noisy_moment(
            self.noise_model,
            circuit=output,
            moment=self.moment,
            system_qubits=self.system_qubits,
            immune_qubits=self.immune_qubits,
            immune_op_tag=self.immune_op_tag,
            immunize_gates=self.immunize_gates,
            pending_idle_moments=self.idle_moments,
        )
        self.moment = []
        self.used_qubits = set()
        self.needs_tick = True

    def finish(self, output: stim.Circuit) -> None:
        """Close the open moment, and emit all pending idle errors."""
        self.close_moment(output)
        _flush_idle_errors(self.noise_model, output, self.idle_moments)


def _is_gate_free(circuit: stim.Circuit, immune_op_tag: str | None) -> bool:
    """Does a circuit consist of annotations (other than TICKs) and noise, at any depth?"""
    return all(
        not (immune_op_tag is not None and immune_op_tag in op.tag)
        and _is_gate_free(op.body_copy(), immune_op_tag)
        if isinstance(op, stim.CircuitRepeatBlock)
        else op.name != "TICK" and op_type(op.name) in (ANNOTATION, NOISE)
        for op in circuit
    )


def _is_empty(circuit: stim.Circuit) -> bool:
    """Is a circuit empty when flattened?"""
    return all(
        isinstance(op, stim.CircuitRepeatBlock) and _is_empty(op.body_copy()) for op in circuit
    )


def _append_noisy_moment(
    noise_model: NoiseModel,
    *,
    circuit: stim.Circuit,
    moment: Collection[stim.CircuitInstruction | stim.CircuitRepeatBlock],
    system_qubits: frozenset[int],
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    immunize_gates: bool,
    pending_idle_moments: dict[int, int],
) -> None:
    """Applies noise to a moment and appends it to a circuit (in-place).

    This method processes all operations in a moment, applies their respective noise rules, and
    adds the resulting noisy operations to the output circuit.

    Args:
        noise_model: The noise model.
        circuit: The circuit to append the noisy operations to.
        moment: Collection of operations happening during the moment in question.
        system_qubits: Set of all qubits in the system that may experience idle errors.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: If an operation contains this string in its tag, that operation is
            noiseless.
        immunize_gates: If True (the default), a gate that touches an immune qubit is treated as
            noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
            noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
            An immune-tagged operation does not suppress idle noise on unrelated qubits in the
            same moment.
        pending_idle_moments: Map from each qubit to its number of idle moments that have not
            yet been emitted as noise.  Updated in-place.
    """
    collapsed_qubits, operation_qubits = categorize_moment_qubits(moment)

    noise_after_moment = stim.Circuit()
    for op in moment:
        if (
            isinstance(op, stim.CircuitRepeatBlock)
            or (immune_op_tag is not None and immune_op_tag in op.tag)
            or (rule := noise_model.get_noise_rule(op)) is None
        ):
            circuit.append(op)
            continue
        effective_rule = _rule_with_immunity(rule, op, immune_qubits, immunize_gates=immunize_gates)
        if effective_rule is None:
            circuit.append(op)
            continue
        noisy_op, after = effective_rule.noisy_operation(op)
        circuit.append(noisy_op)
        noise_after_moment += after
    circuit += noise_after_moment

    moment_was_noisy = any(
        isinstance(op, stim.CircuitInstruction) and op_type(op.name) in GATE_OP_TYPES
        for op in moment
    )
    if moment_was_noisy and (
        noise_model.idle_error
        or noise_model.quadratic_idle_error
        or noise_model.additional_error_waiting_for_m_or_r
    ):
        _append_idle_errors(
            noise_model,
            circuit=circuit,
            collapsed_qubits=collapsed_qubits,
            operation_qubits=operation_qubits,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            pending_idle_moments=pending_idle_moments,
        )


def _append_idle_errors(
    noise_model: NoiseModel,
    *,
    circuit: stim.Circuit,
    collapsed_qubits: list[int],
    operation_qubits: list[int],
    system_qubits: frozenset[int],
    immune_qubits: frozenset[int],
    pending_idle_moments: dict[int, int],
) -> None:
    """Append or count idling errors from the given moment.

    This method identifies which qubits are idle during a moment and applies noise to them
    according to the noise model parameters.  The qubit categorization is precomputed by
    ``categorize_moment_qubits`` and passed in.

    Args:
        noise_model: The noise model.
        circuit: The circuit to append idle error operations to.
        collapsed_qubits: Qubits acted on by measurement / reset ops in this moment.
        operation_qubits: Qubits acted on by non-collapsing (unitary) ops in this moment.
        system_qubits: Set of all qubits in the system that can experience idle errors.
        immune_qubits: Qubits that are declared to be immune to noise.
        pending_idle_moments: Map from each qubit to its number of idle moments that have not
            yet been emitted as noise.  Pauli and quadratic idle errors are counted here, to be
            composed and emitted by ``_flush_idle_errors``; other idle noise is emitted
            immediately.
    """
    non_collapse_qubits = system_qubits - immune_qubits - set(collapsed_qubits)
    idle_qubits = sorted(non_collapse_qubits - set(operation_qubits))

    count_idle_moments = noise_model.quadratic_idle_error is not None
    if noise_model.idle_error is not None:
        if isinstance(noise_model.idle_error.after, (PauliChannel, PauliChannelSequence)):
            count_idle_moments = True
        elif idle_qubits:
            noise_model.idle_error.emit_after(circuit, idle_qubits, context="idle qubits")
    if count_idle_moments:
        for qubit in idle_qubits:
            pending_idle_moments[qubit] = pending_idle_moments.get(qubit, 0) + 1
    if noise_model.additional_error_waiting_for_m_or_r and collapsed_qubits and non_collapse_qubits:
        noise_model.additional_error_waiting_for_m_or_r.emit_after(
            circuit, sorted(non_collapse_qubits), context="qubits waiting for m/r"
        )


def _flush_idle_errors(
    noise_model: NoiseModel,
    circuit: stim.Circuit,
    pending_idle_moments: dict[int, int],
    qubits: Iterable[int] | None = None,
) -> None:
    """Emit and clear the pending idle errors on the given qubits (by default, on all qubits).

    Pending idle errors must be emitted before any instruction that depends on the state of
    their qubit, as identified by ``idle_barrier_qubits``.  They may be deferred past noise and
    classically controlled Paulis, since Pauli errors commute with these up to signs that do
    not affect detectors or observables.
    """
    if not pending_idle_moments:
        return
    qubits_by_count: dict[int, list[int]] = collections.defaultdict(list)
    for qubit in sorted(pending_idle_moments.keys() if qubits is None else qubits):
        if (count := pending_idle_moments.pop(qubit, None)) is not None:
            qubits_by_count[count].append(qubit)
    idle_channel = None if noise_model.idle_error is None else noise_model.idle_error.after
    for count, targets in qubits_by_count.items():
        channels: list[PauliChannel | PauliChannelSequence] = []
        if isinstance(idle_channel, (PauliChannel, PauliChannelSequence)):
            channels.append(
                idle_channel if count == 1 else repeated_idle_channel(idle_channel, count)
            )
        if noise_model.quadratic_idle_error is not None:
            channels.append(quadratic_idle_channel(noise_model.quadratic_idle_error, count))
        if channels:
            after = channels[0] if len(channels) == 1 else compose_idle_channels(*channels)
            NoiseRule(after=after).emit_after(circuit, targets, context="idle qubits")


####################################################################################################
# moments and their qubits


def idle_barrier_qubits(
    instructions: Iterable[stim.CircuitInstruction | stim.CircuitRepeatBlock],
) -> set[int]:
    """Qubits that pending idle errors may not be deferred past, at any depth of repeat blocks.

    These are the qubits whose state an instruction acts on or depends on.  Annotations such as
    ``OBSERVABLE_INCLUDE`` with Pauli targets are included.  ``QUBIT_COORDS`` annotations, noise
    channels, and the targets of classically controlled Paulis are not, since Pauli errors commute
    with these up to signs that do not affect detectors or observables.
    """
    qubits: set[int] = set()
    for instruction in instructions:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(idle_barrier_qubits(instruction.body_copy()))
        elif instruction.name != "QUBIT_COORDS" and op_type(instruction.name) != NOISE:
            for group in instruction.target_groups():
                if not any(
                    target.is_measurement_record_target or target.is_sweep_bit_target
                    for target in group
                ):
                    qubits.update(
                        target.qubit_value for target in group if target.qubit_value is not None
                    )
    return qubits


def categorize_moment_qubits(
    moment: Collection[stim.CircuitInstruction | stim.CircuitRepeatBlock],
) -> tuple[list[int], list[int]]:
    """Categorize a moment's qubit targets and check for reuse.

    Iterates every non-annotation instruction and buckets its qubit targets into three lists —
    collapsed (measurement / reset), classically-controlled, and everything else ("operation").
    Raises if any qubit is used more than once across the three buckets, since noise application
    relies on the "each qubit at most once per moment" invariant that ``NoiseModel.noisy_circuit``
    enforces when ``insert_ticks=True``.  The classically-controlled bucket contributes only to the
    reuse check.

    Args:
        moment: The moment's operations.

    Returns:
        A tuple ``(collapsed_qubits, operation_qubits)``, each in the order the moment referenced
        them.  Consumers use these for idle-error emission (see
        ``NoiseModel._inplace_append_idle_errors``).

    Raises:
        ValueError: If any qubit is operated on multiple times within the moment without a TICK in
            between.
    """
    collapsed_qubits: list[int] = []
    operation_qubits: list[int] = []
    classically_controlled_qubits: list[int] = []
    for op in moment:
        if isinstance(op, stim.CircuitRepeatBlock) or op_type(op.name) in (ANNOTATION, NOISE):
            continue
        target_qubits = [
            target.qubit_value for target in op.targets_copy() if target.qubit_value is not None
        ]
        if op_type(op.name) in (CLIFFORD_PP, JUST_MEASURE_PP):
            # A Pauli product may intentionally address one qubit more than once, e.g.
            # ``MPP X0*Z0``.  It is one atomic operation, not two operations in the moment.
            target_qubits = list(dict.fromkeys(target_qubits))
        if op.name in COLLAPSING_OPS:
            qubits = collapsed_qubits
        elif involves_classical_bits(op):
            qubits = classically_controlled_qubits
        else:
            qubits = operation_qubits
        qubits.extend(target_qubits)

    usage_counts = collections.Counter(
        collapsed_qubits + operation_qubits + classically_controlled_qubits
    )
    qubits_used_multiple_times = {qubit for qubit, count in usage_counts.items() if count != 1}
    if qubits_used_multiple_times:
        raise ValueError(
            f"Qubits were operated on multiple times without a TICK in between:\n"
            f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
            f"moment:\n{moment}"
        )
    return collapsed_qubits, operation_qubits


def gate_qubits(op: stim.CircuitInstruction | stim.CircuitRepeatBlock) -> set[int]:
    """The qubits that an operation acts on, for the purpose of identifying qubit reuse."""
    if isinstance(op, stim.CircuitRepeatBlock) or op_type(op.name) in (ANNOTATION, NOISE):
        return set()
    return {
        target.qubit_value
        for target in op.targets_copy()
        if not target.is_combiner and target.qubit_value is not None
    }


####################################################################################################
# splitting operations into individual gate applications


def split_targets_if_needed(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits operations into pieces as needed.

    This function splits operations like SPP and MPP into each Pauli product, and separates
    classical control operations from quantum operations.

    Args:
        op: The circuit instruction to potentially split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, always decompose broadcast one- and two-qubit gates into individual
            applications, regardless of immunity.

    Yields:
        Circuit instructions, potentially split into smaller pieces.
    """
    this_op_type = op_type(op.name)
    # Two-qubit ops (both Clifford CX/CZ/... and joint measurements MXX/MYY/MZZ) split per-pair via
    # _split_targets_clifford_2q so partial-immunity decisions happen at the pair level.
    if this_op_type in (CLIFFORD_2Q, JUST_MEASURE_2Q):
        yield from _split_targets_clifford_2q(
            op, immune_qubits, immune_op_tag, force_split=force_split
        )
    elif this_op_type == CLIFFORD_PP or this_op_type == JUST_MEASURE_PP:
        yield from _split_targets_pp(op)
    elif this_op_type in (NOISE, ANNOTATION):
        yield op
    else:
        yield from _split_targets_clifford_1q(
            op, immune_qubits, immune_op_tag, force_split=force_split
        )


def _split_targets_clifford_1q(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits single-qubit Clifford operations when immune qubits are present.

    Args:
        op: The single-qubit Clifford operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, split into individual single-target operations unconditionally.

    Yields:
        Circuit instructions split into individual single-target operations.
    """
    if force_split or immune_qubits or (immune_op_tag is not None and immune_op_tag in op.tag):
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args, tag=op.tag)
    else:
        yield op


def _split_targets_clifford_2q(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit operations into individual gate pairs.

    Handles both two-qubit Cliffords (CX/CZ/...) and joint two-qubit measurements (MXX/MYY/MZZ),
    separating classical-control from quantum-only gates.

    Args:
        op: The two-qubit operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, split into individual two-qubit gate pairs unconditionally.

    Yields:
        Circuit instructions split into individual two-qubit gate operations.
    """
    assert op_type(op.name) in (CLIFFORD_2Q, JUST_MEASURE_2Q)
    targets = op.targets_copy()
    if (
        force_split
        or immune_qubits
        or (immune_op_tag is not None and immune_op_tag in op.tag)
        or any(target.is_measurement_record_target for target in targets)
    ):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[k : k + 2], args, tag=op.tag)
    else:
        yield op


def _split_targets_pp(op: stim.CircuitInstruction) -> Iterator[stim.CircuitInstruction]:
    """Splits a Pauli product operation into one operation for each Pauli product.

    Args:
        op: The Pauli product operation to split.

    Yields:
        Circuit instructions, one for each Pauli product.
    """
    assert op_type(op.name) in (CLIFFORD_PP, JUST_MEASURE_PP)
    targets = op.targets_copy()
    args = op.gate_args_copy()
    # group combiner-linked targets so each instruction contains one Pauli product
    start = end = 0
    while end < len(targets):
        if end + 1 == len(targets) or not targets[end + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : end + 1], args, tag=op.tag)
            end += 1
            start = end
        else:
            end += 2
    assert end == len(targets)
