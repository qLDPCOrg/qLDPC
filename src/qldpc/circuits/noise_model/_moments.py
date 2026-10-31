# SPDX-License-Identifier: Apache-2.0

"""Decomposition of circuits into moments, and idle errors accumulated over moments.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Collection, Iterable, Iterator

import stim

from ._operations import (
    ANNOTATION,
    CLIFFORD_1Q,
    CLIFFORD_2Q,
    CLIFFORD_PP,
    COLLAPSING_OPS,
    IDLE_OPS,
    JUST_MEASURE_2Q,
    JUST_MEASURE_PP,
    NOISE,
    involves_classical_bits,
    op_type,
)
from .channels import PauliChannel, PauliChannelSequence

####################################################################################################
# moments and their qubits


def idle_barrier_qubits(
    instructions: Iterable[stim.CircuitInstruction | stim.CircuitRepeatBlock],
) -> set[int]:
    """Qubits that pending idle errors may not be deferred past, at any depth of repeat blocks.

    These are the qubits whose state an instruction acts on or depends on.  Annotations such as
    ``OBSERVABLE_INCLUDE`` with Pauli targets are included.  Explicit idles (``I``/``II``),
    ``QUBIT_COORDS`` annotations, noise channels, and the targets of classically controlled Paulis
    are not, since Pauli errors commute with these up to signs that do not affect detectors or
    observables.
    """
    qubits: set[int] = set()
    for instruction in instructions:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(idle_barrier_qubits(instruction.body_copy()))
        elif (
            instruction.name not in IDLE_OPS
            and instruction.name != "QUBIT_COORDS"
            and op_type(instruction.name) != NOISE
        ):
            for group in instruction.target_groups():
                if not any(
                    target.is_measurement_record_target or target.is_sweep_bit_target
                    for target in group
                ):
                    qubits.update(
                        target.qubit_value for target in group if target.qubit_value is not None
                    )
    return qubits


def get_immune_qubits(circuit: stim.Circuit, immune_qubit_tag: str) -> frozenset[int]:
    """Identify qubits declared immune to noise by tagged coordinate annotations."""
    immune_qubits: set[int] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            immune_qubits.update(get_immune_qubits(instruction.body_copy(), immune_qubit_tag))
        elif instruction.name == "QUBIT_COORDS" and immune_qubit_tag in instruction.tag:
            immune_qubits.update(
                target.qubit_value
                for target in instruction.targets_copy()
                if target.qubit_value is not None
            )
    return frozenset(immune_qubits)


def categorize_moment_qubits(
    moment: Collection[stim.CircuitInstruction | stim.CircuitRepeatBlock],
    *,
    instantaneous_1q_gates: bool = False,
) -> tuple[list[int], list[int]]:
    """Categorize a moment's qubit targets and check for reuse.

    Iterates every non-annotation instruction and buckets its qubit targets into four lists —
    collapsed (measurement / reset), classically-controlled, explicit-idle, and everything else
    ("operation").  Raises if any qubit is used more than once across the four buckets, since noise
    application relies on the "each qubit at most once per moment" invariant that
    ``NoiseModel.noisy_circuit`` enforces when ``insert_ticks=True``.  The classically-controlled
    and explicit-idle buckets contribute only to the reuse check.

    Args:
        moment: The moment's operations.
        instantaneous_1q_gates: If True, ignore single-qubit unitary gates, which take no time.

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
    explicit_idle_qubits: list[int] = []
    for op in moment:
        if (
            isinstance(op, stim.CircuitRepeatBlock)
            or op_type(op.name) in (ANNOTATION, NOISE)
            or (instantaneous_1q_gates and is_instantaneous_1q_gate(op))
        ):
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
        elif op.name in IDLE_OPS:
            qubits = explicit_idle_qubits
        else:
            qubits = operation_qubits
        qubits.extend(target_qubits)

    usage_counts = collections.Counter(
        collapsed_qubits + operation_qubits + classically_controlled_qubits + explicit_idle_qubits
    )
    qubits_used_multiple_times = {qubit for qubit, count in usage_counts.items() if count != 1}
    if qubits_used_multiple_times:
        raise ValueError(
            f"Qubits were operated on multiple times without a TICK in between:\n"
            f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
            f"moment:\n{moment}"
        )
    return collapsed_qubits, operation_qubits


def is_instantaneous_1q_gate(op: stim.CircuitInstruction | stim.CircuitRepeatBlock) -> bool:
    """Is an operation a single-qubit gate that takes no time, if single-qubit gates do not?

    Explicit ``I`` gates mark a moment of idling, so they are not instantaneous.
    """
    return (
        isinstance(op, stim.CircuitInstruction)
        and op_type(op.name) == CLIFFORD_1Q
        and op.name not in IDLE_OPS
    )


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


####################################################################################################
# idle errors


def repeated_idle_channel(
    channel: PauliChannel | PauliChannelSequence, repetitions: int
) -> PauliChannel | PauliChannelSequence:
    """Compose ``repetitions`` independent applications of a one-qubit Pauli channel."""
    if isinstance(channel, PauliChannelSequence):
        # an independent Pauli event that fires with probability p in each of n repetitions has a
        # net effect with probability (1 - (1 - 2p)^n) / 2
        return PauliChannelSequence(
            {
                pauli: (1 - (1 - 2 * probability) ** repetitions) / 2
                for pauli, probability in channel.probabilities.items()
            }
        )

    # compose the distribution over (I, X, Y, Z) by repeated squaring
    probabilities = tuple(channel.probabilities.get(pauli, 0.0) for pauli in "XYZ")
    single: tuple[float, ...] = (max(0.0, 1 - math.fsum(probabilities)), *probabilities)
    result: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    while repetitions:
        if repetitions & 1:
            result = _compose_pauli_distributions(result, single)
        repetitions >>= 1
        if repetitions:
            single = _compose_pauli_distributions(single, single)
    return PauliChannel(dict(zip("XYZ", result[1:], strict=True)))


def quadratic_idle_channel(channel: PauliChannel, num_moments: int) -> PauliChannel:
    """The error of a run of consecutive idle moments, given the quadratic idle error of one.

    If ``channel`` applies Paulis with total probability ``p``, the result applies the same Paulis
    in the same proportions with total probability ``(1 - (1 - 2 * p)**(num_moments**2)) / 2``.
    This is the Pauli twirl of a rotation about a fixed axis by an angle that grows linearly over
    the run, at a Gaussian random rate whose spread is set by ``p``.
    """
    total = math.fsum(channel.probabilities.values())
    # compute (1 - 2 * total)**(num_moments**2) - 1 to full relative precision
    decay = math.expm1(num_moments**2 * math.log1p(-2 * total)) if total < 0.5 else -1.0
    scale = -decay / (2 * total)
    return PauliChannel(
        {pauli: probability * scale for pauli, probability in channel.probabilities.items()}
    )


def compose_idle_channels(
    first: PauliChannel | PauliChannelSequence, second: PauliChannel | PauliChannelSequence
) -> PauliChannel:
    """The one-qubit Pauli channel that applies independent errors from two one-qubit channels."""
    distribution: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    for channel in (first, second):
        if isinstance(channel, PauliChannelSequence):
            # a sequence of independent Pauli events, each of which fires with its own probability
            for pauli, probability in channel.probabilities.items():
                event = [1 - probability, 0.0, 0.0, 0.0]
                event["IXYZ".index(pauli)] = probability
                distribution = _compose_pauli_distributions(distribution, tuple(event))
        else:
            probabilities = tuple(channel.probabilities.get(pauli, 0.0) for pauli in "XYZ")
            single = (max(0.0, 1 - math.fsum(probabilities)), *probabilities)
            distribution = _compose_pauli_distributions(distribution, single)
    return PauliChannel(dict(zip("XYZ", distribution[1:], strict=True)))


def _compose_pauli_distributions(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, ...]:
    """Distribution of the product of independent Paulis, indexed as (I, X, Y, Z).

    Up to phase, Paulis multiply by XOR-ing their indices: e.g. X * Z ~ (1 ^ 3) = 2 ~ Y.  The result
    is renormalized to prevent rounding errors from compounding over repeated compositions.
    """
    distribution = [
        math.fsum(left[index] * right[index ^ outcome] for index in range(4))
        for outcome in range(4)
    ]
    total = math.fsum(distribution)
    return tuple(probability / total for probability in distribution)
