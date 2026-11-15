# SPDX-License-Identifier: Apache-2.0

"""Noise rules, which specify the noise added to individual operations, and their validation.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import stim

from qldpc import _deprecations
from qldpc._utils import get_external_caller_stacklevel

from ..common import StimCircuitLike, _new_instruction
from ..transforms import with_remapped_qubits
from ._operations import (
    JUST_MEASURE_OPS,
    JUST_RESET_OPS,
    MEASURE_AND_RESET_OPS,
    NOISE,
    canonical_rule_key,
    known_gate_shape,
    op_type,
)
from .channels import AbstractPauliChannel, PauliChannel

####################################################################################################
# noise rules


class NoiseRule:
    """Describes how to add noise to an operation.

    A rule can add Pauli noise after each gate of an operation, flip its measurement results, or
    reset its qubits to the wrong state.

    Args:
        after: Noise applied after each gate of a matching operation, on the qubits of that gate.
            It may be any of the following:

            - ``PauliChannel``: a Pauli channel on ``k`` qubits, which
              :meth:`qldpc.circuits.PauliChannel.to_circuit` emits after each block of ``k``
              targets.
            - ``PauliChannelSequence``: a sequence of independent Pauli-string error mechanisms.
            - ``Mapping[str, float]``: a ``PauliChannel``, given by its probabilities.  A Mapping
              whose probabilities are all zero adds no noise.
            - ``stim.Circuit``: a fragment of noise channels on qubits ``0`` to ``k - 1``, where
              ``k`` is ``circuit.num_qubits``.  The fragment is emitted after each block of ``k``
              targets, with its qubits mapped to those of the block.  It must act on at least one
              qubit, and contain only noise channels that record no measurements, such as
              ``X_ERROR``, ``DEPOLARIZE1``, or ``PAULI_CHANNEL_1``.

        readout_error: The probability that a measurement result is reported incorrectly.  Only
            allowed for operations that produce measurement results.  ``None``, the default, leaves
            this field unset.  ``0.0`` adds no error, but is still checked against the gate type.
        reset_error: The probability that a qubit is reset to the wrong state.  Only allowed for
            operations that reset qubits.  ``None`` and ``0.0`` differ as for ``readout_error``.

    Raises:
        ValueError: If a Pauli string is invalid, a probability is not in [0, 1], ``after`` is
            combined with ``readout_error`` or ``reset_error``, or a ``stim.Circuit`` fragment acts
            on no qubits or contains an instruction other than a noise channel without measurement
            results.
        TypeError: If a ``stim.Circuit`` fragment contains a REPEAT block.
    """

    def __init__(
        self,
        *,
        after: AbstractPauliChannel | Mapping[str, float] | stim.Circuit | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
    ) -> None:
        """Initializes a noise rule with specified error channels."""
        self.readout_error = readout_error
        if readout_error is not None and not (0 <= readout_error <= 1):
            raise ValueError(f"{readout_error=} is not between 0 and 1")

        self.reset_error = reset_error
        if reset_error is not None and not (0 <= reset_error <= 1):
            raise ValueError(f"{reset_error=} is not between 0 and 1")

        self.after: AbstractPauliChannel | stim.Circuit | None
        if after is None:
            self.after = None
        elif isinstance(after, AbstractPauliChannel):
            self.after = after
        elif isinstance(after, stim.Circuit):
            _validate_after_circuit(after)
            if after.num_qubits == 0:
                raise ValueError("after (stim.Circuit form) must address at least one qubit")
            self.after = after
        else:
            self.after = PauliChannel(after) or None

        if (readout_error is not None or reset_error is not None) and self.after is not None:
            raise ValueError(
                "NoiseRule cannot combine `after`-noise with readout_error / reset_error.  If you"
                " need this capability, please open an issue at"
                " https://github.com/qLDPCOrg/qLDPC/issues"
            )

    def __bool__(self) -> bool:
        """Is this noise rule nontrivial?"""
        return bool(self.after) or bool(self.readout_error) or bool(self.reset_error)

    def __repr__(self) -> str:
        args = []
        if self.after is not None:
            args.append(f"after={self.after!r}")
        if self.readout_error is not None:
            args.append(f"readout_error={self.readout_error!r}")
        if self.reset_error is not None:
            args.append(f"reset_error={self.reset_error!r}")
        return f"NoiseRule({', '.join(args)})"

    def noisy_operation(
        self, op: stim.CircuitInstruction
    ) -> tuple[stim.CircuitInstruction, stim.Circuit]:
        """Apply this noise rule to the given operation.

        Args:
            op: The operation to add noise to.

        Returns:
            stim.CircuitInstruction: The given operation possibly modified to account for noise.
            stim.Circuit: Noise operations that should follow the given operation.
        """
        targets = op.targets_copy()
        args = op.gate_args_copy()
        qubit_targets = [target.value for target in targets if not target.is_combiner]

        if self.readout_error:
            if op.name not in JUST_MEASURE_OPS and op.name not in MEASURE_AND_RESET_OPS:
                raise ValueError(f"readout_error is not valid for operation {op.name!r}")
            if not args:
                args = [self.readout_error]
            else:
                if len(args) != 1:
                    raise ValueError(f"Expected one gate argument for {op.name!r}")

                # combine independent readout flips using the chance of an odd number of flips
                args = [self.readout_error + args[0] - 2 * self.readout_error * args[0]]

        noisy_op = _new_instruction(op.name, targets, args, op.tag)
        noise_after = stim.Circuit()

        if self.reset_error:
            if op.name not in JUST_RESET_OPS and op.name not in MEASURE_AND_RESET_OPS:
                raise ValueError(f"reset_error is not valid for operation {op.name!r}")
            # a wrong reset is an error that anticommutes with the basis of the reset: a Z error for
            # an X-basis reset, and an X error for a Y- or Z-basis reset
            error_name = "Z_ERROR" if op.name.endswith("X") else "X_ERROR"
            noise_after.append(error_name, qubit_targets, [self.reset_error])

        self.emit_after(noise_after, qubit_targets, context=f"operation {op.name!r}")
        return noisy_op, noise_after

    def emit_after(
        self, circuit: StimCircuitLike, qubit_targets: list[int], *, context: str = "operation"
    ) -> None:
        """Append this rule's ``after`` noise in-place to the provided circuit, once per gate.

        An operation may apply several gates, as ``H 0 1 2`` applies three ``H`` gates.  The noise
        applies to each consecutive block of ``after.num_qubits`` qubits in ``qubit_targets``.

        Args:
            circuit: The circuit to append the noise instructions to.
            qubit_targets: The qubits that the noise applies to, in the order of the targets of the
                operation.
            context: A short description of the operation, used only in error messages.

        Raises:
            ValueError: If ``len(qubit_targets)`` is not a multiple of the number of qubits of
                ``after``.
        """
        n_targets = len(qubit_targets)
        if not self.after:
            return
        num_qubits = self.after.num_qubits
        if n_targets % num_qubits != 0:
            raise ValueError(
                f"This NoiseRule expects a multiple of {num_qubits} qubits but {context} has "
                f"{n_targets} qubit targets"
            )

        if isinstance(self.after, AbstractPauliChannel):
            for i in range(0, n_targets, num_qubits):
                self.after.to_circuit(append_to=circuit, qubits=qubit_targets[i : i + num_qubits])
        else:
            # apply a stim.Circuit fragment to each block of qubits, with its qubits remapped
            assert isinstance(self.after, stim.Circuit)
            for i in range(0, n_targets, num_qubits):
                circuit += with_remapped_qubits(self.after, qubit_targets[i : i + num_qubits])


# the forms of the gate and idle errors of a NoiseModel: a probability of depolarizing noise, Pauli
# noise as a Mapping or channel, or a noise rule (see _as_noise_rule)
ErrorSpec: TypeAlias = float | Mapping[str, float] | AbstractPauliChannel | NoiseRule


def _validate_after_circuit(after_circuit: stim.Circuit) -> None:
    """Validate that ``after_circuit`` contains only noise instructions.

    Args:
        after_circuit: The fragment to validate.

    Raises:
        ValueError: If ``after_circuit`` contains a non-noise or measurement-producing instruction.
        TypeError: If ``after_circuit`` contains a repeat block.
    """
    for op in after_circuit:
        if not isinstance(op, stim.CircuitInstruction):
            raise TypeError(
                f"after (stim.Circuit form) may contain only noise instructions, not "
                f"{type(op).__name__}"
            )
        if op_type(op.name) != NOISE:
            raise ValueError(
                f"after (stim.Circuit form) contains non-noise instruction {op.name!r}; only "
                "NOISE instructions are permitted"
            )
        if stim.gate_data(op.name).produces_measurements:
            raise ValueError(
                f"after (stim.Circuit form) cannot contain measurement-producing noise "
                f"instruction {op.name!r}"
            )


####################################################################################################
# normalization of error specifications


def _as_noise_rule(error: ErrorSpec | None, default_arity: int) -> NoiseRule | None:
    """Normalize an error argument of ``NoiseModel`` to a noise rule, or None for no error.

    A number becomes a depolarizing ``PauliChannel`` on ``default_arity`` qubits, which
    ``PauliChannel.conditioned_on`` can restrict to qubits that are not immune.  A Pauli channel, or
    a Mapping from Pauli strings to probabilities, becomes the ``after`` noise of a rule.  None and
    an empty Mapping give None.  Other errors that add no noise still give a rule, so that callers
    can check the number of qubits of its channel before they drop it.
    """
    if isinstance(error, NoiseRule):
        return error
    if error is False:  # DEPRECATED: see qldpc._deprecations
        _deprecations.warn_deprecated(
            "Passing False for no error to a NoiseModel is deprecated and will be removed in a"
            " future release.  Pass None instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return None
    if error is None or (isinstance(error, Mapping) and not error):
        return None
    if isinstance(error, AbstractPauliChannel):
        return NoiseRule(after=error)
    if isinstance(error, Mapping):
        return NoiseRule(after=PauliChannel(error))
    if isinstance(error, bool) or not isinstance(error, (int, float)):
        raise TypeError(
            f"expected a float, PauliChannel, PauliChannelSequence, Mapping[str, float], or "
            f"NoiseRule; got {type(error).__name__!r}"
        )
    return NoiseRule(after=PauliChannel.depolarizing(default_arity, float(error)))


def _normalize_clifford_nq_error(
    error: Mapping[int, ErrorSpec] | None,
) -> dict[int, NoiseRule]:
    """Normalize the ``clifford_nq_error`` argument of ``NoiseModel`` to rules by number of qubits.

    Entries become rules as in ``_as_noise_rule``, and a number must be a probability.  Each rule
    must fit unitary gates on its number of qubits, and rules that add no noise are dropped.
    """
    if not error:
        return {}
    result: dict[int, NoiseRule] = {}
    for num_qubits, entry in error.items():
        if num_qubits < 1:
            raise ValueError(f"clifford_nq_error key {num_qubits} must be >= 1")
        if isinstance(entry, (NoiseRule, AbstractPauliChannel, Mapping)):
            rule = _as_noise_rule(entry, num_qubits)
            if rule is None:  # an empty Mapping
                continue
        elif isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(
                f"clifford_nq_error[{num_qubits}] has unsupported type {type(entry).__name__}; "
                "expected NoiseRule, PauliChannel, PauliChannelSequence, Mapping[str, float], int, "
                "or float"
            )
        else:
            if not (0 <= entry <= 1):
                raise ValueError(f"clifford_nq_error[{num_qubits}]={entry} is not in [0, 1]")
            if entry == 0:
                continue
            rule = NoiseRule(after=PauliChannel.depolarizing(num_qubits, entry))
        _validate_rule_for_arity(rule, num_qubits, f"clifford_nq_error[{num_qubits}]")
        if rule:
            result[num_qubits] = rule
    return result


def _normalize_rules(rules: Mapping[str, NoiseRule]) -> dict[str, NoiseRule]:
    """Normalize the ``rules`` argument of ``NoiseModel`` to rules by canonical key.

    Each rule whose key fixes its number of qubits must fit gates on those qubits.  Rules for bare
    MPP, SPP, and SPP_DAG are checked when they are used.
    """
    normalized_rules: dict[str, NoiseRule] = {}
    for op_name, rule in rules.items():
        canonical_name = canonical_rule_key(op_name)
        if canonical_name in normalized_rules:
            raise ValueError(f"Duplicate noise rules after canonicalization: {op_name!r}")
        normalized_rules[canonical_name] = rule
        shape = known_gate_shape(canonical_name)
        if shape is not None:
            arity, can_measure, can_reset = shape
            _validate_rule_for_arity(
                rule,
                arity,
                f"rules[{op_name!r}]",
                can_measure=can_measure,
                can_reset=can_reset,
            )
    return normalized_rules


####################################################################################################
# validation of noise rules


def _validate_rule_for_arity(
    rule: NoiseRule,
    num_qubits: int,
    context: str,
    *,
    can_measure: bool = False,
    can_reset: bool = False,
) -> None:
    """Reject a noise rule that does not fit gates on ``num_qubits`` qubits.

    The ``after`` noise of the rule must act on ``num_qubits`` qubits, even if it is an empty
    channel.  A rule may set ``readout_error`` only if ``can_measure``, and ``reset_error`` only if
    ``can_reset``.
    """
    if rule.after is not None and rule.after.num_qubits != num_qubits:
        raise ValueError(
            f"{context}: `after` has arity {rule.after.num_qubits}; expected {num_qubits}"
        )
    if rule.readout_error is not None and not can_measure:
        raise ValueError(f"{context}: `readout_error` is only valid on measurement gates")
    if rule.reset_error is not None and not can_reset:
        raise ValueError(f"{context}: `reset_error` is only valid on reset gates")


def _validate_rule_for_op(rule: NoiseRule, op: stim.CircuitInstruction, context: str) -> None:
    """Reject a noise rule that does not fit the gate application ``op``."""
    num_qubits = sum(1 for target in op.targets_copy() if not target.is_combiner)
    _validate_rule_for_arity(
        rule,
        num_qubits,
        context,
        can_measure=op.name in JUST_MEASURE_OPS or op.name in MEASURE_AND_RESET_OPS,
        can_reset=op.name in JUST_RESET_OPS or op.name in MEASURE_AND_RESET_OPS,
    )


####################################################################################################
# noise rules on immune qubits


def _rule_with_immunity(
    rule: NoiseRule,
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    *,
    immunize_gates: bool,
) -> NoiseRule | None:
    """The noise rule of one gate application on some immune qubits, or None for no noise.

    A gate application on no immune qubits keeps its rule, and one on only immune qubits gets no
    noise.  One on both immune and other qubits gets no noise if ``immunize_gates`` is True, or if
    its rule adds no noise and its class is ``NoiseRule`` itself.  Otherwise, a rule with
    ``PauliChannel`` noise becomes a ``NoiseRule`` with the Pauli strings that act as the identity
    on immune qubits, from ``PauliChannel.conditioned_on``.  Other rules raise an error.
    """
    if not immune_qubits:
        return rule
    qubits = [t.qubit_value for t in op.targets_copy() if t.qubit_value is not None]
    immune_flags = [q in immune_qubits for q in qubits]
    if not any(immune_flags):
        return rule
    # a subclass of NoiseRule may add noise in noisy_operation or emit_after
    if all(immune_flags) or immunize_gates or (type(rule) is NoiseRule and not rule):
        return None
    if isinstance(rule.after, PauliChannel) and rule.after:
        immune_positions = [i for i, immune in enumerate(immune_flags) if immune]
        return NoiseRule(after=rule.after.conditioned_on(immune_positions))
    raise ValueError(
        f"Cannot apply a rule to {op.name!r} with partial immunity (immune qubits: "
        f"{sorted(q for q in qubits if q in immune_qubits)}) under immunize_gates=False.  Only "
        "PauliChannel `after`-noise supports conditioning; set immunize_gates=True to drop noise "
        "on gates that touch immune qubits.  If this edge case matters for your use case, please "
        "open an issue at https://github.com/qLDPCOrg/qLDPC/issues."
    )
