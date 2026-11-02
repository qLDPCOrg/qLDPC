# SPDX-License-Identifier: Apache-2.0

"""Noise rules, which specify the noise added to individual operations, and their validation.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import stim

from ..common import StimCircuitLike
from ..transforms import with_remapped_qubits
from ._operations import JUST_MEASURE_OPS, JUST_RESET_OPS, MEASURE_AND_RESET_OPS, NOISE, op_type
from .channels import AbstractPauliChannel, PauliChannel

####################################################################################################
# noise rules


class NoiseRule:
    """Describes how to add noise to an operation.

    This class encapsulates the noise channels and measurement error probabilities that should be
    applied to a particular type of quantum operation.

    Args:
        after: Noise applied after each matching operation.  The noise is broadcast across all
            of the instruction's qubit targets.  May be any of the following:

            - ``PauliChannel``: a joint Pauli channel of some definite arity ``k``.  Emitted
              natively via ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` when ``k ≤ 2`` or as a
              ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chain per ``k``-qubit block for
              ``k ≥ 3``.
            - ``PauliChannelSequence``: a sequence of independent Pauli-string error mechanisms.
            - ``Mapping[str, float]``: syntactic sugar for a ``PauliChannel``.  Keys are
              non-identity Pauli strings (e.g., ``"X"``, ``"IZ"``, ``"XYZ"``); values are
              probabilities.  Passed directly to ``PauliChannel(mapping)``.
            - ``stim.Circuit``: a fragment of noise instructions emitted verbatim after each
              ``k``-qubit block, with qubit indices remapped from ``[0, k)`` in the fragment
              to the corresponding block's targets.  The fragment's arity is inferred from
              ``circuit.num_qubits``.  Every instruction must be a Stim noise channel, such as
              ``X_ERROR``, ``DEPOLARIZE1``, or ``PAULI_CHANNEL_1``.  Gates, measurements,
              resets, annotations such as ``TICK``, heralded noise channels (which produce
              measurement results), repeat blocks, and fragments that address no qubits are
              rejected.  ``PauliChannel`` and ``Mapping`` arguments are preferred when they
              suffice, as the ``stim.Circuit`` form skips native broadcasting.

        readout_error: The probability that a measurement result is reported incorrectly.  Only
            allowed for operations that produce measurement results.  ``None`` (the default)
            means the field is unset (validation ignores it); ``0.0`` is an explicit no-op flip
            probability that is still validated against the gate type.
        reset_error: The probability that a qubit is reset to the wrong state.  Only allowed for
            operations that reset qubits.  Same ``None`` vs. ``0.0`` distinction as
            ``readout_error``.

    Raises:
        ValueError: If any Pauli string is invalid, any probability is not in [0, 1], or the
            ``stim.Circuit`` form contains an instruction other than a noise channel without
            measurement results or addresses no qubits.
        TypeError: If the ``stim.Circuit`` form contains a repeat block.
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

        noisy_op = stim.CircuitInstruction(op.name, targets, args, tag=op.tag)
        noise_after = stim.Circuit()

        if self.reset_error:
            if op.name not in JUST_RESET_OPS and op.name not in MEASURE_AND_RESET_OPS:
                raise ValueError(f"reset_error is not valid for operation {op.name!r}")
            # Reset gate `RB` (or `MRB`) prepares/resets in Pauli basis B ∈ {X, Y, Z}; a "wrong"
            # reset is modeled by an error that anticommutes with B.  Z-basis (canonical `R`/`MR`)
            # and Y-basis (`RY`/`MRY`) resets get X_ERROR; X-basis (`RX`/`MRX`) gets Z_ERROR.
            error_name = "Z_ERROR" if op.name.endswith("X") else "X_ERROR"
            error_op = stim.CircuitInstruction(error_name, qubit_targets, [self.reset_error])
            noise_after.append(error_op)

        self.emit_after(noise_after, qubit_targets, context=f"operation {op.name!r}")
        return noisy_op, noise_after

    def emit_after(
        self, circuit: StimCircuitLike, qubit_targets: list[int], *, context: str = "operation"
    ) -> None:
        """Append this rule's ``after`` noise in-place to the provided circuit.

        If the operation packages multiple independent gates (e.g., ``H 0 1 2`` — three ``H`` gates
        in one instruction), the noise is applied per gate.  For broadcast forms (``PauliChannel``
        of arity 1 or 2), this is achieved by a single emission on all targets, since stim's
        ``PAULI_CHANNEL_1``/``PAULI_CHANNEL_2`` broadcast natively.  For higher-arity
        ``PauliChannel`` and for the ``stim.Circuit`` form, the fragment is emitted once per gate.

        Args:
            circuit: The circuit to append the noise instructions to.
            qubit_targets: The qubits the noise applies to (in the operation's target order).
            context: A short description of the operation, used only in error messages.

        Raises:
            ValueError: If ``len(qubit_targets)`` is not a multiple of the ``after`` rule's arity
                (i.e. the operation cannot be split into an integer number of individual gates
                matching the rule).
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
            # Emit once per k-qubit chunk.
            for i in range(0, n_targets, num_qubits):
                self.after.to_circuit(append_to=circuit, qubits=qubit_targets[i : i + num_qubits])
        else:
            # stim.Circuit escape hatch: apply the fragment verbatim to each k-qubit block, with
            # qubit indices remapped from [0, k) to the block's targets.
            assert isinstance(self.after, stim.Circuit)
            for i in range(0, n_targets, num_qubits):
                circuit += with_remapped_qubits(self.after, qubit_targets[i : i + num_qubits])


# The accepted forms for a single noise-error specification (as used by `NoiseModel`'s clifford /
# idle error arguments).  Each is normalized to a `NoiseRule` by `_as_noise_rule`: a float is a
# uniform depolarizing channel, a Mapping is auto-wrapped as a `PauliChannel`, and an
# `AbstractPauliChannel` / `NoiseRule` is used directly.
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
    """Normalize a noise-error argument to a NoiseRule (or None if the input is ``None``).

    A float scalar is interpreted as a uniform depolarizing channel of the given ``default_arity``.
    A ``PauliChannel`` / ``PauliChannelSequence`` — or a raw ``Mapping[str, float]`` of Pauli-string
    probabilities, which is auto-wrapped as a ``PauliChannel`` — is used as the rule's ``after``.
    Storing the depolarizing shape as a ``PauliChannel`` (rather than a Mapping-form
    ``DEPOLARIZE{n}`` fragment) preserves the Pauli-channel structure so downstream
    ``PauliChannel.conditioned_on`` can project it under partial immunity.

    Does NOT trivialize an empty NoiseRule to ``None`` — callers must run their guard checks on the
    declared shape (e.g. arity) BEFORE trivializing, so that user typos like
    ``PauliChannel({}, num_qubits=3)`` in a 1-qubit-only slot are surfaced.  Trivialize via
    ``rule or None`` after the guard checks.
    """
    if isinstance(error, NoiseRule):
        return error
    if error is False:
        return None
    if isinstance(error, AbstractPauliChannel):
        return NoiseRule(after=error)
    if isinstance(error, Mapping):
        return NoiseRule(after=PauliChannel(error))
    if error is None:
        return None
    if isinstance(error, bool) or not isinstance(error, (int, float)):
        raise TypeError(
            f"expected a float, PauliChannel, PauliChannelSequence, Mapping[str, float], or "
            f"NoiseRule; got {type(error).__name__!r}"
        )
    return NoiseRule(after=PauliChannel.depolarizing(default_arity, float(error)))


def _normalize_clifford_nq_error(
    error: Mapping[int, ErrorSpec] | None,
) -> dict[int, NoiseRule]:
    """Normalize the ``clifford_nq_error`` argument to a ``dict[int, NoiseRule]``.

    - Floats become uniform ``k``-qubit depolarizing noise: ``DEPOLARIZE1`` for ``k == 1``,
      ``DEPOLARIZE2`` for ``k == 2``, and ``PauliChannel.depolarizing`` otherwise.
    - ``PauliChannel`` / ``PauliChannelSequence`` values (or raw Pauli-string
      ``Mapping[str, float]`` dicts, auto-wrapped as a ``PauliChannel``) become a
      ``NoiseRule(after=<channel>)``.
    - ``NoiseRule`` values are used directly.
    - Falsy entries (0.0, empty NoiseRule, empty channel) are dropped.
    - Entries are rejected if their joint Pauli channel's ``num_qubits`` disagrees with the key, if
      their ``after`` broadcast channels are incompatible with ``k`` qubits, or if they set
      ``readout_error`` / ``reset_error`` (Pauli-product Cliffords are neither).
    """
    if not error:
        return {}
    result: dict[int, NoiseRule] = {}
    for weight, entry in error.items():
        if weight < 1:
            raise ValueError(f"clifford_nq_error key {weight} must be >= 1")
        if isinstance(entry, NoiseRule):
            rule = entry
        elif isinstance(entry, AbstractPauliChannel):
            rule = NoiseRule(after=entry)
        elif isinstance(entry, Mapping):
            rule = NoiseRule(after=PauliChannel(entry))
        elif isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(
                f"clifford_nq_error[{weight}] has unsupported type {type(entry).__name__}; "
                "expected NoiseRule, PauliChannel, PauliChannelSequence, Mapping[str, float], int, "
                "or float"
            )
        else:
            if not (0 <= entry <= 1):
                raise ValueError(f"clifford_nq_error[{weight}]={entry} is not in [0, 1]")
            if entry == 0:
                continue
            rule = NoiseRule(after=PauliChannel.depolarizing(weight, entry))
        _validate_rule_for_arity(rule, weight, f"clifford_nq_error[{weight}]")
        if rule:
            result[weight] = rule
    return result


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
    """Reject a NoiseRule whose channels are ambiguous / incompatible on ``num_qubits`` qubits.

    - The rule's ``after`` arity must equal ``num_qubits`` unless ``after`` is ``None`` (the trivial
      default).  An explicitly-declared arity — even on an empty PauliChannel like
      ``PauliChannel({}, num_qubits=3)`` — is honored so callers can flag user typos.
    - ``readout_error`` is only meaningful if ``can_measure`` is True.
    - ``reset_error`` is only meaningful if ``can_reset`` is True.
    """
    if rule.after is not None and rule.after.num_qubits != num_qubits:
        raise ValueError(
            f"{context}: `after` has arity {rule.after.num_qubits}; expected {num_qubits}"
        )
    if rule.readout_error is not None and not can_measure:
        raise ValueError(f"{context}: `readout_error` is only valid on measurement gates")
    if rule.reset_error is not None and not can_reset:
        raise ValueError(f"{context}: `reset_error` is only valid on reset gates")


def _validate_custom_rule(rule: NoiseRule, op: stim.CircuitInstruction) -> None:
    """Reject a ``rule_func`` result that is incompatible with the gate application ``op``."""
    num_qubits = sum(1 for target in op.targets_copy() if not target.is_combiner)
    _validate_rule_for_arity(
        rule,
        num_qubits,
        context=f"rule_func returned a rule for {op.name!r}",
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
    """Return the ``NoiseRule`` to apply to ``op`` given ``immune_qubits``, or ``None`` to skip.

    Policy — ``op``'s qubits fall into three immunity states:

    - **None immune**: apply ``rule`` unchanged.
    - **All immune**: skip noise entirely (return ``None``); the op still appears in the output
      circuit, but no ``after``, ``readout_error``, or ``reset_error`` is applied.
    - **Mixed** (only possible for atomic ``k>=2`` ops the splitter cannot decompose further —
      2-qubit Clifford pairs, ``MXX``/``MYY``/``MZZ``, and MPP/SPP Pauli products):
        - ``immunize_gates=True`` (default): dead simple — drop everything (``after``,
          ``readout_error``, ``reset_error``).  The gate touches an immune qubit, so no noise is
          applied.
        - ``immunize_gates=False``: only ``PauliChannel`` ``after``-noise supports proper
          conditioning via ``PauliChannel.conditioned_on``.  Anything else (a ``stim.Circuit``
          ``after``, ``readout_error``, or ``reset_error``) has no defined projection on the
          identity-on-immune subspace and raises.
    """
    if not immune_qubits:
        return rule
    qubits = [t.qubit_value for t in op.targets_copy() if t.qubit_value is not None]
    immune_flags = [q in immune_qubits for q in qubits]
    if not any(immune_flags):
        return rule
    if all(immune_flags):
        return None
    # Partial immunity: the atom touches both immune and non-immune qubits.
    if immunize_gates:
        return None  # dead simple: no noise on gates that touch immune qubits.
    # immunize_gates=False: only PauliChannel `after` supports proper conditioning.
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
