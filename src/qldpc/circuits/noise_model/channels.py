# SPDX-License-Identifier: Apache-2.0

"""Pauli channels used to specify noise, and their composition into idle noise.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import abc
import functools
import itertools
import math
from collections.abc import Iterable, Mapping
from typing import overload

import stim

from ..common import StimCircuitLike

# the order of the probability arguments of Stim's PAULI_CHANNEL_1 and PAULI_CHANNEL_2 instructions
_PAULI_CHANNEL_1_ORDER = ("X", "Y", "Z")
_PAULI_CHANNEL_2_ORDER = tuple(a + b for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I"))


####################################################################################################
# Pauli channels


class AbstractPauliChannel(abc.ABC):
    """Base class for sparse Pauli channels.

    An ``AbstractPauliChannel`` maps non-identity Pauli strings to their probabilities.  Each
    subclass sets what the probabilities mean, checks any constraint on their sum, and defines
    ``to_circuit``.

    Args:
        probabilities: Mapping from non-identity Pauli strings to their probabilities.  All strings
            must have the same length ``n`` and contain only ``I``, ``X``, ``Y``, or ``Z``.  The
            all-identity string ``"I" * n`` must not appear.  Entries with probability zero are
            silently dropped.
        num_qubits: The number of qubits of the channel.  Defaults to the length of the Pauli
            strings, or to 0 if ``probabilities`` is empty.  It is needed only for an empty channel
            on one or more qubits.

    Raises:
        ValueError: If a Pauli string is invalid or is the identity, a probability is not in [0, 1],
            or ``num_qubits`` is negative or differs from the length of the Pauli strings.
    """

    def __init__(
        self, probabilities: Mapping[str, float], *, num_qubits: int | None = None
    ) -> None:
        """Validate and store a sparse Pauli-string -> probability mapping."""
        if num_qubits is not None and num_qubits < 0:
            raise ValueError(f"num_qubits={num_qubits} must be >= 0")
        if not probabilities:
            self._num_qubits = num_qubits if num_qubits is not None else 0
            self._probabilities: dict[str, float] = {}
            return

        first_key = next(iter(probabilities))
        derived_num_qubits = len(first_key)
        if num_qubits is not None and num_qubits != derived_num_qubits:
            raise ValueError(
                f"num_qubits={num_qubits} disagrees with Pauli string length {derived_num_qubits}"
            )

        identity = "I" * derived_num_qubits
        for string, prob in probabilities.items():
            if len(string) != derived_num_qubits:
                raise ValueError(
                    f"All Pauli strings must have length {derived_num_qubits}; got {string!r}"
                )
            if any(pauli not in "IXYZ" for pauli in string):
                raise ValueError(
                    f"Pauli string {string!r} contains invalid characters (allowed: I, X, Y, Z)"
                )
            if string == identity:
                raise ValueError(f"Identity string {string!r} is implicit and must not be listed")
            if not (0 <= prob <= 1):
                raise ValueError(f"Probability {prob} for {string!r} is not in [0, 1]")

        nonzero = {
            string: probabilities[string]
            for string in sorted(probabilities)
            if probabilities[string] > 0
        }
        self._num_qubits = derived_num_qubits
        self._probabilities = nonzero

    def __repr__(self) -> str:
        class_name = type(self).__name__
        if not self._probabilities and self._num_qubits:
            return f"{class_name}({{}}, num_qubits={self._num_qubits})"
        return f"{class_name}({dict(self._probabilities)!r})"

    def __hash__(self) -> int:
        return hash((type(self), self._num_qubits, tuple(self._probabilities.items())))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._num_qubits == other._num_qubits and self._probabilities == other._probabilities

    def __bool__(self) -> bool:
        """Is this channel nontrivial?"""
        return bool(self._probabilities)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the channel acts on."""
        return self._num_qubits

    @property
    def probabilities(self) -> Mapping[str, float]:
        """Non-identity Pauli-string → probability mapping."""
        return self._probabilities

    @overload
    def to_circuit(
        self, *, append_to: None = None, qubits: Iterable[int] | None = None, tag: str = ""
    ) -> stim.Circuit: ...

    @overload
    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike,
        qubits: Iterable[int] | None = None,
        tag: str = "",
    ) -> StimCircuitLike: ...

    @abc.abstractmethod
    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike | None = None,
        qubits: Iterable[int] | None = None,
        tag: str = "",
    ) -> StimCircuitLike | stim.Circuit:
        """Convert this channel into a circuit."""

    def _resolve_targets(
        self, append_to: StimCircuitLike | None, qubits: Iterable[int] | None
    ) -> tuple[StimCircuitLike | stim.Circuit, list[int]]:
        """Resolve the ``(circuit, qubit_targets)`` pair shared by every variant of ``to_circuit``.

        Creates a fresh circuit when ``append_to`` is ``None``, defaults ``qubits`` to
        ``range(num_qubits)``, and validates that the number of qubits matches the channel arity.
        """
        circuit = stim.Circuit() if append_to is None else append_to
        qubit_targets = list(range(self._num_qubits)) if qubits is None else list(qubits)
        if len(qubit_targets) != self.num_qubits:
            raise ValueError(
                f"Provided {len(qubit_targets)} qubits for a {self.num_qubits}-qubit channel"
            )
        return circuit, qubit_targets


class PauliChannel(AbstractPauliChannel):
    """A sparse multi-qubit Pauli channel.

    Maps non-identity Pauli strings to their probabilities.  The strings are mutually exclusive
    outcomes, so the channel applies exactly one of them or the identity.  The identity has
    probability 1 minus the sum of the listed probabilities, which must be at most 1.

    Args:
        probabilities: Mapping from non-identity Pauli strings to their probabilities.  All strings
            must have the same length ``n`` and contain only ``I``, ``X``, ``Y``, or ``Z``.  The
            all-identity string ``"I" * n`` must not appear.  Entries with probability zero are
            silently dropped.
        num_qubits: The number of qubits of the channel.  Defaults to the length of the Pauli
            strings, or to 0 if ``probabilities`` is empty.  It is needed only for an empty channel
            on one or more qubits.

    Raises:
        ValueError: If a Pauli string is invalid or is the identity, a probability is not in [0, 1],
            the probabilities sum to more than 1, or ``num_qubits`` is negative or differs from the
            length of the Pauli strings.
    """

    def __init__(
        self, probabilities: Mapping[str, float], *, num_qubits: int | None = None
    ) -> None:
        """Instantiate a Pauli channel."""
        super().__init__(probabilities, num_qubits=num_qubits)

        # the probabilities must sum to at most 1, up to the rounding errors of normalized inputs
        total = math.fsum(self._probabilities.values())
        if total > 1 + 1e-15:
            raise ValueError(f"Sum of Pauli channel probabilities {total} is not in [0, 1]")

    @staticmethod
    def depolarizing(num_qubits: int, probability: float) -> PauliChannel:
        """Uniform ``num_qubits``-qubit depolarizing channel with total error ``probability``.

        Each of the ``4**num_qubits - 1`` non-identity Pauli strings has an equal share of
        ``probability``.

        Raises:
            ValueError: If ``num_qubits`` is less than 1, or ``probability`` is not in [0, 1].
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits={num_qubits} must be >= 1")
        if not (0 <= probability <= 1):
            raise ValueError(f"probability={probability} is not in [0, 1]")
        term_probability = probability / (4**num_qubits - 1)
        # the first Pauli string of the product is the identity
        strings = itertools.islice(itertools.product("IXYZ", repeat=num_qubits), 1, None)
        return PauliChannel({"".join(string): term_probability for string in strings})

    def conditioned_on(self, immune_qubits: Iterable[int]) -> PauliChannel:
        """Return the part of this channel that acts as the identity on the given positions.

        Keeps the Pauli strings that are ``I`` at every position in ``immune_qubits``, with their
        probabilities.  The strings keep their length, so the result has the same ``num_qubits`` as
        this channel.  Its total probability is lower if it drops strings that act on an immune
        position.

        Args:
            immune_qubits: Positions in ``[0, num_qubits)`` at which to keep only the identity.
                Repeats are ignored.

        Returns:
            A ``PauliChannel`` with the same ``num_qubits`` as this channel.  It is empty if no
            strings survive.

        Raises:
            ValueError: If a position is outside ``[0, num_qubits)``.
        """
        immune_qubits = frozenset(immune_qubits)
        for position in immune_qubits:
            if not 0 <= position < self._num_qubits:
                raise ValueError(f"index {position} not in [0, {self._num_qubits})")
        probabilities = {
            string: probability
            for string, probability in self._probabilities.items()
            if all(string[position] == "I" for position in immune_qubits)
        }
        return PauliChannel(probabilities, num_qubits=self._num_qubits)

    @overload
    def to_circuit(
        self,
        *,
        append_to: None = None,
        qubits: Iterable[int] | None = None,
        simplify: bool = True,
        tag: str = "",
    ) -> stim.Circuit: ...

    @overload
    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike,
        qubits: Iterable[int] | None = None,
        simplify: bool = True,
        tag: str = "",
    ) -> StimCircuitLike: ...

    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike | None = None,
        qubits: Iterable[int] | None = None,
        simplify: bool = True,
        tag: str = "",
    ) -> StimCircuitLike | stim.Circuit:
        """Convert this PauliChannel into a circuit.

        With ``simplify=True``, a channel that acts nontrivially on one or two qubits becomes one
        ``X_ERROR``, ``Y_ERROR``, ``Z_ERROR``, ``DEPOLARIZE1``, ``DEPOLARIZE2``,
        ``PAULI_CHANNEL_1``, or ``PAULI_CHANNEL_2`` instruction.  Other channels, and every channel
        with ``simplify=False``, become a ``CORRELATED_ERROR`` followed by one
        ``ELSE_CORRELATED_ERROR`` per remaining Pauli string.  The conditional probabilities of this
        chain give each string its probability in this channel.

        Stim needs ``approximate_disjoint_errors=True`` to make a detector error model from a chain
        with an ``ELSE_CORRELATED_ERROR``, or from a ``PAULI_CHANNEL_1`` or ``PAULI_CHANNEL_2`` with
        more than one nonzero probability.

        Args:
            append_to: The circuit to append to; if ``None`` (default), create a new circuit.
            qubits: The qubit targets, one per position of the channel; if ``None`` (default), set
                to ``range(self.num_qubits)``.
            simplify: If True (the default), emit the channel in the simplest equivalent form.  If
                False, always emit a ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chain.
            tag: An optional tag applied to every emitted instruction.

        Returns:
            The circuit that the noise instructions were appended to.

        Raises:
            ValueError: If the number of ``qubits`` does not match the arity of the channel.
        """
        circuit, qubits = self._resolve_targets(append_to, qubits)
        if not self.probabilities:
            return circuit

        # a channel that acts nontrivially on one or two qubits becomes one instruction on them
        if simplify and self._simple_form is not None:
            name, positions, args = self._simple_form
            circuit.append(name, [qubits[i] for i in positions], args, tag=tag)
            return circuit

        # otherwise, emit a chain of CORRELATED_ERROR and ELSE_CORRELATED_ERROR instructions
        remaining = 1.0
        first = True
        for string, prob in self.probabilities.items():
            pauli_targets = _pauli_string_to_targets(string, qubits)
            if first:
                circuit.append("CORRELATED_ERROR", pauli_targets, [prob], tag=tag)
                first = False
            else:
                # rounding can leave `remaining` at or below `prob`, although the probabilities sum
                # to at most 1.  ELSE_CORRELATED_ERROR(1) then takes all of the remaining
                # probability, and later strings could never fire.
                if remaining <= prob:
                    circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [1.0], tag=tag)
                    break
                circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [prob / remaining], tag=tag)
            remaining -= prob
        return circuit

    @functools.cached_property
    def _simple_form(self) -> tuple[str, tuple[int, ...], tuple[float, ...]] | None:
        """The simplest Stim instruction of this channel, if it acts on one or two qubits.

        Noise models emit the same channel after many gates, so the instruction is found once.

        Returns:
            The name of the instruction, the positions of the qubits of this channel that it acts
            on, and its arguments, or None if this channel acts nontrivially on no qubits or on more
            than two.
        """
        # the positions of the qubits that the channel acts on nontrivially
        positions = tuple(
            sorted({i for string in self.probabilities for i, p in enumerate(string) if p != "I"})
        )

        # Single-qubit channel: PAULI_CHANNEL_1 / DEPOLARIZE1 / [XYZ]_ERROR
        if len(positions) == 1:
            probs_1q = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            for string, prob in self.probabilities.items():
                probs_1q[string[positions[0]]] = prob
            args = [probs_1q[p] for p in _PAULI_CHANNEL_1_ORDER]
            name, name_args = _pauli_channel_1_shortcut(args)
            return name, positions, tuple(name_args)

        # Two-qubit channel: PAULI_CHANNEL_2 / DEPOLARIZE2
        if len(positions) == 2:
            pos0, pos1 = positions
            probs_2q = {pair: 0.0 for pair in _PAULI_CHANNEL_2_ORDER}
            for string, prob in self.probabilities.items():
                probs_2q[string[pos0] + string[pos1]] = prob
            args = [probs_2q[p] for p in _PAULI_CHANNEL_2_ORDER]
            name, name_args = _pauli_channel_2_shortcut(args)
            return name, positions, tuple(name_args)

        return None


class PauliChannelSequence(AbstractPauliChannel):
    """A sequence of independent Pauli-string error mechanisms.

    Like :class:`PauliChannel`, this maps non-identity Pauli strings to probabilities.  Its entries
    are independent events, rather than mutually exclusive outcomes, so several may fire together.
    """

    @overload
    def to_circuit(
        self, *, append_to: None = None, qubits: Iterable[int] | None = None, tag: str = ""
    ) -> stim.Circuit: ...

    @overload
    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike,
        qubits: Iterable[int] | None = None,
        tag: str = "",
    ) -> StimCircuitLike: ...

    def to_circuit(
        self,
        *,
        append_to: StimCircuitLike | None = None,
        qubits: Iterable[int] | None = None,
        tag: str = "",
    ) -> StimCircuitLike | stim.Circuit:
        """Emit each entry of this channel as an independent ``CORRELATED_ERROR``.

        Args:
            append_to: The circuit to append to; if ``None`` (default), create a new circuit.
            qubits: The qubit targets, one per position of the channel; if ``None`` (default), set
                to ``range(self.num_qubits)``.
            tag: An optional tag applied to every emitted instruction.

        Returns:
            The circuit that the noise instructions were appended to.

        Raises:
            ValueError: If the number of ``qubits`` does not match the arity of the channel.
        """
        circuit, qubits = self._resolve_targets(append_to, qubits)
        for string, prob in self.probabilities.items():
            pauli_targets = _pauli_string_to_targets(string, qubits)
            circuit.append("CORRELATED_ERROR", pauli_targets, [prob], tag=tag)
        return circuit


def _pauli_channel_1_shortcut(args: list[float]) -> tuple[str, list[float]]:
    """Return the compact stim ``(name, args)`` for a 1-qubit Pauli-channel probability vector.

    Gives ``DEPOLARIZE1(p)`` if the three probabilities are equal and nonzero and their sum ``p`` is
    at most 3/4, and ``X_ERROR(p)``, ``Y_ERROR(p)``, or ``Z_ERROR(p)`` if only one probability ``p``
    is nonzero.  Gives ``PAULI_CHANNEL_1`` otherwise.
    """
    if _is_uniform_depolarizing(args) and sum(args) <= 0.75:
        return "DEPOLARIZE1", [sum(args)]
    nonzero_positions = [i for i, x in enumerate(args) if x > 0]
    if len(nonzero_positions) == 1:
        (i,) = nonzero_positions
        return f"{_PAULI_CHANNEL_1_ORDER[i]}_ERROR", [args[i]]
    return "PAULI_CHANNEL_1", args


def _pauli_channel_2_shortcut(args: list[float]) -> tuple[str, list[float]]:
    """Return the compact stim ``(name, args)`` for a 2-qubit Pauli-channel probability vector.

    Gives ``DEPOLARIZE2(p)`` if the fifteen probabilities are equal and nonzero and their sum ``p``
    is at most 15/16, and ``PAULI_CHANNEL_2`` otherwise.
    """
    if _is_uniform_depolarizing(args) and sum(args) <= 15 / 16:
        return "DEPOLARIZE2", [sum(args)]
    return "PAULI_CHANNEL_2", args


def _is_uniform_depolarizing(args: list[float]) -> bool:
    """Return True if all ``args`` are nonzero and equal, up to rounding errors.

    Detects a ``PauliChannel.depolarizing(k, p)`` shape, so that it can be written as
    ``DEPOLARIZE{k}(p)`` rather than as ``PAULI_CHANNEL_{k}`` with equal arguments.
    """
    return (
        bool(args)
        and args[0] > 0
        and all(math.isclose(x, args[0], rel_tol=1e-12, abs_tol=1e-15) for x in args[1:])
    )


def _pauli_string_to_targets(string: str, qubit_targets: list[int]) -> list[stim.GateTarget]:
    """Convert a Pauli string over the given qubits to a list of Pauli-typed stim targets."""
    return [
        stim.target_pauli(qubit, pauli)
        for pauli, qubit in zip(string, qubit_targets, strict=True)
        if pauli != "I"
    ]


####################################################################################################
# composing one-qubit Pauli channels, for idle noise


def _repeated_idle_channel(
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
    single = _pauli_distribution(channel)
    result: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    while repetitions:
        if repetitions & 1:
            result = _compose_pauli_distributions(result, single)
        repetitions >>= 1
        if repetitions:
            single = _compose_pauli_distributions(single, single)
    return PauliChannel(dict(zip("XYZ", result[1:], strict=True)))


def _quadratic_idle_channel(channel: PauliChannel, num_moments: int) -> PauliChannel:
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


def _compose_idle_channels(
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
            distribution = _compose_pauli_distributions(distribution, _pauli_distribution(channel))
    return PauliChannel(dict(zip("XYZ", distribution[1:], strict=True)))


def _pauli_distribution(channel: PauliChannel) -> tuple[float, ...]:
    """The probabilities of I, X, Y, and Z in a one-qubit Pauli channel."""
    probabilities = tuple(channel.probabilities.get(pauli, 0.0) for pauli in "XYZ")
    return (max(0.0, 1 - math.fsum(probabilities)), *probabilities)


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
