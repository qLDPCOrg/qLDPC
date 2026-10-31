# SPDX-License-Identifier: Apache-2.0

"""Pauli channels used to specify noise.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import abc
import copy
import itertools
import math
from collections.abc import Iterable, Mapping
from typing import overload

import stim

from ..common import StimCircuitLike

# Floating-point tolerance used for probability comparisons throughout this module.  Small enough to
# catch real bugs, large enough to absorb the ~O(n * eps) drift that accumulates when summing or
# renormalizing many probabilities.
_ABSOLUTE_ERROR_TOLERANCE = 1e-9

# the order of the probability arguments of Stim's PAULI_CHANNEL_1 and PAULI_CHANNEL_2 instructions
_PAULI_CHANNEL_1_ORDER = ("X", "Y", "Z")
_PAULI_CHANNEL_2_ORDER = tuple(a + b for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I"))


####################################################################################################
# Pauli channels


class AbstractPauliChannel(abc.ABC):
    """Base class for sparse Pauli channels.

    Every ``AbstractPauliChannel`` maps non-identity Pauli strings (over the alphabet
    ``{I, X, Y, Z}``) to their probabilities.  The interpretation of those probabilities depends on
    the ``AbstractPauliChannel`` subclass.

    Subclasses of ``AbstractPauliChannel`` are required to define a ``.to_circuit`` method.

    Performs only *per-entry* validation (string shape / alphabet, not-identity, each
    probability in [0, 1]).  Any aggregate constraint (such as normalization) is left to the
    subclass.

    Args:
        probabilities: Mapping from non-identity Pauli strings to their probabilities.  All
            strings must have the same length ``n`` and contain only ``I``, ``X``, ``Y``, or
            ``Z``.  The all-identity string ``"I" * n`` must not appear.  Entries with
            probability zero are silently dropped.
        num_qubits: The arity of the channel.  Required only for an empty ``probabilities`` on a
            nontrivial number of qubits — otherwise defaults to the length of the Pauli strings
            when ``probabilities`` is non-empty, or ``0`` when empty.  If both are supplied and
            inconsistent, a ``ValueError`` is raised.

    Raises:
        ValueError: If the input contains an invalid Pauli string, contains the identity string,
            any probability is not in [0, 1], or ``num_qubits`` disagrees with the length of the
            Pauli strings.
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
        if not isinstance(other, AbstractPauliChannel) or type(other) is not type(self):
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

    Maps non-identity Pauli strings (over the alphabet ``{I, X, Y, Z}``) to their probabilities.
    The strings name mutually exclusive outcomes: this channel applies exactly one Pauli string
    (possibly the identity).  The all-identity string is implicit; its probability is
    ``1 - sum(others)``, so the listed probabilities must sum to at most 1.

    Args:
        probabilities: Mapping from non-identity Pauli strings to their probabilities.  All
            strings must have the same length ``n`` and contain only ``I``, ``X``, ``Y``, or
            ``Z``.  The all-identity string ``"I" * n`` must not appear.  Entries with
            probability zero are silently dropped.
        num_qubits: The arity of the channel.  Required only for an empty ``probabilities`` on a
            nontrivial number of qubits — otherwise defaults to the length of the Pauli strings
            when ``probabilities`` is non-empty, or ``0`` when empty.  If both are supplied and
            inconsistent, a ``ValueError`` is raised.

    Raises:
        ValueError: If the input contains an invalid Pauli string, contains the identity string,
            any probability is not in [0, 1], the sum of probabilities is not in [0, 1], or
            ``num_qubits`` disagrees with the length of the Pauli strings.
    """

    def __init__(
        self, probabilities: Mapping[str, float], *, num_qubits: int | None = None
    ) -> None:
        """Instantiate a Pauli channel."""
        super().__init__(probabilities, num_qubits=num_qubits)

        # Ensure the sum of all probabilities does not exceed 1.  Use math.fsum for a precise sum,
        # and allow a small tolerance so a mathematically normalized input isn't rejected due to
        # per-value rounding (e.g., ``[p_i / sum(p_i) for p_i in ...]`` can accumulate to
        # ~1 + O(n * eps)).
        total = math.fsum(self._probabilities.values())
        if not (-1e-15 <= total <= 1 + 1e-15):
            raise ValueError(f"Sum of Pauli channel probabilities {total} is not in [0, 1]")

    @staticmethod
    def depolarizing(num_qubits: int, probability: float) -> PauliChannel:
        """Uniform ``num_qubits``-qubit depolarizing channel with total error ``probability``.

        Each of the ``4**num_qubits - 1`` non-identity Pauli strings is assigned probability
        ``probability / (4**num_qubits - 1)``.  Strings are inserted in lexicographic order.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits={num_qubits} must be >= 1")
        if not (0 <= probability <= 1):
            raise ValueError(f"probability={probability} is not in [0, 1]")
        num_terms = 4**num_qubits - 1
        weight = probability / num_terms
        identity = "I" * num_qubits
        probs: dict[str, float] = {}
        for tup in itertools.product("IXYZ", repeat=num_qubits):
            string = "".join(tup)
            if string != identity:
                probs[string] = weight
        return PauliChannel(probs)

    def conditioned_on(self, immune_qubits: Iterable[int]) -> PauliChannel:
        """Return the sub-channel of error mechanisms that act as identity on the given qubits.

        Keeps only Pauli strings whose positions in ``immune_qubits`` are all ``I``, with their
        original probabilities.  The returned channel has the same ``num_qubits`` as ``self`` —
        surviving strings retain their length; the immune positions are still present, always as
        ``I``.  Total probability is in general reduced (the dropped weight represents error events
        that would have acted nontrivially on an immune qubit, which are assumed to not occur).

        Args:
            immune_qubits: Positions in ``[0, num_qubits)`` to constrain to identity.  Repeats are
                ignored.

        Returns:
            A ``PauliChannel`` on the same qubits as ``self``.  If no strings survive, the returned
                channel is empty but still reports ``num_qubits == self.num_qubits``.

        Raises:
            ValueError: If any index is outside ``[0, num_qubits)``.
        """
        immune_qubits = frozenset(immune_qubits)
        for qubit in immune_qubits:
            if not 0 <= qubit < self._num_qubits:
                raise ValueError(f"index {qubit} not in [0, {self._num_qubits})")
        result = copy.deepcopy(self)
        if immune_qubits:
            result._probabilities = {
                string: prob
                for string, prob in result._probabilities.items()
                if all(string[i] == "I" for i in immune_qubits)
            }
        return result

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

        If provided a circuit, append to that circuit in-place and return it.

        By default (with ``simplify=True``), the channel is emitted in the simplest form its
        probabilities permit: channels that address only one or two qubits nontrivially become
        ``PAULI_CHANNEL_[12]`` / ``DEPOLARIZE[12]`` / ``[XYZ]_ERROR``.

        With ``simplify=False`` the channel is always emitted as a ``CORRELATED_ERROR`` followed by
        one ``ELSE_CORRELATED_ERROR`` per remaining Pauli string, with conditional probabilities
        renormalized so each string's unconditional firing probability matches this channel.  For
        channels with three or more active qubits, this correlated chain requires
        ``approximate_disjoint_errors=True`` when converting a containing Stim circuit to a detector
        error model.

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

        if simplify:
            # restrict to the qubits addressed nontrivially by this PauliChannel
            positions = sorted(
                {i for string in self.probabilities for i, p in enumerate(string) if p != "I"}
            )
            target_qubits = [qubits[i] for i in positions]

            # Single-qubit channel: PAULI_CHANNEL_1 / DEPOLARIZE1 / [XYZ]_ERROR
            if len(positions) == 1:
                probs_1q = {"X": 0.0, "Y": 0.0, "Z": 0.0}
                for string, prob in self.probabilities.items():
                    probs_1q[string[positions[0]]] = prob
                args = [probs_1q[p] for p in _PAULI_CHANNEL_1_ORDER]
                name, name_args = _pauli_channel_1_shortcut(args)
                circuit.append(name, target_qubits, name_args, tag=tag)
                return circuit

            # Two-qubit channel: PAULI_CHANNEL_2 / DEPOLARIZE2
            if len(positions) == 2:
                pos0, pos1 = positions
                probs_2q = {pair: 0.0 for pair in _PAULI_CHANNEL_2_ORDER}
                for string, prob in self.probabilities.items():
                    probs_2q[string[pos0] + string[pos1]] = prob
                args = [probs_2q[p] for p in _PAULI_CHANNEL_2_ORDER]
                name, name_args = _pauli_channel_2_shortcut(args)
                circuit.append(name, target_qubits, name_args, tag=tag)
                return circuit

        # Emit a CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain
        remaining = 1.0
        first = True
        for string, prob in self.probabilities.items():
            pauli_targets = _pauli_string_to_targets(string, qubits)
            if first:
                circuit.append("CORRELATED_ERROR", pauli_targets, [prob], tag=tag)
                first = False
            else:
                # `remaining` is `1 - sum_of_prior` in exact arithmetic and is guaranteed >= prob
                # because the constructor rejects total > 1.  If floating-point subtraction leaves
                # `remaining` a hair smaller than prob (or zero), emit an ELSE_CORRELATED_ERROR(1)
                # that absorbs the rest of the probability mass and stop — any subsequent
                # ELSE_CORRELATED_ERROR would never fire.
                if remaining <= prob:
                    circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [1.0], tag=tag)
                    break
                circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [prob / remaining], tag=tag)
            remaining -= prob
        return circuit


class PauliChannelSequence(AbstractPauliChannel):
    """A sequence of independent Pauli-string error mechanisms.

    Like :class:`PauliChannel`, this maps non-identity Pauli strings to probabilities, but the
    entries are *independent* events rather than mutually-exclusive outcomes: each fires (or not) on
    its own, and several may fire together.
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

        If provided a circuit, append to that circuit in-place and return it.

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

    Emits ``DEPOLARIZE1(p)`` when the three probabilities are equal and non-zero, or
    ``{X,Y,Z}_ERROR(p)`` when exactly one is non-zero, else falls back to ``PAULI_CHANNEL_1``.
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

    Emits ``DEPOLARIZE2(p)`` when the fifteen probabilities are equal and non-zero, else falls back
    to ``PAULI_CHANNEL_2``.
    """
    if _is_uniform_depolarizing(args) and sum(args) <= 15 / 16:
        return "DEPOLARIZE2", [sum(args)]
    return "PAULI_CHANNEL_2", args


def _is_uniform_depolarizing(args: list[float], *, atol: float = _ABSOLUTE_ERROR_TOLERANCE) -> bool:
    """Return True if all ``args`` are equal (up to ``atol``) and non-zero.

    Detects a ``PauliChannel.depolarizing(k, p)`` shape at emit time so it can be written as
    ``DEPOLARIZE{k}(p)`` instead of ``PAULI_CHANNEL_{k}(p/n, ..., p/n)``.
    """
    return (
        bool(args)
        and args[0] > 0
        and all(math.isclose(x, args[0], rel_tol=1e-12, abs_tol=min(atol, 1e-15)) for x in args[1:])
    )


def _pauli_string_to_targets(string: str, qubit_targets: list[int]) -> list[stim.GateTarget]:
    """Convert a Pauli string over the given qubits to a list of Pauli-typed stim targets."""
    return [
        stim.target_pauli(qubit, pauli)
        for pauli, qubit in zip(string, qubit_targets, strict=True)
        if pauli != "I"
    ]
