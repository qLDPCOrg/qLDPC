"""Implementation of noise models for Stim (and tsim) circuits

The main components of this module are:
- PauliChannel: A sparse multi-qubit Pauli channel used to specify multi-qubit noise.
- NoiseRule: Defines how to add noise to individual operations.
- NoiseModel: Defines how noise is added to circuits.
- Built-in noise models: DepolarizingNoiseModel and the superconducting-inspired SI1000NoiseModel.

Examples of basic usage with a predefined noise model:

    import stim
    from qldpc.circuits.noise_model import DepolarizingNoiseModel, NoiseModel, SI1000NoiseModel

    # Create a simple circuit
    circuit = stim.Circuit("H 0 \n CX 0 1")

    # Apply simple depolarizing noise
    noise_model = DepolarizingNoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Apply superconducting-inspired noise
    noise_model = SI1000NoiseModel(0.001)
    noisy_circuit = noise_model.noisy_circuit(circuit)

    # Create a custom noise model
    custom_model = NoiseModel(
        clifford_1q_error=3e-5,
        clifford_2q_error=1e-3,
        readout_error=1e-3,
        reset_error=1e-3,
        idle_error=2e-4,
    )
    noisy_circuit = custom_model.noisy_circuit(circuit)


Noise on multi-qubit Clifford gates (SPP / MPP):

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule, PauliChannel

    # `clifford_nq_error` maps a qubit count ``k`` to the noise applied after each ``k``-qubit
    # unitary Clifford gate (namely, the Pauli-product Cliffords SPP and SPP_DAG, which are stim's
    # multi-qubit unitary Clifford primitives).  Values may be a float (uniform ``k``-qubit
    # depolarizing channel), a Mapping[str, float] (auto-wrapped as PauliChannel), a PauliChannel,
    # or a full NoiseRule.
    noise_model = NoiseModel(
        clifford_nq_error={
            1: 1e-4,                                 # DEPOLARIZE1(1e-4) after each 1-qubit gate
            2: 1e-3,                                 # DEPOLARIZE2(1e-3) after each 2-qubit gate
            3: PauliChannel.depolarizing(3, 5e-3),   # 3-qubit depolarizing channel, emitted as
                                                     # a chain of CORRELATED_ERROR /
                                                     # ELSE_CORRELATED_ERROR instructions
            4: {"XXXX": 1e-4, "ZZZZ": 2e-4},         # sparse 4-qubit channel (raw dict)
        }
    )

    # Multi-qubit MPP / SPP gates receive ordinary readout_error / clifford_nq_error.  To assign
    # per-basis rules, use "M<paulis>" for MPP (e.g. "MXYZ" for `MPP X*Y*Z`, "MXX" for `MPP X*X`)
    # and "S<paulis>" / "S<paulis>_DAG" for SPP / SPP_DAG (e.g. "SXYZ" for `SPP X*Y*Z`, "SXY_DAG"
    # for `SPP_DAG X*Y`).
    noise_model = NoiseModel(
        readout_error=1e-3,                                # default readout flip probability
        rules={
            "MXYZ": NoiseRule(readout_error=5e-3),         # override for MPP X*Y*Z specifically
            "SXYZ": NoiseRule(after=PauliChannel.depolarizing(3, 1e-2)),  # SPP X*Y*Z override
        },
    )


Per-gate-application noise via a callback (`rule_func`):

    from qldpc.circuits.noise_model import NoiseModel, NoiseRule

    # `rule_func` takes top priority over every other rule in a `NoiseModel`.  Broadcast gates such
    # as `H 0 1 2` and `CX 0 1 2 3` are decomposed into individual gates before being passed to
    # `rule_func`, so the `op` argument is always a stim.CircuitInstruction holding exactly one gate
    # application's worth of targets (e.g. two targets for a two-qubit gate, one Pauli product for
    # an SPP/MPP).  Returning `None` falls back to the ordinary `NoiseModel` rules.
    def bad_qubit_noise(op: stim.CircuitInstruction) -> NoiseRule | None:
        # Give any gate touching qubit 7 a heavier depolarizing kick; leave everything else alone.
        targets = [t for t in op.targets_copy() if not t.is_combiner]
        if any(t.qubit_value == 7 for t in targets):
            channel = "DEPOLARIZE2" if len(targets) == 2 else "DEPOLARIZE1"
            return NoiseRule(after={channel: 1e-2})
        return None

    noise_model = NoiseModel(
        clifford_1q_error=1e-4,
        clifford_2q_error=1e-3,
        rule_func=bad_qubit_noise,
    )
    noisy_circuit = noise_model.noisy_circuit(circuit)

Important note:
---------------

This file was originally taken and modified from
    https://github.com/tqec/tqec/blob/main/src/tqec/utils/noise_model.py
which itself was taken from
    https://zenodo.org/records/7487893
and licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode).

The original code was written for the paper at "Inplace Access to the Surface Code Y Basis", at
    https://quantum-journal.org/papers/q-2024-04-08-1310.
"""

from __future__ import annotations

import collections
import functools
import itertools
import math
import re
import types
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, TypeVar

import stim

try:
    import tsim

    stim_or_tsim_Circuit = TypeVar("stim_or_tsim_Circuit", stim.Circuit, tsim.Circuit)
except ImportError:  # pragma: no cover
    if not TYPE_CHECKING:
        tsim = None
        stim_or_tsim_Circuit = TypeVar("stim_or_tsim_Circuit", bound=stim.Circuit)

from .common import with_remapped_qubits

####################################################################################################
# global constants

DEFAULT_IMMUNE_OP_TAG = "__IMMUNE_TO_NOISE__"

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
CLIFFORD_PP = "CPP"
JUST_MEASURE_1Q = "M1"
JUST_MEASURE_2Q = "M2"
JUST_MEASURE_PP = "MPP"
JUST_RESET_1Q = "R1"
MEASURE_RESET_1Q = "MR1"
ANNOTATION = "info"
NOISE = "!?"
UNSUPPORTED = "unsupported"

# The op-type categories that name a genuine noisy gate (as opposed to pure-noise, annotation, or
# unsupported instructions).  These are the operations a user-provided ``rule_func`` is
# consulted for, once decomposed into individual gate applications.
GATE_OP_TYPES = frozenset(
    {
        CLIFFORD_1Q,
        CLIFFORD_2Q,
        CLIFFORD_PP,
        JUST_MEASURE_1Q,
        JUST_MEASURE_2Q,
        JUST_MEASURE_PP,
        JUST_RESET_1Q,
        MEASURE_RESET_1Q,
    }
)

# Gates that stim knows about but qLDPC does not support (e.g., REPEAT is a block, not a gate,
# and cannot appear as an ordinary CircuitInstruction).
UNSUPPORTED_GATES = frozenset({"REPEAT"})


@functools.cache
def op_type(op_name: str) -> str:
    """Return the qLDPC op-type category for a gate name (canonical or alias)."""
    try:
        data = stim.gate_data(op_name)
    except IndexError:
        return UNSUPPORTED
    if any(alias in UNSUPPORTED_GATES for alias in data.aliases):
        return UNSUPPORTED
    if data.flows is None:
        # Pure noise (X_ERROR, DEPOLARIZE1, HERALDED_ERASE, ...) or annotation (TICK, DETECTOR,
        # MPAD, QUBIT_COORDS, SHIFT_COORDS, OBSERVABLE_INCLUDE).  These are the gates that lack a
        # tableau-based flow specification.
        return NOISE if data.is_noisy_gate else ANNOTATION
    if data.is_reset:
        assert data.is_single_qubit_gate
        if data.produces_measurements:
            return MEASURE_RESET_1Q
        return JUST_RESET_1Q
    if data.produces_measurements:
        if data.is_single_qubit_gate:
            return JUST_MEASURE_1Q
        if data.is_two_qubit_gate:
            return JUST_MEASURE_2Q
        assert data.takes_pauli_targets
        return JUST_MEASURE_PP
    if data.is_unitary:
        if data.is_single_qubit_gate:
            return CLIFFORD_1Q
        if data.is_two_qubit_gate:
            return CLIFFORD_2Q
        assert data.takes_pauli_targets
        return CLIFFORD_PP
    raise NotImplementedError(  # pragma: no cover
        f"qLDPC does not know how to classify the gate {op_name!r}."
        f"  Please open an issue at https://github.com/qLDPCOrg/qLDPC/issues."
    )


# All canonical + alias names stim knows about, plus their aliases (so lookups by any name work).
ALL_STIM_GATE_NAMES = frozenset(alias for gd in stim.gate_data().values() for alias in gd.aliases)

JUST_MEASURE_OPS = frozenset(
    name
    for name in ALL_STIM_GATE_NAMES
    if op_type(name) in (JUST_MEASURE_1Q, JUST_MEASURE_2Q, JUST_MEASURE_PP)
)
JUST_RESET_OPS = frozenset(name for name in ALL_STIM_GATE_NAMES if op_type(name) == JUST_RESET_1Q)
MEASURE_AND_RESET_OPS = frozenset(
    name for name in ALL_STIM_GATE_NAMES if op_type(name) == MEASURE_RESET_1Q
)
COLLAPSING_OPS = JUST_MEASURE_OPS | JUST_RESET_OPS | MEASURE_AND_RESET_OPS

# Noise instructions that stim broadcasts independently per qubit (any number of targets).
BROADCAST_1Q_NOISE = frozenset(
    name
    for name in ALL_STIM_GATE_NAMES
    if op_type(name) == NOISE and stim.gate_data(name).is_single_qubit_gate
)
# Noise instructions that stim broadcasts per (fixed-size) pair; require an even number of targets.
BROADCAST_2Q_NOISE = frozenset(
    name
    for name in ALL_STIM_GATE_NAMES
    if op_type(name) == NOISE and stim.gate_data(name).is_two_qubit_gate
)

CORRELATED_ERROR_NAMES = frozenset({"CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"})

####################################################################################################
# primary methods and classes: as_noiseless_circuit, PauliChannel, NoiseRule, NoiseModel


def as_noiseless_circuit(circuit: stim_or_tsim_Circuit) -> stim_or_tsim_Circuit:
    """Wrap a circuit in a noiseless, one-repitition stim.CircuitRepeatBlock."""
    if tsim is not None and isinstance(circuit, tsim.Circuit):
        return tsim.Circuit.from_stim_program(as_noiseless_circuit(circuit.stim_circuit))
    block = stim.CircuitRepeatBlock(repeat_count=1, body=circuit.copy(), tag=DEFAULT_IMMUNE_OP_TAG)
    noiseless_circuit = stim.Circuit()
    noiseless_circuit.append(block)
    return noiseless_circuit


class PauliChannel:
    """A sparse multi-qubit Pauli channel.

    Maps non-identity Pauli strings (over the alphabet ``{I, X, Y, Z}``) to their probabilities.
    The all-identity string is implicit; its probability is ``1 - sum(others)``.

    Pauli strings are in the absolute Pauli basis: slot ``k`` of a string maps to the ``k``-th
    non-combiner target of the operation the channel is applied to.

    Emitted as a chain of ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` instructions, whose
    conditional firing probabilities are renormalized so that each Pauli string's (unconditional)
    firing probability equals the value provided.
    """

    def __init__(self, probabilities: Mapping[str, float], *, num_qubits: int | None = None):
        """Instantiate a Pauli channel.

        Args:
            probabilities: Mapping from non-identity Pauli strings to their probabilities.  All
                strings must have the same length ``n`` and contain only ``I``, ``X``, ``Y``, or
                ``Z``.  The all-identity string ``"I" * n`` must not appear.  Entries with
                probability zero are silently dropped.
            num_qubits: The arity of the channel.  Required only for an empty ``probabilities``
                on a nontrivial number of qubits — otherwise defaults to the length of the Pauli
                strings when ``probabilities`` is non-empty, or ``0`` when empty.  If both are
                supplied and inconsistent, a ``ValueError`` is raised.

        Raises:
            ValueError: If the input contains an invalid Pauli string, contains the identity
                string, any probability is not in [0, 1], the sum of probabilities is not in
                [0, 1], or ``num_qubits`` disagrees with the length of the Pauli strings.
        """
        if num_qubits is not None and num_qubits < 0:
            raise ValueError(f"num_qubits={num_qubits} must be >= 0")
        if not probabilities:
            self._num_qubits = num_qubits if num_qubits is not None else 0
            self._probabilities: Mapping[str, float] = types.MappingProxyType({})
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
        # Use math.fsum for a precise sum, and allow a small tolerance so a mathematically-
        # normalized input isn't rejected due to per-value rounding (e.g.,
        # ``[p_i / sum(p_i) for p_i in ...]`` can accumulate to ~1 + O(n * eps)).
        total = math.fsum(probabilities.values())
        if not _is_approx_in_unit_interval(total):
            raise ValueError(f"Sum of Pauli channel probabilities {total} is not in [0, 1]")
        # Drop zero-probability entries and canonicalize insertion order (lex over Pauli strings)
        # so `__eq__`-equal channels produce identical noise chains at emission time.  Store
        # behind a MappingProxyType so the object is effectively immutable (needed for hashing).
        nonzero = {
            string: probabilities[string]
            for string in sorted(probabilities)
            if probabilities[string] > 0
        }
        self._num_qubits = derived_num_qubits
        self._probabilities = types.MappingProxyType(nonzero)

    @property
    def num_qubits(self) -> int:
        """Number of qubits the channel acts on."""
        return self._num_qubits

    @property
    def probabilities(self) -> Mapping[str, float]:
        """Read-only view of the non-identity Pauli-string → probability mapping."""
        return self._probabilities

    def __bool__(self) -> bool:
        """Is this channel nontrivial?  (Any zero-prob entries are already dropped in __init__.)"""
        return bool(self._probabilities)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliChannel):
            return NotImplemented
        return self._num_qubits == other._num_qubits and self._probabilities == other._probabilities

    def __hash__(self) -> int:
        # Canonical order is guaranteed by __init__, so tuple(items()) is deterministic.
        return hash((self._num_qubits, tuple(self._probabilities.items())))

    def __repr__(self) -> str:
        if not self._probabilities and self._num_qubits:
            return f"PauliChannel({{}}, num_qubits={self._num_qubits})"
        return f"PauliChannel({dict(self._probabilities)!r})"

    def __getstate__(self) -> tuple[int, dict[str, float]]:
        """Support pickling.  ``types.MappingProxyType`` is not itself picklable."""
        return self._num_qubits, dict(self._probabilities)

    def __setstate__(self, state: tuple[int, dict[str, float]]) -> None:
        num_qubits, probs = state
        self._num_qubits = num_qubits
        self._probabilities = types.MappingProxyType(probs)

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
            immune_qubits: Positions in ``[0, num_qubits)`` to constrain to identity.  Repeats
                are ignored.

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
        if not immune_qubits:
            return PauliChannel(dict(self._probabilities), num_qubits=self._num_qubits)
        surviving = {
            string: prob
            for string, prob in self._probabilities.items()
            if all(string[i] == "I" for i in immune_qubits)
        }
        return PauliChannel(surviving, num_qubits=self._num_qubits)


class NoiseRule:
    """Describes how to add noise to an operation.

    This class encapsulates the noise channels and measurement error probabilities that should be
    applied to a particular type of quantum operation.
    """

    def __init__(
        self,
        *,
        after: PauliChannel | stim.Circuit | Mapping[str, float | Iterable[float]] | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
    ):
        """Initializes a noise rule with specified error channels.

        Args:
            after: Noise applied after each matching operation.  The noise is broadcast across
                all of the instruction's qubit targets — a design choice that assumes each
                instruction addresses every qubit at most once (which is what
                ``_split_moments_with_ticks`` enforces during preprocessing).  Three forms are
                accepted:
                - ``PauliChannel``: a joint Pauli channel of some arity ``k``.  Emitted natively via
                    ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` when ``k ≤ 2`` (stim broadcasts), or
                    as a ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chain per ``k``-qubit
                    block for ``k ≥ 3``.
                - ``stim.Circuit``: an escape hatch — a raw fragment of noise instructions emitted
                    verbatim after each ``k``-qubit block, with qubit indices remapped
                    from ``[0, k)`` in the fragment to the corresponding block's targets.  The
                    fragment's arity is inferred from ``circuit.num_qubits``.  Every instruction
                    must be a noise instruction (``op_type(name) == NOISE``); repeat blocks are
                    rejected.  Prefer ``PauliChannel`` or the ``Mapping`` sugar when they suffice
                    — the ``stim.Circuit`` form skips native broadcasting and requires the user to
                    spell out targets explicitly.  Caveat: measurement-producing noise
                    (``HERALDED_ERASE``, ``HERALDED_PAULI_CHANNEL_1``) emitted via this form will
                    change the number of measurement records the surrounding circuit produces, and
                    can shift indices used by ``DETECTOR`` / ``rec[-k]`` — the caller is responsible
                    for making sure those indices remain consistent.
                - ``Mapping[str, float | Iterable[float]]``: syntactic sugar.  A single-entry
                    mapping over a pure-Pauli broadcast noise (e.g., ``X_ERROR``, ``DEPOLARIZE2``)
                    is converted to the equivalent ``PauliChannel``; everything else (such as
                    heralded noise) is converted to a ``stim.Circuit`` fragment.
                    ``CORRELATED_ERROR`` and ``ELSE_CORRELATED_ERROR`` are not accepted here — pass
                    a ``stim.Circuit`` instead.
            readout_error: The probability that a measurement result is reported incorrectly.  Only
                allowed for operations that produce measurement results.  ``None`` (the default)
                means the field is unset (validation ignores it); ``0.0`` is an explicit no-op flip
                probability that is still validated against the gate type.
            reset_error: The probability that a qubit is reset to the wrong state.  Only allowed for
                operations that reset qubits.  Same ``None`` vs. ``0.0`` distinction as
                ``readout_error``.

        Raises:
            ValueError: If any noise channel name is not recognized, any net probability of an
                error is not in [0, 1], if a broadcast ``after`` mixes 1q and 2q entries, or if
                the ``stim.Circuit`` form contains a non-noise instruction.
        """
        self.readout_error = readout_error
        if readout_error is not None and not (0 <= readout_error <= 1):
            raise ValueError(f"{readout_error=} is not between 0 and 1")

        self.reset_error = reset_error
        if reset_error is not None and not (0 <= reset_error <= 1):
            raise ValueError(f"{reset_error=} is not between 0 and 1")

        self.after: PauliChannel | stim.Circuit | None
        if after is None:
            self.after = None
        elif isinstance(after, PauliChannel):
            self.after = after
        elif isinstance(after, stim.Circuit):
            _validate_after_circuit(after)
            self.after = after
        else:
            # Mapping form is syntactic sugar: normalize into either a PauliChannel (native
            # broadcast) or a stim.Circuit (raw fragment) so the downstream code paths deal only
            # with those two forms.  Full validation happens before we drop all-zero entries so a
            # malformed spelling is rejected loudly even when the probabilities themselves are zero.
            normalized_mapping: dict[str, tuple[float, ...]] = {}
            for op, prob_or_probs in after.items():
                if isinstance(prob_or_probs, Iterable) and not isinstance(
                    prob_or_probs, (str, bytes)
                ):
                    normalized_mapping[op] = tuple(prob_or_probs)
                else:
                    normalized_mapping[op] = (prob_or_probs,)  # type: ignore[assignment]
            has_1q = False
            has_2q = False
            for op, probs in normalized_mapping.items():
                if op_type(op) != NOISE:
                    raise ValueError(f"Invalid or unrecognized noise channel {op!r} in {after=}")
                if op in CORRELATED_ERROR_NAMES:
                    raise ValueError(
                        f"{op} cannot be specified in a broadcast `after` mapping; pass an "
                        "explicit stim.Circuit or a PauliChannel instead"
                    )
                if not _is_approx_in_unit_interval(math.fsum(probs)):
                    raise ValueError(
                        f"The net probability of an error is not between 0 and 1 in {after=}"
                    )
                if op in BROADCAST_1Q_NOISE:
                    has_1q = True
                if op in BROADCAST_2Q_NOISE:
                    has_2q = True
            if has_1q and has_2q:
                raise ValueError(
                    f"broadcast `after` mixes 1-qubit and 2-qubit noise entries in {after=}; "
                    "combine them into a single 2-qubit PauliChannel, or pass an explicit "
                    "stim.Circuit fragment"
                )
            filtered_mapping = {op: probs for op, probs in normalized_mapping.items() if any(probs)}
            if not filtered_mapping:
                self.after = None
            else:
                self.after = _mapping_after_to_channel_or_circuit(filtered_mapping)

        if (readout_error is not None or reset_error is not None) and self.after is not None:
            raise ValueError(
                "NoiseRule cannot combine `after`-noise with readout_error / reset_error.  If you"
                " need this capability, please open an issue at"
                " https://github.com/qLDPCOrg/qLDPC/issues"
            )

    def __bool__(self) -> bool:
        """Is this noise rule nontrivial?"""
        return bool(self.after) or bool(self.readout_error) or bool(self.reset_error)

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
            assert op.name in JUST_MEASURE_OPS or op.name in MEASURE_AND_RESET_OPS
            if not args:
                args = [self.readout_error]
            else:
                assert len(args) == 1
                # combine bit-flip probabilities
                args = [1 - (1 - self.readout_error) * (1 - args[0])]

        noisy_op = stim.CircuitInstruction(op.name, targets, args, tag=op.tag)
        noise_after = stim.Circuit()

        if self.reset_error:
            assert op.name in JUST_RESET_OPS or op.name in MEASURE_AND_RESET_OPS
            # Reset gate `RB` (or `MRB`) prepares/resets in Pauli basis B ∈ {X, Y, Z}; a "wrong"
            # reset is modeled by an error that anticommutes with B.  Z-basis (canonical `R`/`MR`)
            # and Y-basis (`RY`/`MRY`) resets get X_ERROR; X-basis (`RX`/`MRX`) gets Z_ERROR.
            error_name = "Z_ERROR" if op.name.endswith("X") else "X_ERROR"
            error_op = stim.CircuitInstruction(error_name, qubit_targets, [self.reset_error])
            noise_after.append(error_op)

        self.emit_after(noise_after, qubit_targets, context=f"operation {op.name!r}")
        return noisy_op, noise_after

    def emit_after(
        self, circuit: stim.Circuit, qubit_targets: list[int], *, context: str = "operation"
    ) -> None:
        """Append this rule's ``after`` noise in-place to the provided circuit.

        If the operation packages multiple independent gates (e.g., ``H 0 1 2`` — three ``H``
        gates in one instruction), the noise is applied per gate.  For broadcast forms
        (``PauliChannel`` of arity 1 or 2), this is achieved by a single emission on all targets,
        since stim's ``PAULI_CHANNEL_1``/``PAULI_CHANNEL_2`` broadcast natively.  For higher-arity
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
        k = self.after.num_qubits
        if n_targets % k != 0:
            raise ValueError(
                f"This NoiseRule expects a multiple of {k} qubits but {context} has "
                f"{n_targets} qubit targets"
            )

        if isinstance(self.after, PauliChannel):
            # Emit once per k-qubit chunk.
            for i in range(0, n_targets, k):
                _append_pauli_channel(circuit, self.after, qubit_targets[i : i + k])
        else:
            # stim.Circuit escape hatch: apply the fragment verbatim to each k-qubit block, with
            # qubit indices remapped from [0, k) to the block's targets.
            assert isinstance(self.after, stim.Circuit)
            for i in range(0, n_targets, k):
                circuit += with_remapped_qubits(self.after, qubit_targets[i : i + k])


class NoiseModel:
    """A model that defines how to add noise to quantum circuits.

    This class provides a framework for adding various types of noise to quantum circuits, including
    gate errors, readout errors, reset errors, and idling errors.  Classically controlled operations
    are assumed to NOT occur, so the corresponding qubits pick up idling errors, if applicable.
    """

    def __init__(
        self,
        clifford_1q_error: float | PauliChannel | Mapping[str, float] | NoiseRule | None = None,
        clifford_2q_error: float | PauliChannel | Mapping[str, float] | NoiseRule | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
        *,
        clifford_nq_error: (
            Mapping[int, float | PauliChannel | Mapping[str, float] | NoiseRule] | None
        ) = None,
        idle_error: NoiseRule | float | None = None,
        additional_error_waiting_for_m_or_r: NoiseRule | float | None = None,
        rules: Mapping[str, NoiseRule] | None = None,
        rule_func: Callable[[stim.CircuitInstruction], NoiseRule | None] | None = None,
    ):
        """Initializes a noise model with specified parameters.

        Args:
            clifford_1q_error: Default noise applied after each one-qubit unitary Clifford gate.
                A float ``p`` is a uniform 1-qubit depolarizing channel of total error probability
                ``p``.  Also accepts a 1-qubit ``PauliChannel``, a raw ``Mapping[str, float]`` of
                Pauli-string probabilities (auto-wrapped as ``PauliChannel``), or a full
                ``NoiseRule``.
            clifford_2q_error: Default noise applied after each two-qubit unitary Clifford gate.
                A float ``p`` is a uniform 2-qubit depolarizing channel of total error probability
                ``p``.  Also accepts a 2-qubit ``PauliChannel``, a raw ``Mapping[str, float]`` of
                Pauli-string probabilities (auto-wrapped as ``PauliChannel``), or a full
                ``NoiseRule``.
            readout_error: Default probability of flipping measurement results.
            reset_error: Default probability of resetting qubits to the wrong state.
            clifford_nq_error: Optional mapping from a qubit count ``k`` to the noise applied
                after each ``k``-qubit unitary Clifford gate.  Values may be one of
                    - a float ``p`` (interpreted as a uniform ``k``-qubit depolarizing channel of
                        total error probability ``p``),
                    - a ``PauliChannel``,
                    - a raw ``Mapping[str, float]`` (auto-wrapped as ``PauliChannel``), or
                    - a ``NoiseRule``.
                Specifying both ``clifford_nq_error[1]`` and ``clifford_1q_error`` raises an
                ambiguity error; likewise with ``clifford_nq_error[2]`` and ``clifford_2q_error``.
            idle_error: Noise rule or depolarization probability applied to each idling qubit in any
                given moment.  If a NoiseRule is provided, its `after` channels are appended to the
                idle qubits (its readout_error/reset_error fields are ignored).
            additional_error_waiting_for_m_or_r: Additional noise rule or depolarization probability
                applied to qubits that are waiting while other qubits undergo measurement or reset
                operations.  Same NoiseRule semantics as `idle_error`.
            rules: Dictionary mapping specific gate names to their noise rules.  Overrides the
                arity-based defaults for unitary, measurement, and reset gates.
            rule_func: Optional callback function that maps a ``stim.CircuitInstruction`` to a
                ``NoiseRule``.  Takes priority over all other noise rules above.  Any gate that stim
                broadcasts across multiple independent applications (e.g. ``H 0 1 2``,
                ``CX 0 1 2 3``, or ``SPP X1*Y2 Z3*Y4*X5``) is decomposed into its individual
                gate applications before being passed to ``rule_func``, its input ``op``
                always holds exactly one application's worth of targets: one for a one-qubit gate,
                two for a two-qubit gate, and one Pauli product's targets for an SPP/MPP.  The
                callback is consulted only for genuine noisy gates (unitary Cliffords, measurements,
                and resets) that are not classically controlled; it does not affect annotations,
                pure-noise instructions, or idling errors.
        """
        self.rules = rules
        self.rule_func = rule_func
        if rules is not None:
            # Validate rules whose (arity, can_measure, can_reset) is known — fixed-arity stim
            # gates and basis-suffixed rule keys (``"MXYZ"`` for MPP X*Y*Z, ``"SXY_DAG"`` for
            # SPP_DAG X*Y).  Bare ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"`` are variable-arity and
            # deferred to emission-time checks via ``emit_after``.
            for op_name, rule in rules.items():
                shape = _known_gate_shape(op_name)
                if shape is not None:
                    arity, can_measure, can_reset = shape
                    _validate_rule_for_arity(
                        rule,
                        arity,
                        f"rules[{op_name!r}]",
                        can_measure=can_measure,
                        can_reset=can_reset,
                    )
        self.readout_error = readout_error or 0
        self.reset_error = reset_error or 0

        # `clifford_1q_error` / `clifford_2q_error` are syntactic sugar for `clifford_nq_error[1]`
        # / `clifford_nq_error[2]`; internally we normalize everything into a single
        # `clifford_nq_error` dict.  Ambiguity detection is symmetric on RAW inputs — an argument
        # counts as "user-specified" if the user passed anything other than None (including a
        # zero float, an empty NoiseRule, etc.), even if it normalizes to a no-op.
        for size, param, name in (
            (1, clifford_1q_error, "clifford_1q_error"),
            (2, clifford_2q_error, "clifford_2q_error"),
        ):
            if clifford_nq_error is not None and size in clifford_nq_error and param is not None:
                raise ValueError(
                    f"Ambiguous noise specification: both `clifford_nq_error[{size}]` and"
                    f"`{name}` are set.  Specify one or the other."
                )
        merged_nq_input: dict[int, NoiseRule | PauliChannel | Mapping[str, float] | float] = (
            dict(clifford_nq_error) if clifford_nq_error else {}
        )
        rule_1q = _as_noise_rule(clifford_1q_error, 1)
        rule_2q = _as_noise_rule(clifford_2q_error, 2)
        if rule_1q is not None:
            merged_nq_input[1] = rule_1q
        if rule_2q is not None:
            merged_nq_input[2] = rule_2q
        self.clifford_nq_error = _normalize_clifford_nq_error(merged_nq_input)

        self.idle_error = _as_noise_rule(idle_error, 1)
        self.additional_error_waiting_for_m_or_r = _as_noise_rule(
            additional_error_waiting_for_m_or_r, 1
        )
        # Idle noise is applied per-qubit, so anything with a declared arity != 1 has no natural
        # interpretation here.  Validate the declared arity BEFORE trivializing so a user typo like
        # ``PauliChannel({}, num_qubits=3)`` — trivial but wrongly-shaped — is surfaced.
        for field_name, idle_rule in (
            ("idle_error", self.idle_error),
            ("additional_error_waiting_for_m_or_r", self.additional_error_waiting_for_m_or_r),
        ):
            if idle_rule is None:
                continue
            if idle_rule.after is not None and idle_rule.after.num_qubits != 1:
                raise ValueError(
                    f"`{field_name}` does not support a multi-qubit `after` rule "
                    f"(arity={idle_rule.after.num_qubits})."
                )
        # Guards done — now trivialize.
        self.idle_error = self.idle_error or None
        self.additional_error_waiting_for_m_or_r = self.additional_error_waiting_for_m_or_r or None

    @property
    def clifford_1q_error(self) -> NoiseRule | None:
        """Convenience view: ``clifford_nq_error[1]`` if set, else None."""
        return self.clifford_nq_error.get(1)

    @property
    def clifford_2q_error(self) -> NoiseRule | None:
        """Convenience view: ``clifford_nq_error[2]`` if set, else None."""
        return self.clifford_nq_error.get(2)

    def __bool__(self) -> bool:
        """Is this noise model nontrivial?"""
        return (
            bool(self.rules)
            or self.rule_func is not None
            or bool(self.clifford_nq_error)
            or bool(self.readout_error)
            or bool(self.reset_error)
            or bool(self.idle_error)
            or bool(self.additional_error_waiting_for_m_or_r)
        )

    def get_noise_rule(self, op: stim.CircuitInstruction) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific operation.

        Noise rules are consulted in the following order of precedence:
        1. ``rule_func`` (stim.CircuitInstruction -> NoiseRule factory).
        2. ``rules`` (name-based NoiseRules).
        3. ``clifford_nq_error`` (arity-based NoiseRules for unitary Cliffords).
        4. ``readout_error`` and/or ``reset_error`` (per-gate defaults for measurement/reset ops).

        Note: MPP / SPP / SPP_DAG instructions passed to this method must contain exactly one
        Pauli product (e.g. ``MPP X0*Y1*Z2``, not ``MPP X0*Y1 Z2*X3``).  Multi-product
        instructions are decomposed upstream by ``_split_targets_pp`` before this method is
        invoked; hand-calling with an unsplit multi-product op raises ``ValueError`` from
        ``_get_gate_aliases``.

        Args:
            op: The circuit instruction to find a noise rule for.

        Returns:
            The NoiseRule to apply for the given operation, or None for no noise.
        """
        if op_type(op.name) == ANNOTATION or _involves_classical_bits(op):
            return None

        if self.rule_func is not None and op_type(op.name) in GATE_OP_TYPES:
            rule = self.rule_func(op)
            if rule is not None:
                _validate_custom_rule(rule, op)
                return rule

        if self.rules is not None:
            for name in _get_gate_aliases(op):
                rule = self.rules.get(name)
                if rule is not None:
                    return rule

        this_op_type = op_type(op.name)
        if this_op_type in (CLIFFORD_1Q, CLIFFORD_2Q, CLIFFORD_PP):
            if this_op_type == CLIFFORD_1Q:
                num_qubits = 1
            elif this_op_type == CLIFFORD_2Q:
                num_qubits = 2
            else:
                num_qubits = sum(1 for target in op.targets_copy() if not target.is_combiner)
            if num_qubits in self.clifford_nq_error:
                return self.clifford_nq_error[num_qubits]

        if self.readout_error and op.name in JUST_MEASURE_OPS:
            return NoiseRule(readout_error=self.readout_error)
        if self.reset_error and op.name in JUST_RESET_OPS:
            return NoiseRule(reset_error=self.reset_error)
        if (self.readout_error or self.reset_error) and op.name in MEASURE_AND_RESET_OPS:
            return NoiseRule(readout_error=self.readout_error, reset_error=self.reset_error)

        return None

    def noisy_circuit(
        self,
        circuit: stim_or_tsim_Circuit,
        *,
        system_qubits: Iterable[int] | None = None,
        immune_qubits: Iterable[int] = (),
        immune_op_tag: str = DEFAULT_IMMUNE_OP_TAG,
        immunize_gates: bool = True,
        insert_ticks: bool = True,
    ) -> stim_or_tsim_Circuit:
        f"""Construct a noisy version of the given circuit.

        This method first uses TICKs to split the input circuit into moments of operations that can
        be applied in parallel, thereby preventing qubit reuse conflicts.  Noise is then applied to
        each operation according to the rules of this NoiseModel.

        Args:
            circuit: The circuit to apply noise to.
            system_qubits: All qubits that are used by the circuit or are otherwise allowed to
                accumulate idling errors.  Defaults to set(range(circuit.num_qubits)).
            immune_qubits: Qubits that are declared to be immune to noise.  Defaults to none.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.  Default: "{DEFAULT_IMMUNE_OP_TAG}".
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated
                as noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
            insert_ticks: If True, automatically inserts TICK operations to prevent qubit reuse
                conflicts.  If False, assumes that this preprocessing is not necessary.

        Returns:
            The input circuit with added noise.
        """
        if tsim is not None and isinstance(circuit, tsim.Circuit):
            return tsim.Circuit.from_stim_program(
                self.noisy_circuit(
                    circuit.stim_circuit,
                    system_qubits=system_qubits,
                    immune_qubits=immune_qubits,
                    immune_op_tag=immune_op_tag,
                    immunize_gates=immunize_gates,
                    insert_ticks=insert_ticks,
                )
            )

        system_qubits = frozenset(
            range(circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits)

        if insert_ticks:
            # split moments with TICKs to prevent qubit reuse conflicts.  The preprocessing
            # operates purely on gate-level qubit reuse and ignores ``immune_qubits`` (it uses a
            # sentinel to force per-target splitting), so it composes cleanly with immunity.
            circuit = _split_moments_with_ticks(circuit, immune_op_tag)

        noisy_circuit = stim.Circuit()

        first_moment = True
        for moment_or_repeat_block in _iter_moments_and_repeat_blocks(
            circuit, immune_qubits, immune_op_tag, force_split=self.rule_func is not None
        ):
            if first_moment:
                first_moment = False
            elif not isinstance(noisy_circuit[-1], stim.CircuitRepeatBlock):
                noisy_circuit.append("TICK")

            if isinstance(moment_or_repeat_block, stim.CircuitRepeatBlock):
                if immune_op_tag in moment_or_repeat_block.tag:
                    noisy_circuit.append(moment_or_repeat_block)
                else:
                    noisy_body = self.noisy_circuit(
                        moment_or_repeat_block.body_copy(),
                        system_qubits=system_qubits,
                        immune_qubits=immune_qubits,
                        immune_op_tag=immune_op_tag,
                        immunize_gates=immunize_gates,
                        insert_ticks=insert_ticks,
                    )
                    if insert_ticks:
                        noisy_body.append("TICK")
                    noisy_circuit.append(
                        stim.CircuitRepeatBlock(
                            repeat_count=moment_or_repeat_block.repeat_count,
                            body=noisy_body,
                            tag=moment_or_repeat_block.tag,
                        )
                    )
            else:
                self._inplace_append_noisy_moment(
                    circuit=noisy_circuit,
                    moment=moment_or_repeat_block,
                    system_qubits=system_qubits,
                    immune_qubits=immune_qubits,
                    immune_op_tag=immune_op_tag,
                    immunize_gates=immunize_gates,
                )

        return noisy_circuit

    def _inplace_append_noisy_moment(
        self,
        *,
        circuit: stim.Circuit,
        moment: Collection[stim.CircuitInstruction],
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str,
        immunize_gates: bool,
    ) -> None:
        """Apps noise to a moment and appends it to a circuit (in-place).

        This method processes all operations in a moment, applies their respective noise rules, and
        adds the resulting noisy operations to the output circuit.

        Args:
            circuit: The circuit to append the noisy operations to.
            moment: Collection of operations happening during the moment in question.
            system_qubits: Set of all qubits in the system that may experience idle errors.
            immune_qubits: Qubits that are declared to be immune to noise.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated
                as noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.

        Raises:
            ValueError: If qubits are operated on multiple times within the same moment without a
                TICK in between (violating the "each qubit at most once per moment" invariant that
                noise application relies on).
        """
        collapsed_qubits, operation_qubits = _categorize_moment_qubits(moment)

        noise_after_moment = stim.Circuit()
        for op in moment:
            if immune_op_tag in op.tag or (rule := self.get_noise_rule(op)) is None:
                circuit.append(op)
                continue
            effective_rule = _rule_with_immunity(
                rule, op, immune_qubits, immunize_gates=immunize_gates
            )
            if effective_rule is None:
                circuit.append(op)
                continue
            noisy_op, after = effective_rule.noisy_operation(op)
            circuit.append(noisy_op)
            noise_after_moment += after
        circuit += noise_after_moment

        moment_was_noisy = any(immune_op_tag not in op.tag for op in moment)
        if moment_was_noisy and (self.idle_error or self.additional_error_waiting_for_m_or_r):
            self._inplace_append_idle_errors(
                circuit=circuit,
                collapsed_qubits=collapsed_qubits,
                operation_qubits=operation_qubits,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
            )

    def _inplace_append_idle_errors(
        self,
        *,
        circuit: stim.Circuit,
        collapsed_qubits: list[int],
        operation_qubits: list[int],
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
    ) -> None:
        """Append idling errors from the given moment to the given circuit.

        This method identifies which qubits are idle during a moment and applies depolarization
        noise to them according to the noise model parameters.  The qubit categorization is
        precomputed by ``_categorize_moment_qubits`` and passed in.

        Args:
            circuit: The circuit to append idle error operations to.
            collapsed_qubits: Qubits acted on by measurement / reset ops in this moment.
            operation_qubits: Qubits acted on by non-collapsing (unitary) ops in this moment.
            system_qubits: Set of all qubits in the system that can experience idle errors.
            immune_qubits: Qubits that are declared to be immune to noise.
        """
        non_collapse_qubits = system_qubits - immune_qubits - set(collapsed_qubits)
        idle_qubits = sorted(non_collapse_qubits - set(operation_qubits))

        if self.idle_error and idle_qubits:
            self.idle_error.emit_after(circuit, idle_qubits, context="idle qubits")
        if self.additional_error_waiting_for_m_or_r and collapsed_qubits and non_collapse_qubits:
            self.additional_error_waiting_for_m_or_r.emit_after(
                circuit, sorted(non_collapse_qubits), context="qubits waiting for m/r"
            )


####################################################################################################
# custom noise models


class DepolarizingNoiseModel(NoiseModel):
    """Creates a near-standard circuit depolarizing noise model.

    All operations has the same error parameter p:
    - One-qubit Clifford gates get one-qubit depolarization.
    - Two-qubit Clifford gates get two-qubit depolarization.
    - Measurements have their outcomes probabilistically flipped.
    - Reset gates probabalistically reset qubits to the wrong (orthogonal) state.
    - If applicable, every idling qubit in a given moment gets depolarized.

    Multi-qubit Cliffords can also be depolarized by increasing the max_gate_size.
    """

    def __init__(
        self, p: float, *, include_idling_error: bool = False, max_gate_size: int = 2
    ) -> None:
        """Instantiate a depolarizing noise model."""
        self.p = p
        self.include_idling_error = include_idling_error
        super().__init__(
            clifford_nq_error={size: p for size in range(1, max_gate_size + 1)},
            readout_error=p,
            reset_error=p,
            idle_error=p if include_idling_error else False,
        )


class SI1000NoiseModel(NoiseModel):
    """A superconducting-inspired noise model defined in "A Fault-Tolerant Honeycomb Memory"

    This noise model is defined by a two-qubit gate infidelity that determines all error rates.

    See https://arxiv.org/abs/2108.10457.
    """

    def __init__(self, p: float) -> None:
        """Instantiate a superconducting-inspired noise model."""
        self.p = p
        super().__init__(
            clifford_1q_error=p / 10,
            clifford_2q_error=p,
            readout_error=p * 5,
            reset_error=p * 2,
            idle_error=p / 10,
            additional_error_waiting_for_m_or_r=2 * p,
        )


####################################################################################################
# helper methods, roughly in order of use above in the file (sub-helpers grouped with their caller)


# Floating-point tolerance used for probability comparisons throughout this module.  Small enough
# to catch real bugs, large enough to absorb the ~O(n * eps) drift that accumulates when summing
# or renormalizing many probabilities.
_ABSOLUTE_ERROR_TOLERANCE = 1e-9


def _is_approx_in_unit_interval(value: float, *, atol: float = _ABSOLUTE_ERROR_TOLERANCE) -> bool:
    """Return True if ``value`` is in [0, 1], up to floating-point tolerance ``tol``."""
    return -atol <= value <= 1 + atol


def _is_uniform_depolarizing(args: list[float], *, atol: float = _ABSOLUTE_ERROR_TOLERANCE) -> bool:
    """Return True if all ``args`` are equal (up to ``atol``) and non-zero.

    Detects a ``PauliChannel.depolarizing(k, p)`` shape at emit time so it can be written as
    ``DEPOLARIZE{k}(p)`` instead of ``PAULI_CHANNEL_{k}(p/n, ..., p/n)``.
    """
    return bool(args) and args[0] > 0 and all(abs(x - args[0]) <= atol for x in args[1:])


def _pauli_channel_1_shortcut(args: list[float]) -> tuple[str, list[float]]:
    """Return the compact stim ``(name, args)`` for a 1-qubit Pauli-channel probability vector.

    Emits ``DEPOLARIZE1(p)`` when the three probabilities are equal and non-zero, or
    ``{X,Y,Z}_ERROR(p)`` when exactly one is non-zero, else falls back to ``PAULI_CHANNEL_1``.
    """
    if _is_uniform_depolarizing(args):
        return "DEPOLARIZE1", [sum(args)]
    nonzero_positions = [i for i, x in enumerate(args) if x > 0]
    if len(nonzero_positions) == 1:
        (i,) = nonzero_positions
        return f"{_PAULI_CHANNEL_1_ORDER[i]}_ERROR", [args[i]]
    return "PAULI_CHANNEL_1", args


def _pauli_channel_2_shortcut(args: list[float]) -> tuple[str, list[float]]:
    """Return the compact stim ``(name, args)`` for a 2-qubit Pauli-channel probability vector.

    Emits ``DEPOLARIZE2(p)`` when the fifteen probabilities are equal and non-zero, else falls
    back to ``PAULI_CHANNEL_2``.
    """
    if _is_uniform_depolarizing(args):
        return "DEPOLARIZE2", [sum(args)]
    return "PAULI_CHANNEL_2", args


def _get_gate_aliases(op: stim.CircuitInstruction) -> tuple[str, ...]:
    """Return the names by which ``op`` can be matched in ``rules`` and ``rule_func``.

    For a single-product MPP / SPP / SPP_DAG op, the basis-suffixed name (e.g. ``"MXYZ"`` for
    ``MPP X*Y*Z``, ``"SXY_DAG"`` for ``SPP_DAG X*Y``) is yielded first — it's more specific than
    the raw ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"`` — followed by the corresponding stim aliases.
    For every other gate, this is just stim's alias list, which begins with the canonical name
    (so ``M`` / ``MZ`` both yield ``("M", "MZ")``, ``CX`` / ``CNOT`` / ``ZCX`` all yield
    ``("CNOT", "CX", "ZCX")``, etc.).
    """
    aliases = _stim_aliases(op.name)
    if op.name not in ("MPP", "SPP", "SPP_DAG"):
        return aliases
    prefix, suffix = ("S", "_DAG") if op.name == "SPP_DAG" else (op.name[0], "")
    targets = op.targets_copy()
    if not all(target.is_combiner for target in targets[1::2]):
        raise ValueError(
            f"{op.name} must be split into a single Pauli product before alias enumeration"
        )
    basis = ""
    for target in targets[::2]:
        if target.is_x_target:
            basis += "X"
        elif target.is_y_target:
            basis += "Y"
        else:
            assert target.is_z_target
            basis += "Z"
    return (prefix + basis + suffix, *aliases)


@functools.cache
def _stim_aliases(op_name: str) -> tuple[str, ...]:
    """Cached stim aliases of a gate name."""
    return tuple(stim.gate_data(op_name).aliases)


_PAULI_CHANNEL_1_ORDER = ("X", "Y", "Z")
_PAULI_CHANNEL_2_ORDER = tuple(a + b for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I"))


def _append_pauli_channel(
    circuit: stim.Circuit,
    channel: PauliChannel,
    qubit_targets: list[int],
    *,
    tag: str = "",
) -> None:
    """Append noise instructions to ``circuit`` that implement ``channel`` on ``qubit_targets``.

    The emitted form depends on the number of *active* positions — positions where at least one
    Pauli string acts nontrivially.  Channels with 1 or 2 active positions emit a native
    ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` on the corresponding qubit(s), even if the channel's
    formal arity is larger; channels with 3+ active positions emit a chain of one
    ``CORRELATED_ERROR`` followed by one ``ELSE_CORRELATED_ERROR`` per remaining non-zero Pauli
    string, with conditional probabilities renormalized so each Pauli string's unconditional
    firing probability equals its value in ``channel``.  An empty channel (or one whose surviving
    strings are all identity, which cannot occur but is defensively handled) emits nothing.
    """
    active_positions = sorted(
        {i for string in channel.probabilities for i, pauli in enumerate(string) if pauli != "I"}
    )
    if not active_positions:  # pragma: no cover -- callers gate on `bool(channel)`
        return
    active_qubits = [qubit_targets[i] for i in active_positions]

    if len(active_positions) == 1:
        probs_1q = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        for string, prob in channel.probabilities.items():
            probs_1q[string[active_positions[0]]] = prob
        args = [probs_1q[p] for p in _PAULI_CHANNEL_1_ORDER]
        name, name_args = _pauli_channel_1_shortcut(args)
        circuit.append(name, active_qubits, name_args, tag=tag)
        return

    if len(active_positions) == 2:
        pos0, pos1 = active_positions
        probs_2q = {pair: 0.0 for pair in _PAULI_CHANNEL_2_ORDER}
        for string, prob in channel.probabilities.items():
            probs_2q[string[pos0] + string[pos1]] = prob
        args = [probs_2q[p] for p in _PAULI_CHANNEL_2_ORDER]
        name, name_args = _pauli_channel_2_shortcut(args)
        circuit.append(name, active_qubits, name_args, tag=tag)
        return

    remaining = 1.0
    first = True
    for string, prob in channel.probabilities.items():
        pauli_targets = _pauli_string_to_targets(string, qubit_targets)
        if first:
            circuit.append("CORRELATED_ERROR", pauli_targets, [prob], tag=tag)
            first = False
        else:
            # `remaining` is `1 - sum_of_prior` in exact arithmetic and is guaranteed >= prob
            # because the constructor rejects total > 1.  If floating-point subtraction leaves it
            # a hair smaller than prob (or zero), emit an ELSE_CORRELATED_ERROR(1) that absorbs
            # the rest of the probability mass and stop — any subsequent ELSE_CE would never fire.
            if remaining <= prob:
                circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [1.0], tag=tag)
                break
            circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [prob / remaining], tag=tag)
        remaining -= prob


def _pauli_string_to_targets(string: str, qubit_targets: list[int]) -> list[stim.GateTarget]:
    """Convert a Pauli string over the given qubits to a list of Pauli-typed stim targets."""
    return [
        stim.target_pauli(qubit, pauli)
        for pauli, qubit in zip(string, qubit_targets, strict=True)
        if pauli != "I"
    ]


def _validate_after_circuit(after_circuit: stim.Circuit) -> None:
    """Validate that ``after_circuit`` contains only noise instructions.

    Args:
        after_circuit: The fragment to validate.

    Raises:
        ValueError: If ``after_circuit`` contains a non-noise instruction or a repeat block.
    """
    for op in after_circuit:
        if not isinstance(op, stim.CircuitInstruction):
            raise ValueError(
                f"after (stim.Circuit form) may contain only noise instructions, not "
                f"{type(op).__name__}"
            )
        if op_type(op.name) != NOISE:
            raise ValueError(
                f"after (stim.Circuit form) contains non-noise instruction {op.name!r}; only "
                "NOISE instructions are permitted"
            )


# Names of the two Pauli-channel primitives, since their args are already the canonical
# per-Pauli-string probability layout used by ``PauliChannel``.
_PAULI_CHANNEL_MAPPING_NAMES: dict[str, tuple[str, ...]] = {
    "PAULI_CHANNEL_1": _PAULI_CHANNEL_1_ORDER,
    "PAULI_CHANNEL_2": _PAULI_CHANNEL_2_ORDER,
}


def _mapping_after_to_channel_or_circuit(
    mapping: dict[str, tuple[float, ...]],
) -> PauliChannel | stim.Circuit:
    """Convert a Mapping-form ``after`` to a ``PauliChannel`` if possible, else a stim.Circuit.

    A single-entry mapping whose channel name is ``PAULI_CHANNEL_1`` / ``PAULI_CHANNEL_2`` — the
    two stim primitives that already spell out per-Pauli-string probabilities — is exactly a
    ``PauliChannel``.  Every other Mapping-form input (``X_ERROR``, ``DEPOLARIZE1``,
    ``HERALDED_ERASE``, multi-entry mappings, …) round-trips into a stim.Circuit fragment on
    qubits ``[0, k)``, where ``k`` is 2 if any 2-qubit-broadcast entry is present and 1
    otherwise.  Multi-entry Mappings intentionally do NOT merge into a joint ``PauliChannel``:
    users writing ``{"X_ERROR": p, "Z_ERROR": q}`` expect two independent noise ops (which
    compose stochastically, with ``P(Y) = p·q``) — merging would silently give ``P(Y) = 0``.
    """
    if len(mapping) == 1:
        [(name, args)] = mapping.items()
        if name in _PAULI_CHANNEL_MAPPING_NAMES:
            strings = _PAULI_CHANNEL_MAPPING_NAMES[name]
            if len(args) == len(strings):
                return PauliChannel(dict(zip(strings, args)))
    arity = 2 if any(op in BROADCAST_2Q_NOISE for op in mapping) else 1
    fragment = stim.Circuit()
    for name, args in mapping.items():
        fragment.append(name, list(range(arity)), list(args))
    return fragment


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
    - **Mixed** (only possible for atomic k>=2 ops the splitter cannot decompose further —
      2-qubit Clifford pairs, ``MXX``/``MYY``/``MZZ``, and MPP/SPP Pauli products):
        - ``immunize_gates=True`` (default): dead simple — drop everything (``after``,
          ``readout_error``, ``reset_error``).  The gate touches an immune qubit, so no noise
          is applied.
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


_BASIS_SUFFIXED_RULE_KEY = re.compile(r"([MS])([XYZ]+)(_DAG)?")


def _known_gate_shape(op_name: str) -> tuple[int, bool, bool] | None:
    """Return ``(arity, can_measure, can_reset)`` for a rule key with statically-known shape.

    Applies to fixed-arity stim gates (e.g. ``"H"``, ``"CX"``, ``"M"``, ``"R"``, ``"MR"``) and to
    basis-suffixed rule keys (``"MXYZ"`` for ``MPP X*Y*Z``, ``"SXY_DAG"`` for ``SPP_DAG X*Y``).
    Returns ``None`` for variable-arity names (bare ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"``) or for
    names not recognized as either kind.
    """
    this_op_type = op_type(op_name)
    if this_op_type in (CLIFFORD_1Q, JUST_MEASURE_1Q, JUST_RESET_1Q, MEASURE_RESET_1Q):
        gate_data = stim.gate_data(op_name)
        return 1, gate_data.produces_measurements, gate_data.is_reset
    if this_op_type in (CLIFFORD_2Q, JUST_MEASURE_2Q):
        gate_data = stim.gate_data(op_name)
        return 2, gate_data.produces_measurements, gate_data.is_reset
    # Basis-suffixed rule keys parse as `[MS](XYZ)+(_DAG)?` — look up the underlying PP gate
    # (``MPP`` / ``SPP`` / ``SPP_DAG``) for the measurement/reset flags.
    match = _BASIS_SUFFIXED_RULE_KEY.fullmatch(op_name)
    if match is None:
        return None
    prefix, basis, dag = match.groups()
    gate_data = stim.gate_data(f"{prefix}PP{dag or ''}")
    return len(basis), gate_data.produces_measurements, gate_data.is_reset


def _validate_rule_for_arity(
    rule: NoiseRule,
    num_qubits: int,
    context: str,
    *,
    can_measure: bool = False,
    can_reset: bool = False,
) -> None:
    """Reject a NoiseRule whose channels are ambiguous / incompatible on ``num_qubits`` qubits.

    - The rule's ``after`` arity must equal ``num_qubits`` unless ``after`` is ``None`` (the
      trivial default).  An explicitly-declared arity — even on an empty PauliChannel like
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


def _as_noise_rule(
    error: NoiseRule | PauliChannel | Mapping[str, float] | float | None, default_arity: int
) -> NoiseRule | None:
    """Normalize a noise-error argument to a NoiseRule (or None if the input is ``None``).

    A float scalar is interpreted as a uniform depolarizing channel of the given ``default_arity``.
    A ``PauliChannel`` — or a raw ``Mapping[str, float]`` of Pauli-string probabilities, which is
    auto-wrapped as a ``PauliChannel`` — is used as the rule's ``after``.  Storing the depolarizing
    shape as a ``PauliChannel`` (rather than a Mapping-form ``DEPOLARIZE{n}`` fragment) preserves
    the Pauli-channel structure so downstream ``PauliChannel.conditioned_on`` can project it under
    partial immunity.

    Does NOT trivialize an empty NoiseRule to ``None`` — callers must run their guard checks on
    the declared shape (e.g. arity) BEFORE trivializing, so that user typos like
    ``PauliChannel({}, num_qubits=3)`` in a 1-qubit-only slot are surfaced.  Trivialize via
    ``rule or None`` after the guard checks.
    """
    if isinstance(error, NoiseRule):
        return error
    if isinstance(error, PauliChannel):
        return NoiseRule(after=error)
    if isinstance(error, Mapping):
        return NoiseRule(after=PauliChannel(error))
    if error is None:
        return None
    return NoiseRule(after=PauliChannel.depolarizing(default_arity, error))


def _normalize_clifford_nq_error(
    error: Mapping[int, NoiseRule | PauliChannel | Mapping[str, float] | float] | None,
) -> dict[int, NoiseRule]:
    """Normalize the ``clifford_nq_error`` argument to a ``dict[int, NoiseRule]``.

    - Floats become uniform ``k``-qubit depolarizing noise: ``DEPOLARIZE1`` for ``k == 1``,
      ``DEPOLARIZE2`` for ``k == 2``, and ``PauliChannel.depolarizing`` otherwise.
    - ``PauliChannel`` values (or raw Pauli-string ``Mapping[str, float]`` dicts, auto-wrapped)
      become a ``NoiseRule(after=<channel>)``.
    - ``NoiseRule`` values are used directly.
    - Falsy entries (0.0, empty NoiseRule, empty channel) are dropped.
    - Entries are rejected if their joint Pauli channel's ``num_qubits`` disagrees with the key,
      if their ``after`` broadcast channels are incompatible with ``k`` qubits, or if they set
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
        elif isinstance(entry, PauliChannel):
            rule = NoiseRule(after=entry)
        elif isinstance(entry, Mapping):
            rule = NoiseRule(after=PauliChannel(entry))
        elif isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(
                f"clifford_nq_error[{weight}] has unsupported type {type(entry).__name__}; "
                "expected NoiseRule, PauliChannel, Mapping[str, float], int, or float"
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


def _involves_classical_bits(op: stim.CircuitInstruction) -> bool:
    """Determines if an operation involves classical bits.

    Args:
        op: The circuit instruction to check.

    Returns:
        True if the operation involves classical control bits.  False otherwise.
    """
    return any(
        target.is_measurement_record_target or target.is_sweep_bit_target
        for target in op.targets_copy()
    )


def _categorize_moment_qubits(
    moment: Collection[stim.CircuitInstruction],
) -> tuple[list[int], list[int]]:
    """Categorize a moment's qubit targets and check for reuse.

    Iterates every non-annotation instruction and buckets its qubit targets into three lists —
    collapsed (measurement / reset), classically-controlled, and everything else ("operation").
    Raises if any qubit is used more than once across the three buckets, since noise application
    relies on the "each qubit at most once per moment" invariant that ``_split_moments_with_ticks``
    enforces when ``insert_ticks=True``.  The classically-controlled bucket contributes only to
    the reuse check.

    Args:
        moment: The moment's operations.

    Returns:
        A tuple ``(collapsed_qubits, operation_qubits)``, each in the order the moment referenced
        them.  Consumers use these for idle-error emission (see
        ``NoiseModel._inplace_append_idle_errors``).

    Raises:
        ValueError: If any qubit is operated on multiple times within the moment without a TICK
            in between.
    """
    collapsed_qubits: list[int] = []
    operation_qubits: list[int] = []
    classically_controlled_qubits: list[int] = []
    for op in moment:
        if op_type(op.name) == ANNOTATION:
            continue
        target_qubits = [
            target.qubit_value for target in op.targets_copy() if target.qubit_value is not None
        ]
        if op.name in COLLAPSING_OPS:
            qubits = collapsed_qubits
        elif _involves_classical_bits(op):
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


def _split_moments_with_ticks(circuit: stim.Circuit, immune_op_tag: str) -> stim.Circuit:
    """Insert TICKs into a circuit to split stim.CircuitInstruction that reuse qubits.

    This preprocessing ensures that errors are applied correctly to a stim.CircuitInstruction that
    reuses qubits.

    Args:
        circuit: The input circuit to preprocess.
        immune_op_tag: Don't split operations with this tag.

    Returns:
        stim.Circuit: A circuit with TICKs added to prevent instructions from reusing qubits.
    """
    result = stim.Circuit()
    used_qubits: set[int] = set()

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if immune_op_tag in op.tag:
                result.append(op)
                continue

            # Process repeat blocks recursively
            if used_qubits:
                result.append("TICK")
                used_qubits.clear()
            processed_body = _split_moments_with_ticks(op.body_copy(), immune_op_tag)
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=op.repeat_count, body=processed_body, tag=op.tag
                )
            )
            continue

        if op.name == "TICK":
            # Explicit TICK - clear used qubits
            result.append("TICK")
            used_qubits.clear()
            continue

        """
        For preprocessing, we need to force splitting of multi-target operations to detect qubit
        reuse properly.  Use a dummy immune_qubits set with -1 to force splitting of 2-qubit
        operations.
        """
        split_ops = list(_split_targets_if_needed(op, frozenset({-1}), immune_op_tag))

        for split_op in split_ops:
            # Check if this split operation would reuse any qubits
            op_qubits = set()
            if op_type(split_op.name) != ANNOTATION:
                for target in split_op.targets_copy():
                    if not target.is_combiner:
                        op_qubits.add(target.value)

            # If there's qubit reuse, insert a TICK first
            if op_qubits & used_qubits:
                result.append("TICK")
                used_qubits.clear()

            # Add the operation and update used qubits
            result.append(split_op)
            used_qubits.update(op_qubits)

    return result


def _iter_moments_and_repeat_blocks(
    circuit: stim.Circuit,
    immune_qubits: frozenset[int],
    immune_op_tag: str,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitRepeatBlock | list[stim.CircuitInstruction]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like CX rec[-1] 0 are split from quantum operations like
    CX 1 0.  SPP and MPP operations are split into one operation per Pauli product.

    Args:
        circuit: The circuit to split into moments.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, decompose every broadcast gate into its individual applications
            (one target for one-qubit gates, one pair for two-qubit gates) even when no qubit is
            immune.  Used when a ``rule_func`` is present so the callback is consulted once
            per gate application.

    Yields:
        Lists of operations corresponding to one moment in the circuit, with any problematic
        operations like MPPs split into pieces, or CircuitRepeatBlock instances for repeat blocks.

    Note:
        A moment is the time between two TICKs.
    """
    current_moment: list[stim.CircuitInstruction] = []

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if current_moment:
                yield current_moment
                current_moment = []
            yield op
        elif op.name == "TICK":
            if current_moment:
                yield current_moment
                current_moment = []
        else:
            current_moment.extend(
                _split_targets_if_needed(op, immune_qubits, immune_op_tag, force_split=force_split)
            )
    if current_moment:
        yield current_moment


def _split_targets_if_needed(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str,
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
    # Two-qubit ops (both Clifford CX/CZ/... and joint measurements MXX/MYY/MZZ) split per-pair
    # via _split_targets_clifford_2q so partial-immunity decisions happen at the pair level.
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
    immune_op_tag: str,
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
    if force_split or immune_qubits or immune_op_tag in op.tag:
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args, tag=op.tag)
    else:
        yield op


def _split_targets_clifford_2q(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit operations into individual gate pairs.

    Handles both two-qubit Cliffords (CX/CZ/...) and joint two-qubit measurements
    (MXX/MYY/MZZ), separating classical-control from quantum-only gates.

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
        or immune_op_tag in op.tag
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
    start = end = 0
    while end < len(targets):
        if end + 1 == len(targets) or not targets[end + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : end + 1], args, tag=op.tag)
            end += 1
            start = end
        else:
            end += 2
    assert end == len(targets)
