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

    # Multi-qubit MPP gates receive ordinary readout_error.  To assign per-basis rules, key on the
    # standardized name that measurement gates are dispatched under: "M<paulis>", e.g. "MXYZ" for
    # `MPP X*Y*Z`, "MXX" for `MPP X*X`, etc.
    noise_model = NoiseModel(
        readout_error=1e-3,                                # default readout flip probability
        rules={"MXYZ": NoiseRule(readout_error=5e-3)},     # override for MPP X*Y*Z specifically
    )


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
import itertools
import math
import types
import warnings
from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, TypeVar

import stim

try:
    import tsim

    stim_or_tsim_Circuit = TypeVar("stim_or_tsim_Circuit", stim.Circuit, tsim.Circuit)
except ImportError:  # pragma: no cover
    if not TYPE_CHECKING:
        tsim = None
        stim_or_tsim_Circuit = TypeVar("stim_or_tsim_Circuit", bound=stim.Circuit)


####################################################################################################
# global constants


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

OP_TYPES = {
    # one-qubit Cliffords
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "C_NXYZ": CLIFFORD_1Q,
    "C_NZYX": CLIFFORD_1Q,
    "C_XNYZ": CLIFFORD_1Q,
    "C_XYNZ": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZNYX": CLIFFORD_1Q,
    "C_ZYNX": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_NXY": CLIFFORD_1Q,
    "H_NXZ": CLIFFORD_1Q,
    "H_NYZ": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    # two-qubit Cliffords
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CXSWAP": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "CZSWAP": CLIFFORD_2Q,
    "II": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "SWAPCX": CLIFFORD_2Q,
    "SWAPCZ": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    # noise channels
    "CORRELATED_ERROR": NOISE,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "E": NOISE,
    "ELSE_CORRELATED_ERROR": NOISE,
    "HERALDED_ERASE": NOISE,
    "HERALDED_PAULI_CHANNEL_1": NOISE,
    "II_ERROR": NOISE,
    "I_ERROR": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    # collapsing gates
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "MXX": JUST_MEASURE_2Q,
    "MYY": JUST_MEASURE_2Q,
    "MZZ": JUST_MEASURE_2Q,
    "MPP": JUST_MEASURE_PP,
    # Pauli product gates
    "SPP": CLIFFORD_PP,
    "SPP_DAG": CLIFFORD_PP,
    # "REPEAT": ...,  # UNSUPPORTED
    # annotations
    "DETECTOR": ANNOTATION,
    "MPAD": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
}
JUST_MEASURE_OPS = {
    op
    for op, op_type in OP_TYPES.items()
    if op_type == JUST_MEASURE_1Q or op_type == JUST_MEASURE_2Q or op_type == JUST_MEASURE_PP
}
JUST_RESET_OPS = {op for op, op_type in OP_TYPES.items() if op_type == JUST_RESET_1Q}
MEASURE_AND_RESET_OPS = {op for op, op_type in OP_TYPES.items() if op_type == MEASURE_RESET_1Q}
COLLAPSING_OPS = JUST_MEASURE_OPS | JUST_RESET_OPS | MEASURE_AND_RESET_OPS

CORRELATED_ERROR_NAMES = frozenset({"CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"})

# Noise instructions that stim broadcasts independently per qubit (any number of targets).
BROADCAST_1Q_NOISE = frozenset(
    {
        "DEPOLARIZE1",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)
# Noise instructions that stim broadcasts per (fixed-size) pair; require an even number of targets.
BROADCAST_2Q_NOISE = frozenset({"DEPOLARIZE2", "II_ERROR", "PAULI_CHANNEL_2"})

DEFAULT_IMMUNE_OP_TAG = "__IMMUNE_TO_NOISE__"


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
        if not _approx_in_unit_interval(total):
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
        after: Mapping[str, float | Iterable[float]] = {},
        after_pauli_channel: PauliChannel | Mapping[str, float] | None = None,
        readout_error: float = 0,
        reset_error: float = 0,
    ):
        """Initializes a noise rule with specified error channels.

        Args:
            after: A dictionary mapping noise channel names to their probability arguments.  For
                example, {"DEPOLARIZE2": 0.01, "PAULI_CHANNEL_1": [0.02, 0, 0]} will add two-qubit
                depolarization with parameter 0.01, followed by 2% bit-flip noise.  These noise
                channels occur after all other operations in the moment and are applied to the same
                targets as the relevant operation.  CORRELATED_ERROR (alias E) and
                ELSE_CORRELATED_ERROR are not accepted here; use `after_pauli_channel` instead.
            after_pauli_channel: An n-qubit Pauli channel applied jointly to the operation's
                qubits.  Emitted as a chain of CORRELATED_ERROR / ELSE_CORRELATED_ERROR
                instructions whose conditional firing probabilities are renormalized so each
                Pauli string's unconditional firing probability equals the value in the channel.
                The channel's num_qubits must match the number of non-combiner targets of the
                operation it is applied to.  Accepts a `PauliChannel` or a raw dict (auto-wrapped).
            readout_error: The probability that a measurement result is reported incorrectly.  Only
                allowed for operations that produce measurement results.
            reset_error: The probability that a qubit is reset to the wrong state.  Only allowed for
                operations that reset qubits.

        Raises:
            ValueError: If any noise channel name is not recognized or if any net probability of an
                error is not between 0 and 1 (inclusive).
        """
        self.readout_error = readout_error
        if not (0 <= readout_error <= 1):
            raise ValueError(f"{readout_error=} is not between 0 and 1")

        self.reset_error = reset_error
        if not (0 <= reset_error <= 1):
            raise ValueError(f"{reset_error=} is not between 0 and 1")

        self.after = {
            op: tuple(prob_or_probs) if isinstance(prob_or_probs, Iterable) else (prob_or_probs,)
            for op, prob_or_probs in after.items()
        }
        for op, probs in self.after.items():
            if OP_TYPES.get(op) != NOISE:
                raise ValueError(f"Invalid or unrecognized noise channel {op!r} in {after=}")
            if op in CORRELATED_ERROR_NAMES:
                raise ValueError(
                    f"{op} cannot be specified in `after`; use `after_pauli_channel` to specify a "
                    "multi-qubit Pauli channel"
                )
            if not _approx_in_unit_interval(math.fsum(probs)):
                raise ValueError(
                    f"The net probability of an error is not between 0 and 1 in {after=}"
                )

        if after_pauli_channel is not None and not isinstance(after_pauli_channel, PauliChannel):
            after_pauli_channel = PauliChannel(after_pauli_channel)
        # Empty and all-zero channels are treated as absent.
        self.after_pauli_channel = after_pauli_channel or None

    def __bool__(self) -> bool:
        """Is this noise rule nontrivial?"""
        return (
            bool(self.after)
            or bool(self.readout_error)
            or bool(self.reset_error)
            or self.after_pauli_channel is not None
        )

    def noisy_operation(
        self,
        op: stim.CircuitInstruction,
        *,
        immune_qubits: Iterable[int] = (),
        immunize_gates: bool = True,
    ) -> tuple[stim.CircuitInstruction, stim.Circuit]:
        """Apply this noise rule to the given operation.

        Args:
            op: The operation to add noise to.
            immune_qubits: Qubits that are declared to be immune to noise.
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated
                as noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.

        Returns:
            stim.CircuitInstruction: The given operation possibly modified to account for noise.
            stim.Circuit: Noise operations that should follow the given operation.
        """
        immune_qubits = frozenset(immune_qubits)
        targets = op.targets_copy()
        args = op.gate_args_copy()
        qubit_targets = [target.value for target in targets if not target.is_combiner]

        # Drop readout_error only when *every* measured qubit is immune: readout_error is a single
        # per-instruction bit-flip probability, and a measurement whose classical result depends on
        # any non-immune qubit still has a real readout to potentially flip.
        all_immune = bool(qubit_targets) and all(q in immune_qubits for q in qubit_targets)

        if self.readout_error and not all_immune:
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
            error_name = ("X" if _get_standardized_name(op)[-1] != "X" else "Z") + "_ERROR"
            error_op = stim.CircuitInstruction(error_name, qubit_targets, [self.reset_error])
            noise_after.append(error_op)

        self.emit_after(
            noise_after,
            qubit_targets,
            immune_qubits=immune_qubits,
            immunize_gates=immunize_gates,
            context=f"operation {op.name!r}",
        )

        return noisy_op, noise_after

    def emit_after(
        self,
        circuit: stim.Circuit,
        qubit_targets: list[int],
        *,
        immune_qubits: Iterable[int] = (),
        immunize_gates: bool = True,
        context: str = "operation",
    ) -> None:
        """Append this rule's ``after`` and ``after_pauli_channel`` noise in-place.

        This method is the canonical emission path for both ``after`` and ``after_pauli_channel``
        so consumers cannot forget to handle one of them.  It does NOT apply ``readout_error``
        (which modifies a measurement op's own probability argument, not a follow-up instruction)
        or ``reset_error`` (which needs the op name to pick between X_ERROR and Z_ERROR).

        Args:
            circuit: The circuit to append the noise instructions to.
            qubit_targets: The qubits the noise applies to (in the operation's target order).
            immune_qubits: Qubits that are declared to be immune to noise.
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated
                as noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
            context: A short description of the operation, used only in error messages.

        Raises:
            ValueError: If ``after_pauli_channel`` is set and its ``num_qubits`` does not match
                ``len(qubit_targets)``, or if an ``after`` entry demands a specific target arity
                that ``len(qubit_targets)`` does not satisfy (e.g., a 2-qubit-broadcast channel
                like ``DEPOLARIZE2`` applied to an odd number of qubits).
        """
        immune_qubits = frozenset(immune_qubits)
        num_qubits = len(qubit_targets)
        for op_name, args in self.after.items():
            if op_name in BROADCAST_2Q_NOISE and num_qubits % 2 != 0:
                raise ValueError(
                    f"{context}: `after` channel {op_name!r} requires an even number of qubit "
                    f"targets but {num_qubits} were provided"
                )
            circuit.append(op_name, qubit_targets, args)
        if self.after_pauli_channel is not None:
            if self.after_pauli_channel.num_qubits != num_qubits:
                raise ValueError(
                    f"PauliChannel with num_qubits={self.after_pauli_channel.num_qubits} cannot "
                    f"be applied to {context} with {num_qubits} qubit targets"
                )
            immune_positions = [i for i, q in enumerate(qubit_targets) if q in immune_qubits]
            if not immune_positions:
                _append_pauli_channel(circuit, self.after_pauli_channel, qubit_targets)
            elif not immunize_gates:
                sub_channel = self.after_pauli_channel.conditioned_on(immune_positions)
                if sub_channel:
                    _append_pauli_channel(circuit, sub_channel, qubit_targets)


class NoiseModel:
    """A model that defines how to add noise to quantum circuits.

    This class provides a framework for adding various types of noise to quantum circuits, including
    gate errors, readout errors, reset errors, and idling errors.  Classically controlled operations
    are assumed to NOT occur, so the corresponding qubits pick up idling errors, if applicable.
    """

    def __init__(
        self,
        clifford_1q_error: NoiseRule | float | None = None,
        clifford_2q_error: NoiseRule | float | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
        *,
        clifford_nq_error: (
            Mapping[int, NoiseRule | PauliChannel | Mapping[str, float] | float] | None
        ) = None,
        idle_error: NoiseRule | float | None = None,
        additional_error_waiting_for_m_or_r: NoiseRule | float | None = None,
        rules: Mapping[str, NoiseRule] | None = None,
    ):
        """Initializes a noise model with specified parameters.

        Args:
            clifford_1q_error: Default noise rule or depolarization probability for one-qubit
                unitary Clifford gates.
            clifford_2q_error: Default noise rule or depolarization probability for two-qubit
                unitary Clifford gates.
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
            rules: Dictionary mapping specific gate names to their noise rules.  Overrides all other
                rules for unitary, measurement, and reset gates.
        """
        self.rules = rules
        if rules is not None:
            # Validate rules whose gate arity (size) is fixed and known.  Rule keys for MPP are
            # standardized on the Pauli-product basis (e.g., "MXYZ" for MPP X*Y*Z), so their arity
            # varies with the product weight and can only be checked at emission time by
            # `emit_after`, which re-checks against the actual qubit-target count.
            for op_name, rule in rules.items():
                arity = _known_gate_arity(op_name)
                if arity is not None:
                    _validate_rule_for_arity(
                        rule,
                        arity,
                        f"rules[{op_name!r}]",
                        can_measure=_op_can_measure(op_name),
                        can_reset=_op_can_reset(op_name),
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
        rule_1q = _as_noise_rule(clifford_1q_error, "DEPOLARIZE1")
        rule_2q = _as_noise_rule(clifford_2q_error, "DEPOLARIZE2")
        if rule_1q is not None:
            merged_nq_input[1] = rule_1q
        if rule_2q is not None:
            merged_nq_input[2] = rule_2q
        self.clifford_nq_error = _normalize_clifford_nq_error(merged_nq_input)

        self.idle_error = _as_noise_rule(idle_error, "DEPOLARIZE1")
        self.additional_error_waiting_for_m_or_r = _as_noise_rule(
            additional_error_waiting_for_m_or_r, "DEPOLARIZE1"
        )
        # Idle-error emission applies channels per-qubit, so a joint multi-qubit PauliChannel has
        # no natural interpretation here.  Reject rather than silently drop.
        for field_name, idle_rule in (
            ("idle_error", self.idle_error),
            ("additional_error_waiting_for_m_or_r", self.additional_error_waiting_for_m_or_r),
        ):
            if idle_rule is not None and idle_rule.after_pauli_channel is not None:
                raise ValueError(
                    f"`{field_name}` does not support `after_pauli_channel`: idle noise is "
                    "applied per qubit, but a PauliChannel is a joint multi-qubit channel."
                )

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
            or bool(self.clifford_nq_error)
            or bool(self.readout_error)
            or bool(self.reset_error)
            or bool(self.idle_error)
            or bool(self.additional_error_waiting_for_m_or_r)
        )

    def get_noise_rule(self, op: stim.CircuitInstruction) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific operation.

        Args:
            op: The circuit instruction to find a noise rule for.

        Returns:
            The NoiseRule to apply for the given operation, or None for no noise.
        """
        if OP_TYPES[op.name] == ANNOTATION or _involves_classical_bits(op):
            return None

        if self.rules is not None:
            rule = self.rules.get(_get_standardized_name(op)) or self.rules.get(
                op.name
            )  # allows for an MPP rule, but first checks for rules such as MXY
            if rule is not None:
                return rule

        op_type = OP_TYPES[op.name]
        if op_type in (CLIFFORD_1Q, CLIFFORD_2Q, CLIFFORD_PP):
            if op_type == CLIFFORD_1Q:
                num_qubits = 1
            elif op_type == CLIFFORD_2Q:
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
            circuit, immune_qubits, immune_op_tag
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
                        immunize_gates=immunize_gates,
                    )
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
        """
        noise_after_moment = stim.Circuit()
        for op in moment:
            if immune_op_tag in op.tag or (rule := self.get_noise_rule(op)) is None:
                circuit.append(op)
            else:
                noisy_op, after = rule.noisy_operation(
                    op, immune_qubits=immune_qubits, immunize_gates=immunize_gates
                )
                circuit.append(noisy_op)
                noise_after_moment += after

        circuit += _immunize_noise(noise_after_moment, immune_qubits, immunize_gates=immunize_gates)

        moment_was_noisy = any(immune_op_tag not in op.tag for op in moment)
        if moment_was_noisy and (self.idle_error or self.additional_error_waiting_for_m_or_r):
            self._inplace_append_idle_errors(
                circuit=circuit,
                moment=moment,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
            )

    def _inplace_append_idle_errors(
        self,
        *,
        circuit: stim.Circuit,
        moment: Collection[stim.CircuitInstruction],
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
    ) -> None:
        """Append idling errors from the given moment to the given circuit.

        This method identifies which qubits are idle during a moment and applies depolarization
        noise to them according to the noise model parameters.

        Args:
            circuit: The circuit to append idle error operations to.
            moment: The collection of operations happening in the final moment of the circuit.
            system_qubits: Set of all qubits in the system that can experience idle errors.
            immune_qubits: Qubits that are declared to be immune to noise.

        Raises:
            ValueError: If qubits are operated on multiple times within the same moment without a
                TICK in between.
        """
        collapsed_qubits: list[int] = []
        operation_qubits: list[int] = []
        classically_controlled_qubits: list[int] = []
        for op in moment:
            if OP_TYPES[op.name] == ANNOTATION:
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

        # Safety check for operation collisions.
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
_APPROX_TOL = 1e-9


def _approx_in_unit_interval(value: float, *, tol: float = _APPROX_TOL) -> bool:
    """Return True if ``value`` is in [0, 1], up to floating-point tolerance ``tol``."""
    return -tol <= value <= 1 + tol


def _get_standardized_name(op: stim.CircuitInstruction) -> str:
    """Stardardized name of a circuit instruction.

    The primary function of this method is to disambiguate the basis of measurement and reset gates.

    Args:
        op:_name The name of the circuit instruction that we need to standardize.

    Returns:
        str: The standardized name.
    """
    op_name = op.name
    if op_name == "M" or op_name == "R" or op_name == "MR":
        return op_name + "Z"

    if op_name == "MPP":
        name = "M"
        for target in op.targets_copy()[::2]:
            if target.is_x_target:
                name += "X"
            elif target.is_y_target:
                name += "Y"
            else:
                assert target.is_z_target
                name += "Z"
        return name

    return op_name


_PAULI_CHANNEL_1_ORDER = ("X", "Y", "Z")
_PAULI_CHANNEL_2_ORDER = tuple(a + b for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I"))


def _append_pauli_channel(
    circuit: stim.Circuit,
    channel: PauliChannel,
    qubit_targets: list[int],
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
        circuit.append(
            "PAULI_CHANNEL_1", active_qubits, [probs_1q[p] for p in _PAULI_CHANNEL_1_ORDER]
        )
        return

    if len(active_positions) == 2:
        pos0, pos1 = active_positions
        probs_2q = {pair: 0.0 for pair in _PAULI_CHANNEL_2_ORDER}
        for string, prob in channel.probabilities.items():
            probs_2q[string[pos0] + string[pos1]] = prob
        circuit.append(
            "PAULI_CHANNEL_2", active_qubits, [probs_2q[p] for p in _PAULI_CHANNEL_2_ORDER]
        )
        return

    remaining = 1.0
    first = True
    for string, prob in channel.probabilities.items():
        pauli_targets = _pauli_string_to_targets(string, qubit_targets)
        if first:
            circuit.append("CORRELATED_ERROR", pauli_targets, [prob])
            first = False
        else:
            # `remaining` is `1 - sum_of_prior` in exact arithmetic and is guaranteed >= prob
            # because the constructor rejects total > 1.  If floating-point subtraction leaves it
            # a hair smaller than prob (or zero), emit an ELSE_CORRELATED_ERROR(1) that absorbs
            # the rest of the probability mass and stop — any subsequent ELSE_CE would never fire.
            if remaining <= prob:
                circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [1.0])
                break
            circuit.append("ELSE_CORRELATED_ERROR", pauli_targets, [prob / remaining])
        remaining -= prob


def _pauli_string_to_targets(string: str, qubit_targets: list[int]) -> list[stim.GateTarget]:
    """Convert a Pauli string over the given qubits to a list of Pauli-typed stim targets."""
    return [
        stim.target_pauli(qubit, pauli)
        for pauli, qubit in zip(string, qubit_targets, strict=True)
        if pauli != "I"
    ]


# PAULI_CHANNEL_2 arg order:
# IX(0), IY(1), IZ(2), XI(3), XX(4), XY(5), XZ(6),
# YI(7), YX(8), YY(9), YZ(10), ZI(11), ZX(12), ZY(13), ZZ(14)
# Marginal indices when the second qubit is immune (surviving: first qubit; non-cross: XI, YI, ZI):
_PC2_SECOND_IMMUNE_INDICES = (3, 7, 11)
# Marginal indices when the first qubit is immune (surviving: second qubit; non-cross: IX, IY, IZ):
_PC2_FIRST_IMMUNE_INDICES = (0, 1, 2)


def _immunize_noise(
    noise: stim.Circuit,
    immune_qubits: frozenset[int],
    *,
    immunize_gates: bool = True,
) -> stim.Circuit:
    """Return a copy of a flat noise circuit with instructions targeting immune qubits removed.

    An instruction is removed if any of its qubit targets belongs to ``immune_qubits``.  Special
    cases:
    - Broadcast 1-qubit noise (``DEPOLARIZE1``, ``X_ERROR``, etc.): immune targets are dropped,
      non-immune targets kept.
    - Broadcast 2-qubit noise (``DEPOLARIZE2``, ``PAULI_CHANNEL_2``): with ``immunize_gates=True``,
      partially-immune pairs are dropped; with ``immunize_gates=False``, they are conditioned on
      the immune qubit acting as identity and emitted as the resulting 1-qubit sub-channel on the
      surviving qubit (see ``_immunize_2q_noise``).

    ``CORRELATED_ERROR`` / ``ELSE_CORRELATED_ERROR`` chains are always emitted by
    ``NoiseRule.emit_after`` on qubit sets that are either fully immune (dropped upstream) or
    fully non-immune (via ``PauliChannel.conditioned_on``), so they never mention immune qubits
    at this point and pass through the identity branch.

    Args:
        noise: A flat noise circuit (no repeat blocks) to filter.
        immune_qubits: Qubits that are declared to be immune to noise.
        immunize_gates: If True (the default), a gate that touches an immune qubit is treated
            as noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
            the immune qubits, keeping only terms that act as ``I`` on every immune qubit.

    Returns:
        stim.Circuit: A filtered copy of the input circuit.
    """
    if not immune_qubits:
        return noise
    result = stim.Circuit()
    for noise_op in noise:
        assert isinstance(noise_op, stim.CircuitInstruction)
        if all(t.value not in immune_qubits for t in noise_op.targets_copy() if not t.is_combiner):
            result.append(noise_op)
        elif stim.gate_data(noise_op.name).is_two_qubit_gate:
            result += _immunize_2q_noise(noise_op, immune_qubits, immunize_gates=immunize_gates)
        else:
            # 1-qubit noise with multiple targets: keep only non-immune targets
            surviving = [t for t in noise_op.targets_copy() if t.value not in immune_qubits]
            if surviving:
                result.append(
                    stim.CircuitInstruction(noise_op.name, surviving, noise_op.gate_args_copy())
                )
    return result


def _immunize_2q_noise(
    noise_op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    *,
    immunize_gates: bool,
) -> stim.Circuit:
    """Filter or condition a 2-qubit noise instruction on the identity-on-immune subspace.

    Processes each pair of targets independently.  Pairs with no immune qubits are kept as-is.
    Pairs where both qubits are immune are dropped.  For partially-immune pairs, if
    ``immunize_gates`` is True the pair is dropped; otherwise the surviving qubit receives the
    1-qubit sub-channel obtained by keeping only terms that act as ``I`` on the immune position
    (probabilities unchanged).  Only DEPOLARIZE2 and PAULI_CHANNEL_2 support conditioning; other
    2-qubit channels emit a warning and are dropped for partially-immune pairs.

    Args:
        noise_op: A 2-qubit noise instruction.
        immune_qubits: Qubits that are declared to be immune to noise.
        immunize_gates: If True, drop partially-immune pairs.  Otherwise, emit the surviving-qubit
            sub-channel for each partially-immune pair.

    Returns:
        stim.Circuit: The filtered/conditioned circuit for this instruction.
    """
    result = stim.Circuit()
    name = noise_op.name
    args = noise_op.gate_args_copy()
    targets = noise_op.targets_copy()

    for i in range(0, len(targets), 2):
        q1, q2 = targets[i], targets[i + 1]
        q1_immune = q1.value in immune_qubits
        q2_immune = q2.value in immune_qubits

        if not q1_immune and not q2_immune:
            result.append(stim.CircuitInstruction(name, [q1, q2], args))
        elif q1_immune and q2_immune:
            pass  # both immune: skip
        elif immunize_gates:
            pass  # partially immune, gate immunized: drop the pair
        elif name == "DEPOLARIZE2":
            p = args[0]
            surviving = q2.value if q1_immune else q1.value
            result.append(stim.CircuitInstruction("DEPOLARIZE1", [surviving], [p / 5]))
        elif name == "PAULI_CHANNEL_2":
            if q2_immune:
                sub_channel_probs = [args[idx] for idx in _PC2_SECOND_IMMUNE_INDICES]
                surviving = q1.value
            else:
                sub_channel_probs = [args[idx] for idx in _PC2_FIRST_IMMUNE_INDICES]
                surviving = q2.value
            result.append(
                stim.CircuitInstruction("PAULI_CHANNEL_1", [surviving], sub_channel_probs)
            )
        else:  # pragma: no cover
            warnings.warn(
                f"Cannot immunize {name} over immune qubits; noise is dropped.",
                stacklevel=2,
            )

    return result


def _known_gate_arity(op_name: str) -> int | None:
    """Return the fixed number of qubits a gate acts on, or None if variable / unknown."""
    op_type = OP_TYPES.get(op_name)
    if op_type in (CLIFFORD_1Q, JUST_MEASURE_1Q, JUST_RESET_1Q, MEASURE_RESET_1Q):
        return 1
    if op_type in (CLIFFORD_2Q, JUST_MEASURE_2Q):
        return 2
    return None


def _validate_rule_for_arity(
    rule: NoiseRule,
    num_qubits: int,
    context: str,
    *,
    can_measure: bool = False,
    can_reset: bool = False,
) -> None:
    """Reject a NoiseRule whose channels are ambiguous / incompatible on ``num_qubits`` qubits.

    - Any ``after_pauli_channel`` must have matching ``num_qubits``.
    - Any 2-qubit-broadcast entry in ``after`` (DEPOLARIZE2, II_ERROR, PAULI_CHANNEL_2) requires
      ``num_qubits == 2``; on other arities the pairing of targets is ambiguous.
    - 1-qubit-broadcast entries (DEPOLARIZE1, X_ERROR, ...) are always compatible.
    - ``readout_error`` is only meaningful if ``can_measure`` is True.
    - ``reset_error`` is only meaningful if ``can_reset`` is True.
    """
    if rule.after_pauli_channel is not None and rule.after_pauli_channel.num_qubits != num_qubits:
        raise ValueError(
            f"{context} has a PauliChannel with "
            f"num_qubits={rule.after_pauli_channel.num_qubits}; expected {num_qubits}"
        )
    for op_name in rule.after:
        if op_name in BROADCAST_2Q_NOISE and num_qubits % 2 != 0:
            raise ValueError(
                f"{context}: `after` channel {op_name!r} requires an even number of qubit targets "
                f"but the rule is being applied to {num_qubits} qubits"
            )
    if rule.readout_error and not can_measure:
        raise ValueError(f"{context}: `readout_error` is only valid on measurement gates")
    if rule.reset_error and not can_reset:
        raise ValueError(f"{context}: `reset_error` is only valid on reset gates")


def _op_can_measure(op_name: str) -> bool:
    """Is ``op_name`` a gate whose noise rule may set ``readout_error``?"""
    return OP_TYPES.get(op_name) in (
        JUST_MEASURE_1Q,
        JUST_MEASURE_2Q,
        JUST_MEASURE_PP,
        MEASURE_RESET_1Q,
    )


def _op_can_reset(op_name: str) -> bool:
    """Is ``op_name`` a gate whose noise rule may set ``reset_error``?"""
    return OP_TYPES.get(op_name) in (JUST_RESET_1Q, MEASURE_RESET_1Q)


def _as_noise_rule(error: NoiseRule | float | None, default_channel: str) -> NoiseRule | None:
    """Normalize a noise-error argument to a NoiseRule (or None if falsy).

    A falsy scalar (0, False, None) or empty NoiseRule collapses to None.  A truthy scalar is
    wrapped as `NoiseRule(after={default_channel: error})`.
    """
    if isinstance(error, NoiseRule):
        return error or None
    if not error:
        return None
    return NoiseRule(after={default_channel: error})


def _normalize_clifford_nq_error(
    error: Mapping[int, NoiseRule | PauliChannel | Mapping[str, float] | float] | None,
) -> dict[int, NoiseRule]:
    """Normalize the ``clifford_nq_error`` argument to a ``dict[int, NoiseRule]``.

    - Floats become uniform ``k``-qubit depolarizing noise: ``DEPOLARIZE1`` for ``k == 1``,
      ``DEPOLARIZE2`` for ``k == 2``, and ``PauliChannel.depolarizing`` otherwise.
    - ``PauliChannel`` values (or raw ``Mapping[str, float]`` dicts, auto-wrapped) become a
      ``NoiseRule(after_pauli_channel=...)``.
    - ``NoiseRule`` values are used directly.
    - Falsy entries (0.0, empty NoiseRule, empty channel) are dropped.
    - Entries are rejected if their ``after_pauli_channel``'s ``num_qubits`` disagrees with the
      key, or if their ``after`` noise channels are incompatible with ``k`` qubits, or if they
      set ``readout_error`` / ``reset_error`` (Pauli-product Cliffords are neither).
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
            rule = NoiseRule(after_pauli_channel=entry)
        elif isinstance(entry, Mapping):
            rule = NoiseRule(after_pauli_channel=PauliChannel(entry))
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
            if weight == 1:
                rule = NoiseRule(after={"DEPOLARIZE1": entry})
            elif weight == 2:
                rule = NoiseRule(after={"DEPOLARIZE2": entry})
            else:
                rule = NoiseRule(after_pauli_channel=PauliChannel.depolarizing(weight, entry))
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
            if OP_TYPES[split_op.name] != ANNOTATION:
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
    circuit: stim.Circuit, immune_qubits: frozenset[int], immune_op_tag: str
) -> Iterator[stim.CircuitRepeatBlock | list[stim.CircuitInstruction]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like CX rec[-1] 0 are split from quantum operations like
    CX 1 0.  SPP and MPP operations are split into one operation per Pauli product.

    Args:
        circuit: The circuit to split into moments.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.

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
            current_moment.extend(_split_targets_if_needed(op, immune_qubits, immune_op_tag))
    if current_moment:
        yield current_moment


def _split_targets_if_needed(
    op: stim.CircuitInstruction, immune_qubits: frozenset[int], immune_op_tag: str
) -> Iterator[stim.CircuitInstruction]:
    """Splits operations into pieces as needed.

    This function splits operations like SPP and MPP into each Pauli product, and separates
    classical control operations from quantum operations.

    Args:
        op: The circuit instruction to potentially split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.

    Yields:
        Circuit instructions, potentially split into smaller pieces.
    """
    op_type = OP_TYPES[op.name]
    if op_type == CLIFFORD_2Q:
        yield from _split_targets_clifford_2q(op, immune_qubits, immune_op_tag)
    elif op_type == CLIFFORD_PP or op_type == JUST_MEASURE_PP:
        yield from _split_targets_pp(op)
    elif op_type in [NOISE, ANNOTATION]:
        yield op
    else:
        yield from _split_targets_clifford_1q(op, immune_qubits, immune_op_tag)


def _split_targets_clifford_1q(
    op: stim.CircuitInstruction, immune_qubits: frozenset[int], immune_op_tag: str
) -> Iterator[stim.CircuitInstruction]:
    """Splits single-qubit Clifford operations when immune qubits are present.

    Args:
        op: The single-qubit Clifford operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.

    Yields:
        Circuit instructions split into individual single-target operations.
    """
    if immune_qubits or immune_op_tag in op.tag:
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args, tag=op.tag)
    else:
        yield op


def _split_targets_clifford_2q(
    op: stim.CircuitInstruction, immune_qubits: frozenset[int], immune_op_tag: str
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit Clifford operations into individual gate pairs.

    This function separates classical control system operations from quantum operations happening on
    the quantum computer.

    Args:
        op: The two-qubit Clifford operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.

    Yields:
        Circuit instructions split into individual two-qubit gate operations.
    """
    assert OP_TYPES[op.name] == CLIFFORD_2Q
    targets = op.targets_copy()
    if (
        immune_qubits
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
    assert OP_TYPES[op.name] == CLIFFORD_PP or OP_TYPES[op.name] == JUST_MEASURE_PP
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
