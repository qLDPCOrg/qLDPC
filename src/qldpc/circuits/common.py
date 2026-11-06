# SPDX-License-Identifier: Apache-2.0

"""Circuit types, noise-immunity tags, Pauli-product measurements, and the qubit-only guard.

This module is the foundation of ``qldpc.circuits``: it imports no other module of
``qldpc.circuits``, so that all of them can import it.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, ParamSpec, Protocol, Self, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import _deprecations, codes, math

_ReturnType = TypeVar("_ReturnType")
_Params = ParamSpec("_Params")

# Tags that, by default, make an operation or repeat block noiseless, or declare that the qubits of
# a tagged QUBIT_COORDS annotation are immune to noise.  The noise models of qldpc.circuits use
# them, and they are defined here so that circuit transformations can respect them.
DEFAULT_IMMUNE_OP_TAG = "__IMMUNE_TO_NOISE__"
DEFAULT_IMMUNE_QUBIT_TAG = "__IMMUNE_QUBIT_TO_NOISE__"


class StimCircuitProtocol(Protocol):
    """Structural type for circuit objects that behave like a ``stim.Circuit``.

    A ``stim.Circuit`` satisfies this protocol, as do stim-wrapping circuit classes such as
    ``tsim.Circuit``.  qLDPC's circuit-polymorphic functions (e.g. ``noisy_circuit``,
    ``with_remapped_qubits``) are generic over the ``StimCircuitLike`` type variable (bound to this
    protocol) so that they preserve the concrete circuit type of their input.  Those functions read
    a circuit as a sequence of ``stim`` operations (via ``range(len(circuit))`` and indexing), do
    their work on a ``stim.Circuit``, and rebuild a circuit of the original type by constructing an
    empty instance (``type(circuit)()``) and populating it via ``append`` / in-place concatenation.
    A conforming circuit must therefore be default-constructible in addition to supporting the
    methods declared here.  This lets qLDPC support such circuits structurally, without depending on
    the packages that define them.
    """

    def append(
        self,
        name: str | stim.CircuitInstruction | stim.CircuitRepeatBlock | stim.Circuit,
        targets: (
            int
            | stim.GateTarget
            | stim.PauliString
            | Iterable[int | stim.GateTarget | stim.PauliString]
        ) = (),
        arg: float | Iterable[float] | None = None,
        *,
        tag: str = "",
    ) -> None:
        """Append an operation to this circuit in place."""

    def __iadd__(self, other: stim.Circuit) -> Self:
        """Append another circuit to this circuit in place."""

    def __getitem__(self, index: int) -> stim.CircuitInstruction | stim.CircuitRepeatBlock:
        """The operation at the given index."""

    def __len__(self) -> int:
        """The number of operations in this circuit."""


# A type variable for a circuit that behaves like a stim.Circuit (see StimCircuitProtocol).  Used to
# type functions that are generic over stim.Circuit and stim-wrapping circuits while preserving the
# concrete circuit type of their input.
StimCircuitLike = TypeVar("StimCircuitLike", bound=StimCircuitProtocol)


def get_pauli_product_measurements(
    pauli_strings: Sequence[stim.PauliString] | Sequence[Sequence[int]] | npt.NDArray[np.int_],
    qubits: Sequence[int] | None = None,
) -> stim.Circuit:
    """Construct a circuit of MPP instructions that measure the given Pauli strings.

    In addition to a list of Pauli strings, this method accepts a symplectic matrix (or a sequence
    of symplectic rows) in which each row indicates the ``[X|Z]`` support of a Pauli string.  If
    "code" is a QuditCode, for example, then passing "pauli_strings=code.get_stabilizer_ops()" will
    measure the stabilizers of "code".

    Args:
        pauli_strings: Pauli strings or symplectic rows to measure.
        qubits: Optional physical-qubit labels corresponding to the columns of each Pauli string.

    Raises:
        ValueError: If a Pauli string is the identity, if qubits is provided and has repeated
            labels, or if qubits is provided and a Pauli string does not act on exactly len(qubits)
            qubits.
    """
    if isinstance(pauli_strings, np.ndarray):
        pauli_strings = [math.op_to_string(op) for op in np.atleast_2d(pauli_strings)]
    elif pauli_strings and not isinstance(pauli_strings[0], stim.PauliString):
        pauli_strings = [math.op_to_string(np.asarray(op, dtype=int)) for op in pauli_strings]
    if qubits is not None and len(set(qubits)) != len(qubits):
        raise ValueError("Qubit labels for Pauli-product measurements must be distinct")
    circuit = stim.Circuit()
    for string in pauli_strings:
        if not any(pauli in "XYZ" for pauli in str(string)):
            raise ValueError("Pauli-product measurements cannot contain an all-identity row")
        targets = stim.target_combined_paulis(string)
        if qubits is not None:
            if len(string) != len(qubits):
                raise ValueError(
                    f"A Pauli string acts on {len(string)} qubits, but {len(qubits)} qubit labels"
                    " were provided"
                )
            # if applicable, assign qubit indices as provided
            targets = [
                target
                if target.is_combiner
                else stim.target_pauli(
                    qubits[target.value], target.pauli_type, invert=target.is_inverted_result_target
                )
                for target in targets
            ]
        circuit.append("MPP", targets)
    return circuit


def _get_immune_qubits(circuit: stim.Circuit, immune_qubit_tag: str | None) -> frozenset[int]:
    """Identify qubits declared immune to noise by tagged coordinate annotations.

    A qubit is declared immune if a QUBIT_COORDS annotation for it has a tag that contains
    immune_qubit_tag.  If immune_qubit_tag is None, no qubit is declared immune.

    Raises:
        ValueError: If immune_qubit_tag is empty.
    """
    if immune_qubit_tag is None:
        return frozenset()
    if immune_qubit_tag == "":
        raise ValueError("immune_qubit_tag must be non-empty or None")
    immune_qubits: set[int] = set()
    for instruction in _get_instructions(circuit):
        if isinstance(instruction, stim.CircuitRepeatBlock):
            immune_qubits.update(_get_immune_qubits(instruction.body_copy(), immune_qubit_tag))
        elif instruction.name == "QUBIT_COORDS" and immune_qubit_tag in instruction.tag:
            immune_qubits.update(
                target.qubit_value
                for target in instruction.targets_copy()
                if target.qubit_value is not None
            )
    return frozenset(immune_qubits)


def _restrict_to_qubits(
    func: Callable[_Params, _ReturnType],
) -> Callable[_Params, _ReturnType]:
    """Restrict a circuit or tableau constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: _Params.args, **kwargs: _Params.kwargs) -> _ReturnType:
        if any(
            isinstance(arg, codes.QuditCode) and arg.field is not galois.GF2
            for arg in itertools.chain(args, kwargs.values())
        ):
            raise ValueError("Circuit methods are only supported for qubit codes")
        return func(*args, **kwargs)

    return qubit_func


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)


def _as_stim_circuit(circuit: StimCircuitLike) -> stim.Circuit:
    """Copy a circuit into a stim.Circuit."""
    stim_circuit = stim.Circuit()
    for index in range(len(circuit)):
        stim_circuit.append(circuit[index])
    return stim_circuit


def _get_instructions(
    circuit: stim.Circuit,
) -> Iterator[stim.CircuitInstruction | stim.CircuitRepeatBlock]:
    """The instructions and REPEAT blocks of a circuit, in order.

    Stim's iterator over a circuit takes tens of microseconds to finish, which adds up when noise
    models handle many small circuits, so this indexes the circuit instead.
    """
    return (circuit[index] for index in range(len(circuit)))


def _new_instruction(
    name: str, targets: Sequence[stim.GateTarget], args: Sequence[float], tag: str
) -> stim.CircuitInstruction:
    """Build a ``stim.CircuitInstruction`` by appending it to an empty circuit.

    Stim's ``CircuitInstruction`` constructor takes tens of microseconds per target, while appending
    an instruction to a circuit takes about a microsecond.  An instruction that a circuit rejects,
    such as a REPEAT instruction, comes from the constructor, which accepts it or raises its own
    error.
    """
    circuit = stim.Circuit()
    try:
        circuit.append(name, targets, args, tag=tag)
    except (TypeError, ValueError):
        return stim.CircuitInstruction(name, targets, args, tag=tag)
    return circuit[0]
