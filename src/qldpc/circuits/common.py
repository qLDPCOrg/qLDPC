# SPDX-License-Identifier: Apache-2.0

"""Circuit types, Pauli-product measurements, and the guard for qubit-only methods.

This module is the foundation of ``qldpc.circuits``: it imports no other module of
``qldpc.circuits``, so that all of them can import it.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, ParamSpec, Protocol, Self, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import _deprecations, codes, math

_ReturnType = TypeVar("_ReturnType")
_Params = ParamSpec("_Params")


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
            # relabel qubits here, rather than with qubit remapping, which builds on this module
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
