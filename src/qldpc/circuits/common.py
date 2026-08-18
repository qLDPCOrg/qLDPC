"""Miscellaneous circuit utilities.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ParamSpec, Protocol, TypeVar

import galois
import numpy as np
import numpy.typing as npt
import stim
from typing_extensions import Self

from qldpc import codes, math

CircuitOrTableau = TypeVar("CircuitOrTableau", stim.Circuit, stim.Tableau)
Params = ParamSpec("Params")


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


def restrict_to_qubits(
    func: Callable[Params, CircuitOrTableau],
) -> Callable[Params, CircuitOrTableau]:
    """Restrict a circuit or tableau constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: Params.args, **kwargs: Params.kwargs) -> stim.Circuit:
        if any(isinstance(arg, codes.QuditCode) and arg.field is not galois.GF2 for arg in args):
            raise ValueError("Circuit methods are only supported for qubit codes")
        return func(*args, **kwargs)

    return qubit_func


def with_remapped_qubits(
    circuit: StimCircuitLike,
    qubit_map: Mapping[int, int] | Sequence[int],
    *,
    inverse: bool = False,
) -> StimCircuitLike:
    """The same circuit, but with relabeled qubits.

    Qubits not in qubit_map get mapped to themselves.

    Args:
        circuit: The circuit to remap.
        qubit_map: Either a mapping (e.g., dictionary) from old to new qubit indices, or a sequence
            for which the qubit at index old_index gets mapped to new_index = qubit_map[old_index].
        inverse: If True, invert the provided qubit_map.  Default: False.

    Returns:
        The input circuit with remapped qubits.
    """
    qubit_map = (
        qubit_map
        if isinstance(qubit_map, Mapping)
        else {old_index: new_index for old_index, new_index in enumerate(qubit_map)}
    )
    if inverse:
        qubit_map = {val: key for key, val in qubit_map.items()}

    new_circuit = type(circuit)()  # same type as input circuit
    for index in range(len(circuit)):
        op = circuit[index]
        if isinstance(op, stim.CircuitRepeatBlock):
            block = stim.CircuitRepeatBlock(
                repeat_count=op.repeat_count,
                body=with_remapped_qubits(op.body_copy(), qubit_map),
                tag=op.tag,
            )
            new_circuit.append(block)

        else:
            new_targets = [remap_qubit_target(target, qubit_map) for target in op.targets_copy()]
            new_op = stim.CircuitInstruction(
                name=op.name, targets=new_targets, gate_args=op.gate_args_copy(), tag=op.tag
            )
            new_circuit.append(new_op)

    return new_circuit


def remap_qubit_target(target: stim.GateTarget, qubit_map: Mapping[int, int]) -> stim.GateTarget:
    """Remap the qubit addressed by a stim.GateTarget, if any."""
    if target.qubit_value is None:
        return target

    new_qubit_value = qubit_map.get(target.qubit_value, target.qubit_value)
    if target.is_x_target or target.is_z_target or target.is_y_target:
        return stim.target_pauli(
            new_qubit_value,
            target.pauli_type,
            invert=target.is_inverted_result_target,
        )

    if target.is_inverted_result_target:
        return stim.target_inv(new_qubit_value)

    return stim.GateTarget(new_qubit_value)


def get_pauli_product_measurements(
    pauli_strings: Sequence[stim.PauliString] | npt.NDArray[np.int_],
    qubits: Sequence[int] | None = None,
) -> stim.Circuit:
    """Construct a circuit of MPP instructions that measure the given Pauli strings.

    In addition to a list of Pauli strings, this method accepts a symplectic matrix in which each
    row indicates the ``[X|Z]`` support of a Pauli string.  If "code" is a QuditCode, for example,
    then passing "pauli_strings=code.get_stabilizer_ops()" will measure the stabilizers of "code".
    """
    if isinstance(pauli_strings, np.ndarray):
        pauli_strings = [math.op_to_string(op) for op in np.atleast_2d(pauli_strings)]
    circuit = stim.Circuit()
    for string in pauli_strings:
        circuit.append("MPP", stim.target_combined_paulis(string))
    return circuit if qubits is None else with_remapped_qubits(circuit, qubits)


def get_unaddressed_measurements(circuit: stim.Circuit) -> list[int]:
    """Identify measurements, by index, that are not addressed by any detectors in the circuit."""
    measurements: list[int] = []
    addressed_measurements = set()
    for instruction in circuit.flattened():
        new_measurements = range(
            len(measurements),
            len(measurements) + instruction.num_measurements,
        )
        measurements.extend(new_measurements)
        if instruction.name == "DETECTOR":
            addressed_measurements |= {
                measurements[target.value] for target in instruction.targets_copy()
            }
    return sorted(set(measurements) - addressed_measurements)
