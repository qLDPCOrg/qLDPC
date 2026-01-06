"""Methods for constructing miscellaneous useful circuits

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
from collections.abc import Callable, Mapping, Sequence
from typing import ParamSpec, TypeVar, Union

import numpy as np
import stim

from qldpc import codes, math
from qldpc.objects import Pauli

CircuitOrTableau = TypeVar("CircuitOrTableau", bound=Union[stim.Circuit, stim.Tableau])
Params = ParamSpec("Params")


def restrict_to_qubits(
    func: Callable[Params, CircuitOrTableau],
) -> Callable[Params, CircuitOrTableau]:
    """Restrict a circuit or tableau constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: Params.args, **kwargs: Params.kwargs) -> stim.Circuit:
        if any(isinstance(arg, codes.QuditCode) and arg.field.order != 2 for arg in args):
            raise ValueError("Circuit methods are only supported for qubit codes")
        return func(*args, **kwargs)

    return qubit_func


@restrict_to_qubits
def get_encoding_tableau(code: codes.QuditCode, *, only_zero: bool = False) -> stim.Tableau:
    """Tableau to encode physical states at its input into logical states of the given code.

    If only_zero is True, this tableau maps an all-0 physical state at its input to an all-0 logical
    state at its output.  Otherwise, for all j in {0, 1, ..., code.dimension - 1}, this tableau maps
    weight-one X_j and Z_j operators at its input to the logical X and Z operators of the j-th
    logical qubit of the code.  Weight-one Z_j operators for j >= code.dimension get mapped to
    "Z-type" gauge operators and stabilizers, and their conjugate X_j get mapped to "X-type" gauge
    operators and destabilizers.
    """
    if only_zero:
        return stim.Tableau.from_stabilizers(
            [math.op_to_string(op) for op in code.get_stabilizer_ops(symplectic=True)]
            + [math.op_to_string(op) for op in code.get_logical_ops(Pauli.Z, symplectic=True)],
            allow_redundant=True,
            allow_underconstrained=True,
        )

    # identify stabilizers, logical operators, and gauge operators
    stab_ops = code.get_stabilizer_ops(canonicalized=True)
    logical_ops = code.get_logical_ops()
    gauge_ops = code.get_gauge_ops()

    """
    Construct "candidate" destabilizers that have correct pair-wise (anti-)commutation relations
    with the stabilizers, but may contain extra stabilizer, logical, or gauge operator components.
    """
    destab_ops = code.field.Zeros((len(stab_ops), 2 * len(code)), dtype=int)
    pivots = np.argmax(stab_ops.view(np.ndarray).astype(bool), axis=1)
    for destab_op, pivot in zip(destab_ops, pivots):
        destab_op[(pivot + len(code)) % (2 * len(code))] = 1

    # remove logical and gauge operator components
    dual_logical_ops = logical_ops.reshape(2, -1)[::-1, :].reshape(logical_ops.shape)
    dual_gauge_ops = gauge_ops.reshape(2, -1)[::-1, :].reshape(gauge_ops.shape)
    destab_ops -= destab_ops @ math.symplectic_conjugate(dual_logical_ops).T @ logical_ops
    destab_ops -= destab_ops @ math.symplectic_conjugate(dual_gauge_ops).T @ gauge_ops

    """
    Remove stabilizer factors to enforce that destabilizers commute with each other.  This process
    requires updating one destabilizer at a time, since each time we modify a destabilizer by
    stabilizer factors, that changes its commutation relations with other destabilizers.
    """
    for row, destab_op in enumerate(destab_ops[1:], start=1):
        destab_op -= destab_op @ math.symplectic_conjugate(destab_ops[:row]).T @ stab_ops[:row]

    # construct Pauli strings to hand over to Stim
    matrices_x = [logical_ops[: code.dimension], gauge_ops[: code.gauge_dimension], destab_ops]
    matrices_z = [logical_ops[code.dimension :], gauge_ops[code.gauge_dimension :], stab_ops]
    strings_x = [math.op_to_string(op) for matrix in matrices_x for op in matrix]
    strings_z = [math.op_to_string(op) for matrix in matrices_z for op in matrix]
    return stim.Tableau.from_conjugated_generators(xs=strings_x, zs=strings_z)


@restrict_to_qubits
def get_encoding_circuit(code: codes.QuditCode, *, only_zero: bool = False) -> stim.Circuit:
    """Circuit to encode physical states at its input into logical states of the given code.

    If only_zero is True, this circuit maps an all-0 physical state at its input to an all-0 logical
    state at its output.  Otherwise, for all j in {0, 1, ..., code.dimension - 1}, this circuit maps
    weight-one X_j and Z_j operators at its input to the logical X and Z operators of the j-th
    logical qubit of the code.  Weight-one Z_j operators for j >= code.dimension get mapped to
    "Z-type" gauge operators and stabilizers, and their conjugate X_j get mapped to "X-type" gauge
    operators and destabilizers.
    """
    return get_encoding_tableau(code, only_zero=only_zero).to_circuit()


@restrict_to_qubits
def get_encoder_and_decoder(
    code: codes.QuditCode, deformation: stim.Circuit | stim.Tableau | None = None
) -> tuple[stim.Tableau, stim.Tableau]:
    """Encoder for a code, and decoder either the same code or a deformed code."""
    encoder = get_encoding_tableau(code)
    if deformation is None:
        return encoder, encoder.inverse()
    deformation = deformation if isinstance(deformation, stim.Circuit) else deformation.to_circuit()
    deformed_code = code.deformed(deformation, preserve_logicals=True)
    decoder = get_encoding_tableau(deformed_code).inverse()
    return encoder, decoder


@restrict_to_qubits
def get_logical_tableau(
    code: codes.QuditCode,
    physical_circuit_or_tableau: stim.Circuit | stim.Tableau,
    *,
    deform_code: bool = False,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit or tableau.

    If deform_code is True, then the physical circuit is required to have two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to
        code.deformed(physical_circuit, preserve_logicals=True)
    """
    physical_circuit = (
        physical_circuit_or_tableau
        if isinstance(physical_circuit_or_tableau, stim.Circuit)
        else physical_circuit_or_tableau.to_circuit()
    )
    encoder, decoder = get_encoder_and_decoder(code, physical_circuit if deform_code else None)
    return _get_logical_tableau_from_code_data(
        code.dimension, code.gauge_dimension, encoder, decoder, physical_circuit
    )


def _get_logical_tableau_from_code_data(
    dimension: int,  # number of logical qubits of a QuditCode
    gauge_dimension: int,  # number of gauge qubits of a QuditCode
    encoder: stim.Tableau,
    decoder: stim.Tableau,
    physical_circuit: stim.Circuit,
    validate: bool = True,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit."""
    assert len(encoder) == len(decoder) >= dimension + gauge_dimension
    identity_phys = stim.Circuit(f"I {len(encoder) - 1}")
    physical_tableau = (physical_circuit + identity_phys).to_tableau()

    # compute the "upper left" block of the decoded tableau that acts on all logical qubits
    decoded_tableau = encoder.then(physical_tableau).then(decoder)
    x2x, x2z, z2x, z2z, x_signs, z_signs = decoded_tableau.to_numpy()
    logical_tableau = stim.Tableau.from_numpy(
        x2x=x2x[:dimension, :dimension],
        x2z=x2z[:dimension, :dimension],
        z2x=z2x[:dimension, :dimension],
        z2z=z2z[:dimension, :dimension],
        x_signs=x_signs[:dimension],
        z_signs=z_signs[:dimension],
    )

    if validate:
        # identify sectors that address logical, gauge, and stabilizer qubits
        sector_l = slice(dimension)
        sector_g = slice(dimension, dimension + gauge_dimension)
        sector_s = slice(dimension + gauge_dimension, len(encoder))

        # sanity check: stabilizers, logicals, and gauge operators should not pick up destabilizers
        assert not np.any(z2x[:, sector_s])
        assert not np.any(x2x[sector_l, sector_s])
        assert not np.any(x2x[sector_g, sector_s])

        # sanity check: gauge operators should not pick up logical factors
        assert not np.any(x2x[sector_g, sector_l])
        assert not np.any(x2z[sector_g, sector_l])
        assert not np.any(z2x[sector_g, sector_l])
        assert not np.any(z2z[sector_g, sector_l])

    return logical_tableau


def with_remapped_qubits(
    circuit: stim.Circuit, qubit_map: Mapping[int, int] | Sequence[int], *, inverse: bool = False
) -> stim.Circuit:
    """The same circuit, but with relabeled qubits.

    Qubits not in qubit_map get mapped to themselves.

    Args:
        circuit: The circuit to remap.
        qubit_map: Either a mapping (e.g., dictionary) from old to new qubit indices, or a sequence
            for which the qubit at index old_index gets mapped to new_index = qubit_map[old_index].
        inverse: If True, invert the provided qubit_map.  Default: False.

    Returns:
        stim.Circuit: A remapped circuit.
    """
    qubit_map = (
        qubit_map
        if isinstance(qubit_map, Mapping)
        else {old_index: new_index for old_index, new_index in enumerate(qubit_map)}
    )
    if inverse:
        qubit_map = {val: key for key, val in qubit_map.items()}

    new_circuit = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            block = stim.CircuitRepeatBlock(
                repeat_count=op.repeat_count,
                body=with_remapped_qubits(op.body_copy(), qubit_map),
                tag=op.tag,
            )
            new_circuit.append(block)

        else:
            new_targets = [_remap_target(target, qubit_map) for target in op.targets_copy()]
            new_op = stim.CircuitInstruction(
                name=op.name, targets=new_targets, gate_args=op.gate_args_copy(), tag=op.tag
            )
            new_circuit.append(new_op)

    return new_circuit


def _remap_target(target: stim.GateTarget, qubit_map: Mapping[int, int]) -> stim.GateTarget:
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
