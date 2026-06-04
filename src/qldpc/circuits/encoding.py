"""Encoding circuits, logical tableaus, and analysis of logical states

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

from collections.abc import Sequence
from typing import ParamSpec, TypeVar

import numpy as np
import stim

from qldpc import codes, math
from qldpc.objects import Pauli

from .common import restrict_to_qubits

CircuitOrTableau = TypeVar("CircuitOrTableau", stim.Circuit, stim.Tableau)
Params = ParamSpec("Params")


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

    # Construct "candidate" destabilizers that have correct pair-wise (anti-)commutation relations
    # with the stabilizers, but may contain extra stabilizer, logical, or gauge operator components.
    stab_pivots = math.first_nonzero_cols(stab_ops)
    destab_ops = code.field.Zeros((len(stab_ops), 2 * len(code)), dtype=int)
    for destab_op, pivot in zip(destab_ops, stab_pivots):
        destab_op[(pivot + len(code)) % (2 * len(code))] = 1

    # remove logical and gauge operator components
    dual_logical_ops = logical_ops.reshape(2, -1)[::-1, :].reshape(logical_ops.shape)
    dual_gauge_ops = gauge_ops.reshape(2, -1)[::-1, :].reshape(gauge_ops.shape)
    destab_ops -= destab_ops @ math.symplectic_conjugate(dual_logical_ops).T @ logical_ops
    destab_ops -= destab_ops @ math.symplectic_conjugate(dual_gauge_ops).T @ gauge_ops

    # enforce that destabilizers commute with each other by removing stabilizer factors
    for dd in range(len(destab_ops)):
        for ss in range(dd, len(destab_ops)):
            if destab_ops[dd] @ math.symplectic_conjugate(destab_ops[ss]):
                destab_ops[dd] -= stab_ops[ss]

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


def restrict_tableau(tableau: stim.Tableau, qubits: Sequence[int]) -> stim.Tableau:
    """Restrict the given stabilizer tableau to the sub-tableau at the specified qubits."""
    x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
    return stim.Tableau.from_numpy(
        x2x=x2x[np.ix_(qubits, qubits)],
        x2z=x2z[np.ix_(qubits, qubits)],
        z2x=z2x[np.ix_(qubits, qubits)],
        z2z=z2z[np.ix_(qubits, qubits)],
        x_signs=x_signs[qubits],
        z_signs=z_signs[qubits],
    )


@restrict_to_qubits
def get_nontrivial_logical_stabilizers(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit, *, skip_validation: bool = False
) -> list[stim.PauliString]:
    """Identify a complete basis for the nontrivial logical Pauli stabilizers of the prepared state.

    The first len(code) qubits addressed by the circuit must be the data qubits of the code.

    Args:
        code: The code whose logical state is prepared by the provided state_prep_circuit.
        state_prep_circuit: A circuit that prepares a logical state of the provided code.

    Keyword args:
        skip_validation: If True, skip the check to assert that the provided circuit prepares a
            logical state of the provided code.

    Returns:
        A list of logical Pauli operators supported on the data qubits of the provided code.
    """
    if not skip_validation:
        _assert_valid_code_state(code, state_prep_circuit)

    # identify stabilizers of the prepared state that are supported on the data qubits of the code
    decoded_stabilizers = get_state_stabilizers(code, state_prep_circuit, decoded=True)

    # identify stabilizers of the state that are pure logicals
    logical_stab_mat = [math.string_to_op(stab[: code.dimension]) for stab in decoded_stabilizers]
    logical_stab_mat_rref = code.field(logical_stab_mat).row_reduce()
    logical_stab_mat_rref = logical_stab_mat_rref[np.any(logical_stab_mat_rref, axis=1)]
    return logical_stab_mat_rref @ code.get_logical_ops()


@restrict_to_qubits
def get_state_stabilizers(
    code: codes.QuditCode, state_prep_circuit: stim.Circuit, *, decoded: bool = False
) -> list[stim.PauliString]:
    """Identify stabilizers of the prepared state that are supported on the data qubits of the code.

    If 'decoded is True', return these stabilizers in the "decoded" basis of logical operators, gauge
    operators, and stabilizers/destabilizers.  That is, for each string in the returned list of Pauli
    strings, the first code.dimension qubits represent logical operators, the next
    code.gauge_dimension qubits represent gauge operators, and the remaining qubits represent
    stabilizer/destabilizer qubits.

    The strategy in this method is as follows.  If we prepend reset operations to make an initial
    |0...0⟩ initial state explicit, then all stabilizer flows of the circuit should have the form
        1 -> output_generator,
    where each output_generator is an XOR of
        (a) a Pauli string,
        (b) measurements, and
        (c) observables.
    That is, the circuit prepares a state for which, after identifying {+1,-1} <-> {0,1} as
    necessary, the XOR of (a), (b), and (c) for each output_generator is 0.

    To identify a basis of output generators that are supported only on the data qubits of the code,
    we collect all output generators into a binary matrix, and row-reduce this matrix.
    """
    resets = stim.Circuit("R " + " ".join(map(str, range(state_prep_circuit.num_qubits))))
    flow_generators = (resets + state_prep_circuit.without_noise()).flow_generators()

    # identify some useful numbers
    num_qubits = state_prep_circuit.num_qubits
    num_measurements = state_prep_circuit.num_measurements
    num_observables = state_prep_circuit.num_observables
    num_columns = 2 * num_qubits + 1 + num_measurements + num_observables
    num_rows = len(flow_generators) + state_prep_circuit.num_detectors

    # build a matrix of stabilizers for the entire circuit output, determined by the flows
    matrix = code.field.Zeros((num_rows, num_columns))
    for gg, flow in enumerate(flow_generators):
        pauli_string = flow.output_copy()
        if pauli_string:
            xs, zs = pauli_string.to_numpy()
            matrix[gg, :num_qubits] = xs.astype(np.uint8)
            matrix[gg, num_qubits : 2 * num_qubits] = zs.astype(np.uint8)
            matrix[gg, 2 * num_qubits] = 0 if pauli_string.sign == 1 else 1
        for measurement in flow.measurements_copy():
            matrix[gg, -num_observables - num_measurements + measurement] = 1
        for observable in flow.included_observables_copy():  # pragma: no cover technical edge case
            matrix[gg, -num_observables + observable] = 1

    # add stabilizers supported on measurements, as identified by detectors in the provided circuit
    detector_counter = 0
    measurement_counter = 0
    for instruction in state_prep_circuit.flattened():
        measurement_counter += instruction.num_measurements
        if instruction.name == "DETECTOR":
            row = len(flow_generators) + detector_counter
            for target in instruction.targets_copy():
                col = -num_observables - num_measurements + measurement_counter + target.value
                matrix[row, col] = 1
            detector_counter += 1

    # extract stabilizers that are supported entirely on the data qubits of the code
    stabilizers = []
    for row in matrix.row_reduce():
        xs = row[: len(code)]
        zs = row[num_qubits : num_qubits + len(code)]
        other_xs = row[len(code) : num_qubits]
        other_zs = row[num_qubits + len(code) : 2 * num_qubits]
        meas_obs = row[2 * num_qubits + 1 :]
        any_on_others = np.any(other_xs) or np.any(other_zs) or np.any(meas_obs)
        if (np.any(xs) or np.any(zs)) and not any_on_others:
            sign = -1 if row[2 * num_qubits] else 1
            string = stim.PauliString.from_numpy(xs=xs != 0, zs=zs != 0, sign=sign)
            stabilizers.append(string)

    if decoded:
        encoder = get_encoding_tableau(code)
        stabilizers = [stab.before(encoder, targets=range(len(code))) for stab in stabilizers]
    return stabilizers


def _get_logical_tableau_from_code_data(
    dimension: int,  # number of logical qubits of a QuditCode
    gauge_dimension: int,  # number of gauge qubits of a QuditCode
    encoder: stim.Tableau,
    decoder: stim.Tableau,
    physical_circuit: stim.Circuit,
    skip_validation: bool = False,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit."""
    assert len(encoder) == len(decoder) >= dimension + gauge_dimension
    identity_phys = stim.Circuit(f"I {len(encoder) - 1}")
    physical_tableau = (physical_circuit + identity_phys).to_tableau()

    # compute the "upper left" block of the decoded tableau that acts on all logical qubits
    decoded_tableau = encoder.then(physical_tableau).then(decoder)
    logical_tableau = restrict_tableau(decoded_tableau, range(dimension))

    if not skip_validation:
        # identify sectors that address logical, gauge, and stabilizer qubits
        sector_l = slice(dimension)
        sector_g = slice(dimension, dimension + gauge_dimension)
        sector_s = slice(dimension + gauge_dimension, len(encoder))
        x2x, x2z, z2x, z2z, *_ = decoded_tableau.to_numpy()

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


def _assert_valid_code_state(code: codes.QuditCode, state_prep_circuit: stim.Circuit) -> None:
    """Assert that the provided circuit prepares a logical state of the provided code."""
    simulator = stim.TableauSimulator()
    simulator.do(state_prep_circuit)
    for op in code.get_stabilizer_ops():
        string = math.op_to_string(op)
        if not simulator.peek_observable_expectation(string) == 1:
            raise ValueError(
                "The provided circuit does not deterministically prepare a logical code state that"
                " is unentangled from ancillas"
            )
