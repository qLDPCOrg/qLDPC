"""Tools for identifying the transversal logical Clifford gates of a code

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

import collections
from collections.abc import Collection, Sequence

import numpy as np
import stim

from qldpc import abstract, cache, codes
from qldpc.math import op_to_string, symplectic_conjugate
from qldpc.objects import Pauli

from .common import _get_logical_tableau_from_code_data, get_encoder_and_decoder, restrict_to_qubits


@restrict_to_qubits
def get_transversal_ops(
    code: codes.QuditCode,
    local_gates: Collection[str] = ("S", "H", "SWAP"),
    *,
    deform_code: bool = False,
    remove_redundancies: bool = True,
) -> list[tuple[stim.Tableau, stim.Circuit]]:
    """Logical tableaus and physical circuits for transversal logical Clifford gates of a code.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    If deform_code is True, then a physical_circuit returned by this method has two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to
        code.deform(physical_circuit, preserve_logicals=True)

    Uses the methods of https://arxiv.org/abs/2409.18175.
    """
    group_aut = get_transversal_automorphism_group(code, local_gates, deform_code=deform_code)

    transversal_ops: list[tuple[stim.Tableau, stim.Circuit]] = []
    for generator in group_aut.generators:
        logical_tableau, physical_circuit = _get_transversal_automorphism_data(
            code, generator, local_gates, deform_code=deform_code
        )
        if not remove_redundancies or not (
            _is_pauli_tableau(logical_tableau)
            or any(
                _tableaus_are_equivalent_mod_paulis(logical_tableau, tableau)
                for tableau, _ in transversal_ops
            )
        ):
            transversal_ops.append((logical_tableau, physical_circuit))

    return transversal_ops


@restrict_to_qubits
def get_transversal_automorphism_group(
    code: codes.QuditCode,
    local_gates: Collection[str] = ("S", "H", "SWAP"),
    *,
    deform_code: bool = False,
) -> abstract.Group:
    """Construct the transversal Clifford automorphism group of a QuditCode.

    The transversal automorphism group of a QuditCode is the group of logical Clifford operations
    that can be implemented transversally with a given local gate set.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    If deform_code is True, then each member of the automorphism group constructed by this method
    corresponds to a physical_circuit that has two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to
        code.deform(physical_circuit, preserve_logicals=True)

    Uses the methods of https://arxiv.org/abs/2409.18175.
    """
    local_gates = _validate_local_gates(local_gates)
    allow_swaps = "SWAP" in local_gates
    local_gates.discard("SWAP")

    """
    Construct an instrumental classical code whose code words represent Pauli strings that satisfy
    some "effective" parity checks (i.e., commutation requirements).

    If computing the "ordinary" transversal automorphism group of a QuditCode (i.e., if deform_code
    is False), these effective parity checks are just the parity checks of the QuditCode, so the
    automorphism group of the instrumental classical code is the group of transversal physical
    operations that
    (a) preserve commutation with stabilizers (and gauge operators, if applicable), or equivalently
    (b) stabilize the logical Pauli group, thereby implementing logical Clifford operations.

    If deform_code is True, the effective parity checks are the logical Pauli operators of the
    QuditCode, so the automorphism group of the instrumental code represents the group of code
    deformations for which the logical Pauli group of the original QuditCode is a valid choice of
    logical Pauli group for the deformed QuditCode.
    """
    parity_checks = code.matrix if not deform_code else symplectic_conjugate(code.get_logical_ops())
    matrix_x = parity_checks.reshape(-1, 2, len(code))[:, 0, :]
    matrix_z = parity_checks.reshape(-1, 2, len(code))[:, 1, :]
    if not local_gates or local_gates == {"H"}:
        # swapping sectors = swapping X <--> Z
        matrix = np.hstack([matrix_x, matrix_z])
    elif local_gates == {"S"}:
        # swapping sectors = swapping X <--> Y
        matrix = np.hstack([matrix_z, matrix_x + matrix_z])
    elif local_gates == {"SQRT_X"}:
        # swapping sectors = swapping Y <--> Z
        matrix = np.hstack([matrix_x, matrix_x + matrix_z])
    else:
        # we have a complete local Clifford gate set that can arbitrarily permute Pauli ops
        matrix = np.hstack([matrix_x, matrix_z, matrix_x + matrix_z])

    # compute the automorphism group of an instrumental classical code
    instrumental_code = codes.ClassicalCode(matrix)
    group_code = instrumental_code.get_automorphism_group()

    # identify the group of instrumental code transformations generated by the gate set
    num_sectors = len(instrumental_code) // len(code)
    column_perms = []
    if allow_swaps:
        # allow swapping qudits, which swaps corresponding columns in all "sectors" concurrently
        column_perms += [
            [(ss * len(code) + qq, ss * len(code) + qq + 1) for ss in range(num_sectors)]
            for qq in range(len(code) - 1)
        ]
    if local_gates:
        # local gates can permute "sectors" arbitrarily and independently on any qudit
        column_perms += [
            [(ss * len(code) + qq, (ss + 1) * len(code) + qq)]
            for ss in range(num_sectors - 1)
            for qq in range(len(code))
        ]
    group_gates = abstract.Group(*map(abstract.GroupMember, column_perms))

    # intersect the groups above to find the group generated by a transversal gate set
    group_aut_gens = _sympy_group_intersection_generators(group_code, group_gates)
    return abstract.Group(*map(abstract.GroupMember, group_aut_gens))


@restrict_to_qubits
def get_transversal_circuits(
    code: codes.QuditCode,
    logical_circuits_or_tableaus: Sequence[stim.Circuit | stim.Tableau],
    local_gates: Collection[str] = ("S", "H", "SWAP"),
    *,
    deform_code: bool = False,
) -> list[stim.Circuit | None]:
    """Find transversal physical circuits (if any) that implement logical Clifford operations.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    If deform_code is True, then a physical_circuit returned by this method has two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to
        code.deform(physical_circuit, preserve_logicals=True)

    Warning: this method performs a brute-force search over the Clifford automorphisms of a code,
    and thereby generally has exponential runtime.
    """
    physical_circuits = [None] * len(logical_circuits_or_tableaus)

    # convert logical Cliffords into tableaus
    identity = stim.Circuit(f"I {code.dimension - 1}")  # to ensure circuits address all qubits
    logical_tableaus = [
        (
            logical_circuit_or_tableau
            if isinstance(logical_circuit_or_tableau, stim.Tableau)
            else (logical_circuit_or_tableau + identity).to_tableau()
        )
        for logical_circuit_or_tableau in logical_circuits_or_tableaus
    ]

    # compute the group of transversal Cliffords
    group_aut = get_transversal_automorphism_group(code, local_gates, deform_code=deform_code)

    # perform a brute-force search for matching Clifford operations
    matching_ops: list[tuple[stim.Tableau, stim.Circuit] | None] = [None] * len(logical_tableaus)
    for automorphism in group_aut.generate():
        tableau, circuit = _get_transversal_automorphism_data(
            code, automorphism, local_gates, deform_code
        )
        for tt, logical_tableau in enumerate(logical_tableaus):
            if matching_ops[tt] is None and _tableaus_are_equivalent_mod_paulis(
                logical_tableau, tableau
            ):
                matching_ops[tt] = tableau, circuit
        if not any(op is None for op in matching_ops):
            break

    # add logical Pauli corrections to matching circuits
    for tt, (logical_tableau, matching_op) in enumerate(zip(logical_tableaus, matching_ops)):
        if matching_op is None:
            continue
        matching_tableau, matching_circuit = matching_op

        correction = code.field([0] * (2 * len(code)))
        *_, x_signs_l, z_signs_l = logical_tableau.to_numpy()
        *_, x_signs_m, z_signs_m = matching_tableau.to_numpy()
        for logical_qubit in range(code.dimension):
            if x_signs_l[logical_qubit] != x_signs_m[logical_qubit]:  # pragma: no cover
                correction += codes.QuditCode.get_logical_ops(code, Pauli.Z)[logical_qubit]
            if z_signs_l[logical_qubit] != z_signs_m[logical_qubit]:  # pragma: no cover
                correction += codes.QuditCode.get_logical_ops(code, Pauli.X)[logical_qubit]
        correction_circuit = _get_pauli_circuit(op_to_string(correction))

        physical_circuits[tt] = correction_circuit + matching_circuit

    return physical_circuits


def get_transversal_circuit(
    code: codes.QuditCode,
    logical_circuit_or_tableau: stim.Circuit | stim.Tableau,
    local_gates: Collection[str] = ("S", "H", "SWAP"),
    *,
    deform_code: bool = False,
) -> stim.Circuit | None:
    """Find a transversal physical circuit (if any) that implements a logical Clifford operation.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    If deform_code is True, then a physical_circuit returned by this method has two effects, namely
    (a) transforming a logical state of the QuditCode by a corresponding logical Clifford gate, and
    (b) changing the code that encodes the logical state to
        code.deform(physical_circuit, preserve_logicals=True)

    Warning: this method performs a brute-force search over the Clifford automorphisms of a code,
    and thereby generally has exponential runtime.
    """
    return get_transversal_circuits(
        code,
        [logical_circuit_or_tableau],
        local_gates,
        deform_code=deform_code,
    )[0]


def _is_pauli_tableau(tableau: stim.Tableau) -> bool:
    """Does this Tableau represent a Pauli operator?

    If so, it maps Pauli strings to themselves up to sign.
    """
    identity_mat = np.identity(len(tableau), dtype=bool)
    x2x, x2z, z2x, z2z, *_ = tableau.to_numpy()
    return (
        np.array_equal(identity_mat, x2x)
        and np.array_equal(identity_mat, z2z)
        and not np.any(x2z)
        and not np.any(z2x)
    )


def _tableaus_are_equivalent_mod_paulis(tableau_1: stim.Tableau, tableau_2: stim.Tableau) -> bool:
    """Are the two stabilizer tableaus equivalent, up to Paulis?"""
    x2x_1, x2z_1, z2x_1, z2z_1, *_ = tableau_1.to_numpy()
    x2x_2, x2z_2, z2x_2, z2z_2, *_ = tableau_2.to_numpy()
    return (
        np.array_equal(x2x_1, x2x_2)
        and np.array_equal(x2z_1, x2z_2)
        and np.array_equal(z2x_1, z2x_2)
        and np.array_equal(z2z_1, z2z_2)
    )


@cache.use_disk_cache(
    "group_intersection",
    key_func=lambda xx, yy: (xx.hashable_generators(), yy.hashable_generators()),
)
def _sympy_group_intersection_generators(
    group_a: abstract.Group, group_b: abstract.Group
) -> tuple[tuple[int, ...], ...]:
    """Get the generators of the intersection of two Sympy permutation groups."""
    group_sympy = group_a.to_sympy().subgroup_search(group_b.to_sympy().contains)
    return abstract.Group(group_sympy).hashable_generators()


@restrict_to_qubits
def _get_transversal_automorphism_data(
    code: codes.QuditCode,
    automorphism: abstract.GroupMember,
    local_gates: Collection[str],
    deform_code: bool,
) -> tuple[stim.Tableau, stim.Circuit]:
    """Logical tableau and physical circuit for a transversal Clifford automorphism of a code.

    Here the local_gates and deform_code must be the same as those used to construct the
    automorphism group.
    """
    # construct a circuit with the desired action modulo destabilizers
    physical_circuit = stim.Circuit()
    physical_circuit += _get_pauli_permutation_circuit(code, automorphism, local_gates)
    physical_circuit += _get_swap_circuit(code, automorphism)

    # make sure that the physical circuit acts on all physical qubits
    if physical_circuit.num_qubits < len(code):
        physical_circuit.append("I", len(code) - 1)

    # Determine the effect of physical_circuit on "decoded" qubits, for which
    # logicals, stabilizers, and destabilizers are single-qubit Paulis.
    encoder, decoder = get_encoder_and_decoder(code, physical_circuit if deform_code else None)
    decoded_tableau = encoder.then(physical_circuit.to_tableau()).then(decoder)

    # Prepend Pauli corrections to the circuit: a product of destabilizers whose corresponding
    # stabilizers change sign under the physical_circuit.
    decoded_correction = "_" * code.dimension  # identity on the logical qubits
    for aa in range(code.dimension, len(code)):
        decoded_stabilizer_before = "_" * aa + "Z" + "_" * (len(code) - aa - 1)
        decoded_stabilizer_after = decoded_tableau(stim.PauliString(decoded_stabilizer_before))
        decoded_correction += "_" if decoded_stabilizer_after.sign == 1 else "X"
    correction = encoder(stim.PauliString(decoded_correction))
    physical_circuit = _get_pauli_circuit(correction) + physical_circuit

    # identify the logical tableau implemented by the physical circuit
    logical_tableau = _get_logical_tableau_from_code_data(
        code.dimension, code.gauge_dimension, encoder, decoder, physical_circuit
    )

    return logical_tableau, physical_circuit


def _validate_local_gates(local_gates: Collection[str]) -> set[str]:
    """Verify that the provided collection of local gates is supported."""
    allowed_gates = {"S", "H", "SQRT_X", "SWAP"}
    if not allowed_gates.issuperset(local_gates):
        raise ValueError(
            f"Local Clifford gates (provided: {local_gates}) must be subset of {allowed_gates}"
        )
    return set(local_gates)


def _get_swap_circuit(code: codes.QuditCode, automorphism: abstract.GroupMember) -> stim.Circuit:
    """Construct the circuit of SWAPs applied by a transversal Clifford automorphism."""
    circuit = stim.Circuit()
    new_locs = [automorphism(qubit) % len(code) for qubit in range(len(code))]
    for cycle in abstract.GroupMember(new_locs).cyclic_form:
        loc_0 = cycle[0]
        for loc_1 in cycle[1:]:
            circuit.append("SWAP", [loc_0, loc_1])
    return circuit


@restrict_to_qubits
def _get_pauli_permutation_circuit(
    code: codes.QuditCode,
    automorphism: abstract.GroupMember,
    local_gates: Collection[str],
) -> stim.Circuit:
    """Construct the circuit of local Pauli permutations applied by a transversal automorphism."""
    local_gates = _validate_local_gates(local_gates)
    local_gates.discard("SWAP")

    circuit = stim.Circuit()

    if len(local_gates) == 1:
        # there is only one local gate, and all it can do is permute two parity check sectors
        gate = next(iter(local_gates))  # extract the only gate
        for qubit in range(len(code)):
            if automorphism(qubit) >= len(code):
                circuit.append(gate, qubit)

    elif len(local_gates) > 1:
        # we have a complete local Clifford gate set that can permute Pauli sectors arbitrarily
        gate_targets = collections.defaultdict(list)
        for qubit in range(len(code)):
            pauli_perm = [automorphism(qubit + ss * len(code)) // len(code) for ss in range(3)]
            match pauli_perm:
                case [1, 0, 2]:  # X <--> Z
                    gate_targets["H"].append(qubit)
                case [0, 2, 1]:  # X <--> Y
                    gate_targets["S"].append(qubit)
                case [2, 1, 0]:  # Y <--> Z
                    gate_targets["H_YZ"].append(qubit)
                case [2, 0, 1]:  # ZXY <--> XYZ
                    gate_targets["C_ZYX"].append(qubit)  # pragma: no cover
                case [1, 2, 0]:  # ZXY <--> ZYX
                    gate_targets["C_XYZ"].append(qubit)  # pragma: no cover

        for gate, targets in gate_targets.items():
            circuit.append(gate, sorted(targets))

    return circuit


def _get_pauli_circuit(string: stim.PauliString) -> stim.Circuit:
    """Stim circuit to apply a Pauli string."""
    circuit = stim.Circuit()
    for pauli in ["X", "Y", "Z"]:
        if indices := string.pauli_indices(pauli):
            circuit.append(pauli, indices)
    return circuit
