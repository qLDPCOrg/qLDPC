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

import collections
import dataclasses
import functools
import itertools
import operator
from collections.abc import Callable, Iterator

import numpy as np
import stim

from qldpc import codes
from qldpc.math import op_to_string, symplectic_conjugate
from qldpc.objects import Pauli


@dataclasses.dataclass
class QubitIDs:
    """Container to keep track of the indices of qubits in a circuit."""

    data: tuple[int, ...]  # data qubits in an error-correcting code
    check: tuple[int, ...]  # qubits used to measure parity checks in an error-correcting code
    ancilla: tuple[int, ...]  # miscellaneous ancilla qubits

    # identify X-check and Z-check qubits for CSS codes
    checks_x: tuple[int, ...] = ()
    checks_z: tuple[int, ...] = ()

    @staticmethod
    def from_code(code: codes.QuditCode) -> QubitIDs:
        """Initialize from an error-correcting code with specific parity checks."""
        data = tuple(range(len(code)))
        check = tuple(range(len(code), len(code) + code.num_checks))
        checks_x = check[: code.num_checks_x] if isinstance(code, codes.CSSCode) else ()
        checks_z = check[code.num_checks_x :] if isinstance(code, codes.CSSCode) else ()
        return QubitIDs(data, check, (), checks_x, checks_z)

    @staticmethod
    def validated(qubit_ids: QubitIDs, code: codes.QuditCode) -> QubitIDs:
        """Validate qubit IDs for the given code and return."""
        if (
            len(qubit_ids.data) != len(code)
            or len(qubit_ids.check) != code.num_checks
            or (
                isinstance(code, codes.CSSCode)
                and (
                    len(qubit_ids.checks_x) != code.num_checks_x
                    or len(qubit_ids.checks_z) != code.num_checks_z
                )
            )
        ):
            raise ValueError("Qubit IDs are invalid for the given code")
        return qubit_ids

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Iterate over the collections of qubits tracked by this QubitIDs object."""
        yield from (self.data, self.check, self.ancilla)

    def add_ancilla(self, number: int = 1) -> None:
        """Add (one or more) ancilla qubits."""
        start = max(itertools.chain(*self)) + 1
        self.ancilla += tuple(range(start, start + number))


class Record:
    """An organized record of events in a Stim circuit.

    This record assumes that the events it keeps track of (such as measurements or detectors) are
    indexed from zero.  The record is essentially a dictionary that maps some key (such as a qubit
    index) to an ordered list of the events associated with that key.

    Record is subclassed by MeasurementRecord to keep track of measurements in a circuit, and
    by DetectorRecord to keep track of the detectors in a circuit.
    """

    num_events: int
    key_to_events: dict[int, list[int]]

    def __init__(self, initial_record: dict[int, list[int]] | None = None) -> None:
        self.key_to_events = collections.defaultdict(list, initial_record if initial_record else {})
        self.num_events = sum(len(events) for events in self.key_to_events.values())

    def items(self) -> Iterator[tuple[int, list[int]]]:
        """Iterator over keys and their associated events."""
        yield from self.key_to_events.items()

    def get_events(self, *keys: int) -> list[int]:
        """The events associated with a key."""
        return functools.reduce(operator.add, (self.key_to_events.get(key, []) for key in keys))

    def append(self, record: Record | dict[int, list[int]], repeat: int = 1) -> None:
        """Append the given record to this one."""
        assert repeat >= 0
        num_events_in_record = sum(len(events) for _, events in record.items())
        for key, events in record.items():
            self.key_to_events[key].extend(
                [
                    self.num_events + measurement + repetition * num_events_in_record
                    for repetition in range(repeat)
                    for measurement in events
                ]
            )
        self.num_events += num_events_in_record * repeat


class MeasurementRecord(Record):
    """An record of measurements in a Stim circuit, organized by qubit index."""

    def get_target_rec(self, qubit: int, measurement_index: int = -1) -> stim.target_rec:
        """Retrieve a Stim measurement record target for the given qubit.

        Args:
            qubit: The qubit (by index) whose measurement record we want.
            measurement_index: An index specifying which measurement of the specified qubit we want.
                A measurement_index of 0 would be the first measurement of the qubit, while a
                measurement_index of -1 would be the most recent measurement.  Default value: -1.

        Returns:
            stim.target_rec: A Stim measurement record target.
        """
        measurements = self.get_events(qubit)
        if not -len(measurements) <= measurement_index < len(measurements):
            raise ValueError(
                f"Invalid measurement index {measurement_index} for qubit {qubit} with "
                f"{len(measurements)} measurements"
            )
        return stim.target_rec(measurements[measurement_index] - self.num_events)


class DetectorRecord(Record):
    """An record of detectors in a Stim circuit, organized by parity check index."""

    def get_detector(self, check: int, detection_index: int = -1) -> int:
        """Retrieve a Stim detector (by index) for the given parity check.

        Args:
            check: The parity check (by index) whose detector we want.
            detection_index: An index specifying which detector of the specified parity check we
                want.  A detection_index of 0 would be the first detector of the parity check, while
                a detection_index of -1 would be the most recent detector.  Default value: -1.

        Returns:
            int: The index of the detector we want.
        """
        detectors = self.get_events(check)
        if not -len(detectors) <= detection_index < len(detectors):
            raise ValueError(
                f"Invalid detection index {detection_index} for parity check {check} with "
                f"{len(detectors)} detectors"
            )
        return detectors[detection_index]


def restrict_to_qubits(func: Callable[..., stim.Circuit]) -> Callable[..., stim.Circuit]:
    """Restrict a circuit constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: object, **kwargs: object) -> stim.Circuit:
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
            [op_to_string(op) for op in code.get_stabilizer_ops(symplectic=True)]
            + [op_to_string(op) for op in code.get_logical_ops(Pauli.Z, symplectic=True)],
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
    destab_ops -= destab_ops @ symplectic_conjugate(dual_logical_ops).T @ logical_ops
    destab_ops -= destab_ops @ symplectic_conjugate(dual_gauge_ops).T @ gauge_ops

    """
    Remove stabilizer factors to enforce that destabilizers commute with each other.  This process
    requires updating one destabilizer at a time, since each time we modify a destabilizer by
    stabilizer factors, that changes its commutation relations with other destabilizers.
    """
    for row, destab_op in enumerate(destab_ops[1:], start=1):
        destab_op -= destab_op @ symplectic_conjugate(destab_ops[:row]).T @ stab_ops[:row]

    # construct Pauli strings to hand over to Stim
    matrices_x = [logical_ops[: code.dimension], gauge_ops[: code.gauge_dimension], destab_ops]
    matrices_z = [logical_ops[code.dimension :], gauge_ops[code.gauge_dimension :], stab_ops]
    strings_x = [op_to_string(op) for matrix in matrices_x for op in matrix]
    strings_z = [op_to_string(op) for matrix in matrices_z for op in matrix]
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
