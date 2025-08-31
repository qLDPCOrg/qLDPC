"""Helper objects to keep track of qubits, measurements, and detectors

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
from collections.abc import Iterator, Sequence

import stim

from qldpc import codes


@dataclasses.dataclass
class QubitIDs:
    """Container to keep track of the indices of qubits in a circuit."""

    data: tuple[int, ...]  # data qubits in an error-correcting code
    check: tuple[int, ...]  # qubits used to measure parity checks in an error-correcting code
    ancilla: tuple[int, ...]  # miscellaneous ancilla qubits

    # identify X-check and Z-check qubits for CSS codes
    checks_x: tuple[int, ...] = ()
    checks_z: tuple[int, ...] = ()

    def __init__(
        self, data: Sequence[int], check: Sequence[int], ancilla: Sequence[int] = ()
    ) -> None:
        self.data = tuple(data)
        self.check = tuple(check)
        self.ancilla = tuple(ancilla)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Iterate over the collections of qubits tracked by this QubitIDs object."""
        yield from (self.data, self.check, self.ancilla)

    @staticmethod
    def from_code(code: codes.QuditCode) -> QubitIDs:
        """Initialize from an error-correcting code with specific parity checks."""
        data = tuple(range(len(code)))
        check = tuple(range(len(code), len(code) + code.num_checks))
        qubit_ids = QubitIDs(data, check)
        qubit_ids.checks_x = check[: code.num_checks_x] if isinstance(code, codes.CSSCode) else ()
        qubit_ids.checks_z = check[code.num_checks_x :] if isinstance(code, codes.CSSCode) else ()
        return qubit_ids

    @staticmethod
    def validated(qubit_ids: QubitIDs, code: codes.QuditCode) -> QubitIDs:
        """Validate qubit IDs for the given code and return."""
        if len(qubit_ids.data) != len(code) or len(qubit_ids.check) != code.num_checks:
            raise ValueError("Qubit IDs are invalid for the given code")
        if isinstance(code, codes.CSSCode):
            qubit_ids.checks_x = tuple(qubit_ids.check[: code.num_checks_x])
            qubit_ids.checks_z = tuple(qubit_ids.check[code.num_checks_x :])
        return qubit_ids

    def add_ancillas(self, number: int) -> None:
        """Add ancilla qubits."""
        if number > 0:
            start = max(itertools.chain(*self)) + 1
            self.ancilla += tuple(range(start, start + number))


class Record:
    """An organized record of events in a Stim circuit.

    A record is essentially a dictionary that maps some key (such as a qubit index) to an ordered
    list of the events (such as measurements or detectors) associated with that key.  The events that
    a Record keeps track of are assumed to be indexed from zero.

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
        """Append the given record to this one.

        All event numbers in the appended record are increased by the number of events in the current
        record.  That is, if the current record holds n events numbered from 0 to n - 1, then events
        (0, 1, ...) in the appended record are added to the current record as (n, n+1, ...).
        """
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
