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
import copy
import dataclasses
import itertools
from collections.abc import Hashable, ItemsView, Iterator, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import stim
from typing_extensions import Self

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
    def from_code(code: codes.QuditCode, *, num_ancillas: int = 0) -> QubitIDs:
        """Initialize from an error-correcting code with specific parity checks."""
        data = tuple(range(len(code)))
        check = tuple(range(len(code), len(code) + code.num_checks))
        ancilla = tuple(range(check[-1] + 1, check[-1] + 1 + num_ancillas))
        qubit_ids = QubitIDs(data, check, ancilla)
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

    def max(self) -> int:
        """The largest index of any tracked qubit."""
        return max(itertools.chain(*self))

    def shift(self, shift: int) -> QubitIDs:
        """Shift all qubit indices by the given amount and return self."""
        self.data = tuple(qq + shift for qq in self.data)
        self.check = tuple(qq + shift for qq in self.check)
        self.ancilla = tuple(qq + shift for qq in self.ancilla)
        self.checks_x = tuple(qq + shift for qq in self.checks_x)
        self.checks_z = tuple(qq + shift for qq in self.checks_z)
        return self

    def add_ancillas(self, number: int) -> None:
        """Add ancilla qubits."""
        if number > 0:
            start = self.max() + 1
            self.ancilla += tuple(range(start, start + number))


class Record(Mapping[Hashable, list[int]]):
    """An organized record of events in a Stim circuit.

    A record is essentially a dictionary that maps some key (such as a qubit index) to an ordered
    list of the events (such as measurements or detectors) associated with that key.  The events that
    a Record keeps track of are assumed to be indexed from zero.

    Record is subclassed by MeasurementRecord to keep track of measurements in a circuit, and
    by DetectorRecord to keep track of the detectors in a circuit.
    """

    num_events: int
    key_to_events: dict[Hashable, list[int]]

    def __init__(self, initial_record: Mapping[Hashable, Sequence[int]] | None = None) -> None:
        self.key_to_events = collections.defaultdict(list)
        if initial_record:
            self.key_to_events |= {key: list(events) for key, events in initial_record.items()}
        self.num_events = sum(len(events) for events in self.key_to_events.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self.key_to_events)})"

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        """The number of keys associated with events in this record."""
        return len(self.key_to_events)

    def __iter__(self) -> Iterator[Hashable]:
        """Iterator over the keys associated with events in this record."""
        yield from self.key_to_events.keys()

    def __getitem__(self, key: Hashable) -> list[int]:
        """The events associated with a key."""
        return self.key_to_events[key]

    def items(self) -> ItemsView[Hashable, list[int]]:
        """Iterator over keys and their associated events."""
        return self.key_to_events.items()

    def get_events(self, *keys: Hashable) -> list[int]:
        """All events associated with the given keys."""
        return [event for key in keys for event in self.key_to_events.get(key, [])]

    def copy(self) -> Self:
        """A copy of this Record."""
        return type(self)(
            {copy.deepcopy(key): copy.deepcopy(events) for key, events in self.items()}
        )

    def append(self, record: Mapping[Hashable, Sequence[int]], repeat: int = 1) -> None:
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

    def __iadd__(self, other: Mapping[Hashable, Sequence[int]]) -> Self:
        """Append the given record to this one.  See help(qldpc.circuits.Record.append)."""
        self.append(other)
        return self

    def __add__(self, other: Self) -> Self:
        """Combine two records."""
        record = self.copy()
        record.append(other)
        return record


class MeasurementRecord(Record):
    """An organized record of measurements in a Stim circuit."""

    def get_target_rec(self, qubit: Hashable, measurement_index: int = -1) -> stim.target_rec:
        """Retrieve a Stim measurement record target for the given qubit.

        Args:
            qubit: The qubit whose measurement record we want.
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
    """An organized record of detectors in a Stim circuit."""

    def get_detector(self, key: Hashable, detection_index: int = -1) -> int:
        """Retrieve a Stim detector (by index) assoiated with the given key.

        Args:
            key: The name associated with a sequence of detectors in the record.
            detection_index: An index specifying which detector in the specified sequence we want.
                A detection_index of 0 would be the first detector in the sequence, while a
                detection_index of -1 would be the last detector.  Default value: -1.

        Returns:
            int: The index of a detector.
        """
        detectors = self.get_events(key)
        if not -len(detectors) <= detection_index < len(detectors):
            raise ValueError(
                f"Invalid detection index {detection_index} for key '{key}' with {len(detectors)}"
                " detectors"
            )
        return detectors[detection_index]

    def after_post_selection(self, key: Hashable) -> DetectorRecord:
        """A record of the detectors remaining after post-selecting on the detectors of a key.

        If "detector_record" is the record of the detectors in circuit whose detector error model is
        represented by the qldpc.decoders.DetectorErrorModelArrays object "dem_arrays", the record
            new_detector_record = detector_record.after_post_selection(key)
        is the record of the detectors in
            new_dem_arrays = dem_arrays.post_selected_on(detector_record.get_events(key))
        See help(qldpc.decoders.DetectorErrorModelArrays).
        """
        # identify the indices of all detectors, and the detectors to remove
        last_detector = max(max(detectors) for detectors in self.values() if detectors)
        detector_indices = np.arange(last_detector + 1)
        detectors_to_remove = sorted(self.get_events(key))

        # for each detector D, find how many of the detectors_to_remove are <= D
        index_shift = np.searchsorted(detectors_to_remove, detector_indices, side="left")

        # shift detector indices down and remove the post-selection key
        detector_indices -= index_shift
        return DetectorRecord(
            {
                other_key: detector_indices[detectors].tolist()
                for other_key, detectors in self.items()
                if other_key != key
            }
        )


class MemoryExperimentParts(NamedTuple):
    initialization: stim.Circuit
    qec_cycle: stim.Circuit
    readout: stim.Circuit
    measurement_record: MeasurementRecord
    detector_record: DetectorRecord
    qubit_ids: QubitIDs
