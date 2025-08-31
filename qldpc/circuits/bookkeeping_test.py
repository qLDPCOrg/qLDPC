"""Unit tests for bookkeeping.py

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

import pytest
import stim

from qldpc import circuits, codes


def test_qubit_ids() -> None:
    """Default qubit indices."""
    code = codes.SteaneCode()
    qubit_ids = circuits.QubitIDs.from_code(code)
    data_ids, check_ids, ancilla_ids = qubit_ids
    assert data_ids == tuple(range(len(code)))
    assert check_ids == tuple(range(len(code), len(code) + code.num_checks))
    assert not ancilla_ids

    num_ancillas = 3
    qubit_ids.add_ancillas(num_ancillas)
    assert qubit_ids.ancilla == tuple(
        range(len(code) + code.num_checks, len(code) + code.num_checks + num_ancillas)
    )

    assert qubit_ids == circuits.QubitIDs.validated(qubit_ids, code)
    with pytest.raises(ValueError, match="invalid for the given code"):
        circuits.QubitIDs.validated(circuits.QubitIDs((), (), ()), code)


def test_records() -> None:
    """Measurement and detector records."""
    record = circuits.Record({0: [0]})
    assert record.num_events == 1
    record.append({0: [1], 2: [0]})
    assert record.num_events == 3
    record.append({1: [0, 1]}, repeat=3)
    assert record.key_to_events[1] == [3, 4, 5, 6, 7, 8]
    assert dict(record.items()) == record.key_to_events

    measurement_record = circuits.MeasurementRecord(record.key_to_events)
    assert measurement_record.num_events == 9
    assert measurement_record.get_target_rec(2) == stim.target_rec(-8)
    assert measurement_record.get_target_rec(0) == stim.target_rec(-7)
    assert measurement_record.get_target_rec(0, -2) == stim.target_rec(-9)

    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(3)
    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(0, 2)

    detector_record = circuits.DetectorRecord(record.key_to_events)
    assert detector_record.num_events == 9
    assert detector_record.get_detector(2) == 1
    assert detector_record.get_detector(0) == 2
    assert detector_record.get_detector(0, -2) == 0

    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(3)
    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(0, 2)
