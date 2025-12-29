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
    qubit_ids = circuits.QubitIDs.from_code(code, num_ancillas=2)
    data_ids, check_ids, ancilla_ids = qubit_ids
    assert data_ids == tuple(range(len(code)))
    assert check_ids == tuple(range(len(code), len(code) + code.num_checks))
    assert ancilla_ids == tuple(range(len(code) + code.num_checks, len(code) + code.num_checks + 2))

    qubit_ids.add_ancillas(3)
    assert qubit_ids.ancilla == tuple(
        range(len(code) + code.num_checks, len(code) + code.num_checks + 5)
    )

    qubit_ids.shift(3)
    assert qubit_ids.data == tuple(qq + 3 for qq in data_ids)

    assert qubit_ids == circuits.QubitIDs.validated(qubit_ids, code)
    with pytest.raises(ValueError, match="invalid for the given code"):
        circuits.QubitIDs.validated(circuits.QubitIDs((), (), ()), code)


def test_records() -> None:
    """Measurement and detector records."""
    base_record = circuits.Record({0: [0]})
    assert base_record.num_events == 1
    assert str(base_record) == "Record({0: [0]})"

    base_record = base_record + circuits.Record({0: [1], 2: [0]})
    assert base_record.num_events == 3
    base_record += {1: [0, 1]}
    base_record.append({1: [0, 1]}, repeat=2)
    assert base_record[1] == [3, 4, 5, 6, 7, 8]
    assert len(base_record) == 3
    assert list(iter(base_record)) == list(base_record.keys())
    assert dict(base_record.items()) == base_record.key_to_events

    measurement_record = circuits.MeasurementRecord(base_record.key_to_events)
    assert measurement_record.num_events == 9
    assert measurement_record.get_target_rec(2) == stim.target_rec(-8)
    assert measurement_record.get_target_rec(0) == stim.target_rec(-7)
    assert measurement_record.get_target_rec(0, -2) == stim.target_rec(-9)

    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(3)
    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(0, 2)

    detector_record = circuits.DetectorRecord(base_record.key_to_events)
    assert detector_record.num_events == 9
    assert detector_record.get_detector(2) == 1
    assert detector_record.get_detector(0) == 2
    assert detector_record.get_detector(0, -2) == 0

    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(3)
    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(0, 2)

    for record in [base_record, measurement_record, detector_record]:
        record_copy = record.copy()
        assert isinstance(record_copy, type(record))
        assert list(record_copy.items()) == list(record.items())


def test_post_selection() -> None:
    """Update a DetectorRecord after post-selecting on some detectors."""
    record = circuits.DetectorRecord({"flag": [0, 2, 4], "a": [1, 5], "b": [3]})
    expected_record = circuits.DetectorRecord({"a": [0, 2], "b": [1]})
    assert record.after_post_selection("flag") == expected_record
