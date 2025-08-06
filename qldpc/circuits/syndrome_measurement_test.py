"""Unit tests for syndrome_measurement.py

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

import pytest
import stim

from qldpc import circuits


def test_measurement_record() -> None:
    """Build and use a MeasurementRecord."""
    record = circuits.MeasurementRecord()
    record.append({0: [0, 1], 2: [2]})
    assert record.num_measurements == 3
    assert dict(record.items()) == record.qubit_to_measurement
    assert record.get_last_target_rec(2) == stim.target_rec(-1)
    assert record.get_last_target_rec(0) == stim.target_rec(-2)
    with pytest.raises(ValueError, match="Qubit 1 not found"):
        record.get_last_target_rec(1)
