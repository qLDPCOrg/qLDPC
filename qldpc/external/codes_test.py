"""Unit tests for codes.py

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

import unittest.mock

import pytest

from qldpc import codes, external


def test_get_code() -> None:
    """Retrieve parity check matrix from GAP 4."""
    # extract parity check and finite field
    check = [1, 1]
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=f"\n{check}\nGF(3^3)"),
    ):
        assert external.codes.get_code("") == ([check], 27)

    # fail to find parity checks
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.gap.get_output", return_value=r"\nGF(3^3)"),
        pytest.raises(ValueError, match="Code has no parity checks"),
    ):
        assert external.codes.get_code("")


def test_distance_bound() -> None:
    """Compute a bound on code distance using QDistRnd."""
    with unittest.mock.patch("qldpc.external.gap.require_package", return_value=None):
        with pytest.raises(ValueError, match="non-CSS subsystem codes"):
            external.codes.get_distance_bound(codes.QuditCode(codes.SHYPSCode(2).matrix))

        with unittest.mock.patch("qldpc.external.gap.get_output", return_value="3"):
            assert external.codes.get_distance_bound(codes.FiveQubitCode()) == 3
            assert external.codes.get_distance_bound(codes.SteaneCode()) == 3
