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

from qldpc import external


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
