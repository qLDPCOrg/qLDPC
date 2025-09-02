"""Unit tests for decoder.py

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

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders


def test_custom_decoder(pytestconfig: pytest.Config) -> None:
    """Inject custom decoders."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    matrix = np.random.randint(2, size=(2, 2))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder(decoders.Decoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return error

    assert decoders.decode(matrix, syndrome, decoder_constructor=CustomDecoder) is error
    assert decoders.decode(matrix, syndrome, static_decoder=CustomDecoder(matrix)) is error


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    assert np.array_equal(error, decoders.decode(matrix, syndrome))  # with_BP_OSD=True
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BP_LSD=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_RBP="RelayDecoderF32"))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_MWPM=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_ILP=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_GUF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_lookup=True, max_weight=2))
    assert np.array_equal(
        error, decoders.decode(matrix, syndrome, with_weighted_lookup=True, max_weight=2)
    )
