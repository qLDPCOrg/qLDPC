# SPDX-License-Identifier: Apache-2.0

"""Tests for ldpc decoder builders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from qldpc import decoders
from qldpc.decoders.external.ldpc import (
    get_decoder_bf,
    get_decoder_bp_lsd,
    get_decoder_bp_osd,
)


@pytest.mark.parametrize("builder", [get_decoder_bp_osd, get_decoder_bp_lsd, get_decoder_bf])
def test_ldpc_builders(
    builder: Callable[..., decoders.ErrorDecoder],
) -> None:
    """Each ldpc builder decodes matrices and detector error models."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    dem = decoders.DetectorErrorModelArrays.from_arrays(
        matrix, None, np.array([0.1, 0.2, 0.3])
    ).to_dem()
    error = np.array([1, 0, 0])
    syndrome = matrix @ error % 2

    for pcm_or_dem in [matrix, dem]:
        decoded = np.asarray(builder(pcm_or_dem).decode(syndrome), dtype=int)
        assert decoded.shape == error.shape
        assert np.array_equal(matrix @ decoded % 2, syndrome)


@pytest.mark.parametrize(
    ("builder", "name"),
    [
        (get_decoder_bf, "BF"),
        (get_decoder_bp_osd, "BP_OSD"),
        (get_decoder_bp_lsd, "BP_LSD"),
    ],
)
def test_ldpc_builders_reject_erasure(
    builder: Callable[..., decoders.ErrorDecoder], name: str
) -> None:
    """ldpc decoders cannot signal erasure."""
    matrix = np.eye(2, dtype=int)
    with pytest.raises(ValueError, match=rf"The {name} decoder cannot signal erasure"):
        builder(matrix, add_erasure_bit=True)
    assert builder(matrix, add_erasure_bit=False)
