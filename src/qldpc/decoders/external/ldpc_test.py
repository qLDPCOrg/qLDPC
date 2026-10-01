# SPDX-License-Identifier: Apache-2.0

"""Tests for ldpc decoder builders."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import ldpc as ldpc_package
import numpy as np
import pytest
from ldpc import bplsd_decoder

from qldpc import decoders
from qldpc.decoders.external import ldpc as ldpc_integration
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


def test_ldpc_protocol_adapters() -> None:
    """The integration classes are ldpc decoders that satisfy qLDPC's error protocol."""
    assert ldpc_integration.__getattr__("BpOsdDecoder") is ldpc_integration.BpOsdDecoder
    with pytest.raises(AttributeError, match="has no attribute"):
        ldpc_integration.__getattr__("NotAnLdpcDecoder")

    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    adapted_decoders = [
        (
            get_decoder_bp_osd(matrix),
            ldpc_integration.BpOsdDecoder,
            ldpc_package.BpOsdDecoder,
        ),
        (
            get_decoder_bp_lsd(matrix),
            ldpc_integration.BpLsdDecoder,
            bplsd_decoder.BpLsdDecoder,
        ),
        (
            get_decoder_bf(matrix),
            ldpc_integration.BeliefFindDecoder,
            ldpc_package.BeliefFindDecoder,
        ),
    ]
    for decoder, adapter_type, backend_type in adapted_decoders:
        assert isinstance(decoder, adapter_type)
        assert isinstance(decoder, backend_type)
        assert isinstance(decoder, decoders.ErrorDecoder)
        assert pickle.loads(pickle.dumps(adapter_type)) is adapter_type  # noqa: S301
