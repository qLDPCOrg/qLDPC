# SPDX-License-Identifier: Apache-2.0

"""Tests for ldpc decoder builders."""

from __future__ import annotations

import pickle
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import ldpc as ldpc_package
import numpy as np
import pytest
import stim
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


@pytest.mark.parametrize("builder", [get_decoder_bp_osd, get_decoder_bp_lsd, get_decoder_bf])
def test_ldpc_error_channel_compatibility(
    builder: Callable[..., Any],
) -> None:
    """A scalar channel broadcasts, while deprecated and DEM probability inputs are explicit."""
    matrix = np.eye(2, dtype=int)
    scalar_decoder = builder(matrix, error_channel=0.2)
    assert np.array_equal(scalar_decoder.error_channel, [0.2, 0.2])

    with pytest.warns(DeprecationWarning, match="error_rate=0.3.*error_channel=0.3"):
        deprecated_decoder = builder(matrix, error_rate=0.3)
    assert np.array_equal(deprecated_decoder.error_channel, [0.3, 0.3])

    with pytest.raises(ValueError, match="cannot both be specified"):
        builder(matrix, error_rate=0.3, error_channel=0.2)

    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D1")
    for kwargs in ({"error_channel": 0.3}, {"error_rate": 0.3}):
        with pytest.raises(ValueError, match="supplies its own error probabilities"):
            builder(dem, **kwargs)


def test_bp_lsd_random_serial_schedule() -> None:
    """The immediate and deferred BP+LSD builders expose the backend schedule option."""
    matrix = np.eye(2, dtype=int)
    immediate_decoder: Any = get_decoder_bp_lsd(matrix, random_serial_schedule=True)
    deferred_decoder: Any = decoders.bp_lsd(random_serial_schedule=True).build(matrix)
    assert immediate_decoder.random_serial_schedule
    assert deferred_decoder.random_serial_schedule


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


def test_ldpc_import_is_lazy() -> None:
    """Importing the integration does not import ldpc until an adapter is requested."""
    code = """
import sys
import qldpc.decoders.external.ldpc
assert "ldpc" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
