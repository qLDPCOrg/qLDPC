# SPDX-License-Identifier: Apache-2.0

"""Tests for ldpc decoder builders."""

from __future__ import annotations

import pickle
import subprocess
import sys
import unittest.mock
import warnings
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
    _get_decoder_bf,
    _get_decoder_bp_lsd,
    _get_decoder_bp_osd,
)


@pytest.mark.parametrize("builder", [_get_decoder_bp_osd, _get_decoder_bp_lsd, _get_decoder_bf])
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
        (_get_decoder_bf, "BF"),
        (_get_decoder_bp_osd, "BP_OSD"),
        (_get_decoder_bp_lsd, "BP_LSD"),
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


@pytest.mark.parametrize("builder", [_get_decoder_bp_osd, _get_decoder_bp_lsd, _get_decoder_bf])
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
        builder(matrix, error_channel=0.2, error_rate=0.3)

    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D1")
    for kwargs in ({"error_channel": 0.3}, {"error_rate": 0.3}):
        with pytest.raises(ValueError, match="supplies its own error probabilities"):
            builder(dem, **kwargs)


def test_bp_lsd_random_serial_schedule() -> None:
    """The immediate and deferred BP+LSD builders expose the backend schedule option."""
    matrix = np.eye(2, dtype=int)
    immediate_decoder: Any = _get_decoder_bp_lsd(matrix, random_serial_schedule=True)
    deferred_decoder: Any = decoders.bp_lsd(random_serial_schedule=True).build(matrix)
    assert immediate_decoder.random_serial_schedule
    assert deferred_decoder.random_serial_schedule


@pytest.mark.parametrize(
    ("builder", "helper"),
    [
        (_get_decoder_bp_osd, decoders.bp_osd),
        (_get_decoder_bp_lsd, decoders.bp_lsd),
        (_get_decoder_bf, decoders.bf),
    ],
)
def test_ldpc_backend_options(
    builder: Callable[..., decoders.ErrorDecoder],
    helper: Callable[..., decoders.DecoderSpec[decoders.ErrorDecoder]],
) -> None:
    """Named options and backend_options reach the underlying ldpc constructor."""
    matrix = np.eye(2, dtype=int)
    with unittest.mock.patch.object(ldpc_integration, "_build_ldpc_decoder") as backend:
        builder(matrix, max_iter=4, backend_options={"input_vector_type": "syndrome"})
        assert backend.call_args.args[3]["max_iter"] == 4
        assert backend.call_args.args[3]["input_vector_type"] == "syndrome"
        spec = helper(max_iter=4, backend_options={"input_vector_type": "syndrome"})
        assert spec.options["backend_options"] == {"input_vector_type": "syndrome"}
        assert repr(spec).endswith(
            "(max_iter=4, backend_options={'input_vector_type': 'syndrome'})"
        )
        assert pickle.loads(pickle.dumps(spec)).options == spec.options  # noqa: S301
        spec.build(matrix)
        assert backend.call_args.args[3]["input_vector_type"] == "syndrome"

    # an empty mapping is equivalent to no backend options
    assert helper(backend_options={}).options["backend_options"] is None

    # misspelled named options are rejected when the settings are created
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_itr'"):
        helper(max_itr=4)
    with pytest.raises(ValueError, match="lists max_iter by name, so pass it directly"):
        helper(backend_options={"max_iter": 4})
    with pytest.raises(TypeError, match="backend_options must be a mapping"):
        helper(backend_options=[("backend_extension", 12)])

    # the backend rejects unsupported names when the decoder is built, except that BP+LSD silently
    # ignores them, so qLDPC warns about names it does not know
    spec = helper(backend_options={"unsupported_backend_option": True})
    if helper is decoders.bp_lsd:
        with pytest.warns(UserWarning, match=r"\['unsupported_backend_option'\]") as caught:
            spec.build(matrix)
        assert caught[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            helper(backend_options={"input_vector_type": "syndrome"}).build(matrix)
            # osd_method and osd_order are aliases that BP+LSD honors
            aliased: Any = helper(backend_options={"osd_order": 2, "osd_method": "LSD_E"}).build(
                matrix
            )
            assert (aliased.lsd_order, aliased.lsd_method) == (2, "LSD_E")
    else:
        with pytest.raises((TypeError, ValueError), match="unsupported_backend_option"):
            spec.build(matrix)


def test_ldpc_protocol_adapters() -> None:
    """The integration classes are ldpc decoders that satisfy qLDPC's error protocol."""
    assert ldpc_integration.__getattr__("BpOsdDecoder") is ldpc_integration.BpOsdDecoder
    with pytest.raises(AttributeError, match="has no attribute"):
        ldpc_integration.__getattr__("NotAnLdpcDecoder")

    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    adapted_decoders = [
        (
            _get_decoder_bp_osd(matrix),
            ldpc_integration.BpOsdDecoder,
            ldpc_package.BpOsdDecoder,
        ),
        (
            _get_decoder_bp_lsd(matrix),
            ldpc_integration.BpLsdDecoder,
            bplsd_decoder.BpLsdDecoder,
        ),
        (
            _get_decoder_bf(matrix),
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
