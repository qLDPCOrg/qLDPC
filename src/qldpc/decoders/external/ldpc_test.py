# SPDX-License-Identifier: Apache-2.0

"""Tests for ldpc decoder builders."""

from __future__ import annotations

import concurrent.futures
import inspect
import pickle
import sys
import threading
import warnings
from collections.abc import Callable
from typing import Any

import ldpc as ldpc_package
import numpy as np
import pytest
import stim
from ldpc import bplsd_decoder

from qldpc import decoders
from qldpc.conftest import run_python
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


def test_ldpc_builder_error_rate_is_last() -> None:
    """Builders keep error_rate as their last parameter."""
    builders: list[Callable[..., decoders.ErrorDecoder]] = [
        _get_decoder_bp_osd,
        _get_decoder_bp_lsd,
        _get_decoder_bf,
    ]
    for builder in builders:
        assert list(inspect.signature(builder).parameters)[-1] == "error_rate"


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
    decoder: Any
    for decoder in [builder(matrix, max_iter=4), helper(max_iter=4).build(matrix)]:
        assert decoder.max_iter == 4  # the default for this matrix is 2

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
        assert isinstance(decoder, decoders.DetailedErrorDecoder)
        detailed = decoder.decode_errors_detailed(np.array([1, 0], dtype=int))
        assert np.array_equal(detailed.error, decoder.decode(np.array([1, 0], dtype=int)))
        assert pickle.loads(pickle.dumps(adapter_type)) is adapter_type  # noqa: S301

    class NumpyScalarDiagnostics(ldpc_integration._DetailedLdpcDecoderMixin):
        converge = np.bool_(True)
        iterations = np.int64(3)

        def __init__(self) -> None:
            self.log_prob_ratios = np.zeros(2)

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            self.log_prob_ratios[:] = syndrome
            self.iterations = np.int64(syndrome[0] + 1)
            return syndrome.copy()

    decoder = NumpyScalarDiagnostics()
    first = decoder.decode_errors_detailed(np.array([1, 0], dtype=int))
    first_ratios = first.diagnostics["ldpc.log_prob_ratios"]
    assert isinstance(first_ratios, np.ndarray)
    assert first.diagnostics["ldpc.converge"] is True
    assert first.diagnostics["ldpc.iterations"] == 2
    assert np.array_equal(first_ratios, [1, 0])
    second = decoder.decode_errors_detailed(np.array([0, 1], dtype=int))
    second_ratios = second.diagnostics["ldpc.log_prob_ratios"]
    assert isinstance(second_ratios, np.ndarray)
    assert second.diagnostics["ldpc.iterations"] == 1
    assert np.array_equal(second_ratios, decoder.log_prob_ratios)
    assert np.array_equal(first_ratios, [1, 0])


def test_ldpc_protocol_adapters_are_created_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent first uses share one set of adapter classes, as pickling requires."""
    # create the original classes, which monkeypatch restores after the test
    ldpc_integration._get_backend_class("BpOsdDecoder")
    num_threads = 8

    def get_adapter_type(barrier: threading.Barrier) -> type[Any]:
        barrier.wait()
        return ldpc_integration._get_backend_class("BpOsdDecoder")

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads often, so that an unguarded race is likely
    try:
        for _ in range(5):
            monkeypatch.setattr(ldpc_integration, "_BACKEND_CLASSES", None)
            for name in ("BeliefFindDecoder", "BpLsdDecoder", "BpOsdDecoder"):
                monkeypatch.delattr(ldpc_integration, name)
            barriers = [threading.Barrier(num_threads)] * num_threads
            with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
                adapter_types = set(executor.map(get_adapter_type, barriers))
            assert adapter_types == {ldpc_integration.BpOsdDecoder}
    finally:
        sys.setswitchinterval(switch_interval)


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


def test_ldpc_import_is_lazy() -> None:
    """Importing the integration does not import ldpc until an adapter is requested."""
    code = """
import sys
import qldpc.decoders.external.ldpc
assert "ldpc" not in sys.modules
"""
    run_python(code)
