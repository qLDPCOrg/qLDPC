"""Unit tests for retrieval.py.

Copyright 2025 The qLDPC Authors

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

from collections.abc import Callable

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders import retrieval


def test_custom_decoder(pytestconfig: pytest.Config) -> None:
    """Inject custom decoders."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    matrix = np.random.randint(2, size=(2, 2))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder(decoders.Decoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.asarray(error)

    assert decoders.decode(matrix, syndrome, decoder_constructor=CustomDecoder) is error
    assert decoders.decode(matrix, syndrome, static_decoder=CustomDecoder(matrix)) is error

    # injected decoders are validated, which must survive `python -O`
    with pytest.raises(TypeError, match="must be callable"):
        decoders.get_decoder(matrix, decoder_constructor=0)
    with pytest.raises(TypeError, match="callable decode method"):
        decoders.get_decoder(matrix, static_decoder=0)
    with pytest.raises(ValueError, match="cannot process decoding arguments"):
        decoders.get_decoder(matrix, static_decoder=CustomDecoder(matrix), with_BF=True)


def test_decoder_selection() -> None:
    """Exactly one decoder can be requested at a time."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    # a falsy request is consumed rather than passed on to the requested decoder
    assert np.array_equal([1, 1], decoders.decode(matrix, syndrome, with_BF=True, with_MWPM=False))

    with pytest.raises(ValueError, match="Only one decoder"):
        decoders.get_decoder(matrix, with_BF=True, with_MWPM=True)


def test_erasure_bit_request() -> None:
    """A request for an erasure bit is rejected by a decoder that cannot signal erasure."""
    matrix = np.eye(3, 2, dtype=int)

    # a decoder that can signal erasure honours direct and routed requests
    erasing_decoders: list[tuple[Callable[..., decoders.Decoder], dict[str, object]]] = [
        (decoders.get_decoder_RBP, {"with_RBP": True}),
        (decoders.get_decoder_ILP, {"with_ILP": True}),
        (decoders.get_decoder_GUF, {"with_GUF": True}),
        (decoders.get_decoder_lookup, {"with_lookup": True, "max_weight": 2}),
    ]
    for decoder_getter, decoder_args in erasing_decoders:
        direct_args = {"max_weight": 2} if decoder_getter is decoders.get_decoder_lookup else {}
        decoder = decoder_getter(matrix, add_erasure_bit=True, **direct_args)
        assert getattr(decoder, "has_erasure_bit", False)
        decoder = decoders.get_decoder(matrix, add_erasure_bit=True, **decoder_args)
        assert getattr(decoder, "has_erasure_bit", False)

    # every decoder that cannot signal erasure rejects direct and routed requests consistently
    unerasing_decoders: list[tuple[Callable[..., decoders.Decoder], dict[str, object], str]] = [
        (decoders.get_decoder_BF, {"with_BF": True}, "BF"),
        (decoders.get_decoder_BP_OSD, {"with_BP_OSD": True}, "BP_OSD"),
        (decoders.get_decoder_MWPM, {"with_MWPM": True}, "MWPM"),
        (decoders.get_decoder_BP_LSD, {"with_BP_LSD": True}, "BP_LSD"),
    ]
    for decoder_getter, decoder_args, decoder_name in unerasing_decoders:
        with pytest.raises(ValueError, match=rf"The {decoder_name} decoder cannot signal erasure"):
            decoder_getter(matrix, add_erasure_bit=True)
        with pytest.raises(ValueError, match=rf"The {decoder_name} decoder cannot signal erasure"):
            decoders.get_decoder(matrix, add_erasure_bit=True, **decoder_args)
        assert decoder_getter(matrix, add_erasure_bit=False)

    # BP+OSD is the default for a binary matrix
    with pytest.raises(ValueError, match=r"The BP_OSD decoder cannot signal erasure"):
        decoders.get_decoder(matrix, add_erasure_bit=True)


def test_erasure_bit_support_decorator() -> None:
    """A getter declared to support erasure must return a decoder that does so."""

    @retrieval._erasure_bit_support(True)
    def get_decoder_inconsistent(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.Decoder:
        return decoders.get_decoder_BP_OSD(matrix)

    with pytest.raises(ValueError, match=r"The inconsistent decoder cannot signal erasure"):
        get_decoder_inconsistent(np.eye(1, dtype=int), add_erasure_bit=True)


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    assert np.array_equal(error, decoders.decode(matrix, syndrome))  # default, BP+OSD
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BP_LSD=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_BF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_RBP=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_MWPM=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_ILP=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_GUF=True))
    assert np.array_equal(error, decoders.decode(matrix, syndrome, with_lookup=True, max_weight=2))

    # default to GUF with non-binary fields
    field = galois.GF(3)
    matrix = matrix.view(field)
    syndrome = syndrome.view(field)
    error = error.view(field)
    assert np.array_equal(error, decoders.decode(matrix, syndrome))

    # decode from a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    assert np.array_equal(error, decoders.decode(dem, syndrome, with_BP_LSD=True))
    assert np.array_equal(error, decoders.decode(dem, syndrome, with_MWPM=True))
    assert np.array_equal(error, decoders.decode(dem, syndrome, with_ILP=True))
    assert np.array_equal(error, decoders.decode(dem, syndrome, with_GUF=True))

    # a MWPM decoder built from a DEM takes its error weights from that DEM
    with pytest.raises(ValueError, match="Cannot set error weights"):
        decoders.get_decoder(dem, with_MWPM=True, weights=[1.0, 1.0])

    # add a non-graphlike error mechanism, which MWPM can ignore upon request
    matrix = np.hstack([matrix, np.ones((3, 1))])
    error = np.concatenate([error, [0]])
    dem.append("error", 0.125, [stim.DemTarget.relative_detector_id(ii) for ii in range(3)])
    with pytest.raises(ValueError, match="non-graphlike error"):
        np.array_equal(error, decoders.decode(dem, syndrome, with_MWPM=True))
    assert np.array_equal(
        error, decoders.decode(dem, syndrome, with_MWPM=True, ignore_non_graphlike_errors=True)
    )


def test_non_graphlike_over_a_field() -> None:
    """An error is non-graphlike by the number of detectors it addresses, not by their sum."""
    matrix = galois.GF(2)([[1, 1], [1, 0], [1, 0]])  # column 0 addresses three detectors
    syndrome = np.array([1, 0, 0], dtype=int)

    with pytest.raises(ValueError, match="column 0 of the parity check matrix addresses 3"):
        decoders.decode(matrix, syndrome, with_MWPM=True)
    assert np.array_equal(
        [0, 1], decoders.decode(matrix, syndrome, with_MWPM=True, ignore_non_graphlike_errors=True)
    )
