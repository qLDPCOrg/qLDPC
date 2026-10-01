# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deprecated keyword-based decoder construction."""

from __future__ import annotations

import re
import warnings

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.construction import legacy


def test_decoder_selection() -> None:
    """Exactly one decoder can be requested at a time."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    # a falsy request is consumed rather than passed on to the requested decoder
    with pytest.warns(DeprecationWarning, match=r"decoders\.bf\(\.\.\.\)\.build"):
        decoded_error = decoders.decode(matrix, syndrome, with_BF=True, with_MWPM=False)
    assert np.array_equal([1, 1], decoded_error)

    with (
        pytest.warns(DeprecationWarning, match="get_decoder is deprecated"),
        pytest.raises(ValueError, match="Only one decoder"),
    ):
        decoders.get_decoder(matrix, with_BF=True, with_MWPM=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoders.decode(matrix, syndrome, with_BF=True)
    assert caught[0].filename == __file__
    assert "decoders.bf(...).build(pcm_or_dem)" in str(caught[0].message)

    with pytest.warns(DeprecationWarning, match=r"decoders\.bp_osd\(\.\.\.\)\.build"):
        decoders.get_decoder(matrix, max_iter=1)

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = decoders.get_decoder(
            dem,
            with_lookup=True,
            max_weight=1,
            predict_observable_flips=True,
        )
    messages = [str(warning.message) for warning in caught]
    assert any(
        "construct an ObservableLookupDecoder" in message and "decode_observables" in message
        for message in messages
    )
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])


def test_deprecated_decoder_functions(pytestconfig: pytest.Config) -> None:
    """The deprecated get_decoder and decode functions behave as they did, and name replacements."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    matrix = np.random.randint(2, size=(3, 4))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder:
        def __init__(self, matrix: npt.NDArray[np.int_], scale: int = 1) -> None:
            self.scale = scale

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.scale * np.asarray(error)

    # warnings name the replacing call, and point at the caller
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoders.get_decoder(matrix)
        decoders.get_decoder(matrix, decoder_constructor=CustomDecoder, scale=2)
        decoders.decode(matrix, syndrome, with_lookup=True, max_weight=1)
    assert [str(warning.message) for warning in caught] == [
        "decoders.get_decoder is deprecated; use decoders.get_error_decoder(pcm_or_dem) instead",
        "decoders.get_decoder is deprecated; use CustomDecoder(pcm_or_dem, ...) instead",
        (
            "decoders.decode is deprecated; use"
            " decoders.lookup_table(...).build(pcm_or_dem).decode(syndrome) instead"
        ),
    ]
    assert all(warning.filename == __file__ for warning in caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # a decoder constructor receives the remaining arguments, and is returned as is
        decoder = decoders.get_decoder(matrix, decoder_constructor=CustomDecoder, scale=2)
        assert isinstance(decoder, CustomDecoder) and decoder.scale == 2
        with pytest.raises(TypeError, match="must be callable"):
            decoders.get_decoder(matrix, decoder_constructor=0)

        # a static decoder is returned as is, and admits no other arguments
        static_decoder = CustomDecoder(matrix)
        assert decoders.get_decoder(matrix, static_decoder=static_decoder) is static_decoder
        assert decoders.get_decoder(matrix, static_decoder=CustomDecoder) is CustomDecoder
        with pytest.raises(ValueError, match="cannot process decoding arguments"):
            decoders.get_decoder(matrix, static_decoder=static_decoder, with_BF=True)

        # the default decoder depends on the field
        decoder = decoders.get_decoder(galois.GF(3)(matrix))
        assert isinstance(decoder, decoders.GUFDecoder)

    with pytest.warns(DeprecationWarning, match=r"use static_decoder\.decode\(syndrome\)"):
        decoded_error = decoders.decode(matrix, syndrome, static_decoder=static_decoder)
    assert np.array_equal(decoded_error, error)

    # the static_decoder argument has been removed from other methods, in favor of decoder=
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.resolve_decoder(matrix, None, {"static_decoder": static_decoder})
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.resolve_observable_decoder(dem, None, {"static_decoder": static_decoder})
    with pytest.raises(ValueError, match="Cannot combine decoder"):
        decoders.resolve_decoder(matrix, static_decoder, {"with_BF": True})


def test_legacy_decoder_migration_messages() -> None:
    """Deprecated keyword arguments of methods warn with the decoder input that replaces them."""
    matrix = np.eye(2, dtype=int)
    expected_messages: list[tuple[dict[str, object], str]] = [
        ({"decoder_constructor": decoders.LookupDecoder}, "for example decoder=LookupDecoder"),
        ({"with_lookup": True, "predict_observable_flips": True}, "ObservableLookupDecoder"),
        ({"with_BF": True}, r"with_BF keyword .* use decoder=decoders\.bf\(\.\.\.\)"),
        ({"with_BF": True, "with_MWPM": True}, "pass exactly one"),
        ({"max_iter": 5}, r"move them into decoder=decoders\.bp_osd\(\.\.\.\)"),
    ]
    for decoder_args, expected_message in expected_messages:
        message = legacy.get_legacy_decoder_migration_message(matrix, decoder_args)
        assert re.search(expected_message, message), message
    message = legacy.get_legacy_decoder_migration_message(
        galois.GF(3)(matrix), {"max_weight": 1}, argument_name="decoder_x"
    )
    assert "decoder_x=decoders.guf(...)" in message

    # deprecated arguments build a decoder, warning unless an outer API has already warned
    with pytest.warns(DeprecationWarning, match="with_BF keyword"):
        decoder = decoders.resolve_decoder(matrix, None, {"with_BF": True})
    assert np.array_equal(decoder.decode_errors(np.array([1, 0])), [1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.resolve_decoder(matrix, None, {"with_BF": True}, warn_deprecated=False)
        decoders.resolve_decoder(matrix, decoders.bf(), {})
