# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lookup.py."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc._deprecations import lookup
from qldpc.decoders.construction._resolution import build_error_decoder, build_observable_decoder


def test_legacy_observable_lookup_decoder() -> None:
    """LookupDecoder(..., predict_observable_flips=True) decodes to observable flips."""
    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0 L1
        error(0.1) D1 L1
    """)
    with pytest.warns(DeprecationWarning, match=r"lookup\(max_weight=\.\.\.\)\.build_observable"):
        decoder = decoders.custom.LookupDecoder(dem, max_weight=2, predict_observable_flips=True)
    assert isinstance(decoder, lookup.LegacyObservableLookupDecoder)

    # each syndrome identifies the error that caused it, and thereby its observable flips
    syndromes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)
    flips = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int)
    _assert_decodes(decoder, syndromes, flips)
    for method in [decoder.decode_batch, decoder.decode_errors_batch]:
        assert np.array_equal(method(syndromes), flips)
    assert isinstance(decoder, decoders.custom.LookupDecoder)

    # a legacy decoder is an observable decoder, rather than an error decoder
    assert build_observable_decoder(dem, decoder=decoder) is decoder
    with pytest.raises(TypeError, match="observable flips rather than errors"):
        build_error_decoder(dem, decoder=decoder)

    # a matrix-based decoder warns with the matrix-based replacement
    with pytest.warns(DeprecationWarning, match=r"ObservableLookupDecoder\(matrix, max_weight="):
        decoder = decoders.custom.LookupDecoder(
            np.array([[1]], dtype=int),
            max_weight=1,
            observable_flip_matrix=np.array([[1]], dtype=int),
            error_channel=[0.1],
            predict_observable_flips=True,
        )
    assert isinstance(decoder, lookup.LegacyObservableLookupDecoder)
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])


def test_legacy_weighted_observable_lookup_decoder() -> None:
    """WeightedLookupDecoder(..., predict_observable_flips=True) decodes to observable flips."""
    # two errors with the same syndrome, of which only the first flips the observable
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0
    """)
    with pytest.warns(DeprecationWarning, match=r"WeightedObservableLookupDecoder\(\.\.\.\)"):
        decoder = decoders.custom.WeightedLookupDecoder(
            dem, max_weight=1, predict_observable_flips=True
        )
    assert isinstance(decoder, lookup.LegacyWeightedObservableLookupDecoder)
    assert isinstance(decoder, decoders.custom.WeightedLookupDecoder)

    # the penalty function selects the error, whose observable flip is the prediction
    syndromes = np.array([[1], [0]], dtype=int)
    for penalty_index, flip in [(0, 0), (1, 1)]:

        def penalty_func(error: npt.NDArray[np.int_], index: int = penalty_index) -> float:
            return float(error[index])

        flips = np.array([[flip], [0]], dtype=int)
        _assert_decodes(decoder, syndromes, flips, penalty_func)
        for method in [decoder.decode_batch, decoder.decode_errors_batch]:
            assert np.array_equal(method(syndromes, penalty_func), flips)


def _assert_decodes(
    decoder: lookup.LegacyObservableLookupDecoder | lookup.LegacyWeightedObservableLookupDecoder,
    syndromes: npt.NDArray[np.int_],
    flips: npt.NDArray[np.int_],
    *args: object,
) -> None:
    """Assert that a legacy decoder and its pickled copy decode syndromes to observable flips."""
    pickled_decoder = pickle.loads(pickle.dumps(decoder))  # noqa: S301
    for copied_decoder in [decoder, pickled_decoder]:
        assert type(copied_decoder) is type(decoder)
        for syndrome, flip in zip(syndromes, flips):
            for method in [copied_decoder.decode, copied_decoder.decode_errors]:
                assert np.array_equal(method(syndrome, *args), flip)


def test_deprecated_direct_construction() -> None:
    """Calling LookupDecoder or WeightedLookupDecoder directly warns and builds an error decoder."""
    matrix = np.eye(2, dtype=int)
    syndrome = np.array([1, 0], dtype=int)
    cases: list[tuple[type[decoders.custom.LookupDecoder], type[decoders.ErrorDecoder]]] = [
        (decoders.custom.LookupDecoder, decoders.custom.ErrorLookupDecoder),
        (decoders.custom.WeightedLookupDecoder, decoders.custom.WeightedErrorLookupDecoder),
    ]
    for base_class, error_class in cases:
        # an explicit predict_observable_flips=False is accepted, and other options are forwarded
        options_and_decoded: list[tuple[dict[str, bool], list[int]]] = [
            ({}, [1, 0]),
            ({"predict_observable_flips": False, "add_erasure_bit": True}, [1, 0, 0]),
        ]
        for options, decoded in options_and_decoded:
            with pytest.warns(DeprecationWarning, match=rf"\b{error_class.__name__}\b") as caught:
                decoder: Any = base_class(matrix, 1, **options)
            assert len(caught) == 1
            assert caught[0].filename == __file__
            assert type(decoder) is error_class
            assert np.array_equal(decoder.decode(syndrome), decoded)


def test_deprecated_user_subclass() -> None:
    """A user-defined subclass of a base lookup decoder names the classes to subclass instead."""

    class CustomLookupDecoder(decoders.custom.LookupDecoder):
        pass

    class CustomWeightedLookupDecoder(decoders.custom.WeightedLookupDecoder):
        pass

    with pytest.raises(TypeError, match=r"\bErrorLookupDecoder\b.*\bObservableLookupDecoder\b"):
        CustomLookupDecoder(np.eye(1, dtype=int), 1)
    with pytest.raises(
        TypeError, match=r"WeightedErrorLookupDecoder.*WeightedObservableLookupDecoder"
    ):
        CustomWeightedLookupDecoder(np.eye(1, dtype=int), 1, predict_observable_flips=True)


def test_deprecated_legacy_attributes() -> None:
    """Legacy observable lookup decoders have deprecated attributes, which warn and are True."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.warns(DeprecationWarning, match="predict_observable_flips=True"):
        legacy_decoders = [
            decoders.custom.LookupDecoder(dem, 1, predict_observable_flips=True),
            decoders.custom.WeightedLookupDecoder(dem, 1, predict_observable_flips=True),
        ]
    for legacy_decoder, class_name in zip(
        legacy_decoders, ["LookupDecoder", "WeightedLookupDecoder"], strict=True
    ):
        for attribute_name in ["predict_observable_flips", "decode_returns_observables"]:
            match = rf"\b{class_name}\.{attribute_name} is DEPRECATED"
            with pytest.warns(DeprecationWarning, match=match) as caught:
                value = getattr(legacy_decoder, attribute_name)
            assert len(caught) == 1
            assert caught[0].filename == __file__
            assert value is True
