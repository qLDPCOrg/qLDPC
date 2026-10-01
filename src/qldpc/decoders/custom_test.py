# SPDX-License-Identifier: Apache-2.0

"""Tests for the custom-decoder package facade."""

from __future__ import annotations

import pickle

import pytest

from qldpc import decoders
from qldpc.decoders import custom


@pytest.mark.parametrize(
    "decoder_class",
    [
        decoders.CompositeDecoder,
        decoders.DirectDecoder,
        decoders.GUFDecoder,
        decoders.ILPDecoder,
        decoders.LookupDecoder,
        decoders.RelayBPDecoder,
        decoders.WeightedLookupDecoder,
    ],
)
def test_v0_3_3_custom_class_pickle_paths(decoder_class: type) -> None:
    """Classes remain loadable from their v0.3.3 custom-module pickle paths."""
    name = decoder_class.__name__
    payload = pickle.dumps(decoder_class, protocol=0)
    old_payload = payload.replace(
        f"{decoder_class.__module__}\n{name}".encode(),
        f"qldpc.decoders.custom\n{name}".encode(),
    )
    assert pickle.loads(old_payload) is decoder_class  # noqa: S301 - compatibility payload


# Deprecated compatibility


def test_deprecated_protocol_aliases() -> None:
    """The v0.3.3 protocol aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert custom.Decoder is decoders.ErrorDecoder
    with pytest.warns(DeprecationWarning, match="BatchDecoder is deprecated"):
        assert custom.BatchDecoder is decoders.BatchErrorDecoder
