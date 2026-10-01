# SPDX-License-Identifier: Apache-2.0

"""Tests for the custom-decoder package facade."""

from __future__ import annotations

import pickle

import pytest

from qldpc import decoders
from qldpc.decoders import custom


def test_custom_package_exports() -> None:
    """The package facade exposes every custom implementation class."""
    assert custom.CompositeDecoder is decoders.CompositeDecoder
    assert custom.DirectDecoder is decoders.DirectDecoder
    assert custom.GUFDecoder is decoders.GUFDecoder
    assert custom.ILPDecoder is decoders.ILPDecoder
    assert custom.LookupDecoder is decoders.LookupDecoder
    assert custom.ObservableLookupDecoder is decoders.ObservableLookupDecoder
    assert custom.RelayBPDecoder is decoders.RelayBPDecoder
    assert custom.WeightedLookupDecoder is decoders.WeightedLookupDecoder
    assert custom.WeightedObservableLookupDecoder is decoders.WeightedObservableLookupDecoder
    assert custom.PLACEHOLDER_ERROR_RATE == 1e-3


@pytest.mark.parametrize(
    ("decoder_class", "module_name"),
    [
        (decoders.CompositeDecoder, "qldpc.decoders.custom.composition"),
        (decoders.DirectDecoder, "qldpc.decoders.custom.composition"),
        (decoders.GUFDecoder, "qldpc.decoders.custom.guf"),
        (decoders.ILPDecoder, "qldpc.decoders.custom.ilp"),
        (decoders.LookupDecoder, "qldpc.decoders.custom.lookup"),
        (decoders.ObservableLookupDecoder, "qldpc.decoders.custom.lookup"),
        (decoders.RelayBPDecoder, "qldpc.decoders.external.relay_bp"),
        (decoders.WeightedLookupDecoder, "qldpc.decoders.custom.lookup"),
        (decoders.WeightedObservableLookupDecoder, "qldpc.decoders.custom.lookup"),
    ],
)
def test_custom_class_pickle_paths(decoder_class: type, module_name: str) -> None:
    """Custom classes use canonical paths and remain loadable from the v0.3.3 facade path."""
    assert decoder_class.__module__ == module_name
    name = decoder_class.__name__
    payload = pickle.dumps(decoder_class, protocol=0)
    old_payload = payload.replace(
        f"{module_name}\n{name}".encode(),
        f"qldpc.decoders.custom\n{name}".encode(),
    )
    assert pickle.loads(old_payload) is decoder_class  # noqa: S301 - compatibility payload
    assert pickle.loads(payload) is decoder_class  # noqa: S301 - trusted round trip


# Deprecated compatibility


def test_deprecated_protocol_aliases() -> None:
    """The v0.3.3 protocol aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert custom.Decoder is decoders.ErrorDecoder
    with pytest.warns(DeprecationWarning, match="BatchDecoder is deprecated"):
        assert custom.BatchDecoder is decoders.BatchErrorDecoder
