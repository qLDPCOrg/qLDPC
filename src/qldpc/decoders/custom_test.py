# SPDX-License-Identifier: Apache-2.0

"""Tests for the custom-decoder package facade."""

from __future__ import annotations

import pytest

from qldpc import decoders
from qldpc.decoders import custom

# Deprecated compatibility


def test_deprecated_protocol_aliases() -> None:
    """The v0.3.3 protocol aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert custom.Decoder is decoders.ErrorDecoder
    with pytest.warns(DeprecationWarning, match="BatchDecoder is deprecated"):
        assert custom.BatchDecoder is decoders.BatchErrorDecoder


def test_relay_bp_custom_alias() -> None:
    """The external backend remains importable from its old custom-package path."""
    from qldpc.decoders.external.relay_bp import RelayBPDecoder

    with pytest.warns(DeprecationWarning, match=r"use qldpc\.decoders\.external\.relay_bp"):
        assert custom.RelayBPDecoder is RelayBPDecoder
