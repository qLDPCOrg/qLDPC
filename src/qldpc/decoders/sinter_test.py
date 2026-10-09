# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sinter-decoder package facade."""

from __future__ import annotations

import pytest

from qldpc import decoders

# Deprecated compatibility


def test_deprecated_aliases() -> None:
    """Deprecated Sinter aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="SubgraphSinterDecoder is deprecated"):
        assert decoders.sinter.SubgraphSinterDecoder is decoders.SubgraphDecoder
    with pytest.warns(DeprecationWarning, match="SequentialSinterDecoder is deprecated"):
        assert decoders.sinter.SequentialSinterDecoder is decoders.SequentialWindowDecoder
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert decoders.sinter.Decoder is decoders.ErrorDecoder
