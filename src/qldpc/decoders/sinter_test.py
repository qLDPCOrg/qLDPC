# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sinter-decoder package facade."""

from __future__ import annotations

import importlib
import pickle
import warnings
from typing import Any

import pytest

from qldpc import decoders
from qldpc.decoders import sinter as sinter_adapters
from qldpc.decoders.sinter import core, subgraph, window


def test_sinter_facade_imports_are_canonical() -> None:
    """Facade imports expose the split implementations without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        facade = importlib.reload(sinter_adapters)
        assert facade.SinterDecoder is core.SinterDecoder is decoders.SinterDecoder
        assert facade.CompiledSinterDecoder is core.CompiledSinterDecoder
        assert facade.TrivialDecoder is core.TrivialDecoder
        assert facade.CompiledTrivialDecoder is core.CompiledTrivialDecoder
        assert facade.SubgraphDecoder is subgraph.SubgraphDecoder is decoders.SubgraphDecoder
        assert facade.CompiledSubgraphDecoder is subgraph.CompiledSubgraphDecoder
        assert facade.SequentialWindowDecoder is window.SequentialWindowDecoder
        assert facade.CompiledSequentialWindowDecoder is window.CompiledSequentialWindowDecoder
        assert facade.SlidingWindowDecoder is window.SlidingWindowDecoder


@pytest.mark.parametrize(
    "decoder_class",
    [
        core.DecoderNotCompiledError,
        core.SinterDecoder,
        core.CompiledSinterDecoder,
        core.TrivialDecoder,
        core.CompiledTrivialDecoder,
        subgraph.SubgraphDecoder,
        subgraph.CompiledSubgraphDecoder,
        window.SequentialWindowDecoder,
        window.CompiledSequentialWindowDecoder,
        window.SlidingWindowDecoder,
    ],
)
def test_old_sinter_pickle_paths(decoder_class: Any) -> None:
    """Classes pickled under the v0.3.3 Sinter facade path remain loadable."""
    name = decoder_class.__name__
    payload = pickle.dumps(decoder_class, protocol=0)
    payload = payload.replace(
        f"{decoder_class.__module__}\n{name}".encode(),
        f"qldpc.decoders.sinter\n{name}".encode(),
    )
    restored = pickle.loads(payload)  # noqa: S301 - deliberately constructed compatibility data
    assert restored is decoder_class
    assert pickle.loads(pickle.dumps(decoder_class)) is decoder_class  # noqa: S301


# Deprecated compatibility


def test_deprecated_aliases() -> None:
    """Deprecated Sinter aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="SubgraphSinterDecoder is deprecated"):
        assert decoders.sinter.SubgraphSinterDecoder is decoders.SubgraphDecoder
    with pytest.warns(DeprecationWarning, match="SequentialSinterDecoder is deprecated"):
        assert decoders.sinter.SequentialSinterDecoder is decoders.SequentialWindowDecoder
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert decoders.sinter.Decoder is decoders.ErrorDecoder
