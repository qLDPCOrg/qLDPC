# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for the Sinter package facade."""

import importlib
import pickle
import warnings
from typing import Any

import pytest

from qldpc import decoders
from qldpc.decoders import sinter
from qldpc.decoders.sinter import core, subgraph, window

from .. import sinter_test

test_deprecated_aliases = sinter_test.test_deprecated_aliases


def test_sinter_facade_imports_are_canonical() -> None:
    """Old facade imports remain warning-free and expose the split implementations."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        facade = importlib.reload(sinter)
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
    """Classes pickled under the pre-split Sinter facade path remain loadable."""
    name = decoder_class.__name__
    payload = pickle.dumps(decoder_class, protocol=0)
    payload = payload.replace(
        f"{decoder_class.__module__}\n{name}".encode(),
        f"qldpc.decoders.sinter\n{name}".encode(),
    )
    restored = pickle.loads(payload)  # noqa: S301 - deliberately constructed compatibility data
    assert restored is decoder_class
    assert pickle.loads(pickle.dumps(decoder_class)) is decoder_class  # noqa: S301
