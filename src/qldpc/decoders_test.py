# SPDX-License-Identifier: Apache-2.0

"""Tests for the package-root decoder facade."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

from qldpc import decoders
from qldpc.decoders import adapters, construction, custom, sinter

V0_3_3_ROOT_EXPORTS = {
    "BatchDecoder",
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "CompositeDecoder",
    "Decoder",
    "DetectorErrorModelArrays",
    "DirectDecoder",
    "FlipPattern",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "RelayBPDecoder",
    "SequentialSinterDecoder",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "SubgraphSinterDecoder",
    "TrivialDecoder",
    "WeightedLookupDecoder",
    "decode",
    "get_decoder",
    "get_decoder_BF",
    "get_decoder_BP_LSD",
    "get_decoder_BP_OSD",
    "get_decoder_GUF",
    "get_decoder_ILP",
    "get_decoder_MWPM",
    "get_decoder_RBP",
    "get_decoder_lookup",
}


def test_v0_3_3_root_exports_are_preserved() -> None:
    """Every package-root name exported by v0.3.3 remains public."""
    assert V0_3_3_ROOT_EXPORTS <= set(decoders.__all__)

    expected = {
        "CompiledSequentialWindowDecoder": sinter.CompiledSequentialWindowDecoder,
        "CompiledSinterDecoder": sinter.CompiledSinterDecoder,
        "CompiledSubgraphDecoder": sinter.CompiledSubgraphDecoder,
        "CompiledTrivialDecoder": sinter.CompiledTrivialDecoder,
        "CompositeDecoder": custom.CompositeDecoder,
        "DetectorErrorModelArrays": decoders.DetectorErrorModelArrays,
        "DirectDecoder": custom.DirectDecoder,
        "FlipPattern": decoders.FlipPattern,
        "GUFDecoder": custom.GUFDecoder,
        "ILPDecoder": custom.ILPDecoder,
        "LookupDecoder": decoders.LookupDecoder,
        "RelayBPDecoder": custom.RelayBPDecoder,
        "SequentialWindowDecoder": sinter.SequentialWindowDecoder,
        "SinterDecoder": sinter.SinterDecoder,
        "SlidingWindowDecoder": sinter.SlidingWindowDecoder,
        "SubgraphDecoder": sinter.SubgraphDecoder,
        "TrivialDecoder": sinter.TrivialDecoder,
        "WeightedLookupDecoder": decoders.WeightedLookupDecoder,
        "decode": decoders.decode,
        "get_decoder": decoders.get_decoder,
        "get_decoder_lookup": construction.get_decoder_lookup,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name, value in expected.items():
            assert getattr(decoders, name) is value


def test_current_root_facade_exports_are_canonical() -> None:
    """Current root APIs resolve to their owning package implementations."""
    assert decoders.DecoderSpec is construction.DecoderSpec
    assert decoders.bp_osd is construction.bp_osd
    assert decoders.get_error_decoder is construction.get_error_decoder
    assert decoders.resolve_observable_decoder is construction.resolve_observable_decoder
    assert decoders.ErrorsToObservablesDecoder is adapters.ErrorsToObservablesDecoder
    assert decoders.CompositeDecoder is custom.CompositeDecoder
    assert decoders.SinterDecoder is sinter.SinterDecoder

    missing_name = "NotADecoder"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(decoders, missing_name)


def test_optional_decoder_dependencies_are_lazy() -> None:
    """Importing qldpc does not import optional decoder backends."""
    code = """
import sys
import qldpc
import qldpc.decoders.adapters
import qldpc.decoders.construction
import qldpc.decoders.external
assert not {"ldpc", "pymatching", "relay_bp"} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


# Deprecated compatibility


def test_v0_3_3_deprecated_class_aliases() -> None:
    """Deprecated v0.3.3 class aliases warn and resolve to current classes."""
    expected = {
        "BatchDecoder": decoders.BatchErrorDecoder,
        "Decoder": decoders.ErrorDecoder,
        "SequentialSinterDecoder": decoders.SequentialWindowDecoder,
        "SubgraphSinterDecoder": decoders.SubgraphDecoder,
    }
    for name, replacement in expected.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(decoders, name) is replacement
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert replacement.__name__ in str(caught[0].message)


def test_v0_3_3_deprecated_builder_aliases() -> None:
    """Uppercase v0.3.3 builders warn and resolve to lowercase builders."""
    for suffix in ["BF", "BP_LSD", "BP_OSD", "GUF", "ILP", "MWPM", "RBP"]:
        replacement = getattr(construction, f"get_decoder_{suffix.lower()}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert getattr(decoders, f"get_decoder_{suffix}") is replacement
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert replacement.__name__ in str(caught[0].message)


def test_v0_3_3_legacy_construction_functions_work() -> None:
    """The root get_decoder and decode functions retain their v0.3.3 behavior."""
    matrix = np.array([[1, 1]], dtype=int)
    syndrome = np.array([1], dtype=int)
    with pytest.warns(DeprecationWarning, match="get_decoder is deprecated"):
        decoder = decoders.get_decoder(matrix, with_lookup=True, max_weight=1)
    assert np.array_equal(matrix @ decoder.decode(syndrome) % 2, syndrome)

    with pytest.warns(DeprecationWarning, match="decode is deprecated"):
        error = decoders.decode(matrix, syndrome, with_lookup=True, max_weight=1)
    assert np.array_equal(matrix @ error % 2, syndrome)


def test_root_star_import_names_resolve() -> None:
    """Every name advertised by the root facade can be retrieved."""
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        star_imports = {name: getattr(decoders, name) for name in decoders.__all__}
    assert star_imports["SubgraphSinterDecoder"] is decoders.SubgraphDecoder
    assert star_imports["Decoder"] is decoders.ErrorDecoder
