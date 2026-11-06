# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the v0.3.3 and v0.4.1 package-root decoder API."""

import importlib
import inspect
import warnings

import pytest

import qldpc.decoders

# Exported names from src/qldpc/decoders/__init__.py at the v0.3.3 tag.
V033_ROOT_NAMES = {
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

# Exported names from src/qldpc/decoders/__init__.py at the v0.4.1 tag.
V041_ROOT_NAMES = {
    "BatchDecoder",
    "BatchErrorDecoder",
    "BatchObservableDecoder",
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "CompositeDecoder",
    "Decoder",
    "DecoderInput",
    "DecoderSpec",
    "DeferredDecoderInput",
    "DeferredErrorDecoderInput",
    "DetectorErrorModelArrays",
    "DirectDecoder",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "FlipPattern",
    "FrontierObservableDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableDecoder",
    "ObservableDecoderCompiler",
    "ObservableDecoderConstructor",
    "ObservableLookupDecoder",
    "PcmOrDem",
    "RelayBPDecoder",
    "SequentialSinterDecoder",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "SubgraphSinterDecoder",
    "SupportsDecode",
    "TesseractDecoder",
    "TrivialDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
    "WrappedErrorDecoder",
    "as_error_decoder",
    "batch_decode_errors",
    "bf",
    "bp_lsd",
    "bp_osd",
    "compiles_for_dem",
    "decode",
    "decode_observables",
    "frontier",
    "get_decoder",
    "get_decoder_BF",
    "get_decoder_BP_LSD",
    "get_decoder_BP_OSD",
    "get_decoder_GUF",
    "get_decoder_ILP",
    "get_decoder_MWPM",
    "get_decoder_RBP",
    "get_decoder_bf",
    "get_decoder_bp_lsd",
    "get_decoder_bp_osd",
    "get_decoder_guf",
    "get_decoder_ilp",
    "get_decoder_lookup",
    "get_decoder_mwpm",
    "get_decoder_rbp",
    "get_error_and_erasure",
    "get_error_decoder",
    "get_legacy_decoder_migration_message",
    "get_observable_decoder",
    "guf",
    "ilp",
    "is_prebuilt_decoder",
    "is_prebuilt_observable_decoder",
    "lookup",
    "lookup_table",
    "match_error_decoder_to_dem",
    "min_sum_bp",
    "mwpm",
    "reject_prebuilt_decoder",
    "reject_removed_decoder_args",
    "relay_bp",
    "resolve_decoder",
    "resolve_observable_decoder",
    "supports_batch_decoding",
    "tesseract",
    "tesseract_preset",
    "with_erasure_bits",
}


def test_tagged_root_names_are_importable() -> None:
    """Every root name of a tagged release can be imported from the package root."""
    names = sorted(V033_ROOT_NAMES | V041_ROOT_NAMES)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec(f"from qldpc.decoders import {', '.join(names)}", namespace)  # noqa: S102
    assert set(names) <= namespace.keys()


def test_root_star_import() -> None:
    """A star import of the package root resolves every name in its __all__, without warning."""
    namespace: dict[str, object] = {}
    # a star import is only valid at module level, so execute it in a fresh namespace
    exec("from qldpc.decoders import *", namespace)  # noqa: S102
    assert set(qldpc.decoders.__all__) <= namespace.keys()


@pytest.mark.parametrize(
    "name,qualified_name",
    [
        ("LookupDecoder", "qldpc.decoders.custom.lookup.LookupDecoder"),
        ("RelayBPDecoder", "qldpc.decoders.external.relay_bp.RelayBPDecoder"),
        ("CompiledSubgraphDecoder", "qldpc.decoders.sinter.subgraph.CompiledSubgraphDecoder"),
        ("CompiledSinterDecoder", "qldpc.decoders.sinter.core.CompiledSinterDecoder"),
        ("FlipPattern", "qldpc.decoders.dems.FlipPattern"),
    ],
)
def test_historical_root_class_has_qualified_import(name: str, qualified_name: str) -> None:
    """Historical root classes retain identity while naming their canonical package path."""
    assert name not in qldpc.decoders.__all__
    assert name not in vars(qldpc.decoders)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = getattr(qldpc.decoders, name)
    module, _, class_name = qualified_name.rpartition(".")
    assert alias is getattr(importlib.import_module(module), class_name)
    assert [str(warning.message) for warning in caught] == [
        f"{name} is deprecated; use {qualified_name} instead"
    ]
    assert caught[0].filename == __file__


def test_tagged_root_function_signatures() -> None:
    """Keep the legacy calling forms, even if a shim supplies their implementation."""
    root = qldpc.decoders
    # type checkers reject deprecated names, so retrieve them dynamically
    get_decoder, decode = (getattr(root, name) for name in ["get_decoder", "decode"])
    assert list(inspect.signature(get_decoder).parameters) == ["pcm_or_dem", "decoder_args"]
    assert list(inspect.signature(decode).parameters) == [
        "pcm_or_dem",
        "syndrome",
        "decoder_args",
    ]
    assert inspect.signature(get_decoder).parameters["decoder_args"].kind is (
        inspect.Parameter.VAR_KEYWORD
    )
    for name in [
        "get_decoder_BP_OSD",
        "get_decoder_BP_LSD",
        "get_decoder_BF",
        "get_decoder_MWPM",
        "get_decoder_lookup",
        "get_decoder_GUF",
        "get_decoder_ILP",
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder = getattr(root, name)
        signature = inspect.signature(builder)
        assert "pcm_or_dem" in signature.parameters
        assert signature.parameters["decoder_args"].kind is inspect.Parameter.VAR_KEYWORD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # each uppercase alias names the same builder as its lowercase counterpart
        for suffix in ["BF", "BP_LSD", "BP_OSD", "GUF", "ILP", "MWPM", "RBP"]:
            alias = getattr(root, f"get_decoder_{suffix}")
            assert alias is getattr(root, f"get_decoder_{suffix.lower()}")
    # the last alias, get_decoder_RBP, accepts error priors
    assert "error_priors" in inspect.signature(alias).parameters
    for name, first_parameter in [
        ("resolve_decoder", "pcm_or_dem"),
        ("resolve_observable_decoder", "dem"),
    ]:
        with pytest.warns(DeprecationWarning, match=f"{name} is deprecated"):
            function = getattr(root, name)
        assert list(inspect.signature(function).parameters) == [
            first_parameter,
            "decoder",
            "decoder_args",
            "warn_deprecated",
        ]
