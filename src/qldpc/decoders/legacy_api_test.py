# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the v0.3.3 and v0.4.1 package-root decoder API."""

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


@pytest.mark.parametrize("name", sorted(V033_ROOT_NAMES | V041_ROOT_NAMES))
def test_tagged_root_name_is_importable(name: str) -> None:
    assert name in qldpc.decoders.__all__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert hasattr(qldpc.decoders, name)


def test_tagged_root_inventory() -> None:
    """The v0.3.3 root API is a subset of the v0.4.1 root API."""
    assert len(V033_ROOT_NAMES) == 32
    assert len(V041_ROOT_NAMES) == 90
    assert V033_ROOT_NAMES <= V041_ROOT_NAMES


def test_tagged_root_function_signatures() -> None:
    """Keep the legacy calling forms, even if a shim supplies their implementation."""
    root = qldpc.decoders
    assert list(inspect.signature(root.get_decoder).parameters) == ["pcm_or_dem", "decoder_args"]
    assert list(inspect.signature(root.decode).parameters) == [
        "pcm_or_dem",
        "syndrome",
        "decoder_args",
    ]
    assert inspect.signature(root.get_decoder).parameters["decoder_args"].kind is (
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
        rbp = root.get_decoder_RBP
    assert "error_priors" in inspect.signature(rbp).parameters
    assert list(inspect.signature(root.resolve_decoder).parameters) == [
        "pcm_or_dem",
        "decoder",
        "decoder_args",
        "warn_deprecated",
    ]
    assert list(inspect.signature(root.resolve_observable_decoder).parameters) == [
        "dem",
        "decoder",
        "decoder_args",
        "warn_deprecated",
    ]
