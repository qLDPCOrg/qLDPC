# SPDX-License-Identifier: Apache-2.0

"""Tests for the decoder-construction package facade."""

from __future__ import annotations

from qldpc import decoders
from qldpc.decoders import construction
from qldpc.decoders.construction import resolution, specs
from qldpc.decoders.custom import guf, ilp, lookup
from qldpc.decoders.external import ldpc, pymatching, relay_bp


def test_construction_package_exports() -> None:
    """The facade exports modern construction APIs from their canonical modules."""
    expected = {
        "DecoderSpec": specs.DecoderSpec,
        "DeferredErrorDecoderInput": specs.DeferredErrorDecoderInput,
        "DeferredObservableDecoderInput": specs.DeferredObservableDecoderInput,
        "ErrorDecoderConstructor": specs.ErrorDecoderConstructor,
        "ErrorDecoderInput": specs.ErrorDecoderInput,
        "ObservableDecoderConstructor": specs.ObservableDecoderConstructor,
        "ObservableDecoderInput": specs.ObservableDecoderInput,
        "PcmOrDem": specs.PcmOrDem,
        "bf": specs.bf,
        "bp_lsd": specs.bp_lsd,
        "bp_osd": specs.bp_osd,
        "guf": specs.guf,
        "ilp": specs.ilp,
        "lookup_table": specs.lookup_table,
        "min_sum_bp": specs.min_sum_bp,
        "mwpm": specs.mwpm,
        "relay_bp": specs.relay_bp,
        "decode_observables": resolution.decode_observables,
        "get_error_decoder": resolution.get_error_decoder,
        "get_observable_decoder": resolution.get_observable_decoder,
        "is_prebuilt_decoder": resolution.is_prebuilt_decoder,
        "reject_prebuilt_decoder": resolution.reject_prebuilt_decoder,
        "reject_removed_decoder_args": resolution.reject_removed_decoder_args,
        "resolve_decoder": resolution.resolve_decoder,
        "resolve_observable_decoder": resolution.resolve_observable_decoder,
        "get_decoder_bf": ldpc.get_decoder_bf,
        "get_decoder_bp_lsd": ldpc.get_decoder_bp_lsd,
        "get_decoder_bp_osd": ldpc.get_decoder_bp_osd,
        "get_decoder_guf": guf.get_decoder_guf,
        "get_decoder_ilp": ilp.get_decoder_ilp,
        "get_decoder_lookup": lookup.get_decoder_lookup,
        "get_decoder_mwpm": pymatching.get_decoder_mwpm,
        "get_decoder_rbp": relay_bp.get_decoder_rbp,
        "get_error_decoder_mwpm": pymatching.get_error_decoder_mwpm,
        "get_min_sum_bp_decoder": relay_bp.get_min_sum_bp_decoder,
        "get_observable_decoder_lookup": lookup.get_observable_decoder_lookup,
        "get_observable_decoder_mwpm": pymatching.get_observable_decoder_mwpm,
        "get_relay_bp_decoder": relay_bp.get_relay_bp_decoder,
    }
    assert set(construction.__all__) == set(expected)
    for name, value in expected.items():
        assert getattr(construction, name) is value
        if name in decoders.__all__:
            assert getattr(decoders, name) is value

    assert decoders.DecoderSpec.__module__ == "qldpc.decoders.construction.specs"
    builder_modules = {
        "get_decoder_bf": "qldpc.decoders.external.ldpc",
        "get_decoder_bp_lsd": "qldpc.decoders.external.ldpc",
        "get_decoder_bp_osd": "qldpc.decoders.external.ldpc",
        "get_decoder_guf": "qldpc.decoders.custom.guf",
        "get_decoder_ilp": "qldpc.decoders.custom.ilp",
        "get_decoder_lookup": "qldpc.decoders.custom.lookup",
        "get_decoder_mwpm": "qldpc.decoders.external.pymatching",
        "get_decoder_rbp": "qldpc.decoders.external.relay_bp",
        "get_error_decoder_mwpm": "qldpc.decoders.external.pymatching",
        "get_min_sum_bp_decoder": "qldpc.decoders.external.relay_bp",
        "get_observable_decoder_lookup": "qldpc.decoders.custom.lookup",
        "get_observable_decoder_mwpm": "qldpc.decoders.external.pymatching",
        "get_relay_bp_decoder": "qldpc.decoders.external.relay_bp",
    }
    for name, module_name in builder_modules.items():
        assert getattr(construction, name).__module__ == module_name
    assert not hasattr(construction, "get_decoder")
    assert not hasattr(construction, "decode")
