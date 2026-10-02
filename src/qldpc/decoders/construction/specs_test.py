# SPDX-License-Identifier: Apache-2.0

"""Unit tests for typed decoder specifications."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from typing import Any, Never

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.adapters import error_decoders
from qldpc.decoders.custom.lookup import get_observable_decoder_lookup


def _uniform_binary_error_channel(error: npt.NDArray[np.int_] | Sequence[int]) -> float:
    """Return the log probability of a uniformly random binary error."""
    return -float(np.size(error) * np.log(2))


def test_decoder_specs_store_public_builders() -> None:
    """Every callable stored by a DecoderSpec has a stable public construction path."""
    specs = [
        decoders.bp_osd(),
        decoders.bp_lsd(),
        decoders.bf(),
        decoders.mwpm(),
        decoders.frontier(),
        decoders.relay_bp(),
        decoders.min_sum_bp(),
        decoders.tesseract(),
        decoders.lookup_table(1),
        decoders.ilp(),
        decoders.guf(),
    ]
    expected_modules = {
        "bp_osd": "qldpc.decoders.external.ldpc",
        "bp_lsd": "qldpc.decoders.external.ldpc",
        "bf": "qldpc.decoders.external.ldpc",
        "mwpm": "qldpc.decoders.external.pymatching",
        "frontier": "qldpc.decoders.external.frontier",
        "relay_bp": "qldpc.decoders.external.relay_bp",
        "min_sum_bp": "qldpc.decoders.external.relay_bp",
        "tesseract": "qldpc.decoders.external.tesseract",
        "lookup_table": "qldpc.decoders.custom.lookup",
        "ilp": "qldpc.decoders.custom.ilp",
        "guf": "qldpc.decoders.custom.guf",
    }
    for spec in specs:
        for builder in (spec._builder, spec._observable_builder):
            if builder is None:
                continue
            assert builder.__module__ == expected_modules[spec._helper_name]
            assert not builder.__name__.startswith("_")

        restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted round trip
        assert restored.options == spec.options
        assert restored._builder is spec._builder
        assert restored._observable_builder is spec._observable_builder


def test_decoder_spec_observable_modes() -> None:
    """Specs expose native observable construction and error-conversion fallback."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    native_spec = decoders.lookup_table(max_weight=1)
    assert native_spec.predicts_observables_natively
    assert isinstance(native_spec.build_observable_decoder(dem), decoders.ObservableLookupDecoder)

    converted_spec = decoders.guf()
    assert not converted_spec.predicts_observables_natively
    assert isinstance(
        converted_spec.build_observable_decoder(dem),
        error_decoders.ErrorsToObservablesDecoder,
    )
    assert decoders.tesseract().predicts_observables_natively


def test_decoder_specs() -> None:
    """Typed decoder specs defer construction and survive process serialization."""
    matrix = np.eye(2, dtype=int)
    syndrome = np.array([1, 0], dtype=int)

    spec = decoders.lookup_table(max_weight=1, error_channel=_uniform_binary_error_channel)
    assert spec.options["error_channel"] is _uniform_binary_error_channel
    restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted in-memory round trip
    assert np.array_equal(
        decoders.get_error_decoder(matrix, decoder=restored).decode(syndrome), syndrome
    )

    # a spec displays the helper call that reproduces it, omitting default options
    assert repr(decoders.bp_osd()) == "decoders.bp_osd()"
    assert repr(decoders.bp_lsd(max_iter=30, bp_method="ms")) == (
        "decoders.bp_lsd(max_iter=30, bp_method='ms')"
    )
    assert repr(decoders.lookup_table(2)) == "decoders.lookup_table(max_weight=2)"
    assert repr(decoders.relay_bp(gamma0=0.2)) == "decoders.relay_bp(gamma0=0.2)"
    assert repr(decoders.tesseract(det_beam=7)) == "decoders.tesseract(det_beam=7)"
    assert repr(decoders.ilp(verbose=False)) == "decoders.ilp(verbose=False)"
    assert repr(decoders.guf(max_weight=1)) == "decoders.guf(max_weight=1)"
    channel = np.array([0.1, 0.2])
    assert "error_channel=array" in repr(decoders.bf(error_channel=channel))

    # a spec exposes a copy of its options, which cannot modify the spec
    spec = decoders.lookup_table(max_weight=2)
    spec.options["max_weight"] = 3
    assert spec.options["max_weight"] == 2

    # deprecated lookup penalties remain available through deferred construction
    legacy_spec = decoders.lookup_table(max_weight=1, penalty_func=lambda error: -float(error[1]))
    with pytest.warns(DeprecationWarning, match="penalty_func is deprecated"):
        legacy_decoder = legacy_spec.build(np.array([[1, 1]], dtype=int))
    assert np.array_equal(legacy_decoder.decode(np.array([1])), [0, 1])

    # a spec that was not built by a helper still has a (less concise) representation
    spec = decoders.DecoderSpec("custom", decoders.get_decoder_lookup, (("max_weight", 1),))
    assert repr(spec).startswith("DecoderSpec('custom', ")

    # misspelled options are rejected, rather than silently passed to a decoder
    with pytest.raises(TypeError, match="lsd_ordr"):
        decoders.bp_lsd(lsd_ordr=1)  # type: ignore[call-arg]


def test_observable_decoder_specs() -> None:
    """A spec without an error builder builds observable decoders, but not error decoders."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    spec: decoders.DecoderSpec[Never] = decoders.DecoderSpec(
        "observable_lookup", None, (("max_weight", 1),), get_observable_decoder_lookup
    )
    assert decoders.lookup_table(1).infers_errors
    assert not spec.infers_errors
    assert spec.predicts_observables_natively
    assert isinstance(spec.build_observable_decoder(dem), decoders.ObservableLookupDecoder)
    with pytest.raises(TypeError, match="cannot build an error decoder"):
        spec.build(dem)
    with pytest.raises(TypeError, match="cannot build an error decoder"):
        decoders.get_error_decoder(dem, decoder=spec)
    with pytest.raises(ValueError, match="needs an error builder or an observable builder"):
        decoders.DecoderSpec("nothing", None, ())

    # the Frontier helper stores its options without importing Frontier
    options: dict[str, Any] = {
        "K": 64,
        "Delta": 6.0,
        "score_alpha": 0.5,
        "metric_mode": "frontier_lite",
        "int_metric_scale": 512,
        "column_order": "time_order",
        "committee": True,
        "add_erasure_bit": True,
    }
    spec = decoders.frontier(**options)
    assert spec.options == options
    assert not spec.infers_errors
    assert spec.predicts_observables_natively
    assert repr(decoders.frontier(committee=True)) == "decoders.frontier(committee=True)"
    invalid_spec = decoders.frontier(K=0)
    with pytest.raises(ValueError, match="K must be positive"):
        invalid_spec.build_observable_decoder(dem)


def _get_graphlike_inputs() -> tuple[npt.NDArray[np.int_], stim.DetectorErrorModel]:
    """A parity check matrix and a detector error model whose errors flip at most two checks."""
    matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=int)
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.01
    )
    return matrix, circuit.detector_error_model()


def test_decoder_spec_helpers_build_decoders() -> None:
    """Every helper builds a working error decoder for a parity check matrix and a DEM."""
    specs: list[decoders.DecoderSpec[decoders.ErrorDecoder]] = [
        decoders.bp_osd(max_iter=5, osd_method="OSD_CS", osd_order=2),
        decoders.bp_lsd(max_iter=5, lsd_method="LSD_CS", lsd_order=2, always_run_lsd=True),
        decoders.bf(max_iter=5, uf_method="inversion"),
        decoders.mwpm(merge_strategy="independent"),
        decoders.relay_bp(gamma0=0.2),
        decoders.lookup_table(max_weight=1),
        decoders.ilp(),
        decoders.guf(max_weight=2),
    ]
    for pcm_or_dem in _get_graphlike_inputs():
        pcm = (
            decoders.DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix.toarray()
            if (isinstance(pcm_or_dem, stim.DetectorErrorModel))
            else pcm_or_dem
        )
        num_errors = pcm.shape[1]
        for spec in specs:
            decoder = spec.build(pcm_or_dem)
            error = np.zeros(num_errors, dtype=int)
            error[0] = 1
            syndrome = pcm @ error % 2
            decoded_error = np.asarray(decoder.decode(syndrome), dtype=int)
            assert decoded_error.shape == (num_errors,), spec
            assert np.array_equal(pcm @ decoded_error % 2, syndrome), spec

    # options reach the decoders that they configure
    matrix = _get_graphlike_inputs()[0]
    bp_osd_decoder: Any = decoders.bp_osd(osd_method="OSD_CS", osd_order=2).build(matrix)
    bp_lsd_decoder: Any = decoders.bp_lsd(lsd_method="LSD_CS", lsd_order=3).build(matrix)
    bf_decoder: Any = decoders.bf(uf_method="inversion").build(matrix)
    guf_decoder: Any = decoders.guf(max_weight=2).build(matrix)
    assert bp_osd_decoder.osd_order == 2
    assert bp_lsd_decoder.lsd_order == 3
    assert bf_decoder.uf_method == "inversion"
    assert guf_decoder.default_max_weight == 2


def test_decoder_spec_helper_defaults() -> None:
    """Helper defaults agree with the defaults of the decoders that the helpers configure."""
    import inspect

    import ldpc
    import pymatching

    def get_defaults(func: Callable[..., object]) -> dict[str, object]:
        return {
            name: parameter.default
            for name, parameter in inspect.signature(func).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

    # helpers for decoders defined in qLDPC mirror all non-deprecated constructor options
    qldpc_decoders: list[tuple[Callable[..., object], Callable[..., object], set[str]]] = [
        (decoders.frontier, decoders.get_observable_decoder_frontier, set()),
        (decoders.lookup_table, decoders.LookupDecoder, {"predict_observable_flips"}),
        (decoders.guf, decoders.GUFDecoder, set()),
        (decoders.ilp, decoders.ILPDecoder, set()),
        (decoders.tesseract, decoders.TesseractDecoder, set()),
    ]
    for helper, constructor, excluded in qldpc_decoders:
        helper_defaults = get_defaults(helper)
        constructor_defaults = get_defaults(constructor)
        assert helper_defaults.keys() == constructor_defaults.keys() - excluded, helper
        for name, default in helper_defaults.items():
            assert default == constructor_defaults[name], (helper, name)

    entry_points: list[Callable[..., object]] = [
        decoders.LookupDecoder,
        decoders.ObservableLookupDecoder,
        decoders.lookup_table,
    ]
    for entry_point in entry_points:
        assert list(inspect.signature(entry_point).parameters)[-1] == "penalty_func"

    # helpers for relay-bp mirror the options of RelayBPDecoder (other than name, which the
    # precision selects) and of the relay_bp classes that they configure
    import relay_bp

    relay_bp_decoder_defaults = get_defaults(decoders.RelayBPDecoder)
    del relay_bp_decoder_defaults["name"]
    relay_bp_helpers: list[tuple[Callable[..., object], Any]] = [
        (decoders.relay_bp, relay_bp.RelayDecoderF32),
        (decoders.min_sum_bp, relay_bp.MinSumBPDecoderF32),
    ]
    for relay_bp_helper, relay_bp_class in relay_bp_helpers:
        helper_defaults = get_defaults(relay_bp_helper)
        assert helper_defaults.pop("precision") == "F32"
        relay_bp_defaults = relay_bp_decoder_defaults | get_defaults(relay_bp_class)
        assert helper_defaults.keys() == relay_bp_defaults.keys(), relay_bp_helper
        for name, default in helper_defaults.items():
            # relay-bp does not expose some defaults, which the helpers leave to relay-bp
            expected = None if relay_bp_defaults[name] is Ellipsis else relay_bp_defaults[name]
            assert default == expected, (relay_bp_helper, name)

    # helpers for pymatching agree with pymatching wherever they share options
    pymatching_defaults = get_defaults(pymatching.Matching.from_check_matrix)
    for name, default in get_defaults(decoders.mwpm).items():
        if name in pymatching_defaults:
            assert default == pymatching_defaults[name], name

    # ldpc does not expose signatures, so compare decoders built with default helper options
    matrix = _get_graphlike_inputs()[0]
    shared_attributes = [
        "max_iter",
        "bp_method",
        "ms_scaling_factor",
        "schedule",
        "omp_thread_count",
        "random_schedule_seed",
    ]
    ldpc_decoders: list[
        tuple[Callable[[], decoders.DecoderSpec[decoders.ErrorDecoder]], Any, list[str]]
    ] = [
        (decoders.bp_osd, ldpc.BpOsdDecoder, ["osd_method", "osd_order"]),
        (decoders.bp_lsd, ldpc.bplsd_decoder.BpLsdDecoder, ["lsd_method", "lsd_order"]),
        (decoders.bf, ldpc.BeliefFindDecoder, ["uf_method"]),
    ]
    for helper, constructor, attributes in ldpc_decoders:
        helper_decoder = helper().build(matrix)
        ldpc_decoder = constructor(matrix, error_rate=get_defaults(helper)["error_rate"])
        for attribute in shared_attributes + attributes:
            assert getattr(helper_decoder, attribute) == getattr(ldpc_decoder, attribute), (
                helper,
                attribute,
            )


def test_correlated_matching() -> None:
    """Correlated matching uses the decompositions that a detector error model suggests."""
    dem = stim.DetectorErrorModel("""
        error(0.02) D0 D1 ^ D2 D3
        error(0.3) D2 L0
        error(0.3) D3
    """)
    spec = decoders.mwpm(enable_correlations=True)

    # enabling correlations changes the prediction from [1], because the decomposed error that
    # explains D0 D1 also explains D2 D3
    decoder = decoders.get_observable_decoder(dem, decoder=spec)
    assert np.array_equal(decoder.decode_observables(np.array([1, 1, 1, 1])), [0])

    with pytest.raises(ValueError, match="cannot infer errors"):
        decoders.get_error_decoder(dem, decoder=spec)
    with pytest.raises(ValueError, match="not supported with enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, decompose_errors=True)
    with pytest.raises(ValueError, match="not supported with enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, ignore_non_graphlike_errors=True)
