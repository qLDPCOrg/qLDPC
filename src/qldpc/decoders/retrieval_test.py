# SPDX-License-Identifier: Apache-2.0

"""Unit tests for retrieval.py."""

from __future__ import annotations

import pickle
import re
import warnings
from collections.abc import Callable
from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders import retrieval


def test_custom_decoder(pytestconfig: pytest.Config) -> None:
    """Inject custom decoders."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    matrix = np.random.randint(2, size=(2, 2))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder(decoders.ErrorDecoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.asarray(error)

    # a subclass of ErrorDecoder that implements only decode inherits decode_errors
    assert decoders.get_error_decoder(matrix, decoder=CustomDecoder).decode(syndrome) is error
    assert CustomDecoder(matrix).decode_errors(syndrome) is error
    assert (
        decoders.get_error_decoder(matrix, decoder=CustomDecoder(matrix)).decode_errors(syndrome)
        is error
    )

    # an object with only a decode method is wrapped to provide decode_errors
    class BareDecoder:
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.asarray(error)

    bare_decoder = BareDecoder()
    wrapped_decoder: Any = decoders.get_error_decoder(matrix, decoder=bare_decoder)
    assert wrapped_decoder.decoder is bare_decoder
    assert wrapped_decoder.decode_errors(syndrome) is error
    assert not hasattr(wrapped_decoder, "decode_errors_batch")

    # a subclass of ErrorDecoder must implement a decoding method
    class IncompleteDecoder(decoders.ErrorDecoder): ...

    with pytest.raises(NotImplementedError, match="must implement decode_errors"):
        IncompleteDecoder().decode_errors(syndrome)

    # injected decoders are validated, which must survive `python -O`
    with pytest.raises(TypeError, match="must be an ErrorDecoder, or have a decode method"):
        decoders.get_error_decoder(matrix, decoder=lambda _: 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="decoder must be decoder settings"):
        decoders.get_error_decoder(matrix, decoder=0)  # type: ignore[arg-type]


def test_decoder_selection() -> None:
    """Exactly one decoder can be requested at a time."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    # a falsy request is consumed rather than passed on to the requested decoder
    with pytest.warns(DeprecationWarning, match=r"decoders\.bf\(\.\.\.\)\.build"):
        decoded_error = decoders.decode(matrix, syndrome, with_BF=True, with_MWPM=False)
    assert np.array_equal([1, 1], decoded_error)

    with (
        pytest.warns(DeprecationWarning, match="get_decoder is deprecated"),
        pytest.raises(ValueError, match="Only one decoder"),
    ):
        decoders.get_decoder(matrix, with_BF=True, with_MWPM=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoders.decode(matrix, syndrome, with_BF=True)
    assert caught[0].filename == __file__
    assert "decoders.bf(...).build(pcm_or_dem)" in str(caught[0].message)

    with pytest.warns(DeprecationWarning, match=r"decoders\.bp_osd\(\.\.\.\)\.build"):
        decoders.get_decoder(matrix, max_iter=1)

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = decoders.get_decoder(
            dem,
            with_lookup=True,
            max_weight=1,
            predict_observable_flips=True,
        )
    messages = [str(warning.message) for warning in caught]
    assert any(
        "construct an ObservableLookupDecoder" in message and "decode_observables" in message
        for message in messages
    )
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])


def test_deprecated_decoder_functions(pytestconfig: pytest.Config) -> None:
    """The deprecated get_decoder and decode functions behave as they did, and name replacements."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    matrix = np.random.randint(2, size=(3, 4))
    error = np.random.randint(2, size=matrix.shape[1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder:
        def __init__(self, matrix: npt.NDArray[np.int_], scale: int = 1) -> None:
            self.scale = scale

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.scale * np.asarray(error)

    # warnings name the replacing call, and point at the caller
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoders.get_decoder(matrix)
        decoders.get_decoder(matrix, decoder_constructor=CustomDecoder, scale=2)
        decoders.decode(matrix, syndrome, with_lookup=True, max_weight=1)
    assert [str(warning.message) for warning in caught] == [
        "decoders.get_decoder is deprecated; use decoders.get_error_decoder(pcm_or_dem) instead",
        "decoders.get_decoder is deprecated; use CustomDecoder(pcm_or_dem, ...) instead",
        (
            "decoders.decode is deprecated; use"
            " decoders.lookup_table(...).build(pcm_or_dem).decode(syndrome) instead"
        ),
    ]
    assert all(warning.filename == __file__ for warning in caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # a decoder constructor receives the remaining arguments, and is returned as is
        decoder = decoders.get_decoder(matrix, decoder_constructor=CustomDecoder, scale=2)
        assert isinstance(decoder, CustomDecoder) and decoder.scale == 2
        with pytest.raises(TypeError, match="must be callable"):
            decoders.get_decoder(matrix, decoder_constructor=0)

        # a static decoder is returned as is, and admits no other arguments
        static_decoder = CustomDecoder(matrix)
        assert decoders.get_decoder(matrix, static_decoder=static_decoder) is static_decoder
        assert decoders.get_decoder(matrix, static_decoder=CustomDecoder) is CustomDecoder
        with pytest.raises(ValueError, match="cannot process decoding arguments"):
            decoders.get_decoder(matrix, static_decoder=static_decoder, with_BF=True)

        # the default decoder depends on the field
        decoder = decoders.get_decoder(galois.GF(3)(matrix))
        assert isinstance(decoder, decoders.GUFDecoder)

    with pytest.warns(DeprecationWarning, match=r"use static_decoder\.decode\(syndrome\)"):
        decoded_error = decoders.decode(matrix, syndrome, static_decoder=static_decoder)
    assert np.array_equal(decoded_error, error)

    # the static_decoder argument has been removed from other methods, in favor of decoder=
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.resolve_decoder(matrix, None, {"static_decoder": static_decoder})
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.resolve_observable_decoder(dem, None, {"static_decoder": static_decoder})
    with pytest.raises(ValueError, match="Cannot combine decoder"):
        decoders.resolve_decoder(matrix, static_decoder, {"with_BF": True})


def test_legacy_decoder_migration_messages() -> None:
    """Deprecated keyword arguments of methods warn with the decoder input that replaces them."""
    matrix = np.eye(2, dtype=int)
    expected_messages: list[tuple[dict[str, object], str]] = [
        ({"decoder_constructor": decoders.LookupDecoder}, "for example decoder=LookupDecoder"),
        ({"with_lookup": True, "predict_observable_flips": True}, "ObservableLookupDecoder"),
        ({"with_BF": True}, r"with_BF keyword .* use decoder=decoders\.bf\(\.\.\.\)"),
        ({"with_BF": True, "with_MWPM": True}, "pass exactly one"),
        ({"max_iter": 5}, r"move them into decoder=decoders\.bp_osd\(\.\.\.\)"),
    ]
    for decoder_args, expected_message in expected_messages:
        message = retrieval.get_legacy_decoder_migration_message(matrix, decoder_args)
        assert re.search(expected_message, message), message
    message = retrieval.get_legacy_decoder_migration_message(
        galois.GF(3)(matrix), {"max_weight": 1}, argument_name="decoder_x"
    )
    assert "decoder_x=decoders.guf(...)" in message

    # deprecated arguments build a decoder, warning unless an outer API has already warned
    with pytest.warns(DeprecationWarning, match="with_BF keyword"):
        decoder = decoders.resolve_decoder(matrix, None, {"with_BF": True})
    assert np.array_equal(decoder.decode_errors(np.array([1, 0])), [1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.resolve_decoder(matrix, None, {"with_BF": True}, warn_deprecated=False)
        decoders.resolve_decoder(matrix, decoders.bf(), {})


def test_deprecated_aliases() -> None:
    """Deprecated names of decoder protocols warn, and refer to their replacements."""
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert retrieval.Decoder is decoders.ErrorDecoder
    with pytest.warns(DeprecationWarning, match="BatchDecoder is deprecated"):
        assert retrieval.BatchDecoder is decoders.BatchErrorDecoder


def test_decoder_specs() -> None:
    """Typed decoder specs defer construction and survive process serialization."""
    matrix = np.eye(2, dtype=int)
    syndrome = np.array([1, 0], dtype=int)

    spec = decoders.lookup_table(max_weight=1)
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
    assert repr(decoders.ilp(verbose=False)) == "decoders.ilp(verbose=False)"
    assert repr(decoders.guf(max_weight=1)) == "decoders.guf(max_weight=1)"
    channel = np.array([0.1, 0.2])
    assert "error_channel=array" in repr(decoders.bf(error_channel=channel))

    # a spec that was not built by a helper still has a (less concise) representation
    spec = decoders.DecoderSpec("custom", decoders.get_decoder_lookup, (("max_weight", 1),))
    assert repr(spec).startswith("DecoderSpec('custom', ")

    # misspelled options are rejected, rather than silently passed to a decoder
    with pytest.raises(TypeError, match="lsd_ordr"):
        decoders.bp_lsd(lsd_ordr=1)  # type: ignore[call-arg]


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
        (decoders.lookup_table, decoders.LookupDecoder, {"predict_observable_flips"}),
        (decoders.guf, decoders.GUFDecoder, set()),
        (decoders.ilp, decoders.ILPDecoder, set()),
    ]
    for helper, constructor, excluded in qldpc_decoders:
        helper_defaults = get_defaults(helper)
        constructor_defaults = get_defaults(constructor)
        assert helper_defaults.keys() == constructor_defaults.keys() - excluded, helper
        for name, default in helper_defaults.items():
            assert default == constructor_defaults[name], (helper, name)

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


def _get_circuit_data() -> tuple[stim.DetectorErrorModel, npt.NDArray[np.int_]]:
    """A repetition-code memory experiment's detector error model and sampled syndromes."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.02
    )
    syndromes = circuit.compile_detector_sampler(seed=0).sample(200).astype(int)
    return circuit.detector_error_model(), syndromes


def test_native_observable_decoders() -> None:
    """Decoders that predict observable flips natively agree with ones that convert errors."""
    dem, syndromes = _get_circuit_data()
    native_decoder_types: list[tuple[decoders.DecoderSpec[Any], type[Any]]] = [
        (decoders.mwpm(), retrieval._MatchingObservableDecoder),
        (decoders.relay_bp(), decoders.RelayBPDecoder),
        (decoders.min_sum_bp(gamma0=0.5), decoders.RelayBPDecoder),
        (decoders.lookup_table(max_weight=2), decoders.ObservableLookupDecoder),
    ]
    for spec, native_decoder_type in native_decoder_types:
        assert spec.predicts_observables_natively
        native_decoder: Any = decoders.get_observable_decoder(dem, decoder=spec)
        assert isinstance(native_decoder, native_decoder_type)
        converted_decoder = retrieval.ErrorsToObservablesDecoder(spec.build(dem), dem)
        assert np.array_equal(
            native_decoder.decode_observables_batch(syndromes),
            converted_decoder.decode_observables_batch(syndromes),
        ), spec
        assert np.array_equal(
            native_decoder.decode_observables(syndromes[0]),
            converted_decoder.decode_observables(syndromes[0]),
        ), spec
        assert np.array_equal(
            decoders.decode_observables(dem, syndromes[0], decoder=spec),
            native_decoder.decode_observables(syndromes[0]),
        )

        # an empty batch yields empty predictions
        for observable_decoder in [native_decoder, converted_decoder]:
            no_flips = observable_decoder.decode_observables_batch(syndromes[:0])
            assert no_flips.shape == (0, dem.num_observables), spec
            assert no_flips.dtype == np.uint8, spec

    # other decoders predict observable flips by converting the errors that they infer
    spec = decoders.bp_osd()
    assert not spec.predicts_observables_natively
    bp_osd_decoder = spec.build_observable_decoder(dem)
    assert isinstance(bp_osd_decoder, retrieval.ErrorsToObservablesDecoder)
    assert isinstance(decoders.get_observable_decoder(dem), retrieval.ErrorsToObservablesDecoder)
    no_flips = bp_osd_decoder.decode_observables_batch(syndromes[:0])
    assert no_flips.shape == (0, dem.num_observables)

    # native observable decoders support detector error models without observables
    dem_without_observables = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.1) D0
        error(0.1) D1
    """)
    specs: list[decoders.DecoderSpec[Any]] = [decoders.relay_bp(), decoders.lookup_table(1)]
    for spec in specs:
        observable_decoder = decoders.get_observable_decoder(dem_without_observables, decoder=spec)
        assert observable_decoder.decode_observables(np.array([1, 0])).shape == (0,)

    # an erasure bit of an error decoder is appended to predicted observable flips
    erasing_decoder = decoders.get_observable_decoder(
        dem, decoder=decoders.guf(add_erasure_bit=True)
    )
    assert getattr(erasing_decoder, "has_erasure_bit", False)
    assert erasing_decoder.decode_observables(syndromes[0]).shape == (dem.num_observables + 1,)


def test_correlated_matching() -> None:
    """Correlated matching exploits the decompositions of a detector error model."""
    import pymatching

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.02,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes = circuit.compile_detector_sampler(seed=0).sample(1000).astype(np.uint8)

    spec = decoders.mwpm(enable_correlations=True)
    assert spec.predicts_observables_natively
    assert repr(spec) == "decoders.mwpm(enable_correlations=True)"
    correlated_decoder: Any = decoders.get_observable_decoder(dem, decoder=spec)
    assert isinstance(correlated_decoder, retrieval._MatchingObservableDecoder)

    # predictions agree with those of pymatching, and differ from those of uncorrelated matching
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    expected_flips = matching.decode_batch(syndromes, enable_correlations=True)
    uncorrelated_decoder: Any = decoders.get_observable_decoder(
        dem, decoder=decoders.mwpm(decompose_errors=True)
    )
    uncorrelated_flips = uncorrelated_decoder.decode_observables_batch(syndromes)
    correlated_flips = correlated_decoder.decode_observables_batch(syndromes)
    assert np.array_equal(correlated_flips, expected_flips)
    assert not np.array_equal(correlated_flips, uncorrelated_flips)
    for syndrome, flips in zip(syndromes[:10], expected_flips[:10]):
        assert np.array_equal(correlated_decoder.decode_observables(syndrome), flips)
        assert np.array_equal(decoders.decode_observables(dem, syndrome, decoder=spec), flips)

    # correlated matching cannot infer errors
    with pytest.raises(ValueError, match="cannot infer errors"):
        decoders.get_error_decoder(dem, decoder=spec)
    with pytest.raises(ValueError, match="cannot infer errors"):
        decoders.get_decoder_MWPM(dem, enable_correlations=True)

    # options that configure an uncorrelated matching graph are rejected
    unsupported_options: list[dict[str, Any]] = [
        {"decompose_errors": True},
        {"weights": 1.0},
        {"error_probabilities": 0.1},
        {"repetitions": 2},
        {"timelike_weights": 1.0},
        {"measurement_error_probabilities": 0.1},
        {"merge_strategy": "independent"},
        {"use_virtual_boundary_node": True},
    ]
    for options in unsupported_options:
        (name,) = options
        with pytest.raises(ValueError, match=rf"option {name}=.* not supported"):
            decoders.mwpm(enable_correlations=True, **options)
    with pytest.raises(ValueError, match=r"option weights=1\.0 is not supported"):
        retrieval._get_observable_decoder_MWPM(dem, enable_correlations=True, weights=1.0)

    # errors that are not graphlike, even after decomposition, are rejected or ignored
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 D4 L1
        error(0.1) D0 D1 ^ D1 D3 L0
        error(0.1) D0
    """)
    with pytest.raises(ValueError, match="component that flips 3 detectors"):
        decoders.get_observable_decoder(dem, decoder=spec)
    spec = decoders.mwpm(enable_correlations=True, ignore_non_graphlike_errors=True)
    correlated_decoder = decoders.get_observable_decoder(dem, decoder=spec)
    # the dropped error was the only one to flip the last detector and the last observable
    assert np.array_equal(correlated_decoder.decode_observables(np.array([1, 0, 0, 0, 0])), [0, 0])
    assert np.array_equal(correlated_decoder.decode_observables(np.array([1, 0, 0, 1, 0])), [1, 0])


def test_observable_decoder_inputs() -> None:
    """Observable decoders are built from settings, constructors, and prebuilt decoders."""
    dem, syndromes = _get_circuit_data()
    observable_lookup = decoders.ObservableLookupDecoder(dem, max_weight=2)
    error_lookup = decoders.LookupDecoder(dem, max_weight=2)
    expected_flips = observable_lookup.decode_observables_batch(syndromes)

    # prebuilt decoders, and constructors of either kind of decoder
    decoder_inputs: list[decoders.ObservableDecoderInput] = [
        observable_lookup,
        error_lookup,
        lambda dem: decoders.ObservableLookupDecoder(dem, max_weight=2),
        lambda dem: decoders.LookupDecoder(dem, max_weight=2),
    ]
    for decoder_input in decoder_inputs:
        observable_decoder: Any = decoders.get_observable_decoder(dem, decoder=decoder_input)
        assert np.array_equal(
            observable_decoder.decode_observables_batch(syndromes), expected_flips
        )
    assert decoders.get_observable_decoder(dem, decoder=observable_lookup) is observable_lookup

    # deprecated decoder-selection arguments are converted by an internal path
    observable_decoder = retrieval.resolve_observable_decoder(
        dem, None, {"with_lookup": True, "max_weight": 2}, warn_deprecated=False
    )
    assert np.array_equal(
        [observable_decoder.decode_observables(syndrome) for syndrome in syndromes], expected_flips
    )

    # invalid inputs
    with pytest.raises(TypeError, match="decoder must be decoder settings"):
        decoders.get_observable_decoder(dem, decoder=object())  # type: ignore[arg-type]

    def build_invalid_decoder(dem: stim.DetectorErrorModel) -> Any:
        return object()

    spec = decoders.DecoderSpec("custom", decoders.get_decoder_lookup, (), build_invalid_decoder)
    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        spec.build_observable_decoder(dem)


def test_error_decoder_output_is_validated() -> None:
    """A decoder whose .decode returns observable flips is not mistaken for an error decoder."""
    import pymatching

    dem, _ = _get_circuit_data()

    # a matching decoder built from a DEM returns observable flips from its decode method
    with pytest.raises(ValueError, match="inferred an error of length 1"):
        decoders.get_observable_decoder(
            dem, decoder=lambda dem: pymatching.Matching.from_detector_error_model(dem)
        )

    # the deprecated LookupDecoder(..., predict_observable_flips=True) is rejected explicitly
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        decoders.get_observable_decoder(
            dem,
            decoder=lambda dem: decoders.LookupDecoder(
                dem, max_weight=1, predict_observable_flips=True
            ),
        )


def test_merged_error_mechanisms() -> None:
    """Errors inferred by a decoder that merges equivalent error mechanisms are expanded."""
    # the first two error mechanisms are equivalent
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    # GUF decoders merge equivalent mechanisms, and decode one syndrome at a time
    for add_erasure_bit in [False, True]:
        merging_decoder = decoders.get_decoder_GUF(dem, add_erasure_bit=add_erasure_bit)
        assert len(merging_decoder.decode(syndromes[0])) == 2 + add_erasure_bit
        decoder: Any = retrieval.match_error_decoder_to_dem(merging_decoder, dem)
        assert isinstance(decoder, retrieval.ExpandedErrorDecoder)
        errors = decoder.decode_batch(syndromes)
        assert errors.shape == (2, 3 + add_erasure_bit)
        assert np.array_equal(errors, [decoder.decode(syndrome) for syndrome in syndromes])
        assert np.array_equal(errors[:, [0, 1]].sum(axis=1), [1, 0])  # one of the merged errors
        assert decoder.decode_batch(syndromes[:0]).shape == (0, 3 + add_erasure_bit)

    # matching decoders merge equivalent mechanisms, and decode in batches
    decoder = retrieval.match_error_decoder_to_dem(decoders.get_decoder_MWPM(dem), dem)
    assert isinstance(decoder, retrieval.ExpandedErrorDecoder)
    assert decoder.decode_batch(syndromes).shape == (2, 3)


def test_decomposed_error_mechanisms() -> None:
    """Errors in the components of decomposed error mechanisms are not read as errors of a DEM."""
    # decomposition splits the first error, but leaves as many (merged) errors as the model has
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        error(0.1) D0
        error(0.01) D1
    """)
    syndrome = np.array([1, 0], dtype=int)

    # a matching decoder can predict observable flips natively
    spec = decoders.mwpm(decompose_errors=True)
    assert np.array_equal(decoders.decode_observables(dem, syndrome, decoder=spec), [0])

    # an error decoder that infers decomposed errors cannot predict observable flips
    decoder_inputs: list[decoders.ObservableDecoderInput] = [
        spec.build(dem),
        lambda dem: decoders.get_decoder_MWPM(dem, decompose_errors=True),
    ]
    for decoder_input in decoder_inputs:
        with pytest.raises(ValueError, match="components of decomposed error mechanisms"):
            decoders.get_observable_decoder(dem, decoder=decoder_input)

    # decomposition that splits no error leaves the error mechanisms of a model intact
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1")
    decoder = decoders.get_observable_decoder(
        dem, decoder=lambda dem: decoders.get_decoder_MWPM(dem, decompose_errors=True)
    )
    assert np.array_equal(decoder.decode_observables(syndrome), [1])


def test_reject_prebuilt_decoder() -> None:
    """Prebuilt decoders are rejected where a decoder must be built for a new matrix."""
    matrix = np.eye(2, dtype=int)
    prebuilt = decoders.LookupDecoder(matrix, max_weight=1)
    reason = "the matrix is new"
    for decoder in [None, decoders.lookup_table(max_weight=1), decoders.LookupDecoder]:
        retrieval.reject_prebuilt_decoder(decoder, reason)
    with pytest.raises(ValueError, match="cannot be passed as decoder= here because the matrix"):
        retrieval.reject_prebuilt_decoder(prebuilt, reason)


def test_invalid_explicit_decoder_inputs() -> None:
    """The explicit input rejects observable predictors and invalid factories."""
    matrix = np.eye(1, dtype=int)
    observable = decoders.ObservableLookupDecoder(
        stim.DetectorErrorModel("error(0.1) D0 L0"), max_weight=1
    )

    with pytest.raises(TypeError, match="observable flips rather than errors"):
        decoders.get_error_decoder(matrix, decoder=observable)  # type: ignore[arg-type]

    def observable_factory(_matrix: object) -> object:
        return observable

    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        decoders.get_error_decoder(
            matrix,
            decoder=observable_factory,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="decoder must be decoder settings"):
        decoders.get_error_decoder(matrix, decoder=object())  # type: ignore[arg-type]


def test_erasure_bit_request() -> None:
    """A request for an erasure bit is rejected by a decoder that cannot signal erasure."""
    matrix = np.eye(3, 2, dtype=int)

    # a decoder that can signal erasure honours direct and routed requests
    erasing_decoders: list[
        tuple[Callable[..., decoders.ErrorDecoder], decoders.DecoderSpec[decoders.ErrorDecoder]]
    ] = [
        (decoders.get_decoder_RBP, decoders.relay_bp(add_erasure_bit=True)),
        (decoders.get_decoder_ILP, decoders.ilp(add_erasure_bit=True)),
        (decoders.get_decoder_GUF, decoders.guf(add_erasure_bit=True)),
        (decoders.get_decoder_lookup, decoders.lookup_table(max_weight=2, add_erasure_bit=True)),
    ]
    for decoder_getter, decoder_spec in erasing_decoders:
        direct_args = {"max_weight": 2} if decoder_getter is decoders.get_decoder_lookup else {}
        decoder = decoder_getter(matrix, add_erasure_bit=True, **direct_args)
        assert getattr(decoder, "has_erasure_bit", False)
        decoder = decoders.get_error_decoder(matrix, decoder=decoder_spec)
        assert getattr(decoder, "has_erasure_bit", False)

    # every decoder that cannot signal erasure rejects direct and routed requests consistently; the
    # typed helpers for these decoders do not accept an add_erasure_bit argument at all
    unerasing_decoders: list[
        tuple[Callable[..., decoders.ErrorDecoder], dict[str, object], str]
    ] = [
        (decoders.get_decoder_BF, {"with_BF": True}, "BF"),
        (decoders.get_decoder_BP_OSD, {"with_BP_OSD": True}, "BP_OSD"),
        (decoders.get_decoder_MWPM, {"with_MWPM": True}, "MWPM"),
        (decoders.get_decoder_BP_LSD, {"with_BP_LSD": True}, "BP_LSD"),
    ]
    for decoder_getter, decoder_args, decoder_name in unerasing_decoders:
        with pytest.raises(ValueError, match=rf"The {decoder_name} decoder cannot signal erasure"):
            decoder_getter(matrix, add_erasure_bit=True)
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match=rf"The {decoder_name} decoder cannot signal erasure"),
        ):
            decoders.get_decoder(matrix, add_erasure_bit=True, **decoder_args)
        assert decoder_getter(matrix, add_erasure_bit=False)

    # BP+OSD is the default for a binary matrix
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match=r"The BP_OSD decoder cannot signal erasure"),
    ):
        decoders.get_decoder(matrix, add_erasure_bit=True)


def test_erasure_bit_support_decorator() -> None:
    """A getter declared to support erasure must return a decoder that does so."""

    @retrieval._erasure_bit_support(True)
    def get_decoder_inconsistent(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        return decoders.get_decoder_BP_OSD(matrix)

    with pytest.raises(ValueError, match=r"The inconsistent decoder cannot signal erasure"):
        get_decoder_inconsistent(np.eye(1, dtype=int), add_erasure_bit=True)


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    assert np.array_equal(
        error, decoders.get_error_decoder(matrix).decode(syndrome)
    )  # default, BP+OSD
    for decoder in [
        decoders.bp_lsd(),
        decoders.bf(),
        decoders.relay_bp(),
        decoders.mwpm(),
        decoders.ilp(),
        decoders.guf(),
        decoders.lookup_table(max_weight=2),
    ]:
        assert np.array_equal(
            error, decoders.get_error_decoder(matrix, decoder=decoder).decode(syndrome)
        )

    # default to GUF with non-binary fields
    field = galois.GF(3)
    matrix = matrix.view(field)
    syndrome = syndrome.view(field)
    error = error.view(field)
    assert np.array_equal(error, decoders.get_error_decoder(matrix).decode(syndrome))
    with pytest.warns(DeprecationWarning, match=r"decoders\.guf\(\.\.\.\)\.build"):
        assert decoders.get_decoder(matrix, max_weight=1)

    # decode from a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    for decoder in [decoders.bp_lsd(), decoders.mwpm(), decoders.ilp(), decoders.guf()]:
        assert np.array_equal(
            error, decoders.get_error_decoder(dem, decoder=decoder).decode(syndrome)
        )

    # a MWPM decoder built from a DEM takes its error weights from that DEM
    with pytest.raises(ValueError, match="Cannot set error weights"):
        decoders.get_error_decoder(dem, decoder=decoders.mwpm(weights=[1.0, 1.0]))

    # add a non-graphlike error mechanism, which MWPM can ignore upon request
    matrix = np.hstack([matrix, np.ones((3, 1))])
    error = np.concatenate([error, [0]])
    dem.append("error", 0.125, [stim.DemTarget.relative_detector_id(ii) for ii in range(3)])
    with pytest.raises(ValueError, match="non-graphlike error"):
        decoders.get_error_decoder(dem, decoder=decoders.mwpm()).decode(syndrome)
    decoder = decoders.mwpm(ignore_non_graphlike_errors=True)
    assert np.array_equal(error, decoders.get_error_decoder(dem, decoder=decoder).decode(syndrome))


def test_non_graphlike_over_a_field() -> None:
    """An error is non-graphlike by the number of detectors it addresses, not by their sum."""
    matrix = galois.GF(2)([[1, 1], [1, 0], [1, 0]])  # column 0 addresses three detectors
    syndrome = np.array([1, 0, 0], dtype=int)

    with pytest.raises(ValueError, match="column 0 of the parity check matrix addresses 3"):
        decoders.get_error_decoder(matrix, decoder=decoders.mwpm()).decode(syndrome)
    decoder = decoders.mwpm(ignore_non_graphlike_errors=True)
    assert np.array_equal(
        [0, 1], decoders.get_error_decoder(matrix, decoder=decoder).decode(syndrome)
    )
