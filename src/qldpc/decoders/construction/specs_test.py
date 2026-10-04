# SPDX-License-Identifier: Apache-2.0

"""Tests for generic typed decoder specifications."""

from __future__ import annotations

import inspect
import pathlib
import pickle
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Never, cast

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders import common
from qldpc.decoders.adapters import error_decoders
from qldpc.decoders.construction import specs
from qldpc.decoders.construction.resolution import _get_error_decoder, _get_observable_decoder
from qldpc.decoders.custom.lookup import _get_decoder_lookup, _get_observable_decoder_lookup
from qldpc.decoders.external.frontier import _get_observable_decoder_frontier
from qldpc.decoders.external.ldpc import _get_decoder_bf, _get_decoder_bp_lsd, _get_decoder_bp_osd


def _get_graphlike_inputs() -> tuple[npt.NDArray[np.int_], stim.DetectorErrorModel]:
    """A parity check matrix and a detector error model whose errors flip at most two checks."""
    matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=int)
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.01
    )
    return matrix, circuit.detector_error_model()


def _uniform_binary_error_channel(error: npt.NDArray[np.int_] | Sequence[int]) -> float:
    """Return the log probability of a uniformly random binary error."""
    return -float(np.size(error) * np.log(2))


def test_decoder_specs() -> None:
    """Typed decoder specs defer construction and survive process serialization."""
    matrix = np.eye(2, dtype=int)
    syndrome = np.array([1, 0], dtype=int)

    spec = decoders.lookup(max_weight=1, error_channel=_uniform_binary_error_channel)
    assert spec.options["error_channel"] is _uniform_binary_error_channel
    restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted in-memory round trip
    assert np.array_equal(_get_error_decoder(matrix, decoder=restored).decode(syndrome), syndrome)

    # a spec displays the helper call that reproduces it, omitting default options
    assert repr(decoders.bp_osd()) == "decoders.bp_osd()"
    assert repr(decoders.bp_lsd(max_iter=30, bp_method="ms")) == (
        "decoders.bp_lsd(max_iter=30, bp_method='ms')"
    )
    assert repr(decoders.lookup(2)) == "decoders.lookup(max_weight=2)"
    assert repr(decoders.relay_bp(gamma0=0.2)) == "decoders.relay_bp(gamma0=0.2)"
    assert repr(decoders.tesseract(det_beam=7)) == "decoders.tesseract(det_beam=7)"
    assert repr(decoders.ilp(verbose=False)) == "decoders.ilp(verbose=False)"
    assert repr(decoders.guf(max_weight=1)) == "decoders.guf(max_weight=1)"
    channel = np.array([0.1, 0.2])
    assert "error_channel=array" in repr(decoders.bf(error_channel=channel))

    # a spec exposes a copy of its options, which cannot modify the spec
    spec = decoders.lookup(max_weight=2)
    spec.options["max_weight"] = 3
    assert spec.options["max_weight"] == 2

    # deprecated lookup penalties remain available through deferred construction
    legacy_spec = decoders.lookup(max_weight=1, penalty_func=lambda error: -float(error[1]))
    with pytest.warns(DeprecationWarning, match="penalty_func is deprecated"):
        legacy_decoder = legacy_spec.build(np.array([[1, 1]], dtype=int))
    assert np.array_equal(legacy_decoder.decode(np.array([1])), [0, 1])

    # a spec that was not built by a helper still has a (less concise) representation
    spec = decoders.DecoderSpec("custom", _get_decoder_lookup, (("max_weight", 1),))
    assert repr(spec).startswith("DecoderSpec('custom', ")

    # misspelled options are rejected when specifications are created, including by helpers that
    # forward additional options to their backends through backend_options
    with pytest.raises(TypeError, match=r"bp_lsd\(\).*unexpected keyword argument 'lsd_ordr'"):
        decoders.bp_lsd(lsd_ordr=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match=r"guf\(\).*unexpected keyword argument 'max_weigth'"):
        decoders.guf(max_weigth=1)  # type: ignore[call-arg]


def test_decoder_spec_observable_modes() -> None:
    """Specs expose native observable construction and error-conversion fallback."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    native_spec = decoders.lookup(max_weight=1)
    assert native_spec.predicts_observables_natively
    assert isinstance(
        native_spec.build_observable_decoder(dem), decoders.custom.ObservableLookupDecoder
    )

    converted_spec = decoders.guf()
    assert not converted_spec.predicts_observables_natively
    assert isinstance(
        converted_spec.build_observable_decoder(dem),
        error_decoders.ErrorsToObservablesDecoder,
    )
    assert decoders.tesseract().predicts_observables_natively


def test_decoder_specs_store_module_builders() -> None:
    """Every callable stored by a DecoderSpec is a module-level builder, pickled by reference."""
    specs = [
        decoders.bp_osd(),
        decoders.bp_lsd(),
        decoders.bf(),
        decoders.mwpm(),
        decoders.frontier(),
        decoders.relay_bp(),
        decoders.min_sum_bp(),
        decoders.tesseract(),
        decoders.lookup(1),
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
        "lookup": "qldpc.decoders.custom.lookup",
        "ilp": "qldpc.decoders.custom.ilp",
        "guf": "qldpc.decoders.custom.guf",
    }
    for spec in specs:
        for builder in (spec._builder, spec._observable_builder):
            if builder is None:
                continue
            assert builder.__module__ == expected_modules[spec._helper_name]
            assert builder.__name__.startswith(("_get_decoder_", "_get_observable_decoder_"))

        restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted round trip
        assert restored.options == spec.options
        assert restored._builder is spec._builder
        assert restored._observable_builder is spec._observable_builder


def test_decoder_spec_backend_options() -> None:
    """A backend_options mapping is stored as a plain dict, and cannot repeat named options."""

    def builder(
        matrix: npt.NDArray[np.int_],
        *,
        max_weight: int | None = None,
        backend_options: Mapping[str, object] | None = None,
    ) -> decoders.custom.GUFDecoder:
        options: dict[str, Any] = dict(backend_options or {})
        return decoders.custom.GUFDecoder(matrix, max_weight=max_weight, **options)

    helper = specs.decoder_spec("custom", builder)
    assert helper().options["backend_options"] is None
    assert helper(backend_options={}).options["backend_options"] is None
    backend_options = types.MappingProxyType({"symplectic": False})
    spec = helper(backend_options=backend_options)
    assert type(spec.options["backend_options"]) is dict
    assert spec.options["backend_options"] == {"symplectic": False}
    assert isinstance(spec.build(np.eye(1, dtype=int)), decoders.custom.GUFDecoder)

    with pytest.raises(TypeError, match=r"custom\(\) backend_options must be a mapping"):
        helper(backend_options=cast(Any, ["symplectic"]))
    with pytest.raises(ValueError, match="lists max_weight by name, so pass it directly"):
        helper(backend_options={"max_weight": 1})
    with pytest.raises(ValueError, match="lists backend_options, max_weight by name, so pass them"):
        helper(backend_options={"max_weight": 1, "backend_options": {}})


def test_decoder_spec_factory_validation() -> None:
    """Factories transform observable options and reject unusable source signatures."""

    def observable_builder(
        dem: stim.DetectorErrorModel, *, scale: int = 1
    ) -> decoders.ObservableDecoder:
        del scale
        return _get_observable_decoder_lookup(dem, max_weight=1)

    helper = specs.observable_decoder_spec(
        "observable",
        observable_builder,
        option_transform=lambda options, explicit: options | {"scale": len(explicit)},
    )
    observable_spec = helper(scale=3)
    assert observable_spec.options["scale"] == 1
    assert isinstance(
        observable_spec.build_observable_decoder(stim.DetectorErrorModel()),
        decoders.ObservableDecoder,
    )

    missing_input_builder: Any = lambda: decoders.custom.GUFDecoder(np.eye(1, dtype=int))
    with pytest.raises(TypeError, match="must accept a matrix or DEM"):
        specs.decoder_spec("missing_input", missing_input_builder)

    def variadic_builder(matrix: npt.NDArray[np.int_], *options: object) -> decoders.ErrorDecoder:
        del options
        return decoders.custom.GUFDecoder(matrix)

    assert isinstance(variadic_builder(np.eye(1, dtype=int), "option"), decoders.ErrorDecoder)
    variadic_helper = specs.decoder_spec("variadic", variadic_builder)
    with pytest.raises(TypeError, match="do not support variadic positional arguments"):
        variadic_helper("option")


def test_observable_decoder_specs() -> None:
    """A spec without an error builder builds observable decoders, but not error decoders."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    spec: decoders.DecoderSpec[Never] = decoders.DecoderSpec(
        "observable_lookup", None, (("max_weight", 1),), _get_observable_decoder_lookup
    )
    assert decoders.lookup(1).infers_errors
    assert not spec.infers_errors
    assert spec.predicts_observables_natively
    assert isinstance(spec.build_observable_decoder(dem), decoders.custom.ObservableLookupDecoder)
    with pytest.raises(TypeError, match="cannot build an error decoder"):
        spec.build(dem)
    with pytest.raises(TypeError, match="cannot build an error decoder"):
        _get_error_decoder(dem, decoder=spec)
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


def test_decoder_spec_helpers_build_decoders() -> None:
    """Every helper builds a working error decoder for a parity check matrix and a DEM."""
    specs: list[decoders.DecoderSpec[decoders.ErrorDecoder]] = [
        decoders.bp_osd(max_iter=5, osd_method="OSD_CS", osd_order=2),
        decoders.bp_lsd(max_iter=5, lsd_method="LSD_CS", lsd_order=2, always_run_lsd=True),
        decoders.bf(max_iter=5, uf_method="inversion"),
        decoders.mwpm(merge_strategy="independent"),
        decoders.relay_bp(gamma0=0.2),
        decoders.lookup(max_weight=1),
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
        (decoders.frontier, _get_observable_decoder_frontier, set()),
        (decoders.lookup, decoders.custom.LookupDecoder, {"predict_observable_flips"}),
        (decoders.guf, decoders.custom.GUFDecoder, set()),
        (decoders.ilp, decoders.custom.ILPDecoder, set()),
        (decoders.tesseract, decoders.external.TesseractDecoder, set()),
    ]
    for helper, constructor, excluded in qldpc_decoders:
        helper_defaults = get_defaults(helper)
        constructor_defaults = get_defaults(constructor)
        assert helper_defaults.keys() == constructor_defaults.keys() - excluded, helper
        for name, default in helper_defaults.items():
            assert default == constructor_defaults[name], (helper, name)

    entry_points: list[Callable[..., object]] = [
        decoders.custom.LookupDecoder,
        decoders.custom.ObservableLookupDecoder,
        decoders.lookup,
    ]
    for entry_point in entry_points:
        assert list(inspect.signature(entry_point).parameters)[-1] == "penalty_func"

    # helpers for relay-bp mirror the options of RelayBPDecoder (other than name, which the
    # precision selects) and of the relay_bp classes that they configure
    import relay_bp

    relay_bp_decoder_defaults = get_defaults(decoders.external.RelayBPDecoder)
    del relay_bp_decoder_defaults["name"]
    relay_bp_helpers: list[tuple[Callable[..., object], Any]] = [
        (decoders.relay_bp, relay_bp.RelayDecoderF32),
        (decoders.min_sum_bp, relay_bp.MinSumBPDecoderF32),
    ]
    for relay_bp_helper, relay_bp_class in relay_bp_helpers:
        helper_defaults = get_defaults(relay_bp_helper)
        assert helper_defaults.pop("precision") == "F32"
        assert helper_defaults.pop("backend_options") is None
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
        default_channel = [common.PLACEHOLDER_ERROR_RATE] * matrix.shape[1]
        ldpc_decoder = constructor(matrix, error_channel=default_channel)
        for attribute in shared_attributes + attributes:
            assert getattr(helper_decoder, attribute) == getattr(ldpc_decoder, attribute), (
                helper,
                attribute,
            )


def test_decoder_spec_helper_annotations() -> None:
    """Generated helper annotations match their public parameters and result."""
    helpers = (decoders.bp_osd, decoders.lookup, decoders.frontier)
    for helper in helpers:
        signature = inspect.signature(helper)
        annotations = typing.get_type_hints(helper)
        annotated_parameters = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.annotation is not inspect.Parameter.empty
        }
        assert annotations.keys() == annotated_parameters | {"return"}
        assert typing.get_origin(annotations["return"]) is decoders.DecoderSpec

    # the runtime annotations name the decoder that the specifications build, as type checkers infer
    expected_decoder_types: list[tuple[Callable[..., object], object]] = [
        (decoders.bp_osd, decoders.ErrorDecoder),
        (decoders.mwpm, decoders.BatchErrorDecoder),
        (decoders.relay_bp, decoders.external.RelayBPDecoder),
        (decoders.tesseract, decoders.external.TesseractDecoder),
        (decoders.lookup, decoders.custom.LookupDecoder),
        (decoders.guf, decoders.custom.GUFDecoder),
        (decoders.ilp, decoders.custom.ILPDecoder),
        (decoders.frontier, Never),
    ]
    for annotated_helper, decoder_type in expected_decoder_types:
        return_annotation = typing.get_type_hints(annotated_helper)["return"]
        assert typing.get_args(return_annotation) == (decoder_type,), annotated_helper
        assert inspect.signature(annotated_helper).return_annotation == return_annotation

    # a builder without a return annotation is assumed to build an error decoder
    def unannotated_builder(matrix: npt.NDArray[np.int_]):  # type: ignore[no-untyped-def]
        return decoders.custom.GUFDecoder(matrix)

    unannotated_helper = specs.decoder_spec("unannotated", unannotated_builder)
    assert typing.get_args(inspect.signature(unannotated_helper).return_annotation) == (
        decoders.ErrorDecoder,
    )
    assert isinstance(unannotated_helper().build(np.eye(1, dtype=int)), decoders.custom.GUFDecoder)


def test_decoder_spec_helper_docstrings() -> None:
    """Exported generated helpers must document their options and appear in the guide."""
    helpers = [
        (name, helper)
        for name in decoders.__all__
        if (helper := vars(decoders).get(name)) is not None
        and isinstance(getattr(helper, "__signature__", None), inspect.Signature)
        and typing.get_origin(getattr(helper, "__annotations__", {}).get("return"))
        is decoders.DecoderSpec
    ]
    assert helpers
    guide = (pathlib.Path(__file__).parents[4] / "docs/source/decoders.rst").read_text()
    migration = guide.split("Migrating from qLDPC 0.3.3", maxsplit=1)[1]
    for name, helper in helpers:
        assert callable(helper) and getattr(helper, "__name__", None) == name
        docstring = inspect.getdoc(helper)
        assert docstring is not None and docstring.startswith("Configure ")
        arguments = docstring.split("Args:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        documented_names = {
            line.strip().split(":", maxsplit=1)[0].lstrip("*")
            for line in arguments.splitlines()
            if line.startswith("    ") and not line.startswith("        ")
        }
        legacy_options = {"error_rate", "penalty_func"} & set(inspect.signature(helper).parameters)
        assert documented_names == set(inspect.signature(helper).parameters) - legacy_options
        assert ".. deprecated::" not in docstring
        assert all(option in migration for option in legacy_options)
        assert "A decoder specification." in docstring
        assert f".. autofunction:: qldpc.decoders.{name}\n" in guide

    # builders document their built decoders, and helpers independently describe specifications
    builders: list[tuple[Callable[..., object], Callable[..., object]]] = [
        (_get_decoder_bp_osd, decoders.bp_osd),
        (_get_decoder_lookup, decoders.lookup),
        (_get_observable_decoder_frontier, decoders.frontier),
    ]
    for builder, documented_helper in builders:
        builder_docstring = inspect.getdoc(builder)
        helper_docstring = inspect.getdoc(documented_helper)
        assert builder_docstring is not None and helper_docstring is not None
        assert builder_docstring.startswith("Build ")
        builder_returns = builder_docstring.split("Returns:\n", maxsplit=1)[1].split("\n\n")[0]
        helper_returns = helper_docstring.split("Returns:\n", maxsplit=1)[1].split("\n\n")[0]
        assert "decoder specification" not in builder_returns
        assert helper_returns.lstrip().startswith("A decoder specification.")
        assert "    pcm_or_dem:" not in helper_docstring


def test_helper_docstring_is_written_independently() -> None:
    """Builder details must not accidentally become the public helper's documentation."""
    docstring = "Configure a custom decoder.\n\nReturns:\n    A decoder specification."
    helper = specs.decoder_spec("custom", _get_decoder_bp_osd, doc=docstring)
    assert helper.__doc__ == docstring
    assert specs.decoder_spec("undocumented", _get_decoder_bp_osd).__doc__ is None


def test_deprecated_error_rate_option_is_last_and_warns() -> None:
    """Deferred helpers keep deprecated options last and translate them with a warning."""
    entry_points = (
        decoders.bp_osd,
        decoders.bp_lsd,
        decoders.bf,
        decoders.tesseract,
        decoders.tesseract_preset,
        _get_decoder_bp_osd,
        _get_decoder_bp_lsd,
        _get_decoder_bf,
        decoders.external.TesseractDecoder,
    )
    for entry_point in entry_points:
        assert list(inspect.signature(entry_point).parameters)[-1] == "error_rate"

    documented_entry_points: tuple[Callable[..., object], ...] = (
        _get_decoder_bp_osd,
        _get_decoder_bp_lsd,
        _get_decoder_bf,
        decoders.external.TesseractDecoder.__init__,
    )
    for documented_entry_point in documented_entry_points:
        docstring = inspect.getdoc(documented_entry_point)
        assert docstring is not None
        arguments = docstring.split("Args:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        documented_names = [
            line.strip().split(":", maxsplit=1)[0]
            for line in arguments.splitlines()
            if line.startswith("    ") and not line.startswith("        ")
        ]
        assert documented_names[-1] == "error_rate"

    with pytest.warns(DeprecationWarning, match="error_rate=0.2.*error_channel=0.2"):
        spec = decoders.bp_osd(error_rate=0.2)
    assert "error_rate" not in spec.options
    assert spec.options["error_channel"] == 0.2


@pytest.mark.parametrize(
    "preset",
    ["long-beam", "short-beam"],
)
@pytest.mark.parametrize("sparsify", [None, "surface-code-like", "color-code-like"])
def test_tesseract_presets(preset: Any, sparsify: Any) -> None:
    """Tesseract preset helpers accept every named family."""
    spec = decoders.tesseract_preset(preset, sparsify=sparsify)
    assert spec.infers_errors
    assert spec.predicts_observables_natively


def test_tesseract_preset_options() -> None:
    """Tesseract presets preserve qLDPC options and reject unknown selectors."""
    channel = np.array([0.1, 0.2])
    default = decoders.tesseract_preset()
    assert default.options == decoders.tesseract_preset("long-beam").options
    spec = decoders.tesseract_preset("short-beam", error_channel=channel, add_erasure_bit=True)
    assert "error_rate" not in spec.options
    assert spec.options["error_channel"] is channel
    assert spec.options["add_erasure_bit"] is True

    with pytest.warns(DeprecationWarning, match="error_rate=0.3.*error_channel=0.3"):
        deprecated_spec = decoders.tesseract_preset("short-beam", error_rate=0.3)
    assert deprecated_spec.options["error_channel"] == 0.3
    with pytest.raises(ValueError, match="cannot both be specified"):
        decoders.tesseract_preset("short-beam", error_channel=channel, error_rate=0.3)

    with pytest.raises(ValueError, match="Unknown Tesseract preset"):
        decoders.tesseract_preset("medium-beam")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown Tesseract sparsify preset"):
        decoders.tesseract_preset(sparsify="generic")  # type: ignore[arg-type]


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
    decoder = _get_observable_decoder(dem, decoder=spec)
    assert np.array_equal(decoder.decode_observables(np.array([1, 1, 1, 1])), [0])

    with pytest.raises(ValueError, match="cannot infer errors"):
        _get_error_decoder(dem, decoder=spec)
    with pytest.raises(ValueError, match="not supported with enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, decompose_errors=True)
    with pytest.raises(ValueError, match="not supported with enable_correlations=True"):
        decoders.mwpm(enable_correlations=True, ignore_non_graphlike_errors=True)
