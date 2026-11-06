# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deprecated keyword-based decoder construction."""

from __future__ import annotations

import functools
import pickle
import re
import types
import warnings
from collections.abc import Callable
from typing import Any, cast

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc._deprecations import decoders as deprecated_decoders
from qldpc.conftest import run_python
from qldpc.decoders.custom.guf import GUFDecoder
from qldpc.decoders.custom.ilp import ILPDecoder
from qldpc.decoders.custom.lookup import ErrorLookupDecoder, LookupDecoder, ObservableLookupDecoder


def test_decoder_selection() -> None:
    """Exactly one decoder can be requested at a time."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    # a falsy request is consumed rather than passed on to the requested decoder
    with pytest.warns(DeprecationWarning, match=r"decoders\.bf\(\.\.\.\)\.build"):
        decoded_error = deprecated_decoders.decode(matrix, syndrome, with_BF=True, with_MWPM=False)
    assert np.array_equal([1, 1], decoded_error)

    with (
        pytest.warns(DeprecationWarning, match="get_decoder is deprecated"),
        pytest.raises(ValueError, match="Only one decoder"),
    ):
        deprecated_decoders.get_decoder(matrix, with_BF=True, with_MWPM=True)

    with pytest.warns(DeprecationWarning, match=r"decoders\.bp_osd\(\.\.\.\)\.build"):
        deprecated_decoders.get_decoder(matrix, max_iter=1)

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = deprecated_decoders.get_decoder(
            dem,
            with_lookup=True,
            max_weight=1,
            predict_observable_flips=True,
        )
    messages = [str(warning.message) for warning in caught]
    assert any(
        "decoders.lookup(max_weight=...).build_observable_decoder(dem)" in message
        for message in messages
    )
    assert np.array_equal(decoder.decode(np.array([1], dtype=int)), [1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert deprecated_decoders.decode(
            dem,
            np.array([1], dtype=int),
            with_lookup=True,
            max_weight=1,
            predict_observable_flips=True,
        ).tolist() == [1]
    assert any(
        "decoders.lookup(max_weight=...).build_observable_decoder(dem).decode_observables(syndrome)"
        in str(warning.message)
        for warning in caught
    )

    with pytest.warns(
        DeprecationWarning,
        match=r"ObservableLookupDecoder\(matrix, max_weight=.*decode_observables",
    ):
        matrix_decoder = deprecated_decoders.get_decoder(
            np.array([[1]], dtype=int),
            with_lookup=True,
            max_weight=1,
            observable_flip_matrix=np.array([[1]], dtype=int),
            error_channel=[0.1],
            predict_observable_flips=True,
        )
    assert matrix_decoder.decode(np.array([1], dtype=int)).tolist() == [1]


def test_deprecated_decoder_functions() -> None:
    """The deprecated get_decoder and decode functions behave as they did, and name replacements."""
    matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    error = np.array([1, 0, 0, 1])
    syndrome = (matrix @ error) % 2

    class CustomDecoder:
        def __init__(self, matrix: npt.NDArray[np.int_], scale: int = 1) -> None:
            self.scale = scale

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.scale * np.asarray(error)

    # warnings name the replacing call, and point at the caller
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_decoders.get_decoder(matrix)
        deprecated_decoders.get_decoder(matrix, decoder_constructor=CustomDecoder, scale=2)
        deprecated_decoders.decode(matrix, syndrome, with_lookup=True, max_weight=1)
    assert [str(warning.message) for warning in caught] == [
        "decoders.get_decoder is deprecated; use decoders.bp_osd().build(pcm_or_dem) instead",
        "decoders.get_decoder is deprecated; use CustomDecoder(pcm_or_dem, ...) instead",
        (
            "decoders.decode is deprecated; use"
            " decoders.lookup(...).build(pcm_or_dem).decode(syndrome) instead"
        ),
    ]
    assert all(warning.filename == __file__ for warning in caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # a decoder constructor receives the remaining arguments, and is returned as is
        decoder = deprecated_decoders.get_decoder(
            matrix, decoder_constructor=CustomDecoder, scale=2
        )
        assert isinstance(decoder, CustomDecoder) and decoder.scale == 2
        with pytest.raises(TypeError, match="must be callable"):
            deprecated_decoders.get_decoder(matrix, decoder_constructor=0)

        # a static decoder is returned as is, and admits no other arguments
        static_decoder = CustomDecoder(matrix)
        assert (
            deprecated_decoders.get_decoder(matrix, static_decoder=static_decoder) is static_decoder
        )
        assert (
            deprecated_decoders.get_decoder(matrix, static_decoder=CustomDecoder) is CustomDecoder
        )
        with pytest.raises(ValueError, match="cannot process decoding arguments"):
            deprecated_decoders.get_decoder(matrix, static_decoder=static_decoder, with_BF=True)

        # the default decoder depends on the field
        decoder = deprecated_decoders.get_decoder(galois.GF(3)(matrix))
        assert isinstance(decoder, GUFDecoder)

    with pytest.warns(DeprecationWarning, match=r"use static_decoder\.decode\(syndrome\)"):
        decoded_error = deprecated_decoders.decode(matrix, syndrome, static_decoder=static_decoder)
    assert np.array_equal(decoded_error, error)

    # the static_decoder argument has been removed from other methods, in favor of decoder=
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        deprecated_decoders.resolve_decoder(matrix, None, {"static_decoder": static_decoder})
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        deprecated_decoders.resolve_observable_decoder(
            dem, None, {"static_decoder": static_decoder}
        )
    with pytest.raises(ValueError, match="Cannot combine decoder"):
        deprecated_decoders.resolve_decoder(matrix, static_decoder, {"with_BF": True})


def test_legacy_flat_backend_options() -> None:
    """Deprecated APIs move flat backend options into the backend_options of a builder."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    with pytest.warns(DeprecationWarning):
        decoder: Any = deprecated_decoders.get_decoder_bp_osd(
            matrix,
            osd_method="OSD_CS",
            input_vector_type="syndrome",
            backend_options={"osd_order": 2},
        )
    assert decoder.osd_order == 2
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Unknown parameter"):
        deprecated_decoders.get_decoder(matrix, with_BP_OSD=True, unsupported_option=1)

    # add_erasure_bit is not a backend option, so unsupported decoders still reject it clearly
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="cannot signal erasure"),
    ):
        deprecated_decoders.get_decoder(matrix, with_BF=True, add_erasure_bit=True)

    # MWPM rejects flat options that PyMatching would silently ignore
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match=r"\['typo'\]"):
        deprecated_decoders.get_decoder_mwpm(matrix, typo=1)
    with pytest.warns(DeprecationWarning):
        matching: Any = deprecated_decoders.get_decoder_mwpm(matrix, merge_strategy="disallow")
    assert np.array_equal(matching.decode(np.array([1, 0])), [1, 0, 0])

    # builders without backend_options receive their keyword arguments unchanged, and the legacy
    # constructors remain pickleable for Sinter
    pickled = pickle.dumps(deprecated_decoders._get_decoder_constructors()["BP_OSD"])
    constructor = pickle.loads(pickled)  # noqa: S301
    assert np.array_equal(constructor(matrix).decode(np.array([1, 0])), [1, 0, 0])
    with pytest.warns(DeprecationWarning):
        decoder = deprecated_decoders.get_decoder(matrix, with_lookup=True, max_weight=1)
    assert np.array_equal(decoder.decode(np.array([1, 0])), [1, 0, 0])


def test_deprecated_builders() -> None:
    """Deprecated builders warn about replacement specifications and build the same decoder."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    syndrome = np.array([1, 0])
    builders: list[tuple[Callable[..., Any], str, dict[str, object]]] = [
        (deprecated_decoders.get_decoder_bp_osd, "decoders.bp_osd(...).build(pcm_or_dem)", {}),
        (deprecated_decoders.get_decoder_bp_lsd, "decoders.bp_lsd(...).build(pcm_or_dem)", {}),
        (deprecated_decoders.get_decoder_bf, "decoders.bf(...).build(pcm_or_dem)", {}),
        (deprecated_decoders.get_decoder_guf, "decoders.guf(...).build(pcm_or_dem)", {}),
        (
            deprecated_decoders.get_decoder_lookup,
            "decoders.lookup(...).build(pcm_or_dem)",
            {"max_weight": 1},
        ),
        (deprecated_decoders.get_decoder_mwpm, "decoders.mwpm(...).build(pcm_or_dem)", {}),
        (
            deprecated_decoders.get_decoder_rbp,
            (
                "decoders.relay_bp(...).build(pcm_or_dem)"
                " or decoders.min_sum_bp(...).build(pcm_or_dem)"
            ),
            {},
        ),
    ]
    for builder, replacement, options in builders:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decoder = builder(matrix, **options)
        assert [str(warning.message) for warning in caught] == [
            f"decoders.{builder.__name__} is deprecated; use {replacement} instead"
        ]
        assert caught[0].filename == __file__
        assert np.array_equal(decoder.decode(syndrome), [1, 0, 0])

    with pytest.warns(DeprecationWarning, match=r"decoders\.ilp\(\.\.\.\)\.build"):
        decoder = deprecated_decoders.get_decoder_ilp(matrix)
    assert isinstance(decoder, ILPDecoder)

    # additional options are forwarded to the backend, which rejects unsupported names
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Unknown parameter"):
        deprecated_decoders.get_decoder_bp_osd(matrix, unsupported_backend_option=True)
    with pytest.warns(DeprecationWarning):
        bp_osd_decoder: Any = deprecated_decoders.get_decoder_bp_osd(matrix, max_iter=7)
    assert bp_osd_decoder.max_iter == 7


def test_deprecated_lookup_builders() -> None:
    """Deprecated lookup builders build an error lookup decoder, or one that predicts flips."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0 L1")
    syndrome = np.array([1], dtype=int)

    # an explicit predict_observable_flips=False is dropped, and builds an error lookup decoder
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder: Any = deprecated_decoders.get_decoder(
            dem, with_lookup=True, max_weight=1, predict_observable_flips=False
        )
    assert len(caught) == 1  # only get_decoder warns
    assert type(decoder) is ErrorLookupDecoder
    assert np.array_equal(decoder.decode(syndrome), [1])

    # predict_observable_flips=True builds a legacy lookup decoder that predicts observable flips
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder = deprecated_decoders.get_decoder_lookup(
            dem, max_weight=1, predict_observable_flips=True
        )
    flag_warnings = [w for w in caught if "predict_observable_flips=True" in str(w.message)]
    assert len(flag_warnings) == 1
    assert "build_observable_decoder" in str(flag_warnings[0].message)
    assert all(warning.filename == __file__ for warning in caught)
    assert np.array_equal(decoder.decode(syndrome), [1, 1])
    assert isinstance(decoder, LookupDecoder)

    # Sinter pickles the legacy lookup constructor
    pickled = pickle.dumps(deprecated_decoders._get_decoder_constructors()["lookup"])
    constructor = pickle.loads(pickled)  # noqa: S301
    assert np.array_equal(constructor(dem, max_weight=1).decode(syndrome), [1])


def test_deprecated_resolution_functions() -> None:
    """Deprecated resolution functions warn with the call that replaces them for each input."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")

    def build_lookup(pcm_or_dem: decoders.PcmOrDem) -> ErrorLookupDecoder:
        return ErrorLookupDecoder(pcm_or_dem, max_weight=1)

    def build_observable_lookup(
        dem: stim.DetectorErrorModel,
    ) -> ObservableLookupDecoder:
        return ObservableLookupDecoder(dem, max_weight=1)

    def get_message(function: Callable[..., Any], *args: object, **kwargs: object) -> str:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = function(*args, **kwargs)
        assert len(caught) == 1 and caught[0].filename == __file__
        name = function.__name__
        assert str(caught[0].message).startswith(f"decoders.{name} is deprecated; use ")
        if name == "get_error_decoder":
            assert np.array_equal(result.decode_errors(np.array([1, 0])), [1, 0, 0])
        elif name == "get_observable_decoder":
            assert np.array_equal(result.decode_observables(np.array([1, 0])), [1])
        else:
            assert np.array_equal(result, [1])
        return str(caught[0].message).split("; use ", maxsplit=1)[1]

    error_cases: list[tuple[decoders.ErrorDecoderInput, str]] = [
        (decoders.bf(max_iter=2), "decoders.bf(max_iter=2).build(pcm_or_dem)"),
        (None, "decoders.bp_osd().build(pcm_or_dem)"),
        (build_lookup(matrix), "the prebuilt decoder directly"),
        (build_lookup, "build_lookup(pcm_or_dem)"),
        (functools.partial(build_lookup), "decoder_constructor(pcm_or_dem)"),
    ]
    for decoder_input, replacement in error_cases:
        message = get_message(deprecated_decoders.get_error_decoder, matrix, decoder=decoder_input)
        assert message == f"{replacement} instead"
    with pytest.warns(DeprecationWarning, match=r"decoders\.guf\(\)\.build"):
        decoder = deprecated_decoders.get_error_decoder(galois.GF(3)(matrix))
    assert isinstance(decoder, GUFDecoder)

    observable_cases: list[tuple[decoders.DecoderInput, str | None, str]] = [
        (decoders.lookup(1), "decoders.lookup(max_weight=1).build_observable_decoder(dem)", ""),
        (None, "decoders.bp_osd().build_observable_decoder(dem)", ""),
        (
            decoders.SinterDecoder(decoder=decoders.lookup(1)),
            "decoder.compile_decoder_for_dem(dem)",
            "",
        ),
        (build_observable_lookup(dem), None, ""),
        (build_lookup(dem), "decoders.ErrorsToObservablesDecoder(decoder, dem)", ""),
        (
            build_lookup,
            "build_lookup(dem)",
            (
                " (if it builds an error decoder, use"
                " decoders.ErrorsToObservablesDecoder(build_lookup(dem), dem))"
            ),
        ),
        (
            build_observable_lookup,
            "build_observable_lookup(dem)",
            (
                " (if it builds an error decoder, use"
                " decoders.ErrorsToObservablesDecoder(build_observable_lookup(dem), dem))"
            ),
        ),
        (decoders.from_dem(build_observable_lookup), "decoder.build(dem)", ""),
        (
            functools.partial(build_observable_lookup),
            "decoder_constructor(dem)",
            (
                " (if it builds an error decoder, use"
                " decoders.ErrorsToObservablesDecoder(decoder_constructor(dem), dem))"
            ),
        ),
    ]
    for observable_input, expression, note in observable_cases:
        message = get_message(
            deprecated_decoders.get_observable_decoder, dem, decoder=observable_input
        )
        if expression is None:
            assert message == "the prebuilt observable decoder directly instead"
        else:
            assert message == f"{expression}{note} instead"
        message = get_message(
            deprecated_decoders.decode_observables, dem, np.array([1, 0]), decoder=observable_input
        )
        if expression is None:
            assert message == (
                "the decode_observables method of the prebuilt observable decoder instead"
            )
        else:
            assert message == f"{expression}.decode_observables(syndrome){note} instead"

    with (
        pytest.warns(DeprecationWarning, match="a factory that accepts a detector error model"),
        pytest.raises(ValueError, match="needs a parity-check matrix"),
    ):
        deprecated_decoders.get_observable_decoder(dem, decoder=decoders.from_matrix(build_lookup))

    # an invalid input still names a replacement before it is rejected
    with (
        pytest.warns(DeprecationWarning, match=r"decoders\.bp_osd\(\.\.\.\)\.build"),
        pytest.raises(TypeError, match="decoder must be"),
    ):
        deprecated_decoders.get_error_decoder(matrix, decoder=cast(Any, 1))
    with (
        pytest.warns(DeprecationWarning, match=r"decoders\.mwpm\(\.\.\.\)\.build"),
        pytest.raises(TypeError, match="decoder must be"),
    ):
        deprecated_decoders.get_observable_decoder(dem, decoder=cast(Any, 1))


def test_legacy_decoder_migration_messages() -> None:
    """Deprecated keyword arguments of methods warn with the decoder input that replaces them."""
    matrix = np.eye(2, dtype=int)
    expected_messages: list[tuple[dict[str, object], str]] = [
        ({"decoder_constructor": ErrorLookupDecoder}, "for example decoder=ErrorLookupDecoder"),
        (
            {"with_lookup": True, "predict_observable_flips": True},
            r"ObservableLookupDecoder\(matrix, .*observable_flip_matrix=",
        ),
        ({"with_BF": True}, r"with_BF keyword .* use decoder=decoders\.bf\(\.\.\.\)"),
        ({"with_BF": True, "with_MWPM": True}, "pass exactly one"),
        ({"max_iter": 5}, r"move them into decoder=decoders\.bp_osd\(\.\.\.\)"),
    ]
    for decoder_args, expected_message in expected_messages:
        message = deprecated_decoders.get_legacy_decoder_migration_message(matrix, decoder_args)
        assert re.search(expected_message, message), message
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    assert "decoders.lookup(max_weight=...).build_observable_decoder(dem)" in (
        deprecated_decoders.get_legacy_decoder_migration_message(
            dem, {"with_lookup": True, "predict_observable_flips": True}
        )
    )
    assert "decoder=decoders.lookup(max_weight=...)" in (
        deprecated_decoders.get_legacy_decoder_migration_message(
            None, {"with_lookup": True, "predict_observable_flips": True}
        )
    )
    message = deprecated_decoders.get_legacy_decoder_migration_message(
        galois.GF(3)(matrix), {"max_weight": 1}, argument_name="decoder_x"
    )
    assert "decoder_x=decoders.guf(...)" in message

    # deprecated arguments build a decoder, warning unless an outer API has already warned
    with pytest.warns(DeprecationWarning, match="with_BF keyword"):
        decoder = deprecated_decoders.resolve_decoder(matrix, None, {"with_BF": True})
    assert np.array_equal(decoder.decode_errors(np.array([1, 0])), [1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deprecated_decoders.resolve_decoder(matrix, None, {"with_BF": True}, warn_deprecated=False)
        deprecated_decoders.resolve_decoder(matrix, decoders.bf(), {})

    # deprecated arguments build an observable decoder with error-decoder semantics
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")
    observable_decoder = deprecated_decoders.resolve_observable_decoder(
        dem, None, {"with_lookup": True, "max_weight": 1}, warn_deprecated=False
    )
    assert isinstance(observable_decoder, decoders.ErrorsToObservablesDecoder)
    assert np.array_equal(observable_decoder.decode_observables(np.array([1, 0])), [1])
    assert np.array_equal(observable_decoder.decode_observables(np.array([1, 1])), [0])

    # without deprecated arguments, decoder specifications may build a native observable decoder
    observable_decoder = deprecated_decoders.resolve_observable_decoder(dem, decoders.lookup(1), {})
    assert isinstance(observable_decoder, ObservableLookupDecoder)


def test_merge_legacy_css_decoder_args() -> None:
    """Deprecated arguments of the two CSS sectors warn per argument, and translate per sector."""
    matrix = galois.GF2([[1, 1, 0], [0, 1, 1]])
    merge = deprecated_decoders.merge_legacy_css_decoder_args

    def decode(decoder_input: decoders.DecoderInput) -> npt.NDArray[np.int_]:
        assert callable(decoder_input)
        decoder: Any = decoder_input(matrix)
        return decoder.decode_errors(np.array([1, 0]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder_x, decoder_z = merge(
            matrix, matrix, None, None, {"with_lookup": True, "max_weight": 1}, None, None
        )
    assert ["decoder=decoders.lookup(...)" in str(warning.message) for warning in caught] == [True]
    assert decoder_x is decoder_z  # so that the sectors can share one decoder
    assert np.array_equal(decode(decoder_x), [1, 0, 0])

    # sector-specific arguments override shared ones
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoder_x, decoder_z = merge(
            matrix,
            matrix,
            None,
            None,
            {"with_lookup": True, "max_weight": 1},
            None,
            {"max_weight": 0},
        )
    assert len(caught) == 2
    assert sum("decoder_z=" in str(warning.message) for warning in caught) == 1
    assert np.array_equal(decode(decoder_x), [1, 0, 0])
    assert np.array_equal(decode(decoder_z), [0, 0, 0])

    # sector-specific arguments that hold equal arrays also share a translated decoder input
    sector_args = [{"with_lookup": True, "max_weight": 1, "error_channel": np.full(3, 0.1)}]
    sector_args.append({**sector_args[0], "error_channel": np.full(3, 0.1)})
    with pytest.warns(DeprecationWarning):
        decoder_x, decoder_z = merge(matrix, matrix, None, None, {}, *sector_args)
    assert decoder_x is decoder_z

    # a sector without deprecated arguments keeps its decoder input
    spec = decoders.lookup(1)
    with pytest.warns(DeprecationWarning, match="decoder_x="):
        decoder_x, decoder_z = merge(matrix, matrix, None, spec, {}, {"with_BF": True}, None)
    assert np.array_equal(decode(decoder_x), [1, 0, 0]) and decoder_z is spec

    # deprecated arguments cannot be combined with a decoder input or the removed static_decoder
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Cannot combine"):
        merge(matrix, matrix, None, spec, {}, None, {"with_BF": True})
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        merge(matrix, matrix, None, None, {}, {"static_decoder": spec}, None)


def test_import_order() -> None:
    """Importing this module before the rest of qLDPC does not break the import of qLDPC."""
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.find_spec('qldpc')\n"
        "package = importlib.util.module_from_spec(spec)\n"
        "sys.modules['qldpc'] = package\n"
        "import qldpc._deprecations.decoders\n"
        "spec.loader.exec_module(package)\n"
    )
    run_python(code)


def test_deprecate_error_rate_option() -> None:
    """Deferred options translate explicit error_rate values and reject ambiguity."""
    options: dict[str, object] = {"error_channel": None, "error_rate": None}
    assert deprecated_decoders.deprecate_error_rate_option(options, frozenset()) == {
        "error_channel": None
    }

    with pytest.warns(DeprecationWarning, match="error_rate=0.2.*error_channel=0.2") as caught:
        translated = deprecated_decoders.deprecate_error_rate_option(
            {"error_channel": None, "error_rate": 0.2},
            frozenset({"error_rate"}),
        )
    assert caught[0].filename == __file__
    assert translated == {"error_channel": 0.2}

    with pytest.raises(ValueError, match="cannot both be specified"):
        deprecated_decoders.deprecate_error_rate_option(
            {"error_channel": 0.1, "error_rate": 0.2},
            frozenset({"error_channel", "error_rate"}),
        )

    # an explicit error_rate=None is equivalent to omitting it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        translated = deprecated_decoders.deprecate_error_rate_option(
            {"error_channel": 0.1, "error_rate": None},
            frozenset({"error_channel", "error_rate"}),
        )
        assert translated == {"error_channel": 0.1}
        for helper in (decoders.bp_osd, decoders.tesseract):
            assert helper(error_rate=None).options == helper().options
            assert helper(error_rate=None, error_channel=0.1).options["error_channel"] == 0.1
        assert (
            decoders.tesseract_preset(error_rate=None, error_channel=0.1).options["error_channel"]
            == 0.1
        )


def test_decode_returns_observables() -> None:
    """Decoders whose decode methods return observable flips are identified, warning for others."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    with pytest.warns(DeprecationWarning, match="predict_observable_flips=True"):
        legacy_decoders = [
            LookupDecoder(dem, max_weight=1, predict_observable_flips=True),
            decoders.custom.WeightedLookupDecoder(dem, max_weight=1, predict_observable_flips=True),
        ]
    for legacy_decoder in legacy_decoders:
        assert deprecated_decoders.decode_returns_observables(legacy_decoder)

    # neither classes, nor error decoders, nor objects without the declaration return observables
    for decoder in [type(legacy_decoders[0]), ErrorLookupDecoder(dem, max_weight=1), object()]:
        assert not deprecated_decoders.decode_returns_observables(decoder)

    # other objects that declare decode_returns_observables=True use a deprecated convention
    declaring_decoder = types.SimpleNamespace(decode_returns_observables=True)
    with pytest.warns(DeprecationWarning, match="implement decode_observables") as caught:
        assert deprecated_decoders.decode_returns_observables(declaring_decoder)
    assert caught[0].filename == __file__
    assert deprecated_decoders.decode_returns_observables(declaring_decoder, warn=False)


def test_selects_legacy_decoder() -> None:
    """Deprecated keyword arguments select a decoder by decoder_constructor or with_<NAME>."""
    assert deprecated_decoders.selects_legacy_decoder({"with_RBP": True, "seed": 1})
    assert deprecated_decoders.selects_legacy_decoder({"decoder_constructor": print})
    assert not deprecated_decoders.selects_legacy_decoder({"with_RBP": False, "seed": 1})
