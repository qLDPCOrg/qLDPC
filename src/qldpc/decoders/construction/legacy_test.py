# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deprecated keyword-based decoder construction."""

from __future__ import annotations

import functools
import pickle
import re
import unittest.mock
import warnings
from collections.abc import Callable
from typing import Any, cast

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.construction import legacy
from qldpc.decoders.external import ldpc as ldpc_integration


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


def test_legacy_flat_backend_options() -> None:
    """Deprecated APIs move flat backend options into the backend_options of a builder."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    with (
        warnings.catch_warnings(),
        unittest.mock.patch.object(ldpc_integration, "_build_ldpc_decoder") as backend,
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        decoders.get_decoder_bp_osd(
            matrix, max_iter=7, backend_extension=1, backend_options={"other": 2}
        )
        expected_options = {"max_iter": 7, "backend_extension": 1, "other": 2}
        assert expected_options.items() <= backend.call_args.args[3].items()
        decoders.get_decoder(matrix, with_BF=True, backend_extension=3)
        assert backend.call_args.args[3]["backend_extension"] == 3

    # add_erasure_bit is not a backend option, so unsupported decoders still reject it clearly
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="cannot signal erasure"),
    ):
        decoders.get_decoder(matrix, with_BF=True, add_erasure_bit=True)

    # MWPM rejects flat options that PyMatching would silently ignore
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match=r"\['typo'\]"):
        decoders.get_decoder_mwpm(matrix, typo=1)
    with pytest.warns(DeprecationWarning):
        matching: Any = decoders.get_decoder_mwpm(matrix, merge_strategy="disallow")
    assert np.array_equal(matching.decode(np.array([1, 0])), [1, 0, 0])

    # builders without backend_options receive their keyword arguments unchanged, and the legacy
    # constructors remain pickleable for Sinter
    constructor = legacy.DECODER_CONSTRUCTORS["BP_OSD"]
    assert isinstance(constructor, functools.partial)
    assert pickle.loads(pickle.dumps(constructor)).func is constructor.func  # noqa: S301
    with pytest.warns(DeprecationWarning):
        decoder = decoders.get_decoder(matrix, with_lookup=True, max_weight=1)
    assert np.array_equal(decoder.decode(np.array([1, 0])), [1, 0, 0])


def test_deprecated_builders() -> None:
    """Deprecated builders warn with the settings that replace them, and build the same decoder."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    syndrome = np.array([1, 0])
    builders: list[tuple[Callable[..., Any], str, dict[str, object]]] = [
        (decoders.get_decoder_bp_osd, "decoders.bp_osd(...).build(pcm_or_dem)", {}),
        (decoders.get_decoder_bp_lsd, "decoders.bp_lsd(...).build(pcm_or_dem)", {}),
        (decoders.get_decoder_bf, "decoders.bf(...).build(pcm_or_dem)", {}),
        (decoders.get_decoder_guf, "decoders.guf(...).build(pcm_or_dem)", {}),
        (decoders.get_decoder_lookup, "decoders.lookup(...).build(pcm_or_dem)", {"max_weight": 1}),
        (decoders.get_decoder_mwpm, "decoders.mwpm(...).build(pcm_or_dem)", {}),
        (
            decoders.get_decoder_rbp,
            "decoders.relay_bp(...).build(pcm_or_dem) or decoders.min_sum_bp(...).build(pcm_or_dem)",
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
        decoder = decoders.get_decoder_ilp(matrix)
    assert isinstance(decoder, decoders.ILPDecoder)

    # additional options are forwarded to the backend, which rejects unsupported names
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Unknown parameter"):
        decoders.get_decoder_bp_osd(matrix, unsupported_backend_option=True)
    with pytest.warns(DeprecationWarning):
        bp_osd_decoder: Any = decoders.get_decoder_bp_osd(matrix, max_iter=7)
    assert bp_osd_decoder.max_iter == 7

    # the deprecated uppercase aliases resolve to the deprecated lowercase builders
    with pytest.warns(DeprecationWarning, match="get_decoder_BP_OSD"):
        assert decoders.get_decoder_BP_OSD is decoders.get_decoder_bp_osd


def test_deprecated_resolution_functions() -> None:
    """Deprecated resolution functions warn with the call that replaces them for each input."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")

    def build_lookup(pcm_or_dem: decoders.PcmOrDem) -> decoders.LookupDecoder:
        return decoders.LookupDecoder(pcm_or_dem, max_weight=1)

    def build_observable_lookup(
        dem: stim.DetectorErrorModel,
    ) -> decoders.ObservableLookupDecoder:
        return decoders.ObservableLookupDecoder(dem, max_weight=1)

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
        message = get_message(decoders.get_error_decoder, matrix, decoder=decoder_input)
        assert message == f"{replacement} instead"
    with pytest.warns(DeprecationWarning, match=r"decoders\.guf\(\)\.build"):
        decoder = decoders.get_error_decoder(galois.GF(3)(matrix))
    assert isinstance(decoder, decoders.GUFDecoder)

    observable_cases: list[tuple[decoders.DecoderInput, str | None]] = [
        (decoders.lookup(1), "decoders.lookup(max_weight=1).build_observable_decoder(dem)"),
        (None, "decoders.bp_osd().build_observable_decoder(dem)"),
        (
            decoders.SinterDecoder(decoder=decoders.lookup(1)),
            "decoder.compile_decoder_for_dem(dem)",
        ),
        (build_observable_lookup(dem), None),
        (build_lookup(dem), "decoders.ErrorsToObservablesDecoder(decoder, dem)"),
        (build_lookup, "decoders.ErrorsToObservablesDecoder(build_lookup(dem), dem)"),
        (build_observable_lookup, "build_observable_lookup(dem)"),
    ]
    for observable_input, expression in observable_cases:
        message = get_message(decoders.get_observable_decoder, dem, decoder=observable_input)
        if expression is None:
            assert message == "the prebuilt observable decoder directly instead"
        else:
            assert message == f"{expression} instead"
        message = get_message(
            decoders.decode_observables, dem, np.array([1, 0]), decoder=observable_input
        )
        if expression is None:
            assert message == (
                "the decode_observables method of the prebuilt observable decoder instead"
            )
        else:
            assert message == f"{expression}.decode_observables(syndrome) instead"

    # a partial constructor is classified by the constructor that it wraps, and a constructor that
    # does not declare what it builds gets advice for both kinds of decoder
    def build_unannotated(dem: stim.DetectorErrorModel):  # type: ignore[no-untyped-def]
        return decoders.ObservableLookupDecoder(dem, max_weight=1)

    def build_unresolvable(dem: stim.DetectorErrorModel) -> decoders.ObservableLookupDecoder:
        return decoders.ObservableLookupDecoder(dem, max_weight=1)

    build_unresolvable.__annotations__["return"] = "UndefinedDecoder"
    wrapped = "decoders.ErrorsToObservablesDecoder({}(dem), dem)"
    constructor_cases: list[tuple[Callable[..., object], str, str]] = [
        (functools.partial(build_observable_lookup), "decoder_constructor(dem)", ""),
        (
            functools.partial(decoders.LookupDecoder, max_weight=1),
            wrapped.format("decoder_constructor"),
            "",
        ),
        (
            build_unannotated,
            "build_unannotated(dem)",
            f" (if it builds an error decoder, use {wrapped.format('build_unannotated')})",
        ),
        (
            build_unresolvable,
            "build_unresolvable(dem)",
            f" (if it builds an error decoder, use {wrapped.format('build_unresolvable')})",
        ),
    ]
    for constructor, expression, note in constructor_cases:
        message = get_message(decoders.get_observable_decoder, dem, decoder=constructor)
        assert message == f"{expression}{note} instead"
        message = get_message(
            decoders.decode_observables, dem, np.array([1, 0]), decoder=constructor
        )
        assert message == f"{expression}.decode_observables(syndrome){note} instead"

    # an invalid input still names a replacement before it is rejected
    with (
        pytest.warns(DeprecationWarning, match=r"decoders\.bp_osd\(\.\.\.\)\.build"),
        pytest.raises(TypeError, match="decoder must be"),
    ):
        decoders.get_error_decoder(matrix, decoder=cast(Any, 1))
    with (
        pytest.warns(DeprecationWarning, match=r"decoders\.mwpm\(\.\.\.\)\.build"),
        pytest.raises(TypeError, match="decoder must be"),
    ):
        decoders.get_observable_decoder(dem, decoder=cast(Any, 1))

    # the renamed lookup_table helper is a deprecated alias of lookup
    with pytest.warns(DeprecationWarning, match="lookup_table is deprecated; use lookup") as caught:
        assert decoders.lookup_table is decoders.lookup
    assert caught[0].filename == __file__

    # internal resolution, used by methods that accept decoder=, does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.resolve_decoder(matrix, decoders.bf(), {})
        decoders.resolve_observable_decoder(dem, None, {})


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
        message = legacy.get_legacy_decoder_migration_message(matrix, decoder_args)
        assert re.search(expected_message, message), message
    message = legacy.get_legacy_decoder_migration_message(
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

    # deprecated arguments build an observable decoder with error-decoder semantics
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")
    observable_decoder = decoders.resolve_observable_decoder(
        dem, None, {"with_lookup": True, "max_weight": 1}, warn_deprecated=False
    )
    assert isinstance(observable_decoder, decoders.ErrorsToObservablesDecoder)
    assert np.array_equal(observable_decoder.decode_observables(np.array([1, 0])), [1])
    assert np.array_equal(observable_decoder.decode_observables(np.array([1, 1])), [0])

    # without deprecated arguments, decoder settings may build a native observable decoder
    observable_decoder = decoders.resolve_observable_decoder(dem, decoders.lookup(1), {})
    assert isinstance(observable_decoder, decoders.ObservableLookupDecoder)
