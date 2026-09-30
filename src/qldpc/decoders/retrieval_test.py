# SPDX-License-Identifier: Apache-2.0

"""Unit tests for retrieval.py."""

from __future__ import annotations

import pickle
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

    with pytest.warns(DeprecationWarning, match="decoder_constructor.*decoder="):
        assert decoders.decode(matrix, syndrome, decoder_constructor=CustomDecoder) is error
    assert decoders.decode(matrix, syndrome, decoder=CustomDecoder) is error
    assert decoders.decode(matrix, syndrome, decoder=CustomDecoder(matrix)) is error

    # injected decoders are validated, which must survive `python -O`
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="must be callable"):
        decoders.get_decoder(matrix, decoder_constructor=0)
    with pytest.raises(TypeError, match="callable decode method"):
        decoders.get_decoder(matrix, decoder=lambda _: 0)  # type: ignore[arg-type]

    # the static_decoder argument has been removed, in favor of decoder=
    for decoder_args in [{"static_decoder": CustomDecoder(matrix)}, {"static_decoder": None}]:
        with pytest.raises(TypeError, match="static_decoder argument has been removed"):
            decoders.get_decoder(matrix, **decoder_args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Cannot combine decoder"):
        decoders.get_decoder(matrix, decoder=CustomDecoder(matrix), with_BF=True)


def test_decoder_selection() -> None:
    """Exactly one decoder can be requested at a time."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    # a falsy request is consumed rather than passed on to the requested decoder
    with pytest.warns(DeprecationWarning, match=r"decoder=decoders\.bf"):
        decoded_error = decoders.decode(matrix, syndrome, with_BF=True, with_MWPM=False)
    assert np.array_equal([1, 1], decoded_error)

    with (
        pytest.warns(DeprecationWarning, match="pass exactly one"),
        pytest.raises(ValueError, match="Only one decoder"),
    ):
        decoders.get_decoder(matrix, with_BF=True, with_MWPM=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decoders.decode(matrix, syndrome, with_BF=True)
    assert caught[0].filename == __file__
    assert "decoder=decoders.bf(...)" in str(caught[0].message)

    with pytest.warns(DeprecationWarning, match=r"decoder=decoders\.bp_osd"):
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


def test_decoder_specs() -> None:
    """Typed decoder specs defer construction and survive process serialization."""
    matrix = np.eye(2, dtype=int)
    syndrome = np.array([1, 0], dtype=int)

    spec = decoders.lookup_table(max_weight=1)
    restored = pickle.loads(pickle.dumps(spec))  # noqa: S301 - trusted in-memory round trip
    assert np.array_equal(decoders.decode(matrix, syndrome, decoder=restored), syndrome)

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
        (decoders.relay_bp, decoders.RelayBPDecoder, {"observable_error_matrix"}),
        (decoders.ilp, decoders.ILPDecoder, set()),
    ]
    for helper, constructor, excluded in qldpc_decoders:
        helper_defaults = get_defaults(helper)
        constructor_defaults = get_defaults(constructor)
        assert helper_defaults.keys() == constructor_defaults.keys() - excluded, helper
        for name, default in helper_defaults.items():
            assert default == constructor_defaults[name], (helper, name)

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


def test_reject_prebuilt_decoder() -> None:
    """Prebuilt decoders are rejected where a decoder must be built for a new matrix."""
    matrix = np.eye(2, dtype=int)
    prebuilt = decoders.LookupDecoder(matrix, max_weight=1)
    reason = "the matrix is new"
    for decoder in [None, decoders.lookup_table(max_weight=1), decoders.LookupDecoder]:
        retrieval._reject_prebuilt_decoder(decoder, reason)
    with pytest.raises(ValueError, match="cannot be passed as decoder= here because the matrix"):
        retrieval._reject_prebuilt_decoder(prebuilt, reason)


def test_invalid_explicit_decoder_inputs() -> None:
    """The explicit input rejects observable predictors and invalid factories."""
    matrix = np.eye(1, dtype=int)
    observable = decoders.ObservableLookupDecoder(
        stim.DetectorErrorModel("error(0.1) D0 L0"), max_weight=1
    )

    with pytest.raises(TypeError, match="observable flips rather than errors"):
        decoders.get_decoder(matrix, decoder=observable)  # type: ignore[arg-type]

    def observable_factory(_matrix: object) -> object:
        return observable

    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        decoders.get_decoder(
            matrix,
            decoder=observable_factory,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="DecoderSpec"):
        decoders.get_decoder(matrix, decoder=object())  # type: ignore[arg-type]


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
        decoder = decoders.get_decoder(matrix, decoder=decoder_spec)
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
            decoders.get_decoder(
                matrix,
                add_erasure_bit=True,
                **decoder_args,  # type: ignore[arg-type]
            )
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

    assert np.array_equal(error, decoders.decode(matrix, syndrome))  # default, BP+OSD
    for decoder in [
        decoders.bp_lsd(),
        decoders.bf(),
        decoders.relay_bp(),
        decoders.mwpm(),
        decoders.ilp(),
        decoders.guf(),
        decoders.lookup_table(max_weight=2),
    ]:
        assert np.array_equal(error, decoders.decode(matrix, syndrome, decoder=decoder))

    # default to GUF with non-binary fields
    field = galois.GF(3)
    matrix = matrix.view(field)
    syndrome = syndrome.view(field)
    error = error.view(field)
    assert np.array_equal(error, decoders.decode(matrix, syndrome))
    with pytest.warns(DeprecationWarning, match=r"decoder=decoders\.guf"):
        assert decoders.get_decoder(matrix, max_weight=1)

    # decode from a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    for decoder in [decoders.bp_lsd(), decoders.mwpm(), decoders.ilp(), decoders.guf()]:
        assert np.array_equal(error, decoders.decode(dem, syndrome, decoder=decoder))

    # a MWPM decoder built from a DEM takes its error weights from that DEM
    with pytest.raises(ValueError, match="Cannot set error weights"):
        decoders.get_decoder(dem, decoder=decoders.mwpm(weights=[1.0, 1.0]))

    # add a non-graphlike error mechanism, which MWPM can ignore upon request
    matrix = np.hstack([matrix, np.ones((3, 1))])
    error = np.concatenate([error, [0]])
    dem.append("error", 0.125, [stim.DemTarget.relative_detector_id(ii) for ii in range(3)])
    with pytest.raises(ValueError, match="non-graphlike error"):
        decoders.decode(dem, syndrome, decoder=decoders.mwpm())
    decoder = decoders.mwpm(ignore_non_graphlike_errors=True)
    assert np.array_equal(error, decoders.decode(dem, syndrome, decoder=decoder))


def test_non_graphlike_over_a_field() -> None:
    """An error is non-graphlike by the number of detectors it addresses, not by their sum."""
    matrix = galois.GF(2)([[1, 1], [1, 0], [1, 0]])  # column 0 addresses three detectors
    syndrome = np.array([1, 0, 0], dtype=int)

    with pytest.raises(ValueError, match="column 0 of the parity check matrix addresses 3"):
        decoders.decode(matrix, syndrome, decoder=decoders.mwpm())
    decoder = decoders.mwpm(ignore_non_graphlike_errors=True)
    assert np.array_equal([0, 1], decoders.decode(matrix, syndrome, decoder=decoder))
