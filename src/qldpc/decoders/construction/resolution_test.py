# SPDX-License-Identifier: Apache-2.0

"""Unit tests for decoder input resolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.adapters import error_decoders
from qldpc.decoders.construction import resolution
from qldpc.decoders.external.pymatching import MatchingObservableDecoder


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
        (decoders.mwpm(), MatchingObservableDecoder),
        (decoders.relay_bp(), decoders.RelayBPDecoder),
        (decoders.min_sum_bp(gamma0=0.5), decoders.RelayBPDecoder),
        (decoders.lookup_table(max_weight=2), decoders.ObservableLookupDecoder),
    ]
    for spec, native_decoder_type in native_decoder_types:
        assert spec.predicts_observables_natively
        native_decoder: Any = decoders.get_observable_decoder(dem, decoder=spec)
        assert isinstance(native_decoder, native_decoder_type)
        converted_decoder = error_decoders.ErrorsToObservablesDecoder(spec.build(dem), dem)
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
    assert isinstance(bp_osd_decoder, error_decoders.ErrorsToObservablesDecoder)
    assert isinstance(
        decoders.get_observable_decoder(dem), error_decoders.ErrorsToObservablesDecoder
    )
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

    # invalid inputs
    with pytest.raises(TypeError, match="decoder must be decoder settings"):
        decoders.get_observable_decoder(dem, decoder=object())  # type: ignore[arg-type]

    def build_invalid_decoder(dem: stim.DetectorErrorModel) -> Any:
        return object()

    spec = decoders.DecoderSpec("custom", decoders.get_decoder_lookup, (), build_invalid_decoder)
    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        spec.build_observable_decoder(dem)


def test_observable_decoder_compilers() -> None:
    """An observable-decoder compiler, such as a SinterDecoder, is compiled for the given model."""
    dem, syndromes = _get_circuit_data()
    expected_flips = decoders.ObservableLookupDecoder(dem, max_weight=2).decode_observables_batch(
        syndromes
    )
    lookup_compiler = decoders.SinterDecoder(decoder=decoders.lookup_table(max_weight=2))
    assert not decoders.is_prebuilt_decoder(lookup_compiler)

    observable_decoder = decoders.get_observable_decoder(dem, decoder=lookup_compiler)
    assert isinstance(observable_decoder, decoders.CompiledSinterDecoder)
    assert not observable_decoder.has_erasure_bit
    assert np.array_equal(
        [observable_decoder.decode_observables(syndrome) for syndrome in syndromes],
        expected_flips,
    )
    assert np.array_equal(
        decoders.decode_observables(dem, syndromes[0], decoder=lookup_compiler), expected_flips[0]
    )

    # a nested compiler is compiled for each detector error model that the outer decoder decodes
    nested_compiler = decoders.SinterDecoder(decoder=lookup_compiler)
    nested_decoder = nested_compiler.compile_decoder_for_dem(dem)
    assert isinstance(nested_decoder.decoder, decoders.CompiledSinterDecoder)
    assert np.array_equal(nested_decoder.decode_shots(syndromes.astype(np.uint8)), expected_flips)

    # a compiled decoder that only decodes bit-packed shots signals discards with an erasure bit
    class BitPackedCompiler:
        def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> object:
            return _BitPackedOnly(lookup_compiler.compile_decoder_for_dem(dem))

    bit_packed_decoder = decoders.get_observable_decoder(
        dem,
        decoder=BitPackedCompiler(),  # type: ignore[arg-type]
    )
    assert bit_packed_decoder.has_erasure_bit  # type: ignore[attr-defined]
    for syndrome, flips in zip(syndromes, expected_flips):
        assert np.array_equal(bit_packed_decoder.decode_observables(syndrome), [*flips, 0])

    class InvalidCompiler:
        def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> object:
            return object()

    with pytest.raises(TypeError, match="compiled by compile_decoder_for_dem must provide"):
        decoders.get_observable_decoder(dem, decoder=InvalidCompiler())  # type: ignore[arg-type]


class _BitPackedOnly:
    """A compiled decoder that only exposes decode_shots_bit_packed."""

    def __init__(self, compiled_decoder: decoders.CompiledSinterDecoder) -> None:
        self.decode_shots_bit_packed = compiled_decoder.decode_shots_bit_packed


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
        merging_decoder = decoders.get_decoder_guf(dem, add_erasure_bit=add_erasure_bit)
        assert len(merging_decoder.decode(syndromes[0])) == 2 + add_erasure_bit
        decoder: Any = error_decoders.match_error_decoder_to_dem(merging_decoder, dem)
        assert isinstance(decoder, error_decoders.ExpandedErrorDecoder)
        errors = decoder.decode_batch(syndromes)
        assert errors.shape == (2, 3 + add_erasure_bit)
        assert np.array_equal(errors, [decoder.decode(syndrome) for syndrome in syndromes])
        assert np.array_equal(errors[:, [0, 1]].sum(axis=1), [1, 0])  # one of the merged errors
        assert decoder.decode_batch(syndromes[:0]).shape == (0, 3 + add_erasure_bit)

    # matching decoders merge equivalent mechanisms, and decode in batches
    decoder = error_decoders.match_error_decoder_to_dem(decoders.get_decoder_mwpm(dem), dem)
    assert isinstance(decoder, error_decoders.ExpandedErrorDecoder)
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
        lambda dem: decoders.get_decoder_mwpm(dem, decompose_errors=True),
    ]
    for decoder_input in decoder_inputs:
        with pytest.raises(ValueError, match="components of decomposed error mechanisms"):
            decoders.get_observable_decoder(dem, decoder=decoder_input)

    # decomposition that splits no error leaves the error mechanisms of a model intact
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1")
    decoder = decoders.get_observable_decoder(
        dem, decoder=lambda dem: decoders.get_decoder_mwpm(dem, decompose_errors=True)
    )
    assert np.array_equal(decoder.decode_observables(syndrome), [1])


def test_reject_prebuilt_decoder() -> None:
    """Prebuilt decoders are rejected where a decoder must be built for a new matrix."""
    matrix = np.eye(2, dtype=int)
    prebuilt = decoders.LookupDecoder(matrix, max_weight=1)
    reason = "the matrix is new"
    for decoder in [None, decoders.lookup_table(max_weight=1), decoders.LookupDecoder]:
        resolution.reject_prebuilt_decoder(decoder, reason)
    with pytest.raises(ValueError, match="cannot be passed as decoder= here because the matrix"):
        resolution.reject_prebuilt_decoder(prebuilt, reason)


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
        (decoders.get_decoder_rbp, decoders.relay_bp(add_erasure_bit=True)),
        (decoders.get_decoder_ilp, decoders.ilp(add_erasure_bit=True)),
        (decoders.get_decoder_guf, decoders.guf(add_erasure_bit=True)),
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
        (decoders.get_decoder_bf, {"with_BF": True}, "BF"),
        (decoders.get_decoder_bp_osd, {"with_BP_OSD": True}, "BP_OSD"),
        (decoders.get_decoder_mwpm, {"with_MWPM": True}, "MWPM"),
        (decoders.get_decoder_bp_lsd, {"with_BP_LSD": True}, "BP_LSD"),
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


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    assert np.array_equal(
        error, decoders.get_error_decoder(matrix).decode(syndrome)
    )  # default, BP+OSD
    assert np.array_equal(
        error, decoders.get_error_decoder(matrix.astype(np.int32)).decode(syndrome)
    )  # ldpc itself rejects int32 matrices
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
