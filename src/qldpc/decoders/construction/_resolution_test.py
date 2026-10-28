# SPDX-License-Identifier: Apache-2.0

"""Unit tests for decoder input resolution."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc._deprecations import decoders as deprecated_decoders
from qldpc.decoders._capabilities import is_prebuilt_decoder
from qldpc.decoders.adapters import error_decoders
from qldpc.decoders.construction import _resolution
from qldpc.decoders.construction._resolution import build_error_decoder, build_observable_decoder
from qldpc.decoders.custom.guf import GUFDecoder
from qldpc.decoders.external.pymatching import MatchingObservableDecoder


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)

    assert np.array_equal(error, build_error_decoder(matrix).decode(syndrome))  # default, BP+OSD

    # default to GUF with non-binary fields
    field = galois.GF(3)
    matrix = matrix.view(field)
    syndrome = syndrome.view(field)
    error = error.view(field)
    decoder: decoders.ErrorDecoder = build_error_decoder(matrix)
    assert isinstance(decoder, GUFDecoder)
    assert np.array_equal(error, decoder.decode(syndrome))

    # add a non-graphlike error mechanism to a detector error model, which MWPM can ignore upon
    # request
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    error = np.concatenate([error, [0]])
    dem.append("error", 0.125, [stim.DemTarget.relative_detector_id(ii) for ii in range(3)])
    with pytest.raises(ValueError, match="non-graphlike error"):
        build_error_decoder(dem, decoder=decoders.mwpm()).decode(syndrome)
    decoder = build_error_decoder(dem, decoder=decoders.mwpm(ignore_non_graphlike_errors=True))
    assert np.array_equal(error, decoder.decode(syndrome))


def test_custom_decoder() -> None:
    """Inject custom decoders."""
    matrix = np.array([[1, 1], [0, 1]])
    error = np.array([1, 0])
    syndrome = np.array([1, 0])

    class CustomDecoder(decoders.ErrorDecoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return error

    # a subclass of ErrorDecoder is accepted as a constructor or as a prebuilt decoder
    assert build_error_decoder(matrix, decoder=CustomDecoder).decode(syndrome) is error
    assert (
        build_error_decoder(matrix, decoder=CustomDecoder(matrix)).decode_errors(syndrome) is error
    )

    # an object with only a decode method is wrapped to provide decode_errors
    class BareDecoder:
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return error

    bare_decoder = BareDecoder()
    wrapped_decoder: Any = build_error_decoder(matrix, decoder=bare_decoder)
    assert wrapped_decoder.decoder is bare_decoder
    assert wrapped_decoder.decode_errors(syndrome) is error

    # injected decoders are validated, which must survive `python -O`
    with pytest.raises(TypeError, match="must be an ErrorDecoder, or have a decode method"):
        build_error_decoder(matrix, decoder=lambda _: 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="decoder must be a decoder specification"):
        build_error_decoder(matrix, decoder=0)  # type: ignore[arg-type]


def test_explicit_factory_inputs() -> None:
    """Factory wrappers specify both the input representation and output contract."""
    matrix = galois.GF(3)([[1, 1]])
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    seen_matrices: list[galois.FieldArray] = []
    seen_dems: list[stim.DetectorErrorModel] = []

    def build_errors(matrix: galois.FieldArray) -> decoders.ErrorDecoder:
        seen_matrices.append(matrix)
        return decoders.custom.ErrorLookupDecoder(matrix, max_weight=1)

    def build_observables(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        seen_dems.append(dem)
        return decoders.custom.ObservableLookupDecoder(dem, max_weight=1)

    matrix_factory = decoders.from_matrix(build_errors)
    dem_factory = decoders.from_dem(build_observables)
    assert build_error_decoder(matrix, decoder=matrix_factory).decode_errors(
        np.array([1])
    ).shape == (2,)
    assert len(seen_matrices) == 1 and seen_matrices[0] is matrix
    assert np.array_equal(
        build_observable_decoder(dem, decoder=dem_factory).decode_observables(np.array([1])), [1]
    )
    assert len(seen_dems) == 1 and seen_dems[0] is dem
    with pytest.raises(ValueError, match="needs a parity-check matrix"):
        build_error_decoder(dem, decoder=matrix_factory)
    with pytest.raises(ValueError, match="needs a parity-check matrix"):
        build_observable_decoder(dem, decoder=matrix_factory)
    with pytest.raises(TypeError, match="predicts observables, but this workflow needs inferred"):
        build_error_decoder(matrix, decoder=dem_factory)  # type: ignore[arg-type]

    def build_invalid_decoder(dem: stim.DetectorErrorModel) -> Any:
        return object()

    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        build_observable_decoder(dem, decoder=decoders.from_dem(build_invalid_decoder))
    with pytest.raises(TypeError, match="must be an ErrorDecoder"):
        build_error_decoder(matrix, decoder=decoders.from_matrix(build_invalid_decoder))


def test_invalid_explicit_decoder_inputs() -> None:
    """The explicit input rejects observable predictors and invalid factories."""
    matrix = np.eye(1, dtype=int)
    observable = decoders.custom.ObservableLookupDecoder(
        stim.DetectorErrorModel("error(0.1) D0 L0"), max_weight=1
    )

    with pytest.raises(TypeError, match="observable flips rather than errors"):
        build_error_decoder(matrix, decoder=observable)  # type: ignore[arg-type]

    def observable_factory(_matrix: object) -> object:
        return observable

    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        build_error_decoder(
            matrix,
            decoder=observable_factory,  # type: ignore[arg-type]
        )


def test_reject_prebuilt_decoder() -> None:
    """Prebuilt decoders are rejected where a decoder must be built for a new matrix."""
    matrix = np.eye(2, dtype=int)
    prebuilt = decoders.custom.ErrorLookupDecoder(matrix, max_weight=1)
    reason = "the matrix is new"
    for decoder in [None, decoders.lookup(max_weight=1), decoders.custom.ErrorLookupDecoder]:
        _resolution.reject_prebuilt_decoder(decoder, reason)
    with pytest.raises(ValueError, match="cannot be passed as decoder= here because the matrix"):
        _resolution.reject_prebuilt_decoder(prebuilt, reason)


def test_error_decoder_output_is_validated() -> None:
    """A decoder whose .decode returns observable flips is not mistaken for an error decoder."""
    import pymatching

    dem, _ = _get_circuit_data()

    # a matching decoder built from a DEM returns observable flips from its decode method
    with pytest.raises(ValueError, match="inferred an error of length 1"):
        build_observable_decoder(
            dem, decoder=lambda dem: pymatching.Matching.from_detector_error_model(dem)
        )


def test_erasure_bit_request() -> None:
    """A request for an erasure bit is rejected by a decoder that cannot signal erasure."""
    matrix = np.eye(3, 2, dtype=int)

    # a decoder that can signal erasure honours routed requests
    erasing_specs: list[decoders.DecoderSpec[decoders.ErrorDecoder]] = [
        decoders.relay_bp(add_erasure_bit=True),
        decoders.ilp(add_erasure_bit=True),
        decoders.guf(add_erasure_bit=True),
        decoders.lookup(max_weight=2, add_erasure_bit=True),
    ]
    for decoder_spec in erasing_specs:
        decoder = build_error_decoder(matrix, decoder=decoder_spec)
        assert getattr(decoder, "has_erasure_bit", False)

    # every decoder that cannot signal erasure rejects routed requests; the typed helpers for these
    # decoders do not accept an add_erasure_bit argument at all
    unerasing_decoders: list[tuple[dict[str, object], str]] = [
        ({"with_BF": True}, "BF"),
        ({"with_BP_OSD": True}, "BP_OSD"),
        ({"with_MWPM": True}, "MWPM"),
        ({"with_BP_LSD": True}, "BP_LSD"),
    ]
    for decoder_args, decoder_name in unerasing_decoders:
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match=rf"The {decoder_name} decoder cannot signal erasure"),
        ):
            deprecated_decoders.get_decoder(matrix, add_erasure_bit=True, **decoder_args)

    # BP+OSD is the default for a binary matrix
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match=r"The BP_OSD decoder cannot signal erasure"),
    ):
        deprecated_decoders.get_decoder(matrix, add_erasure_bit=True)


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
    assert np.array_equal(
        build_observable_decoder(dem, decoder=spec).decode_observables(syndrome), [0]
    )

    # an error decoder that infers decomposed errors cannot predict observable flips
    decoder_inputs: list[decoders.DecoderInput] = [
        spec.build(dem),
        lambda dem: decoders.mwpm(decompose_errors=True).build(dem),
    ]
    for decoder_input in decoder_inputs:
        with pytest.raises(ValueError, match="components of decomposed error mechanisms"):
            build_observable_decoder(dem, decoder=decoder_input)

    # decomposition that splits no error leaves the error mechanisms of a model intact
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1")
    decoder = build_observable_decoder(
        dem, decoder=lambda dem: decoders.mwpm(decompose_errors=True).build(dem)
    )
    assert np.array_equal(decoder.decode_observables(syndrome), [1])


def test_native_observable_decoders() -> None:
    """Decoders that predict observable flips natively agree with ones that convert errors."""
    dem, syndromes = _get_circuit_data()
    native_decoder_types: list[tuple[decoders.DecoderSpec[Any], type[Any]]] = [
        (decoders.mwpm(), MatchingObservableDecoder),
        (decoders.relay_bp(), decoders.external.RelayBPDecoder),
        (decoders.min_sum_bp(gamma0=0.5), decoders.external.RelayBPDecoder),
        (decoders.lookup(max_weight=2), decoders.custom.ObservableLookupDecoder),
    ]
    for spec, native_decoder_type in native_decoder_types:
        assert spec.predicts_observables_natively
        native_decoder: Any = build_observable_decoder(dem, decoder=spec)
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
            build_observable_decoder(dem, decoder=spec).decode_observables(syndromes[0]),
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
    assert isinstance(build_observable_decoder(dem), error_decoders.ErrorsToObservablesDecoder)
    no_flips = bp_osd_decoder.decode_observables_batch(syndromes[:0])
    assert no_flips.shape == (0, dem.num_observables)

    # native observable decoders support detector error models without observables
    dem_without_observables = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.1) D0
        error(0.1) D1
    """)
    specs: list[decoders.DecoderSpec[Any]] = [decoders.relay_bp(), decoders.lookup(1)]
    for spec in specs:
        observable_decoder = build_observable_decoder(dem_without_observables, decoder=spec)
        assert observable_decoder.decode_observables(np.array([1, 0])).shape == (0,)

    # an erasure bit of an error decoder is appended to predicted observable flips
    erasing_decoder = build_observable_decoder(dem, decoder=decoders.guf(add_erasure_bit=True))
    assert getattr(erasing_decoder, "has_erasure_bit", False)
    assert erasing_decoder.decode_observables(syndromes[0]).shape == (dem.num_observables + 1,)


def test_observable_decoder_inputs() -> None:
    """Observable decoders are built from specifications, constructors, and prebuilt decoders."""
    dem, syndromes = _get_circuit_data()
    observable_lookup = decoders.custom.ObservableLookupDecoder(dem, max_weight=2)
    error_lookup = decoders.custom.ErrorLookupDecoder(dem, max_weight=2)
    expected_flips = observable_lookup.decode_observables_batch(syndromes)

    # prebuilt decoders, and constructors of either kind of decoder
    decoder_inputs: list[decoders.DecoderInput] = [
        observable_lookup,
        error_lookup,
        lambda dem: decoders.custom.ObservableLookupDecoder(dem, max_weight=2),
        lambda dem: decoders.custom.ErrorLookupDecoder(dem, max_weight=2),
    ]
    for decoder_input in decoder_inputs:
        observable_decoder: Any = build_observable_decoder(dem, decoder=decoder_input)
        assert np.array_equal(
            observable_decoder.decode_observables_batch(syndromes), expected_flips
        )
    assert build_observable_decoder(dem, decoder=observable_lookup) is observable_lookup

    # invalid inputs
    with pytest.raises(TypeError, match="decoder must be a decoder specification"):
        build_observable_decoder(dem, decoder=object())  # type: ignore[arg-type]

    def build_invalid_decoder(dem: stim.DetectorErrorModel) -> Any:
        return object()

    spec = decoders.DecoderSpec(
        "custom", decoders.custom.ErrorLookupDecoder, (), build_invalid_decoder
    )
    with pytest.raises(TypeError, match="must provide a decode_observables method"):
        spec.build_observable_decoder(dem)


def test_observable_decoder_compilers() -> None:
    """An observable-decoder compiler, such as a SinterDecoder, is compiled for the given model."""
    dem, syndromes = _get_circuit_data()
    expected_flips = decoders.custom.ObservableLookupDecoder(
        dem, max_weight=2
    ).decode_observables_batch(syndromes)
    lookup_compiler = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=2))
    assert not is_prebuilt_decoder(lookup_compiler)

    observable_decoder = build_observable_decoder(dem, decoder=lookup_compiler)
    assert isinstance(observable_decoder, decoders.sinter.CompiledSinterDecoder)
    assert not observable_decoder.has_erasure_bit
    assert np.array_equal(
        [observable_decoder.decode_observables(syndrome) for syndrome in syndromes],
        expected_flips,
    )
    assert np.array_equal(
        build_observable_decoder(dem, decoder=lookup_compiler).decode_observables(syndromes[0]),
        expected_flips[0],
    )

    # a nested compiler is compiled for each detector error model that the outer decoder decodes
    nested_compiler = decoders.SinterDecoder(decoder=lookup_compiler)
    nested_decoder = nested_compiler.compile_decoder_for_dem(dem)
    assert isinstance(nested_decoder.decoder, decoders.sinter.CompiledSinterDecoder)
    assert np.array_equal(nested_decoder.decode_shots(syndromes.astype(np.uint8)), expected_flips)

    # a compiled decoder that only decodes bit-packed shots signals discards with an erasure bit
    class BitPackedCompiler:
        def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> object:
            return _BitPackedOnly(lookup_compiler.compile_decoder_for_dem(dem))

    # this compiler builds an untyped object, and the protocols do not declare has_erasure_bit
    bit_packed_decoder = build_observable_decoder(
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
        build_observable_decoder(dem, decoder=InvalidCompiler())  # type: ignore[arg-type]


def _get_circuit_data() -> tuple[stim.DetectorErrorModel, npt.NDArray[np.int_]]:
    """A repetition-code memory experiment's detector error model and sampled syndromes."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.02
    )
    syndromes = circuit.compile_detector_sampler(seed=0).sample(200).astype(int)
    return circuit.detector_error_model(), syndromes


class _BitPackedOnly:
    """A compiled decoder that only exposes decode_shots_bit_packed."""

    def __init__(self, compiled_decoder: decoders.sinter.CompiledSinterDecoder) -> None:
        self.decode_shots_bit_packed = compiled_decoder.decode_shots_bit_packed
