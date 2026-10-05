# SPDX-License-Identifier: Apache-2.0

"""Unit tests for code_capacity.py."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import warnings
from typing import Any, Never

import galois
import numpy as np
import numpy.typing as npt
import pytest
import sinter
import stim

from qldpc import codes, decoders
from qldpc.codes import code_capacity
from qldpc.decoders.adapters import observable_decoders
from qldpc.decoders.external.pymatching import MatchingObservableDecoder


class _FixedObservableDecoder(decoders.ObservableDecoder):
    """Observable decoder that returns a fixed output, and records the syndromes it decodes."""

    def __init__(
        self,
        output: npt.ArrayLike,
        has_erasure_bit: bool = False,
        *,
        num_detectors: int = 1,
        num_observables: int | None = None,
    ) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit
        self.num_detectors = num_detectors
        self.num_observables = (
            len(self.output) - int(has_erasure_bit) if num_observables is None else num_observables
        )
        self.syndromes: list[npt.NDArray[np.int_]] = []

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        self.syndromes.append(syndrome)
        return self.output


class _FixedErrorDecoder(decoders.ErrorDecoder):
    """Error decoder that returns a fixed output."""

    def __init__(self, output: npt.ArrayLike, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output


class _BitPackedSinterDecoder(sinter.Decoder):
    """Sinter decoder whose compiled form has only the bit-packed methods that Sinter requires."""

    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.dems: list[stim.DetectorErrorModel] = []

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> Any:
        self.dems.append(dem)
        return _BitPackedCompiledDecoder(self.packed_prediction)


class _BitPackedCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.packed_syndromes: list[npt.NDArray[np.uint8]] = []

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        self.packed_syndromes.append(bit_packed_detection_event_data)
        return self.packed_prediction


def test_get_code_capacity_dem() -> None:
    """A code-capacity DEM has one error mechanism per column of the syndrome matrix."""
    syndrome_matrix = galois.GF2([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    observable_matrix = galois.GF2([[1, 0, 0, 1]])
    dem = code_capacity.get_code_capacity_dem(syndrome_matrix, observable_matrix)
    assert dem.num_detectors == 3  # an untriggered detector is still declared
    assert dem.num_observables == 1
    assert dem.num_errors == 4

    # each error mechanism flips the detectors and observables in its columns, with a probability
    # that does not depend on the physical error rate
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), syndrome_matrix)
    assert np.array_equal(dem_arrays.observable_flip_matrix.toarray(), observable_matrix)
    assert len(set(dem_arrays.error_probs)) == 1

    # error mechanisms may be combinations of error locations
    dem_errors = galois.GF2([[1, 0], [0, 1], [0, 1], [0, 1]])
    dem = code_capacity.get_code_capacity_dem(syndrome_matrix, observable_matrix, dem_errors)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), syndrome_matrix @ dem_errors)
    assert np.array_equal(
        dem_arrays.observable_flip_matrix.toarray(), observable_matrix @ dem_errors
    )

    # symplectic X, Z, and Y mechanisms are constructed directly, with caller-supplied probabilities
    syndrome_matrix = galois.GF2([[1, 0, 0, 1]])
    observable_matrix = galois.GF2([[0, 1, 1, 0]])
    error_probs = np.arange(1, 7) / 100
    dem = code_capacity.get_code_capacity_dem(
        syndrome_matrix,
        observable_matrix,
        symplectic_errors=True,
        error_probs=error_probs,
    )
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), [[1, 0, 0, 1, 1, 1]])
    assert np.array_equal(dem_arrays.observable_flip_matrix.toarray(), [[0, 1, 1, 0, 1, 1]])
    assert np.array_equal(dem_arrays.error_probs, error_probs)
    with pytest.raises(ValueError, match="requires an observable_matrix"):
        code_capacity.get_code_capacity_dem(syndrome_matrix, None, symplectic_errors=True)
    with pytest.raises(ValueError, match="cannot both be specified"):
        code_capacity.get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            galois.GF2.Identity(4),
            symplectic_errors=True,
        )

    # Stim detector error models are binary
    field = galois.GF(3)
    with pytest.raises(ValueError, match="cannot decode a code over GF"):
        code_capacity.get_code_capacity_dem(field(syndrome_matrix), field(observable_matrix))


def test_code_capacity_decoder_from_error_decoder() -> None:
    """An error decoder predicts the observable values of the errors that it infers."""
    field = galois.GF(3)
    syndrome_matrix = field([[1, 2, 0]])
    observable_matrix = field([[1, 1, 1], [0, 2, 1]])

    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([2, 1, 0])
    )
    assert not decoder.can_discard
    observables, erased = decoder.decode(field([1]))
    assert isinstance(observables, field) and not erased
    assert np.array_equal(observables, observable_matrix @ field([2, 1, 0]))

    # decoding fails if and only if the predicted observable values differ from the true ones
    assert decoder.get_failure_and_erasure(field([2, 1, 0])) == (False, False)
    assert decoder.get_failure_and_erasure(field([2, 1, 1])) == (True, False)
    assert decoder.get_failure_and_erasure(field([1, 0, 2])) == (False, False)  # same logical class

    # an erasure bit is passed through, and an erased sample is not a failure
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([0, 0, 0, 1], has_erasure_bit=True)
    )
    assert decoder.can_discard
    assert decoder.get_failure_and_erasure(field([2, 1, 1])) == (False, True)

    # malformed errors are rejected with actionable messages
    malformed_outputs: list[tuple[list[Any], str]] = [
        ([0, 0], r"inferred an error of shape \(2,\), but expected shape \(3,\)"),
        ([0, 0, 3], "not elements of GF"),
        ([0.0, 0.0, 0.0], "expected integers"),
    ]
    for output, message in malformed_outputs:
        decoder = code_capacity.get_code_capacity_decoder(
            syndrome_matrix, observable_matrix, _FixedErrorDecoder(output)
        )
        with pytest.raises(ValueError, match=message):
            decoder.decode(field([0]))
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([0, 0, 0, 2], has_erasure_bit=True)
    )
    with pytest.raises(ValueError, match="erasure flags that are not 0 or 1"):
        decoder.decode(field([0]))

    # binary specifications with a native observable builder use the sector's detector error model
    decoder = code_capacity.get_code_capacity_decoder(
        galois.GF2([[1, 1]]), galois.GF2([[1, 0]]), decoders.lookup(max_weight=1)
    )
    assert isinstance(decoder.decoder, decoders.custom.ObservableLookupDecoder)

    # nonbinary codes cannot use a Stim DEM, but can still infer field-valued errors
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, decoders.lookup(max_weight=1)
    )
    assert isinstance(decoder.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert isinstance(decoder.decoder.error_decoder, decoders.custom.LookupDecoder)

    # None represents an identity observable map without materializing a dense identity matrix
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, None, _FixedErrorDecoder([2, 1, 0])
    )
    assert decoder.observable_matrix is None and decoder.num_observables == 3
    assert decoder.reuse_for(syndrome_matrix, None) is decoder
    assert decoder.get_failure_and_erasure(field([2, 1, 0])) == (False, False)


def test_code_capacity_decoder_from_observable_decoder() -> None:
    """A prebuilt observable decoder is used as is, and its predictions are validated."""
    field = galois.GF(3)
    syndrome_matrix = field([[1, 2, 0]])
    observable_matrix = field([[1, 1, 1], [0, 2, 1]])

    observable_decoder = _FixedObservableDecoder([2, 1])
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, observable_decoder
    )
    assert decoder.decoder is observable_decoder
    error = field([1, 0, 1])  # its observable values are [2, 1]
    assert decoder.get_failure_and_erasure(error) == (False, False)
    assert np.array_equal(observable_decoder.syndromes[-1], syndrome_matrix @ error)
    assert type(observable_decoder.syndromes[-1]) is np.ndarray
    assert decoder.get_failure_and_erasure(field([1, 0, 0])) == (True, False)

    # an erasure flag discards a sample
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedObservableDecoder([0, 0, 1], True)
    )
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(error) == (False, True)

    # a prediction of the wrong shape or field is rejected
    for output, message in [
        ([0, 0, 0], r"shape \(3,\), but expected shape \(2,\): 2 value\(s\)"),
        ([0, 5], "not elements of GF"),
    ]:
        decoder = code_capacity.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder(output, num_observables=2),
        )
        with pytest.raises(ValueError, match=message):
            decoder.decode(field([0]))

    # a decoder that infers errors and predicts observable flips is used as an error decoder
    class _BothDecoder(_FixedErrorDecoder):
        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            raise NotImplementedError  # pragma: no cover

    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _BothDecoder([1, 0, 1])
    )
    assert isinstance(decoder.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # an object with both decode and decode_observables keeps its legacy error-decoder semantics
    class _DecodeOnlyBoth:
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.array([1, 0, 1])

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            raise AssertionError("decode_observables should not be called")  # pragma: no cover

    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _DecodeOnlyBoth()
    )
    assert isinstance(decoder.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # the ObservableDecoder protocol does not require optional dimension metadata
    class _NoDimensions:
        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.array([2, 1])

    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _NoDimensions()
    )
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # prebuilt observable decoders are rejected where a caller asks for that
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        code_capacity.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder([0, 0]),
            prebuilt_rejection_reason="it is a test",
        )

    # deprecated keyword arguments cannot be combined with an observable decoder
    with pytest.raises(ValueError, match="Cannot combine decoder="):
        code_capacity.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder([0, 0]),
            {"with_lookup": True},
        )


def test_code_capacity_error_and_observable_output_contracts() -> None:
    """An ambiguous syndrome can have distinct error and observable predictions."""
    field = galois.GF2
    syndrome_matrix = field([[1, 1]])
    observable_matrix = field([[1, 0]])
    errors = [field([1, 0]), field([0, 1])]
    assert all(np.array_equal(syndrome_matrix @ error, [1]) for error in errors)
    assert [int((observable_matrix @ error)[0]) for error in errors] == [1, 0]

    projected = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([1, 0])
    )
    native = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedObservableDecoder([0])
    )
    assert isinstance(projected.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert native.decoder.decode_observables(np.array([1])).tolist() == [0]
    assert [projected.get_failure_and_erasure(error) for error in errors] == [
        (False, False),
        (True, False),
    ]
    assert [native.get_failure_and_erasure(error) for error in errors] == [
        (True, False),
        (False, False),
    ]

    erasing = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([1, 0, 1], True)
    )
    prediction, erased = erasing.decode(field([1]))
    assert erasing.num_erasure_flags == 1
    assert prediction.tolist() == [1] and erased
    assert erasing.get_failure_and_erasure(errors[1]) == (False, True)
    with pytest.raises(ValueError, match=r"expected shape \(3,\)"):
        code_capacity.get_code_capacity_decoder(
            syndrome_matrix, observable_matrix, _FixedErrorDecoder([1, 0], True)
        ).decode(field([1]))


def test_code_capacity_dem_mechanism_order_and_native_prediction() -> None:
    """DEM columns keep the caller's error order, including tied syndromes."""
    field = galois.GF2
    syndrome_matrix = field([[1, 1, 0], [0, 0, 1]])
    observable_matrix = field([[1, 0, 1], [0, 1, 1]])
    dem_errors = field([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    dem = code_capacity.get_code_capacity_dem(
        syndrome_matrix,
        observable_matrix,
        dem_errors,
        error_probs=np.array([0.12, 0.04, 0.06]),
    )
    arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(arrays.detector_flip_matrix.toarray(), [[1, 1, 0], [0, 0, 1]])
    assert np.array_equal(arrays.observable_flip_matrix.toarray(), [[0, 1, 1], [1, 0, 1]])
    assert np.allclose(arrays.error_probs, [0.12, 0.04, 0.06])

    native = decoders.lookup(max_weight=1).build_observable_decoder(dem)
    decoder = code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, native)
    errors = [field([0, 1, 0]), field([1, 0, 0]), field([0, 0, 1])]
    assert [decoder.get_failure_and_erasure(error) for error in errors] == [
        (False, False),
        (True, False),
        (False, False),
    ]


def test_code_capacity_prefers_most_likely_observable_class() -> None:
    """Native decoding can outperform selecting one representative physical error."""
    field = galois.GF2
    syndrome_matrix = field([[1, 1, 1]])
    observable_matrix = field([[0, 0, 1]])
    errors = [field([1, 0, 0]), field([0, 1, 0]), field([0, 0, 1])]
    assert all(np.array_equal(syndrome_matrix @ error, [1]) for error in errors)
    assert [int((observable_matrix @ error)[0]) for error in errors] == [0, 0, 1]

    specification = decoders.lookup(max_weight=1)
    inferred_error = specification.build(syndrome_matrix).decode_errors(np.array([1]))
    assert inferred_error.tolist() == [0, 0, 1]
    native = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, specification
    )
    projected = code_capacity.get_code_capacity_decoder(
        syndrome_matrix,
        observable_matrix,
        decoders.from_matrix(specification.build),
    )
    assert isinstance(native.decoder, decoders.custom.ObservableLookupDecoder)
    assert isinstance(projected.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert native.decode(field([1]))[0].tolist() == [0]
    assert projected.decode(field([1]))[0].tolist() == [1]
    assert [native.get_failure_and_erasure(error) for error in errors] == [
        (False, False),
        (False, False),
        (True, False),
    ]
    assert [projected.get_failure_and_erasure(error) for error in errors] == [
        (True, False),
        (True, False),
        (False, False),
    ]

    with pytest.warns(DeprecationWarning, match="with_lookup"):
        legacy = code_capacity.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            None,
            {"with_lookup": True, "max_weight": 1},
        )
    assert isinstance(legacy.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert legacy.decode(field([1]))[0].tolist() == [1]

    # The DEM's relative mechanism weights affect the native class prediction.
    weighted = code_capacity.get_code_capacity_decoder(
        syndrome_matrix,
        observable_matrix,
        specification,
        dem_error_weights=np.array([1.0, 1.0, 4.0]),
    )
    assert weighted.decode(field([1]))[0].tolist() == [1]

    # Even a most likely individual error can lose to the combined probability of another class.
    weighted_observables = field([[1, 0, 0]])
    weighted_native = code_capacity.get_code_capacity_decoder(
        syndrome_matrix,
        weighted_observables,
        specification,
        dem_error_weights=np.array([3.0, 2.0, 2.0]),
    )
    weighted_projected = code_capacity.get_code_capacity_decoder(
        syndrome_matrix,
        weighted_observables,
        decoders.from_matrix(
            lambda matrix: decoders.custom.LookupDecoder(
                matrix, max_weight=1, error_channel=[0.003, 0.002, 0.002]
            )
        ),
    )
    assert weighted_native.decode(field([1]))[0].tolist() == [0]
    assert weighted_projected.decode(field([1]))[0].tolist() == [1]

    erasing = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, decoders.lookup(max_weight=0, add_erasure_bit=True)
    )
    prediction, erased = erasing.decode(field([1]))
    assert prediction.tolist() == [0] and erased
    assert erasing.get_failure_and_erasure(errors[2]) == (False, True)


def test_code_capacity_native_build_fallback() -> None:
    """A specification that cannot decode the code-capacity DEM natively falls back to errors."""
    field = galois.GF2
    syndrome_matrix = field([[1, 1, 0], [0, 1, 1]])
    observable_matrix = field([[1, 0, 0]])
    single_errors = [field([1, 0, 0]), field([0, 1, 0]), field([0, 0, 1])]

    # matrix-only options cannot be applied to a detector error model
    for specification in [
        decoders.lookup(max_weight=1, error_channel=[0.1, 0.1, 0.1]),
        decoders.mwpm(weights=np.ones(3)),
    ]:
        with pytest.warns(decoders.ObservableDecodingFallbackWarning, match="from_matrix"):
            decoder = code_capacity.get_code_capacity_decoder(
                syndrome_matrix, observable_matrix, specification
            )
        assert isinstance(decoder.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
        assert not any(decoder.get_failure_and_erasure(error)[0] for error in single_errors)

    # a backend can reject the structure of the model, such as Y mechanisms that are not graphlike
    code = codes.QuditCode(codes.SurfaceCode(3).matrix)
    with pytest.warns(decoders.ObservableDecodingFallbackWarning, match="non-graphlike"):
        code.get_logical_error_rate_func(10, decoder=decoders.mwpm())
    with pytest.warns(decoders.ObservableDecodingFallbackWarning, match="cannot be symplectic"):
        code.get_logical_error_rate_func(10, decoder=decoders.lookup(max_weight=1, symplectic=True))

    # explicit error decoding, error-only specifications, and successful native builds do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", decoders.ObservableDecodingFallbackWarning)
        quiet_inputs: list[decoders.DecoderInput] = [
            decoders.from_matrix(decoders.mwpm(weights=np.ones(3)).build),
            decoders.bp_osd(),
            decoders.lookup(max_weight=1),
        ]
        for quiet_input in quiet_inputs:
            code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, quiet_input)

    # if error decoding also fails, both errors are reported
    bad_shape = decoders.lookup(max_weight=1, error_channel=[0.1, 0.1])
    with (
        pytest.warns(decoders.ObservableDecodingFallbackWarning),
        pytest.raises(ValueError, match="must have shape") as error_info,
    ):
        code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, bad_shape)
    assert isinstance(error_info.value.__cause__, ValueError)
    assert "stim.DetectorErrorModel" in str(error_info.value.__cause__)

    # an observable-only specification has nothing to fall back to
    observable_only: decoders.DecoderSpec[Never] = decoders.DecoderSpec(
        "observable_only", None, (), _raise_value_error
    )
    with pytest.raises(ValueError, match="cannot build"):
        code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, observable_only)

    # only a ValueError falls back; other errors from the native build propagate
    native_type_error: decoders.DecoderSpec[decoders.ErrorDecoder] = decoders.DecoderSpec(
        "native_type_error", decoders.bp_osd().build, (), _raise_type_error
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", decoders.ObservableDecodingFallbackWarning)
        with pytest.raises(TypeError, match="wrong type"):
            code_capacity.get_code_capacity_decoder(
                syndrome_matrix, observable_matrix, native_type_error
            )

    # long option values are truncated in the warning
    long_channel = decoders.lookup(max_weight=1, error_channel=[0.1] * 3 + [0.2] * 50)
    with (
        pytest.warns(decoders.ObservableDecodingFallbackWarning) as records,
        pytest.raises(ValueError),
    ):
        code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, long_channel)
    assert str(records[0].message).startswith("decoders.lookup(")
    assert "...) could not build" in str(records[0].message)


def _raise_value_error(dem: stim.DetectorErrorModel) -> Never:
    raise ValueError("cannot build")


def _raise_type_error(dem: stim.DetectorErrorModel) -> Never:
    raise TypeError("wrong type")


def test_code_capacity_decoder_from_sinter_decoder() -> None:
    """A Sinter-style decoder is compiled for the code-capacity DEM of its sector."""
    code = codes.RepetitionCode(3)
    observable_matrix = code.field([[1, 0, 0]])

    native = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.mwpm()
    )
    assert isinstance(native.decoder, MatchingObservableDecoder)
    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.SinterDecoder(decoder=decoders.mwpm())
    )
    assert isinstance(decoder.decoder, decoders.sinter.CompiledSinterDecoder)
    assert decoder.decoder.num_detectors == 2 and decoder.decoder.num_observables == 1
    for bit in range(3):
        error = code.field.Zeros(3)
        error[bit] = 1
        assert native.get_failure_and_erasure(error) == (False, False)
        assert decoder.get_failure_and_erasure(error) == (False, False)

    # a trivial decoder fails whenever an error flips an observable
    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.TrivialDecoder()
    )
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (True, False)
    assert decoder.get_failure_and_erasure(code.field([0, 1, 0])) == (False, False)

    # an erasure-enabled inner decoder discards samples
    sinter_decoder = decoders.SinterDecoder(
        decoder=decoders.lookup(max_weight=0, add_erasure_bit=True)
    )
    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, sinter_decoder
    )
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (False, True)
    assert decoder.get_failure_and_erasure(code.field([0, 0, 0])) == (False, False)

    # a Sinter-style decoder is not rejected as prebuilt, since it is compiled here
    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        decoders.TrivialDecoder(),
        prebuilt_rejection_reason="it is a test",
    )
    assert isinstance(decoder.decoder, decoders.sinter.CompiledTrivialDecoder)

    # but a compiled decoder is, and it must fit the sector that it decodes
    compiled_decoder = decoders.TrivialDecoder().compile_decoder_for_dem(
        code_capacity.get_code_capacity_dem(code.matrix, observable_matrix)
    )
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        code_capacity.get_code_capacity_decoder(
            code.matrix, observable_matrix, compiled_decoder, prebuilt_rejection_reason="test"
        )
    with pytest.raises(ValueError, match="has num_observables=1, but this code-capacity sector"):
        code_capacity.get_code_capacity_decoder(
            code.matrix, code.field.Identity(3), compiled_decoder
        )
    with pytest.raises(ValueError, match="has num_detectors=2, but this code-capacity sector"):
        code_capacity.get_code_capacity_decoder(
            code.field.Zeros((3, 3)), observable_matrix, compiled_decoder
        )

    # a DEM factory receives the code-capacity detector error model, not the parity-check matrix
    def observable_constructor(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        return decoders.custom.ObservableLookupDecoder(dem, max_weight=1)

    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.from_dem(observable_constructor)
    )
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (False, False)

    # return annotations no longer choose the factory input or output contract
    def annotated_observable_factory(_input: object) -> decoders.ObservableDecoder:
        return _FixedObservableDecoder([0])

    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        code_capacity.get_code_capacity_decoder(
            code.matrix, observable_matrix, annotated_observable_factory
        )

    # so are specifications that build an observable decoder, but cannot infer errors
    observable_spec: decoders.DecoderSpec[Never] = decoders.DecoderSpec(
        "observable_lookup", None, (), observable_constructor
    )
    decoder = code_capacity.get_code_capacity_decoder(
        code.matrix, observable_matrix, observable_spec
    )
    assert isinstance(decoder.decoder, decoders.custom.ObservableLookupDecoder)
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (False, False)

    # relative mechanism weights are scaled by a fixed placeholder probability
    weighted_sinter_decoder = _BitPackedSinterDecoder(np.zeros((1, 1), dtype=np.uint8))
    code_capacity.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        weighted_sinter_decoder,
        dem_error_weights=np.array([1, 2, 3]),
    )
    weighted_dem_arrays = decoders.DetectorErrorModelArrays(
        weighted_sinter_decoder.dems[0], simplify=False
    )
    assert np.array_equal(
        weighted_dem_arrays.error_probs / weighted_dem_arrays.error_probs[0], [1, 2, 3]
    )

    # external compiled Sinter decoders are recognized as prebuilt
    external_compiled = _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        code_capacity.get_code_capacity_decoder(
            code.matrix,
            observable_matrix,
            external_compiled,
            prebuilt_rejection_reason="it is a test",
        )

    # a Sinter-style decoder cannot decode a nonbinary code
    field = galois.GF(3)
    with pytest.raises(ValueError, match="cannot decode a code over GF"):
        code_capacity.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), decoders.TrivialDecoder()
        )
    with pytest.raises(ValueError, match="cannot decode a code over GF"):
        code_capacity.get_code_capacity_decoder(
            field(code.matrix),
            field(observable_matrix),
            decoders.from_dem(observable_constructor),
        )
    with pytest.raises(ValueError, match="is binary, so it cannot decode a code over GF"):
        code_capacity.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), compiled_decoder
        )
    with pytest.raises(ValueError, match="is binary, so it cannot decode a code over GF"):
        code_capacity.get_code_capacity_decoder(
            field(code.matrix),
            field(observable_matrix),
            _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8)),
        )

    binary_lookup = decoders.custom.ObservableLookupDecoder(
        code.matrix,
        max_weight=1,
        observable_flip_matrix=observable_matrix,
        error_channel=[0.1] * len(code),
    )
    with pytest.raises(ValueError, match=r"built over GF\(2\).+sector is over GF\(3\)"):
        code_capacity.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), binary_lookup
        )


def test_code_capacity_decoder_from_bit_packed_sinter_decoder() -> None:
    """A Sinter decoder whose compiled form only decodes bit-packed data is supported."""
    syndrome_matrix = galois.GF2(np.eye(9, dtype=int))
    observable_matrix = galois.GF2(np.eye(9, dtype=int))
    error = galois.GF2([1, 0, 0, 0, 0, 0, 0, 0, 1])

    # a raw precompiled decoder needs dimension metadata before it can be used directly
    with pytest.raises(ValueError, match="does not declare num_detectors"):
        code_capacity.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _BitPackedCompiledDecoder(np.array([[1, 1]], dtype=np.uint8)),
        )

    # nine observables take two bytes, so the prediction [1, 0, ..., 0, 1] is packed as [1, 1]
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1, 1]], dtype=np.uint8))
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    assert sinter_decoder.dems[0].num_detectors == sinter_decoder.dems[0].num_observables == 9
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(error) == (False, False)
    compiled_decoder = decoder.decoder.compiled_decoder  # type:ignore[attr-defined]
    assert np.array_equal(compiled_decoder.packed_syndromes[-1], [[1, 1]])
    assert decoder.get_failure_and_erasure(galois.GF2.Zeros(9)) == (True, False)

    # a nonzero byte added past the observables discards the sample
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1, 1, 1]], dtype=np.uint8))
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    assert decoder.get_failure_and_erasure(error) == (False, True)

    # a prediction of any other width is rejected
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1]], dtype=np.uint8))
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    with pytest.raises(ValueError, match=r"shape \(1, 1\) for one shot"):
        decoder.get_failure_and_erasure(error)

    # a compiled decoder that can do neither is rejected
    class _NotADecoder(sinter.Decoder):
        def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> object:
            return object()

    with pytest.raises(TypeError, match="must provide a decode_observables or"):
        code_capacity.get_code_capacity_decoder(syndrome_matrix, observable_matrix, _NotADecoder())


def test_code_capacity_decoder_reuse() -> None:
    """A decoder is reused for another sector only when it predicts the right observables."""
    syndrome_matrix = galois.GF2([[1, 1, 0], [0, 1, 1]])
    observables_a = galois.GF2([[1, 1, 1]])
    observables_b = galois.GF2([[1, 0, 0]])

    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observables_a, decoders.lookup(max_weight=1)
    )
    assert decoder.reuse_for(syndrome_matrix, observables_a) is decoder
    assert decoder.reuse_for(galois.GF2([[1, 1, 0]]), observables_a) is None
    assert decoder.reuse_for(syndrome_matrix, observables_b) is None

    # a matrix-only error decoder may be reused with other observables
    projected = code_capacity.get_code_capacity_decoder(
        syndrome_matrix,
        observables_a,
        decoders.from_matrix(lambda matrix: decoders.custom.LookupDecoder(matrix, max_weight=1)),
    )
    reused_decoder = projected.reuse_for(syndrome_matrix, observables_b)
    assert reused_decoder is not None
    assert isinstance(projected.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert isinstance(reused_decoder.decoder, observable_decoders.ErrorsToFieldObservablesDecoder)
    assert reused_decoder.decoder.error_decoder is projected.decoder.error_decoder
    assert reused_decoder.observable_matrix is not None
    assert np.array_equal(reused_decoder.observable_matrix, observables_b)

    # an observable decoder is not
    decoder = code_capacity.get_code_capacity_decoder(
        syndrome_matrix, observables_a, decoders.TrivialDecoder()
    )
    assert decoder.reuse_for(syndrome_matrix, observables_b) is None


@pytest.mark.parametrize(
    "module",
    [
        "qldpc.codes.code_capacity",
        "qldpc.codes.monte_carlo",
        "qldpc.decoders.capabilities",
        "qldpc.decoders.custom",
        "qldpc.decoders.custom.lookup",
        "qldpc.decoders.adapters.observable_decoders",
        "qldpc.circuits",
    ],
)
def test_decoder_import_order(module: str) -> None:
    """Decoder-related modules can each be the first qLDPC import in a fresh interpreter."""
    repo_root = pathlib.Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(repo_root / "src")},
        text=True,
    )
    assert result.returncode == 0, result.stderr
