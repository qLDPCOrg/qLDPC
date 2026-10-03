# SPDX-License-Identifier: Apache-2.0

"""Unit tests for common.py."""

from __future__ import annotations

import warnings
from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders import common
from qldpc.decoders.external.ldpc import _get_decoder_bp_osd


def test_get_error_and_erasure() -> None:
    """Decode a syndrome, with and without an erasure bit."""
    field = galois.GF2
    syndrome = field([1, 0, 1])

    class _Decoder:
        def __init__(self, output: np.ndarray, has_erasure_bit: bool = False) -> None:
            self.output = output
            if has_erasure_bit:
                self.has_erasure_bit = True

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            return self.output

    # a plain decoder returns the inferred error and no erasure
    decoder = _Decoder(np.array([1, 1, 0, 0], dtype=np.uint8))
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert not erasure and isinstance(error, field) and np.array_equal(error, field([1, 1, 0, 0]))

    # an erasure-enabled decoder strips the last (erasure) bit and reports it.  The first and last
    # entries differ, so reading the wrong end of the vector fails here
    decoder = _Decoder(np.array([0, 1, 1, 0, 1], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert erasure and np.array_equal(error, field([0, 1, 1, 0]))

    # the same decoder reports no erasure when the syndrome was recognized
    decoder = _Decoder(np.array([1, 1, 0, 0, 0], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = decoders.get_error_and_erasure(decoder, syndrome)
    assert not erasure and np.array_equal(error, field([1, 1, 0, 0]))


def test_with_erasure_bits() -> None:
    """Erasure flags are appended to individual and batched inferred errors."""
    error = np.array([1, 0], dtype=int)
    assert np.array_equal(decoders.with_erasure_bits(error, True), [1, 0, 1])

    errors = np.array([[1, 0], [0, 1]], dtype=int)
    erased = np.array([False, True])
    assert np.array_equal(decoders.with_erasure_bits(errors, erased), [[1, 0, 0], [0, 1, 1]])


def test_detailed_decode_helpers() -> None:
    """Detailed helpers preserve predictions and normalize erasure flags."""
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    class BareErrorDecoder:
        has_erasure_bit = True

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.append(syndrome, int(syndrome[0]))

    error_results = decoders.decode_errors_detailed_batch(BareErrorDecoder(), syndromes)
    assert [result.error.tolist() for result in error_results] == [[1, 0], [0, 1]]
    assert [result.erasure for result in error_results] == [True, False]
    assert not error_results[0].diagnostics

    class BareObservableDecoder:
        has_erasure_bit = True

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.append(syndrome, int(syndrome[1]))

    observable_results = decoders.decode_observables_detailed_batch(
        BareObservableDecoder(), syndromes
    )
    assert [result.observable_flips.tolist() for result in observable_results] == [
        [1, 0],
        [0, 1],
    ]
    assert [result.erasure for result in observable_results] == [False, True]

    class DetailedErrorDecoder(BareErrorDecoder, decoders.ErrorDecoder):
        def decode_errors_detailed(
            self, syndrome: npt.NDArray[np.int_]
        ) -> decoders.ErrorDecodeResult:
            return decoders.ErrorDecodeResult(syndrome, diagnostics={"backend.score": 2.5})

        def decode_errors_detailed_batch(
            self, values: npt.NDArray[np.int_]
        ) -> tuple[decoders.ErrorDecodeResult, ...]:
            return tuple(self.decode_errors_detailed(value) for value in values)

    detailed_errors = decoders.decode_errors_detailed_batch(DetailedErrorDecoder(), syndromes)
    assert [result.error.tolist() for result in detailed_errors] == [[1, 0], [0, 1]]
    assert [result.diagnostics["backend.score"] for result in detailed_errors] == [2.5, 2.5]
    assert decoders.decode_errors_detailed(DetailedErrorDecoder(), syndromes[0]).error.tolist() == [
        1,
        0,
    ]
    assert isinstance(DetailedErrorDecoder(), decoders.DetailedErrorDecoder)
    assert isinstance(DetailedErrorDecoder(), decoders.BatchDetailedErrorDecoder)

    class InvalidDetailedErrorDecoder(BareErrorDecoder):
        def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> object:
            return object()

        def decode_errors_detailed_batch(self, values: npt.NDArray[np.int_]) -> tuple[object, ...]:
            return ()

    with pytest.raises(TypeError, match="must return an ErrorDecodeResult"):
        decoders.decode_errors_detailed(InvalidDetailedErrorDecoder(), syndromes[0])
    with pytest.raises(TypeError, match="one ErrorDecodeResult per syndrome"):
        decoders.decode_errors_detailed_batch(InvalidDetailedErrorDecoder(), syndromes)

    class DetailedObservableDecoder(BareObservableDecoder):
        def decode_observables_detailed(
            self, syndrome: npt.NDArray[np.int_]
        ) -> decoders.ObservableDecodeResult:
            return decoders.ObservableDecodeResult(syndrome, diagnostics={"backend.iterations": 4})

        def decode_observables_detailed_batch(
            self, values: npt.NDArray[np.int_]
        ) -> tuple[decoders.ObservableDecodeResult, ...]:
            return tuple(self.decode_observables_detailed(value) for value in values)

    detailed_observables = decoders.decode_observables_detailed_batch(
        DetailedObservableDecoder(), syndromes
    )
    assert [result.observable_flips.tolist() for result in detailed_observables] == [
        [1, 0],
        [0, 1],
    ]
    assert [result.diagnostics["backend.iterations"] for result in detailed_observables] == [
        4,
        4,
    ]
    assert decoders.decode_observables_detailed(
        DetailedObservableDecoder(), syndromes[0]
    ).observable_flips.tolist() == [1, 0]
    assert isinstance(DetailedObservableDecoder(), decoders.DetailedObservableDecoder)
    assert isinstance(DetailedObservableDecoder(), decoders.BatchDetailedObservableDecoder)

    class InvalidDetailedObservableDecoder(BareObservableDecoder):
        def decode_observables_detailed(self, syndrome: npt.NDArray[np.int_]) -> object:
            return object()

        def decode_observables_detailed_batch(
            self, values: npt.NDArray[np.int_]
        ) -> tuple[object, ...]:
            return ()

    with pytest.raises(TypeError, match="must return an ObservableDecodeResult"):
        decoders.decode_observables_detailed(InvalidDetailedObservableDecoder(), syndromes[0])
    with pytest.raises(TypeError, match="one ObservableDecodeResult per syndrome"):
        decoders.decode_observables_detailed_batch(InvalidDetailedObservableDecoder(), syndromes)

    assert decoders.decode_errors_detailed_batch(BareErrorDecoder(), syndromes[:0]) == ()
    assert decoders.decode_observables_detailed_batch(BareObservableDecoder(), syndromes[:0]) == ()
    ordinary = decoders.LookupDecoder(np.eye(2, dtype=int), max_weight=1)
    assert decoders.decode_errors_detailed(ordinary, syndromes[0]).error.tolist() == [1, 0]


def test_detailed_batch_uses_hard_batch_fallback() -> None:
    """A decoder without detailed methods retains its native hard-batch path."""
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    class BatchErrors(decoders.BatchErrorDecoder):
        has_erasure_bit = True

        def decode_errors_batch(self, values: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.column_stack((values[:, ::-1], values[:, 0]))

    errors = decoders.decode_errors_detailed_batch(BatchErrors(), syndromes)
    assert [result.error.tolist() for result in errors] == [[0, 1], [1, 0]]
    assert [result.erasure for result in errors] == [True, False]

    class BatchObservables(decoders.BatchObservableDecoder):
        has_erasure_bit = True

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.append(syndrome, 1)

        def decode_observables_batch(self, values: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.column_stack((values, values[:, 1]))

    assert decoders.decode_observables_detailed(BatchObservables(), syndromes[0]).erasure
    observables = decoders.decode_observables_detailed_batch(BatchObservables(), syndromes)
    assert [result.observable_flips.tolist() for result in observables] == [[1, 0], [0, 1]]
    assert [result.erasure for result in observables] == [False, True]

    class SingleDetailedErrors(BatchErrors):
        def decode_errors_detailed(
            self, syndrome: npt.NDArray[np.int_]
        ) -> decoders.ErrorDecodeResult:
            return decoders.ErrorDecodeResult(syndrome, diagnostics={"source": "detailed"})

    class SingleDetailedObservables(BatchObservables):
        def decode_observables_detailed(
            self, syndrome: npt.NDArray[np.int_]
        ) -> decoders.ObservableDecodeResult:
            return decoders.ObservableDecodeResult(syndrome, diagnostics={"source": "detailed"})

    assert [
        result.diagnostics["source"]
        for result in decoders.decode_errors_detailed_batch(SingleDetailedErrors(), syndromes)
    ] == ["detailed", "detailed"]
    assert [
        result.diagnostics["source"]
        for result in decoders.decode_observables_detailed_batch(
            SingleDetailedObservables(), syndromes
        )
    ] == ["detailed", "detailed"]


def test_erasure_bit_support_decorator() -> None:
    """Erasure support declarations enforce capabilities and use an explicit display name."""

    @common._erasure_bit_support("Friendly Name", supported=True)
    def unusually_named_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del add_erasure_bit
        return _get_decoder_bp_osd(matrix)

    with pytest.raises(ValueError, match="The Friendly Name decoder cannot signal erasure"):
        unusually_named_builder(np.eye(1, dtype=int), add_erasure_bit=True)

    @common._erasure_bit_support("unsupported", supported=False)
    def unsupported_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del add_erasure_bit
        return _get_decoder_bp_osd(matrix)

    assert unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=False)
    with pytest.raises(ValueError, match="The unsupported decoder cannot signal erasure"):
        unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=True)


def test_to_pcm() -> None:
    """Matrices pass through unchanged and DEM detector matrices are densified."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    assert common._to_pcm(matrix) is matrix

    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1).to_dem()
    assert np.array_equal(common._to_pcm(dem), matrix)


def test_get_matrix_error_channel() -> None:
    """Matrix probabilities are normalized and DEM-owned probabilities are protected."""
    matrix = np.eye(2, dtype=int)
    default_channel = common._get_matrix_error_channel(matrix, None, None)
    assert default_channel is not None
    assert np.array_equal(
        default_channel,
        [common.PLACEHOLDER_ERROR_RATE] * 2,
    )
    channel = np.array([0.1, 0.2])
    normalized_channel = common._get_matrix_error_channel(matrix, channel, None)
    assert normalized_channel is not None
    assert np.array_equal(normalized_channel, channel)

    with pytest.warns(DeprecationWarning, match="error_rate=0.3.*error_channel=0.3") as caught:
        deprecated_channel = common._get_matrix_error_channel(matrix, None, 0.3)
    assert deprecated_channel is not None
    assert np.array_equal(deprecated_channel, [0.3, 0.3])
    assert caught[0].filename == __file__

    with pytest.raises(ValueError, match="cannot both be specified"):
        common._get_matrix_error_channel(matrix, 0.2, 0.3)
    with pytest.raises(ValueError, match="error probabilities of shape"):
        common._get_matrix_error_channel(matrix, [0.1], None)
    for invalid_channel in ([np.nan, 0.2], [-0.1, 0.2], [0.1, 1.1]):
        with pytest.raises(ValueError, match="finite and between 0 and 1"):
            common._get_matrix_error_channel(matrix, invalid_channel, None)

    dem = stim.DetectorErrorModel("error(0.1) D0")
    assert common._get_matrix_error_channel(dem, None, None) is None
    for error_channel, error_rate, specified in (
        (None, 0.2, "error_rate=0.2"),
        (0.2, None, "error_channel=0.2"),
        (0.2, 0.3, "error_channel=0.2 and error_rate=0.3"),
    ):
        with pytest.raises(ValueError) as error:
            common._get_matrix_error_channel(dem, error_channel, error_rate)
        message = str(error.value)
        assert message.startswith(
            f"A detector error model supplies its own error probabilities, so {specified} cannot"
        )
        assert "let error_channel override its probabilities" in message
        assert "SinterDecoder" in message
        assert "DetectorErrorModelArrays(dem).detector_flip_matrix" in message


def test_dem_error_probabilities_through_public_paths() -> None:
    """Explicit probabilities for a DEM fail with migration advice through every entry point."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")
    match = r"supplies its own error probabilities.*detector_flip_matrix"
    with pytest.raises(ValueError, match=match):
        decoders.bp_osd(error_channel=0.1).build(dem)
    with pytest.raises(ValueError, match=match):
        decoders.SinterDecoder(decoder=decoders.bf(error_channel=0.1)).compile_decoder_for_dem(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=match):
            decoders.SinterDecoder(error_rate=0.1).compile_decoder_for_dem(dem)
        with pytest.raises(ValueError, match=match):
            decoders.get_decoder(dem, with_BP_LSD=True, error_rate=0.1)

    # the suggested alternative decodes the detector-flip matrix with the given probabilities
    matrix = decoders.DetectorErrorModelArrays(dem).detector_flip_matrix
    decoder: Any = decoders.bp_osd(error_channel=0.3).build(matrix)
    assert np.array_equal(decoder.error_channel, [0.3, 0.3, 0.3])


def test_deprecate_error_rate_option() -> None:
    """Deferred options translate explicit error_rate values and reject ambiguity."""
    options: dict[str, object] = {"error_channel": None, "error_rate": None}
    assert common._deprecate_error_rate_option(options, frozenset()) == {"error_channel": None}

    with pytest.warns(DeprecationWarning, match="error_rate=0.2.*error_channel=0.2") as caught:
        translated = common._deprecate_error_rate_option(
            {"error_channel": None, "error_rate": 0.2},
            frozenset({"error_rate"}),
        )
    assert caught[0].filename == __file__
    assert translated == {"error_channel": 0.2}

    with pytest.raises(ValueError, match="cannot both be specified"):
        common._deprecate_error_rate_option(
            {"error_channel": 0.1, "error_rate": 0.2},
            frozenset({"error_channel", "error_rate"}),
        )

    # an explicit error_rate=None is equivalent to omitting it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        translated = common._deprecate_error_rate_option(
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
