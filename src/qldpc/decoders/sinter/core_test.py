# SPDX-License-Identifier: Apache-2.0

"""Tests for core Sinter decoders."""

from __future__ import annotations

import ldpc
import numpy as np
import numpy.typing as npt
import pytest
import sinter
import stim

from qldpc import decoders
from qldpc.decoders.custom.lookup import _get_decoder_lookup


def test_sinter_decoder() -> None:
    """Try out a simple decoding problem."""
    dem = stim.DetectorErrorModel("""
        error(0.0001) D0
        error(0.0002) D0 D1
        error(0.0003) D2 L1
    """)

    circuit_errors = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    observable_flips = [[0, 0], [0, 0], [0, 1]]
    bit_packed_shots = np.packbits(circuit_errors, bitorder="little", axis=1)
    expected_flips = np.packbits(observable_flips, bitorder="little", axis=1)

    for decoder in [
        decoders.SinterDecoder(decoder=decoders.bp_osd()),
        decoders.SinterDecoder(decoder=decoders.min_sum_bp()),
        decoders.SinterDecoder(decoder=decoders.mwpm()),
    ]:
        compiled_decoder = decoder.compile_decoder_for_dem(dem)
        predicted_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots)
        assert np.array_equal(predicted_flips, expected_flips)

        no_flips = compiled_decoder.decode_shots_bit_packed(bit_packed_shots[:0])
        assert no_flips.shape == (0, expected_flips.shape[1])

        assert np.array_equal(
            [compiled_decoder.decode_observables(np.asarray(error)) for error in circuit_errors],
            observable_flips,
        )

    compiled_decoder = decoders.SinterDecoder().compile_decoder_for_dem(dem)
    assert isinstance(compiled_decoder.decoder, ldpc.BpOsdDecoder)
    assert isinstance(compiled_decoder.observable_decoder, decoders.ErrorsToObservablesDecoder)
    compiled_decoder = decoders.SinterDecoder(decoder=decoders.mwpm()).compile_decoder_for_dem(dem)
    assert compiled_decoder.decoder is compiled_decoder.observable_decoder

    error_decoder = _get_decoder_lookup(dem, max_weight=3)
    compiled_decoder = decoders.sinter.CompiledSinterDecoder(
        decoders.DetectorErrorModelArrays(dem), error_decoder
    )
    assert compiled_decoder.decoder is error_decoder
    assert np.array_equal(
        compiled_decoder.decode_shots_bit_packed(bit_packed_shots), expected_flips
    )

    decoder = decoders.TrivialDecoder()
    compiled_decoder = decoder.compile_decoder_for_dem(dem)
    assert np.array_equal(
        compiled_decoder.decode_shots(np.array(circuit_errors)),
        np.zeros_like(observable_flips),
    )
    assert np.array_equal(
        compiled_decoder.decode_shots_bit_packed(bit_packed_shots),
        np.zeros_like(expected_flips),
    )


def test_sinter_decoder_classes() -> None:
    """Only compiled Sinter decoders implement qLDPC's observable protocol."""
    assert sinter.Decoder in decoders.SinterDecoder.__mro__
    assert decoders.ObservableDecoder not in decoders.SinterDecoder.__mro__
    assert sinter.CompiledDecoder in decoders.sinter.CompiledSinterDecoder.__mro__
    assert decoders.ObservableDecoder in decoders.sinter.CompiledSinterDecoder.__mro__

    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    decoder = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=1))
    assert not hasattr(decoder, "decode")
    compiled = decoder.compile_decoder_for_dem(dem)
    assert not hasattr(compiled, "decode")
    with pytest.raises(TypeError, match="cannot decode until it is compiled"):
        decoders.as_error_decoder(decoder)
    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        decoders.as_error_decoder(compiled)
    assert np.array_equal(compiled.decode_observables(np.array([1], dtype=int)), [1])


def test_sinter_decoder_correlated_matching() -> None:
    """A SinterDecoder keeps the error decompositions that correlated matching uses."""
    dem = stim.DetectorErrorModel("""
        error(0.02) D0 D1 ^ D2 D3
        error(0.3) D2 L0
        error(0.3) D3
    """)
    spec = decoders.mwpm(enable_correlations=True)
    compiled_decoder = decoders.SinterDecoder(decoder=spec).compile_decoder_for_dem(dem)
    assert np.array_equal(compiled_decoder.decode_shots(np.array([[1, 1, 1, 1]])), [[0]])

    with pytest.raises(ValueError, match="decompose_errors=True discards"):
        decoders.SinterDecoder(decompose_errors=True, decoder=spec)


def test_unsimplified_dense_decoder() -> None:
    """A dense decoder is built for every error mechanism that its model declares."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    compiled = decoders.SinterDecoder(
        simplify=False, decoder=decoders.guf()
    ).compile_decoder_for_dem(dem)
    assert compiled.decode_shots(np.array([[1, 0]], dtype=np.uint8)).tolist() == [[1]]


def test_rejected_decoder_arguments() -> None:
    """Decoder arguments that an ObservableDecoder cannot honor are rejected."""
    with (
        pytest.warns(DeprecationWarning, match="free-form decoder options"),
        pytest.raises(ValueError, match="DEFUNCT"),
    ):
        decoders.SinterDecoder(priors_arg="error_channel")

    dem = stim.DetectorErrorModel("""
        error(0.3) D0 L0 L1
        error(0.1) D1 L1
    """)

    def build_observable_decoder(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        return decoders.custom.ObservableLookupDecoder(dem, 2)

    decoder = decoders.SinterDecoder(decoder=build_observable_decoder)
    compiled = decoder.compile_decoder_for_dem(dem)
    assert isinstance(compiled.observable_decoder, decoders.custom.ObservableLookupDecoder)
    assert np.array_equal(compiled.decode_observables(np.array([1, 0])), [1, 1])

    compiled = decoders.SinterDecoder(
        decoder=decoders.from_dem(build_observable_decoder)
    ).compile_decoder_for_dem(dem)
    assert np.array_equal(compiled.decode_observables(np.array([1, 0])), [1, 1])
    with pytest.raises(ValueError, match="needs a parity-check matrix"):
        decoders.SinterDecoder(
            decoder=decoders.from_matrix(lambda matrix: decoders.custom.LookupDecoder(matrix, 1))
        ).compile_decoder_for_dem(dem)

    window_decoder = decoders.SequentialWindowDecoder(
        [[0], [1]],
        decoder=build_observable_decoder,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="predicts observable flips rather than errors"):
        window_decoder.compile_decoder_for_dem(dem)

    class SingleShotDecoder:
        def __init__(self, dem: stim.DetectorErrorModel) -> None:
            self.lookup = decoders.custom.ObservableLookupDecoder(dem, 2)

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.lookup.decode_observables(syndrome)

    compiled = decoders.SinterDecoder(decoder=SingleShotDecoder).compile_decoder_for_dem(dem)
    assert np.array_equal(compiled.decode_shots(np.array([[1, 0], [0, 1]])), [[1, 1], [0, 1]])

    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        decoders.SinterDecoder(
            decoder=decoders.custom.ObservableLookupDecoder(dem, 2)  # type: ignore[arg-type]
        )

    with pytest.warns(DeprecationWarning):
        decoder = decoders.SinterDecoder(
            with_lookup=True, max_weight=2, predict_observable_flips=True
        )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        decoder.compile_decoder_for_dem(dem)

    with pytest.warns(DeprecationWarning):
        window_decoder = decoders.SequentialWindowDecoder(
            [[0], [1]], with_lookup=True, max_weight=1, predict_observable_flips=True
        )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="observable flips rather than errors"),
    ):
        window_decoder.compile_decoder_for_dem(dem)


def test_observable_decoders_reject_prebuilt_decoders() -> None:
    """Observable decoders reject an error decoder that cannot be rebuilt for each model."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    prebuilt = decoders.custom.LookupDecoder(dem, max_weight=1)
    for build_decoder, reason in [
        (
            lambda: decoders.SinterDecoder(decoder=prebuilt),  # type: ignore[arg-type]
            "detector error model",
        ),
        (
            lambda: decoders.SubgraphDecoder([[0]], decoder=prebuilt),  # type: ignore[arg-type]
            "each subgraph",
        ),
        (
            lambda: decoders.SequentialWindowDecoder([[0]], decoder=prebuilt),  # type: ignore[arg-type]
            "each window",
        ),
        (
            lambda: decoders.SlidingWindowDecoder(1, 1, decoder=prebuilt),  # type: ignore[arg-type]
            "each window",
        ),
    ]:
        with pytest.raises(
            ValueError, match=f"prebuilt decoder cannot be passed as decoder=.*{reason}"
        ):
            build_decoder()
    with pytest.raises(TypeError, match="static_decoder argument has been removed"):
        decoders.SinterDecoder(static_decoder=prebuilt)

    decoder = decoders.SequentialWindowDecoder(
        [[0]], decoder=lambda dem: decoders.custom.LookupDecoder(dem, max_weight=1)
    )
    assert np.array_equal(
        decoder.compile_decoder_for_dem(dem).decode_observables(np.array([1])), [1]
    )


def test_sinter_decoder_with_erasure() -> None:
    """A compiled decoder appends an erasure bit if its inner decoder erases."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        error(0.1) D1 L0
    """)
    decoder = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=1, add_erasure_bit=True))
    compiled = decoder.compile_decoder_for_dem(dem)

    assert isinstance(compiled.observable_decoder, decoders.custom.ObservableLookupDecoder)
    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 1
    assert compiled.has_erasure_bit
    assert not decoders.TrivialDecoder().compile_decoder_for_dem(dem).has_erasure_bit

    shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (2, dem.num_observables + 1)
    assert np.array_equal(result[:, :-1], [[0], [1]])
    assert np.all(result[:, -1] == 0)
    assert compiled.decode_shots(np.array([[1, 1]], dtype=np.uint8))[0, -1] == 1

    # a nested compiled decoder passes its erasure bit on to the decoder that wraps it
    nested = decoders.SinterDecoder(decoder=decoder).compile_decoder_for_dem(dem)
    assert nested.num_erasure_bits == 1
    assert np.array_equal(nested.decode_shots(np.array([[1, 1]], dtype=np.uint8)), [[0, 1]])


@pytest.mark.parametrize("num_observables", [7, 8])
def test_erasure_signalled_in_an_added_byte(num_observables: int) -> None:
    """An erasure bit is bit-packed into one byte past the observable flips."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.1) D{oo} L{oo}" for oo in range(num_observables))
    )
    decoder = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=1, add_erasure_bit=True))
    compiled = decoder.compile_decoder_for_dem(dem)

    erased_syndrome = np.zeros(num_observables, dtype=np.uint8)
    erased_syndrome[:2] = 1
    shots = np.array([np.zeros(num_observables, dtype=np.uint8), erased_syndrome])
    packed_flips = compiled.decode_shots_bit_packed(compiled.packbits(shots))
    assert np.array_equal(
        compiled.pack_observable_flips(compiled.decode_shots(shots)), packed_flips
    )

    assert packed_flips.shape == (2, (num_observables + 7) // 8 + 1)
    assert packed_flips[0, -1] == 0
    assert packed_flips[1, -1] == 1


@pytest.mark.parametrize("num_observables", [7, 8])
def test_predict_observables_with_erasure(num_observables: int) -> None:
    """Predictions written to a file omit the discard byte."""
    dem = stim.DetectorErrorModel(
        "\n".join(f"error(0.2) D{oo} L{oo}" for oo in range(num_observables))
    )
    detection_events = np.zeros((3, num_observables), dtype=bool)
    detection_events[1, 0] = True
    detection_events[2, :2] = True

    decoder = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=1, add_erasure_bit=True))
    predictions = sinter.predict_observables(
        dem=dem, dets=detection_events, decoder="qldpc", custom_decoders={"qldpc": decoder}
    )

    expected_flips = np.zeros((3, num_observables), dtype=int)
    expected_flips[1, 0] = 1
    assert np.array_equal(np.asarray(predictions, dtype=int), expected_flips)

    class WideCompiledDecoder(decoders.sinter.CompiledSinterDecoder):
        """A compiled decoder whose bit-packed predictions are two bytes too wide."""

        def pack_observable_flips(
            self, observable_flips: npt.NDArray[np.uint8]
        ) -> npt.NDArray[np.uint8]:
            packed_flips = super().pack_observable_flips(observable_flips)
            padding = np.zeros((len(packed_flips), 2), dtype=np.uint8)
            return np.hstack([packed_flips, padding])

    class WideDecoder(decoders.SinterDecoder):
        """A decoder whose bit-packed predictions are two bytes too wide."""

        def compile_decoder_for_dem(
            self, dem: stim.DetectorErrorModel
        ) -> decoders.sinter.CompiledSinterDecoder:
            compiled = super().compile_decoder_for_dem(dem)
            return WideCompiledDecoder(compiled.dem_arrays, compiled.decoder)

    with pytest.raises(ValueError, match="bytes of observable flips per shot"):
        sinter.predict_observables(
            dem=dem,
            dets=detection_events,
            decoder="qldpc",
            custom_decoders={"qldpc": WideDecoder(decoder=decoders.lookup(max_weight=1))},
        )


def test_compiled_sinter_decoder_delegates_bit_packed_shots() -> None:
    """A compiled decoder delegates bit-packed shots to an inner decoder that decodes them."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    class BitPackedShotDecoder(decoders.ObservableDecoder):
        output: npt.NDArray[np.uint8]

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.asarray(syndrome)

        def decode_shots_bit_packed(
            self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
        ) -> npt.NDArray[np.uint8]:
            return self.output

    inner = BitPackedShotDecoder()
    compiled = decoders.sinter.CompiledSinterDecoder(decoders.DetectorErrorModelArrays(dem), inner)
    shots = np.array([[0], [1]], dtype=np.uint8)
    assert np.array_equal(compiled.decode_shots(shots), shots)

    inner.output = np.array([[5], [9]], dtype=np.uint8)
    assert np.array_equal(compiled.decode_shots_bit_packed(shots), inner.output)
    inner.output = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="bit-packed shots of shape"):
        compiled.decode_shots_bit_packed(shots)


def test_compiled_sinter_decoder_subclass_decode_shots() -> None:
    """A subclass that post-processes decode_shots gets the same packed and unpacked results."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    class InvertingDecoder(decoders.sinter.CompiledSinterDecoder):
        def decode_shots(
            self, detection_event_data: npt.NDArray[np.uint8]
        ) -> npt.NDArray[np.uint8]:
            return 1 - super().decode_shots(detection_event_data)

    compiled = InvertingDecoder(
        decoders.DetectorErrorModelArrays(dem),
        decoders.custom.ObservableLookupDecoder(dem, max_weight=1),
    )
    shot = np.zeros((1, 1), dtype=np.uint8)
    assert np.array_equal(compiled.decode_shots(shot), [[1]])
    assert compiled.decode_shots_bit_packed(np.zeros((1, 1), dtype=np.uint8)).tolist() == [[1]]
