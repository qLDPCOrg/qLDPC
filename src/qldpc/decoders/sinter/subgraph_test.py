# SPDX-License-Identifier: Apache-2.0

"""Tests for subgraph Sinter decoders."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import stim

from qldpc import decoders


def test_subgraph_decoding() -> None:
    """Decode independent subgraphs."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D1 L1
        error(0.1) D2 L2
    """)
    sampler = dem.compile_sampler()
    det_data, obs_data, _err_data = sampler.sample(100)

    decoder_1 = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=3))
    compiled_decoder_1 = decoder_1.compile_decoder_for_dem(dem)
    predicted_flips_1 = compiled_decoder_1.decode_shots_bit_packed(
        compiled_decoder_1.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, compiled_decoder_1.packbits(obs_data))

    decoder_2 = decoders.SubgraphDecoder([[0], [1], [2]], decoder=decoders.lookup(max_weight=1))
    compiled_decoder_2 = decoder_2.compile_decoder_for_dem(dem)
    predicted_flips_2 = compiled_decoder_2.decode_shots_bit_packed(
        compiled_decoder_2.packbits(det_data)
    )
    assert np.array_equal(predicted_flips_1, predicted_flips_2)

    with pytest.raises(ValueError, match="inconsistent"):
        decoders.SubgraphDecoder([[0], [1], [2]], [[0]])


def test_compiled_subgraph_input_validation() -> None:
    """Compiled subgraph decoders reject inconsistent regions and detector counts."""
    dem = stim.DetectorErrorModel("""
        detector(0) D0
        detector(1) D1
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    wide_shots = np.zeros((1, dem.num_detectors + 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="per subgraph"):
        decoders.CompiledSubgraphDecoder([[0]], [[0], [1]], [], 2, 2)
    subgraph_decoder = decoders.SubgraphDecoder(
        [[0], [1]], decoder=decoders.lookup(max_weight=1)
    ).compile_decoder_for_dem(dem)
    with pytest.raises(ValueError, match="detectors per shot"):
        subgraph_decoder.decode_shots(wide_shots)


def test_native_observable_decoders_on_subgraphs() -> None:
    """Native observable decoders agree with constructors, even without observables."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.01
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    detection_events = circuit.compile_detector_sampler(seed=0).sample(200).astype(np.uint8)
    num_detectors = dem.num_detectors
    subgraph_detectors = [range(num_detectors // 2), range(num_detectors // 2, num_detectors)]

    for spec in [
        decoders.mwpm(),
        decoders.relay_bp(),
        decoders.min_sum_bp(gamma0=0.5),
        decoders.lookup(max_weight=1),
    ]:
        assert spec.predicts_observables_natively
        predicted_flips = []
        decoder_inputs: list[decoders.DeferredDecoderInput] = [spec, spec.build]
        for decoder in decoder_inputs:
            sinter_decoder = decoders.SubgraphDecoder(
                subgraph_detectors, [[0], []], decompose_errors=True, decoder=decoder
            )
            compiled = sinter_decoder.compile_decoder_for_dem(dem)
            predicted_flips.append(compiled.decode_shots(detection_events))
        assert predicted_flips[0].shape == (len(detection_events), dem.num_observables)
        assert np.array_equal(predicted_flips[0], predicted_flips[1]), spec


def test_subgraph_partition_warnings() -> None:
    """Compiling warns about a partition whose predictions do not add up."""
    contested_dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0")
    with pytest.warns(UserWarning, match="can be predicted by more than one subgraph") as contested:
        decoders.SubgraphDecoder(
            [[0], [1]], decoder=decoders.lookup(max_weight=1)
        ).compile_decoder_for_dem(contested_dem)
    assert all(warning.filename == __file__ for warning in contested)

    uncovered_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L0")
    with pytest.warns(UserWarning, match="belong to no subgraph"):
        decoders.SubgraphDecoder(
            [[0]], [[0]], decoder=decoders.lookup(max_weight=1)
        ).compile_decoder_for_dem(uncovered_dem)

    sound_dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D1 L1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoders.SubgraphDecoder(
            [[0], [1]],
            [[0], [1]],
            decoder=decoders.lookup(max_weight=1),
        ).compile_decoder_for_dem(sound_dem)


def test_subgraph_decoder_with_erasure() -> None:
    """SubgraphDecoder collects one erasure bit per subgraph past the observables."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.1) D2 L1
    """)
    decoder = decoders.SubgraphDecoder(
        [[0, 1], [2]], decoder=decoders.lookup(max_weight=1, add_erasure_bit=True)
    )
    compiled = decoder.compile_decoder_for_dem(dem)

    assert compiled.num_observables == dem.num_observables
    assert compiled.num_erasure_bits == 2

    shots = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    result = compiled.decode_shots(shots)
    assert result.shape == (3, dem.num_observables + 2)
    assert np.array_equal(result[:, :2], [[1, 0], [0, 1], [0, 0]])
    assert np.all(result[:, 2:] == 0)

    unknown_result = compiled.decode_shots(np.array([[1, 0, 0]], dtype=np.uint8))
    assert unknown_result[0, 2] == 1
    assert unknown_result[0, 3] == 0

    packed_flips = compiled.decode_shots_bit_packed(compiled.packbits(shots))
    assert packed_flips.shape == (3, 1 + 1)
    packed_unknown = compiled.decode_shots_bit_packed(
        compiled.packbits(np.array([[1, 0, 0]], dtype=np.uint8))
    )
    assert packed_unknown[0, -1] == 1
