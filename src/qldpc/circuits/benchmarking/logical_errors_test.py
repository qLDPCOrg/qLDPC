# SPDX-License-Identifier: Apache-2.0

"""Tests for sampled logical error rates and discard rates."""

from __future__ import annotations

from collections.abc import Collection
from typing import cast

import numpy as np
import pytest
import sinter
import stim

from qldpc import circuits, decoders
from qldpc.circuits.benchmarking import logical_errors


def test_logical_error_and_discard_rate() -> None:
    """Sample a noiseless circuit and post-select detectors or incorrect observables."""
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")

    # bypass sinter to compute logical error rates
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit,
        decoder=decoders.TrivialDecoder(),
        num_samples=1,
        post_select=range(circuit.num_detectors),
    )
    assert logical_error_rate == 0
    assert discard_rate == 0

    # post-selecting on a detector that always fires discards every shot, also for arrays and True
    dem = stim.DetectorErrorModel("error(1) D0 L0")
    for post_select in ([0], np.array([0]), True):
        logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
            dem, decoder=decoders.TrivialDecoder(), num_samples=2, post_select=post_select
        )
        assert np.isnan(logical_error_rate)
        assert discard_rate == 1

    # post-select on observables
    dem = circuit.detector_error_model()
    dem.append("error", [1], [stim.DemTarget.logical_observable_id(circuit.num_observables)])
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        dem,
        decoder=decoders.TrivialDecoder(),
        num_samples=1,
        post_select_observables=[circuit.num_observables],
    )
    assert np.isnan(logical_error_rate)
    assert discard_rate == 1

    # a decoder built into Sinter, whose compile_decoder_for_dem takes the model by keyword only
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        stim.DetectorErrorModel("error(0.1) D0 L0"),
        decoder=sinter.BUILT_IN_DECODERS["pymatching"],
        num_samples=10,
    )
    assert logical_error_rate == 0
    assert discard_rate == 0

    # sampling nothing reports no rates rather than dividing by zero
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit, decoder=decoders.TrivialDecoder(), num_samples=0
    )
    assert np.isnan(logical_error_rate)
    assert discard_rate == 0

    # incompatible DEMs for sampling and decoding
    with pytest.raises(ValueError, match="Incompatible detector error models"):
        circuits.get_logical_error_and_discard_rate(
            circuit,
            decoder=decoders.TrivialDecoder(),
            num_samples=1,
            dem_to_decode=stim.DetectorErrorModel(),
        )


def test_erasure_counts_as_a_discard() -> None:
    """A decoder that signals erasure has its erased shots discarded, not counted as errors."""
    # seven independent qubits, so an erasure bit does not fit in the last byte of the observables
    num_qubits = 7
    circuit = stim.Circuit()
    for qubit in range(num_qubits):
        circuit.append("X_ERROR", [qubit], 0.2)
        circuit.append("M", [qubit])
    for index in range(num_qubits):
        circuit.append("DETECTOR", [stim.target_rec(-num_qubits + index)])
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits + index)], [index])

    # every error of weight one is decoded exactly, and every heavier error is erased, so the
    # logical error rate is exactly zero and the discard rate is 1 - 0.8**7 - 7 * 0.2 * 0.8**6
    # a decoder specification is built for the sampled detector error model
    decoder = decoders.lookup(max_weight=1, add_erasure_bit=True)
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit, decoder, num_samples=1000
    )
    assert logical_error_rate == 0
    assert 0.30 < discard_rate < 0.55


def test_postselection_mask() -> None:
    """Post-selection masks are bit-packed, accept negative indices, and reject other indices."""
    assert logical_errors._get_postselection_mask((), 3, "Detector") is None
    mask = logical_errors._get_postselection_mask(np.array([0, -1]), 10, "Detector")
    assert mask is not None
    assert np.array_equal(mask, [1, 2])
    with pytest.raises(ValueError, match="Observable post-selection"):
        logical_errors._get_postselection_mask([1], 1, "Observable")
    # invalid indices are passed on unchanged, to test their errors
    for indices in (0, [0.5], [True]):
        with pytest.raises(TypeError, match="Observable post-selection"):
            logical_errors._get_postselection_mask(
                cast("Collection[int]", indices), 1, "Observable"
            )
