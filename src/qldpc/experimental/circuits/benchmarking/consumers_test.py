# SPDX-License-Identifier: Apache-2.0

"""Unit tests for consumers.py."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import scipy.stats
import stim

from qldpc import circuits, codes, decoders
from qldpc.decoders.construction.specs import observable_decoder_spec
from qldpc.experimental.circuits.benchmarking import consumers
from qldpc.experimental.circuits.benchmarking.consumers import Consumer, ConsumerCalibration
from qldpc.experimental.circuits.benchmarking.residual_weights import (
    ResidualWeightHistogram,
    estimate_residual_weight_histogram,
)
from qldpc.objects import Pauli

_CODE = codes.SteaneCode()
_NUM_QUBITS = len(_CODE)


def test_calibration_matches_brute_force() -> None:
    """Stratified estimates for a noisy Steane-code readout match brute force within sampling error.

    The consumer flips each qubit of a logical |0> state with probability q.  It then reads out the
    Z-type stabilizers and logical Z operator noiselessly, and decodes with a lookup table.  It
    therefore fails exactly when a minimum-weight decoder mispredicts the logical Z operator.  Its
    seven error mechanisms are X errors on the seven qubits, with equal odds, so a set of g of them
    is a uniformly random X error of weight g.  Z-type errors do not affect the consumer, so their
    paired changes vanish exactly, but their standard errors stay positive.  No Z-type error on
    logical |0> has residual weight above one, so that row is unsampled.
    """
    error_rate, max_residual_weight, num_samples = 0.05, 2, 2_000
    consumer = _get_readout_consumer(error_rate)
    calibration = consumer.calibrate(
        max_residual_weight, num_samples, max_fault_weight=_NUM_QUBITS, seed=0
    )
    assert calibration.truncated_probability == 0
    assert calibration.excess_standard_errors is not None
    assert calibration.excess_num_samples is not None
    assert calibration.baseline_standard_errors is not None

    failures, residual_weights = _get_readout_failures()
    weights = _VECTORS.sum(axis=1)
    # each error weight w contributes num_samples uniformly random errors of weight w
    frequencies = (weights <= max_residual_weight + 1) / np.array(
        [math.comb(_NUM_QUBITS, w) for w in weights]
    )
    excesses = np.asarray(calibration.excess_failure_probabilities[Pauli.X])
    errors = np.asarray(calibration.excess_standard_errors[Pauli.X])
    baseline = np.asarray(calibration.baseline_failure_probabilities)
    baseline_errors = np.asarray(calibration.baseline_standard_errors)
    # the baseline pools the samples of both Pauli types and the physical error weights 1, 2, and 3
    num_baseline_samples = 2 * (max_residual_weight + 1) * num_samples
    baseline_failures = np.round(baseline * num_baseline_samples)
    beta = scipy.stats.beta(baseline_failures + 0.5, num_baseline_samples - baseline_failures + 0.5)
    assert np.allclose(baseline_errors, beta.std())
    for fault_weight in range(_NUM_QUBITS + 1):
        consumer_faults = weights == fault_weight
        # the failure rate of each X-type block error with g random consumer faults
        failure_rates = np.array(
            [failures[error ^ _VECTORS_INDEX][consumer_faults].mean() for error in range(128)]
        )
        expected_baseline = failure_rates[0]
        assert (
            abs(baseline[fault_weight] - expected_baseline)
            <= 5 * baseline_errors[fault_weight] + 1e-12
        )
        for residual_weight in range(1, max_residual_weight + 1):
            sampled = frequencies * (residual_weights == residual_weight)
            expected = sampled @ failure_rates / sampled.sum() - expected_baseline
            excess, error = (
                excesses[residual_weight - 1, fault_weight],
                errors[residual_weight - 1, fault_weight],
            )
            assert abs(excess - expected) <= 5 * error + 1e-12

    # at the consumer's noise, the stratified estimates combine to the target excess probability
    consumer_noise = _get_error_probabilities(error_rate)
    changes = np.array([consumer_noise @ failures[error ^ _VECTORS_INDEX] for error in range(128)])
    changes -= consumer_noise @ failures
    for residual_weight in range(1, max_residual_weight + 1):
        sampled = frequencies * (residual_weights == residual_weight)
        expected = sampled @ changes / sampled.sum()
        x_histogram = _get_histogram(np.eye(max_residual_weight + 2)[residual_weight], "X")
        z_histogram = _get_histogram(np.eye(max_residual_weight + 2)[0], "Z")
        excess, error = calibration.get_excess_failure_probability(x_histogram, z_histogram)
        assert abs(excess - expected) < 5 * error

    excesses = np.asarray(calibration.excess_failure_probabilities[Pauli.Z])
    errors = np.asarray(calibration.excess_standard_errors[Pauli.Z])
    counts = np.asarray(calibration.excess_num_samples[Pauli.Z])
    assert np.all(excesses[0] == 0) and np.all(counts[0] > 0)
    assert np.allclose(errors[0], [_get_dirichlet_deviation(0, count, 0) for count in counts[0]])
    assert np.array_equal(excesses[1], 1 - baseline) and np.all(counts[1] == 0)
    assert np.all(errors[1] == 0)


def test_certain_failures() -> None:
    """Excesses are the paired estimates, with their own standard errors, also at the boundary.

    The block is two qubits in |00>.  The decoder of the consumer predicts no flips, so the consumer
    fails if either of its observables flips.  These observables are the measurements of block
    qubit 0 and of qubit 2.  Error mechanism A flips qubit 2, and mechanism B flips a detector that
    the decoder ignores.  Without a block error, the consumer therefore fails exactly when A occurs.
    The block error X0*X1 of residual weight two flips the first observable, so the consumer always
    fails with it.  Its paired change in failure is 1 exactly when A does not occur.  Given one
    fault, A occurs with probability odds(A) / (odds(A) + odds(B)).  Every sample of fault weight
    zero changes, and none of fault weight two does, but their standard errors stay positive.
    """
    probability_a, probability_b = 0.05, 0.2
    circuit = stim.Circuit(
        f"""
        CORRELATED_ERROR({probability_a}) X2
        CORRELATED_ERROR({probability_b}) X3
        M 0 1 2 3
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(1) rec[-2]
        """
    )
    decoder = _UnbatchedDecoder(lambda syndrome: np.zeros(2, dtype=int))
    consumer = Consumer(stim.Circuit("R 0 1"), circuit, decoder, block_qubits=(0, 1))
    calibration = consumer.calibrate(2, 200, max_fault_weight=2, paulis=["X"], seed=0)
    assert calibration.excess_standard_errors is not None
    assert calibration.excess_num_samples is not None
    excesses = np.asarray(calibration.excess_failure_probabilities[Pauli.X])[1]
    errors = np.asarray(calibration.excess_standard_errors[Pauli.X])[1]
    counts = np.asarray(calibration.excess_num_samples[Pauli.X])[1]
    assert excesses[0] == 1 and excesses[2] == 0
    assert errors[0] == pytest.approx(_get_dirichlet_deviation(counts[0], 0, 0))
    assert errors[2] == pytest.approx(_get_dirichlet_deviation(0, counts[2], 0))

    odds_a, odds_b = (p / (1 - p) for p in (probability_a, probability_b))
    mean = excesses[1]
    increases = round(mean * counts[1])
    expected_error = _get_dirichlet_deviation(increases, counts[1] - increases, 0)
    assert errors[1] == pytest.approx(expected_error)
    assert abs(mean - odds_b / (odds_a + odds_b)) <= 5 * errors[1]

    # the excess of residual weight two is that of a certain failure, beyond the table
    excess, error = calibration.get_excess_failure_probability(_get_histogram([0, 0, 1, 0], "X"))
    certain, certain_error = calibration.get_excess_failure_probability(
        _get_histogram([0, 0, 0, 1], "X")
    )
    assert abs(excess - certain) <= 5 * math.hypot(error, certain_error)


def test_block_errors_through_a_clifford_network() -> None:
    """Direct checks propagate X, Y, and Z block errors as Stim does.

    A Bell pair on qubits 0 and 1 passes through a noiseless Clifford network.  Its Hadamard and
    CNOT gates turn X errors into Z errors and spread errors onto qubit 2.  A preparation leaves a
    fixed Pauli error on the Bell pair, so the excess is one if the consumer then fails, and zero
    otherwise.  Decoders that predict the observable flip to be zero, or the flip of either
    detector, together reveal every detector and observable flip of a Stim simulation.
    """
    preparation = stim.Circuit("H 0\nCX 0 1")
    for detector in [None, 0, 1]:
        decoder = _get_detector_decoder(detector)
        consumer = Consumer(preparation, _get_network([0, 0, 0]), decoder)
        for paulis in [("X", "I"), ("I", "Z"), ("Y", "I"), ("Z", "Y")]:
            circuit = preparation.copy()
            for qubit, pauli in enumerate(paulis):
                if pauli != "I":
                    circuit.append(f"{pauli}_ERROR", [qubit], 1)
            detectors, observables = _sample_once(circuit + _get_network([0, 0, 0]))
            expected = np.any(decoder.decode_observables(detectors) != observables)
            value, _ = consumer.estimate_excess_failure_probability(circuit, 1, seed=0)
            assert value == expected


def test_simulated_calibration_matches_brute_force() -> None:
    """A calibration of a Clifford network agrees with brute force over its error mechanisms.

    The consumer runs the network on a Bell pair.  Brute force simulates the Bell pair, with each
    X-type or Z-type block error, through the network with every set of its error mechanisms.  Given
    g mechanisms, a set has a probability proportional to the product of their odds.  The two block
    errors of weight one of each type leave the same state and have residual weight one, and the
    error of weight two has residual weight zero.  With zero or three mechanisms, every sample has
    the same paired change, so the estimates are exact.  The other estimates agree with brute force
    within their standard errors.
    """
    probabilities = [0.1, 0.2, 0.15]
    odds = np.array([p / (1 - p) for p in probabilities])
    preparation = stim.Circuit("H 0\nCX 0 1")
    decoder = _get_detector_decoder(0)
    consumer = Consumer(preparation, _get_network(probabilities), decoder)
    calibration = consumer.calibrate(1, 2_000, max_fault_weight=3, seed=0)
    assert calibration.excess_standard_errors is not None
    assert calibration.baseline_standard_errors is not None
    assert np.allclose(
        sorted(np.asarray(calibration.mechanism_probabilities)), sorted(probabilities)
    )

    def get_failure(block_error: str, faults: Sequence[int]) -> bool:
        circuit = preparation + stim.Circuit(block_error) + _get_network(faults)
        detectors, observables = _sample_once(circuit)
        return bool(np.any(decoder.decode_observables(detectors) != observables))

    for fault_weight in range(4):
        sets = [
            faults for faults in itertools.product([0, 1], repeat=3) if sum(faults) == fault_weight
        ]
        weights = np.array([np.prod(odds[np.flatnonzero(faults)]) for faults in sets])
        weights /= weights.sum()
        expected_baseline = weights @ [get_failure("", faults) for faults in sets]
        baseline = np.asarray(calibration.baseline_failure_probabilities)[fault_weight]
        baseline_error = np.asarray(calibration.baseline_standard_errors)[fault_weight]
        exact = fault_weight in (0, 3)
        assert baseline_error > 0
        assert abs(baseline - expected_baseline) <= (0 if exact else 5 * baseline_error) + 1e-12
        for pauli in (Pauli.X, Pauli.Z):
            failures = weights @ [get_failure(f"{pauli}_ERROR(1) 0", faults) for faults in sets]
            excess = np.asarray(calibration.excess_failure_probabilities[pauli])[0, fault_weight]
            error = np.asarray(calibration.excess_standard_errors[pauli])[0, fault_weight]
            assert error > 0
            tolerance = 0 if exact else 5 * error
            assert abs(excess - (failures - expected_baseline)) <= tolerance + 1e-12


def test_jeffreys_deviations() -> None:
    """Posterior standard deviations follow from the Dirichlet and Beta distributions of SciPy.

    Counts of paired changes 1, 0, and -1 have a Dirichlet posterior.  Counts of failures among
    samples have a Beta posterior.  The cases include counts of zero, for which the plug-in standard
    error of a mean vanishes.
    """
    for increases, unchanged, decreases in [(0, 0, 0), (0, 5, 0), (3, 0, 0), (2, 7, 4), (0, 1, 9)]:
        counts = increases + unchanged + decreases
        deviation = consumers._get_jeffreys_deviations(increases, decreases, counts)
        expected = _get_dirichlet_deviation(increases, unchanged, decreases)
        assert deviation == pytest.approx(expected) and deviation > 0
        beta = scipy.stats.beta(increases + 0.5, counts - increases + 0.5)
        deviation = consumers._get_jeffreys_deviations(increases, None, counts)
        assert deviation == pytest.approx(beta.std()) and deviation > 0


def test_hard_threshold_calibration() -> None:
    """A threshold consumer has binomial excess failure probabilities.

    With n mechanisms of probability p, the fault weight g is binomial, also after rescaling p.  A
    block error of residual weight m <= t adds the failures with t - m < g <= t.  A heavier error,
    beyond the table, makes every shot fail and adds the probability of g <= t.
    """
    num_mechanisms, probability, correctable_weight, max_fault_weight = 300, 1e-4, 2, 3
    fault_weights = np.arange(max_fault_weight + 1)
    baseline = (fault_weights > correctable_weight).astype(float)
    residual_weights = np.arange(1, correctable_weight + 1)[:, None]
    excesses = (residual_weights + fault_weights > correctable_weight) - baseline
    calibration = ConsumerCalibration(
        {"X": excesses}, baseline, np.full(num_mechanisms, probability)
    )
    for scale, scaled in [(1, calibration), (2, calibration.rescaled(2))]:
        distribution = scipy.stats.binom(num_mechanisms, scale * probability)
        for residual_weight in range(1, correctable_weight + 2):
            histogram = _get_histogram(np.eye(correctable_weight + 2)[residual_weight], "X")
            value, error = scaled.get_excess_failure_probability(histogram)
            lowest = max(correctable_weight - residual_weight + 1, 0)
            expected = distribution.cdf(correctable_weight) - distribution.cdf(lowest - 1)
            assert value == pytest.approx(expected) and error == 0
        assert scaled.truncated_probability == pytest.approx(distribution.sf(max_fault_weight))
    assert calibration.rescaled(2).description == "(consumer error probabilities scaled by 2)"
    with pytest.raises(ValueError):
        calibration.rescaled(1e5)
    with pytest.raises(ValueError):
        ConsumerCalibration({"X": [[0.5]]}, [0.1]).rescaled(2)


def test_excess_failure_matches_brute_force() -> None:
    """Paired samples match the exact excess failure of a noisy preparation within sampling error.

    The preparation flips each qubit of logical |0> with probability r, and also applies Z errors,
    which do not affect the readout consumer.  A correlated error that fires a post-selected flag
    detector leaves a logical error, which post-selection removes.
    """
    error_rate, preparation_error_rate, num_samples = 0.05, 0.03, 40_000
    consumer = _get_readout_consumer(error_rate)
    qubits = " ".join(map(str, range(_NUM_QUBITS)))
    circuit = consumer.block_preparation + stim.Circuit(
        f"""
        X_ERROR({preparation_error_rate}) {qubits}
        Z_ERROR({preparation_error_rate}) {qubits}
        CORRELATED_ERROR(0.2) X0 X1 X2 X3 X4 X5 X6 X7
        M 7
        DETECTOR rec[-1]
        """
    )
    value, error = consumer.estimate_excess_failure_probability(
        circuit, num_samples, post_select=True, seed=0
    )
    failures, _ = _get_readout_failures()
    consumer_noise = _get_error_probabilities(error_rate)
    changes = np.array([consumer_noise @ failures[error ^ _VECTORS_INDEX] for error in range(128)])
    changes -= consumer_noise @ failures
    expected = _get_error_probabilities(preparation_error_rate) @ changes
    assert 0 < error and abs(value - expected) < 5 * error

    for pauli in (Pauli.X, Pauli.Z):
        value, _ = consumer.estimate_excess_failure_probability(
            circuit, 1000, pauli=pauli, post_select=True, seed=0
        )
        assert (value == 0) == (pauli is Pauli.Z)


def test_decoders() -> None:
    """Consumers accept decoders without batch decoding, and reject decoders of the wrong kind.

    With a noiseless readout, every X-type error of residual weight one is corrected, and no sample
    fails, but the standard errors of the baseline stay positive.  A decoder that signals erasure
    predicts one flip too many.
    """
    consumer, noisy = _get_readout_consumer(0.0), _get_readout_consumer(0.1)
    decoder = decoders.lookup(max_weight=1).build_observable_decoder(
        (noisy.block_preparation + noisy.circuit).detector_error_model()
    )
    unbatched = Consumer(
        consumer.block_preparation, consumer.circuit, _UnbatchedDecoder(decoder.decode_observables)
    )
    calibration = unbatched.calibrate(1, 10, paulis=["X"], seed=0)
    assert list(calibration.excess_failure_probabilities) == [Pauli.X]
    assert np.all(np.asarray(calibration.excess_failure_probabilities[Pauli.X]) == 0)
    assert np.all(np.asarray(calibration.baseline_failure_probabilities) == 0)
    assert calibration.baseline_standard_errors is not None
    # 10 samples of each of the physical error weights 1 and 2, none of which fails
    expected_error = scipy.stats.beta(0.5, 20.5).std()
    assert np.allclose(np.asarray(calibration.baseline_standard_errors), expected_error)

    erasing = dataclasses.replace(
        noisy, decoder=decoders.lookup(max_weight=1, add_erasure_bit=True)
    )
    with pytest.raises(ValueError, match=r"of shape \(\d+, 1\)"):
        erasing.calibrate(1, 10)
    with pytest.raises(TypeError, match="must be an observable decoder"):
        Consumer(consumer.block_preparation, consumer.circuit, None)  # type: ignore[arg-type]


def test_prediction_shapes() -> None:
    """A consumer requires one prediction per shot and observable, and rejects a transposed batch.

    The consumer measures two qubits as two observables, and three shots give a batch of shape
    (3, 2).  A decoder of one observable may instead predict one flip per shot.
    """
    circuit = stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]")
    preparation = stim.Circuit("R 0 1\nX_ERROR(1) 0")
    transposed = Consumer(stim.Circuit("R 0 1"), circuit, _BatchDecoder(lambda batch: batch.T))
    with pytest.raises(ValueError, match=r"of shape \(3, 2\)"):
        transposed.estimate_excess_failure_probability(preparation, 3, seed=0)

    one_observable = stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    flat = Consumer(stim.Circuit("R 0"), one_observable, _BatchDecoder(lambda batch: batch[:, 0]))
    value, _ = flat.estimate_excess_failure_probability(stim.Circuit("R 0\nX_ERROR(1) 0"), 3)
    assert value == 1


def test_decoder_specifications() -> None:
    """Decoder specifications get the detector error model of the consumer with its metadata.

    A rotated surface-code memory with depolarizing noise has Y errors that flip three detectors.
    Matching decodes it with the decompositions that Stim suggests, and corrects any single fault.
    A specification also sees the detector coordinates of the circuit.  An error that Stim cannot
    decompose, here one that flips three detectors and nothing else, stays whole.
    """
    memory = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=1e-3
    )
    data = [1, 3, 5, 8, 10, 12, 15, 17, 19]
    preparation = stim.Circuit(f"R {' '.join(map(str, data))}")
    consumer = Consumer(
        preparation, memory, decoders.mwpm(decompose_errors=True), block_qubits=data
    )
    calibration = consumer.calibrate(1, 10, seed=0)
    assert np.all(np.asarray(calibration.baseline_failure_probabilities)[:2] == 0)

    models: list[stim.DetectorErrorModel] = []

    def build(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        models.append(dem)
        return _get_detector_decoder(None)

    recording = observable_decoder_spec("recording", build)()
    dataclasses.replace(consumer, decoder=recording).calibrate(0, 1, seed=0)
    undecomposable = stim.Circuit(
        """
        CORRELATED_ERROR(0.1) X0 X1 X2
        M 0 1 2
        DETECTOR(0, 1) rec[-3]
        DETECTOR(1, 1) rec[-2]
        DETECTOR(2, 1) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    Consumer(stim.Circuit("R 0 1 2"), undecomposable, recording).calibrate(0, 1, seed=0)
    for circuit, decomposed in [(preparation + memory, True), (undecomposable, False)]:
        model = models.pop(0)
        assert model.get_detector_coordinates() == circuit.get_detector_coordinates()
        separators = [
            target.is_separator()
            for instruction in model.flattened()
            for target in instruction.targets_copy()
            if isinstance(target, stim.DemTarget)
        ]
        assert any(separators) == decomposed


def test_output_qubits_and_description() -> None:
    """Output qubits map to block qubits in order, and calibrations keep the consumer description.

    The block is two qubits in |00>, and the consumer measures block qubit 0 as its observable.  Its
    decoder predicts no flip.  A preparation leaves an X error on its qubit 0, which becomes block
    qubit 0 or 1, depending on the order of the output qubits.  The consumer then fails exactly when
    that block qubit is 0.  Every sample has the same paired change, which still has a positive
    standard error.
    """
    circuit = stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-2]")
    preparation = stim.Circuit("R 0 1\nX_ERROR(1) 0")
    num_samples = 3
    consumer = Consumer(
        stim.Circuit("R 0 1"),
        circuit,
        _get_detector_decoder(None),
        description="two measured qubits",
    )
    for output_qubits in [(0, 1), (1, 0)]:
        value, error = consumer.estimate_excess_failure_probability(
            preparation, num_samples, output_qubits=output_qubits, seed=0
        )
        changes = int(output_qubits.index(0) == 0)
        assert value == changes
        expected = _get_dirichlet_deviation(changes * num_samples, (1 - changes) * num_samples, 0)
        assert error == pytest.approx(expected)
    assert consumer.calibrate(0, 1, seed=0).description == "two measured qubits"


def test_default_max_residual_weight() -> None:
    """By default, a calibration resolves residual weights up to the correctable weight plus one."""
    consumer = dataclasses.replace(_get_readout_consumer(0.0), correctable_weight=1)
    calibration = consumer.calibrate(num_samples=10, paulis=["X"], seed=0)
    assert len(np.asarray(calibration.excess_failure_probabilities[Pauli.X])) == 2


def test_invalid_consumers() -> None:
    """Invalid consumers, blocks, preparation circuits, and codes raise errors."""
    consumer = _get_readout_consumer(0.0)
    preparation, circuit, decoder = consumer.block_preparation, consumer.circuit, consumer.decoder
    with pytest.raises(ValueError, match="must be noiseless"):
        Consumer(preparation + stim.Circuit("X_ERROR(0.1) 0"), circuit, decoder)
    with pytest.raises(TypeError, match="block_qubits"):
        Consumer(preparation, circuit, decoder, block_qubits=(0.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block_qubits"):
        Consumer(preparation, circuit, decoder, block_qubits=(0, 0))
    with pytest.raises(ValueError, match="correctable_weight must be at least 0"):
        Consumer(preparation, circuit, decoder, correctable_weight=-1)
    with pytest.raises(ValueError, match="give the consumer a correctable_weight"):
        consumer.calibrate()
    with pytest.raises(ValueError, match="paulis must have"):
        consumer.calibrate(1, 10, paulis=["X", "x"])
    with pytest.raises(ValueError, match="probability one"):
        _get_readout_consumer(1.0).calibrate(1, 10)

    trivial_decoder = _UnbatchedDecoder(lambda syndrome: np.zeros(1, dtype=int))
    with pytest.raises(ValueError, match="at least one observable"):
        Consumer(preparation, stim.Circuit(), trivial_decoder)
    observable = stim.Circuit("OBSERVABLE_INCLUDE(0) Z0")
    bell_pair = Consumer(
        stim.Circuit("H 0\nCX 0 1"), observable, trivial_decoder, block_qubits=(0,)
    )
    with pytest.raises(ValueError, match="definite pure stabilizer state"):
        bell_pair.calibrate(1)
    not_css = Consumer(
        stim.Circuit("H 0\nS 0"), stim.Circuit("OBSERVABLE_INCLUDE(0) Y0"), trivial_decoder
    )
    with pytest.raises(ValueError, match="require a CSS state"):
        not_css.calibrate(1)

    flagged = preparation + stim.Circuit("X_ERROR(1) 7\nM 7\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="discarded every shot"):
        consumer.estimate_excess_failure_probability(flagged, 10, post_select=True)


def test_fixed_noise_costs() -> None:
    """Costs at a fixed noise level, in one column, score a preparation exactly.

    A preparation of logical |0> of the Steane code flips each qubit with probability r and applies
    a Z error to each qubit with probability s.  The X-type residual weight of an error is the least
    weight of an error with the same syndrome and logical Z value.  The Z-type residual weight is 0
    for a Z error with a trivial syndrome, and 1 otherwise.  With c(0) = 0.1, X-type residual
    weights above two cost 1 - c(0) = 0.9.
    """
    x_rate, z_rate = 0.02, 0.03
    calibration = ConsumerCalibration(
        {"z": [[0.3]], Pauli.X: [[0.1], [0.2]]},
        [0.1],
        excess_standard_errors={"z": [[0.02]], Pauli.X: [[0], [0]]},
    )
    _, x_residual_weights = _get_readout_failures()
    x_distribution = np.bincount(
        np.minimum(x_residual_weights, 3), weights=_get_error_probabilities(x_rate), minlength=4
    )
    checks = _CODE.get_stabilizer_ops(Pauli.X).view(np.ndarray).astype(int)
    z_error = _get_error_probabilities(z_rate) @ np.any(_VECTORS @ checks.T % 2, axis=1)
    expected = x_distribution @ [0, 0.1, 0.2, 0.9] + 0.3 * z_error
    value, error = calibration.estimate_excess_failure_probability(
        _get_noisy_preparation(x_rate, z_rate), output_qubits=range(_NUM_QUBITS)
    )
    assert value == pytest.approx(expected) and error == pytest.approx(0.02 * z_error)


def test_state_check() -> None:
    """A calibration with the stabilizers of its state checks the state of a preparation."""
    circuit = _get_noisy_preparation(0.02, 0.03)
    stabilizers = circuits.get_state_stabilizers(circuit, _NUM_QUBITS)
    calibration = ConsumerCalibration({"X": [[0.5]]}, [0.1], state_stabilizers=stabilizers)
    value, _ = calibration.estimate_excess_failure_probability(circuit, 10, seed=0)
    assert value > 0
    plus = circuits.get_encoding_circuit(_CODE, only_zero=True)
    plus.append("H", range(_NUM_QUBITS))
    with pytest.raises(ValueError):
        calibration.estimate_excess_failure_probability(plus)
    with pytest.raises(ValueError):
        calibration.estimate_excess_failure_probability(circuit, output_qubits=range(6))


def test_histogram_requirements() -> None:
    """Scores need one histogram per calibrated type, which resolves the weights of its table."""
    calibration = ConsumerCalibration({"X": [[0.1], [0.2]], "Z": [[0.3]]}, [0.1])
    x_histogram, z_histogram = _get_histogram([1, 0, 0, 0], "X"), _get_histogram([1, 0, 0], "Z")
    with pytest.raises(ValueError):
        calibration.get_excess_failure_probability(x_histogram)
    with pytest.raises(ValueError):
        calibration.get_excess_failure_probability(x_histogram, _get_histogram([1, 0, 0], None))
    with pytest.raises(ValueError):
        calibration.get_excess_failure_probability(_get_histogram([1, 0, 0], "X"), z_histogram)
    assert calibration.get_excess_failure_probability(x_histogram, z_histogram) == (0, 0)


def test_cell_standard_errors() -> None:
    """The standard errors of the cells add in quadrature, weighted by Pr[m] and Pr[g].

    With one mechanism of probability 1/2, Pr[g] = 1/2 for g = 0 and 1.  A histogram with
    Pr[m = 1] = 0.3 and Pr[m = 2] = 0.2 weights the cells of residual weight one by 0.15 and those
    of residual weight two by 0.1.
    """
    calibration = ConsumerCalibration(
        {"X": [[0.1, 0.3], [0.2, 0.4]]},
        [0, 0],
        [0.5],
        excess_standard_errors={"X": [[0.02, 0.04], [0.06, 0.08]]},
    )
    value, error = calibration.get_excess_failure_probability(
        _get_histogram([0.5, 0.3, 0.2, 0], "X")
    )
    assert value == pytest.approx(0.3 * 0.2 + 0.2 * 0.3)
    assert error == pytest.approx(math.sqrt(0.003**2 + 0.006**2 + 0.006**2 + 0.008**2))


def test_shared_baseline_error() -> None:
    """Every residual weight beyond a table shares the error of the baseline.

    Each of four qubits of |0000> flips with probability 0.3.  A nonzero X-type residual weight
    therefore has probability 1 - 0.7**4, however finely a histogram resolves it, and costs
    1 - c(0) = 0.8 without a table.  Qubits in |+> instead suffer Z errors with probability 0.2.
    The X-type and Z-type certain failures share the same baseline, so their errors add linearly.
    """
    calibration = ConsumerCalibration({"X": []}, [0.2], baseline_standard_errors=[0.1])
    circuit = stim.Circuit("X_ERROR(0.3) 0 1 2 3")
    for max_residual_weight in [0, 1, 4]:
        histogram = estimate_residual_weight_histogram(
            circuit, max_residual_weight, output_qubits=range(4), pauli="X"
        )
        value, error = calibration.get_excess_failure_probability(histogram)
        assert value == pytest.approx((1 - 0.7**4) * 0.8)
        assert error == pytest.approx((1 - 0.7**4) * 0.1)

    calibration = ConsumerCalibration({"X": [], "Z": []}, [0.2], baseline_standard_errors=[0.1])
    circuit = stim.Circuit("H 2 3\nX_ERROR(0.3) 0 1\nZ_ERROR(0.2) 2 3")
    value, error = calibration.estimate_excess_failure_probability(circuit, output_qubits=range(4))
    probability = (1 - 0.7**2) + (1 - 0.8**2)
    assert value == pytest.approx(probability * 0.8)
    assert error == pytest.approx(probability * 0.1)


def test_unsampled_cells() -> None:
    """A cell that no sample reached counts as certain failure, with the error of the baseline.

    With one mechanism of probability p = 0.1, Pr[g = 0] = 0.9 and Pr[g = 1] = 0.1.  The cell
    (m, g) = (1, 1) is unsampled, so its excess is 1 - f(0, 1) = 0.8.  It shares the error 0.02 of
    f(0, 1) with residual weight two, beyond the table, which costs 1 - c(0) = 1 - 0.11.
    """
    calibration = ConsumerCalibration(
        {"X": [[0.2, 0.8]]},
        [0.1, 0.2],
        [0.1],
        baseline_standard_errors=[0.01, 0.02],
        excess_num_samples={"X": [[10, 0]]},
    )
    value, error = calibration.get_excess_failure_probability(_get_histogram([0.5, 0.3, 0.2], "X"))
    assert value == pytest.approx(0.3 * (0.9 * 0.2 + 0.1 * 0.8) + 0.2 * 0.89)
    assert error == pytest.approx(math.hypot(0.9 * 0.2 * 0.01, 0.1 * (0.2 + 0.3) * 0.02))

    # scores keep the signs of excesses
    negative = dataclasses.replace(
        calibration, excess_failure_probabilities={"X": [[-0.1, -0.1]]}, excess_num_samples=None
    )
    value, _ = negative.get_excess_failure_probability(_get_histogram([0.5, 0.3, 0.2], "X"))
    assert value == pytest.approx(0.3 * -0.1 + 0.2 * 0.89)


def test_same_seed() -> None:
    """The same seed gives the same sampled estimate."""
    calibration = ConsumerCalibration({"X": [[0.5]], "Z": [[-0.5]]}, [0.5])
    circuit = _get_flagged_bell_pair()
    estimate = calibration.estimate_excess_failure_probability(circuit, 300, seed=0)
    assert estimate == calibration.estimate_excess_failure_probability(circuit, 300, seed=0)
    assert estimate[1] > 0


def test_independent_histograms_of_pauli_types() -> None:
    """The histograms of the two Pauli types of a score use independent random numbers.

    In the flagged Bell-pair circuit, the X-type and Z-type residual weights have the same
    distribution, and the histograms sample the heavier fault weights.  The two types have opposite
    excesses, so shared samples would cancel up to the overflow probability of about 1e-10.
    """
    calibration = ConsumerCalibration({"X": [[0.5]], "Z": [[-0.5]]}, [0.5])
    value, error = calibration.estimate_excess_failure_probability(
        _get_flagged_bell_pair(), 300, seed=0
    )
    assert abs(value) > 1e-6 and error > 0


def test_serialization() -> None:
    """A fully populated calibration survives saving and loading as JSON."""
    calibration = ConsumerCalibration(
        {"Z": [[0.1, 0.8]], "X": []},
        [0.1, 0.2],
        [0.01, 0.02],
        excess_standard_errors={"Z": [[0.01, 0]], "X": []},
        baseline_standard_errors=[0.001, 0.002],
        excess_num_samples={"Z": [[100, 0]], "X": []},
        state_stabilizers=["+ZZ", "-XX"],
        description="a consumer",
    )
    data = calibration.to_dict()
    restored = ConsumerCalibration.from_dict(json.loads(json.dumps(data)))
    assert restored.to_dict() == data
    assert data["excess_failure_probabilities"] == {"X": [], "Z": [[0.1, 0.8]]}
    assert data["state_stabilizers"] == ["+ZZ", "-XX"]
    for field in dataclasses.fields(ConsumerCalibration):
        value, restored_value = getattr(calibration, field.name), getattr(restored, field.name)
        if isinstance(value, dict):
            assert value.keys() == restored_value.keys()
            assert all(np.array_equal(value[key], restored_value[key]) for key in value)
        elif isinstance(value, tuple):
            assert value == restored_value
        else:
            assert np.array_equal(value, restored_value)
    with pytest.raises(TypeError):
        ConsumerCalibration.from_dict(data | {"extra": 0})


def test_invalid_tables() -> None:
    """Invalid tables, Pauli keys, and baselines raise errors."""
    table = np.zeros((2, 3))
    cases: list[tuple[dict[Any, Any], list[float]]] = [
        ({}, [0] * 3),
        ({"X": table, "x": table}, [0] * 3),
        ({"Y": table}, [0] * 3),
        ({"X": table[:, :2]}, [0] * 3),
        ({"X": table + np.nan}, [0] * 3),
        ({"X": table + 2}, [0] * 3),
        ({"X": table}, []),
        ({"X": table}, [0, 0, 2]),
    ]
    for tables, baseline in cases:
        with pytest.raises(ValueError):
            ConsumerCalibration(tables, baseline)


def test_exact_cells_are_probabilities() -> None:
    """Exact cells must have failure probabilities in [0, 1].  Estimated cells may stray outside.

    With the baseline 0.5, the cells (1, 0) and (1, 1) have failure probabilities 1.1 and -0.1.  A
    cell is exact if its excess and its baseline both have zero standard error.
    """
    tables: dict[Any, Any] = {"X": [[0.6, -0.6]]}
    baseline = [0.5, 0.5]
    with pytest.raises(ValueError):
        ConsumerCalibration(tables, baseline)
    with pytest.raises(ValueError):
        ConsumerCalibration(tables, baseline, excess_standard_errors={"X": [[0.1, 0]]})
    with pytest.raises(ValueError):
        ConsumerCalibration(tables, baseline, baseline_standard_errors=[0, 0.1])
    ConsumerCalibration(tables, baseline, excess_standard_errors={"X": [[0.1, 0.1]]})
    ConsumerCalibration(tables, baseline, baseline_standard_errors=[0.1, 0.1])
    ConsumerCalibration(
        tables,
        baseline,
        excess_standard_errors={"X": [[0.1, 0]]},
        baseline_standard_errors=[0, 0.1],
    )


def test_invalid_standard_errors() -> None:
    """Standard errors must be nonnegative and finite, in the layout of the tables."""
    tables: dict[Any, Any] = {"X": [[0.1, 0.2]]}
    baseline = [0, 0]
    for errors in [[[0.1]], [[-0.1, 0.1]], [[np.inf, 0.1]], [[np.nan, 0.1]]]:
        with pytest.raises(ValueError):
            ConsumerCalibration(tables, baseline, excess_standard_errors={"X": errors})
    for baseline_errors in [[0.1], [-0.1, 0.1], [np.inf, 0.1], [np.nan, 0.1]]:
        with pytest.raises(ValueError):
            ConsumerCalibration(tables, baseline, baseline_standard_errors=baseline_errors)


def test_invalid_sample_counts() -> None:
    """Sample counts are nonnegative integers, and unsampled cells are certain failures."""
    tables: dict[Any, Any] = {"X": [[0.1, 0.8]]}
    baseline = [0.1, 0.2]
    invalid_counts: list[list[list[float]]] = [[[1.5, 0]], [[-1, 0]], [[1, 0, 0]], [[0, 0]]]
    for counts in invalid_counts:
        with pytest.raises(ValueError):
            ConsumerCalibration(tables, baseline, excess_num_samples={"X": counts})
    with pytest.raises(ValueError):
        ConsumerCalibration(
            tables,
            baseline,
            excess_standard_errors={"X": [[0.01, 0.01]]},
            excess_num_samples={"X": [[1, 0]]},
        )


def test_invalid_mechanism_probabilities() -> None:
    """Mechanism probabilities must lie in [0, 1]."""
    for probabilities in [[-0.1], [1.5], [np.nan]]:
        with pytest.raises(ValueError):
            ConsumerCalibration({"X": [[0.1]]}, [0], probabilities)


def test_invalid_state_stabilizers() -> None:
    """State stabilizers must be independent commuting generators of a state."""
    with pytest.raises(ValueError):
        ConsumerCalibration({"X": [[0.1]]}, [0], state_stabilizers=["XX"])


class _BatchDecoder:
    """A batch decoder whose predictions are a function of zeros of shape (shots, 2)."""

    def __init__(self, predict: Callable[[np.ndarray], np.ndarray]) -> None:
        self._predict = predict

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes."""
        return self._predict(np.zeros((len(syndromes), 2), dtype=int))

    # consumers decode in batches, but the protocol also requires this method
    decode_observables = decode_observables_batch


class _UnbatchedDecoder:
    """An observable decoder without batch decoding."""

    def __init__(self, decode: Callable[[np.ndarray], np.ndarray]) -> None:
        self._decode = decode

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode one syndrome."""
        return self._decode(syndrome)


def _get_detector_decoder(detector: int | None) -> _UnbatchedDecoder:
    """A decoder of one observable that predicts the flip of a given detector, or no flip."""
    return _UnbatchedDecoder(
        lambda syndrome: np.full(1, 0 if detector is None else syndrome[detector])
    )


def _get_dirichlet_deviation(increases: int, unchanged: int, decreases: int) -> float:
    """The standard deviation of p₊ - p₋ under the Jeffreys-prior posterior of paired changes.

    The posterior of the probabilities of the changes 1, 0, and -1 is a Dirichlet distribution,
    whose covariance SciPy computes.
    """
    alpha = np.array([increases, unchanged, decreases]) + 0.5
    weights = np.array([1, 0, -1])
    return float(np.sqrt(weights @ scipy.stats.dirichlet(alpha).cov() @ weights))


def _sample_once(circuit: stim.Circuit) -> tuple[np.ndarray, np.ndarray]:
    """The detector and observable flips of one shot of a circuit."""
    detectors, observables = circuit.compile_detector_sampler().sample(1, separate_observables=True)
    return detectors[0].astype(np.uint8), observables[0].astype(np.uint8)


# every binary vector on the qubits of the Steane code, with bit j for qubit j
_VECTORS = np.array(list(itertools.product([0, 1], repeat=_NUM_QUBITS)), dtype=int)[:, ::-1]
_VECTORS_INDEX = np.arange(len(_VECTORS))


def _get_histogram(distribution: Sequence[float], pauli: Any) -> ResidualWeightHistogram:
    """An exact histogram with given probabilities of residual weights 0, 1, ... and overflow."""
    weight_sums = np.array([[0.0, *distribution]])
    return ResidualWeightHistogram(
        len(distribution) - 2,
        True,
        np.ones(1),
        0.0,
        weight_sums,
        np.zeros_like(weight_sums),
        np.zeros(1, dtype=int),
        np.ones(1, dtype=bool),
        pauli,
    )


def _get_noisy_preparation(x_rate: float, z_rate: float) -> stim.Circuit:
    """A preparation of logical |0> of the Steane code with X and Z errors on every qubit."""
    qubits = " ".join(map(str, range(_NUM_QUBITS)))
    return circuits.get_encoding_circuit(_CODE, only_zero=True) + stim.Circuit(
        f"X_ERROR({x_rate}) {qubits}\nZ_ERROR({z_rate}) {qubits}"
    )


def _get_flagged_bell_pair() -> stim.Circuit:
    """A Bell pair with many flag-flipping error mechanisms, every other one with a Y error.

    Each mechanism flips a flag qubit, whose measurement feeds a detector that is not post-selected.
    """
    circuit = stim.Circuit("H 0\nCX 0 1")
    for index in range(185):
        targets = [stim.target_y(0)] if index % 2 == 0 else []
        circuit.append("CORRELATED_ERROR", [*targets, stim.target_x(2)], 0.01)
        circuit.append("MR", [2])
        circuit.append("DETECTOR", [stim.target_rec(-1)])
    return circuit


def _get_readout_consumer(error_rate: float) -> Consumer:
    """A consumer that flips each qubit of logical |0> and noiselessly reads out its Z operators."""
    qubits = " ".join(map(str, range(_NUM_QUBITS)))
    circuit = stim.Circuit(f"X_ERROR({error_rate}) {qubits}")
    checks = _CODE.get_stabilizer_ops(Pauli.Z, symplectic=True)
    circuit += circuits.get_pauli_product_measurements(checks)
    for index in range(-len(checks), 0):
        circuit.append("DETECTOR", [stim.target_rec(index)])
    circuit += circuits.get_observables(_CODE, basis=Pauli.Z)
    return Consumer(
        circuits.get_encoding_circuit(_CODE, only_zero=True),
        circuit,
        decoders.lookup(max_weight=1),
        block_qubits=tuple(range(_NUM_QUBITS)),
    )


def _get_network(probabilities: Sequence[float]) -> stim.Circuit:
    """A Clifford network after a Bell pair, with three correlated errors of given probabilities."""
    return stim.Circuit(
        f"""
        CORRELATED_ERROR({probabilities[0]}) X0
        H 0
        CX 0 2
        CORRELATED_ERROR({probabilities[1]}) Z1 X2
        CX 1 2
        H 1
        CORRELATED_ERROR({probabilities[2]}) Y0
        MPP Z0*Z1*X2 Z0*X1*Z2 X0*X1*X2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def _get_readout_failures() -> tuple[np.ndarray, np.ndarray]:
    """Failures of the readout consumer, and residual weights, of each X-type error on logical |0>.

    Errors are indexed by integers, with bit j for qubit j.  The consumer fails if a lightest error
    with the same syndrome flips the logical Z operator differently.  The residual weight of an
    X-type error is the least weight of an error with the same syndrome and logical Z value.
    """
    checks = _CODE.get_stabilizer_ops(Pauli.Z).view(np.ndarray).astype(int)
    logicals = _CODE.get_logical_ops(Pauli.Z).view(np.ndarray).astype(int)
    syndromes = [tuple(row) for row in _VECTORS @ checks.T % 2]
    flips = [tuple(row) for row in _VECTORS @ logicals.T % 2]
    weights = _VECTORS.sum(axis=1)
    lightest: dict[tuple[int, ...], int] = {}
    residual: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for index in np.argsort(weights, kind="stable"):
        lightest.setdefault(syndromes[index], int(index))
        residual.setdefault((syndromes[index], flips[index]), int(weights[index]))
    failures = np.array(
        [flips[index] != flips[lightest[syndromes[index]]] for index in range(len(_VECTORS))]
    )
    residual_weights = np.array(
        [residual[syndromes[index], flips[index]] for index in range(len(_VECTORS))]
    )
    return failures.astype(float), residual_weights


def _get_error_probabilities(error_rate: float) -> np.ndarray:
    """The probability of each binary vector when each bit is set independently."""
    weights = _VECTORS.sum(axis=1)
    return error_rate**weights * (1 - error_rate) ** (_NUM_QUBITS - weights)
