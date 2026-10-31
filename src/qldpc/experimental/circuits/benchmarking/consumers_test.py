# SPDX-License-Identifier: Apache-2.0

"""Unit tests for consumers.py."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import scipy.stats
import stim

from qldpc import circuits, codes
from qldpc.experimental.circuits.benchmarking.consumers import ConsumerCalibration
from qldpc.experimental.circuits.benchmarking.residual_weights import (
    ResidualWeightHistogram,
    estimate_residual_weight_histogram,
)
from qldpc.objects import Pauli

_CODE = codes.SteaneCode()
_NUM_QUBITS = len(_CODE)


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


# every binary vector on the qubits of the Steane code, with bit j for qubit j
_VECTORS = np.array(list(itertools.product([0, 1], repeat=_NUM_QUBITS)), dtype=int)[:, ::-1]


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


def _get_readout_failures() -> tuple[np.ndarray, np.ndarray]:
    """Failures of a readout, and residual weights, of each X-type error on logical |0>.

    Errors are indexed by integers, with bit j for qubit j.  A lookup table decodes a noiseless
    readout of the Z-type stabilizers and logical Z operator.  The readout fails if a lightest error
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
