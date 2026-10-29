# SPDX-License-Identifier: Apache-2.0

"""Unit tests for consumer_profiles.py."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
import stim

from qldpc import circuits, codes
from qldpc.experimental.circuits.benchmarking import consumer_profiles
from qldpc.experimental.circuits.benchmarking.consumer_profiles import (
    ConsumerProfile,
    estimate_downstream_fault_weight,
)

# Resets of six qubits (R 0 1 2 3, RX 4, MR 2), three two-qubit gates, and five measurements (M 0 1,
# MX 4, MR 2, MPP X0*X1).  Qubit 3 is never measured.
_CIRCUIT = stim.Circuit(
    "R 0 1 2 3\nRX 4\nH 0\nCX 0 1 2 3\nTICK\nCZ 1 2\nM 0 1\nMX 4\nMR 2\nMPP X0*X1"
)


def test_consumer_profile() -> None:
    """Profiles store their failure probabilities, and fail beyond them."""
    profile = ConsumerProfile([0, 0.25, 1])  # type: ignore[arg-type]
    assert profile.failure_probabilities == (0.0, 0.25, 1.0)
    assert profile.max_residual_weight == 2
    assert [profile.get_failure_probability(weight) for weight in range(4)] == [0, 0.25, 1, 1]
    assert ConsumerProfile.hard(2).failure_probabilities == (0, 0, 0)

    with pytest.raises(ValueError, match="at least one failure probability"):
        ConsumerProfile(())
    for invalid in ((0.5, 1.5), (np.nan,)):
        with pytest.raises(ValueError, match="must lie in"):
            ConsumerProfile(invalid)
    with pytest.raises(TypeError, match="must be an integer"):
        profile.get_failure_probability(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be nonnegative"):
        profile.get_failure_probability(-1)


def test_downstream_fault_profiles() -> None:
    """Consumers fail when the residual weight plus their own fault weight exceeds a budget."""
    # with Pr[g = 0, 1, 2] = 0.5, 0.3, 0.1 and Pr[g > 2] = 0.1, the guarantee m + g <= 2 fails with
    # probabilities 0.1, 0.2, and 0.5 for residual weights m = 0, 1, and 2
    profile = ConsumerProfile.from_downstream_fault_weights([0.5, 0.3, 0.1], 2)
    assert np.allclose(profile.failure_probabilities, [0.1, 0.2, 0.5])
    # a distribution that sums to one needs no entries beyond its largest fault weight
    profile = ConsumerProfile.from_downstream_fault_weights([0.5, 0.5], 2)
    assert np.allclose(profile.failure_probabilities, [0.0, 0.0, 0.5])

    expected_fault_weight, correctable_weight = 0.4, 3
    profile = ConsumerProfile.poisson(expected_fault_weight, correctable_weight)
    expected = [
        scipy.stats.poisson.sf(correctable_weight - weight, expected_fault_weight)
        for weight in range(correctable_weight + 1)
    ]
    assert np.allclose(profile.failure_probabilities, expected)

    # a partner state adds its residual weight to g, counting its overflow beyond weight 1 as 2
    partner = [0.9, 0.08, 0.02]
    combined = ConsumerProfile.from_downstream_fault_weights(np.convolve([0.7, 0.3], partner), 1)
    expected = [1 - 0.7 * 0.9 - 0.3 * 0.9 - 0.7 * 0.08, 1 - 0.7 * 0.9]
    assert np.allclose(combined.failure_probabilities, expected)

    # [0.5] leaves undetermined whether the remaining 0.5 has fault weight 1 or more
    for invalid in ([], [[0.5]], [-0.1], [0.6, 0.6], [np.nan], [0.5]):
        with pytest.raises(ValueError, match="Fault-weight probabilities"):
            ConsumerProfile.from_downstream_fault_weights(invalid, 1)
    with pytest.raises(TypeError, match="correctable weight must be an integer"):
        ConsumerProfile.hard(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="correctable weight must be nonnegative"):
        ConsumerProfile.hard(-1)
    for invalid_mean in (-0.1, np.inf):
        with pytest.raises(ValueError, match="nonnegative and finite"):
            ConsumerProfile.poisson(invalid_mean, 1)


def test_downstream_fault_weight() -> None:
    """Expected fault weights count the operations of a consumer at a circuit's error rates."""
    code = codes.SteaneCode()
    error_rate = 1e-3
    noisy = circuits.DepolarizingNoiseModel(error_rate).noisy_circuit(_CIRCUIT)
    assert estimate_downstream_fault_weight(noisy) == pytest.approx(2 * 1 * 2 * error_rate)
    assert estimate_downstream_fault_weight(noisy, code=code) == pytest.approx(
        2 * 7 * 2 * error_rate
    )
    assert estimate_downstream_fault_weight(noisy, num_data_qubits=5) == pytest.approx(
        20 * error_rate
    )

    # SI1000 noise has two-qubit gates at p, measurements at 5 p, and resets at 2 p, and the Steane
    # code has six checks of total weight 24
    noisy = circuits.SI1000NoiseModel(error_rate).noisy_circuit(_CIRCUIT)
    steane = estimate_downstream_fault_weight(noisy, code=code)
    assert steane == pytest.approx(2 * 7 * 6 * error_rate)
    syndrome = estimate_downstream_fault_weight(noisy, "syndrome", code=code)
    assert syndrome == pytest.approx((24 + 6 * (5 + 2)) * error_rate)
    explicit = estimate_downstream_fault_weight(
        noisy,
        "syndrome",
        code=code,
        two_qubit_error=0.0,
        measurement_error=0.0,
        reset_error=error_rate,
    )
    assert explicit == pytest.approx(6 * error_rate)


def test_downstream_fault_weight_errors() -> None:
    """Unknown presets, invalid arguments, and missing error rates raise errors."""
    noisy = circuits.DepolarizingNoiseModel(1e-3).noisy_circuit(_CIRCUIT)
    with pytest.raises(ValueError, match="preset must be"):
        estimate_downstream_fault_weight(noisy, "memory")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two_qubit_error must lie in"):
        estimate_downstream_fault_weight(noisy, two_qubit_error=2.0)
    with pytest.raises(ValueError, match="requires a code"):
        estimate_downstream_fault_weight(noisy, "syndrome")
    with pytest.raises(ValueError, match="must be a qubit code"):
        estimate_downstream_fault_weight(noisy, "syndrome", code=codes.SurfaceCode(2, field=3))
    with pytest.raises(TypeError, match="num_data_qubits must be an integer"):
        estimate_downstream_fault_weight(noisy, num_data_qubits=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_data_qubits must be positive"):
        estimate_downstream_fault_weight(noisy, num_data_qubits=0)
    with pytest.raises(ValueError, match="no two-qubit gates"):
        estimate_downstream_fault_weight(stim.Circuit("M(0.001) 0"))
    with pytest.raises(ValueError, match="no resets"):
        estimate_downstream_fault_weight(
            stim.Circuit("CX 0 1\nM 0"), "syndrome", code=codes.SteaneCode()
        )


def test_reset_error_reading() -> None:
    """Reset noise is the first single-qubit noise channel on a reset qubit before any operation."""
    probability = 1e-3
    for operation, expected in (("TICK", probability), ("H 0", 0.0)):
        circuit = stim.Circuit(f"R 0\n{operation}\nX_ERROR({probability}) 0")
        assert consumer_profiles._get_mean_error_rates(circuit)[2] == pytest.approx(expected)
