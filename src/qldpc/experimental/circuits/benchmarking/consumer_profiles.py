# SPDX-License-Identifier: Apache-2.0

"""Consumer profiles: how the circuits after a state preparation respond to its faults.

A prepared state is handed to a consumer, such as a round of Steane-type error correction, which has
faults of its own.  A ``ConsumerProfile`` gives the probability that the consumer fails, as a
function of the residual weight of the fault of the preparation circuit (see
``estimate_residual_weight_histogram``).  ``ResidualWeightHistogram.get_excess_failure_probability``
averages a profile over the residual weights of a preparation circuit.
``estimate_downstream_fault_weight`` estimates the expected fault weight of a consumer from the
noise of a circuit, for ``ConsumerProfile.poisson``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes
from qldpc.circuits._post_selection import is_integer
from qldpc.circuits.certification.fault_propagation import _get_unmeasured_qubits

# Rounding slack when checking that probabilities sum to at most one.
_TOLERANCE = 1e-9

_TWO_QUBIT_NOISE = frozenset({"DEPOLARIZE2", "PAULI_CHANNEL_2"})
_SINGLE_QUBIT_NOISE = frozenset({"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "PAULI_CHANNEL_1"})
_RESETS = frozenset({"R", "RX", "RY", "MR", "MRX", "MRY"})
_ANNOTATIONS = frozenset(
    {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"}
)

####################################################################################################
# consumer profiles


@dataclasses.dataclass(frozen=True)
class ConsumerProfile:
    """Failure probabilities of a consumer of a prepared state, as a function of residual weight.

    Entry ``m`` of ``failure_probabilities``, written ``c(m)``, is the probability that the consumer
    fails when the fault of the preparation circuit has residual weight ``m`` (see
    ``estimate_residual_weight_histogram``).  Every residual weight beyond the last entry is taken
    to make the consumer fail.  Entry 0 is the failure probability of the consumer when the
    preparation leaves no Pauli error that a stabilizer cannot remove, so ``c(m) - c(0)`` is the
    failure that a residual weight ``m`` causes, which
    ``ResidualWeightHistogram.get_excess_failure_probability`` averages over a preparation circuit.
    This excess is nonnegative if the failure probabilities do not decrease with ``m``, as for every
    profile that the constructors below return; other profiles are accepted as given.

    The constructors ``hard``, ``from_downstream_fault_weights``, and ``poisson`` model a consumer
    that succeeds exactly when ``m + g <= correctable_weight``, where ``g`` is the fault weight of
    its own fault.  This is the guarantee of a fault-tolerant consumer for a code that corrects
    every Pauli error of Pauli weight at most ``correctable_weight``, and it is pessimistic for a
    decoder that also corrects many heavier Pauli errors.  Failure probabilities that were measured
    for a real consumer and decoder can be passed to the class directly instead.

    Attributes:
        failure_probabilities: The failure probabilities ``c(0), c(1), ...``, each in [0, 1].

    Raises:
        ValueError: If there are no failure probabilities, or if one is not in [0, 1].
    """

    failure_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        """Store the failure probabilities as a tuple of floats, and check that they are valid."""
        values = tuple(float(value) for value in self.failure_probabilities)
        if not values:
            raise ValueError("A consumer profile needs at least one failure probability")
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"Failure probabilities must lie in [0, 1], not {values}")
        object.__setattr__(self, "failure_probabilities", values)

    @property
    def max_residual_weight(self) -> int:
        """The largest residual weight with an explicit failure probability."""
        return len(self.failure_probabilities) - 1

    def get_failure_probability(self, weight: int) -> float:
        """The probability that the consumer fails on a state with the given residual weight.

        Args:
            weight: A nonnegative residual weight.

        Returns:
            The failure probability, which is 1 for weights above ``max_residual_weight``.

        Raises:
            TypeError: If the weight is not an integer.
            ValueError: If the weight is negative.
        """
        if not is_integer(weight):
            raise TypeError(f"The residual weight must be an integer, not {weight!r}")
        if weight < 0:
            raise ValueError(f"The residual weight must be nonnegative, not {weight}")
        return self.failure_probabilities[weight] if weight <= self.max_residual_weight else 1.0

    @classmethod
    def hard(cls, correctable_weight: int) -> ConsumerProfile:
        """A consumer with no fault of its own that corrects every residual weight up to a bound.

        Its excess failure probability is the probability that the residual weight exceeds
        ``correctable_weight``.

        Args:
            correctable_weight: The largest residual weight that the consumer corrects, typically
                ``(d - 1) // 2`` for a code of distance ``d``.

        Returns:
            The profile ``c(m) = 0`` for ``m <= correctable_weight``, and ``c(m) = 1`` otherwise.

        Raises:
            TypeError: If ``correctable_weight`` is not an integer.
            ValueError: If ``correctable_weight`` is negative.
        """
        return cls.from_downstream_fault_weights([1.0], correctable_weight)

    @classmethod
    def from_downstream_fault_weights(
        cls, fault_weight_probabilities: npt.ArrayLike, correctable_weight: int
    ) -> ConsumerProfile:
        """A consumer that succeeds if a residual weight m plus its own fault weight g is at most t.

        Here ``t`` is ``correctable_weight``, and the fault weight ``g`` of the consumer's own fault
        has the given distribution, so ``c(m) = Pr[g > t - m]``.  The excess failure probability of
        this profile is then the probability that a fault-tolerant consumer, which would otherwise
        succeed, loses its guarantee because of the residual weight of the preparation.

        Other prepared states that the consumer uses, such as the resource states of Steane-type
        error correction, add their residual weights to ``g``.  To include them, convolve their
        residual-weight distributions (see
        ``ResidualWeightHistogram.get_residual_weight_distribution``) into the fault-weight
        distribution.  The last entry of such a distribution, for residual weights above its
        ``max_residual_weight``, then counts as ``max_residual_weight + 1``, which is exact as long
        as ``max_residual_weight >= correctable_weight``.

        Args:
            fault_weight_probabilities: Entry ``g`` is the probability that the consumer's own fault
                has fault weight ``g``.  These are nonnegative and sum to at most one, and any
                remaining probability is for fault weights above the last entry.  So that the
                profile is determined, they must either sum to one or have at least ``t + 1``
                entries.
            correctable_weight: The largest total weight ``t`` that the consumer corrects, typically
                ``(d - 1) // 2`` for a code of distance ``d``.

        Returns:
            The profile with entries ``c(m) = Pr[g > t - m]`` for ``m = 0, ..., t``.

        Raises:
            TypeError: If ``correctable_weight`` is not an integer.
            ValueError: If ``correctable_weight`` is negative, if the fault-weight probabilities are
                not a nonempty one-dimensional array of nonnegative numbers that sum to at most one,
                or if they neither sum to one nor have at least ``t + 1`` entries.
        """
        max_total_weight = _check_correctable_weight(correctable_weight)
        probabilities = np.asarray(fault_weight_probabilities, dtype=float)
        if (
            probabilities.ndim != 1
            or not probabilities.size
            or not np.all(probabilities >= 0)
            or probabilities.sum() > 1 + _TOLERANCE
        ):
            raise ValueError(
                "Fault-weight probabilities must be a nonempty one-dimensional array of nonnegative"
                f" numbers that sum to at most one, not {probabilities}"
            )
        if len(probabilities) <= max_total_weight and probabilities.sum() < 1 - _TOLERANCE:
            raise ValueError(
                "Fault-weight probabilities that do not sum to one must cover every fault weight up"
                f" to the correctable weight {max_total_weight}, not only {len(probabilities) - 1}"
            )
        # entry j is Pr[g <= j]
        cumulative = np.cumsum(probabilities)
        return cls(
            tuple(
                float(
                    np.clip(
                        1 - cumulative[min(max_total_weight - weight, len(cumulative) - 1)], 0, 1
                    )
                )
                for weight in range(max_total_weight + 1)
            )
        )

    @classmethod
    def poisson(cls, expected_fault_weight: float, correctable_weight: int) -> ConsumerProfile:
        """A consumer whose fault weight has a Poisson distribution.

        The fault weight of a consumer with many independent rare error mechanisms, whose
        probabilities sum to ``expected_fault_weight``, has approximately a Poisson distribution
        (see ``estimate_downstream_fault_weight``).  This approximation comes with no error bound,
        and it can be poor for a consumer with a few error mechanisms of large probability.

        Args:
            expected_fault_weight: The expected fault weight of the consumer's own fault, a
                nonnegative number.
            correctable_weight: The largest total weight that the consumer corrects, typically
                ``(d - 1) // 2`` for a code of distance ``d``.

        Returns:
            The profile of ``from_downstream_fault_weights`` for a Poisson-distributed fault weight.

        Raises:
            TypeError: If ``correctable_weight`` is not an integer.
            ValueError: If ``expected_fault_weight`` is negative or not finite, or if
                ``correctable_weight`` is negative.
        """
        max_total_weight = _check_correctable_weight(correctable_weight)
        if not 0 <= expected_fault_weight < math.inf:
            raise ValueError(
                f"The expected fault weight must be nonnegative and finite, not"
                f" {expected_fault_weight}"
            )
        # Poisson probabilities by the recurrence Pr[g + 1] = Pr[g] * nu / (g + 1), which cannot
        # overflow
        probabilities = [math.exp(-expected_fault_weight)]
        for weight in range(max_total_weight):
            probabilities.append(probabilities[-1] * expected_fault_weight / (weight + 1))
        return cls.from_downstream_fault_weights(probabilities, correctable_weight)


def _check_correctable_weight(correctable_weight: int) -> int:
    """Return a correctable weight, which must be a nonnegative integer."""
    if not is_integer(correctable_weight):
        raise TypeError(f"The correctable weight must be an integer, not {correctable_weight!r}")
    if correctable_weight < 0:
        raise ValueError(f"The correctable weight must be nonnegative, not {correctable_weight}")
    return int(correctable_weight)


####################################################################################################
# downstream fault weights


def estimate_downstream_fault_weight(
    circuit: stim.Circuit,
    preset: Literal["steane", "syndrome"] = "steane",
    *,
    code: codes.QuditCode | None = None,
    num_data_qubits: int | None = None,
    two_qubit_error: float | None = None,
    measurement_error: float | None = None,
    reset_error: float | None = None,
) -> float:
    """Estimate the expected fault weight of a consumer, from the noise of a circuit.

    The estimate multiplies mean error probabilities, read from the noise instructions of a noisy
    circuit (typically the state-preparation circuit itself), by the numbers of operations of a
    reference consumer.  It simulates no circuit.  The presets are:

    - ``"steane"`` (default): one round of Steane-type error correction, in which the data block
      meets a resource state for each type of check through transversal two-qubit gates, and each
      resource state is measured transversally.  That makes ``2 n`` two-qubit gates and ``2 n``
      measurements, where ``n`` is the number of data qubits, so the estimate is
      ``2 n (p_2 + p_m)``.  It does not depend on the checks of the code, which suits codes with
      high-weight checks, such as concatenated codes.  The error mechanisms of the preparations of
      the resource states are not included; ``ConsumerProfile.from_downstream_fault_weights``
      explains how to include their residual weights instead.
    - ``"syndrome"``: one round of syndrome extraction with one bare ancilla per check of ``code``,
      with ``E`` two-qubit gates for checks of total weight ``E``, and with ``m`` resets and ``m``
      measurements for ``m`` checks.  The estimate is ``E p_2 + m (p_m + p_r)``.

    The mean error probabilities ``p_2``, ``p_m``, and ``p_r`` of two-qubit gates, measurements, and
    resets are read from the circuit as follows, unless they are passed explicitly:

    - ``p_2`` is the total probability of the two-qubit noise channels (``DEPOLARIZE2`` and
      ``PAULI_CHANNEL_2``), divided by the number of two-qubit unitary gates.
    - ``p_m`` is the mean flip probability (the argument) of the measurements.
    - ``p_r`` is the mean, over resets, of the probability of the first single-qubit noise channel
      that acts on the reset qubit before any other operation does.

    Other noise, such as idling noise, is not counted.  The expected fault weight of a consumer is
    the sum of the probabilities of its error mechanisms, which ``ConsumerProfile.poisson`` takes.
    This estimate counts each noisy operation of the reference consumer once, with its total error
    probability, rather than building the detector error model of the consumer, so it agrees with
    that sum only to first order in the error probabilities, and only when the components of each
    noise channel are separate error mechanisms.

    Args:
        circuit: A noisy Stim circuit from which to read error probabilities.
        preset: The reference consumer, ``"steane"`` (default) or ``"syndrome"``.

    Keyword args:
        code: The qubit code of the data block.  Required for the ``"syndrome"`` preset.
        num_data_qubits: The number of data qubits for the ``"steane"`` preset.  Default: the length
            of ``code`` if provided, and otherwise the number of qubits of ``circuit`` that some
            instruction acts on and no measurement targets.
        two_qubit_error: The mean error probability of a two-qubit gate, instead of ``p_2``.
        measurement_error: The mean error probability of a measurement, instead of ``p_m``.
        reset_error: The mean error probability of a reset, instead of ``p_r``.

    Returns:
        The expected fault weight of one round of the reference consumer.

    Raises:
        TypeError: If ``num_data_qubits`` is not an integer.
        ValueError: If ``preset`` is unknown, if the ``"syndrome"`` preset lacks a code, if the code
            is not a qubit code, if ``num_data_qubits`` is not positive, if an explicit error
            probability is not in [0, 1], or if the circuit has no operations from which to read a
            required error probability.
    """
    if preset not in ("steane", "syndrome"):
        raise ValueError(f'The preset must be "steane" or "syndrome", not {preset!r}')
    if code is not None and code.field.order != 2:
        raise ValueError(f"The code must be a qubit code, not a code over GF({code.field.order})")
    for name, value in (
        ("two_qubit_error", two_qubit_error),
        ("measurement_error", measurement_error),
        ("reset_error", reset_error),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1], not {value}")
    rates = _get_mean_error_rates(circuit)
    two_qubit_error = _pick_rate(two_qubit_error, rates[0], "two_qubit_error", "two-qubit gates")
    measurement_error = _pick_rate(measurement_error, rates[1], "measurement_error", "measurements")

    if preset == "steane":
        if num_data_qubits is None:
            num_data_qubits = (
                len(code) if code is not None else len(_get_unmeasured_qubits(circuit))
            )
        if not is_integer(num_data_qubits):
            raise TypeError(f"num_data_qubits must be an integer, not {num_data_qubits!r}")
        if num_data_qubits < 1:
            raise ValueError(f"num_data_qubits must be positive, not {num_data_qubits}")
        return 2 * int(num_data_qubits) * (two_qubit_error + measurement_error)

    if code is None:
        raise ValueError('The "syndrome" preset requires a code')
    reset_error = _pick_rate(reset_error, rates[2], "reset_error", "resets")
    checks = np.asarray(code.matrix.view(np.ndarray)).reshape(-1, 2, len(code))
    total_check_weight = int(np.count_nonzero(np.any(checks, axis=1)))
    num_checks = len(checks)
    return total_check_weight * two_qubit_error + num_checks * (measurement_error + reset_error)


def _get_mean_error_rates(circuit: stim.Circuit) -> tuple[float | None, float | None, float | None]:
    """Mean error probabilities of two-qubit gates, measurements, and resets in a noisy circuit.

    See ``estimate_downstream_fault_weight`` for how each is read.  An entry is None if the circuit
    has no operation of that kind.
    """
    num_gates = num_measurements = num_resets = 0
    gate_noise = measurement_noise = reset_noise = 0.0
    # reset qubits that no noise channel or other operation has acted on since their reset
    fresh_resets: set[int] = set()
    for instruction in circuit.flattened():
        name = instruction.name
        if name in _ANNOTATIONS:
            continue
        args = instruction.gate_args_copy()
        qubits = [
            target.qubit_value
            for target in instruction.targets_copy()
            if target.qubit_value is not None
        ]
        if name in _SINGLE_QUBIT_NOISE:
            for qubit in qubits:
                if qubit in fresh_resets:
                    reset_noise += sum(args)
                    fresh_resets.discard(qubit)
            continue
        fresh_resets.difference_update(qubits)
        if name in _TWO_QUBIT_NOISE:
            gate_noise += sum(args) * (len(qubits) // 2)
            continue
        gate_data = stim.gate_data(name)
        if gate_data.is_two_qubit_gate and gate_data.is_unitary:
            num_gates += len(qubits) // 2
        if gate_data.produces_measurements and not name.startswith("HERALDED_"):
            num_measurements += instruction.num_measurements
            measurement_noise += (args[0] if args else 0.0) * instruction.num_measurements
        if name in _RESETS:
            num_resets += len(qubits)
            fresh_resets.update(qubits)
    return (
        gate_noise / num_gates if num_gates else None,
        measurement_noise / num_measurements if num_measurements else None,
        reset_noise / num_resets if num_resets else None,
    )


def _pick_rate(explicit: float | None, inferred: float | None, name: str, kind: str) -> float:
    """Return an explicit error probability if provided, or else the one read from a circuit."""
    if explicit is not None:
        return float(explicit)
    if inferred is None:
        raise ValueError(
            f"The circuit has no {kind} from which to read an error probability; pass {name}"
        )
    return inferred
