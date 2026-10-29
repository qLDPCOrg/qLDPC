# SPDX-License-Identifier: Apache-2.0

"""Residual-weight histograms of noisy state-preparation circuits.

``qldpc.circuits.certify_bounded_fault_propagation`` checks that every fault of a noisy Clifford
state-preparation circuit, up to a chosen fault weight, has residual weight at most its fault
weight.  Circuits that pass this check can still differ in how likely their faults are to have each
residual weight.  ``estimate_residual_weight_histogram`` estimates, over the shots of a circuit that
pass post-selection, the probability distribution of the residual weight of the fault that occurs in
a shot, and returns a ``ResidualWeightHistogram``.  The residual weight needs no decoder, and it
accounts for Pauli errors that a chosen logical readout cannot detect, such as Z errors on a logical
``|0>`` state, which later circuits must still correct.

``ResidualWeightHistogram.get_excess_failure_probability`` turns a histogram into a single figure of
merit for a ``ConsumerProfile``, a model of the circuits that consume the prepared state.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Sequence
from typing import Literal, Protocol

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import circuits
from qldpc import math as qldpc_math
from qldpc.circuits._post_selection import get_flag_detectors, get_selected_detectors, is_integer
from qldpc.circuits.bookkeeping import DetectorRecord
from qldpc.circuits.certification.fault_propagation import (
    _get_primitive_fault_signatures,
    _get_unmeasured_qubits,
    _is_css_state,
    _validate_output_qubits,
)
from qldpc.codes.distance._bounded_weight_search import PauliCosetSearch

from .consumer_profiles import ConsumerProfile

# Sampled faults are processed in chunks of this many, to bound memory use.
_CHUNK_SIZE = 10_000
# The proposal distribution multiplies the odds of error mechanisms that fire a post-selection
# detector by this factor, so that sampled faults pass post-selection more often.
_FLAGGED_ODDS_FACTOR = 0.05
# By default, faults are neglected beyond the smallest fault weight at which the probability of all
# heavier faults is below this fraction of the probability of fault weight max_residual_weight + 1,
# the smallest fault weight at which a circuit with bounded fault propagation can overflow.
_TRUNCATION_TOLERANCE = 1e-9
# The default maximum fault weight is at most this much more than max_residual_weight.
_MAX_EXTRA_FAULT_WEIGHT = 40
# The CSS tables pack a syndrome sector into one 64-bit machine word.
_MAX_TABLE_QUBITS = 64

####################################################################################################
# residual-weight histograms


@dataclasses.dataclass(frozen=True, eq=False)
class ResidualWeightHistogram:
    """Estimated residual weights of the faults of a noisy state-preparation circuit.

    An instance is built by ``estimate_residual_weight_histogram``, whose docstring defines error
    mechanisms, faults, fault weights, and residual weights.  The fault of a shot is the set of
    error mechanisms that occur in it, and each shot has one outcome: post-selection discards it, or
    accepts it with the residual weight ``m`` of its fault.  The histogram resolves ``m = 0, ...,
    max_residual_weight`` and lumps larger residual weights into one overflow outcome.

    The raw data are sorted by fault weight ``f``, for ``f = 0, ..., max_fault_weight``.  The
    probability of each fault weight is known exactly.  Row ``f`` of ``weight_sums`` holds, for each
    outcome, the summed importance weights of the sampled faults of fault weight ``f`` with that
    outcome, so that dividing by the row sum estimates the probability of each outcome given the
    fault weight.  A row that was computed exactly holds these probabilities, and has no samples.
    Faults of fault weight above ``max_fault_weight`` are never sampled, and count as accepted with
    overflow, as do the faults of any fault weight with neither samples nor an exact row.  These
    conventions can only overstate residual weights.

    The columns of ``weight_sums`` and ``squared_weight_sums`` are, in order: discarded, residual
    weight ``0, 1, ..., max_residual_weight``, and overflow.

    Attributes:
        max_residual_weight: The largest residual weight that the histogram resolves.
        css_weight: Whether the Pauli weight is the CSS weight (True) or the physical weight
            (False).
        fault_weight_probabilities: Entry ``f`` is the probability that exactly ``f`` error
            mechanisms occur, for ``f = 0, ..., max_fault_weight``.
        truncated_probability: The probability that more than ``max_fault_weight`` error mechanisms
            occur.
        weight_sums: Summed importance weights, with one row per fault weight and one column per
            outcome, as above.
        squared_weight_sums: Summed squares of the importance weights, in the layout of
            ``weight_sums``.  They are zero in exact rows.
        num_samples: Entry ``f`` is the number of faults of fault weight ``f`` that were sampled.
        exact: Entry ``f`` is True if row ``f`` was computed exactly, by listing every fault of
            fault weight ``f``.

    Raises:
        ValueError: If the raw data have inconsistent shapes, or are negative or not finite.
    """

    max_residual_weight: int
    css_weight: bool
    fault_weight_probabilities: npt.NDArray[np.float64]
    truncated_probability: float
    weight_sums: npt.NDArray[np.float64]
    squared_weight_sums: npt.NDArray[np.float64]
    num_samples: npt.NDArray[np.int_]
    exact: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Check that the raw data have consistent shapes and valid values."""
        num_rows = len(self.fault_weight_probabilities)
        shape = (num_rows, self.max_residual_weight + 3)
        if (
            self.max_residual_weight < 0
            or num_rows < 1
            or self.weight_sums.shape != shape
            or self.squared_weight_sums.shape != shape
            or self.num_samples.shape != (num_rows,)
            or self.exact.shape != (num_rows,)
        ):
            raise ValueError("The raw data of a residual-weight histogram have inconsistent shapes")
        arrays = (self.fault_weight_probabilities, self.weight_sums, self.squared_weight_sums)
        if not all(np.all(np.isfinite(array)) for array in arrays) or not math.isfinite(
            self.truncated_probability
        ):
            raise ValueError("The raw data of a residual-weight histogram must be finite")
        if (
            np.any(self.fault_weight_probabilities < 0)
            or not 0 <= self.truncated_probability <= 1
            or np.any(self.weight_sums < 0)
            or np.any(self.squared_weight_sums < 0)
            or np.any(self.num_samples < 0)
        ):
            raise ValueError("The raw data of a residual-weight histogram must be nonnegative")

    @property
    def max_fault_weight(self) -> int:
        """The largest fault weight that the histogram covers."""
        return len(self.fault_weight_probabilities) - 1

    @property
    def acceptance_probability(self) -> float:
        """The estimated probability that a shot passes post-selection.

        Shots with more than ``max_fault_weight`` error mechanisms, or with an unsampled fault
        weight, count as accepted, so this can overstate the probability.
        """
        fractions = self._get_outcome_fractions()
        return float(self.fault_weight_probabilities @ fractions[:, 1:].sum(axis=1)) + float(
            self.truncated_probability
        )

    @property
    def discard_probability(self) -> float:
        """The estimated probability that post-selection discards a shot.

        This is one minus ``acceptance_probability``, so it can understate the probability.
        """
        return float(self.fault_weight_probabilities @ self._get_outcome_fractions()[:, 0])

    @property
    def truncation_error_bound(self) -> float:
        """The most by which neglected faults can change a conditional probability.

        Faults of fault weight above ``max_fault_weight`` count as accepted with overflow.  This
        bounds the change, in any probability given acceptance that the histogram reports, if they
        had any other outcome.
        """
        return self.truncated_probability / self.acceptance_probability

    def get_residual_weight_distribution(self) -> npt.NDArray[np.float64]:
        """The distribution of the residual weight of a shot that passes post-selection.

        Returns:
            An array with ``max_residual_weight + 2`` entries.  Entry ``m <= max_residual_weight``
            is the probability that an accepted shot has residual weight ``m``, and the last entry
            is the probability that it has a residual weight above ``max_residual_weight``.
        """
        fractions = self._get_outcome_fractions()
        masses = self.fault_weight_probabilities @ fractions[:, 1:]
        masses[-1] += self.truncated_probability
        return masses / self.acceptance_probability

    def get_tail_probabilities(self) -> npt.NDArray[np.float64]:
        """The probabilities that an accepted shot has a residual weight above each threshold.

        Returns:
            An array with ``max_residual_weight + 1`` entries, whose entry ``r`` is the probability
            that a shot that passes post-selection has residual weight above ``r``.  Entry ``(d - 1)
            // 2`` is the probability that the state alone exceeds the weight that a code of
            distance ``d`` is guaranteed to correct.
        """
        distribution = self.get_residual_weight_distribution()
        return np.cumsum(distribution[::-1])[::-1][1:]

    def get_excess_failure_probability(self, profile: ConsumerProfile) -> tuple[float, float]:
        """Estimate the failure that the faults of the circuit cause in a consumer of its state.

        For a consumer profile ``c``, this is the expectation of ``c(m) - c(0)`` over the residual
        weight ``m`` of a shot that passes post-selection: the probability that the consumer fails
        with this state, less its probability of failing with a perfect state.  It is a difference
        of probabilities, which is nonnegative if the failure probabilities of the profile do not
        decrease with ``m``, as for every profile that the constructors of ``ConsumerProfile``
        return.  For ``ConsumerProfile.hard(r)`` it is the tail probability of ``r`` (see
        ``get_tail_probabilities``).

        The standard error propagates the sampling variance of the importance-weighted estimate of
        each fault weight to first order.  It is zero for exact rows, and it understates the
        uncertainty of a sampled fault weight in which few or no samples have the outcomes that
        matter, so check ``num_samples`` when the standard error looks too good to be true.

        Args:
            profile: A model of the consumer of the prepared state.

        Returns:
            The excess failure probability and its standard error.

        Raises:
            ValueError: If the profile has explicit failure probabilities for residual weights above
                ``max_residual_weight``, which the histogram does not resolve.
        """
        if profile.max_residual_weight > self.max_residual_weight:
            raise ValueError(
                "The consumer profile resolves residual weights up to"
                f" {profile.max_residual_weight}, but the histogram only up to"
                f" {self.max_residual_weight}.  Estimate a histogram with"
                f" max_residual_weight >= {profile.max_residual_weight}."
            )
        failures = [
            profile.get_failure_probability(weight)
            for weight in range(self.max_residual_weight + 2)
        ]
        excess = np.array([0.0, *failures]) - failures[0]
        excess[0] = 0.0
        return self._get_conditional_mean(excess)

    def _get_outcome_fractions(self) -> npt.NDArray[np.float64]:
        """The estimated probability of each outcome, given each fault weight.

        A fault weight with neither samples nor an exact row counts as accepted overflow.
        """
        totals = self.weight_sums.sum(axis=1)
        fractions = np.zeros_like(self.weight_sums)
        known = totals > 0
        fractions[known] = self.weight_sums[known] / totals[known, None]
        fractions[~known, -1] = 1.0
        return fractions

    def _get_conditional_mean(self, values: npt.NDArray[np.float64]) -> tuple[float, float]:
        """Estimate the mean of a function of the outcome over accepted shots, with an error bar.

        Args:
            values: The value of each outcome, in the column order of ``weight_sums``.  The value of
                the discarded outcome is ignored.

        Returns:
            The estimated mean and its first-order standard error.
        """
        values = values.copy()
        values[0] = 0.0
        accepted = np.ones(len(values))
        accepted[0] = 0.0
        fractions = self._get_outcome_fractions()
        probabilities = self.fault_weight_probabilities
        numerators = fractions @ values
        acceptances = fractions @ accepted
        numerator = probabilities @ numerators + self.truncated_probability * values[-1]
        denominator = probabilities @ acceptances + self.truncated_probability
        mean = numerator / denominator

        # self-normalized importance sampling: a sample with outcome k deviates from the ratio
        # estimate of row f by (values[k] - numerators[f]) - mean (accepted[k] - acceptances[f])
        deviations = (values[None, :] - numerators[:, None]) - mean * (
            accepted[None, :] - acceptances[:, None]
        )
        totals = self.weight_sums.sum(axis=1)
        sampled = (totals > 0) & ~self.exact
        row_variances = np.zeros(len(totals))
        row_variances[sampled] = (self.squared_weight_sums[sampled] * deviations[sampled] ** 2).sum(
            axis=1
        ) / totals[sampled] ** 2
        error = math.sqrt(float(probabilities**2 @ row_variances)) / float(denominator)
        return float(mean), error


####################################################################################################
# estimation


def estimate_residual_weight_histogram(
    circuit: stim.Circuit,
    max_residual_weight: int,
    num_samples: int = 100_000,
    *,
    output_qubits: Sequence[int] | None = None,
    css_weight: bool | None = None,
    add_flags: bool = False,
    post_select: bool | Collection[int] | Literal["flags"] = False,
    max_fault_weight: int | None = None,
    max_exact_faults: int = 10**6,
    seed: int | None = None,
) -> ResidualWeightHistogram:
    """Estimate the residual weights of the faults of a noisy Clifford state-preparation circuit.

    This function uses the terms of ``qldpc.circuits.certify_bounded_fault_propagation``.  A fault
    is a set of error mechanisms, which are the error instructions of the detector error model of
    the circuit after adding any flag detectors, and its fault weight is its number of error
    mechanisms.  A fault leaves a Pauli error, its output error, on the output qubits that hold the
    prepared state.  The residual weight of the fault is the smallest Pauli weight of its output
    error after multiplying it by any stabilizer of the prepared state, where the Pauli weight of a
    Pauli error is either its CSS weight ``max(|x|, |z|)`` or its physical weight.

    In each shot of the circuit, error mechanisms occur independently with their own probabilities,
    and the error mechanisms that occur form the fault of the shot.  A shot is discarded if its
    fault fires a post-selected detector.  As for ``certify_bounded_fault_propagation``, error
    instructions with probability zero are not error mechanisms.  Stim converts some noise channels,
    such as ``ELSE_CORRELATED_ERROR`` chains, into independent error mechanisms only approximately,
    and the results that are called exact below are exact for these error mechanisms.

    The estimate is stratified by fault weight.  The probability of each fault weight is computed
    exactly.  Fault weights with few enough faults are listed exactly; the others are sampled from
    proposal distributions that make more error mechanisms occur and that suppress the error
    mechanisms that fire a post-selected detector, and importance weights correct for both.  Given
    its fault weight, a fault has the same distribution under every proposal with the same
    suppression, so all samples of a fault weight are pooled.

    The residual weight of each fault is computed exactly from the syndrome of its output error.
    For the CSS weight of a CSS state on at most 64 qubits, two tables of the syndromes of light
    X-type and Z-type Pauli errors give it by a meet-in-the-middle lookup.  Otherwise, the slower
    exact search of ``certify_bounded_fault_propagation`` gives it.

    Args:
        circuit: A noisy Stim Clifford circuit starting in Stim's all-zero state.
        max_residual_weight: The largest residual weight to resolve, a nonnegative integer; heavier
            residual weights are lumped together.  Use at least ``(d - 1) // 2`` for a code of
            distance ``d``, and at least the ``max_residual_weight`` of any consumer profile to
            score.
        num_samples: The number of faults to sample, spread evenly over proposal distributions for
            the fault weights that are not listed exactly.  Default: 100_000.

    Keyword args:
        output_qubits: Distinct indices of the output qubits, at least one.  Default: all qubits
            that the circuit acts on and never measures.
        css_weight: True to use the CSS weight as the Pauli weight, or False to use the physical
            weight.  Default: the CSS weight if the prepared state is a CSS state in the physical
            Pauli basis, and the physical weight otherwise.
        add_flags: Whether to add a flag detector for each measurement that no detector or
            observable addresses, as in ``qldpc.circuits.get_state_prep_diagnostic_circuit``.
            Default: False.
        post_select: The detectors to post-select on, among the circuit's detectors followed by any
            added flag detectors: True for all of them, False (the default) for none, a collection
            of indices (negative indices count back from the last detector), or "flags" for exactly
            the added flag detectors, which requires ``add_flags=True``.
        max_fault_weight: The largest fault weight to cover; the ``truncated_probability`` of the
            histogram is the probability of larger fault weights.  Default: the smallest fault
            weight from ``max_residual_weight + 1`` on beyond which the remaining probability is
            below a billionth of the probability of fault weight ``max_residual_weight + 1``,
            searching no further than ``max_residual_weight + 41``, and in any case at most the
            number of error mechanisms.
        max_exact_faults: Fault weights with at most this many faults (and every smaller fault
            weight) are computed exactly rather than sampled.  Default: ``10**6``.
        seed: A seed for the random number generator.  Default: None, for fresh randomness.

    Returns:
        A ``ResidualWeightHistogram``.

    Raises:
        TypeError: If ``max_residual_weight``, ``num_samples``, ``max_fault_weight``, or
            ``max_exact_faults`` is not an integer, if ``css_weight`` is not a bool or None, or if
            ``output_qubits`` or ``post_select`` has an invalid type.
        ValueError: If an integer argument is out of range, if ``output_qubits`` or ``post_select``
            is invalid, if the output qubits do not end in a definite stabilizer state, if the
            circuit has an error instruction with probability one, or if Stim cannot convert the
            circuit to a detector error model.
    """
    max_residual_weight = _check_integer(max_residual_weight, "max_residual_weight", 0)
    num_samples = _check_integer(num_samples, "num_samples", 0)
    max_exact_faults = _check_integer(max_exact_faults, "max_exact_faults", 0)
    if max_fault_weight is not None:
        max_fault_weight = _check_integer(max_fault_weight, "max_fault_weight", 0)
    if css_weight is not None and not isinstance(css_weight, bool):
        raise TypeError("css_weight must be a bool or None")
    if output_qubits is None:
        output_qubits = _get_unmeasured_qubits(circuit)
        if not output_qubits:
            raise ValueError("The circuit has no unmeasured qubits to use as output_qubits")
    qubits = _validate_output_qubits(output_qubits, circuit.num_qubits)

    # flag detectors come after the detectors that the circuit already has
    detector_record = DetectorRecord({"prep": range(circuit.num_detectors)})
    flag_detectors = get_flag_detectors(circuit) if add_flags else stim.Circuit()
    detector_record.append({"flags": range(flag_detectors.num_detectors)})
    selected = get_selected_detectors(post_select, detector_record, add_flags)
    full_circuit = circuit + flag_detectors

    stabilizers = circuits.get_state_stabilizers(full_circuit, qubits)
    if len(stabilizers) != len(qubits):
        raise ValueError("The output qubits must end in a definite pure stabilizer state")
    matrix = np.array([qldpc_math.string_to_op(op) for op in stabilizers], dtype=np.uint8)
    is_css = _is_css_state(matrix)
    css_weight = is_css if css_weight is None else css_weight

    oracle: _ResidualWeightOracle
    if css_weight and is_css and len(qubits) <= _MAX_TABLE_QUBITS:
        # syndromes are taken with respect to pure-type generators, X-type ones first
        x_supports, z_supports = _get_css_supports(matrix)
        oracle = _CSSWeightTables(x_supports, z_supports, max_residual_weight)
        stabilizers = [
            *(stim.PauliString("".join("X" if bit else "_" for bit in row)) for row in x_supports),
            *(stim.PauliString("".join("Z" if bit else "_" for bit in row)) for row in z_supports),
        ]
    else:
        oracle = _CosetSearchWeights(matrix, css_weight, max_residual_weight)

    dem, flags, syndromes = _get_primitive_fault_signatures(
        full_circuit, stabilizers, qubits, selected
    )
    probabilities = [
        instruction.args_copy()[0] for instruction in dem if instruction.type == "error"
    ]
    if any(probability >= 1 for probability in probabilities):
        raise ValueError("The circuit has an error instruction with probability one")
    mechanisms = _Mechanisms.from_signatures(probabilities, flags, syndromes, len(selected), oracle)

    probabilities_by_fault_weight, truncated_probability = _get_fault_weight_probabilities(
        mechanisms.probabilities, max_residual_weight, max_fault_weight
    )
    max_fault_weight = len(probabilities_by_fault_weight) - 1
    num_columns = max_residual_weight + 3
    weight_sums = np.zeros((max_fault_weight + 1, num_columns))
    squared_weight_sums = np.zeros_like(weight_sums)
    sample_counts = np.zeros(max_fault_weight + 1, dtype=int)
    exact = np.zeros(max_fault_weight + 1, dtype=bool)

    # no error mechanism occurs in fault weight zero, so the shot is accepted with no error
    weight_sums[0, 1] = 1.0
    exact[0] = True
    cache: dict[bytes, int] = {}
    max_exact_fault_weight = 0
    for fault_weight in range(1, max_fault_weight + 1):
        num_faults = math.comb(len(mechanisms), fault_weight)
        if num_faults > max_exact_faults:
            break
        # a fault weight above the number of error mechanisms has probability zero
        if num_faults:
            weight_sums[fault_weight] = _list_faults(mechanisms, fault_weight, oracle, cache)
        exact[fault_weight] = True
        max_exact_fault_weight = fault_weight

    if max_exact_fault_weight < max_fault_weight:
        rng = np.random.default_rng(seed)
        targets = range(max_exact_fault_weight + 1, max_fault_weight + 1)
        for index, fault_weight in enumerate(targets):
            shots = num_samples // len(targets) + (index < num_samples % len(targets))
            _sample_faults(
                mechanisms,
                fault_weight,
                shots,
                (max_exact_fault_weight + 1, max_fault_weight),
                oracle,
                cache,
                rng,
                (weight_sums, squared_weight_sums, sample_counts),
            )

    return ResidualWeightHistogram(
        max_residual_weight=max_residual_weight,
        css_weight=css_weight,
        fault_weight_probabilities=probabilities_by_fault_weight,
        truncated_probability=truncated_probability,
        weight_sums=weight_sums,
        squared_weight_sums=squared_weight_sums,
        num_samples=sample_counts,
        exact=exact,
    )


@dataclasses.dataclass(frozen=True)
class _Mechanisms:
    """Independent error mechanisms, with their effects packed into 64-bit machine words.

    Attributes:
        probabilities: The probability of each error mechanism.
        flag_words: Row ``j`` holds the post-selected detectors that mechanism ``j`` fires.
        key_words: Row ``j`` holds the syndrome of the output error of mechanism ``j``, in the
            packing of the residual-weight oracle.
    """

    probabilities: npt.NDArray[np.float64]
    flag_words: npt.NDArray[np.uint64]
    key_words: npt.NDArray[np.uint64]

    def __len__(self) -> int:
        """The number of error mechanisms."""
        return len(self.probabilities)

    @property
    def odds(self) -> npt.NDArray[np.float64]:
        """The odds ``p / (1 - p)`` of each error mechanism."""
        return self.probabilities / (1 - self.probabilities)

    @property
    def flagged(self) -> npt.NDArray[np.bool_]:
        """Whether each error mechanism fires a post-selected detector."""
        return np.any(self.flag_words, axis=1)

    @staticmethod
    def from_signatures(
        probabilities: Sequence[float],
        flags: Sequence[int],
        syndromes: Sequence[int],
        num_flags: int,
        oracle: _ResidualWeightOracle,
    ) -> _Mechanisms:
        """Pack the error mechanisms of a detector error model, with one for each error instruction.

        Error instructions with probability zero are not error mechanisms, and are dropped.
        """
        kept = [
            (probability, flag, syndrome)
            for probability, flag, syndrome in zip(probabilities, flags, syndromes)
            if probability > 0
        ]
        num_flag_words = (num_flags + 63) // 64
        return _Mechanisms(
            np.array([probability for probability, _, _ in kept], dtype=float),
            np.array(
                [_to_words(flag, num_flag_words) for _, flag, _ in kept], dtype=np.uint64
            ).reshape(len(kept), num_flag_words),
            np.array(
                [oracle.get_key_words(syndrome) for _, _, syndrome in kept], dtype=np.uint64
            ).reshape(len(kept), oracle.num_key_words),
        )


def _check_integer(value: int, name: str, minimum: int) -> int:
    """Return an integer argument, which must be at least a minimum value."""
    if not is_integer(value):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {value}")
    return int(value)


def _get_combinations(num_items: int, size: int) -> npt.NDArray[np.intp]:
    """All increasing tuples of ``size`` distinct indices below ``num_items``, one per row."""
    combinations = np.zeros((1, 0), dtype=np.intp)
    for _ in range(size):
        last = combinations[:, -1] if combinations.shape[1] else np.full(len(combinations), -1)
        counts = num_items - 1 - last
        rows = np.repeat(np.arange(len(combinations)), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        combinations = np.concatenate(
            [combinations[rows], (last[rows] + 1 + offsets)[:, None]], axis=1
        )
    return combinations


def _get_fault_weight_probabilities(
    probabilities: npt.NDArray[np.float64], max_residual_weight: int, max_fault_weight: int | None
) -> tuple[npt.NDArray[np.float64], float]:
    """The probability of each fault weight up to a maximum, and of any larger fault weight.

    If ``max_fault_weight`` is None, choose it as ``estimate_residual_weight_histogram`` describes.
    """
    num_mechanisms = len(probabilities)
    cap = (
        min(num_mechanisms, max_residual_weight + 1 + _MAX_EXTRA_FAULT_WEIGHT)
        if max_fault_weight is None
        else max_fault_weight
    )
    # the last entry accumulates the probability of more than cap error mechanisms
    distribution = np.zeros(cap + 2)
    distribution[0] = 1.0
    for probability in probabilities:
        distribution[-1] += distribution[-2] * probability
        distribution[1:-1] = (
            distribution[1:-1] * (1 - probability) + distribution[:-2] * probability
        )
        distribution[0] *= 1 - probability
    if max_fault_weight is None:
        # tails[f] is the probability of more than f error mechanisms
        tails = np.cumsum(distribution[::-1])[::-1][1:]
        base = min(max_residual_weight + 1, cap)
        small = np.flatnonzero(tails[base:] <= _TRUNCATION_TOLERANCE * distribution[base])
        max_fault_weight = base + int(small[0]) if len(small) else cap
        return distribution[: max_fault_weight + 1], float(tails[max_fault_weight])
    return distribution[:-1], float(distribution[-1])


def _list_faults(
    mechanisms: _Mechanisms,
    fault_weight: int,
    oracle: _ResidualWeightOracle,
    cache: dict[bytes, int],
) -> npt.NDArray[np.float64]:
    """The exact probability of each outcome, given that a fixed number of error mechanisms occur.

    Given its fault weight, a fault has a probability proportional to the product of the odds of its
    error mechanisms.
    """
    faults = _get_combinations(len(mechanisms), fault_weight)
    log_weights = np.log(mechanisms.odds)[faults].sum(axis=1)
    weights = np.exp(log_weights - log_weights.max())
    outcomes = _get_outcomes(mechanisms, faults, oracle, cache)
    sums = np.bincount(outcomes, weights=weights, minlength=oracle.max_residual_weight + 3)
    return sums / sums.sum()


def _sample_faults(
    mechanisms: _Mechanisms,
    target_fault_weight: int,
    num_shots: int,
    fault_weight_range: tuple[int, int],
    oracle: _ResidualWeightOracle,
    cache: dict[bytes, int],
    rng: np.random.Generator,
    accumulators: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int_]],
) -> None:
    """Sample faults from one proposal distribution, and accumulate their importance weights.

    The proposal multiplies the odds of every error mechanism by a common factor, chosen so that the
    expected fault weight is ``target_fault_weight``, and multiplies the odds of flagged error
    mechanisms by ``_FLAGGED_ODDS_FACTOR`` on top.  The common factor cancels from the distribution
    of the faults of a given fault weight, so the importance weight of a fault is
    ``_FLAGGED_ODDS_FACTOR ** -k`` for its number ``k`` of flagged error mechanisms, up to a factor
    that depends only on the fault weight.  Only samples with a fault weight in
    ``fault_weight_range`` (inclusive) are kept.
    """
    weight_sums, squared_weight_sums, sample_counts = accumulators
    num_columns = oracle.max_residual_weight + 3
    lowest, highest = fault_weight_range
    flagged = mechanisms.flagged
    probabilities = _get_proposal_probabilities(
        mechanisms.odds * np.where(flagged, _FLAGGED_ODDS_FACTOR, 1.0), target_fault_weight
    )
    for start in range(0, num_shots, _CHUNK_SIZE):
        chunk = min(_CHUNK_SIZE, num_shots - start)
        # an exact Bernoulli sample: draw how many shots each mechanism occurs in, then which
        counts = rng.binomial(chunk, probabilities)
        mechanism_indices = np.repeat(np.arange(len(mechanisms)), counts)
        shot_indices = np.concatenate(
            [np.zeros(0, dtype=np.intp)]
            + [rng.choice(chunk, size=count, replace=False) for count in counts if count]
        )
        fault_weights = np.bincount(shot_indices, minlength=chunk)
        num_flagged = np.bincount(shot_indices[flagged[mechanism_indices]], minlength=chunk)
        importance = _FLAGGED_ODDS_FACTOR ** -num_flagged.astype(float)

        # group the occurring mechanisms by shot, and keep the shots of the requested fault weights
        order = np.argsort(shot_indices, kind="stable")
        grouped_mechanisms = mechanism_indices[order]
        keep = (fault_weights >= lowest) & (fault_weights <= highest)
        outcomes = np.zeros(chunk, dtype=np.intp)
        for fault_weight in np.unique(fault_weights[keep]):
            shots = np.flatnonzero(fault_weights == fault_weight)
            starts = np.concatenate([[0], np.cumsum(fault_weights)])[shots]
            members = grouped_mechanisms[starts[:, None] + np.arange(fault_weight)]
            outcomes[shots] = _get_outcomes(mechanisms, members, oracle, cache)
        bins = fault_weights[keep] * num_columns + outcomes[keep]
        size = weight_sums.size
        weight_sums += np.bincount(bins, weights=importance[keep], minlength=size).reshape(
            weight_sums.shape
        )
        squared_weight_sums += np.bincount(
            bins, weights=importance[keep] ** 2, minlength=size
        ).reshape(weight_sums.shape)
        sample_counts += np.bincount(fault_weights[keep], minlength=len(sample_counts))


def _get_proposal_probabilities(
    odds: npt.NDArray[np.float64], target_fault_weight: float
) -> npt.NDArray[np.float64]:
    """Probabilities with odds proportional to the given odds, and a given expected fault weight.

    The expected fault weight is capped just below the number of error mechanisms.
    """
    target = min(target_fault_weight, len(odds) - 0.5)
    low, high = -100.0, 100.0
    for _ in range(200):
        middle = (low + high) / 2
        scaled = math.exp(middle) * odds
        if np.sum(scaled / (1 + scaled)) < target:
            low = middle
        else:
            high = middle
    scaled = math.exp((low + high) / 2) * odds
    return scaled / (1 + scaled)


def _get_outcomes(
    mechanisms: _Mechanisms,
    faults: npt.NDArray[np.intp],
    oracle: _ResidualWeightOracle,
    cache: dict[bytes, int],
) -> npt.NDArray[np.intp]:
    """The outcome column of each fault, given as a row of error-mechanism indices.

    Column 0 is for a fault that fires a post-selected detector, and column ``1 + m`` for one of
    residual weight ``m``, with ``m = max_residual_weight + 1`` for overflow.
    """
    flags = np.bitwise_xor.reduce(mechanisms.flag_words[faults], axis=1)
    accepted = ~np.any(flags, axis=1)
    keys = np.bitwise_xor.reduce(mechanisms.key_words[faults[accepted]], axis=1)
    outcomes = np.zeros(len(faults), dtype=np.intp)
    outcomes[accepted] = 1 + _get_cached_weights(keys, oracle, cache)
    return outcomes


def _get_cached_weights(
    keys: npt.NDArray[np.uint64], oracle: _ResidualWeightOracle, cache: dict[bytes, int]
) -> npt.NDArray[np.intp]:
    """Residual weights of packed syndromes, capped at ``max_residual_weight + 1``, with a cache."""
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    names = [row.tobytes() for row in unique]
    new = [index for index, name in enumerate(names) if name not in cache]
    if new:
        for index, weight in zip(new, oracle.get_weights(unique[new])):
            cache[names[index]] = int(weight)
    return np.array([cache[name] for name in names], dtype=np.intp)[inverse.reshape(-1)]


def _to_words(value: int, num_words: int) -> list[int]:
    """Split a nonnegative integer into 64-bit machine words, least significant first."""
    return [(value >> (64 * index)) & (2**64 - 1) for index in range(num_words)]


####################################################################################################
# exact residual weights of syndromes


class _ResidualWeightOracle(Protocol):
    """Exact residual weights of syndromes of a stabilizer state, up to a maximum weight."""

    @property
    def max_residual_weight(self) -> int:
        """The largest residual weight that the oracle resolves."""

    @property
    def num_key_words(self) -> int:
        """The number of 64-bit machine words in a packed syndrome."""

    def get_key_words(self, syndrome: int) -> list[int]:
        """Pack a syndrome, with bit ``j`` for stabilizer ``j``, into machine words."""

    def get_weights(self, keys: npt.NDArray[np.uint64]) -> npt.NDArray[np.intp]:
        """Residual weights of rows of packed syndromes, capped at ``max_residual_weight + 1``."""


class _CSSWeightTables:
    """Exact CSS weights of the syndromes of a CSS state, by a meet-in-the-middle lookup.

    The stabilizers of a CSS state are generated by X-type generators, which detect Z errors, and
    Z-type generators, which detect X errors.  The X and Z parts of a Pauli error reduce
    independently, so the CSS residual weight of a syndrome is the larger of two minima: the
    smallest weight of a Z error with the syndrome of the X-type generators, and the smallest weight
    of an X error with the syndrome of the Z-type generators.  Each minimum is the smallest weight
    of a binary vector with a given syndrome, which is at most ``max_residual_weight`` exactly when
    the vector splits into two vectors of weights at most ``max_residual_weight // 2`` and
    ``(max_residual_weight + 1) // 2``.  The tables list the syndromes of all such vectors, so a
    lookup finds the smallest weight exactly if it is at most ``max_residual_weight``.

    A packed syndrome has two words: the syndrome of the X-type generators, then that of the Z-type
    generators, each with one bit per generator.
    """

    num_key_words = 2

    def __init__(
        self,
        x_supports: npt.NDArray[np.uint8],
        z_supports: npt.NDArray[np.uint8],
        max_residual_weight: int,
    ) -> None:
        """Build the tables for the generators with the given supports, one row per generator."""
        self.max_residual_weight = max_residual_weight
        self._num_x_generators = len(x_supports)
        self._sectors = [
            (
                _get_syndrome_table(supports, max_residual_weight // 2),
                _get_syndrome_table(supports, (max_residual_weight + 1) // 2),
            )
            for supports in (x_supports, z_supports)
        ]

    def get_key_words(self, syndrome: int) -> list[int]:
        """Split a syndrome into the syndromes of the X-type and the Z-type generators."""
        mask = (1 << self._num_x_generators) - 1
        return [syndrome & mask, syndrome >> self._num_x_generators]

    def get_weights(self, keys: npt.NDArray[np.uint64]) -> npt.NDArray[np.intp]:
        """CSS weights of packed syndromes, capped at ``max_residual_weight + 1``."""
        weights = [
            _get_sector_weights(keys[:, column], small, large, self.max_residual_weight)
            for column, (small, large) in enumerate(self._sectors)
        ]
        return np.maximum(*weights)


class _CosetSearchWeights:
    """Exact residual weights of the syndromes of any stabilizer state, by a coset search.

    This uses the exact minimum-weight search of ``certify_bounded_fault_propagation``, which is
    slower than ``_CSSWeightTables`` but works for any stabilizer state and either Pauli weight.
    """

    def __init__(
        self, stabilizers: npt.NDArray[np.uint8], css_weight: bool, max_residual_weight: int
    ) -> None:
        """Prepare the search for the given stabilizer generators, one row per generator."""
        self.max_residual_weight = max_residual_weight
        self.num_key_words = (len(stabilizers) + 63) // 64
        self._search = PauliCosetSearch(stabilizers, css_weight=css_weight)

    def get_key_words(self, syndrome: int) -> list[int]:
        """Split a syndrome into machine words, least significant first."""
        return _to_words(syndrome, self.num_key_words)

    def get_weights(self, keys: npt.NDArray[np.uint64]) -> npt.NDArray[np.intp]:
        """Residual weights of packed syndromes, capped at ``max_residual_weight + 1``."""
        weights = np.zeros(len(keys), dtype=np.intp)
        for row, words in enumerate(keys):
            syndrome = sum(int(word) << (64 * index) for index, word in enumerate(words))
            # raise the weight to each new lower bound until a Pauli error that light exists
            weight = 0
            while weight <= self.max_residual_weight:
                bound = self._search.get_min_weight_bound(syndrome, weight)
                if bound.within_max_weight:
                    break
                weight = max(weight + 1, bound.lower_bound)
            weights[row] = min(weight, self.max_residual_weight + 1)
        return weights


def _get_css_supports(
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Supports of X-type and Z-type generators of a CSS stabilizer group in ``[X|Z]`` form.

    The rows of a row-reduced matrix whose leading block is zero form a basis of the vectors in its
    row space with a zero leading block.  Putting the Z part first gives the pure-X elements, and
    putting the X part first gives the pure-Z elements.
    """
    num_qubits = stabilizers.shape[1] // 2
    x_part, z_part = stabilizers[:, :num_qubits], stabilizers[:, num_qubits:]
    supports = []
    for first, second in ((z_part, x_part), (x_part, z_part)):
        reduced = np.asarray(galois.GF2(np.concatenate([first, second], axis=1)).row_reduce())
        pure = reduced[np.any(reduced, axis=1) & ~np.any(reduced[:, :num_qubits], axis=1)]
        supports.append(pure[:, num_qubits:].astype(np.uint8))
    return supports[0], supports[1]


def _get_syndrome_table(
    supports: npt.NDArray[np.uint8], max_residual_weight: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.intp]]:
    """The syndromes of all binary vectors of weight at most a maximum, with their least weights.

    Bit ``k`` of the syndrome of a vector is its parity on the support of generator ``k``.

    Returns:
        The distinct syndromes in increasing order, and the smallest weight of a vector with each.
    """
    columns = np.zeros(supports.shape[1], dtype=np.uint64)
    for generator, support in enumerate(supports):
        columns |= support.astype(np.uint64) << np.uint64(generator)
    syndromes = [np.zeros(1, dtype=np.uint64)]
    weights = [np.zeros(1, dtype=np.intp)]
    for weight in range(1, max_residual_weight + 1):
        vectors = _get_combinations(len(columns), weight)
        syndromes.append(np.bitwise_xor.reduce(columns[vectors], axis=1))
        weights.append(np.full(len(vectors), weight, dtype=np.intp))
    # the stable sort keeps the lightest vector first among those with the same syndrome
    all_syndromes = np.concatenate(syndromes)
    all_weights = np.concatenate(weights)
    order = np.argsort(all_syndromes, kind="stable")
    unique, first = np.unique(all_syndromes[order], return_index=True)
    return unique, all_weights[order][first]


def _get_sector_weights(
    syndromes: npt.NDArray[np.uint64],
    small: tuple[npt.NDArray[np.uint64], npt.NDArray[np.intp]],
    large: tuple[npt.NDArray[np.uint64], npt.NDArray[np.intp]],
    max_residual_weight: int,
) -> npt.NDArray[np.intp]:
    """Smallest weights of binary vectors with given syndromes, at most ``max_residual_weight + 1``.

    A syndrome has a vector of weight ``w <= max_residual_weight`` exactly when it is the sum of a
    syndrome in the small table and one in the large table with weights summing to at most ``w``.
    """
    small_syndromes, small_weights = small
    large_syndromes, large_weights = large
    weights = np.full(len(syndromes), max_residual_weight + 1, dtype=np.intp)
    batch = max(1, 2**22 // len(small_syndromes))
    for start in range(0, len(syndromes), batch):
        others = syndromes[start : start + batch, None] ^ small_syndromes[None, :]
        positions = np.minimum(np.searchsorted(large_syndromes, others), len(large_syndromes) - 1)
        found = large_syndromes[positions] == others
        totals = np.where(
            found, small_weights[None, :] + large_weights[positions], max_residual_weight + 1
        )
        weights[start : start + batch] = np.minimum(totals.min(axis=1), max_residual_weight + 1)
    return weights
