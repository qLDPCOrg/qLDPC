# SPDX-License-Identifier: Apache-2.0

"""Monte-Carlo helpers for code-capacity logical error rate estimation.

These utilities turn the failure and discard counts collected by the .get_logical_error_rate_func
methods of the code classes into logical error and discard rate estimates, and support the sampling
that those methods perform.  They depend only on the decoder interface and on the binomial weight
distribution, not on the code classes themselves, so they live in their own module.

Code-capacity sampling only ever asks a decoder which observables an error flips (see
CodeCapacityDecoder and get_code_capacity_decoder).  An error decoder is used by converting the
errors that it infers into observable flips.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar, cast, get_type_hints

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.special
import stim

from qldpc import decoders, math
from qldpc.decoders.custom import PLACEHOLDER_ERROR_RATE

OneOrManyFloats = TypeVar("OneOrManyFloats", float, Iterable[float])


@dataclasses.dataclass
class ErrorRateFunc:
    """Container for raw simulation data used to compute logical error and discard rates.

    An instance of this class is built and returned by the .get_logical_error_rate_func method of
    ClassicalCode, QuditCode, and CSSCode.  If::

        func = code.get_logical_error_rate_func(...),

    then ``func`` takes a physical error rate ``p`` as an argument and returns two numbers:

    (1) A logical error rate, estimated over the error weights that were sampled.
    (2) A statistical uncertainty in that rate: the standard deviation propagated from the
        per-weight Jeffreys posterior variances.

    If called with an array of physical error rates, this function returns two arrays.  If called
    with ``discard_rate=True``, it computes a discard rate instead of an error rate.

    Errors of weight above the max_error_weight are never sampled, and this class counts every one
    of them as a failure.  The reported error rate therefore sits above the true error rate, by at
    most func.truncation_error_bound(p), which is the probability of drawing such a heavy error.
    A plot of the reported rate usually carries a vertical bar spanning the range in which the true
    rate might lie.  That bar is asymmetric here: the statistical uncertainty spreads in both
    directions, but counting the unsampled errors as failures only pushes the reported rate up.
    Writing ``value, error = func(p)``, the bottom and top of the bar are::

        ``lower = max(value - error - func.truncation_error_bound(p), 0)``,
        ``upper = value + error``.

    The error bar defined by ``lower`` and ``upper`` is almost, but not exactly a confidence
    interval: its lower edge mixes a posterior standard deviation with a bound that is not
    statistical at all, and its center is the observed failure rate rather than the mean of the
    posterior that supplies its width (see infidelities).  Read it as a rough indication of what is
    unknown.

    No truncation bound enters a discard rate, because an error counted as a failure is not counted
    as a discard.  A discard rate's bar runs from ``value - error`` to ``value + error``.

    Errors of weight below min_error_weight are assumed to decode perfectly, which keeps light
    errors from dominating the reported uncertainty at small physical error rates.
    """

    # number of times we sampled each error weight
    num_samples: npt.NDArray[np.int_]

    # number of failures and discards by error weight
    num_failures: npt.NDArray[np.int_]
    num_discards: npt.NDArray[np.int_]

    num_error_locations: int  # total number of error locations
    max_error_rate: float  # largest physical error rate we can consider

    # smallest error weight that the decoder is taken to be capable of failing on; every lighter
    # error is treated as decoded perfectly, contributing neither a rate nor an uncertainty
    min_error_weight: int = 1

    def __post_init__(self) -> None:
        """Check that the counts form a consistent set of per-weight binomial observations."""
        if not self.num_samples.shape == self.num_failures.shape == self.num_discards.shape:
            raise ValueError("num_samples, num_failures, and num_discards must have equal shape")
        if self.num_samples.size == 0:
            raise ValueError("at least one error weight is required")
        if np.any(self.num_failures < 0) or np.any(self.num_discards < 0):
            raise ValueError("failure and discard counts must be non-negative")
        if np.any(self.num_failures + self.num_discards > self.num_samples):
            raise ValueError("failures plus discards cannot exceed the samples at any weight")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate must lie in [0, 1]")
        _check_error_weight(self.min_error_weight)
        # the weight-0 case is the min_error_weight=1 instance of this check: a no-error sample can
        # neither fail nor be discarded
        below = slice(None, self.min_error_weight)
        if np.any(self.num_failures[below]) or np.any(self.num_discards[below]):
            raise ValueError(
                f"errors of weight below min_error_weight={self.min_error_weight} are taken to be"
                " decoded perfectly, so they cannot fail or be discarded"
            )

    @property
    def max_error_weight(self) -> int:
        """Max error weight considered."""
        return self.num_samples.size - 1

    @property
    def infidelities(self) -> npt.NDArray[np.floating]:
        """Mean infidelity at each error weight.

        This is the observed failure rate.  The posterior whose variance infidelity_variances
        reports sits above it, by 1/(2n + 2) at a weight with n kept samples and no observed
        failure, so the reported rate and its uncertainty do not come from the same estimator.
        """
        return self.num_failures / self._as_divisor(self.num_samples - self.num_discards)

    @property
    def discard_rates(self) -> npt.NDArray[np.floating]:
        """Discard rate at each error weight."""
        return self.num_discards / self._as_divisor(self.num_samples)

    @staticmethod
    def _as_divisor(counts: npt.NDArray[np.int_]) -> npt.NDArray[np.floating]:
        """Cast sample counts to float for use as a divisor, mapping zeros to infinity.

        Dividing by infinity yields zero, so a weight with no (kept) samples is recorded with a
        zero rate rather than nan or inf.  That is optimistic for a weight with no data; the
        reported variance does not share this optimism (see jeffreys_variance).
        """
        divisor = counts.astype(float)
        divisor[divisor == 0] = np.inf
        return divisor

    @property
    def infidelity_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the infidelity at each error weight.

        See help(qldpc.codes.monte_carlo.jeffreys_variance) for additional info.
        """
        num_samples_kept = self.num_samples - self.num_discards
        variances = jeffreys_variance(self.num_failures, num_samples_kept)
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight cannot fail
        return variances

    @property
    def discard_rate_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the discard rate at each error weight.

        See help(qldpc.codes.monte_carlo.jeffreys_variance) for additional info.
        """
        variances = jeffreys_variance(self.num_discards, self.num_samples)
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight is never discarded
        return variances

    def __call__(
        self, error_rate: OneOrManyFloats, *, discard_rate: bool = False
    ) -> tuple[OneOrManyFloats, OneOrManyFloats]:
        """Compute the logical error rate (or discard rate) at a given physical error rate."""
        if isinstance(error_rate, Iterable):
            results = [self(rate, discard_rate=discard_rate) for rate in error_rate]
            return (  # type:ignore[return-value]
                np.array([result[0] for result in results]),
                np.array([result[1] for result in results]),
            )
        if error_rate > self.max_error_rate:
            raise ValueError(
                "This ErrorRateFunc does not cover physical error rates greater than"
                f" {self.max_error_rate}.  Try calling <YOUR_CODE>.get_logical_error_rate_func with"
                " a larger max_error_rate."
            )
        weight_probs = _get_error_probs_by_weight(
            self.num_error_locations, error_rate, self.max_error_weight
        )
        value: OneOrManyFloats
        if discard_rate:
            # an error treated as a failure is not a discard, so no such term applies here
            value = float(weight_probs @ self.discard_rates)
            variances = self.discard_rate_variances
        else:
            # errors heavier than max_error_weight are all treated as failures, so their
            # probability enters the rate in full
            truncation = self.truncation_error_bound(error_rate)
            value = float(weight_probs @ self.infidelities) + truncation
            variances = self.infidelity_variances
        error = float(np.sqrt(weight_probs**2 @ variances))
        return value, error

    def truncation_error_bound(self, error_rate: OneOrManyFloats) -> OneOrManyFloats:
        """Contribution to the reported infidelity from errors too heavy to have been sampled.

        Errors heavier than max_error_weight are all treated as failures, so this is the probability
        of such an error.  Takes one physical error rate or an iterable of them, and returns one
        bound or an array of them to match, as the constructed function itself does.

        The probability of an error heavier than a given weight is the upper tail of a binomial
        distribution, which the regularized incomplete beta function gives in closed form.  With
        ``n = num_error_locations`` error locations each erring with probability p, and weights up
        to k covered,

            ``sum_(j=k+1)^(n) (n choose j) p**j (1-p)**(n-j) = I_p(k + 1, n - k)``.
        """
        if isinstance(error_rate, Iterable):
            values = [self.truncation_error_bound(rate) for rate in error_rate]
            return np.array(values)  # type:ignore[return-value]
        if self.max_error_weight >= self.num_error_locations:
            # no error is heavier than the number of error locations, so nothing is truncated.  The
            # closed form cannot be relied on to say so, because its second parameter is zero here:
            # it then returns zero at every rate below one, which is right, but one at a rate of
            # exactly one, which is not.
            return 0.0
        return float(
            scipy.special.betainc(
                self.max_error_weight + 1,
                self.num_error_locations - self.max_error_weight,
                error_rate,
            )
        )


def _check_error_weight(min_error_weight: int) -> None:
    """Reject a minimum failing error weight below one."""
    if min_error_weight < 1:
        raise ValueError("min_error_weight must be at least 1: weight 0 is a no-error case")


def jeffreys_variance(
    num_events: npt.NDArray[np.int_], num_trials: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    """Posterior variance of a binomial rate under a Jeffreys prior.

    With x events observed in n trials, the rate posterior is Beta(x + 1/2, n - x + 1/2), whose
    mean is (x + 1/2) / (n + 1) and whose variance is mean * (1 - mean) / (n + 2).  Unlike the
    plug-in variance f (1 - f) / n, this is positive at x = 0, so a weight with no observed events
    still carries uncertainty, and finite at n = 0, where it reverts to the prior variance 1/8 for
    a weight with no data at all.

    See Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion," Statist. Sci. 16
    (2001) 101-133, https://doi.org/10.1214/ss/1009213286, for this posterior and its behaviour at
    small counts.  That work recommends quoting an interval between two quantiles of the posterior,
    which callers here cannot use: they need one uncertainty on a weighted sum over error weights,
    and variances add in quadrature across that sum while quantiles do not.  A variance is therefore
    what this function returns, at the price of a symmetric half-width that does not inherit the
    coverage of the recommended interval.
    """
    smoothed_rate = (num_events + 0.5) / (num_trials + 1)
    return smoothed_rate * (1 - smoothed_rate) / (num_trials + 2)


def get_sample_allocation(
    num_samples: int, block_length: int, max_error_rate: float, min_error_weight: int = 1
) -> npt.NDArray[np.int_]:
    """Construct an allocation of samples by error weight.

    This method returns an array whose k-th entry is the number of samples to devote to errors of
    weight k, given a maximum error rate that we care about.

    A single allocation has to serve every physical error rate ``p <= max_error_rate``, so samples
    are apportioned in proportion to the largest probability that each error weight is ever
    assigned over that range of p (see _get_max_error_probs_by_weight).  Every sampled weight is
    guaranteed at least one sample, so no weight below the maximum sampled weight is silently
    recorded as failure-free for want of data.

    The heaviest weight sampled is the heaviest whose share of the budget reaches half a sample (see
    _get_max_error_weight), so a larger budget covers more weights and leaves less probability above
    them.  Covering a weight costs at least one sample, so a budget spread thinly over many weights
    can be overspent.

    Weights below min_error_weight are taken to be decoded perfectly and get no samples: there is
    nothing to learn about them, so spending samples there would only take samples away from the
    weights that the decoder can actually fail on.  The default of one excludes weight 0, the
    no-error case.
    """
    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must lie in [0, 1]")
    _check_error_weight(min_error_weight)
    # an empty budget measures nothing, so cover only the weights a caller has declared cannot fail:
    # every heavier error then lies above the covered range and is treated as a failure.  A
    # min_error_weight past the block length is clamped to it, as every other return path here is.
    if num_samples <= 0:
        return np.zeros(min(min_error_weight, block_length + 1), dtype=int)

    max_weight = _get_max_error_weight(block_length, max_error_rate, num_samples, min_error_weight)
    probs = _get_max_error_probs_by_weight(block_length, max_error_rate, max_weight)
    probs[:min_error_weight] = 0
    total = np.sum(probs)
    if total == 0:
        # nothing in range is worth sampling: no error of weight >= min_error_weight is possible, or
        # every weight in range is taken to decode perfectly, or the envelope has underflowed, the
        # error rate being so small that a qualifying error has no representable probability.  All
        # three say the weights in range contribute nothing measurable, so cover the range and spend
        # nothing on it.  An empty budget is the different case, where nothing is known rather than
        # nothing can happen.
        return np.zeros(max_weight + 1, dtype=int)
    probs /= total

    # apportion the budget by the method of largest remainders: hand each weight its floored share,
    # then give the leftover samples to the weights with the largest discarded fractions
    shares = probs * num_samples
    sample_allocation = np.floor(shares).astype(int)
    leftovers = num_samples - sample_allocation.sum()
    ranked = (
        min_error_weight
        + np.argsort(shares[min_error_weight:] - sample_allocation[min_error_weight:])[::-1]
    )
    sample_allocation[ranked[:leftovers]] += 1

    sample_allocation[min_error_weight:] = np.maximum(sample_allocation[min_error_weight:], 1)
    return sample_allocation


def _get_max_error_weight(
    block_length: int, max_error_rate: float, num_samples: int, min_error_weight: int = 1
) -> int:
    """Largest error weight to sample, given a maximum error rate that we care about.

    A weight qualifies when its share of the budget reaches half a sample, that share being the one
    the allocation would hand it were it the heaviest weight covered (see
    _get_max_error_probs_by_weight).  The heaviest qualifying weight is taken.

    Weights above it are treated as certain failures rather than sampled, by an amount
    ErrorRateFunc.truncation_error_bound reports.
    """
    envelope = _get_max_error_probs_by_weight(block_length, max_error_rate, block_length)
    envelope[:min_error_weight] = 0
    if envelope.sum() == 0:
        # no error of weight >= 1 is possible, or every weight in range is taken to decode
        # perfectly.  Nothing anywhere in range can fail, so cover all of it and truncate nothing.
        return block_length
    # a weight's share is measured against the weights covered when it is the heaviest one, which is
    # the range get_sample_allocation apportions over, so the share tested is the one that weight
    # would receive.  The cumulative sum is zero below min_error_weight, where nothing is eligible.
    covered = np.cumsum(envelope)
    fractions = np.divide(envelope, covered, out=np.zeros_like(envelope), where=covered > 0)
    shares = fractions * num_samples
    reaches_a_sample = np.nonzero(shares >= 0.5)[0]
    return int(reaches_a_sample[-1])


def _get_max_error_probs_by_weight(
    block_length: int, max_error_rate: float, max_weight: int
) -> npt.NDArray[np.floating]:
    """Build an array whose k-th entry, for k >= 1, is ``max_(p <= max_error_rate) q_k(p)``.

    Entry 0 is held at zero rather than at the one that ``q_0(0)`` attains, because the no-error
    case is never sampled and callers apportion a budget across these entries, where it must take
    no share.

    Here ``q_k(p)`` is the probability of a weight-k error at physical error rate p, as built by
    _get_error_probs_by_weight.  As a function of p, ``q_k(p)`` peaks at ``p = k / block_length``,
    so the maximum over ``p <= max_error_rate`` is attained at ``p = min(k / block_length,
    max_error_rate)``.  This envelope is the natural stand-in for a single ``q_k(p)`` when one
    allocation must serve a whole range of physical error rates: a weight contributes
    ``q_k(p)**2`` to the variance of an estimate at error rate p, so the envelope is the largest
    that contribution ever gets over the range of p being served.
    """
    probs = np.zeros(max_weight + 1)
    if max_error_rate == 0:
        # no error of weight >= 1 is possible, so every entry of the envelope is zero
        return probs
    for weight in range(1, max_weight + 1):
        error_rate = min(weight / block_length, max_error_rate)
        if error_rate == 1:
            # every location errs, so all probability sits at block_length, which is this weight:
            # an error rate of one requires both max_error_rate == 1 and weight == block_length
            probs[weight] = 1
        else:
            probs[weight] = np.exp(
                math.log_choose(block_length, weight)
                + weight * np.log(error_rate)
                + (block_length - weight) * np.log(1 - error_rate)
            )
    return probs


def _get_error_probs_by_weight(
    block_length: int, error_rate: float, max_weight: int | None = None
) -> npt.NDArray[np.floating]:
    """Build an array whose k-th entry is the probability of a weight-k error in a code.

    If a code has block_length n and each bit has an independent probability ``p = error_rate`` of
    an error, then the probability of k errors is ``(n choose k) p**k (1-p)**(n-k)``.

    We compute the above probability using logarithms because otherwise the combinatorial factor
    ``(n choose k)`` might be too large to handle.
    """
    max_weight = block_length if max_weight is None else max_weight

    # deal with some pathological cases
    if error_rate == 0:
        probs = np.zeros(max_weight + 1)
        probs[0] = 1
        return probs
    elif error_rate == 1:
        # every location has an error, so the weight is exactly block_length with probability 1.
        # If block_length exceeds max_weight then this weight lies outside the array and its
        # probability is fully truncated, so leave all entries at zero (the missing mass is
        # reported by truncation_error_bound) rather than indexing past the end of the array.
        probs = np.zeros(max_weight + 1)
        if block_length <= max_weight:
            probs[block_length] = 1
        return probs

    log_error_rate = np.log(error_rate)
    log_one_minus_error_rate = np.log(1 - error_rate)
    log_probs = [
        math.log_choose(block_length, kk)
        + kk * log_error_rate
        + (block_length - kk) * log_one_minus_error_rate
        for kk in range(max_weight + 1)
    ]
    return np.exp(log_probs)


def get_error_and_erasure(
    decoder: decoders.ErrorDecoder | decoders.SupportsDecode,
    syndrome: galois.FieldArray,
) -> tuple[galois.FieldArray, bool]:
    """Decode a syndrome and return the inferred error together with an erasure flag.

    If the decoder has a has_erasure_bit attribute set to True (e.g., a LookupDecoder constructed
    with ``add_erasure_bit=True``), the last element of the decoded vector is treated as the erasure
    bit: 1 means the syndrome was not recognized and the sample should be discarded, 0 means a
    correction was found normally.  The erasure bit is stripped before returning the error.
    """
    error = decoders.as_error_decoder(decoder).decode_errors(syndrome.view(np.ndarray))
    if getattr(decoder, "has_erasure_bit", False):
        return error[:-1].view(type(syndrome)), bool(error[-1])
    return error.view(type(syndrome)), False


################################################################################
# decoders for code-capacity sampling


def get_code_capacity_dem(
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    dem_errors: galois.FieldArray | None = None,
    *,
    symplectic_errors: bool = False,
    error_probs: npt.NDArray[np.floating] | float = PLACEHOLDER_ERROR_RATE,
) -> stim.DetectorErrorModel:
    """Build the detector error model of one sector of a code-capacity experiment.

    Error mechanism j of the model is the error in column j of dem_errors.  By default, each error
    location is an error mechanism.  With symplectic_errors, the mechanisms are instead the
    single-qudit X, Z, and Y errors, in that order.  An error mechanism flips the detectors of its
    syndrome ``syndrome_matrix @ error``, and the observables of its observable values
    ``observable_matrix @ error``.  If observable_matrix is None, every error location is itself an
    observable.

    The error probabilities must be fixed while the returned code-capacity estimator is evaluated
    at different physical error rates.  A code-capacity estimate reuses the decoding outcomes of
    fixed-weight errors at every physical error rate, so it requires such a decoder.

    Raises:
        ValueError: If the matrices are not binary, since Stim detector error models are binary.
    """
    field = type(syndrome_matrix)
    if getattr(field, "order", 2) != 2:
        raise ValueError(
            "A Sinter-style decoder is compiled for a Stim detector error model, which is binary, so"
            f" it cannot decode a code over {field.name}.  Pass decoder settings such as"
            " decoders.guf(), or a prebuilt observable decoder (such as an ObservableLookupDecoder)"
            " built for this code, instead"
        )
    if symplectic_errors:
        if dem_errors is not None:
            raise ValueError("dem_errors and symplectic_errors cannot both be specified")
        if observable_matrix is None:
            raise ValueError("symplectic_errors requires an observable_matrix")
        detector_flip_matrix = _get_single_qudit_error_effects(syndrome_matrix)
        observable_flip_matrix = _get_single_qudit_error_effects(observable_matrix)
    elif dem_errors is not None:
        detector_flip_matrix = syndrome_matrix @ dem_errors
        observable_flip_matrix = (
            dem_errors if observable_matrix is None else observable_matrix @ dem_errors
        )
    else:
        detector_flip_matrix = syndrome_matrix
        observable_flip_matrix = (
            observable_matrix
            if observable_matrix is not None
            else scipy.sparse.identity(syndrome_matrix.shape[1], dtype=np.uint8, format="csc")
        )
    dem_arrays = decoders.DetectorErrorModelArrays.from_arrays(
        np.asarray(detector_flip_matrix, dtype=np.uint8),
        observable_flip_matrix,
        error_probs,
    )
    return dem_arrays.to_dem()


def _get_single_qudit_error_effects(matrix: galois.FieldArray) -> galois.FieldArray:
    """Apply a symplectic map to single-qudit X, Z, and Y errors without forming those errors."""
    components_x, components_z = np.hsplit(matrix, 2)
    return np.hstack([components_x, components_z, components_x + components_z]).view(type(matrix))


@dataclasses.dataclass(frozen=True)
class CodeCapacityDecoder:
    """Observable decoder for one sector of a code-capacity experiment.

    A code-capacity experiment samples errors, and decodes the syndrome ``syndrome_matrix @ error``
    of each error to predict the values ``observable_matrix @ error`` of its observables.  Decoding
    fails if the prediction differs from these values, and is discarded if the decoder signals
    erasure by setting any of the num_erasure_flags flags that it appends to each prediction.
    Matrices are arrays over the field of the code; None denotes an identity observable map.

    Build a CodeCapacityDecoder with get_code_capacity_decoder.
    """

    # predicts the values of the observables of an error from its syndrome
    decoder: decoders.ObservableDecoder
    # maps an error to the syndrome that the decoder decodes
    syndrome_matrix: galois.FieldArray
    # maps an error to the values of its observables
    observable_matrix: galois.FieldArray | None
    # number of erasure flags that the decoder appends to each prediction
    num_erasure_flags: int = 0

    @property
    def field(self) -> type[galois.FieldArray]:
        """The field of the syndromes and observables of this decoder."""
        return type(self.syndrome_matrix)

    @property
    def num_observables(self) -> int:
        """The number of observables that this decoder predicts."""
        return (
            self.syndrome_matrix.shape[1]
            if self.observable_matrix is None
            else len(self.observable_matrix)
        )

    @property
    def can_discard(self) -> bool:
        """Whether this decoder can signal erasure, and thereby discard a sample."""
        return self.num_erasure_flags > 0

    @staticmethod
    def from_error_decoder(
        error_decoder: decoders.ErrorDecoder,
        syndrome_matrix: galois.FieldArray,
        observable_matrix: galois.FieldArray | None,
    ) -> CodeCapacityDecoder:
        """Predict observables by converting the errors that an error decoder infers."""
        observable_decoder = _ErrorsToFieldObservablesDecoder(
            error_decoder,
            observable_matrix,
            type(syndrome_matrix),
            syndrome_matrix.shape[1],
        )
        return CodeCapacityDecoder(
            observable_decoder,
            syndrome_matrix,
            observable_matrix,
            int(observable_decoder.has_erasure_bit),
        )

    def reuse_for(
        self,
        syndrome_matrix: galois.FieldArray,
        observable_matrix: galois.FieldArray | None,
    ) -> CodeCapacityDecoder | None:
        """Reuse this decoder for another sector with the same syndrome matrix, if possible.

        The decoder is reused as is if the observable matrices are also equal.  An error decoder is
        reused with a different observable matrix, since the errors that it infers do not depend on
        observables.  Otherwise, return None: an observable decoder only predicts the observables
        that it was built for.
        """
        if not np.array_equal(syndrome_matrix, self.syndrome_matrix):
            return None
        if _observable_matrices_equal(observable_matrix, self.observable_matrix):
            return self
        if isinstance(self.decoder, _ErrorsToFieldObservablesDecoder):
            return CodeCapacityDecoder.from_error_decoder(
                self.decoder.error_decoder, syndrome_matrix, observable_matrix
            )
        return None

    def decode(self, syndrome: galois.FieldArray) -> tuple[galois.FieldArray, bool]:
        """Predict the observable values of an error from its syndrome, and whether it was erased.

        Raises:
            ValueError: If the decoder returns a prediction of the wrong shape, with entries that
                are not elements of the field of the code, or with erasure flags other than 0 or 1.
        """
        prediction = np.asarray(self.decoder.decode_observables(syndrome.view(np.ndarray)))
        _validate_decoder_output(
            prediction,
            self.num_observables,
            self.num_erasure_flags,
            self.field,
            "An observable decoder predicted observable values",
        )
        observables = self.field(prediction[: self.num_observables].astype(int))
        return observables, bool(np.any(prediction[self.num_observables :]))

    def get_failure_and_erasure(self, error: galois.FieldArray) -> tuple[bool, bool]:
        """Decode the syndrome of an error, and report whether decoding failed or was erased.

        Decoding fails if the predicted observable values differ from those of the error.  An erased
        sample is not a failure.
        """
        predicted_observables, erased = self.decode(self.syndrome_matrix @ error)
        if erased:
            return False, True
        actual_observables = (
            error if self.observable_matrix is None else self.observable_matrix @ error
        )
        return bool(np.any(predicted_observables != actual_observables)), False


def _observable_matrices_equal(
    matrix_a: galois.FieldArray | None, matrix_b: galois.FieldArray | None
) -> bool:
    """Whether two observable maps, where None denotes the identity, are equal."""
    if matrix_a is None or matrix_b is None:
        return matrix_a is matrix_b
    return bool(np.array_equal(matrix_a, matrix_b))


def get_code_capacity_decoder(
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    decoder: decoders.ErrorDecoderInput | decoders.ObservableDecoderInput,
    decoder_args: Mapping[str, object] | None = None,
    *,
    dem_errors: galois.FieldArray | None = None,
    symplectic_dem_errors: bool = False,
    dem_error_weights: npt.NDArray[np.floating] | None = None,
    prebuilt_rejection_reason: str | None = None,
    warn_deprecated: bool = True,
) -> CodeCapacityDecoder:
    """Build an observable decoder for one sector of a code-capacity experiment.

    Code-capacity sampling decodes the syndrome ``syndrome_matrix @ error`` of each sampled error to
    predict its observable values ``observable_matrix @ error``.  The decoder input is resolved into
    an observable decoder as follows:

    - A Sinter-style decoder (an object with a compile_decoder_for_dem method, such as a
      decoders.SinterDecoder), or a constructor explicitly declared to return an observable decoder,
      is built for the detector error model that get_code_capacity_dem constructs.  This requires
      the matrices to be binary.
    - A prebuilt observable decoder (an object with a decode_observables method, or a compiled
      Sinter decoder with a decode_shots_bit_packed method, that is not also an error decoder) is
      used as is.  It must predict the observable values ``observable_matrix @ error`` from the
      syndrome ``syndrome_matrix @ error``.
    - Anything else (None, decoder settings, a constructor, or a prebuilt error decoder, together
      with any deprecated decoder_args) builds an error decoder for syndrome_matrix, exactly as
      qldpc.decoders.resolve_decoder does.  The observable values of the errors that it infers are
      its predictions.  A decoder that is both an error decoder and an observable decoder, such as a
      RelayBPDecoder, is used as an error decoder.

    Args:
        syndrome_matrix: The matrix that maps an error to its syndrome.
        observable_matrix: The matrix that maps an error to its observable values, or None if every
            error location is itself an observable.
        decoder: The decoder input.
        decoder_args: Deprecated keyword-based decoder options, which build an error decoder.
        dem_errors: The errors of the error mechanisms of the detector error model for which a
            Sinter-style decoder is compiled, as columns of a matrix.  Defaults to the identity
            matrix, making each error location an error mechanism.
        symplectic_dem_errors: Whether the detector error model has one X, Z, and Y mechanism per
            qudit.  Cannot be combined with dem_errors.
        dem_error_weights: Relative probabilities for the detector error model's error mechanisms.
            These are scaled by a fixed placeholder error rate, so decoder decisions do not change
            when the returned estimator is evaluated at different physical error rates.
        prebuilt_rejection_reason: If not None, reject a prebuilt (error or observable) decoder,
            with this reason; see help(qldpc.decoders.reject_prebuilt_decoder).  A Sinter-style
            decoder is compiled here, so it is not rejected.
        warn_deprecated: Whether to warn when decoder_args is nonempty.

    Returns:
        A CodeCapacityDecoder.
    """
    decoder_args = decoder_args or {}
    dem_error_probs: npt.NDArray[np.floating] | float = PLACEHOLDER_ERROR_RATE
    if dem_error_weights is not None:
        dem_error_probs = PLACEHOLDER_ERROR_RATE * np.asarray(dem_error_weights, dtype=float)
    if not decoder_args and compiles_for_dem(decoder):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)  # type:ignore[union-attr]
        return _get_observable_code_capacity_decoder(
            compiled_decoder,
            syndrome_matrix,
            observable_matrix,
            "A decoder compiled by compile_decoder_for_dem",
            require_dimensions=False,
        )

    if prebuilt_rejection_reason is not None:
        decoders.reject_prebuilt_decoder(decoder, prebuilt_rejection_reason)

    if not decoder_args and is_prebuilt_observable_decoder(decoder):
        return _get_observable_code_capacity_decoder(
            decoder,
            syndrome_matrix,
            observable_matrix,
            "A prebuilt observable decoder",
            require_dimensions=True,
        )

    if not decoder_args and constructs_observable_decoder(decoder):
        dem = get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            dem_errors,
            symplectic_errors=symplectic_dem_errors,
            error_probs=dem_error_probs,
        )
        constructor = cast(decoders.ObservableDecoderConstructor, decoder)
        return _get_observable_code_capacity_decoder(
            constructor(dem),
            syndrome_matrix,
            observable_matrix,
            "An observable decoder constructor",
            require_dimensions=False,
        )

    error_decoder = decoders.resolve_decoder(
        syndrome_matrix,
        decoder,  # type:ignore[arg-type]
        decoder_args,
        warn_deprecated=warn_deprecated,
    )
    return CodeCapacityDecoder.from_error_decoder(error_decoder, syndrome_matrix, observable_matrix)


def compiles_for_dem(decoder: object) -> bool:
    """Whether a decoder input is a Sinter-style decoder, compiled for a detector error model.

    Such a decoder, like a decoders.SinterDecoder, has a compile_decoder_for_dem method.
    """
    return not isinstance(decoder, type) and callable(
        getattr(decoder, "compile_decoder_for_dem", None)
    )


def is_prebuilt_observable_decoder(decoder: object) -> bool:
    """Whether a decoder input is a prebuilt decoder that predicts observables, but not errors.

    Such a decoder has a decode_observables method, or is a compiled Sinter decoder with a
    decode_shots_bit_packed method.  It is not a Sinter-style decoder that still has to be compiled
    for a detector error model (see compiles_for_dem), and it is not an error decoder: a decoder
    that can do both, such as a RelayBPDecoder, is used as an error decoder.
    """
    returns_observables = bool(getattr(decoder, "decode_returns_observables", False))
    is_error_decoder = isinstance(decoder, decoders.ErrorDecoder) or (
        isinstance(decoder, decoders.SupportsDecode) and not returns_observables
    )
    return (
        not isinstance(decoder, (type, decoders.DecoderSpec))
        and not is_error_decoder
        and not compiles_for_dem(decoder)
        and (
            isinstance(decoder, decoders.ObservableDecoder)
            or callable(getattr(decoder, "decode_shots_bit_packed", None))
        )
    )


def constructs_observable_decoder(decoder: object) -> bool:
    """Whether a callable explicitly declares that it constructs an observable decoder."""
    if not callable(decoder) or decoders.is_prebuilt_decoder(decoder):
        return False
    if isinstance(decoder, type):
        return issubclass(decoder, decoders.ObservableDecoder) and not issubclass(
            decoder, decoders.ErrorDecoder
        )
    try:
        return_annotation = get_type_hints(decoder).get("return")
    except (NameError, TypeError):
        return False
    return _annotation_is_observable_decoder(return_annotation)


def _annotation_is_observable_decoder(annotation: object) -> bool:
    """Whether a return annotation identifies an observable decoder, but not an error decoder."""
    if not isinstance(annotation, type):
        return False
    return issubclass(annotation, decoders.ObservableDecoder) and not issubclass(
        annotation, decoders.ErrorDecoder
    )


def _get_observable_code_capacity_decoder(
    decoder: object,
    syndrome_matrix: galois.FieldArray,
    observable_matrix: galois.FieldArray | None,
    source: str,
    *,
    require_dimensions: bool,
) -> CodeCapacityDecoder:
    """Wrap a prebuilt or compiled observable decoder, checking that it fits the given matrices."""
    field = type(syndrome_matrix)
    num_detectors = len(syndrome_matrix)
    num_observables = (
        syndrome_matrix.shape[1] if observable_matrix is None else len(observable_matrix)
    )
    observable_decoder: decoders.ObservableDecoder
    if isinstance(decoder, decoders.CompiledSinterDecoder):
        if field.order != 2:
            raise ValueError(f"{source} is binary, so it cannot decode a code over {field.name}")
        observable_decoder, num_erasure_flags = decoder, decoder.num_erasure_bits
    elif isinstance(decoder, decoders.ObservableDecoder):
        num_erasure_flags = int(bool(getattr(decoder, "has_erasure_bit", False)))
        observable_decoder = decoder
    elif callable(getattr(decoder, "decode_shots_bit_packed", None)):
        if field.order != 2:
            raise ValueError(f"{source} is binary, so it cannot decode a code over {field.name}")
        observable_decoder = _BitPackedObservableDecoder(decoder, num_observables)
        num_erasure_flags = 1
    else:
        raise TypeError(
            f"{source} must provide a decode_observables or decode_shots_bit_packed method"
        )

    dimensions = [("num_detectors", num_detectors), ("num_observables", num_observables)]
    for name, expected in dimensions:
        value = getattr(decoder, name, None)
        if (
            require_dimensions
            and not isinstance(decoder, decoders.ObservableDecoder)
            and not isinstance(value, (int, np.integer))
        ):
            raise ValueError(
                f"{source} does not declare {name}, so its compatibility with this code-capacity"
                " sector cannot be validated"
            )
        if isinstance(value, (int, np.integer)) and value != expected:
            raise ValueError(
                f"{source} has {name}={value}, but this code-capacity sector has {expected}.  An"
                " observable decoder must be built for the syndromes and observables of the sector"
                " that it decodes"
            )
    decoder_field = getattr(decoder, "field", None)
    if (
        isinstance(decoder_field, type)
        and issubclass(decoder_field, galois.FieldArray)
        and decoder_field is not field
    ):
        raise ValueError(
            f"{source} is built over {decoder_field.name}, but this code-capacity sector is over"
            f" {field.name}"
        )
    return CodeCapacityDecoder(
        observable_decoder, syndrome_matrix, observable_matrix, num_erasure_flags
    )


def _validate_decoder_output(
    output: npt.NDArray[Any],
    num_values: int,
    num_erasure_flags: int,
    field: type[galois.FieldArray],
    source: str,
) -> None:
    """Check that a decoder output holds num_values field elements followed by erasure flags."""
    expected_shape = (num_values + num_erasure_flags,)
    if output.shape != expected_shape:
        flags = f" and {num_erasure_flags} erasure flag(s)" if num_erasure_flags else ""
        raise ValueError(
            f"{source} of shape {output.shape}, but expected shape {expected_shape}:"
            f" {num_values} value(s){flags}"
        )
    if not (np.issubdtype(output.dtype, np.integer) or np.issubdtype(output.dtype, np.bool_)):
        raise ValueError(f"{source} of dtype {output.dtype}, but expected integers")
    values, erasure_flags = output[:num_values].astype(int), output[num_values:].astype(int)
    if np.any(values < 0) or np.any(values >= field.order):
        raise ValueError(f"{source} with entries that are not elements of {field.name}")
    if np.any((erasure_flags != 0) & (erasure_flags != 1)):
        raise ValueError(f"{source} with erasure flags that are not 0 or 1")


class _ErrorsToFieldObservablesDecoder(decoders.ObservableDecoder):
    """Observable decoder that converts the errors that an error decoder infers into observables.

    The observable values of an inferred error are ``observable_matrix @ error``, over the field of
    observable_matrix.  If the error decoder signals erasure, its erasure bit is appended to each
    prediction.
    """

    def __init__(
        self,
        error_decoder: decoders.ErrorDecoder,
        observable_matrix: galois.FieldArray | None,
        field: type[galois.FieldArray],
        num_error_locations: int,
    ) -> None:
        self.error_decoder = error_decoder
        self.observable_matrix = observable_matrix
        self.field = field
        self.num_error_locations = num_error_locations
        self.has_erasure_bit = bool(getattr(error_decoder, "has_erasure_bit", False))

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable values."""
        num_error_locations = self.num_error_locations
        error = np.asarray(self.error_decoder.decode_errors(syndrome))
        _validate_decoder_output(
            error,
            num_error_locations,
            int(self.has_erasure_bit),
            self.field,
            "An error decoder inferred an error",
        )
        inferred_error = self.field(error[:num_error_locations].astype(int))
        observables = (
            inferred_error
            if self.observable_matrix is None
            else self.observable_matrix @ inferred_error
        )
        return np.concatenate([observables.view(np.ndarray), error[num_error_locations:]])


class _BitPackedObservableDecoder(decoders.ObservableDecoder):
    """Observable decoder that wraps a compiled Sinter decoder with bit-packed inputs and outputs.

    The compiled decoder predicts one bit-packed byte per eight observables, and may add one byte,
    which asks for the shot to be discarded if it is nonzero.  Each prediction of this decoder ends
    with an erasure flag that is set if the shot is to be discarded.
    """

    has_erasure_bit = True

    def __init__(self, compiled_decoder: Any, num_observables: int) -> None:
        self.compiled_decoder = compiled_decoder
        self.num_observables = num_observables

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips and an erasure flag."""
        packed_syndrome = np.packbits(
            np.asarray(syndrome, dtype=np.uint8).reshape(1, -1), bitorder="little", axis=1
        )
        packed_prediction = np.asarray(
            self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=packed_syndrome
            ),
            dtype=np.uint8,
        )
        num_bytes = -(-self.num_observables // 8)
        if packed_prediction.shape not in [(1, num_bytes), (1, num_bytes + 1)]:
            raise ValueError(
                f"A compiled Sinter decoder predicted bit-packed observable flips of shape"
                f" {packed_prediction.shape} for one shot, but {self.num_observables} observables"
                f" take shape (1, {num_bytes}), or (1, {num_bytes + 1}) with a byte added to signal"
                " discards"
            )
        flips = np.unpackbits(
            packed_prediction[0, :num_bytes], count=self.num_observables, bitorder="little"
        )
        erased = bool(np.any(packed_prediction[0, num_bytes:]))
        return np.append(flips, np.uint8(erased))
