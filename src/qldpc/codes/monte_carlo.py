# SPDX-License-Identifier: Apache-2.0

"""Statistical helpers for code-capacity logical error rate estimation.

These utilities turn the failure and discard counts collected by the .get_logical_error_rate_func
methods of the code classes into logical error and discard rate estimates, and support the sampling
that those methods perform.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, overload

import numpy as np
import numpy.typing as npt
import scipy.special

from qldpc import _deprecations, math

####################################################################################################
# logical error rate estimation


@dataclasses.dataclass
class ErrorRateFunc:
    """Container for raw simulation data used to compute logical error and discard rates.

    An instance of this class is built with ErrorRateFunc.from_sampler and returned by the
    .get_logical_error_rate_func method of ClassicalCode, QuditCode, and CSSCode.  If::

        func = code.get_logical_error_rate_func(...),

    then ``func`` takes a physical error rate ``p`` as an argument and returns two numbers:

    (1) A logical error rate: the probability that a shot fails given that the decoder keeps it, as
        defined below.
    (2) A statistical uncertainty in that rate: the standard deviation propagated to first order
        from the Jeffreys posterior over the failed, discarded, and corrected samples of each weight
        (a Dirichlet distribution), which grows large, and loses meaning, as the discard rate
        approaches one.

    If called with an array of physical error rates, this function returns two arrays.  If called
    with ``discard_rate=True``, it computes a discard rate instead of an error rate.

    The logical error rate is the probability that a shot fails given that the decoder keeps it,
    that is, the number of failed kept shots divided by the number of kept shots.  Writing ``q_w``
    for the probability of a weight-w error, and ``f_w``, ``d_w`` and ``n_w`` for the failures,
    discards and samples at weight w, it is::

        ``(sum_w q_w f_w / n_w + T) / (1 - sum_w q_w d_w / n_w)``,

    where the denominator is one minus the discard rate, and ``T`` is the probability of an error
    too heavy to have been sampled.  Without discards, the rate is ``sum_w q_w f_w / n_w + T``.

    Errors of weight above the max_error_weight are never sampled, and this class counts every one
    of them as a kept failure, which gives ``T`` above.  The reported error rate therefore sits
    above the true error rate, by at most func.truncation_error_bound(p).  A plot of the reported
    rate usually carries a vertical bar spanning the range in which the true rate might lie.  That
    bar is asymmetric here: the statistical uncertainty spreads in both directions, but counting the
    unsampled errors as failures only pushes the reported rate up.  Writing ``value, error =
    func(p)``, the bottom and top of the bar are::

        ``lower = max(value - error - func.truncation_error_bound(p), 0)``,
        ``upper = value + error``.

    The error bar defined by ``lower`` and ``upper`` is almost, but not exactly a confidence
    interval: its lower edge mixes a posterior standard deviation with a bound that is not
    statistical at all, and its center is the observed failure rate rather than the mean of the
    posterior that supplies its width.  Read it as a rough indication of what is unknown.

    No truncation bound enters a discard rate, because an error counted as a failure is not counted
    as a discard.  A discard rate's bar runs from ``value - error`` to ``value + error``.

    Errors of weight below min_error_weight are assumed to decode perfectly, which keeps light
    errors from dominating the reported uncertainty at small physical error rates.
    """

    # number of times we sampled each error weight
    num_samples: npt.NDArray[np.int_]

    # number of samples at each error weight that the decoder kept but decoded incorrectly
    # (failures), and that it discarded (discards); no sample is both
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

    @classmethod
    def from_sampler(
        cls,
        sample_failures_and_discards: Callable[[int, int], tuple[int, int]],
        num_samples: int,
        num_error_locations: int,
        max_error_rate: float,
        min_error_weight: int = 1,
    ) -> ErrorRateFunc:
        """Sample errors of fixed weight, and collect the counts.

        The sample budget is apportioned in proportion to the largest probability that each error
        weight has at physical error rates up to max_error_rate, with at least one sample for every
        weight from min_error_weight up to the heaviest weight sampled.

        Args:
            sample_failures_and_discards: A function that samples and decodes errors of a given
                weight a given number of times, and returns the numbers of failures and discards.
            num_samples: The total sample budget.
            num_error_locations: The number of error locations, or block length.
            max_error_rate: The largest physical error rate that the ErrorRateFunc can evaluate.
            min_error_weight: The smallest error weight that the decoder is taken to be able to fail
                on; lighter errors are taken to be decoded perfectly.

        Returns:
            An ErrorRateFunc.
        """
        sample_allocation = _get_sample_allocation(
            num_samples, num_error_locations, max_error_rate, min_error_weight
        )
        num_failures = np.zeros(sample_allocation.size, dtype=int)
        num_discards = np.zeros(sample_allocation.size, dtype=int)
        for weight in np.nonzero(sample_allocation)[0].tolist():
            num_failures[weight], num_discards[weight] = sample_failures_and_discards(
                weight, sample_allocation[weight]
            )
        return cls(
            sample_allocation,
            num_failures,
            num_discards,
            num_error_locations,
            float(max_error_rate),
            min_error_weight,
        )

    @property
    def max_error_weight(self) -> int:
        """Max error weight considered."""
        return self.num_samples.size - 1

    @property
    def infidelities(self) -> npt.NDArray[np.floating]:
        """Mean infidelity at each error weight: the failure rate among its kept samples.

        This is the observed failure rate.  The posterior whose variance infidelity_variances
        reports sits above it, by 1/(2n + 2) at a weight with n kept samples and no observed
        failure, so the two do not come from the same estimator.  The logical error rate averages
        these rates over the weights of kept shots: it weights the rate at weight w by
        ``q_w (1 - d_w / n_w)``, the probability of a kept weight-w shot, adds the probability ``T``
        of an error too heavy to have been sampled, and divides by the probability that a shot is
        kept (see help(ErrorRateFunc)).
        """
        return self.num_failures / self._as_divisor(self.num_samples - self.num_discards)

    @property
    def discard_rates(self) -> npt.NDArray[np.floating]:
        """Discard rate at each error weight."""
        return self.num_discards / self._as_divisor(self.num_samples)

    @property
    def infidelity_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the infidelity at each error weight.

        With x failures in n kept samples, this is the variance of the Beta(x + 1/2, n - x + 1/2)
        posterior, which is positive even if x = 0.

        These are per-weight diagnostics.  The uncertainty that the function reports comes from the
        joint posterior of failures and discards instead, so it differs slightly from propagating
        these variances, even when nothing is discarded.
        """
        num_samples_kept = self.num_samples - self.num_discards
        variances = _jeffreys_variance(self.num_failures, num_samples_kept)
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight cannot fail
        return variances

    @property
    def discard_rate_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the discard rate at each error weight.

        With x discards in n samples, this is the variance of the Beta(x + 1/2, n - x + 1/2)
        posterior, which is positive even if x = 0.
        """
        variances = _jeffreys_variance(self.num_discards, self.num_samples)
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight is never discarded
        return variances

    @overload
    def __call__(self, error_rate: float, *, discard_rate: bool = False) -> tuple[float, float]: ...

    @overload
    def __call__(
        self, error_rate: Iterable[float], *, discard_rate: bool = False
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    def __call__(
        self, error_rate: float | Iterable[float], *, discard_rate: bool = False
    ) -> tuple[float, float] | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Compute the logical error rate (or discard rate) at one or more physical error rates.

        Args:
            error_rate: A physical error rate, at most max_error_rate, or an iterable of them.
            discard_rate: If True, compute the discard rate instead of the logical error rate.

        Returns:
            The rate and its statistical uncertainty: two floats for one physical error rate, or two
            one-dimensional arrays with one entry per physical error rate.  The logical error rate
            is taken among kept shots (see help(ErrorRateFunc)), so if every shot is discarded at a
            physical error rate, the logical error rate and its uncertainty are both NaN there, as
            in qldpc.circuits.get_logical_error_and_discard_rate.

        Raises:
            ValueError: If a physical error rate exceeds max_error_rate.
        """
        if isinstance(error_rate, Iterable):
            results = [self(rate, discard_rate=discard_rate) for rate in error_rate]
            return (
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
        if discard_rate:
            # an error treated as a failure is not a discard, so no such term applies here
            value = float(weight_probs @ self.discard_rates)
            error = float(np.sqrt(weight_probs**2 @ self.discard_rate_variances))
            return value, error

        # errors heavier than max_error_weight are all treated as kept failures
        tail = self._get_tail_probability(error_rate)
        kept = self._get_kept_probability(weight_probs, tail)
        if kept == 0:
            # every shot is discarded, so no kept shot can define a failure rate
            return float("nan"), float("nan")
        failures = self.num_failures / self._as_divisor(self.num_samples)
        value = (float(weight_probs @ failures) + tail) / kept

        # propagate the per-weight posterior covariances of the failure and discard fractions to the
        # rate, whose derivatives with respect to them are q_w / kept and value * q_w / kept
        failure_variances, discard_variances, covariances = _jeffreys_multinomial_covariances(
            self.num_failures, self.num_discards, self.num_samples
        )
        variances = failure_variances + value**2 * discard_variances + 2 * value * covariances
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight is certain
        error = float(np.sqrt(weight_probs**2 @ variances)) / kept
        return value, error

    @overload
    def truncation_error_bound(self, error_rate: float) -> float: ...

    @overload
    def truncation_error_bound(self, error_rate: Iterable[float]) -> npt.NDArray[np.floating]: ...

    def truncation_error_bound(
        self, error_rate: float | Iterable[float]
    ) -> float | npt.NDArray[np.floating]:
        """Contribution to the reported logical error rate from errors above max_error_weight.

        Errors heavier than max_error_weight are never sampled, and are all treated as kept
        failures.  This is their contribution to the reported rate: the probability ``T`` of such an
        error, divided by one minus the discard rate (see help(ErrorRateFunc)), which is ``T``
        itself without discards.  It is also the most by which the reported rate can exceed the true
        rate, which it does if every such error is in fact kept and decoded correctly.  Takes one
        physical error rate or an iterable of them, and returns one bound or an array of them to
        match, as the constructed function itself does.
        """
        if isinstance(error_rate, Iterable):
            values = [self.truncation_error_bound(rate) for rate in error_rate]
            return np.array(values)
        tail = self._get_tail_probability(error_rate)
        if tail == 0:
            return 0.0
        weight_probs = _get_error_probs_by_weight(
            self.num_error_locations, error_rate, self.max_error_weight
        )
        return tail / self._get_kept_probability(weight_probs, tail)

    def _get_kept_probability(self, weight_probs: npt.NDArray[np.floating], tail: float) -> float:
        """The probability that a shot is kept, counting every unsampled error as kept.

        This is one minus the discard rate, summed from nonnegative terms so that it is never
        negative and never below the tail probability, as subtracting the discard rate from one can
        be when nearly every shot is discarded.
        """
        return float(weight_probs @ (1 - self.discard_rates)) + tail

    def _get_tail_probability(self, error_rate: float) -> float:
        """The probability of an error too heavy to have been sampled.

        The probability of an error heavier than a given weight is the upper tail of a binomial
        distribution, which the regularized incomplete beta function gives in closed form.  With
        ``n = num_error_locations`` error locations each erring with probability p, and weights up
        to k covered,

            ``sum_(j=k+1)^(n) (n choose j) p**j (1-p)**(n-j) = I_p(k + 1, n - k)``.
        """
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

    @staticmethod
    def _as_divisor(counts: npt.NDArray[np.int_]) -> npt.NDArray[np.floating]:
        """Cast sample counts to float for use as a divisor, mapping zeros to infinity.

        Dividing by infinity yields zero, so a weight with no (kept) samples is recorded with a zero
        rate rather than nan or inf.  That is optimistic for a weight with no data; the reported
        variance does not share this optimism (see _jeffreys_variance and
        _jeffreys_multinomial_covariances).
        """
        divisor = counts.astype(float)
        divisor[divisor == 0] = np.inf
        return divisor


def _get_sample_allocation(
    num_samples: int, block_length: int, max_error_rate: float, min_error_weight: int = 1
) -> npt.NDArray[np.int_]:
    """Construct an allocation of samples by error weight.

    This method returns an array whose k-th entry is the number of samples to devote to errors of
    weight k, given a maximum error rate that we care about.

    A single allocation has to serve every physical error rate ``p <= max_error_rate``, so samples
    are apportioned in proportion to the largest probability that each error weight is ever assigned
    over that range of p (see _get_max_error_probs_by_weight).  Every sampled weight is guaranteed
    at least one sample, so no weight below the maximum sampled weight is silently recorded as
    failure-free for want of data.

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


def _jeffreys_variance(
    num_events: npt.NDArray[np.int_], num_trials: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    """Posterior variance of a binomial rate under a Jeffreys prior.

    With x events observed in n trials, the rate posterior is Beta(x + 1/2, n - x + 1/2), whose mean
    is (x + 1/2) / (n + 1) and whose variance is mean * (1 - mean) / (n + 2).  Unlike the plug-in
    variance f (1 - f) / n, this is positive at x = 0, so a weight with no observed events still
    carries uncertainty, and finite at n = 0, where it reverts to the prior variance 1/8 for a
    weight with no data at all.

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


def _jeffreys_multinomial_covariances(
    num_failures: npt.NDArray[np.int_],
    num_discards: npt.NDArray[np.int_],
    num_samples: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Posterior covariances of the failure and discard rates under a Jeffreys prior.

    Each sample fails, is discarded, or is corrected, so the three counts are multinomial, and their
    Jeffreys posterior is Dirichlet(f + 1/2, d + 1/2, c + 1/2) for f failures, d discards and c
    corrected samples.  With ``a = n + 3/2`` for n samples and the posterior means ``m = (x + 1/2) /
    a``, this returns the variances ``m (1 - m) / (a + 1)`` of the failure and discard rates and
    their covariance ``-m_f m_d / (a + 1)``.  This covariance matrix is positive definite, so no
    outcome is taken to be certain from a finite number of samples, and with no data it reverts to
    the prior Dirichlet(1/2, 1/2, 1/2).
    """
    concentration = num_samples + 1.5
    failure_mean = (num_failures + 0.5) / concentration
    discard_mean = (num_discards + 0.5) / concentration
    return (
        failure_mean * (1 - failure_mean) / (concentration + 1),
        discard_mean * (1 - discard_mean) / (concentration + 1),
        -failure_mean * discard_mean / (concentration + 1),
    )


def _check_error_weight(min_error_weight: int) -> None:
    """Reject a minimum failing error weight below one."""
    if min_error_weight < 1:
        raise ValueError("min_error_weight must be at least 1: weight 0 is a no-error case")


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
    # the range _get_sample_allocation apportions over, so the share tested is the one that weight
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
    case is never sampled and callers apportion a budget across these entries, where it must take no
    share.

    Here ``q_k(p)`` is the probability of a weight-k error at physical error rate p, as built by
    _get_error_probs_by_weight.  As a function of p, ``q_k(p)`` peaks at ``p = k / block_length``,
    so the maximum over ``p <= max_error_rate`` is attained at ``p = min(k / block_length,
    max_error_rate)``.  This envelope is the natural stand-in for a single ``q_k(p)`` when one
    allocation must serve a whole range of physical error rates: a weight contributes ``q_k(p)**2``
    to the variance of an estimate at error rate p, so the envelope is the largest that contribution
    ever gets over the range of p being served.
    """
    probs = np.zeros(max_weight + 1)
    if max_error_rate == 0:
        # no error of weight >= 1 is possible, so every entry of the envelope is zero
        return probs
    for weight in range(1, max_weight + 1):
        error_rate = min(weight / block_length, max_error_rate)
        if error_rate == 1:
            # every location errs, so all probability sits at block_length, which is this weight: an
            # error rate of one requires both max_error_rate == 1 and weight == block_length
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
        # every location has an error, so the weight is exactly block_length with probability 1.  If
        # block_length exceeds max_weight then this weight lies outside the array and its
        # probability is fully truncated, so leave all entries at zero (the missing mass is reported
        # by truncation_error_bound) rather than indexing past the end of the array.
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


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
