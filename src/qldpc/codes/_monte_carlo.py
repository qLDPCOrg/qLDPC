"""Monte-Carlo helpers for code-capacity logical error rate estimation.

These utilities turn the failure and discard counts collected by the .get_logical_error_rate_func
methods of the code classes into logical error and discard rate estimates, and support the sampling
that those methods perform.  They depend only on the decoder interface and on the binomial weight
distribution, not on the code classes themselves, so they live in their own module.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TypeVar

import galois
import numpy as np
import numpy.typing as npt
import scipy.special

from qldpc import decoders, math

OneOrManyFloats = TypeVar("OneOrManyFloats", float, Iterable[float])


@dataclasses.dataclass
class ErrorRateFunc:
    """Container for raw simulation data used to compute logical error and discard rates.

    An instance of this class is built and returned by the .get_logical_error_rate_func method of
    ClassicalCode, QuditCode, and CSSCode.  If

        func = code.get_logical_error_rate_func(...),

    then "func" takes a physical error rate "p" as an argument, and returns two numbers:
    (1) A logical error rate.
    (2) A statistical uncertainty in that rate: the standard deviation propagated from the
        per-weight Jeffreys posterior variances.
    If called with an array of physical error rates, this function returns two arrays.

    An error bar covering the true rate is asymmetric.  Given ``value, error = func(p)``, draw it as

        ``lower = value - error - func.truncation_error_bound(p)``,
        ``upper = value + error``.

    The asymmetry is there because errors heavier than the heaviest sampled weight are charged as
    certain failures, which makes (1) an upper estimate: the true rate is lower by somewhere between
    zero and that bound.  The charge is a one-sided systematic rather than a standard deviation, so
    it is deliberately not part of (2).  A symmetric bar drawn from (2) alone understates what is
    not known: where the sample budget could not reach the bulk of the weight distribution, the
    charge can be the whole of the reported rate, leaving (2) a tight bar around pure artifact.

    If called with the keyword argument discard_rate=True, compute a discard rate rather than an
    error rate.

    Errors of weight below min_error_weight are taken to be decoded perfectly, so they contribute
    no uncertainty at all rather than the uncertainty that finitely many samples would leave
    behind.  This matters because the weight distribution puts most of its mass on light errors
    when the physical error rate is small, so at small error rates those weights would otherwise
    dominate the reported uncertainty while carrying no information.

    A min_error_weight above one is taken on trust.  Those weights are not sampled, so nothing
    here can test the claim, and a claim that is false makes the reported error rate too small --
    unboundedly so at small physical error rates, where the weights it excludes are the ones that
    carry the error rate.  The claim is about a particular decoder rather than about the code: a
    decoder that does not return a minimum-weight correction can fail on errors far lighter than
    half the code distance, so min_error_weight cannot be read off the distance.  Establishing it
    means checking the decoder, for example by enumerating every error of the weights in question.
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
        if self.min_error_weight < 1:
            raise ValueError("min_error_weight must be at least 1: weight 0 is a no-error case")
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
        """Mean infidelity at each error weight."""
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
        reported variance does not share this optimism (see _jeffreys_variance).
        """
        divisor = counts.astype(float)
        divisor[divisor == 0] = np.inf
        return divisor

    @property
    def infidelity_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the infidelity at each error weight.

        See help(qldpc.codes._monte_carlo._jeffreys_variance) for additional info.
        """
        num_samples_kept = self.num_samples - self.num_discards
        variances = _jeffreys_variance(self.num_failures, num_samples_kept)
        variances[: self.min_error_weight] = 0.0  # a perfectly decoded weight cannot fail
        return variances

    @property
    def discard_rate_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the discard rate at each error weight.

        See help(qldpc.codes._monte_carlo._jeffreys_variance) for additional info.
        """
        variances = _jeffreys_variance(self.num_discards, self.num_samples)
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
        if discard_rate:
            rates = self.discard_rates
            variances = self.discard_rate_variances
        else:
            rates = self.infidelities
            variances = self.infidelity_variances
        # errors heavier than max_error_weight are charged as certain failures, so their probability
        # enters the rate in full
        value = float(weight_probs @ rates) + self.truncation_error_bound(error_rate)
        error = float(np.sqrt(weight_probs**2 @ variances))
        return value, error

    def truncation_error_bound(self, error_rate: OneOrManyFloats) -> OneOrManyFloats:
        """Upper bound on the truncation error in the infidelity or discard rate estimate.

        Errors heavier than max_error_weight are charged as certain failures, so this is the
        probability of such an error: the amount by which truncation inflates a reported rate.
        Subtract it from a rate to get the low end of an error bar that covers the true rate, as
        described in the class docstring.  It covers only that tail: a min_error_weight that a
        decoder does not live up to biases a reported rate the other way, by an amount no bound here
        can speak to, since the weights that such a claim excludes are never sampled.

        The probability of an error heavier than a given weight is the upper tail of a binomial
        distribution, which the regularized incomplete beta function gives in closed form.  With
        ``n = num_error_locations`` error locations each erring with probability p, and weights up
        to k covered,

            ``sum_(j=k+1)^(n) (n choose j) p**j (1-p)**(n-j) = I_p(k + 1, n - k)``.

        Beware the argument order: scipy spells ``I_p(a, b)`` as ``betainc(a, b, p)``, taking the
        variable last where the usual notation puts it first.
        """
        if isinstance(error_rate, Iterable):
            values = [self.truncation_error_bound(rate) for rate in error_rate]
            return np.array(values)  # type:ignore[return-value]
        if self.max_error_weight >= self.num_error_locations:
            # no error is heavier than the number of error locations, so nothing is truncated.  The
            # closed form has no range left to integrate here and reports all of the mass instead.
            return 0.0
        return float(
            scipy.special.betainc(
                self.max_error_weight + 1,
                self.num_error_locations - self.max_error_weight,
                error_rate,
            )
        )


def _jeffreys_variance(
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
    small counts.  What that work recommends is the interval between two quantiles of the posterior,
    which is not what is built from this variance: callers propagate it into a symmetric half-width
    about a plug-in rate, and such a half-width does not inherit the quantile interval's coverage.
    """
    smoothed_rate = (num_events + 0.5) / (num_trials + 1)
    return smoothed_rate * (1 - smoothed_rate) / (num_trials + 2)


def _get_sample_allocation(
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

    The heaviest weight sampled is the heaviest the budget can pay for (see _get_max_error_weight),
    so a larger budget covers more weights and leaves less probability above them.

    Apportioning by that envelope, rather than by the weight distribution at max_error_rate alone,
    trades precision at the top of the range of p for precision below it.  The quantity it improves
    is the uncertainty relative to the rate being estimated, across the whole range of p served: a
    small error rate draws its rate from light errors, which the distribution at max_error_rate
    assigns little weight and therefore barely samples, so an allocation built from that
    distribution is least precise, relative to the rate, exactly where the rate is smallest.  The
    price is at ``p = max_error_rate`` itself, where the samples moved to lighter weights are no
    longer available.  Both effects are of order some tens of percent; which way the trade is worth
    making depends on how much of the range of p a caller actually reads.

    Weights below min_error_weight are taken to be decoded perfectly and get no samples: there is
    nothing to learn about them, so spending samples there would only take samples away from the
    weights that the decoder can actually fail on.  The default of one excludes weight 0, the
    no-error case.
    """
    if not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must lie in [0, 1]")
    if min_error_weight < 1:
        raise ValueError("min_error_weight must be at least 1: weight 0 is a no-error case")
    # an empty budget measures nothing, so cover nothing: the lone weight-0 bin leaves every error
    # of weight >= 1 above the covered range, where ErrorRateFunc charges it as a certain failure.
    # Knowing nothing is then reported as a rate lost entirely to truncation.
    if num_samples <= 0:
        return np.zeros(1, dtype=int)

    max_weight = _get_max_error_weight(block_length, max_error_rate, num_samples, min_error_weight)
    probs = _get_max_error_probs_by_weight(block_length, max_error_rate, max_weight)
    probs[:min_error_weight] = 0
    total = np.sum(probs)
    if total == 0:
        # nothing in range is worth sampling, either because no error of weight >= 1 is possible or
        # because every weight in range is taken to decode perfectly.  Unlike an empty budget, both
        # say that the weights in range do not fail, so cover the range and spend nothing on it.
        return np.zeros(max_weight + 1, dtype=int)
    probs /= total

    # apportion the budget by the method of largest remainders: hand each weight its floored share,
    # then give the leftover samples to the weights with the largest discarded fractions
    shares = probs * num_samples
    sample_allocation = np.floor(shares).astype(int)
    leftovers = int(num_samples - sample_allocation.sum())
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
    _get_max_error_probs_by_weight).  Half a sample is the point at which a share rounds to a whole
    one, which makes it a rounding convention rather than a derived threshold.  Every weight up to
    the heaviest that qualifies is then sampled, and is guaranteed a sample by the floor in
    _get_sample_allocation whether or not its share earned one.

    The heaviest weight anywhere above the threshold is taken, rather than the last of an unbroken
    run of them.  The two differ only when the envelope is not monotonic in weight, which happens
    above a maximum error rate of one half: the envelope of a weight below block_length / 2 is a
    peak height that falls as weight grows, and past that midpoint it rises again, so weights over
    the threshold can sit beyond a run of weights under it.  Those heavy weights carry most of the
    probability at the top of the range of error rates served, so excluding them would truncate
    exactly what matters most there.  Including them costs samples, because every weight between
    them and min_error_weight is then guaranteed one by the floor in _get_sample_allocation: that is
    how a request for a handful of samples can return an allocation of hundreds.

    Weights above the largest included one are charged as certain failures (see
    ErrorRateFunc.truncation_error_bound).  That is an upper bound on what they contribute, since a
    failure rate cannot exceed one, so a reported rate is never too small on this account.  How
    loose the bound is depends on the code and on how far the budget reached.  For a qubit
    stabilizer code the failure rate at large weight approaches 1 - 4**-dimension, because a heavy
    Pauli error is corrected to a near-uniform choice among the 4**dimension logical classes.  So
    the charge is near-exact for a code carrying many logical qubits -- on a [[144,12,12]] bivariate
    bicycle code at max_error_rate 0.1 a thousand samples reach weight 25, where the decoder fails
    on 998 errors in 1000 and every heavier weight fails on all of them.  A code carrying one
    approaches 4/3, but only well above the weights a budget of any usual size stops at: an
    [[81,1,9]] surface code measures 0.71 at weight 23 and settles at 0.75 by about weight 28, and
    at max_error_rate 0.1 the charge exceeds what the omitted weights truly contribute by 1.7 times
    at the weight a thousand samples reach and 2.2 times at the weight a hundred reach.  A classical
    code is the opposite case, since a heavy binary error carries no per-location randomness and
    simply fails: there the charge is tight.  Rather than rest on an assumption either way, the
    charge is reported with every rate, so a caller can see its size.

    At least one weight the decoder can fail on is always included, so that a small budget yields a
    poor estimate rather than no estimate at all.  That follows from the criterion itself: the
    lightest eligible weight is the only one covered when it is the heaviest, so its share is the
    whole budget and it qualifies for any budget at all.
    """
    envelope = _get_max_error_probs_by_weight(block_length, max_error_rate, block_length)
    envelope[:min_error_weight] = 0
    if envelope.sum() == 0:
        # no error of weight >= 1 is possible, or every weight in range is taken to decode
        # perfectly.  Nothing anywhere in range can fail, so cover all of it and charge nothing.
        return block_length
    # a weight's share is measured against the weights covered when it is the heaviest one, which is
    # the range _get_sample_allocation apportions over, so the share tested is the one that weight
    # would receive.  The cumulative sum is zero below min_error_weight, where nothing is eligible.
    covered = np.cumsum(envelope)
    fractions = np.divide(envelope, covered, out=np.zeros_like(envelope), where=covered > 0)
    shares = fractions * num_samples
    reaches_a_sample = np.nonzero(shares >= 0.5)[0]
    weight_from_budget = int(reaches_a_sample[-1]) if reaches_a_sample.size else 0
    return max(weight_from_budget, min_error_weight)


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


def _get_error_and_erasure(
    decoder: decoders.Decoder,
    syndrome: galois.FieldArray,
) -> tuple[galois.FieldArray, bool]:
    """Decode a syndrome and return the inferred error together with an erasure flag.

    If the decoder has a has_erasure_bit attribute set to True (e.g., a LookupDecoder constructed
    with add_erasure_bit=True), the last element of the decoded vector is treated as the erasure
    bit: 1 means the syndrome was not recognized and the sample should be discarded, 0 means a
    correction was found normally.  The erasure bit is stripped before returning the error.
    """
    error = decoder.decode(syndrome.view(np.ndarray))
    if getattr(decoder, "has_erasure_bit", False):
        return error[:-1].view(type(syndrome)), bool(error[-1])
    return error.view(type(syndrome)), False
