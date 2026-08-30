"""Monte-Carlo helpers for code-capacity logical error rate estimation.

These utilities turn the failure and discard counts collected by the .get_logical_error_rate_func
methods of the code classes into logical error and discard rate estimates, and support the sampling
that those methods perform.  They depend only on the decoder interface and elementary combinatorics,
so they live in their own module.

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
    (2) An uncertainty in the logical error rate: the standard deviation propagated from the
        per-weight Jeffreys posterior variances (see infidelity_variances).
    If called with an array of physical error rates, this function returns two arrays.

    If called with the keyword argument discard_rate=True, compute a discard rate rather than an
    error rate.
    """

    # number of times we sampled each error weight
    num_samples: npt.NDArray[np.int_]

    # number of failures and discards by error weight
    num_failures: npt.NDArray[np.int_]
    num_discards: npt.NDArray[np.int_]

    num_error_locations: int  # total number of error locations
    max_error_rate: float  # largest physical error rate we can consider

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
        if self.num_failures[0] != 0 or self.num_discards[0] != 0:
            raise ValueError("weight 0 is a no-error case: it cannot fail or be discarded")

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
        """The Jeffreys posterior variance of the infidelity at each error weight."""
        num_samples_kept = self.num_samples - self.num_discards
        variances = _jeffreys_variance(self.num_failures, num_samples_kept)
        variances[0] = 0.0  # weight 0 (no error) cannot fail
        return variances

    @property
    def discard_rate_variances(self) -> npt.NDArray[np.floating]:
        """The Jeffreys posterior variance of the discard rate at each error weight."""
        variances = _jeffreys_variance(self.num_discards, self.num_samples)
        variances[0] = 0.0  # weight 0 (no error) is never discarded
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
            values = 1 - self.discard_rates
            variances = self.discard_rate_variances
        else:
            values = 1 - self.infidelities
            variances = self.infidelity_variances
        value = weight_probs @ values
        error = np.sqrt(weight_probs**2 @ variances)
        return 1 - float(value), float(error)

    def truncation_error_bound(self, error_rate: OneOrManyFloats) -> OneOrManyFloats:
        """Upper bound on the truncation error in the infidelity or discard rate estimate."""
        if isinstance(error_rate, Iterable):
            values = [self.truncation_error_bound(rate) for rate in error_rate]
            return np.array(values)  # type:ignore[return-value]
        weight_probs = _get_error_probs_by_weight(
            self.num_error_locations, error_rate, self.max_error_weight
        )
        return float(1.0 - weight_probs.sum())


def _jeffreys_variance(
    num_events: npt.NDArray[np.int_], num_trials: npt.NDArray[np.int_]
) -> npt.NDArray[np.floating]:
    """Posterior variance of a binomial rate under a Jeffreys prior.

    With x events observed in n trials, the rate posterior is Beta(x + 1/2, n - x + 1/2), whose
    mean is (x + 1/2) / (n + 1) and whose variance is mean * (1 - mean) / (n + 2).  Unlike the
    plug-in variance f (1 - f) / n, this is positive at x = 0, so a weight with no observed events
    still carries uncertainty, and finite at n = 0, where it reverts to the prior variance 1/8 for
    a weight with no data at all.
    """
    smoothed_rate = (num_events + 0.5) / (num_trials + 1)
    return smoothed_rate * (1 - smoothed_rate) / (num_trials + 2)


def _get_sample_allocation(
    num_samples: int, block_length: int, max_error_rate: float
) -> npt.NDArray[np.int_]:
    """Construct an allocation of samples by error weight.

    This method returns an array whose k-th entry is the number of samples to devote to errors of
    weight k, given a maximum error rate that we care about.
    """
    probs = _get_error_probs_by_weight(block_length, max_error_rate)

    # zero out the distribution at k=0, flatten it out to the left of its peak, and renormalize
    probs[0] = 0
    if not probs.any():
        # no error of weight >= 1 is possible (e.g. max_error_rate 0 or an empty code), so there
        # is nothing to sample; return a lone weight-0 bin
        return np.zeros(1, dtype=int)
    probs[1 : np.argmax(probs)] = probs.max()
    probs /= np.sum(probs)

    # assign sample numbers according to the probability distribution constructed above,
    # increasing num_samples if necessary to deal with weird edge cases from round-off errors
    while np.sum(sample_allocation := np.round(probs * num_samples).astype(int)) < num_samples:
        num_samples += 1  # pragma: no cover

    # weight 0 (no error) decodes deterministically and is not sampled, so its allocation stays
    # zero; truncate trailing zeros, keeping just the weight-0 bin when nothing is allocated
    nonzero = np.nonzero(sample_allocation)[0]
    end = nonzero[-1] + 1 if nonzero.size else 1
    return sample_allocation[:end]


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
