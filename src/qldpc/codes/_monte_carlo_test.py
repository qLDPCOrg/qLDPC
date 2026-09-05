"""Unit tests for _monte_carlo.py.

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

import galois
import numpy as np
import pytest

from qldpc.codes import _monte_carlo


def test_get_error_probs_by_weight() -> None:
    """Probability of a weight-k error under an i.i.d. error model."""
    # a zero error rate puts all probability on the weight-0 error
    probs = _monte_carlo._get_error_probs_by_weight(5, 0.0)
    assert probs[0] == 1 and probs[1:].sum() == 0

    # a unit error rate puts all probability on the maximum weight
    probs = _monte_carlo._get_error_probs_by_weight(5, 1.0)
    assert probs.shape == (6,) and probs[5] == 1 and probs[:5].sum() == 0

    # an intermediate error rate gives a normalized truncated binomial distribution
    probs = _monte_carlo._get_error_probs_by_weight(5, 0.3, max_weight=3)
    assert probs.shape == (4,) and np.all(probs >= 0)
    assert np.isclose(probs.sum(), sum(_monte_carlo._get_error_probs_by_weight(5, 0.3)[:4]))

    # max_weight=0 keeps only the weight-0 entry rather than being treated as "unset"
    probs = _monte_carlo._get_error_probs_by_weight(5, 0.3, max_weight=0)
    assert probs.shape == (1,) and np.isclose(probs[0], 0.7**5)


def test_get_max_error_probs_by_weight() -> None:
    """Envelope of the weight distribution over a range of physical error rates."""
    block_length, max_error_rate = 20, 0.1
    probs = _monte_carlo._get_max_error_probs_by_weight(block_length, max_error_rate, 8)
    assert probs.shape == (9,) and probs[0] == 0  # weight 0 carries no weight in an allocation

    # each entry bounds its weight's probability at every error rate in range, and is attained
    for weight in range(1, 9):
        scanned = max(
            _monte_carlo._get_error_probs_by_weight(block_length, rate, weight)[weight]
            for rate in np.linspace(0, max_error_rate, 200)
        )
        assert scanned <= probs[weight] <= scanned * (1 + 1e-4)

    # the top weight's envelope is one: q_n(p) peaks at p = 1, where every location errs
    probs = _monte_carlo._get_max_error_probs_by_weight(5, 1.0, 5)
    assert probs[5] == 1
    assert probs[1] == pytest.approx(5 * 0.2 * 0.8**4)  # q_1 peaks at p = 1/5


def test_get_max_error_weight() -> None:
    """Choice of the largest error weight to sample."""
    block_length, max_error_rate = 50, 0.2

    # coverage reaches the heaviest weight whose share of the budget reaches half a sample -- that
    # share taken over the weights it would cover -- and no weight above it comes that close
    num_samples = 10**4
    max_weight = _monte_carlo._get_max_error_weight(block_length, max_error_rate, num_samples)
    envelope = _monte_carlo._get_max_error_probs_by_weight(
        block_length, max_error_rate, block_length
    )
    envelope[0] = 0
    covered = np.cumsum(envelope)
    fractions = np.divide(envelope, covered, out=np.zeros_like(envelope), where=covered > 0)
    shares = fractions * num_samples
    assert shares[max_weight] >= 0.5 and np.all(shares[max_weight + 1 :] < 0.5)

    # half a sample is the boundary itself, not merely somewhere near it: the largest budget that
    # leaves a weight's share below half excludes that weight, and one more sample brings it in
    weight = 20
    just_under = int(np.floor(0.5 / fractions[weight]))
    assert fractions[weight] * just_under < 0.5 <= fractions[weight] * (just_under + 1)
    for budget, expected in [(just_under, weight - 1), (just_under + 1, weight)]:
        assert _monte_carlo._get_max_error_weight(block_length, max_error_rate, budget) == expected

    # a larger budget covers more weights
    weights = [
        _monte_carlo._get_max_error_weight(block_length, max_error_rate, num_samples)
        for num_samples in [1, 10**3, 10**6, 10**9]
    ]
    assert weights == sorted(weights) and weights[0] < weights[-1]

    # the smallest budget still covers a weight the decoder can fail on, so that a small budget
    # gives a poor estimate rather than none at all.  The lightest eligible weight is measured
    # against itself alone, so its share is the whole budget and it qualifies however small that is.
    assert _monte_carlo._get_max_error_weight(block_length, max_error_rate, 1) == 1
    assert _monte_carlo._get_max_error_weight(block_length, max_error_rate, 1, 7) == 7

    # a zero error rate, or a claim that every possible weight decodes perfectly, leaves nothing
    # that can fail in range, so the whole range is covered and nothing in it is charged as failure
    assert _monte_carlo._get_max_error_weight(block_length, 0.0, 10**9) == block_length
    assert (
        _monte_carlo._get_max_error_weight(
            block_length, max_error_rate, 10**9, min_error_weight=block_length + 1
        )
        == block_length
    )

    # an empty code has no error weights to cover at all
    assert _monte_carlo._get_max_error_weight(0, max_error_rate, 10**9) == 0


def test_get_sample_allocation() -> None:
    """Allocation of samples across error weights."""
    allocation = _monte_carlo._get_sample_allocation(1000, block_length=10, max_error_rate=0.2)
    assert allocation[0] == 0  # weight 0 (no error) is not sampled
    assert np.sum(allocation) >= 1000  # every requested sample is allocated
    assert np.all(allocation[1:] > 0)  # no sampled weight is left without data

    # a larger budget covers more weights, so that spending more samples also shrinks the mass
    # charged as certain failure above them (see _get_max_error_weight)
    sizes = [
        _monte_carlo._get_sample_allocation(num_samples, 40, 0.2).size
        for num_samples in [10, 1000, 100000, 10**9]
    ]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]

    # the budget is apportioned in proportion to the envelope of the weight distribution
    num_samples, block_length, max_error_rate = 10**6, 40, 0.2
    allocation = _monte_carlo._get_sample_allocation(num_samples, block_length, max_error_rate)
    probs = _monte_carlo._get_max_error_probs_by_weight(
        block_length, max_error_rate, allocation.size - 1
    )
    # each weight gets its exact share, up to the one sample that integer apportionment can shift
    # and the floor of one sample that keeps a lightly weighted tail weight from going unsampled
    shares = probs / probs.sum() * num_samples
    assert np.all(np.abs(allocation[1:] - np.maximum(shares[1:], 1)) <= 1)

    # when every weight earns a sample outright, so that the floor never lifts one, the largest
    # remainders apportion the budget exactly rather than losing samples to rounding.  Taking the
    # floor of each share is what makes that hold, which these parameters are chosen to expose:
    # rounding to nearest instead would spend more than the budget, leaving nothing to redistribute
    num_samples, block_length = 1000, 5
    allocation = _monte_carlo._get_sample_allocation(num_samples, block_length, 0.2)
    probs = _monte_carlo._get_max_error_probs_by_weight(block_length, 0.2, allocation.size - 1)
    shares = probs[1:] / probs.sum() * num_samples
    assert np.min(shares) > 1 and np.round(shares).sum() > num_samples
    assert np.sum(allocation) == num_samples

    # weights that the decoder is taken to decode perfectly get no samples at all
    allocation = _monte_carlo._get_sample_allocation(1000, 40, 0.2, min_error_weight=4)
    assert not allocation[:4].any() and np.all(allocation[4:] > 0)

    # weight 0 is a no-error case, so a min_error_weight below one is rejected
    with pytest.raises(ValueError, match="min_error_weight must be at least 1"):
        _monte_carlo._get_sample_allocation(1000, 10, 0.2, min_error_weight=0)

    # an empty budget covers nothing, leaving every error of weight >= 1 charged as a failure: with
    # nothing measured, the whole reported rate is truncation
    assert np.array_equal(_monte_carlo._get_sample_allocation(0, 10, 0.2), [0])
    assert np.array_equal(_monte_carlo._get_sample_allocation(1000, 0, 0.2), [0])  # no locations

    # nothing worth sampling is a different statement: it says the weights in range do not fail, so
    # the range is covered without spending anything on it, and nothing in it is charged as failure
    for allocation in [
        _monte_carlo._get_sample_allocation(1000, 10, 0.0),  # no error possible
        _monte_carlo._get_sample_allocation(1000, 10, 0.05, min_error_weight=20),  # none can fail
    ]:
        assert np.array_equal(allocation, np.zeros(11))

    # an error rate outside [0, 1] is rejected rather than building a nonsense weight distribution
    for max_error_rate in [-0.1, 1.5, float("nan")]:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            _monte_carlo._get_sample_allocation(1000, 10, max_error_rate)


def test_get_error_and_erasure() -> None:
    """Decoding a syndrome, with and without an erasure bit."""
    field = galois.GF2
    syndrome = field([1, 0, 1])

    class _Decoder:
        def __init__(self, output: np.ndarray, has_erasure_bit: bool = False) -> None:
            self.output = output
            if has_erasure_bit:
                self.has_erasure_bit = True

        def decode(self, syndrome: np.ndarray) -> np.ndarray:
            return self.output

    # a plain decoder returns the inferred error and no erasure
    decoder = _Decoder(np.array([1, 1, 0, 0], dtype=np.uint8))
    error, erasure = _monte_carlo._get_error_and_erasure(decoder, syndrome)
    assert not erasure and isinstance(error, field) and np.array_equal(error, field([1, 1, 0, 0]))

    # an erasure-enabled decoder strips the last (erasure) bit and reports it
    decoder = _Decoder(np.array([1, 1, 0, 0, 1], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = _monte_carlo._get_error_and_erasure(decoder, syndrome)
    assert erasure and np.array_equal(error, field([1, 1, 0, 0]))


def test_jeffreys_variance() -> None:
    """Posterior variance of a binomial rate under a Jeffreys prior."""
    events = np.array([0, 0, 5])
    trials = np.array([0, 100, 100])
    variances = _monte_carlo._jeffreys_variance(events, trials)

    # no data reverts to the prior variance 1/8 (Beta(1/2, 1/2))
    assert np.isclose(variances[0], 1 / 8)

    # zero observed events over many trials still carries positive uncertainty
    assert variances[1] > 0

    # the general entry matches mean * (1 - mean) / (n + 2) with mean = (x + 1/2) / (n + 1)
    mean = (5 + 0.5) / (100 + 1)
    assert np.isclose(variances[2], mean * (1 - mean) / (100 + 2))


def test_error_rate_func_validation() -> None:
    """Inconsistent count arrays are rejected at construction."""

    def make(
        samples: list[int], failures: list[int], discards: list[int], min_error_weight: int = 1
    ) -> None:
        _monte_carlo.ErrorRateFunc(
            min_error_weight=min_error_weight,
            num_samples=np.array(samples),
            num_failures=np.array(failures),
            num_discards=np.array(discards),
            num_error_locations=5,
            max_error_rate=0.5,
        )

    # the count arrays must share a shape
    with pytest.raises(ValueError, match="equal shape"):
        make([10, 10], [0], [0])

    # at least one error weight is required
    with pytest.raises(ValueError, match="at least one error weight"):
        make([], [], [])

    # counts cannot be negative
    with pytest.raises(ValueError, match="non-negative"):
        make([10], [-1], [0])

    # failures plus discards cannot exceed the samples at any weight
    with pytest.raises(ValueError, match="cannot exceed"):
        make([10], [7], [5])

    # weight 0 is the no-error case and cannot record failures or discards
    with pytest.raises(ValueError, match="cannot fail or be discarded"):
        make([10, 10], [3, 0], [0, 0])
    with pytest.raises(ValueError, match="cannot fail or be discarded"):
        make([10, 10], [0, 0], [3, 0])

    # neither can a heavier weight that min_error_weight claims is decoded perfectly
    with pytest.raises(ValueError, match="cannot fail or be discarded"):
        make([0, 10, 10], [0, 3, 0], [0, 0, 0], min_error_weight=2)

    # a min_error_weight below one would un-zero the no-error case
    with pytest.raises(ValueError, match="min_error_weight must be at least 1"):
        make([10], [0], [0], min_error_weight=0)

    # the maximum error rate must be a probability
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _monte_carlo.ErrorRateFunc(
            num_samples=np.array([0, 10]),
            num_failures=np.array([0, 1]),
            num_discards=np.array([0, 0]),
            num_error_locations=5,
            max_error_rate=1.5,
        )


def test_error_bar_survives_zero_failures() -> None:
    """The reported uncertainty stays positive when contributing weights see zero failures.

    This is the pathology the Jeffreys variance fixes: a plug-in f(1 - f)/n variance is exactly
    zero at every weight with no observed failures, so the aggregate error bar collapses to zero
    in precisely the rare-event regime the estimate exists to measure.
    """
    func = _monte_carlo.ErrorRateFunc(
        num_samples=np.array([1, 100, 100]),
        num_failures=np.array([0, 0, 0]),  # no observed failures at any weight
        num_discards=np.array([0, 0, 0]),
        num_error_locations=5,
        max_error_rate=0.5,
    )
    # every sampled weight (>0) carries positive variance despite zero observed failures
    assert np.all(func.infidelity_variances[1:] > 0)

    # so the aggregate error bar is positive at a physical error rate that weights those bins
    _, uncertainty = func(0.1)
    assert uncertainty > 0


def test_error_rate_rises_from_zero() -> None:
    """A reported rate stays positive and grows with the physical error rate.

    Taking the probability above the covered weights as one minus the probability below them loses
    every significant digit once the covered weights hold nearly all of it, which is the ordinary
    case at a small physical error rate.  The reported rate then flattens onto zero and goes
    negative, which no probability may do.  The counts below describe a decoder that corrects every
    error of weight at most four and fails on every heavier one.
    """
    num_samples = np.full(9, 10)
    num_samples[0] = 0
    func = _monte_carlo.ErrorRateFunc(
        num_samples=num_samples,
        num_failures=np.where(np.arange(9) > 4, num_samples, 0),
        num_discards=np.zeros(9, dtype=int),
        num_error_locations=9,
        max_error_rate=0.05,
    )
    rates = np.asarray(func(np.logspace(-9, np.log10(0.05), 50))[0])
    assert np.all(rates > 0) and np.all(np.diff(rates) > 0)

    # a covered range reaching the block length has no weight above it left to charge, including at
    # an error rate of one, where the closed form for that charge has an empty range to report
    counts = np.zeros(2, dtype=int)
    func = _monte_carlo.ErrorRateFunc(counts, counts, counts, 1, 1.0)
    assert func(1.0)[0] == 0 and func.truncation_error_bound(1.0) == 0


def test_error_rate_func_min_error_weight() -> None:
    """Weights taken to be decoded perfectly contribute no uncertainty.

    Without min_error_weight these unsampled weights would each revert to the Jeffreys prior
    variance 1/8, and at a small physical error rate they carry the largest weight probabilities,
    so they would dominate the reported uncertainty while carrying no information at all.
    """
    func = _monte_carlo.ErrorRateFunc(
        num_samples=np.array([0, 0, 0, 100]),
        num_failures=np.array([0, 0, 0, 5]),
        num_discards=np.array([0, 0, 0, 0]),
        num_error_locations=10,
        max_error_rate=0.5,
        min_error_weight=3,
    )
    assert np.all(func.infidelity_variances[:3] == 0)
    assert np.all(func.discard_rate_variances[:3] == 0)
    assert func.infidelity_variances[3] > 0


def test_error_rate_func() -> None:
    """Convert raw failure and discard counts into error and discard rate estimates."""
    func = _monte_carlo.ErrorRateFunc(
        num_samples=np.array([1, 100, 100]),
        num_failures=np.array([0, 10, 0]),
        num_discards=np.array([0, 0, 100]),  # every weight-2 sample is discarded
        num_error_locations=5,
        max_error_rate=0.5,
    )
    assert func.max_error_weight == 2

    # a weight whose samples are all discarded has an undefined infidelity, recorded as zero
    assert func.infidelities[2] == 0
    assert np.isclose(func.infidelities[1], 0.1)
    assert np.array_equal(func.discard_rates, [0, 0, 1])

    # the deterministic weight-0 (no-error) case carries no uncertainty
    assert func.infidelity_variances[0] == 0
    assert func.discard_rate_variances[0] == 0

    # a weight with observed events has positive variance, as does the discard path at a weight
    # with zero observed discards (0 of 100) -- the Jeffreys variance does not collapse there
    assert func.infidelity_variances[1] > 0
    assert func.discard_rate_variances[1] > 0

    # a weight with no kept samples reverts to the Jeffreys prior variance 1/8 (weight 2: every
    # sample discarded)
    assert np.isclose(func.infidelity_variances[2], 1 / 8)

    # a scalar physical error rate yields a (rate, uncertainty) pair, for errors and discards alike
    error_rate, uncertainty = func(0.1)
    assert 0 <= error_rate <= 1 and uncertainty >= 0
    discard_rate, uncertainty = func(0.1, discard_rate=True)
    assert 0 <= discard_rate <= 1 and uncertainty >= 0

    # an iterable of physical error rates yields arrays of rates and uncertainties
    rates, uncertainties = func([0.0, 0.1])
    rates, uncertainties = np.asarray(rates), np.asarray(uncertainties)
    assert rates.shape == (2,) and uncertainties.shape == (2,)
    assert rates[0] == 0  # a zero physical error rate gives a zero logical error rate

    # physical error rates beyond the constructed range are rejected
    with pytest.raises(ValueError, match="does not cover"):
        func(0.9)

    # the truncation error bound is available for scalar and iterable inputs
    assert 0 <= func.truncation_error_bound(0.1) <= 1
    assert np.asarray(func.truncation_error_bound([0.1, 0.2])).shape == (2,)


def test_error_rate_func_single_weight() -> None:
    """A degenerate func covering only the weight-0 bin evaluates without crashing."""
    func = _monte_carlo.ErrorRateFunc(
        num_samples=np.array([0]),
        num_failures=np.array([0]),
        num_discards=np.array([0]),
        num_error_locations=5,
        max_error_rate=0.5,
    )
    assert func.max_error_weight == 0

    # every error of weight >= 1 lies outside the covered range, so it is fully truncated: the
    # reported rate is the whole of that charge, with no statistical uncertainty behind it
    error_rate, uncertainty = func(0.1)
    assert np.isclose(error_rate, 1 - 0.9**5) and uncertainty == 0
    assert error_rate == func.truncation_error_bound(0.1)
