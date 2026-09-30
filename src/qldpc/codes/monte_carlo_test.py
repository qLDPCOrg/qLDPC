# SPDX-License-Identifier: Apache-2.0

"""Unit tests for monte_carlo.py."""

from __future__ import annotations

from typing import Any

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.stats
import sinter
import stim

from qldpc import codes, decoders
from qldpc.codes import monte_carlo


def test_get_error_probs_by_weight() -> None:
    """Probability of a weight-k error under an i.i.d. error model."""
    # a zero error rate puts all probability on the weight-0 error
    probs = monte_carlo._get_error_probs_by_weight(5, 0.0)
    assert probs[0] == 1 and probs[1:].sum() == 0

    # a unit error rate puts all probability on the maximum weight
    probs = monte_carlo._get_error_probs_by_weight(5, 1.0)
    assert probs.shape == (6,) and probs[5] == 1 and probs[:5].sum() == 0

    # if that weight lies above the covered range, its probability is truncated away entirely rather
    # than written past the end of the array, leaving a distribution that carries no mass at all
    probs = monte_carlo._get_error_probs_by_weight(5, 1.0, max_weight=3)
    assert probs.shape == (4,) and probs.sum() == 0

    # an intermediate error rate gives a normalized truncated binomial distribution
    probs = monte_carlo._get_error_probs_by_weight(5, 0.3, max_weight=3)
    assert probs.shape == (4,)
    assert np.isclose(probs.sum(), sum(monte_carlo._get_error_probs_by_weight(5, 0.3)[:4]))

    # max_weight=0 keeps only the weight-0 entry rather than being treated as "unset"
    probs = monte_carlo._get_error_probs_by_weight(5, 0.3, max_weight=0)
    assert probs.shape == (1,) and np.isclose(probs[0], 0.7**5)


def test_get_max_error_probs_by_weight() -> None:
    """Envelope of the weight distribution over a range of physical error rates."""
    block_length, max_error_rate = 20, 0.1
    probs = monte_carlo._get_max_error_probs_by_weight(block_length, max_error_rate, 8)
    assert probs.shape == (9,) and probs[0] == 0  # weight 0 carries no weight in an allocation

    # each entry bounds its weight's probability at every error rate in range, and is attained
    for weight in range(1, 9):
        scanned = max(
            monte_carlo._get_error_probs_by_weight(block_length, rate, weight)[weight]
            for rate in np.linspace(0, max_error_rate, 200)
        )
        assert scanned <= probs[weight] <= scanned * (1 + 1e-4)

    # the top weight's envelope is one: q_n(p) peaks at p = 1, where every location errs
    probs = monte_carlo._get_max_error_probs_by_weight(5, 1.0, 5)
    assert probs[5] == 1
    assert probs[1] == pytest.approx(5 * 0.2 * 0.8**4)  # q_1 peaks at p = 1/5


def test_get_max_error_weight() -> None:
    """Choice of the largest error weight to sample."""
    block_length, max_error_rate = 50, 0.2

    # a weight's share of the budget is taken over the weights it would cover
    envelope = monte_carlo._get_max_error_probs_by_weight(
        block_length, max_error_rate, block_length
    )
    envelope[0] = 0
    covered = np.cumsum(envelope)
    fractions = np.divide(envelope, covered, out=np.zeros_like(envelope), where=covered > 0)

    # one sample either side of the boundary changes the answer: the largest budget that leaves a
    # weight's share below half excludes that weight, and one more sample brings it in.  That
    # brackets the threshold to within one share, about 3% of a sample here, rather than pinning it
    # at exactly half -- which is a rounding choice rather than a derived quantity
    weight = 20
    just_under = int(np.floor(0.5 / fractions[weight]))
    assert fractions[weight] * just_under < 0.5 <= fractions[weight] * (just_under + 1)
    for budget, expected in [(just_under, weight - 1), (just_under + 1, weight)]:
        assert monte_carlo._get_max_error_weight(block_length, max_error_rate, budget) == expected

    # a larger budget covers more weights
    weights = [
        monte_carlo._get_max_error_weight(block_length, max_error_rate, num_samples)
        for num_samples in [1, 10**3, 10**6, 10**9]
    ]
    assert weights == sorted(weights) and weights[0] < weights[-1]

    # the smallest budget still covers a weight the decoder can fail on, so that a small budget
    # gives a poor estimate rather than none at all.  The lightest eligible weight is measured
    # against itself alone, so its share is the whole budget, which reaches half a sample whenever
    # that budget is one or more
    assert monte_carlo._get_max_error_weight(block_length, max_error_rate, 1) == 1
    assert monte_carlo._get_max_error_weight(block_length, max_error_rate, 1, 7) == 7

    # a zero error rate, or a claim that every possible weight decodes perfectly, leaves nothing
    # that can fail in range, so the whole range is covered and nothing in it is a failure
    assert monte_carlo._get_max_error_weight(block_length, 0.0, 10**9) == block_length
    assert (
        monte_carlo._get_max_error_weight(
            block_length, max_error_rate, 10**9, min_error_weight=block_length + 1
        )
        == block_length
    )

    # the heaviest qualifying weight is taken, not the end of the first unbroken run: above a
    # maximum error rate of one half the envelope turns, so weight 6 qualifies while 4 and 5 do not.
    # This is also the share rule reaching block_length, rather than the no-failure shortcut above
    assert monte_carlo._get_max_error_weight(6, 0.95, 2) == 6

    # an empty code has no error weights to cover at all
    assert monte_carlo._get_max_error_weight(0, max_error_rate, 10**9) == 0


def test_get_sample_allocation() -> None:
    """Allocation of samples across error weights."""
    allocation = monte_carlo.get_sample_allocation(1000, block_length=10, max_error_rate=0.2)
    assert allocation[0] == 0  # weight 0 (no error) is not sampled
    assert np.sum(allocation) >= 1000  # every requested sample is allocated
    assert np.all(allocation[1:] > 0)  # no sampled weight is left without data

    # the budget is apportioned in proportion to the envelope of the weight distribution
    num_samples, block_length, max_error_rate = 10**6, 40, 0.2
    allocation = monte_carlo.get_sample_allocation(num_samples, block_length, max_error_rate)
    probs = monte_carlo._get_max_error_probs_by_weight(
        block_length, max_error_rate, allocation.size - 1
    )
    # each weight takes the floor of its share and the leftover samples go to the weights whose
    # discarded fractions were largest, with a floor of one sample keeping a lightly weighted tail
    # weight from going unsampled.  Spelling that rule out here, rather than allowing every weight a
    # sample either way, is what pins which weights the leftovers land on
    shares = probs / probs.sum() * num_samples
    expected = np.floor(shares).astype(int)
    leftovers = num_samples - int(expected.sum())
    largest_remainders = 1 + np.argsort(shares[1:] - expected[1:])[::-1]
    expected[largest_remainders[:leftovers]] += 1
    expected[1:] = np.maximum(expected[1:], 1)
    assert leftovers > 0 and np.array_equal(allocation, expected)

    # when every weight earns a sample outright, so that the floor never lifts one, the largest
    # remainders apportion the budget exactly rather than losing samples to rounding.  Taking the
    # floor of each share is what makes that hold, which these parameters are chosen to expose:
    # rounding to nearest instead would spend more than the budget, leaving nothing to redistribute
    num_samples, block_length = 1000, 5
    allocation = monte_carlo.get_sample_allocation(num_samples, block_length, 0.2)
    probs = monte_carlo._get_max_error_probs_by_weight(block_length, 0.2, allocation.size - 1)
    shares = probs[1:] / probs.sum() * num_samples
    assert np.min(shares) > 1 and np.round(shares).sum() > num_samples
    assert np.sum(allocation) == num_samples

    # weights that the decoder is taken to decode perfectly get no samples at all
    allocation = monte_carlo.get_sample_allocation(1000, 40, 0.2, min_error_weight=4)
    assert not allocation[:4].any() and np.all(allocation[4:] > 0)

    # weight 0 is a no-error case, so a min_error_weight below one is rejected
    with pytest.raises(ValueError, match="min_error_weight must be at least 1"):
        monte_carlo.get_sample_allocation(1000, 10, 0.2, min_error_weight=0)

    # an empty budget measures nothing, so it covers exactly the weights a caller has declared
    # cannot fail and treats every heavier error as a failure: the whole reported rate is then
    # truncation, and the declared weights stay out of it
    assert np.array_equal(monte_carlo.get_sample_allocation(0, 10, 0.2), [0])
    assert np.array_equal(monte_carlo.get_sample_allocation(0, 10, 0.2, 4), np.zeros(4))

    # a claim reaching past the block length covers the same weights as one that stops there, and is
    # clamped to it rather than sizing the returned array by the claim
    assert np.array_equal(
        monte_carlo.get_sample_allocation(0, 10, 0.2, 10**8),
        monte_carlo.get_sample_allocation(1000, 10, 0.2, 10**8),
    )

    # nothing worth sampling is a different statement: it says the weights in range do not fail, so
    # the range is covered without spending anything on it, and nothing in it is a failure
    for allocation, covered in [
        (monte_carlo.get_sample_allocation(1000, 10, 0.0), 11),  # no error possible
        (monte_carlo.get_sample_allocation(1000, 10, 0.05, 20), 11),  # none of them can fail
        (monte_carlo.get_sample_allocation(1000, 0, 0.2), 1),  # no error locations to err
    ]:
        assert np.array_equal(allocation, np.zeros(covered))

    # an error rate outside [0, 1] is rejected rather than building a nonsense weight distribution.
    # nan is the case that pins the two-sided comparison: rejecting it with "< 0 or > 1" instead
    # would silently admit it, every nan comparison being false
    for max_error_rate in [1.5, float("nan")]:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            monte_carlo.get_sample_allocation(1000, 10, max_error_rate)


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
    error, erasure = monte_carlo.get_error_and_erasure(decoder, syndrome)
    assert not erasure and isinstance(error, field) and np.array_equal(error, field([1, 1, 0, 0]))

    # an erasure-enabled decoder strips the last (erasure) bit and reports it.  The first and last
    # entries differ, so reading the wrong end of the vector fails here
    decoder = _Decoder(np.array([0, 1, 1, 0, 1], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = monte_carlo.get_error_and_erasure(decoder, syndrome)
    assert erasure and np.array_equal(error, field([0, 1, 1, 0]))

    # the same decoder reports no erasure when the syndrome was recognized
    decoder = _Decoder(np.array([1, 1, 0, 0, 0], dtype=np.uint8), has_erasure_bit=True)
    error, erasure = monte_carlo.get_error_and_erasure(decoder, syndrome)
    assert not erasure and np.array_equal(error, field([1, 1, 0, 0]))


def test_jeffreys_variance() -> None:
    """Posterior variance of a binomial rate under a Jeffreys prior."""
    events = np.array([0, 0, 5])
    trials = np.array([0, 100, 100])
    variances = monte_carlo.jeffreys_variance(events, trials)

    # no data reverts to the prior variance 1/8 (Beta(1/2, 1/2))
    assert np.isclose(variances[0], 1 / 8)

    # the general entry matches mean * (1 - mean) / (n + 2) with mean = (x + 1/2) / (n + 1)
    mean = (5 + 0.5) / (100 + 1)
    assert np.isclose(variances[2], mean * (1 - mean) / (100 + 2))


def test_error_rate_func_validation() -> None:
    """Inconsistent count arrays are rejected at construction."""

    def make(
        samples: list[int], failures: list[int], discards: list[int], min_error_weight: int = 1
    ) -> None:
        monte_carlo.ErrorRateFunc(
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

    # counts cannot be negative, on either the failure or the discard path
    with pytest.raises(ValueError, match="non-negative"):
        make([10], [-1], [0])
    with pytest.raises(ValueError, match="non-negative"):
        make([0, 10], [0, 0], [0, -1])

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
        monte_carlo.ErrorRateFunc(
            num_samples=np.array([0, 10]),
            num_failures=np.array([0, 1]),
            num_discards=np.array([0, 0]),
            num_error_locations=5,
            max_error_rate=1.5,
        )


def test_error_rate_rises_from_zero() -> None:
    """A reported rate, and the bound above it, both stay positive and grow with the error rate.

    The bound is the probability of an error heavier than the covered weights.  Taking it as one
    minus the probability of the weights below loses every significant digit once those weights hold
    nearly all of it, which is the ordinary case at a small physical error rate: the bound flattens
    onto zero and then goes negative, which no probability may do.  The counts below describe a
    decoder that corrects every error of weight at most four and fails on every heavier one.
    """
    num_samples = np.full(9, 10)
    num_samples[0] = 0
    func = monte_carlo.ErrorRateFunc(
        num_samples=num_samples,
        num_failures=np.where(np.arange(9) > 4, num_samples, 0),
        num_discards=np.zeros(9, dtype=int),
        num_error_locations=9,
        max_error_rate=0.05,
    )
    error_rates = np.logspace(-9, np.log10(0.05), 50)
    rates = np.asarray(func(error_rates)[0])
    assert np.all(rates > 0) and np.all(np.diff(rates) > 0)

    # the cancelling form reaches -2e-16 at the light end of this range, where the closed form
    # correctly reports 1e-81
    bounds = np.asarray(func.truncation_error_bound(error_rates))
    assert np.all(bounds > 0) and np.all(np.diff(bounds) > 0)

    # a covered range reaching the block length has no weight above it left to bound, including at
    # an error rate of one, where the closed form has an empty range to report
    counts = np.zeros(2, dtype=int)
    func = monte_carlo.ErrorRateFunc(counts, counts, counts, 1, 1.0)
    assert func(1.0)[0] == 0 and func.truncation_error_bound(1.0) == 0


def test_error_rate_func() -> None:
    """Convert raw failure and discard counts into error and discard rate estimates."""
    func = monte_carlo.ErrorRateFunc(
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

    # an iterable of physical error rates yields arrays of rates and uncertainties
    rates, uncertainties = func([0.0, 0.1])
    rates, uncertainties = np.asarray(rates), np.asarray(uncertainties)
    assert rates.shape == (2,) and uncertainties.shape == (2,)
    assert rates[0] == 0  # a zero physical error rate gives a zero logical error rate

    # physical error rates beyond the constructed range are rejected
    with pytest.raises(ValueError, match="does not cover"):
        func(0.9)

    # an iterable input yields one bound per error rate, rather than losing entries
    assert np.asarray(func.truncation_error_bound([0.1, 0.2])).shape == (2,)


def _expected_rate_and_error(
    func: monte_carlo.ErrorRateFunc, error_rate: float, *, discard_rate: bool = False
) -> tuple[float, float]:
    """Recompute a reported rate and uncertainty by an independent route.

    Everything here comes from scipy rather than from the module under test: the weight distribution
    from a binomial mass function, and each per-weight posterior variance from a beta distribution.
    Only the final propagation is spelled the same way, and spelling it separately is what pins it.
    """
    weights = np.arange(func.num_samples.size)
    weight_probs = scipy.stats.binom.pmf(weights, func.num_error_locations, error_rate)
    if discard_rate:
        events, trials = func.num_discards, func.num_samples
    else:
        events, trials = func.num_failures, func.num_samples - func.num_discards

    # a weight with no trials is recorded with a zero rate, but keeps the prior's variance
    safe_trials = np.where(trials > 0, trials, 1)
    rates = np.where(trials > 0, events / safe_trials, 0.0)
    variances = scipy.stats.beta(events + 0.5, trials - events + 0.5).var()
    variances = np.where(weights < func.min_error_weight, 0.0, variances)

    # weights above the covered range are all treated as failures, and only on that path
    truncation = 0.0
    if not discard_rate:
        truncation = float(
            scipy.stats.binom.sf(func.max_error_weight, func.num_error_locations, error_rate)
        )
    return float(weight_probs @ rates) + truncation, float(np.sqrt(weight_probs**2 @ variances))


def test_reported_uncertainty_is_the_propagated_posterior() -> None:
    """A reported rate and uncertainty match an independent computation of both.

    The counts below are chosen so that the two paths cannot be confused for one another: the
    infidelity is taken over kept samples while the discard rate is taken over all of them, so
    swapping either the rates or the variances between the paths changes both answers.  They also
    put a weight at each extreme -- one that always fails, one whose every sample is discarded and
    which therefore has no kept samples at all -- and leave weights above the covered range for the
    truncation term to act on.
    """
    cases = [
        monte_carlo.ErrorRateFunc(
            num_samples=np.array([0, 8, 5, 4]),
            num_failures=np.array([0, 3, 5, 0]),
            num_discards=np.array([0, 2, 0, 4]),
            num_error_locations=6,
            max_error_rate=0.5,
        ),
        # the same counts held to be perfectly decoded below weight two, so that the weights
        # min_error_weight excludes are the ones carrying the most probability
        monte_carlo.ErrorRateFunc(
            num_samples=np.array([0, 0, 8, 5, 4]),
            num_failures=np.array([0, 0, 3, 5, 0]),
            num_discards=np.array([0, 0, 2, 0, 4]),
            num_error_locations=7,
            max_error_rate=0.5,
            min_error_weight=2,
        ),
    ]
    error_rates = [0.05, 0.25, 0.5]
    for func in cases:
        for discards in [False, True]:
            for error_rate in error_rates:
                expected = _expected_rate_and_error(func, error_rate, discard_rate=discards)
                # atol=0 so that rtol is what gets enforced; its 1e-8 default would dominate here
                assert np.allclose(
                    func(error_rate, discard_rate=discards), expected, rtol=1e-12, atol=0
                )

        # an array argument gives the same numbers as the scalar calls, and carries discard_rate
        # through with it rather than silently reporting infidelities
        for discards in [False, True]:
            values, errors = func(error_rates, discard_rate=discards)
            expected_pairs = [
                _expected_rate_and_error(func, rate, discard_rate=discards) for rate in error_rates
            ]
            expected_values = [pair[0] for pair in expected_pairs]
            expected_errors = [pair[1] for pair in expected_pairs]
            assert np.allclose(np.asarray(values), expected_values, rtol=1e-12, atol=0)
            assert np.allclose(np.asarray(errors), expected_errors, rtol=1e-12, atol=0)

    # the two paths genuinely differ, so the checks above would catch them being exchanged
    func = cases[0]
    assert not np.allclose(func(0.25), func(0.25, discard_rate=True))
    assert not np.allclose(func.infidelity_variances, func.discard_rate_variances)


def test_error_rate_func_single_weight() -> None:
    """A degenerate func covering only the weight-0 bin evaluates without crashing."""
    func = monte_carlo.ErrorRateFunc(
        num_samples=np.array([0]),
        num_failures=np.array([0]),
        num_discards=np.array([0]),
        num_error_locations=5,
        max_error_rate=0.5,
    )
    assert func.max_error_weight == 0

    # every error of weight >= 1 lies outside the covered range, so nothing measured contributes and
    # the reported rate is the truncation term alone, with no uncertainty around it.  A discard
    # rate takes no such term, so it stays at zero
    truncation = func.truncation_error_bound(0.1)
    assert np.isclose(truncation, 1 - 0.9**5)
    assert func(0.1) == (truncation, 0)
    assert func(0.1, discard_rate=True) == (0, 0)


class _FixedObservableDecoder(decoders.ObservableDecoder):
    """Observable decoder that returns a fixed output, and records the syndromes it decodes."""

    def __init__(
        self,
        output: npt.ArrayLike,
        has_erasure_bit: bool = False,
        *,
        num_detectors: int = 1,
        num_observables: int | None = None,
    ) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit
        self.num_detectors = num_detectors
        self.num_observables = (
            len(self.output) - int(has_erasure_bit) if num_observables is None else num_observables
        )
        self.syndromes: list[npt.NDArray[np.int_]] = []

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        self.syndromes.append(syndrome)
        return self.output


class _FixedErrorDecoder(decoders.ErrorDecoder):
    """Error decoder that returns a fixed output."""

    def __init__(self, output: npt.ArrayLike, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output


class _BitPackedSinterDecoder(sinter.Decoder):
    """Sinter decoder whose compiled form has only the bit-packed methods that Sinter requires."""

    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.dems: list[stim.DetectorErrorModel] = []

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> Any:
        self.dems.append(dem)
        return _BitPackedCompiledDecoder(self.packed_prediction)


class _BitPackedCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, packed_prediction: npt.NDArray[np.uint8]) -> None:
        self.packed_prediction = packed_prediction
        self.packed_syndromes: list[npt.NDArray[np.uint8]] = []

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        self.packed_syndromes.append(bit_packed_detection_event_data)
        return self.packed_prediction


def test_get_code_capacity_dem() -> None:
    """A code-capacity DEM has one error mechanism per column of the syndrome matrix."""
    syndrome_matrix = galois.GF2([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    observable_matrix = galois.GF2([[1, 0, 0, 1]])
    dem = monte_carlo.get_code_capacity_dem(syndrome_matrix, observable_matrix)
    assert dem.num_detectors == 3  # an untriggered detector is still declared
    assert dem.num_observables == 1
    assert dem.num_errors == 4

    # each error mechanism flips the detectors and observables in its columns, with a probability
    # that does not depend on the physical error rate
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), syndrome_matrix)
    assert np.array_equal(dem_arrays.observable_flip_matrix.toarray(), observable_matrix)
    assert len(set(dem_arrays.error_probs)) == 1

    # error mechanisms may be combinations of error locations
    dem_errors = galois.GF2([[1, 0], [0, 1], [0, 1], [0, 1]])
    dem = monte_carlo.get_code_capacity_dem(syndrome_matrix, observable_matrix, dem_errors)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), syndrome_matrix @ dem_errors)
    assert np.array_equal(
        dem_arrays.observable_flip_matrix.toarray(), observable_matrix @ dem_errors
    )

    # symplectic X, Z, and Y mechanisms are constructed directly, with caller-supplied probabilities
    syndrome_matrix = galois.GF2([[1, 0, 0, 1]])
    observable_matrix = galois.GF2([[0, 1, 1, 0]])
    error_probs = np.arange(1, 7) / 100
    dem = monte_carlo.get_code_capacity_dem(
        syndrome_matrix,
        observable_matrix,
        symplectic_errors=True,
        error_probs=error_probs,
    )
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    assert np.array_equal(dem_arrays.detector_flip_matrix.toarray(), [[1, 0, 0, 1, 1, 1]])
    assert np.array_equal(dem_arrays.observable_flip_matrix.toarray(), [[0, 1, 1, 0, 1, 1]])
    assert np.array_equal(dem_arrays.error_probs, error_probs)
    with pytest.raises(ValueError, match="requires an observable_matrix"):
        monte_carlo.get_code_capacity_dem(syndrome_matrix, None, symplectic_errors=True)
    with pytest.raises(ValueError, match="cannot both be specified"):
        monte_carlo.get_code_capacity_dem(
            syndrome_matrix,
            observable_matrix,
            galois.GF2.Identity(4),
            symplectic_errors=True,
        )

    # Stim detector error models are binary
    field = galois.GF(3)
    with pytest.raises(ValueError, match="cannot decode a code over GF"):
        monte_carlo.get_code_capacity_dem(field(syndrome_matrix), field(observable_matrix))


def test_code_capacity_decoder_from_error_decoder() -> None:
    """An error decoder predicts the observable values of the errors that it infers."""
    field = galois.GF(3)
    syndrome_matrix = field([[1, 2, 0]])
    observable_matrix = field([[1, 1, 1], [0, 2, 1]])

    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([2, 1, 0])
    )
    assert not decoder.can_discard
    observables, erased = decoder.decode(field([1]))
    assert isinstance(observables, field) and not erased
    assert np.array_equal(observables, observable_matrix @ field([2, 1, 0]))

    # decoding fails if and only if the predicted observable values differ from the true ones
    assert decoder.get_failure_and_erasure(field([2, 1, 0])) == (False, False)
    assert decoder.get_failure_and_erasure(field([2, 1, 1])) == (True, False)
    assert decoder.get_failure_and_erasure(field([1, 0, 2])) == (False, False)  # same logical class

    # an erasure bit is passed through, and an erased sample is not a failure
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([0, 0, 0, 1], has_erasure_bit=True)
    )
    assert decoder.can_discard
    assert decoder.get_failure_and_erasure(field([2, 1, 1])) == (False, True)

    # malformed errors are rejected with actionable messages
    malformed_outputs: list[tuple[list[Any], str]] = [
        ([0, 0], r"inferred an error of shape \(2,\), but expected shape \(3,\)"),
        ([0, 0, 3], "not elements of GF"),
        ([0.0, 0.0, 0.0], "expected integers"),
    ]
    for output, message in malformed_outputs:
        decoder = monte_carlo.get_code_capacity_decoder(
            syndrome_matrix, observable_matrix, _FixedErrorDecoder(output)
        )
        with pytest.raises(ValueError, match=message):
            decoder.decode(field([0]))
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedErrorDecoder([0, 0, 0, 2], has_erasure_bit=True)
    )
    with pytest.raises(ValueError, match="erasure flags that are not 0 or 1"):
        decoder.decode(field([0]))

    # settings build an error decoder, even if it could predict observable flips natively
    decoder = monte_carlo.get_code_capacity_decoder(
        galois.GF2([[1, 1]]), galois.GF2([[1, 0]]), decoders.lookup_table(max_weight=1)
    )
    assert isinstance(decoder.decoder, monte_carlo._ErrorsToFieldObservablesDecoder)
    assert isinstance(decoder.decoder.error_decoder, decoders.LookupDecoder)

    # None represents an identity observable map without materializing a dense identity matrix
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, None, _FixedErrorDecoder([2, 1, 0])
    )
    assert decoder.observable_matrix is None and decoder.num_observables == 3
    assert decoder.reuse_for(syndrome_matrix, None) is decoder
    assert decoder.get_failure_and_erasure(field([2, 1, 0])) == (False, False)


def test_code_capacity_decoder_from_observable_decoder() -> None:
    """A prebuilt observable decoder is used as is, and its predictions are validated."""
    field = galois.GF(3)
    syndrome_matrix = field([[1, 2, 0]])
    observable_matrix = field([[1, 1, 1], [0, 2, 1]])

    observable_decoder = _FixedObservableDecoder([2, 1])
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, observable_decoder
    )
    assert decoder.decoder is observable_decoder
    error = field([1, 0, 1])  # its observable values are [2, 1]
    assert decoder.get_failure_and_erasure(error) == (False, False)
    assert np.array_equal(observable_decoder.syndromes[-1], syndrome_matrix @ error)
    assert type(observable_decoder.syndromes[-1]) is np.ndarray
    assert decoder.get_failure_and_erasure(field([1, 0, 0])) == (True, False)

    # an erasure flag discards a sample
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _FixedObservableDecoder([0, 0, 1], True)
    )
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(error) == (False, True)

    # a prediction of the wrong shape or field is rejected
    for output, message in [
        ([0, 0, 0], r"shape \(3,\), but expected shape \(2,\): 2 value\(s\)"),
        ([0, 5], "not elements of GF"),
    ]:
        decoder = monte_carlo.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder(output, num_observables=2),
        )
        with pytest.raises(ValueError, match=message):
            decoder.decode(field([0]))

    # a decoder that infers errors and predicts observable flips is used as an error decoder
    class _BothDecoder(_FixedErrorDecoder):
        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            raise NotImplementedError  # pragma: no cover

    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _BothDecoder([1, 0, 1])
    )
    assert isinstance(decoder.decoder, monte_carlo._ErrorsToFieldObservablesDecoder)
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # an object with both decode and decode_observables keeps its legacy error-decoder semantics
    class _DecodeOnlyBoth:
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.array([1, 0, 1])

        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            raise AssertionError("decode_observables should not be called")  # pragma: no cover

    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _DecodeOnlyBoth()
    )
    assert isinstance(decoder.decoder, monte_carlo._ErrorsToFieldObservablesDecoder)
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # the ObservableDecoder protocol does not require optional dimension metadata
    class _NoDimensions:
        def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.array([2, 1])

    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, _NoDimensions()
    )
    assert decoder.get_failure_and_erasure(error) == (False, False)

    # prebuilt observable decoders are rejected where a caller asks for that
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        monte_carlo.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder([0, 0]),
            prebuilt_rejection_reason="it is a test",
        )

    # deprecated keyword arguments cannot be combined with an observable decoder
    with pytest.raises(ValueError, match="Cannot combine decoder="):
        monte_carlo.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _FixedObservableDecoder([0, 0]),
            {"with_lookup": True},
            warn_deprecated=False,
        )


def test_code_capacity_decoder_from_sinter_decoder() -> None:
    """A Sinter-style decoder is compiled for the code-capacity DEM of its sector."""
    code = codes.RepetitionCode(3)
    observable_matrix = code.field([[1, 0, 0]])

    decoder = monte_carlo.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.SinterDecoder(decoder=decoders.mwpm())
    )
    assert isinstance(decoder.decoder, decoders.CompiledSinterDecoder)
    assert decoder.decoder.num_detectors == 2 and decoder.decoder.num_observables == 1
    for bit in range(3):
        error = code.field.Zeros(3)
        error[bit] = 1
        assert decoder.get_failure_and_erasure(error) == (False, False)

    # a trivial decoder fails whenever an error flips an observable
    decoder = monte_carlo.get_code_capacity_decoder(
        code.matrix, observable_matrix, decoders.TrivialDecoder()
    )
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (True, False)
    assert decoder.get_failure_and_erasure(code.field([0, 1, 0])) == (False, False)

    # an erasure-enabled inner decoder discards samples
    sinter_decoder = decoders.SinterDecoder(
        decoder=decoders.lookup_table(max_weight=0, add_erasure_bit=True)
    )
    decoder = monte_carlo.get_code_capacity_decoder(code.matrix, observable_matrix, sinter_decoder)
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (False, True)
    assert decoder.get_failure_and_erasure(code.field([0, 0, 0])) == (False, False)

    # a Sinter-style decoder is not rejected as prebuilt, since it is compiled here
    decoder = monte_carlo.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        decoders.TrivialDecoder(),
        prebuilt_rejection_reason="it is a test",
    )
    assert isinstance(decoder.decoder, decoders.CompiledTrivialDecoder)

    # but a compiled decoder is, and it must fit the sector that it decodes
    compiled_decoder = decoders.TrivialDecoder().compile_decoder_for_dem(
        monte_carlo.get_code_capacity_dem(code.matrix, observable_matrix)
    )
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        monte_carlo.get_code_capacity_decoder(
            code.matrix, observable_matrix, compiled_decoder, prebuilt_rejection_reason="test"
        )
    with pytest.raises(ValueError, match="has num_observables=1, but this code-capacity sector"):
        monte_carlo.get_code_capacity_decoder(code.matrix, code.field.Identity(3), compiled_decoder)
    with pytest.raises(ValueError, match="has num_detectors=2, but this code-capacity sector"):
        monte_carlo.get_code_capacity_decoder(
            code.field.Zeros((3, 3)), observable_matrix, compiled_decoder
        )

    # a typed observable constructor is built from the code-capacity detector error model
    def observable_constructor(dem: stim.DetectorErrorModel) -> decoders.ObservableDecoder:
        return decoders.ObservableLookupDecoder(dem, max_weight=1)

    decoder = monte_carlo.get_code_capacity_decoder(
        code.matrix, observable_matrix, observable_constructor
    )
    assert decoder.get_failure_and_erasure(code.field([1, 0, 0])) == (False, False)

    # relative mechanism weights are scaled by a fixed placeholder probability
    weighted_sinter_decoder = _BitPackedSinterDecoder(np.zeros((1, 1), dtype=np.uint8))
    monte_carlo.get_code_capacity_decoder(
        code.matrix,
        observable_matrix,
        weighted_sinter_decoder,
        dem_error_weights=np.array([1, 2, 3]),
    )
    weighted_dem_arrays = decoders.DetectorErrorModelArrays(
        weighted_sinter_decoder.dems[0], simplify=False
    )
    assert np.array_equal(
        weighted_dem_arrays.error_probs / weighted_dem_arrays.error_probs[0], [1, 2, 3]
    )

    # external compiled Sinter decoders are recognized as prebuilt
    external_compiled = _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="prebuilt decoder cannot be passed as decoder="):
        monte_carlo.get_code_capacity_decoder(
            code.matrix,
            observable_matrix,
            external_compiled,
            prebuilt_rejection_reason="it is a test",
        )

    # a Sinter-style decoder cannot decode a nonbinary code
    field = galois.GF(3)
    with pytest.raises(ValueError, match="cannot decode a code over GF"):
        monte_carlo.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), decoders.TrivialDecoder()
        )
    with pytest.raises(ValueError, match="is binary, so it cannot decode a code over GF"):
        monte_carlo.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), compiled_decoder
        )
    with pytest.raises(ValueError, match="is binary, so it cannot decode a code over GF"):
        monte_carlo.get_code_capacity_decoder(
            field(code.matrix),
            field(observable_matrix),
            _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8)),
        )

    binary_lookup = decoders.ObservableLookupDecoder(
        code.matrix,
        max_weight=1,
        observable_flip_matrix=observable_matrix,
        error_channel=[0.1] * len(code),
    )
    with pytest.raises(ValueError, match=r"built over GF\(2\).+sector is over GF\(3\)"):
        monte_carlo.get_code_capacity_decoder(
            field(code.matrix), field(observable_matrix), binary_lookup
        )


def test_code_capacity_decoder_from_bit_packed_sinter_decoder() -> None:
    """A Sinter decoder whose compiled form only decodes bit-packed data is supported."""
    syndrome_matrix = galois.GF2(np.eye(9, dtype=int))
    observable_matrix = galois.GF2(np.eye(9, dtype=int))
    error = galois.GF2([1, 0, 0, 0, 0, 0, 0, 0, 1])

    # a raw precompiled decoder needs dimension metadata before it can be used directly
    with pytest.raises(ValueError, match="does not declare num_detectors"):
        monte_carlo.get_code_capacity_decoder(
            syndrome_matrix,
            observable_matrix,
            _BitPackedCompiledDecoder(np.array([[1, 1]], dtype=np.uint8)),
        )

    # nine observables take two bytes, so the prediction [1, 0, ..., 0, 1] is packed as [1, 1]
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1, 1]], dtype=np.uint8))
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    assert sinter_decoder.dems[0].num_detectors == sinter_decoder.dems[0].num_observables == 9
    assert decoder.num_erasure_flags == 1
    assert decoder.get_failure_and_erasure(error) == (False, False)
    compiled_decoder = decoder.decoder.compiled_decoder  # type:ignore[attr-defined]
    assert np.array_equal(compiled_decoder.packed_syndromes[-1], [[1, 1]])
    assert decoder.get_failure_and_erasure(galois.GF2.Zeros(9)) == (True, False)

    # a nonzero byte added past the observables discards the sample
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1, 1, 1]], dtype=np.uint8))
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    assert decoder.get_failure_and_erasure(error) == (False, True)

    # a prediction of any other width is rejected
    sinter_decoder = _BitPackedSinterDecoder(np.array([[1]], dtype=np.uint8))
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observable_matrix, sinter_decoder
    )
    with pytest.raises(ValueError, match=r"shape \(1, 1\) for one shot"):
        decoder.get_failure_and_erasure(error)

    # a compiled decoder that can do neither is rejected
    class _NotADecoder(sinter.Decoder):
        def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> object:
            return object()

    with pytest.raises(TypeError, match="must provide a decode_observables or"):
        monte_carlo.get_code_capacity_decoder(syndrome_matrix, observable_matrix, _NotADecoder())


def test_code_capacity_decoder_reuse() -> None:
    """A decoder is reused for another sector only when it predicts the right observables."""
    syndrome_matrix = galois.GF2([[1, 1, 0], [0, 1, 1]])
    observables_a = galois.GF2([[1, 1, 1]])
    observables_b = galois.GF2([[1, 0, 0]])

    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observables_a, decoders.lookup_table(max_weight=1)
    )
    assert decoder.reuse_for(syndrome_matrix, observables_a) is decoder
    assert decoder.reuse_for(galois.GF2([[1, 1, 0]]), observables_a) is None

    # an error decoder is reused with other observables
    reused_decoder = decoder.reuse_for(syndrome_matrix, observables_b)
    assert reused_decoder is not None
    assert reused_decoder.decoder.error_decoder is decoder.decoder.error_decoder  # type:ignore
    assert reused_decoder.observable_matrix is not None
    assert np.array_equal(reused_decoder.observable_matrix, observables_b)

    # an observable decoder is not
    decoder = monte_carlo.get_code_capacity_decoder(
        syndrome_matrix, observables_a, decoders.TrivialDecoder()
    )
    assert decoder.reuse_for(syndrome_matrix, observables_b) is None


def test_is_prebuilt_observable_decoder() -> None:
    """Classify decoder inputs by explicit capabilities."""
    matrix = galois.GF2([[1, 1]])
    assert monte_carlo.is_prebuilt_observable_decoder(_FixedObservableDecoder([0]))
    assert monte_carlo.is_prebuilt_observable_decoder(
        _BitPackedCompiledDecoder(np.zeros((1, 1), dtype=np.uint8))
    )
    assert not monte_carlo.is_prebuilt_observable_decoder(_FixedErrorDecoder([0, 0]))
    assert not monte_carlo.is_prebuilt_observable_decoder(decoders.lookup_table(max_weight=1))
    assert not monte_carlo.is_prebuilt_observable_decoder(decoders.ObservableLookupDecoder)
    assert not monte_carlo.is_prebuilt_observable_decoder(decoders.TrivialDecoder())
    assert not monte_carlo.is_prebuilt_observable_decoder(decoders.relay_bp().build(matrix))
    assert monte_carlo.compiles_for_dem(decoders.TrivialDecoder())
    assert not monte_carlo.compiles_for_dem(decoders.TrivialDecoder)

    class _UnannotatedConstructor:
        def __call__(self, dem: stim.DetectorErrorModel) -> Any:
            return None  # pragma: no cover

    assert monte_carlo.constructs_observable_decoder(_FixedObservableDecoder)
    assert not monte_carlo.constructs_observable_decoder(lambda matrix: None)
    assert not monte_carlo.constructs_observable_decoder(_UnannotatedConstructor())
