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


def test_get_sample_allocation() -> None:
    """Allocation of samples across error weights."""
    allocation = _monte_carlo._get_sample_allocation(1000, block_length=10, max_error_rate=0.2)
    assert allocation[0] == 1  # exactly one sample is reserved for the weight-0 error
    assert np.sum(allocation) >= 1000  # every requested sample is allocated
    assert allocation[-1] > 0  # trailing zeros are truncated


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

    def make(samples: list[int], failures: list[int], discards: list[int]) -> None:
        _monte_carlo.ErrorRateFunc(
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
    # every sampled weight (>0) carries positive variance despite zero observed failures, while
    # the deterministic weight-0 case remains exempt
    assert func.infidelity_variances[0] == 0
    assert np.all(func.infidelity_variances[1:] > 0)

    # so the aggregate error bar is positive at a physical error rate that weights those bins
    _, uncertainty = func(0.1)
    assert uncertainty > 0


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
