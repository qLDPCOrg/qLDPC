# SPDX-License-Identifier: Apache-2.0

"""Tests for the basis families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

import qldpc

from . import _binary as binary
from . import _compilation, _information_sets
from .conftest import get_random_full_rank_matrix

# Two disjoint full blocks and a rank-two remainder.
PARTIAL_BASIS = np.hstack(
    [
        np.eye(3, dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
        np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
    ]
)
EVEN_BASIS = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)


def test_information_set_packing() -> None:
    """Information sets pack full blocks first and retain the residual rank."""
    get_family = _information_sets.get_disjoint_family
    assert get_family(PARTIAL_BASIS, np.eye(3, dtype=np.uint8)).ranks.tolist() == [3, 3, 2]
    assert not len(get_family(np.zeros((1, 3), dtype=np.uint8), None).ranks)

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert get_family(basis, None).ranks.tolist() == [3, 3]

    # Only one column carries the second row, so no column order packs a second full set.
    basis = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    assert get_family(basis, None).ranks.tolist() == [2, 1]


def test_a_basis_family_stacks_its_sets() -> None:
    """The sets of a basis family are reduced on their pivots, which are padded for partial sets."""
    labels = np.eye(3, dtype=np.uint8)
    basis_family = _information_sets.get_disjoint_family(PARTIAL_BASIS, labels)
    assert basis_family.generators.shape == (3, 3, PARTIAL_BASIS.shape[1])
    assert basis_family.labels.shape == (3, 3, 3)
    assert basis_family.certificate.dimension == 3
    for generators, set_labels, pivots, rank in zip(*basis_family[:4], strict=True):
        # The labels record the row operations that reduce the basis.
        np.testing.assert_array_equal(generators, set_labels @ PARTIAL_BASIS % 2)
        np.testing.assert_array_equal(generators[:rank, pivots[:rank]], np.eye(rank))
        assert np.all(pivots[rank:] == -1)
    no_labels = _information_sets.get_disjoint_family(PARTIAL_BASIS, None)
    assert no_labels.labels.shape == (3, 3, 0)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("basis", [PARTIAL_BASIS, EVEN_BASIS])
def test_family_certificates(basis: npt.NDArray[np.uint8]) -> None:
    """Rank-deficient blocks and inferred parity give certificates valid for every word."""
    # A target above every weight keeps the rank-deficient block.
    basis_family = _information_sets.get_disjoint_family(basis, None)
    basis_family = _information_sets.prune_family(basis_family, basis.shape[1] + 1)
    _assert_certificate_bounds_unvisited_words(basis, basis_family)


def test_weight_divisor_inference() -> None:
    """Even rows span only even weights, and a divisor proven by the caller is retained."""
    odd = np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    get_family = _information_sets.get_disjoint_family
    assert get_family(EVEN_BASIS, None).certificate.weight_divisor == 2
    assert get_family(odd, None).certificate.weight_divisor == 1
    assert get_family(odd, None, weight_divisor=2).certificate.weight_divisor == 2


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("ranks, upper_bound", [([10, 6, 2], 6), ([10, 6, 2], 12), ([6, 2], 6)])
def test_prune_family(ranks: list[int], upper_bound: int) -> None:
    """Pruning partial search bases does not increase estimated enumeration work."""
    no_columns = np.empty((len(ranks), 10, 0), dtype=np.uint8)
    pivots = np.full((len(ranks), 10), -1, dtype=np.int64)
    basis_family = _information_sets.BasisFamily(
        no_columns, no_columns, pivots, np.array(ranks, dtype=np.int64), 1, 1
    )
    pruned = _information_sets.prune_family(basis_family, upper_bound)
    assert len(pruned.ranks)
    assert pruned.certificate.work(upper_bound) <= basis_family.certificate.work(upper_bound)


@pytest.mark.usefixtures("compilation_mode")
def test_overlapping_family_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = binary.get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(basis, labels)
    # One full set certifies a target of one after its first layer, which no replacement improves.
    assert not len(_get_overlapping_family(basis, labels, disjoint, 1, minimum_work=0).ranks)
    basis_family = _get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0)
    assert np.all(basis_family.ranks == len(basis))
    loads = np.bincount(basis_family.pivots.ravel(), minlength=basis.shape[1])
    assert basis_family.max_coverage >= loads.max() > 1
    for generators, set_labels, pivots in zip(*basis_family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(len(basis)))
    _assert_certificate_bounds_unvisited_words(basis, basis_family)


@pytest.mark.usefixtures("compilation_mode")
def test_a_cheap_search_keeps_its_disjoint_basis_family() -> None:
    """No overlapping basis family is built if the work of the disjoint one is below the minimum."""
    basis = binary.get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(basis, labels)
    assert not len(_get_overlapping_family(basis, labels, disjoint, 3).ranks)
    assert len(_get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0).ranks)


@pytest.mark.usefixtures("compilation_mode")
def test_overlapping_family_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Complete or nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    complete = _information_sets.get_disjoint_family(basis, None)
    assert not len(_get_overlapping_family(basis, None, complete, 26).ranks)
    shortened = basis[:, :-1]
    disjoint = _information_sets.get_disjoint_family(shortened, None)
    assert not len(_get_overlapping_family(shortened, None, disjoint, 26).ranks)

    # A zero basis has no information sets to replace.
    zeros = np.zeros((1, 3), dtype=np.uint8)
    empty = _information_sets.get_disjoint_family(zeros, None)
    assert not len(empty.ranks)
    assert not len(_get_overlapping_family(zeros, None, empty, 2).ranks)

    # Only three pivot pairs exist, so later attempts rediscover them.  The search must terminate
    # without a replacement that beats the disjoint family.
    small = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(small, None)
    assert not len(_get_overlapping_family(small, None, disjoint, 2, minimum_work=0).ranks)


@pytest.mark.usefixtures("compilation_mode")
def test_certificate_follows_its_definition(pytestconfig: pytest.Config) -> None:
    """The lower bound and work estimate equal their definitions, evaluated directly."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(30):
        dimension = int(rng.integers(1, 13))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 6)))
        weight_divisor, max_coverage = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        certificate = _information_sets.Certificate(dimension, ranks, weight_divisor, max_coverage)

        oracle = [
            _get_lower_bound_from_definition(dimension, ranks, weight_divisor, max_coverage, weight)
            for weight in range(dimension + 1)
        ]
        assert [certificate.lower_bound(weight) for weight in range(dimension + 1)] == oracle
        for upper_bound in range(1, 2 * dimension + 2):
            candidates = 0
            for weight in range(1, dimension + 1):
                candidates += len(ranks) * math.comb(dimension, weight)
                if oracle[weight] >= upper_bound:
                    break
            assert certificate.work(upper_bound) == candidates


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("dimension, num_sets", [(300, 16), (60, 8192)])
def test_work_estimates_saturate(dimension: int, num_sets: int) -> None:
    """Work estimates for huge searches stay large and positive, rather than overflowing."""
    certificate = _information_sets.Certificate(dimension, ranks=(dimension,) * num_sets)
    assert 2**50 < certificate.work(upper_bound=10**6) < 2**63


def test_binomials_are_exact_where_they_fit_and_cannot_be_modified() -> None:
    binomials = _information_sets.get_capped_binomials(70)
    assert not binomials.flags.writeable
    assert binomials[:5].tolist() == [math.comb(70, weight) for weight in range(5)]
    assert 0 < binomials[35] < math.comb(70, 35)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets, num_sets", [(1, 0), (3, 0), (4, 4), (16, 4)])
def test_the_size_limit_bounds_the_overlapping_basis_family(max_sets: int, num_sets: int) -> None:
    """A cheaper basis family of overlapping sets needs four sets, so a lower limit finds none."""
    basis = binary.get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(basis, labels)
    overlapping = _get_overlapping_family(basis, labels, disjoint, 3, max_sets, minimum_work=0)
    assert len(overlapping.ranks) == num_sets


def test_functions_agree_in_plain_python_and_compiled(pytestconfig: pytest.Config) -> None:
    """Pruned and replacement basis families are identical, bit for bit, plain and compiled."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The Hamming code has an overlapping basis family that does less work.  Random bases rarely do.
    hamming = binary.get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    cases: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, int]] = [
        (hamming, np.eye(len(hamming), dtype=np.uint8), 3)
    ]
    for trial in range(12):
        rows = int(rng.integers(3, 10))
        basis = get_random_full_rank_matrix(rng, rows, int(rng.integers(2 * rows, 5 * rows)))
        labels = np.eye(rows, dtype=np.uint8) if trial % 2 else None
        cases.append((basis, labels, int(rng.integers(2, rows + 2))))
    for case, (basis, labels, upper_bound) in enumerate(cases):
        results = []
        for compiled in (False, True):
            _compilation.set_on_request_functions_compiled(compiled)
            try:
                disjoint = _information_sets.get_disjoint_family(basis, labels)
                pruned = _information_sets.prune_family(disjoint, upper_bound)
                replacement = _get_overlapping_family(
                    basis, labels, pruned, upper_bound, minimum_work=0
                )
            finally:
                _compilation.set_on_request_functions_compiled(False)
            results.append((pruned, replacement))
        assert case or len(results[0][1].ranks)
        for plain, compiled_result in zip(*results, strict=True):
            for field, value in zip(plain._fields, plain, strict=True):
                np.testing.assert_array_equal(value, getattr(compiled_result, field), field)


def _get_lower_bound_from_definition(
    dimension: int, ranks: tuple[int, ...], weight_divisor: int, max_coverage: int, weight: int
) -> int:
    """The certificate lower bound after the given completed weight, from its definition."""
    occupancy = sum(max(0, weight + 1 - (dimension - rank)) for rank in ranks)
    bound = -(-occupancy // max_coverage)
    return bound + (-bound) % weight_divisor


def _get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    basis_family: _information_sets.BasisFamily,
    upper_bound: int,
    max_sets: int = _information_sets.MAX_SETS,
    *,
    minimum_work: int = _information_sets.MINIMUM_WORK,
) -> _information_sets.BasisFamily:
    """Stage three, with the capped binomials and a fixed seed, as the searches call it."""
    return _information_sets.get_overlapping_family(
        basis,
        np.empty((len(basis), 0), dtype=np.uint8) if labels is None else labels,
        basis_family,
        upper_bound,
        max_sets,
        minimum_work,
        _information_sets.get_capped_binomials(len(basis)),
        np.random.default_rng(0),
    )


def _assert_certificate_bounds_unvisited_words(
    basis: npt.NDArray[np.uint8], family: _information_sets.BasisFamily
) -> None:
    """Check a basis family's certificate against every word that its search bases span.

    A search basis visits a word after layer ``t`` if the word combines at most ``t`` of its rows.
    """

    def get_coefficient_weights(generators: npt.NDArray[np.uint8]) -> dict[bytes, int]:
        return {
            (np.asarray(bits) @ generators % 2).astype(np.uint8).tobytes(): sum(bits)
            for bits in itertools.product((0, 1), repeat=len(generators))
        }

    words = get_coefficient_weights(basis)
    coefficient_weights = [get_coefficient_weights(generators) for generators in family.generators]
    certificate = family.certificate
    # Every search basis spans the same space as the original basis.
    assert all(weights.keys() == words.keys() for weights in coefficient_weights)
    weights = {word: np.count_nonzero(np.frombuffer(word, dtype=np.uint8)) for word in words}
    for word, weight in weights.items():
        assert weight % certificate.weight_divisor == 0
        for completed in range(len(basis)):
            if all(item[word] > completed for item in coefficient_weights):
                assert weight >= certificate.lower_bound(completed)

    # The certificate is not vacuous: before the last layer, it reaches the minimum nonzero weight.
    assert certificate.lower_bound(len(basis) - 1) >= min(filter(None, weights.values()))
