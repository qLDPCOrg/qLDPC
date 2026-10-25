# SPDX-License-Identifier: Apache-2.0

"""Tests for the information-set families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

import qldpc

from . import _binary as binary
from . import _information_sets

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
    information_sets = _information_sets.get_information_set_generators(
        PARTIAL_BASIS, np.eye(3, dtype=np.uint8)
    )
    assert [rank for _, _, rank, _ in information_sets] == [3, 3, 2]
    assert not _information_sets.get_information_set_generators(
        np.zeros((1, 3), dtype=np.uint8), None
    )

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    information_sets = _information_sets.get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [3, 3]

    # Only one column carries the second row, so no column order packs a second full set.
    basis = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    information_sets = _information_sets.get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [2, 1]


@pytest.mark.parametrize("basis", [PARTIAL_BASIS, EVEN_BASIS])
def test_family_certificates(basis: npt.NDArray[np.uint8]) -> None:
    """Rank-deficient blocks and inferred parity give certificates valid for every word."""
    # A target above every weight keeps the rank-deficient block.
    family = _information_sets.prepare_family(basis, None, upper_bound=basis.shape[1] + 1)
    _assert_certificate_bounds_unvisited_words(basis, family)


def test_weight_divisor_inference() -> None:
    """Even rows span only even weights, and a divisor proven by the caller is retained."""
    odd = np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    get_family = _information_sets.get_disjoint_family
    assert get_family(EVEN_BASIS, None).certificate.weight_divisor == 2
    assert get_family(odd, None).certificate.weight_divisor == 1
    assert get_family(odd, None, weight_divisor=2).certificate.weight_divisor == 2


@pytest.mark.parametrize("ranks, upper_bound", [([10, 6, 2], 6), ([10, 6, 2], 12), ([6, 2], 6)])
def test_prune_family(ranks: list[int], upper_bound: int) -> None:
    """Pruning partial search bases does not increase estimated enumeration work."""
    generators = np.empty((10, 0), dtype=np.uint8)
    pivots = np.empty(0, dtype=int)
    information_sets = tuple(
        _information_sets.InformationSet(generators, None, rank, pivots) for rank in ranks
    )
    family = _information_sets.BasisFamily(
        information_sets, _information_sets.Certificate(10, tuple(ranks))
    )
    pruned = _information_sets.prune_family(family, upper_bound=upper_bound)
    assert pruned.information_sets
    assert pruned.certificate.work(upper_bound) <= family.certificate.work(upper_bound)


def test_overlapping_family_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = binary.get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(basis, labels)
    # One full set certifies a target of one after its first layer, which no replacement improves.
    assert (
        _information_sets.get_overlapping_family(
            basis, labels, disjoint, upper_bound=1, minimum_work=0
        )
        is None
    )
    family = _information_sets.get_overlapping_family(
        basis, labels, disjoint, upper_bound=3, minimum_work=0
    )
    assert family is not None
    loads = np.bincount(
        np.concatenate([item.pivots for item in family.information_sets]),
        minlength=basis.shape[1],
    )
    assert family.certificate.max_coverage >= loads.max() > 1
    for item in family.information_sets:
        assert item.labels is not None
        np.testing.assert_array_equal(item.generators, item.labels @ basis % 2)
        np.testing.assert_array_equal(item.generators[:, item.pivots], np.eye(len(basis)))
    _assert_certificate_bounds_unvisited_words(basis, family)


def test_overlapping_family_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Complete or nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    complete = _information_sets.get_disjoint_family(basis, None)
    assert _information_sets.get_overlapping_family(basis, None, complete, upper_bound=26) is None
    shortened = basis[:, :-1]
    disjoint = _information_sets.get_disjoint_family(shortened, None)
    assert (
        _information_sets.get_overlapping_family(shortened, None, disjoint, upper_bound=26) is None
    )

    # Only three pivot pairs exist, so later attempts rediscover them.  The search must terminate
    # without a replacement that beats the disjoint family.
    small = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    disjoint = _information_sets.get_disjoint_family(small, None)
    assert (
        _information_sets.get_overlapping_family(
            small, None, disjoint, upper_bound=2, minimum_work=0
        )
        is None
    )


def _assert_certificate_bounds_unvisited_words(
    basis: npt.NDArray[np.uint8], family: _information_sets.BasisFamily
) -> None:
    """Check a family's certificate against every word that its search bases span.

    A search basis visits a word after layer ``t`` if the word combines at most ``t`` of its rows.
    """

    def get_coefficient_weights(generators: npt.NDArray[np.uint8]) -> dict[bytes, int]:
        return {
            (np.asarray(bits) @ generators % 2).astype(np.uint8).tobytes(): sum(bits)
            for bits in itertools.product((0, 1), repeat=len(generators))
        }

    words = get_coefficient_weights(basis)
    coefficient_weights = [
        get_coefficient_weights(item.generators) for item in family.information_sets
    ]
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
