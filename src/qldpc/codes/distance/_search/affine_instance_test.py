# SPDX-License-Identifier: Apache-2.0

"""Tests for the instances of the affine BZ search."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary, unpack_bits

from ..conftest import get_random_full_rank_matrix
from . import affine_instance, information_sets


def test_coset_bounds_hold_for_every_unvisited_word(pytestconfig: pytest.Config) -> None:
    """Every word of a coset that a step leaves unvisited weighs at least the bound of that step.

    The codes are random, so some of their disjoint information sets are rank deficient, and the
    words of each coset are enumerated through every set (see ``_get_unvisited_minima``).  The
    instances hold the disjoint information sets, or random information sets, which may overlap, and
    whose certificate divides by their coverage.  A word weighs at least its Hamming weight divided
    by the weight factor, which is one or two.  About half of the codes get a last column with the
    parity of each row, which makes them even, and the bounds of the cosets of even codes are
    rounded to the parity of each coset.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(40):
        num_rows = int(rng.integers(1, 6))
        generators = get_random_full_rank_matrix(rng, num_rows, int(rng.integers(num_rows, 11)))
        if rng.integers(2):
            generators = np.hstack([generators, generators.sum(axis=1, keepdims=True) % 2])
        factor = int(rng.integers(1, 3))
        num_sets = int(rng.integers(2, 5))
        sets = [
            row_reduce_binary(generators, None, columns=rng.permutation(generators.shape[1]))
            for _ in range(num_sets)
        ]
        pivots = np.array([found for *_, found in sets], dtype=np.int64)
        overlapping = information_sets.BasisFamily(
            np.array([rows for rows, *_ in sets]),
            np.zeros((num_sets, num_rows, 0), dtype=np.uint8),
            pivots,
            np.full(num_sets, num_rows, dtype=np.int64),
            1,
            int(np.bincount(pivots.ravel()).max()),
        )
        instances = [
            affine_instance.get_coset_instance(
                generators, random_column_orders=0, weight_factor=factor
            ),
            affine_instance.get_coset_family_instance(overlapping, weight_factor=factor),
        ]
        offset = np.asarray(rng.integers(2, size=generators.shape[1]), dtype=np.uint8)
        even = not np.any(generators.sum(axis=1) % 2)
        for instance in instances:
            bounds = instance.get_weight_bounds(int(offset.sum()) % 2 if even else None)
            for step, minimum in _get_unvisited_minima(instance, generators, offset).items():
                assert -(-minimum // factor) >= bounds[step]


def test_coset_bounds_round_up_to_the_parity_of_each_coset() -> None:
    """The bounds of the cosets of an even code round up to the parity of each coset, not to even.

    The code is spanned by ``110000``, ``001100``, and ``000011``, so the words of the code itself
    have even weight, and those of the coset of ``100000`` odd weight.  The bound of each step
    equals the least weight of the words of the coset that it leaves unvisited.  Once the first set
    of pivots ``0, 2, 4`` has completed layer one and the other set layer zero, the bound without
    parity would be three.  The least weight of the unvisited words is four in the code and three
    in the coset of ``100000``.  A bound rounded to the even divisor of the code would be four for
    both.  The entry before the first step is not the bound of a step, and is left out.
    """
    generators = np.kron(np.eye(3), np.ones((1, 2))).astype(np.uint8)
    instance = affine_instance.get_coset_instance(
        generators, random_column_orders=0, weight_factor=1
    )
    for parity, offset in enumerate(([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0])):
        bounds = instance.get_weight_bounds(parity)
        minima = _get_unvisited_minima(instance, generators, np.array(offset, dtype=np.uint8))
        assert minima[1, 1] == 4 - parity
        assert all(bounds[step] == minimum for step, minimum in minima.items() if step != (0, 0))


def test_distance_instances_delete_pivot_columns(pytestconfig: pytest.Config) -> None:
    """A distance instance packs each set without its pivot columns, and keeps the certificate."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    basis = get_random_full_rank_matrix(rng, 4, 70)
    labels = np.eye(4, dtype=np.uint8)
    family = information_sets.get_disjoint_family(basis, labels, random_column_orders=0)
    certificate = family.get_search_certificate(3, 0, 1)
    instance = affine_instance.get_distance_instance(family, certificate)
    assert instance.certificate is certificate
    assert instance.weight_factor == 1
    assert np.array_equal(instance.ranks, family.ranks)
    assert np.array_equal(instance.pivots, family.pivots)
    for index, (rows, set_labels, pivots, rank) in enumerate(
        zip(family.generators, family.labels, family.pivots, family.ranks, strict=True)
    ):
        kept = np.delete(rows, pivots[:rank], axis=1)
        width = instance.widths[index]
        assert width == (kept.shape[1] + 63) // 64
        assert np.array_equal(
            unpack_bits(instance.generators[index, :, :width], kept.shape[1]), kept
        )
        assert instance.label_widths[index] == 1
        assert np.array_equal(unpack_bits(instance.labels[index], 4), set_labels)


def test_coset_instances_keep_every_column(pytestconfig: pytest.Config) -> None:
    """A coset instance packs the full rows of its sets, and the empty code has one set."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    generators = get_random_full_rank_matrix(rng, 5, 70)
    family = information_sets.get_disjoint_family(generators, None, random_column_orders=0)
    instance = affine_instance.get_coset_instance(
        generators, random_column_orders=0, weight_factor=2
    )
    assert instance.weight_factor == 2
    assert np.array_equal(instance.ranks, family.ranks)
    assert np.array_equal(instance.pivots, family.pivots)
    assert instance.certificate == information_sets.Certificate(5, tuple(family.ranks.tolist()))
    assert np.array_equal(instance.widths, [2] * len(family.ranks))
    for rows, packed in zip(family.generators, instance.generators, strict=True):
        assert np.array_equal(unpack_bits(packed, 70), rows)
    assert instance.labels.shape == (len(family.ranks), 5, 0)
    assert not instance.label_widths.any()

    # An instance of two information sets that share a pivot column keeps their coverage of two.
    first = family.pivots[0]
    order = np.concatenate([first[:1], np.setdiff1d(np.arange(70), first)])
    reduced, _, second = row_reduce_binary(generators, None, columns=order)
    overlapping = information_sets.BasisFamily(
        np.stack([family.generators[0], reduced]),
        np.zeros((2, 5, 0), dtype=np.uint8),
        np.stack([first, second]),
        np.full(2, 5, dtype=np.int64),
        1,
        2,
    )
    instance = affine_instance.get_coset_family_instance(overlapping, weight_factor=1)
    assert np.array_equal(instance.pivots, overlapping.pivots)
    assert instance.certificate == information_sets.Certificate(5, (5, 5), max_coverage=2)
    for rows, packed in zip(overlapping.generators, instance.generators, strict=True):
        assert np.array_equal(unpack_bits(packed, 70), rows)

    empty = affine_instance.get_coset_instance(
        np.zeros((0, 70), dtype=np.uint8), random_column_orders=0, weight_factor=2
    )
    assert empty.generators.shape == (1, 0, 2)
    assert np.array_equal(empty.widths, [2])
    assert empty.pivots.shape == (1, 0)
    assert np.array_equal(empty.ranks, [0])
    assert np.array_equal(empty.get_weight_bounds(), [[0, information_sets.NO_WORD_BOUND]])


def _get_unvisited_minima(
    instance: affine_instance.AffineInstance,
    generators: npt.NDArray[np.uint8],
    offset: npt.NDArray[np.uint8],
) -> dict[tuple[int, int], int]:
    """The least Hamming weight of the words of a coset that each step leaves unvisited.

    Each set reduces ``offset`` on its pivots to its own offset, and a word of the coset is in layer
    ``w`` of the set if it adds ``w`` of the set's rows to that offset.  After step ``(w, j)``, a
    word is unvisited if each of the first ``j`` sets has it in a layer above ``w``, and each other
    set in a layer above ``w - 1``.  Steps that leave no word unvisited are omitted.
    """
    num_rows, num_columns = generators.shape
    layers: dict[bytes, list[int]] = {}
    for rows, pivots, rank in zip(instance.generators, instance.pivots, instance.ranks):
        rows = unpack_bits(rows, num_columns)
        set_offset = offset.copy()
        for row, pivot in zip(rows, pivots[:rank]):
            if set_offset[pivot]:
                set_offset ^= row
        assert not set_offset[pivots[:rank]].any()
        for coefficients in itertools.product((0, 1), repeat=num_rows):
            word = (set_offset + np.array(coefficients) @ rows) % 2
            layers.setdefault(word.astype(np.uint8).tobytes(), []).append(sum(coefficients))
    minima: dict[tuple[int, int], int] = {}
    num_sets = len(instance.ranks)
    for key, word_layers in layers.items():
        assert len(word_layers) == num_sets
        weight = sum(key)
        for layer, num_completed in itertools.product(range(num_rows + 1), range(num_sets + 1)):
            completed = [layer if index < num_completed else layer - 1 for index in range(num_sets)]
            if all(item > done for item, done in zip(word_layers, completed, strict=True)):
                step = (layer, num_completed)
                minima[step] = min(minima.get(step, weight), weight)
    return minima
