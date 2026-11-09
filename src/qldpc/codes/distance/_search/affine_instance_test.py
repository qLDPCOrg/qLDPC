# SPDX-License-Identifier: Apache-2.0

"""Tests for the instances of the affine BZ search."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import unpack_bits

from ..conftest import get_random_full_rank_matrix
from . import affine_instance, information_sets

_MAX = np.iinfo(np.int64).max


@pytest.mark.usefixtures("compilation_mode")
def test_coset_bounds_hold_for_every_unvisited_word(pytestconfig: pytest.Config) -> None:
    """Every word of a coset that a step leaves unvisited weighs at least the bound of that step.

    The codes are random, so some of their disjoint information sets are rank deficient, and the
    words of each coset are enumerated through every set (see ``_get_unvisited_minima``).
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(40):
        num_rows = int(rng.integers(1, 6))
        generators = get_random_full_rank_matrix(rng, num_rows, int(rng.integers(num_rows, 11)))
        instance = affine_instance.get_coset_instance(
            generators, random_column_orders=0, weight_factor=1
        )
        offset = np.asarray(rng.integers(2, size=generators.shape[1]), dtype=np.uint8)
        bounds = instance.get_weight_bounds()
        for step, minimum in _get_unvisited_minima(instance, generators, offset).items():
            assert minimum >= bounds[step]


def test_coset_bounds_take_no_divisor_from_an_even_code() -> None:
    """The cosets of an even code can have odd weight, so their bounds are not rounded to even.

    The code is spanned by ``110000``, ``001100`` and ``000011``, and every word of the coset of
    ``101010`` has weight three.  Once the first set of pivots ``0, 2, 4`` has completed layer one
    and the other set layer zero, the words with two ones on the first set remain, and the bound of
    that step is exactly three.  A bound rounded to the even divisor of the code would be four.
    """
    generators = np.kron(np.eye(3), np.ones((1, 2))).astype(np.uint8)
    family = information_sets.get_disjoint_family(generators, None, random_column_orders=0)
    assert family.weight_divisor == 2
    instance = affine_instance.get_coset_instance(
        generators, random_column_orders=0, weight_factor=1
    )
    offset = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
    minima = _get_unvisited_minima(instance, generators, offset)
    assert minima[1, 1] == instance.get_weight_bounds()[1, 1] == 3


def test_weight_bounds_follow_the_steps(pytestconfig: pytest.Config) -> None:
    """The weight bounds round the certificate up, and do not decrease from step to step.

    Column zero of each row repeats the last column of the row before, and the bounds after the last
    layer are the largest native integer, for sets of every rank, including zero, and both factors.
    Each call returns a table of its own.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(60):
        dimension = int(rng.integers(0, 7))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 5)))
        certificate = information_sets.Certificate(
            dimension, ranks, int(rng.integers(1, 3)), int(rng.integers(1, 3))
        )
        for factor in (1, 2):
            instance = _get_instance(certificate, factor)
            bounds = instance.get_weight_bounds()
            assert bounds.shape == (dimension + 1, len(ranks) + 1)
            for layer, num_completed in itertools.product(
                range(dimension + 1), range(len(ranks) + 1)
            ):
                expected = (
                    _MAX
                    if layer == dimension and num_completed
                    else -(-certificate.step_lower_bound(layer, num_completed) // factor)
                )
                assert bounds[layer, num_completed] == expected
            steps = bounds[:, 1:].ravel()
            assert np.all(steps[1:] >= steps[:-1])
            assert np.array_equal(bounds[1:, 0], bounds[:-1, -1])
            expected_bounds = bounds.copy()
            bounds[:] = -1
            assert np.array_equal(instance.get_weight_bounds(), expected_bounds)


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
        assert not instance.generators[index, :, width:].any()
        assert instance.label_widths[index] == 1
        assert np.array_equal(unpack_bits(instance.labels[index], 4), set_labels)


def test_coset_instances_keep_every_column(pytestconfig: pytest.Config) -> None:
    """A coset instance packs the full rows of the disjoint sets, and the empty code has one set.

    The rows of a rank-one code span one column each, so its table of bounds grows with the number
    of sets, not with its square.
    """
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

    empty = affine_instance.get_coset_instance(
        np.zeros((0, 70), dtype=np.uint8), random_column_orders=0, weight_factor=2
    )
    assert empty.generators.shape == (1, 0, 2)
    assert np.array_equal(empty.widths, [2])
    assert empty.pivots.shape == (1, 0)
    assert np.array_equal(empty.ranks, [0])
    assert np.array_equal(empty.get_weight_bounds(), [[0, _MAX]])

    rank_one = affine_instance.get_coset_instance(
        np.ones((1, 256), dtype=np.uint8), random_column_orders=0, weight_factor=1
    )
    assert np.array_equal(rank_one.ranks, [1] * 256)
    assert rank_one.get_weight_bounds().shape == (2, 257)


def _get_instance(
    certificate: information_sets.Certificate, factor: int
) -> affine_instance.AffineInstance:
    """An instance with a given certificate and weight factor, whose sets have no columns."""
    num_sets, dimension = len(certificate.ranks), certificate.dimension
    return affine_instance.AffineInstance(
        np.zeros((num_sets, dimension, 0), dtype=np.uint64),
        np.zeros((num_sets, dimension, 0), dtype=np.uint64),
        np.zeros(num_sets, dtype=np.int64),
        np.zeros(num_sets, dtype=np.int64),
        np.full((num_sets, dimension), -1, dtype=np.int64),
        np.array(certificate.ranks, dtype=np.int64),
        certificate,
        factor,
    )


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
