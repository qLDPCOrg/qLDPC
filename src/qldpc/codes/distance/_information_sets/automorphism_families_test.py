# SPDX-License-Identifier: Apache-2.0

"""Unit tests for automorphism families of information sets."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

from qldpc import codes
from qldpc._binary import get_rank_packed, pack_bits
from qldpc.objects import Pauli

from .._binary import get_eligible_rows
from ..conftest import get_sector_problem
from . import automorphism_families, certificates


def test_automorphisms_must_map_both_spaces_onto_themselves() -> None:
    """Only permutations that preserve the searched space and its excluded subspace pass."""
    code = codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    basis, labels = get_sector_problem(code, Pauli.X)
    shift = np.concatenate([np.roll(np.arange(7), 1), np.roll(np.arange(7), 1) + 7])
    automorphism_families.verify_automorphisms(basis, labels, shift[None])
    automorphism_families.verify_automorphisms(basis, labels, np.empty((0, 14), dtype=int))

    transposition = np.arange(14)
    transposition[[0, 1]] = [1, 0]
    not_permutations = [np.zeros((1, 14), dtype=int), np.arange(13)[None], np.arange(14)]
    for automorphisms in [np.stack([shift, transposition]), *not_permutations]:
        with pytest.raises(ValueError):
            automorphism_families.verify_automorphisms(basis, labels, automorphisms)

    # Swapping the last two columns preserves the excluded subspace, spanned by 100, but not the
    # searched space; swapping the first two preserves the searched space, but not the excluded one.
    basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    for swap in ([0, 2, 1], [1, 0, 2]):
        with pytest.raises(ValueError, match="not an automorphism"):
            automorphism_families.verify_automorphisms(basis, labels, np.array([swap]))


def test_automorphism_families_certify_like_the_family_of_all_images() -> None:
    """The certificate is that of the images of the scanned set under every group element.

    Each image is an information set, and the coverage of the family is counted directly.  The
    groups include one that acts on each block with several orbits.
    """
    rng = np.random.default_rng(0)
    for code in _get_random_mb_codes(rng, 12):
        translations = code._get_qudit_automorphisms()
        for automorphisms in (translations, translations[:1]):
            elements = _get_group_elements(automorphisms)
            for pauli in (Pauli.X, Pauli.Z):
                basis, labels = get_sector_problem(code, pauli)
                divisor = 2 if np.all(basis.sum(axis=1) % 2 == 0) else 1
                family, certificate = automorphism_families.get_automorphism_family(
                    basis, labels, automorphisms, divisor
                )
                [generators], [set_labels], [pivots] = (
                    family.generators,
                    family.labels,
                    family.pivots,
                )
                dimension = len(basis)
                assert family.ranks.tolist() == [dimension]

                # The scanned set is systematic on its pivots, spans the searched space, and labels
                # exactly the rows outside the excluded subspace as eligible.
                np.testing.assert_array_equal(
                    generators[:, pivots], np.eye(dimension, dtype=np.uint8)
                )
                assert _get_rank(np.vstack([basis, generators])) == dimension
                excluded = basis[~get_eligible_rows(labels)]
                for row, eligible in zip(generators, get_eligible_rows(set_labels)):
                    assert eligible == (_get_rank(np.vstack([excluded, row])) > len(excluded))

                coverage = np.zeros(basis.shape[1], dtype=int)
                for element in elements:
                    assert _get_rank(basis[:, element[pivots]]) == dimension
                    coverage[element[pivots]] += 1
                for weight in range(1, dimension):
                    expected = math.ceil(len(elements) * (weight + 1) / coverage.max())
                    expected += -expected % divisor
                    assert certificate.lower_bound(weight) == expected


def test_scanned_sets_balance_two_orbits() -> None:
    """For two orbits, the scanned set has the fewest pivots in its fuller orbit that any can have.

    An information set has at most ``rank(O)`` pivots in an orbit ``O``, and so at least ``r -
    rank(O')`` in the other orbit ``O'``, and every count between these bounds is attained.
    """
    rng = np.random.default_rng(1)
    for code in _get_random_mb_codes(rng, 12):
        for pauli in (Pauli.X, Pauli.Z):
            basis, labels = get_sector_problem(code, pauli)
            dimension, num_columns = basis.shape
            family, _ = automorphism_families.get_automorphism_family(
                basis, labels, code._get_qudit_automorphisms(), 1
            )
            blocks = (np.arange(num_columns // 2), np.arange(num_columns // 2, num_columns))
            counts = [np.isin(family.pivots[0], block).sum() for block in blocks]
            ranks = [_get_rank(basis[:, block]) for block in blocks]
            assert max(counts) == max((dimension + 1) // 2, dimension - min(ranks))


def test_automorphism_family_work_scans_one_set_per_layer() -> None:
    """The work estimate counts each layer of one set, through the layer that certifies a bound.

    This certificate of three images of an information set of a five-row basis, with coverage two,
    bounds unvisited words by ``ceil(3 (t + 1) / 2)`` once every image has completed layer ``t``.
    """
    certificate = certificates.Certificate(5, (5,) * 3, max_coverage=2)
    # layer one certifies weight three; layer two certifies five; no layer certifies 100
    assert automorphism_families.estimate_automorphism_family_work(certificate, 3) == 5
    assert automorphism_families.estimate_automorphism_family_work(certificate, 4) == 5 + 10
    assert automorphism_families.estimate_automorphism_family_work(certificate, 100) == 2**5 - 1


def _get_random_mb_codes(rng: np.random.Generator, count: int) -> list[codes.MBCode]:
    """Random binary multivariate bicycle codes over small groups that encode logical qubits."""
    groups = [
        ((7,), (x,)),
        ((12,), (x,)),
        ((4, 3), (x, y)),
        ((6, 2), (x, y)),
        ((2, 2, 3), (x, y, z)),
    ]
    found: list[codes.MBCode] = []
    while len(found) < count:
        orders, symbols = groups[len(found) % len(groups)]
        polys = [
            sum(
                np.prod(
                    [symbol ** int(rng.integers(order)) for symbol, order in zip(symbols, orders)]
                )
                for _ in range(3)
            )
            for _ in range(2)
        ]
        code = codes.MBCode(dict(zip(symbols, orders)), *polys)
        if code.dimension:
            found.append(code)
    return found


def _get_group_elements(generators: npt.NDArray[np.int_]) -> list[npt.NDArray[np.int_]]:
    """Every permutation in the group that some permutations generate."""
    identity = np.arange(generators.shape[1])
    elements = {identity.tobytes(): identity}
    frontier = [identity]
    while frontier:
        products = {
            product.tobytes(): product
            for element in frontier
            for product in (generator[element] for generator in generators)
        }
        frontier = [product for key, product in products.items() if key not in elements]
        elements.update(products)
    return list(elements.values())


def _get_rank(matrix: npt.NDArray[np.uint8]) -> int:
    """The rank of a binary matrix."""
    return get_rank_packed(pack_bits(np.asarray(matrix, dtype=np.uint8)))
