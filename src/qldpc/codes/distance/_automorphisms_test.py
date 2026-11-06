# SPDX-License-Identifier: Apache-2.0

"""Unit tests for families of translates of an information set by automorphisms."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

from qldpc import codes
from qldpc._binary import get_rank_packed, pack_bits
from qldpc.objects import Pauli, PauliXZ

from . import _automorphisms
from ._binary import get_eligible_rows, get_nested_code_basis


def test_automorphisms_must_map_both_spaces_onto_themselves() -> None:
    """Only permutations that preserve the searched space and its excluded subspace pass."""
    code = codes.QCCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    basis, labels = _get_sector(code, Pauli.X)
    shift = np.concatenate([np.roll(np.arange(7), 1), np.roll(np.arange(7), 1) + 7])
    _automorphisms.verify_automorphisms(basis, labels, shift[None])
    _automorphisms.verify_automorphisms(basis, labels, np.empty((0, 14), dtype=int))

    transposition = np.arange(14)
    transposition[[0, 1]] = [1, 0]
    not_permutations = [np.zeros((1, 14), dtype=int), np.arange(13)[None], np.arange(14)]
    for automorphisms in [np.stack([shift, transposition]), *not_permutations]:
        with pytest.raises(ValueError):
            _automorphisms.verify_automorphisms(basis, labels, automorphisms)

    # Swapping the last two columns preserves the excluded subspace, spanned by 100, but not the
    # searched space; swapping the first two preserves the searched space, but not the excluded one.
    basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    for swap in ([0, 2, 1], [1, 0, 2]):
        with pytest.raises(ValueError, match="not an automorphism"):
            _automorphisms.verify_automorphisms(basis, labels, np.array([swap]))


@pytest.mark.usefixtures("compilation_mode")
def test_translates_certify_like_the_family_of_all_translates() -> None:
    """The certificate is that of the translates of the scanned set by every group element.

    Each translate is an information set, and the coverage of the family is counted directly.  The
    groups include one that acts on each block with several orbits.
    """
    rng = np.random.default_rng(0)
    for code in _get_random_qc_codes(rng, 12):
        translations = code._get_qudit_automorphisms()
        for automorphisms in (translations, translations[:1]):
            elements = _get_group_elements(automorphisms)
            for pauli in (Pauli.X, Pauli.Z):
                basis, labels = _get_sector(code, pauli)
                divisor = 2 if np.all(basis.sum(axis=1) % 2 == 0) else 1
                family, certificate = _automorphisms.get_translate_family(
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


@pytest.mark.usefixtures("compilation_mode")
def test_scanned_sets_balance_two_orbits() -> None:
    """For two orbits, the scanned set has the fewest pivots in its fuller orbit that any can have.

    An information set has at most ``rank(O)`` pivots in an orbit ``O``, and so at least ``r -
    rank(O')`` in the other orbit ``O'``, and every count between these bounds is attained.
    """
    rng = np.random.default_rng(1)
    for code in _get_random_qc_codes(rng, 12):
        for pauli in (Pauli.X, Pauli.Z):
            basis, labels = _get_sector(code, pauli)
            dimension, num_columns = basis.shape
            family, _ = _automorphisms.get_translate_family(
                basis, labels, code._get_qudit_automorphisms(), 1
            )
            blocks = (np.arange(num_columns // 2), np.arange(num_columns // 2, num_columns))
            counts = [np.isin(family.pivots[0], block).sum() for block in blocks]
            ranks = [_get_rank(basis[:, block]) for block in blocks]
            assert max(counts) == max((dimension + 1) // 2, dimension - min(ranks))


def test_translate_work_scans_one_set_per_layer() -> None:
    """The work estimate counts each layer of one set, through the layer that certifies a bound."""
    basis, labels = _get_sector(codes.QCCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4), Pauli.X)
    shift = np.concatenate([np.roll(np.arange(7), 1), np.roll(np.arange(7), 1) + 7])
    _, certificate = _automorphisms.get_translate_family(basis, labels, shift[None], 1)
    dimension = len(basis)
    for upper_bound in range(1, 2 * dimension):
        last = min(
            weight
            for weight in range(1, dimension + 1)
            if weight == dimension or certificate.lower_bound(weight) >= upper_bound
        )
        expected = sum(math.comb(dimension, weight) for weight in range(1, last + 1))
        assert _automorphisms.estimate_translate_work(certificate, upper_bound) == expected


def _get_sector(
    code: codes.CSSCode, pauli: PauliXZ
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The adapted basis of one sector of a binary CSS code, and its quotient labels."""
    return get_nested_code_basis(
        np.asarray(code.get_logical_ops(pauli), dtype=np.uint8),
        np.asarray(code.get_stabilizer_ops(pauli, canonicalized=True), dtype=np.uint8),
    )


def _get_random_qc_codes(rng: np.random.Generator, count: int) -> list[codes.QCCode]:
    """Random binary quasi-cyclic codes that encode logical qubits, over small abelian groups."""
    groups = [
        ((7,), (x,)),
        ((12,), (x,)),
        ((4, 3), (x, y)),
        ((6, 2), (x, y)),
        ((2, 2, 3), (x, y, z)),
    ]
    found: list[codes.QCCode] = []
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
        code = codes.QCCode(dict(zip(symbols, orders)), *polys)
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
