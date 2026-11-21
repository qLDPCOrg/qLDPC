# SPDX-License-Identifier: Apache-2.0

"""Instances of the affine BZ search: the packed sets of a problem and their certificate.

The distance search (``distance.py``) and the coset searches (``coset.py``) scan the words of an
affine space: an offset plus sums of rows of a set.  Each search packs its problem once for each
family in an :class:`AffineInstance`.  Sections cited in this module are those of
``brouwer_zimmermann.md`` and ``brouwer_zimmermann_code.md``, whose Section 4.1 says how the
searches use instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_num_words, pack_bits

from .._information_sets import basis_families, certificates

####################################################################################################
# instances


@dataclass(frozen=True, eq=False)
class AffineInstance:
    """The packed sets of one problem of an affine BZ search, and the certificate of their steps.

    Attributes:
        generators: The packed rows of each set, with shape ``(num_sets, dimension, width)``,
            zero-padded to a common width.  A distance instance deletes the pivot columns of each
            set (Section 3.5).
        labels: The packed quotient labels of the rows, with the same first two axes, and no words
            for a problem without labels.
        widths: The number of words of ``generators`` that each set uses.
        pivots: The pivot columns of each set, padded with -1, with shape ``(num_sets, dimension)``.
            They index the full rows, which are the rows of ``generators`` only in a coset instance.
        ranks: The number of pivots of each set that are not padding.
        certificate: The certificate of the steps, in Hamming units, whose weight divisor is one
            in a coset instance.  For an automorphism family, it covers every image of the one
            packed set (Section 3.7).
        weight_factor: The factor ``η`` such that a lower bound ``L`` on Hamming weights bounds
            weights below by ``ceil(L / η)`` (Section 4.1).
    """

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    widths: npt.NDArray[np.int64]
    pivots: npt.NDArray[np.int64]
    ranks: npt.NDArray[np.int64] = field(init=False)
    certificate: certificates.Certificate
    weight_factor: int = 1

    def __post_init__(self) -> None:
        assert len(self.generators) == len(self.labels) == len(self.pivots)
        assert self.certificate.dimension == self.generators.shape[1]
        ranks = np.count_nonzero(self.pivots >= 0, axis=1).astype(np.int64)
        object.__setattr__(self, "ranks", ranks)

    def get_weight_bounds(self, parity: int | None = None) -> npt.NDArray[np.int64]:
        """Bounds on the weight of the unvisited words after each step of the certificate.

        ``parity``, if given, must be the parity of the Hamming weight of every word.  The Hamming
        bounds are rounded up to it, and then divided by the weight factor, rounding up
        (Section 3.2).  Each call returns a new table.
        """
        return self.certificate.get_weight_bounds(self.weight_factor, parity)


def get_distance_instance(
    basis_family: basis_families.BasisFamily, certificate: certificates.Certificate
) -> AffineInstance:
    """Pack the sets of a basis family for the distance search, with their pivot columns deleted.

    ``certificate`` is ``basis_family.certificate``, or that of the automorphism family whose one
    set ``basis_family`` holds.
    """
    # The scans count the ones of the deleted pivot columns from the support (Section 3.5).
    packed_rows = [
        pack_bits(np.delete(generators, pivots[:rank], axis=1))
        for generators, pivots, rank in zip(
            basis_family.generators, basis_family.pivots, basis_family.ranks, strict=True
        )
    ]

    # Zero-pad the sets to a common width, so that the compiled scans take single arrays.  The
    # labels of every set already share a width.
    widths = np.array([rows.shape[1] for rows in packed_rows], dtype=np.int64)
    packed_generators = np.zeros(
        (len(packed_rows), basis_family.generators.shape[1], int(widths.max(initial=0))),
        dtype=np.uint64,
    )
    for index, rows in enumerate(packed_rows):
        packed_generators[index, :, : widths[index]] = rows
    packed_labels = np.stack([pack_bits(set_labels) for set_labels in basis_family.labels])
    return AffineInstance(
        generators=packed_generators,
        labels=packed_labels,
        widths=widths,
        pivots=basis_family.pivots,
        certificate=certificate,
    )


def get_coset_instance(
    generators: npt.NDArray[np.uint8], *, random_column_orders: int, weight_factor: int
) -> AffineInstance:
    """Pack the disjoint information sets of independent generators for the coset search.

    The rows keep every column, and the certificate has no weight divisor, even if the code has one
    (Section 3.2).  Without generators, each coset is a single word, which the search visits on
    layer zero of one empty set.

    Args:
        generators: Independent generators of the code, one in each row.
        random_column_orders: Passed to ``basis_families.get_disjoint_family``.
        weight_factor: Two for a search of Pauli errors, and one for a search of X or Z parts
            (Section 4.1).
    """
    num_generators, num_columns = generators.shape
    if not num_generators:
        width = get_num_words(num_columns)
        return AffineInstance(
            generators=np.zeros((1, 0, width), dtype=np.uint64),
            labels=np.zeros((1, 0, 0), dtype=np.uint64),
            widths=np.full(1, width, dtype=np.int64),
            pivots=np.full((1, 0), -1, dtype=np.int64),
            certificate=certificates.Certificate(0, (0,)),
            weight_factor=weight_factor,
        )
    family = basis_families.get_disjoint_family(
        generators, None, random_column_orders=random_column_orders
    )
    return get_coset_family_instance(family, weight_factor=weight_factor)


def get_coset_family_instance(
    basis_family: basis_families.BasisFamily, *, weight_factor: int
) -> AffineInstance:
    """Pack the sets of any basis family of a code for the coset search, with every column.

    The sets may overlap, as in the families of coset routes (Section 4.4).  The certificate takes
    the ranks and maximum coverage of the family, and no weight divisor, and holds for every coset
    (Section 3.1, "Families for coset searches").  ``weight_factor`` is as for
    ``get_coset_instance``.
    """
    num_sets, num_generators, num_columns = basis_family.generators.shape
    packed_generators = pack_bits(basis_family.generators.reshape(-1, num_columns)).reshape(
        num_sets, num_generators, -1
    )
    return AffineInstance(
        generators=packed_generators,
        labels=np.zeros((num_sets, num_generators, 0), dtype=np.uint64),
        widths=np.full(num_sets, packed_generators.shape[2], dtype=np.int64),
        pivots=basis_family.pivots,
        certificate=certificates.Certificate(
            num_generators,
            tuple(int(rank) for rank in basis_family.ranks),
            max_coverage=basis_family.max_coverage,
        ),
        weight_factor=weight_factor,
    )
