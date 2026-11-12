# SPDX-License-Identifier: Apache-2.0

"""Instances of the affine BZ search: the packed sets of a problem and their certificate.

The distance search (``distance.py``) and the coset searches of the coset-leader search
(``coset.py``) run the same Brouwer-Zimmermann (BZ) search over the words of an affine space (see
``brouwer_zimmermann.md``).  The layers of this search add rows of an information set to an offset.
An :class:`AffineInstance` holds what a search builds for its problem before it scans, rather than
for each layer or query: the packed rows and quotient labels of the sets that it scans, their pivot
columns and ranks, the certificate that bounds the Hamming weight of the words that it has not yet
visited, and the factor that converts these bounds into bounds on the weights that the search
compares.  A distance search builds another instance when it switches to an automorphism family.
Each family with which a coset search of ``CosetLeaderSearch`` answers queries has a coset instance
(see ``coset_routes.py``).  The offsets and stopping step of a coset search depend on the query, so
the instance holds neither.

A distance instance packs the rows of each set with its pivot columns deleted, since a scan of a
space restores their weight arithmetically.  Its certificate is the one that the distance search
selects, with the proven divisor of eligible weights.  A coset instance packs the full rows of the
sets of a family of information sets of its code, disjoint or overlapping, which a coset search
weighs whole or, in a search that weighs Hamming weights, on its routes by tails, by tails derived
from them (see ``coset_tables.CosetTails``).  Its certificate takes the maximum coverage of the
family, and no divisor, since the words of a coset need not share the divisibility of the code.  If
every word of the code has even Hamming weight, the Hamming weights of the words of each coset all
have one parity instead, and the bounds of a coset may be rounded up to its parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits

from .._information_sets import basis_families, certificates

####################################################################################################
# instances


@dataclass(frozen=True, eq=False)
class AffineInstance:
    """The packed sets of one problem of an affine BZ search, and the certificate of their steps.

    Attributes:
        generators: The packed rows of each scanned set, with shape ``(num_sets, dimension,
            width)``, zero-padded to a common width.  The rows of a distance instance have the
            pivot columns of their set deleted.
        labels: The packed quotient labels of the rows of each scanned set, with the same first two
            axes, and no words for a problem without labels.
        widths: The number of words of ``generators`` that each scanned set uses.
        label_widths: The number of words of ``labels`` that each scanned set uses.  Every set
            uses all of them, and the array holds a number for each set because the compiled scans
            take a width for each set.
        pivots: The pivot columns of each scanned set, padded with -1, with shape ``(num_sets,
            dimension)``.  They are columns of the full rows, and are bit positions of the rows of
            ``generators`` only in a coset instance.
        ranks: The rank of each scanned set: the number of its pivots that are not padding.
        certificate: The certificate of the steps of the search, in Hamming units.  Its weight
            divisor, which is one for a coset instance, rounds the Hamming bounds up to multiples of
            a proven divisor of eligible weights.  Its sets need not be the scanned sets: a search
            that scans one set of an automorphism family is certified by all its images.
        weight_factor: The factor ``η`` such that every word weighs at least its Hamming weight
            divided by ``η``, so that a bound ``L`` on Hamming weights bounds weights below by
            ``ceil(L / η)``.
    """

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    widths: npt.NDArray[np.int64]
    label_widths: npt.NDArray[np.int64] = field(init=False)
    pivots: npt.NDArray[np.int64]
    ranks: npt.NDArray[np.int64] = field(init=False)
    certificate: certificates.Certificate
    weight_factor: int = 1

    def __post_init__(self) -> None:
        assert len(self.generators) == len(self.labels) == len(self.pivots)
        assert self.certificate.dimension == self.generators.shape[1]
        label_widths = np.full(len(self.labels), self.labels.shape[2], dtype=np.int64)
        ranks = np.count_nonzero(self.pivots >= 0, axis=1).astype(np.int64)
        object.__setattr__(self, "label_widths", label_widths)
        object.__setattr__(self, "ranks", ranks)

    def get_weight_bounds(self, parity: int | None = None) -> npt.NDArray[np.int64]:
        """Bounds on the weight of the unvisited words after each step of the certificate.

        The table of ``Certificate.get_weight_bounds`` for the weight factor of the instance and
        ``parity``: the bounds on Hamming weight are rounded up to ``parity``, if it is given,
        before they are divided by the weight factor.  The Hamming weights of the words of each
        coset of an even code share a parity, zero or one (see ``get_coset_instance``), so the
        bounds of such a coset may be rounded up to it.  Each call returns a new table.
        """
        return self.certificate.get_weight_bounds(self.weight_factor, parity)


def get_distance_instance(
    basis_family: basis_families.BasisFamily, certificate: certificates.Certificate
) -> AffineInstance:
    """Pack the sets of a basis family for the distance search, with their pivot columns deleted.

    The certificate is the one that the search selects for the family (see
    ``BasisFamily.get_search_certificate``), or that of an automorphism family whose one set the
    basis family holds.
    """
    # Remove systematic pivots; the scans restore their known weight arithmetically.
    packed_rows = [
        pack_bits(np.delete(generators, pivots[:rank], axis=1))
        for generators, pivots, rank in zip(
            basis_family.generators, basis_family.pivots, basis_family.ranks, strict=True
        )
    ]

    # Zero-pad every set to a common word width, so the Numba kernel receives single arrays.  The
    # labels of every set already have the same width, which is zero for a basis without labels.
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

    The rows keep every column.  The certificate takes no weight divisor, since the words of a coset
    need not share the divisibility of the code, even if the family infers one.  Its coverage is
    one, since the sets are disjoint.  ``random_column_orders`` is passed to
    ``basis_families.get_disjoint_family``.  ``weight_factor`` is the weight factor of the instance.
    It is two for a search of Pauli errors, since an error weighs at least half the Hamming weight
    of the binary word that represents it in the search (see ``get_pauli_search`` in ``coset.py``).
    It is one for a search of X or Z parts.  Without generators, each coset is a single word, which
    the search visits on layer zero of one empty set.

    If every generator has even Hamming weight, so does every word ``v`` of the code, and every word
    ``u + v`` of a coset has the parity of ``|u|``, since ``|u + v| = |u| + |v| - 2 |u ∩ v|``.  The
    bounds of each coset may then be rounded up to its parity (see ``get_weight_bounds``).  The
    coset search decides whether to, since only it knows whether some coset is odd (see
    ``coset_tables.get_search_tables``).
    """
    num_generators, num_columns = generators.shape
    if not num_generators:
        width = max(1, (num_columns + 63) // 64)
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

    The family may hold disjoint sets, as in ``get_coset_instance``, or overlapping ones, such as
    those of ``basis_families.get_overlapping_family``.  The certificate takes the ranks and the
    maximum coverage of the family, and no weight divisor.  Its coverage bound holds for the words
    of every coset as it does for the words of the code: a word of a coset adds a unique sum of the
    rows of each set to the offset of the set, which is zero on the pivots of the set, so a word in
    layer ``w`` or above of a set has at least as many ones on its pivots as a word of the code
    does.  ``weight_factor`` is the weight factor of the instance (see ``get_coset_instance``).
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
