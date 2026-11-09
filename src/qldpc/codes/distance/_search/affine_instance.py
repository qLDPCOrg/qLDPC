# SPDX-License-Identifier: Apache-2.0

"""Instances of the affine BZ search: the packed sets of a problem and their certificate.

The distance search (``brouwer_zimmermann.py``) and the coset-leader search
(``coset_leader_search.py``) are two instances of one Brouwer-Zimmermann (BZ) search over the words
of an affine space, which add rows of an information set to an offset (see
``brouwer_zimmermann.md``).  An :class:`AffineInstance` holds what such a search builds once for a
problem: the packed rows and quotient labels of the sets that it scans, their pivot columns and
ranks, the certificate that bounds the Hamming weight of the words that it has not yet visited, and
the factor that converts these bounds into bounds on the weights that the search compares.  The
offsets and the stopping rule are set per query by the search.

A distance instance packs the rows of each set with its pivot columns deleted, since a scan of a
space restores their weight arithmetically, and its certificate is the one that the distance search
selects, with the proven divisor of eligible weights.  A coset instance packs the full rows, which
the coset search weighs whole, and its certificate takes no divisor, since the words of a coset need
not share the divisibility of the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits

from . import information_sets

####################################################################################################
# instances


@dataclass(frozen=True, eq=False)
class AffineInstance:
    """The packed sets of one problem of an affine BZ search, and the certificate of their steps.

    Attributes:
        generators: The packed rows of each scanned set, with shape ``(num_sets, dimension,
            width)``, zero-padded to a common width.
        labels: The packed quotient labels of the rows of each scanned set, with the same first two
            axes, and no words for a problem without labels.
        widths: The number of words of ``generators`` that each scanned set uses.
        label_widths: The number of words of ``labels`` that each scanned set uses.
        pivots: The pivot columns of each scanned set, padded with -1, with shape ``(num_sets,
            dimension)``.
        ranks: The rank of the pivot block of each scanned set.
        certificate: The certificate of the steps of the search, in Hamming units.  Its weight
            divisor is the rounding rule of the instance.  Its sets need not be the scanned sets:
            a search that scans one set of an automorphism family is certified by all its images.
        weight_factor: The factor ``η`` such that every word weighs at least its Hamming weight
            divided by ``η``, so that a bound ``L`` on Hamming weights bounds weights by
            ``⌈L/η⌉``.
    """

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    widths: npt.NDArray[np.int64]
    label_widths: npt.NDArray[np.int64]
    pivots: npt.NDArray[np.int64]
    ranks: npt.NDArray[np.int64]
    certificate: information_sets.Certificate
    weight_factor: int = 1

    def __post_init__(self) -> None:
        assert len(self.ranks) == len(self.generators) == len(self.pivots)
        assert self.certificate.dimension == self.generators.shape[1]

    def get_weight_bounds(self) -> npt.NDArray[np.int64]:
        """Bounds on the weight of the unvisited words after each step of the certificate.

        Entry ``[w, j]`` is ``⌈L/η⌉`` for the entry ``L`` of the coverage bounds of the certificate
        (see ``information_sets.get_step_lower_bounds``), except that the entries after the last
        layer stay at the largest native integer, since no word remains unvisited.  The bounds do
        not decrease in the order of the steps, and column zero of each row repeats the last column
        of the row before.  Each call returns a new table.
        """
        certificate = self.certificate
        bounds = information_sets.get_step_lower_bounds(
            certificate.dimension,
            np.array(certificate.ranks, dtype=np.int64),
            certificate.weight_divisor,
            certificate.max_coverage,
        )
        # Round up without overflowing the bounds after the last layer, which are then restored.
        bounds = -(-bounds // self.weight_factor)
        bounds[certificate.dimension, 1:] = np.iinfo(np.int64).max
        return bounds


def get_distance_instance(
    basis_family: information_sets.BasisFamily, certificate: information_sets.Certificate
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
    label_widths = np.full(len(packed_labels), packed_labels.shape[2], dtype=np.int64)
    return AffineInstance(
        packed_generators,
        packed_labels,
        widths,
        label_widths,
        basis_family.pivots,
        np.array(basis_family.ranks, dtype=np.int64),
        certificate,
    )


def get_coset_instance(
    generators: npt.NDArray[np.uint8], *, random_column_orders: int, weight_factor: int
) -> AffineInstance:
    """Pack the disjoint information sets of independent generators for the coset search.

    The rows keep every column.  The certificate takes no weight divisor, since the words of a coset
    need not share the divisibility of the code, even if the family infers one, and a coverage of
    one, since the sets are disjoint.  ``random_column_orders`` is passed to
    ``information_sets.get_disjoint_family``.  Without generators, each coset is a single word,
    which the search visits on layer zero of one empty set.
    """
    num_generators, num_columns = generators.shape
    if not num_generators:
        width = max(1, (num_columns + 63) // 64)
        return AffineInstance(
            np.zeros((1, 0, width), dtype=np.uint64),
            np.zeros((1, 0, 0), dtype=np.uint64),
            np.full(1, width, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.full((1, 0), -1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            information_sets.Certificate(0, (0,)),
            weight_factor,
        )
    family = information_sets.get_disjoint_family(
        generators, None, random_column_orders=random_column_orders
    )
    num_sets = len(family.ranks)
    packed_generators = pack_bits(family.generators.reshape(-1, num_columns)).reshape(
        num_sets, num_generators, -1
    )
    ranks = np.array(family.ranks, dtype=np.int64)
    return AffineInstance(
        packed_generators,
        np.zeros((num_sets, num_generators, 0), dtype=np.uint64),
        np.full(num_sets, packed_generators.shape[2], dtype=np.int64),
        np.zeros(num_sets, dtype=np.int64),
        family.pivots,
        ranks,
        information_sets.Certificate(num_generators, tuple(int(rank) for rank in ranks)),
        weight_factor,
    )
