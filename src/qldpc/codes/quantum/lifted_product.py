# SPDX-License-Identifier: Apache-2.0

"""Lifted product codes and their subsystem variant."""

from __future__ import annotations

from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt

import qldpc
from qldpc import abstract

from ..common import CSSCode
from .hypergraph_product import HGPCode, SHPCode
from .two_block import _get_abelian_block_translations

####################################################################################################
# lifted product codes


class LPCode(CSSCode):
    """Lifted product code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are RingArrays, or matrices whose entries are members of a group algebra
    over a finite field ``F_q``.  Each of these entries can be "lifted" to a matrix over ``F_q``, in
    which case the RingArray is interpreted as a block matrix; this is called "lifting" the
    RingArray.  Lifts to orthogonal matrices, such as permutation matrices, give X and Z checks that
    commute.  For other lifts, LPCode raises a ValueError if the checks do not commute.  LPCode also
    needs lifts with ``right=True``, which a noncommutative group has only with the default lift,
    its regular representation.

    As an example, the lift-connected surface code in Eq. (2) of https://arxiv.org/pdf/2401.02911v2
    can be constructed by::

        import numpy as np
        from qldpc.abstract import CyclicGroup, GroupRing, RingArray
        from qldpc.codes import RepetitionCode, LPCode

        num_copies = 5  # the number of surface codes to stitch together
        group = CyclicGroup(num_copies)
        x = group.generators[0]  # generator of the cyclic group

        # stitch together small surface codes by hand
        ring = GroupRing(group)
        rep_matrix = RingArray.build([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], ring)
        int_matrix = RingArray.build([[0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]], ring)
        code = LPCode(rep_matrix + int_matrix)

        # stitch together larger surface codes
        dist = 10  # distance of the individual surface codes
        rep_matrix = RingArray.build(RepetitionCode(dist).matrix, ring)
        int_matrix = RingArray.build(np.zeros((dist - 1, dist), dtype=int), ring)
        for row in range(len(int_matrix)):
            int_matrix[row, row + 1] = x
        code = LPCode(rep_matrix + int_matrix)

    Notes:

    - A lifted product code with ``RingArray`` instances of size ``1×1`` is a two-block code (more
      specifically, a two-block group-algebra code).  If the base group of the ``RingArray``
      instances is a cyclic group, the resulting lifted product code is a generalized bicycle code.
    - A lifted product code with ``RingArray`` instances whose entries get lifted to ``1×1``
      matrices is a hypergraph product code built from the lifted ``RingArray`` instances.
    - One way to get an ``LPCode``: take a classical code with parity check matrix ``H`` and
      multiply it by a diagonal matrix ``D = diag(a_1, a_2, ... a_n)``, where all ``{a_j}`` are
      elements of a group algebra.  The ``RingArray`` ``P = H @ D`` can then be used for one of the
      ``RingArray`` inputs of an ``LPCode``.

    References:

    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    - https://arxiv.org/abs/2306.16400
    - https://arxiv.org/abs/2401.02911
    """

    matrix_a: abstract.RingArray
    matrix_b: abstract.RingArray
    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        matrix_a: npt.NDArray[np.object_] | Sequence[Sequence[object]],
        matrix_b: npt.NDArray[np.object_] | Sequence[Sequence[object]] | None = None,
        *,
        set_logicals: bool = False,
    ) -> None:
        """Lifted product of two RingArrays, as in arXiv:2012.04068.

        If ``set_logicals`` is True, set the canonical logical line operators of the code (see
        ``get_canonical_logical_line_ops``).  These use semisimple Howell normal forms, which need a
        field characteristic that does not divide the order of the group.  Lifted products of binary
        codes over groups of even order are common but fail this condition, so ``set_logicals``
        defaults to False, and if it is True for a group algebra that fails it, a ValueError is
        raised.
        """
        if matrix_b is None:
            matrix_b = matrix_a
        self.matrix_a = abstract.RingArray(matrix_a)
        self.matrix_b = abstract.RingArray(matrix_b)

        ring = self.matrix_a.ring
        field = ring.field

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(self.matrix_a, self.matrix_b)

        # identify the number of qudits in each sector
        self.sector_size = self.matrix_a.group.lift_dim * np.outer(
            self.matrix_a.shape[::-1],
            self.matrix_b.shape[::-1],
        )

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self.bias_tailoring_qubits = slice(self.sector_size[0, 0], None)

        # lifts to orthogonal matrices give commuting checks; otherwise, verify that checks commute
        super().__init__(matrix_x.lift(), matrix_z.lift(), field)
        if _has_orthogonal_lifts(ring):
            self._is_subsystem_code = False
        elif self.is_subsystem_code:
            raise ValueError(
                "The X and Z checks of this LPCode do not commute, and its group algebra lifts"
                " members to matrices that are not orthogonal"
            )

        if set_logicals:
            try:
                logical_ops_xz = self.get_canonical_logical_line_ops(self.matrix_a, self.matrix_b)
            except (ValueError, NotImplementedError) as exception:
                raise ValueError(
                    "Cannot set canonical logical operators for this code, likely due to a"
                    " choice of group algebra for which some features are not yet supported"
                ) from exception
            # this call sits outside the try above because a validation failure here is a real bug
            # in the computed operators, not the "unsupported group algebra" the try/except reports
            self.set_logical_ops_xz(*logical_ops_xz, skip_validation=False)

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a lifted product code."""
        logical_ops_xz = LPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return LPCode.to_lifted_logical_ops(*logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a lifted product code.

        Generalizes HGPCode.get_canonical_logical_line_ops.
        """
        # put each ring kernel in a standard form before choosing paired logical operators
        generator_a = matrix_a.null_space().howell_normal_form_semisimple()
        generator_b = matrix_b.null_space(right=True).howell_normal_form_semisimple(right=True)
        generator_a_T = matrix_a.T.null_space().howell_normal_form_semisimple()
        generator_b_T = matrix_b.T.null_space(right=True).howell_normal_form_semisimple(right=True)

        dual_a = abstract.get_howell_dual(generator_a)  # generator_a @ dual_a.T is diagonal
        dual_b = abstract.get_howell_dual(generator_b)
        dual_a_T = abstract.get_howell_dual(generator_a_T)
        dual_b_T = abstract.get_howell_dual(generator_b_T)

        logical_ops_x_l = abstract.kron(dual_a, generator_b)
        logical_ops_z_l = abstract.kron(generator_a, dual_b)
        logical_ops_x_r = abstract.kron(generator_a_T, dual_b_T)
        logical_ops_z_r = abstract.kron(dual_a_T, generator_b_T)

        logical_ops_x = abstract.block_diag(logical_ops_x_l, logical_ops_x_r)
        logical_ops_z = abstract.block_diag(logical_ops_z_l, logical_ops_z_r)
        return logical_ops_x, logical_ops_z

    @staticmethod
    def to_lifted_logical_ops(
        logical_ops_x: abstract.RingArray, logical_ops_z: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Lift logical operators over a ring to logical operators over the base field."""
        ring = logical_ops_x.ring
        num_columns = logical_ops_x.shape[1]
        block_size = ring.group.lift_dim
        num_lifted_columns = num_columns * block_size
        identity = np.eye(block_size, dtype=int)
        lifted_ops_x = ring.field.Zeros((0, num_lifted_columns))
        lifted_ops_z = ring.field.Zeros((0, num_lifted_columns))
        for op_x, op_z in zip(logical_ops_x, logical_ops_z):
            ops_x = op_x.reshape(1, *op_x.shape).lift()
            ops_z = op_z.reshape(1, *op_z.shape).lift()
            # choose independent lifted operators and make their overlap matrix the identity
            inner_product = ops_x @ ops_z.T
            if not np.array_equal(inner_product, identity):
                sector_rank = np.linalg.matrix_rank(inner_product)
                if ring.is_commutative:
                    basis_change = np.linalg.inv(inner_product[:sector_rank, :sector_rank]).T
                    ops_x = ops_x[:sector_rank]
                    ops_z = basis_change @ ops_z[:sector_rank]
                else:
                    pivot_rows = qldpc.math.first_nonzero_cols(inner_product.T.row_reduce())
                    rows = pivot_rows[pivot_rows < block_size]
                    cols = qldpc.math.first_nonzero_cols(inner_product[rows].row_reduce())
                    basis_change = np.linalg.inv(inner_product[np.ix_(rows, cols)]).T
                    ops_x = ops_x[rows]
                    ops_z = basis_change @ ops_z[cols]
            lifted_ops_x = np.vstack([lifted_ops_x, ops_x])
            lifted_ops_z = np.vstack([lifted_ops_z, ops_z])

        return lifted_ops_x, lifted_ops_z

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Translations of every block of qudits, or None (see _get_abelian_block_translations)."""
        group = self.matrix_a.group
        return _get_abelian_block_translations(group, len(self) // group.lift_dim)


class SLPCode(CSSCode):
    """Subsystem lifted product code.

    The subsystem lifted product code is a lifted version of the subsystem hypergraph product code.
    That is, the SLPCode is to the SHPCode what the LPCode is to the HGPCode.  See
    help(qldpc.codes.LPCode) for additional information.

    As an example, the SLPCode in example 1 on page 6 of https://arxiv.org/pdf/2404.18302v1 can be
    constructed by::

        from qldpc.abstract import CyclicGroup, GroupRing, RingArray
        from qldpc.codes import SLPCode

        group = CyclicGroup(2)
        ring = GroupRing(group)
        x = ring.generators[0]  # generator of the cyclic group
        matrix = RingArray.build([[1, x, x], [x, x, 1]], ring)  # Eq. 21 of arXiv:2404.18302v1
        code = SLPCode(matrix)
        assert code.get_code_params() == (18, 4, 2)

    while the SLPCode in example 2 is::

        group = CyclicGroup(3)
        ring = GroupRing(group)
        x = ring.generators[0]    # generator of the cyclic group, as a member of the ring
        matrix = RingArray([[ring.one + x + x**2, ring.one + x, x]])  # Eq. 23 of arXiv:2404.18302v1
        code = SLPCode(matrix)
        assert code.get_code_params() == (27, 12, 2)

    References:

    - https://errorcorrectionzoo.org/c/subsystem_lifted_product
    - https://arxiv.org/abs/2404.18302
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.object_] | Sequence[Sequence[object]],
        matrix_b: npt.NDArray[np.object_] | Sequence[Sequence[object]] | None = None,
        *,
        set_logicals: bool = False,
    ) -> None:
        """Subsystem lifted product of two RingArrays, as in arXiv:2404.18302.

        ``set_logicals`` is as for :class:`LPCode`.
        """
        if matrix_b is None:
            matrix_b = matrix_a
        self.matrix_a = abstract.RingArray(matrix_a)
        self.matrix_b = abstract.RingArray(matrix_b)

        ring = self.matrix_a.ring
        field = ring.field

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = SHPCode.get_matrix_product(self.matrix_a, self.matrix_b)
        super().__init__(matrix_x.lift(), matrix_z.lift(), field, is_subsystem_code=True)

        # TODO: set self._stabilizer_ops

        if set_logicals:
            try:
                logical_ops_xz = self.get_canonical_logical_line_ops(self.matrix_a, self.matrix_b)
            except (ValueError, NotImplementedError) as exception:
                raise ValueError(
                    "Cannot set canonical logical operators for this code, likely due to a"
                    " choice of group algebra for which some features are not yet supported"
                ) from exception
            # this call sits outside the try above because a validation failure here is a real bug
            # in the computed operators, not the "unsupported group algebra" the try/except reports
            self.set_logical_ops_xz(*logical_ops_xz, skip_validation=False)

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a subsystem lifted product code."""
        logical_ops_xz = SLPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return SLPCode.to_lifted_logical_ops(*logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a subsystem lifted product code.

        Generalizes SHPCode.get_canonical_logical_line_ops.
        """
        generator_a = matrix_a.null_space().howell_normal_form_semisimple()
        generator_b = matrix_b.null_space(right=True).howell_normal_form_semisimple(right=True)

        dual_a = abstract.get_howell_dual(generator_a)  # generator_a @ dual_a.T is diagonal
        dual_b = abstract.get_howell_dual(generator_b)

        logical_ops_x = abstract.kron(dual_a, generator_b)
        logical_ops_z = abstract.kron(generator_a, dual_b)
        return logical_ops_x, logical_ops_z

    @staticmethod
    def to_lifted_logical_ops(
        logical_ops_x: abstract.RingArray, logical_ops_z: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Lift logical operators over a ring to logical operators over the base field."""
        return LPCode.to_lifted_logical_ops(logical_ops_x, logical_ops_z)

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Translations of every block of qudits, or None (see _get_abelian_block_translations)."""
        group = self.matrix_a.group
        return _get_abelian_block_translations(group, len(self) // group.lift_dim)


def _has_orthogonal_lifts(ring: abstract.GroupRing) -> bool:
    """Does a group algebra lift the members of its group to orthogonal matrices?

    A lift L is orthogonal if ``L(g.T) = L(g).T``, where ``g.T = ~g``.  Orthogonal matrices form a
    group, so if the generators lift to orthogonal matrices, every member does.
    """
    return all(
        np.array_equal(generator.T.lift(), generator.lift().T) for generator in ring.generators
    )
