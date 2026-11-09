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
from .hypergraph_product import HGPCode, SHPCode, _check_same_ring
from .two_block import _get_abelian_block_translations

####################################################################################################
# lifted product codes


class LPCode(CSSCode):
    """Lifted product code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are RingArrays, or matrices whose entries are members of a ring over a
    finite field ``F_q``, such as a group algebra.  Each of these entries can be "lifted" to a
    matrix over ``F_q``, in which case the RingArray is interpreted as a block matrix; this is
    called "lifting" the RingArray.

    The ring must satisfy the following requirements:

    1. The ring is commutative or a group algebra, since ``abstract.kron`` implements the Kronecker
       product over a noncommutative ring only for group algebras.
    2. The group of a noncommutative group algebra uses the default lift, its regular
       representation.  Lifting a Kronecker product over a noncommutative group algebra needs lifts
       with ``right=True``, which a noncommutative group has only with its default lift.
    3. The ring has an involution (see ``Ring.has_involution``).

    LPCode raises a NotImplementedError if requirement 1 fails, and otherwise a ValueError if
    requirement 2 or 3 fails.  If the involution of the ring transposes lifts, the X and Z checks
    commute.  Otherwise, LPCode verifies that they commute, and raises a ValueError if they do not.
    The involution of a ``MatrixRing`` transposes lifts, as does that of a group algebra whose group
    members lift to orthogonal matrices, such as permutation matrices.

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

    - A lifted product code with ``RingArray`` instances of size ``1×1`` is a two-block code, and a
      two-block group-algebra code if the ring is a group algebra.  If the ring is the algebra of a
      cyclic group, the resulting lifted product code is a generalized bicycle code.
    - A lifted product code with ``RingArray`` instances whose entries get lifted to ``1×1``
      matrices is a hypergraph product code built from the lifted ``RingArray`` instances.
    - One way to get an ``LPCode``: take a classical code with parity check matrix ``H`` and
      multiply it by a diagonal matrix ``D = diag(a_1, a_2, ... a_n)``, where all ``{a_j}`` are
      elements of the ring.  The ``RingArray`` ``P = H @ D`` can then be used for one of the
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

        The two RingArrays must be over the same ring, or else a ValueError is raised.

        If ``set_logicals`` is True, set the canonical logical line operators of the code.  These
        operators need a group algebra whose group order is not divisible by the field
        characteristic (see ``get_canonical_logical_line_ops``, which also describes the side
        effects of computing them).  Lifted products of binary codes over groups of even order are
        common but fail this condition, so ``set_logicals`` defaults to False.  If it is True for a
        ring that fails this condition, a ValueError is raised.  A ValueError is also raised if
        these operators are invalid, which can happen if the involution of the ring does not
        transpose lifts.
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
        self.sector_size = ring.lift_dim * np.outer(
            self.matrix_a.shape[::-1],
            self.matrix_b.shape[::-1],
        )

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self.bias_tailoring_qubits = slice(self.sector_size[0, 0], None)

        # an involution that transposes lifts gives commuting checks; otherwise, verify that checks
        # commute
        super().__init__(matrix_x.lift(), matrix_z.lift(), field)
        if _involution_transposes_lifts(ring):
            self._is_subsystem_code = False
        elif self.is_subsystem_code:
            raise ValueError(
                "The X and Z checks of this LPCode do not commute, and the involution of its ring"
                " does not transpose lifts"
            )

        if set_logicals:
            _set_canonical_logical_line_ops(self)

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a lifted product code.

        These operators use semisimple Howell normal forms, which are implemented only for group
        algebras, and need a field characteristic that does not divide the group order.  This method
        raises a NotImplementedError for a ring that is not a group algebra, and a ValueError for a
        group algebra that fails this condition or for matrices over different rings.  Unless the
        primitive central idempotents of the group algebra are known, this method may run GAP,
        prompt on standard input, and use the system clipboard; see GroupRing.get_transformer.
        These operators also assume that the involution of the ring transposes lifts.
        """
        logical_ops_xz = LPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return LPCode.to_lifted_logical_ops(*logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a lifted product code.

        Generalizes HGPCode.get_canonical_logical_line_ops.  See
        ``LPCode.get_canonical_logical_line_ops`` for the rings that this method needs, and its side
        effects.  Matrices over different rings raise a ValueError.
        """
        _require_group_algebras(matrix_a, matrix_b)
        _check_same_ring(matrix_a, matrix_b)

        # put each ring null space in a standard form before choosing paired logical operators
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
        """Lift logical operators over a ring to logical operators over the base field.

        Each row of ``logical_ops_x`` and the matching row of ``logical_ops_z`` lift to blocks of X
        and Z operators, whose overlap matrix is ``X @ Z.T``.  Lifting keeps as many pairs of these
        X and Z operators as the rank of this overlap matrix, and makes their overlap matrix the
        identity.  If operators lifted from different rows have zero overlap, as for canonical
        logical operators if the involution of the ring transposes lifts, the overlap matrix of all
        lifted operators is the identity.  Arrays with different numbers of rows raise a ValueError.
        """
        if len(logical_ops_x) != len(logical_ops_z):
            raise ValueError(
                f"Cannot lift {len(logical_ops_x)} X and {len(logical_ops_z)} Z logical operators,"
                " which must come in pairs"
            )
        ring = logical_ops_x.ring
        num_columns = logical_ops_x.shape[1]
        block_size = ring.lift_dim
        num_lifted_columns = num_columns * block_size
        lifted_ops_x = ring.field.Zeros((0, num_lifted_columns))
        lifted_ops_z = ring.field.Zeros((0, num_lifted_columns))
        for op_x, op_z in zip(logical_ops_x, logical_ops_z):
            # lift a pair of rows to blocks of X and Z operators, keep the X operators at pivot rows
            # of their overlap matrix, which are independent and as many as its rank, and combine
            # the Z operators into a dual basis for the kept X operators
            ops_x = op_x.reshape(1, *op_x.shape).lift()
            ops_z = op_z.reshape(1, *op_z.shape).lift()
            overlap = ops_x @ ops_z.T
            pivots = qldpc.math.first_nonzero_cols(overlap.T.row_reduce())
            rows = pivots[pivots < block_size]
            basis_change = qldpc.math.get_dual_basis(overlap[rows], validate=False)
            lifted_ops_x = np.vstack([lifted_ops_x, ops_x[rows]])
            lifted_ops_z = np.vstack([lifted_ops_z, basis_change @ ops_z])

        return lifted_ops_x, lifted_ops_z

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Translations of every block of qudits, or None (see _get_lifted_block_translations)."""
        return _get_lifted_block_translations(self.matrix_a.ring, len(self))


class SLPCode(CSSCode):
    """Subsystem lifted product code.

    The subsystem lifted product code is a lifted version of the subsystem hypergraph product code.
    That is, the SLPCode is to the SHPCode what the LPCode is to the HGPCode.  See
    help(qldpc.codes.LPCode) for additional information.  The ring of an SLPCode must satisfy
    requirements 1 and 2 of an LPCode.

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

        The two RingArrays must be over the same ring, or else a ValueError is raised.
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
            _set_canonical_logical_line_ops(self)

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a subsystem lifted product code.

        See ``LPCode.get_canonical_logical_line_ops`` for the rings that this method needs, and its
        side effects.  These operators assume that the involution of the ring transposes lifts.
        """
        logical_ops_xz = SLPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return SLPCode.to_lifted_logical_ops(*logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a subsystem lifted product code.

        Generalizes SHPCode.get_canonical_logical_line_ops.  See
        ``LPCode.get_canonical_logical_line_ops`` for the rings that this method needs, and its side
        effects.  Matrices over different rings raise a ValueError.
        """
        _require_group_algebras(matrix_a, matrix_b)
        _check_same_ring(matrix_a, matrix_b)
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
        """Lift logical operators over a ring to logical operators over the base field.

        See :meth:`LPCode.to_lifted_logical_ops`.
        """
        return LPCode.to_lifted_logical_ops(logical_ops_x, logical_ops_z)

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Translations of every block of qudits, or None (see _get_lifted_block_translations)."""
        return _get_lifted_block_translations(self.matrix_a.ring, len(self))


def _get_lifted_block_translations(
    ring: abstract.Ring, num_qudits: int
) -> npt.NDArray[np.int_] | None:
    """Block translations that preserve a code built from lifts of the members of a ring, or None.

    Returns:
        The result of ``_get_abelian_block_translations`` for the group of a group algebra, and None
        for any other ring.
    """
    if not isinstance(ring, abstract.GroupRing):
        return None
    return _get_abelian_block_translations(ring.group, num_qudits // ring.lift_dim)


def _involution_transposes_lifts(ring: abstract.Ring) -> bool:
    """Does the involution of a ring transpose the lifts of its members?

    For a group algebra, this means that group members lift to orthogonal matrices.  The maps
    ``x -> x.T.lift()`` and ``x -> x.lift().T`` are anti-homomorphisms that both send ``ring.one``
    to the identity matrix, so they agree on every member if they agree on the generators of the
    ring.
    """
    return all(
        np.array_equal(generator.T.lift(), generator.lift().T) for generator in ring.generators
    )


def _require_group_algebras(*matrices: abstract.RingArray) -> None:
    """Raise a NotImplementedError unless all matrices are over group algebras.

    Canonical logical operators of lifted products use semisimple Howell normal forms, which are
    implemented only for group algebras.  This check comes before computing null spaces, which raise
    a ValueError for a ring without an involution.
    """
    for matrix in matrices:
        if not isinstance(matrix.ring, abstract.GroupRing):
            raise NotImplementedError(
                "Canonical logical operators of lifted products are implemented only for group"
                f" algebras, not {matrix.ring}"
            )


def _set_canonical_logical_line_ops(code: LPCode | SLPCode) -> None:
    """Set the canonical logical line operators of a lifted product code."""
    try:
        ring_logical_ops_xz = code.get_canonical_ring_logical_line_ops(code.matrix_a, code.matrix_b)
    except (ValueError, NotImplementedError) as exception:
        raise ValueError(
            "Cannot set canonical logical operators for this code, likely due to a choice of ring"
            " for which some features are not yet supported"
        ) from exception

    # lift these operators to the base field
    logical_ops_xz = code.to_lifted_logical_ops(*ring_logical_ops_xz)

    # if the involution of the ring transposes lifts, invalid operators are a bug
    if _involution_transposes_lifts(code.matrix_a.ring):
        code.set_logical_ops_xz(*logical_ops_xz, skip_validation=False)
        return

    # these operators assume that the involution transposes lifts, so otherwise they can be invalid
    try:
        code.set_logical_ops_xz(*logical_ops_xz, skip_validation=False)
    except ValueError as exception:
        raise ValueError(
            "Cannot set canonical logical operators for this code, since the involution of its ring"
            " does not transpose lifts"
        ) from exception
