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
    - https://arxiv.org/abs/2607.28795
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
        characteristic, or seed matrices that split at trailing pivot blocks over a ring whose
        involution transposes lifts.  See ``get_canonical_logical_line_ops`` for these conditions
        and for the side effects of computing the operators.  Lifted products of binary codes over
        groups of even order are common but often fail both conditions, so ``set_logicals``
        defaults to False.  If it is True for matrices that fail both conditions, a ValueError is
        raised.  A ValueError is also raised if these operators are invalid, which can happen for a
        group algebra whose involution does not transpose lifts.
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

        These operators come from one of two routes.

        1. Semisimple route, for group algebras whose group order the field characteristic does
           not divide.  This route lifts the operators of ``get_canonical_ring_logical_line_ops``,
           which use semisimple Howell normal forms.  Unless the primitive central idempotents of
           the group algebra are known, this route may run GAP, prompt on standard input, and use
           the system clipboard; see GroupRing.get_transformer.  These operators assume that the
           involution of the ring transposes lifts.  This route takes precedence wherever it
           applies, even for seed matrices that also fit the split route.
        2. Split route, for seed matrices ``matrix_a`` and ``matrix_b`` that split at trailing
           pivot blocks (defined below), following Appendix B of https://arxiv.org/abs/2607.28795v2.
           This route is for any other ring that meets the requirements on the ring of an
           ``LPCode``, and whose involution transposes lifts.

        Let ``matrix_a`` and ``matrix_b`` have shapes ``(ma, na)`` and ``(mb, nb)``, and let ``d``
        be the size of lifted blocks.  The first block of qudits, which is sector (0, 0), is a grid
        of cells ``(i, j, g)``, where ``i`` indexes the columns of ``matrix_a``, ``j`` indexes the
        columns of ``matrix_b``, and ``g`` indexes the coordinates of a lifted block.  A line of
        qudits is a row (fixed ``i``) or a column (fixed ``j``) of this grid.  Write
        ``matrix_a = [A_free | A_pivot]`` and ``matrix_b = [B_free | B_pivot]``, where the square
        pivot blocks are the last ``ma`` and ``mb`` columns.  A cell is free if its ``i`` and ``j``
        index free columns, and a pivot cell otherwise.  The seed matrices split at these pivot
        blocks if the lifts of the pivot blocks are invertible in the actions that the code uses on
        lines of qudits, where ``I`` is an identity matrix: ``A_pivot`` as in ``kron(matrix_a, I)``,
        and ``B_pivot`` as in ``kron(I, matrix_b)``.  This condition generalizes the square
        invertibility condition of Definition 16 in arXiv:2607.28795v2 to other rings, lifts, and
        square seed matrices, with the pivot columns last.  Over a noncommutative group algebra,
        these actions multiply from the left and from the right, and a ring matrix can be
        invertible in one action but not the other.

        The split route builds one pair of operators for each free column ``i`` of ``matrix_a``,
        free column ``j`` of ``matrix_b``, and coordinate ``g`` of a lifted block.  The X operator
        lies on row ``i`` of the grid, and the Z operator lies on column ``j``.  On the free cells
        of its line, each operator is the unit vector at ``(i, j, g)``, and its values on pivot
        cells solve the checks.  For these ``(na - ma) * (nb - mb) * d`` pairs,
        ``logical_ops_x @ logical_ops_z.T`` is the identity.  These pairs span all logical
        operators because the involution of the ring transposes lifts.  Without this property, a
        subsystem lifted product can encode more logical qudits, so the split route declines such
        rings.  This route recognizes only trailing pivot blocks, so seed matrices with invertible
        pivot blocks elsewhere need their columns reordered first.

        For matrices that fit neither route, this method raises a NotImplementedError if either
        ring is not a group algebra, and otherwise a ValueError, such as for matrices over
        different rings or for a group algebra whose group order the field characteristic divides.
        """
        # the split route declines semisimple group algebras, so the semisimple route takes
        # precedence
        logical_ops_xz = _get_split_logical_line_ops(matrix_a, matrix_b, with_second_block=True)
        if logical_ops_xz is not None:
            return logical_ops_xz
        _check_semisimple_route(matrix_a, matrix_b, mention_split_route=True)
        ring_logical_ops_xz = LPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return LPCode.to_lifted_logical_ops(*ring_logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a lifted product code.

        Generalizes HGPCode.get_canonical_logical_line_ops.  These are the ring-valued operators of
        the semisimple route of ``LPCode.get_canonical_logical_line_ops``, which describes the rings
        that this route needs, and the side effects of this route.  This method raises a
        NotImplementedError if either ring is not a group algebra, and a ValueError for a group
        algebra whose group order the field characteristic divides or for matrices over different
        group algebras.
        """
        _check_semisimple_route(matrix_a, matrix_b)

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
    - https://arxiv.org/abs/2607.28795
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

        These operators are built by the routes of ``LPCode.get_canonical_logical_line_ops``,
        which describes the matrices that each route needs, and the side effects and errors of each
        route.  The operators of the split route are those of ``LPCode(matrix_a, matrix_b)``,
        restricted to its first block of qudits.
        """
        # the split route declines semisimple group algebras, so the semisimple route takes
        # precedence
        logical_ops_xz = _get_split_logical_line_ops(matrix_a, matrix_b, with_second_block=False)
        if logical_ops_xz is not None:
            return logical_ops_xz
        _check_semisimple_route(matrix_a, matrix_b, mention_split_route=True)
        ring_logical_ops_xz = SLPCode.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        return SLPCode.to_lifted_logical_ops(*ring_logical_ops_xz)

    @staticmethod
    def get_canonical_ring_logical_line_ops(
        matrix_a: abstract.RingArray, matrix_b: abstract.RingArray
    ) -> tuple[abstract.RingArray, abstract.RingArray]:
        """Canonical ring-logical line operators for a subsystem lifted product code.

        Generalizes SHPCode.get_canonical_logical_line_ops.  These are the ring-valued operators of
        the semisimple route of ``LPCode.get_canonical_logical_line_ops``, which describes the rings
        that this route needs, and the side effects of this route.  This method raises the errors of
        ``LPCode.get_canonical_ring_logical_line_ops``.
        """
        _check_semisimple_route(matrix_a, matrix_b)
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


def _check_semisimple_route(
    matrix_a: abstract.RingArray,
    matrix_b: abstract.RingArray,
    *,
    mention_split_route: bool = False,
) -> None:
    """Raise an error unless two matrices fit the semisimple route of canonical logical operators.

    The semisimple route of ``LPCode.get_canonical_logical_line_ops`` uses semisimple Howell normal
    forms, which are implemented only for group algebras, and need a field characteristic that does
    not divide the group order.  This function raises a NotImplementedError if either matrix is not
    over a group algebra, and then a ValueError for matrices over different group algebras, or over
    a group algebra that is not semisimple.  Callers run it before computing null spaces, so a ring
    that is not a group algebra raises a NotImplementedError rather than the ValueError of a null
    space over a ring without an involution.  If ``mention_split_route`` is True, the errors for a
    ring that is not a group algebra or not semisimple also say that the matrices do not fit the
    split route, for callers that tried it first.
    """
    split_route_hint = (
        ", and the seed matrices do not fit the split route (see"
        " LPCode.get_canonical_logical_line_ops)"
        if mention_split_route
        else ""
    )
    ring_a, ring_b = matrix_a.ring, matrix_b.ring
    if not isinstance(ring_a, abstract.GroupRing) or not isinstance(ring_b, abstract.GroupRing):
        ring = ring_b if isinstance(ring_a, abstract.GroupRing) else ring_a
        raise NotImplementedError(
            "The semisimple route of canonical logical operators of lifted products is implemented"
            f" only for group algebras, not {ring}{split_route_hint}"
        )
    _check_same_ring(matrix_a, matrix_b)
    if not ring_a.is_semisimple:
        raise ValueError(
            "The semisimple route of canonical logical operators of lifted products needs a field"
            f" characteristic that does not divide the group order, unlike that of {ring_a}"
            f"{split_route_hint}"
        )


def _set_canonical_logical_line_ops(code: LPCode | SLPCode) -> None:
    """Set the canonical logical line operators of a lifted product code."""
    ring = code.matrix_a.ring
    try:
        logical_ops_xz = code.get_canonical_logical_line_ops(code.matrix_a, code.matrix_b)
    except (ValueError, NotImplementedError) as exception:
        raise ValueError(
            f"Cannot set canonical logical operators for this code: {exception}"
        ) from exception

    # the split route requires an involution that transposes lifts, and the semisimple route
    # assumes one, so invalid operators are a bug if the involution transposes lifts, and otherwise
    # they can come from the semisimple route
    if _involution_transposes_lifts(ring):
        code.set_logical_ops_xz(*logical_ops_xz)
        return
    try:
        code.set_logical_ops_xz(*logical_ops_xz)
    except ValueError as exception:
        raise ValueError(
            "Cannot set canonical logical operators for this code, since the involution of its ring"
            " does not transpose lifts"
        ) from exception


def _get_split_logical_line_ops(
    matrix_a: abstract.RingArray, matrix_b: abstract.RingArray, *, with_second_block: bool
) -> tuple[galois.FieldArray, galois.FieldArray] | None:
    """Canonical logical line operators of the split route, or None for matrices outside it.

    See ``LPCode.get_canonical_logical_line_ops`` for the split route.  This function returns None
    for matrices over semisimple group algebras, which the semisimple route takes, and for other
    matrices that the split route does not support.  If ``with_second_block`` is True, the operators
    also cover the second block of qudits of an LPCode, on which they vanish.
    """
    ring = matrix_a.ring
    if (
        matrix_b.ring != ring
        or (isinstance(ring, abstract.GroupRing) and ring.is_semisimple)
        or not (ring.is_commutative or isinstance(ring, abstract.GroupRing))
        or not ring.has_involution
        or not _involution_transposes_lifts(ring)
    ):
        return None

    # lift the checks on one column and on one row of the first block of qudits, using the actions
    # of the lifted product
    one = np.ones((1, 1), dtype=int)
    checks_a = abstract.kron(matrix_a, one).lift()
    checks_b = abstract.kron(one, matrix_b).lift()
    logical_ops_xz = _get_paired_grid_line_ops(checks_a, checks_b, ring.lift_dim)
    if logical_ops_xz is None or not with_second_block:
        return logical_ops_xz

    # the operators vanish on the second block of qudits of an LPCode
    second_block_size = len(matrix_a) * len(matrix_b) * ring.lift_dim
    zeros = ring.field.Zeros((len(logical_ops_xz[0]), second_block_size))
    logical_ops_x, logical_ops_z = (np.hstack([ops, zeros]) for ops in logical_ops_xz)
    return logical_ops_x.view(ring.field), logical_ops_z.view(ring.field)  # views for type checks


def _get_paired_grid_line_ops(
    checks_a: galois.FieldArray, checks_b: galois.FieldArray, block_size: int
) -> tuple[galois.FieldArray, galois.FieldArray] | None:
    """Paired X and Z line operators on a grid of blocks, or None if a pivot block is singular.

    Let ``checks_a`` and ``checks_b`` have shapes ``(ma * block_size, na * block_size)`` and
    ``(mb * block_size, nb * block_size)``.  The grid has shape ``(na, nb, block_size)``, flattened
    in row-major order.  Z operators are supported on one column of the grid, where they must
    satisfy ``checks_a``.  X operators are supported on one row, where they must satisfy
    ``checks_b``.  The pivot block of each check matrix is its last square block of columns.

    If both pivot blocks are invertible, then for each free cell ``(i, j, g)``, with
    ``i < na - ma`` and ``j < nb - mb``, this function returns the X operator on row ``i`` and the
    Z operator on column ``j`` that equal the unit vector at ``(i, j, g)`` on free cells of that
    line.  Their values on pivot cells solve the checks.  An X operator on row ``i`` and a Z
    operator on column ``j'`` meet only at the free cells ``(i, j', g')``, so the X operator for
    ``(i, j, g)`` and the Z operator for ``(i', j', g')`` overlap only if they are the same cell,
    and ``logical_ops_x @ logical_ops_z.T`` is the identity.

    Returns:
        The X operators and Z operators, both ordered by ``(i, j, g)``, or None if a check matrix
        has more rows than columns or its pivot block is not invertible.
    """
    # on one line, a pivot block is invertible if and only if the reduced null space of the checks
    # is the identity on free coordinates, followed by the values that solve the checks
    line_ops = []
    for checks in (checks_a, checks_b):
        num_free = checks.shape[1] - len(checks)
        if num_free < 0:
            return None
        null_space = checks.null_space().row_reduce()
        if not np.array_equal(null_space[:, :num_free], np.eye(num_free, dtype=int)):
            return None
        line_ops.append(null_space)
    line_ops_z, line_ops_x = line_ops

    # place each X line operator on a free row of the grid
    field = type(checks_a)
    num_rows, num_cols = checks_a.shape[1] // block_size, checks_b.shape[1] // block_size
    num_free_rows = len(line_ops_z) // block_size
    num_free_cols = len(line_ops_x) // block_size
    logical_ops_x = np.kron(field.Identity(num_rows)[:num_free_rows], line_ops_x).view(field)

    # place each Z line operator on a free column of the grid
    logical_ops_z = field.Zeros(
        (num_free_rows, num_free_cols, block_size, num_rows, num_cols, block_size)
    )
    line_ops_z_grid = line_ops_z.reshape(num_free_rows, block_size, num_rows, block_size)
    for col in range(num_free_cols):
        logical_ops_z[:, col, :, :, col, :] = line_ops_z_grid
    grid_size = num_rows * num_cols * block_size
    return logical_ops_x, logical_ops_z.reshape(len(logical_ops_x), grid_size)
