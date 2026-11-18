# SPDX-License-Identifier: Apache-2.0

"""Quantum Tanner codes.

A quantum Tanner code (QTCode) is a CSS code built from two symmetric subsets A and B of a group G,
and two classical local codes C_A and C_B, with one coordinate per member of A and of B,
respectively.  Its qudits are the faces of a left-right Cayley complex built from A and B, and its
parity checks sit at the vertices of the complex.  The vertices are members of a cover of G, and
fall into classes: four classes in the quadripartite complex, which is the default, or two classes
in the bipartite complex (see help(qldpc.objects.CayleyComplex)).  For each vertex g with X-type
checks, a in A, and b in B, the complex has a face ``f(g,a,b) = {g, ag, gb, agb}``, whose corners g
and agb have X-type checks, and whose corners ag and gb have Z-type checks (see help(QTCode)).

A lifted quantum Tanner code (LQTCode) is built in the same way from an explicit group G, ordered
lists A and B of members of G that may repeat, and four classical local codes, two for each list.
The qudits of an LQTCode are the faces, which it calls squares, of a quadripartite complex with all
of G in each class of vertices.  An LQTCode is the code of Eq. (2) of
https://arxiv.org/abs/2512.20532, in other notation (see LQTCode.from_lrz).  A quadripartite QTCode
has the qudits and stabilizer group of an LQTCode, or of a direct summand of an LQTCode if the
complex of the LQTCode is disconnected.  A bipartite QTCode is the quotient, by an involution, of an
LQTCode or of a direct summand of an LQTCode.  Under these correspondences, the parity checks of the
two codes agree only for some choices of local checks (see help(LQTCode)).

Local views.  Write ``f(v,a,b) = {v, av, vb, avb}`` for any vertex v.  Every face at a vertex v of a
QTCode is ``f(v,a,b)`` for exactly one pair (a, b) of members of A and B, which is the label of the
face at v.  So the face ``f(g,a,b)`` has the label (a, b) at its corner g, and the label (a^-1, b)
at its corner ag, since it is ``f(ag,a^-1,b)``.  Coordinate (i, j) of the local view of a vertex is
the face whose label is ``(members_a[i], members_b[j])``, where ``members_a`` and ``members_b`` list
the members of A and B in increasing order.  So the local view of a vertex is an ``|A| x |B|`` array
of qudits (QTCode.get_local_views).  In the coordinates of their local views, the X-type checks at a
vertex span the local check space ``C_A ⨂ C_B``, and the Z-type checks span ``~C_A ⨂ ~C_B``, where
``~C`` is the dual code of C.  LQTCode labels squares by positions in A and B instead (see
help(LQTCode)).

Local checks.  The checks at each vertex are the words of a spanning set of its local check space.
The argument ``local_checks`` of QTCode and LQTCode chooses this set:

- "min_weight" (the default): a minimum-weight basis, whose i-th lightest word is no heavier than
  the i-th lightest word of any basis of the space, for each i.
- "from_codes": the rows of ``G_A ⨂ G_B`` for X-type checks, and of ``H_A ⨂ H_B`` for Z-type checks.
  Here H_A and G_A are the parity check matrix and generator matrix that C_A carries
  (``code.matrix`` and ``code.generator``), and likewise for C_B.  Rows of these matrices that are
  linearly dependent give redundant checks.

Both choices span the same local check space at every vertex, so they give the same stabilizer
group, number of qudits, and dimension.

The "min_weight" checks that span ``C_A ⨂ C_B`` are the products ``u_i ⨂ v_j`` of minimum-weight
bases {u_i} of C_A and {v_j} of C_B, which qldpc.math.get_min_weight_basis finds for each factor,
and likewise for ``~C_A ⨂ ~C_B``.  These products form a minimum-weight basis of the space, as
follows.  Write ``|x|`` for the weight of a word x.  A basis {w_k} of a space is a minimum-weight
basis iff every word ``x = sum_k c_k w_k`` has ``|x| >= |w_k|`` whenever ``c_k != 0``.  This
condition is necessary: if ``|x| < |w_k|`` for some ``c_k != 0``, replacing w_k by x gives a lighter
basis.  It is also sufficient, since it says that the words of the basis of weight at most m span
all words of weight at most m, for every m, so no basis has more words of weight at most m.  Now let
``y = sum_ij c_ij u_i ⨂ v_j``, and read y as a matrix whose row r is ``sum_j x_j[r] v_j``, where
``x_j = sum_i c_ij u_i``.  Fix indices i and j with ``c_ij != 0``.  Then ``|x_j| >= |u_i|``, since
x_j has the coefficient c_ij on u_i.  Each of the ``|x_j|`` rows r with ``x_j[r] != 0`` has a
nonzero coefficient on v_j, so it has weight at least ``|v_j|``.  So
``|y| >= |u_i| |v_j| = |u_i ⨂ v_j|`` whenever ``c_ij != 0``, and by the condition above,
``{u_i ⨂ v_j}`` is a minimum-weight basis.  The checks are minimum-weight bases of the local check
spaces, but generating sets of the stabilizer group that combine checks at different vertices may be
lighter.

Canonical local checks.  With the optional pybliss package, bliss finds qudit permutations that
preserve a code (see QuditCode.get_known_permutation_automorphism_group), but only those that map a
set of checks onto itself.  The parity checks of QTCode and LQTCode are one spanning set of each
local check space, chosen by local_checks, so a permutation that preserves the code can map parity
checks to other words of these spaces, and then bliss cannot find it among the parity checks.
QTCode and LQTCode therefore also have bliss search their canonical local checks, which depend only
on the local check spaces.

The canonical local checks of a vertex are the nonzero words of weight at most w_max of its local
check space, placed on its local view, where w_max is the largest weight of a word in a
minimum-weight basis of the space.  All minimum-weight bases of a space have the same weights, so
w_max depends only on the space.  By the argument above, w_max for ``C_A ⨂ C_B`` is the product of
the largest weights of words in minimum-weight bases of C_A and C_B, and likewise for
``~C_A ⨂ ~C_B``.

- The canonical checks of a vertex include a minimum-weight basis of its local check space, so they
  span it, and the canonical checks of all vertices span the row spaces of matrix_x and matrix_z.
  So a permutation of the qudits that maps the set of canonical checks onto itself preserves the
  code.
- Suppose that a permutation of the qudits maps the local view of each vertex v to the local view of
  a vertex v' with checks of the same type, and maps the local check space of v onto that of v'.  It
  preserves weights, so it maps the canonical checks of v onto those of v', even if it does not map
  parity checks to parity checks.  So bliss can find such permutations among the canonical checks.

Two caps bound the cost of the canonical checks, and the docstrings of QTCode and LQTCode give their
values:

- a cap on the number of words of a local check space, since finding its canonical checks
  enumerates its words; and
- a cap on the number of canonical checks per vertex, since the matrix that bliss searches has a row
  for each canonical check at each vertex.

For each type of checks, if a local check space of that type exceeds either cap, bliss searches the
parity checks of that type in place of the canonical checks of that type.
"""

from __future__ import annotations

import abc
import ast
import functools
import heapq
import itertools
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, get_args

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.linalg

from qldpc import abstract
from qldpc.math import get_min_weight_basis
from qldpc.objects import (
    PAULIS_XZ,
    CayleyComplex,
    Pauli,
    PauliXZ,
    PauliXZLike,
)

from ..classical import (
    TannerCode,
)
from ..common import (
    ClassicalCode,
    CSSCode,
    _get_scrambled_seed,
    _mark_read_only,
    _preserves_row_space,
)

LocalChecks = Literal["min_weight", "from_codes"]
"""The choices of the checks that span each local check space (see the module docstring)."""

# the classes (s, t) of the vertices of an LQTCode with checks of each type, in the order of their
# rows in matrix_x and matrix_z
_VERTEX_CLASSES: dict[PauliXZ, tuple[tuple[int, int], ...]] = {
    Pauli.X: ((0, 0), (1, 1)),
    Pauli.Z: ((0, 1), (1, 0)),
}

# the most words that a local code or its dual may have for "min_weight" local checks
_MAX_WORDS_FOR_MIN_WEIGHT_BASIS = 2**20

# The most words in a local check space, and the most canonical local checks at each vertex, for
# which a QTCode or an LQTCode has bliss search its canonical checks of a type, as well as its
# parity checks.  The first limit bounds the time to enumerate words.  The second bounds the size of
# the matrix that bliss searches, which has a row for each canonical check at each vertex.
_MAX_WORDS_FOR_CANONICAL_CHECKS = 2**20
_MAX_CANONICAL_CHECKS_PER_VERTEX = 2**10

# the number of words of a local check space that _get_canonical_local_checks enumerates at a time
_WORDS_PER_CHUNK = 2**14

####################################################################################################
# quantum Tanner codes


class _SquareComplexCode(CSSCode):
    """A CSS code on a square complex, whose checks at each vertex span a product of local codes.

    The local check space of each vertex is the tensor product of two local codes, or of their
    duals, read on the local view of the vertex (see help(qldpc.codes.quantum.quantum_tanner)).
    This class finds the canonical local checks of the code from the blocks of vertices that
    _get_vertex_blocks gives, and has bliss search them as well as the parity checks.
    """

    @abc.abstractmethod
    def _get_vertex_blocks(
        self, pauli: PauliXZ
    ) -> list[tuple[ClassicalCode, ClassicalCode, npt.NDArray[np.int_]]]:
        """The blocks of vertices with checks of one type, whose vertices share their local codes.

        Each block is given by its local codes C_A and C_B, and by the local views of its vertices,
        as an array whose entry ``[v, i, j]`` is the qudit at coordinate (i, j) of the local view of
        its v-th vertex.  Each block of vertices gives a block of rows of matrix_x or matrix_z, and
        the blocks are in the order of these rows.
        """

    def _get_permutation_automorphism_search_matrices(self) -> tuple[galois.FieldArray, ...]:
        """The parity checks of this code and its canonical local checks, for bliss to search.

        Bliss can find more automorphisms among the canonical checks (see
        help(qldpc.codes.quantum.quantum_tanner)), but it can also find automorphisms of the parity
        checks that are not automorphisms of the canonical checks, such as when a local code is the
        direct sum of codes on disjoint sets of coordinates.  So bliss searches both.
        """
        canonical_checks = self._get_canonical_checks()
        if np.array_equal(canonical_checks, self.matrix):
            return (self.matrix,)
        return (self.matrix, canonical_checks)

    def _get_canonical_checks(self) -> galois.FieldArray:
        """The canonical local checks of this code, in the layout of its parity check matrix.

        The module docstring defines the canonical local checks, and shows that they span the row
        spaces of matrix_x and matrix_z.  For each type of checks, if a block of vertices with that
        type has a local check space with more than _MAX_WORDS_FOR_CANONICAL_CHECKS words, or more
        than _MAX_CANONICAL_CHECKS_PER_VERTEX canonical checks at each vertex, the returned matrix
        holds the parity checks of that type instead, which depend on the choice of local checks.
        """
        blocks = []
        for pauli, parity_checks in zip(PAULIS_XZ, [self.matrix_x, self.matrix_z]):
            checks = self._get_canonical_checks_of_type(pauli)
            blocks.append(parity_checks if checks is None else checks)
        return scipy.linalg.block_diag(*blocks).view(self.field)

    def _get_canonical_checks_of_type(self, pauli: PauliXZ) -> galois.FieldArray | None:
        """The canonical local checks of one type at all vertices, or None if there are too many."""
        blocks = []
        for code_a, code_b, local_views in self._get_vertex_blocks(pauli):
            # find the canonical checks of one local view of the block, or give up
            local_checks = _get_canonical_local_checks(code_a, code_b, pauli)
            if local_checks is None:
                return None

            # place them on the local view of every vertex of the block
            blocks.append(_place_local_checks(local_views, local_checks, len(self)))
        return np.vstack(blocks).view(self.field)


class QTCode(_SquareComplexCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    A quantum Tanner code is defined by:

    - two symmetric subsets A and B of a group G, which hold the inverse of each of their members
      and are not both empty, and
    - two classical local codes C_A and C_B, with block lengths ``|A|`` and ``|B|``, respectively.

    The qudits of a quantum Tanner code are the faces of a left-right Cayley complex built from A
    and B.  The vertices of the complex are members of a cover of G, on which members of A act from
    the left and members of B from the right (see help(qldpc.objects.CayleyComplex)).  By default,
    the complex is quadripartite: each vertex has a class (s, t) in {0,1} x {0,1}, multiplying a
    vertex by a member of A flips s, and multiplying it by a member of B flips t.  Vertices of
    classes (0, 0) and (1, 1) carry X-type checks, and those of classes (1, 0) and (0, 1) carry
    Z-type checks.  If bipartite is True, the complex is bipartite: each vertex has a class s in
    {0,1}, multiplying a vertex by a member of A or of B flips s, and vertices of class 0 carry
    X-type checks.  Either way, QTCode keeps only the connected component of the complex that
    contains the identity, whose class is (0, 0) or 0.  For each vertex g with X-type checks, a in
    A, and b in B, the complex has a face ``f(g,a,b) = {g, ag, gb, agb}``::

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

    The corners g and agb of the face carry X-type checks, and its corners ag and gb carry Z-type
    checks.  Each face is ``f(g,a,b)`` for two triples (g, a, b), one for each of its corners with
    X-type checks, so the code has ``|A| |B| / 2`` qudits for each vertex with X-type checks.

    The X-type checks at each vertex span the local check space ``C_A ⨂ C_B``, and the Z-type checks
    span ``~C_A ⨂ ~C_B``, where ``~C`` is the dual code of C.  These spaces are read in the
    coordinates of the local view of the vertex, an ``|A| x |B|`` array of the faces at the vertex
    (see QTCode.get_local_views and help(qldpc.codes.quantum.quantum_tanner)).  Coordinate i of C_A
    belongs to member i of members_a, which lists the members of A in increasing order.  Likewise,
    coordinate j of C_B belongs to member j of members_b.  QTCode.from_ordered_members takes local
    codes whose coordinates follow another order of the members.

    The argument local_checks chooses the checks that span each local check space: a minimum-weight
    basis ("min_weight", the default), or products of the rows of the matrices that the local codes
    carry ("from_codes").  The module docstring describes these choices, and proves that
    "min_weight" gives minimum-weight bases.  The choice can change the parity checks, but not the
    stabilizer group.  The "min_weight" checks at each vertex are as light as any basis of its local
    check space, though generating sets of the stabilizer group that combine checks at different
    vertices can be lighter.  The "min_weight" checks are also not chosen to minimize the number of
    checks that act on each qudit.

    QTCode is a CSS stabilizer code over the field of its local codes, which can be any finite
    field.  Its X-type and Z-type checks commute: the local views of two vertices with checks of
    different types share either no faces, or exactly one row of each local view, or exactly one
    column of each, where words of a local code meet words of its dual (help(LQTCode) gives the
    argument in detail).  QTCode records no distance.  The constructor merges repeated members of A
    and B.  It checks that A and B are symmetric and not both empty, that the local codes are over
    one field and have one coordinate per member of their subsets, and, for the bipartite complex,
    that A and B satisfy the Total No Conjugacy condition (see help(qldpc.objects.CayleyComplex)).
    With "min_weight" local checks, it enumerates the words of each local code and of its dual,
    which takes time exponential in their dimensions, so it checks that each of them has at most
    2^20 words.

    Two QTCodes are equal if they have the same generating data and the same parity checks.
    QTCode.save records the choice of local checks, and the generator matrices of the local codes.
    For lists of members that repeat or are not symmetric, four local codes, or a complex on all of
    a given group, use LQTCode, whose docstring also relates QTCodes to it.

    Symmetries: QTCode.get_known_permutation_automorphism_group gives a group of qudit permutations
    that preserve the code.  Its generators include translations of the Cayley complex.  Left
    translation by a vertex x with X-type checks multiplies every vertex by x from the left.  It is
    included if conjugation by x permutes A, and the induced permutation of the coordinates of C_A
    maps C_A onto itself.  Right translation by x multiplies every vertex by x from the right.  It
    is included if conjugation by x permutes B, and the induced permutation of the coordinates of
    C_B maps C_B onto itself.  With the optional pybliss package, the generators also include the
    automorphisms that bliss finds among the parity checks of the code, and among its canonical
    local checks (see help(qldpc.codes.quantum.quantum_tanner)).  If the local check spaces of
    X-type or Z-type checks have more than 2^20 words, or more than 2^10 canonical checks at each
    vertex, bliss searches no canonical checks of that type.  The group can omit other symmetries,
    such as those from automorphisms of the group G or from exchanging A and B.  It contains no
    ZX-dualities, which exchange X-type and Z-type operators.

    Notes:

    - Asymptotically good families of quantum Tanner codes use projective special linear groups and
      random local codes (arXiv:2202.13641).

    References:

    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571
    - https://arxiv.org/abs/2202.13641
    - https://arxiv.org/abs/2508.05095 (explicit instances)
    """

    complex: CayleyComplex

    def __init__(
        self,
        subset_a: Iterable[abstract.GroupMember],
        subset_b: Iterable[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        bipartite: bool = False,
        local_checks: LocalChecks = "min_weight",
    ) -> None:
        """Construct a quantum Tanner code.

        Args:
            subset_a: The members of the symmetric subset A of a group.  Repeated members are
                merged.
            subset_b: The members of the symmetric subset B of the same group.  Repeated members are
                merged.
            code_a: The local code C_A, or its parity check matrix, with one coordinate per member
                of A.
            code_b: The local code C_B, or its parity check matrix, with one coordinate per member
                of B.  By default, ``~C_A``, whose generator matrix is the parity check matrix of
                C_A, if A and B have equally many members.
            field: The field of the local codes, if they are given as matrices.
            bipartite: Whether to build the code on the bipartite Cayley complex, rather than the
                quadripartite one.
            local_checks: The checks that span each local check space: "min_weight" (the default) or
                "from_codes" (see help(qldpc.codes.quantum.quantum_tanner)).  Pass "from_codes" for
                local codes that have too many words for "min_weight".

        Raises:
            ValueError: If any of the following holds:

                - A or B is not symmetric, or both are empty.
                - bipartite is True, and A and B do not satisfy the Total No Conjugacy condition.
                - code_b is not given, and A and B have different numbers of members.
                - The local codes are over different fields, or not over the given field.
                - A local code does not have one coordinate per member of its subset.
                - local_checks is not a choice of local checks.
                - local_checks is "min_weight", and a local code or its dual has more than 2^20
                  words.
        """
        local_checks = _parse_local_checks(local_checks)

        # build the complex first, since it merges repeated members of the subsets
        self.complex = CayleyComplex(subset_a, subset_b, bipartite=bipartite)

        # read the local codes; C_B defaults to the dual of C_A if the subsets have equal sizes
        num_members_a, num_members_b = len(self.complex.subset_a), len(self.complex.subset_b)
        if code_b is None and num_members_a != num_members_b:
            raise ValueError(
                "Underspecified generating data for quantum Tanner code: code_b is required when"
                f" the subsets A and B have different numbers of distinct members ({num_members_a}"
                f" and {num_members_b})"
            )
        code_a, code_b = _read_local_codes(code_a, code_b, field)

        # check that the local codes share a field, and each have one coordinate per member of their
        # subset
        if field is None and code_a.field is not code_b.field:
            raise ValueError("The local codes provided for this QTCode are over different fields")
        for name, code, subset in [
            ("A", code_a, self.complex.subset_a),
            ("B", code_b, self.complex.subset_b),
        ]:
            if len(code) != len(subset):
                raise ValueError(
                    f"The local code C_{name} has block length {len(code)}, but the subset {name}"
                    f" has {len(subset)} distinct members"
                )

        # place the local checks at every vertex of the complex
        self.code_a = code_a
        self.code_b = code_b
        self._local_checks = local_checks
        code_x, code_z = self.get_subcodes(self.complex, code_a, code_b, local_checks=local_checks)
        super().__init__(code_x, code_z, field)
        self._is_subsystem_code = False  # the checks commute (see help(QTCode))

    def __eq__(self, other: object) -> bool:
        if not (
            isinstance(other, QTCode)
            and self.code_a == other.code_a
            and self.code_b == other.code_b
            and self.complex.bipartite == other.complex.bipartite
        ):
            return False

        # The members of the two complexes may be written on different numbers of points, and SymPy
        # permutations on different numbers of points are never equal.  Read all members on the
        # largest number of points.
        subsets = [
            self.complex.subset_a,
            self.complex.subset_b,
            other.complex.subset_a,
            other.complex.subset_b,
        ]
        size = max(member.size for subset in subsets for member in subset)
        subset_a, subset_b, other_subset_a, other_subset_b = [
            {_get_member_on_points(member, size) for member in subset} for subset in subsets
        ]

        # equal generating data can give different parity checks, depending on local_checks
        return (
            subset_a == other_subset_a
            and subset_b == other_subset_b
            and np.array_equal(self.matrix_x, other.matrix_x)
            and np.array_equal(self.matrix_z, other.matrix_z)
        )

    @property
    def members_a(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the subset A, in increasing order.

        Member i labels coordinate i of C_A (see help(qldpc.codes.quantum.quantum_tanner)).
        """
        return tuple(sorted(self.complex.subset_a))

    @property
    def members_b(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the subset B, in increasing order.

        Member j labels coordinate j of C_B (see help(qldpc.codes.quantum.quantum_tanner)).
        """
        return tuple(sorted(self.complex.subset_b))

    def get_local_views(self, pauli: PauliXZLike) -> npt.NDArray[np.int_]:
        """The local views of the vertices with parity checks of one type.

        Entry ``[v, i, j]`` of the returned array is the qudit at coordinate (i, j) of the local
        view of the v-th vertex with checks of the given type: the face whose label at that vertex
        is ``(members_a[i], members_b[j])`` (see help(qldpc.codes.quantum.quantum_tanner)).  The
        checks of each vertex are consecutive rows of ``matrix_x`` or ``matrix_z``, and the vertices
        appear in the order of these rows.  Every vertex has the same number k of checks, so the
        checks of the v-th vertex are the rows ``v k`` to ``v k + k - 1``.

        Args:
            pauli: Pauli.X for the vertices with X-type checks, or Pauli.Z for those with Z-type
                checks.

        Returns:
            A read-only integer array of shape ``(number of vertices, |A|, |B|)``.
        """
        # mark the cached array here, since deep copies and unpickling of a code make it writable
        return _mark_read_only(self._local_views[Pauli.coerce_xz(pauli)])

    @staticmethod
    def get_subcodes(
        cayplex: CayleyComplex,
        code_a: ClassicalCode,
        code_b: ClassicalCode,
        *,
        local_checks: LocalChecks = "min_weight",
    ) -> tuple[TannerCode, TannerCode]:
        """Get the classical Tanner subcodes of a quantum Tanner code.

        Return the classical Tanner codes on subgraph_x and subgraph_z (see QTCode.get_subgraphs),
        whose parity checks are the X-type and Z-type checks of the quantum Tanner code.  Each
        places the local checks of its type at every source, so its subcode, in the sense of
        TannerCode, is ``~(C_A ⨂ C_B)`` or ``~(~C_A ⨂ ~C_B)``.  The argument local_checks is as for
        QTCode.
        """
        local_checks = _parse_local_checks(local_checks)
        if local_checks == "min_weight":
            _check_min_weight_sizes([code_a, code_b])
        subgraph_x, subgraph_z = QTCode.get_subgraphs(cayplex)

        # the subcodes in the sense of TannerCode: the classical codes whose parity checks are the
        # local checks
        checks_x, checks_z = (
            ClassicalCode(_get_local_checks(code_a, code_b, pauli, local_checks))
            for pauli in PAULIS_XZ
        )
        return TannerCode(subgraph_x, checks_x), TannerCode(subgraph_z, checks_z)

    @staticmethod
    def get_subgraphs(cayplex: CayleyComplex) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs of the inner (classical) Tanner codes for a quantum Tanner code.

        The vertices of the complex are members of a cover of the group.  Each member a of A acts on
        the cover as its image, (a,1,0) in the quadripartite complex or (a,1) in the bipartite
        complex, and each member b of B as (b,0,1) or (b,1) (see CayleyComplex.cover_subset_a and
        CayleyComplex.cover_subset_b).  Below, a and b stand for these images.  The vertices with
        X-type checks are the identity and the vertices that it reaches through steps ``g -> agb``
        (see help(QTCode)).  For each vertex g with X-type checks, a in A, and b in B, the complex
        has a face ``f(g,a,b) = {g, ag, gb, agb}``::

             g ―――――――――― gb

             |  f(g,a,b)  |

            ag ――――――――― agb

        The (directed) subgraphs are:

        - subgraph_x, with an edge ``(g, f(g,a,b))`` for each such triple (g, a, b), and
        - subgraph_z, with an edge ``(ag, f(g,a,b))`` for each such triple (g, a, b).

        Each edge has a "sort" attribute, which is the label of the face at the source of the edge:
        (a, b) for the edge ``(g, f(g,a,b))``, and ``(a^-1, b)`` for the edge ``(ag, f(g,a,b))``,
        since ``f(g,a,b) = f(ag,a^-1,b)``.  Classical Tanner codes on these subgraphs order the
        edges at each source by their labels, with members in increasing order, as in
        QTCode.members_a and QTCode.members_b.  Each face also has a "sort" attribute, which numbers
        the faces in the order in which they are found.  This fixes the order of the bits of the
        classical Tanner codes, and so of the qudits of the quantum Tanner code.
        """
        # sort the subsets by the total order on the group, which fixes the order in which faces are
        # added below, and thereby the qudit that each face of the Cayley complex is identified with
        subset_a = sorted(cayplex.cover_subset_a)
        subset_b = sorted(cayplex.cover_subset_b)

        # identify the identity element
        member = next(itertools.chain(subset_a, subset_b))
        identity = member * ~member

        # Build the subgraphs one node at a time, always adding the faces of the least node that has
        # been found but not yet visited.  The heap holds the nodes that are waiting to be visited.
        subgraph_x = nx.DiGraph()
        subgraph_z = nx.DiGraph()
        nodes_found = {identity}
        nodes_to_visit = [identity]
        num_faces = 0
        while nodes_to_visit:
            gg = heapq.heappop(nodes_to_visit)

            # add all faces adjacent to this node, numbering each face when it is first found
            for aa, bb in itertools.product(subset_a, subset_b):
                aa_gg, gg_bb = aa * gg, gg * bb
                aa_gg_bb = aa_gg * bb
                face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
                if face not in subgraph_x:
                    subgraph_x.add_node(face, sort=num_faces)
                    subgraph_z.add_node(face, sort=num_faces)
                    num_faces += 1
                subgraph_x.add_edge(gg, face, sort=(aa, bb))
                subgraph_z.add_edge(aa_gg, face, sort=(~aa, bb))

                # queue the corner agb, which is a source of subgraph_x like gg, if it is new
                if aa_gg_bb not in nodes_found:
                    nodes_found.add(aa_gg_bb)
                    heapq.heappush(nodes_to_visit, aa_gg_bb)

        return subgraph_x, subgraph_z

    @staticmethod
    def from_ordered_members(
        members_a: Sequence[abstract.GroupMember],
        members_b: Sequence[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        bipartite: bool = False,
        local_checks: LocalChecks = "min_weight",
    ) -> QTCode:
        """Construct a quantum Tanner code whose local codes label coordinates by given members.

        Coordinate i of code_a belongs to members_a[i], and coordinate j of code_b to members_b[j],
        whereas QTCode labels the coordinates of its local codes by the members of each subset in
        increasing order.  This method permutes the columns of the local codes, and of the generator
        matrices that they carry, into that order, and returns the QTCode of the permuted codes,
        which are its code_a and code_b.

        Args:
            members_a: The distinct members of the symmetric subset A of a group, in the order of
                the coordinates of code_a.
            members_b: The distinct members of the symmetric subset B of the same group, in the
                order of the coordinates of code_b.
            code_a: The local code C_A, or its parity check matrix.
            code_b: The local code C_B, or its parity check matrix.  By default, ``~C_A``, whose
                generator matrix is the parity check matrix of C_A, and whose coordinate j belongs
                to members_b[j].
            field: The field of the local codes, if they are given as matrices.
            bipartite: Whether to build the code on the bipartite Cayley complex, as for QTCode.
            local_checks: The choice of local checks, as for QTCode.

        Raises:
            ValueError: If a list of members repeats a member (LQTCode allows repeats), or if a
                local code does not have one coordinate per member of its list.  Also in the cases
                that QTCode lists, such as lists of members that are not symmetric.
        """
        members_a, members_b = list(members_a), list(members_b)
        code_a, code_b = _read_local_codes(code_a, code_b, field)

        # sort the members of each list, and the coordinates of its local code alike
        size = max((member.size for member in members_a + members_b), default=0)
        sorted_codes = []
        for name, members, code in [("a", members_a, code_a), ("b", members_b, code_b)]:
            resized = [_get_member_on_points(member, size) for member in members]
            if len(set(resized)) != len(resized):
                raise ValueError(
                    f"members_{name} repeats a member, but QTCode needs distinct members; use"
                    " LQTCode for repeated members"
                )
            if len(code) != len(members):
                raise ValueError(
                    f"The local code C_{name.upper()} has block length {len(code)}, but members_{name}"
                    f" has {len(members)} members"
                )
            order = sorted(range(len(members)), key=resized.__getitem__)
            sorted_code = ClassicalCode(code.matrix[:, order])
            sorted_codes.append(sorted_code.set_generator(code.generator[:, order]))
        return QTCode(
            members_a,
            members_b,
            sorted_codes[0],
            sorted_codes[1],
            bipartite=bipartite,
            local_checks=local_checks,
        )

    @staticmethod
    def random(
        group: abstract.Group,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        bipartite: bool = False,
        one_subset: bool = False,
        exclude_identity: bool = False,
        seed: int | None = None,
        local_checks: LocalChecks = "min_weight",
    ) -> QTCode:
        """Construct a random quantum Tanner code from a group and local codes.

        Unless exclude_identity is True, the random subsets A and B can contain the identity element
        e of the group.  Then the faces ``f(g,e,b) = {g, eg, gb, egb}`` have vertices g and eg that
        are the same member of the group, and similarly for B.  The double or quadruple covers built
        by CayleyComplex keep these vertices apart, so the code is still defined on the
        quadripartite complex, or on the bipartite complex if A and B do not both contain e.

        Args:
            group: The group from which to draw the random symmetric subsets A and B.
            code_a: The local code C_A, or its parity check matrix.  A has one member per coordinate
                of C_A.
            code_b: The local code C_B, or its parity check matrix.  B has one member per coordinate
                of C_B.  By default, ``~C_A``, whose generator matrix is the parity check matrix of
                C_A.
            field: The field of the local codes, if they are given as matrices.
            bipartite: Whether to build the code on the bipartite Cayley complex, as for QTCode.
            one_subset: If True, draw only A, and use it as B as well, which needs local codes of
                the same length.  A = B never satisfies the Total No Conjugacy condition, since
                ``a e = e a``, so one_subset=True with bipartite=True always raises a ValueError.
            exclude_identity: If True, the random subsets exclude the identity element of the group,
                as the generating sets of Cayley graphs usually do.  See
                help(qldpc.abstract.Group.random_symmetric_subset) for the sizes of subsets without
                the identity.
            seed: A seed for the random subsets, from which the construction is reproducible.
            local_checks: The choice of local checks, as for QTCode.

        Raises:
            ValueError: If any of the following holds:

                - A local code has length 0.
                - The group has no symmetric subset whose size is the length of C_A, or of C_B
                  unless one_subset is True, without the identity if exclude_identity is True (see
                  help(qldpc.abstract.Group.random_symmetric_subset)).
                - QTCode raises a ValueError for the subsets and local codes, such as when
                  one_subset is True and C_B does not have one coordinate per member of A, or when
                  one_subset and bipartite are both True.
        """
        code_a, code_b = _read_local_codes(code_a, code_b, field)
        subset_a = group.random_symmetric_subset(
            code_a.num_bits, exclude_identity=exclude_identity, seed=seed
        )
        if one_subset:
            subset_b = subset_a
        else:
            # scramble the seed so that the second subset is drawn independently of the first, while
            # keeping the construction as a whole reproducible from the given seed.  Reduce it
            # first, so that a seed of any magnitude is accepted, as it is for the first subset.
            seed_b = _get_scrambled_seed(seed % 2**32) if seed is not None else None
            subset_b = group.random_symmetric_subset(
                code_b.num_bits, exclude_identity=exclude_identity, seed=seed_b
            )
        return QTCode(
            subset_a, subset_b, code_a, code_b, bipartite=bipartite, local_checks=local_checks
        )

    def save(self, path: str, *headers: str) -> None:
        """Save the generating data of this code to a file.

        The file records the subsets, the local codes, the order of the field, the complex, the
        choice of local checks, and the generator matrices of the local codes.  From these data,
        QTCode.load rebuilds an equal code.  In later releases, it rebuilds the same parity check
        matrices, with the same order of checks and qudits.  Since the file records only the order
        of the field, this method saves only codes over galois.GF(order), with its default
        irreducible polynomial and primitive element.  The members of each subset are written in
        sorted order, so that the file does not depend on the order in which the subsets list their
        members.  Equal codes can still give different files, if they chose their local checks
        differently, or if their local codes carry different generator matrices.

        Raises:
            ValueError: If the field of this code is not galois.GF(order), with its default
                irreducible polynomial and primitive element.  This method then writes nothing.
        """
        # refuse fields other than galois.GF(order), in which QTCode.load reads saved values
        order = self.field.order
        if self.field is not galois.GF(order):
            raise ValueError(
                "Saved files record only the order of the field, so QTCode.save saves only codes"
                f" over galois.GF({order}), with its default irreducible polynomial and primitive"
                " element, not over a field with irreducible polynomial"
                f" {self.field.irreducible_poly} and primitive element"
                f" {self.field.primitive_element}"
            )

        # convert subsets to arrays
        subset_a = np.array([gen.array_form for gen in sorted(self.complex.subset_a)])
        subset_b = np.array([gen.array_form for gen in sorted(self.complex.subset_b)])

        # create save directory if necessary
        save_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(save_dir, exist_ok=True)

        with open(path, "w") as file:
            # write provided headers
            for header in headers:
                file.writelines(f"# {line}\n" for line in header.splitlines())

            # write subsets
            file.write("# subset_a:\n")
            np.savetxt(file, subset_a, fmt="%d")
            file.write("# subset_b:\n")
            np.savetxt(file, subset_b, fmt="%d")

            # write seed codes
            file.write("# code_a.matrix:\n")
            np.savetxt(file, self.code_a.matrix, fmt="%d")
            file.write("# code_b.matrix:\n")
            np.savetxt(file, self.code_b.matrix, fmt="%d")

            # Record the data that only choose the parity checks on comment lines.  QTCode.load in
            # released versions up to 0.4.1 reads only the lines of integers between comment lines,
            # and the field and complex from the last two lines, so it reads these files as it reads
            # files without these lines.
            file.write(f"# local checks: {self._local_checks}\n")
            file.write(f"# code_a.generator: {self.code_a.generator.tolist()}\n")
            file.write(f"# code_b.generator: {self.code_b.generator.tolist()}\n")

            # write other data
            file.write(f"# base field: {order}\n")
            file.write(f"# bipartite: {bool(self.complex.bipartite)}\n")

    @staticmethod
    def load(path: str) -> QTCode:
        """Load a QTCode from a file written by QTCode.save.

        The loaded code chooses its local checks as the saved code did, if the file records that
        choice.  For files that record a choice of local checks, QTCode.load rebuilds the same
        parity check matrices in every later release, with the same order of checks and qudits.

        Files written by QTCode.save in releases up to 0.4.1 and the dev releases 0.4.2.dev20261004
        and 0.4.2.dev20261023 record no choice of local checks, and no generator matrices of the
        local codes, and load with "min_weight".  Their codes had local checks in reduced row
        echelon form, which span the same local check spaces.  So for a code over galois.GF(order),
        with its default irreducible polynomial and primitive element, a file of 0.4.0, 0.4.1, or
        these dev releases loads with the same stabilizer group, but can load with other parity
        checks.  Loading the file of a code over another representation of its field gives a code
        over galois.GF(order), which can differ from the saved code, or raises an error.

        Releases before 0.4.0 ordered qudits in a way that varied between sessions, so their files
        can also load with the qudits in another order.  Releases 0.0.11 to 0.0.17 also wrote a last
        line that lists conjugated qudits, whose X-type and Z-type operators the saved code
        exchanged.  QTCode conjugates no qudits, so loading such a file raises a ValueError unless
        the line is exactly "# conjugate: ()".
        """
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")

        with open(path, "r") as file:
            lines = file.read().splitlines()

        # Find the comment line that starts each section of generating data.  Headers come first and
        # may mimic these lines, so search backwards from the end of the file, looking for each
        # section's line above the line of the section after it.
        names = [
            "subset_a",
            "subset_b",
            "code_a.matrix",
            "code_b.matrix",
            "base field",
            "bipartite",
        ]
        starts: dict[str, int] = {}
        end = len(lines)
        for name in reversed(names):
            start = next(
                (
                    index
                    for index in range(end - 1, -1, -1)
                    if lines[index].startswith(f"# {name}:")
                ),
                None,
            )
            if start is None:
                raise ValueError(f"Saved code has no '# {name}:' line: {path}")
            starts[name] = end = start

        # Find the sections that choose the parity checks, which files written by QTCode.save since
        # it recorded them hold between the "code_b.matrix" and "base field" lines.
        choice_names = ["local checks", "code_a.generator", "code_b.generator"]
        choice_starts = {
            name: index
            for index in range(starts["code_b.matrix"] + 1, starts["base field"])
            for name in choice_names
            if lines[index].startswith(f"# {name}:")
        }
        if choice_starts:
            if missing := [name for name in choice_names if name not in choice_starts]:
                raise ValueError(f"Saved code has no '# {missing[0]}:' line: {path}")
            starts |= choice_starts

        # Releases 0.0.11 to 0.0.17 ended files with a "# conjugate:" line that lists conjugated
        # qudits, or "()" if there are none.  A QTCode cannot conjugate qudits, so raise unless the
        # line is exactly "# conjugate: ()".
        last_lines = lines[starts["bipartite"] + 1 :]
        if any(
            line.startswith("# conjugate:") and line != "# conjugate: ()" for line in last_lines
        ):
            raise ValueError(f"Saved code conjugates qudits, which a QTCode cannot do: {path}")

        def get_malformed_error(name: str) -> ValueError:
            """The error for a section of the file that QTCode.load cannot read."""
            return ValueError(f"Saved code has a malformed '# {name}:' section: {path}")

        def read_rows(name: str) -> list[list[int]]:
            """Read the rows of integers in a section, which can be empty or have empty rows.

            Raise a ValueError if a row holds anything other than integers.
            """
            next_start = min(start for start in starts.values() if start > starts[name])
            section_lines = lines[starts[name] + 1 : next_start]
            try:
                return [[int(value) for value in line.split()] for line in section_lines]
            except ValueError as error:
                raise get_malformed_error(name) from error

        def read_value(name: str) -> str:
            """Read the value on the line that starts a section, as "2" in "# base field: 2"."""
            return lines[starts[name]].removeprefix(f"# {name}:").strip()

        def read_literal(name: str, is_valid: Callable[[object], bool]) -> Any:
            """Read the Python literal on the line that starts a section.

            Raise a ValueError if the literal does not parse or is not valid.
            """
            # ast.literal_eval also raises MemoryError or RecursionError for some deeply nested
            # input
            try:
                value = ast.literal_eval(read_value(name))
            except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as error:
                raise get_malformed_error(name) from error
            if not is_valid(value):
                raise get_malformed_error(name)
            return value

        def is_integer_matrix(value: object) -> bool:
            """Whether a value is a list of lists of integers, as the rows of a matrix."""
            return isinstance(value, list) and all(
                isinstance(row, list) and all(type(entry) is int for entry in row) for row in value
            )

        def to_matrix(name: str, rows: list[list[int]], num_columns: int) -> npt.NDArray[np.int_]:
            """Convert rows of a section to a matrix with the given number of columns.

            Raise a ValueError if the rows do not form such a matrix, or hold integers too large
            for it.
            """
            try:
                return np.array(rows, dtype=int).reshape(len(rows), num_columns)
            except (OverflowError, ValueError) as error:
                raise get_malformed_error(name) from error

        def read_members(name: str) -> set[abstract.GroupMember]:
            """Read the members of a subset from their array forms, one per row of a section.

            Raise a ValueError if a row is not a permutation.
            """
            rows = read_rows(name)
            try:
                return {abstract.GroupMember(row) for row in rows}
            except ValueError as error:
                raise get_malformed_error(name) from error

        # construct subsets and generating codes
        field: int = read_literal("base field", lambda value: type(value) is int)
        # 0.4.1 wrote the argument bipartite as given, such as 1
        bipartite = bool(read_literal("bipartite", lambda value: type(value) in (bool, int)))
        subset_a, subset_b = read_members("subset_a"), read_members("subset_b")

        # a seed code without parity checks still has one bit per member of its subset
        code_a, code_b = [
            ClassicalCode(to_matrix(name, read_rows(name), len(subset)), field)
            for name, subset in [("code_a.matrix", subset_a), ("code_b.matrix", subset_b)]
        ]

        # restore the choice of local checks, and the generator matrices of the local codes, which
        # files without them leave to the defaults
        local_checks: LocalChecks = "min_weight"
        if choice_starts:
            for code, name in [(code_a, "code_a.generator"), (code_b, "code_b.generator")]:
                rows: list[list[int]] = read_literal(name, is_integer_matrix)
                code.set_generator(to_matrix(name, rows, len(code)))
            local_checks = _parse_local_checks(read_value("local checks"))
        return QTCode(
            subset_a, subset_b, code_a, code_b, bipartite=bipartite, local_checks=local_checks
        )

    @functools.cached_property
    def _local_views(self) -> dict[PauliXZ, npt.NDArray[np.int_]]:
        """The local views of the vertices with checks of each type."""
        # cover members compare by their members of G first, so these orders agree with members_a/b
        members_a = sorted(self.complex.cover_subset_a)
        members_b = sorted(self.complex.cover_subset_b)
        positions_a = {member: index for index, member in enumerate(members_a)}
        positions_b = {member: index for index, member in enumerate(members_b)}
        local_views = {}
        for pauli, subgraph in zip(PAULIS_XZ, self.get_subgraphs(self.complex)):
            # TannerCode sorts the sources of a subgraph, which carry its checks, by the nodes
            # themselves, since they have no "sort" attribute
            sources = sorted(node for node in subgraph if subgraph.out_degree(node))
            local_view = np.zeros((len(sources), len(positions_a), len(positions_b)), dtype=int)
            for index, source in enumerate(sources):
                for _, face, (member_a, member_b) in subgraph.out_edges(source, data="sort"):
                    position_a, position_b = positions_a[member_a], positions_b[member_b]
                    local_view[index, position_a, position_b] = subgraph.nodes[face]["sort"]
            local_views[pauli] = local_view
        return local_views

    def _get_vertex_blocks(
        self, pauli: PauliXZ
    ) -> list[tuple[ClassicalCode, ClassicalCode, npt.NDArray[np.int_]]]:
        """The vertices with checks of one type, as one block, since they share C_A and C_B."""
        return [(self.code_a, self.code_b, self.get_local_views(pauli))]

    def _get_permutation_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Generators of translations of the Cayley complex that preserve this code, or None.

        Here A and B are the cover subsets of the complex, whose vertices are members of the cover
        group (see help(qldpc.objects.CayleyComplex)), and V is the set of vertices.  Let x be a
        vertex with X-type checks such that conjugation by x permutes A.  Number the members of A in
        increasing order, let π map each number i to the number of the conjugate of member i, and
        suppose that π, as a permutation of coordinates, maps C_A onto itself.  Then left
        translation by x, which maps each vertex v to xv, preserves this code:

        - It maps V onto itself.  The vertices are the products uw of a member u of the group <A>
          generated by A and a member w of <B>, since left and right multiplications commute.  So
          ``x = u'w'`` for some such u' and w', and since conjugation by x maps <A> onto itself,
          ``xV = x<A><B> = <A>x<B> = <A>u'w'<B> = V``.
        - It maps vertices with X-type checks to vertices with X-type checks.  The cover group
          assigns each member a class in Z_2 (bipartite complex) or Z_2 x Z_2 (quadripartite
          complex), which multiplication adds, and the vertices with X-type checks are those whose
          class is in the subgroup {0} or {(0,0), (1,1)}, which contains the class of x.
        - It maps the face ``f(g,a,b) = {g, ag, gb, agb}`` to the face ``f(xg,a',b)``, where
          ``a' = xax^-1``.  So at a vertex g with X-type checks, the face with label (a, b) goes to
          the face with label (a', b) at xg, and at a vertex ag with Z-type checks, the face with
          label (a^-1, b) goes to the face with label (a'^-1, b) at xag, since
          ``a'^-1 = xa^-1x^-1``.  Either way, coordinate (i, j) of a local view goes to coordinate
          (π(i), j) of the local view of the image of its vertex.
        - Since π maps C_A onto itself, it maps ~C_A onto itself, so the translation maps the local
          check space of each vertex onto that of its image, and so the row spaces of matrix_x and
          matrix_z onto themselves.

        Right translation by x, which maps each vertex v to vx, preserves this code in the same way
        if x is a vertex with X-type checks, conjugation ``b -> x^-1 b x`` permutes B, and the
        induced permutation of coordinates maps C_B onto itself.  In each case, the vertices x that
        satisfy the conditions form a group, since their products satisfy them, and this method
        returns qudit permutations of generators of the two groups.  In a quadripartite complex, the
        group of left translations includes translation by the central member (e,1,1) of the cover
        group, where e is the identity of the group G, if (e,1,1) is a vertex.

        These conditions are sufficient, not necessary.  The translations omit other symmetries,
        such as those from automorphisms of the group, from exchanging A and B, or from permutations
        that preserve the stabilizer group but not each local check space.
        """
        # the faces of the complex, which are its qudits, and the vertices with X-type checks
        subgraph_x, _ = self.get_subgraphs(self.complex)
        faces = {face: qudit for face, qudit in subgraph_x.nodes(data="sort") if qudit is not None}
        vertices = sorted(node for node in subgraph_x if subgraph_x.out_degree(node))

        # Find generators of the left and right translations, and permute the faces by each.  Cover
        # members compare by their members of G first, so these orders are those of members_a/b.
        permutations = []
        for left, members, code in [
            (True, sorted(self.complex.cover_subset_a), self.code_a),
            (False, sorted(self.complex.cover_subset_b), self.code_b),
        ]:
            for vertex, _ in _get_translation_generators(vertices, members, [code], left=left):
                permutation = np.empty(len(self), dtype=int)
                for face, qudit in faces.items():
                    image = frozenset(vertex * node if left else node * vertex for node in face)
                    permutation[qudit] = faces[image]
                permutations.append(permutation)
        return np.array(permutations, dtype=int) if permutations else None


class LQTCode(_SquareComplexCode):
    """Lifted quantum Tanner code (not an LPCode): products of local codes, repeated over a group.

    A lifted quantum Tanner code is a CSS code defined by:

    - a finite group G, given explicitly;
    - ordered lists ``A = (a_0, ..., a_{m-1})`` and ``B = (b_0, ..., b_{n-1})`` of members of G,
      whose members may repeat and need not be symmetric, and which may be longer than G or empty;
      and
    - classical local codes C_A^0 and C_A^1 of length m, and C_B^0 and C_B^1 of length n, over one
      finite field.

    Complex and qudits.  The code is built on a quadripartite square complex, whose vertices are the
    triples (g, s, t) in G x {0,1} x {0,1}, and (s, t) is the class of a vertex.  The squares of the
    complex are the triples (i, j, g) for ``i < m``, ``j < n``, and g in G, and the corners of
    square (i, j, g) are::

           (g, 0, 0) ――――――――――――― (g b_j, 0, 1)

               |         (i, j, g)         |

        (a_i g, 1, 0) ―――――――――――― (a_i g b_j, 1, 1)

    Here products such as ``a_i g`` are products of qLDPC group members (``a_i * g``).  The square
    has the label (i, j) at each of its corners.  So a vertex (h, s, t) is a corner of exactly one
    square with each label (i, j), namely ``(i, j, a_i^-s h b_j^-t)``, which sits at coordinate
    (i, j) of the m x n local view of the vertex (see LQTCode.get_local_views).  Since squares are
    labeled by positions in A and B, rather than by members, members can repeat.  The qudits of the
    code are the squares, and qudit (i, j, g) has the index ``(i n + j) |G| + r(g)``, where r(g) is
    the position of g among the members of G in increasing order.

    Checks.  A vertex of class (s, t) uses the local codes C_A^s and C_B^t, in the coordinates of
    its local view.  The vertices with s = t carry X-type checks that span the local check space
    ``C_A^s ⨂ C_B^t``, and those with s != t carry Z-type checks that span ``~C_A^s ⨂ ~C_B^t``,
    where ``~C`` is the dual code of C.  The argument local_checks chooses the checks of each
    vertex: "min_weight" chooses a minimum-weight basis of its local check space, and "from_codes"
    the rows of ``G_A^s ⨂ G_B^t`` (X-type) or ``H_A^s ⨂ H_B^t`` (Z-type).  Here H_A^s and G_A^s are
    the parity check matrix and generator matrix that C_A^s carries (``code.matrix`` and
    ``code.generator``), and likewise for C_B^t (see help(qldpc.codes.quantum.quantum_tanner)).  The
    rows of matrix_x are the checks of the vertices of class (0, 0) and then (1, 1), and the rows of
    matrix_z those of the vertices of class (0, 1) and then (1, 0).  Within each class, vertices
    (g, s, t) are ordered by increasing g, and the checks of each vertex are consecutive rows.

    Commutation.  The checks commute over any finite field.  An X-type vertex (h, s, s) and a Z-type
    vertex (h', s, 1-s) share the first entry s of their classes.  Square (i, j, g) is at coordinate
    (i, j) of both of their local views if ``g = a_i^-s h b_j^-s = a_i^-s h' b_j^(s-1)``, which is
    to say if ``h' = h b_j^(1-2s)``, a condition on j alone.  Let J be the set of positions j of B
    with ``b_j = (h^-1 h')^(1-2s)``; if this member repeats in B, then J has several positions.  The
    two local views overlap on all rows i of the columns j in J.  On this overlap, a word ``x ⨂ y``
    of the X-type space, with x in C_A^s, and a word ``u ⨂ w`` of the Z-type space, with u in
    ~C_A^s, have the inner product ``(x · u) (sum_{j in J} y_j w_j) = 0``, and linearity extends
    this to all words of the two spaces.  Likewise, an X-type vertex (h, s, s) and a Z-type vertex
    (h', 1-s, s) share the second entry s of their classes, and their local views overlap on all
    columns of the rows i with ``h' = a_i^(1-2s) h``, where the inner product factors through C_B^s
    and its dual.  So ``matrix_x @ matrix_z.T = 0``.

    Relation to QTCode.  Let A and B be symmetric sets of distinct members of a group, listed in
    increasing order as by QTCode.members_a and QTCode.members_b, and let G be the group that they
    generate, with identity e.  Let ι_A be the permutation of positions in A that maps each member
    to its inverse, and let ``C∘ι_A`` be the code whose coordinate i is coordinate ι_A(i) of a code
    C, and likewise for B.

    - The quadripartite ``QTCode(A, B, C_A, C_B)`` has a face ``f(g,a,b) = {g, ag, gb, agb}`` for
      each vertex g of class (0, 0) of its complex, with the labels (a, b), (a^-1, b), (a, b^-1),
      and (a^-1, b^-1) at its corners of classes (0, 0), (1, 0), (0, 1), and (1, 1).  This face is
      the square (i, j, g) with ``a = a_i`` and ``b = b_j``, so in the labels of squares, the local
      check spaces of QTCode are those of the local codes ``C_A^0 = C_A``, ``C_A^1 = C_A∘ι_A``,
      ``C_B^0 = C_B``, and ``C_B^1 = C_B∘ι_B``.  QTCode keeps only the connected component of the
      complex that contains the identity, whereas an LQTCode on a disconnected complex is the direct
      sum of the codes on its components.  So if the complex is connected, the QTCode and the
      LQTCode with these local codes have the same qudits, up to their order, and the same
      stabilizer group.
    - The bipartite complex of a QTCode, whose subsets satisfy the Total No Conjugacy condition
      (TNC), is a quotient of the quadripartite complex.  The involution
      ``σ: (g, s, t) -> (g, 1-s, 1-t)`` maps square (i, j, g) to square
      ``(ι_A(i), ι_B(j), a_i g b_j)``, and since ``C^1 = C^0∘ι``, it maps the local check space of
      each vertex onto that of its image.  It changes the class of every vertex, and fixes square
      (i, j, g) only if ``a_i g b_j = g``, which TNC excludes.  The quotient by σ is the bipartite
      complex of QTCode, whose vertex (g, 0) has X-type checks and is the orbit of (g, 0, 0) and
      (g, 1, 1), and whose vertex (g, 1) is the orbit of (g, 1, 0) and (g, 0, 1).  Let K be the
      component of the quadripartite complex that contains (e, 0, 0).  If σ maps K onto itself, the
      bipartite QTCode is the quotient of the quadripartite QTCode by σ.  Otherwise, the quotient
      maps K isomorphically onto the component of the bipartite complex that contains (e, 0), and
      the bipartite QTCode has the qudits, up to their order, and the stabilizer group of the
      quadripartite QTCode.

    These relations are between stabilizer groups.  The parity checks correspond as well, up to the
    order of qudits and checks, if both codes use local_checks="from_codes", and C_A^1 and C_B^1
    carry the matrices of C_A and C_B with permuted columns, as
    ``ClassicalCode(H_A[:, ι_A]).set_generator(G_A[:, ι_A])`` does for the parity check matrix H_A
    and generator matrix G_A of C_A.  Otherwise the parity checks can differ.  For example, with
    "min_weight" they can differ because the minimum-weight basis that qLDPC picks can depend on the
    order of coordinates.

    LQTCode is a CSS stabilizer code.  It records no distance.  The constructor checks that the
    members of A and B are members of G, that the local codes have lengths m and n, and that they
    are over one field, which can be any finite field (the codes of arXiv:2512.20532 are binary).
    With "min_weight" local checks, it enumerates the words of each local code and of its dual,
    which takes time exponential in their dimensions, so it checks that each of them has at most
    2^20 words.  The "min_weight" checks at each vertex are as light as any basis of its local check
    space, though generating sets of the stabilizer group that combine checks at different vertices
    can be lighter.  The "min_weight" checks are also not chosen to minimize the number of checks
    that act on each qudit.  A local code given as a parity check matrix carries a generator matrix
    that qLDPC computes, which "from_codes" uses.  LQTCode.from_lrz builds a code from the data of
    arXiv:2512.20532, in that paper's notation.

    Symmetries: LQTCode.get_known_permutation_automorphism_group gives a group of qudit permutations
    that preserve the code.  Its generators include translations of the complex.  Left translation
    by a member x of G maps each vertex (g, s, t) to (xg, s, t).  It is included if conjugation by
    x, ``a -> x a x^-1``, permutes the list A, which is to say that the conjugate of each member
    occurs in A as often as the member does.  Repeated members are matched in order: the k-th
    occurrence of a member goes to the k-th occurrence of its conjugate, which gives a permutation π
    of the positions in A.  The translation is included if, moreover, π maps C_A^0 and C_A^1 onto
    themselves as a permutation of coordinates.  Right translation by x maps each vertex (g, s, t)
    to (gx, s, t), and is included under the same conditions for the conjugation ``b -> x^-1 b x``
    of B, and for C_B^0 and C_B^1.  With the optional pybliss package, the generators also include
    the automorphisms that bliss finds among the parity checks of the code, and among its canonical
    local checks (see help(qldpc.codes.quantum.quantum_tanner)).  For each type of checks, if either
    class of vertices with that type has a local check space with more than 2^20 words or 2^10
    canonical checks per vertex, bliss searches no canonical checks of that type.  The group can
    omit other symmetries, such as those from automorphisms of G or from exchanging A and B.  It
    contains no ZX-dualities, which exchange X-type and Z-type operators.

    References:

    - https://arxiv.org/abs/2512.20532 (Eq. (2))
    - https://arxiv.org/abs/2202.13641
    - https://arxiv.org/abs/2206.07571
    - https://errorcorrectionzoo.org/c/quantum_tanner
    """

    group: abstract.Group
    members_a: tuple[abstract.GroupMember, ...]
    members_b: tuple[abstract.GroupMember, ...]
    code_a0: ClassicalCode
    code_a1: ClassicalCode
    code_b0: ClassicalCode
    code_b1: ClassicalCode

    def __init__(
        self,
        group: abstract.Group,
        members_a: Sequence[abstract.GroupMember],
        members_b: Sequence[abstract.GroupMember],
        code_a0: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b0: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | type[galois.FieldArray] | None = None,
        *,
        code_a1: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        code_b1: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        local_checks: LocalChecks = "min_weight",
    ) -> None:
        """Construct a lifted quantum Tanner code.

        Args:
            group: The group G.
            members_a: The members a_i of G, in order.  Members may repeat.  The attribute members_a
                holds them on as many points as G acts on.
            members_b: The members b_j of G, in order.  Members may repeat.  The attribute members_b
                holds them on as many points as G acts on.
            code_a0: The local code C_A^0, or its parity check matrix, with one coordinate per
                position i in members_a.
            code_b0: The local code C_B^0, or its parity check matrix, with one coordinate per
                position j in members_b.
            field: The field of the local codes, if they are given as matrices.
            code_a1: The local code C_A^1, or its parity check matrix.  By default, C_A^0.
            code_b1: The local code C_B^1, or its parity check matrix.  By default, C_B^0.
            local_checks: The checks that span each local check space: "min_weight" (the default) or
                "from_codes" (see help(LQTCode)).

        Raises:
            ValueError: If any of the following holds:

                - A member of A or B is not a member of G.
                - A local code does not have one coordinate per position in its list.
                - The local codes are over different fields, or not over the given field.
                - local_checks is not a choice of local checks.
                - local_checks is "min_weight", and a local code or its dual has more than 2^20
                  words.
        """
        local_checks = _parse_local_checks(local_checks)

        # read the local codes; C_A^1 and C_B^1 default to C_A^0 and C_B^0
        codes_a = [
            ClassicalCode(code, field)
            for code in [code_a0, code_a0 if code_a1 is None else code_a1]
        ]
        codes_b = [
            ClassicalCode(code, field)
            for code in [code_b0, code_b0 if code_b1 is None else code_b1]
        ]
        if len({code.field for code in codes_a + codes_b}) > 1:
            raise ValueError("The local codes provided for this LQTCode are over different fields")

        # check the members and the lengths of the local codes, and read the members on as many
        # points as the group acts on, without merging repeated members
        lists = []
        for name, members, codes in [("a", members_a, codes_a), ("b", members_b, codes_b)]:
            members = list(members)
            _check_members_in_group(group, members, f"members_{name}")
            for index, code in enumerate(codes):
                if len(code) != len(members):
                    raise ValueError(
                        f"The local code C_{name.upper()}^{index} has block length {len(code)}, but"
                        f" members_{name} has {len(members)} members"
                    )
            lists.append(tuple(_get_member_on_points(member, group.degree) for member in members))

        # save the generating data, and check that "min_weight" can enumerate the local codes
        self.group = group
        self.members_a, self.members_b = lists
        self.code_a0, self.code_a1 = codes_a
        self.code_b0, self.code_b1 = codes_b
        if local_checks == "min_weight":
            _check_min_weight_sizes(codes_a + codes_b)

        # place the local checks of each class of vertices on their local views
        num_qudits = len(self.members_a) * len(self.members_b) * group.order
        matrices = []
        for pauli in PAULIS_XZ:
            blocks = [
                _place_local_checks(
                    local_views, _get_local_checks(code_a, code_b, pauli, local_checks), num_qudits
                )
                for code_a, code_b, local_views in self._get_vertex_blocks(pauli)
            ]
            matrices.append(np.vstack(blocks).view(codes_a[0].field))
        super().__init__(matrices[0], matrices[1])
        self._is_subsystem_code = False  # the checks commute (see help(LQTCode))

    @staticmethod
    def from_lrz(
        group: abstract.Group,
        members_a: Sequence[abstract.GroupMember],
        members_b: Sequence[abstract.GroupMember],
        *,
        code_0: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_1: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_0_prime: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_1_prime: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | type[galois.FieldArray] | None = None,
        local_checks: LocalChecks = "min_weight",
    ) -> LQTCode:
        """The lifted quantum Tanner code of Eq. (2) of arXiv:2512.20532, in that paper's notation.

        That paper builds a code from a group G, ordered lists ``A = (a_1, ..., a_{n_A})`` and
        ``B = (b_1, ..., b_{n_B})`` of members of G, and binary codes C_0 and C_1 of length n_A and
        C'_0 and C'_1 of length n_B, whose parity check matrices H_s and H'_t and generator matrices
        G_s and G'_t it fixes.  Below, as in LQTCode, positions i in A and j in B count from 0, so
        a_i is the paper's a_{i+1}.  The qubits of the paper are the triples (i, j, g), in the order
        of the index ``(i n_B + j) |G| + r(g)`` of LQTCode, and its parity check matrices are

            ``H_X = [H_0 ⨂ G'_0 ⨂ I; (H_1 ⨂ G'_1 ⨂ I) L_A R_B]``, and
            ``H_Z = [(G_0 ⨂ H'_1 ⨂ I) R_B; (G_1 ⨂ H'_0 ⨂ I) L_A]``,

        where I is the identity on the members of G, L_A maps qubit (i, j, g) to (i, j, a_i g), and
        R_B maps it to ``(i, j, g b_j^-1)``.  So the checks of the four blocks, for each g in G, are
        products of local words on the qubits:

        - ``(i, j, g)``, with words of ``~C_0 ⨂ C'_0`` (X-type);
        - ``(i, j, a_i g b_j^-1)``, with words of ``~C_1 ⨂ C'_1`` (X-type);
        - ``(i, j, g b_j^-1)``, with words of ``C_0 ⨂ ~C'_1`` (Z-type); and
        - ``(i, j, a_i g)``, with words of ``C_1 ⨂ ~C'_0`` (Z-type).

        These are the local views of the vertices (g, s, t) of classes (0, 0), (1, 1), (0, 1), and
        (1, 0) of the LQTCode with members ``a_i^-1`` in place of a_i, members b_j, and local codes
        ``C_A^s = ~C_s`` and ``C_B^t = C'_t``, whose local check spaces are the same products.  This
        method returns that LQTCode.  It builds each code ``~C_s`` with the matrices of C_s
        exchanged: its parity check matrix is G_s, and its generator matrix is H_s.  So with
        local_checks="from_codes", the X-type checks at the vertices of class (s, s) are the rows of
        ``H_s ⨂ G'_s``, and the Z-type checks at the vertices of class (s, t) are the rows of
        ``G_s ⨂ H'_t``.  These are the rows of H_X and H_Z, up to their order, if each code carries
        the matrices of the paper: its parity check matrix as code.matrix, and its generator matrix
        as code.generator, which ClassicalCode.set_generator sets.  For a code given only as a
        parity check matrix, qLDPC computes a generator matrix, so the checks of the returned code
        span the same spaces as those of the paper, but can differ from them.  The published check
        matrices of its [[144, 12, 11]] code order the qubits by g first: qubit (i, j, g) is the
        published qubit ``36 h + 6 i + j``, where h is the number that the files give g (they number
        the members of Z_2 × Z_2 from 0 to 3 under bitwise XOR).  So a comparison with those files
        needs a relabeling of the qubits.

        The paper's codes are binary.  Over other finite fields, this method builds the LQTCode in
        the same way.

        Args:
            group: The group G.
            members_a: The members a_i of G, in order, as in the paper.  Members may repeat.
            members_b: The members b_j of G, in order, as in the paper.  Members may repeat.
            code_0: The code C_0, or its parity check matrix H_0.
            code_1: The code C_1, or its parity check matrix H_1.
            code_0_prime: The code C'_0, or its parity check matrix H'_0.
            code_1_prime: The code C'_1, or its parity check matrix H'_1.
            field: The field of the codes, if they are given as matrices.
            local_checks: The checks that span each local check space, as for LQTCode.

        Raises:
            ValueError: As for LQTCode.
        """
        # check the members of A before inverting them, so that an error names a member as given
        members_a = list(members_a)
        _check_members_in_group(group, members_a, "members_a")

        # build the LQTCode of the inverses of the members of A, and the duals of C_0 and C_1
        codes = [
            ClassicalCode(code, field) for code in [code_0, code_1, code_0_prime, code_1_prime]
        ]
        return LQTCode(
            group,
            [~member for member in members_a],
            members_b,
            _get_dual_code(codes[0]),
            codes[2],
            code_a1=_get_dual_code(codes[1]),
            code_b1=codes[3],
            local_checks=local_checks,
        )

    def get_local_views(self, pauli: PauliXZLike) -> npt.NDArray[np.int_]:
        """The local views of the vertices with parity checks of one type.

        Entry ``[v, i, j]`` of the returned array is the qudit at coordinate (i, j) of the local
        view of the v-th vertex with checks of the given type: if that vertex is (h, s, t), the
        square ``(i, j, a_i^-s h b_j^-t)``.  The vertices with X-type checks are those of class
        (0, 0) and then (1, 1), and the vertices with Z-type checks are those of class (0, 1) and
        then (1, 0).  Within each class, vertices (h, s, t) are ordered by increasing h.  The checks
        of each vertex are consecutive rows of matrix_x or matrix_z, and the vertices appear in the
        order of these rows.  All vertices of a class have the same number of checks, but the two
        classes of vertices with checks of one type can have different numbers of checks, since
        their local codes can differ.

        Args:
            pauli: Pauli.X for the vertices with X-type checks, or Pauli.Z for those with Z-type
                checks.

        Returns:
            A read-only integer array of shape ``(2 |G|, m, n)``.
        """
        vertex_classes = _VERTEX_CLASSES[Pauli.coerce_xz(pauli)]
        local_views = [self._class_local_views[vertex_class] for vertex_class in vertex_classes]
        return _mark_read_only(np.concatenate(local_views))

    @functools.cached_property
    def _members_of_group(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the group G, in increasing order, on as many points as G acts on."""
        degree = self.group.degree
        return tuple(
            sorted(_get_member_on_points(member, degree) for member in self.group.generate())
        )

    @functools.cached_property
    def _class_local_views(self) -> dict[tuple[int, int], npt.NDArray[np.int_]]:
        """The local views of the vertices of each class (s, t), in increasing order of their h."""
        members = self._members_of_group
        positions = {member: position for position, member in enumerate(members)}
        num_a, num_b, order = len(self.members_a), len(self.members_b), len(members)

        # the positions of a_i^-1 h and of h b_j^-1, for each member h of G
        left_positions = {
            member: [positions[~member * hh] for hh in members] for member in set(self.members_a)
        }
        right_positions = {
            member: [positions[hh * ~member] for hh in members] for member in set(self.members_b)
        }
        shifts_a = np.array([left_positions[member] for member in self.members_a], dtype=int)
        shifts_b = np.array([right_positions[member] for member in self.members_b], dtype=int)

        # reshape, since an empty list of members gives an array of shape (0,)
        shifts_a = shifts_a.reshape(num_a, order)
        shifts_b = shifts_b.reshape(num_b, order)

        # the position of a_i^-s h b_j^-t at coordinate (i, j) of each vertex (h, s, t), and the
        # qudit of the square (i, j, a_i^-s h b_j^-t)
        first_qudits = np.add.outer(np.arange(num_a) * num_b, np.arange(num_b)) * order
        local_views = {}
        for ss, tt in itertools.product(range(2), repeat=2):
            if ss:
                square_positions = np.broadcast_to(shifts_a.T[:, :, None], (order, num_a, num_b))
            else:
                square_positions = np.broadcast_to(
                    np.arange(order)[:, None, None], (order, num_a, num_b)
                )
            if tt:
                square_positions = shifts_b[np.arange(num_b), square_positions]
            local_views[ss, tt] = first_qudits + square_positions
        return local_views

    def _get_vertex_blocks(
        self, pauli: PauliXZ
    ) -> list[tuple[ClassicalCode, ClassicalCode, npt.NDArray[np.int_]]]:
        """The blocks of vertices with checks of one type, which are the classes of these vertices.

        The vertices of class (s, t) have the local codes C_A^s and C_B^t.
        """
        codes_a, codes_b = [self.code_a0, self.code_a1], [self.code_b0, self.code_b1]
        return [
            (codes_a[ss], codes_b[tt], self._class_local_views[ss, tt])
            for ss, tt in _VERTEX_CLASSES[pauli]
        ]

    def _get_permutation_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Generators of translations of the complex that preserve this code, or None.

        Let x be a member of G such that conjugation by x permutes the list A: the conjugate of each
        member occurs in A as often as the member does.  Then ``x a_i x^-1 = a_π(i)`` for the
        permutation π of positions that maps the k-th occurrence of each member to the k-th
        occurrence of its conjugate.  Suppose that π, as a permutation of coordinates, maps C_A^0
        and C_A^1 onto themselves.  Then left translation by x, which maps each vertex (g, s, t) to
        (xg, s, t), preserves this code:

        - It maps the corners g, a_i g, g b_j, and a_i g b_j of square (i, j, g) to the corners xg,
          ``a_π(i) xg``, xg b_j, and ``a_π(i) xg b_j`` of square ``(π(i), j, xg)``, of the same
          classes.  So it maps squares to squares, which is the qudit permutation
          ``(i, j, g) -> (π(i), j, xg)``, and vertices to vertices of the same class.
        - The square at coordinate (i, j) of the local view of a vertex has the label (i, j) at that
          vertex, so its image, which has the label (π(i), j), is at coordinate (π(i), j) of the
          local view of the image of the vertex.
        - Since π maps C_A^s onto itself, it maps ~C_A^s onto itself.  So the translation maps the
          local check space of each vertex onto that of its image, and so the row spaces of matrix_x
          and matrix_z onto themselves.

        Right translation by x, which maps each vertex (g, s, t) to (gx, s, t), likewise maps square
        (i, j, g) to square ``(i, ρ(j), gx)``, where ``x^-1 b_j x = b_ρ(j)``, and preserves this
        code if ρ matches repeated members in order and maps C_B^0 and C_B^1 onto themselves.
        Matching in order makes the permutation of positions of a product xy the product of the
        permutations of x and y.  So the members x that satisfy the conditions form a group,
        translations compose as their members do, and this method returns the qudit permutations of
        generators of the groups of left and right translations.

        These conditions are sufficient, not necessary.  The translations omit other symmetries,
        such as permutations that match repeated members in other orders, that come from
        automorphisms of G, that exchange A and B, that exchange classes of vertices (such as the
        involution σ of the quadripartite complex of a QTCode), or that preserve the stabilizer
        group but not each local check space.
        """
        group_members = self._members_of_group
        positions = {member: position for position, member in enumerate(group_members)}
        num_a, num_b = len(self.members_a), len(self.members_b)
        permutations = []
        for left, listed_members, codes in [
            (True, self.members_a, [self.code_a0, self.code_a1]),
            (False, self.members_b, [self.code_b0, self.code_b1]),
        ]:
            generators = _get_translation_generators(
                group_members, listed_members, codes, left=left
            )
            for xx, permutation in generators:
                # square (i, j, g) goes to square (π(i), j, xg) or (i, ρ(j), gx)
                images_g = [positions[xx * hh if left else hh * xx] for hh in group_members]
                images_a = permutation if left else range(num_a)
                images_b = range(num_b) if left else permutation
                squares = np.add.outer(np.array(images_a, dtype=int) * num_b, list(images_b))
                permutations.append(np.add.outer(squares * len(group_members), images_g).ravel())
        return np.array(permutations, dtype=int) if permutations else None


def _parse_local_checks(value: object) -> LocalChecks:
    """Read a choice of local checks, raising a ValueError if it is not one."""
    for local_checks in get_args(LocalChecks):
        if isinstance(value, str) and value == local_checks:
            return local_checks
    raise ValueError(
        f"Unrecognized choice of local checks: {value!r} (expected one of {get_args(LocalChecks)})"
    )


def _get_dual_code(code: ClassicalCode) -> ClassicalCode:
    """The dual of a classical code, whose generator matrix is the parity check matrix of the code.

    So the dual carries the same two matrices as the code, with their roles exchanged.  The dual
    ``~code`` would instead carry a generator matrix that qLDPC computes, so "from_codes" local
    checks would not use the parity check matrix that the code carries.
    """
    return ClassicalCode(code.generator).set_generator(code.matrix)


def _read_local_codes(
    code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
    code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None,
    field: int | type[galois.FieldArray] | None,
) -> tuple[ClassicalCode, ClassicalCode]:
    """Read the local codes C_A and C_B of a QTCode, where C_B defaults to the dual of C_A."""
    code_a = ClassicalCode(code_a, field)
    code_b = ClassicalCode(code_b, field) if code_b is not None else _get_dual_code(code_a)
    return code_a, code_b


def _check_members_in_group(
    group: abstract.Group, members: Sequence[abstract.GroupMember], name: str
) -> None:
    """Raise a ValueError if a member of a list is not a member of a group."""
    for member in members:
        if member not in group:
            raise ValueError(
                f"{name} contains {member}, which is not a member of the group {group}"
            )


def _get_member_on_points(member: abstract.GroupMember, num_points: int) -> abstract.GroupMember:
    """The same member, represented on num_points points, fixing any added points.

    The member must fix every point from num_points on, which are dropped.
    """
    return abstract.GroupMember(
        member.array_form[:num_points] + list(range(member.size, num_points))
    )


def _get_factor_matrices(
    code_a: ClassicalCode, code_b: ClassicalCode, pauli: PauliXZ
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """Matrices whose row spaces are the factors of the local check space of one type.

    The local check space is ``C_A ⨂ C_B`` for X-type checks, and ``~C_A ⨂ ~C_B`` for Z-type checks.
    Its factors are spanned by the generator matrices G_A and G_B that code_a and code_b carry, or
    by their parity check matrices H_A and H_B.
    """
    if pauli is Pauli.X:
        return code_a.generator, code_b.generator
    return code_a.matrix, code_b.matrix


def _get_local_checks(
    code_a: ClassicalCode, code_b: ClassicalCode, pauli: PauliXZ, local_checks: LocalChecks
) -> galois.FieldArray:
    """The local checks of one type at a vertex whose local codes are code_a and code_b.

    The checks span the local check space ``C_A ⨂ C_B`` for X-type checks, or ``~C_A ⨂ ~C_B`` for
    Z-type checks, in the coordinates of a local view.  They are the products of minimum-weight
    bases of the two factors if local_checks is "min_weight", and otherwise the rows of
    ``G_A ⨂ G_B`` or ``H_A ⨂ H_B``, where G_A and H_A are the generator and parity check matrices
    that code_a carries, and likewise for code_b (see help(qldpc.codes.quantum.quantum_tanner)).
    Before asking for minimum-weight bases, callers check the sizes of all local codes with
    _check_min_weight_sizes, so that a request that cannot be met fails before any enumeration.
    """
    factors = _get_factor_matrices(code_a, code_b, pauli)
    if local_checks == "min_weight":
        factors = (get_min_weight_basis(factors[0]), get_min_weight_basis(factors[1]))
    return np.kron(*factors).view(code_a.field)


def _check_min_weight_sizes(codes: Iterable[ClassicalCode]) -> None:
    """Raise a ValueError if a local code or its dual has too many words for "min_weight" checks.

    The limit is _MAX_WORDS_FOR_MIN_WEIGHT_BASIS words.
    """
    for code in codes:
        for num_words, space in [
            (code.field.order**code.dimension, "local code"),
            (code.field.order**code.rank, "dual of a local code"),
        ]:
            if num_words > _MAX_WORDS_FOR_MIN_WEIGHT_BASIS:
                raise ValueError(
                    f"Minimum-weight local checks enumerate the words of each local code and of its"
                    f" dual, but a {space} has {num_words} words, more than"
                    f" {_MAX_WORDS_FOR_MIN_WEIGHT_BASIS}.  Pass local_checks='from_codes' to build"
                    " the checks from the matrices that the local codes carry."
                )


def _get_translation_generators(
    candidates: Sequence[abstract.GroupMember],
    members: Sequence[abstract.GroupMember],
    codes: Sequence[ClassicalCode],
    *,
    left: bool,
) -> list[tuple[abstract.GroupMember, tuple[int, ...]]]:
    """Generators of the group of members x whose translations from one side preserve a code.

    A candidate x qualifies if conjugation by x, ``m -> x m x^-1`` for left translations or
    ``m -> x^-1 m x`` for right translations, permutes the list of members, and the induced
    permutation of positions, which matches repeated members in order, maps each code onto itself as
    a permutation of its coordinates (see _get_conjugation_permutation).  The candidates must
    include the product of any two candidates that qualify.  The permutations of two candidates
    compose as the candidates do, so the candidates that qualify then form a group, of which this
    function returns generators.

    Args:
        candidates: The members x of the group that may qualify, in increasing order, such as the
            vertices with X-type checks of a QTCode or the members of G for an LQTCode.
        members: The list of members whose positions label the coordinates of the codes.
        codes: The local codes whose coordinates the positions of the members label.
        left: Whether to find left translations, rather than right translations.

    Returns:
        Pairs of a generator x and the permutation π of positions by which conjugation by x maps the
        member at each position i to the member at position π(i).
    """
    # find the candidates whose conjugation permutes the members and keeps the codes
    preserves_codes: dict[tuple[int, ...], bool] = {}
    qualifying = []
    for candidate in candidates:
        permutation = _get_conjugation_permutation(candidate, members, left=left)
        if permutation is None:
            continue
        if permutation not in preserves_codes:
            preserves_codes[permutation] = all(
                _preserves_row_space(code.matrix, np.array(permutation, dtype=int), len(code))
                for code in codes
            )
        if preserves_codes[permutation]:
            qualifying.append((candidate, permutation))

    # keep each qualifying candidate that those kept before it do not generate
    generators: list[tuple[abstract.GroupMember, tuple[int, ...]]] = []
    for candidate, permutation in qualifying:
        kept = [generator for generator, _ in generators]
        if not candidate.is_Identity and not (kept and candidate in abstract.Group(*kept)):
            generators.append((candidate, permutation))
            if abstract.Group(*kept, candidate).order == len(qualifying):
                break
    return generators


def _get_conjugation_permutation(
    conjugator: abstract.GroupMember, members: Sequence[abstract.GroupMember], *, left: bool
) -> tuple[int, ...] | None:
    """The permutation of positions by which conjugation by a member x permutes a list of members.

    Conjugation by the conjugator x is ``m -> x m x^-1`` if left is True, and ``m -> x^-1 m x``
    otherwise.  The permutation π maps each position i to the position π(i) of the conjugate of the
    member at position i.  It matches repeated members in order: the k-th occurrence of a member
    goes to the k-th occurrence of its conjugate.  So the permutations of two conjugators compose as
    the conjugators do.  The result is None if conjugation does not permute the list, which is to
    say if some member and its conjugate occur different numbers of times.
    """
    # the positions at which each member occurs in the list, in increasing order
    positions: dict[abstract.GroupMember, list[int]] = {}
    for position, member in enumerate(members):
        positions.setdefault(member, []).append(position)

    # map the k-th occurrence of each member to the k-th occurrence of its conjugate
    permutation = [0] * len(members)
    for member, member_positions in positions.items():
        if left:
            conjugate = conjugator * member * ~conjugator
        else:
            conjugate = ~conjugator * member * conjugator
        conjugate_positions = positions.get(conjugate, [])
        if len(conjugate_positions) != len(member_positions):
            return None
        for position, conjugate_position in zip(member_positions, conjugate_positions):
            permutation[position] = conjugate_position
    return tuple(permutation)


def _get_canonical_local_checks(
    code_a: ClassicalCode, code_b: ClassicalCode, pauli: PauliXZ
) -> galois.FieldArray | None:
    """The canonical checks of the local check space of one type, or None.

    The local check space is ``C_A ⨂ C_B`` for X-type checks, and ``~C_A ⨂ ~C_B`` for Z-type checks,
    where C_A and C_B are code_a and code_b.  Its canonical checks are its nonzero words of weight
    at most w_max, where w_max is the largest weight of a word in a minimum-weight basis of the
    space (see help(qldpc.codes.quantum.quantum_tanner)).  The result is None if the space has more
    than _MAX_WORDS_FOR_CANONICAL_CHECKS words, or more than _MAX_CANONICAL_CHECKS_PER_VERTEX
    canonical checks.
    """
    # find a basis of the space, and give up if it has too many words to enumerate
    field = code_a.field
    factors = _get_factor_matrices(code_a, code_b, pauli)
    basis = np.kron(factors[0].row_space(), factors[1].row_space()).view(field)
    num_words = field.order ** len(basis)
    if num_words > _MAX_WORDS_FOR_CANONICAL_CHECKS:
        return None
    if not len(basis):
        return field.Zeros((0, basis.shape[1]))

    # Products of minimum-weight bases are minimum-weight bases, so their largest weights multiply.
    # Finding a minimum-weight basis of a factor enumerates its words.  Each factor has at most as
    # many words as the space, since neither factor is zero, so this costs no more than the
    # enumeration below.
    max_weight = 1
    for factor in factors:
        weights = np.count_nonzero(get_min_weight_basis(factor).view(np.ndarray), axis=1)
        max_weight *= int(np.max(weights, initial=0))

    # Enumerate the words of the space in chunks, keeping the light ones.  This copies the
    # enumeration of qldpc.math._iter_coefficients, which is private to qldpc.math.
    place_values = field.order ** np.arange(len(basis), dtype=np.int64)
    checks = [field.Zeros((0, basis.shape[1]))]
    num_checks = 0
    for start in range(0, num_words, _WORDS_PER_CHUNK):
        indices = np.arange(start, min(start + _WORDS_PER_CHUNK, num_words), dtype=np.int64)
        words = field(indices[:, None] // place_values % field.order) @ basis
        weights = np.count_nonzero(words.view(np.ndarray), axis=1)
        checks.append(words[(weights > 0) & (weights <= max_weight)])
        num_checks += len(checks[-1])
        if num_checks > _MAX_CANONICAL_CHECKS_PER_VERTEX:
            return None
    return np.vstack(checks).view(field)


def _place_local_checks(
    local_views: npt.NDArray[np.int_], local_checks: galois.FieldArray, num_qudits: int
) -> galois.FieldArray:
    """Copies of the same local checks at several vertices, as the rows of a parity check matrix.

    Args:
        local_views: An array of shape (number of vertices, m, n), whose entry ``[v, i, j]`` is the
            qudit at coordinate (i, j) of the local view of vertex v.
        local_checks: The local checks, as the rows of a matrix with one column per coordinate
            (i, j) of a local view, in row-major order.
        num_qudits: The number of qudits of the code.

    Returns:
        A matrix whose rows are the local checks on the local view of each vertex, with the checks
        of each vertex in consecutive rows, and the vertices in order.
    """
    num_vertices, num_checks = len(local_views), len(local_checks)
    checks = type(local_checks).Zeros((num_vertices * num_checks, num_qudits))
    rows = np.arange(len(checks)).reshape(num_vertices, num_checks, 1)
    columns = local_views.reshape(num_vertices, 1, local_checks.shape[1])
    checks[rows, columns] = local_checks
    return checks
