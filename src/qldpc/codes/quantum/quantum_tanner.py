# SPDX-License-Identifier: Apache-2.0

"""Quantum Tanner codes.

A quantum Tanner code is a CSS code whose qudits are the faces of a square complex, and whose parity
checks at each vertex span its local check space: the tensor product of two classical local codes,
or of their duals, on the faces at that vertex.  This module provides two classes of these codes:

- QTCode builds a code on the left-right Cayley complex of two symmetric subsets of a group, from
  two local codes, as in arXiv:2202.13641 and arXiv:2206.07571.
- LQTCode builds the lifted quantum Tanner codes of Eq. (2) of arXiv:2512.20532, from an explicit
  group, two ordered lists of its members, and four local codes.

Use QTCode for symmetric sets of distinct members and for the bipartite complex.  Use LQTCode for
lists whose members repeat or are not symmetric, or for four local codes; its docstring relates the
two classes.  LocalChecks lists the choices of parity checks of both classes.
"""

from __future__ import annotations

import abc
import functools
import heapq
import itertools
import json
import os
import re
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
"""The choices of the parity checks of a QTCode or an LQTCode that span each local check space.

The local check space of a vertex with X-type checks is ``C_A ⨂ C_B``, and that of a vertex with
Z-type checks is ``~C_A ⨂ ~C_B``, where C_A and C_B are the local codes of the vertex, and ``~C`` is
the dual code of C (see QTCode and LQTCode).  The parity checks at the vertex are the words of a
spanning set of this space, so every choice gives the same stabilizer group, but the parity checks
can differ:

- "min_weight" (the default): a minimum-weight basis, whose i-th lightest word is no heavier than
  the i-th lightest word of any basis of the space, for each i.  It consists of the tensor products
  u ⨂ v of the words u and v of minimum-weight bases of the two factors of the space, which
  qldpc.math.get_min_weight_basis finds.  Finding these bases enumerates the words of each local
  code and of its dual, so QTCode and LQTCode raise a ValueError if a local code or its dual has
  more than 2^20 words.  Generating sets of the stabilizer group that combine checks at different
  vertices can still be lighter, and the checks are not chosen to minimize the number of checks that
  act on each qudit.
- "from_codes": the rows of ``G_A ⨂ G_B`` for X-type checks, and of ``H_A ⨂ H_B`` for Z-type checks,
  where H_A and G_A are the parity check matrix and generator matrix that C_A carries
  (``code.matrix`` and ``code.generator``), and likewise for C_B.  A local code carries the
  generator matrix set by ClassicalCode.set_generator, or else one that qLDPC computes.  Linearly
  dependent rows of these matrices give redundant checks.
"""

# the classes (s, t) of the vertices of an LQTCode with checks of each type, in the order of their
# rows in matrix_x and matrix_z
_VERTEX_CLASSES: dict[PauliXZ, tuple[tuple[int, int], ...]] = {
    Pauli.X: ((0, 0), (1, 1)),
    Pauli.Z: ((0, 1), (1, 0)),
}

# the most words that a local code or its dual may have for "min_weight" local checks
_MAX_WORDS_FOR_MIN_WEIGHT_BASIS = 2**20

# caps on the canonical local checks (see _SquareComplexCode._get_canonical_checks)
_MAX_WORDS_FOR_CANONICAL_CHECKS = 2**20
_MAX_CANONICAL_CHECKS_PER_VERTEX = 2**10

# the number of words of a local check space that _get_canonical_local_checks enumerates at a time
_WORDS_PER_CHUNK = 2**14

####################################################################################################
# quantum Tanner codes


class _SquareComplexCode(CSSCode):
    """A CSS code on a square complex, whose checks at each vertex span a product of local codes.

    The local check space of each vertex is the tensor product of two local codes, or of their
    duals, read on the local view of the vertex (see LocalChecks).  This class finds the canonical
    local checks of the code from the blocks of vertices that _get_vertex_blocks gives, and has
    bliss search them as well as the parity checks.
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

        Bliss is a graph-automorphism tool, which the optional pybliss package provides.  It finds
        qudit permutations that map a set of checks onto itself (see
        QuditCode.get_known_permutation_automorphism_group).  A permutation that preserves this code
        can map its parity checks to other words of the local check spaces, and bliss then misses it
        among the parity checks.  Such a permutation can still map the canonical local checks (see
        _get_canonical_checks) onto themselves, since they depend only on the local check spaces.
        For example, a permutation that maps the local view and local check space of each vertex
        onto those of a vertex with checks of the same type preserves weights, so it maps canonical
        checks to canonical checks.  Bliss can also find automorphisms of the parity checks that are
        not automorphisms of the canonical checks, such as when a local code is the direct sum of
        codes on disjoint sets of coordinates.  So bliss searches both.
        """
        canonical_checks = self._get_canonical_checks()
        if np.array_equal(canonical_checks, self.matrix):
            return (self.matrix,)
        return (self.matrix, canonical_checks)

    def _get_canonical_checks(self) -> galois.FieldArray:
        """The canonical local checks of this code, in the layout of its parity check matrix.

        The canonical local checks of a vertex are the nonzero words of its local check space that
        are no heavier than the heaviest word of a minimum-weight basis of the space, placed on its
        local view.  All minimum-weight bases of a space have the same weights, so these checks
        depend only on the space.  They include a minimum-weight basis of each local check space, so
        they span the row spaces of matrix_x and matrix_z, and a qudit permutation that maps the set
        of them onto itself preserves the code.

        For each type of checks, the returned matrix holds the parity checks of that type instead if
        a local check space of that type has more than _MAX_WORDS_FOR_CANONICAL_CHECKS words, or
        more than _MAX_CANONICAL_CHECKS_PER_VERTEX canonical checks.  The first cap bounds the
        enumeration of the words of the space, and the second the size of the matrix that bliss
        searches, which has a row for each canonical check at each vertex.
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
            canonical_local_checks = _get_canonical_local_checks(code_a, code_b, pauli)
            if canonical_local_checks is None:
                return None

            # place them on the local view of every vertex of the block
            blocks.append(_place_local_checks(local_views, canonical_local_checks, len(self)))
        return np.vstack(blocks).view(self.field)


class QTCode(_SquareComplexCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    A quantum Tanner code is defined by:

    - two symmetric subsets A and B of a group G, which hold the inverse of each of their members
      and are not both empty, and
    - two classical local codes C_A and C_B, with block lengths ``|A|`` and ``|B|``, respectively.

    The qudits of the code are the faces of the left-right Cayley complex of A and B, whose vertices
    are members of a cover of G (see help(qldpc.objects.CayleyComplex)).  By default, the complex is
    quadripartite: each vertex has a class (s, t), where multiplying the vertex by a member of A
    flips s, and multiplying it by a member of B flips t.  Vertices of classes (0, 0) and (1, 1)
    have X-type checks.  If bipartite is True, the complex is bipartite: each vertex has a class s,
    which multiplying the vertex by a member of A or of B flips, and vertices of class 0 have X-type
    checks.  The bipartite complex needs A and B to satisfy the Total No Conjugacy condition of
    arXiv:2202.13641, Eq. (1): ``ag != gb`` for all a in A, b in B, and g in the group that A and B
    generate.  QTCode keeps only the component of the complex that contains the identity.  For each
    vertex g with X-type checks, a in A, and b in B, the complex has a face
    ``f(g,a,b) = {g, ag, gb, agb}``, whose corners ag and gb have Z-type checks::

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

    On the local view of each vertex, an ``|A| x |B|`` array of the faces at the vertex (see
    QTCode.get_local_views), the X-type checks span the local check space ``C_A ⨂ C_B``, and the
    Z-type checks span ``~C_A ⨂ ~C_B``, where ``~C`` is the dual code of C.  The argument
    local_checks chooses which words of each local check space are parity checks (see LocalChecks).
    Coordinate i of C_A belongs to ``members_a[i]``, and coordinate j of C_B to ``members_b[j]``,
    which list the members of A and B in increasing order.  QTCode.from_ordered_members takes other
    orders.  The local codes can be over any finite field.

    This is the construction of arXiv:2202.13641, Sec. 4.1, with the roles of X and Z exchanged, on
    the bipartite complex or on the quadripartite complex of Sec. 3.2 of that paper.

    Notes:

    Two QTCodes are equal if both are bipartite or both quadripartite, and they have the same
    subsets, local codes over the same field with the same parity check matrices, and the same
    parity checks, whatever choice of local checks gave them.  QTCode.save_json writes a code to a
    file, from which QTCode.load_json rebuilds an equal code (see QTCode.save_json for what the file
    records and which fields it supports).

    The group that QTCode.get_known_permutation_automorphism_group returns includes translations of
    the complex that preserve the local codes.  With the optional pybliss package, it also includes
    the automorphisms that bliss finds among the parity checks and among the canonical local checks,
    which depend only on the local check spaces.  It can omit other symmetries, such as those from
    automorphisms of G or from exchanging A and B.

    References:

    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571 (Sec. 4)
    - https://arxiv.org/abs/2202.13641 (Secs. 3.2 and 4.1, and Theorem 2 for asymptotically good
      families from projective special linear groups and random local codes)
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
                "from_codes" (see LocalChecks).  Pass "from_codes" for local codes that have too
                many words for "min_weight".

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

        # The checks commute, over any field, by the argument of arXiv:2202.13641, Sec. 4.1, on the
        # bipartite complex, and of arXiv:2206.07571, Sec. 2 and Fig. 2, on the quadripartite one.
        self._is_subsystem_code = False

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

        # the same generating data can give different parity checks, depending on the choice of
        # local checks and the generator matrices of the local codes
        return (
            subset_a == other_subset_a
            and subset_b == other_subset_b
            and np.array_equal(self.matrix_x, other.matrix_x)
            and np.array_equal(self.matrix_z, other.matrix_z)
        )

    @property
    def members_a(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the subset A, in increasing order.

        Member i labels coordinate i of C_A, and row i of each local view (see
        QTCode.get_local_views).  Increasing order is the order in which sorted() puts qLDPC group
        members (see help(qldpc.abstract.GroupMember)).  The members are written on the largest
        number of points of any member of A or B, so they can differ from the members given to
        QTCode, since SymPy permutations on different numbers of points are never equal.
        """
        return tuple(sorted(self.complex.subset_a))

    @property
    def members_b(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the subset B, in increasing order.

        Member j labels coordinate j of C_B, and column j of each local view (see
        QTCode.get_local_views).  The order, and the number of points on which the members are
        written, are as for members_a.
        """
        return tuple(sorted(self.complex.subset_b))

    def get_local_views(self, pauli: PauliXZLike) -> npt.NDArray[np.int_]:
        """The local views of the vertices with parity checks of one type.

        Each face at a vertex v is ``{v, av, vb, avb}`` for exactly one member a of A and one member
        b of B, which form the label (a, b) of the face at v, as in arXiv:2202.13641, Sec. 4.1,
        whereas arXiv:2206.07571, Sec. 4.2, gives a face the same label at all four corners.  So the
        face ``f(g,a,b) = {g, ag, gb, agb}`` has the labels (a, b), (a^-1, b), (a, b^-1), and
        (a^-1, b^-1) at its corners g, ag, gb, and agb.  Entry ``[v, i, j]`` of the returned array
        is the qudit at coordinate (i, j) of the local view of the v-th vertex with checks of the
        given type, which is the face whose label at that vertex is
        ``(members_a[i], members_b[j])``.  The checks of each vertex are consecutive rows of
        ``matrix_x`` or ``matrix_z``: every vertex has the same number k of checks, and the checks
        of the v-th vertex are rows ``v k`` to ``v k + k - 1``.

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
        """Get the classical Tanner codes of the X-type and Z-type checks of a quantum Tanner code.

        Return the classical Tanner codes on subgraph_x and subgraph_z (see QTCode.get_subgraphs),
        whose parity checks are the X-type and Z-type checks of the quantum Tanner code.  The parity
        checks of the subcode of each (TannerCode.subcode) are the local checks of its type, which
        it places at every vertex with checks of that type, so the subcode is ``~(C_A ⨂ C_B)`` or
        ``~(~C_A ⨂ ~C_B)``.  The argument local_checks is as for QTCode.
        """
        local_checks = _parse_local_checks(local_checks)
        if local_checks == "min_weight":
            _check_min_weight_sizes([code_a, code_b])
        subgraph_x, subgraph_z = QTCode.get_subgraphs(cayplex)

        # the subcodes in the sense of TannerCode: the classical codes whose parity checks are the
        # local checks
        subcode_x, subcode_z = (
            ClassicalCode(_get_local_checks(code_a, code_b, pauli, local_checks))
            for pauli in PAULIS_XZ
        )
        return TannerCode(subgraph_x, subcode_x), TannerCode(subgraph_z, subcode_z)

    @staticmethod
    def get_subgraphs(cayplex: CayleyComplex) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs of the inner (classical) Tanner codes for a quantum Tanner code.

        The vertices of the complex are members of a cover of the group.  Each member a of A acts on
        the cover as its image, (a,1,0) in the quadripartite complex or (a,1) in the bipartite
        complex, and each member b of B as (b,0,1) or (b,1) (see CayleyComplex.cover_subset_a and
        CayleyComplex.cover_subset_b).  Below, a and b stand for these images.  The vertices with
        X-type checks are the identity and the vertices that it reaches through steps ``g -> agb``.
        Each such vertex g has the faces ``f(g,a,b) = {g, ag, gb, agb}`` for a in A and b in B (see
        help(QTCode)).  For each such triple (g, a, b), the (directed) subgraphs have:

        - an edge ``(g, f(g,a,b))`` in subgraph_x, and
        - an edge ``(ag, f(g,a,b))`` in subgraph_z.

        Each edge has a "sort" attribute, which is the label of the face at the source of the edge
        (see QTCode.get_local_views): (a, b) for the edge ``(g, f(g,a,b))``, and ``(a^-1, b)`` for
        the edge ``(ag, f(g,a,b))``.  Classical Tanner codes on these subgraphs order the edges at
        each source by their labels, with members in increasing order, as in QTCode.members_a and
        QTCode.members_b.  Each face also has a "sort" attribute, which numbers the faces in the
        order in which they are found.  This fixes the order of the bits of the classical Tanner
        codes, and so of the qudits of the quantum Tanner code.
        """
        # sort the cover members of the subsets by the total order on the cover group, which fixes
        # the order in which faces are added below, and thereby the qudit that each face of the
        # Cayley complex is identified with
        cover_members_a = sorted(cayplex.cover_subset_a)
        cover_members_b = sorted(cayplex.cover_subset_b)

        # identify the identity element
        member = next(itertools.chain(cover_members_a, cover_members_b))
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
            for aa, bb in itertools.product(cover_members_a, cover_members_b):
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

        # Sort the members of each list, and the coordinates of its local code alike.  Read the
        # members on a common number of points first, so that members that differ only in trailing
        # fixed points count as repeats, since SymPy permutations on different numbers of points are
        # never equal.
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

        Unless exclude_identity is True, the random subsets A and B can contain the identity e of
        the group.  The code is still defined, since the cover that CayleyComplex builds keeps apart
        the corners of a face that are the same member of the group, such as g and eg if e is in A.
        With bipartite=True, subsets that both contain e violate the Total No Conjugacy condition,
        so this method raises a ValueError for them.

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
                the same length.  With bipartite=True, this always raises a ValueError, since A = B
                violates the Total No Conjugacy condition of arXiv:2202.13641, Eq. (1).
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
                  one_subset is True and C_B does not have one coordinate per member of A.
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

    def save_json(self, path: str, *notes: str) -> None:
        """Save this code to a JSON file.

        The file records the subsets, the parity check and generator matrices of the local codes,
        the order of the field, whether the complex is bipartite, and the choice of local checks.
        From these data, QTCode.load_json rebuilds an equal code with the same parity check
        matrices, including the order of checks and qudits.  Equal codes can give different files,
        such as when they chose their local checks differently.

        Args:
            path: The file to write, which is overwritten if it exists.  Missing directories on its
                path are created.
            *notes: Text that the file records, and that QTCode.load_json ignores.

        Raises:
            ValueError: If the field of this code is not galois.GF(order), with its default
                irreducible polynomial and primitive element, since the file records only the order
                of the field.  This method then writes nothing.
        """
        # refuse fields other than galois.GF(order), in which QTCode.load_json reads saved values
        order = self.field.order
        if self.field is not galois.GF(order):
            raise ValueError(
                "Saved files record only the order of the field, so QTCode.save_json saves only"
                f" codes over galois.GF({order}), with its default irreducible polynomial and"
                " primitive element, not over a field with irreducible polynomial"
                f" {self.field.irreducible_poly} and primitive element"
                f" {self.field.primitive_element}"
            )

        data = {
            "notes": list(notes),
            "subset_a": [member.array_form for member in self.members_a],
            "subset_b": [member.array_form for member in self.members_b],
            "code_a": {
                "matrix": self.code_a.matrix.tolist(),
                "generator": self.code_a.generator.tolist(),
            },
            "code_b": {
                "matrix": self.code_b.matrix.tolist(),
                "generator": self.code_b.generator.tolist(),
            },
            "field_order": order,
            "bipartite": bool(self.complex.bipartite),
            "local_checks": self._local_checks,
        }

        # Indent the data, but put each list of integers, such as a member or a row of a matrix, on
        # one line.  The pattern matches a bracketed list that holds only digits, commas, and
        # whitespace, so only the innermost lists of integers.  It cannot match inside a note, since
        # json.dumps escapes line breaks inside strings.
        text = re.sub(
            r"\[\n[\d,\s]*\]",
            lambda match: json.dumps(json.loads(match.group())),
            json.dumps(data, indent=2),
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(text + "\n")

    @staticmethod
    def load_json(path: str) -> QTCode:
        """Load a QTCode from a JSON file written by QTCode.save_json.

        Args:
            path: A file written by QTCode.save_json.

        Raises:
            ValueError: If any of the following holds, with a message that names the file:

                - The file does not exist, does not parse as JSON, or does not hold a JSON object.
                - The object lacks a key that QTCode.save_json writes, or has another key.
                - An entry is malformed, such as a matrix whose rows have different lengths, or a
                  subset whose rows are not distinct permutations in increasing order.
                - The file records an unrecognized choice of local checks, or other data that do not
                  define a QTCode.
        """
        # read the JSON object in the file, and check its keys
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except (ValueError, RecursionError) as error:  # RecursionError for deeply nested input
            raise ValueError(f"Saved code does not parse as JSON: {path}") from error
        if type(data) is not dict:
            raise ValueError(f"Saved code is not a JSON object: {path}")
        keys = [
            "notes",
            "subset_a",
            "subset_b",
            "code_a",
            "code_b",
            "field_order",
            "bipartite",
            "local_checks",
        ]
        if missing := [key for key in keys if key not in data]:
            raise ValueError(f"Saved code has no {missing[0]!r} entry: {path}")
        if unknown := [key for key in data if key not in keys]:
            raise ValueError(f"Saved code has an unknown {unknown[0]!r} entry: {path}")

        def read(
            key: str, is_valid: Callable[[Any], bool], convert: Callable[[Any], Any] = lambda x: x
        ) -> Any:
            """Read an entry, raising a ValueError if it is not valid, or does not convert."""
            try:
                if is_valid(data[key]):
                    return convert(data[key])
            except (OverflowError, ValueError) as error:
                raise ValueError(f"Saved code has a malformed {key!r} entry: {path}") from error
            raise ValueError(f"Saved code has a malformed {key!r} entry: {path}")

        def is_rows(value: object) -> bool:
            """Whether a value is a list of lists of integers, other than bools."""
            return isinstance(value, list) and all(
                isinstance(row, list) and all(type(entry) is int for entry in row) for row in value
            )

        def is_local_code(value: object) -> bool:
            """Whether a value holds the rows of a parity check matrix and a generator matrix."""
            return (
                isinstance(value, dict)
                and value.keys() == {"matrix", "generator"}
                and all(is_rows(rows) for rows in value.values())
            )

        def to_members(rows: list[list[int]]) -> set[abstract.GroupMember]:
            """The members of a subset.

            They must be distinct and in increasing order when written on a common number of points.
            """
            members = [abstract.GroupMember(row) for row in rows]
            size = max((member.size for member in members), default=0)
            resized = [_get_member_on_points(member, size) for member in members]
            if resized != sorted(set(resized)):
                raise ValueError("The members of a subset repeat or are not in increasing order")
            return set(members)

        def to_matrices(
            value: dict[str, list[list[int]]], num_columns: int
        ) -> list[npt.NDArray[np.int_]]:
            """The parity check and generator matrices of a local code, with num_columns columns.

            A matrix has num_columns columns even if it has no rows.
            """
            return [
                np.array(value[name], dtype=int).reshape(len(value[name]), num_columns)
                for name in ["matrix", "generator"]
            ]

        # read and check the entries
        read(
            "notes",
            lambda value: isinstance(value, list) and all(type(note) is str for note in value),
        )
        field_order: int = read("field_order", lambda value: type(value) is int)
        bipartite: bool = read("bipartite", lambda value: type(value) is bool)
        subset_a, subset_b = [read(key, is_rows, to_members) for key in ["subset_a", "subset_b"]]
        matrices_a = read("code_a", is_local_code, lambda value: to_matrices(value, len(subset_a)))
        matrices_b = read("code_b", is_local_code, lambda value: to_matrices(value, len(subset_b)))

        # rebuild the code
        try:
            code_a, code_b = [
                ClassicalCode(matrix, field_order).set_generator(generator)
                for matrix, generator in [matrices_a, matrices_b]
            ]
            return QTCode(
                subset_a,
                subset_b,
                code_a,
                code_b,
                bipartite=bipartite,
                local_checks=data["local_checks"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Saved code does not define a QTCode ({error}): {path}") from error

    @functools.cached_property
    def _local_views(self) -> dict[PauliXZ, npt.NDArray[np.int_]]:
        """The local views of the vertices with checks of each type."""
        # cover members compare by their members of G first, so these orders agree with members_a/b
        cover_members_a = sorted(self.complex.cover_subset_a)
        cover_members_b = sorted(self.complex.cover_subset_b)
        positions_a = {member: index for index, member in enumerate(cover_members_a)}
        positions_b = {member: index for index, member in enumerate(cover_members_b)}
        local_views = {}
        for pauli, subgraph in zip(PAULIS_XZ, self.get_subgraphs(self.complex)):
            # TannerCode sorts the sources of a subgraph, which carry its checks, by the nodes
            # themselves, since they have no "sort" attribute
            sources = sorted(node for node in subgraph if subgraph.out_degree(node))
            views = np.zeros((len(sources), len(positions_a), len(positions_b)), dtype=int)
            for index, source in enumerate(sources):
                for _, face, (member_a, member_b) in subgraph.out_edges(source, data="sort"):
                    position_a, position_b = positions_a[member_a], positions_b[member_b]
                    views[index, position_a, position_b] = subgraph.nodes[face]["sort"]
            local_views[pauli] = views
        return local_views

    def _get_vertex_blocks(
        self, pauli: PauliXZ
    ) -> list[tuple[ClassicalCode, ClassicalCode, npt.NDArray[np.int_]]]:
        """The vertices with checks of one type, as one block, since they share C_A and C_B."""
        return [(self.code_a, self.code_b, self.get_local_views(pauli))]

    def _get_permutation_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Generators of translations of the Cayley complex that preserve this code, or None.

        Let A and B be the cover subsets of the complex, whose vertices are members of the cover
        group (see help(qldpc.objects.CayleyComplex)).  Let x be a vertex with X-type checks such
        that conjugation by x permutes A, and let π map the position i of each member a of A, in
        increasing order, to the position of ``x a x^-1``.  Left translation by x, which maps each
        vertex v to xv, maps the vertices with X-type checks onto themselves, and the face
        ``f(g,a,b) = {g, ag, gb, agb}`` to the face ``f(xg,xax^-1,b)``.  So it maps coordinate
        (i, j) of the local view of each vertex to coordinate (π(i), j) of the local view of its
        image, and it preserves this code if π, as a permutation of coordinates, maps C_A onto
        itself.

        Right translation by x, which maps each vertex v to vx, likewise preserves this code if x is
        a vertex with X-type checks, conjugation ``b -> x^-1 b x`` permutes B, and the induced
        permutation of coordinates maps C_B onto itself.  For each side, the vertices x that satisfy
        these conditions form a group, and this method returns the qudit permutations of generators
        of the two groups.  These conditions are sufficient, not necessary.  The translations omit
        other symmetries, such as those from automorphisms of G, from exchanging A and B, or from
        permutations that preserve the stabilizer group but not each local check space.
        """
        # the faces of the complex, which are its qudits, and the vertices with X-type checks
        subgraph_x, _ = self.get_subgraphs(self.complex)
        faces = {face: qudit for face, qudit in subgraph_x.nodes(data="sort") if qudit is not None}
        vertices_x = sorted(node for node in subgraph_x if subgraph_x.out_degree(node))

        # Find generators of the left and right translations, and permute the faces by each.  Cover
        # members compare by their members of G first, so these orders are those of members_a/b.
        permutations = []
        for left, members, code in [
            (True, sorted(self.complex.cover_subset_a), self.code_a),
            (False, sorted(self.complex.cover_subset_b), self.code_b),
        ]:
            for vertex, _ in _get_translation_generators(vertices_x, members, [code], left=left):
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
    - classical local codes C_A^0 and C_A^1 (code_a0 and code_a1) of length m, and C_B^0 and C_B^1
      (code_b0 and code_b1) of length n, over one finite field.

    The code is built on a quadripartite square complex, whose vertices are the triples (g, s, t) in
    G x {0,1} x {0,1}, where (s, t) is the class of a vertex.  The squares of the complex are the
    triples (i, j, g) for ``i < m``, ``j < n``, and g in G, and the corners of square (i, j, g)
    are::

           (g, 0, 0) ――――――――――――― (g b_j, 0, 1)

               |         (i, j, g)         |

        (a_i g, 1, 0) ―――――――――――― (a_i g b_j, 1, 1)

    Products such as ``a_i g`` are products of qLDPC group members (``a_i * g``).  Square (i, j, g)
    is at coordinate (i, j) of the m x n local view of each of its corners (see
    LQTCode.get_local_views).  The squares are the qudits, and qudit (i, j, g) has the index
    ``(i n + j) |G| + r(g)``, where r(g) is the position of g in ``sorted(group.generate())``.

    A vertex of class (s, t) uses the local codes C_A^s and C_B^t on its local view.  The vertices
    with s = t have X-type checks that span the local check space ``C_A^s ⨂ C_B^t``, and those with
    s != t have Z-type checks that span ``~C_A^s ⨂ ~C_B^t``, where ``~C`` is the dual code of C.
    The argument local_checks chooses which words of each local check space are parity checks
    (see LocalChecks).  LQTCode.get_local_views gives the order of the checks.

    Over GF(2), LQTCode is the code of Eq. (2) of arXiv:2512.20532, in other notation, and
    LQTCode.from_lrz builds it from the data of that paper, in its notation.  LQTCode builds the
    code in the same way over any finite field.

    Notes:

    To relate QTCode to LQTCode, let A and B be symmetric subsets of a group G with identity e,
    listed in increasing order as by QTCode.members_a and QTCode.members_b.  Let ι_A be the
    permutation of positions in A that maps each member to its inverse, let ``C∘ι_A`` be the code
    whose coordinate i is coordinate ι_A(i) of a code C, and likewise for B.  Then the quadripartite
    ``QTCode(A, B, C_A, C_B)`` has the qudits, up to their order, and the stabilizer group of
    ``LQTCode(G, A, B, C_A, C_B, code_a1=C_A∘ι_A, code_b1=C_B∘ι_B)`` restricted to the component K
    of its complex that contains (e, 0, 0) (arXiv:2512.20532, Sec. 1).  The face ``f(g,a_i,b_j)`` of
    this QTCode for a vertex g of class (0, 0) is the square (i, j, g).  The bipartite
    ``QTCode(A, B, C_A, C_B, bipartite=True)`` has the qudits, up to their order, and the stabilizer
    group of the quotient of this LQTCode by the involution ``σ: (g, s, t) -> (g, 1-s, 1-t)`` of its
    complex, restricted to the image of K.  If σ does not map K onto itself, the two QTCodes have
    the same qudits, up to their order, and the same stabilizer group.

    These relations are between stabilizer groups.  The parity checks also correspond, up to the
    order of qudits and checks, if the QTCode and the LQTCode both use local_checks="from_codes",
    and C_A^1 and C_B^1 carry the matrices of C_A and C_B with permuted columns, as
    ``ClassicalCode(H_A[:, ι_A]).set_generator(G_A[:, ι_A])`` does for the parity check matrix H_A
    and generator matrix G_A of C_A.  Then the checks of the quadripartite QTCode are those of the
    LQTCode at the vertices of K, and the checks of the bipartite QTCode are their images under the
    quotient by σ.  Each check of the bipartite QTCode is the image of two checks at vertices of K
    if σ maps K onto itself, and of one otherwise.  Without these conditions on local_checks and on
    C_A^1 and C_B^1, the parity checks can differ: for example, the minimum-weight basis that
    "min_weight" picks can depend on the order of coordinates.

    The group that LQTCode.get_known_permutation_automorphism_group returns includes translations of
    the complex that preserve the local codes.  With the optional pybliss package, it also includes
    the automorphisms that bliss finds, as for QTCode.  It can omit other symmetries, such as those
    from automorphisms of G or from exchanging A and B.

    References:

    - https://arxiv.org/abs/2512.20532 (Sec. 1 and Eq. (2))
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
                "from_codes" (see LocalChecks).

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
            blocks = []
            for code_a, code_b, local_views in self._get_vertex_blocks(pauli):
                local_check_matrix = _get_local_checks(code_a, code_b, pauli, local_checks)
                blocks.append(_place_local_checks(local_views, local_check_matrix, num_qudits))
            matrices.append(np.vstack(blocks).view(codes_a[0].field))
        super().__init__(matrices[0], matrices[1])

        # The checks commute, over any field, by the argument of arXiv:2512.20532, Sec. 1, which
        # also holds when repeated members make two local views share several rows or columns.
        self._is_subsystem_code = False

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

        That paper, by Leverrier, Rozendaal, and Zémor (LRZ), builds a code from a group G, ordered
        lists ``A = (a_1, ..., a_{n_A})`` and ``B = (b_1, ..., b_{n_B})`` of members of G, and
        binary codes C_0 and C_1 of length n_A and C'_0 and C'_1 of length n_B, whose parity check
        matrices H_s and H'_t and generator matrices G_s and G'_t it fixes.  Below, as in LQTCode,
        positions i in A and j in B count from 0, so a_i is the paper's a_{i+1}.  The qubits of the
        paper are the triples (i, j, g), in the order of the index ``(i n_B + j) |G| + r(g)`` of
        LQTCode, and its parity check matrices are

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
        method returns that LQTCode.

        With local_checks="from_codes", the parity checks of the returned code are the rows of H_X
        and H_Z, up to their order, if each code carries the matrices of the paper: its parity check
        matrix as code.matrix, and its generator matrix as code.generator, which
        ClassicalCode.set_generator sets.  This method builds each code ``~C_s`` with the matrices
        of C_s exchanged: its parity check matrix is G_s, and its generator matrix is H_s.  So the
        X-type checks at the vertices of class (s, s) are the rows of ``H_s ⨂ G'_s``, and the Z-type
        checks at the vertices of class (s, t) are the rows of ``G_s ⨂ H'_t``.  For a code given
        only as a parity check matrix, qLDPC computes a generator matrix, so the checks of the
        returned code span the same spaces as those of the paper, but can differ from them.

        The check matrices that the authors published for the paper's [[144, 12, 11]] code, whose
        group G is Z_2 × Z_2, order the qubits by g first.  With G written as {0, 1, 2, 3} under
        bitwise XOR, qubit (i, j, g) is their qubit ``36 h + 6 i + j``, where h is the number that
        stands for g.  So a comparison with those matrices needs a relabeling of the qubits.

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
            ValueError: As for LQTCode, whose local codes C_A^s and C_B^t are ~C_s and C'_t, so an
                error about C_A^s concerns code_s, and one about C_B^t concerns code_t_prime.
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

        # the position r(g) of the member ``g = a_i^-s h b_j^-t`` of the square at coordinate (i, j)
        # of each vertex (h, s, t), and the qudit of that square (i, j, g)
        first_qudits = np.add.outer(np.arange(num_a) * num_b, np.arange(num_b)) * order
        local_views = {}
        for ss, tt in itertools.product(range(2), repeat=2):
            if ss:
                positions_g = np.broadcast_to(shifts_a.T[:, :, None], (order, num_a, num_b))
            else:
                positions_g = np.broadcast_to(
                    np.arange(order)[:, None, None], (order, num_a, num_b)
                )
            if tt:
                positions_g = shifts_b[np.arange(num_b), positions_g]
            local_views[ss, tt] = first_qudits + positions_g
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

        Let x be a member of G such that conjugation by x permutes the list A, and let π be the
        permutation of positions with ``x a_i x^-1 = a_π(i)`` that matches repeated members in order
        (see _get_conjugation_permutation).  Left translation by x, which maps each vertex (g, s, t)
        to (xg, s, t), maps square (i, j, g) to square ``(π(i), j, xg)``, which is the qudit
        permutation of the translation.  It maps coordinate (i, j) of the local view of each vertex
        to coordinate (π(i), j) of the local view of its image, which has the same class, so it
        preserves this code if π, as a permutation of coordinates, maps C_A^0 and C_A^1 onto
        themselves.

        Right translation by x, which maps each vertex (g, s, t) to (gx, s, t), likewise maps square
        (i, j, g) to square ``(i, ρ(j), gx)``, where ``x^-1 b_j x = b_ρ(j)`` as for π, and preserves
        this code if ρ maps C_B^0 and C_B^1 onto themselves.  For each side, the members x that
        satisfy these conditions form a group (see _get_translation_generators), and this method
        returns the qudit permutations of the translations by generators of each group.  These
        conditions are sufficient, not necessary.  The translations omit other symmetries, such as
        permutations that match repeated members in other orders, that come from automorphisms of G,
        that exchange A and B, or that preserve the stabilizer group but not each local check space.
        They also omit permutations that exchange classes of vertices, such as the involution σ of
        the complex (see LQTCode).
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
                image_labels = np.add.outer(np.array(images_a, dtype=int) * num_b, list(images_b))
                permutations.append(
                    np.add.outer(image_labels * len(group_members), images_g).ravel()
                )
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

    The dual ``~code`` would instead carry a generator matrix that qLDPC computes, so "from_codes"
    local checks would not use the parity check matrix that the code carries.
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
            generators = ", ".join(str(generator) for generator in group.generators)
            raise ValueError(
                f"{name} contains {member}, which is not a member of the group of order"
                f" {group.order} generated by {generators}"
            )


def _get_member_on_points(member: abstract.GroupMember, num_points: int) -> abstract.GroupMember:
    """The same member, represented on num_points points, fixing any added points.

    The member must fix every point from num_points on, which are dropped.
    """
    assert member.array_form[num_points:] == list(range(num_points, member.size))
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
    Z-type checks, in the coordinates of a local view, and local_checks chooses them
    (see LocalChecks).  Before asking for minimum-weight bases, callers check the sizes of all local
    codes with _check_min_weight_sizes, so that a request that cannot be met fails before any
    enumeration.
    """
    factors = _get_factor_matrices(code_a, code_b, pauli)
    if local_checks == "min_weight":
        factors = (get_min_weight_basis(factors[0]), get_min_weight_basis(factors[1]))
    return np.kron(*factors).view(code_a.field)


def _check_min_weight_sizes(codes: Iterable[ClassicalCode]) -> None:
    """Raise a ValueError if a local code or its dual has too many words for "min_weight" checks."""
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
    """Generators of the conjugators that permute members by a permutation that preserves codes.

    A candidate x qualifies if conjugation by x, ``m -> x m x^-1`` for left translations or
    ``m -> x^-1 m x`` for right translations, permutes the list of members, and the induced
    permutation of positions (see _get_conjugation_permutation) maps each code onto itself as a
    permutation of its coordinates.  The candidates must include the product of any two candidates
    that qualify.  The candidates that qualify then form a group (see _get_conjugation_permutation),
    and this function returns generators of the group.

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

            # stop once the generators generate every candidate that qualifies: these candidates
            # form a group that contains the group generated, so this holds once their orders agree
            if abstract.Group(*kept, candidate).order == len(qualifying):
                break
    return generators


def _get_conjugation_permutation(
    conjugator: abstract.GroupMember, members: Sequence[abstract.GroupMember], *, left: bool
) -> tuple[int, ...] | None:
    """The permutation of positions by which conjugation by a member x permutes a list of members.

    Conjugation by x is ``m -> x m x^-1`` if left is True, and ``m -> x^-1 m x`` otherwise.  The
    permutation π maps each position i to the position π(i) of the image under conjugation of the
    member at position i.  It matches repeated members in order: the k-th occurrence of a member
    goes to the k-th occurrence of its image.  So the permutation of a product xy of conjugators is
    the permutation of y followed by that of x if left is True, and the permutation of x followed by
    that of y otherwise, as for conjugation by xy itself.  The result is None if conjugation does
    not permute the list, which is to say if some member and its image occur different numbers of
    times.
    """
    # the positions at which each member occurs in the list, in increasing order
    positions: dict[abstract.GroupMember, list[int]] = {}
    for position, member in enumerate(members):
        positions.setdefault(member, []).append(position)

    # map the k-th occurrence of each member to the k-th occurrence of its image under conjugation
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
    space (see _SquareComplexCode._get_canonical_checks).  The result is None if the space has more
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
    local_views: npt.NDArray[np.int_], local_check_matrix: galois.FieldArray, num_qudits: int
) -> galois.FieldArray:
    """Copies of the same local checks at several vertices, as the rows of a parity check matrix.

    Args:
        local_views: An array of shape (number of vertices, m, n), whose entry ``[v, i, j]`` is the
            qudit at coordinate (i, j) of the local view of vertex v.
        local_check_matrix: The local checks, as the rows of a matrix with one column per
            coordinate (i, j) of a local view, in row-major order.
        num_qudits: The number of qudits of the code.

    Returns:
        A matrix whose rows are the local checks on the local view of each vertex, with the checks
        of each vertex in consecutive rows, and the vertices in order.
    """
    num_vertices, num_checks = len(local_views), len(local_check_matrix)
    checks = type(local_check_matrix).Zeros((num_vertices * num_checks, num_qudits))
    rows = np.arange(len(checks)).reshape(num_vertices, num_checks, 1)
    columns = local_views.reshape(num_vertices, 1, local_check_matrix.shape[1])
    checks[rows, columns] = local_check_matrix
    return checks
