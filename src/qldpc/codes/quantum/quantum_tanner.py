# SPDX-License-Identifier: Apache-2.0

"""Quantum Tanner codes.

A quantum Tanner code (QTCode) is a CSS code built from two symmetric subsets A and B of a group,
and two classical local codes C_A and C_B, with one coordinate per member of A and of B,
respectively.  Its qudits are the faces ``f(g,a,b) = {g, ag, gb, agb}`` of a left-right Cayley
complex built from A and B, and its parity checks sit at the vertices of the complex: X-type checks
at the corners g of the faces, and Z-type checks at the corners ag (see help(QTCode)).

Local views.  The faces at a vertex are labeled by pairs (a, b) of members of A and B, and each pair
labels one face.  At a vertex g with X-type checks, face ``f(g,a,b)`` has the label (a, b).  At a
vertex ag with Z-type checks, the same face has the label (a^-1, b), since it is ``f(ag,a^-1,b)``.
Coordinate (i, j) of the local view of a vertex is the face whose label is
``(members_a[i], members_b[j])``, where ``members_a`` and ``members_b`` list the members of A and B
in increasing order.  So the local view of a vertex is an ``|A| x |B|`` array of qudits
(QTCode.get_local_views).  In the coordinates of their local views, the X-type checks at a vertex
span the local check space ``C_A ⨂ C_B``, and the Z-type checks span ``~C_A ⨂ ~C_B``, where ``~C``
is the dual code of C.

Local checks.  The checks at each vertex are the words of a spanning set of its local check space,
which the argument ``local_checks`` of QTCode chooses:

- "min_weight" (the default): a minimum-weight basis, whose i-th lightest word is as light as the
  i-th lightest word of any basis of the space, for each i.
- "from_codes": the rows of ``G_A ⨂ G_B`` for X-type checks and of ``H_A ⨂ H_B`` for Z-type checks,
  where H and G are the parity check matrix and generator matrix that each local code carries
  (``code.matrix`` and ``code.generator``).  Rows of H or G that are linearly dependent give
  redundant checks.

Both choices span the same local check space at every vertex, so they give the same stabilizer
group, number of qudits, and dimension.

The minimum-weight bases are tensor products of minimum-weight bases of the factors, such as
``{u_i ⨂ v_j}`` for minimum-weight bases {u_i} of C_A and {v_j} of C_B, which
qldpc.math.get_min_weight_basis finds.  Such a product is a minimum-weight basis of ``C_A ⨂ C_B``,
as follows.  Write ``|x|`` for the weight of a word x.  A basis {u_i} of a space is a minimum-weight
basis iff every word ``x = sum_i c_i u_i`` has ``|x| >= |u_i|`` whenever ``c_i != 0``.  (If
``|x| < |u_i|``, replacing u_i by x gives a lighter basis.  Conversely, the condition says that the
words of the basis of weight at most w span all words of weight at most w, for every w, so no basis
has more words of weight at most w.)  Now let ``X = sum_ij c_ij u_i ⨂ v_j``, and read X as a
matrix whose row r is ``sum_j x_j[r] v_j``, where ``x_j = sum_i c_ij u_i``.  Fix indices i and j
with ``c_ij != 0``.  Then ``|x_j| >= |u_i|``, since x_j has the coefficient c_ij on u_i, and each
of the ``|x_j|`` rows r with ``x_j[r] != 0`` has a nonzero coefficient on v_j, so it has weight at
least ``|v_j|``.  So ``|X| >= |u_i| |v_j| = |u_i ⨂ v_j|``, and ``{u_i ⨂ v_j}`` is a minimum-weight
basis.  The checks are minimum-weight bases of the local check spaces, but generating sets of the
stabilizer group that combine checks at different vertices may be lighter.
"""

from __future__ import annotations

import ast
import functools
import heapq
import itertools
import os
from collections.abc import Iterable, Sequence
from typing import Literal, get_args

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.linalg

from qldpc import abstract
from qldpc.math import get_min_weight_basis
from qldpc.objects import (
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

_LocalChecks = Literal["min_weight", "from_codes"]

# the most words that a local code or its dual may have for "min_weight" local checks
_MAX_WORDS_FOR_MIN_WEIGHT_BASIS = 2**20

# The most words in a local check space of a QTCode, and the most canonical checks at each vertex,
# for which bliss searches the canonical checks of that type.  The first limit bounds the time to
# enumerate words.  The second bounds the size of the matrix that bliss searches, which has a row
# for each canonical check at each vertex.
_MAX_WORDS_FOR_CANONICAL_CHECKS = 2**20
_MAX_CANONICAL_CHECKS_PER_VERTEX = 2**10

####################################################################################################
# quantum Tanner codes


# TODO: example notebook featuring this code
class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    Altogether, a quantum Tanner code is defined by:

    - two symmetric (self-inverse) subsets A and B of a group G, not both empty, and
    - two classical codes ``C_A`` and ``C_B``, respectively with block lengths ``|A|`` and ``|B|``.

    The qudits of a quantum Tanner code live on the faces of a Cayley complex built out of A and B.
    Each face of the Cayley complex looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

    where (g,a,b) is an element of (G,A,B), and ``f(g,a,b) = {g, ag, gb, agb}``.  We define two
    (directed) subgraphs on the Cayley complex:

    - subgraph_x with edges ``( g, f(g,a,b))``, and
    - subgraph_z with edges ``(ag, f(g,a,b))``.

    The X-type parity checks of a quantum Tanner code are then given by the classical Tanner code on
    subgraph_x with subcode ``~(C_A ⨂ C_B)``, where ``~C`` is the dual code to C.  Z-type parity
    checks are similarly given by the classical Tanner code on subgraph_z with subcode
    ``~(~C_A ⨂ ~C_B)``.

    The parity checks at each vertex span the local check space ``C_A ⨂ C_B`` (X-type) or
    ``~C_A ⨂ ~C_B`` (Z-type), in the coordinates of the local view of the vertex.  Coordinate i of
    the local codes belongs to member i of ``members_a`` (members of A in increasing order), and
    likewise for B.  To label coordinates in another order, permute the columns of the local codes.
    The argument ``local_checks`` chooses the checks that span each local check space: a
    minimum-weight basis ("min_weight", the default), or products of the matrices that the local
    codes carry ("from_codes").  The module docstring describes these choices, and proves that
    "min_weight" gives minimum-weight bases.  The choice can change the parity checks, but not the
    stabilizer group.  Minimum-weight bases are exact, and are not chosen to minimize the number of
    checks that act on each qudit.

    Two QTCodes are equal if they have the same generating data and the same parity checks.
    QTCode.save records the choice of local checks, and the generator matrices of the local codes.

    Symmetries: QTCode.get_known_qudit_automorphism_group gives a group of qudit permutations that
    preserve the code.  Its generators include translations of the Cayley complex.  Left
    translation by a vertex x with X-type checks multiplies every vertex by x from the left; it is
    included if conjugation by x permutes A, and the induced permutation of the coordinates of C_A
    maps C_A onto itself.  Right translation by x multiplies every vertex by x from the right; it is
    included if conjugation by x permutes B, and the induced permutation of the coordinates of C_B
    maps C_B onto itself.  With the optional pybliss package, the generators also include the
    automorphisms that bliss finds among the parity checks of the code, and among its canonical
    local checks: the nonzero words of each local check space that are no heavier than the heaviest
    word of a minimum-weight basis of the space, at every vertex.  Unlike the parity checks, the
    canonical checks do not depend on the choice of local_checks.  A permutation of the qudits that
    maps the local views of vertices with checks of each type to local views of vertices with checks
    of the same type, and their local check spaces onto each other, maps the canonical checks onto
    themselves, so bliss can find a much larger group from them.  If the local check spaces of
    X-type or Z-type checks have more than 2^20 words, or more than 2^10 canonical checks at each
    vertex, the canonical checks of that type are replaced by its parity checks.  The group can
    omit other symmetries, such as those from automorphisms of the group G or from exchanging A and
    B.  It contains no ZX-dualities, which exchange X-type and Z-type operators.

    Notes:

    - "Good" quantum Tanner code: projective special linear group and random classical codes.

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
        local_checks: Literal["min_weight", "from_codes"] = "min_weight",
    ) -> None:
        """Construct a quantum Tanner code.

        Args:
            subset_a: The members of the symmetric subset A of a group.  Repeated members are
                merged.
            subset_b: The members of the symmetric subset B of the same group.
            code_a: The local code C_A, or its parity check matrix, with one coordinate per member
                of A.
            code_b: The local code C_B, or its parity check matrix, with one coordinate per member
                of B.
                By default, ``~C_A``, whose generator matrix is the parity check matrix of C_A.
            field: The field of the local codes, if they are given as matrices.
            bipartite: Whether to build the code on the bipartite Cayley complex, rather than the
                quadripartite one (see help(qldpc.objects.CayleyComplex)).
            local_checks: The checks that span each local check space: "min_weight" (the default)
                or "from_codes" (see help(qldpc.codes.quantum.quantum_tanner)).  "min_weight"
                enumerates the words of each local code and of its dual, and raises a ValueError if
                one has more than 2^20 words; pass "from_codes" for such local codes.
        """
        local_checks = _parse_local_checks(local_checks)

        # build the complex first, since it merges repeated members of the subsets
        self.complex = CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        code_a = ClassicalCode(code_a, field)
        if code_b is not None:
            code_b = ClassicalCode(code_b, field)
        elif len(self.complex.subset_a) == len(self.complex.subset_b):
            code_b = _get_dual_code(code_a)
        else:
            raise ValueError(
                "Underspecified generating data for quantum Tanner code:\n"
                "no seed code provided for one of the generating subsets"
            )

        if field is None and code_a.field is not code_b.field:
            raise ValueError("The sub-codes provided for this QTCode are over different fields")
        for name, code, subset in [
            ("A", code_a, self.complex.subset_a),
            ("B", code_b, self.complex.subset_b),
        ]:
            if len(code) != len(subset):
                raise ValueError(
                    f"The seed code C_{name} has block length {len(code)}, but the generating"
                    f" subset {name} has {len(subset)} distinct members"
                )

        self.code_a = code_a
        self.code_b = code_b
        self._local_checks = local_checks
        code_x, code_z = self.get_subcodes(self.complex, code_a, code_b, local_checks=local_checks)
        super().__init__(code_x, code_z, field)
        self._is_subsystem_code = False  # known by construction

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
            {abstract.GroupMember(member.array_form, size=size) for member in subset}
            for subset in subsets
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

        Member i labels coordinate i of the local codes on the A side (see
        help(qldpc.codes.quantum.quantum_tanner)).
        """
        return tuple(sorted(self.complex.subset_a))

    @property
    def members_b(self) -> tuple[abstract.GroupMember, ...]:
        """The members of the subset B, in increasing order.

        Member j labels coordinate j of the local codes on the B side (see
        help(qldpc.codes.quantum.quantum_tanner)).
        """
        return tuple(sorted(self.complex.subset_b))

    def get_local_views(self, pauli: PauliXZLike) -> npt.NDArray[np.int_]:
        """The local views of the vertices with parity checks of one type.

        Entry ``[v, i, j]`` of the returned array is the qudit at coordinate (i, j) of the local
        view of the v-th vertex with checks of the given type: the face whose label at that vertex
        is ``(members_a[i], members_b[j])`` (see help(qldpc.codes.quantum.quantum_tanner)).  The
        checks of each vertex are consecutive rows of ``matrix_x`` or ``matrix_z``, and the vertices
        appear in the order of these rows.

        Args:
            pauli: Pauli.X for the vertices with X-type checks, or Pauli.Z for those with Z-type
                checks.

        Returns:
            A read-only integer array of shape ``(number of vertices, |A|, |B|)``.
        """
        return self._local_views[Pauli.coerce_xz(pauli) is Pauli.Z]

    @staticmethod
    def get_subcodes(
        cayplex: CayleyComplex,
        code_a: ClassicalCode,
        code_b: ClassicalCode,
        *,
        local_checks: Literal["min_weight", "from_codes"] = "min_weight",
    ) -> tuple[TannerCode, TannerCode]:
        """Get the classical Tanner subcodes of a quantum Tanner code.

        The parity checks of the subcodes are the X-type and Z-type checks of the quantum Tanner
        code.  The argument local_checks is as for QTCode.
        """
        local_checks = _parse_local_checks(local_checks)
        subgraph_x, subgraph_z = QTCode.get_subgraphs(cayplex)
        checks_x, checks_z = _get_local_check_spaces(code_a, code_b, local_checks)
        return TannerCode(subgraph_x, checks_x), TannerCode(subgraph_z, checks_z)

    @staticmethod
    def get_subgraphs(cayplex: CayleyComplex) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs of the inner (classical) Tanner codes for a quantum Tanner code.

        These subgraphs are defined using the faces of a Cayley complex.  Each face looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

        where ``f(g,a,b) = {g, ag, gb, agb}``.  Specifically, the (directed) subgraphs are:

        - subgraph_x with edges ``( g, f(g,a,b))``, and
        - subgraph_z with edges ``(ag, f(g,a,b))``.

        Classical Tanner codes on these subgraphs are used to construct a quantum Tanner code.

        As a matter of practice, defining classical Tanner codes on subgraph_x and subgraph_z
        requires choosing an ordering on the edges incident to every source node of these graphs.
        If the group G is equipped with a total order, a natural ordering of edges incident to every
        source node is induced by assigning the label (a, b) to edge ``(g, f(g,a,b))``.  Consistency
        then requires that edge ``(ag, f(g,a,b))`` has label ``(a^-1, b)``, as verified by defining
        ``g' = ag`` and checking that ``f(g,a,b) = f(g',a^-1,b)``.

        Each face also has a "sort" attribute, which numbers the faces in the order in which they
        are found.  This fixes the order of the bits of the classical Tanner codes, and so of the
        qudits of the quantum Tanner code.
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
        local_checks: Literal["min_weight", "from_codes"] = "min_weight",
    ) -> QTCode:
        """Construct a random quantum Tanner code from a base group and seed code(s).

        If only one code C is provided, use its dual ~C for the second code.  If exclude_identity is
        True, the random subsets exclude the identity element e of the group, as the generating sets
        of Cayley graphs usually do.  If A contains the identity, the faces
        ``f(g,e,b) = {g, eg, gb, egb}`` have vertices g and eg that are the same group member, and
        similarly for B.  The double or quadruple covers built by CayleyComplex keep these vertices
        apart, so the code is still defined.  See help(qldpc.abstract.Group.random_symmetric_subset)
        for the sizes of subsets without the identity, and the error raised for other sizes.  The
        argument local_checks is as for QTCode.
        """
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b if code_b is not None else _get_dual_code(code_a), field)
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
        QTCode.load rebuilds an equal code if the field is galois.GF(order), with its default
        irreducible polynomial and primitive element.  For a field with any other representation,
        QTCode.load can raise an error or build a different code.  The members of each subset are
        written in sorted order, so that the file does not depend on the order in which the subsets
        list their members.  Equal codes can still give different files, if they chose their local
        checks differently, or if their local codes carry different generator matrices.
        """
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
            file.write(f"# base field: {self.field.order}\n")
            file.write(f"# bipartite: {self.complex.bipartite}\n")

    @staticmethod
    def load(
        path: str,
        *,
        local_checks: Literal["min_weight", "from_codes"] | None = None,
    ) -> QTCode:
        """Load a QTCode from a file written by QTCode.save.

        The code chooses its local checks as the saved code did, unless the argument local_checks
        (as for QTCode) is given.  Files written by QTCode.save in earlier releases record no choice
        of local checks, and no generator matrices of the local codes, and load with "min_weight".
        Their codes had parity checks in reduced row echelon form, which span the same local check
        spaces, so the loaded code has the same stabilizer group, but can have other parity checks.
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

        def read_rows(name: str) -> list[list[int]]:
            """Read the rows of integers in a section, which can be empty or have empty rows."""
            next_start = min(start for start in starts.values() if start > starts[name])
            section_lines = lines[starts[name] + 1 : next_start]
            return [[int(value) for value in line.split()] for line in section_lines]

        # construct subsets and generating codes
        field = ast.literal_eval(lines[starts["base field"]].split(":")[-1])
        bipartite = ast.literal_eval(lines[starts["bipartite"]].split(":")[-1])
        subset_a = {abstract.GroupMember(row) for row in read_rows("subset_a")}
        subset_b = {abstract.GroupMember(row) for row in read_rows("subset_b")}

        # a seed code without parity checks still has one bit per member of its subset
        code_a, code_b = [
            ClassicalCode(np.array(rows, dtype=int).reshape(len(rows), len(subset)), field)
            for rows, subset in [
                (read_rows("code_a.matrix"), subset_a),
                (read_rows("code_b.matrix"), subset_b),
            ]
        ]

        # restore the choice of local checks, and the generator matrices of the local codes
        if choice_starts:
            for code, name in [(code_a, "code_a"), (code_b, "code_b")]:
                rows = ast.literal_eval(lines[starts[f"{name}.generator"]].split(":", 1)[1].strip())
                code.set_generator(np.array(rows, dtype=int).reshape(len(rows), len(code)))
            if local_checks is None:
                local_checks = _parse_local_checks(lines[starts["local checks"]].split(":")[-1])
        return QTCode(
            subset_a,
            subset_b,
            code_a,
            code_b,
            bipartite=bipartite,
            local_checks="min_weight" if local_checks is None else local_checks,
        )

    @functools.cached_property
    def _local_views(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """The local views of the vertices with X-type checks, and of those with Z-type checks."""
        members_a = sorted(self.complex.cover_subset_a)
        members_b = sorted(self.complex.cover_subset_b)
        positions_a = {member: index for index, member in enumerate(members_a)}
        positions_b = {member: index for index, member in enumerate(members_b)}
        local_views = []
        for subgraph in self.get_subgraphs(self.complex):
            # TannerCode orders the sources of a subgraph, which carry checks, by the sources alone
            sources = sorted(node for node in subgraph if subgraph.out_degree(node))
            local_view = np.zeros((len(sources), len(positions_a), len(positions_b)), dtype=int)
            for index, source in enumerate(sources):
                for _, face, (member_a, member_b) in subgraph.out_edges(source, data="sort"):
                    position_a, position_b = positions_a[member_a], positions_b[member_b]
                    local_view[index, position_a, position_b] = subgraph.nodes[face]["sort"]
            local_views.append(_mark_read_only(local_view))
        return local_views[0], local_views[1]

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Generators of translations of the Cayley complex that preserve this code, or None.

        Here A and B are the cover subsets of the complex, whose vertices are members of the cover
        group (see help(qldpc.objects.CayleyComplex)), and V is the set of vertices.  Let x be a
        vertex with X-type checks, such that conjugation by x permutes A, mapping member i of A to
        member π(i) in increasing order, and such that π, as a permutation of coordinates, maps C_A
        onto itself.  Then left translation by x, which maps each vertex v to xv, preserves this
        code:

        - It maps V onto itself.  The vertices are the products uw of a member u of the group <A>
          generated by A and a member w of <B>, since left and right multiplications commute.  So
          x = u'w' for some such u' and w', and since conjugation by x maps <A> onto itself,
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
        group, if it is a vertex.

        These conditions are sufficient, not necessary.  The translations omit other symmetries,
        such as those from automorphisms of the group, from exchanging A and B, or from permutations
        that preserve the stabilizer group but not each local check space.
        """
        # the faces of the complex, which are its qudits, and the vertices with X-type checks
        subgraph_x, _ = self.get_subgraphs(self.complex)
        faces = {face: qudit for face, qudit in subgraph_x.nodes(data="sort") if qudit is not None}
        vertices = sorted(node for node in subgraph_x if subgraph_x.out_degree(node))

        # find generators of the left and right translations, and permute the faces by each
        permutations = []
        for left, members, code in [
            (True, sorted(self.complex.cover_subset_a), self.code_a),
            (False, sorted(self.complex.cover_subset_b), self.code_b),
        ]:
            for vertex in _get_translations(vertices, members, code, left=left):
                permutation = np.empty(len(self), dtype=int)
                for face, qudit in faces.items():
                    image = frozenset(vertex * node if left else node * vertex for node in face)
                    permutation[qudit] = faces[image]
                permutations.append(permutation)
        return np.array(permutations, dtype=int) if permutations else None

    def _get_automorphism_search_matrices(self) -> tuple[galois.FieldArray, ...]:
        """The parity checks of this code and its canonical local checks, for bliss to search.

        Bliss can find more automorphisms among the canonical checks (see _get_canonical_checks),
        but it can also find automorphisms among the parity checks that do not map the canonical
        checks onto themselves, such as when a local code is the direct sum of codes on disjoint
        sets of coordinates.  So bliss searches both.
        """
        canonical_checks = self._get_canonical_checks()
        if np.array_equal(canonical_checks, self.matrix):
            return (self.matrix,)
        return (self.matrix, canonical_checks)

    def _get_canonical_checks(self) -> galois.FieldArray:
        """The canonical local checks of this code, in the layout of its parity check matrix.

        At each vertex, the canonical local checks are the nonzero words of its local check space
        whose weight is at most w*, the largest weight of a word in a minimum-weight basis of the
        space (all such bases have the same weights), on the local view of the vertex.  For the
        space ``C_A ⨂ C_B``, w* is the product of the largest weights of words in minimum-weight
        bases of C_A and C_B, since tensor products of minimum-weight bases are minimum-weight bases
        (see help(qldpc.codes.quantum.quantum_tanner)), and likewise for ``~C_A ⨂ ~C_B``.

        - The canonical checks of a vertex include a minimum-weight basis of its local check space,
          so they span it, and the canonical checks of all vertices span the row spaces of matrix_x
          and matrix_z.  So a permutation of the qudits that maps the set of canonical checks onto
          itself preserves this code.
        - Unlike the parity checks, the canonical checks depend only on the local check spaces.  A
          permutation of coordinates that maps one local check space onto another, of checks of the
          same type, preserves weights, so it maps canonical checks to canonical checks.  So bliss
          can find automorphisms that map local views to local views of the same type without
          mapping parity checks to parity checks.

        Finding the canonical checks takes time proportional to the number of words of each local
        check space.  If a local check space has more than 2^20 words, or more than 2^10 canonical
        checks at each vertex, the matrix holds the parity checks of that type instead, which depend
        on the choice of local checks.
        """
        blocks = []
        sectors: list[tuple[PauliXZ, list[galois.FieldArray], galois.FieldArray]] = [
            (Pauli.X, [self.code_a.generator, self.code_b.generator], self.matrix_x),
            (Pauli.Z, [self.code_a.matrix, self.code_b.matrix], self.matrix_z),
        ]
        for pauli, factors, parity_checks in sectors:
            # find the canonical checks of one local view, or fall back to the parity checks
            canonical_checks = _get_canonical_local_checks(*factors)
            if canonical_checks is None:
                blocks.append(parity_checks)
                continue

            # place the canonical checks on the local view of every vertex
            local_views = self.get_local_views(pauli)
            local_views = local_views.reshape(len(local_views), canonical_checks.shape[1])
            checks = self.field.Zeros((len(local_views) * len(canonical_checks), len(self)))
            rows = np.arange(len(checks)).reshape(len(local_views), len(canonical_checks))
            for vertex_rows, local_view in zip(rows, local_views):
                checks[np.ix_(vertex_rows, local_view)] = canonical_checks
            blocks.append(checks)
        return scipy.linalg.block_diag(*blocks).view(self.field)


def _parse_local_checks(value: object) -> _LocalChecks:
    """Read a choice of local checks, raising a ValueError if it is not one."""
    for local_checks in get_args(_LocalChecks):
        if isinstance(value, str) and value.strip() == local_checks:
            return local_checks
    raise ValueError(
        f"Unrecognized choice of local checks for a QTCode: {value!r} (expected one of"
        f" {get_args(_LocalChecks)})"
    )


def _get_dual_code(code: ClassicalCode) -> ClassicalCode:
    """The dual of a classical code, whose generator matrix is the parity check matrix of the code.

    So the dual carries the same two matrices as the code, with their roles exchanged.
    """
    return ClassicalCode(code.generator).set_generator(code.matrix)


def _get_local_check_spaces(
    code_a: ClassicalCode, code_b: ClassicalCode, local_checks: _LocalChecks
) -> tuple[ClassicalCode, ClassicalCode]:
    """Classical codes whose parity checks are the local X-type and Z-type checks of a QTCode.

    Their parity checks span the local check spaces ``C_A ⨂ C_B`` and ``~C_A ⨂ ~C_B``, as the
    argument local_checks of QTCode chooses (see help(qldpc.codes.quantum.quantum_tanner)).
    """
    # spanning sets of the local codes C_A and C_B, and of their duals ~C_A and ~C_B
    factors_x = [code_a.generator, code_b.generator]
    factors_z = [code_a.matrix, code_b.matrix]
    if local_checks == "min_weight":
        for code in [code_a, code_b]:
            for num_words, space in [
                (code.field.order**code.dimension, "local code"),
                (code.field.order**code.rank, "dual of a local code"),
            ]:
                if num_words > _MAX_WORDS_FOR_MIN_WEIGHT_BASIS:
                    raise ValueError(
                        f"Minimum-weight local checks enumerate the words of each local code and"
                        f" of its dual, but a {space} has {num_words} words, more than 2^20.  Pass"
                        " local_checks='from_codes' to build the checks from the matrices that the"
                        " local codes carry."
                    )
        factors_x = [get_min_weight_basis(factor) for factor in factors_x]
        factors_z = [get_min_weight_basis(factor) for factor in factors_z]
    return ClassicalCode(np.kron(*factors_x)), ClassicalCode(np.kron(*factors_z))


def _get_translations(
    vertices: Sequence[abstract.GroupMember],
    members: Sequence[abstract.GroupMember],
    code: ClassicalCode,
    *,
    left: bool,
) -> list[abstract.GroupMember]:
    """Generators of the group of vertices that translate a QTCode from the left or the right.

    A vertex x qualifies if conjugation by x, ``m -> x m x^-1`` for left translations or
    ``m -> x^-1 m x`` for right translations, permutes the members, and the induced permutation of
    the coordinates of the code maps the code onto itself (see QTCode._get_qudit_automorphisms).

    Args:
        vertices: The vertices with X-type checks, in increasing order.
        members: The members of a cover subset of the complex, in increasing order.
        code: The local code whose coordinates the members label.
        left: Whether to find left translations, rather than right translations.
    """
    # index the members, and find the vertices whose conjugation permutes them and keeps the code
    positions = {member: index for index, member in enumerate(members)}
    preserves_code: dict[tuple[int, ...], bool] = {}
    qualifying = []
    for vertex in vertices:
        conjugates = [
            vertex * member * ~vertex if left else ~vertex * member * vertex for member in members
        ]
        if not all(conjugate in positions for conjugate in conjugates):
            continue
        permutation = tuple(positions[conjugate] for conjugate in conjugates)
        if permutation not in preserves_code:
            preserves_code[permutation] = _preserves_row_space(
                code.matrix, np.array(permutation, dtype=int), len(code)
            )
        if preserves_code[permutation]:
            qualifying.append(vertex)

    # keep each qualifying vertex that those kept before it do not generate
    generators: list[abstract.GroupMember] = []
    for vertex in qualifying:
        if not vertex.is_Identity and not (generators and vertex in abstract.Group(*generators)):
            generators.append(vertex)
            if abstract.Group(*generators).order == len(qualifying):
                break
    return generators


def _get_canonical_local_checks(
    factor_a: galois.FieldArray, factor_b: galois.FieldArray
) -> galois.FieldArray | None:
    """The canonical checks of the tensor product of the row spaces of two matrices, or None.

    The canonical checks are the nonzero words of the space whose weight is at most the largest
    weight of a word in a minimum-weight basis of the space (see
    QTCode._get_canonical_checks).  The result is None if the space has more than 2^20
    words, or more than 2^10 canonical checks.
    """
    field = type(factor_a)
    basis = np.kron(factor_a.row_space(), factor_b.row_space()).view(field)
    num_words = field.order ** len(basis)
    if num_words > _MAX_WORDS_FOR_CANONICAL_CHECKS:
        return None
    if not len(basis):
        return field.Zeros((0, basis.shape[1]))

    # Products of minimum-weight bases are minimum-weight bases, so their largest weights multiply.
    # Each factor has at most as many words as the space, since neither factor is zero.
    max_weight = 1
    for factor in [factor_a, factor_b]:
        weights = np.count_nonzero(get_min_weight_basis(factor).view(np.ndarray), axis=1)
        max_weight *= int(np.max(weights, initial=0))

    # enumerate the words of the space in chunks, keeping the light ones
    place_values = field.order ** np.arange(len(basis), dtype=np.int64)
    checks = [field.Zeros((0, basis.shape[1]))]
    num_checks = 0
    for start in range(0, num_words, chunk_size := 2**14):
        indices = np.arange(start, min(start + chunk_size, num_words), dtype=np.int64)
        words = field(indices[:, None] // place_values % field.order) @ basis
        weights = np.count_nonzero(words.view(np.ndarray), axis=1)
        checks.append(words[(weights > 0) & (weights <= max_weight)])
        num_checks += len(checks[-1])
        if num_checks > _MAX_CANONICAL_CHECKS_PER_VERTEX:
            return None
    return np.vstack(checks).view(field)
