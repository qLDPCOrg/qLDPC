# SPDX-License-Identifier: Apache-2.0

"""Quantum Tanner codes."""

from __future__ import annotations

import ast
import itertools
import os
from collections.abc import Collection, Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc import abstract
from qldpc._utils import networkx as nx
from qldpc.objects import (
    CayleyComplex,
)

from ..classical import (
    TannerCode,
)
from ..common import ClassicalCode, CSSCode, _get_scrambled_seed

####################################################################################################
# quantum Tanner codes


# TODO: example notebook featuring this code
class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    Altogether, a quantum Tanner code is defined by:

    - two symmetric (self-inverse) subsets A and B of a group G, and
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

    Notes:

    - "Good" quantum Tanner code: projective special linear group and random classical codes.

    References:

    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571
    - https://arxiv.org/abs/2202.13641
    """

    complex: CayleyComplex

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        bipartite: bool = False,
    ) -> None:
        """Construct a quantum Tanner code."""
        code_a = ClassicalCode(code_a, field)
        if code_b is not None:
            code_b = ClassicalCode(code_b, field)
        elif len(subset_a) == len(subset_b):
            code_b = ~code_a
        else:
            raise ValueError(
                "Underspecified generating data for quantum Tanner code:\n"
                "no seed code provided for one of the generating subsets"
            )

        if field is None and code_a.field is not code_b.field:
            raise ValueError("The sub-codes provided for this QTCode are over different fields")

        self.code_a = code_a
        self.code_b = code_b
        self.complex = CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        code_x, code_z = self.get_subcodes(self.complex, code_a, code_b)
        super().__init__(code_x, code_z, field)
        self._is_subsystem_code = False  # known by construction

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QTCode)
            and self.code_a == other.code_a
            and self.code_b == other.code_b
            and self.complex.subset_a == other.complex.subset_a
            and self.complex.subset_b == other.complex.subset_b
            and self.complex.bipartite == other.complex.bipartite
        )

    @staticmethod
    def get_subcodes(
        cayplex: CayleyComplex, code_a: ClassicalCode, code_b: ClassicalCode
    ) -> tuple[TannerCode, TannerCode]:
        """Get the classical Tanner subcodes of a quantum Tanner code."""
        subgraph_x, subgraph_z = QTCode.get_subgraphs(cayplex)
        subcode_x = ~ClassicalCode.tensor_product(code_a, code_b)
        subcode_z = ~ClassicalCode.tensor_product(~code_a, ~code_b)
        return TannerCode(subgraph_x, subcode_x), TannerCode(subgraph_z, subcode_z)

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
        """
        # sort the subsets by the total order on the group, which fixes the order in which faces are
        # added below, and thereby the qudit that each face of the Cayley complex is identified with
        subset_a = sorted(cayplex.cover_subset_a)
        subset_b = sorted(cayplex.cover_subset_b)

        # identify the identity element
        identity = subset_a[0] * ~subset_a[0]

        # identify the set of nodes for which we still need to add faces
        nodes_to_add = {identity}

        # build the subgraphs one node at a time
        subgraph_x = nx.DiGraph()
        subgraph_z = nx.DiGraph()
        while nodes_to_add:
            gg = min(nodes_to_add)
            nodes_to_add.remove(gg)

            # identify nodes we have already covered, and new nodes we may need to cover
            old_nodes = set(subgraph_x.nodes())
            new_nodes = set()

            # add all faces adjacent to this node
            for aa, bb in itertools.product(subset_a, subset_b):
                aa_gg, gg_bb = aa * gg, gg * bb
                aa_gg_bb = aa_gg * bb
                face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
                subgraph_x.add_edge(gg, face, sort=(aa, bb))
                subgraph_z.add_edge(aa_gg, face, sort=(~aa, bb))

                new_nodes.add(aa_gg_bb)

            nodes_to_add |= new_nodes - old_nodes

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
        seed: int | None = None,
    ) -> QTCode:
        """Construct a random quantum Tanner code from a base group and seed code(s).

        If only one code C is provided, use its dual ~C for the second code.
        """
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b if code_b is not None else ~code_a, field)
        subset_a = group.random_symmetric_subset(code_a.num_bits, seed=seed)
        if one_subset:
            subset_b = subset_a
        else:
            # scramble the seed so that the second subset is drawn independently of the first, while
            # keeping the construction as a whole reproducible from the given seed.  Reduce it
            # first, so that a seed of any magnitude is accepted, as it is for the first subset.
            seed_b = _get_scrambled_seed(seed % 2**32) if seed is not None else None
            subset_b = group.random_symmetric_subset(code_b.num_bits, seed=seed_b)
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)

    def save(self, path: str, *headers: str) -> None:
        """Save the generating data of this code to a file.

        The members of each subset are written in sorted order, so that equal codes give equal
        files.
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

            # write other data
            file.write(f"# base field: {self.field.order}\n")
            file.write(f"# bipartite: {self.complex.bipartite}\n")

    @staticmethod
    def load(path: str) -> QTCode:
        """Load a QTCode from a file."""
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")

        with open(path, "r") as file:
            lines = file.read().splitlines()

        # load miscellaneous data
        field = ast.literal_eval(lines[-2].split(":")[-1])
        bipartite = ast.literal_eval(lines[-1].split(":")[-1])

        # load integer arrays separated by comments
        arrays = []
        last_index = 0
        for index, line in enumerate(lines):
            if line.startswith("#"):
                if index > last_index + 1:
                    array = np.genfromtxt(lines[last_index + 1 : index], dtype=int, ndmin=2)
                    arrays.append(array)
                last_index = index

        # construct subsets and generating codes
        subset_a = {abstract.GroupMember(gen) for gen in arrays[0]}
        subset_b = {abstract.GroupMember(gen) for gen in arrays[1]}
        code_a = ClassicalCode(arrays[2], field)
        code_b = ClassicalCode(arrays[3], field)
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)
