"""Instrumental objects used to construct error-correcting codes

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
from collections.abc import Collection, Iterator, Sequence
from typing import Literal

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt

from qldpc import abstract
from qldpc.abstract import DEFAULT_FIELD_ORDER


class Pauli(enum.Enum):
    """Pauli operators."""

    I = (0, 0)  # noqa: E741
    Z = (0, 1)
    X = (1, 0)
    Y = (1, 1)

    def __mul__(self, other: Pauli) -> Pauli:
        """Product of two Pauli operators."""
        val_x = (self.value[0] + other.value[0]) % 2
        val_z = (self.value[1] + other.value[1]) % 2
        return Pauli((val_x, val_z))

    def __invert__(self) -> Pauli:
        """Hadamard-transform this Pauli operator."""
        return Pauli(self.value[::-1])

    def swap_xz(self) -> PauliXZ:
        """Convert between Pauli.X and Pauli.Z, promising a PauliXZ return type (or an error)."""
        if self is Pauli.X:
            return Pauli.Z
        if self is Pauli.Z:
            return Pauli.X
        raise ValueError(
            f"Pauli.dual_xz only converts between Pauli.X and Pauli.Z (provided: {self})"
        )

    def __str__(self) -> str:
        if self is Pauli.I:
            return "I"
        elif self is Pauli.Z:
            return "Z"
        elif self is Pauli.X:
            return "X"
        return "Y"

    @staticmethod
    def from_string(string: str) -> Pauli:
        """Build a Pauli operator from a string."""
        if string == "I":
            return Pauli.I
        elif string == "Z":
            return Pauli.Z
        elif string == "X":
            return Pauli.X
        elif string == "Y":
            return Pauli.Y
        raise ValueError(f"Invalid Pauli operator: {string}")

    @property
    def index(self) -> int:
        """Numerical index for Pauli operators."""
        return self.__index__()

    def __index__(self) -> int:
        """Allow indexing arrays with Pauli operators."""
        if self == Pauli.X:
            return 0
        if self == Pauli.Z:
            return 1
        raise AttributeError(f"No index for {self}.")


PauliXZ = Literal[Pauli.X, Pauli.Z]
PAULIS_XZ: list[PauliXZ] = [Pauli.X, Pauli.Z]


@dataclasses.dataclass(frozen=True)
class QuditPauli:
    """A qudit operator of the form X(val_x)*Z(val_z)."""

    value: tuple[int, int] = (0, 0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuditPauli) and self.value == other.value

    def __invert__(self) -> QuditPauli:
        """Fourier-transform this qudit operator."""
        return QuditPauli(self.value[::-1])

    def __neg__(self) -> QuditPauli:
        """Invert the shifts and phases on this qudit operator."""
        return QuditPauli((-self.value[0], -self.value[1]))

    def __str__(self) -> str:
        val_x, val_z = self.value
        if not val_x and not val_z:
            return "I"
        if val_x == val_z:
            return f"Y({val_z})"
        ops = []
        if val_x:
            ops.append(f"X({val_x})")
        if val_z:
            ops.append(f"Z({val_z})")
        return "*".join(ops)

    @staticmethod
    def from_string(string: str) -> QuditPauli:
        """Build a qudit operator from its string representation."""
        if string == "I":
            return QuditPauli((0, 0))

        invalid_op = f"Invalid qudit operator: {string}"

        val_x, val_z = 0, 0
        factors = string.split("*")
        if len(factors) > 2:
            raise ValueError(invalid_op)

        for factor in factors:
            pauli = factor[0]
            val_str = factor[2:-1]
            _factor = f"{pauli}({val_str})"
            if pauli not in "XYZ" or not val_str.isnumeric() or factor != _factor:
                raise ValueError(invalid_op)

            val = int(val_str)
            if pauli == "X":
                val_x = val
            elif pauli == "Z":
                val_z = val
            else:  # pauli == "Y"
                val_x = val_z = val

        return QuditPauli((val_x, val_z))


@dataclasses.dataclass(frozen=True)
class Node:
    """Node in a Tanner graph.

    A node essentially an integer index, together with a boolean flag to distinguish "data" node
    from a "check" node in an error-correcting code.
    """

    index: int
    is_data: bool = True

    def __hash__(self) -> int:
        return hash((self.index, self.is_data))

    def __lt__(self, other: Node) -> bool:
        if self.is_data == other.is_data:
            return self.index < other.index
        return self.is_data  # data bits "precede" check bits

    def __str__(self) -> str:
        tag = "d" if self.is_data else "c"
        return f"{tag}_{self.index}"


class CayleyComplex:
    """Left-right Cayley complex, used for constructing quantum Tanner codes.

    A Cayley complex is a geometric structure that is built out of two subsets A and B of a group G.
    The subsets respectively act on elements of G from the left and right, and must be symmetric,
    which is to say (for example) that if a ∈ A, then a^-1 ∈ A.

    The generating data (A,B) is used to build vertices V, edges E, and faces F as follows:
    - vertices are members of G,
    - edges have the form (g, ag) and (g, gb), and
    - faces f(g,a,b) have the form {g, ab, gb, agb}:

         g  →  gb
         ↓     ↓
        ag  → agb

    If the complex is disconnected, we keep only the component connected to the identity element.

    After constructing the complex (V,E,F), we can define two bipartite directed graphs:
    - subgraph_x with edges ( g, f(g,a,b)), and
    - subgraph_z with edges (ag, f(g,a,b)).
    These graphs are used to construct classical Tanner codes that serve as the X and Z sectors of a
    quantum CSS code known as a quatnum Tanner code.

    There are, however, two complications to keep in mind.  First, in order for the faces to be
    nondegenerate (that is, for each face to contain four vertices), the generating data (A,B) must
    satisfy the Total No Conjugacy condition:

    [1] ag != gb for all g,a,b in (G,A,B).

    Second, in order to construct a valid quantum Tanner code out of subgraph_x and subgraph_z, the
    graph (V,E) must be bipartite, V = V_0 ∪ V_1, such that (for example) nodes {g, agb} are in one
    partition, while nodes {ag, gb} are in the other partition.  The nodes V_0 and V_1 are then used
    as the source nodes of subgraph_x and subgraph_z.  The graph (V,E) is bipartite if:

    [2] The Cayley graphs (G;A) and (G;B) both are bipartite.

    The Cayley graphs (G;A) and (G;B) are graphs whose
    - vertices are members of G, and
    - edges are pairs of vertices connected by A or B, as in (g, ag) or (g, gb).

    In fact, condition [2] can be enforced at no added cost by taking the double cover of G and
    modifying members of A and B as:
    - G --> G ⨂ Z_2,
    - a --> (a,1), and
    - b --> (b,1),
    where Z_2 ~ {0,1} is the 2-element group (under addition), such that (a,1) * (g,i) = (ag,i+1)
    and (g,i) * (b,1) = (gb,i+1).  The complex generated by A and B after this modification is the
    "bipartite" Cayley complex constructed in https://arxiv.org/abs/2202.13641.

    If requirement [1] is not satisfied, then we can construct a "quadripartite" complex that
    enforces [1] and [2] by taking the quadruple cover of G and modifying members of A and B as:
    - G -->  G ⨂ Z_2 ⨂ Z_2,
    - a --> (a,1,0), and
    - b --> (b,0,1),
    where similarly to before (a,1,0) * (g,i,j) = (ag,i+1,j) and (g,i,j) * (b,0,1) = (gb,i,j+1).
    This modification of A and B corresponds to the "quadripartite" Cayley complex constructed in
    https://arxiv.org/abs/2206.07571.

    References:
    - https://arxiv.org/abs/2202.13641
    - https://arxiv.org/abs/2206.07571
    - https://www.youtube.com/watch?v=orWcstqWGGo
    """

    # generating data
    subset_a: set[abstract.GroupMember]
    subset_b: set[abstract.GroupMember]
    bipartite: bool

    # geometric data
    _graph: nx.Graph | None = None

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember] | None = None,
        *,
        bipartite: bool = False,
    ) -> None:
        """Construct a left-right Cayley complex."""
        if subset_b is None:
            subset_b = subset_a

        # assert that the generating subsets are symmetric
        for subset, name in [(subset_a, "subset_a"), (subset_b, "subset_b")]:
            for member in subset:
                if ~member not in subset:
                    message = (
                        "Provided generating subsets are not symmetric\n"
                        + f"Generating {name} contains {member} but not its inverse, {~member}"
                    )
                    raise ValueError(message)

        # save generating data
        self.subset_a = set(subset_a)
        self.subset_b = set(subset_b)
        self.bipartite = bipartite

        # if asked for a bipartite complex, run a validity check
        if bipartite and not CayleyComplex.satisfies_total_no_conjugacy(subset_a, subset_b):
            raise ValueError("Provided group and subsets do not satisfy Total No Conjugacy")

    @functools.cached_property
    def graph(self) -> nx.Graph:
        """Graph consisting of the nodes and edges of the complex."""
        return CayleyComplex.build_cayley_graph(self.cover_subset_a, self.cover_subset_b)

    @functools.cached_property
    def cover_subset_a(self) -> set[abstract.GroupMember]:
        """Subset induced by taking the double cover(s) of the group for this complex."""
        identity, shift = abstract.CyclicGroup(2).generate()
        if not self.bipartite:
            shift = shift @ identity
        return set(aa @ shift for aa in self.subset_a)

    @functools.cached_property
    def cover_subset_b(self) -> set[abstract.GroupMember]:
        """Subset induced by taking the double cover(s) of the group for this complex."""
        identity, shift = abstract.CyclicGroup(2).generate()
        if not self.bipartite:
            shift = identity @ shift
        return set(bb @ shift for bb in self.subset_b)

    @staticmethod
    def build_cayley_graph(
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember] = (),
    ) -> nx.Graph:
        """Build a left-right Cayley graph generated from the identity element of a group."""
        # identify the identity element
        member = next(iter(subset_a))
        identity = member * ~member

        # identify the set of nodes for which we still need to add edges
        nodes_to_add = set([identity])

        # build the graph one node at a time
        graph = nx.Graph()
        while nodes_to_add:
            gg = nodes_to_add.pop()

            # identify nodes we have already covered, and new nodes we may need to cover
            old_nodes = set(graph.nodes())
            new_nodes = set()

            # add all edges adjacent to this node
            for aa in subset_a:
                aa_gg = aa * gg
                graph.add_edge(gg, aa_gg, type="L")  # "L" for left-acting
                new_nodes.add(aa_gg)
            for bb in subset_b:
                gg_bb = gg * bb
                graph.add_edge(gg, gg_bb, type="R")  # "R" for right-acting
                new_nodes.add(gg_bb)

            nodes_to_add |= new_nodes - old_nodes

        return graph

    @staticmethod
    def satisfies_total_no_conjugacy(
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
    ) -> bool:
        """Check the Total No-Conjugacy condition: aa gg != gg bb for all gg, aa, bb."""
        group = abstract.Group(*subset_a, *subset_b)
        return all(
            aa * gg != gg * bb
            for gg, aa, bb in itertools.product(group.generate(), subset_a, subset_b)
        )


class ChainComplex:
    """Chain complex: a sequence modules with "boundary operators" that map between them.

    An n-chain complex with modules (A_0, A_1, ..., A_n) can be written as

    {} <--[d_0] A_0 <--[d_1] A_1 <-- ... <--[d_n] A_n <--[d_{n+1}] {}

    Here j is called the "degree" of A_j, and d_j : A_j --> A_{j-1} is a "boundary operator" or
    "differential".  Neighboring boundary operators annihilate, in the sense that d_{j-1} d_j = 0.

    In practice, we represent a chain complex by the boundary operators (d_1, d_2, ..., d_n), which
    are in turn represented by matrices over (i) a finite field, or (ii) a group algebra.  The
    boundary operators d_0 and d_{n+1} are formally treated as 0 × dim(A_0) and dim(A_n) × 0
    matrices.

    References:
    - https://en.wikipedia.org/wiki/Chain_complex
    - https://arxiv.org/abs/1810.01519
    - https://arxiv.org/abs/2103.06309
    """

    _field: type[galois.FieldArray]
    _ops: tuple[npt.NDArray[np.int_] | abstract.RingArray, ...]

    # if boundary operators are defined over a group algebra, keep track of their base group
    _group: abstract.Group | None

    def __init__(
        self,
        ops: Sequence[npt.NDArray[np.int_] | abstract.RingArray],
        field: int | None = None,
        skip_validation: bool = False,
    ) -> None:
        # check that either all or none of the operators are defined over a group algebra
        if not (
            all(isinstance(op, abstract.RingArray) for op in ops)
            or not any(isinstance(op, abstract.RingArray) for op in ops)
        ):
            raise ValueError("Invalid or inconsistent operator types provided for a ChainComplex")

        # identify the base field and group for the boundary operators of this chain complex
        fields = set([galois.GF(field)]) if field is not None else set()
        groups = set()
        for op in ops:
            if isinstance(op, abstract.RingArray):
                fields.add(op.field)
                groups.add(op.group)
            elif isinstance(op, galois.FieldArray):
                fields.add(type(op))
        if len(fields) > 1 or len(groups) > 1:
            raise ValueError("Inconsistent base fields (or groups) provided for chain complex")
        self._field = fields.pop() if fields else galois.GF(DEFAULT_FIELD_ORDER)
        self._group = groups.pop() if groups else None

        # identify the boundary operators of this chain complex
        self._ops = tuple(op.view(self.field) for op in ops) if self._group is None else tuple(ops)

        if not skip_validation:
            self._validate_ops()

    def _validate_ops(self) -> None:
        """Validate the consistency of this the boundary operators in this chain complex."""
        for op_a, op_b in zip(self.ops, self.ops[1:]):
            if op_a.shape[1] != op_b.shape[0] or np.any(op_a @ op_b):
                raise ValueError(
                    "Condition for a chain complex not satisfied:\n"
                    "Neighboring boundary operators of a chain complex must compose to zero"
                )

    @property
    def field(self) -> type[galois.FieldArray]:
        """The base field of this chain complex."""
        return self._field

    @property
    def group(self) -> abstract.Group | None:
        """The base group of this chain complex."""
        return self._group

    @property
    def num_links(self) -> int:
        """The number of "internal" links in this chain complex."""
        return len(self.ops)

    @property
    def ops(self) -> tuple[npt.NDArray[np.int_] | abstract.RingArray, ...]:
        """The boundary operators of this chain complex."""
        return self._ops

    def dim(self, degree: int) -> int:
        """The dimension of the module of the given degree."""
        return self.op(degree).shape[1]

    @property
    def T(self) -> ChainComplex:
        """Transpose and reverse the order of the boundary operators in this chain complex."""
        dual_ops = [op.T for op in self.ops[::-1]]
        return ChainComplex(dual_ops, skip_validation=True)

    def op(self, degree: int) -> npt.NDArray[np.int_] | abstract.RingArray:
        """The boundary operator of this chain complex that acts on the module of a given degree."""
        assert 0 <= degree <= self.num_links + 1
        if degree == 0:
            return self.field.Zeros((0, self.ops[0].shape[0]))
        if degree == len(self._ops) + 1:
            return self.field.Zeros((self.ops[-1].shape[1], 0))
        return self.ops[degree - 1]

    @staticmethod
    def tensor_product(  # noqa: C901 ignore complexity check
        chain_a: ChainComplex | npt.NDArray[np.int_] | abstract.RingArray,
        chain_b: ChainComplex | npt.NDArray[np.int_] | abstract.RingArray,
        field: int | None = None,
    ) -> ChainComplex:
        """Tensor product of two chain complexes.

        The tensor product of chain complexes C_A and C_B, respectively with modules (A_0, A_1, ...)
        and (B_0, B_1, ...), is a new chain complex C_P with modules (P_0, P_1, ...).  The module
        P_k of degree k can be written as a direct sum of tensor products A_i ⨂ B_j for which i+j=k,
        that is:

        [1] P_k = ⨁_{i+j=k} A_i ⨂ B_j.

        Elements of P_2, for example, can be written as vectors [a_2 ⨂ b_0, a_1 ⨂ b_1, a_0 ⨂ b_2],
        that concatenate different a_i ⨂ b_j ∈ A_i ⨂ B_j.

        The boundary operator d_k in C_P is defined by its action on each "sector" (i, j), namely

        [2] d_{i+j}(a_i ⨂ b_j) = d_i^A(a_i) ⨂ b_j + (-1)^i a_i ⨂ d_j^B(b_j),

        where d_i^A and d_j^B are boundary operators of C_A and C_B.

        In practice, to construct a boundary operator d_k we build a block matrix whose rows and
        columns correspond, respectively, to sectors of P_{k-1} and P_k.  We then populate this
        block matrix by the maps between sectors of P_k and P_{k-1} that are induced by the
        definition of d_{i+j}.
        """
        if not isinstance(chain_a, ChainComplex):
            chain_a = ChainComplex([chain_a], field=field)
        if not isinstance(chain_b, ChainComplex):
            chain_b = ChainComplex([chain_b], field=field)
        if chain_a.field is not chain_b.field or chain_a.group != chain_b.group:
            raise ValueError("Incompatible chain complexes: different base fields or groups")
        chain_field = chain_a.field

        def get_degree_pairs(degree: int) -> Iterator[tuple[int, int]]:
            """Pairs of degrees that add up to the given total degree."""
            min_deg_a = max(degree - chain_b.num_links, 0)
            max_deg_a = min(chain_a.num_links, degree)
            for deg_a in range(max_deg_a, min_deg_a - 1, -1):
                yield deg_a, degree - deg_a

        def get_block_index(deg_a: int, deg_b: int) -> int:
            """Index of the "sector" with the given degrees in the direct sum of two chains."""
            max_deg_a = min(chain_a.num_links, deg_a + deg_b)
            return max_deg_a - deg_a

        def get_zero_block(
            row_degs: tuple[int, int], col_degs: tuple[int, int]
        ) -> npt.NDArray[np.int_]:
            """Get a zero matrix to fill in a block of a total boundary operator."""
            row_deg_a, row_deg_b = row_degs
            col_deg_a, col_deg_b = col_degs
            rows = chain_a.dim(row_deg_a) * chain_b.dim(row_deg_b)
            cols = chain_a.dim(col_deg_a) * chain_b.dim(col_deg_b)
            return chain_field.Zeros((rows, cols))

        ops = []
        for degree in range(1, chain_a.num_links + chain_b.num_links + 1):
            # fill in zero blocks of the total boundary operator
            blocks = [
                [get_zero_block(row_degs, col_degs) for col_degs in get_degree_pairs(degree)]
                for row_degs in get_degree_pairs(degree - 1)
            ]

            # fill in nonzero blocks of the total boundary operator
            for col, (deg_a, deg_b) in enumerate(get_degree_pairs(degree)):
                op_a = chain_a.op(deg_a)
                op_b = chain_b.op(deg_b)
                if deg_a:
                    row = get_block_index(deg_a - 1, deg_b)
                    iden_b = np.identity(op_b.shape[1], dtype=op_b.dtype)
                    blocks[row][col] = np.kron(op_a, iden_b)  # type:ignore[assignment,arg-type]
                if deg_b:
                    row = get_block_index(deg_a, deg_b - 1)
                    iden_a = np.identity(op_a.shape[1], dtype=op_a.dtype)
                    blocks[row][col] = (
                        np.kron(iden_a, op_b) * (-1) ** deg_a  # type:ignore[arg-type]
                    )

            ops.append(np.block(blocks))

        if chain_a.group is None:
            ops = [op.view(chain_field) for op in ops]
        else:
            ops = [op.view(abstract.RingArray) for op in ops]
        return ChainComplex(ops, skip_validation=True)
