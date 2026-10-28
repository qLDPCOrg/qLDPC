# SPDX-License-Identifier: Apache-2.0

"""Objects for building codes: Pauli operators, Tanner graphs, Cayley and chain complexes."""

from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.csgraph

from qldpc._utils import networkx as nx
from qldpc.math import resolve_field

if TYPE_CHECKING:
    from qldpc import abstract


class Pauli(enum.Enum):
    """Pauli operators."""

    I = (0, 0)
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
            f"Pauli.swap_xz only converts between Pauli.X and Pauli.Z (provided: {self})"
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

    @staticmethod
    def coerce_xz(value: PauliXZLike) -> PauliXZ:
        """Convert Pauli.X, Pauli.Z, or a case-insensitive "X" or "Z" string into a PauliXZ.

        Raises:
            ValueError: If the value does not identify Pauli.X or Pauli.Z.
        """
        if isinstance(value, str):
            if value.upper() == "X":
                return Pauli.X
            if value.upper() == "Z":
                return Pauli.Z
        elif value is Pauli.X or value is Pauli.Z:
            return value
        raise ValueError(
            f"Expected Pauli.X or Pauli.Z, or a case-insensitive 'X' or 'Z' string (got {value!r})"
        )

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
        raise TypeError(f"Only Pauli.X and Pauli.Z have an index (got {self})")


PauliXZ = Literal[Pauli.X, Pauli.Z]
PAULIS_XZ: list[PauliXZ] = [Pauli.X, Pauli.Z]

# inputs that Pauli.coerce_xz converts into Pauli.X or Pauli.Z
PauliXZLike = PauliXZ | Literal["X", "Z", "x", "z"]


@dataclasses.dataclass(frozen=True)
class QuditPauli:
    """A qudit operator of the form ``X(val_x)*Z(val_z)``."""

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
            pauli = factor[:1]
            val_str = factor[2:-1]
            _factor = f"{pauli}({val_str})"
            if pauli not in "XYZ" or not val_str.isdecimal() or factor != _factor:
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

    A node is essentially an integer index, together with a boolean flag to distinguish "data" node
    from a "check" node in an error-correcting code.
    """

    index: int
    is_data: bool = True

    def __lt__(self, other: Node) -> bool:
        if self.is_data == other.is_data:
            return self.index < other.index
        return self.is_data  # data bits "precede" check bits

    def __str__(self) -> str:
        tag = "d" if self.is_data else "c"
        return f"{tag}_{self.index}"


class TannerGraph(nx.DiGraph):
    """Tanner graph of a parity check matrix, with edges directed from check nodes to data nodes.

    A Tanner graph is a bipartite graph whose nodes are the data (qu)dits and the checks of a code,
    identified by Node objects.  The check node c and the data node d share an edge iff check c
    addresses d.  Every row of the parity check matrix has a check node, so a check that addresses
    nothing is an isolated node, and the graph records the number of checks.

    A Tanner graph of a classical parity check matrix H (with ``symplectic=False``) has a data node
    for every column of H.  Edge (c, d) is in the graph iff ``H[c, d] != 0``, and its "val" data is
    the entry ``H[c, d]``.

    A Tanner graph of a symplectic parity check matrix (with ``symplectic=True``), whose rows have
    the form ``[X|Z]``, has a data node for every qudit, that is, for every pair of columns (j, j+n)
    of the matrix, where n is the number of qudits.  The data of edge (c, d) maps the Pauli class to
    the operator that check c applies to qudit d: a Pauli if the field is GF(2), and a QuditPauli
    otherwise.

    The field and symplectic attributes are stored in the graph attribute dictionary, ``graph``, so
    that copies, subgraph views, and other graphs that networkx derives from this one keep them.
    """

    @staticmethod
    def from_matrix(
        matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]], *, symplectic: bool = False
    ) -> TannerGraph:
        """Build the Tanner graph of a parity check matrix.

        Args:
            matrix: The parity check matrix.  Its field is that of a galois.FieldArray, or GF(2)
                otherwise.
            symplectic: Whether the rows of the matrix are symplectic vectors of the form
                ``[X|Z]``.  Defaults to False.

        Returns:
            The Tanner graph of the matrix, with data nodes before check nodes.
        """
        matrix = np.asanyarray(matrix)
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF2
        graph = TannerGraph(field=field, symplectic=symplectic)
        if not symplectic:
            graph.add_nodes_from(Node(index=bit, is_data=True) for bit in range(matrix.shape[-1]))
            graph.add_nodes_from(Node(index=check, is_data=False) for check in range(len(matrix)))
            for row, col in zip(*np.nonzero(matrix)):
                node_check = Node(index=int(row), is_data=False)
                node_bit = Node(index=int(col), is_data=True)
                graph.add_edge(node_check, node_bit, val=matrix[row][col])
            return graph

        matrix = np.reshape(matrix, (len(matrix), 2, matrix.shape[-1] // 2))
        graph.add_nodes_from(Node(index=qudit, is_data=True) for qudit in range(matrix.shape[-1]))
        graph.add_nodes_from(Node(index=check, is_data=False) for check in range(len(matrix)))
        _Pauli = Pauli if field is galois.GF2 else QuditPauli
        for row, xz, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)
            qudit_op = graph[node_check][node_qudit].get(Pauli, _Pauli((0, 0)))
            vals_xz = list(qudit_op.value)
            vals_xz[xz] += int(matrix[row, xz, col])
            graph[node_check][node_qudit][Pauli] = _Pauli(tuple(vals_xz))
        return graph

    @property
    def field(self) -> type[galois.FieldArray]:
        """The field of the parity check matrix of this graph, GF(2) by default."""
        field: type[galois.FieldArray] = self.graph.get("field", galois.GF2)
        return field

    @field.setter
    def field(self, field: type[galois.FieldArray]) -> None:
        self.graph["field"] = field

    @property
    def symplectic(self) -> bool:
        """Whether this graph represents a symplectic parity check matrix, False by default."""
        return bool(self.graph.get("symplectic", False))

    def to_matrix(self) -> galois.FieldArray:
        """The parity check matrix of this Tanner graph."""
        num_data = sum(node.is_data for node in self.nodes())
        num_checks = len(self.nodes()) - num_data
        if not self.symplectic:
            matrix = self.field.Zeros((num_checks, num_data))
            for node_check, node_bit, data in self.edges(data=True):
                matrix[node_check.index, node_bit.index] = data.get("val", 1)
            return matrix

        int_matrix = np.zeros((num_checks, 2, num_data), dtype=int)
        for node_check, node_qudit, data in self.edges(data=True):
            int_matrix[node_check.index, :, node_qudit.index] = data.get(Pauli).value
        return self.field(int_matrix.reshape(num_checks, 2 * num_data))


class CayleyComplex:
    """Left-right Cayley complex, used for constructing quantum Tanner codes.

    A Cayley complex is a geometric structure that is built out of two subsets A and B of a group G.
    The subsets respectively act on elements of G from the left and right, and must be symmetric,
    which is to say (for example) that if ``a ∈ A``, then ``a^-1 ∈ A``.

    The generating data (A,B) is used to build vertices V, edges E, and faces F as follows:

    - vertices are members of G,
    - edges have the form (g, ag) and (g, gb), and
    - faces f(g,a,b) have the form {g, ab, gb, agb}::

         g  →  gb
         ↓     ↓
        ag  → agb

    If the complex is disconnected, we keep only the component connected to the identity element.

    After constructing the complex (V,E,F), we can define two bipartite directed graphs:

    - subgraph_x with edges ( g, f(g,a,b)), and
    - subgraph_z with edges (ag, f(g,a,b)).

    These graphs are used to construct classical Tanner codes that serve as the X and Z sectors of a
    quantum CSS code known as a quantum Tanner code.

    There are, however, two complications to keep in mind.  First, in order for the faces to be
    nondegenerate (that is, for each face to contain four vertices), the generating data (A,B) must
    satisfy the Total No Conjugacy condition:

    [1] ``ag != gb`` for all g,a,b in (G,A,B).

    Second, in order to construct a valid quantum Tanner code out of subgraph_x and subgraph_z, the
    graph (V,E) must be bipartite, ``V = V_0 ∪ V_1``, such that (for example) nodes {g, agb} are in
    one partition, while nodes {ag, gb} are in the other partition.  The nodes V_0 and V_1 are then
    used as the source nodes of subgraph_x and subgraph_z.  The graph (V,E) is bipartite if:

    [2] The Cayley graphs (G;A) and (G;B) both are bipartite.

    The Cayley graphs (G;A) and (G;B) are graphs whose

    - vertices are members of G, and
    - edges are pairs of vertices connected by A or B, as in (g, ag) or (g, gb).

    In fact, condition [2] can be enforced at no added cost by taking the double cover of G and
    modifying members of A and B as:

    - ``G --> G ⨂ Z_2``,
    - ``a --> (a,1)``, and
    - ``b --> (b,1)``,

    where ``Z_2 ~ {0,1}`` is the 2-element group (under addition), such that
    ``(a,1) * (g,i) = (ag,i+1)`` and ``(g,i) * (b,1) = (gb,i+1)``.  The complex generated by A and B
    after this modification is the "bipartite" Cayley complex constructed in
    https://arxiv.org/abs/2202.13641.

    If requirement [1] is not satisfied, then we can construct a "quadripartite" complex that
    enforces [1] and [2] by taking the quadruple cover of G and modifying members of A and B as:

    - ``G --> G ⨂ Z_2 ⨂ Z_2``,
    - ``a --> (a,1,0)``, and
    - ``b --> (b,0,1)``,

    where similarly to before ``(a,1,0) * (g,i,j) = (ag,i+1,j)`` and
    ``(g,i,j) * (b,0,1) = (gb,i,j+1)``.  This modification of A and B corresponds to the
    "quadripartite" Cayley complex constructed in
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
        from qldpc import abstract

        identity, shift = abstract.CyclicGroup(2).generate()
        if not self.bipartite:
            shift = shift @ identity
        return {aa @ shift for aa in self.subset_a}

    @functools.cached_property
    def cover_subset_b(self) -> set[abstract.GroupMember]:
        """Subset induced by taking the double cover(s) of the group for this complex."""
        from qldpc import abstract

        identity, shift = abstract.CyclicGroup(2).generate()
        if not self.bipartite:
            shift = identity @ shift
        return {bb @ shift for bb in self.subset_b}

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
        nodes_to_add = {identity}

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
        """Check the Total No-Conjugacy condition: ``aa gg != gg bb`` for all gg, aa, bb."""
        from qldpc import abstract

        group = abstract.Group(*subset_a, *subset_b)
        return all(
            aa * gg != gg * bb
            for gg, aa, bb in itertools.product(group.generate(), subset_a, subset_b)
        )


class ChainComplex:
    """Chain complex: a sequence modules with "boundary operators" that map between them.

    An n-chain complex with modules (A_0, A_1, ..., A_n) can be written as

    ``{} <--[d_0] A_0 <--[d_1] A_1 <-- ... <--[d_n] A_n <--[d_{n+1}] {}``

    Here j is called the "degree" of A_j, and ``d_j : A_j --> A_{j-1}`` is a "boundary operator" or
    "differential".  Neighboring boundary operators annihilate, in the sense that
    ``d_{j-1} d_j = 0``.

    In practice, we represent a chain complex by the boundary operators (d_1, d_2, ..., d_n), which
    are in turn represented by matrices over (i) a finite field, or (ii) a group algebra.  The
    boundary operators d_0 and d_{n+1} are formally treated as ``0 × dim(A_0)`` and ``dim(A_n) × 0``
    matrices.

    References:

    - https://en.wikipedia.org/wiki/Chain_complex
    - https://arxiv.org/abs/1810.01519
    - https://arxiv.org/abs/2103.06309
    """

    _field: type[galois.FieldArray]
    _ops: tuple[npt.NDArray[np.int_] | abstract.RingArray, ...]

    # if boundary operators are defined over a group algebra, keep track of the base ring
    _ring: abstract.GroupRing | None

    def __init__(
        self,
        ops: Sequence[npt.NDArray[np.int_] | abstract.RingArray],
        field: int | type[galois.FieldArray] | None = None,
        *,
        skip_validation: bool = False,
    ) -> None:
        from qldpc import abstract

        # check that either all or none of the operators are defined over a group algebra
        if not (
            all(isinstance(op, abstract.RingArray) for op in ops)
            or not any(isinstance(op, abstract.RingArray) for op in ops)
        ):
            raise ValueError("Invalid or inconsistent operator types provided for a ChainComplex")

        # identify the base field and/or ring for the boundary operators of this chain complex
        fields = {resolve_field(field)} if field is not None else set()
        rings = set()
        for op in ops:
            if isinstance(op, abstract.RingArray):
                fields.add(op.field)
                rings.add(op.ring)
            elif isinstance(op, galois.FieldArray):
                fields.add(type(op))
        if len(fields) > 1 or len(rings) > 1:
            raise ValueError("Inconsistent base fields or rings provided to a chain complex")
        self._field = fields.pop() if fields else galois.GF2
        self._ring = rings.pop() if rings else None

        # identify the boundary operators of this chain complex
        self._ops = tuple(op.view(self.field) for op in ops) if self._ring is None else tuple(ops)

        if not skip_validation:
            self._validate_ops()

    @property
    def field(self) -> type[galois.FieldArray]:
        """The base field of this chain complex."""
        return self._field

    @property
    def ring(self) -> abstract.GroupRing | None:
        """The base ring of this chain complex."""
        return self._ring

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
        from qldpc import abstract

        if not 0 <= degree <= self.num_links + 1:
            raise ValueError(
                f"Degree {degree} is out of range for a chain complex with {self.num_links} links"
            )
        if degree == 0:
            matrix = self.field.Zeros((0, self.ops[0].shape[0]))
            return matrix if self.ring is None else abstract.RingArray.build(matrix, self.ring)
        if degree == len(self._ops) + 1:
            matrix = self.field.Zeros((self.ops[-1].shape[1], 0))
            return matrix if self.ring is None else abstract.RingArray.build(matrix, self.ring)
        return self.ops[degree - 1]

    @staticmethod
    def tensor_product(
        chain_a: ChainComplex | npt.NDArray[np.int_] | abstract.RingArray,
        chain_b: ChainComplex | npt.NDArray[np.int_] | abstract.RingArray,
        field: int | type[galois.FieldArray] | None = None,
    ) -> ChainComplex:
        """Tensor product of two chain complexes.

        The tensor product of chain complexes C_A and C_B, respectively with modules (A_0, A_1, ...)
        and (B_0, B_1, ...), is a new chain complex C_P with modules (P_0, P_1, ...).  The module
        P_k of degree k can be written as a direct sum of tensor products ``A_i ⨂ B_j`` for which
        ``i+j=k``, that is:

        [1] ``P_k = ⨁_{i+j=k} A_i ⨂ B_j``.

        Elements of P_2, for example, can be written as vectors
        ``[a_2 ⨂ b_0, a_1 ⨂ b_1, a_0 ⨂ b_2]``, that concatenate different ``a_i ⨂ b_j ∈ A_i ⨂ B_j``.

        The boundary operator d_k in C_P is defined by its action on each "sector" (i, j), namely

        [2] ``d_{i+j}(a_i ⨂ b_j) = d_i^A(a_i) ⨂ b_j + (-1)^i a_i ⨂ d_j^B(b_j)``,

        where ``d_i^A`` and ``d_j^B`` are boundary operators of C_A and C_B.

        In practice, to construct a boundary operator d_k we build a block matrix whose rows and
        columns correspond, respectively, to sectors of P_{k-1} and P_k.  We then populate this
        block matrix by the maps between sectors of P_k and P_{k-1} that are induced by the
        definition of d_{i+j}.
        """
        from qldpc import abstract

        if not isinstance(chain_a, ChainComplex):
            chain_a = ChainComplex([chain_a], field=field)
        if not isinstance(chain_b, ChainComplex):
            chain_b = ChainComplex([chain_b], field=field)
        if chain_a.field is not chain_b.field or chain_a.ring != chain_b.ring:
            raise ValueError("Incompatible chain complexes: different base fields or rings")
        if chain_a.ring is not None and not chain_a.ring.is_commutative:
            raise ValueError(
                "Tensor products of chain complexes over non-commutative rings are not supported"
            )

        chain_field = chain_a.field

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

        matrices = []
        for degree in range(1, chain_a.num_links + chain_b.num_links + 1):
            # fill in zero blocks of the total boundary operator
            blocks = [
                [
                    get_zero_block(row_degs, col_degs)
                    for col_degs in _get_degree_pairs(chain_a, chain_b, degree)
                ]
                for row_degs in _get_degree_pairs(chain_a, chain_b, degree - 1)
            ]

            # fill in nonzero blocks of the total boundary operator
            for col, (deg_a, deg_b) in enumerate(_get_degree_pairs(chain_a, chain_b, degree)):
                op_a = chain_a.op(deg_a)
                op_b = chain_b.op(deg_b)
                if deg_a:
                    row = get_block_index(deg_a - 1, deg_b)
                    iden_b = np.identity(op_b.shape[1], dtype=np.uint8)
                    # NumPy types np.kron as possibly returning an object array
                    blocks[row][col] = np.kron(op_a, iden_b)  # type:ignore[assignment]
                if deg_b:
                    row = get_block_index(deg_a, deg_b - 1)
                    iden_a = np.identity(op_a.shape[1], dtype=np.uint8)
                    blocks[row][col] = np.kron(iden_a, op_b) * (-1) ** deg_a

            matrices.append(np.block(blocks))

        if chain_a.ring is None:
            return ChainComplex([op.view(chain_field) for op in matrices], skip_validation=True)
        # rebuild each operator through RingArray.build to coerce every entry into a ring member
        ops = [abstract.RingArray.build(matrix, chain_a.ring) for matrix in matrices]
        return ChainComplex(ops, skip_validation=True)

    @staticmethod
    def balanced_product(
        chain_a: ChainComplex,
        chain_b: ChainComplex,
        symmetries_a: Sequence[Sequence[Iterable[int]]],
        symmetries_b: Sequence[Sequence[Iterable[int]]],
    ) -> ChainComplex:
        """Balanced product of two chain complexes.

        A symmetry of a chain complex is a list of permutations ``(p_0, p_1, ..., p_n)`` of the
        bases of its modules ``(A_0, A_1, ..., A_n)``, such that every boundary operator satisfies
        ``d_k[p_{k-1}][:, p_k] == d_k``.  Here each permutation is an integer sequence or a SymPy
        permutation, which maps basis element ``j`` to ``p_k[j]``.

        Pairing symmetries ``g_a`` and ``g_b`` from ``zip(symmetries_a, symmetries_b)``, the
        balanced product is the quotient of the tensor product of two chain complexes that
        identifies every basis element ``a ⨂ b`` with ``g_a(a) ⨂ g_b(b)``.  The basis of the
        balanced product thereby consists of orbits of the group generated by these pairs, which
        need not act freely.  Within each sector ``A_i ⨂ B_j`` of the tensor product, orbits are
        ordered by their smallest element ``(a, b)``.  Without symmetries, the balanced product is
        the tensor product.  Unlike tensor products, balanced products require chain complexes over
        a finite field, rather than a group ring.

        References:

        - https://errorcorrectionzoo.org/c/balanced_product
        - Section IV of https://arxiv.org/abs/2012.09271
        """
        if chain_a.ring is not None or chain_b.ring is not None:
            raise ValueError("Balanced products of chain complexes over rings are not supported")
        if chain_a.field is not chain_b.field:
            raise ValueError("Balanced products require chain complexes over the same field")
        if len(symmetries_a) != len(symmetries_b):
            raise ValueError("Balanced products require the same number of symmetries per complex")
        perms_a = [
            chain_a._get_symmetry_perms(symmetry, f"symmetries_a[{index}]")
            for index, symmetry in enumerate(symmetries_a)
        ]
        perms_b = [
            chain_b._get_symmetry_perms(symmetry, f"symmetries_b[{index}]")
            for index, symmetry in enumerate(symmetries_b)
        ]

        # for each degree, map each sector (deg_a, deg_b) to the quotient-basis indices of its pairs
        orbit_labels: list[dict[tuple[int, int], npt.NDArray[np.int_]]] = []
        dims = []
        for degree in range(chain_a.num_links + chain_b.num_links + 1):
            sector_labels = {}
            dim = 0
            for deg_a, deg_b in _get_degree_pairs(chain_a, chain_b, degree):
                labels = _get_diagonal_orbits(
                    (chain_a.dim(deg_a), chain_b.dim(deg_b)),
                    [(perm_a[deg_a], perm_b[deg_b]) for perm_a, perm_b in zip(perms_a, perms_b)],
                )
                sector_labels[deg_a, deg_b] = labels + dim
                dim += int(labels.max(initial=-1)) + 1
            orbit_labels.append(sector_labels)
            dims.append(dim)

        # map one representative of each orbit through the tensor-product boundary operator
        ops = []
        for degree in range(1, len(orbit_labels)):
            op = chain_a.field.Zeros((dims[degree - 1], dims[degree]))
            target_labels = orbit_labels[degree - 1]
            for (deg_a, deg_b), labels in orbit_labels[degree].items():
                cols, representatives = np.unique(labels, return_index=True)
                reps_a, reps_b = np.divmod(representatives, labels.shape[1])
                if deg_a:
                    rows = target_labels[deg_a - 1, deg_b][:, reps_b]
                    np.add.at(op, (rows, cols), chain_a.op(deg_a)[:, reps_a])
                if deg_b:
                    rows = target_labels[deg_a, deg_b - 1][reps_a, :].T
                    vals = chain_b.op(deg_b)[:, reps_b]
                    np.add.at(op, (rows, cols), -vals if deg_a % 2 else vals)
            ops.append(op)

        return ChainComplex(ops)

    def _validate_ops(self) -> None:
        """Validate the consistency of the boundary operators in this chain complex."""
        for op_a, op_b in zip(self.ops, self.ops[1:]):
            if op_a.shape[1] != op_b.shape[0] or np.any(op_a @ op_b):
                raise ValueError(
                    "Condition for a chain complex not satisfied:\n"
                    "Neighboring boundary operators of a chain complex must compose to zero"
                )

    def _get_symmetry_perms(self, symmetry: Sequence[Iterable[int]], name: str) -> list[list[int]]:
        """Validate a symmetry of this chain complex, and return its permutations as lists."""
        from qldpc import abstract

        if len(symmetry) != self.num_links + 1:
            raise ValueError(f"{name} must have {self.num_links + 1} permutations")
        perms = [
            abstract.GroupMember(list(perm), size=self.dim(degree)).array_form
            for degree, perm in enumerate(symmetry)
        ]
        for degree in range(1, self.num_links + 1):
            op = self.op(degree)
            if not np.array_equal(op[perms[degree - 1]][:, perms[degree]], op):
                raise ValueError(
                    f"{name} is not a symmetry: it changes the boundary operator on degree {degree}"
                )
        return perms


def _get_degree_pairs(
    chain_a: ChainComplex, chain_b: ChainComplex, degree: int
) -> Iterator[tuple[int, int]]:
    """Yield pairs of chain degrees that add up to a total degree."""
    min_deg_a = max(degree - chain_b.num_links, 0)
    max_deg_a = min(chain_a.num_links, degree)
    for deg_a in range(max_deg_a, min_deg_a - 1, -1):
        yield deg_a, degree - deg_a


def _get_diagonal_orbits(
    shape: tuple[int, int], perm_pairs: Sequence[tuple[list[int], list[int]]]
) -> npt.NDArray[np.int_]:
    """Label pairs of points by their orbits under permutations (a, b) --> (p_a[a], p_b[b]).

    Orbits are labeled 0, 1, ... in order of their smallest pair in row-major order.
    """
    pairs = np.arange(shape[0] * shape[1]).reshape(shape)
    sources = np.tile(pairs.ravel(), len(perm_pairs))
    targets = np.array(
        [pairs[perm_a][:, perm_b].ravel() for perm_a, perm_b in perm_pairs], dtype=int
    )
    graph = scipy.sparse.coo_array(
        (np.ones(sources.size), (sources, targets.ravel())), shape=(pairs.size,) * 2
    )
    _, components = scipy.sparse.csgraph.connected_components(graph, directed=False)
    _, smallest_pairs = np.unique(components, return_index=True)
    return np.argsort(np.argsort(smallest_pairs))[components].reshape(shape)
