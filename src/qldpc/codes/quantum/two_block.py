# SPDX-License-Identifier: Apache-2.0

"""Two-block codes, including multivariate bicycle, bivariate bicycle, and GALA codes."""

from __future__ import annotations

import collections
import functools
import itertools
import math
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt
import sympy

from qldpc import abstract
from qldpc._utils import networkx as nx
from qldpc.objects import (
    Node,
)

from ..common import ClassicalCode, CSSCode

####################################################################################################
# two-block codes


class TBCode(CSSCode):
    """Two-block code.

    A TBCode code is built out of two commuting matrices A and B, which are combined to define

    (a) ``matrix_x = [A, B]``, and
    (b) ``matrix_z = [B.T, -A.T]``.

    Commutativity of A and B ensures that ``matrix_x @ matrix_z.T = AB - BA = 0``, which makes
    matrix_x and matrix_z a valid choice of parity check matrices of a CSSCode.

    Two-block codes whose matrices A and B are elements of an abelian group algebra are multivariate
    bicycle codes (MBCodes).

    References:

    - https://errorcorrectionzoo.org/c/two_block_quantum
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | type[galois.FieldArray] | None = None,
        *,
        promise_equal_distance_xz: bool = False,
        skip_validation: bool = False,
    ) -> None:
        """Construct a two-block quantum code."""
        matrix_a = ClassicalCode(matrix_a, field).matrix
        matrix_b = ClassicalCode(matrix_b, field).matrix
        if not skip_validation and not np.array_equal(matrix_a @ matrix_b, matrix_b @ matrix_a):
            raise ValueError("The matrices provided for this TBCode do not commute")

        matrix_x = np.block([matrix_a, matrix_b])
        matrix_z = np.block([matrix_b.T, -matrix_a.T])
        super().__init__(
            matrix_x, matrix_z, field, promise_equal_distance_xz=promise_equal_distance_xz
        )
        if not skip_validation:
            self._is_subsystem_code = False  # verified above


class GALACode(CSSCode):
    """Group-Action Lift with Active orthogonality (GALA) code.

    A GALA code is a two-block CSS code built from two sequences of elements in a binary group
    algebra.  Given generator sequences (F_0, ..., F_{L/2-1}) and (G_0, ..., G_{L/2-1}), construct
    block-circulant matrices F and G with entries::

        F[i, j] = F_{j-i},
        G[i, j] = G_{j-i},

    where generator indices are taken modulo L/2.  These matrices define parent check matrices::

        parent_matrix_x = [F, G],
        parent_matrix_z = [G.T, F.T].

    The first J block rows of each parent matrix are the active rows that define the parity checks
    of the code.  The remaining rows are latent rows and do not define stabilizers.

    Each generator is a RingMember and may therefore be either a monomial, representing one group
    element, or a polynomial sum of group elements.  GALA codes are currently supported only over
    GF(2).

    The compact self-dual [[132, 30, 12]] code from arXiv:2608.07431 can be constructed by::

        from qldpc import abstract, codes

        ring = abstract.GroupRing(abstract.CyclicGroup(11))
        x = ring.generators[0]
        code = codes.GALACode(
            generators_f=[x**2, x**4, x**3, x**6, x**3, x**9],
            generators_g=[x**9, x**2, x**8, x**5, x**8, x**7],
            num_active_rows=5,
        )
        assert code.num_qubits == 132  # the block length n
        assert code.dimension == 30  # the number of logical qubits k

        # the maximum stabilizer weight, which for this code coincides numerically with the distance
        # d = 12 reported in arXiv:2608.07431; computing the distance itself is intractable at this
        # block length, so it is taken from the reference rather than checked
        assert code.get_weight() == 12

    References:

    - https://arxiv.org/abs/2608.07431

    Args:
        generators_f: The sequence (F_0, ..., F_{L/2-1}) of group-ring elements.
        generators_g: The sequence (G_0, ..., G_{L/2-1}) of group-ring elements.
        num_active_rows: The number J of active block rows, with 1 <= J <= L/2.

    Keyword args:
        skip_validation: If True, skip the check that the active X-type and Z-type parity checks
            commute.  Structural input validation is performed regardless.  Default: False.
    """

    generators_f: tuple[abstract.RingMember, ...]
    generators_g: tuple[abstract.RingMember, ...]

    matrix_f: abstract.RingArray
    matrix_g: abstract.RingArray
    parent_matrix_x: abstract.RingArray
    parent_matrix_z: abstract.RingArray

    ring: abstract.GroupRing
    group: abstract.Group

    num_blocks: int
    num_active_rows: int

    def __init__(
        self,
        generators_f: Sequence[abstract.RingMember],
        generators_g: Sequence[abstract.RingMember],
        num_active_rows: int,
        *,
        skip_validation: bool = False,
    ) -> None:
        """Construct a GALA code from two sequences of group-ring generators."""
        generators_f, generators_g, ring = self._validate_generators(
            generators_f, generators_g, num_active_rows
        )
        self.generators_f = generators_f
        self.generators_g = generators_g
        self.ring = ring
        self.group = self.ring.group
        self.num_blocks = 2 * len(self.generators_f)
        self.num_active_rows = num_active_rows

        self.matrix_f = self._get_block_circulant(self.generators_f)
        self.matrix_g = self._get_block_circulant(self.generators_g)
        self.parent_matrix_x = abstract.RingArray(np.hstack([self.matrix_f, self.matrix_g]))
        self.parent_matrix_z = abstract.RingArray(np.hstack([self.matrix_g.T, self.matrix_f.T]))

        # lift only active block rows so inactive GALA checks are not added to the code
        active_matrix_x = abstract.RingArray(self.parent_matrix_x[: self.num_active_rows])
        active_matrix_z = abstract.RingArray(self.parent_matrix_z[: self.num_active_rows])
        matrix_x = active_matrix_x.lift()
        matrix_z = active_matrix_z.lift()
        if not skip_validation and np.any(matrix_x @ matrix_z.T):
            raise ValueError("The active parity checks of this GALACode do not commute")
        super().__init__(matrix_x, matrix_z)
        if not skip_validation:
            self._is_subsystem_code = False  # verified above

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = f"{self.name} on {self.num_qubits} qubits"
        text += f" with {self.num_blocks} blocks, {self.num_active_rows} active rows,"
        text += f" and lift group {self.group}"
        return text

    @staticmethod
    def _validate_generators(
        generators_f: Sequence[abstract.RingMember],
        generators_g: Sequence[abstract.RingMember],
        num_active_rows: int,
    ) -> tuple[
        tuple[abstract.RingMember, ...],
        tuple[abstract.RingMember, ...],
        abstract.GroupRing,
    ]:
        """Validate and normalize GALA generator data."""
        generators_f = tuple(generators_f)
        generators_g = tuple(generators_g)
        if not generators_f or not generators_g:
            raise ValueError("GALA generator sequences must be nonempty")
        if len(generators_f) != len(generators_g):
            raise ValueError("GALA generator sequences must have equal lengths")

        generators = generators_f + generators_g
        if not all(isinstance(generator, abstract.RingMember) for generator in generators):
            raise ValueError("GALA generators must be RingMember objects")
        ring = generators[0].ring
        if any(generator.ring != ring for generator in generators[1:]):
            raise ValueError("All GALA generators must belong to the same group ring")
        if ring.field.order != 2:
            raise ValueError("GALA codes are currently supported only over GF(2)")

        if not isinstance(num_active_rows, (int, np.int_)):
            raise TypeError("The number of active rows must be an integer")
        if not 1 <= num_active_rows <= len(generators_f):
            raise ValueError(
                "The number of active rows must lie between 1 and"
                f" {len(generators_f)} (provided: {num_active_rows})"
            )

        generators_f = tuple(generator.copy() for generator in generators_f)
        generators_g = tuple(generator.copy() for generator in generators_g)
        return generators_f, generators_g, ring

    @staticmethod
    def _get_block_circulant(
        generators: Sequence[abstract.RingMember],
    ) -> abstract.RingArray:
        """Construct the block-circulant matrix induced by a sequence of generators."""
        size = len(generators)
        return abstract.RingArray(
            [[generators[(col - row) % size] for col in range(size)] for row in range(size)]
        )

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_] | None:
        """Translations of every block of qudits, or None (see _get_abelian_block_translations)."""
        return _get_abelian_block_translations(self.group, self.num_blocks)


class MBCode(TBCode):
    """Multivariate bicycle code, also known as an abelian two-block group-algebra code.

    An MBCode is a two block code (TBCode) built out of matrices A and B that are chosen from a ring
    over an abelian group to ensure that A and B commute.  More specifically, an MBCode is a CSS
    code with subcode parity check matrices

    - ``matrix_x = [A, B]``, and
    - ``matrix_z = [B.T, -A.T]``.

    Here A and B are polynomials of the form ``A = sum_{i,j,k,...} A_{ijk...} x^i y^j z^k ...``,
    where

    - A_{ijk...} is a scalar coefficient (over some finite field),
    - x, y, z, ... are generators of cyclic groups of orders ``R_x``, ``R_y``, ``R_z``, ...
    - the monomial ``x^i y^j z^k ...`` represents a tensor product of cyclic shift matrices.

    A multivariate bicycle code is defined by...
    [1] a sequence of cyclic group orders, and
    [2] two multivariate polynomials.
    The polynomials should be sympy expressions such as 1 + x + x * y**2 with sympy.abc variables x
    and y.  Group orders are, by default, associated with the free variables of the polynomials in
    lexicographic order.  Group orders can also be assigned to variables explicitly with a
    dictionary, as in {x: 12, y: 6}.

    References:

    - https://errorcorrectionzoo.org/c/abelian_2bga
    - https://arxiv.org/abs/2406.19151
    - https://arxiv.org/abs/2306.16400

    With polynomials in one variable, multivariate bicycle codes are generalized bicycle codes:

    - https://errorcorrectionzoo.org/c/generalized_bicycle
    - https://arxiv.org/abs/2203.17216

    With polynomials in two variables, they are bivariate bicycle codes; see the BBCode class.

    The X-distance and Z-distance of a multivariate bicycle code are equal.  Swapping the two
    blocks, inverting group elements, and negating one block maps the X-type checks onto the Z-type
    checks, and preserves the weights of operators.
    """

    poly_a: sympy.Poly
    poly_b: sympy.Poly
    orders: tuple[int, ...]

    symbols: tuple[sympy.Symbol, ...]
    symbol_gens: dict[sympy.Symbol, abstract.GroupMember]

    group: abstract.AbelianGroup
    ring: abstract.GroupRing

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | type[galois.FieldArray] | None = None,
    ) -> None:
        """Construct a multivariate bicycle code."""
        # identify the symbols used to denote cyclic group generators
        symbols = poly_a.free_symbols | poly_b.free_symbols
        if len(orders) < len(symbols):
            raise ValueError(f"Provided {len(symbols)} symbols, but only {len(orders)} orders.")

        # identify cyclic group orders with symbols in the polynomials
        if isinstance(orders, dict):
            symbol_to_order = orders.copy()
        else:
            symbol_to_order = {}
            for symbol, order in zip(sorted(symbols, key=str), orders):
                assert isinstance(symbol, sympy.Symbol), f"Invalid symbol: {symbol}"
                symbol_to_order[symbol] = order

        # add placeholder symbols for any orders that the polynomials do not account for the "~"
        # prefix and the index keep each placeholder distinct from the others and from the symbols
        # appearing in the polynomials; sorting makes the names deterministic
        placeholder_prefix = "~" + "".join(sorted(map(str, symbols)))
        while len(symbol_to_order) < len(orders):
            index = len(symbol_to_order)
            symbol_to_order[sympy.Symbol(f"{placeholder_prefix}_{index}")] = orders[index]

        self.symbols = tuple(symbol_to_order.keys())
        self.orders = tuple(symbol_to_order.values())

        # identify the group generator associated with each symbol
        self.group = abstract.AbelianGroup(*self.orders)
        self.ring = abstract.GroupRing(self.group, field)
        # an abelian group provides a generator only for a nontrivial factor, so generators go to
        # the factors that have them, and a symbol whose cyclic group is trivial takes the identity
        generators = iter(self.group.generators)
        self.symbol_gens = {}
        for symbol, order in zip(self.symbols, self.orders):
            self.symbol_gens[symbol] = next(generators) if order > 1 else self.group.identity

        # simplify the polynomials, whose monomials can denote the same group element
        self.poly_a = self.get_simplified_form(poly_a)
        self.poly_b = self.get_simplified_form(poly_b)

        # build defining matrices of a multivariate bicycle code; transpose the lift by convention
        matrix_a = self.ring.eval(self.poly_a, self.symbol_gens).lift().T
        matrix_b = self.ring.eval(self.poly_b, self.symbol_gens).lift().T
        super().__init__(
            matrix_a, matrix_b, field, promise_equal_distance_xz=True, skip_validation=True
        )
        self._is_subsystem_code = False  # matrices over an abelian group ring commute

    def get_simplified_form(
        self, poly: sympy.Basic, orders: tuple[int, ...] | None = None
    ) -> sympy.Poly:
        """Simplify the given polynomial with the relations satisfied by the cyclic generators.

        A generator of a cyclic group of order R satisfies x**R = 1, so the exponent of a symbol
        matters only modulo its order, and two monomials whose exponents agree modulo the orders
        denote the same group element.  Reducing every exponent into [0, order) therefore collects
        such monomials into a single term, whose coefficient is the sum of theirs in the base field,
        and a coefficient that sums to zero leaves no term at all.  The polynomial is returned over
        all of the symbols of this code, including any that it does not address.
        """
        orders = orders or self.orders
        assert len(orders) == len(self.symbols)

        coefficients: dict[sympy.Expr, galois.FieldArray] = {}
        for term in abstract.iter_monomial_terms(poly):
            _, _exponents = abstract.get_coefficient_and_exponents(term)
            exponents = dict(_exponents)  # convert into a dictionary, {symbol: exponent}
            monomial = sympy.prod(
                symbol ** (exponents.get(symbol, 0) % order)
                for symbol, order in zip(self.symbols, orders)
            )
            # a monomial term evaluates to a single group element, whose coefficient is in the field
            [(coefficient, _group_member)] = self.ring.eval(term, self.symbol_gens)
            coefficients[monomial] = coefficients.get(monomial, self.ring.field(0)) + coefficient

        terms = [int(coefficient) * monomial for monomial, coefficient in coefficients.items()]
        return sympy.Poly(sum(terms), *self.symbols)

    def get_canonical_form(
        self, poly: sympy.Basic, orders: tuple[int, ...] | None = None
    ) -> sympy.Expr:
        """Canonicalize the given polynomial, shifting exponents to (-order/2, order/2].

        The polynomial is simplified first, so that its monomials denote distinct group elements and
        carry the coefficients that the base field gives them.  Shifting the exponents of distinct
        simplified monomials leaves them distinct, so the terms below combine by addition alone.
        """
        orders = orders or self.orders

        # shift the exponents of one simplified monomial term at a time
        new_poly: sympy.Expr = sympy.Integer(0)
        for term in abstract.iter_monomial_terms(self.get_simplified_form(poly, orders)):
            coeff, _exponents = abstract.get_coefficient_and_exponents(term)
            exponents = dict(_exponents)  # convert into a dictionary, {symbol: exponent}

            new_term = sympy.Integer(coeff)
            for symbol, order in zip(self.symbols, orders):
                new_exponent = exponents.get(symbol, 0)
                if new_exponent > order / 2:
                    new_exponent -= order
                new_term *= symbol**new_exponent

            new_poly += new_term

        return new_poly

    def get_syndrome_subgraphs(self, *, strategy: str = "") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        The syndrome measurement circuit induced by the sequence of subgraphs constructed here
        generalizes the syndrome measurement circuit for BBCodes in arXiv:2308.07915 via the
        techniques used to construct a circuit for HGPCodes for Algorithm 2 in arXiv:2109.14609.

        Let ``L`` and ``R`` denote, respectively, the data qubits addressed by the left and right
        half of the parity check matrix for X-type stabilizers (``self.matrix_x``).  The sequence of
        subgraphs constructed here is as follows:

        1. Group together edges of the Tanner graph by ``XLA``, ``XRB``, ``ZLB``, and ``ZRA`` type,
           where ``XLA``, for example, refers to the edges associated for X-type parity checks that
           address data qubits in ``L``, whose connections are determined by the polynomial ``A``.
           The sequence of subgraphs ``(XLA, XRB, ZLB, ZRA)`` corresponds to a valid syndrome
           measurement circuit.
        2. Split ``A`` into two terms, ``A = A_1 + A_2``, and correspondingly split the graphs
           ``XLA`` and ``ZRA`` into the pairs of graphs ``(XLA_1, XLA_2)`` and ``(ZRA_1, ZRA_2)``.
           Push ``XLA_1`` to the end of the subgraph sequence for syndrome measurement, and push
           ``ZRA_1`` to the beginning, thereby arriving at the final subgraph sequence
           ``(ZRA_1, XLA_2, XRB, ZLB, ZRA_2, XLA_1)``.

        Pushing ``XLA_1`` to the end of the subgraph sequence corresponds to commuting associated
        gates to the right of the syndrome measurement circuit.  Similarly to the situation in
        Figure 2c of arXiv:2109.14609v1, commuting ``XLA_1`` to the right of ``ZLB`` introduces CNOT
        gates between X and Z check qubits; the X and Z support of these gates is given,
        respectively, by the row and column of ``A_1 @ B.T``.  These CNOTs get cancelled out by
        pushing ``ZRA_1`` to the left of ``XLB``.
        """
        if strategy:
            raise ValueError(
                f"{type(self)}.get_syndrome_subgraphs does not use an edge coloration strategy"
                f" (provided: {strategy})"
            )

        # build matrices for each term in A and B, transposed to match the lift in __init__
        terms_a = abstract.iter_monomial_terms(self.poly_a)
        terms_b = abstract.iter_monomial_terms(self.poly_b)
        matrices_a = [self.ring.eval(term, self.symbol_gens).lift().T for term in terms_a]
        matrices_b = [self.ring.eval(term, self.symbol_gens).lift().T for term in terms_b]

        # collect edges by type and index of a term in A or B
        edges_XL: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_XR: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_ZL: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_ZR: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        for term_index, matrix in enumerate(matrices_a):
            for xx, ll in zip(*np.where(matrix)):
                zz = ll + self.num_checks_x
                rr = xx + len(self) // 2
                edges_XL[term_index].append((Node(xx, is_data=False), Node(ll, is_data=True)))
                edges_ZR[term_index].append((Node(zz, is_data=False), Node(rr, is_data=True)))
        for term_index, matrix in enumerate(matrices_b):
            for xx, col in zip(*np.where(matrix)):
                ll = xx
                rr = col + len(self) // 2
                zz = col + self.num_checks_x
                edges_XR[term_index].append((Node(xx, is_data=False), Node(rr, is_data=True)))
                edges_ZL[term_index].append((Node(zz, is_data=False), Node(ll, is_data=True)))

        # convert edge sets into subgraphs and return (ZR_1, XL_2, XR, ZL, ZR_2, XL_1)
        subgraphs_XL = tuple(self.graph.edge_subgraph(edges_XL[term]) for term in edges_XL)
        subgraphs_XR = tuple(self.graph.edge_subgraph(edges_XR[term]) for term in edges_XR)
        subgraphs_ZL = tuple(self.graph.edge_subgraph(edges_ZL[term]) for term in edges_ZL)
        subgraphs_ZR = tuple(self.graph.edge_subgraph(edges_ZR[term]) for term in edges_ZR)
        return (
            subgraphs_ZR[::2]
            + subgraphs_XL[1::2]
            + subgraphs_XR
            + subgraphs_ZL
            + subgraphs_ZR[1::2]
            + subgraphs_XL[::2]
        )

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_]:
        """Translations of both blocks of qudits by the generators of the group of this code.

        The qudits of each block, and the checks of each type, are indexed by group elements.
        Translating every qudit by a group element maps each check to the check of the translated
        group element, so it maps the rows of both parity check matrices onto themselves.
        """
        return _get_block_translations(self.group, 2)


class BBCode(MBCode):
    """Bivariate bicycle code, or a multivariate bicycle code with polynomials in two variables.

    A bivariate bicycle code is a CSS code with subcode parity check matrices

    - ``matrix_x = [A, B]``, and
    - ``matrix_z = [B.T, -A.T]``.

    Here A and B are polynomials of the form ``A = sum_{i,j} A_{ij} x^i y^j``, where

    - A_{ij} is a scalar coefficient (over some finite field),
    - x and y are, respectively, generators of cyclic groups of orders ``R_x`` and ``R_y``, and
    - the monomial ``x^i y^j`` represents a tensor product of cyclic shift matrices.

    A bivariate bicycle code is defined by...
    [1] two cyclic group orders, and
    [2] two bivariate polynomials.
    The polynomials should be sympy expressions such as 1 + x + x * y**2 with sympy.abc variables x
    and y.  Group orders are, by default, associated with the free variables of the polynomials in
    lexicographic order.  Group orders can also be assigned to variables explicitly with a
    dictionary, as in {x: 12, y: 6}.

    The polynomials A and B induce a "canonical" layout of the data and check qubits of a BBCode.
    In the canonical layout, qubits are organized into plaquettes of four qubits that look like::

        L X
        Z R

    where L and R are data qubits, and X and Z are check qubits.  More specifically:

    - L and R data qubits are addressed by the left and right halves of matrix_x (or matrix_z).
    - X are check qubits measure X-type parity checks, and are associated with rows of matrix_x.
    - Z are check qubits measure Z-type parity checks, and are associated with rows of matrix_z.

    These four-qubit plaquettes are arranged into a rectangular grid that is ``R_x`` plaquettes wide
    and ``R_y`` plaquettes tall, where ``R_x`` and ``R_y`` are the orders of the cyclic groups
    generated by x and y.  Each qubit can then be labeled by coordinates (a, b) of a plaquette,
    corresponding to a row and column in the grid of plaquettes, and a "sector" (L, R, X, or Z)
    within a plaquette.

    If we associate ``(L, R) ~ (0, 1)``, then the data qubit addressed by column qq of matrix_x (or
    matrix_z) has the label ``(sector, a, b) = numpy.unravel_index(qq, [2, R_x, R_y])``.  The
    integer index of a data qubit and its label are thereby related to each other by array
    reshaping.  The label of a check qubit, whose numerical index is the index of a corresponding
    row in the full parity check matrix of a BBCode, is similarly obtained by associating
    ``(X, Z) ~ (0, 1)``.

    The connections between data and check qubits can be read directly from the polynomials A and B:

    - If ``A_{ij} != 0``, then...

      - every X qubit addresses an L qubit that is (i, j) plaquettes (right, up), and
      - every Z qubit addresses an R qubit that is (i, j) plaquettes (left, down).

    - If ``B_{ij} != 0``, then...

      - every X qubit addresses an R qubit that is (i, j) plaquettes (right, up), and
      - every Z qubit addresses an L qubit that is (i, j) plaquettes (left, down).

    Here the grid of plaquettes is assumed to have periodic boundary conditions, so going one
    plaquette "up" from the top row of the grid gets you to the bottom row of the grid.

    References:

    - https://errorcorrectionzoo.org/c/qcga
    - https://arxiv.org/abs/2308.07915
    - https://arxiv.org/abs/2408.10001
    - https://arxiv.org/abs/2404.18809
    """

    orders: tuple[int, int]
    symbols: tuple[sympy.Symbol, sympy.Symbol]

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | type[galois.FieldArray] | None = None,
    ) -> None:
        """Construct a bivariate bicycle code."""
        symbols = poly_a.free_symbols | poly_b.free_symbols
        if len(orders) != 2 or len(symbols) != 2:
            raise ValueError(
                "BBCodes should have exactly two cyclic group orders and two symbols, not "
                f"{len(orders)} orders and {len(symbols)} symbols."
            )
        super().__init__(orders, poly_a, poly_b, field)

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field is galois.GF2:
            text += f"{self.name} on {self.num_qubits} qubits"
        else:
            text += f"{self.name} on {self.num_qudits} qudits over {self.field_name}"
        orders = dict(zip(self.symbols, self.orders))
        text += f" with cyclic group orders {orders} and generating polynomials"
        text += f"\n  A = {self.poly_a.as_expr()}"
        text += f"\n  B = {self.poly_b.as_expr()}"
        return text

    def get_node_label(self, node: Node) -> tuple[str, int, int]:
        """Convert a node of this code's Tanner graph into a qubit label.

        The qubit label identifies the sector (L, R, X, Y) within a plaquette, and the coordinates
        of the plaquette that contains the given node (qubit).
        """
        return self.get_node_label_from_orders(node, self.orders)

    @staticmethod
    @functools.cache
    def get_node_label_from_orders(node: Node, orders: tuple[int, int]) -> tuple[str, int, int]:
        """Get the label of a qubit in a BBCode with cyclic groups of the given orders.

        The qubit label identifies the sector (L, R, X, Y) within a plaquette, and the coordinates
        of the plaquette that contains the given node (qubit).
        """
        ss, aa, bb = np.unravel_index(node.index, (2,) + orders)
        if node.is_data:
            sector = "L" if ss == 0 else "R"
        else:
            sector = "X" if ss == 0 else "Z"
        return sector, int(aa), int(bb)

    def get_qubit_pos(
        self, qubit: Node | tuple[str, int, int], folded_layout: bool = False
    ) -> tuple[int, int]:
        """Get the canonical position of a qubit in this code.

        If ``folded_layout is True``, "fold" the array of qubits as in Figure 2 of arXiv:2404.18809.
        """
        return self.get_qubit_pos_from_orders(qubit, folded_layout, self.orders)

    @staticmethod
    @functools.cache
    def get_qubit_pos_from_orders(
        qubit: Node | tuple[str, int, int],
        folded_layout: bool,
        orders: tuple[int, int],
    ) -> tuple[int, int]:
        """Get the canonical position of a qubit in a BBCode with cyclic groups of the given orders.

        If ``folded_layout is True``, "fold" the array of qubits as in Figure 2 of arXiv:2404.18809.
        """
        if isinstance(qubit, Node):
            qubit = BBCode.get_node_label_from_orders(qubit, orders)
        ss, aa, bb = qubit

        # convert sector and plaquette coordinates into qubit coordinates
        xx = 2 * aa + int(ss == "R" or ss == "X")
        yy = 2 * bb + int(ss == "L" or ss == "X")
        if folded_layout:
            order_a, order_b = orders
            xx = 2 * xx if xx < order_a else (2 * order_a - 1 - xx) * 2 + 1
            yy = 2 * yy if yy < order_b else (2 * order_b - 1 - yy) * 2 + 1
        return xx, yy

    def get_equivalent_toric_layout_code_data(
        self,
    ) -> Sequence[tuple[tuple[int, int], sympy.Poly, sympy.Poly]]:
        """Get the generating data for equivalent BBCodes with "manifestly toric" layouts.

        For simplicity, we consider BBCodes for qubits (with base field ``F_2``) in the text below.

        A BBCode has a manifestly toric layout if it is generated by polynomials that look like

            ``poly_a = 1 + x + ...``, and
            ``poly_b = 1 + y + ...``,

        We say that two BBCodes are "equivalent" if they can be obtained from one another by a
        permutation of data and check qubits.  For qudit codes, we also allow "equivalent" BBCodes
        to differ by an overall sign of their Z-type parity checks, which does not change the code
        that these checks define.

        To find an equivalent BBCode with a manifestly toric layout, we take

            ``poly_a = sum_j A_j --> poly_a / A_k = 1 + sum_{j != k} A_j/A_k``, and
            ``poly_b = sum_j B_j --> poly_b / B_l = 1 + sum_{j != l} B_j/B_l``.

        Each pair of terms ``(A_j/A_k, B_j/B_l)`` is then a candidate for cyclic group generators
        (g, h) for an equivalent BBCode.

        This modification of polynomials and change-of-basis from the original generators (x, y) to
        (g, h) produces an equivalent BBCode so long as g and h satisfy the conditions in Lemma 4 of
        arXiv:2308.07915, which boils down to the requirement that

            ``order(g) * order(h) = order(<g, h>) = order(<x, y>)``,

        where (for example) ``<x, y>`` is the abelian group generated by x and y.
        """
        if not nx.is_weakly_connected(self.graph):
            # a connected tanner graph is required for a toric layout to exist
            return []

        # identify individual monomials (terms without their coefficients) in the polynomials
        monomials_a = [term.as_coeff_Mul()[1] for term in abstract.iter_monomial_terms(self.poly_a)]
        monomials_b = [term.as_coeff_Mul()[1] for term in abstract.iter_monomial_terms(self.poly_b)]

        # identify collections of monomials that can be combined to obtain a toric layout
        toric_params = []
        for (a_1, a_2), (b_1, b_2) in itertools.product(
            itertools.combinations(monomials_a, 2),
            itertools.combinations(monomials_b, 2),
        ):
            vec_g = self.as_exponent_vector(a_2 / a_1)
            vec_h = self.as_exponent_vector(b_2 / b_1)
            if self.is_valid_basis(vec_g, vec_h):
                toric_params.append((a_1, a_2, b_1, b_2))
                toric_params.append((a_1, a_2, b_2, b_1))
                toric_params.append((a_2, a_1, b_1, b_2))
                toric_params.append((a_2, a_1, b_2, b_1))

        toric_layout_generating_data = []
        for a_1, a_2, b_1, b_2 in toric_params:
            # new generators and their cyclic group orders
            gen_g = a_2 / a_1
            gen_h = b_2 / b_1
            vec_g = self.as_exponent_vector(gen_g)
            vec_h = self.as_exponent_vector(gen_h)
            orders = (self.get_order(vec_g), self.get_order(vec_h))

            # new "shifted" polynomials
            shifted_poly_a = (self.poly_a / a_1).expand()
            shifted_poly_b = (self.poly_b / b_1).expand()

            # without loss of generality, enforce that the toric layout "width" >= "height"
            if orders[0] < orders[1]:
                orders = orders[::-1]
                gen_g, gen_h = gen_h, gen_g
                shifted_poly_a, shifted_poly_b = shifted_poly_b, shifted_poly_a

            # change polynomial basis to gen_g and gen_h
            new_poly_a, new_poly_b = self.change_poly_basis(
                gen_g, gen_h, shifted_poly_a, shifted_poly_b
            )

            # add new generating data
            generating_data = (orders, new_poly_a, new_poly_b)
            if generating_data not in toric_layout_generating_data:
                toric_layout_generating_data.append(generating_data)

        return toric_layout_generating_data

    def as_exponent_vector(self, monomial: sympy.Mul) -> tuple[int, int]:
        """Express the given monomial as a vector of exponents, as in ``x**3/y**2 -> (3, -2)``."""
        _, _exponents = abstract.get_coefficient_and_exponents(monomial)
        exponents = dict(_exponents)  # convert into a dictionary, {symbol: exponent}
        return (exponents.get(self.symbols[0], 0), exponents.get(self.symbols[1], 0))

    def change_poly_basis(
        self, new_x: sympy.Mul, new_y: sympy.Mul, *polys: sympy.Basic
    ) -> list[sympy.Basic]:
        """Change polynomial bases from (old_x, old_y) = self.symbols to (new_x, new_y)."""
        # identify vectors of exponents, as in new_x = old_x**pp * old_y**qq -> (pp, qq)
        vec_new_x = self.as_exponent_vector(new_x)
        vec_new_y = self.as_exponent_vector(new_y)

        # identify the orders of new_x and new_y
        orders = self.get_order(vec_new_x), self.get_order(vec_new_y)

        # invert the system of equations for each of old_x and old_y
        new_basis = vec_new_x, vec_new_y
        xx, xy = self._get_basis_coefficients(new_basis, 1, 0)
        yx, yy = self._get_basis_coefficients(new_basis, 0, 1)

        # express generators old_x, old_y in terms of new_x and new_y
        symbol_new_x = sympy.Symbol("".join(map(str, self.symbols)))
        symbol_new_y = sympy.Symbol("".join(map(str, self.symbols * 2)))
        old_x = symbol_new_x**xx * symbol_new_y**xy
        old_y = symbol_new_x**yx * symbol_new_y**yy

        # build polynomials for an equivalent BBCode with a manifestly toric layout
        new_polys = []
        for poly in polys:
            # expand (x, y) in terms of (g, h), then "rename" (g, h) to (x, y)
            poly = poly.subs({self.symbols[0]: old_x, self.symbols[1]: old_y})
            poly = poly.subs({symbol_new_x: self.symbols[0], symbol_new_y: self.symbols[1]})

            # add canonical form of this polynomial, with exponents in (-order/2, order/2]
            new_polys.append(self.get_canonical_form(poly, orders))

        return new_polys

    def get_order(self, vec: tuple[int, int]) -> int:
        """What multiple of the vector hits the "origin" on the torus of plaquettes for this code?

        The plaquettes for this code tile a torus with shape self.orders.
        """
        period_0 = self.orders[0] // math.gcd(vec[0], self.orders[0])
        period_1 = self.orders[1] // math.gcd(vec[1], self.orders[1])
        return period_0 * period_1 // math.gcd(period_0, period_1)

    def is_valid_basis(self, vec_a: tuple[int, int], vec_b: tuple[int, int]) -> bool:
        """Are the given vectors a valid basis for the plaquettes of this code?

        The plaquettes for this code tile a torus with shape self.orders.
        """
        order_a = self.get_order(vec_a)
        order_b = self.get_order(vec_b)
        if order_a * order_b != len(self) // 2:
            return False

        # brute-force determine whether every plaquette can be reached by the basis vectors
        reached = np.zeros(self.orders, dtype=bool)
        for aa in range(order_a):
            for bb in range(order_b):
                xx = (aa * vec_a[0] + bb * vec_b[0]) % self.orders[0]
                yy = (aa * vec_a[1] + bb * vec_b[1]) % self.orders[1]
                reached[xx, yy] = True
        return bool(np.all(reached))

    def _get_basis_coefficients(
        self, basis: tuple[tuple[int, int], tuple[int, int]], aa: int, bb: int
    ) -> tuple[int, int]:
        """Brute-force solve for the basis coefficients ``xx, yy`` of a target vector.

        Solves ``xx * basis[0] + yy * basis[1] == (aa, bb) % self.orders`` for ``xx, yy``.
        """
        aa = aa % self.orders[0]
        bb = bb % self.orders[1]
        order_0 = self.get_order(basis[0])
        order_1 = self.get_order(basis[1])
        for xx in range(order_0):
            for yy in range(order_1):
                if (
                    aa == (xx * basis[0][0] + yy * basis[1][0]) % self.orders[0]
                    and bb == (xx * basis[0][1] + yy * basis[1][1]) % self.orders[1]
                ):
                    return xx, yy
        raise ValueError(f"Uninvertible system of equations: {basis}, {aa}, {bb}")


def _get_block_translations(group: abstract.Group, num_blocks: int) -> npt.NDArray[np.int_]:
    """Qudit permutations that translate several blocks of qudits at once by group generators.

    The qudits form ``num_blocks`` consecutive blocks.  The qudits of each block are indexed like
    the rows of the lifts of ``group``, which must be permutation matrices.  Row ``i`` of the result
    moves the qudits of every block as the lift of generator ``i`` moves its rows.

    These permutations preserve a code whose checks they permute, as when the parity check matrices
    are arrays of lifts of elements of an abelian group algebra, as in a quasi-cyclic code.  Such a
    code can return them from ``_get_qudit_automorphisms``.
    """
    block_size = len(group.lift(group.identity))
    offsets = block_size * np.arange(num_blocks)[:, None]
    return np.array(
        [
            (np.argmax(group.lift(generator), axis=0) + offsets).ravel()
            for generator in group.generators
        ],
        dtype=int,
    ).reshape(-1, num_blocks * block_size)


def _get_abelian_block_translations(
    group: abstract.Group, num_blocks: int
) -> npt.NDArray[np.int_] | None:
    """Block translations that preserve a code built from lifts of an abelian group algebra.

    The parity check matrices of such a code are arrays of blocks, each of which is a linear
    combination of the lifts of group elements.  The qudits form ``num_blocks`` blocks, and the
    checks of each type form blocks, which are indexed like the rows of the lifts.  Translating
    every block of qudits by a group element multiplies a parity check matrix ``H`` on the right by
    a block-diagonal matrix ``D`` of copies of ``T``, the lift of that element or of its inverse.
    If the group is abelian, ``T`` commutes with every block of ``H``, so ``H D = D' H``, where
    ``D'`` is a block-diagonal matrix with one copy of ``T`` for each block of checks.  If ``T`` is
    a permutation matrix, ``D'`` permutes the rows of ``H``, so the translation maps the rows of
    ``H`` onto themselves.

    Returns:
        The translations by the generators of the group (see ``_get_block_translations``), or None
        if the group is not abelian or the lift of a generator is not a permutation matrix.
    """
    if not group.is_abelian:
        return None
    for generator in group.generators:
        lift = group.lift(generator)
        is_permutation = (
            np.isin(lift, (0, 1)).all()
            and (lift.sum(axis=0) == 1).all()
            and (lift.sum(axis=1) == 1).all()
        )
        if not is_permutation:
            return None
    return _get_block_translations(group, num_blocks)
