"""Quantum error-correcting codes

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

import ast
import collections
import functools
import itertools
import math
import operator
import os
from collections.abc import Collection, Sequence

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy
import sympy

from qldpc import abstract
from qldpc.abstract import DEFAULT_FIELD_ORDER
from qldpc.math import first_nonzero_cols
from qldpc.objects import CayleyComplex, ChainComplex, Node, Pauli, PauliXZ, QuditPauli

from .classical import HammingCode, RepetitionCode, RingCode, SimplexCode, TannerCode
from .common import ClassicalCode, CSSCode, QuditCode


class FiveQuditCode(QuditCode):
    """Smallest quantum error-correcting code.

    Generalizes the better-known FiveQubitCode.

    References:
    - https://errorcorrectionzoo.org/c/galois_5_1_3
    """

    def __init__(self, field: int | None = None) -> None:
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        matrix = [
            [1, 0, 0, -1, 0, 0, 1, -1, 0, 0],
            [0, 1, 0, 0, -1, 0, 0, 1, -1, 0],
            [-1, 0, 1, 0, 0, 0, 0, 0, 1, -1],
            [0, -1, 0, 1, 0, -1, 0, 0, 0, 1],
        ]
        super().__init__(
            code_field(1) * np.array(matrix, dtype=int),
            is_subsystem_code=False,
        )
        self._dimension = 1
        self._distance = 3


class FiveQubitCode(FiveQuditCode):
    """Smallest quantum error-correcting code.

    References:
    - https://errorcorrectionzoo.org/c/stab_5_1_3
    """

    def __init__(self) -> None:
        super().__init__(field=2)


class QuantumHammingCode(CSSCode):
    """Quantum Hamming code, whose parity check matrices are classical Hamming codes.

    References:
    - https://errorcorrectionzoo.org/c/quantum_hamming_css
    """

    def __init__(self, rank: int, field: int | None = None) -> None:
        code = HammingCode(rank, field)
        super().__init__(code, code, is_subsystem_code=False)
        self._distance_x = self._distance_z = 3


class SteaneCode(QuantumHammingCode):
    """Smallest quantum error-correcting CSS code.

    Also the smallest error-correcting color code.

    References:
    - https://errorcorrectionzoo.org/c/steane
    """

    def __init__(self) -> None:
        super().__init__(rank=3)
        self.set_logical_ops_xz([[1] * 7], [[1] * 7], validate=False)


class IcebergCode(CSSCode):
    """A quantum error detecting code: [2m, 2m-2, 2].

    The m = 3 IcebergCode is the [6, 4, 2] code that is used to construct concatenated
    many-hypercube codes.

    References:
    - https://errorcorrectionzoo.org/c/iceberg
    - https://errorcorrectionzoo.org/c/stab_6_4_2
    - https://arxiv.org/abs/2403.16054
    """

    def __init__(self, size: int, *, alternative_logicals: bool = False) -> None:
        checks = [[1] * (2 * size)]
        super().__init__(checks, checks, is_subsystem_code=False)
        self._dimension = 2 * size - 2
        self._distance_x = self._distance_z = 2

        if alternative_logicals and size == 3:
            # make a specific choice of logical operators for the [6, 4, 2] code, splitting the
            # four logical qubits into pairs with disjoint support on the physical qubits
            sector_ops_x = [[1, 1, 0], [0, 1, 1]]
            sector_ops_z = sector_ops_x[::-1]
            ops_x = scipy.linalg.block_diag(sector_ops_x, sector_ops_x)
            ops_z = scipy.linalg.block_diag(sector_ops_z, sector_ops_z)
            self.set_logical_ops_xz(ops_x, ops_z)


class C4Code(IcebergCode):
    """A [4, 2, 2] code, commonly known as the "C4" code.

    References:
    - https://errorcorrectionzoo.org/c/stab_4_2_2
    """

    def __init__(self) -> None:
        super().__init__(2)


class C6Code(CSSCode):
    """A [6, 2, 2] code, commonly known as the "C6" code.

    References:
    - https://errorcorrectionzoo.org/c/stab_6_2_2
    """

    def __init__(self) -> None:
        checks = [[1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]]
        super().__init__(checks, checks, is_subsystem_code=False)
        logical_ops_xz = scipy.linalg.block_diag([1, 1, 1], [1, 1, 1])
        self.set_logical_ops_xz(logical_ops_xz, logical_ops_xz)
        self._dimension = 2
        self._distance_x = self._distance_z = 2


####################################################################################################
# two-block and quasi-cyclic codes


class TBCode(CSSCode):
    """Two-block code.

    A TBCode code is built out of two commuting matrices A and B, which are combined to define
    (a) matrix_x = [A, B], and
    (b) matrix_z = [B.T, -A.T].
    Commutativity of A and B ensures that matrix_x @ matrix_z.T = AB - BA = 0, which makes matrix_x
    and matrix_z a valid choice of parity check matrices of a CSSCode.

    Two-block codes constructed out of circulant matrices (i.e., matrices chosen from a ring over an
    Abelian group) are known as quasi-cyclic codes (QCCodes).

    References:
    - https://errorcorrectionzoo.org/c/two_block_quantum
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        promise_equal_distance_xz: bool = False,
        validate: bool = True,
    ) -> None:
        """Construct a two-block quantum code."""
        matrix_a = ClassicalCode(matrix_a, field).matrix
        matrix_b = ClassicalCode(matrix_b, field).matrix
        if validate and not np.array_equal(matrix_a @ matrix_b, matrix_b @ matrix_a):
            raise ValueError("The matrices provided for this TBCode do not commute")

        matrix_x = np.block([matrix_a, matrix_b])
        matrix_z = np.block([matrix_b.T, -matrix_a.T])
        super().__init__(
            matrix_x,
            matrix_z,
            field,
            promise_equal_distance_xz=promise_equal_distance_xz,
            is_subsystem_code=False,
        )


class QCCode(TBCode):
    """Quasi-cyclic code.

    A QCCode is a two block code (TBCode) built out of matrices A and B that are chosen from a ring
    over an Abelian group to ensure that A and B commute.  More specifically, a QCCode is a CSS code
    with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j,k,...} A_{ijk...} x^i y^j z^k ..., where
    - A_{ijk...} is a scalar coefficient (over some finite field),
    - x, y, z, ... are generators of cyclic groups of orders R_x, R_y, R_z, ...
    - the monomial x^i y^j z^k ... represents a tensor product of cyclic shift matrices.

    A quasi-cyclic code code is defined by...
    [1] a sequence of cyclic group orders, and
    [2] two multivariate polynomials.
    The polynomials should be sympy expressions such as 1 + x + x * y**2 with sympy.abc variables x
    and y.  Group orders are, by default, associated with the free variables of the polynomials in
    lexicographic order.  Group orders can also be assigned to variables explicitly with a
    dictionary, as in {x: 12, y: 6}.

    References:
    - https://errorcorrectionzoo.org/c/quantum_quasi_cyclic

    Univariate quasi-cyclic codes are generalized bicycle codes:
    - https://errorcorrectionzoo.org/c/generalized_bicycle
    - https://arxiv.org/abs/2203.17216

    Bivariate quasi-cyclic codes are bivariate bicycle codes; see the BBCode class.
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
        field: int | None = None,
    ) -> None:
        """Construct a generalized bicycle code."""
        self.poly_a = sympy.Poly(poly_a)
        self.poly_b = sympy.Poly(poly_b)

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

        # add more placeholder symbols if necessary
        while len(symbol_to_order) < len(orders):
            unique_symbol = sympy.Symbol("~" + "".join(map(str, symbols)))
            symbol_to_order[unique_symbol] = orders[len(symbol_to_order)]

        self.symbols = tuple(symbol_to_order.keys())
        self.orders = tuple(symbol_to_order.values())

        # identify the group generator associated with each symbol
        self.group = abstract.AbelianGroup(*self.orders)
        self.ring = abstract.GroupRing(self.group, field)
        self.symbol_gens = dict(zip(self.symbols, self.group.generators))

        # build defining matrices of a quasi-cyclic code; transpose the lift by convention
        matrix_a = self.eval(self.poly_a).lift().T
        matrix_b = self.eval(self.poly_b).lift().T
        super().__init__(matrix_a, matrix_b, field, promise_equal_distance_xz=True, validate=False)

    def eval(self, expression: sympy.Basic) -> abstract.RingMember:
        """Convert a sympy expression into an element of this code's group algebra."""
        if isinstance(expression, sympy.Poly):
            terms = sympy.Add.make_args(expression.as_expr())
            return functools.reduce(operator.add, [self.eval(term) for term in terms])

        coeff, monomial = expression.as_coeff_Mul()
        member = self.to_group_member(monomial)
        if not 0 <= int(coeff) < self.ring.field.order:
            raise ValueError(
                f"Coefficient {coeff} in expression {expression} is invalid over the finite"
                f" field GF({self.ring.field.order})"
            )
        return abstract.RingMember(self.ring, (int(coeff), member))

    def to_group_member(
        self, monomial: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul
    ) -> abstract.GroupMember:
        """Convert a monomial into an associated member of this code's base group."""
        _, exponents = self.get_coefficient_and_exponents(monomial)

        output = self.group.identity
        for base, exponent in exponents.items():
            output *= self.symbol_gens[base] ** exponent
        return output

    @staticmethod
    def get_coefficient_and_exponents(
        monomial: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul,
    ) -> tuple[int, dict[sympy.Symbol, int]]:
        """Extract the coefficients and exponents in a monomial expression.

        For example, this method takes 5 x**3 y**2 to (5, {x: 3, y: 2})."""
        coeff, monomial = monomial.as_coeff_Mul()
        exponents = {}
        if isinstance(monomial, sympy.Integer):
            coeff *= int(monomial)
        elif isinstance(monomial, sympy.Symbol):
            exponents[monomial] = 1
        elif isinstance(monomial, sympy.Pow):
            base, exponent = monomial.as_base_exp()
            exponents[base] = exponent
        elif isinstance(monomial, sympy.Mul):
            for factor in monomial.args:
                base, exponent = factor.as_base_exp()
                exponents[base] = exponent
        return coeff, exponents

    def get_canonical_form(
        self, poly: sympy.Poly, orders: tuple[int, ...] | None = None
    ) -> tuple[sympy.Poly, sympy.Poly]:
        """Canonicalize the given polynomial, shifting exponents to (-order/2, order/2]."""
        orders = orders or self.orders
        assert len(orders) == len(self.symbols)

        # canonialize and add one term ata time
        new_poly = sympy.core.numbers.Zero()
        for term in poly.args:
            coeff, exponents = self.get_coefficient_and_exponents(term)

            new_term = sympy.core.numbers.One()
            for symbol, order in zip(self.symbols, orders):
                new_exponent = exponents.get(symbol, 0) % order
                if new_exponent > order / 2:
                    new_exponent -= order
                new_term *= coeff * symbol**new_exponent

            new_poly += new_term

        return new_poly

    def get_syndrome_subgraphs(self, *, strategy: str = "") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        The syndrome measurement circuit induced by the sequence of subgraphs constructed here
        generalizes the syndrome measurement circuit for BBCodes in arXiv:2308.07915 via the
        techniques used to construct a circuit for HGPCodes for Algorithm 2 in arXiv:2109.14609.

        Let L and R denote, respectively, the data qubits addressed by the left and right half of
        the parity check matrix for X-type stabilizers (self.matrix_x).  The sequence of subgraphs
        constructed here is as follows:
        1. Group together edges of the Tanner graph by XLA, XRB, ZLB, and ZRA type, where XLA, for
            example, refers to the edges associated for X-type parity checks that address data
            qubits in L, whose connections are determined by the polynomial A.  The sequence of
            subgraphs (XLA, XRB, ZLB, ZRA) corresponds to a valid syndrome measurement circuit.
        2. Split A in into two terms, A = A_1 + A_2, and correspondingly split the graphs XLA and
            ZRA into the pairs of graphs (XLA_1, XLA_2) and (ZRA_1, ZRA_2).  Push XLA_1 to the end
            of the subgraph sequence for syndrome measurement, and push ZRA_1 to the beginning,
            thereby arriving at the final subgraph sequence (ZRA_1, XLA_2, XRB, ZLB, ZRA_2, XLA_1).
        Pushing XLA_1 to the end of the subgraph sequence corresponds to commuting associated gates
        to the right of the syndrome measurement circuit.  Similarly to the situation in Figure 2c
        of arXiv:2109.14609v1, commuting XLA_1 to the right of ZLB introduces CNOT gates between X
        and Z check qubits; the X and Z support of these gates is given, respectively, by the row
        and column of A_1 @ B.T.  These CNOTs get cancelled out by pushing ZRA_1 to the left of XLB.
        """
        assert not strategy, (
            f"{type(self)}.get_syndrome_subgraphs does not use an edge coloration strategy"
            f" (provided: {strategy})"
        )

        # build matrices for each term in A and B
        terms_a = sympy.Add.make_args(self.poly_a.as_expr())
        terms_b = sympy.Add.make_args(self.poly_b.as_expr())
        matrices_a = [self.eval(term).lift().T for term in terms_a]
        matrices_b = [self.eval(term).lift().T for term in terms_b]

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


class BBCode(QCCode):
    """Bivariate bicycle code, or a quasi-cyclic code with polynomials in two variables.

    A bivariate bicycle code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j} A_{ij} x^i y^j, where
    - A_{ij} is a scalar coefficient (over some finite field),
    - x and y are, respectively, generators of cyclic groups of orders R_x and R_y, and
    - the monomial x^i y^j represents a tensor product of cyclic shift matrices.

    A bivariate bicycle code is defined by...
    [1] two cyclic group orders, and
    [2] two bivariate polynomials.
    The polynomials should be sympy expressions such as 1 + x + x * y**2 with sympy.abc variables x
    and y.  Group orders are, by default, associated with the free variables of the polynomials in
    lexicographic order.  Group orders can also be assigned to variables explicitly with a
    dictionary, as in {x: 12, y: 6}.

    The polynomials A and B induce a "canonical" layout of the data and check qubits of a BBCode.
    In the canonical layout, qubits are organized into plaquettes of four qubits that look like
        L X
        Z R
    where L and R are data qubits, and X and Z are check qubits.  More specifically:
    - L and R data qubits are addressed by the left and right halves of matrix_x (or matrix_z).
    - X are check qubits measure X-type parity checks, and are associated with rows of matrix_x.
    - Z are check qubits measure Z-type parity checks, and are associated with rows of matrix_z.
    These four-qubit plaquettes are arranged into a rectangular grid that is R_x plaquettes wide and
    R_y plaquettes tall, where R_x and R_y are the orders of the cyclic groups generated by x and y.
    Each qubit can then be labeled by coordinates (a, b) of a plaquette, corresponding to a row
    and column in the grid of plaquettes, and a "sector" (L, R, X, or Z) within a plaquette.

    If we associate (L, R) ~ (0, 1), then the data qubit addressed by column qq of matrix_x (or
    matrix_z) has the label (sector, a, b) = numpy.unravel_index(qq, [2, R_x, R_y]).  The integer
    index of a data qubit its label are thereby related to each other by array reshaping.  The label
    of a check qubit, whose numerical index is the index of a corresponding row in the full parity
    check matrix of a BBCode, is similarly obtained by associating (X, Z) ~ (0, 1).

    The connections between data and check qubits can be read directly from the polynomials A and B:
    - If A_{ij} != 0, then...
      - every X qubit addresses an L qubit that is (i, j) plaquettes (right, up), and
      - every Z qubit addresses an R qubit that is (i, j) plaquettes (left, down).
    - If B_{ij} != 0, then...
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
        field: int | None = None,
    ) -> None:
        """Construct a bivariate bicycle code."""
        symbols = sympy.Poly(poly_a).free_symbols | sympy.Poly(poly_b).free_symbols
        if len(orders) != 2 or len(symbols) != 2:
            raise ValueError(
                "BBCodes should have exactly two cyclic group orders and two symbols, not "
                f"{len(orders)} orders and {len(symbols)} symbols."
            )
        super().__init__(orders, poly_a, poly_b, field)

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
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

        If folded_layout is True, "fold" the array of qubits as in Figure 2 of arXiv:2404.18809.
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

        If folded_layout is True, "fold" the array of qubits as in Figure 2 of arXiv:2404.18809.
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

        For simplicity, we consider BBCodes for qubits (with base field F_2) in the text below.

        A BBCode has a manifestly toric layout if it is generated by polynomials that look like
            poly_a = 1 + x + ..., and
            poly_b = 1 + y + ...,
        We say that two BBCodes are "equivalent" if they can be obtained from one another by a
        permutation of data and check qubits.

        To an find equivalent BBCode with a manifestly toric layout, we take
            poly_a = sum_j A_j --> poly_a / A_k = 1 + sum_{j != k} A_j/A_k, and
            poly_b = sum_j B_j --> poly_b / B_l = 1 + sum_{j != l} B_j/B_l.
        Each pair of terms (A_j/A_k, B_j/B_l) is then a candidate for cyclic group generators (g, h)
        for an equivalent BBCode.

        This modification of polynomials and change-of-basis from the original generators (x, y)
        to (g, h) produces an equivalent BBCode so long as g and h satisfy the conditions in Lemma 4
        of arXiv:2308.07915, which boils down to the requirement that
            order(g) * order(h) = order(<g, h>) = order(<x, y>),
        where (for example) <x, y> is the Abelian group generated by x and y.
        """
        if not nx.is_weakly_connected(self.graph):
            # a connected tanner graph is required for a toric layout to exist
            return []

        # identify individual monomials (terms without their coefficients) in the polynomials
        monomials_a = [term.as_coeff_Mul()[1] for term in self.poly_a.as_expr().args]
        monomials_b = [term.as_coeff_Mul()[1] for term in self.poly_b.as_expr().args]

        # identify collections of monomials that can be combined to obtain a toric layout
        toric_params = []
        for (a_1, a_2), (b_1, b_2) in itertools.product(
            itertools.combinations(monomials_a, 2), itertools.combinations(monomials_b, 2)
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
            # new generators and their their cyclic group orders
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
        """Express the given monomial as a vector of exponents, as in x**3/y**2 -> (3, -2)."""
        _, exponents = self.get_coefficient_and_exponents(monomial)
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
        xx, xy = self.modular_inverse(new_basis, 1, 0)
        yx, yy = self.modular_inverse(new_basis, 0, 1)

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

    def modular_inverse(
        self, basis: tuple[tuple[int, int], tuple[int, int]], aa: int, bb: int
    ) -> tuple[int, int]:
        """Brute force: solve xx * basis[0] + yy * basis[1] == (aa, bb) % self.orders for xx, yy.

        If provided orders, treat them as the orders of the basis vectors.
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
        if not order_a * order_b == len(self) // 2:
            return False

        # brute-force determine whether every plaquette can be reached by the basis vectors
        reached = np.zeros(self.orders, dtype=bool)
        for aa in range(order_a):
            for bb in range(order_b):
                xx = (aa * vec_a[0] + bb * vec_b[0]) % self.orders[0]
                yy = (aa * vec_a[1] + bb * vec_b[1]) % self.orders[1]
                reached[xx, yy] = True
        return bool(np.all(reached))


####################################################################################################
# hypergraph product code, lifted product code, and their subsystem variants


class HGPCode(CSSCode):
    """Hypergraph product code.

    A hypergraph product code AB is constructed from the parity check matrices of two classical
    codes, A and B.

    Consider the following:
    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.
    We represent data bits/qudits by circles (○) and check bits/qudits by squares (□).

    Denode the Tanner graph of code C by G_C.  The nodes of G_AB can be arranged into a matrix.  The
    rows of this matrix are labeled by nodes of G_A, and columns by nodes of G_B.  The matrix of
    nodes in G_AB can thus be organized into four sectors:

    ――――――――――――――――――――――――――――――――――
      | ○ ○ ○ ○ | □ □ □ ← nodes of G_B
    ――+―――――――――+――――――
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ――+―――――――――+――――――
    □ | □ □ □ □ | ○ ○ ○
    □ | □ □ □ □ | ○ ○ ○
    ↑ nodes of G_A
    ――――――――――――――――――――――――――――――――――

    We identify each sector by two bits.
    In the example above:
    - sector (0, 0) has 3×4=12 data qudits
    - sector (0, 1) has 3×3=9 check qudits
    - sector (1, 0) has 2×4=8 check qudits
    - sector (1, 1) has 2×3=6 data qudits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qudits in sectors...
    - (1, 0) of G_AB measure X-type operators, and
    - (0, 1) of G_AB measure Z-type operators.

    This class contains two equivalent constructions of an HGPCode:
    - A construction based on Tanner graphs (as discussed above).
    - A construction based on check matrices, as originally introduced in arXiv:0903.0566.
    The latter construction is less intuitive, but more efficient.

    References:
    - https://errorcorrectionzoo.org/c/hypergraph_product
    - https://arxiv.org/abs/0903.0566
    - https://arxiv.org/abs/1202.0928
    - https://arxiv.org/abs/2202.01702
    - https://www.youtube.com/watch?v=iehMcUr2saM
    """

    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        """Hypergraph product of two classical codes, as in arXiv:0903.0566.

        The parity check matrices of the hypergraph product code are:

        matrix_x = [ H1 ⨂ In2, Im1 ⨂ H2.T]
        matrix_z = [-In1 ⨂ H2, H1.T ⨂ Im2]

        Here (H1, H2) == (matrix_a, matrix_b), and I[m/n][1/2] are identity matrices,
        with (m1, n1) = H1.shape and (m2, n2) = H2.shape.

        A minus sign in one sector of matrix_x or matrix_z is necessary to satisfy CSS code
        requirements with nonbinary fields.  The placement of this sign is chosen for consistency
        with the tensor product of chain complexes.
        """
        if code_b is None:
            code_b = code_a
        self.code_a = ClassicalCode(code_a, field)
        self.code_b = ClassicalCode(code_b, field)
        field = self.code_a.field.order

        # use a matrix-based hypergraph product to identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(self.code_a.matrix, self.code_b.matrix)

        # identify the number of qudits in each sector
        self.sector_size = np.outer(
            [self.code_a.num_bits, self.code_a.num_checks],
            [self.code_b.num_bits, self.code_b.num_checks],
        )

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self._default_conjugate = slice(self.sector_size[0, 0], None)

        super().__init__(
            matrix_x.view(np.ndarray).astype(int),
            matrix_z.view(np.ndarray).astype(int),
            field,
            is_subsystem_code=False,
        )

        if set_logicals:
            logical_ops_xz = HGPCode.get_canonical_logical_ops(self.code_a, self.code_b)
            self.set_logical_ops_xz(*logical_ops_xz, validate=False)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        The sequence here is essentially the sequence used for hypergraph product codes in Algorithm
        2 of arXiv:2109.14609, modified to obviate the need to find a balanced ordering of Tanner
        graph vertices.

        More specifically, this method constructs Tanner subgraphs as follows:
        1. For the classical seed code that defines vertical edges of this HGPCode (self.code_a),
            color the edges of its Tanner graph, and number these colors starting from zero.
        2. Even edges get assigned a "north" or "south" direction if they are associated,
            respectively, with X-type or Z-type parity checks.  Odd edges get assigned the opposite
            direction.
        3. Steps 1 and 2 are repeated for (horizontal, self.code_b, east, west) in place of
            (vertical, self.code_a, north, south).

        Args:
            strategy: The strategy used by nx.greedy_color to color edges of the Tanner graph.
                Default: "smallest_last".
        """
        node_map = HGPCode.get_product_node_map(self.code_a.graph.nodes, self.code_b.graph.nodes)

        # collect subgraphs of North and South edges
        edges_n: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_s: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        coloring_a = nx.greedy_color(nx.line_graph(self.code_a.graph.to_undirected()), strategy)
        for (check_a, data_a), color in coloring_a.items():
            for node_b in self.code_b.graph.nodes:
                node_0 = node_map[check_a, node_b]
                node_1 = node_map[data_a, node_b]
                data, check = sorted([node_0, node_1])
                edges_ns = edges_s if (color + node_b.is_data) % 2 == 0 else edges_n
                edges_ns[color].append((check, data))
        graphs_n = tuple(self.graph.edge_subgraph(edges) for edges in edges_n.values())
        graphs_s = tuple(self.graph.edge_subgraph(edges) for edges in edges_s.values())

        # collect subgraphs of East and West edges
        edges_e: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_w: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        coloring_b = nx.greedy_color(nx.line_graph(self.code_b.graph.to_undirected()), strategy)
        for (check_b, data_b), color in coloring_b.items():
            for node_a in self.code_a.graph.nodes:
                node_0 = node_map[node_a, check_b]
                node_1 = node_map[node_a, data_b]
                data, check = sorted([node_0, node_1])
                edges_ew = edges_e if (color + node_b.is_data) % 2 == 0 else edges_w
                edges_ew[color].append((check, data))
        graphs_e = tuple(self.graph.edge_subgraph(edges) for edges in edges_e.values())
        graphs_w = tuple(self.graph.edge_subgraph(edges) for edges in edges_w.values())

        return graphs_n + graphs_e + graphs_w + graphs_s

    @staticmethod
    def get_matrix_product(
        matrix_a: npt.NDArray[np.int_ | np.object_],
        matrix_b: npt.NDArray[np.int_ | np.object_],
    ) -> tuple[npt.NDArray[np.int_ | np.object_], npt.NDArray[np.int_ | np.object_]]:
        """Hypergraph product of two parity check matrices."""
        # construct the nontrivial blocks of the final parity check matrices
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_T_Im2 = np.kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the X-sector and Z-sector parity check matrices
        matrix_x = np.block([mat_H1_In2, mat_Im1_H2_T])
        matrix_z = np.block([-mat_In1_H2, mat_H1_T_Im2])
        return matrix_x.view(type(matrix_a)), matrix_z.view(type(matrix_a))

    @staticmethod
    def get_graph_product(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> nx.DiGraph:
        """Hypergraph product of two Tanner graphs."""
        graph = nx.DiGraph()
        field = getattr(graph_a, "field", galois.GF(DEFAULT_FIELD_ORDER))
        _Pauli = Pauli if field.order == 2 else QuditPauli

        # start with a cartesian products of the input graphs
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation, and tag each edge with a QuditPauli
        for node_fst, node_snd, data in graph_product.edges(data=True):
            # identify the sectors of two nodes
            sector_fst = HGPCode.get_sector(*node_fst)
            sector_snd = HGPCode.get_sector(*node_snd)

            # identify data-qudit vs. check nodes, and their sectors
            if sector_fst in [(0, 0), (1, 1)]:
                node_qudit, sector_qudit = node_fst, sector_fst
                node_check, sector_check = node_snd, sector_snd
            else:
                node_check, sector_check = node_fst, sector_fst
                node_qudit, sector_qudit = node_snd, sector_snd

            # start with an X-type operator
            op = _Pauli((data.get("val", 0), 0))

            # switch to Z-type operator for check qudits in the (0, 1) sector
            if sector_check == (0, 1):
                op = ~op

            # account for the minus sign in the (0, 0) sector of the Z-type subcode
            if isinstance(op, QuditPauli) and sector_qudit == (0, 0) and op.value[Pauli.Z]:
                op = -op

            graph.add_edge(node_check, node_qudit)
            graph[node_check][node_qudit][Pauli] = op

        # relabel nodes, from (node_a, node_b) --> node_combined
        node_map = HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes)
        graph = nx.relabel_nodes(graph, node_map)
        graph.field = field
        return graph

    @staticmethod
    def get_sector(node_a: Node, node_b: Node) -> tuple[int, int]:
        """Get the sector of a node in a graph product."""
        return int(not node_a.is_data), int(not node_b.is_data)

    @staticmethod
    def get_product_node_map(
        nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        # identify the number of data/check nodes, and the total numer of X/Z checks
        num_data_a = sum(node.is_data for node in nodes_a)
        num_data_b = sum(node.is_data for node in nodes_b)
        num_checks_a = len(nodes_a) - num_data_a
        num_checks_b = len(nodes_b) - num_data_b
        num_checks_x = num_checks_a * num_data_b
        num_checks_z = num_checks_b * num_data_a
        num_checks = num_checks_x + num_checks_z

        node_map = {}
        index_data, index_check = 0, 0
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            if HGPCode.get_sector(node_a, node_b) in [(0, 0), (1, 1)]:
                node = Node(index=index_data, is_data=True)
                index_data += 1
            else:
                # shift check node indices for consistency with the CSSCode parity check matrix
                index = (index_check + num_checks_x) % num_checks
                node = Node(index=index, is_data=False)
                index_check += 1
            node_map[node_a, node_b] = node
        return node_map

    @staticmethod
    def get_canonical_logical_ops(
        code_a: ClassicalCode, code_b: ClassicalCode
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical operators for the hypergraph product code.

        These operators are essentially those in Lemma 1 of arXiv:2204.10812v3, modified using pivot
        matrices similarly to Theorem VIII.10 of arXiv:2502.07150v1 to ensure pair-wise
        anti-commutation relations.
        """
        assert code_a.field is code_b.field
        code_field = code_a.field

        generator_a = code_a.generator.row_reduce()
        generator_b = code_b.generator.row_reduce()
        generator_a_T = code_a.matrix.T.null_space()
        generator_b_T = code_b.matrix.T.null_space()

        pivots_a = code_field.Zeros(generator_a.shape)
        pivots_b = code_field.Zeros(generator_b.shape)
        pivots_a[range(len(pivots_a)), first_nonzero_cols(generator_a)] = 1
        pivots_b[range(len(pivots_b)), first_nonzero_cols(generator_b)] = 1

        pivots_a_T = code_field.Zeros(generator_a_T.shape)
        pivots_b_T = code_field.Zeros(generator_b_T.shape)
        pivots_a_T[range(len(pivots_a_T)), first_nonzero_cols(generator_a_T)] = 1
        pivots_b_T[range(len(pivots_b_T)), first_nonzero_cols(generator_b_T)] = 1

        logical_ops_x_l = np.kron(pivots_a, generator_b)
        logical_ops_x_r = np.kron(generator_a_T, pivots_b_T)

        logical_ops_z_l = np.kron(generator_a, pivots_b)
        logical_ops_z_r = np.kron(pivots_a_T, generator_b_T)

        logical_ops_x = scipy.linalg.block_diag(logical_ops_x_l, logical_ops_x_r)
        logical_ops_z = scipy.linalg.block_diag(logical_ops_z_l, logical_ops_z_r)
        return logical_ops_x.view(code_field), logical_ops_z.view(code_field)

    def _get_distance_exact(self, pauli: PauliXZ | None) -> int | float:
        """Exact distance calculation for hypergraph product codes, from arXiv:2308.15520."""
        if pauli is not None:
            # TODO: address the case of X and Z distance
            return NotImplemented  # pragma: no cover
        code_a = self.code_a
        code_b = self.code_b
        code_a_T = ClassicalCode(self.code_a.matrix.T)
        code_b_T = ClassicalCode(self.code_b.matrix.T)
        if code_a_T.get_distance() is np.nan or code_b_T.get_distance() is np.nan:
            return min(code_a.get_distance(), code_b.get_distance())  # pragma: no cover
        if code_a.get_distance() is np.nan or code_b.get_distance() is np.nan:
            return min(code_a_T.get_distance(), code_b_T.get_distance())  # pragma: no cover
        return min(
            code_a.get_distance(),
            code_b.get_distance(),
            code_a_T.get_distance(),
            code_b_T.get_distance(),
        )


class SHPCode(CSSCode):
    """Subsystem hypergraph product code.

    A subsystem hypergraph product code (SHPCode) is constructed from two classical codes.  Unlike
    the ordinary hypergraph product code, an SHPCode depends only on the actual classical codes it
    is built from; in particular, an SHPCode does not depend on the choice of parity check matrices
    for the underlying classical codes.

    If the classical generating codes of an SHPCode have code parameters [n1, k1, d1], [n2, k2, d2],
    the SHPCode has parameters [n1 n2, k1 k2, min(d1, d2)].

    References:
    - https://errorcorrectionzoo.org/c/subsystem_quantum_parity
    - https://arxiv.org/abs/2002.06257
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        """Subsystem hypergraph product of two classical codes, as in arXiv:2002.06257."""
        if code_b is None:
            code_b = code_a
        self.code_a = ClassicalCode(code_a, field)
        self.code_b = ClassicalCode(code_b, field)
        code_field = self.code_a.field

        matrix_x, matrix_z = SHPCode.get_matrix_product(self.code_a.matrix, self.code_b.matrix)
        super().__init__(
            matrix_x.view(np.ndarray).astype(int),
            matrix_z.view(np.ndarray).astype(int),
            code_field.order,
            is_subsystem_code=True,
        )

        stab_ops_x = np.kron(self.code_a.matrix, self.code_b.generator)
        stab_ops_z = np.kron(-self.code_a.generator, self.code_b.matrix)
        self._stabilizer_ops = scipy.linalg.block_diag(stab_ops_x, stab_ops_z).view(code_field)

        if set_logicals:
            logical_ops_xz = SHPCode.get_canonical_logical_ops(self.code_a, self.code_b)
            self.set_logical_ops_xz(*logical_ops_xz, validate=False)

    @staticmethod
    def get_matrix_product(
        matrix_a: npt.NDArray[np.int_ | np.object_],
        matrix_b: npt.NDArray[np.int_ | np.object_],
    ) -> tuple[npt.NDArray[np.int_ | np.object_], npt.NDArray[np.int_ | np.object_]]:
        """Subsystem hypergraph product of two parity check matrices."""
        matrix_x = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int)).view(type(matrix_a))
        matrix_z = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b).view(type(matrix_a))
        return matrix_x, matrix_z

    @staticmethod
    def get_canonical_logical_ops(
        code_x: ClassicalCode, code_z: ClassicalCode
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical operators for the subsystem hypergraph product code.

        These operators are essentially those in Theorem VIII.10 of arXiv:2502.07150v1, generalized
        slightly to account for the possibility that code_x != code_z.
        """
        assert code_x.field is code_z.field
        code_field = code_x.field

        generator_x = code_x.generator.row_reduce()
        generator_z = code_z.generator.row_reduce()

        pivots_x = code_field.Zeros(generator_x.shape)
        pivots_z = code_field.Zeros(generator_z.shape)
        pivots_x[range(len(pivots_x)), first_nonzero_cols(generator_x)] = 1
        pivots_z[range(len(pivots_z)), first_nonzero_cols(generator_z)] = 1

        logical_ops_x = np.kron(pivots_x, generator_z)
        logical_ops_z = np.kron(generator_x, pivots_z)
        return logical_ops_x.view(code_field), logical_ops_z.view(code_field)

    def _get_distance_exact(self, pauli: PauliXZ | None) -> int | float:
        """Exact distance calculation for subsystem hypergraph product codes."""
        match pauli:
            case Pauli.X:
                return self.code_b.get_distance()
            case Pauli.Z:
                return self.code_a.get_distance()
            case _:
                return min(self.code_a.get_distance(), self.code_b.get_distance())


class LPCode(CSSCode):
    """Lifted product code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are RingArrays, or matrices whose entries are members of a group algebra
    over a finite field F_q.  Each of these entries can be "lifted" to a representation as
    orthogonal matrices over F_q, in which case the RingArray is interpreted as a block matrix; this
    is called "lifting" the RingArray.

    As an example, the lift-connected surface code in Eq. (2) of https://arxiv.org/pdf/2401.02911v2
    can be constructed by

        import numpy as np
        from qldpc.abstract import CyclicGroup, GroupRing, RingArray, RingMember
        from qldpc.codes import RepetitionCode, LPCode

        num_copies = 5  # the number of surface codes to stitch together
        group = CyclicGroup(num_copies)
        ring = GroupRing(group)
        x = RingMember(ring, group.generators[0])  # generator of the cyclic group

        # stitch together small surface codes by hand
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
    - A lifted product code with RingArrays of size 1×1 is a two-block code (more specifically, a
        two-block group-algebra code).  If the base group of the RingArrays is a cyclic group, the
        resulting lifted product code is a generalized bicycle code.
    - A lifted product code with RingArrays whose entries get lifted to 1×1 matrices is a
        hypergraph product code built from the lifted RingArrays.
    - One way to get an LPCode: take a classical code with parity check matrix H and multiply it by
        a diagonal matrix D = diag(a_1, a_2, ... a_n), where all {a_j} are elements of a group
        algebra.  The RingArray P = H @ D can then be used for one of the RingArrays of an LPCode.

    References:
    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    - https://arxiv.org/abs/2306.16400
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.object_] | Sequence[Sequence[object]],
        matrix_b: npt.NDArray[np.object_] | Sequence[Sequence[object]] | None = None,
    ) -> None:
        """Lifted product of two RingArrays, as in arXiv:2012.04068."""
        if matrix_b is None:
            matrix_b = matrix_a
        matrix_a = abstract.RingArray(matrix_a)
        matrix_b = abstract.RingArray(matrix_b)
        field = matrix_a.field.order

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(matrix_a, matrix_b)
        assert isinstance(matrix_x, abstract.RingArray)
        assert isinstance(matrix_z, abstract.RingArray)

        # identify the number of qudits in each sector
        self.sector_size = matrix_a.group.lift_dim * np.outer(
            matrix_a.shape[::-1],
            matrix_b.shape[::-1],
        )

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self._default_conjugate = slice(self.sector_size[0, 0], None)

        super().__init__(matrix_x.lift(), matrix_z.lift(), field, is_subsystem_code=False)


class SLPCode(CSSCode):
    """Subsystem lifted product code.

    The subsystem lifted product code is a lifted version of the subsystem hypergraph product code.
    That is, the SLPCode is to the SHPCode what the LPCode is to the HGPCode.
    See help(qldpc.codes.LPCode) for additional information.

    As an example, the SLPCode in example 1 on page 6 of https://arxiv.org/pdf/2404.18302v1 can be
    constructed by

        from qldpc.abstract import CyclicGroup, GroupRing, RingMember, RingArray
        from qldpc.codes import SLPCode

        group = CyclicGroup(2)
        ring = GroupRing(group)
        x = group.generators[0]
        matrix = RingArray.build([[1, x, x], [x, x, 1]], ring)  # Eq. 21 of arXiv:2404.18302v1
        code = SLPCode(matrix)
        assert code.get_code_params() == (18, 4, 2)

    while the SLPCode in example 2 is

        group = CyclicGroup(3)
        ring = GroupRing(group)
        x = RingMember(ring, group.generators[0])
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
    ) -> None:
        """Subsystem lifted product of two RingArrays."""
        if matrix_b is None:
            matrix_b = matrix_a
        matrix_a = abstract.RingArray(matrix_a)
        matrix_b = abstract.RingArray(matrix_b)
        field = matrix_a.field.order

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = SHPCode.get_matrix_product(matrix_a, matrix_b)
        assert isinstance(matrix_x, abstract.RingArray)
        assert isinstance(matrix_z, abstract.RingArray)

        super().__init__(matrix_x.lift(), matrix_z.lift(), field, is_subsystem_code=True)


####################################################################################################
# quantum Tanner code


# TODO: example notebook featuring this code
class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    Altogether, a quantum Tanner code is defined by:
    - two symmetric (self-inverse) subsets A and B of a group G, and
    - two classical codes C_A and C_B, respectively with block lengths |A| and |B|.

    The qudits of a quantum Tanner code live on the faces of a Cayley complex built out of A and B.
    Each face of the Cayley complex looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

    where (g,a,b) is an element of (G,A,B), and f(g,a,b) = {g, ab, gb, agb}.  We define two
    (directed) subgraphs on the Cayley complex:
    - subgraph_x with edges ( g, f(g,a,b)), and
    - subgraph_z with edges (ag, f(g,a,b)).

    The X-type parity checks of a quantum Tanner code are then given by the classical Tanner code on
    subgraph_x with subcode ~(C_A ⨂ C_B), where ~C is the dual code to C.  Z-type parity checks are
    similarly given by the classical Tanner code on subgraph_z with subcode ~(~C_A ⨂ ~C_B).

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
        field: int | None = None,
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
        super().__init__(code_x, code_z, field, is_subsystem_code=False)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QTCode)
            and other.code_a == other.code_a
            and other.code_b == other.code_b
            and other.complex.subset_a == other.complex.subset_a
            and other.complex.subset_b == other.complex.subset_b
            and other.complex.bipartite == other.complex.bipartite
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

        where f(g,a,b) = {g, ab, gb, agb}.  Specifically, the (directed) subgraphs are:
        - subgraph_x with edges ( g, f(g,a,b)), and
        - subgraph_z with edges (ag, f(g,a,b)).
        Classical Tanner codes on these subgraphs are used as to construct quantum Tanner code.

        As a matter of practice, defining classical Tanner codes on subgraph_x and subgraph_z
        requires choosing an ordering on the edges incident to every source node of these graphs.
        If the group G is equipped with a total order, a natural ordering of edges incident to every
        source node is induced by assigning the label (a, b) to edge (g, f(g,a,b)).  Consistency
        then requires that edge (ag, f(g,a,b)) has label (a^-1, b), as verified by defining g' = ag
        and checking that f(g,a,b) = f(g',a^-1,b).
        """
        subset_a = cayplex.cover_subset_a
        subset_b = cayplex.cover_subset_b

        # identify the identity element
        member = next(iter(subset_a))
        identity = member * ~member

        # identify the set of nodes for which we still need to add faces
        nodes_to_add = set([identity])

        # build the subgraphs one node at a time
        subgraph_x = nx.DiGraph()
        subgraph_z = nx.DiGraph()
        while nodes_to_add:
            gg = nodes_to_add.pop()

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
        field: int | None = None,
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
        subset_b = group.random_symmetric_subset(code_b.num_bits) if not one_subset else subset_a
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)

    def save(self, path: str, *headers: str) -> None:
        """Save the generating data of this code to a file."""
        # convert subsets to arrays
        subset_a = np.array([gen.array_form for gen in self.complex.subset_a])
        subset_b = np.array([gen.array_form for gen in self.complex.subset_b])

        # create save directory if necessary
        save_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(save_dir, exist_ok=True)

        with open(path, "w") as file:
            # write provided headers
            for header in headers:
                for line in header.splitlines():
                    file.write(f"# {line}\n")

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
        subset_a = set(abstract.GroupMember(gen) for gen in arrays[0])
        subset_b = set(abstract.GroupMember(gen) for gen in arrays[1])
        code_a = ClassicalCode(arrays[2], field)
        code_b = ClassicalCode(arrays[3], field)
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)


####################################################################################################
# surface code and friends


class SurfaceCode(CSSCode):
    """The one and only!

    Actually, there are two variants: "ordinary" and "rotated" surface codes.
    The rotated code is more qubit-efficient.

    References:
    - https://errorcorrectionzoo.org/c/toric
    - https://errorcorrectionzoo.org/c/surface
    - https://errorcorrectionzoo.org/c/rotated_surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | None = None,
        *,
        rotated: bool = True,
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols
        self.rotated = rotated

        # save known distances and dimension
        self._distance_x = cols
        self._distance_z = rows
        self._dimension = 1

        if not rotated:
            # "original" surface code
            code_a = RepetitionCode(rows, field)
            code_b = RepetitionCode(cols, field)
            self.parent_code = HGPCode(code_a, code_b)
            matrix_x = self.parent_code.matrix_x.view(np.ndarray)
            matrix_z = self.parent_code.matrix_z.view(np.ndarray)
            self._default_conjugate: list[int] | slice = slice(
                self.parent_code.sector_size[0, 0], None
            )

        else:
            # rotated surface code
            matrix_x, matrix_z = SurfaceCode.get_rotated_checks(rows, cols)
            self._default_conjugate = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

            # invert Z-type Pauli on every other qubit
            code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
            if code_field.order > 2:
                matrix_z = matrix_z.view(code_field)
                matrix_z[:, self._default_conjugate] *= -1

        super().__init__(matrix_x, matrix_z, field=field, promise_equal_distance_xz=rows == cols)

    @staticmethod
    def get_rotated_checks(
        rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Example 5x5 rotated surface code layout:

             ―――     ―――
            | ⋅ |   | ⋅ |
            ○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○
                | ⋅ |   | ⋅ |
                 ―――     ―――

        Here:
        - Circles (○) denote data qubits (of which there are 5×5 = 25 total).
        - Tiles with a cross (×) denote X-type parity checks (12 total).
        - Tiles with a dot (⋅) denote Z-type parity checks (12 total).
        """

        def get_check_pauli(row: int, col: int) -> PauliXZ:
            """What type of stabilizer does this check measure?"""
            return Pauli.X if (row + col) % 2 == 0 else Pauli.Z

        def check_is_used(row: int, col: int) -> bool:
            """Is the check qubit with these coordinates used?"""
            if row == 0 or row == rows:
                return 0 < col < cols and get_check_pauli(row, col) is Pauli.Z
            if col == 0 or col == cols:
                return 0 < row < rows and get_check_pauli(row, col) is Pauli.X
            return 0 < row < rows and 0 < col < cols

        def get_check(row: int, col: int) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            row_indices = [row - 1, row, row - 1, row]
            col_indices = [col - 1, col - 1, col, col]
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                if 0 <= row < rows and 0 <= col < cols:
                    check[row, col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row, col in itertools.product(range(rows + 1), range(cols + 1)):
            if check_is_used(row, col):
                check = get_check(row, col)
                if get_check_pauli(row, col) is Pauli.X:
                    checks_x.append(check)
                else:
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        If this is an unrotated surface code, return the syndrome subgraphs of the parent HGPCode.
        Otherwise, organize edges of the Tanner graph by an orientation in {NW, NE, SW, SE}, and by
        whether they are used for the readout of X-type or Z-type syndromes.  Interleave these edges
        in such a way as to minimize circuit depth and avoid hook errors.

        Args:
            strategy: Only used if self.rotated is False, in which case this argument is passed to
                HGPCode.get_syndrome_subgraphs to color the edges of the Tanner graph of this code.
                Default: "smallest_last".
        """
        if not self.rotated:
            return self.parent_code.get_syndrome_subgraphs(strategy=strategy)

        def get_check_pauli(row: int, col: int) -> PauliXZ:
            """What type of stabilizer does this check measure?"""
            return Pauli.X if (row + col) % 2 == 0 else Pauli.Z

        def check_is_used(row: int, col: int) -> bool:
            """Is the check qubit with these coordinates used?"""
            if row == 0 or row == self.rows:
                return 0 < col < self.cols and get_check_pauli(row, col) is Pauli.Z
            if col == 0 or col == self.cols:
                return 0 < row < self.rows and get_check_pauli(row, col) is Pauli.X
            return 0 < row < self.rows and 0 < col < self.cols

        # identify all coordinates of check qubits, and a map from coordinates to a Node
        check_node_coords = sorted(
            [
                (row, col)
                for row, col in itertools.product(range(self.rows + 1), range(self.cols + 1))
                if check_is_used(row, col)
            ],
            key=lambda row_col: (int(get_check_pauli(*row_col)), *row_col),
        )
        node_map = {
            (row, col): Node(index, is_data=False)
            for index, (row, col) in enumerate(check_node_coords)
        }

        # collect edges of the Tanner graph by type: (pauli, orientation)
        edges: dict[tuple[Pauli, str], list[tuple[Node, Node]]] = collections.defaultdict(list)
        for qubit, (row, col) in enumerate(itertools.product(range(self.rows), range(self.cols))):
            data_node = Node(qubit, is_data=True)

            check_nw = (row, col)
            check_ne = (row, col + 1)
            check_sw = (row + 1, col)
            check_se = (row + 1, col + 1)
            if check_is_used(*check_nw):
                check_pauli = get_check_pauli(*check_nw)
                check_node = node_map[check_nw]
                edges[check_pauli, "nw"].append((check_node, data_node))
            if check_is_used(*check_ne):
                check_pauli = get_check_pauli(*check_ne)
                check_node = node_map[check_ne]
                edges[check_pauli, "ne"].append((check_node, data_node))
            if check_is_used(*check_sw):
                check_pauli = get_check_pauli(*check_sw)
                check_node = node_map[check_sw]
                edges[check_pauli, "sw"].append((check_node, data_node))
            if check_is_used(*check_se):
                check_pauli = get_check_pauli(*check_se)
                check_node = node_map[check_se]
                edges[check_pauli, "se"].append((check_node, data_node))

        # return subgraphs in the order that minimizes hook errors
        subgraphs = {key: self.graph.edge_subgraph(edge_group) for key, edge_group in edges.items()}
        return (
            subgraphs[Pauli.X, "nw"],
            subgraphs[Pauli.Z, "nw"],
            subgraphs[Pauli.X, "sw"],
            subgraphs[Pauli.Z, "ne"],
            subgraphs[Pauli.X, "ne"],
            subgraphs[Pauli.Z, "sw"],
            subgraphs[Pauli.X, "se"],
            subgraphs[Pauli.Z, "se"],
        )


class ToricCode(CSSCode):
    """Surface code with periodic boundary conditions, encoding two logical qudits.

    References:
    - https://errorcorrectionzoo.org/c/toric
    - https://errorcorrectionzoo.org/c/surface
    - https://errorcorrectionzoo.org/c/rotated_surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | None = None,
        *,
        rotated: bool = True,
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols
        self.rotated = rotated

        # save known distances and dimension
        self._distance_x = self._distance_z = min(rows, cols)
        self._dimension = 2

        if not rotated:
            # "original" toric code
            code_a = RingCode(rows, field)
            code_b = RingCode(cols, field)
            self.parent_code = HGPCode(code_a, code_b)
            matrix_x = self.parent_code.matrix_x.view(np.ndarray)
            matrix_z = self.parent_code.matrix_z.view(np.ndarray)
            self._default_conjugate: list[int] | slice = slice(
                self.parent_code.sector_size[0, 0], None
            )

        else:
            # rotated toric code
            if rows % 2 or cols % 2:
                raise ValueError(
                    f"Rotated toric code must have even side lengths, not {rows} and {cols}"
                )

            matrix_x, matrix_z = ToricCode.get_rotated_checks(rows, cols)
            self._default_conjugate = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

            # invert Z-type Pauli on every other qubit
            code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
            if code_field.order > 2:
                matrix_z = matrix_z.view(code_field)
                matrix_z[:, self._default_conjugate] *= -1

            if rows == cols == 2 and rotated:
                # All Toric codes have redundant parity checks, but the case of the rotated 2x2
                # Toric code is particularly egregious: the two X/Z checks are *equal*.  So remove
                # the extra checks in this case.
                matrix_x = matrix_x[:-1]
                matrix_z = matrix_z[:-1]

        super().__init__(matrix_x, matrix_z, field=field, promise_equal_distance_xz=rows == cols)

    @staticmethod
    def get_rotated_checks(
        rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Same as in SurfaceCode.get_rotated_checks, but with periodic boundary conditions.
        """

        def get_check_pauli(row: int, col: int) -> PauliXZ:
            """What type of stabilizer does this check measure?"""
            return Pauli.X if (row + col) % 2 == 0 else Pauli.Z

        def get_check(row: int, col: int) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            row_indices = np.array([row - 1, row, row - 1, row]) % rows
            col_indices = np.array([col - 1, col - 1, col, col]) % cols
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                check[row, col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row, col in itertools.product(range(rows), range(cols)):
            check = get_check(row, col)
            if get_check_pauli(row, col) is Pauli.X:
                checks_x.append(check)
            else:
                checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        If this is an unrotated toric code, return the syndrome subgraphs of the parent HGPCode.
        Otherwise, return the subgraphs of edges oriented along (NW, NE, SW, SE) directions.

        Args:
            strategy: Only used if self.rotated is False, in which case this argument is passed to
                HGPCode.get_syndrome_subgraphs to color the edges of the Tanner graph of this code.
                Default: "smallest_last".
        """
        if not self.rotated:
            return self.parent_code.get_syndrome_subgraphs(strategy=strategy)

        def get_check_pauli(row: int, col: int) -> PauliXZ:
            """What type of stabilizer does this check measure?"""
            return Pauli.X if (row + col) % 2 == 0 else Pauli.Z

        # identify all coordinates of check qubits, and a map from coordinates to a Node
        check_node_coords = sorted(
            [(row, col) for row, col in itertools.product(range(self.rows), range(self.cols))],
            key=lambda row_col: (int(get_check_pauli(*row_col)), *row_col),
        )
        node_map = {
            (row, col): Node(index, is_data=False)
            for index, (row, col) in enumerate(check_node_coords)
        }

        # collect edges of the Tanner graph by type (orientation)
        edges: dict[str, list[tuple[Node, Node]]] = collections.defaultdict(list)
        for qubit, (row, col) in enumerate(itertools.product(range(self.rows), range(self.cols))):
            node_data = Node(qubit, is_data=True)
            edges["nw"].append((node_map[row, col], node_data))
            edges["ne"].append((node_map[row, (col + 1) % self.cols], node_data))
            edges["sw"].append((node_map[(row + 1) % self.rows, col], node_data))
            edges["se"].append((node_map[(row + 1) % self.rows, (col + 1) % self.cols], node_data))

        subgraphs = {key: self.graph.edge_subgraph(edge_group) for key, edge_group in edges.items()}
        return subgraphs["nw"], subgraphs["ne"], subgraphs["sw"], subgraphs["se"]


class GeneralizedSurfaceCode(CSSCode):
    """Surface or toric code defined on a multi-dimensional hypercubic lattice.

    References:
    - https://errorcorrectionzoo.org/c/higher_dimensional_surface
    """

    def __init__(
        self,
        size: int,
        dim: int,
        field: int | None = None,
        *,
        periodic: bool = False,
    ) -> None:
        if dim < 2:
            raise ValueError(
                f"The dimension of a generalized surface code should be >= 2 (provided: {dim})"
            )

        # save known distances
        # TODO: find and link source for these
        self._distance_x = size ** (dim - 1)
        self._distance_z = size

        base_code = RingCode(size, field) if periodic else RepetitionCode(size, field)

        # build a chain complex one link at a time
        chain = ChainComplex(base_code.matrix)
        link = ChainComplex(base_code.matrix.T)
        for _ in range(dim - 1):
            chain = ChainComplex.tensor_product(chain, link)

            # to reduce computational overhead, remove chain links that we don't care about
            chain = ChainComplex(*chain.ops[:2])

        matrix_x, matrix_z = chain.op(1), chain.op(2).T
        assert not isinstance(matrix_x, abstract.RingArray)
        assert not isinstance(matrix_z, abstract.RingArray)
        super().__init__(matrix_x, matrix_z, field)


####################################################################################################
# miscellaneous codes


class BaconShorCode(SHPCode):
    """Bacon-Shor code on a square grid, implemented as a subsystem hypergraph product code.

    References:
    - https://errorcorrectionzoo.org/c/bacon_shor
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        code_x = RepetitionCode(rows, field)
        code_z = RepetitionCode(cols, field) if cols is not None else None
        super().__init__(code_x, code_z, field, set_logicals=set_logicals)

        self._distance_x = cols
        self._distance_z = rows


class SHYPSCode(SHPCode):
    """Subsystem hypergraph product simplex (SHYPS) code.

    Subsystem hypergraph product codes naturally inherit the automorphisms (symmetries) of the
    classical codes that they are built from.  The SHYPSCode is built from classical SimplexCodes
    that have a very large automorphism group, which gives SHYPSCodes a large set of
    SWAP-transversal Clifford operations.

    References:
    - https://errorcorrectionzoo.org/c/shyps
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int | None = None,
        field: int | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        dim_z = dim_z if dim_z is not None else dim_x

        code_x = SimplexCode(dim_x, field)
        code_z = SimplexCode(dim_z, field)
        super().__init__(code_x, code_z, set_logicals=set_logicals)

        self._dimension = dim_x * dim_z
