"""Classical error-correcting codes.

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

import itertools
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt
import sympy

from qldpc import abstract
from qldpc._util import networkx as nx

from .common import ClassicalCode


class RepetitionCode(ClassicalCode):
    """Classical repetition code: the ``[n, 1, n]`` code whose code words are constant vectors.

    References:

    - https://errorcorrectionzoo.org/c/repetition
    """

    def __init__(self, bits: int, field: int | type[galois.FieldArray] | None = None) -> None:
        self._field = abstract.resolve_field(field)
        self._matrix = self.field.Zeros((bits - 1, bits))
        for row in range(bits - 1):
            self._matrix[row, row] = 1
            self._matrix[row, row + 1] = -self.field(1)

        self._dimension = 1
        self._distance = bits


class RingCode(ClassicalCode):
    """Classical ring code: repetition code with periodic boundary conditions.

    The periodic boundary adds one (redundant) parity check, so a RingCode has the same code words
    as a RepetitionCode of the same block length, and hence the same ``[n, 1, n]`` parameters.

    References:

    - https://errorcorrectionzoo.org/c/repetition
    """

    def __init__(self, bits: int, field: int | type[galois.FieldArray] | None = None) -> None:
        self._field = abstract.resolve_field(field)
        self._matrix = self.field.Zeros((bits, bits))
        for row in range(bits):
            self._matrix[row, row] = 1
            self._matrix[row, (row + 1) % bits] = -self.field(1)

        self._dimension = 1
        self._distance = bits


class CyclicCode(ClassicalCode):
    """Classical cyclic code.

    A CyclicCode is determined by an integer block length and an integer polynomial in one variable.
    The parity check matrix of a CyclicCode is obtained by interpreting the polynomial as an element
    of a cyclic group algebra, and lifting this polynomial to a square matrix.  The multiplicative
    identity and the generator of the group correspond, respectively, to the identity and shift
    matrices.

    The CyclicCode with polynomial ``1 - x`` is a RingCode.

    References:

    - https://errorcorrectionzoo.org/c/cyclic
    """

    def __init__(
        self, bits: int, poly: sympy.Basic, field: int | type[galois.FieldArray] | None = None
    ) -> None:
        """Construct a cyclic code from a block length and a polynomial in one variable."""
        if not isinstance(poly, sympy.Basic) or not len(poly.free_symbols) == 1:
            raise ValueError(f"{poly} is not a univariate polynomial")
        group = abstract.CyclicGroup(bits)
        ring = abstract.GroupRing(group, field)
        symbols = dict(zip(poly.free_symbols, group.generators))
        matrix = ring.eval(poly, symbols).lift().T  # transpose the lift by convention
        super().__init__(matrix, field)


class GolayCode(CyclicCode):
    """Classical binary [23, 12, 7] Golay code.

    A famous "perfect" cyclic code: every 23-bit word lies within distance 3 of a unique codeword,
    so it corrects any 3 bit-flip errors.  As a cyclic code, its codewords are the multiples
    (mod ``x^23 - 1``) of the Golay generator polynomial ``g(x)``, a degree-11 factor of
    ``x^23 - 1``.

    The parity checks are the cyclic shifts of the weight-8 polynomial
    ``h(x) = (x + 1) g(x)``, which generates the dual code ``G^perp = [23, 11, 8]``.

    References:

    - https://errorcorrectionzoo.org/c/golay
    """

    def __init__(self) -> None:
        """Construct the [23, 12, 7] Golay code."""
        # the Golay generator polynomial g(x), a degree-11 factor of x^23 - 1
        x = sympy.Symbol("x")
        generator = x**11 + x**9 + x**7 + x**6 + x**5 + x + 1

        # use the cyclic shifts of the weight-8 polynomial (x + 1) g(x) as parity checks
        super().__init__(23, (x + 1) * generator, field=2)

        self._dimension = 12
        self._distance = 7


class HammingCode(ClassicalCode):
    """Classical Hamming code.

    When working over the binary field (0s and 1s), the parity check matrix of the HammingCode is
    built by stacking together (as columns) all nonzero bitstrings.  More generally, the parity
    check matrix is built from a maximal set of linearly independent nonzero vectors over a finite
    field; equivalently, from all vectors whose first nonzero element is a 1.

    References:

    - https://errorcorrectionzoo.org/c/hamming
    - https://errorcorrectionzoo.org/c/q-ary_hamming
    """

    def __init__(self, size: int, field: int | type[galois.FieldArray] | None = None) -> None:
        """Construct a Hamming code of a given rank."""
        if size < 2:
            raise ValueError(f"Hamming codes require a rank of at least 2 (provided: {size})")
        self._field = abstract.resolve_field(field)
        if self.field is galois.GF2:
            # collect all nonzero bitstrings
            bitstrings = list(itertools.product([0, 1], repeat=size))
            self._matrix = self.field(bitstrings[1:]).T

        else:
            # collect all nonzero vectors whose first nonzero element is a 1
            vectors = [
                (0,) * top_row + (1,) + rest
                for top_row in range(size - 1, -1, -1)
                for rest in itertools.product(range(self.field.order), repeat=size - top_row - 1)
            ]
            self._matrix = self.field(vectors).T

        self._dimension = len(self) - len(self._matrix)
        self._distance = 3


class ExtendedHammingCode(ClassicalCode):
    """Classical extended Hamming code: the ordinary Hamming code with an extra parity bit.

    The extended Hamming code of size m is also equal to ``ReedMullerCode(m - 2, m)``.

    References:

    - https://errorcorrectionzoo.org/c/extended_hamming
    """

    def __init__(self, size: int) -> None:
        """Construct an extended Hamming code of a given rank."""
        if size < 2:
            raise ValueError(
                f"Extended Hamming codes require a rank of at least 2 (provided: {size})"
            )
        matrix: npt.NDArray[np.int_] = HammingCode(size).matrix
        matrix = np.column_stack([np.zeros(matrix.shape[0], dtype=int), matrix])
        matrix = np.vstack([np.ones(matrix.shape[1], dtype=int), matrix])
        matrix[0] += matrix[1]
        super().__init__(matrix)

        self._dimension = len(self) - len(self._matrix)
        self._distance = 4


class ReedMullerCode(ClassicalCode):
    """Classical Reed-Muller code.

    A Reed-Muller code with order r and size m, denoted ``RM(r, m)``, has code parameters

        ``[2**m, k, 2**(m-r)]``

    where

        ``k = sum_(j = 0)^r (m choose j)``.

    References:

    - https://errorcorrectionzoo.org/c/reed_muller
    - https://feog.github.io/10-coding.pdf
    """

    def __init__(
        self, order: int, size: int, field: int | type[galois.FieldArray] | None = None
    ) -> None:
        self._assert_valid_params(order, size)
        self._order = order
        self._size = size

        generator = ReedMullerCode.get_generator(order, size)
        self._matrix = ClassicalCode(generator, field).generator
        self._field = abstract.resolve_field(field)

        self._dimension = len(generator)
        self._distance = 2 ** (size - order)

    @property
    def size(self) -> int:
        """The size parameter of this code."""
        return self._size

    @property
    def order(self) -> int:
        """The order parameter of this code."""
        return self._order

    @staticmethod
    def get_generator(order: int, size: int) -> npt.NDArray[np.int_]:
        """Get the generator matrix for the specified Reed-Muller code."""
        ReedMullerCode._assert_valid_params(order, size)

        if order == 0:
            return np.ones((1, 2**size), dtype=int)
        if order == size:
            return np.identity(2**size, dtype=int)

        mat_a = ReedMullerCode.get_generator(order, size - 1)
        mat_b = ReedMullerCode.get_generator(order - 1, size - 1)
        mat_z = np.zeros_like(mat_b)
        return np.block([[mat_a, mat_a], [mat_z, mat_b]]).astype(int)

    @staticmethod
    def _assert_valid_params(order: int, size: int) -> None:
        if not (size >= 0 and 0 <= order <= size):
            raise ValueError(
                "Reed-Muller code R(r,m) must have m >= 0 and 0 <= r <= m\n"
                + f"Provided: (r,m) = ({order},{size})"
            )


class ReedSolomonCode(ClassicalCode):
    """Classical Reed-Solomon code.

    Source: https://mhostetter.github.io/galois/latest/api/galois.ReedSolomon

    References:

    - https://errorcorrectionzoo.org/c/reed_solomon
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        super().__init__(galois.ReedSolomon(bits, dimension).H)
        self._dimension = dimension


class BCHCode(ClassicalCode):
    """Classical BCH (Bose-Chaudhuri-Hocquenghem) code.

    Source: https://mhostetter.github.io/galois/latest/api/galois.BCH

    References:

    - https://errorcorrectionzoo.org/c/bch
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(
        self, length: int, dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> None:
        field = abstract.resolve_field(field)
        if not BCHCode._is_valid_bch_length(length, field.order):
            raise ValueError(
                f"BCH codes over F_{field.order} are only defined for block lengths"
                f" {field.order}^m - 1 with integer m."
            )
        super().__init__(galois.BCH(length, dimension, field=field).H)
        self._dimension = dimension

    @staticmethod
    def _is_valid_bch_length(length: int, field_order: int) -> bool:
        """Is the given block length valid for a BCH code over a field of the given order?

        BCH codes over ``F_q`` are defined for block lengths ``q**m - 1`` with integer ``m >= 1``.
        """
        power, exponent = length + 1, 0
        while power > 1 and power % field_order == 0:
            power //= field_order
            exponent += 1
        return power == 1 and bool(exponent)


class SimplexCode(ClassicalCode):
    """Classical simplex code: the dual of the Hamming code.

    A simplex code of dimension k over a field of order q has code parameters

        ``[(q**k - 1) / (q - 1), k, q ** (k - 1)]``.

    Its generator matrix has one column for each point of the projective space ``PG(k - 1, q)``,
    that is, one representative of each family of nonzero vectors of ``F_q**k`` that are scalar
    multiples of one another.  Those columns are precisely the parity checks of a Hamming code of
    the same rank, which is what makes a simplex code the dual of a Hamming code.

    Over the binary field the scalar multiples of a nonzero vector are just the vector itself, so
    the block length is ``2**k - 1`` and the code has a cyclic presentation: its parity checks are
    the cyclic shifts of a three-term polynomial, and therefore all have weight 3.  That
    presentation is used here for binary codes, since low-weight parity checks are what make simplex
    codes attractive as the building blocks of a SHYPSCode.  It does not carry over to ``q > 2``,
    where a cyclic code of length ``q**k - 1`` would instead be the ``(q - 1)``-fold repetition of
    the simplex code.

    Over the binary field the automorphism group of this code is the general linear group
    ``GL(k, 2)``.

    References:

    - https://errorcorrectionzoo.org/c/simplex
    - https://errorcorrectionzoo.org/c/hamming
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(self, dim: int, field: int | type[galois.FieldArray] | None = None) -> None:
        field = abstract.resolve_field(field)
        if dim < 2:
            raise ValueError(f"Simplex codes require a dimension of at least 2 (provided: {dim})")

        if field is galois.GF2:
            # the cyclic presentation, whose parity checks all have weight 3
            polynomial = SimplexCode.get_defining_polynomial(dim, field)
            coefficients = polynomial.coefficients(size=2**dim - 1, order="asc")
            matrix = np.array([np.roll(coefficients, shift) for shift in range(len(coefficients))])
            super().__init__(matrix, field=field)
        else:
            # one generator column per point of PG(dim - 1, q), i.e. the dual of a Hamming code
            generator = HammingCode(dim, field).matrix
            super().__init__(ClassicalCode.from_generator(generator, field), field)

        self._dimension = dim
        self._distance = field.order ** (dim - 1)

    @staticmethod
    def get_defining_polynomial(
        dim: int, field: int | type[galois.FieldArray] | None = None
    ) -> galois.Poly:
        """The polynomial defining the cyclic presentation of a simplex code.

        The cyclic code of length ``field.order**dim - 1`` with this check polynomial is a simplex
        code when ``field.order == 2``, and the ``(field.order - 1)``-fold repetition of one
        otherwise.

        Returns a three-term polynomial of the form ``h(x) = 1 + a * x**c + b * x**d``, where

        - the coefficients a and b are elements of a finite field,
        - the exponents c and d are integers, and
        - ``gcd(h(x), x ** (field**dim - 1) - 1)`` is a primitive polynomial of degree dim.

        A dimension of at least 2 is required: a one-dimensional simplex code has block length
        ``field.order - 1 == 1``, for which no nontrivial check polynomial exists.
        """
        field = abstract.resolve_field(field)
        if dim < 2:
            raise ValueError(f"Simplex codes require a dimension of at least 2 (provided: {dim})")

        # first try finding a primitive three-term polynomial of degree dim
        try:
            primitive_polys = galois.primitive_polys(order=field.order, degree=dim, terms=3)
            return next(primitive_polys)
        except StopIteration:
            pass

        # find a suitable polynomial by brute force

        order = field.order**dim - 1
        mod_poly_coefficients = [0] * (order + 1)
        mod_poly_coefficients[0] = -1
        mod_poly_coefficients[-1] = 1
        mod_poly = galois.Poly(mod_poly_coefficients, field=field)

        for aa, bb in itertools.product(range(1, field.order), repeat=2):
            for cc, dd in itertools.combinations(range(1, order + 1), 2):
                coefficients = [0] * (order + 1)
                coefficients[0] = 1
                coefficients[cc] = aa
                coefficients[dd] = bb
                poly = galois.Poly(coefficients[::-1], field=field)
                gcd_poly = galois.gcd(poly, mod_poly)
                if gcd_poly.degree == dim and gcd_poly.is_primitive():
                    return poly

        raise ValueError(
            "Suitable primitive polynomial not found.  This should not be possible."
        )  # pragma: no cover


class TannerCode(ClassicalCode):
    """Classical Tanner code.

    A Tanner code ``T(G,C)`` is constructed from:
    [1] A bipartite "half-regular" graph G.  That is, a graph...

        ... with two sets of nodes, V and W.
        ... in which all nodes in V have degree n.

    [2] A classical code C on n bits.

    For convenience, we make G directed, with edges directed from V to W.  The node sets V and W can
    then be identified, respectively, by the sources and sinks of G.

    The Tanner code ``T(G,C)`` is defined on ``|W|`` bits.  A ``|W|``-bit string x is a code word
    of ``T(G,C)`` iff, for every node v in V, the bits of x incident to v are a code word of C.

    This construction requires an ordering of the edges E(v) adjacent to each vertex v.  This class
    sorts E(v) by the value of the "sort" attribute attached to each edge.  If there is no "sort"
    attribute, its value is treated as the corresponding neighbor of v.

    Tanner codes can similarly be defined on regular (undirected) graphs ``G' = (V',E')`` by
    placing checks on V' and bits on E'.

    Notes:

    - If the subcode C has m checks, its parity matrix has shape ``(m,n)``.
    - The code ``T(G,C)`` has ``|W|`` bits and ``|V|m`` checks.

    References:

    - https://doi.org/10.1109/TIT.1981.1056404
    """

    subgraph: nx.DiGraph
    subcode: ClassicalCode

    def __init__(self, subgraph: nx.Graph, subcode: ClassicalCode) -> None:
        """Construct a classical Tanner code."""
        if not isinstance(subgraph, nx.DiGraph):
            subgraph = TannerCode.as_directed_subgraph(subgraph)

        self.subgraph = subgraph
        self.subcode = subcode
        sources = [node for node in subgraph if subgraph.in_degree(node) == 0]
        sinks = [node for node in subgraph if subgraph.out_degree(node) == 0]
        sink_indices = {sink: idx for idx, sink in enumerate(sorted(sinks))}

        num_bits = len(sinks)
        num_checks = len(sources) * subcode.num_checks
        matrix = np.zeros((num_checks, num_bits), dtype=int)
        for idx, source in enumerate(sorted(sources)):
            checks = range(subcode.num_checks * idx, subcode.num_checks * (idx + 1))
            bits = [sink_indices[sink] for sink in self._get_sorted_neighbors(source)]
            if len(bits) != len(subcode):
                raise ValueError(
                    f"Node {source} has degree {len(bits)}, but the subcode of this Tanner code is"
                    f" defined on {len(subcode)} bits.  Every source node of the subgraph must have"
                    " degree equal to the block length of the subcode."
                )
            matrix[np.ix_(checks, bits)] = subcode.matrix
        super().__init__(matrix, subcode.field)

    def _get_sorted_neighbors(self, node: object) -> Sequence[object]:
        """Sorted neighbors of the given node."""
        return sorted(
            self.subgraph.neighbors(node),
            key=lambda neighbor: self.subgraph[node][neighbor].get("sort", neighbor),
        )

    @staticmethod
    def as_directed_subgraph(subgraph: nx.Graph) -> nx.DiGraph:
        """Convert an undirected graph for a Tanner code into a directed graph for the same code."""
        directed_subgraph = nx.DiGraph()
        for node_a, node_b, edge_data in subgraph.edges(data=True):
            edge = frozenset([node_a, node_b])
            directed_subgraph.add_edge(node_a, edge)
            directed_subgraph.add_edge(node_b, edge)
            if (sort_data := edge_data.get("sort")) is not None:
                directed_subgraph[node_a][edge]["sort"] = sort_data[node_a]
                directed_subgraph[node_b][edge]["sort"] = sort_data[node_b]
        return directed_subgraph
