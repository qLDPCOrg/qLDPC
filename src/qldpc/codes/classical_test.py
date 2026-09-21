"""Unit tests for classical.py.

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

import networkx as nx
import numpy as np
import pytest
from sympy.abc import x, y

from qldpc import codes


def test_basic() -> None:
    """Repetition, ring, and Hamming codes."""
    num_bits = 4
    assert codes.RepetitionCode(num_bits).get_code_params() == (num_bits, 1, num_bits)
    assert codes.RingCode(num_bits).get_code_params() == (num_bits, 1, num_bits)

    # the rank of repetition and Hamming codes is independent of the field
    assert codes.RepetitionCode(3, 2).rank == codes.RepetitionCode(3, 3).rank
    assert codes.HammingCode(3, 2).rank == codes.HammingCode(3, 3).rank


def test_cyclic_codes() -> None:
    """Cyclic codes."""

    # the RingCode is a CyclicCode
    code_a = codes.CyclicCode(5, 1 - x, field=9)
    code_b = codes.RingCode(5, field=9)
    assert np.array_equal(code_a.matrix, code_b.matrix)

    # reproduce Table 2 from arxiv:2511.09683v2
    cyclic_codes = {
        (15, 1 + x + x**4): (15, 4, 8),
        (21, 1 + x + x**5): (21, 5, 10),
        (28, 1 + x**2 + x**4 + x**10): (28, 10, 6),
        (21, 1 + x + x**3 + x**8): (21, 7, 8),
        (30, 1 + x + x**2 + x**7): (30, 6, 14),
        (31, 1 + x + x**2 + x**6 + x**27): (31, 10, 10),
        (31, 1 + x + x**3 + x**9 + x**10): (31, 10, 12),
    }
    for (bits, poly), params in cyclic_codes.items():
        assert codes.CyclicCode(bits, poly).get_code_params() == params

    with pytest.raises(ValueError, match="not a univariate polynomial"):
        codes.CyclicCode(3, 4)

    with pytest.raises(ValueError, match="not a univariate polynomial"):
        codes.CyclicCode(5, x + y)


def test_special_codes() -> None:
    """More complicated classical codes."""
    code: codes.ClassicalCode

    bits, dimension = 3, 2
    assert codes.ReedSolomonCode(bits, dimension).dimension == dimension

    bits, dimension, field = 7, 4, 2
    assert codes.BCHCode(bits, dimension, field).dimension == dimension
    with pytest.raises(ValueError, match=rf"block lengths {field}\^m - 1"):
        codes.BCHCode(bits - 1, dimension, field)

    order, size, field = 1, 3, 2
    code = codes.ReedMullerCode(order, size, field)
    assert (code.order, code.size) == (order, size)
    assert ~code == codes.ReedMullerCode(size - order - 1, size, field)
    assert code.dimension == len(code) - np.linalg.matrix_rank(code.matrix)

    with pytest.raises(ValueError, match="0 <= r <= m"):
        codes.ReedMullerCode(-1, 0)

    # the Hamming code can be recovered by puncturing the extended Hamming code
    assert codes.ClassicalCode.equiv(
        codes.HammingCode(4), codes.ExtendedHammingCode(4).punctured([0])
    )

    # Hamming and extended Hamming codes report the parameters their parity checks bear out
    for size in [2, 3, 4]:
        for code, params in [
            (codes.HammingCode(size), (2**size - 1, 2**size - 1 - size, 3)),
            (codes.ExtendedHammingCode(size), (2**size, 2**size - 1 - size, 4)),
        ]:
            assert code.get_code_params() == params
            assert codes.ClassicalCode(code.matrix).get_code_params() == params

    # classical simplex codes.  Rebuilding a code from its parity check matrix alone carries none of
    # the parameters that its constructor caches, so the rebuilt code has to compute them.
    for dimension in [2, 3, 8]:
        code = codes.SimplexCode(dimension)
        params = (2**dimension - 1, dimension, 2 ** (dimension - 1))
        assert code.get_code_params() == params
        assert codes.ClassicalCode(code.matrix).get_code_params() == params

    # the Golay code is a [23, 12, 7] code with minimum-weight (weight-8) parity checks
    code = codes.GolayCode()
    assert code.get_code_params() == (23, 12, 7)
    assert codes.ClassicalCode(code.matrix).get_code_params() == (23, 12, 7)
    assert set(code.matrix.view(np.ndarray).sum(axis=1)) == {8}


def test_simplex_codes_over_fields() -> None:
    """A simplex code is the dual of a Hamming code: [(q**k - 1)/(q - 1), k, q**(k - 1)].

    Over F_2 the code has a cyclic presentation whose parity checks all have weight 3, which makes
    simplex codes useful as the building blocks of a SHYPSCode, so binary codes use it.  A cyclic
    code of length q**k - 1 over a larger field would instead be the (q - 1)-fold repetition of a
    simplex code, with (q - 1) times the block length and distance.
    """
    for field in [2, 3, 4, 5]:
        for dim in [2, 3]:
            code = codes.SimplexCode(dim, field)
            params = ((field**dim - 1) // (field - 1), dim, field ** (dim - 1))

            # the parameters the constructor reports, and the same parameters computed by a code
            # rebuilt from the parity check matrix alone, which caches nothing
            assert code.get_code_params() == params
            assert codes.ClassicalCode(code.matrix).get_code_params() == params

            # over a larger field the code is built as, and equals, the dual of a Hamming code;
            # the binary cyclic presentation is the same code only up to a permutation of bits
            if field > 2:
                assert codes.ClassicalCode.equiv(
                    codes.SimplexCode(dim, field), ~codes.HammingCode(dim, field)
                )

    # the binary presentation has weight-3 parity checks, which SHYPS codes inherit
    for dim in [2, 3, 4, 5]:
        matrix = codes.SimplexCode(dim).matrix.view(np.ndarray)
        assert np.all(np.count_nonzero(matrix, axis=1) == 3)


def test_reed_muller_order_zero() -> None:
    """The order-zero Reed-Muller code RM(0, m) is the [2**m, 1, 2**m] repetition code."""
    for size in range(5):
        generator = codes.ReedMullerCode.get_generator(0, size)
        assert np.asarray(generator).ndim == 2  # a single row, not a flat vector
        assert codes.ReedMullerCode(0, size).get_code_params() == (2**size, 1, 2**size)

    # the documented duality RM(r, m)^perp == RM(m - r - 1, m) reaches order zero at r == m - 1
    assert codes.ClassicalCode.equiv(~codes.ReedMullerCode(2, 3), codes.ReedMullerCode(0, 3))


def test_degenerate_code_sizes() -> None:
    """Code families reject parameters for which they are not defined."""
    for size in [-1, 0, 1]:
        with pytest.raises(ValueError, match="rank of at least 2"):
            codes.HammingCode(size)
        with pytest.raises(ValueError, match="rank of at least 2"):
            codes.ExtendedHammingCode(size)
        with pytest.raises(ValueError, match="dimension of at least 2"):
            codes.SimplexCode(size)
        with pytest.raises(ValueError, match="dimension of at least 2"):
            codes.SimplexCode.get_defining_polynomial(size)


def test_bch_block_lengths() -> None:
    """A BCH block length is valid exactly when it is field_order**m - 1 for an integer m >= 1."""
    # valid: q**m - 1.  Digits of these lengths are non-decimal in base q > 10, which a
    # string-based check on the base-q representation would reject.
    for length, order in [(1, 2), (15, 2), (8, 3), (120, 11), (168, 13), (16, 17)]:
        assert codes.BCHCode._is_valid_bch_length(length, order)

    # invalid: negative, or not one less than a power of the field order, or m == 0
    for length, order in [(-4, 2), (-1, 2), (0, 2), (6, 2), (14, 2), (7, 3), (119, 11)]:
        assert not codes.BCHCode._is_valid_bch_length(length, order)

    # a valid length over a field of order greater than 10 builds a code of the asked-for dimension,
    # which its parity checks have to agree with
    code = codes.BCHCode(120, 100, field=11)
    assert code.dimension == len(code) - code.rank == 100


def test_tanner_code_preserves_input_graph() -> None:
    """Building a Tanner code leaves the given graph, and hence the resulting code, unchanged."""
    # a subcode whose automorphism group cannot absorb a relabeling of the subgraph edges
    subcode = codes.ClassicalCode([[0, 0, 1], [1, 1, 0]])
    subgraph = nx.complete_graph(4)
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: -node_b, node_b: -node_a}

    code = codes.TannerCode(subgraph, subcode)
    assert all("sort" in subgraph[node_a][node_b] for node_a, node_b in subgraph.edges)

    # a second code built from the same graph is the same code
    assert codes.ClassicalCode.equiv(code, codes.TannerCode(subgraph, subcode))


def test_tanner_code_requires_matching_degree() -> None:
    """A Tanner code requires every source node to have degree equal to the subcode block length."""
    subgraph = nx.Graph([(0, 3), (1, 3), (0, 4)])
    with pytest.raises(ValueError, match="but the subcode of this Tanner code"):
        codes.TannerCode(subgraph, codes.RepetitionCode(3))


def test_tanner_code() -> None:
    """Classical Tanner codes on random regular graphs."""
    subcode = codes.ClassicalCode.random(5, 3)
    subgraph = nx.random_regular_graph(subcode.num_bits, subcode.num_bits * 2 + 2)

    tag = "sort_label"
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: tag, node_b: tag}

    code = codes.TannerCode(subgraph, subcode)
    assert code.num_bits == subgraph.number_of_edges()
    assert code.num_checks == subgraph.number_of_nodes() * code.subcode.num_checks
    assert all(code.subgraph.get_edge_data(*edge)["sort"] == tag for edge in code.subgraph.edges)
