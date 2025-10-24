"""Unit tests for classical.py

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

from qldpc import codes


def test_basic() -> None:
    """Repetition, ring, and Hamming codes."""
    num_bits = 4
    assert codes.RepetitionCode(num_bits).get_code_params() == (num_bits, 1, num_bits)
    assert codes.RingCode(num_bits).get_code_params() == (num_bits, 1, num_bits)

    # the rank of repetition and Hamming codes is independent of the field
    assert codes.RepetitionCode(3, 2).rank == codes.RepetitionCode(3, 3).rank
    assert codes.HammingCode(3, 2).rank == codes.HammingCode(3, 3).rank


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

    # classical simplex codes
    for dimension in [2, 3, 8]:
        code = codes.SimplexCode(dimension)
        params = (2**dimension - 1, dimension, 2 ** (dimension - 1))
        assert code.get_code_params() == params
        code._dimension = None
        code.forget_distance()
        assert code.get_code_params() == params


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
