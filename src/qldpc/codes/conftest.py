# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures and helpers for tests of codes."""

from __future__ import annotations

import contextlib
import unittest.mock
from collections.abc import Callable, Iterator

import networkx as nx
import pytest

from qldpc import codes
from qldpc.codes.distance import _tuning
from qldpc.objects import Pauli

# Changes fields of the active search tuning until the end of a test (see the tune fixture).
Tune = Callable[..., _tuning.SearchTuning]


@pytest.fixture
def tune() -> Iterator[Tune]:
    """Change fields of the active search tuning, as ``use_tuning`` does, until the test ends.

    Fixtures and tests that change the tuning share this one stack of changes, so that every change
    is undone in reverse order, whatever order pytest tears fixtures down in.
    """
    with contextlib.ExitStack() as stack:
        yield lambda **changes: stack.enter_context(_tuning.use_tuning(**changes))
    # A change that outlives its test, as one made by tune within a use_tuning block would, fails.
    assert _tuning.get_tuning() == _tuning.DEFAULT_TUNING


def assert_valid_subgraphs(code: codes.QuditCode) -> None:
    """Every edge of the Tanner graph occurs in exactly one syndrome measurement subgraph."""
    edges = [edge for graph in code.get_syndrome_subgraphs() for edge in graph.edges]
    assert len(edges) == len(set(edges))
    assert set(edges) == set(code.graph.edges)


def tanner_graphs_are_isomorphic(code_a: codes.QuditCode, code_b: codes.QuditCode) -> bool:
    """Are two codes equal up to the order of their qudits and parity checks?

    The codes are compared by their Pauli-labelled Tanner graphs, keeping qudits and checks apart.
    """

    def tagged_graph(code: codes.QuditCode) -> nx.DiGraph:
        """Copy a Tanner graph with its data/check partition available to the matcher."""
        graph = code.graph.copy()
        nx.set_node_attributes(graph, {node: node.is_data for node in graph}, "is_data")
        return graph

    return nx.is_isomorphic(
        tagged_graph(code_a),
        tagged_graph(code_b),
        node_match=nx.algorithms.isomorphism.categorical_node_match(["is_data"], [None]),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match([Pauli], [None]),
    )


def assert_cached_parameters_are_genuine(
    expected: list[tuple[codes.CSSCode, tuple[int, int, int]]],
) -> None:
    """Codes that cache their parameters agree with a computation that ignores the cache.

    Several constructors assign a dimension and distance taken from the literature rather than
    computing them, and the public getters then return those values verbatim.  Rebuild each code
    from its parity check matrices alone, which carry no cached parameters at all, and compare what
    the code reports against what the rebuilt code computes, so that a constant which disagrees
    with the code it describes cannot pass unnoticed.

    Args:
        expected: Pairs of a code and its expected ``(n, k, d)`` parameters.
    """
    for code, params in expected:
        rebuilt = codes.CSSCode(code.matrix_x, code.matrix_z)
        with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
            assert code.get_code_params() == params
            assert rebuilt.get_code_params() == params

            # compare the X and Z distances separately, since the parameters above report only the
            # smaller of the two
            assert code.get_distance(Pauli.X) == rebuilt.get_distance(Pauli.X)
            assert code.get_distance(Pauli.Z) == rebuilt.get_distance(Pauli.Z)
