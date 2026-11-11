# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantum_tanner.py."""

from __future__ import annotations

import pathlib
import random
import unittest.mock

import networkx as nx
import numpy as np
import pytest
import sympy

from qldpc import abstract, codes
from qldpc.conftest import run_python


def test_quantum_tanner(pytestconfig: pytest.Config, tmp_path: pathlib.Path) -> None:
    """Quantum Tanner code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    # random quantum Tanner code
    group = abstract.CyclicGroup(12)
    subcode = codes.RepetitionCode(4, field=3)
    code = codes.QTCode.random(group, subcode, seed=np.random.randint(2**31))

    # assert that subgraphs have the right number of nodes, edges, and node degrees
    subgraph_x, subgraph_z = codes.QTCode.get_subgraphs(code.complex)
    group_size = code.complex.graph.number_of_nodes()
    num_faces = code.num_qudits
    for graph in [subgraph_x, subgraph_z]:
        assert graph.number_of_nodes() == num_faces + group_size / 2
        assert graph.number_of_edges() == num_faces * 2
        sources = [node for node in graph.nodes if graph.in_degree(node) == 0]
        assert {graph.out_degree(node) for node in sources} == {subcode.num_bits**2}

    # raise error if the generating data is underspecified
    subset_a = code.complex.subset_a
    subset_b = group.random_symmetric_subset(len(subset_a) - 1)
    subcode_a = codes.RepetitionCode(len(subset_a))
    with pytest.raises(ValueError, match="Underspecified generating data"):
        codes.QTCode(subset_a, subset_b, subcode_a)

    # raise error if seed codes are over different fields
    subcode_b = codes.RepetitionCode(2, field=3)
    with pytest.raises(ValueError, match="different fields"):
        codes.QTCode(subset_a, subset_a, subcode_a, subcode_b)

    # saving and loading a QTCode, creating a missing directory along the way
    path = tmp_path / "subdirectory" / "code.txt"
    code.save(str(path), "header")
    assert codes.QTCode.load(str(path)) == code
    with pytest.raises(ValueError, match="Path does not exist"):
        codes.QTCode.load(str(tmp_path / "missing.txt"))

    # saved files do not depend on the iteration order of subsets, which varies between sessions
    saved_files = []
    for reverse in [False, True]:
        ordered_a = sorted(code.complex.subset_a, reverse=reverse)
        ordered_b = sorted(code.complex.subset_b, reverse=reverse)
        with (
            unittest.mock.patch.object(code.complex, "subset_a", ordered_a),
            unittest.mock.patch.object(code.complex, "subset_b", ordered_b),
        ):
            code.save(str(path))
        saved_files.append(path.read_text())
    assert saved_files[0] == saved_files[1]

    # codes whose saved data has empty rows or sections, such as members of the trivial group or
    # seed codes without parity checks, load back too, even with headers that mimic section names
    generator = abstract.CyclicGroup(2).generators[0]
    for saved_code in [
        codes.QTCode.random(abstract.TrivialGroup(), codes.ClassicalCode([[1]])),
        codes.QTCode([generator], [generator], np.zeros((0, 1), dtype=int), [[1]]),
    ]:
        saved_code.save(str(path), "subset_a:\nsubset_b:")
        assert codes.QTCode.load(str(path)) == saved_code

    # loading fails clearly if a section is missing
    path.write_text("# subset_a:\n")
    with pytest.raises(ValueError, match="Saved code has no"):
        codes.QTCode.load(str(path))

    # Random subsets can exclude the identity element.  Without that exclusion, about a third of
    # random symmetric subsets of size 3 in this group contain it, so several seeds would catch it.
    symmetric_group = abstract.SymmetricGroup(3)
    first_seed = np.random.randint(2**31)
    for seed in range(first_seed, first_seed + 10):
        code = codes.QTCode.random(
            symmetric_group, codes.HammingCode(2), exclude_identity=True, seed=seed
        )
        assert not any(
            member.is_Identity for member in code.complex.subset_a | code.complex.subset_b
        )


def test_quantum_tanner_generating_data() -> None:
    """Quantum Tanner codes validate and compare their generating data."""
    shift = abstract.CyclicGroup(6).generators[0]
    members = [shift, ~shift]
    subcode = codes.RepetitionCode(2)
    code = codes.QTCode(members, members, subcode)

    # repeated members are merged before the sizes of the subsets are compared
    assert codes.QTCode(members * 2, members, subcode) == code

    # codes are equal if their members are equal after adding fixed points to the end of them
    padded_members = [
        abstract.GroupMember(member.array_form, size=member.size + 1) for member in members
    ]
    padded_code = codes.QTCode(padded_members, padded_members, subcode)
    assert padded_code.complex.subset_a != code.complex.subset_a
    assert padded_code == code
    assert code == padded_code
    assert code != code.complex

    # raise error if a seed code does not have one bit per distinct member of its subset, even if
    # the product of the block lengths of the seed codes is right
    other_members = [shift**2, shift**3, ~(shift**2)]
    for code_a, code_b, wrong_code in [
        (codes.RepetitionCode(3), codes.RepetitionCode(3), "C_A"),
        (codes.RepetitionCode(2), codes.RepetitionCode(2), "C_B"),
        (codes.RepetitionCode(3), codes.RepetitionCode(2), "C_[AB]"),
    ]:
        with pytest.raises(ValueError, match=f"seed code {wrong_code} has block length"):
            codes.QTCode(members, other_members, code_a, code_b)

    # one generating subset may be empty, which leaves a complex without faces, but not both
    no_bits = np.zeros((0, 0), dtype=int)
    assert codes.QTCode([], members, no_bits, subcode).num_qudits == 0
    assert codes.QTCode(members, [], subcode, no_bits).num_qudits == 0
    with pytest.raises(ValueError, match="at least one generating member"):
        codes.QTCode([], [], no_bits)


def test_quantum_tanner_nonabelian_faces() -> None:
    """QTCode realizes the defining face incidence over a noncommutative group."""
    group = abstract.QuaternionGroup()
    generator_i, generator_j = group.generators
    assert generator_i * generator_j != generator_j * generator_i

    code = codes.QTCode(
        [generator_i, ~generator_i],
        [generator_j, ~generator_j],
        codes.RepetitionCode(2),
        bipartite=False,
    )
    cayplex = code.complex
    subgraph_x, subgraph_z = codes.QTCode.get_subgraphs(cayplex)

    colors = nx.bipartite.color(cayplex.graph)
    member = next(iter(cayplex.cover_subset_a))
    identity = member * ~member
    sources_x = {element for element, color in colors.items() if color == colors[identity]}

    expected_x = set()
    expected_z = set()
    for element in sources_x:
        for subset_a in cayplex.cover_subset_a:
            for subset_b in cayplex.cover_subset_b:
                neighbor = subset_a * element
                face = frozenset([element, neighbor, element * subset_b, neighbor * subset_b])
                expected_x.add((element, face, (subset_a, subset_b)))
                expected_z.add((neighbor, face, (~subset_a, subset_b)))

    def labelled_edges(graph: nx.DiGraph) -> set[tuple[object, object, object]]:
        """Edges together with the generator pair that defines each face."""
        return {
            (source, face, edge_data["sort"]) for source, face, edge_data in graph.edges(data=True)
        }

    assert labelled_edges(subgraph_x) == expected_x
    assert labelled_edges(subgraph_z) == expected_z

    def matrix_supports(matrix: np.ndarray, graph: nx.DiGraph) -> list[frozenset[object]]:
        """Matrix-row supports expressed using the graph's face nodes."""
        faces = sorted(
            (node for node in graph if graph.out_degree(node) == 0),
            key=lambda face: graph.nodes[face]["sort"],
        )
        return [frozenset(faces[col] for col in np.flatnonzero(row)) for row in matrix]

    # the faces of both subgraphs are numbered in one order, from zero
    face_labels_x, face_labels_z = [
        {node: label for node, label in graph.nodes(data="sort") if graph.out_degree(node) == 0}
        for graph in [subgraph_x, subgraph_z]
    ]
    assert face_labels_x == face_labels_z
    assert sorted(face_labels_x.values()) == list(range(code.num_qudits))

    for matrix, graph in [(code.matrix_x, subgraph_x), (code.matrix_z, subgraph_z)]:
        expected = {
            frozenset(graph.successors(source)) for source in graph if graph.in_degree(source) == 0
        }
        actual = matrix_supports(matrix, graph)
        assert len(actual) == len(expected)
        assert set(actual) == expected

    assert not np.any(code.matrix_x @ code.matrix_z.T)


def test_quantum_tanner_qudit_order() -> None:
    """Quantum Tanner codes have a fixed qudit order.

    A code that is saved and loaded is rebuilt from its generating data, so changing the order of
    its qudits would invalidate anything that refers to them by index.
    """

    def get_supports(code: codes.QTCode) -> list[tuple[int, ...]]:
        """Supports of the X-type parity checks of a code."""
        return [tuple(np.flatnonzero(row)) for row in code.matrix_x.view(np.ndarray)]

    shift = abstract.CyclicGroup(3).generators[0]
    code = codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2))
    assert get_supports(code) == [
        (0, 1, 2, 3),
        (1, 2, 4, 5),
        (5, 6, 7, 8),
        (3, 6, 7, 9),
        (4, 9, 10, 11),
        (0, 8, 10, 11),
    ]

    # a bipartite complex of the symmetric group S_3, generated by 3-cycles and by transpositions
    cycle = abstract.GroupMember([1, 2, 0])
    swaps = [abstract.GroupMember([1, 0, 2]), abstract.GroupMember([2, 1, 0])]
    code = codes.QTCode([cycle, ~cycle], swaps, codes.RepetitionCode(2), bipartite=True)
    assert get_supports(code) == [
        (0, 1, 2, 3),
        (0, 3, 4, 5),
        (1, 6, 7, 8),
        (5, 6, 9, 10),
        (4, 7, 8, 11),
        (2, 9, 10, 11),
    ]


def test_random_quantum_tanner_code_is_reproducible() -> None:
    """A seed fixes both of the random subsets that define a random quantum Tanner code."""
    group = abstract.CyclicGroup(8)
    subcode = codes.RepetitionCode(2)

    def matrix_for(seed: int | None = None, one_subset: bool = False) -> bytes:
        code = codes.QTCode.random(group, subcode, seed=seed, one_subset=one_subset)
        return np.asarray(code.matrix).tobytes()

    # the same seed gives the same code every time, and different seeds give different codes
    assert len({matrix_for(seed=7) for _ in range(4)}) == 1
    assert matrix_for(seed=7) != matrix_for(seed=8)

    # a seed of any magnitude is accepted
    assert matrix_for(seed=2**40) == matrix_for(seed=2**40)

    # reusing one subset for both sides is likewise reproducible
    assert len({matrix_for(seed=7, one_subset=True) for _ in range(3)}) == 1

    # without a seed the code is still drawn at random, from SymPy's global generator.  Replacing
    # that generator with a seeded one for this check keeps it from depending on chance.
    with unittest.mock.patch.object(sympy.core.random, "rng", random.Random(0)):
        assert len({matrix_for() for _ in range(4)}) > 1

    # the code is also independent of the hash seed, which sets the iteration order of the sets of
    # group members that the construction is built from.  Child interpreters with fixed hash seeds
    # must reproduce the code built here, under this interpreter's (randomized) hash seed.
    script = (
        "import numpy as np;"
        "from qldpc import abstract, codes;"
        "code = codes.QTCode.random(abstract.CyclicGroup(8), codes.RepetitionCode(2), seed=7);"
        "print(np.asarray(code.matrix).tobytes().hex())"
    )
    matrices = {run_python(script, env={"PYTHONHASHSEED": hash_seed}) for hash_seed in ["0", "1"]}
    assert matrices == {matrix_for(seed=7).hex() + "\n"}


def test_toric_tanner_code(size: int = 4) -> None:
    """Rotated toric code as a quantum Tanner code."""
    group = abstract.Group.product(abstract.CyclicGroup(size), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    subcode_a = codes.RepetitionCode(2)
    code = codes.QTCode(subset_a, subset_b, subcode_a, bipartite=False)
    assert code.get_code_params() == (size**2, 2, size)
    assert code.get_weight() == 4
