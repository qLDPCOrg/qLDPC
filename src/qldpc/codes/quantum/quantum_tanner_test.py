# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantum_tanner.py."""

from __future__ import annotations

import itertools
import pathlib
import random
import unittest.mock
from typing import Literal

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
import sympy

from qldpc import abstract, codes
from qldpc.codes.quantum import quantum_tanner
from qldpc.conftest import run_python
from qldpc.objects import Pauli, PauliXZ

LocalChecks = Literal["min_weight", "from_codes"]
LOCAL_CHECKS: tuple[LocalChecks, ...] = ("min_weight", "from_codes")


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

    # a code in which visiting the least node first and visiting nodes in the order in which they
    # are found give different orders of faces
    shift = abstract.CyclicGroup(6).generators[0]
    code = codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2))
    assert get_supports(code) == [
        (0, 1, 2, 3),
        (1, 2, 4, 5),
        (4, 6, 7, 8),
        (0, 7, 8, 9),
        (5, 9, 10, 11),
        (3, 6, 10, 11),
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


def get_min_weights(space: galois.FieldArray) -> list[int]:
    """The weights of a minimum-weight basis of the row space of a matrix, by brute force.

    A minimum-weight basis has exactly r_w - r_(w-1) vectors of weight w, where r_w is the dimension
    of the span of all vectors of the row space with weight at most w.
    """
    field = type(space)
    basis = space.row_space()
    coefficients = field(list(itertools.product(range(field.order), repeat=len(basis))))
    words = coefficients.reshape(field.order ** len(basis), len(basis)) @ basis
    word_weights = np.count_nonzero(words.view(np.ndarray), axis=1)
    weights: list[int] = []
    for weight in range(1, space.shape[1] + 1):
        light_words = words[(word_weights > 0) & (word_weights <= weight)]
        light_rank = np.linalg.matrix_rank(light_words) if len(light_words) else 0
        weights += [weight] * (light_rank - len(weights))
    return weights


def get_sorted_rows(matrix: npt.NDArray[np.int_]) -> list[tuple[int, ...]]:
    """The rows of a matrix, in sorted order, with their multiplicities."""
    return sorted(map(tuple, matrix.tolist()))


def assert_same_row_space(matrix: galois.FieldArray, other: galois.FieldArray) -> None:
    """Assert that two matrices over a field have the same row space."""
    rank = np.linalg.matrix_rank(matrix) if matrix.size else 0
    other_rank = np.linalg.matrix_rank(other) if other.size else 0
    stacked = np.vstack([matrix, other]).view(type(matrix))
    assert rank == other_rank == (np.linalg.matrix_rank(stacked) if stacked.size else 0)


def test_quantum_tanner_local_checks(pytestconfig: pytest.Config) -> None:
    """The choices of local checks span the local check spaces of the local views of vertices."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    # random codes over abelian and nonabelian groups, on both complexes, and over GF(2) and GF(3)
    cases: list[
        tuple[abstract.Group, codes.ClassicalCode, codes.ClassicalCode | None, bool, int | None]
    ] = [
        (abstract.CyclicGroup(6), codes.HammingCode(2), None, False, None),
        (
            abstract.SymmetricGroup(3),
            codes.ClassicalCode([[1, 1, 0, 1], [0, 1, 1, 1]]),
            None,
            False,
            None,
        ),
        (
            abstract.CyclicGroup(4),
            codes.ClassicalCode([[1, 2, 0], [0, 1, 1]], field=3),
            None,
            False,
            None,
        ),
        # a bipartite complex needs subsets that satisfy Total No Conjugacy, as those of seed 36 do
        (
            abstract.DihedralGroup(5),
            codes.HammingCode(2),
            codes.ClassicalCode([[1, 1, 1, 0]]),
            True,
            36,
        ),
    ]
    for group, code_a, code_b, bipartite, seed in cases:
        code = codes.QTCode.random(
            group,
            code_a,
            code_b,
            bipartite=bipartite,
            seed=int(rng.integers(2**31)) if seed is None else seed,
        )
        code_a, code_b = code.code_a, code.code_b

        # give C_A a generator matrix with redundant rows, which "from_codes" uses as given
        generator = code_a.generator
        code_a = codes.ClassicalCode(code_a).set_generator(np.vstack([generator, generator[:1]]))

        # the faces and their qudits, and the members of the subsets in the cover of the complex
        subgraphs = codes.QTCode.get_subgraphs(code.complex)
        face_qudits = dict(subgraphs[0].nodes(data="sort"))
        members_a = sorted(code.complex.cover_subset_a)
        members_b = sorted(code.complex.cover_subset_b)

        # the members of the subsets label local coordinates in the order of their covers
        size = code.members_a[0].size
        for cover_members, members in [(members_a, code.members_a), (members_b, code.members_b)]:
            assert [member.array_form[:size] for member in cover_members] == [
                member.array_form for member in members
            ]

        for local_checks in LOCAL_CHECKS:
            new_code = codes.QTCode(
                code.complex.subset_a,
                code.complex.subset_b,
                code_a,
                code_b,
                bipartite=bipartite,
                local_checks=local_checks,
            )
            assert len(new_code) == len(code)
            assert new_code.dimension == code.dimension
            assert_same_row_space(new_code.matrix_x, code.matrix_x)
            assert_same_row_space(new_code.matrix_z, code.matrix_z)
            assert not np.any(new_code.matrix_x @ new_code.matrix_z.T)

            sectors: list[
                tuple[PauliXZ, nx.DiGraph, galois.FieldArray, list[galois.FieldArray]]
            ] = [
                (Pauli.X, subgraphs[0], new_code.matrix_x, [code_a.generator, code_b.generator]),
                (Pauli.Z, subgraphs[1], new_code.matrix_z, [code_a.matrix, code_b.matrix]),
            ]
            for pauli, subgraph, matrix, local_spaces in sectors:
                # the local view of each vertex v labels face f(v,a,b) by its members a and b
                sources = sorted(node for node in subgraph if subgraph.out_degree(node))
                local_views = new_code.get_local_views(pauli)
                assert local_views.shape == (len(sources), len(members_a), len(members_b))
                assert not local_views.flags.writeable
                for source, local_view in zip(sources, local_views):
                    for (row, member_a), (col, member_b) in itertools.product(
                        enumerate(members_a), enumerate(members_b)
                    ):
                        corners = [source, member_a * source, source * member_b]
                        face = frozenset([*corners, member_a * source * member_b])
                        assert local_view[row, col] == face_qudits[face]

                # the checks of each vertex lie in its local view, and span its local check space
                local_space = np.kron(*local_spaces).view(new_code.field)
                num_checks = len(matrix) // len(local_views)
                for vertex, local_view in enumerate(local_views):
                    checks = matrix[vertex * num_checks : (vertex + 1) * num_checks]
                    outside = np.setdiff1d(np.arange(len(new_code)), local_view.ravel())
                    assert not np.any(checks[:, outside])
                    local_checks_at_vertex = checks[:, local_view.ravel()]
                    if local_checks == "from_codes":
                        assert np.array_equal(local_checks_at_vertex, local_space)
                    else:
                        assert_same_row_space(local_checks_at_vertex, local_space)
                    if local_checks == "min_weight":
                        weights = np.count_nonzero(local_checks_at_vertex.view(np.ndarray), axis=1)
                        assert weights.tolist() == get_min_weights(local_space)


def get_c4_a4_code(local_checks: LocalChecks = "min_weight") -> codes.QTCode:
    """A [[144, 8, 12]] code on a bipartite complex of C4 x A4.

    The code was found in a search for small quantum Tanner codes.
    """
    subset_a = [[1, 2, 3, 0, 4, 7, 5, 6], [3, 0, 1, 2, 4, 6, 7, 5]]
    subset_b = [
        [1, 2, 3, 0, 5, 4, 7, 6],
        [1, 2, 3, 0, 6, 7, 4, 5],
        [1, 2, 3, 0, 7, 6, 5, 4],
        [3, 0, 1, 2, 5, 4, 7, 6],
        [3, 0, 1, 2, 6, 7, 4, 5],
        [3, 0, 1, 2, 7, 6, 5, 4],
    ]
    return codes.QTCode(
        [abstract.GroupMember(member) for member in subset_a],
        [abstract.GroupMember(member) for member in subset_b],
        [[1, 1]],
        [[1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 0, 0]],
        bipartite=True,
        local_checks=local_checks,
    )


def test_quantum_tanner_min_weight_regression() -> None:
    """Minimum-weight local checks give a [[144, 8, 12]] code checks of weight 6."""
    code = get_c4_a4_code()
    assert (len(code), code.dimension) == (144, 8)
    assert code.get_weight() == 6

    # the checks are deterministic, as a saved code relies on: pin those of the first vertices
    expected_checks: dict[PauliXZ, list[list[int]]] = {
        Pauli.X: [
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        ],
        Pauli.Z: [
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        ],
    }
    for pauli, matrix in zip(expected_checks, [code.matrix_x, code.matrix_z]):
        local_view = code.get_local_views(pauli)[0]
        assert np.array_equal(matrix[:3][:, local_view.ravel()], expected_checks[pauli])


def test_quantum_tanner_default_dual_code() -> None:
    """The default C_B is ~C_A, whose generator matrix is the parity check matrix of C_A."""
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2, shift**4]
    matrix_a = [[1, 1, 0, 1], [0, 1, 1, 1]]  # not in reduced row echelon form
    code = codes.QTCode(subset, subset, matrix_a, local_checks="from_codes")
    assert np.array_equal(code.code_b.generator, matrix_a)
    local_views = code.get_local_views(Pauli.X)
    num_checks = len(code.matrix_x) // len(local_views)
    local_checks = code.matrix_x[:num_checks][:, local_views[0].ravel()]
    assert np.array_equal(local_checks, np.kron(code.code_a.generator, matrix_a))


def test_quantum_tanner_default_local_checks(tmp_path: pathlib.Path) -> None:
    """QTCodes choose minimum-weight local checks by default, also for files of earlier releases."""
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2, shift**4]
    code_a = codes.ClassicalCode([[1, 0, 0, 0], [0, 1, 1, 0]])
    code_a.set_generator([[0, 1, 1, 1], [0, 0, 0, 1]])  # not a minimum-weight basis
    code = codes.QTCode(subset, subset, code_a)
    from_codes_code = codes.QTCode(subset, subset, code_a, local_checks="from_codes")
    assert code == codes.QTCode(subset, subset, code_a, local_checks="min_weight")
    assert code != from_codes_code

    subcode_x, subcode_z = codes.QTCode.get_subcodes(code.complex, code.code_a, code.code_b)
    assert np.array_equal(subcode_x.matrix, code.matrix_x)
    assert np.array_equal(subcode_z.matrix, code.matrix_z)
    random_code = codes.QTCode.random(abstract.CyclicGroup(4), code_a, seed=0)
    assert random_code == codes.QTCode.random(
        abstract.CyclicGroup(4), code_a, seed=0, local_checks="min_weight"
    )

    # Files saved by earlier releases record no choice of local checks and no generator matrices of
    # the local codes, and load with minimum-weight local checks.
    path = tmp_path / "code.txt"
    from_codes_code.save(str(path))
    lines = path.read_text().splitlines()
    first, last = lines.index("# local checks: from_codes"), lines.index("# base field: 2")
    path.write_text("\n".join(lines[:first] + lines[last:]) + "\n")
    loaded_code = codes.QTCode.load(str(path))
    loaded_codes = (loaded_code.code_a, loaded_code.code_b)
    assert loaded_code == codes.QTCode(subset, subset, *loaded_codes, local_checks="min_weight")
    assert loaded_code != codes.QTCode(subset, subset, *loaded_codes, local_checks="from_codes")
    assert_same_row_space(loaded_code.matrix_x, code.matrix_x)
    assert_same_row_space(loaded_code.matrix_z, code.matrix_z)

    for local_checks in ["row_reduced", None]:
        with pytest.raises(ValueError, match="Unrecognized choice of local checks"):
            codes.QTCode(subset, subset, code_a, local_checks=local_checks)  # type: ignore[arg-type]


def test_quantum_tanner_saved_local_checks(tmp_path: pathlib.Path) -> None:
    """Saved codes keep their choice of local checks, and the generator matrices of local codes."""
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2, shift**4]
    code_a = codes.ClassicalCode([[1, 0, 0, 0], [0, 1, 1, 0]])
    code_a.set_generator([[0, 1, 1, 1], [0, 0, 0, 1]])
    code_b = codes.ClassicalCode([[1, 1, 0, 1]])
    code_b.set_generator([[1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
    path = tmp_path / "code.txt"
    for local_checks in LOCAL_CHECKS:
        code = codes.QTCode(subset, subset, code_a, code_b, local_checks=local_checks)
        code.save(str(path))
        loaded_code = codes.QTCode.load(str(path))
        assert loaded_code == code
        assert np.array_equal(loaded_code.code_a.generator, code_a.generator)
        assert np.array_equal(loaded_code.code_b.generator, code_b.generator)

    # Released versions of QTCode.load up to 0.4.1 read the lines of integers between comment lines
    # as the subsets and local codes, and the field and complex from the last two lines.  The lines
    # that record the choice of local checks are comment lines, so these versions ignore them.
    lines = path.read_text().splitlines()
    assert lines[-2:] == ["# base field: 2", "# bipartite: False"]
    first, last = lines.index("# local checks: from_codes"), lines.index("# base field: 2")
    assert all(line.startswith("#") for line in lines[first:last])

    # the choice of local checks given to load takes precedence over the saved one
    min_weight_code = codes.QTCode(subset, subset, code_a, code_b, local_checks="min_weight")
    assert codes.QTCode.load(str(path), local_checks="min_weight") == min_weight_code

    # loading fails clearly if one of the sections that choose the parity checks is missing
    lines = path.read_text().splitlines()
    lines = [line for line in lines if not line.startswith("# code_b.generator:")]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="no '# code_b\\.generator:' line"):
        codes.QTCode.load(str(path))

    # QTCodes with different checks are different, even with the same generating data
    toric_codes = [
        codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2), local_checks=choice)
        for choice in LOCAL_CHECKS
    ]
    assert toric_codes[0] == toric_codes[1]
    assert min_weight_code != codes.QTCode(
        subset, subset, code_a, code_b, local_checks="from_codes"
    )


def test_quantum_tanner_min_weight_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum-weight local checks need local codes and duals with few enough words."""
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2]
    code_a = codes.HammingCode(2)  # the [3, 1] repetition code, whose dual has 4 words
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_MIN_WEIGHT_BASIS", 4)
    codes.QTCode(subset, subset, code_a, local_checks="min_weight")
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_MIN_WEIGHT_BASIS", 3)
    with pytest.raises(ValueError, match="local_checks='from_codes'"):
        codes.QTCode(subset, subset, code_a, local_checks="min_weight")
    codes.QTCode(subset, subset, code_a, local_checks="from_codes")


def test_quantum_tanner_without_checks() -> None:
    """Local check spaces of dimension zero give no checks."""
    shift = abstract.CyclicGroup(3).generators[0]
    code = codes.QTCode(
        [shift, ~shift],
        [shift, ~shift],
        np.eye(2, dtype=int),  # the code with no nonzero words
        np.zeros((0, 2), dtype=int),  # the code with all words
    )
    assert code.matrix_x.shape == code.matrix_z.shape == (0, len(code))
    assert code.get_weight() == 0
    assert code.get_local_views(Pauli.X).shape == (len(code) // 2, 2, 2)
    with pytest.raises(ValueError, match=r"Expected Pauli\.X or Pauli\.Z"):
        code.get_local_views(Pauli.Y)  # type: ignore[arg-type]


def get_small_qt_codes(rng: np.random.Generator) -> list[codes.QTCode]:
    """Small QTCodes over abelian and nonabelian groups, on both complexes, over GF(2) and GF(3).

    The codes on the quadripartite complex are random.  A bipartite complex needs subsets that
    satisfy Total No Conjugacy, as those of the fixed seeds do.
    """
    cases: list[
        tuple[abstract.Group, codes.ClassicalCode, codes.ClassicalCode | None, bool, int | None]
    ] = [
        (abstract.CyclicGroup(6), codes.HammingCode(2), None, False, None),
        (
            abstract.SymmetricGroup(3),
            codes.ClassicalCode([[1, 1, 0, 1], [0, 1, 1, 1]]),
            None,
            False,
            None,
        ),
        (
            abstract.CyclicGroup(4),
            codes.ClassicalCode([[1, 2, 0], [0, 1, 1]], field=3),
            None,
            False,
            None,
        ),
        (
            abstract.DihedralGroup(5),
            codes.HammingCode(2),
            codes.ClassicalCode([[1, 1, 1, 0]]),
            True,
            36,
        ),
        (abstract.DihedralGroup(6), codes.HammingCode(2), None, True, 9),
    ]
    return [
        codes.QTCode.random(
            group,
            code_a,
            code_b,
            bipartite=bipartite,
            seed=int(rng.integers(2**31)) if seed is None else seed,
        )
        for group, code_a, code_b, bipartite, seed in cases
    ]


def test_quantum_tanner_translations(pytestconfig: pytest.Config) -> None:
    """The translations that QTCode._get_qudit_automorphisms returns preserve the code."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for code in get_small_qt_codes(rng):
        translations = code._get_qudit_automorphisms()
        for translation in translations if translations is not None else []:
            for matrix in [code.matrix_x, code.matrix_z]:
                assert_same_row_space(matrix[:, np.argsort(translation)], matrix)


def test_quantum_tanner_all_translations() -> None:
    """On small codes, the returned translations generate all translations that preserve the code.

    A brute-force search finds every translation by a vertex, from the left or the right, that
    preserves the code.  The conditions under which QTCode._get_qudit_automorphisms returns a
    translation are sufficient but not known to be necessary, so this test uses fixed codes.
    """
    for code in get_small_qt_codes(np.random.default_rng(0)):
        translations = code._get_qudit_automorphisms()
        assert translations is not None

        # translate the faces by every vertex, keeping the translations that preserve the code
        subgraph_x, subgraph_z = codes.QTCode.get_subgraphs(code.complex)
        faces = {face: qudit for face, qudit in subgraph_x.nodes(data="sort") if qudit is not None}
        vertices = [node for node in subgraph_x if subgraph_x.out_degree(node)]
        vertices += [node for node in subgraph_z if subgraph_z.out_degree(node)]
        symmetries = []
        for vertex, left in itertools.product(vertices, [True, False]):
            images = [frozenset(vertex * v if left else v * vertex for v in face) for face in faces]
            if all(image in faces for image in images):
                permutation = np.array([faces[image] for image in images])
                if all(
                    np.linalg.matrix_rank(np.vstack([matrix, matrix[:, np.argsort(permutation)]]))
                    == np.linalg.matrix_rank(matrix)
                    for matrix in [code.matrix_x, code.matrix_z]
                ):
                    symmetries.append(abstract.GroupMember(permutation.tolist()))
        all_translations = abstract.Group(*symmetries)
        generators = [abstract.GroupMember(row.tolist()) for row in translations]
        assert all(generator in all_translations for generator in generators)
        assert abstract.Group(*generators).order == all_translations.order > 1


def test_quantum_tanner_canonical_checks(pytestconfig: pytest.Config) -> None:
    """Bliss searches all light words of the local check spaces of a QTCode, at every vertex.

    The light words of a local check space are its nonzero words that are no heavier than the
    heaviest word of a minimum-weight basis.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    # include a code whose local check spaces for Z-type checks are zero
    shift = abstract.CyclicGroup(4).generators[0]
    code_without_z_checks = codes.QTCode(
        [shift, ~shift],
        [shift, ~shift, shift**2],
        np.zeros((0, 2), dtype=int),  # the code with all words
        codes.HammingCode(2),
    )
    for code in [*get_small_qt_codes(rng), code_without_z_checks]:
        search_matrix = code._get_canonical_checks()
        num_qudits = len(code)
        is_x_check = np.any(search_matrix[:, :num_qudits], axis=1)
        assert not np.any(search_matrix[is_x_check, num_qudits:])
        sectors: list[tuple[PauliXZ, galois.FieldArray, list[galois.FieldArray]]] = [
            (
                Pauli.X,
                search_matrix[is_x_check, :num_qudits],
                [code.code_a.generator, code.code_b.generator],
            ),
            (
                Pauli.Z,
                search_matrix[~is_x_check, num_qudits:],
                [code.code_a.matrix, code.code_b.matrix],
            ),
        ]
        for pauli, checks, local_spaces in sectors:
            # all light words of the local check space, by brute force
            field = code.field
            basis = np.kron(*local_spaces).view(field).row_space()
            coefficients = field(list(itertools.product(range(field.order), repeat=len(basis))))
            words = coefficients.reshape(field.order ** len(basis), len(basis)) @ basis
            weights = np.count_nonzero(words.view(np.ndarray), axis=1)
            max_weight = max(get_min_weights(basis), default=0)
            light_words = words[(weights > 0) & (weights <= max_weight)]

            # every vertex has the light words of its local check space on its local view
            expected_checks = []
            for local_view in code.get_local_views(pauli):
                for word in light_words:
                    check = field.Zeros(num_qudits)
                    check[local_view.ravel()] = word
                    expected_checks.append(tuple(check.tolist()))
            assert get_sorted_rows(checks) == sorted(expected_checks)
            assert_same_row_space(checks, code.matrix_x if pauli is Pauli.X else code.matrix_z)


def test_quantum_tanner_canonical_checks_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bliss searches the parity checks of X-type or Z-type, if their local spaces are too large.

    In this code, each local check space of X-type checks has 2 words, of which 1 is light, and each
    local check space of Z-type checks has 16 words, of which 9 are light: the words of weight 4 in
    the tensor product of two [3, 2] codes.
    """
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2]
    code = codes.QTCode(subset, subset, codes.HammingCode(2), codes.HammingCode(2))
    num_qudits = len(code)
    num_vertices_x = len(code.get_local_views(Pauli.X))
    num_vertices_z = len(code.get_local_views(Pauli.Z))
    canonical_search_matrix = code._get_canonical_checks()
    assert len(canonical_search_matrix) == num_vertices_x + 9 * num_vertices_z

    # the limit on canonical checks applies to each vertex, not to all vertices together
    monkeypatch.setattr(quantum_tanner, "_MAX_CANONICAL_CHECKS_PER_VERTEX", 10)
    assert 9 * num_vertices_z > 10
    assert get_sorted_rows(code._get_canonical_checks()) == get_sorted_rows(canonical_search_matrix)

    # limits that only the local check spaces of Z-type checks exceed
    for name, limit in [
        ("_MAX_WORDS_FOR_CANONICAL_CHECKS", 8),
        ("_MAX_CANONICAL_CHECKS_PER_VERTEX", 4),
    ]:
        with monkeypatch.context() as context:
            context.setattr(quantum_tanner, name, limit)
            search_matrix = code._get_canonical_checks()
        assert len(search_matrix) == num_vertices_x + len(code.matrix_z)
        assert_same_row_space(search_matrix[:num_vertices_x, :num_qudits], code.matrix_x)
        assert np.array_equal(search_matrix[num_vertices_x:, num_qudits:], code.matrix_z)

    # a limit that all local check spaces exceed
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_CANONICAL_CHECKS", 0)
    assert np.array_equal(code._get_canonical_checks(), code.matrix)


def test_quantum_tanner_canonical_checks_without_qudits() -> None:
    """Canonical checks of codes without qudits, and of local check spaces without nonzero words."""
    shift = abstract.CyclicGroup(4).generators[0]
    repetition_code = codes.ClassicalCode([[1, 1]])
    empty_code = codes.ClassicalCode(np.zeros((0, 0), dtype=int))
    for subset_a, subset_b, code_a, code_b in [
        ([shift, ~shift], [], repetition_code, empty_code),
        ([], [shift, ~shift], empty_code, repetition_code),
    ]:
        code = codes.QTCode(subset_a, subset_b, code_a, code_b)
        assert len(code) == 0
        assert code._get_canonical_checks().shape == (0, 0)
        assert code._get_qudit_automorphisms() is None
        assert code.get_known_qudit_automorphism_group().order == 1

    # a zero factor makes a zero space, whose canonical checks need no search of the other factor
    field = galois.GF(2)
    canonical_checks = quantum_tanner._get_canonical_local_checks(
        field.Zeros((0, 3)), field.Identity(62)
    )
    assert canonical_checks is not None and canonical_checks.shape == (0, 3 * 62)


def test_quantum_tanner_symmetries() -> None:
    """The symmetries of a [[144, 8, 12]] code.

    The code has 72 translations.  With pybliss, bliss finds 1152 symmetries among its canonical
    local checks, but only 48 among its parity checks, and the known group of the code has order
    1152.  Exact distance searches, which can use the symmetries, find the distance of the code with
    either method.
    """
    code = get_c4_a4_code("min_weight")
    translations = code._get_qudit_automorphisms()
    assert translations is not None
    group = abstract.Group(*[abstract.GroupMember(row.tolist()) for row in translations])
    assert group.order == 72

    methods: list[Literal["brouwer_zimmermann", "connected_cluster"]] = [
        "brouwer_zimmermann",
        "connected_cluster",
    ]
    for method in methods:
        assert get_c4_a4_code("min_weight").get_distance_exact(method=method) == 12

    # the canonical checks do not depend on the choice of local checks, unlike the parity checks
    search_matrix = code._get_canonical_checks()
    for local_checks in LOCAL_CHECKS:
        other_code = get_c4_a4_code(local_checks)
        assert get_sorted_rows(other_code._get_canonical_checks()) == get_sorted_rows(search_matrix)
    assert not np.array_equal(get_c4_a4_code("from_codes").matrix, code.matrix)

    # plain codes, which know no symmetries of a construction, give the groups that bliss finds
    pytest.importorskip("pybliss")
    for matrix, order in [(search_matrix, 1152), (code.matrix, 48)]:
        assert codes.QuditCode(matrix).get_known_qudit_automorphism_group().order == order
    assert code.get_known_qudit_automorphism_group().order == 1152


def test_quantum_tanner_symmetries_of_parity_checks() -> None:
    """The known group of a QTCode contains the symmetries that bliss finds among its parity checks.

    In this [[200, 12]] code over the dihedral group D5, the local code C_B is the direct sum of
    repetition codes on two disjoint pairs of coordinates.  Bliss finds a group of order 800 among
    the parity checks, but only 8 among the canonical local checks, which some of those symmetries
    do not map onto themselves.
    """
    pytest.importorskip("pybliss")
    subset_a = [[1, 0, 4, 3, 2], [2, 1, 0, 4, 3], [2, 3, 4, 0, 1], [3, 2, 1, 0, 4], [3, 4, 0, 1, 2]]
    subset_b = [[1, 0, 4, 3, 2], [1, 2, 3, 4, 0], [3, 2, 1, 0, 4], [4, 0, 1, 2, 3]]
    code = codes.QTCode(
        [abstract.GroupMember(member) for member in subset_a],
        [abstract.GroupMember(member) for member in subset_b],
        [[1, 0, 0, 0, 1], [1, 1, 1, 1, 0]],
        [[1, 1, 1, 1], [0, 1, 0, 1]],
    )
    assert (len(code), code.dimension) == (200, 12)
    canonical_checks_code = codes.QuditCode(code._get_canonical_checks())
    parity_checks_group = codes.CSSCode(
        code.matrix_x, code.matrix_z
    ).get_known_qudit_automorphism_group()
    assert canonical_checks_code.get_known_qudit_automorphism_group().order == 8
    assert parity_checks_group.order == 800
    group = code.get_known_qudit_automorphism_group()
    assert all(member in group for member in parity_checks_group.generators)
