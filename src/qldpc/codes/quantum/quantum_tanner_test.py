# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quantum_tanner.py."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import pathlib
import pickle
import random
import re
import textwrap
import unittest.mock
from typing import get_args

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
import sympy

from qldpc import abstract, codes
from qldpc.codes.quantum import quantum_tanner
from qldpc.codes.quantum.quantum_tanner import LocalChecks
from qldpc.conftest import run_python
from qldpc.objects import PAULIS_XZ, Pauli, PauliXZ

LOCAL_CHECKS: tuple[LocalChecks, ...] = get_args(LocalChecks)


def test_quantum_tanner(pytestconfig: pytest.Config, tmp_path: pathlib.Path) -> None:
    """Quantum Tanner code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    # random quantum Tanner code
    group = abstract.CyclicGroup(12)
    local_code = codes.RepetitionCode(4, field=3)
    code = codes.QTCode.random(group, local_code, seed=np.random.randint(2**31))

    # raise error if C_B is not given, and the subsets have different sizes
    subset_a = code.complex.subset_a
    subset_b = group.random_symmetric_subset(len(subset_a) - 1)
    code_a = codes.RepetitionCode(len(subset_a))
    with pytest.raises(ValueError, match="code_b is required"):
        codes.QTCode(subset_a, subset_b, code_a)

    # raise error if the local codes are over different fields
    code_b = codes.RepetitionCode(2, field=3)
    with pytest.raises(ValueError, match="different fields"):
        codes.QTCode(subset_a, subset_a, code_a, code_b)

    # saving and loading QTCodes on both complexes, creating a missing directory, overwriting an
    # existing file, and keeping notes as written, even one that looks like a list of integers
    cycle = abstract.GroupMember([1, 2, 0])
    swaps = [abstract.GroupMember(image) for image in [[1, 0, 2], [2, 1, 0], [0, 2, 1]]]
    bipartite_code = codes.QTCode([cycle, ~cycle], swaps, [[1, 1]], [[1, 1, 1]], bipartite=True)
    path = tmp_path / "subdirectory" / "code.json"
    notes = ["a note", "a note\nwith two lines", "[ 1,2]"]
    for saved_code in [code, bipartite_code]:
        saved_code.save_json(str(path), *notes)
        assert json.loads(path.read_text())["notes"] == notes
        assert_loads_equal_code(saved_code, path)

    # saved files do not depend on the iteration order of subsets, which varies between sessions
    saved_files = []
    for reverse in [False, True]:
        ordered_a = sorted(code.complex.subset_a, reverse=reverse)
        ordered_b = sorted(code.complex.subset_b, reverse=reverse)
        with (
            unittest.mock.patch.object(code.complex, "subset_a", ordered_a),
            unittest.mock.patch.object(code.complex, "subset_b", ordered_b),
        ):
            code.save_json(str(path))
        saved_files.append(path.read_text())
    assert saved_files[0] == saved_files[1]

    # codes whose saved data has empty rows or matrices, such as members of the trivial group or
    # local codes without parity checks, load back too
    generator = abstract.CyclicGroup(2).generators[0]
    for saved_code in [
        codes.QTCode.random(abstract.TrivialGroup(), codes.ClassicalCode([[1]])),
        codes.QTCode([generator], [generator], np.zeros((0, 1), dtype=int), [[1]]),
    ]:
        saved_code.save_json(str(path))
        assert_loads_equal_code(saved_code, path)

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
    local_code = codes.RepetitionCode(2)
    code = codes.QTCode(members, members, local_code)

    # repeated members are merged before the sizes of the subsets are compared
    assert codes.QTCode(members * 2, members, local_code) == code

    # codes are equal if their members are equal after adding fixed points to the end of them
    padded_members = [
        abstract.GroupMember(member.array_form, size=member.size + 1) for member in members
    ]
    padded_code = codes.QTCode(padded_members, padded_members, local_code)
    assert padded_code.complex.subset_a != code.complex.subset_a
    assert padded_code == code
    assert code == padded_code
    assert code != code.complex

    # members_a and members_b hold the members of A and B on the largest number of points of any
    # member of A or B
    swap_a, swap_b = abstract.GroupMember([1, 0]), abstract.GroupMember([0, 1, 3, 2])
    members_a = codes.QTCode([swap_a], [swap_b], [[1]]).members_a
    assert [member.array_form for member in members_a] == [[1, 0, 2, 3]]
    members_b = codes.QTCode([swap_b], [swap_a], [[1]]).members_b
    assert [member.array_form for member in members_b] == [[1, 0, 2, 3]]

    # codes whose local codes have different parity check matrices are not equal, even if they have
    # the same subsets and parity checks
    subset = [shift, ~shift, shift**3]
    code_1 = codes.QTCode(subset, subset, [[1, 1, 0], [0, 1, 1]], [[1, 1, 1]])
    code_2 = codes.QTCode(subset, subset, [[1, 0, 1], [0, 1, 1]], [[1, 1, 1]])
    assert np.array_equal(code_1.matrix, code_2.matrix)
    assert code_1 != code_2

    # codes whose local codes carry different generator matrices are equal, if they have the same
    # generating data and parity checks
    redundant_code = codes.ClassicalCode([[1, 1, 0], [0, 1, 1]]).set_generator([[1, 1, 1]] * 2)
    assert codes.QTCode(subset, subset, redundant_code, [[1, 1, 1]]) == code_1

    # codes whose local codes are over different fields are not equal, even with equal matrices
    code_gf4 = codes.QTCode(members, members, codes.RepetitionCode(2, field=4))
    assert np.array_equal(code_gf4.matrix, code.matrix)
    assert code_gf4 != code

    # raise error if a local code does not have one bit per distinct member of its subset, even if
    # the product of the block lengths of the local codes is right
    other_members = [shift**2, shift**3, ~(shift**2)]
    for code_a, code_b, wrong_code in [
        (codes.RepetitionCode(3), codes.RepetitionCode(3), "C_A"),
        (codes.RepetitionCode(2), codes.RepetitionCode(2), "C_B"),
        (codes.RepetitionCode(3), codes.RepetitionCode(2), "C_[AB]"),
    ]:
        with pytest.raises(ValueError, match=f"local code {wrong_code} has block length"):
            codes.QTCode(members, other_members, code_a, code_b)

    # one generating subset may be empty, which leaves a complex without faces, but not both
    no_bits = np.zeros((0, 0), dtype=int)
    assert codes.QTCode([], members, no_bits, local_code).num_qudits == 0
    assert codes.QTCode(members, [], local_code, no_bits).num_qudits == 0
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
        for member_a in cayplex.cover_subset_a:
            for member_b in cayplex.cover_subset_b:
                neighbor = member_a * element
                face = frozenset([element, neighbor, element * member_b, neighbor * member_b])
                expected_x.add((element, face, (member_a, member_b)))
                expected_z.add((neighbor, face, (~member_a, member_b)))

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
    face_qudits_x, face_qudits_z = [
        {node: qudit for node, qudit in graph.nodes(data="sort") if graph.out_degree(node) == 0}
        for graph in [subgraph_x, subgraph_z]
    ]
    assert face_qudits_x == face_qudits_z
    assert sorted(face_qudits_x.values()) == list(range(code.num_qudits))

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
    local_code = codes.RepetitionCode(2)

    def matrix_for(seed: int | None = None, one_subset: bool = False) -> bytes:
        code = codes.QTCode.random(group, local_code, seed=seed, one_subset=one_subset)
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
    # must reproduce the code built here, under this interpreter's (randomized) hash seed.  The hash
    # seed does not interact with compilation, so the children skip compiling qLDPC's functions.
    script = (
        "import numpy as np;"
        "from qldpc import abstract, codes;"
        "code = codes.QTCode.random(abstract.CyclicGroup(8), codes.RepetitionCode(2), seed=7);"
        "print(np.asarray(code.matrix).tobytes().hex())"
    )
    matrices = {
        run_python(script, env={"PYTHONHASHSEED": hash_seed, "QLDPC_DISABLE_COMPILATION": "1"})
        for hash_seed in ["0", "1"]
    }
    assert matrices == {matrix_for(seed=7).hex() + "\n"}


def test_quantum_tanner_random_defaults() -> None:
    """QTCode.random uses the dual of C_A for C_B by default, and can use the subset A for B.

    Without one_subset, seed 0 draws different subsets A and B, so the last assertion can fail.
    """
    group = abstract.CyclicGroup(8)
    code_a = codes.HammingCode(2)
    code = codes.QTCode.random(group, code_a, seed=0)
    assert_same_row_space(code.code_b.matrix, code_a.generator)
    assert code.complex.subset_a != code.complex.subset_b
    code = codes.QTCode.random(group, code_a, seed=0, one_subset=True)
    assert code.complex.subset_a == code.complex.subset_b


def test_toric_tanner_code(size: int = 4) -> None:
    """Rotated toric code as a quantum Tanner code."""
    group = abstract.Group.product(abstract.CyclicGroup(size), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    local_code = codes.RepetitionCode(2)
    code = codes.QTCode(subset_a, subset_b, local_code, bipartite=False)
    assert code.get_code_params() == (size**2, 2, size)
    assert code.get_weight() == 4


def get_words(space: galois.FieldArray) -> galois.FieldArray:
    """All words of the row space of a matrix, by brute force."""
    field = type(space)
    basis = space.row_space()
    coefficients = field(list(itertools.product(range(field.order), repeat=len(basis))))
    return coefficients.reshape(field.order ** len(basis), len(basis)) @ basis


def get_min_weights(space: galois.FieldArray) -> list[int]:
    """The weights of a minimum-weight basis of the row space of a matrix, by brute force.

    A minimum-weight basis has exactly r_w - r_(w-1) vectors of weight w, where r_w is the dimension
    of the span of all vectors of the row space with weight at most w.
    """
    words = get_words(space)
    word_weights = np.count_nonzero(words.view(np.ndarray), axis=1)
    weights: list[int] = []
    for weight in range(1, space.shape[1] + 1):
        light_words = words[(word_weights > 0) & (word_weights <= weight)]
        light_rank = np.linalg.matrix_rank(light_words) if len(light_words) else 0
        weights += [weight] * (light_rank - len(weights))
    return weights


def get_light_words(space: galois.FieldArray) -> galois.FieldArray:
    """The light words of the row space of a matrix, by brute force.

    These are the nonzero words that are no heavier than the heaviest word of a minimum-weight basis
    of the space.
    """
    words = get_words(space)
    weights = np.count_nonzero(words.view(np.ndarray), axis=1)
    return words[(weights > 0) & (weights <= max(get_min_weights(space), default=0))]


def get_local_space(
    code_a: codes.ClassicalCode, code_b: codes.ClassicalCode, pauli: PauliXZ
) -> galois.FieldArray:
    """A matrix whose rows span the local check space of one type, C_A ⨂ C_B or ~C_A ⨂ ~C_B.

    Its rows are the products of the rows of the generator matrices of the local codes for X-type
    checks, and of their parity check matrices for Z-type checks.
    """
    if pauli is Pauli.X:
        return np.kron(code_a.generator, code_b.generator).view(code_a.field)
    return np.kron(code_a.matrix, code_b.matrix).view(code_a.field)


def place_words(
    words: galois.FieldArray, local_views: npt.NDArray[np.int_], num_qudits: int
) -> list[tuple[int, ...]]:
    """The rows of the checks that place each word on each local view.

    Entry ``[v, i, j]`` of local_views is the qudit at coordinate (i, j) of the local view of vertex
    v, and the words have one entry per coordinate, in row-major order.
    """
    checks = []
    for local_view in local_views:
        for word in words:
            check = type(words).Zeros(num_qudits)
            check[local_view.ravel()] = word
            checks.append(tuple(check.tolist()))
    return checks


def get_sorted_rows(matrix: npt.ArrayLike) -> list[tuple[int, ...]]:
    """The rows of a matrix, in sorted order, with their multiplicities."""
    return sorted(map(tuple, np.asarray(matrix).tolist()))


def assert_same_row_space(matrix: galois.FieldArray, other: galois.FieldArray) -> None:
    """Assert that two matrices over a field have the same row space."""
    rank = np.linalg.matrix_rank(matrix) if matrix.size else 0
    other_rank = np.linalg.matrix_rank(other) if other.size else 0
    stacked = np.vstack([matrix, other]).view(type(matrix))
    assert rank == other_rank == (np.linalg.matrix_rank(stacked) if stacked.size else 0)


def test_quantum_tanner_local_checks(pytestconfig: pytest.Config) -> None:
    """The choices of local checks span the local check spaces of the local views of vertices."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for code in get_small_qt_codes(rng):
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
                bipartite=code.complex.bipartite,
                local_checks=local_checks,
            )
            assert len(new_code) == len(code)
            assert new_code.dimension == code.dimension
            assert_same_row_space(new_code.matrix_x, code.matrix_x)
            assert_same_row_space(new_code.matrix_z, code.matrix_z)
            assert not np.any(new_code.matrix_x @ new_code.matrix_z.T)

            sectors: list[tuple[PauliXZ, nx.DiGraph, galois.FieldArray]] = [
                (Pauli.X, subgraphs[0], new_code.matrix_x),
                (Pauli.Z, subgraphs[1], new_code.matrix_z),
            ]
            for pauli, subgraph, matrix in sectors:
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
                local_space = get_local_space(code_a, code_b, pauli)
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
                        weights = np.count_nonzero(local_checks_at_vertex.view(np.ndarray), axis=1)
                        assert sorted(weights.tolist()) == get_min_weights(local_space)


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
    """Minimum-weight local checks give a [[144, 8, 12]] code with checks of weight 6."""
    code = get_c4_a4_code()
    assert (len(code), code.dimension) == (144, 8)
    assert code.get_weight() == 6


def test_quantum_tanner_parity_check_regression(tmp_path: pathlib.Path) -> None:
    """Saved files load with pinned parity check matrices.

    This test pins the parity check matrices, including the order of checks and qudits, that
    QTCode.load_json builds from the data of files that QTCode.save_json writes.  The codes are over
    GF(2) on both complexes, over GF(3) with each choice of local checks, and over GF(4).  Each
    local check space of the "min_weight" codes has several minimum-weight bases, so the pins also
    fix which one QTCode chooses.
    """

    def get_digest(matrix: galois.FieldArray) -> tuple[tuple[int, ...], str]:
        """The shape of a matrix over a field of order at most 256, and a hash of its entries."""
        entries = matrix.view(np.ndarray).astype(np.uint8).tobytes()
        return matrix.shape, hashlib.sha256(entries).hexdigest()[:16]

    path = tmp_path / "code.json"
    members_z4 = [[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    for data, (digest_x, digest_z) in [
        (
            # GF(2), on the quadripartite complex of Z_6
            {
                "subset_a": [[1, 2, 3, 4, 5, 0], [3, 4, 5, 0, 1, 2], [5, 0, 1, 2, 3, 4]],
                "subset_b": [[1, 2, 3, 4, 5, 0], [3, 4, 5, 0, 1, 2], [5, 0, 1, 2, 3, 4]],
                "code_a": {"matrix": [[1, 1, 0], [0, 1, 1]], "generator": [[1, 1, 1]]},
                "code_b": {"matrix": [[1, 1, 1]], "generator": [[1, 1, 0], [0, 1, 1]]},
                "field_order": 2,
                "bipartite": False,
                "local_checks": "min_weight",
            },
            (((12, 27), "b00585d05a7219f9"), ((12, 27), "d07562358f9a1120")),
        ),
        (
            # GF(2), on the bipartite complex of S_3, with the 3-cycles in A, and the identity and
            # the transpositions in B
            {
                "subset_a": [[1, 2, 0], [2, 0, 1]],
                "subset_b": [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]],
                "code_a": {"matrix": [[1, 1]], "generator": [[1, 1]]},
                "code_b": {
                    "matrix": [[1, 1, 1, 0], [0, 1, 1, 1]],
                    "generator": [[1, 0, 1, 1], [0, 1, 1, 0]],
                },
                "field_order": 2,
                "bipartite": True,
                "local_checks": "min_weight",
            },
            (((12, 24), "665216ea24e4f1c0"), ((12, 24), "a23deb3c02e0831d")),
        ),
        (
            # GF(3), on the quadripartite complex of Z_4
            {
                "subset_a": members_z4,
                "subset_b": members_z4,
                "code_a": {"matrix": [[1, 1, 1]], "generator": [[1, 2, 0], [1, 1, 1]]},
                "code_b": {"matrix": [[1, 2, 0], [1, 1, 1]], "generator": [[1, 1, 1]]},
                "field_order": 3,
                "bipartite": False,
                "local_checks": "min_weight",
            },
            (((16, 36), "a0a948a348718a44"), ((16, 36), "3a21b7d13b9ce30e")),
        ),
        (
            # the same subsets and local codes with "from_codes", and generator matrices that are
            # not the defaults, which give the X-type checks
            {
                "subset_a": members_z4,
                "subset_b": members_z4,
                "code_a": {"matrix": [[1, 1, 1]], "generator": [[1, 2, 0], [1, 1, 1]]},
                "code_b": {"matrix": [[1, 2, 0], [1, 1, 1]], "generator": [[2, 2, 2]]},
                "field_order": 3,
                "bipartite": False,
                "local_checks": "from_codes",
            },
            (((16, 36), "15b0a39fa21e11be"), ((16, 36), "69cae8398f44678f")),
        ),
        (
            # GF(4), on the quadripartite complex of the Klein group
            {
                "subset_a": [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
                "subset_b": [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
                "code_a": {"matrix": [[1, 2, 3]], "generator": [[1, 0, 2], [0, 1, 3]]},
                "code_b": {"matrix": [[1, 0, 2], [0, 1, 3]], "generator": [[1, 2, 3]]},
                "field_order": 4,
                "bipartite": False,
                "local_checks": "min_weight",
            },
            (((16, 36), "0000e37ac9968dad"), ((16, 36), "b8f9705a40a9fe41")),
        ),
    ]:
        saved_data = {"notes": [], **data}
        path.write_text(json.dumps(saved_data))
        code = codes.QTCode.load_json(str(path))
        assert get_digest(code.matrix_x) == digest_x
        assert get_digest(code.matrix_z) == digest_z

        # these are the data that QTCode.save_json writes for the code
        code.save_json(str(path))
        assert json.loads(path.read_text()) == saved_data


def test_quantum_tanner_default_dual_code() -> None:
    """The default C_B is ~C_A, whose generator matrix is the parity check matrix of C_A."""
    shift = abstract.CyclicGroup(4).generators[0]
    subset = [shift, ~shift, shift**2, shift**4]
    matrix_a = [[1, 1, 0, 1], [0, 1, 1, 1]]  # not in reduced row echelon form
    code = codes.QTCode(subset, subset, matrix_a)
    assert np.array_equal(code.code_b.generator, matrix_a)


def test_quantum_tanner_default_local_checks() -> None:
    """QTCodes choose minimum-weight local checks by default."""
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
    members = [~shift, shift, shift**2, shift**4]
    assert codes.QTCode.from_ordered_members(
        members, members, code_a
    ) == codes.QTCode.from_ordered_members(members, members, code_a, local_checks="min_weight")

    for local_checks in ["row_reduced", " min_weight", None]:
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
    path = tmp_path / "code.json"
    for local_checks in LOCAL_CHECKS:
        code = codes.QTCode(subset, subset, code_a, code_b, local_checks=local_checks)
        code.save_json(str(path))
        assert json.loads(path.read_text())["local_checks"] == local_checks
        assert_loads_equal_code(code, path)

    # QTCodes with the same generating data and the same checks are equal, whatever choice of local
    # checks gave them
    toric_codes = [
        codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2), local_checks=choice)
        for choice in LOCAL_CHECKS
    ]
    assert toric_codes[0] == toric_codes[1]


def test_quantum_tanner_save_fields(tmp_path: pathlib.Path) -> None:
    """QTCodes save only over default representations of fields, since files record only the order.

    Over a field with another irreducible polynomial or primitive element, QTCode.load_json would
    read the saved values in the default field, so QTCode.save_json raises before it writes
    anything.
    """
    shift = abstract.CyclicGroup(4).generators[0]
    path = tmp_path / "subdirectory" / "code.json"
    for field in [
        galois.GF(8, irreducible_poly="x^3 + x^2 + 1"),
        galois.GF(8, primitive_element=3),
        galois.GF(5, primitive_element=3),
    ]:
        # given both local codes, and checks taken from them, these codes need neither a dual code
        # nor minimum-weight checks, whose computations are costly in a new field
        code = codes.QTCode(
            [shift, ~shift], [shift, ~shift], [[1, 2]], [[2, 1]], field, local_checks="from_codes"
        )
        with pytest.raises(ValueError, match="record only the order of the field"):
            code.save_json(str(path))
        assert not path.parent.exists()

    # the default representation of an extension field saves and loads
    code = codes.QTCode([shift, ~shift], [shift, ~shift], [[1, 2]], field=4)
    code.save_json(str(path))
    assert_loads_equal_code(code, path)


def test_quantum_tanner_save_json_text(tmp_path: pathlib.Path) -> None:
    """QTCode.save_json writes indented JSON, with each member and each matrix row on one line."""
    shift = abstract.CyclicGroup(3).generators[0]
    code = codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2))
    path = tmp_path / "code.json"
    code.save_json(str(path), "a note", "another note")
    assert path.read_text() == textwrap.dedent(
        """\
        {
          "notes": [
            "a note",
            "another note"
          ],
          "subset_a": [
            [1, 2, 0],
            [2, 0, 1]
          ],
          "subset_b": [
            [1, 2, 0],
            [2, 0, 1]
          ],
          "code_a": {
            "matrix": [
              [1, 1]
            ],
            "generator": [
              [1, 1]
            ]
          },
          "code_b": {
            "matrix": [
              [1, 1]
            ],
            "generator": [
              [1, 1]
            ]
          },
          "field_order": 2,
          "bipartite": false,
          "local_checks": "min_weight"
        }
        """
    )


def test_quantum_tanner_load_json_errors(tmp_path: pathlib.Path) -> None:
    """QTCode.load_json raises a ValueError that names the file and the problem."""
    shift = abstract.CyclicGroup(4).generators[0]
    path = tmp_path / "code.json"
    codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2)).save_json(str(path))
    data = json.loads(path.read_text())

    def assert_load_fails(message: str) -> None:
        """Assert that loading the file fails with an error whose message names it."""
        with pytest.raises(ValueError, match=f"(?s){message}.*: {re.escape(str(path))}$"):
            codes.QTCode.load_json(str(path))

    # a file that does not exist, does not parse as JSON, or does not hold a JSON object
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match=f"Path does not exist: {re.escape(str(missing))}$"):
        codes.QTCode.load_json(str(missing))
    for text in ["{", "[" * 10**5]:  # the second is nested too deeply for json to parse
        path.write_text(text)
        assert_load_fails("does not parse as JSON")
    path.write_bytes(b'{"notes": ["\xff"]}')  # not UTF-8
    assert_load_fails("does not parse as JSON")
    path.write_text("[]")
    assert_load_fails("is not a JSON object")

    # an object without a key that QTCode.save_json writes, or with another key
    path.write_text(json.dumps({key: value for key, value in data.items() if key != "bipartite"}))
    assert_load_fails("has no 'bipartite' entry")
    path.write_text(json.dumps(data | {"distance": 2}))
    assert_load_fails("has an unknown 'distance' entry")

    # malformed entries
    rows = data["code_a"]["matrix"]
    for key, value in [
        ("notes", "a note"),
        ("notes", [1]),
        ("field_order", True),
        ("bipartite", 1),
        ("subset_a", [[1, 2, 3, 0.0], [3, 0, 1, 2]]),
        ("subset_a", [[1, 1, 3, 0], [3, 0, 1, 2]]),  # not a permutation
        ("subset_a", [[3, 0, 1, 2], [1, 2, 3, 0]]),  # not in increasing order
        ("subset_a", [[1, 2, 3, 0], [1, 2, 3, 0], [3, 0, 1, 2]]),  # not distinct
        # not distinct on 5 points, with the repeated rows in either order
        ("subset_a", [[1, 2, 3, 0], [1, 2, 3, 0, 4], [3, 0, 1, 2]]),
        ("subset_a", [[1, 2, 3, 0, 4], [1, 2, 3, 0], [3, 0, 1, 2]]),
        ("code_a", rows),
        ("code_a", {"matrix": rows}),
        ("code_a", {"matrix": [[1, True]], "generator": [[1, 1]]}),
        ("code_b", {"matrix": [[1, 1], [1]], "generator": []}),
        ("code_b", {"matrix": [[1, 1, 1]], "generator": []}),  # one column per member of B
        ("code_b", {"matrix": [[1, 2**64]], "generator": [[1, 1]]}),
    ]:
        path.write_text(json.dumps(data | {key: value}))
        assert_load_fails(f"has a malformed {key!r} entry")

    # data that do not define a QTCode: a choice of local checks that is not one, field orders that
    # the constructors reject, a generator matrix whose rows are not words of the code, and a subset
    # that is not symmetric
    for key, value, message in [
        ("local_checks", "row_reduced", "Unrecognized choice of local checks"),
        ("field_order", 6, ""),
        ("field_order", 2**70, ""),
        ("code_a", {"matrix": rows, "generator": [[1, 0]]}, "nontrivial syndromes"),
        ("subset_a", [[1, 2, 3, 0], [2, 3, 0, 1]], "not symmetric"),
    ]:
        path.write_text(json.dumps(data | {key: value}))
        assert_load_fails(f"does not define a QTCode.*{message}")


def assert_loads_equal_code(code: codes.QTCode, path: pathlib.Path) -> None:
    """Assert that QTCode.load_json rebuilds a code from the file that code.save_json wrote.

    The loaded code is equal to the code, with the same parity check matrices, including the order
    of checks and qudits, and the same generator matrices of its local codes.  Saving it writes the
    same file, so it also has the same choice of local checks.
    """
    loaded_code = codes.QTCode.load_json(str(path))
    assert loaded_code == code
    assert np.array_equal(loaded_code.matrix_x, code.matrix_x)
    assert np.array_equal(loaded_code.matrix_z, code.matrix_z)
    assert np.array_equal(loaded_code.code_a.generator, code.code_a.generator)
    assert np.array_equal(loaded_code.code_b.generator, code.code_b.generator)
    text = path.read_text()
    loaded_code.save_json(str(path), *json.loads(text)["notes"])
    assert path.read_text() == text


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


def test_quantum_tanner_local_views_of_copies() -> None:
    """The local views of a copy of a QTCode are read-only, as those of the code are."""
    shift = abstract.CyclicGroup(4).generators[0]
    code = codes.QTCode([shift, ~shift], [shift, ~shift], codes.RepetitionCode(2))

    # read the views first, so that the copy carries the cached arrays
    for pauli in PAULIS_XZ:
        assert not code.get_local_views(pauli).flags.writeable
    unpickled_code = pickle.loads(pickle.dumps(code))  # noqa: S301 - trusted in-memory round trip
    for copied_code in [copy.deepcopy(code), unpickled_code]:
        for pauli in PAULIS_XZ:
            assert not copied_code.get_local_views(pauli).flags.writeable


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


def test_quantum_tanner_all_translations() -> None:
    """On small codes, the returned translations generate all translations that preserve the code.

    A brute-force search finds every translation by a vertex, from the left or the right, that
    preserves the code.  The conditions under which QTCode._get_permutation_automorphisms returns a
    translation are sufficient but not known to be necessary, so this test uses fixed codes.
    """
    for code in get_small_qt_codes(np.random.default_rng(0)):
        translations = code._get_permutation_automorphisms()
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
        sectors: list[tuple[PauliXZ, galois.FieldArray]] = [
            (Pauli.X, search_matrix[is_x_check, :num_qudits]),
            (Pauli.Z, search_matrix[~is_x_check, num_qudits:]),
        ]
        for pauli, checks in sectors:
            # every vertex has the light words of its local check space on its local view
            light_words = get_light_words(get_local_space(code.code_a, code.code_b, pauli))
            expected_checks = place_words(light_words, code.get_local_views(pauli), num_qudits)
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
        assert code._get_permutation_automorphisms() is None
        assert code.get_known_permutation_automorphism_group().order == 1

    # a zero factor makes a zero space, whose canonical checks need no search of the other factor
    canonical_checks = quantum_tanner._get_canonical_local_checks(
        codes.ClassicalCode(np.zeros((0, 3), dtype=int)),
        codes.ClassicalCode(np.eye(62, dtype=int)),
        Pauli.Z,
    )
    assert canonical_checks is not None and canonical_checks.shape == (0, 3 * 62)


def test_quantum_tanner_canonical_checks_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical local checks are the light words of a space, also when found in chunks.

    The local check spaces of Z-type checks here have 16 and 9 words, which a chunk size of 5 splits
    into several chunks.  The first space has 6 light words.
    """
    binary_codes = (
        codes.ClassicalCode([[1, 1, 0, 0], [0, 0, 1, 1]]),
        codes.ClassicalCode([[1, 1, 1, 0], [0, 1, 1, 1]]),
    )
    ternary_codes = (
        codes.ClassicalCode([[1, 2, 0], [0, 1, 1]], field=3),
        codes.ClassicalCode([[1, 1, 1]], field=3),
    )
    monkeypatch.setattr(quantum_tanner, "_WORDS_PER_CHUNK", 5)
    for code_a, code_b in [binary_codes, ternary_codes]:
        light_words = get_light_words(get_local_space(code_a, code_b, Pauli.Z))
        canonical_checks = quantum_tanner._get_canonical_local_checks(code_a, code_b, Pauli.Z)
        assert canonical_checks is not None
        assert get_sorted_rows(canonical_checks) == get_sorted_rows(light_words)

    # the cap on canonical checks per vertex counts the light words of all chunks
    monkeypatch.setattr(quantum_tanner, "_WORDS_PER_CHUNK", 1)
    monkeypatch.setattr(quantum_tanner, "_MAX_CANONICAL_CHECKS_PER_VERTEX", 1)
    assert quantum_tanner._get_canonical_local_checks(*binary_codes, Pauli.Z) is None


def test_quantum_tanner_symmetries() -> None:
    """The symmetries of a [[144, 8, 12]] code.

    The code has 72 translations.  With pybliss, bliss finds 1152 symmetries among its canonical
    local checks, but only 48 among its parity checks, and the known group of the code has order
    1152.
    """
    code = get_c4_a4_code("min_weight")
    translations = code._get_permutation_automorphisms()
    assert translations is not None
    group = abstract.Group(*[abstract.GroupMember(row.tolist()) for row in translations])
    assert group.order == 72

    # the canonical checks do not depend on the choice of local checks, unlike the parity checks
    search_matrix = code._get_canonical_checks()
    other_code = get_c4_a4_code("from_codes")
    assert get_sorted_rows(other_code._get_canonical_checks()) == get_sorted_rows(search_matrix)
    assert not np.array_equal(other_code.matrix, code.matrix)

    # plain codes, which know no symmetries of a construction, give the groups that bliss finds
    pytest.importorskip("pybliss")
    for matrix, order in [(search_matrix, 1152), (code.matrix, 48)]:
        assert codes.QuditCode(matrix).get_known_permutation_automorphism_group().order == order
    assert code.get_known_permutation_automorphism_group().order == 1152


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
    ).get_known_permutation_automorphism_group()
    assert canonical_checks_code.get_known_permutation_automorphism_group().order == 8
    assert parity_checks_group.order == 800
    group = code.get_known_permutation_automorphism_group()
    assert all(member in group for member in parity_checks_group.generators)


def get_d5_subsets() -> tuple[list[abstract.GroupMember], list[abstract.GroupMember]]:
    """Subsets of the dihedral group D5 that satisfy Total No Conjugacy, in increasing order.

    The subset A holds the identity and two reflections, and B holds the four rotations.  The
    quadripartite complex of these subsets is connected.
    """
    rotation = abstract.GroupMember([1, 2, 3, 4, 0])
    subset_a = [abstract.GroupMember(image) for image in ([0, 1, 2, 3, 4], [0, 4, 3, 2, 1])]
    subset_a.append(abstract.GroupMember([4, 3, 2, 1, 0]))
    return subset_a, [rotation**power for power in range(1, 5)]


def get_permuted_code(code: codes.ClassicalCode, order: list[int]) -> codes.ClassicalCode:
    """The code whose coordinate i is coordinate order[i] of a code, with its generator matrix."""
    return codes.ClassicalCode(code.matrix[:, order]).set_generator(code.generator[:, order])


def test_quantum_tanner_from_ordered_members() -> None:
    """QTCode.from_ordered_members permutes local coordinates into the order of their members."""
    shift = abstract.CyclicGroup(6).generators[0]
    members_a = [shift**5, shift**2, shift, shift**4]
    members_b = [shift**3, shift**5, shift]
    code_a = codes.ClassicalCode([[1, 1, 0, 1], [0, 1, 1, 1]])
    code_a.set_generator([[1, 1, 1, 0], [1, 0, 1, 1]])  # not in reduced row echelon form
    code_b = codes.ClassicalCode([[1, 1, 0]])

    # members in increasing order are those with increasing exponents, which permutes coordinates
    order_a, order_b = [2, 1, 3, 0], [2, 0, 1]
    code = codes.QTCode.from_ordered_members(
        members_a, members_b, code_a, code_b, local_checks="from_codes"
    )
    expected_code_a = get_permuted_code(code_a, order_a)
    expected_code = codes.QTCode(
        members_a,
        members_b,
        expected_code_a,
        get_permuted_code(code_b, order_b),
        local_checks="from_codes",
    )
    assert code == expected_code
    assert np.array_equal(code.code_a.generator, expected_code_a.generator)

    # The default C_B is ~C_A, whose coordinate j belongs to members_b[j].  Here members_b lists the
    # members of members_a in reverse order, so its members in increasing order are at [1, 2, 0, 3].
    code = codes.QTCode.from_ordered_members(
        members_a, members_a[::-1], code_a, local_checks="from_codes"
    )
    assert np.array_equal(code.code_b.generator, code_a.matrix[:, [1, 2, 0, 3]])

    # local codes given as matrices over another field
    matrix_a, matrix_b = np.array([[1, 2, 0, 1], [0, 1, 1, 2]]), np.array([[1, 2, 0]])
    code = codes.QTCode.from_ordered_members(members_a, members_b, matrix_a, matrix_b, field=3)
    expected_code = codes.QTCode(
        members_a, members_b, matrix_a[:, order_a], matrix_b[:, order_b], field=3
    )
    assert code.field.order == 3
    assert code == expected_code

    # Bipartite complexes work alike.  The members of A in increasing order are at [1, 2, 0], and
    # those of B at [1, 3, 0, 2].
    subset_a, subset_b = get_d5_subsets()
    members_a = [subset_a[2], subset_a[0], subset_a[1]]
    members_b = [subset_b[2], subset_b[0], subset_b[3], subset_b[1]]
    code_a, code_b = codes.ClassicalCode([[1, 1, 0]]), codes.ClassicalCode([[1, 1, 1, 0]])
    code = codes.QTCode.from_ordered_members(
        members_a, members_b, code_a, code_b, bipartite=True, local_checks="from_codes"
    )
    expected_code = codes.QTCode(
        members_a,
        members_b,
        get_permuted_code(code_a, [1, 2, 0]),
        get_permuted_code(code_b, [1, 3, 0, 2]),
        bipartite=True,
        local_checks="from_codes",
    )
    assert code.complex.bipartite
    assert code == expected_code

    # members must be distinct, even if they act on different numbers of points
    repeated_members = [shift, abstract.GroupMember(shift.array_form, size=7), ~shift]
    with pytest.raises(ValueError, match="members_a repeats a member"):
        codes.QTCode.from_ordered_members(repeated_members, members_b, [[1, 1, 1]], code_b)
    with pytest.raises(ValueError, match="C_B has block length 4, but members_b has 3"):
        codes.QTCode.from_ordered_members(members_a[:3], members_b[:3], code_a, code_b)


# The supports of the rows of the check matrices HX_C2C2_144_12_11.mtx and HZ_C2C2_144_12_11.mtx of
# the [[144, 12, 11]] code in the ancillary files of arXiv:2512.20532, with nine qubits per row.
# The files index qubit (i, j, h) by ``36 h + 6 i + j``, where the members h of C2 x C2 are 0, 1, 2,
# and 3 under bitwise XOR.
LRZ_144_12_11_SUPPORTS_X = (
    "1 2 3 25 26 27 31 32 33 0 2 4 24 26 28 30 32 34 0 1 5 24 25 29 30 31 35 7 8 9 19 20 21 31 "
    "32 33 6 8 10 18 20 22 30 32 34 6 7 11 18 19 23 30 31 35 13 14 15 19 20 21 25 26 27 12 14 "
    "16 18 20 22 24 26 28 12 13 17 18 19 23 24 25 29 37 38 39 61 62 63 67 68 69 36 38 40 60 62 "
    "64 66 68 70 36 37 41 60 61 65 66 67 71 43 44 45 55 56 57 67 68 69 42 44 46 54 56 58 66 68 "
    "70 42 43 47 54 55 59 66 67 71 49 50 51 55 56 57 61 62 63 48 50 52 54 56 58 60 62 64 48 49 "
    "53 54 55 59 60 61 65 73 74 75 97 98 99 103 104 105 72 74 76 96 98 100 102 104 106 72 73 77 "
    "96 97 101 102 103 107 79 80 81 91 92 93 103 104 105 78 80 82 90 92 94 102 104 106 78 79 83 "
    "90 91 95 102 103 107 85 86 87 91 92 93 97 98 99 84 86 88 90 92 94 96 98 100 84 85 89 90 91 "
    "95 96 97 101 109 110 111 133 134 135 139 140 141 108 110 112 132 134 136 138 140 142 108 "
    "109 113 132 133 137 138 139 143 115 116 117 127 128 129 139 140 141 114 116 118 126 128 "
    "130 138 140 142 114 115 119 126 127 131 138 139 143 121 122 123 127 128 129 133 134 135 "
    "120 122 124 126 128 130 132 134 136 120 121 125 126 127 131 132 133 137 1 13 29 39 51 77 "
    "89 97 135 0 12 28 29 76 77 88 89 96 0 1 12 13 38 50 96 97 134 7 13 21 45 51 55 83 89 131 6 "
    "12 54 82 83 88 89 130 131 6 7 12 13 20 44 50 54 55 21 29 55 71 97 105 131 135 139 28 29 54 "
    "70 71 96 130 131 138 20 54 55 96 97 104 134 138 139 3 15 37 49 65 99 113 125 133 36 48 64 "
    "65 112 113 124 125 132 2 14 36 37 48 49 98 132 133 9 15 19 43 49 57 95 119 125 18 42 48 94 "
    "95 118 119 124 125 8 14 18 19 42 43 48 49 56 19 35 57 65 95 99 103 133 141 18 34 35 64 65 "
    "94 95 102 132 18 19 56 98 102 103 132 133 140 5 17 25 63 73 85 101 111 123 4 5 16 17 24 72 "
    "84 100 101 24 25 62 72 73 84 85 110 122 11 17 59 79 85 93 117 123 127 10 11 16 17 58 59 78 "
    "84 126 78 79 84 85 92 116 122 126 127 25 33 59 63 67 93 101 127 143 24 58 59 66 100 101 "
    "126 142 143 24 25 32 62 66 67 92 126 127 27 41 53 61 75 87 109 121 137 40 41 52 53 60 108 "
    "120 136 137 26 60 61 74 86 108 109 120 121 23 47 53 81 87 91 115 121 129 22 23 46 47 52 53 "
    "90 114 120 80 86 90 91 114 115 120 121 128 23 27 31 61 69 91 107 129 137 22 23 30 60 90 "
    "106 107 136 137 26 30 31 60 61 68 90 91 128"
)
LRZ_144_12_11_SUPPORTS_Z = (
    "6 10 11 54 58 59 138 142 143 7 9 11 55 57 59 139 141 143 8 9 10 56 57 58 140 141 142 0 4 5 "
    "96 100 101 138 142 143 1 3 5 97 99 101 139 141 143 2 3 4 98 99 100 140 141 142 0 4 5 6 10 "
    "11 12 16 17 1 3 5 7 9 11 13 15 17 2 3 4 8 9 10 14 15 16 18 22 23 42 46 47 102 106 107 19 "
    "21 23 43 45 47 103 105 107 20 21 22 44 45 46 104 105 106 36 40 41 102 106 107 132 136 137 "
    "37 39 41 103 105 107 133 135 137 38 39 40 104 105 106 134 135 136 36 40 41 42 46 47 48 52 "
    "53 37 39 41 43 45 47 49 51 53 38 39 40 44 45 46 50 51 52 66 70 71 78 82 83 126 130 131 67 "
    "69 71 79 81 83 127 129 131 68 69 70 80 81 82 128 129 130 24 28 29 66 70 71 72 76 77 25 27 "
    "29 67 69 71 73 75 77 26 27 28 68 69 70 74 75 76 72 76 77 78 82 83 84 88 89 73 75 77 79 81 "
    "83 85 87 89 74 75 76 80 81 82 86 87 88 30 34 35 90 94 95 114 118 119 31 33 35 91 93 95 115 "
    "117 119 32 33 34 92 93 94 116 117 118 30 34 35 60 64 65 108 112 113 31 33 35 61 63 65 109 "
    "111 113 32 33 34 62 63 64 110 111 112 108 112 113 114 118 119 120 124 125 109 111 113 115 "
    "117 119 121 123 125 110 111 112 116 117 118 122 123 124 6 12 18 44 50 56 82 88 94 7 13 19 "
    "44 45 50 51 56 57 45 51 57 82 83 88 89 94 95 0 12 24 38 50 62 76 88 100 1 13 25 38 39 50 "
    "51 62 63 39 51 63 76 77 88 89 100 101 0 6 30 38 44 68 76 82 106 1 7 31 38 39 44 45 68 69 "
    "39 45 69 76 77 82 83 106 107 8 14 20 42 48 54 118 124 130 8 9 14 15 20 21 43 49 55 9 15 21 "
    "118 119 124 125 130 131 2 14 26 36 48 60 112 124 136 2 3 14 15 26 27 37 49 61 3 15 27 112 "
    "113 124 125 136 137 2 8 32 36 42 66 112 118 142 2 3 8 9 32 33 37 43 67 3 9 33 112 113 118 "
    "119 142 143 10 16 22 78 84 90 116 122 128 79 85 91 116 117 122 123 128 129 10 11 16 17 22 "
    "23 117 123 129 4 16 28 72 84 96 110 122 134 73 85 97 110 111 122 123 134 135 4 5 16 17 28 "
    "29 111 123 135 4 10 34 72 78 102 110 116 140 73 79 103 110 111 116 117 140 141 4 5 10 11 "
    "34 35 111 117 141 46 52 58 80 86 92 114 120 126 80 81 86 87 92 93 115 121 127 46 47 52 53 "
    "58 59 81 87 93 40 52 64 74 86 98 108 120 132 74 75 86 87 98 99 109 121 133 40 41 52 53 64 "
    "65 75 87 99 40 46 70 74 80 104 108 114 138 74 75 80 81 104 105 109 115 139 40 41 46 47 70 "
    "71 75 81 105"
)


def get_lrz_144_12_11_data() -> tuple[
    abstract.Group,
    list[abstract.GroupMember],
    list[abstract.GroupMember],
    list[codes.ClassicalCode],
]:
    """The group, lists of members, and codes C_0, C_1, C'_0, and C'_1 of an LRZ code.

    The code is the [[144, 12, 11]] code of arXiv:2512.20532 (LRZ), whose check matrices are in
    LRZ_144_12_11_SUPPORTS_X and LRZ_144_12_11_SUPPORTS_Z.  The members 1, 2, and 3 of C2 x C2 in
    those matrices are x, y, and xy here.  The lists of members and the codes were read off these
    matrices.  The codes C_0 = C'_0 carry the matrices H and G of the [6, 3, 3] code in Section 2 of
    the paper, and C_1 = C'_1 is C_0 with its coordinates shifted cyclically by one.
    """
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    shift_x, shift_y = group.generators
    identity = group.identity
    members_a = [identity, identity, identity, shift_x, shift_y, shift_x * shift_y]
    members_b = [identity, identity, shift_x, shift_x, shift_y, shift_y]
    matrix = np.array([[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
    generator = np.array([[0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])
    code_0 = codes.ClassicalCode(matrix).set_generator(generator)
    code_1 = codes.ClassicalCode(np.roll(matrix, -1, axis=1))
    code_1.set_generator(np.roll(generator, -1, axis=1))
    return group, members_a, members_b, [code_0, code_1, code_0, code_1]


def get_lrz_checks(
    group: abstract.Group,
    members_a: list[abstract.GroupMember],
    members_b: list[abstract.GroupMember],
    lrz_codes: list[codes.ClassicalCode],
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """The check matrices H_X and H_Z of Eq. (2) of arXiv:2512.20532, built as in that paper.

    The codes are C_0, C_1, C'_0, and C'_1, with parity check matrices H = code.matrix and generator
    matrices G = code.generator.  The members of the group, in increasing order, index the rows and
    columns of the permutation matrices ``(λ_a)_{g,h} = δ_{h,ag}`` and
    ``(ρ_b)_{g,h} = δ_{h,gb^-1}``.
    """
    field = lrz_codes[0].field
    members = sorted(group.generate())
    positions = {member: position for position, member in enumerate(members)}

    def get_lift(member: abstract.GroupMember, right: bool) -> npt.NDArray[np.int_]:
        """The permutation matrix λ_a of a member a, or ρ_b of a member b."""
        lift = np.zeros((len(members), len(members)), dtype=int)
        for row, element in enumerate(members):
            lift[row, positions[element * ~member if right else member * element]] = 1
        return lift

    lift_a = field(
        scipy.linalg.block_diag(*[get_lift(aa, False) for aa in members_a for _ in members_b])
    )
    lift_b = field(
        scipy.linalg.block_diag(*[get_lift(bb, True) for _ in members_a for bb in members_b])
    )
    identity = field.Identity(len(members))
    h_0, h_1, h_0_prime, h_1_prime = [code.matrix for code in lrz_codes]
    g_0, g_1, g_0_prime, g_1_prime = [code.generator for code in lrz_codes]
    matrix_x = np.vstack(
        [
            np.kron(np.kron(h_0, g_0_prime), identity),
            np.kron(np.kron(h_1, g_1_prime), identity) @ lift_a @ lift_b,
        ]
    )
    matrix_z = np.vstack(
        [
            np.kron(np.kron(g_0, h_1_prime), identity) @ lift_b,
            np.kron(np.kron(g_1, h_0_prime), identity) @ lift_a,
        ]
    )
    return matrix_x.view(field), matrix_z.view(field)


def test_lifted_quantum_tanner_lrz_published() -> None:
    """LQTCode.from_lrz rebuilds the [[144, 12, 11]] code of arXiv:2512.20532, and its checks.

    With local_checks="from_codes", the checks are the rows of the published check matrices, in
    which each qubit is in at most 10 checks.
    """
    group, members_a, members_b, lrz_codes = get_lrz_144_12_11_data()
    lrz_qt_codes = {
        local_checks: codes.LQTCode.from_lrz(
            group,
            members_a,
            members_b,
            code_0=lrz_codes[0],
            code_1=lrz_codes[1],
            code_0_prime=lrz_codes[2],
            code_1_prime=lrz_codes[3],
            local_checks=local_checks,
        )
        for local_checks in LOCAL_CHECKS
    }
    code = lrz_qt_codes["from_codes"]
    assert (len(code), code.dimension, code.get_weight()) == (144, 12, 9)
    default_code = codes.LQTCode.from_lrz(
        group,
        members_a,
        members_b,
        code_0=lrz_codes[0],
        code_1=lrz_codes[1],
        code_0_prime=lrz_codes[2],
        code_1_prime=lrz_codes[3],
    )
    assert np.array_equal(default_code.matrix, lrz_qt_codes["min_weight"].matrix)
    assert not np.array_equal(default_code.matrix, code.matrix)

    # read the qubit (i, j, g) of the code as the published qubit 36 h + 6 i + j, where h labels g
    identity, shift_x, shift_y = group.identity, *group.generators
    labels = {member: h for h, member in enumerate([identity, shift_x, shift_y, shift_x * shift_y])}
    columns = [
        36 * labels[member] + 6 * ii + jj
        for ii, jj, member in itertools.product(range(6), range(6), sorted(group.generate()))
    ]
    sectors = [
        (code.matrix_x, lrz_qt_codes["min_weight"].matrix_x, LRZ_144_12_11_SUPPORTS_X),
        (code.matrix_z, lrz_qt_codes["min_weight"].matrix_z, LRZ_144_12_11_SUPPORTS_Z),
    ]
    for matrix, min_weight_matrix, supports in sectors:
        published_matrix = np.zeros((72, 144), dtype=int)
        np.put_along_axis(
            published_matrix, np.array(supports.split(), dtype=int).reshape(72, 9), 1, 1
        )
        relabeled_matrix = np.zeros_like(published_matrix)
        relabeled_matrix[:, columns] = matrix.view(np.ndarray)
        assert get_sorted_rows(relabeled_matrix) == get_sorted_rows(published_matrix)
        assert_same_row_space(min_weight_matrix, matrix)
    checks = np.vstack([code.matrix_x, code.matrix_z]).view(np.ndarray)
    assert np.max(np.count_nonzero(checks, axis=0)) == 10


def test_lifted_quantum_tanner_lrz(pytestconfig: pytest.Config) -> None:
    """LQTCode.from_lrz builds the code of Eq. (2) of arXiv:2512.20532 from the given bases."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    def get_random_codes(length_a: int, length_b: int, field: int) -> list[codes.ClassicalCode]:
        """Random codes C_0 and C_1 of length length_a, and C'_0 and C'_1 of length length_b."""
        return [
            codes.ClassicalCode.random(length, length // 2, field, seed=int(rng.integers(2**31)))
            for length in [length_a, length_a, length_b, length_b]
        ]

    # S3, a nonabelian group, and one of its transpositions, which generates a proper subgroup
    symmetric_group = abstract.SymmetricGroup(3)
    transposition = abstract.GroupMember([1, 0, 2])
    random_members = [symmetric_group.random(seed=int(rng.integers(2**31))) for _ in range(5)]

    # the cyclic group of order 2, which is shorter than the list A that repeats its identity
    parity_group = abstract.CyclicGroup(2)
    parity_identity, flip = parity_group.identity, parity_group.generators[0]

    cases: list[
        tuple[
            abstract.Group,
            list[abstract.GroupMember],
            list[abstract.GroupMember],
            list[codes.ClassicalCode],
        ]
    ] = [
        (symmetric_group, random_members[:3], random_members[3:], get_random_codes(3, 2, 3)),
        (
            symmetric_group,
            [transposition, transposition, symmetric_group.identity],
            [symmetric_group.identity, transposition],
            get_random_codes(3, 2, 2),
        ),
        (
            parity_group,
            [parity_identity, parity_identity, parity_identity, flip],
            [parity_identity, flip],
            get_random_codes(4, 2, 2),
        ),
    ]
    for group, members_a, members_b, lrz_codes in cases:
        matrix_x, matrix_z = get_lrz_checks(group, members_a, members_b, lrz_codes)
        for local_checks in LOCAL_CHECKS:
            code = codes.LQTCode.from_lrz(
                group,
                members_a,
                members_b,
                code_0=lrz_codes[0],
                code_1=lrz_codes[1],
                code_0_prime=lrz_codes[2],
                code_1_prime=lrz_codes[3],
                local_checks=local_checks,
            )
            assert len(code) == len(members_a) * len(members_b) * group.order
            assert not np.any(code.matrix_x @ code.matrix_z.T)
            if local_checks == "from_codes":
                assert get_sorted_rows(code.matrix_x) == get_sorted_rows(matrix_x)
                assert get_sorted_rows(code.matrix_z) == get_sorted_rows(matrix_z)
            else:
                assert_same_row_space(code.matrix_x, matrix_x)
                assert_same_row_space(code.matrix_z, matrix_z)


def test_lifted_quantum_tanner_local_views() -> None:
    """The local view of a vertex (h, s, t) holds the squares (i, j, a_i^-s h b_j^-t).

    The checks of the vertices of each class are consecutive rows of matrix_x or matrix_z.  With
    "from_codes", they are the rows of products of the matrices that the local codes carry.  With
    "min_weight", the default, they form a minimum-weight basis of each local check space, and a
    brute-force search finds the weights of such a basis independently.  The generator matrix of
    C_A^1 is not a minimum-weight basis, so the two choices give different checks.
    """
    group = abstract.SymmetricGroup(3)
    rotation, transposition = abstract.GroupMember([1, 2, 0]), abstract.GroupMember([1, 0, 2])
    members_a = [transposition, transposition, rotation]
    members_b = [rotation, group.identity]
    code_a1 = codes.ClassicalCode([[1, 1, 0]]).set_generator([[1, 1, 1], [0, 0, 1]])
    codes_a = [codes.HammingCode(2), code_a1]
    codes_b = [codes.RepetitionCode(2), codes.ClassicalCode([[1, 0]])]
    lifted_codes = {
        local_checks: codes.LQTCode(
            group,
            members_a,
            members_b,
            codes_a[0],
            codes_b[0],
            code_a1=codes_a[1],
            code_b1=codes_b[1],
            local_checks=local_checks,
        )
        for local_checks in LOCAL_CHECKS
    }
    default_code = codes.LQTCode(
        group, members_a, members_b, codes_a[0], codes_b[0], code_a1=codes_a[1], code_b1=codes_b[1]
    )
    assert np.array_equal(default_code.matrix, lifted_codes["min_weight"].matrix)
    assert not np.array_equal(lifted_codes["min_weight"].matrix, lifted_codes["from_codes"].matrix)
    members = sorted(group.generate())

    for local_checks, code in lifted_codes.items():
        assert len(code) == 3 * 2 * 6
        sectors: list[tuple[PauliXZ, galois.FieldArray, list[tuple[int, int]]]] = [
            (Pauli.X, code.matrix_x, [(0, 0), (1, 1)]),
            (Pauli.Z, code.matrix_z, [(0, 1), (1, 0)]),
        ]
        for pauli, matrix, vertex_classes in sectors:
            local_views = code.get_local_views(pauli)
            assert local_views.shape == (2 * len(members), 3, 2)
            assert not local_views.flags.writeable
            vertices = [(hh, ss, tt) for ss, tt in vertex_classes for hh in members]
            first_row = 0
            for (hh, ss, tt), local_view in zip(vertices, local_views):
                # the qudit (i, j, g) at coordinate (i, j) has the index (2 i + j) |G| + r(g)
                for (ii, member_a), (jj, member_b) in itertools.product(
                    enumerate(members_a), enumerate(members_b)
                ):
                    square = (~member_a if ss else group.identity) * hh
                    square *= ~member_b if tt else group.identity
                    assert local_view[ii, jj] == (2 * ii + jj) * len(members) + members.index(
                        square
                    )

                # the local check space of the vertex is spanned by G_A^s ⨂ G_B^t or H_A^s ⨂ H_B^t
                local_space = get_local_space(codes_a[ss], codes_b[tt], pauli)
                if local_checks == "from_codes":
                    num_checks = len(local_space)
                else:
                    num_checks = np.linalg.matrix_rank(local_space) if local_space.size else 0

                # the checks of the vertex lie in its local view
                checks = matrix[first_row : first_row + num_checks]
                vertex_checks = checks[:, local_view.ravel()]
                assert np.count_nonzero(checks) == np.count_nonzero(vertex_checks)
                if local_checks == "from_codes":
                    assert np.array_equal(vertex_checks, local_space)
                else:
                    assert_same_row_space(vertex_checks, local_space)
                    weights = np.count_nonzero(vertex_checks.view(np.ndarray), axis=1)
                    assert sorted(weights.tolist()) == get_min_weights(local_space)
                first_row += num_checks
            assert first_row == len(matrix)


def get_squares_of_faces(
    code: codes.QTCode, members: list[abstract.GroupMember]
) -> list[list[int]]:
    """The squares of an LQTCode, as qudit indices, that are the faces of a QTCode.

    The face ``f(g,a,b)`` at a vertex g of class (0, 0) of a quadripartite complex, or of class 0 of
    a bipartite complex, is the square (i, j, g) whose members of A and B are a and b.  In a
    bipartite complex, this face is also ``f(agb,a^-1,b^-1)``, and so two squares.  The members of
    the group, in increasing order, give the squares their indices.
    """
    num_points = members[0].size
    subgraph_x, _ = codes.QTCode.get_subgraphs(code.complex)
    cover_a, cover_b = sorted(code.complex.cover_subset_a), sorted(code.complex.cover_subset_b)
    squares: list[list[int]] = [[] for _ in range(len(code))]
    for vertex in subgraph_x:
        # the vertices of class (0, 0) or 0 are those that fix the points of the covering factors
        if not subgraph_x.out_degree(vertex):
            continue
        image = vertex.array_form
        if image[num_points:] == sorted(image[num_points:]):
            position = members.index(abstract.GroupMember(image[:num_points]))
            for _, face, (member_a, member_b) in subgraph_x.out_edges(vertex, data="sort"):
                label = cover_a.index(member_a) * len(cover_b) + cover_b.index(member_b)
                squares[subgraph_x.nodes[face]["sort"]].append(label * len(members) + position)
    return squares


def test_lifted_quantum_tanner_qtcode() -> None:
    """QTCodes have the stabilizer groups of LQTCodes with C^1 = C^0∘ι, or of their quotients.

    The quotients are by an involution σ.  The quadripartite complex of the code over D5 is
    connected, so σ maps it onto itself.  That of the code over D6 has two components, which σ
    exchanges.  With local_checks="from_codes", and local codes C^1 that carry the matrices of C^0
    with permuted columns, the parity checks correspond as well.
    """
    rotation = abstract.GroupMember([2, 3, 4, 5, 0, 1])
    reflections = [[1, 0, 5, 4, 3, 2], [3, 2, 1, 0, 5, 4], [5, 4, 3, 2, 1, 0]]
    cases: list[
        tuple[
            list[abstract.GroupMember],
            list[abstract.GroupMember],
            codes.ClassicalCode | None,
            bool,
        ]
    ] = [
        (*get_d5_subsets(), codes.ClassicalCode([[1, 1, 1, 0]]), True),
        (
            [abstract.GroupMember(image) for image in reflections],
            [abstract.GroupMember(list(range(6))), rotation, ~rotation],
            None,
            False,
        ),
    ]
    for subset_a, subset_b, code_b, connected in cases:
        qt_codes = [
            codes.QTCode(
                subset_a,
                subset_b,
                codes.HammingCode(2),
                code_b,
                bipartite=bipartite,
                local_checks="from_codes",
            )
            for bipartite in [False, True]
        ]
        quadripartite_code, bipartite_code = qt_codes

        # the codes on the two complexes differ, even when they have the same parity checks, as over
        # D6, where σ maps the component of the identity onto another component
        assert quadripartite_code != bipartite_code
        if not connected:
            assert np.array_equal(quadripartite_code.matrix, bipartite_code.matrix)

        # the LQTCode on all of the group that A and B generate, with C^1 = C^0∘ι
        members_a = list(quadripartite_code.members_a)
        members_b = list(quadripartite_code.members_b)
        local_codes = []
        for members, local_code in [
            (members_a, quadripartite_code.code_a),
            (members_b, quadripartite_code.code_b),
        ]:
            inversion = [members.index(~member) for member in members]
            local_codes.append((local_code, get_permuted_code(local_code, inversion)))
        generated_group = abstract.Group(*members_a, *members_b)
        lifted_code = codes.LQTCode(
            generated_group,
            members_a,
            members_b,
            local_codes[0][0],
            local_codes[1][0],
            code_a1=local_codes[0][1],
            code_b1=local_codes[1][1],
            local_checks="from_codes",
        )
        num_points = members_a[0].size
        members = sorted(
            abstract.GroupMember(member.array_form, size=num_points)
            for member in generated_group.generate()
        )
        assert (len(lifted_code) == len(quadripartite_code)) == connected
        assert lifted_code.dimension == (1 if connected else 2) * quadripartite_code.dimension
        assert len(bipartite_code) == len(quadripartite_code) // (2 if connected else 1)

        # Fold the squares onto the faces of each QTCode.  Each check of the LQTCode lies on these
        # squares or off them, and those on them fold onto the checks of the QTCode, each of which
        # appears once for each square of a face.
        for qt_code in qt_codes:
            squares_of_faces = get_squares_of_faces(qt_code, members)
            num_copies = 2 if qt_code.complex.bipartite else 1
            assert all(len(squares) == num_copies for squares in squares_of_faces)
            faces = {
                square: face for face, squares in enumerate(squares_of_faces) for square in squares
            }
            for matrix, qt_matrix in [
                (lifted_code.matrix_x, qt_code.matrix_x),
                (lifted_code.matrix_z, qt_code.matrix_z),
            ]:
                folded_checks = []
                for check in matrix.view(np.ndarray):
                    support = set(np.flatnonzero(check))
                    if support <= faces.keys():
                        folded_check = np.zeros(len(qt_code), dtype=int)
                        for square in support:
                            folded_check[faces[int(square)]] += check[square]
                        folded_checks.append(folded_check)
                    else:
                        assert not support & faces.keys()
                assert get_sorted_rows(folded_checks) == sorted(
                    get_sorted_rows(qt_matrix) * num_copies
                )


def get_s3_lifted_qt_code(exchange_classes: bool = False) -> codes.LQTCode:
    """An LQTCode over S3 whose lists repeat members, with a group of 6 translations.

    The list A holds each transposition twice, in adjacent positions.  Conjugation by any member of
    S3 permutes A, and since it matches repeated members in order, it maps the first and second
    positions of each pair to the first and second positions of a pair.  So it maps C_A^0, whose
    words have equal entries at the first positions of the pairs, onto itself.  The words of C_A^1
    have equal entries on the first pair, and even weight on the other four positions, so only
    conjugation by the identity and by the transposition of the first pair, which exchanges the
    other two transpositions, maps C_A^1 onto itself.  Conjugation by any member of S3 permutes
    B = (e, r, r^-1): rotations fix each member, and transpositions exchange r and r^-1.  So all
    conjugations map C_B^0, whose words have equal entries at positions 1 and 2, onto itself, but
    only those by rotations map C_B^1, whose words have a zero at position 1, onto itself.  So the
    translations from the left and from the right form groups of orders 2 and 3, which commute, and
    generate a group of order 6.

    If exchange_classes is True, the local codes of classes 0 and 1 are exchanged on each side,
    which does not change the translations.
    """
    group = abstract.SymmetricGroup(3)
    rotation = abstract.GroupMember([1, 2, 0])
    transpositions = [abstract.GroupMember(image) for image in [[1, 0, 2], [2, 1, 0], [0, 2, 1]]]
    codes_a = [[[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]], [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]]]
    codes_b = [[[0, 1, 1]], [[0, 1, 0]]]
    if exchange_classes:
        codes_a, codes_b = codes_a[::-1], codes_b[::-1]
    return codes.LQTCode(
        group,
        [member for member in transpositions for _ in range(2)],
        [group.identity, rotation, ~rotation],
        codes_a[0],
        codes_b[0],
        code_a1=codes_a[1],
        code_b1=codes_b[1],
    )


def test_lifted_quantum_tanner_translations() -> None:
    """Translations of the complex preserve an LQTCode, matching repeated members in order.

    Exchanging the local codes of classes 0 and 1 checks that translations need to preserve both.
    """
    for exchange_classes in [False, True]:
        code = get_s3_lifted_qt_code(exchange_classes)
        translations = code._get_permutation_automorphisms()
        assert translations is not None
        for translation in translations:
            for matrix in [code.matrix_x, code.matrix_z]:
                assert_same_row_space(matrix[:, np.argsort(translation)], matrix)
        group = abstract.Group(*[abstract.GroupMember(row.tolist()) for row in translations])
        assert group.order == 6


def test_lifted_quantum_tanner_translations_of_order_3() -> None:
    """Translations by members whose conjugation permutes A and B in 3-cycles preserve an LQTCode.

    Conjugation by a 3-cycle of A4 permutes its three double transpositions cyclically, and every
    permutation of coordinates maps a repetition code onto itself.  A 3-cycle differs from its
    inverse, so this code fixes the direction of conjugation on both sides, which codes whose
    conjugations are their own inverses do not.
    """
    group = abstract.AlternatingGroup(4)
    images = [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    members = [abstract.GroupMember(image) for image in images]
    code = codes.LQTCode(group, members, members, codes.RepetitionCode(3), codes.RepetitionCode(3))
    translations = code._get_permutation_automorphisms()
    assert translations is not None
    for translation in translations:
        for matrix in [code.matrix_x, code.matrix_z]:
            assert_same_row_space(matrix[:, np.argsort(translation)], matrix)


def test_lifted_quantum_tanner_canonical_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bliss searches the parity checks, and the light words of the local check space of each class.

    In this code, the local check spaces of the vertices with X-type checks have 256 words, and
    those of the vertices with Z-type checks have 4 words.
    """
    code = get_s3_lifted_qt_code()
    codes_a, codes_b = [code.code_a0, code.code_a1], [code.code_b0, code.code_b1]
    num_qudits = len(code)
    search_matrix = code._get_canonical_checks()
    search_matrices = code._get_permutation_automorphism_search_matrices()
    assert len(search_matrices) == 2
    for matrix in [code.matrix, search_matrix]:
        assert any(np.array_equal(matrix, other) for other in search_matrices)
    is_x_check = np.any(search_matrix[:, :num_qudits], axis=1)
    assert not np.any(search_matrix[is_x_check, num_qudits:])
    sectors: list[tuple[PauliXZ, galois.FieldArray, list[tuple[int, int]]]] = [
        (Pauli.X, search_matrix[is_x_check, :num_qudits], [(0, 0), (1, 1)]),
        (Pauli.Z, search_matrix[~is_x_check, num_qudits:], [(0, 1), (1, 0)]),
    ]
    for pauli, checks, vertex_classes in sectors:
        # every vertex has the light words of the local check space of its class on its local view,
        # and the vertices of each class are half of the vertices with checks of the given type
        expected_checks = []
        class_local_views = np.split(code.get_local_views(pauli), 2)
        for (ss, tt), local_views in zip(vertex_classes, class_local_views):
            light_words = get_light_words(get_local_space(codes_a[ss], codes_b[tt], pauli))
            expected_checks += place_words(light_words, local_views, num_qudits)
        assert get_sorted_rows(checks) == sorted(expected_checks)
        assert_same_row_space(checks, code.matrix_x if pauli is Pauli.X else code.matrix_z)

    # if the local check spaces of the vertices with X-type checks are too large, the X-type parity
    # checks replace the canonical X-type checks
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_CANONICAL_CHECKS", 16)
    search_matrix = code._get_canonical_checks()
    num_x_checks = len(code.matrix_x)
    assert np.array_equal(search_matrix[:num_x_checks, :num_qudits], code.matrix_x)
    assert_same_row_space(search_matrix[num_x_checks:, num_qudits:], code.matrix_z)
    assert len(search_matrix) - num_x_checks > len(code.matrix_z)

    # if they replace all canonical checks, bliss searches only the parity checks
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_CANONICAL_CHECKS", 0)
    search_matrices = code._get_permutation_automorphism_search_matrices()
    assert len(search_matrices) == 1
    assert np.array_equal(search_matrices[0], code.matrix)


def test_lifted_quantum_tanner_canonical_checks_of_one_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One X-type class over a cap makes parity checks replace all canonical X-type checks.

    The local check spaces of the vertices of classes (0, 0) and (1, 1) have 8 and 4 words, so under
    a cap of 4 words, only class (0, 0) exceeds it, but the X-type parity checks replace the
    canonical X-type checks of both classes.  Class (1, 1) has 3 canonical checks at each vertex but
    2 parity checks, so the assertion fails if only the checks of class (0, 0) are replaced.
    """
    group = abstract.CyclicGroup(2)
    code = codes.LQTCode(
        group,
        [group.identity] * 3,
        [group.identity],
        np.zeros((0, 3), dtype=int),
        np.zeros((0, 1), dtype=int),
        code_a1=[[1, 1, 1]],
    )
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_CANONICAL_CHECKS", 4)
    search_matrix = code._get_canonical_checks()
    x_block = search_matrix[:, : len(code)]
    assert np.array_equal(x_block[np.any(x_block, axis=1)], code.matrix_x)


def test_lifted_quantum_tanner_generating_data() -> None:
    """LQTCode reads and validates its generating data."""
    group = abstract.CyclicGroup(4)
    shift = group.generators[0]
    code_a = codes.RepetitionCode(2)

    # members are read on as many points as the group acts on, keeping repeated members
    padded_shift = abstract.GroupMember(shift.array_form + [4])
    code = codes.LQTCode(group, [shift, padded_shift], [shift, ~shift], code_a, [[1, 0]])
    assert code.members_a == (shift, shift)

    # the local codes of class 1 default to those of class 0
    assert code.code_a1 == code.code_a0
    assert code.code_b1 == code.code_b0
    assert code == codes.LQTCode(
        group, [shift, shift], [shift, ~shift], code_a, [[1, 0]], code_a1=code_a, code_b1=[[1, 0]]
    )

    # either list, or both, may be empty, which leaves no qudits
    no_bits = codes.ClassicalCode(np.zeros((0, 0), dtype=int))
    one_bit = codes.ClassicalCode([[1]])
    empty_cases: list[
        tuple[
            list[abstract.GroupMember],
            list[abstract.GroupMember],
            codes.ClassicalCode,
            codes.ClassicalCode,
        ]
    ] = [
        ([], [shift], no_bits, one_bit),
        ([shift], [], one_bit, no_bits),
        ([], [], no_bits, no_bits),
    ]
    for members_a, members_b, code_a0, code_b0 in empty_cases:
        code = codes.LQTCode(group, members_a, members_b, code_a0, code_b0)
        assert len(code) == 0
        assert code.matrix_x.shape == code.matrix_z.shape == (0, 0)
        for pauli in PAULIS_XZ:
            assert code.get_local_views(pauli).shape == (8, len(members_a), len(members_b))
        assert code._get_canonical_checks().shape == (0, 0)
        assert code.get_known_permutation_automorphism_group().order == 1

    # invalid generating data
    swap = abstract.GroupMember([1, 0])
    with pytest.raises(ValueError, match="members_b contains"):
        codes.LQTCode(group, [shift], [swap], [[1]], [[1]])
    with pytest.raises(ValueError, match=re.escape(f"members_a contains {shift},")):
        codes.LQTCode.from_lrz(
            abstract.Group(shift**2),
            [shift],
            [shift],
            code_0=[[1]],
            code_1=[[1]],
            code_0_prime=[[1]],
            code_1_prime=[[1]],
        )
    with pytest.raises(ValueError, match="C_A\\^1 has block length 3, but members_a has 2"):
        codes.LQTCode(group, [shift, shift], [shift], code_a, [[1]], code_a1=[[1, 1, 1]])
    # LQTCode.from_lrz leaves length errors to LQTCode, whose C_A^s is ~C_s and C_B^t is C'_t
    for wrong_index, name in enumerate(["C_A\\^0", "C_A\\^1", "C_B\\^0", "C_B\\^1"]):
        members_name = "members_a" if wrong_index < 2 else "members_b"
        matrices = [[[1] * (2 if index == wrong_index else 1)] for index in range(4)]
        with pytest.raises(
            ValueError, match=f"{name} has block length 2, but {members_name} has 1 members"
        ):
            codes.LQTCode.from_lrz(
                group,
                [shift],
                [shift],
                code_0=matrices[0],
                code_1=matrices[1],
                code_0_prime=matrices[2],
                code_1_prime=matrices[3],
            )
    with pytest.raises(ValueError, match="different fields"):
        codes.LQTCode(group, [shift], [shift], [[1]], codes.ClassicalCode([[1]], field=3))
    for local_checks in ["row_reduced", None]:
        with pytest.raises(ValueError, match="Unrecognized choice of local checks"):
            codes.LQTCode(
                group,
                [shift],
                [shift],
                [[1]],
                [[1]],
                local_checks=local_checks,  # type: ignore[arg-type]
            )


def test_lifted_quantum_tanner_matrices_over_other_fields() -> None:
    """LQTCode and LQTCode.from_lrz read parity check matrices over a given field."""
    group = abstract.CyclicGroup(3)
    shift = group.generators[0]
    members_a = [shift, shift, group.identity]
    members_b = [shift, group.identity]
    matrices = [[[1, 2, 0], [0, 1, 1]], [[1, 1, 2]], [[1, 2]], [[2, 1]]]
    lrz_codes = [codes.ClassicalCode(matrix, field=3) for matrix in matrices]
    matrix_x, matrix_z = get_lrz_checks(group, members_a, members_b, lrz_codes)

    # from_lrz builds the code of Eq. (2) from the given parity check matrices of C_0, C_1, C'_0,
    # and C'_1, and their computed generator matrices
    code = codes.LQTCode.from_lrz(
        group,
        members_a,
        members_b,
        code_0=matrices[0],
        code_1=matrices[1],
        code_0_prime=matrices[2],
        code_1_prime=matrices[3],
        field=3,
        local_checks="from_codes",
    )
    assert code.field.order == 3
    assert get_sorted_rows(code.matrix_x) == get_sorted_rows(matrix_x)
    assert get_sorted_rows(code.matrix_z) == get_sorted_rows(matrix_z)

    # the local code C_A^s is the dual of C_s, with the parity check and generator matrices of C_s
    # exchanged
    for local_code, lrz_code, matrix in [
        (code.code_a0, lrz_codes[0], matrices[0]),
        (code.code_a1, lrz_codes[1], matrices[1]),
    ]:
        assert np.array_equal(local_code.matrix, lrz_code.generator)
        assert np.array_equal(local_code.generator, code.field(matrix))

    # the constructor reads matrices over the given field in the same way
    code = codes.LQTCode(
        group,
        [~member for member in members_a],
        members_b,
        lrz_codes[0].generator.tolist(),
        matrices[2],
        field=3,
        code_a1=lrz_codes[1].generator.tolist(),
        code_b1=matrices[3],
    )
    assert code.field.order == 3
    assert_same_row_space(code.matrix_x, matrix_x)
    assert_same_row_space(code.matrix_z, matrix_z)


def test_quantum_tanner_min_weight_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum-weight local checks need local codes and their duals to have few enough words."""
    group = abstract.CyclicGroup(4)
    shift = group.generators[0]
    subset = [shift, ~shift, shift**2]
    repetition_code = codes.RepetitionCode(3)  # 2 words, and its dual has 4
    even_code = codes.ClassicalCode([[1, 1, 1]])  # 4 words, and its dual has 2
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_MIN_WEIGHT_BASIS", 3)
    for local_code in [repetition_code, even_code]:
        with pytest.raises(ValueError, match="local_checks='from_codes'"):
            codes.QTCode(subset, subset, local_code, local_code)
        with pytest.raises(ValueError, match="local_checks='from_codes'"):
            codes.LQTCode(group, subset, subset, local_code, local_code)
    codes.QTCode(subset, subset, repetition_code, even_code, local_checks="from_codes")
    codes.LQTCode(group, subset, subset, repetition_code, even_code, local_checks="from_codes")
    monkeypatch.setattr(quantum_tanner, "_MAX_WORDS_FOR_MIN_WEIGHT_BASIS", 4)
    codes.QTCode(subset, subset, repetition_code, even_code)
    codes.LQTCode(group, subset, subset, repetition_code, even_code)

    # the local codes of class 1 of an LQTCode are checked too
    full_code = np.zeros((0, 3), dtype=int)  # 8 words, and its dual has 1
    with pytest.raises(ValueError, match="local_checks='from_codes'"):
        codes.LQTCode(group, subset, subset, repetition_code, even_code, code_a1=full_code)
