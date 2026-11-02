# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical forms of products of π/2 rotations, using stim tableaus as the oracle."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc.experimental.objects import _packed_paulis as packed
from qldpc.experimental.objects import _symplectic as symplectic
from qldpc.experimental.objects.conftest import get_random_pauli_string

_KERNELS = (
    "canonicalize",
    "_row_reduce",
    "_get_bit",
    "_set_dual",
    "_get_image_of_dual",
    "_triangularize",
    "_restricted_entry",
    "_choose_vector",
    "_is_continuable",
)


@pytest.fixture(params=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled kernels, or their Python versions for coverage."""
    if request.param == "python":
        for name in _KERNELS:
            monkeypatch.setattr(symplectic, name, getattr(symplectic, name).py_func)


def _pack_axes(
    axes: list[stim.PauliString], num_qubits: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Packed words and phases of a list of Hermitian axes."""
    num_words = 2 * packed.get_num_words(num_qubits)
    words = np.zeros((len(axes), num_words), dtype=np.uint64)
    phases = np.zeros(len(axes), dtype=np.int64)
    for row, axis in enumerate(axes):
        words[row], phases[row] = packed.from_stim(axis, num_qubits)
    return words, phases


def _get_tableau(
    core: npt.NDArray[np.uint64],
    axes: npt.NDArray[np.uint64],
    phases: npt.NDArray[np.int64],
    num_qubits: int,
) -> stim.Tableau:
    """The stim tableau of ``S·P``, which determines it up to a global phase."""
    circuit = stim.Circuit()
    circuit.append("I", [num_qubits - 1])
    pauli = packed.to_stim(core, 0, num_qubits)
    for qubit in range(num_qubits):
        if pauli[qubit]:
            circuit.append("_XYZ"[pauli[qubit]], [qubit])
    for words, phase in zip(axes, phases):
        axis = packed.to_stim(words, int(phase), num_qubits)
        if not axis.weight:  # a rotation about the identity is a global phase
            continue
        gate = "SPP" if axis.sign == 1 else "SPP_DAG"
        circuit.append(gate, stim.target_combined_paulis(axis * axis.sign))
    return circuit.to_tableau()


def _get_symplectic_matrix(tableau: stim.Tableau) -> npt.NDArray[np.uint8]:
    """The symplectic matrix of a tableau, whose columns are the images of X_0, ..., Z_0, ..."""
    num_qubits = len(tableau)
    outputs = [tableau.x_output(qubit) for qubit in range(num_qubits)]
    outputs += [tableau.z_output(qubit) for qubit in range(num_qubits)]
    return np.array([np.concatenate(output.to_numpy()) for output in outputs], dtype=np.uint8).T


def _get_rank(matrix: npt.NDArray[np.uint8]) -> int:
    """The rank of a binary matrix over GF(2)."""
    matrix = matrix.copy() % 2
    rank = 0
    for column in range(matrix.shape[1]):
        rows = [row for row in range(rank, len(matrix)) if matrix[row, column]]
        if rows:
            matrix[[rank, rows[0]]] = matrix[[rows[0], rank]]
            matrix[(matrix[:, column] == 1) & (np.arange(len(matrix)) != rank)] ^= matrix[rank]
            rank += 1
    return rank


@pytest.mark.usefixtures("kernels")
def test_canonical_forms(rng: np.random.Generator) -> None:
    """Canonical forms equal their inputs, use at most rank(M - I) + 1 axes, and are canonical."""
    for num_qubits in (1, 2, 3, 65):
        for _ in range(6):
            axes = [get_random_pauli_string(rng, num_qubits) for _ in range(rng.integers(6))]
            words, phases = _pack_axes(axes, num_qubits)
            core = packed.from_stim(get_random_pauli_string(rng, num_qubits))[0]
            new_core, new_words, new_phases = symplectic.canonicalize(core, words, phases)
            tableau = _get_tableau(core, words, phases, num_qubits)
            assert _get_tableau(new_core, new_words, new_phases, num_qubits) == tableau
            identity = np.eye(2 * num_qubits, dtype=np.uint8)
            rank = _get_rank(_get_symplectic_matrix(tableau) ^ identity)
            assert len(new_words) - rank in (0, 1)
            assert all(
                packed.to_stim(*row, num_qubits).sign == 1 for row in zip(new_words, new_phases)
            )

            # another product of rotations for the same operator has the same canonical form
            extra = get_random_pauli_string(rng, num_qubits, nontrivial=True)
            position = rng.integers(len(axes) + 1)
            others = [*axes[:position], extra, -extra, *axes[position:], extra, extra, extra, extra]
            other_words, other_phases = _pack_axes(others, num_qubits)
            result = symplectic.canonicalize(core, other_words, other_phases)
            np.testing.assert_array_equal(result[0], new_core)
            np.testing.assert_array_equal(result[1], new_words)
            np.testing.assert_array_equal(result[2], new_phases)


def test_minimal_forms_on_two_qubits() -> None:
    """On two qubits, canonical forms use the fewest π/2 rotations possible, found by search."""
    letters = ("".join(pair) for pair in itertools.product("IXYZ", repeat=2))
    nontrivial = [stim.PauliString(pair) for pair in letters if pair != "II"]
    form = np.kron(np.array([[0, 1], [1, 0]], dtype=np.uint8), np.eye(2, dtype=np.uint8))
    transvections = []
    for axis in nontrivial:
        vector = np.concatenate(axis.to_numpy()).astype(np.uint8)
        transvections.append((np.eye(4, dtype=np.uint8) + np.outer(vector, vector @ form)) % 2)
    identity = np.eye(4, dtype=np.uint8)
    shortest: dict[bytes, list[stim.PauliString]] = {identity.tobytes(): []}
    frontier: list[tuple[npt.NDArray[np.uint8], list[stim.PauliString]]] = [(identity, [])]
    while frontier:  # breadth-first search over products of rotations, ignoring signs
        next_frontier = []
        for matrix, axes in frontier:
            for axis, transvection in zip(nontrivial, transvections):
                product = transvection @ matrix % 2
                if product.tobytes() not in shortest:
                    shortest[product.tobytes()] = [*axes, axis]
                    next_frontier.append((product, [*axes, axis]))
        frontier = next_frontier
    assert len(shortest) == 720  # the symplectic group Sp(4, 2)
    core = np.zeros(2, dtype=np.uint64)
    for axes in shortest.values():
        _, new_words, _ = symplectic.canonicalize(core, *_pack_axes(axes, 2))
        assert len(new_words) == len(axes)


def test_controlled_z_needs_an_extra_rotation() -> None:
    """CZ = R_{π/2}(Z0 Z1) R_{π/2}(-Z0) R_{π/2}(-Z1) has rank(M - I) = 2, but needs 3 rotations."""
    axes = [stim.PauliString(axis) for axis in ("+ZZ", "-Z_", "-_Z")]
    words, phases = _pack_axes(axes, 2)
    core = np.zeros(2, dtype=np.uint64)
    new_core, new_words, new_phases = symplectic.canonicalize(core, words, phases)
    assert len(new_words) == 3
    expected = stim.Tableau.from_named_gate("CZ")
    assert _get_tableau(new_core, new_words, new_phases, 2) == expected


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize(
    "letters", [("YY", "Y_", "XX"), ("XY_", "ZZY", "YZ_", "Y_X")], ids=["pair", "pair-lookahead"]
)
def test_choices_of_sums_of_basis_vectors(letters: tuple[str, ...]) -> None:
    """Inputs found by search, whose decompositions choose sums of two basis vectors."""
    num_qubits = len(letters[0])
    words, phases = _pack_axes([stim.PauliString(axis) for axis in letters], num_qubits)
    core = np.zeros(len(words[0]), dtype=np.uint64)
    tableau = _get_tableau(core, words, phases, num_qubits)
    assert _get_tableau(*symplectic.canonicalize(core, words, phases), num_qubits) == tableau


@pytest.mark.usefixtures("kernels")
def test_lookahead_avoids_an_extra_rotation() -> None:
    """An input found by search, for which choices without lookahead need an extra rotation.

    The three rotations act as a matrix ``M`` with ``rank(M - I) = 3``, so their canonical form has
    the fewest rotations possible.
    """
    words, phases = _pack_axes([stim.PauliString(axis) for axis in ("+_YX", "-YY_", "+__X")], 3)
    core = np.zeros(len(words[0]), dtype=np.uint64)
    tableau = _get_tableau(core, words, phases, 3)
    new_core, new_words, new_phases = symplectic.canonicalize(core, words, phases)
    assert _get_tableau(new_core, new_words, new_phases, 3) == tableau
    assert _get_rank(_get_symplectic_matrix(tableau) ^ np.eye(6, dtype=np.uint8)) == 3
    assert len(new_words) == 3


@pytest.mark.usefixtures("kernels")
def test_steps_without_continuable_choices() -> None:
    """An input found by search, whose decomposition reaches steps without continuable choices.

    At one step, candidates ``g`` with ``β(g, g) = 1`` exist but none is continuable, and at another
    step, no candidate has ``β(g, g) = 1``.
    """
    words, phases = _pack_axes([stim.PauliString(axis) for axis in ("_X", "_Y", "X_", "XX")], 2)
    core = np.zeros(len(words[0]), dtype=np.uint64)
    tableau = _get_tableau(core, words, phases, 2)
    new_core, new_words, new_phases = symplectic.canonicalize(core, words, phases)
    assert _get_tableau(new_core, new_words, new_phases, 2) == tableau
    identity = np.eye(4, dtype=np.uint8)
    assert len(new_words) <= _get_rank(_get_symplectic_matrix(tableau) ^ identity) + 1
