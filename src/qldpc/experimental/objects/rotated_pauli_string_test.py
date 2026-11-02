# SPDX-License-Identifier: Apache-2.0

"""Tests for Pauli strings followed by rotations, against dense matrices."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from operator import mul

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc.experimental.objects import RotatedPauliString, rotated_pauli_string
from qldpc.experimental.objects import _dense as dense
from qldpc.experimental.objects.conftest import get_random_pauli_string

_KERNELS = ("_multiply", "_simplify", "_merge_t_axes", "_equal", "_remove_row")

# A core, π/2 axes, and π/4 axes.
_Rotations = tuple[stim.PauliString, list[stim.PauliString], list[stim.PauliString]]


@pytest.fixture(params=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled kernels, or their Python versions for coverage."""
    if request.param == "python":
        for name in _KERNELS:
            monkeypatch.setattr(
                rotated_pauli_string, name, getattr(rotated_pauli_string, name).py_func
            )


def _get_rotations_matrix(
    core: stim.PauliString,
    s_axes: Sequence[stim.PauliString],
    t_axes: Sequence[stim.PauliString],
    num_qubits: int,
) -> npt.NDArray[np.complex128]:
    """The matrix of ``R_{π/4}(t_k) ··· R_{π/4}(t_1) · R_{π/2}(s_j) ··· R_{π/2}(s_1) · P``."""
    matrix = dense.get_pauli_matrix(core, num_qubits)
    for axes, angle in ((s_axes, np.pi / 2), (t_axes, np.pi / 4)):
        for axis in axes:
            matrix = dense.get_rotation_matrix(axis, angle, num_qubits) @ matrix
    return matrix


def _get_matrix(operator: RotatedPauliString) -> npt.NDArray[np.complex128]:
    """The matrix of an operator, built from its core and rotation axes."""
    return _get_rotations_matrix(
        operator.core, operator.s_axes, operator.t_axes, operator.num_qubits
    )


def _get_random_rotations(
    rng: np.random.Generator, num_qubits: int, *, clifford: bool = False
) -> _Rotations:
    """A random core with up to 3 random π/2 axes and, unless ``clifford``, up to 4 π/4 axes."""
    core = get_random_pauli_string(rng, num_qubits)
    s_axes = [get_random_pauli_string(rng, num_qubits) for _ in range(rng.integers(4))]
    num_t_axes = 0 if clifford else rng.integers(5)
    t_axes = [get_random_pauli_string(rng, num_qubits) for _ in range(num_t_axes)]
    return core, s_axes, t_axes


def _get_random_operator(rng: np.random.Generator, num_qubits: int) -> RotatedPauliString:
    """A random operator with up to 3 π/2 rotations and up to 4 π/4 rotations."""
    core, s_axes, t_axes = _get_random_rotations(rng, num_qubits)
    return RotatedPauliString.from_rotations(core, s_axes=s_axes, t_axes=t_axes)


@pytest.mark.usefixtures("kernels")
def test_simplification_preserves_operators(rng: np.random.Generator) -> None:
    for num_qubits in (1, 2, 3):
        for _ in range(10):
            core, s_axes, t_axes = _get_random_rotations(rng, num_qubits)
            operator = RotatedPauliString.from_rotations(core, s_axes=s_axes, t_axes=t_axes)
            expected = _get_rotations_matrix(core, s_axes, t_axes, num_qubits)
            assert dense.equal_up_to_phase(_get_matrix(operator), expected)
            assert operator == RotatedPauliString.from_rotations(
                operator.core, s_axes=operator.s_axes, t_axes=operator.t_axes
            )


@pytest.mark.usefixtures("kernels")
def test_simplified_forms() -> None:
    """Simplification merges and cancels rotations, and drops π/4 rotations about the identity.

    Two rotations about the same axis merge, and two about opposite axes ``a`` and ``-a`` cancel.
    """
    operator = RotatedPauliString.from_rotations(
        "XY", s_axes=["+ZZ", "-ZZ", "+_X"], t_axes=["+X_", "-__", "+Z_", "-Z_", "+_Z", "+_Z"]
    )
    expected = RotatedPauliString.from_rotations("XY", s_axes=["+_X", "+_Z"], t_axes=["+X_"])
    assert operator == expected and hash(operator) == hash(expected)
    assert dense.equal_up_to_phase(_get_matrix(operator), _get_matrix(expected))

    # merging R_{π/4}(X_) R_{π/4}(X_) = R_{π/2}(X_) conjugates the inner axis Z_ to -Y_
    operator = RotatedPauliString.from_rotations("__", t_axes=["+Z_", "+X_", "+X_"])
    assert operator == RotatedPauliString.from_rotations("__", s_axes=["+X_"], t_axes=["-Y_"])
    assert operator != RotatedPauliString.from_rotations("__", s_axes=["+X_"], t_axes=["+Y_"])
    assert not operator.is_clifford

    clifford = RotatedPauliString.from_rotations("X", s_axes=["Z"])
    assert clifford.is_clifford and not clifford.is_pauli
    assert RotatedPauliString("X").is_pauli


def test_clifford_operators_have_canonical_forms(rng: np.random.Generator) -> None:
    """Clifford operators are equal objects if and only if they are equal up to a global phase.

    Each random operator also appears as a different product of rotations, with an extra pair of
    rotations ``R_{π/2}(a) R_{π/2}(-a) = I`` inserted, and a Clifford operator is a Pauli string if
    and only if it has no rotations.
    """
    for num_qubits in (1, 2):
        paulis = [
            dense.get_pauli_matrix(stim.PauliString("".join(letters)), num_qubits)
            for letters in itertools.product("IXYZ", repeat=num_qubits)
        ]
        operators = []
        matrices = []
        for _ in range(30):
            core, s_axes, _ = _get_random_rotations(rng, num_qubits, clifford=True)
            extra = get_random_pauli_string(rng, num_qubits)
            position = rng.integers(len(s_axes) + 1)
            other_s_axes = [*s_axes[:position], extra, -extra, *s_axes[position:]]
            for axes in (s_axes, other_s_axes):
                operators.append(RotatedPauliString.from_rotations(core, s_axes=axes))
                matrices.append(_get_rotations_matrix(core, axes, [], num_qubits))

        for operator, matrix in zip(operators, matrices):
            is_pauli = any(dense.equal_up_to_phase(matrix, pauli) for pauli in paulis)
            assert operator.is_pauli == is_pauli
        for (first, first_matrix), (second, second_matrix) in itertools.combinations(
            zip(operators, matrices), 2
        ):
            assert (first == second) == dense.equal_up_to_phase(first_matrix, second_matrix)
            assert first != second or hash(first) == hash(second)


@pytest.mark.usefixtures("kernels")
def test_products_and_adjoints(rng: np.random.Generator) -> None:
    """Products and adjoints of a fixed pair of operators, and of random operators."""
    pairs = [
        (
            RotatedPauliString.from_rotations("Z", s_axes=["X"], t_axes=["Y"]),
            RotatedPauliString.from_rotations("Y", s_axes=["Z"], t_axes=["X"]),
        )
    ]
    for num_qubits in (1, 2, 3):
        for _ in range(8):
            pairs.append(
                (_get_random_operator(rng, num_qubits), _get_random_operator(rng, num_qubits))
            )

    for first, second in pairs:
        first_matrix = _get_matrix(first)
        product = first_matrix @ _get_matrix(second)
        assert dense.equal_up_to_phase(_get_matrix(first * second), product)
        assert first != second or dense.equal_up_to_phase(first_matrix, _get_matrix(second))
        assert dense.equal_up_to_phase(_get_matrix(first.adjoint()), first_matrix.conj().T)
        identity = first * first.adjoint()
        assert dense.equal_up_to_phase(_get_matrix(identity), np.eye(len(first_matrix)))
        if first.is_clifford:
            assert identity == RotatedPauliString(num_qubits=first.num_qubits)


def test_strings_and_widths() -> None:
    operator = RotatedPauliString.from_rotations("X_", s_axes=["+ZZ"], t_axes=["-XY", "+XZ"])
    assert operator.num_qubits == 2
    assert str(operator).splitlines()[0] == "X_" and "T -XY" in str(operator)
    for value in (operator, RotatedPauliString("Z")):
        assert eval(repr(value), {"RotatedPauliString": RotatedPauliString}) == value
    assert RotatedPauliString().num_qubits == 0 and str(RotatedPauliString()) == ""

    product = operator * RotatedPauliString("__Y")
    assert product.num_qubits == 3
    expected = np.kron(_get_matrix(operator), dense.get_pauli_matrix(stim.PauliString("Y"), 1))
    assert dense.equal_up_to_phase(_get_matrix(product), expected)
    assert RotatedPauliString("Z", num_qubits=3) == RotatedPauliString("Z__")
    assert RotatedPauliString("Z") != RotatedPauliString("Z_")


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        RotatedPauliString.from_rotations("X", s_axes=["iZ"])
    with pytest.raises(ValueError, match="more than"):
        RotatedPauliString("XX", num_qubits=1)
    with pytest.raises(TypeError):
        mul(RotatedPauliString("X"), "X")
    assert RotatedPauliString("X") != "X"
