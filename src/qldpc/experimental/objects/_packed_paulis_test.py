# SPDX-License-Identifier: Apache-2.0

"""Tests for bit-packed Pauli strings, using stim as an independent oracle."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import stim

from qldpc.experimental.objects import _packed_paulis as packed
from qldpc.experimental.objects.conftest import get_random_pauli_string

# Qubit counts with a partial word, a full word, and more than one word.
_NUM_QUBITS = (1, 64, 65)
_KERNELS = (
    "count_y",
    "is_scalar",
    "anticommutes",
    "left_multiply",
    "conjugate_by_s_rotation",
    "conjugate_by_local_gate",
)


@pytest.fixture(params=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled kernels, or their Python versions for coverage."""
    if request.param == "python":
        for name in _KERNELS:
            monkeypatch.setattr(packed, name, getattr(packed, name).py_func)


def _random_string(rng: np.random.Generator, num_qubits: int) -> stim.PauliString:
    """A random stim Pauli string, with a random sign that need not be real."""
    letters = "".join(rng.choice(list("IXYZ"), size=num_qubits))
    return (1, 1j, -1, -1j)[rng.integers(4)] * stim.PauliString(letters)


def test_packing_round_trip(rng: np.random.Generator) -> None:
    for num_qubits in _NUM_QUBITS:
        xs, zs = rng.integers(2, size=(2, 3, 4, num_qubits)).astype(bool)
        words = packed.pack(xs, zs)
        assert words.shape == (3, 4, 2 * packed.get_num_words(num_qubits))
        unpacked_xs, unpacked_zs = packed.unpack(words, num_qubits)
        np.testing.assert_array_equal(unpacked_xs, xs)
        np.testing.assert_array_equal(unpacked_zs, zs)
    with pytest.raises(ValueError, match="same shape"):
        packed.pack([0, 1], [0, 1, 1])


def test_stim_round_trip(rng: np.random.Generator) -> None:
    for num_qubits in _NUM_QUBITS:
        string = _random_string(rng, num_qubits)
        words, phase = packed.from_stim(string)
        assert packed.to_stim(words, phase, num_qubits) == string
        padded = packed.to_stim(*packed.from_stim(string, num_qubits + 3), num_qubits + 3)
        assert padded == string + stim.PauliString(3)
    with pytest.raises(ValueError, match="more than"):
        packed.from_stim(stim.PauliString("XX"), 1)

    strings = [stim.PauliString("-XY"), stim.PauliString("Z")]
    words, phases = packed.pack_axes(strings, 3)
    assert [packed.to_stim(*row, 3) for row in zip(words, phases)] == [
        string + stim.PauliString(3 - len(string)) for string in strings
    ]
    assert packed.pack_axes([], 65)[0].shape == (0, 4)


@pytest.mark.usefixtures("kernels")
def test_products_and_commutation(rng: np.random.Generator) -> None:
    for num_qubits in _NUM_QUBITS:
        first, second = _random_string(rng, num_qubits), _random_string(rng, num_qubits)
        first_words, first_phase = packed.from_stim(first)
        second_words, second_phase = packed.from_stim(second)
        assert packed.count_y(first_words) == len(first.pauli_indices("Y"))
        assert packed.anticommutes(first_words, second_words) == (not first.commutes(second))

        words = first_words.copy()
        phase = packed.left_multiply(words, first_phase, second_words, second_phase)
        assert packed.to_stim(words, phase, num_qubits) == second * first

        assert not packed.is_scalar(packed.from_stim(stim.PauliString({num_qubits - 1: "Z"}))[0])
        assert packed.is_scalar(packed.from_stim(-stim.PauliString(num_qubits))[0])


@pytest.mark.usefixtures("kernels")
def test_conjugation_by_s_rotations(rng: np.random.Generator) -> None:
    """All one-qubit Pauli strings and axes, and random ones on more qubits."""
    pairs = [
        (sign * stim.PauliString(target), axis_sign * stim.PauliString(axis))
        for target in "IXYZ"
        for sign in (1, 1j, -1, -1j)
        for axis in "XYZ"
        for axis_sign in (1, -1)
    ]
    for num_qubits in _NUM_QUBITS[1:]:
        for _ in range(4):
            # stim has no rotation about the identity
            axis = get_random_pauli_string(rng, num_qubits, nontrivial=True)
            pairs.append((_random_string(rng, num_qubits), axis))

    for target, axis in pairs:
        gate = "SPP" if axis.sign == 1 else "SPP_DAG"
        circuit = stim.Circuit()
        circuit.append(gate, stim.target_combined_paulis(axis * axis.sign))
        words, phase = packed.from_stim(target)
        phase = packed.conjugate_by_s_rotation(words, phase, *packed.from_stim(axis))
        assert packed.to_stim(words, phase, len(target)) == target.after(circuit)


def test_conjugation_tables() -> None:
    """The shared conjugation tables cover all unitary gates but SPP and SPP_DAG, in order."""
    conjugation_tables = packed.get_conjugation_tables()
    expected_names = [
        name
        for name, gate_data in stim.gate_data().items()
        if gate_data.is_unitary and name not in ("SPP", "SPP_DAG")
    ]
    assert sorted(conjugation_tables.gate_indices) == sorted(expected_names)
    assert list(conjugation_tables.gate_indices.values()) == list(range(len(expected_names)))
    for array in (conjugation_tables.images, conjugation_tables.image_phases):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize(
    "name, qubits",
    [
        ("S", [63]),
        ("SQRT_X", [64]),
        ("H", [63]),
        ("CX", [63, 64]),
        ("CX", [64, 63]),
        ("ISWAP", [64, 63]),
        ("SWAP", [63, 64]),
    ],
)
def test_conjugation_by_local_gates(name: str, qubits: list[int]) -> None:
    """Conjugation through a conjugation table, at qubits on both sides of a word boundary."""
    conjugation_tables = packed.get_conjugation_tables()
    gate_index = conjugation_tables.gate_indices[name]
    num_qubits = 130
    for sign, letters in itertools.product((1, -1j), itertools.product("IXYZ", repeat=len(qubits))):
        target = sign * stim.PauliString(num_qubits)
        for qubit, letter in zip(qubits, letters):
            target[qubit] = letter
        words, phase = packed.from_stim(target)
        phase = packed.conjugate_by_local_gate(
            words,
            phase,
            np.array(qubits, dtype=np.int64),
            conjugation_tables.images[gate_index],
            conjugation_tables.image_phases[gate_index],
        )
        expected = target.after(stim.CircuitInstruction(name, qubits))
        assert packed.to_stim(words, phase, num_qubits) == expected, (sign, letters)
