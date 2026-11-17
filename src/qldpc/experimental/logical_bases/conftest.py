# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and brute-force helpers for logical-basis tests."""

import itertools
from typing import Literal

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import PauliXZ


@pytest.fixture
def stabilizer_code() -> codes.CSSCode:
    """A [[10, 5]] code with a redundant X-type check, and lightest bases of mixed weights."""
    return codes.CSSCode(
        [
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        ],
        [[1, 1, 0, 0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 0, 0]],
    )


@pytest.fixture
def subsystem_code() -> codes.CSSCode:
    """A [[6, 3]] subsystem code whose dressed and bare logical operators have different weights."""
    return codes.CSSCode(
        [[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]], [[0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1]]
    )


def get_class(code: codes.CSSCode, pauli: PauliXZ, op: npt.ArrayLike) -> tuple[int, ...]:
    """The class of an operator: its pairing with each of the code's opposite-type operators."""
    dual_ops = np.asarray(code.get_logical_ops(pauli.swap_xz()), dtype=int)
    return tuple(int(bit) for bit in dual_ops @ np.asarray(op, dtype=int) % 2)


def get_class_words(
    code: codes.CSSCode, pauli: PauliXZ, logicals: Literal["dressed", "bare"]
) -> dict[tuple[int, ...], list[npt.NDArray[np.int_]]]:
    """Every nontrivial logical operator of one Pauli type, by class, by enumeration.

    Dressed operators commute with the stabilizers of the opposite type, and bare ones with its
    gauge operators.
    """
    other = pauli.swap_xz()
    checks = code.get_stabilizer_ops(other) if logicals == "dressed" else code.get_matrix(other)
    space = np.asarray(galois.GF2(np.asarray(checks)).null_space(), dtype=int)
    words: dict[tuple[int, ...], list[npt.NDArray[np.int_]]] = {}
    for combination in itertools.product([0, 1], repeat=len(space)):
        word = np.array(combination, dtype=int) @ space % 2
        if any(bits := get_class(code, pauli, word)):
            words.setdefault(bits, []).append(word)
    return words


def get_class_weights(
    code: codes.CSSCode, pauli: PauliXZ, logicals: Literal["dressed", "bare"]
) -> dict[tuple[int, ...], int]:
    """The minimum weight of each nontrivial logical class of one Pauli type, by enumeration."""
    words = get_class_words(code, pauli, logicals)
    return {
        bits: min(int(word.sum()) for word in class_words) for bits, class_words in words.items()
    }
