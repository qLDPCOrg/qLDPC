# SPDX-License-Identifier: Apache-2.0

"""Initial codes and local Clifford proposal distributions.

A search moves between codes of the same size by proposals: each proposal conjugates the current
code by a random gate on a few qubits, which keeps its numbers of physical and logical qubits.  A
search starts from a random initial code, drawn by conjugating a trivial code by a random Clifford.

A SWEL code is a self-dual CSS code that has a basis of logical operators with equal X and Z parts
(self-dual with equivalent logicals), so that a transversal Hadamard gate acts as a logical Hadamard
gate on each logical qubit.  The SWEL search keeps every code SWEL by applying only gates that
preserve this property.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, get_args

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes import CSSCode, CSSLogicalCounter

# The types of codes that a search covers: general CSS codes, and SWEL codes.
CodeType = Literal["css", "swel"]
CODE_TYPES: tuple[CodeType, ...] = get_args(CodeType)

# Separates the stream of initial codes of a seed from the search, which seeds its generator with
# the same integer.
_INITIAL_CODE_STREAM = 0x494E4954  # the ASCII bytes of "INIT"

####################################################################################################
# initial codes


def random_initial_codes(
    num_qubits: int,
    num_logicals: int,
    *,
    seed: int = 0,
    self_dual: bool = False,
) -> Iterator[CSSLogicalCounter]:
    """An endless stream of random initial codes, which are not checked for their distance.

    Each code is the trivial code of the given size conjugated by a random Clifford: a uniformly
    random invertible matrix for a general CSS code, or a random walk of SWEL gates for a self-dual
    code, of 30 gates per qubit, which makes no claim of uniform sampling from the orthogonal group.
    The stream is determined by the seed.

    Args:
        num_qubits: Physical block length.
        num_logicals: Positive number of encoded qubits, at most num_qubits.
        seed: NumPy initialization seed.  The stream of codes differs from the search's generator,
            which uses the same integer.
        self_dual: Draw self-dual (SWEL) codes, by scrambling a code of trivial logical qubits and
            Bell pairs with SWEL gates, instead of general CSS codes.

    Raises:
        ValueError: If the code size is invalid, or a self-dual code cannot have it.
    """
    if not 0 < num_logicals <= num_qubits:
        raise ValueError("Invalid code size")
    if self_dual and (num_qubits < 4 or (num_qubits - num_logicals) % 2):
        raise ValueError("Self-dual initialization requires n>=4 and even n-k")
    rng = np.random.default_rng([seed, _INITIAL_CODE_STREAM])
    checks_x, checks_z = _get_trivial_code(num_qubits, num_logicals, self_dual=self_dual)

    def scramble() -> Iterator[CSSLogicalCounter]:
        while True:
            if self_dual:
                yield _scramble_self_dual(checks_x, checks_z, rng)
            else:
                yield _scramble_css(checks_x, checks_z, rng)

    return scramble()


def _get_trivial_code(
    num_qubits: int, num_logicals: int, *, self_dual: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Checks of a code with trivial logicals on its first qubits.

    The remaining qubits hold Bell pairs for a self-dual code, and single-qubit X and Z checks
    otherwise.
    """
    rx = (num_qubits - num_logicals) // 2
    rz = num_qubits - num_logicals - rx
    x = np.zeros((rx, num_qubits), dtype=np.uint8)
    if self_dual:
        for jj in range(rx):
            x[jj, num_logicals + jj] = x[jj, num_logicals + rx + jj] = 1
        return x, x.copy()
    z = np.zeros((rz, num_qubits), dtype=np.uint8)
    for jj in range(rx):
        x[jj, num_logicals + jj] = 1
    for jj in range(rz):
        z[jj, num_logicals + rx + jj] = 1
    return x, z


def _get_counter(x: npt.NDArray[np.uint8], z: npt.NDArray[np.uint8]) -> CSSLogicalCounter:
    return CSSLogicalCounter(CSSCode(x.astype(int), z.astype(int)))


def _scramble_self_dual(
    x: npt.NDArray[np.uint8],
    z: npt.NDArray[np.uint8],
    rng: np.random.Generator,
) -> CSSLogicalCounter:
    """A random walk of 30 SWEL gates per qubit, which preserves the self-dual structure."""
    num_qubits = x.shape[1]
    counter = _get_counter(x, z)
    for _ in range(30 * num_qubits):
        targets = list(map(int, rng.choice(num_qubits, 4, replace=False)))
        counter = counter.deformed_swel(*targets)
    return counter


def _scramble_css(
    x: npt.NDArray[np.uint8], z: npt.NDArray[np.uint8], rng: np.random.Generator
) -> CSSLogicalCounter:
    """Conjugate by a uniformly random invertible matrix, acting by its inverse transpose on Z."""
    field, size = galois.GF(2), x.shape[1]
    matrix = rng.integers(2, size=(size, size), dtype=np.uint8)
    while np.linalg.matrix_rank(field(matrix)) < size:  # reject singular samples
        matrix = rng.integers(2, size=(size, size), dtype=np.uint8)
    inverse = np.linalg.inv(field(matrix)).view(np.ndarray)
    return _get_counter((x @ matrix.T) % 2, (z @ inverse) % 2)


####################################################################################################
# local Clifford proposals


def propose(
    counter: CSSLogicalCounter,
    rng: np.random.Generator,
    *,
    code_type: CodeType,
    gate_size: int,
) -> CSSLogicalCounter:
    """Sample a neighbor of a code: the code conjugated by one random gate on random qubits.

    The source counter is not changed.

    Args:
        counter: The current code.
        rng: Random generator of the search.
        code_type: ``css`` for a CNOT or a random CSS gate, or ``swel`` for the SWEL gate on four
            qubits in random order.
        gate_size: Number of qubits of the gate.  For a css search, 2 for a CNOT, 3 for a random
            three-qubit CNOT circuit, or 4 for a uniformly random invertible matrix on X, which acts
            on Z by its inverse transpose.  A swel search uses 4, for the SWEL gate.
    """
    targets = list(map(int, rng.choice(counter.num_qubits, gate_size, replace=False)))
    if code_type == "swel":
        return counter.deformed_swel(*targets)
    if gate_size == 2:
        return counter.deformed_cnot(*targets)
    matrix_x = _sample_three_qubit_gate(rng) if gate_size == 3 else _sample_four_qubit_gate(rng)
    return counter.deformed(targets, matrix_x)


def _sample_three_qubit_gate(rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """The X matrix of a weighted three-qubit CNOT circuit.

    The first three gate classes appear twice among the nine equally likely circuits, and the
    other three once, so the six classes have the relative frequencies 2:2:2:1:1:1.
    """
    kind = int(rng.integers(9))
    sequences = (
        ((0, 1),),
        ((0, 1),),
        ((0, 1), (1, 2)),
        ((0, 1), (1, 2)),
        ((0, 1), (2, 0)),
        ((0, 1), (2, 0)),
        ((0, 1), (0, 2)),
        ((0, 1), (2, 1)),
        ((0, 1), (2, 1), (1, 0), (1, 2)),
    )
    x = np.eye(3, dtype=np.uint8)
    for control, target in sequences[kind]:
        x[target] ^= x[control]
    return x


def _sample_four_qubit_gate(rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """A uniform GL(4,2) matrix for X."""
    rows, span = [], {0}
    for _ in range(4):
        row = int(rng.choice([value for value in range(1, 16) if value not in span]))
        rows.append(row)
        span |= {value ^ row for value in span}
    return np.array([[(row >> jj) & 1 for jj in range(4)] for row in rows], dtype=np.uint8)
