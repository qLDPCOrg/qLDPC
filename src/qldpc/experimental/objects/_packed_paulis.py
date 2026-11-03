# SPDX-License-Identifier: Apache-2.0

"""Bit-packed Pauli strings, and compiled kernels that multiply and conjugate them.

A Pauli string on ``n`` qubits is stored as ``2W`` unsigned 64-bit words, where ``W = ceil(n/64)``:

- words ``0, ..., W - 1`` hold its X bits, and words ``W, ..., 2W - 1`` hold its Z bits;
- qubit ``q`` is bit ``q % 64`` of word ``q // 64`` of each half.

A batch of Pauli strings is a 2D array with one string per row.

Bits ``(x, z)`` and an integer phase ``r`` modulo 4 represent the operator ``i^r X^x Z^z``, where
``X^x`` and ``Z^z`` are tensor products of single-qubit ``X`` and ``Z`` operators.  Since
``Y = i X Z``:

- a Hermitian Pauli string with sign ``s = ±1`` has phase ``r = |x ∧ z| + 1 - s`` modulo 4, where
  ``|x ∧ z|`` is its number of ``Y`` letters;
- products follow ``X^a Z^b X^c Z^d = (-1)^{b·c} X^{a+c} Z^{b+d}``.

The kernels are compiled by numba.  Those that multiply or conjugate a Pauli string act on its
words in place and return its new phase.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc._binary import popcount_word
from qldpc._compilation import always_compiled

# The phase r of a stim sign s, with s = i^r.
_SIGN_TO_PHASE = {1: 0, 1j: 1, -1: 2, -1j: 3}
_PHASE_TO_SIGN = (1, 1j, -1, -1j)

####################################################################################################
# packing and conversion


def get_num_words(num_qubits: int) -> int:
    """The number of 64-bit words that hold the X bits, or the Z bits, of ``num_qubits`` qubits."""
    return (num_qubits + 63) // 64


def pack(xs: npt.ArrayLike, zs: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Pack X and Z bits, with qubits along the last axis, into the words of Pauli strings."""
    xs = np.asarray(xs, dtype=bool)
    zs = np.asarray(zs, dtype=bool)
    if xs.shape != zs.shape or xs.ndim == 0:
        raise ValueError(f"X and Z bits must have the same shape, not {xs.shape} and {zs.shape}")
    num_words = get_num_words(xs.shape[-1])
    words = np.zeros((*xs.shape[:-1], 2 * num_words), dtype=np.uint64)
    for part, bits in enumerate((xs, zs)):
        packed = np.packbits(bits, axis=-1, bitorder="little")
        padded = np.zeros((*bits.shape[:-1], 8 * num_words), dtype=np.uint8)
        padded[..., : packed.shape[-1]] = packed
        words[..., part * num_words : (part + 1) * num_words] = padded.view("<u8")
    return words


def unpack(
    words: npt.ArrayLike, num_qubits: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Unpack the words of Pauli strings into X and Z bits, with qubits along the last axis."""
    words = np.ascontiguousarray(words, dtype="<u8")
    num_words = words.shape[-1] // 2
    bits = np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little").astype(bool)
    return bits[..., :num_qubits], bits[..., 64 * num_words : 64 * num_words + num_qubits]


def from_stim(
    string: stim.PauliString, num_qubits: int | None = None
) -> tuple[npt.NDArray[np.uint64], int]:
    """The words and phase of a stim Pauli string, padded with identities to ``num_qubits``."""
    num_qubits = len(string) if num_qubits is None else num_qubits
    if len(string) > num_qubits:
        raise ValueError(f"Pauli string {string} acts on more than {num_qubits} qubits")
    xs = np.zeros(num_qubits, dtype=bool)
    zs = np.zeros(num_qubits, dtype=bool)
    xs[: len(string)], zs[: len(string)] = string.to_numpy()
    phase = (int(np.count_nonzero(xs & zs)) + _SIGN_TO_PHASE[string.sign]) % 4
    return pack(xs, zs), phase


def to_stim(words: npt.ArrayLike, phase: int, num_qubits: int) -> stim.PauliString:
    """The stim Pauli string with the given words and phase."""
    xs, zs = unpack(words, num_qubits)
    sign = _PHASE_TO_SIGN[(phase - int(np.count_nonzero(xs & zs))) % 4]
    return stim.PauliString.from_numpy(xs=xs, zs=zs, sign=sign)


def pack_axes(
    axes: Sequence[stim.PauliString], num_qubits: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """The words and phases of stim Pauli strings, such as rotation axes, one per row.

    The Pauli strings are padded with identities to ``num_qubits`` qubits, as by ``from_stim``.
    """
    words = np.zeros((len(axes), 2 * get_num_words(num_qubits)), dtype=np.uint64)
    phases = np.zeros(len(axes), dtype=np.int64)
    for row, axis in enumerate(axes):
        words[row], phases[row] = from_stim(axis, num_qubits)
    return words, phases


####################################################################################################
# compiled kernels


@always_compiled(inline=True)
def count_y(words: npt.NDArray[np.uint64]) -> int:
    """The number of qubits on which a Pauli string acts as ``Y``."""
    num_words = len(words) // 2
    count = 0
    for word in range(num_words):
        count += int(popcount_word(words[word] & words[num_words + word]))
    return count


@always_compiled(inline=True)
def is_scalar(words: npt.NDArray[np.uint64]) -> bool:
    """Whether a Pauli string is a scalar: the identity up to its phase."""
    for word in range(len(words)):
        if words[word]:
            return False
    return True


@always_compiled(inline=True)
def anticommutes(first: npt.NDArray[np.uint64], second: npt.NDArray[np.uint64]) -> bool:
    """Whether two Pauli strings anticommute."""
    num_words = len(first) // 2
    parity = np.uint64(0)
    for word in range(num_words):
        parity ^= (first[word] & second[num_words + word]) ^ (
            first[num_words + word] & second[word]
        )
    return popcount_word(parity) % 2 == 1


@always_compiled(inline=True)
def left_multiply(
    target: npt.NDArray[np.uint64],
    target_phase: int,
    factor: npt.NDArray[np.uint64],
    factor_phase: int,
) -> int:
    """Replace ``target`` by ``factor · target``, and return the phase of the product."""
    num_words = len(target) // 2
    count = 0
    for word in range(num_words):
        count += int(popcount_word(factor[num_words + word] & target[word]))
        target[word] ^= factor[word]
        target[num_words + word] ^= factor[num_words + word]
    return (target_phase + factor_phase + 2 * count) % 4


@always_compiled(inline=True)
def conjugate_by_s_rotation(
    target: npt.NDArray[np.uint64],
    target_phase: int,
    axis: npt.NDArray[np.uint64],
    axis_phase: int,
) -> int:
    """Conjugate ``target`` by the π/2 rotation ``exp(-iπ a/4)`` about a Hermitian Pauli ``a``.

    The π/2 rotation, which is an S gate if ``a = Z``, leaves ``target`` unchanged if they commute,
    and otherwise maps it to ``-i a · target``.  The rotation equals the gate ``SPP a`` of stim up
    to a global phase.
    """
    if not anticommutes(target, axis):
        return target_phase
    return (left_multiply(target, target_phase, axis, axis_phase) + 3) % 4


@always_compiled(inline=True)
def conjugate_by_local_clifford(
    target: npt.NDArray[np.uint64],
    target_phase: int,
    qubits: npt.NDArray[np.int64],
    images: npt.NDArray[np.int64],
    image_phases: npt.NDArray[np.int64],
) -> int:
    """Conjugate ``target`` by a Clifford gate on a few qubits, given by its conjugation table.

    The conjugation table, ``images`` and ``image_phases``, is that of ``get_clifford_gate_table``,
    and ``qubits`` lists the qubits that the gate addresses, in order.
    """
    num_words = len(target) // 2
    num_local = len(qubits)

    # read the index of the Pauli string of the target on the qubits of the gate
    index = np.int64(0)
    for position in range(num_local):
        word = qubits[position] // 64
        shift = np.uint64(qubits[position] % 64)
        index |= np.int64((target[word] >> shift) & np.uint64(1)) << position
        index |= np.int64((target[num_words + word] >> shift) & np.uint64(1)) << (
            num_local + position
        )

    # overwrite the bits of those qubits with the image, and add its phase
    image = images[index]
    for position in range(num_local):
        word = qubits[position] // 64
        mask = np.uint64(1) << np.uint64(qubits[position] % 64)
        for part in range(2):
            if (image >> (part * num_local + position)) & 1:
                target[part * num_words + word] |= mask
            else:
                target[part * num_words + word] &= ~mask
    return (target_phase + image_phases[index]) % 4


####################################################################################################
# conjugation tables of Clifford gates


@functools.cache
def get_clifford_gate_table(name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The conjugation table of a stim Clifford gate: the images of the Pauli strings on its qubits.

    A Pauli string ``X^x Z^z`` on the ``k`` qubits that the gate addresses has the index ``Σ_j (x_j
    2^j + z_j 2^{k + j})``.  Entry ``i`` of the returned arrays holds the index and the phase of the
    image of the Pauli string with index ``i``, in the convention of this module.
    """
    # list the images of the generators X_0, ..., X_{k-1}, Z_0, ..., Z_{k-1}, in the order of the
    # bits of an index
    tableau = stim.Tableau.from_named_gate(name)
    num_local = len(tableau)
    generators = [tableau.x_output(qubit) for qubit in range(num_local)]
    generators += [tableau.z_output(qubit) for qubit in range(num_local)]
    factors = [from_stim(generator, num_local) for generator in generators]

    # multiply the images of the generators in each index, in the order in which X^x Z^z lists
    # them; left multiplication builds the product from its last factor
    images = np.zeros(4**num_local, dtype=np.int64)
    image_phases = np.zeros(4**num_local, dtype=np.int64)
    for index in range(4**num_local):
        words = np.zeros(2, dtype=np.uint64)
        phase = 0
        for position in range(len(factors) - 1, -1, -1):
            if (index >> position) & 1:
                factor, factor_phase = factors[position]
                phase = left_multiply(words, phase, factor, factor_phase)
        images[index] = int(words[0]) | (int(words[1]) << num_local)
        image_phases[index] = phase
    return images, image_phases
