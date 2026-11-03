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

import dataclasses
import functools
import types
from collections.abc import Mapping, Sequence

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


####################################################################################################
# conjugation by local gates


def get_conjugation_table(name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The conjugation table of a stim Clifford gate: the images of the Pauli strings on its qubits.

    A Pauli string ``X^x Z^z`` on the ``k`` qubits that the gate addresses has the index ``Σ_j (x_j
    2^j + z_j 2^{k + j})``.  Entry ``i`` of the returned arrays holds the index and the phase of the
    image, under conjugation by the gate, of the Pauli string with index ``i``, in the convention
    of this module.  A conjugation table has ``4^k`` entries, so it suits gates on few qubits.
    """
    # list the images of the generators X_0, ..., X_{k-1}, Z_0, ..., Z_{k-1}, in the order of the
    # bits of an index
    tableau = stim.Tableau.from_named_gate(name)
    num_gate_qubits = len(tableau)
    generator_images = [tableau.x_output(qubit) for qubit in range(num_gate_qubits)]
    generator_images += [tableau.z_output(qubit) for qubit in range(num_gate_qubits)]
    packed_generator_images = [from_stim(image, num_gate_qubits) for image in generator_images]

    # for each index, multiply the images of the generators whose bits are set, in the order in
    # which X^x Z^z lists the generators; left multiplication builds the product from its last
    # factor
    images = np.zeros(4**num_gate_qubits, dtype=np.int64)
    image_phases = np.zeros(4**num_gate_qubits, dtype=np.int64)
    for pauli_index in range(4**num_gate_qubits):
        image = np.zeros(2, dtype=np.uint64)
        image_phase = 0
        for generator in range(2 * num_gate_qubits - 1, -1, -1):
            if (pauli_index >> generator) & 1:
                generator_image, generator_phase = packed_generator_images[generator]
                image_phase = left_multiply(image, image_phase, generator_image, generator_phase)
        images[pauli_index] = int(image[0]) | (int(image[1]) << num_gate_qubits)
        image_phases[pauli_index] = image_phase
    return images, image_phases


@dataclasses.dataclass(frozen=True)
class ConjugationTables:
    """The conjugation tables of all local gates of stim, shared by all packed circuits.

    The local gates are the unitary gates of stim that act on one or two qubits per target group,
    which are all unitary gates except the Pauli-product rotations ``SPP`` and ``SPP_DAG``.  Local
    gates are indexed in the order of ``stim.gate_data``, and the arrays are read-only.

    Attributes:
        gate_indices: The index of each local gate, by its stim name, listed in the order of the
            indices.
        images: Indexed by local gate, a row with the conjugation table of ``get_conjugation_table``
            for the gate, padded with zeros to the 16 entries of two-qubit gates.
        image_phases: Indexed by local gate, a row with the phases of these images, padded in the
            same way.
    """

    gate_indices: Mapping[str, int]
    images: npt.NDArray[np.int64]
    image_phases: npt.NDArray[np.int64]


@functools.cache
def get_conjugation_tables() -> ConjugationTables:
    """The conjugation tables of all local gates of stim, built once on first use."""
    # list the local gates
    gate_names = [
        name
        for name, gate_data in stim.gate_data().items()
        if gate_data.is_unitary and (gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate)
    ]

    # pad the conjugation table of each local gate to the 16 entries of two-qubit gates
    images = np.zeros((len(gate_names), 16), dtype=np.int64)
    image_phases = np.zeros((len(gate_names), 16), dtype=np.int64)
    for gate_index, gate_name in enumerate(gate_names):
        gate_images, gate_image_phases = get_conjugation_table(gate_name)
        images[gate_index, : len(gate_images)] = gate_images
        image_phases[gate_index, : len(gate_image_phases)] = gate_image_phases

    # make the tables read-only, since they are cached and shared, and index the gates by name
    images.flags.writeable = False
    image_phases.flags.writeable = False
    gate_indices = {gate_name: gate_index for gate_index, gate_name in enumerate(gate_names)}
    return ConjugationTables(types.MappingProxyType(gate_indices), images, image_phases)


@always_compiled(inline=True)
def conjugate_by_local_gate(
    target: npt.NDArray[np.uint64],
    target_phase: int,
    qubits: npt.NDArray[np.int64],
    images: npt.NDArray[np.int64],
    image_phases: npt.NDArray[np.int64],
) -> int:
    """Conjugate ``target`` by a local gate, a Clifford gate given by its conjugation table.

    The conjugation table, ``images`` and ``image_phases``, is that of ``get_conjugation_table``,
    and ``qubits[j]`` is the qubit that the gate addresses as its qubit ``j``.
    """
    num_words = len(target) // 2
    num_gate_qubits = len(qubits)

    # read the index of the restriction of the target to the qubits of the gate
    pauli_index = np.int64(0)
    for position in range(num_gate_qubits):
        word = qubits[position] // 64
        shift = np.uint64(qubits[position] % 64)
        pauli_index |= np.int64((target[word] >> shift) & np.uint64(1)) << position
        pauli_index |= np.int64((target[num_words + word] >> shift) & np.uint64(1)) << (
            num_gate_qubits + position
        )

    # overwrite the X bits (half 0) and Z bits (half 1) of the target on those qubits with the
    # image of the restriction, and add the phase of the image
    image = images[pauli_index]
    for position in range(num_gate_qubits):
        word = qubits[position] // 64
        mask = np.uint64(1) << np.uint64(qubits[position] % 64)
        for half in range(2):
            if (image >> (half * num_gate_qubits + position)) & 1:
                target[half * num_words + word] |= mask
            else:
                target[half * num_words + word] &= ~mask
    return (target_phase + image_phases[pauli_index]) % 4
