# SPDX-License-Identifier: Apache-2.0

"""Pauli strings followed by layers of π/2 and π/4 Pauli rotations.

``RotatedPauliString`` represents a Clifford+T operator, up to a global phase, as a Pauli string
followed by a layer of π/2 Pauli rotations and a layer of π/4 Pauli rotations.  Pauli strings
conjugated by Clifford+T unitaries are the main example.  Products and adjoints keep this form,
since a Clifford operator ``C`` moves past a π/4 rotation by conjugating its axis:
``C R_{π/4}(t) = R_{π/4}(C t C†) C``.  The Pauli string and axes are stored bit-packed, so
operations cost time polynomial in the numbers of qubits and rotations, rather than in the size of
a matrix.

The rotation by an angle ``θ`` about a Hermitian Pauli string ``a`` is ``R_θ(a) = exp(-iθ a/2)``.
Up to global phases:

- ``R_{π/2}(Z) = exp(-iπ Z/4)`` is the S gate, and ``R_{π/2}(a)`` is stim's generalized S gate
  ``SPP a``;
- ``R_{π/4}(Z) = exp(-iπ Z/8)`` is the T gate.

The rotations satisfy:

- ``R_{π/4}(a)² = R_{π/2}(a)`` and ``R_{π/4}(a) R_{π/4}(-a) = I``;
- ``R_{π/2}(a) b R_{π/2}(a)† = -i a b`` for a Pauli string ``b`` that anticommutes with ``a``.
"""

# Graduation plan: once the API of this module settles, it moves to qldpc.objects, as follows.
#
# 1. Turn the module qldpc/objects.py into a package qldpc/objects/ that keeps every name of the
#    module importable from qldpc.objects.
# 2. Move this module, _packed_paulis.py, _symplectic.py, their test files, and the fixtures and
#    helpers of qldpc/experimental/objects/conftest.py into qldpc/objects/.
# 3. Re-export RotatedPauliString from qldpc.objects, and remove qldpc.experimental.objects;
#    experimental modules need no deprecation shims.
# 4. Mention RotatedPauliString in the description of qldpc.objects in docs/source/library_map.rst.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc._compilation import always_compiled

from . import _packed_paulis as packed
from . import _symplectic

# The packed core, π/2 axes, π/2 phases, π/4 axes, and π/4 phases of an operator.
_Arrays = tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int64],
]

####################################################################################################
# Pauli strings followed by rotations


class RotatedPauliString:
    """A Pauli string followed by layers of π/2 and π/4 Pauli rotations, up to a global phase.

    A ``RotatedPauliString`` on ``n`` qubits represents the operator

        ``V = R_{π/4}(t_k) ··· R_{π/4}(t_1) · R_{π/2}(s_j) ··· R_{π/2}(s_1) · P``,

    where ``P`` is its ``core``, the ``s_i`` are its ``s_axes``, and the ``t_i`` are its ``t_axes``,
    all Hermitian Pauli strings, and ``R_θ(a) = exp(-iθ a/2)`` is the rotation by ``θ`` about
    ``a``.  Axes are listed innermost (first applied) first.  "Rotated" means that the rotations are
    applied after ``P``, not that they conjugate ``P``.  Such operators are exactly the Clifford+T
    operators, and those without π/4 rotations (``k = 0``) are the Clifford operators.  Unlike a
    ``stim.PauliString``, a ``RotatedPauliString`` has no sign.

    Objects are stored in a simplified form, which:

    - drops π/4 rotations about the identity;
    - merges π/4 rotations that can be brought next to each other: two about the same axis ``a``
      become a π/2 rotation about ``a``, and two about ``a`` and ``-a`` cancel;
    - puts the Clifford part ``R_{π/2}(s_j) ··· R_{π/2}(s_1) · P`` in a canonical form.

    Equal objects represent operators that are equal up to a global phase.  The converse holds for
    operators without π/4 rotations, whose simplified forms are canonical.  With π/4 rotations,
    equal operators can be unequal objects: for example, the π/4 rotations about the 15 products of
    ``Z`` operators on nonempty subsets of 4 qubits multiply to a global phase, but their simplified
    form keeps all 15 rotations.

    Args:
        pauli: A Pauli string, whose sign does not matter.  Default: the identity on no qubits.
        num_qubits: The number of qubits to act on, padding ``pauli`` with identities.  Default:
            the length of ``pauli``.

    Raises:
        ValueError: If ``pauli`` is longer than ``num_qubits``.
    """

    _num_qubits: int
    _core: npt.NDArray[np.uint64]
    _s_axes: npt.NDArray[np.uint64]
    _s_phases: npt.NDArray[np.int64]
    _t_axes: npt.NDArray[np.uint64]
    _t_phases: npt.NDArray[np.int64]

    def __init__(
        self, pauli: str | stim.PauliString = "", *, num_qubits: int | None = None
    ) -> None:
        string = stim.PauliString(pauli)
        num_qubits = len(string) if num_qubits is None else num_qubits
        core, _ = packed.from_stim(string, num_qubits)
        no_axes = np.zeros((0, len(core)), dtype=np.uint64)
        no_phases = np.zeros(0, dtype=np.int64)
        self._set(num_qubits, (core, no_axes, no_phases, no_axes, no_phases))

    @staticmethod
    def from_rotations(
        core: str | stim.PauliString,
        *,
        s_axes: Sequence[str | stim.PauliString] = (),
        t_axes: Sequence[str | stim.PauliString] = (),
        num_qubits: int | None = None,
    ) -> RotatedPauliString:
        """The simplified product of a Pauli string and layers of π/2 and π/4 rotations.

        The product is

            ``R_{π/4}(t_k) ··· R_{π/4}(t_1) · R_{π/2}(s_j) ··· R_{π/2}(s_1) · P``.

        Args:
            core: The Pauli string ``P``, whose sign does not matter.
            s_axes: The Hermitian axes ``s_1, ..., s_j`` of the π/2 rotations, innermost first.
            t_axes: The Hermitian axes ``t_1, ..., t_k`` of the π/4 rotations, innermost first.
            num_qubits: The number of qubits.  Default: the length of the longest Pauli string.

        Returns:
            The simplified ``RotatedPauliString``.

        Raises:
            ValueError: If an axis is not Hermitian, or a Pauli string is longer than
                ``num_qubits``.
        """
        strings = [stim.PauliString(string) for string in (core, *s_axes, *t_axes)]
        if num_qubits is None:
            num_qubits = max(len(string) for string in strings)
        for axis in strings[1:]:
            if axis.sign not in (1, -1):
                raise ValueError(f"Rotation axes must be Hermitian, but {axis} is not")
        core_words, _ = packed.from_stim(strings[0], num_qubits)
        s_words, s_phases = _pack_axes(strings[1 : 1 + len(s_axes)], num_qubits)
        t_words, t_phases = _pack_axes(strings[1 + len(s_axes) :], num_qubits)
        return RotatedPauliString._new(
            num_qubits, _simplify(core_words, s_words, s_phases, t_words, t_phases)
        )

    ################################################################################################
    # properties

    @property
    def num_qubits(self) -> int:
        """The number of qubits that this operator acts on."""
        return self._num_qubits

    @property
    def core(self) -> stim.PauliString:
        """The Pauli string ``P``, with sign +1."""
        string = packed.to_stim(self._core, 0, self._num_qubits)
        return string * string.sign.conjugate()

    @property
    def s_axes(self) -> list[stim.PauliString]:
        """The axes of the π/2 rotations, innermost (first applied) first."""
        return _unpack_axes(self._s_axes, self._s_phases, self._num_qubits)

    @property
    def t_axes(self) -> list[stim.PauliString]:
        """The axes of the π/4 rotations, innermost (first applied) first."""
        return _unpack_axes(self._t_axes, self._t_phases, self._num_qubits)

    @property
    def is_pauli(self) -> bool:
        """Whether this operator has no rotations, so that it is a Pauli string.

        An operator without π/4 rotations is a Pauli string, up to a global phase, if and only if it
        has no rotations.  An operator with π/4 rotations can be a Pauli string too (see the class
        docstring).
        """
        return not len(self._s_axes) and not len(self._t_axes)

    @property
    def is_clifford(self) -> bool:
        """Whether this operator has no π/4 rotations, so that it is a Clifford operator.

        An operator with π/4 rotations can be a Clifford operator too (see the class docstring).
        """
        return not len(self._t_axes)

    ################################################################################################
    # algebra

    def __mul__(self, other: RotatedPauliString) -> RotatedPauliString:
        """The product of two operators, acting on enough qubits for both."""
        if not isinstance(other, RotatedPauliString):
            return NotImplemented
        num_qubits = max(self._num_qubits, other._num_qubits)
        lhs, rhs = self._padded(num_qubits), other._padded(num_qubits)
        product = _multiply(*lhs._get_arrays(), *rhs._get_arrays())
        return RotatedPauliString._new(num_qubits, _simplify(*product))

    def adjoint(self) -> RotatedPauliString:
        """The Hermitian adjoint of this operator.

        The adjoint is

            ``V† = P · R_{π/2}(-s_1) ··· R_{π/2}(-s_j) · R_{π/4}(-t_1) ··· R_{π/4}(-t_k)``.
        """
        no_axes = self._s_axes[:0]
        no_phases = self._s_phases[:0]
        identity = np.zeros_like(self._core)

        # write each inverse layer as an operator with the identity as its core, and multiply
        inverse_s = (self._s_axes[::-1].copy(), (self._s_phases[::-1] + 2) % 4)
        inverse_t = (self._t_axes[::-1].copy(), (self._t_phases[::-1] + 2) % 4)
        product = _multiply(
            *(self._core, no_axes, no_phases, no_axes, no_phases),
            *(identity, *inverse_s, no_axes, no_phases),
        )
        product = _multiply(*product, *(identity, no_axes, no_phases, *inverse_t))
        return RotatedPauliString._new(self._num_qubits, _simplify(*product))

    ################################################################################################
    # comparison and printing

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RotatedPauliString):
            return NotImplemented
        return self._num_qubits == other._num_qubits and all(
            np.array_equal(mine, theirs)
            for mine, theirs in zip(self._get_arrays(), other._get_arrays())
        )

    def __hash__(self) -> int:
        return hash((self._num_qubits, *(array.tobytes() for array in self._get_arrays())))

    def __str__(self) -> str:
        """The core, then a line ``S a`` for each π/2 axis and ``T a`` for each π/4 axis."""
        lines = [str(self.core)[1:]]
        lines.extend(f"S {axis}" for axis in self.s_axes)
        lines.extend(f"T {axis}" for axis in self.t_axes)
        return "\n".join(lines)

    def __repr__(self) -> str:
        arguments = [repr(str(self.core)[1:])]
        if len(self._s_axes):
            arguments.append(f"s_axes={[str(axis) for axis in self.s_axes]!r}")
        if len(self._t_axes):
            arguments.append(f"t_axes={[str(axis) for axis in self.t_axes]!r}")
        arguments.append(f"num_qubits={self._num_qubits}")
        return f"RotatedPauliString.from_rotations({', '.join(arguments)})"

    ################################################################################################
    # internals

    @staticmethod
    def _new(num_qubits: int, arrays: _Arrays) -> RotatedPauliString:
        """An operator with the given simplified arrays."""
        result = object.__new__(RotatedPauliString)
        result._set(num_qubits, arrays)
        return result

    def _set(self, num_qubits: int, arrays: _Arrays) -> None:
        """Set the number of qubits and the arrays of this operator, which must be simplified."""
        self._num_qubits = num_qubits
        self._core, self._s_axes, self._s_phases, self._t_axes, self._t_phases = arrays

    def _get_arrays(self) -> _Arrays:
        """The core, and the π/2 and π/4 axes with their phases."""
        return self._core, self._s_axes, self._s_phases, self._t_axes, self._t_phases

    def _padded(self, num_qubits: int) -> RotatedPauliString:
        """This operator, padded with identities to act on ``num_qubits`` qubits."""
        if num_qubits == self._num_qubits:
            return self
        arrays = (
            _pad(self._core, self._num_qubits, num_qubits),
            _pad(self._s_axes, self._num_qubits, num_qubits),
            self._s_phases,
            _pad(self._t_axes, self._num_qubits, num_qubits),
            self._t_phases,
        )
        return RotatedPauliString._new(num_qubits, arrays)


####################################################################################################
# compiled kernels


@always_compiled
def _multiply(
    lhs_core: npt.NDArray[np.uint64],
    lhs_s_axes: npt.NDArray[np.uint64],
    lhs_s_phases: npt.NDArray[np.int64],
    lhs_t_axes: npt.NDArray[np.uint64],
    lhs_t_phases: npt.NDArray[np.int64],
    rhs_core: npt.NDArray[np.uint64],
    rhs_s_axes: npt.NDArray[np.uint64],
    rhs_s_phases: npt.NDArray[np.int64],
    rhs_t_axes: npt.NDArray[np.uint64],
    rhs_t_phases: npt.NDArray[np.int64],
) -> _Arrays:
    """The product ``(T_A S_A P_A) · (T_B S_B P_B)`` in the form ``T · S · P``, unsimplified.

    Here ``T_A`` and ``T_B`` are layers of π/4 rotations, ``S_A`` and ``S_B`` are layers of π/2
    rotations, and ``P_A`` and ``P_B`` are Pauli strings.  The product equals ``T_A T_B' · S_A S_B ·
    P_A' P_B``, where ``T_B'`` conjugates the π/4 axes of ``T_B`` by ``S_A P_A``, and ``P_A' = S_B†
    P_A S_B``.
    """
    num_words = len(lhs_core)

    # move T_B left past S_A P_A, conjugating its axes by P_A and then by the rotations of S_A
    moved = rhs_t_axes.copy()
    moved_phases = rhs_t_phases.copy()
    for row in range(len(moved)):
        if packed.anticommutes(lhs_core, moved[row]):
            moved_phases[row] = (moved_phases[row] + 2) % 4
        for index in range(len(lhs_s_axes)):
            moved_phases[row] = packed.conjugate_by_s_rotation(
                moved[row], moved_phases[row], lhs_s_axes[index], lhs_s_phases[index]
            )

    # move P_A right past S_B, conjugating it by R_{π/2}(-s) for the axes s of S_B, outermost first
    core = lhs_core.copy()
    for index in range(len(rhs_s_axes) - 1, -1, -1):
        packed.conjugate_by_s_rotation(core, 0, rhs_s_axes[index], (rhs_s_phases[index] + 2) % 4)
    for word in range(num_words):
        core[word] ^= rhs_core[word]

    # list the axes of each layer innermost first: T_B' before T_A, and S_B before S_A
    t_axes = np.empty((len(moved) + len(lhs_t_axes), num_words), dtype=np.uint64)
    t_axes[: len(moved)] = moved
    t_axes[len(moved) :] = lhs_t_axes
    t_phases = np.empty(len(t_axes), dtype=np.int64)
    t_phases[: len(moved)] = moved_phases
    t_phases[len(moved) :] = lhs_t_phases
    s_axes = np.empty((len(rhs_s_axes) + len(lhs_s_axes), num_words), dtype=np.uint64)
    s_axes[: len(rhs_s_axes)] = rhs_s_axes
    s_axes[len(rhs_s_axes) :] = lhs_s_axes
    s_phases = np.empty(len(s_axes), dtype=np.int64)
    s_phases[: len(rhs_s_axes)] = rhs_s_phases
    s_phases[len(rhs_s_axes) :] = lhs_s_phases
    return core, s_axes, s_phases, t_axes, t_phases


@always_compiled
def _simplify(
    core: npt.NDArray[np.uint64],
    s_axes: npt.NDArray[np.uint64],
    s_phases: npt.NDArray[np.int64],
    t_axes: npt.NDArray[np.uint64],
    t_phases: npt.NDArray[np.int64],
) -> _Arrays:
    """Merge π/4 rotations where possible, and put the Clifford part in canonical form."""
    s_axes, s_phases, t_axes, t_phases = _merge_t_axes(s_axes, s_phases, t_axes, t_phases)
    core, s_axes, s_phases = _symplectic.canonicalize(core, s_axes, s_phases)
    return core, s_axes, s_phases, t_axes, t_phases


@always_compiled
def _merge_t_axes(
    s_axes: npt.NDArray[np.uint64],
    s_phases: npt.NDArray[np.int64],
    t_axes: npt.NDArray[np.uint64],
    t_phases: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.uint64], npt.NDArray[np.int64], npt.NDArray[np.uint64], npt.NDArray[np.int64]
]:
    """Remove π/4 rotations about the identity, and merge pairs of π/4 rotations about ``±a``.

    Two π/4 rotations about ``a`` and ``±a`` can be brought next to each other if the inner one
    commutes with every rotation between them.  Then ``R_{π/4}(a) R_{π/4}(-a) = I``, and
    ``R_{π/4}(a) R_{π/4}(a) = R_{π/2}(a)``.  The π/2 rotation of a merged pair moves inward past
    the remaining inner π/4 rotations, conjugating their axes, and joins the π/2 layer as its
    outermost rotation.
    """
    num_words = t_axes.shape[1]

    # remove rotations about the identity, which are global phases
    axes = t_axes.copy()
    phases = t_phases.copy()
    count = 0
    for row in range(len(t_axes)):
        if not packed.is_scalar(t_axes[row]):
            axes[count] = t_axes[row]
            phases[count] = t_phases[row]
            count += 1

    # move each rotation outward to a partner about ±a, merge the two, and start over after a merge
    merged_axes = np.empty((count // 2, num_words), dtype=np.uint64)
    merged_phases = np.empty(count // 2, dtype=np.int64)
    num_merged = 0
    first = 0
    while first < count:
        merged = False
        for second in range(first + 1, count):
            if _equal(axes[first], axes[second]):
                axis = axes[first].copy()
                axis_phase = phases[first]
                same_sign = phases[first] == phases[second]
                count = _remove_row(axes, phases, second, count)
                count = _remove_row(axes, phases, first, count)
                if same_sign:
                    for inner in range(first):
                        phases[inner] = packed.conjugate_by_s_rotation(
                            axes[inner], phases[inner], axis, axis_phase
                        )
                    merged_axes[num_merged] = axis
                    merged_phases[num_merged] = axis_phase
                    num_merged += 1
                merged = True
                break
            if packed.anticommutes(axes[first], axes[second]):
                break
        first = 0 if merged else first + 1

    # append the π/2 rotations of the merged pairs to the π/2 layer, as its outermost rotations
    new_s_axes = np.empty((len(s_axes) + num_merged, num_words), dtype=np.uint64)
    new_s_axes[: len(s_axes)] = s_axes
    new_s_axes[len(s_axes) :] = merged_axes[:num_merged]
    new_s_phases = np.empty(len(s_axes) + num_merged, dtype=np.int64)
    new_s_phases[: len(s_axes)] = s_phases
    new_s_phases[len(s_axes) :] = merged_phases[:num_merged]
    return new_s_axes, new_s_phases, axes[:count].copy(), phases[:count].copy()


@always_compiled(inline=True)
def _equal(first: npt.NDArray[np.uint64], second: npt.NDArray[np.uint64]) -> bool:
    """Whether two Pauli strings have the same words."""
    for word in range(len(first)):
        if first[word] != second[word]:
            return False
    return True


@always_compiled(inline=True)
def _remove_row(
    axes: npt.NDArray[np.uint64], phases: npt.NDArray[np.int64], row: int, count: int
) -> int:
    """Remove a row from the first ``count`` rows of axes and phases, and return the new count."""
    for index in range(row, count - 1):
        axes[index] = axes[index + 1]
        phases[index] = phases[index + 1]
    return count - 1


####################################################################################################
# helpers


def _pack_axes(
    axes: Sequence[stim.PauliString], num_qubits: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """The packed words and phases of Hermitian axes, one per row."""
    words = np.zeros((len(axes), 2 * packed.get_num_words(num_qubits)), dtype=np.uint64)
    phases = np.zeros(len(axes), dtype=np.int64)
    for row, axis in enumerate(axes):
        words[row], phases[row] = packed.from_stim(axis, num_qubits)
    return words, phases


def _unpack_axes(
    words: npt.NDArray[np.uint64], phases: npt.NDArray[np.int64], num_qubits: int
) -> list[stim.PauliString]:
    """The Hermitian axes with the given packed words and phases."""
    return [packed.to_stim(row, int(phase), num_qubits) for row, phase in zip(words, phases)]


def _pad(
    words: npt.NDArray[np.uint64], num_qubits: int, new_num_qubits: int
) -> npt.NDArray[np.uint64]:
    """Pad packed Pauli strings with identities, from ``num_qubits`` to ``new_num_qubits``."""
    xs, zs = packed.unpack(words, num_qubits)
    padding = np.zeros((*xs.shape[:-1], new_num_qubits - num_qubits), dtype=bool)
    return packed.pack(
        np.concatenate([xs, padding], axis=-1), np.concatenate([zs, padding], axis=-1)
    )
