# SPDX-License-Identifier: Apache-2.0

"""Canonical forms of Clifford operators that are products of π/2 Pauli rotations.

A product ``S = R_{π/2}(s_j) ··· R_{π/2}(s_1)`` of π/2 rotations ``R_{π/2}(a) = exp(-iπ a/4)`` about
Hermitian Pauli strings, followed by a Pauli string ``P``, is a Clifford operator ``S·P``.  In
symplectic coordinates, it acts on Pauli strings as the matrix ``M = T_{s_j} ··· T_{s_1}``, where
each transvection ``T_a: v ↦ v + ω(a, v) a`` is the action of ``R_{π/2}(a)``, and the symplectic
form ``ω`` is 1 on anticommuting Pauli strings.  ``canonicalize`` rewrites ``S·P`` as ``S'·P'``,
equal up to a global phase, where ``S'`` depends only on the operator, so that equal operators get
equal representations.  ``S'`` has at most ``rank(M - I) + 1`` rotations, and no product of fewer
than ``rank(M - I)`` π/2 rotations acts as ``M``.

The rewriting uses only the rotation axes, and never the ``2n × 2n`` matrix ``M``.  The image of
``M - I`` lies in the span of the axes.  For a basis ``a_1, ..., a_r`` of this image in reduced row
echelon form, ``M v = v + Σ_{k,l} a_k Q_{kl} ω(a_l, v)`` for a unique invertible ``r × r`` matrix
``Q``.  Applying the transvections of axes ``b_i = Σ_k g_{ik} a_k`` in the order ``b_1, b_2, ...``
gives ``M`` if the bilinear form ``β(x, y) = x^T Q^{-1} y`` has ``β(g_i, g_i) = 1`` and
``β(g_i, g_l) = 0`` for ``i < l``.  Such vectors are found greedily, with a one-step lookahead.  If
no vector in the remaining subspace has ``β(g, g) = 1``, one extra transvection makes one exist, as
some symplectic matrices are a product of no fewer than ``rank(M - I) + 1`` transvections.  Finally,
the Pauli string ``P'`` reconciles the signs of the images of Pauli strings under ``S·P`` and
``S'``.

For ``j`` rotations, ``r = rank(M - I) ≤ j``, and ``n`` qubits, the cost is ``O(j² n/64 + r³)``
in the common case, and ``O(r⁵)`` at worst for the lookahead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_inverse, row_reduce_packed
from qldpc._compilation import always_compiled

from . import _packed_paulis as packed

####################################################################################################
# canonical forms


@always_compiled
def canonicalize(
    core: npt.NDArray[np.uint64],
    axes: npt.NDArray[np.uint64],
    axis_phases: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Rewrite ``S·P`` in canonical form, up to a global phase.

    Args:
        core: The words of the Pauli string ``P``, whose phase does not matter.
        axes: The words of the axes of the rotations of ``S``, one per row, innermost (first
            applied) first.
        axis_phases: The phases of the axes, which must be Hermitian.

    Returns:
        The words of the canonical core ``P'``, and the words and phases of the canonical axes of
        ``S'``, which are Hermitian with sign +1.
    """
    num_words = len(core)
    new_core = core.copy()

    # the image of M - I is spanned by the images of the duals of the pivots of the span of the axes
    span, span_pivots = _row_reduce(axes)
    images = np.empty_like(span)
    for row in range(len(span)):
        _get_image_of_dual(axes, axis_phases, span_pivots[row], images[row])
    basis, pivots = _row_reduce(images)

    # column l of Q holds the coordinates of (M - I) h_l in the basis, for the dual h_l of pivot l
    rank = len(basis)
    q_matrix = np.zeros((rank, rank), dtype=np.uint8)
    image = np.empty(num_words, dtype=np.uint64)
    for column in range(rank):
        _get_image_of_dual(axes, axis_phases, pivots[column], image)
        for row in range(rank):
            q_matrix[row, column] = _get_bit(image, pivots[row])

    # the canonical axes, with sign +1
    new_axes = _triangularize(basis, get_inverse(q_matrix))
    new_phases = np.empty(len(new_axes), dtype=np.int64)
    for row in range(len(new_axes)):
        new_phases[row] = packed.count_y(new_axes[row])

    # S·P = S'·P' with P' = δ P, where δ is in the span of the axes, so δ is fixed by its symplectic
    # products with the duals of the pivots of the span, read off from signs of images of the duals.
    test = np.empty(num_words, dtype=np.uint64)
    for row in range(len(span)):
        _set_dual(span_pivots[row], test)
        anticommutes_with_core = packed.anticommutes(core, test)
        phase = 2 if anticommutes_with_core else 0
        for index in range(len(axes)):
            phase = packed.conjugate_by_s_rotation(test, phase, axes[index], axis_phases[index])
        _set_dual(span_pivots[row], test)
        new_phase = 0
        for index in range(len(new_axes)):
            new_phase = packed.conjugate_by_s_rotation(
                test, new_phase, new_axes[index], new_phases[index]
            )
        signs_differ = (phase - new_phase) % 4 == 2
        if signs_differ != anticommutes_with_core:
            for word in range(num_words):
                new_core[word] ^= span[row, word]
    return new_core, new_axes, new_phases


####################################################################################################
# helpers


@always_compiled
def _row_reduce(
    rows: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """The reduced row echelon form of packed rows over GF(2), without zero rows, and its pivots.

    Bit positions are ordered as in the packed words: bit ``b`` of word ``w`` is position ``64 w +
    b``, so X bits come before Z bits.
    """
    labels = np.empty((len(rows), 0), dtype=np.uint64)
    reduced, _, pivots = row_reduce_packed(rows, labels, np.arange(64 * rows.shape[1]))
    return reduced[: len(pivots)].copy(), pivots


@always_compiled(inline=True)
def _get_bit(words: npt.NDArray[np.uint64], position: int) -> np.uint8:
    """The bit at a position of packed words."""
    return np.uint8((words[position // 64] >> np.uint64(position % 64)) & np.uint64(1))


@always_compiled(inline=True)
def _set_dual(position: int, out: npt.NDArray[np.uint64]) -> None:
    """Set ``out`` to the Pauli string ``h`` with ``ω(v, h) = v[position]`` for every ``v``.

    The string ``h`` is ``Z_q`` for the X bit of qubit ``q``, and ``X_q`` for its Z bit.
    """
    out[:] = 0
    num_bits = 32 * len(out)  # the number of X bits, or of Z bits
    dual = (position + num_bits) % (2 * num_bits)
    out[dual // 64] = np.uint64(1) << np.uint64(dual % 64)


@always_compiled(inline=True)
def _get_image_of_dual(
    axes: npt.NDArray[np.uint64],
    axis_phases: npt.NDArray[np.int64],
    position: int,
    out: npt.NDArray[np.uint64],
) -> None:
    """Set ``out`` to ``(M - I) h`` for the dual ``h`` of a position (see ``_set_dual``)."""
    _set_dual(position, out)
    phase = 0
    for index in range(len(axes)):
        phase = packed.conjugate_by_s_rotation(out, phase, axes[index], axis_phases[index])
    dual = (position + 32 * len(out)) % (64 * len(out))
    out[dual // 64] ^= np.uint64(1) << np.uint64(dual % 64)


@always_compiled
def _triangularize(
    basis: npt.NDArray[np.uint64], form: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint64]:
    """Transvection axes, first applied first, of ``M`` with ``M - I`` given by a basis and a form.

    Here ``form`` is the matrix of ``β``, as in the module docstring.  Each step chooses a vector
    ``g`` with ``β(g, g) = 1``, emits the axis ``Σ_k g_k basis_k``, and continues in the subspace of
    vectors ``y`` with ``β(g, y) = 0``.  If no vector has ``β(g, g) = 1``, then ``β`` is symmetric,
    and the step emits the first basis vector ``c = basis_0`` and replaces ``Q`` by
    ``Q + e_0 e_0^T`` (so ``β`` by ``β + (β e_0)(β e_0)^T``), which accounts for applying ``T_c``
    first.
    """
    vectors = basis.copy()
    current = form.copy()
    size, num_words = vectors.shape
    axes = np.empty((2 * size, num_words), dtype=np.uint64)
    num_axes = 0
    while size > 0:
        first, second = _choose_vector(current, size)
        if first < 0:
            axes[num_axes] = vectors[0]
            num_axes += 1
            column = current[:size, 0].copy()
            for row in range(size):
                for col in range(size):
                    current[row, col] ^= column[row] & column[col]
            continue
        axes[num_axes] = vectors[first]
        product = current[first, :size].copy()
        if second >= 0:
            for word in range(num_words):
                axes[num_axes, word] ^= vectors[second, word]
            for col in range(size):
                product[col] ^= current[second, col]
        num_axes += 1

        # restrict to {y : product · y = 0}, with basis y_k = e_k + product_k e_pivot for k ≠ pivot
        pivot = 0
        while not product[pivot]:
            pivot += 1
        new_vectors = np.empty((size - 1, num_words), dtype=np.uint64)
        new_form = np.empty((size - 1, size - 1), dtype=np.uint8)
        new_row = 0
        for row in range(size):
            if row == pivot:
                continue
            for word in range(num_words):
                new_vectors[new_row, word] = vectors[row, word]
                if product[row]:
                    new_vectors[new_row, word] ^= vectors[pivot, word]
            new_col = 0
            for col in range(size):
                if col == pivot:
                    continue
                new_form[new_row, new_col] = _restricted_entry(current, product, pivot, row, col)
                new_col += 1
            new_row += 1
        vectors = new_vectors
        current = new_form
        size -= 1
    return axes[:num_axes].copy()


@always_compiled(inline=True)
def _restricted_entry(
    form: npt.NDArray[np.uint8], product: npt.NDArray[np.uint8], pivot: int, row: int, col: int
) -> np.uint8:
    """``β(y_row, y_col)`` for ``y_k = e_k + product_k e_pivot``."""
    return (
        form[row, col]
        ^ (product[col] & form[row, pivot])
        ^ (product[row] & form[pivot, col])
        ^ (product[row] & product[col] & form[pivot, pivot])
    )


@always_compiled
def _choose_vector(form: npt.NDArray[np.uint8], size: int) -> tuple[int, int]:
    """Choose ``g = e_first (+ e_second)`` with ``β(g, g) = 1``, preferring a continuable choice.

    Candidates are unit vectors, then sums of two unit vectors, in order.  A candidate is
    continuable if the next subspace is empty or has a vector ``y`` with ``β(y, y) = 1``.  Returns
    ``(-1, -1)`` if no candidate has ``β(g, g) = 1``, and ``second = -1`` for a unit vector.
    """
    fallback_first, fallback_second = -1, -1
    for first in range(size):
        if form[first, first]:
            if fallback_first < 0:
                fallback_first = first
            if size == 1 or _is_continuable(form, size, first, -1):
                return first, -1
    for first in range(size):
        for second in range(first + 1, size):
            value = form[first, first] ^ form[second, second]
            value ^= form[first, second] ^ form[second, first]
            if not value:
                continue
            if fallback_first < 0:
                fallback_first, fallback_second = first, second
            if _is_continuable(form, size, first, second):
                return first, second
    return fallback_first, fallback_second


@always_compiled
def _is_continuable(form: npt.NDArray[np.uint8], size: int, first: int, second: int) -> bool:
    """Whether the subspace after choosing ``g = e_first (+ e_second)`` has ``β(y, y) = 1``."""
    product = form[first, :size].copy()
    if second >= 0:
        for col in range(size):
            product[col] ^= form[second, col]
    pivot = 0
    while not product[pivot]:
        pivot += 1
    for row in range(size):
        if row == pivot:
            continue
        if _restricted_entry(form, product, pivot, row, row):
            return True
    for row in range(size):
        for col in range(row + 1, size):
            if row == pivot or col == pivot:
                continue
            value = _restricted_entry(form, product, pivot, row, row)
            value ^= _restricted_entry(form, product, pivot, col, col)
            value ^= _restricted_entry(form, product, pivot, row, col)
            value ^= _restricted_entry(form, product, pivot, col, row)
            if value:
                return True
    return False
