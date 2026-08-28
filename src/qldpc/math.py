"""Miscellaneous mathematical and linear algebra methods.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TypeVar

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.special
import stim

DenseIntegerArray = galois.FieldArray | npt.NDArray[np.int_]
SparseIntegerArray = scipy.sparse.spmatrix | scipy.sparse.sparray
IntegerArray = DenseIntegerArray | SparseIntegerArray

DenseIntegerArrayType = TypeVar("DenseIntegerArrayType", galois.FieldArray, npt.NDArray[np.int_])


####################################################################################################
# general math


@functools.cache
def log_choose(n: int, k: int) -> float:
    """Natural logarithm of ``(n choose k) = n! / ( k! * (n-k)! )``."""
    return (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )


####################################################################################################
# manipulating Pauli strings and their symplectic vector representations


def op_to_string(op: npt.NDArray[np.int_]) -> stim.PauliString:
    """Convert an integer array that represents a Pauli string into a stim.PauliString.

    The (first, second) half the array indicates the support of (X, Z) Paulis.
    """
    support_xz = np.array(op, dtype=int).reshape(2, -1)
    paulis = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return stim.PauliString([paulis[xx, zz] for xx, zz in support_xz.T])


def string_to_op(string: stim.PauliString, num_qubits: int | None = None) -> npt.NDArray[np.int_]:
    """Convert a stim.PauliString into an integer array, inverting qldpc.math.op_to_string.

    The (first, second) half the array indicates the support of (X, Z) Paulis.
    """
    num_qubits = num_qubits or len(string)
    string = stim.PauliString(string)  # copy: avoid mutating the caller's input via *=
    string *= stim.PauliString(f"I{num_qubits - 1}")
    return np.hstack(string.to_numpy()).astype(int)


def symplectic_conjugate(vectors: DenseIntegerArrayType) -> DenseIntegerArrayType:
    """Take symplectic vectors to their duals.

    The symplectic conjugate of a Pauli string swaps its X and Z support, and multiplies its Z
    sector by -1, taking ``Q = [Q_x|Q_z] -> [Q_z|-Q_x]``, such that the symplectic inner product
    between Pauli strings P and Q is
    ``⟨P,Q⟩_s = P_x @ Q_z - P_z @ Q_x = P @ symplectic_conjugate(Q)``.
    """
    assert vectors.shape[-1] % 2 == 0
    conjugated_vectors = vectors.copy().reshape(-1, 2, vectors.shape[-1] // 2)[:, ::-1, :]
    conjugated_vectors[:, 1, :] *= -1
    return conjugated_vectors.reshape(vectors.shape).view(type(vectors))


def symplectic_weight(vectors: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """The symplectic weight of vectors.

    The symplectic weight of a Pauli string is the number of qudits that it addresses nontrivially.
    Returns one weight per input vector (a 0-dimensional array for a single vector).
    """
    assert vectors.shape[-1] % 2 == 0
    vectors_xz = vectors.reshape(-1, 2, vectors.shape[-1] // 2)
    vectors_x = np.asarray(vectors_xz[:, 0, :], dtype=int)
    vectors_z = np.asarray(vectors_xz[:, 1, :], dtype=int)
    return np.count_nonzero(vectors_x | vectors_z, axis=-1).reshape(vectors.shape[:-1])


####################################################################################################
# matrix helper functions


def first_nonzero_cols(
    matrix: npt.NDArray[np.generic] | Sequence[npt.NDArray[np.generic]],
) -> npt.NDArray[np.int_]:
    """Get the first nonzero column for every row in a matrix.

    If all columns are zero in a particular row, the column value is set to the number of columns.

    If the input ``matrix`` has more than two dimensions, return an ``output`` vector of length
    ``matrix.shape[0]``, where ``output[r]`` is the first "column" ``c`` for which
    ``np.any(matrix[r, c])`` is nonzero; (or set ``output[r] = matrix.shape[1]`` if
    ``not np.any(matrix[r])``).
    """
    _matrix = np.atleast_2d(np.asarray(matrix))
    if _matrix.shape[1] == 0:
        # no columns: every row is all-zero, so its first-nonzero column is the column count (0)
        return np.zeros(_matrix.shape[0], dtype=int)
    nonzero_mask = np.any(_matrix.view(np.ndarray).astype(bool), axis=tuple(range(2, _matrix.ndim)))
    has_any_nonzero_in_row = np.any(nonzero_mask, axis=1)
    first_nonzero_col_index = np.argmax(nonzero_mask, axis=1)
    first_nonzero_col_index[~has_any_nonzero_in_row] = nonzero_mask.shape[1]
    return first_nonzero_col_index.astype(np.int_, copy=False)


def block_matrix(
    blocks: Sequence[Sequence[npt.NDArray[np.generic] | int | object]],
) -> npt.NDArray[np.int_]:
    """Build a block matrix.

    Literal 0 entries are replaced by zero matrices, and literal 1 entries are replaced by an
    identity matrix (padded below and to the right with zeros, if necessary).
    """
    if not len({len(row) for row in blocks}) == 1:
        raise ValueError("Inconsistent numbers of blocks in each row")

    # consistency checks
    row_sizes = np.array(
        [[bb.shape[0] if isinstance(bb, np.ndarray) else -1 for bb in row] for row in blocks]
    )
    col_sizes = np.array(
        [[bb.shape[1] if isinstance(bb, np.ndarray) else -1 for bb in row] for row in blocks]
    )
    dtypes = [bb.dtype for row in blocks for bb in row if isinstance(bb, np.ndarray)]
    if not all(len(set(row[row != -1])) == 1 for row in row_sizes):
        raise ValueError("Inconsistent row numbers")
    if not all(len(set(col[col != -1])) == 1 for col in col_sizes.T):
        raise ValueError("Inconsistent column numbers")
    if not len(set(dtypes)) == 1:
        raise ValueError("Inconsistent block data types")

    # row numbers, column numbers, and data type
    row_nums = [next(size for size in row if size != -1) for row in row_sizes]
    col_nums = [next(size for size in col if size != -1) for col in col_sizes.T]
    dtype = dtypes[0]

    # initialize a zero matrix and populate blocks
    matrix = np.zeros((sum(row_nums), sum(col_nums)), dtype=dtype)
    for rr, row in enumerate(blocks):
        row_slice = slice(sum(row_nums[:rr]), sum(row_nums[: rr + 1]))
        for cc, block in enumerate(row):
            col_slice = slice(sum(col_nums[:cc]), sum(col_nums[: cc + 1]))
            if not isinstance(block, int):
                matrix[row_slice, col_slice] = block
            elif block == 1:
                matrix[row_slice, col_slice] = np.eye(row_nums[rr], col_nums[cc], dtype=dtype)
            elif block != 0:
                # a literal 0 leaves the already-zero block; any other literal is an error
                raise ValueError(f"Unrecognized block: {block}")
    return matrix


####################################################################################################
# basis builders


def get_dual_basis(basis: galois.FieldArray, *, validate: bool = True) -> galois.FieldArray:
    """Construct a dual basis, for which ``dual_basis @ basis.T = identity_matrix``.

    The rows of ``basis`` must be linearly independent, and it must have at least as many columns
    as rows; a dual basis exists only in that case.

    Args:
        basis: A full-row-rank matrix with at least as many columns as rows, whose rows form the
            basis to dualize.
        validate: If True (default), check the precondition above and raise a ``ValueError`` when
            it fails.  Pass False to skip the check when the precondition is already guaranteed.
    """
    if validate and (
        basis.shape[0] > basis.shape[1] or np.linalg.matrix_rank(basis) != basis.shape[0]
    ):
        raise ValueError("A dual basis can only be found for wide matrices of full rank")
    pivot_cols = first_nonzero_cols(basis.row_reduce())
    linearly_independent_cols = basis[:, pivot_cols].view(type(basis))
    dual_basis = np.zeros(basis.shape, dtype=int).view(type(basis))
    dual_basis[:, pivot_cols] = np.linalg.inv(linearly_independent_cols).T
    return dual_basis.view(type(basis))


def get_orthonormal_basis(
    matrix: galois.FieldArray, *, promise_full_rank: bool = False
) -> galois.FieldArray | None:
    """An orthonormal basis for the row space of a matrix over a finite field.

    Given a matrix whose rows span a subspace V of ``GF(q)^n``, return a matrix L whose rows are a
    basis for V with ``L @ L.T = identity`` -- that is, a basis of V whose vectors each have
    self-overlap ``v @ v = 1`` and are pairwise orthogonal under the dot product.  If V has no such
    basis, return None.

    The rows of the matrix may be linearly dependent; they are first reduced to a basis of V.  Pass
    promise_full_rank=True to skip this reduction when the rows are already independent.

    An orthonormal basis for V exists if and only if (a) V is nondegenerate, meaning no nonzero
    vector of V is orthogonal to all of V, and (b), according to the field's characteristic:

    - over a field of characteristic 2, some vector of V has nonzero self-overlap ``v @ v``;
    - over a field of odd characteristic, an even number of the vectors in any mutually orthogonal
      basis of V have a self-overlap with no square root in ``GF(q)``.

    The construction is a variant of Gram-Schmidt orthogonalization; the characteristic-2 case
    follows Algorithm 1 and Lemma 2 of https://arxiv.org/abs/2503.19790.
    """
    field = type(matrix)
    dimension = matrix.shape[1]
    if not promise_full_rank:
        matrix = matrix.row_space()  # reduce to a basis, discarding linearly dependent rows
    words = list(matrix)
    units = (
        _orthonormalize_char_2(words)
        if field.characteristic == 2
        else _orthonormalize_odd(words, field)
    )
    if units is None:
        return None
    return field(units) if units else field.Zeros((0, dimension))


def _orthonormalize_char_2(words: list[galois.FieldArray]) -> list[galois.FieldArray] | None:
    """Try to orthonormalize linearly independent vectors over a field of characteristic 2.

    Reduce the row space to mutually orthogonal "unit" vectors u with ``u @ u = 1`` and "hyperbolic
    pairs" (b, c) with ``b @ b = c @ c = 0`` and ``b @ c = 1`` (Algorithm 1 of
    https://arxiv.org/abs/2503.19790).  Every element of a characteristic-2 field is a square, so
    any vector with nonzero self-overlap can be rescaled to a unit vector.  Lemma 2 of
    https://arxiv.org/abs/2503.19790 then rewrites one unit vector and one hyperbolic pair into
    three unit vectors, eliminating every hyperbolic pair.
    """
    units: list[galois.FieldArray] = []  # vectors u with u @ u = 1
    pairs: list[tuple[galois.FieldArray, galois.FieldArray]] = []  # (b, c) with b @ c = 1
    while words:
        index = next((ii for ii, word in enumerate(words) if word @ word), None)
        if index is not None:
            # extract a unit vector and orthogonalize other words against it
            pivot = words.pop(index)
            unit = pivot / _sqrt(pivot @ pivot)
            words = [word - (word @ unit) * unit for word in words]
            units.append(unit)
        else:
            # extract a hyperbolic pair and orthogonalize other words against both of its vectors
            first = words.pop(0)
            index = next((ii for ii, word in enumerate(words) if first @ word), None)
            if index is None:
                return None  # "first" is orthogonal to all of V, so no orthonormal basis exists
            partner = words.pop(index)
            partner = partner / (first @ partner)  # rescale so that first @ partner == 1
            words = [word - (word @ partner) * first - (word @ first) * partner for word in words]
            pairs.append((first, partner))

    # an orthonormal basis needs at least one unit vector; pure hyperbolic pairs have none
    if not units and pairs:
        return None

    # Lemma 2: rewrite a unit vector a and a hyperbolic pair (b, c) as three unit vectors
    while pairs:
        vec_b, vec_c = pairs.pop()
        vec_a = units.pop()
        units += [vec_a + vec_b + vec_c, vec_a + vec_b, vec_a + vec_c]
    return units


def _orthonormalize_odd(
    words: list[galois.FieldArray], field: type[galois.FieldArray]
) -> list[galois.FieldArray] | None:
    """Try to orthonormalize linearly independent vectors over a field of odd characteristic.

    First, find an orthogonal basis in which each vector has nonzero self-overlap (with Gram-Schmidt
    orthogonalization): at each step, pick a vector with nonzero self-overlap ``v @ v`` and subtract
    its projection from the others.  Out of the remaining null vectors (with zero self-overlap), if
    a pair (u, v) have nonzero overlap ``u @ v``, then their sum has nonzero self-overlap
    (in a field with odd characteristic), ``(u + v) @ (u + v) = 2 * u @ v``, so replace
    ``u <- u + v``, and orthogonalize remaining vectors against u.  If no remaining null vectors
    have nonzero overlap, no orthonormal basis exists.

    A vector with nonzero self-overlap can be rescaled to a unit vector (self-overlap 1) exactly
    when its self-overlap has a square root in the field, by taking ``v -> v / sqrt(v @ v)``.
    Vectors with non-square self-overlaps cannot be rescaled on their own, but a pair of them may
    be combined into two unit vectors, allowing an orthonormal basis when the number of such
    vectors is even.
    """
    # diagonalize: build an orthogonal basis of vectors with nonzero self-overlap
    diagonal: list[tuple[galois.FieldArray, galois.FieldArray]] = []
    while words:
        index = next((ii for ii, word in enumerate(words) if word @ word), None)
        if index is None:
            # every self-overlap vanishes; sum an overlapping pair to get a nonzero self-overlap
            index = _combine_null_vectors(words)
            if index is None:
                return None  # every overlap vanishes on the remaining space; no basis exists
        # extract a vector with nonzero overlap, and orthogonalize the remaining vectors against it
        pivot = words.pop(index)
        overlap = pivot @ pivot
        words = [word - (word @ pivot) / overlap * pivot for word in words]
        diagonal.append((pivot, overlap))

    # rescale square-norm vectors to unit vectors; collect the non-square-norm vectors
    units: list[galois.FieldArray] = []
    non_squares: list[tuple[galois.FieldArray, galois.FieldArray]] = []
    for pivot, overlap in diagonal:
        if field.is_square(overlap):
            units.append(pivot / _sqrt(overlap))
        else:
            non_squares.append((pivot, overlap))

    # non-square self-overlaps pair off in twos, so an odd count means no orthonormal basis
    if len(non_squares) % 2:
        return None

    # pair non-square-norm vectors into unit vectors, using a fixed non-square element epsilon
    if non_squares:
        epsilon = field.primitive_element  # a generator of GF(q)* is always a non-square
        # solve alpha**2 + beta**2 = 1 / epsilon, so that alpha u + beta w has self-overlap 1
        alpha, beta = _sum_of_two_squares(field, field(1) / epsilon)
        for (u_vec, u_norm), (w_vec, w_norm) in zip(non_squares[::2], non_squares[1::2]):
            # rescale u_vec and w_vec to self-overlap epsilon, then combine into two unit vectors
            u_vec = u_vec * _sqrt(epsilon / u_norm)
            w_vec = w_vec * _sqrt(epsilon / w_norm)
            units += [alpha * u_vec + beta * w_vec, -beta * u_vec + alpha * w_vec]
    return units


def _combine_null_vectors(words: list[galois.FieldArray]) -> int | None:
    """Add one word to another to obtain a nonzero self-overlap; return its index, or None.

    Assumes every word has zero self-overlap.  In odd characteristic, if some pair has nonzero
    overlap then their sum has nonzero self-overlap; if no pair does, every overlap is zero.
    """
    for ii in range(len(words)):
        for jj in range(ii + 1, len(words)):
            if words[ii] @ words[jj]:
                words[ii] = words[ii] + words[jj]
                return ii
    return None


def _sum_of_two_squares(
    field: type[galois.FieldArray], target: galois.FieldArray
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """Field elements (a, b) with ``a**2 + b**2 == target``; these exist in odd characteristic."""
    for alpha in field.elements:  # roughly half of all "alpha" succeed, so this exits quickly
        remainder = target - alpha * alpha
        if field.is_square(remainder):
            return alpha, _sqrt(remainder)
    raise AssertionError("no solution to a**2 + b**2 = target")  # pragma: no cover


def _sqrt(value: galois.FieldArray) -> galois.FieldArray:
    """A square root of a (0-dimensional) field element that is known to be a square.

    Delegates to np.sqrt, which computes finite-field square roots but must be given an array rather
    than a 0-dimensional scalar (which it rejects over some extension fields).
    """
    return np.sqrt(np.atleast_1d(value))[0]


def symplectic_gram_schmidt(
    vectors: galois.FieldArray, *, promise_full_rank: bool = False
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """Reduce vectors to symplectic hyperbolic pairs and a symplectic radical.

    The rows of ``vectors`` span a subspace V of ``GF(q)^(2n)`` equipped with the symplectic inner
    product ``⟨a, b⟩_s = a @ symplectic_conjugate(b)`` (see symplectic_conjugate).  Return a pair
    ``(hyperbolic, radical)``:

    - ``hyperbolic`` has shape ``(2m, 2n)`` and holds ``m`` mutually orthogonal hyperbolic pairs.
      Its rows are ordered ``[b_0, ..., b_{m-1}, c_0, ..., c_{m-1}]``, so that
      ``hyperbolic @ symplectic_conjugate(hyperbolic).T`` is the block matrix ``[[0, I], [-I, 0]]``:
      ``⟨b_i, c_j⟩_s = δ_ij`` and all other products vanish.
    - ``radical`` spans the symplectic radical of V -- the vectors of V that are orthogonal to all
      of V.  Its rows are isotropic and orthogonal to every row of ``hyperbolic`` and ``radical``.

    Together the rows of ``hyperbolic`` and ``radical`` form a basis for V.  The rows of ``vectors``
    may be linearly dependent; they are first reduced to a basis of V.  Pass promise_full_rank=True
    to skip this reduction when the rows are already independent; passing it for dependent rows
    leaves the dependent directions in ``radical`` as spurious (possibly zero) rows, though the
    ``hyperbolic`` pairs stay correct.

    Because the symplectic form is alternating, ``⟨v, v⟩_s = 0`` for every vector in every
    characteristic, so -- unlike get_orthonormal_basis -- there is no unit-vector case: the
    construction peels off one hyperbolic pair at a time and collects the leftover radical.
    """
    field = type(vectors)
    dimension = vectors.shape[1]
    if not promise_full_rank:
        vectors = vectors.row_space()  # reduce to a basis, discarding linearly dependent rows
    words = list(vectors)

    firsts: list[galois.FieldArray] = []  # the b_j
    partners: list[galois.FieldArray] = []  # the c_j, with ⟨b_j, c_j⟩_s = 1
    radical: list[galois.FieldArray] = []
    while words:
        first = words.pop(0)
        index = next(
            (ii for ii, word in enumerate(words) if first @ symplectic_conjugate(word) != 0), None
        )
        if index is None:
            # "first" is orthogonal to every remaining word (and, by prior projections, to the
            # extracted pairs and radical), so it belongs to the symplectic radical
            radical.append(first)
            continue
        partner = words.pop(index)
        # rescale so that ⟨first, partner⟩_s = 1
        partner = partner / (first @ symplectic_conjugate(partner))
        # project the remaining words to be symplectically orthogonal to both "first" and "partner"
        conj_first = symplectic_conjugate(first)
        conj_partner = symplectic_conjugate(partner)
        words = [
            word - (word @ conj_partner) * first + (word @ conj_first) * partner for word in words
        ]
        firsts.append(first)
        partners.append(partner)

    hyperbolic = field(firsts + partners) if firsts else field.Zeros((0, dimension))
    return hyperbolic, field(radical) if radical else field.Zeros((0, dimension))
