# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, and distance method and backend types and validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, SupportsIndex, get_args

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_independent_rows

from ._binary import (
    as_binary_matrix,
    as_integer,
    assert_binary,
    get_nested_code_basis,
    symplectic_to_hamming,
)
from ._brouwer_zimmermann import (
    get_distance_brouwer_zimmermann,
    get_distance_brouwer_zimmermann_many,
)
from ._brute_force import get_distance_quantum_brute_force, get_nested_distance_brute_force

DistanceBackend = Literal["auto", "decoder", "gap", "sqetch"]
DistanceMethod = Literal["brouwer_zimmermann", "brute_force", "information_sets"]

_CanonicalDistanceMethod = Literal["brouwer_zimmermann", "brute_force"]

# The canonical name of every accepted exact-distance method name, including aliases.  Logical
# counts accept the same names.
_DISTANCE_METHODS: dict[str, _CanonicalDistanceMethod] = {
    "brouwer_zimmermann": "brouwer_zimmermann",
    "brute_force": "brute_force",
    "information_sets": "brouwer_zimmermann",
}


####################################################################################################
# exact binary distance


def get_distance_classical(
    generators: npt.ArrayLike,
    *,
    cutoff: int = 1,
    block_size: int = 15,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
) -> int:
    """Distance of a classical linear binary code.

    Args:
        generators: The generator matrix of the classical code whose distance we want to compute.
            Both methods reduce redundant rows to a linearly independent basis.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) enumerates fixed-weight
            combinations in several information-set bases and stops once its lower and upper bounds
            meet.  ``"brute_force"`` enumerates every nonzero codeword.  ``"information_sets"`` is
            an alias for ``"brouwer_zimmermann"``.
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1``.  The search stops as soon as it
            proves that every nontrivial code word is heavier than ``max_weight``, which can be much
            faster than computing a large distance exactly.  The limit also caps ``cutoff``; pass
            ``cutoff=max_weight`` as well to stop at the first code word within the limit.  A code
            without nontrivial code words gives ``n + 1``, capped likewise.

    Returns:
        The minimum Hamming distance between different code words, or equivalently the minimum
        Hamming weight of a nontrivial code word.  A code of length ``n`` without nontrivial code
        words has no distance and gives ``n + 1``.  No code word has this weight, so it is not an
        observed weight, even if it is at most ``cutoff``.  If ``max_weight`` is given, the result
        is the smaller of this value and ``max_weight + 1``, so a result of ``max_weight + 1`` means
        that every nontrivial code word is heavier than ``max_weight``.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool.
        ValueError: If ``max_weight`` is negative, if the method is unknown, or if the generators
            are not a binary matrix.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    assert_binary(generators, "generators")
    matrix = as_binary_matrix(generators)
    basis = get_independent_rows(matrix)
    if not len(basis):
        return _apply_upper_bound(matrix.shape[1] + 1, upper_bound)

    if method == "brouwer_zimmermann":
        # Labels with no columns exclude no subcode.
        return get_distance_brouwer_zimmermann(
            basis,
            labels=np.empty((len(basis), 0), dtype=np.uint8),
            cutoff=cutoff,
            block_size=block_size,
            upper_bound=upper_bound,
        )

    # The reduced classical basis needs no excluded subspace.
    distance = get_distance_quantum_brute_force(
        basis,
        np.empty((0, matrix.shape[1]), dtype=np.uint8),
        cutoff=cutoff,
        block_size=block_size,
        homogeneous=True,
    )
    return _apply_upper_bound(distance, upper_bound)


def get_distance_quantum(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int = 1,
    block_size: int = 15,
    homogeneous: bool = False,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
) -> int:
    """Distance of a binary quantum code.

    Args:
        logical_ops: Rows that generate the logical space together with the stabilizers.
            Both methods remove redundant rows and exclude the stabilizer row space from witnesses.
        stabilizers: A matrix whose rows represent stabilizers of the code.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        homogeneous: If True, all Pauli strings (represented by rows of logical_ops and stabilizers)
            are assumed to have the same homogeneous (X or Z) type.  If False, Pauli strings may
            have mixed (X, Y, or Z) support on different qubits.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) uses an exclusion-aware
            search over the logical space modulo stabilizers.  ``"brute_force"`` enumerates eligible
            words using independent bases constructed internally.  ``"information_sets"`` is an
            alias for ``"brouwer_zimmermann"``.
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1``.  The search stops as soon as it
            proves that every nontrivial logical operator is heavier than ``max_weight``, which can
            be much faster than computing a large distance exactly.  The limit also caps ``cutoff``,
            so an early exit still returns the weight of an operator within the limit; pass
            ``cutoff=max_weight`` as well to stop at the first such operator.  Inputs without
            nontrivial logical operators give ``n + 1``, capped likewise.

    Returns:
        The exact minimum weight of a nontrivial logical operator in ``logical_ops`` modulo
        stabilizers (the code distance).  As an optimization, as soon as the lightest operator seen
        so far has weight ``<= cutoff`` the search stops early and returns that weight -- an upper
        bound on the true distance, in ``[distance, cutoff]``.  With the default ``cutoff=1`` this
        still returns the exact distance for every valid input (an early return can exceed the
        distance only when ``cutoff >= 2``).  Pass ``cutoff=0`` to disable the early exit and force
        the exact minimum.  Inputs on ``n`` qubits without nontrivial logical operators have no
        distance and give ``n + 1``.  No operator has this weight, so it is not an observed weight,
        even if it is at most ``cutoff``.  If ``max_weight`` is given, the result is the smaller of
        this value and ``max_weight + 1``, so a result of ``max_weight + 1`` means that every
        nontrivial logical operator is heavier than ``max_weight``.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool.
        ValueError: If ``max_weight`` is negative, if the method is unknown, or if the operators
            are not binary matrices of equal width, with an even width if ``homogeneous`` is False.

    More specifically, if homogeneous is True, then::

        (a) each Pauli string is represented by a binary vector of length equal to the number of
            data qubits in a code, indicating the nontrivial support of the Pauli string; and
        (b) the weight of a Pauli string is the Hamming weight of the corresponding bitstring.

    If ``homogeneous is False``, then::

        (a) each Pauli string is represented by a symplectic binary vector (of length 2 * data
            qubits) with the first and second halves indicating the X and Z Pauli support; and
        (b) the weight of a Pauli string is the symplectic weight of the corresponding bitstring.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    logical_matrix, stabilizer_matrix = _as_operator_matrices(logical_ops, stabilizers)
    num_bits = logical_matrix.shape[1]
    physical_size = num_bits if homogeneous else num_bits // 2
    if not homogeneous and num_bits % 2:
        raise ValueError("Symplectic operators must have an even number of columns")
    # Without a nontrivial logical operator there is no distance, and the result is n + 1, which
    # exceeds the weight of every operator.
    no_distance = _apply_upper_bound(physical_size + 1, upper_bound)
    if not len(logical_matrix):
        return no_distance

    # Separate an independent excluded basis from a complement representing the quotient.
    basis, labels = get_nested_code_basis(logical_matrix, stabilizer_matrix)
    if labels.shape[1] == 0:
        return no_distance
    if method == "brute_force":
        distance = get_nested_distance_brute_force(
            basis, labels, cutoff=cutoff, block_size=block_size, homogeneous=homogeneous
        )
        return _apply_upper_bound(distance, upper_bound)

    divisor = 1
    if not homogeneous:
        basis = symplectic_to_hamming(basis)
        divisor = 2

    distance = get_distance_brouwer_zimmermann(
        basis,
        labels,
        cutoff=cutoff * divisor,
        block_size=block_size,
        weight_divisor=divisor,
        upper_bound=None if upper_bound is None else upper_bound * divisor,
    )
    return distance // divisor


def _get_distance_css_brouwer_zimmermann(
    sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    cutoff: int,
    block_size: int = 15,
    upper_bound: int | None = None,
) -> int:
    """Compute the minimum distance across homogeneous binary CSS sectors.

    Args:
        sectors: Logical operators and stabilizers for every sector to search.  All operators use
            homogeneous binary support vectors rather than symplectic vectors.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent, as in get_distance_quantum.
        upper_bound: Weight of a known logical operator from an omitted sector, if any.  The result
            is the minimum of this bound and the distances of the supplied sectors.

    Returns:
        The minimum exact distance across the supplied sectors and the optional upper bound.  A
        sector without nontrivial logical operators counts as one more than its width, as in
        get_distance_quantum, and no operator has that weight.  Any other result at or below
        ``cutoff`` is an observed upper bound.

    Raises:
        ValueError: If there are neither sectors nor an upper bound, or if the operators of a sector
            are not binary matrices of equal width.
    """
    if not sectors and upper_bound is None:
        raise ValueError("Require at least one sector or an upper bound")
    problems: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = []
    no_distance_bound: int | None = None
    for logical_ops, stabilizers in sectors:
        logical_matrix, stabilizer_matrix = _as_operator_matrices(logical_ops, stabilizers)
        basis, labels = get_nested_code_basis(logical_matrix, stabilizer_matrix)
        if labels.shape[1]:
            problems.append((basis, labels))
            continue

        # A sector without nontrivial logical operators has no distance.  It contributes one more
        # than its width, as get_distance_quantum returns, and no operator has that weight.
        sector_bound = logical_matrix.shape[1] + 1
        no_distance_bound = (
            sector_bound if no_distance_bound is None else min(no_distance_bound, sector_bound)
        )

    # Only the weight of a known operator can end the search early or bound the searches of the
    # other sectors.  The bound of a sector without logical operators joins the minimum at the end.
    bounds = [bound for bound in (upper_bound, no_distance_bound) if bound is not None]
    if (upper_bound is not None and upper_bound <= cutoff) or not problems:
        return min(bounds)
    distance = get_distance_brouwer_zimmermann_many(
        problems,
        cutoff=cutoff,
        block_size=block_size,
        upper_bound=upper_bound,
    )
    return min([distance, *bounds])


def _as_operator_matrices(
    logical_ops: npt.ArrayLike, stabilizers: npt.ArrayLike
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Check that logical operators and stabilizers are binary, and return them as matrices.

    The stabilizers must have as many columns as the logical operators.
    """
    assert_binary(logical_ops, "logical_ops")
    assert_binary(stabilizers, "stabilizers")
    logical_matrix = as_binary_matrix(logical_ops)
    return logical_matrix, as_binary_matrix(stabilizers, num_cols=logical_matrix.shape[1])


def _get_weight_limit(cutoff: int, max_weight: SupportsIndex | None) -> tuple[int, int | None]:
    """The cutoff to use under a weight limit, and the upper bound ``max_weight + 1`` (or None).

    The limit caps the cutoff, so that an early exit still returns the weight of a word within it.
    """
    if max_weight is None:
        return cutoff, None
    max_weight = as_integer(max_weight, "max_weight")
    if max_weight < 0:
        raise ValueError(f"max_weight must be nonnegative, not {max_weight}")
    return min(cutoff, max_weight), max_weight + 1


def _apply_upper_bound(distance: int, upper_bound: int | None) -> int:
    """The smaller of a distance and an optional upper bound."""
    return distance if upper_bound is None else min(distance, upper_bound)


####################################################################################################
# method and backend selectors


def _validate_distance_backend(backend: DistanceBackend) -> None:
    """Validate a distance-bound backend selector."""
    if backend not in get_args(DistanceBackend):
        raise ValueError(
            f"Unknown distance backend {backend!r}; choose from 'auto', 'decoder', 'gap', or "
            "'sqetch'."
        )


def _validate_distance_method(
    method: DistanceMethod,
) -> _CanonicalDistanceMethod:
    """Validate an exact-distance method selector, and return its canonical name.

    The alias ``"information_sets"`` resolves to ``"brouwer_zimmermann"``.
    """
    if method not in _DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' or "
            "'brute_force'."
        )
    return _DISTANCE_METHODS[method]
