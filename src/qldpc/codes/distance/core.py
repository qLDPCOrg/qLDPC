# SPDX-License-Identifier: Apache-2.0

"""Exact binary distances of codes given by plain arrays, and the methods that compute them."""

from __future__ import annotations

import enum
import functools
import math
from collections.abc import Callable
from typing import Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_independent_rows, get_kernel
from qldpc.objects import PAULIS_XZ, Pauli, PauliXZ, PauliXZLike

from ._binary import (
    as_binary_matrix,
    as_binary_matrix_pair,
    as_integer,
    get_css_stabilizers,
    get_nested_code_basis,
    get_pauli_stabilizers,
    multiply_binary,
    swap_pauli_halves,
    symplectic_to_hamming,
)
from ._brouwer_zimmermann.distance import get_distance_brouwer_zimmermann_many
from ._brute_force import get_nested_distance_brute_force
from ._connected_cluster.search import get_distance_connected_cluster
from ._connected_cluster.sparse_checks import get_row_weights, get_sparser_rows
from ._tuning import SearchTuning, get_tuning

DistanceMethod = Literal[
    "brouwer_zimmermann", "information_sets", "connected_cluster", "brute_force"
]
"""The methods of exact distance searches.

- ``"brouwer_zimmermann"`` (the default) and its alias ``"information_sets"``: the
  Brouwer-Zimmermann algorithm, which enumerates words of increasing weight in several information
  sets, and stops once a lower bound that the sets prove meets the lightest word found.
- ``"connected_cluster"``: the connected-cluster algorithm of Dumer, Kovalev, and Pryadko
  (`arXiv:1611.07164 <https://arxiv.org/abs/1611.07164>`_), with the branch and bound of Okada and
  Kasai (`arXiv:2607.14091 <https://arxiv.org/abs/2607.14091>`_).  It searches the words whose
  supports the parity checks of a code connect, so it can be fast only if the checks are sparse,
  and functions that take generators reject it.  On quantum LDPC codes it is often much faster than
  ``"brouwer_zimmermann"``, but it can be much slower on some of them, and on classical LDPC codes
  whose checks are only moderately sparse.  For a subsystem code, :func:`get_distance_css` and
  :func:`get_distance_pauli` search sparser generators of the stabilizers that they derive; a
  caller who knows sparse ones can pass them to :func:`get_distance_from_checks`.
- ``"brute_force"``: the enumeration of every word, for small codes and tests.  It is also the
  fastest method for classical codes of low dimension whose distance is a large fraction of their
  length, on which the time and setup memory of ``"brouwer_zimmermann"`` can grow with the square
  of the length.  :class:`~qldpc.codes.distance.CosetLeaderSearch` and
  :func:`~qldpc.circuits.certify_bounded_fault_propagation` reject it.

:doc:`/exact_search_tuning` says how to make costly searches faster.
"""


class NoDistance(float, enum.ReprEnum):
    """The distance of an object with no nontrivial logical operator to minimize over.

    A code that encodes nothing, or a circuit with no fault that flips a logical observable without
    flipping a detector, has no distance, and qLDPC then returns ``NO_DISTANCE``, the only member
    of this enum.  It behaves as a float ``nan``:

    - It prints as ``nan``, also in the representation of a tuple.
    - It equals nothing, not even itself, and ordering comparisons with it are False.
    - Arithmetic with it gives ``nan``, and NumPy and pandas treat it as missing.

    Test for it with ``distance is NO_DISTANCE``, or with ``math.isnan(distance)``, which also
    catches ``nan`` from arithmetic.  Results that can be ``NO_DISTANCE`` are annotated
    ``int | Literal[NoDistance.NAN]``, which type checkers narrow to ``int`` once
    ``distance is NO_DISTANCE`` is ruled out.
    """

    NAN = math.nan

    def __repr__(self) -> str:
        return float.__repr__(self)

    @classmethod
    def _missing_(cls, value: object) -> NoDistance | None:
        # Pickling and copying look the member up by its value, which differs from itself.
        return cls.NAN if isinstance(value, float) and math.isnan(value) else None


NO_DISTANCE = NoDistance.NAN
"""The distance returned for an object without one: the only member of :class:`NoDistance`.

It behaves as a float ``nan``.  Test for it with ``distance is NO_DISTANCE``.
"""

_CanonicalDistanceMethod = Literal["brouwer_zimmermann", "connected_cluster", "brute_force"]

# The canonical name of every accepted exact-distance method name, including aliases.
_DISTANCE_METHODS: dict[str, _CanonicalDistanceMethod] = {
    "brouwer_zimmermann": "brouwer_zimmermann",
    "information_sets": "brouwer_zimmermann",
    "connected_cluster": "connected_cluster",
    "brute_force": "brute_force",
}


####################################################################################################
# exact binary distances


def get_distance_from_generators(
    generators: npt.ArrayLike,
    excluded: npt.ArrayLike | None = None,
    *,
    symplectic: bool = False,
    cutoff: int = 1,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """The minimum weight of a nontrivial word in the row space of ``generators`` and ``excluded``.

    The words are the sums of rows of ``generators`` and ``excluded``, and a word is trivial if it
    is a sum of rows of ``excluded`` alone.  Rows may be redundant.

    For a classical code, pass its generator matrix to get its distance.  For a quantum code, pass
    its logical operators, and exclude its stabilizers.  For a subsystem code, excluding its gauge
    generators, such as the rows of the ``matrix`` of a :class:`~qldpc.codes.QuditCode`, gives its
    distance, which is dressed.  With bare logical operators as generators, such as those of
    :meth:`~qldpc.codes.common.QuditCode.get_logical_ops`, excluding only its stabilizers gives its
    bare distance instead (:func:`get_distance_from_checks` defines both).

    Args:
        generators: A binary matrix whose rows, together with those of ``excluded``, span the words.
        excluded: A binary matrix with as many columns, whose rows span the trivial words.  By
            default, only the zero word is trivial.  If either matrix is an empty list or has shape
            ``(0, 0)``, it takes the width of the other.
        symplectic: If False (the default), each row is a binary word, which weighs its number of
            nonzero entries.  If True, each row is a Pauli string on ``n`` qubits, whose ``2 n``
            entries are the supports of its X part and then of its Z part, and which weighs the
            number of qubits on which it acts nontrivially.
        cutoff: Exit early once a nontrivial word of weight at most ``cutoff`` is found (see
            "Returns").
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``, or
            ``"brute_force"`` (see :data:`DistanceMethod`).  ``"connected_cluster"`` needs parity
            checks: use :func:`get_distance_from_checks`.
        max_weight: If given, a nonnegative integer limit on the weights to search (see "Returns").
            The search stops once it proves that every nontrivial word is heavier, which can be much
            faster than finding a large distance exactly.  The limit caps ``cutoff``, so that an
            early exit still returns the weight of a word within the limit.  Pass
            ``cutoff=max_weight`` as well to stop at the first such word.
        automorphisms: Permutations of the bits, or of the qubits if ``symplectic`` is True, that
            map the row space of ``excluded``, and that of both matrices together, onto themselves.
            Pass an array whose row ``p`` maps bit or qubit ``j`` to ``p[j]``, or a function that
            returns such an array or None.  A search asks for them at most once, when it reaches a
            costly stage (see :doc:`/exact_search_tuning`), and only then calls a function.  The
            Brouwer-Zimmermann method then verifies that they are automorphisms, and uses them to
            search fewer information sets.  The brute-force method ignores them.

    Returns:
        The minimum weight of a nontrivial word.  Once the lightest nontrivial word found weighs at
        most ``cutoff``, the search stops early and returns its weight, an upper bound on the
        distance in ``[distance, cutoff]``.  With the default ``cutoff=1``, the result is always
        exact, since a nontrivial word weighs at least one; pass ``cutoff=0`` to make that
        explicit.  If ``max_weight`` is given, the result is the smaller of this value and
        ``max_weight + 1``, so ``max_weight + 1`` means that every nontrivial word is heavier than
        ``max_weight``.  Inputs without nontrivial words give ``NO_DISTANCE`` (see
        :class:`NoDistance`), whatever ``cutoff`` and ``max_weight`` are.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``generators`` or
            ``excluded`` is not array-like, as when a code is given instead of a matrix.
        ValueError: If the method is unknown or is ``"connected_cluster"``, if ``max_weight`` is
            negative, if the matrices are not binary matrices with the same number of columns, if
            neither has a width (``generators`` is an empty list, and ``excluded`` is one too or is
            not given), if ``symplectic`` is True and the number of columns is odd, if
            ``automorphisms`` are not permutations of the bits or qubits, or if a search finds that
            one of them is not an automorphism.
    """
    method = _reject_connected_cluster(_validate_distance_method(method))
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    generator_matrix, excluded_matrix = _as_matrix_and_excluded(generators, excluded, "generators")
    num_variables = _get_num_variables(generator_matrix.shape[1], symplectic)
    return _search_generators(
        generator_matrix,
        excluded_matrix,
        symplectic=symplectic,
        cutoff=cutoff,
        upper_bound=upper_bound,
        method=method,
        get_automorphisms=_get_automorphism_getter(automorphisms, num_variables),
    )


def get_distance_from_checks(
    checks: npt.ArrayLike,
    excluded: npt.ArrayLike | None = None,
    *,
    symplectic: bool = False,
    cutoff: int = 1,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """The minimum weight of a nontrivial word that satisfies the checks.

    A binary word satisfies a check if their dot product is zero modulo two, and a Pauli string
    satisfies a check if they commute.  A word is trivial if it is in the row space of ``excluded``.
    For a classical code, pass its parity check matrix to get its distance.  For the X logical
    operators of a CSS stabilizer code, pass its Z checks, and exclude its X checks.  Checks may be
    redundant, which can speed up the connected-cluster method.

    The distance of a subsystem code is dressed: its logical operators may be multiplied by gauge
    operators.  This function finds it with the stabilizers as the checks and the gauge generators
    as ``excluded``, and :func:`get_distance_css` and :func:`get_distance_pauli` find it from the
    gauge generators.  With these roles exchanged, this function finds the bare distance instead:
    the minimum weight of a nontrivial logical operator that commutes with every gauge operator.
    For a stabilizer code, both are its distance.  The bare distance of a binary quantum code
    ``code``, whose ``matrix`` holds its gauge generators, is::

        get_distance_from_checks(code.matrix, code.get_stabilizer_ops(), symplectic=True)

    For a CSS code, two smaller classical searches give it too.  The bare X-distance is::

        get_distance_from_checks(code.matrix_z, code.get_stabilizer_ops(Pauli.X))

    The bare Z-distance exchanges X and Z, and the bare distance is the smaller of the two.

    Excluding nothing, or only some stabilizers, counts stabilizers as logical operators.  For a
    subsystem code, passing stabilizers as both the checks and ``excluded``, such as its Z-type
    stabilizers and its X-type ones, counts gauge operators that are not stabilizers as logical
    operators.

    Args:
        checks: A binary matrix whose rows are the checks.
        excluded: A binary matrix with as many columns, whose rows satisfy the checks and span the
            trivial words.  By default, only the zero word is trivial.  It is required if
            ``symplectic`` is True, since the stabilizers of a quantum code satisfy its checks but
            are trivial.  If either matrix is an empty list or has shape ``(0, 0)``, it takes the
            width of the other.
        symplectic: Whether each row is a Pauli string rather than a binary word (see
            :func:`get_distance_from_generators`).
        cutoff: Exit early once a nontrivial word of weight at most ``cutoff`` is found (see
            :func:`get_distance_from_generators`).
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_cluster"``, or ``"brute_force"`` (see :data:`DistanceMethod`).
        max_weight: If given, a nonnegative integer limit on the weights to search (see
            :func:`get_distance_from_generators`).
        automorphisms: Permutations of the bits, or of the qubits if ``symplectic`` is True, that
            map the row space of ``checks``, and that of ``excluded``, onto themselves, given as for
            :func:`get_distance_from_generators`.  The Brouwer-Zimmermann and connected-cluster
            methods verify that they are automorphisms, and use them to search less.  The
            brute-force method ignores them.

    Returns:
        The minimum weight of a nontrivial word, or ``NO_DISTANCE`` if there is none, with the
        early exit at ``cutoff`` and the limit ``max_weight`` of
        :func:`get_distance_from_generators`.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks`` or
            ``excluded`` is not array-like, as when a code is given instead of a matrix.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the matrices are not
            binary matrices with the same number of columns, if neither has a width (``checks`` is
            an empty list, and ``excluded`` is one too or is not given), if ``symplectic`` is True
            and the number of columns is odd or ``excluded`` is not given, if a row of ``excluded``
            does not satisfy the checks, if ``automorphisms`` are not permutations of the bits or
            qubits, or if a search finds that one of them is not an automorphism.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    check_matrix, excluded_matrix = _as_matrix_and_excluded(checks, excluded, "checks")
    num_variables = _get_num_variables(check_matrix.shape[1], symplectic)
    if symplectic and excluded is None:
        raise ValueError(
            "Symplectic checks need excluded rows, which satisfy the checks but are trivial.  For"
            " the distance of a stabilizer or subsystem code, use get_distance_pauli; for the bare"
            " distance of a subsystem code, pass its gauge generators as the checks and its"
            " stabilizers as excluded (see get_distance_from_checks)"
        )
    dot_checks = swap_pauli_halves(check_matrix) if symplectic else check_matrix
    if np.any(multiply_binary(excluded_matrix, dot_checks.T)):
        raise ValueError(
            "Every row of excluded must satisfy the checks.  For the distance of a subsystem code,"
            " which is dressed, use get_distance_css or get_distance_pauli; for its bare distance,"
            " pass its gauge generators as the checks and its stabilizers as excluded (see"
            " get_distance_from_checks)"
        )
    return _search_checks(
        check_matrix,
        excluded_matrix,
        symplectic=symplectic,
        cutoff=cutoff,
        upper_bound=upper_bound,
        method=method,
        get_automorphisms=_get_automorphism_getter(automorphisms, num_variables),
    )


def get_distance_css(
    checks_x: npt.ArrayLike,
    checks_z: npt.ArrayLike,
    pauli: PauliXZLike | None = None,
    *,
    cutoff: int = 1,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """The distance of a binary CSS code, or of its X or Z logical operators, from its checks.

    The rows of ``checks_x`` are the supports of the X-type checks of the code, which detect Z
    errors, and the rows of ``checks_z`` are those of its Z-type checks.  If the checks commute,
    they generate the stabilizer group.  An X logical operator then has its support in the null
    space of ``checks_z``, and is trivial if its support is in the row space of ``checks_x``, and
    likewise for Z.  If the checks do not all commute, they generate the gauge group of a subsystem
    code.  An X logical operator then commutes with every Z-type stabilizer, and is trivial if it is
    an X-type gauge operator, so the result is the dressed distance (see
    :func:`get_distance_from_checks`, which also gives the bare distance).
    ``CSSCode(checks_x, checks_z, is_subsystem_code=False)`` raises an error if the checks do not
    commute.

    Args:
        checks_x: A binary matrix of the X-type checks, which may be redundant.
        checks_z: A binary matrix of the Z-type checks, with as many columns, which may be
            redundant.  If either matrix is an empty list or has shape ``(0, 0)``, it takes the
            width of the other.
        pauli: ``Pauli.X`` (or ``"X"``) for the X-distance, the minimum weight of a nontrivial X
            logical operator, or ``Pauli.Z`` (or ``"Z"``) for the Z-distance.  If None (the
            default), the result is the distance of the code, the smaller of the two, which the
            Brouwer-Zimmermann and connected-cluster methods find faster than two calls would.
        cutoff: Exit early once a nontrivial logical operator of weight at most ``cutoff`` is found
            (see :func:`get_distance_from_generators`).
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_cluster"``, or ``"brute_force"`` (see :data:`DistanceMethod`, which also
            covers subsystem codes).
        max_weight: If given, a nonnegative integer limit on the weights to search (see
            :func:`get_distance_from_generators`).
        automorphisms: Permutations of the qubits that map the row space of ``checks_x``, and that
            of ``checks_z``, onto themselves, given as for :func:`get_distance_from_generators`.
            The Brouwer-Zimmermann and connected-cluster methods verify that they are automorphisms,
            and use them to search less.  The brute-force method ignores them.

    Returns:
        The minimum weight of a nontrivial logical operator of the requested types, with the early
        exit at ``cutoff`` and the limit ``max_weight`` of :func:`get_distance_from_generators`, or
        ``NO_DISTANCE`` for a code that encodes no qubits.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks_x`` or
            ``checks_z`` is not array-like, as when a code is given instead of its checks.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the checks are not
            binary matrices with the same number of columns, if both are empty lists, which have no
            width, if ``pauli`` is not X, Z, or None, if ``automorphisms`` are not permutations of
            the qubits, or if a search finds that one of them is not an automorphism.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    gauge_x, gauge_z = as_binary_matrix_pair(checks_x, checks_z, ("checks_x", "checks_z"))
    paulis = PAULIS_XZ if pauli is None else [Pauli.coerce_xz(pauli)]
    get_automorphisms = _get_automorphism_getter(automorphisms, gauge_x.shape[1])
    return _search_css(
        gauge_x,
        gauge_z,
        paulis,
        cutoff=cutoff,
        upper_bound=upper_bound,
        method=method,
        get_automorphisms=get_automorphisms,
    )


def get_distance_pauli(
    checks: npt.ArrayLike,
    *,
    cutoff: int = 1,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """The distance of a binary stabilizer or subsystem code, from its parity check matrix.

    Each row of ``checks`` is a Pauli string on ``n`` qubits, ``[P_x|P_z]`` with the supports
    ``P_x`` and ``P_z`` of its X and Z parts, as in :class:`~qldpc.codes.QuditCode`.  A Pauli string
    weighs the number of qubits on which it acts nontrivially.  If the rows commute, they generate
    the stabilizer group, and the distance is the minimum weight of a Pauli string that commutes
    with every row but is not a stabilizer.  Otherwise, they generate the gauge group of a subsystem
    code, whose stabilizers are the gauge operators that commute with every gauge operator, and the
    result is its dressed distance: the minimum weight of a Pauli string that commutes with every
    stabilizer but is not a gauge operator.  :func:`get_distance_from_checks` says how to find the
    bare distance instead.  ``QuditCode(checks, is_subsystem_code=False)`` raises an error if the
    rows do not commute.

    If X-type and Z-type operators generate the group that the rows generate, even if the rows mix
    X and Z, this function searches the code as :func:`get_distance_css` does, which is faster.

    Args:
        checks: A binary matrix whose rows represent Pauli strings, which may be redundant.
        cutoff: Exit early once a nontrivial logical operator of weight at most ``cutoff`` is found
            (see :func:`get_distance_from_generators`).
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_cluster"``, or ``"brute_force"`` (see :data:`DistanceMethod`, which also
            covers subsystem codes).
        max_weight: If given, a nonnegative integer limit on the weights to search (see
            :func:`get_distance_from_generators`).
        automorphisms: Permutations of the qubits that map the row space of ``checks`` onto itself,
            given as for :func:`get_distance_from_generators`.  The Brouwer-Zimmermann and
            connected-cluster methods verify that they are automorphisms, and use them to search
            less.  The brute-force method ignores them.

    Returns:
        The minimum weight of a nontrivial logical operator, with the early exit at ``cutoff`` and
        the limit ``max_weight`` of :func:`get_distance_from_generators`, or ``NO_DISTANCE`` for a
        code that encodes no qubits.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks`` is not
            array-like, as when a code is given instead of its checks.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the checks are not a
            binary matrix with an even number of columns, if they are an empty list, which has no
            width, if ``automorphisms`` are not permutations of the qubits, or if a search finds
            that one of them is not an automorphism.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    gauge = as_binary_matrix(checks, "checks")
    num_qubits = _get_num_variables(gauge.shape[1], symplectic=True)
    get_automorphisms = _get_automorphism_getter(automorphisms, num_qubits)

    # Each row is the product of its X part, a row of gauge_x, and its Z part, a row of gauge_z.  So
    # the rows generate a CSS group exactly if they generate the group of all of these parts, whose
    # rank is rank(gauge_x) + rank(gauge_z).  Two sectors of n bits are much cheaper to search than
    # one of 2n bits.
    gauge_x, gauge_z = gauge[:, :num_qubits], gauge[:, num_qubits:]
    rank = len(get_independent_rows(gauge))
    if rank == len(get_independent_rows(gauge_x)) + len(get_independent_rows(gauge_z)):
        return _search_css(
            gauge_x,
            gauge_z,
            PAULIS_XZ,
            cutoff=cutoff,
            upper_bound=upper_bound,
            method=method,
            get_automorphisms=get_automorphisms,
        )

    stabilizers = get_pauli_stabilizers(gauge)

    # get_pauli_stabilizers returns new rows only for a subsystem code.  The gauge generators that
    # commute with every gauge generator are stabilizers, which join the checks (Section 3.9 of
    # connected_cluster.md).
    kept_stabilizers = None
    if method == "connected_cluster" and stabilizers is not gauge:
        overlaps = multiply_binary(gauge, swap_pauli_halves(gauge).T)
        kept_stabilizers = gauge[~np.any(overlaps, axis=1)]

    return _search_checks(
        stabilizers,
        gauge,
        symplectic=True,
        cutoff=cutoff,
        upper_bound=upper_bound,
        method=method,
        get_automorphisms=get_automorphisms,
        kept_stabilizers=kept_stabilizers,
    )


def _search_css(
    gauge_x: npt.NDArray[np.uint8],
    gauge_z: npt.NDArray[np.uint8],
    paulis: list[PauliXZ],
    *,
    cutoff: int,
    upper_bound: int | None,
    method: _CanonicalDistanceMethod,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None,
) -> int | Literal[NoDistance.NAN]:
    """Search the requested sectors of a CSS code, given its X-type and Z-type gauge generators.

    The arguments are those of ``get_distance_css``, validated, with ``upper_bound`` the
    ``max_weight + 1`` (or None) of ``_get_weight_limit``, and ``get_automorphisms`` from
    ``_get_automorphism_getter``.
    """
    # Each sector takes the stabilizers of the other type as checks, and excludes the gauge
    # generators of its own type.
    stabilizers_x, stabilizers_z = get_css_stabilizers(gauge_x, gauge_z)
    sectors = {Pauli.X: (stabilizers_z, gauge_x), Pauli.Z: (stabilizers_x, gauge_z)}
    problems = [sectors[sector_pauli] for sector_pauli in paulis]

    # Each type has one independent nontrivial logical operator for each logical qubit, so the X
    # type alone shows whether the code encodes any.
    num_logical_x = (
        gauge_x.shape[1]
        - len(get_independent_rows(stabilizers_z))
        - len(get_independent_rows(gauge_x))
    )
    if num_logical_x == 0:
        return NO_DISTANCE

    # Search the requested types at once, except by brute force, which searches each type in turn.
    if method == "connected_cluster":
        tuning = get_tuning()
        if stabilizers_x is not gauge_x:
            # get_css_stabilizers returns new rows, a row-reduced basis that can be dense, only for
            # a subsystem code (Section 3.9 of connected_cluster.md).
            overlaps = multiply_binary(gauge_x, gauge_z.T)
            kept_stabilizers = {
                Pauli.X: gauge_z[~np.any(overlaps, axis=0)],
                Pauli.Z: gauge_x[~np.any(overlaps, axis=1)],
            }
            problems = [
                (
                    _get_sparser_stabilizers(
                        checks, kept_stabilizers[sector_pauli], symplectic=False, tuning=tuning
                    ),
                    excluded,
                )
                for sector_pauli, (checks, excluded) in zip(paulis, problems, strict=True)
            ]
        return get_distance_connected_cluster(
            problems,
            cutoff=cutoff,
            upper_bound=upper_bound,
            get_automorphisms=get_automorphisms,
            tuning=tuning,
        )
    if method == "brute_force":
        return min(
            _search_checks(
                checks,
                excluded,
                symplectic=False,
                cutoff=cutoff,
                upper_bound=upper_bound,
                method=method,
                get_automorphisms=None,
            )
            for checks, excluded in problems
        )
    return get_distance_brouwer_zimmermann_many(
        [get_nested_code_basis(get_kernel(checks), excluded) for checks, excluded in problems],
        cutoff=cutoff,
        upper_bound=upper_bound,
        get_automorphisms=get_automorphisms,
        tuning=get_tuning(),
    )


def _search_checks(
    checks: npt.NDArray[np.uint8],
    excluded: npt.NDArray[np.uint8],
    *,
    symplectic: bool,
    cutoff: int,
    upper_bound: int | None,
    method: _CanonicalDistanceMethod,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None,
    kept_stabilizers: npt.NDArray[np.uint8] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """Search the words that satisfy the checks, outside the row space of the excluded rows.

    The excluded rows satisfy the checks, and the other arguments are as for ``_search_css``.  If
    ``kept_stabilizers`` is given, the checks are the derived stabilizers of a subsystem code, which
    a connected-cluster search replaces with those of ``_get_sparser_stabilizers``.
    """
    if method != "connected_cluster":
        kernel = get_kernel(swap_pauli_halves(checks) if symplectic else checks)
        return _search_generators(
            kernel,
            excluded,
            symplectic=symplectic,
            cutoff=cutoff,
            upper_bound=upper_bound,
            method=method,
            get_automorphisms=get_automorphisms,
        )

    # The words that satisfy the checks span a space of dimension checks.shape[1] - rank(checks),
    # which holds the row space of the excluded rows, so equal dimensions leave no nontrivial word.
    dimension = checks.shape[1] - len(get_independent_rows(checks))
    if dimension == len(get_independent_rows(excluded)):
        return NO_DISTANCE
    tuning = get_tuning()
    if kept_stabilizers is not None:
        checks = _get_sparser_stabilizers(
            checks, kept_stabilizers, symplectic=symplectic, tuning=tuning
        )
    return get_distance_connected_cluster(
        [(checks, excluded)],
        cutoff=cutoff,
        upper_bound=upper_bound,
        symplectic=symplectic,
        get_automorphisms=get_automorphisms,
        tuning=tuning,
    )


def _get_sparser_stabilizers(
    stabilizers: npt.NDArray[np.uint8],
    kept_stabilizers: npt.NDArray[np.uint8],
    *,
    symplectic: bool,
    tuning: SearchTuning,
) -> npt.NDArray[np.uint8]:
    """Checks that span the stabilizers of a subsystem code, for a connected-cluster search.

    ``stabilizers`` is the basis that ``get_css_stabilizers`` or ``get_pauli_stabilizers`` derives,
    and ``kept_stabilizers`` holds the gauge generators that are stabilizers.  The checks are the
    rows of ``kept_stabilizers`` no heavier than some sparser row, followed by ``stabilizers`` made
    sparser (Section 3.9 of ``connected_cluster.md``).
    """
    rows = stabilizers
    if tuning.max_distance_sparsifying_rounds:
        rows, _ = get_sparser_rows(
            stabilizers,
            symplectic=symplectic,
            max_rounds=tuning.max_distance_sparsifying_rounds,
            rounds_without_improvement=tuning.sparsifying_rounds_without_improvement,
        )

    # Kept rows heavier than every sparser row would slow the search, and the sparser rows alone
    # span the stabilizers.
    max_sparser_weight = get_row_weights(rows, symplectic=symplectic).max(initial=0)
    light = get_row_weights(kept_stabilizers, symplectic=symplectic) <= max_sparser_weight
    return np.vstack([kept_stabilizers[light], rows])


def _search_generators(
    generators: npt.NDArray[np.uint8],
    excluded: npt.NDArray[np.uint8],
    *,
    symplectic: bool,
    cutoff: int,
    upper_bound: int | None,
    method: Literal["brouwer_zimmermann", "brute_force"],
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None,
) -> int | Literal[NoDistance.NAN]:
    """Search the words that the generators and excluded rows span, outside the excluded row space.

    ``upper_bound`` and ``get_automorphisms`` are as for ``_search_css``, and the other arguments
    are those of ``get_distance_from_generators``, validated.
    """
    tuning = get_tuning()
    if len(excluded):
        # An adapted basis and its quotient labels (Section 3.4 of brouwer_zimmermann.md).
        basis, labels = get_nested_code_basis(generators, excluded)
        if labels.shape[1] == 0:
            return NO_DISTANCE
    else:
        basis = get_independent_rows(generators)
        if not len(basis):
            return NO_DISTANCE
        # Labels with no columns exclude nothing.
        labels = np.empty((len(basis), 0), dtype=np.uint8)

    if method == "brute_force":
        distance = get_nested_distance_brute_force(
            basis,
            labels,
            cutoff=cutoff,
            batch_exponent=tuning.enumeration_batch_exponent,
            symplectic=symplectic,
        )
        return _apply_upper_bound(distance, upper_bound)

    # A Pauli string weighs half the Hamming weight of its image under symplectic_to_hamming
    # (Section 2.7 of brouwer_zimmermann.md).
    divisor = 1
    if symplectic:
        num_qubits = basis.shape[1] // 2
        basis = symplectic_to_hamming(basis)
        divisor = 2
        if get_automorphisms is not None:
            get_automorphisms = functools.partial(
                _get_lifted_automorphisms, get_automorphisms, num_qubits
            )
    distance = get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=cutoff * divisor,
        weight_divisor=divisor,
        upper_bound=None if upper_bound is None else upper_bound * divisor,
        get_automorphisms=get_automorphisms,
        tuning=tuning,
    )
    return distance // divisor


def _as_matrix_and_excluded(
    matrix: npt.ArrayLike, excluded: npt.ArrayLike | None, name: str
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Check a matrix and the excluded rows, and return them as binary matrices of one width.

    ``name`` names the matrix in errors.  Without ``excluded``, the excluded matrix has no rows.
    """
    if excluded is None:
        binary_matrix = as_binary_matrix(matrix, name)
        return binary_matrix, np.empty((0, binary_matrix.shape[1]), dtype=np.uint8)
    return as_binary_matrix_pair(matrix, excluded, (name, "excluded"))


def _get_num_variables(num_cols: int, symplectic: bool) -> int:
    """The number of bits, or of qubits if the rows of a matrix are symplectic."""
    if not symplectic:
        return num_cols
    if num_cols % 2:
        raise ValueError("Symplectic rows must have an even number of columns")
    return num_cols // 2


def _get_automorphism_getter(
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None, num_variables: int
) -> Callable[[], npt.NDArray[np.int_] | None] | None:
    """A function that returns the given automorphisms, checked to be permutations, or None.

    A function that the caller gives is called at most once, when the searches first ask for it.
    """
    if automorphisms is None:
        return None
    if not callable(automorphisms):
        permutations = _as_permutations(automorphisms, num_variables)
        return lambda: permutations

    get_automorphisms = automorphisms

    @functools.cache
    def get_permutations() -> npt.NDArray[np.int_] | None:
        result = get_automorphisms()
        return None if result is None else _as_permutations(result, num_variables)

    return get_permutations


def _as_permutations(automorphisms: npt.ArrayLike, num_variables: int) -> npt.NDArray[np.int_]:
    """Check that the rows of an array are permutations of the variables, and return the array."""
    array = np.asarray(automorphisms)
    if (
        array.ndim != 2
        or array.shape[1] != num_variables
        or not np.array_equal(
            np.sort(array, axis=1), np.broadcast_to(np.arange(num_variables), array.shape)
        )
    ):
        raise ValueError(
            "The automorphisms must be a two-dimensional array whose rows are permutations of"
            f" range({num_variables})"
        )
    return array.astype(np.int_)


def _get_weight_limit(cutoff: int, max_weight: SupportsIndex | None) -> tuple[int, int | None]:
    """The cutoff, capped by a weight limit, and the upper bound ``max_weight + 1`` (or None)."""
    if max_weight is None:
        return cutoff, None
    max_weight = as_integer(max_weight, "max_weight")
    if max_weight < 0:
        raise ValueError(f"max_weight must be nonnegative, not {max_weight}")
    return min(cutoff, max_weight), max_weight + 1


def _apply_upper_bound(distance: int, upper_bound: int | None) -> int:
    """The smaller of a distance and an optional upper bound."""
    return distance if upper_bound is None else min(distance, upper_bound)


def _get_lifted_automorphisms(
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None], num_qubits: int
) -> npt.NDArray[np.int_] | None:
    """Call a function that returns qubit permutations, and lift them to Hamming coordinates.

    The coordinates are ``(x | z | x + z)``, as in ``symplectic_to_hamming`` (Section 3.7 of
    ``brouwer_zimmermann.md``).  The result is None if ``get_automorphisms`` returns None.
    """
    automorphisms = get_automorphisms()
    if automorphisms is None:
        return None
    return np.hstack([automorphisms + block * num_qubits for block in range(3)])


####################################################################################################
# method selectors


def _validate_distance_method(
    method: DistanceMethod,
) -> _CanonicalDistanceMethod:
    """Validate the name of a distance method, and return the canonical name of the method."""
    if not isinstance(method, str) or method not in _DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' (or its alias"
            " 'information_sets'), 'brute_force', or 'connected_cluster'."
        )
    return _DISTANCE_METHODS[method]


def _reject_connected_cluster(
    method: _CanonicalDistanceMethod,
) -> Literal["brouwer_zimmermann", "brute_force"]:
    """Return a method other than ``"connected_cluster"``, which needs parity checks."""
    if method == "connected_cluster":
        raise ValueError(
            "The 'connected_cluster' distance method searches parity checks rather than"
            " generators: use get_distance_from_checks, get_distance_css, or get_distance_pauli"
        )
    return method
