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
from qldpc.objects import PAULIS_XZ, Pauli, PauliXZLike

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
from ._connected_support.search import get_distance_connected_support
from ._tuning import get_tuning

DistanceMethod = Literal[
    "brouwer_zimmermann", "brute_force", "information_sets", "connected_support"
]
"""The methods of exact distance searches.

- ``"brouwer_zimmermann"``, the default of every search, and its alias ``"information_sets"``: the
  Brouwer--Zimmermann algorithm, which enumerates words of increasing weight in several information
  sets, and stops once a lower bound that the sets prove meets the lightest word found.
- ``"connected_support"``: a search of the words whose supports the parity checks of a code connect,
  which is fast for codes with sparse checks.  Functions that take generators rather than checks
  reject it.  For a subsystem code, :func:`get_distance_css` and :func:`get_distance_pauli` derive
  its stabilizers from its gauge generators and use them as the checks.  These can be dense even if
  the gauge generators are sparse.
- ``"brute_force"``: the enumeration of every word, for small codes and tests.

:class:`~qldpc.codes.distance.CosetLeaderSearch` and
:func:`~qldpc.circuits.certify_bounded_fault_propagation`, which uses it, reject ``"brute_force"``.
The page :doc:`/exact_search_tuning` of the documentation says how to make costly searches faster.
"""


class NoDistance(float, enum.ReprEnum):
    """The distance of an object with no nontrivial logical operator to minimize over.

    A code that encodes nothing, or a circuit with no fault that flips a logical observable without
    flipping a detector, has no distance.  qLDPC then returns ``NO_DISTANCE``, the only member of
    this enum, which behaves as a float ``nan``:

    - It prints as ``nan``, also in the representation of a tuple.
    - It equals nothing, not even itself, and ordering comparisons with it are False.
    - Arithmetic with it gives ``nan``, and NumPy and pandas treat it as missing.

    Test for it with ``distance is NO_DISTANCE``, or with ``math.isnan(distance)``, which also
    catches ``nan`` from arithmetic.  A result that can be ``NO_DISTANCE`` is annotated
    ``int | Literal[NoDistance.NAN]``, so that type checkers treat it as an ``int`` once
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

_CanonicalDistanceMethod = Literal["brouwer_zimmermann", "brute_force", "connected_support"]

# The canonical name of every accepted exact-distance method name, including aliases.
_DISTANCE_METHODS: dict[str, _CanonicalDistanceMethod] = {
    "brouwer_zimmermann": "brouwer_zimmermann",
    "brute_force": "brute_force",
    "information_sets": "brouwer_zimmermann",
    "connected_support": "connected_support",
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
    is a sum of rows of ``excluded`` alone.  For a classical code, pass its generator matrix, and
    nothing excluded, to get its distance: the minimum weight of a nonzero code word.  For a quantum
    code, pass its logical operators, and exclude its stabilizers, to get its distance: the minimum
    weight of a nontrivial logical operator.  For a subsystem code, exclude its gauge generators,
    such as the rows of the ``matrix`` of a :class:`~qldpc.codes.QuditCode`, to get its distance,
    which is dressed: its logical operators may be multiplied by gauge operators.  If the generators
    are bare logical operators, such as those that
    :meth:`~qldpc.codes.common.QuditCode.get_logical_ops` returns, excluding only the stabilizers of
    the code gives its bare distance instead (see :func:`get_distance_from_checks`).  Rows may be
    redundant.

    Args:
        generators: A binary matrix whose rows, together with those of ``excluded``, span the words.
        excluded: A binary matrix with as many columns, whose rows span the trivial words.  By
            default, only the zero word is trivial.  If either matrix is an empty list or has shape
            ``(0, 0)``, it takes the width of the other.
        symplectic: If False (the default), each row is a binary word, which weighs its number of
            nonzero entries.  If True, each row is a Pauli string on ``n`` qubits, with ``2 n``
            entries: the support of its X part, then the support of its Z part.  A Pauli string
            weighs the number of qubits on which it acts nontrivially.
        cutoff: Exit early once a nontrivial word of weight at most ``cutoff`` is found (see
            "Returns").
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``, or
            ``"brute_force"`` (see :data:`DistanceMethod`).  The ``"connected_support"`` method
            needs parity checks: use :func:`get_distance_from_checks`.
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1``.  The search stops as soon as it
            proves that every nontrivial word is heavier than ``max_weight``, which can be much
            faster than computing a large distance exactly.  The limit also caps ``cutoff``, so an
            early exit still returns the weight of a word within the limit; pass
            ``cutoff=max_weight`` as well to stop at the first such word.
        automorphisms: Permutations of the bits, or of the qubits if ``symplectic`` is True, that
            map the row space of ``excluded``, and that of ``generators`` and ``excluded`` together,
            onto themselves.  Pass them as an array whose row ``p`` maps bit or qubit ``j`` to
            ``p[j]``, or as a function that returns such an array or None.  A search asks for them
            at most once, when it reaches a costly stage (see :doc:`/exact_search_tuning`), and only
            then calls a function.  The Brouwer--Zimmermann method then verifies that the
            permutations are automorphisms, and uses them to search fewer information sets.  The
            brute-force method does not use them.

    Returns:
        The minimum weight of a nontrivial word.  As soon as the lightest nontrivial word seen so
        far has weight ``<= cutoff``, the search stops early and returns that weight, an upper bound
        on the distance in ``[distance, cutoff]``.  With the default ``cutoff=1``, the result is
        always exact, since a nontrivial word weighs at least one; pass ``cutoff=0`` to make that
        explicit.  If ``max_weight`` is given, the result is the smaller of this value and
        ``max_weight + 1``, so a result of ``max_weight + 1`` means that every nontrivial word is
        heavier than ``max_weight``.  Inputs without nontrivial words have no distance and give
        ``NO_DISTANCE`` (see :class:`NoDistance`), whatever ``cutoff`` and ``max_weight`` are.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``generators`` or
            ``excluded`` is not array-like, as when a code is given instead of a matrix.
        ValueError: If the method is unknown or is ``"connected_support"``, if ``max_weight`` is
            negative, if the matrices are not binary matrices with the same number of columns, if
            neither matrix has a width, as when both are empty lists, if
            they have an odd number of columns while ``symplectic`` is True, if ``automorphisms``
            are not permutations of the bits or qubits, or if a search uses them and finds that one
            of them is not an automorphism.
    """
    method = _reject_connected_support(_validate_distance_method(method))
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
    For a classical code, pass its parity check matrix, and nothing excluded, to get its distance.
    For the X logical operators of a CSS stabilizer code, pass its Z checks, and exclude its X
    checks.  Checks may be redundant, which can speed up the connected-support method.

    The distance of a subsystem code is dressed: its logical operators may be multiplied by gauge
    operators.  :func:`get_distance_css` and :func:`get_distance_pauli` find it from the gauge
    generators, and so does this function, given the stabilizers as the checks and the gauge
    generators as ``excluded``.  With these roles exchanged, it finds the bare distance instead:
    the minimum weight of a nontrivial logical operator that commutes with every gauge operator.
    For a stabilizer code, both are its distance.  The bare distance of a binary quantum code
    ``code``, whose ``matrix`` holds its gauge generators, is::

        get_distance_from_checks(code.matrix, code.get_stabilizer_ops(), symplectic=True)

    For a CSS code, two smaller classical searches give it too.  The bare X-distance is::

        get_distance_from_checks(code.matrix_z, code.get_stabilizer_ops(Pauli.X))

    The bare Z-distance exchanges X and Z, and the bare distance is the smaller of the two.

    A word that satisfies the checks counts as a logical operator unless it is a sum of excluded
    rows.  So exclude all of the stabilizers for the bare distance, and all of the gauge generators
    for the dressed distance.  Excluding nothing, or only some stabilizers, counts stabilizers as
    logical operators.  For a subsystem code, passing its stabilizers as both the checks and
    ``excluded``, such as its Z-type and X-type stabilizers, counts gauge operators that are not
    stabilizers as logical operators.

    Args:
        checks: A binary matrix whose rows are the checks.
        excluded: A binary matrix with as many columns, whose rows satisfy the checks and span the
            trivial words.  By default, only the zero word is trivial.  It is required if
            ``symplectic`` is True, since the stabilizers of a quantum code satisfy its checks but
            are trivial.  If either matrix is an empty list or has shape ``(0, 0)``, it takes the
            width of the other.
        symplectic: If False (the default), each row is a binary word, which weighs its number of
            nonzero entries.  If True, each row is a Pauli string on ``n`` qubits, with ``2 n``
            entries: the support of its X part, then the support of its Z part.  A Pauli string
            weighs the number of qubits on which it acts nontrivially.
        cutoff: Exit early once a nontrivial word of weight at most ``cutoff`` is found (see
            "Returns").
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_support"``, or ``"brute_force"`` (see :data:`DistanceMethod`).  The
            Brouwer--Zimmermann and brute-force methods search among the words that satisfy the
            checks, and the connected-support method searches the words whose supports the checks
            connect.
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1``.  The search stops as soon as it
            proves that every nontrivial word is heavier than ``max_weight``, which can be much
            faster than computing a large distance exactly.  The limit also caps ``cutoff``, so an
            early exit still returns the weight of a word within the limit; pass
            ``cutoff=max_weight`` as well to stop at the first such word.
        automorphisms: Permutations of the bits, or of the qubits if ``symplectic`` is True, that
            map the row space of ``checks``, and that of ``excluded``, onto themselves.  Pass them
            as an array whose row ``p`` maps bit or qubit ``j`` to ``p[j]``, or as a function that
            returns such an array or None.  A search asks for them at most once, when it reaches a
            costly stage (see :doc:`/exact_search_tuning`), and only then calls a function.  The
            Brouwer--Zimmermann and connected-support methods then verify that the permutations are
            automorphisms, and use them to search less.  The brute-force method does not use them.

    Returns:
        The minimum weight of a nontrivial word, as for :func:`get_distance_from_generators`,
        including its early exit at ``cutoff``, its limit ``max_weight``, and ``NO_DISTANCE`` for
        inputs without nontrivial words.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks`` or
            ``excluded`` is not array-like, as when a code is given instead of a matrix.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the matrices are not
            binary matrices with the same number of columns, if neither matrix has a width, as when
            both are empty lists, if they have an odd number of columns
            while ``symplectic`` is True, if ``symplectic`` is True and ``excluded`` is not given,
            if a row of ``excluded`` does not satisfy the checks, if ``automorphisms`` are not
            permutations of the bits or qubits, or if a search uses them and finds that one of them
            is not an automorphism.
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
    errors, and the rows of ``checks_z`` are those of its Z-type checks.  If the X and Z checks
    commute, they generate the stabilizer group.  An X logical operator is then an X-type operator
    that commutes with every Z-type check, so its support is in the null space of ``checks_z``.  It
    is trivial if it is an X-type stabilizer, whose support is in the row space of ``checks_x``.  Z
    logical operators are likewise in the null space of ``checks_x``, and trivial in the row space
    of ``checks_z``.

    If the X and Z checks do not all commute, they generate the gauge group of a subsystem code.
    Its stabilizers are the gauge operators that commute with every gauge operator, and its logical
    operators are dressed: they may be multiplied by gauge operators.  An X logical operator then
    commutes with every Z-type stabilizer, and it is trivial if it is an X-type gauge operator.  The
    result is the dressed distance of the code, and :func:`get_distance_from_checks` says how to
    find its bare distance instead.  To be sure that the checks define a stabilizer code, build a
    code with ``CSSCode(checks_x, checks_z, is_subsystem_code=False)``, which raises an error if
    they do not commute.

    Args:
        checks_x: A binary matrix of the X-type checks, which may be redundant.
        checks_z: A binary matrix of the Z-type checks, with as many columns, which may be
            redundant.  If either matrix is an empty list or has shape ``(0, 0)``, it takes the
            width of the other.
        pauli: ``Pauli.X`` (or ``"X"``) for the X-distance, the minimum weight of a nontrivial X
            logical operator, or ``Pauli.Z`` (or ``"Z"``) for the Z-distance.  If None (the
            default), the result is the distance of the code, the smaller of the two.  The
            Brouwer--Zimmermann and connected-support methods then search both types at once, which
            is faster than two calls.
        cutoff: Exit early once a nontrivial logical operator of weight at most ``cutoff`` is found
            (see :func:`get_distance_from_generators`).
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_support"``, or ``"brute_force"`` (see :data:`DistanceMethod`, also for
            subsystem codes).
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1`` (see
            :func:`get_distance_from_generators`).
        automorphisms: Permutations of the qubits that map the row space of ``checks_x``, and that
            of ``checks_z``, onto themselves.  Pass them as an array whose row ``p`` maps qubit
            ``j`` to ``p[j]``, or as a function that returns such an array or None.  A search asks
            for them at most once, when it reaches a costly stage (see :doc:`/exact_search_tuning`),
            and only then calls a function.  The Brouwer--Zimmermann and connected-support methods
            then verify that the permutations are automorphisms, and use them to search less.  The
            brute-force method does not use them.

    Returns:
        The minimum weight of a nontrivial logical operator of the requested types, as for
        :func:`get_distance_from_generators`, including its early exit at ``cutoff``, its limit
        ``max_weight``, and ``NO_DISTANCE`` for a code that encodes no qubits.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks_x`` or
            ``checks_z`` is not array-like, as when a code is given instead of its checks.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the checks are not
            binary matrices with the same number of columns, if both are empty lists, which have no
            width, if ``pauli`` is not X, Z, or None, if
            ``automorphisms`` are not permutations of the qubits, or if a search uses them and finds
            that one of them is not an automorphism.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    gauge_x, gauge_z = as_binary_matrix_pair(checks_x, checks_z, ("checks_x", "checks_z"))
    paulis = PAULIS_XZ if pauli is None else [Pauli.coerce_xz(pauli)]
    get_automorphisms = _get_automorphism_getter(automorphisms, gauge_x.shape[1])

    # The logical operators of each type satisfy the stabilizers of the other type, and are trivial
    # if they are gauge operators of their own type.
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

    # Search the requested types at once, except by brute force, which has no joint search, and so
    # searches each type in turn and takes the minimum.
    if method == "connected_support":
        return get_distance_connected_support(
            problems,
            cutoff=cutoff,
            upper_bound=upper_bound,
            get_automorphisms=get_automorphisms,
            tuning=get_tuning(),
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


def get_distance_pauli(
    checks: npt.ArrayLike,
    *,
    cutoff: int = 1,
    method: DistanceMethod = "brouwer_zimmermann",
    max_weight: SupportsIndex | None = None,
    automorphisms: npt.ArrayLike | Callable[[], npt.ArrayLike | None] | None = None,
) -> int | Literal[NoDistance.NAN]:
    """The distance of a binary stabilizer or subsystem code, from its parity check matrix.

    Each row of ``checks`` represents a Pauli string on ``n`` qubits as a vector ``[P_x|P_z]`` of
    length ``2 n``, where ``P_x`` and ``P_z`` are the supports of its X-type and Z-type parts, as in
    :class:`~qldpc.codes.QuditCode`.  If all of these Pauli strings commute, the code is a
    stabilizer code, and the rows generate its stabilizer group.  Its distance is the minimum weight
    of a nontrivial logical operator: a Pauli string that commutes with every row, but is not a
    stabilizer.  A Pauli string weighs the number of qubits on which it acts nontrivially.

    Otherwise, the code is a subsystem code, and the rows generate its gauge group.  Its stabilizers
    are the gauge operators that commute with every gauge operator, and its logical operators are
    dressed: they may be multiplied by gauge operators.  Its distance is then the dressed distance:
    the minimum weight of a Pauli string that commutes with every stabilizer, but is not a gauge
    operator.  :func:`get_distance_from_checks` says how to find its bare distance instead.  To be
    sure that the checks define a stabilizer code, build a code with
    ``QuditCode(checks, is_subsystem_code=False)``, which raises an error if they do not commute.

    Args:
        checks: A binary matrix whose rows represent Pauli strings, which may be redundant.
        cutoff: Exit early once a nontrivial logical operator of weight at most ``cutoff`` is found
            (see :func:`get_distance_from_generators`).
        method: ``"brouwer_zimmermann"`` (the default), its alias ``"information_sets"``,
            ``"connected_support"``, or ``"brute_force"`` (see :data:`DistanceMethod`, also for
            subsystem codes).
        max_weight: If given, a nonnegative integer limit on the weights to search: the result is
            then the smaller of the distance and ``max_weight + 1`` (see
            :func:`get_distance_from_generators`).
        automorphisms: Permutations of the qubits that map the row space of ``checks`` onto itself.
            Pass them as an array whose row ``p`` maps qubit ``j`` to ``p[j]``, or as a function
            that returns such an array or None.  A search asks for them at most once, when it
            reaches a costly stage (see :doc:`/exact_search_tuning`), and only then calls a
            function.  The Brouwer--Zimmermann and connected-support methods then verify that the
            permutations are automorphisms, and use them to search less.  The brute-force method
            does not use them.

    Returns:
        The minimum weight of a nontrivial logical operator, as for
        :func:`get_distance_from_generators`, including its early exit at ``cutoff``, its limit
        ``max_weight``, and ``NO_DISTANCE`` for a code that encodes no qubits.

    Raises:
        TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``checks`` is not
            array-like, as when a code is given instead of its checks.
        ValueError: If the method is unknown, if ``max_weight`` is negative, if the checks are not a
            binary matrix with an even number of columns, if ``automorphisms`` are not permutations
            of the qubits, or if a search uses them and finds that one of them is not an
            automorphism.
    """
    method = _validate_distance_method(method)
    cutoff, upper_bound = _get_weight_limit(cutoff, max_weight)
    gauge = as_binary_matrix(checks, "checks")
    num_qubits = _get_num_variables(gauge.shape[1], symplectic=True)
    return _search_checks(
        get_pauli_stabilizers(gauge),
        gauge,
        symplectic=True,
        cutoff=cutoff,
        upper_bound=upper_bound,
        method=method,
        get_automorphisms=_get_automorphism_getter(automorphisms, num_qubits),
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
) -> int | Literal[NoDistance.NAN]:
    """Search the words that satisfy the checks, outside the row space of excluded rows.

    The excluded rows satisfy the checks.  ``upper_bound``, if given, is ``max_weight + 1``, which
    the result does not exceed, and ``get_automorphisms`` returns checked permutations or None (see
    ``_get_automorphism_getter``).  The other arguments are those of the public distance functions,
    validated.
    """
    if method != "connected_support":
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

    # The words that satisfy the checks form a space of dimension the number of columns minus the
    # rank of the checks, which holds the row space of the excluded rows.
    dimension = checks.shape[1] - len(get_independent_rows(checks))
    if dimension == len(get_independent_rows(excluded)):
        return NO_DISTANCE
    return get_distance_connected_support(
        [(checks, excluded)],
        cutoff=cutoff,
        upper_bound=upper_bound,
        homogeneous=not symplectic,
        get_automorphisms=get_automorphisms,
        tuning=get_tuning(),
    )


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

    ``upper_bound``, if given, is ``max_weight + 1``, which the result does not exceed, and
    ``get_automorphisms`` returns checked permutations or None (see ``_get_automorphism_getter``).
    The other arguments are those of ``get_distance_from_generators``, validated.
    """
    tuning = get_tuning()
    if len(excluded):
        # Separate an independent excluded basis from a complement representing the quotient.
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
            homogeneous=not symplectic,
        )
        return _apply_upper_bound(distance, upper_bound)

    # A Pauli string weighs half the Hamming weight of its image under symplectic_to_hamming.
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

    The matrix is named ``name`` in error messages.  If ``excluded`` is None, the excluded matrix
    has no rows, and otherwise either matrix can take the width of the other (see
    ``as_binary_matrix_pair``).
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


def _get_lifted_automorphisms(
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None], num_qubits: int
) -> npt.NDArray[np.int_] | None:
    """Call a function that returns qubit permutations, and lift them to Hamming coordinates.

    The Hamming coordinates are ``(x | z | x + z)``, as in ``symplectic_to_hamming``.  Each lifted
    permutation moves the coordinates of each of the three blocks as the permutation moves the
    qubits.  The map of ``symplectic_to_hamming`` commutes with these permutations, so a lifted
    permutation preserves the image of a space exactly if the permutation preserves the space.  The
    result is None if ``get_automorphisms`` returns None.
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
    """Validate an exact-distance method selector, and return its canonical name.

    The alias ``"information_sets"`` resolves to ``"brouwer_zimmermann"``.
    """
    if not isinstance(method, str) or method not in _DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' (or its alias"
            " 'information_sets'), 'brute_force', or 'connected_support'."
        )
    return _DISTANCE_METHODS[method]


def _reject_connected_support(
    method: _CanonicalDistanceMethod,
) -> Literal["brouwer_zimmermann", "brute_force"]:
    """Return a method other than ``"connected_support"``, which needs parity checks."""
    if method == "connected_support":
        raise ValueError(
            "The 'connected_support' distance method searches parity checks rather than"
            " generators: use get_distance_from_checks, get_distance_css, or get_distance_pauli"
        )
    return method
