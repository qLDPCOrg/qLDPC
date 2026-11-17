# SPDX-License-Identifier: Apache-2.0

"""Choose logical operators of binary CSS codes, one per class of a basis, for parallel rounds."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import time
from collections.abc import Sequence
from typing import Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.sparse

from qldpc import codes
from qldpc._binary import get_kernel, pack_bits, row_reduce_binary, unpack_bits
from qldpc.codes.distance import count_logical_ops_css
from qldpc.codes.distance._binary import get_nested_code_basis
from qldpc.objects import Pauli, PauliXZ

from ._sectors import Sector, get_sector
from .evaluation import (
    LogicalBasisMetrics,
    LogicalOpsMetrics,
    _as_binary_matrix,
    evaluate_logical_ops,
)
from .lightest_basis import _get_min_weight, _get_min_weight_word

# Candidates are listed from information sets, that of the column order and then random ones, each
# contributing the words that combine at most this many of its rows, until the list matches the
# exact count or the samples run out.
_MAX_ROWS_PER_SAMPLE = 3
_MAX_SAMPLES = 2000

####################################################################################################
# rounds


def optimize_logical_rounds(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike | None = None,
    *,
    weight_slack: int = 0,
    logicals: Literal["dressed", "bare"] = "dressed",
    time_limit: float | None = None,
) -> OptimizedLogicalRounds:
    """Choose an operator in each class of a basis of logical operators, for few parallel rounds.

    Operators that share no qubit can be acted on at the same time, for example by parallel lattice
    surgery.  For each Pauli type, this function keeps the logical class of every operator of
    ``logical_ops``, and chooses an operator in each class from its candidates: the operators of the
    class with weight at most the minimum weight of the class plus ``weight_slack``.  It minimizes
    the number of rounds of disjoint operators needed to act on all of the chosen operators.  Among
    the choices with the fewest rounds, it minimizes the largest load (the number of chosen
    operators on one qubit), and then the total weight.  Keeping the classes keeps the X-type and
    Z-type operators paired, so the result is a complete basis.

    The candidates of each class are listed by sampling information sets, and a list is proven
    complete when its length equals an exact count from
    :func:`qldpc.codes.distance.count_logical_ops_css`.  Then an integer program for each number of
    rounds, from a lower bound upward, is solved with ``scipy.optimize.milp`` until one has a
    solution.  The choices that the solver returns are checked against the constraints, but its
    proofs are trusted.  These are its proofs that fewer rounds are impossible, and that no choice
    with as few rounds is better.  The result is optimal unless a list is incomplete or the time
    limit runs out.  Counting, listing, and solving each generally take time exponential in the
    size of the code.

    In some codes, such as bivariate bicycle codes, all operators of a class have weights of the
    same parity, so a slack of 1 allows no new operators, and 2 is the next useful value.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        logical_ops: A complete basis with shape ``(2k, 2n)``, laid out as by
            :meth:`qldpc.codes.CSSCode.get_logical_ops`, whose classes are kept.  Defaults to the
            code's own logical operators; the ``logical_ops`` of the result of
            :func:`~qldpc.experimental.logical_bases.optimize_logical_basis` are a good choice.
        weight_slack: How much heavier than the minimum weight of its class an operator may be.
        logicals: For a subsystem code, ``"dressed"`` (the default) uses logical operators that
            commute with the stabilizers, while ``"bare"`` uses those that commute with every gauge
            operator.  The two are the same for a stabilizer code.  Only a basis of bare operators
            can be passed to :meth:`qldpc.codes.CSSCode.set_logical_ops`.
        time_limit: If given, a limit in seconds on the time spent solving integer programs, for
            both Pauli types together.  The X-type operators are chosen first, so they can use up
            the limit.  Finding minimum weights, and counting and listing candidates, are not
            limited.

    Returns:
        The chosen basis, with the round of each operator, a lower bound on the rounds, and why
        the search stopped.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, if ``logicals`` is invalid, if ``logical_ops`` is not
            a complete basis as :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`
            describes, or if ``weight_slack`` or ``time_limit`` is negative.
        RuntimeError: If the integer-program solver fails, or returns a result that cannot be
            right, such as a choice that breaks its constraints.
    """
    # check the arguments and the basis
    if weight_slack < 0 or (time_limit is not None and time_limit < 0):
        raise ValueError("weight_slack and time_limit must be nonnegative")
    ops = code.get_logical_ops() if logical_ops is None else logical_ops
    evaluate_logical_ops(code, ops, logicals=logicals)
    k, n = code.dimension, len(code)
    matrix = _as_binary_matrix(ops, (2 * k, 2 * n))

    # choose the operators of each Pauli type, sharing the time limit
    remaining = time_limit
    chosen = np.zeros_like(matrix)
    rounds = {}
    paulis: list[tuple[PauliXZ, slice, slice]] = [
        (Pauli.X, slice(None, k), slice(None, n)),
        (Pauli.Z, slice(k, None), slice(n, None)),
    ]
    for pauli, rows, cols in paulis:
        sector = get_sector(code, pauli, logicals)
        chosen[rows, cols], rounds[pauli], solve_time = _choose_operators(
            sector, matrix[rows, cols], weight_slack, remaining
        )
        if remaining is not None:
            remaining = max(remaining - solve_time, 0)
    return OptimizedLogicalRounds(chosen, rounds[Pauli.X], rounds[Pauli.Z])


@dataclasses.dataclass(frozen=True, eq=False)
class OptimizedLogicalRounds:
    """A complete basis of logical operators, chosen so that they can be acted on in few rounds.

    Attributes:
        logical_ops: The basis, with shape ``(2k, 2n)``, laid out as by
            :meth:`qldpc.codes.CSSCode.get_logical_ops`.
        x: The rounds of the X-type operators.
        z: The rounds of the Z-type operators.
    """

    logical_ops: npt.NDArray[np.uint8]
    x: LogicalOpsRounds
    z: LogicalOpsRounds

    @property
    def optimal(self) -> bool:
        """Whether the choice of operators of both Pauli types is proven optimal."""
        return self.x.optimal and self.z.optimal

    @property
    def metrics(self) -> LogicalBasisMetrics:
        """The metrics of the X-type and Z-type operators.

        These are the metrics that :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`
        returns for this basis.
        """
        k, n = len(self.logical_ops) // 2, self.logical_ops.shape[1] // 2
        ops_x, ops_z = self.logical_ops[:k, :n], self.logical_ops[k:, n:]
        return LogicalBasisMetrics(
            x=LogicalOpsMetrics._from_ops(ops_x), z=LogicalOpsMetrics._from_ops(ops_z)
        )


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalOpsRounds:
    """The rounds in which to act on the logical operators of one Pauli type.

    Operators in one round share no qubit.

    Attributes:
        rounds: The round of each operator, numbered from 0.
        lower_bound: A lower bound on the number of rounds of every choice of candidates in the
            classes of these operators, with candidates as
            :func:`~qldpc.experimental.logical_bases.optimize_logical_rounds` defines them.
        stop_reason: Why the search stopped.  ``"optimal"``: the choice is proven optimal, with
            the fewest rounds, then the smallest largest load, then the smallest total weight.
            ``"incomplete_candidates"``: the sampling did not find every candidate, so a better
            choice may exist.  ``"time_limit"``: the candidates are complete, but the time limit
            ran out before the choice was proven optimal.
    """

    rounds: npt.NDArray[np.int_]
    lower_bound: int
    stop_reason: Literal["optimal", "time_limit", "incomplete_candidates"]

    @property
    def num_rounds(self) -> int:
        """The number of rounds."""
        return int(self.rounds.max(initial=-1)) + 1

    @property
    def optimal(self) -> bool:
        """Whether the choice of operators is proven optimal."""
        return self.stop_reason == "optimal"


####################################################################################################
# choosing operators


def _choose_operators(
    sector: Sector, ops: npt.NDArray[np.uint8], weight_slack: int, time_limit: float | None
) -> tuple[npt.NDArray[np.uint8], LogicalOpsRounds, float]:
    """Choose an operator in the class of each row of ``ops``, to act on them in few rounds.

    Also returns the time spent solving integer programs, which ``time_limit`` limits.
    """
    k, n = ops.shape
    if k == 0:
        return ops, LogicalOpsRounds(np.zeros(0, dtype=int), 0, "optimal"), 0.0

    # list the candidates of each class
    candidates = []
    min_weights = []
    complete = True
    for op in ops:
        space = np.vstack([sector.trivial, op])
        min_weights.append(_get_min_weight(space, sector.trivial))
        words, found_all = _list_candidates(op, sector.trivial, min_weights[-1], weight_slack)
        candidates.append(words)
        complete &= found_all

    # bound the rounds from below: a round of disjoint operators covers at most n qubits, and the
    # bound from common qubits holds only if no candidate is missing
    lower_bound = max(1, math.ceil(sum(min_weights) / n))
    if complete:
        lower_bound = max(lower_bound, _get_common_qubit_bound(candidates))

    # try more and more rounds, until a choice exists or the time limit runs out; k rounds, with
    # one class in each, always suffice
    solve_time = 0.0
    for num_rounds in range(lower_bound, k + 1):
        timeout = None if time_limit is None else max(time_limit - solve_time, 0)
        start = time.monotonic()
        status, choice = _solve_rounds(candidates, num_rounds, timeout)
        solve_time += time.monotonic() - start
        if status != "infeasible":
            break
        # infeasibility is proven only over the listed candidates
        if complete:
            lower_bound = num_rounds + 1
    if status == "infeasible":
        raise RuntimeError("The integer-program solver claims that no choice has k rounds")

    # without a choice, use the lightest candidates, in greedily colored rounds
    if choice is None:
        lightest = np.array([words[0] for words in candidates])
        choice = lightest, _get_greedy_rounds(lightest)
    chosen, rounds_of_ops = choice

    # missing candidates take precedence over the time limit as the reason to stop
    stop_reason: Literal["optimal", "time_limit", "incomplete_candidates"]
    if not complete:
        stop_reason = "incomplete_candidates"
    elif status == "optimal":
        stop_reason = "optimal"
    else:
        stop_reason = "time_limit"
    return chosen, LogicalOpsRounds(rounds_of_ops, lower_bound, stop_reason), solve_time


def _list_candidates(
    op: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8], min_weight: int, weight_slack: int
) -> tuple[list[npt.NDArray[np.uint8]], bool]:
    """The operators of the class of ``op`` within ``weight_slack`` of its minimum weight.

    Returns the operators found, lightest first, and whether they are all of them.
    """
    # count the candidates, and list the sets of rows to combine
    max_weight = min_weight + weight_slack
    expected = _count_candidates(op, trivial, min_weight, max_weight)
    basis, labels = get_nested_code_basis(op[None, :], trivial)
    rank, n = basis.shape
    combinations = [
        np.array(list(itertools.combinations(range(rank), size)), dtype=np.int64).reshape(-1, size)
        for size in range(1, _MAX_ROWS_PER_SAMPLE + 1)
    ]

    # combine few rows of the basis reduced on the information set of the column order, then on
    # random ones
    found: dict[bytes, npt.NDArray[np.uint64]] = {}
    rng = np.random.default_rng(0)
    for sample in range(_MAX_SAMPLES):
        order = np.arange(n) if sample == 0 else rng.permutation(n)
        reduced, reduced_labels, _ = row_reduce_binary(basis, labels, columns=order)
        assert reduced_labels is not None
        packed = pack_bits(reduced)
        for rows in combinations:
            words = np.bitwise_xor.reduce(packed[rows], axis=1)
            in_class = np.bitwise_xor.reduce(reduced_labels[rows, 0], axis=1).astype(bool)
            weights = np.bitwise_count(words).sum(axis=1)
            for word in words[in_class & (weights <= max_weight)]:
                found[word.tobytes()] = word
        if len(found) == expected:
            break

    # sampling rarely misses every candidate, but the list must not be empty
    if not found:
        space = np.vstack([trivial, op])
        word = pack_bits(_get_min_weight_word(space, trivial, min_weight)[None, :])[0]
        found[word.tobytes()] = word

    # sort the words lightest first, and otherwise in an order that does not depend on the sampling
    words = unpack_bits(np.array(list(found.values())), n)
    words = words[np.lexsort((*words.T[::-1], words.sum(axis=1)))]
    return list(words), len(found) == expected


def _count_candidates(
    op: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8], min_weight: int, max_weight: int
) -> int:
    """The number of operators of the class of ``op`` with weights in ``[min_weight, max_weight]``.

    They are the nontrivial X-type logical operators of an auxiliary stabilizer code, whatever the
    Pauli type of ``op``.  Its X-type checks are ``trivial``, and its Z-type checks have the span of
    ``trivial`` and ``op`` as their kernel.
    """
    checks_z = get_kernel(np.vstack([trivial, op]))
    return sum(
        count_logical_ops_css(trivial, checks_z, weight).count_x
        for weight in range(min_weight, min(max_weight, len(op)) + 1)
    )


def _get_common_qubit_bound(candidates: Sequence[Sequence[npt.NDArray[np.uint8]]]) -> int:
    """A lower bound on rounds: the most classes whose candidates all contain the same qubit."""
    common = np.array([np.bitwise_and.reduce(np.array(words), axis=0) for words in candidates])
    return int(common.astype(int).sum(axis=0).max())


def _solve_rounds(
    candidates: Sequence[Sequence[npt.NDArray[np.uint8]]],
    num_rounds: int,
    time_limit: float | None,
) -> tuple[
    Literal["optimal", "feasible", "infeasible", "time_limit"],
    tuple[npt.NDArray[np.uint8], npt.NDArray[np.int_]] | None,
]:
    """The best choice of one candidate and one round per class, if any choice exists.

    Operators in one round share no qubit.  The best choice minimizes the largest number of chosen
    operators on one qubit, and then their total weight.  Returns a status, and the chosen
    operators and their rounds, if any.  The status is ``"optimal"`` if the choice is proven best,
    ``"feasible"`` if it is found but not proven best within the time limit, ``"infeasible"`` if
    no choice exists, and ``"time_limit"`` if none is found within the time limit.

    Variable ``(c, j, r)`` is 1 if class ``c`` takes candidate ``j`` in round ``r``.  Class 0 is put
    in round 0, since rounds can be relabeled.  One more variable bounds the load of every qubit.
    """
    # index the variables by class, candidate, and round
    k, n = len(candidates), len(candidates[0][0])
    assignments = [
        (c, j, r)
        for c, words in enumerate(candidates)
        for j in range(len(words))
        for r in range(num_rounds if c else 1)
    ]
    supports = np.array([candidates[c][j] for c, j, _ in assignments], dtype=np.int64)
    num_vars = len(assignments)

    # each class takes one candidate in one round, no qubit is used twice in one round, and the
    # last variable is at least the number of chosen operators on each qubit
    classes = np.array([c for c, _, _ in assignments])
    rounds = np.array([r for _, _, r in assignments])
    variables, qubits = np.nonzero(supports)
    ones = np.ones(len(variables))
    one_per_class = scipy.sparse.csr_array(
        (np.ones(num_vars), (classes, np.arange(num_vars))), shape=(k, num_vars + 1)
    )
    disjoint = scipy.sparse.csr_array(
        (ones, (rounds[variables] * n + qubits, variables)), shape=(num_rounds * n, num_vars + 1)
    )
    loads = scipy.sparse.csr_array(
        (
            np.append(ones, -np.ones(n)),
            (np.append(qubits, np.arange(n)), np.append(variables, np.full(n, num_vars))),
        ),
        shape=(n, num_vars + 1),
    )
    constraints = [
        scipy.optimize.LinearConstraint(one_per_class, 1, 1),
        scipy.optimize.LinearConstraint(disjoint, 0, 1),
        scipy.optimize.LinearConstraint(loads, -np.inf, 0),
    ]

    # minimize the largest load first, then the total weight, which is at most k * n; a zero gap
    # makes the solver prove the total weight optimal, rather than within a relative tolerance
    weights = supports.sum(axis=1)
    milp = functools.partial(
        scipy.optimize.milp,
        np.append(weights, k * n + 1).astype(float),
        constraints=constraints,
        integrality=np.ones(num_vars + 1),
        bounds=scipy.optimize.Bounds(0, np.append(np.ones(num_vars), k)),
    )
    options: dict[str, float] = {"mip_rel_gap": 0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    start = time.monotonic()
    result = milp(options=options)

    # HiGHS's presolve fails on some small programs (status 4) that it solves without presolve
    if result.status == 4:
        if time_limit is not None:
            options["time_limit"] = max(time_limit - (time.monotonic() - start), 0)
        result = milp(options={**options, "presolve": False})

    # statuses 0, 1, and 2 are optimal, a time or iteration limit (these programs set no iteration
    # limit), and infeasible (or a malformed program, which these are not)
    if result.status == 2:
        return "infeasible", None
    if result.status not in (0, 1):
        raise RuntimeError(f"The integer program failed: {result.message}")
    if result.x is None:
        return "time_limit", None
    picked = [assignments[i] for i in np.flatnonzero(result.x[:num_vars] > 0.5)]
    status: Literal["optimal", "feasible"] = "optimal" if result.status == 0 else "feasible"
    return status, _check_choice(candidates, picked)


def _check_choice(
    candidates: Sequence[Sequence[npt.NDArray[np.uint8]]], picked: Sequence[tuple[int, int, int]]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
    """The operators and rounds of a choice from the solver, checked rather than trusted.

    ``picked`` holds the class, candidate, and round of each chosen operator.  The choice must take
    one candidate of each class, and operators in one round must share no qubit.
    """
    picked = sorted(picked)
    if [c for c, _, _ in picked] != list(range(len(candidates))):
        raise RuntimeError("The integer program returned a choice that breaks its constraints")
    chosen = np.array([candidates[c][j] for c, j, _ in picked], dtype=np.uint8)
    rounds = np.array([r for _, _, r in picked])
    if any(chosen[rounds == r].astype(int).sum(axis=0).max() > 1 for r in np.unique(rounds)):
        raise RuntimeError("The integer program returned a choice that breaks its constraints")
    return chosen, rounds


def _get_greedy_rounds(ops: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """Rounds for operators, by greedy coloring of their overlap graph."""
    overlap = ops.astype(int) @ ops.T.astype(int) > 0
    np.fill_diagonal(overlap, False)
    coloring = nx.greedy_color(nx.from_numpy_array(overlap), strategy="DSATUR")
    return np.array([coloring[i] for i in range(len(ops))], dtype=int)
