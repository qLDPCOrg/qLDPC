# SPDX-License-Identifier: Apache-2.0

"""Choose the operators of a basis of logical operators to act on them in few parallel rounds."""

from __future__ import annotations

import dataclasses
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
from .lightest_basis import _get_min_weight

# Candidates are listed from random information sets, each contributing the words that combine at
# most this many of its rows, until the list matches the exact count or the samples run out.
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
    """Choose the operators of a basis of logical operators to act on them in few parallel rounds.

    Operators that share no qubit can be acted on at the same time, for example by parallel lattice
    surgery.  For each Pauli type, this function keeps the logical class of every operator of
    ``logical_ops``, and chooses an operator in each class, of weight at most the minimum weight of
    its class plus ``weight_slack``.  It minimizes the number of rounds needed to act on all of
    them, if operators in one round must share no qubit.  Among the choices with the fewest rounds,
    it minimizes the largest number of operators on one qubit, and then the total weight.  Keeping
    the classes keeps the X-type and Z-type operators paired, so the result is a complete basis.

    The candidates for each class are listed by sampling information sets, and the exact counts of
    :func:`qldpc.codes.distance.count_logical_ops_css` prove each list complete.  An integer program
    (:func:`scipy.optimize.milp`) then tries 1, 2, ... rounds, starting from a lower bound, until a
    choice exists, so the result is optimal unless a list is incomplete or the time limit runs out.
    Counting, listing, and solving each generally take time exponential in the size of the code.

    In some codes, such as bivariate bicycle codes, all operators of a class have weights of the
    same parity, so a slack of 1 allows no new operators, and 2 is the next useful value.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        logical_ops: A complete basis with shape ``(2k, 2n)``, laid out as by
            :meth:`qldpc.codes.CSSCode.get_logical_ops`, whose classes are kept.  Defaults to the
            code's own logical operators; the basis of
            :func:`~qldpc.experimental.logical_bases.optimize_logical_basis` is a good choice.
        weight_slack: How much heavier than the minimum weight of its class an operator may be.
        logicals: For a subsystem code, ``"dressed"`` (the default) uses logical operators that
            commute with the stabilizers, while ``"bare"`` uses those that commute with every gauge
            operator.  The two are the same for a stabilizer code.  Only a basis of bare operators
            can be passed to :meth:`qldpc.codes.CSSCode.set_logical_ops`.
        time_limit: If given, a limit in seconds on the time spent solving integer programs, for
            both Pauli types together.  Counting and listing candidates are not limited.

    Returns:
        The chosen basis, with the round of each operator and a lower bound on the rounds.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, if ``logicals`` is invalid, if ``logical_ops`` is not
            a complete basis as :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`
            describes, or if ``weight_slack`` or ``time_limit`` is negative.
        RuntimeError: If the integer-program solver returns a choice that breaks its constraints.
    """
    if weight_slack < 0 or (time_limit is not None and time_limit < 0):
        raise ValueError("weight_slack and time_limit must be nonnegative")
    ops = code.get_logical_ops() if logical_ops is None else logical_ops
    evaluate_logical_ops(code, ops, logicals=logicals)
    k, n = code.dimension, len(code)
    matrix = _as_binary_matrix(ops, (2 * k, 2 * n))
    deadline = None if time_limit is None else time.monotonic() + time_limit

    chosen = np.zeros_like(matrix)
    rounds = {}
    paulis: list[tuple[PauliXZ, slice, slice]] = [
        (Pauli.X, slice(None, k), slice(None, n)),
        (Pauli.Z, slice(k, None), slice(n, None)),
    ]
    for pauli, rows, cols in paulis:
        sector = get_sector(code, pauli, logicals)
        chosen[rows, cols], rounds[pauli] = _choose_operators(
            sector, matrix[rows, cols], weight_slack, deadline
        )
    return OptimizedLogicalRounds(chosen, rounds[Pauli.X], rounds[Pauli.Z])


@dataclasses.dataclass(frozen=True, eq=False)
class OptimizedLogicalRounds:
    """A complete basis of logical operators chosen to act on them in few parallel rounds.

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
        lower_bound: A lower bound on the number of rounds of every choice of operators of these
            classes, within the weight limit.
        stop_reason: Why the choice stopped: ``"optimal"`` if it is proven optimal (the fewest
            rounds, then the smallest maximum load, then the smallest total weight),
            ``"time_limit"`` if the time limit ran out first, or ``"incomplete_candidates"`` if the
            sampling of candidates did not find them all, so that a better choice may exist.
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
    sector: Sector, ops: npt.NDArray[np.uint8], weight_slack: int, deadline: float | None
) -> tuple[npt.NDArray[np.uint8], LogicalOpsRounds]:
    """Choose an operator in the class of each row of ``ops``, to act on them in few rounds."""
    k, n = ops.shape
    if k == 0:
        return ops, LogicalOpsRounds(np.zeros(0, dtype=int), 0, "optimal")

    # list the candidates of each class, and bound the rounds from below
    candidates = []
    min_weights = []
    complete = True
    for op in ops:
        space = np.vstack([sector.trivial, op])
        min_weights.append(_get_min_weight(space, sector.trivial))
        words, found_all = _list_candidates(op, sector.trivial, min_weights[-1], weight_slack)
        candidates.append(words)
        complete &= found_all
    lower_bound = max(1, math.ceil(sum(min_weights) / n))
    if complete:
        lower_bound = max(lower_bound, _get_common_qubit_bound(candidates))

    # try more and more rounds, until a choice exists or the time limit runs out
    status = "time_limit"
    for num_rounds in range(lower_bound, k + 1):
        timeout = None if deadline is None else max(deadline - time.monotonic(), 0)
        status, choice = _solve_rounds(candidates, num_rounds, timeout)
        if status == "infeasible" and complete:
            lower_bound = num_rounds + 1
        if status != "infeasible":
            break

    # without a choice, the lightest candidates in greedily coloured rounds are one
    if choice is None:
        chosen = np.array([words[0] for words in candidates])
        choice = chosen, _get_greedy_rounds(chosen)
    stop_reason: Literal["optimal", "time_limit", "incomplete_candidates"] = (
        "incomplete_candidates"
        if not complete
        else "optimal"
        if status == "optimal"
        else "time_limit"
    )
    return choice[0], LogicalOpsRounds(choice[1], lower_bound, stop_reason)


def _list_candidates(
    op: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8], min_weight: int, weight_slack: int
) -> tuple[list[npt.NDArray[np.uint8]], bool]:
    """The operators of the class of ``op`` within ``weight_slack`` of its minimum weight.

    Returns the operators found, lightest first, and whether they are all of them.
    """
    max_weight = min_weight + weight_slack
    expected = _count_candidates(op, trivial, min_weight, max_weight)
    basis, labels = get_nested_code_basis(op[None, :], trivial)
    rank, n = basis.shape
    combinations = [
        np.array(list(itertools.combinations(range(rank), size)), dtype=np.int64).reshape(-1, size)
        for size in range(1, _MAX_ROWS_PER_SAMPLE + 1)
    ]

    # combine few rows of the basis reduced on random information sets
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

    words = unpack_bits(
        np.array(list(found.values()), dtype=np.uint64).reshape(-1, packed.shape[1]), n
    )
    order = np.lexsort(words.T[::-1])
    words = words[order]
    words = words[np.argsort(words.sum(axis=1), kind="stable")]
    return list(words), len(found) == expected


def _count_candidates(
    op: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8], min_weight: int, max_weight: int
) -> int:
    """The number of operators of the class of ``op`` with weights in ``[min_weight, max_weight]``.

    They are the nontrivial X-type logical operators of a stabilizer code whose X-type checks are
    ``trivial``, and whose Z-type checks have the span of ``trivial`` and ``op`` as their kernel.
    """
    checks_z = get_kernel(np.vstack([trivial, op]))
    return sum(
        count_logical_ops_css(trivial, checks_z, weight).count_x
        for weight in range(min_weight, min(max_weight, len(op)) + 1)
    )


def _get_common_qubit_bound(candidates: Sequence[Sequence[npt.NDArray[np.uint8]]]) -> int:
    """A lower bound on rounds: the most classes whose candidates all contain one qubit."""
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
    operators on one qubit, and then their total weight.  Returns whether the choice is proven best
    (``"optimal"``), found but not proven best within the time limit (``"feasible"``), proven not
    to exist (``"infeasible"``), or not found within the time limit (``"time_limit"``), and the
    chosen operators and their rounds, if any.

    The variables are 1 if class ``c`` takes candidate ``j`` in round ``r``.  Class 0 is put in
    round 0, since rounds can be relabelled.  One more variable bounds the load of every qubit.
    """
    k, n = len(candidates), len(candidates[0][0])
    keys = [
        (c, j, r)
        for c, words in enumerate(candidates)
        for j in range(len(words))
        for r in range(num_rounds if c else 1)
    ]
    supports = np.array([candidates[c][j] for c, j, _ in keys], dtype=np.int64).reshape(-1, n)
    num_vars = len(keys)

    # each class takes one candidate in one round, no qubit is used twice in one round, and the
    # last variable is at least the number of chosen operators on each qubit
    classes = np.array([c for c, _, _ in keys])
    rounds = np.array([r for _, _, r in keys])
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

    # minimize the largest load first, then the total weight
    weights = supports.sum(axis=1)
    cost = np.append(weights, k * n + 1).astype(float)
    result = scipy.optimize.milp(
        cost,
        constraints=constraints,
        integrality=np.ones(num_vars + 1),
        bounds=scipy.optimize.Bounds(0, np.append(np.ones(num_vars), k)),
        options={} if time_limit is None else {"time_limit": time_limit},
    )
    if result.status == 2:
        return "infeasible", None
    if result.x is None:
        if result.status == 1:
            return "time_limit", None
        raise RuntimeError(f"The integer program failed: {result.message}")

    # check the choice, rather than trusting the solver
    picked = [keys[i] for i in np.flatnonzero(result.x[:num_vars] > 0.5)]
    chosen = np.array([candidates[c][j] for c, j, _ in sorted(picked)], dtype=np.uint8)
    chosen_rounds = np.array([r for _, _, r in sorted(picked)])
    if sorted(c for c, _, _ in picked) != list(range(k)) or any(
        chosen[chosen_rounds == r].astype(int).sum(axis=0).max(initial=0) > 1
        for r in range(num_rounds)
    ):
        raise RuntimeError("The integer program returned a choice that breaks its constraints")
    used = np.unique(chosen_rounds)
    status: Literal["optimal", "feasible"] = "optimal" if result.status == 0 else "feasible"
    return status, (chosen, np.searchsorted(used, chosen_rounds))


def _get_greedy_rounds(ops: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """Rounds for operators, by greedy colouring of their overlap graph."""
    overlap = ops.astype(int) @ ops.T.astype(int) > 0
    np.fill_diagonal(overlap, False)
    colouring = nx.coloring.greedy_color(nx.from_numpy_array(overlap), strategy="DSATUR")
    return np.array([colouring[i] for i in range(len(ops))], dtype=int)
