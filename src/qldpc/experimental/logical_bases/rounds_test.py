# SPDX-License-Identifier: Apache-2.0

"""Unit tests for choosing logical operators, one per class of a basis, for parallel rounds."""

import itertools
import types
from typing import Any, Literal
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from . import rounds
from .conftest import get_class, get_class_weights, get_class_words
from .evaluation import LogicalOpsMetrics, evaluate_logical_ops
from .rounds import (
    LogicalOpsRounds,
    OptimizedLogicalRounds,
    _check_choice,
    _list_candidates,
    optimize_logical_rounds,
)

# a [[7, 4]] code whose choices of X-type operators at minimum weight need from 2 to 4 rounds
_CHOICE_CHECKS = ([[1, 0, 1, 1, 1, 0, 0]], [[0, 1, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0]])

# an [[8, 4]] code in which a 2-round choice exists; but if only single rows of the basis reduced on
# the column-order information set are listed, the X-type candidates of three classes all contain
# qubit 1, which would wrongly bound the rounds below by 3
_SHARED_CHECKS = (
    [[1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1, 0, 0]],
    [[0, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1]],
)

# a [[7, 4]] code whose X-type choices at slack 3 with the fewest rounds differ in their largest
# load; the choice with the smallest largest load is 3 heavier than the lightest choice
_ORDER_CHECKS = ([[1, 1, 1, 1, 1, 1, 1]], [[1, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]])

# a [[10, 6]] code with an integer program that HiGHS (in SciPy 1.18) fails to solve with presolve
_PRESOLVE_CHECKS = (
    [[1, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]],
)


@pytest.mark.parametrize(
    "checks, weight_slack",
    [(_ORDER_CHECKS, 3), (_PRESOLVE_CHECKS, 2)],
)
def test_optimum(checks: tuple[Any, Any], weight_slack: int) -> None:
    """The chosen operators are optimal, by brute force over every choice of operators."""
    code = codes.CSSCode(*checks)
    result = optimize_logical_rounds(code, weight_slack=weight_slack)
    _check_result(code, code.get_logical_ops(), result, weight_slack, "dressed")
    assert result.optimal


@pytest.mark.parametrize("logicals", ["dressed", "bare"])
def test_subsystem_code(
    subsystem_code: codes.CSSCode, logicals: Literal["dressed", "bare"]
) -> None:
    """Operators of a subsystem code are chosen optimally, in the classes of the given basis."""
    k = subsystem_code.dimension
    # a paired basis other than the code's own
    ops = np.asarray(subsystem_code.get_logical_ops(), dtype=int)
    ops[0] = (ops[0] + ops[1]) % 2
    ops[k + 1] = (ops[k + 1] + ops[k]) % 2
    result = optimize_logical_rounds(subsystem_code, ops, weight_slack=2, logicals=logicals)
    _check_result(subsystem_code, ops, result, 2, logicals)
    assert result.optimal


def test_time_limit() -> None:
    """Without time to solve, a valid choice is returned, and not claimed to be optimal."""
    code = codes.CSSCode(*_CHOICE_CHECKS)
    result = optimize_logical_rounds(code, time_limit=0)
    _check_valid(code, code.get_logical_ops(), result, 0, "dressed")
    assert (result.x.stop_reason, result.z.stop_reason) == ("time_limit", "time_limit")
    assert not result.optimal

    # a choice found but not proven best within the time limit
    milp = scipy.optimize.milp

    def unproven_milp(*args: Any, **kwargs: Any) -> Any:
        solution = milp(*args, **kwargs)
        if solution.status == 0:
            solution.status = 1
        return solution

    with mock.patch.object(scipy.optimize, "milp", unproven_milp):
        result = optimize_logical_rounds(code)
    _check_valid(code, code.get_logical_ops(), result, 0, "dressed")
    assert result.x.stop_reason == "time_limit"


def test_time_limit_fallback() -> None:
    """Without any choice in time, the lightest candidates are used, even with slack."""
    code = codes.CSSCode(*_CHOICE_CHECKS)
    out_of_time = scipy.optimize.OptimizeResult(status=1, x=None, message="time limit")
    with mock.patch.object(scipy.optimize, "milp", return_value=out_of_time):
        result = optimize_logical_rounds(code, weight_slack=2)
    _check_valid(code, code.get_logical_ops(), result, 2, "dressed")
    assert (result.x.stop_reason, result.z.stop_reason) == ("time_limit", "time_limit")
    k, n = code.dimension, len(code)
    paulis: list[tuple[PauliXZ, slice, slice]] = [
        (Pauli.X, slice(None, k), slice(None, n)),
        (Pauli.Z, slice(k, None), slice(n, None)),
    ]
    for pauli, rows, cols in paulis:
        weights = get_class_weights(code, pauli, "dressed")
        for op in result.logical_ops[rows, cols].astype(int):
            assert op.sum() == weights[get_class(code, pauli, op)]


def test_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampling with random information sets gives the same result every time."""
    monkeypatch.setattr(rounds, "_MAX_ROWS_PER_SAMPLE", 1)
    monkeypatch.setattr(rounds, "_MAX_SAMPLES", 3)
    code = codes.CSSCode(*_ORDER_CHECKS)
    results = [optimize_logical_rounds(code, weight_slack=2) for _ in range(4)]
    for result in results[1:]:
        np.testing.assert_array_equal(result.logical_ops, results[0].logical_ops)
        np.testing.assert_array_equal(result.x.rounds, results[0].x.rounds)
        np.testing.assert_array_equal(result.z.rounds, results[0].z.rounds)

    # the Z-type lists, which these samples leave incomplete, are the same every time
    trivial = code.get_matrix(Pauli.Z).view(np.ndarray).astype(np.uint8)
    weights = get_class_weights(code, Pauli.Z, "dressed")
    for op in code.get_logical_ops(Pauli.Z).view(np.ndarray).astype(np.uint8):
        min_weight = weights[get_class(code, Pauli.Z, op)]
        lists = [_list_candidates(op, trivial, min_weight, 2)[0] for _ in range(4)]
        for listed in lists[1:]:
            np.testing.assert_array_equal(listed, lists[0])


def test_time_limit_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time limit covers all solving, including retries, but not listing candidates.

    The X-type operators are solved first.
    """
    # each reading of the clock takes one tick, and each count takes 50
    count, milp = rounds._count_candidates, scipy.optimize.milp
    now = [0.0]
    limits: list[float] = []

    def tick() -> float:
        now[0] += 1
        return now[0]

    def slow_count(*args: Any) -> int:
        now[0] += 50
        return count(*args)

    def recorded_milp(*args: Any, **kwargs: Any) -> Any:
        limits.append(kwargs["options"]["time_limit"])
        return milp(*args, **kwargs)

    def failing_presolve(*args: Any, **kwargs: Any) -> Any:
        if kwargs["options"].get("presolve", True):
            now[0] += 10
            limits.append(kwargs["options"]["time_limit"])
            return scipy.optimize.OptimizeResult(status=4, x=None, message="error")
        return recorded_milp(*args, **kwargs)

    monkeypatch.setattr(rounds, "time", types.SimpleNamespace(monotonic=tick))
    monkeypatch.setattr(rounds, "_count_candidates", slow_count)
    with mock.patch.object(scipy.optimize, "milp", recorded_milp):
        result = optimize_logical_rounds(codes.CSSCode(*_CHOICE_CHECKS), time_limit=100)
    assert result.optimal
    assert len(limits) > 2 and limits[0] == 100
    assert all(0 < limit - next_limit < 50 for limit, next_limit in itertools.pairwise(limits))

    # a retry without presolve gets only the time left after the failed solve, which takes 10
    limits.clear()
    with mock.patch.object(scipy.optimize, "milp", failing_presolve):
        optimize_logical_rounds(codes.CSSCode(*_CHOICE_CHECKS), time_limit=100)
    assert limits[0] == 100 and 80 < limits[1] <= 90

    # the X-type operators are chosen first, and here use up the time limit
    def slow_milp(*args: Any, **kwargs: Any) -> Any:
        now[0] += 100
        return milp(*args, **kwargs)

    with mock.patch.object(scipy.optimize, "milp", slow_milp):
        result = optimize_logical_rounds(codes.CSSCode(*_CHOICE_CHECKS), time_limit=100)
    assert (result.x.stop_reason, result.z.stop_reason) == ("optimal", "time_limit")


def test_incomplete_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sampling misses candidates, the result is valid but not claimed to be optimal."""
    monkeypatch.setattr(rounds, "_MAX_ROWS_PER_SAMPLE", 1)
    monkeypatch.setattr(rounds, "_MAX_SAMPLES", 1)
    code = codes.CSSCode(*_SHARED_CHECKS)
    result = optimize_logical_rounds(code)
    _check_valid(code, code.get_logical_ops(), result, 0, "dressed")
    assert result.x.stop_reason == "incomplete_candidates"
    assert not result.optimal

    # the lower bound holds for every choice, including those whose candidates were missed
    best = _get_best(code, code.get_logical_ops(), Pauli.X, 0, "dressed")
    assert result.x.lower_bound <= best[0]

    # missing candidates are reported even if the time limit also runs out
    out_of_time = scipy.optimize.OptimizeResult(status=1, x=None, message="time limit")
    with mock.patch.object(scipy.optimize, "milp", return_value=out_of_time):
        result = optimize_logical_rounds(code)
    assert result.x.stop_reason == "incomplete_candidates"

    # a class with no candidate found still gets one of minimum weight, even if the operator of the
    # basis in that class (the first Z-type one) is heavier
    code = codes.CSSCode([[1, 1, 0, 1, 1, 0]], [[1, 0, 1, 1, 0, 0]])
    k, n = code.dimension, len(code)
    ops = np.asarray(code.get_logical_ops(), dtype=int)
    ops[k, n:] = (ops[k, n:] + code.get_matrix(Pauli.Z)[0].view(np.ndarray)) % 2
    assert (
        ops[k, n:].sum()
        > get_class_weights(code, Pauli.Z, "dressed")[get_class(code, Pauli.Z, ops[k, n:])]
    )
    result = optimize_logical_rounds(code, ops)
    _check_result(code, ops, result, 0, "dressed")


def test_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random information sets find candidates that need many rows of the column-order basis."""
    monkeypatch.setattr(rounds, "_MAX_ROWS_PER_SAMPLE", 1)
    code = codes.CSSCode(*_CHOICE_CHECKS)
    trivial = code.get_matrix(Pauli.X).view(np.ndarray).astype(np.uint8)
    weights = get_class_weights(code, Pauli.X, "dressed")
    for op in code.get_logical_ops(Pauli.X).view(np.ndarray).astype(np.uint8):
        assert _list_candidates(op, trivial, weights[get_class(code, Pauli.X, op)], 2)[1]


def test_solver_failures() -> None:
    """Solver failures, and results that cannot be right, raise errors."""
    code = codes.CSSCode(*_CHOICE_CHECKS)
    milp = scipy.optimize.milp

    def broken_milp(*args: Any, **kwargs: Any) -> Any:
        solution = milp(*args, **kwargs)
        if solution.x is not None:
            solution.x[:] = 0
        return solution

    with (
        mock.patch.object(scipy.optimize, "milp", broken_milp),
        pytest.raises(RuntimeError, match="breaks its constraints"),
    ):
        optimize_logical_rounds(code)

    infeasible = scipy.optimize.OptimizeResult(status=2, x=None, message="infeasible")
    with (
        mock.patch.object(scipy.optimize, "milp", return_value=infeasible),
        pytest.raises(RuntimeError, match="no choice has 4 rounds"),
    ):
        optimize_logical_rounds(code)

    failed = scipy.optimize.OptimizeResult(status=4, x=None, message="error")
    with (
        mock.patch.object(scipy.optimize, "milp", return_value=failed),
        pytest.raises(RuntimeError, match="failed"),
    ):
        optimize_logical_rounds(code)


def test_check_choice() -> None:
    """A choice must take one candidate per class, and keep the operators of a round disjoint."""
    candidates = [[np.array([1, 1, 0]), np.array([0, 0, 1])], [np.array([0, 1, 1])]]
    chosen, rounds_of_ops = _check_choice(candidates, [(1, 0, 1), (0, 0, 0)])
    np.testing.assert_array_equal(chosen, [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(rounds_of_ops, [0, 1])
    with pytest.raises(RuntimeError, match="breaks its constraints"):
        _check_choice(candidates, [(0, 0, 0), (1, 0, 0)])
    with pytest.raises(RuntimeError, match="breaks its constraints"):
        _check_choice(candidates, [(0, 0, 0), (0, 1, 1)])
    with pytest.raises(RuntimeError, match="breaks its constraints"):
        _check_choice(candidates, [(0, 0, 0), (0, 1, 1), (1, 0, 2)])


def test_special_inputs(subsystem_code: codes.CSSCode) -> None:
    """Edge cases: no logical qubits, mixed optimality, a slack beyond n, and invalid inputs."""
    code = codes.CSSCode([[1, 1]], [[1, 1]])
    result = optimize_logical_rounds(code)
    assert result.logical_ops.shape == (0, 4)
    assert (result.x.num_rounds, result.x.lower_bound) == (0, 0)
    assert result.optimal
    assert result.metrics.x.weights.shape == (0,)

    # the result is optimal only if the choices of both Pauli types are
    x_rounds = LogicalOpsRounds(np.zeros(1, dtype=int), 1, "optimal")
    z_rounds = LogicalOpsRounds(np.zeros(1, dtype=int), 1, "time_limit")
    assert not OptimizedLogicalRounds(np.zeros((2, 2), dtype=np.uint8), x_rounds, z_rounds).optimal

    # a slack beyond the number of qubits allows every operator of each class
    code = codes.SteaneCode()
    result = optimize_logical_rounds(code, weight_slack=7)
    _check_result(code, code.get_logical_ops(), result, 7, "dressed")

    with pytest.raises(ValueError, match="nonnegative"):
        optimize_logical_rounds(code, weight_slack=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        optimize_logical_rounds(code, time_limit=-1)
    with pytest.raises(ValueError, match="shape"):
        optimize_logical_rounds(codes.SteaneCode(), np.zeros((1, 14), dtype=int))

    # dressed operators of a subsystem code are not a basis of bare operators
    dressed = optimize_logical_rounds(subsystem_code).logical_ops
    with pytest.raises(ValueError, match="not logical"):
        optimize_logical_rounds(subsystem_code, dressed, logicals="bare")


def _check_result(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike,
    result: OptimizedLogicalRounds,
    weight_slack: int,
    logicals: Literal["dressed", "bare"],
) -> None:
    """Check a result against brute force over every choice of operators."""
    _check_valid(code, logical_ops, result, weight_slack, logicals)
    k, n = code.dimension, len(code)
    paulis: list[tuple[PauliXZ, LogicalOpsRounds, slice, slice]] = [
        (Pauli.X, result.x, slice(None, k), slice(None, n)),
        (Pauli.Z, result.z, slice(k, None), slice(n, None)),
    ]
    for pauli, rounds_of_type, rows, cols in paulis:
        best = _get_best(code, logical_ops, pauli, weight_slack, logicals)
        chosen = result.logical_ops[rows, cols].astype(int)
        assert (rounds_of_type.num_rounds, *_get_value(chosen)[1:]) == best
        assert rounds_of_type.lower_bound == rounds_of_type.num_rounds


def _check_valid(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike,
    result: OptimizedLogicalRounds,
    weight_slack: int,
    logicals: Literal["dressed", "bare"],
) -> None:
    """Check that a result is a basis with the given classes, weights, and valid rounds."""
    evaluate_logical_ops(code, result.logical_ops, logicals=logicals)
    k, n = code.dimension, len(code)
    ops = np.asarray(logical_ops, dtype=int)
    paulis: list[tuple[PauliXZ, LogicalOpsRounds, LogicalOpsMetrics, slice, slice]] = [
        (Pauli.X, result.x, result.metrics.x, slice(None, k), slice(None, n)),
        (Pauli.Z, result.z, result.metrics.z, slice(k, None), slice(n, None)),
    ]
    for pauli, rounds_of_type, metrics, rows, cols in paulis:
        weights = get_class_weights(code, pauli, logicals)
        chosen = result.logical_ops[rows, cols].astype(int)
        for op, new_op in zip(ops[rows, cols], chosen):
            assert get_class(code, pauli, new_op) == get_class(code, pauli, op)
            assert new_op.sum() <= weights[get_class(code, pauli, op)] + weight_slack
        for r in range(rounds_of_type.num_rounds):
            assert chosen[rounds_of_type.rounds == r].sum(axis=0).max(initial=0) <= 1
        assert rounds_of_type.lower_bound <= _get_value(chosen)[0]
        np.testing.assert_array_equal(metrics.weights, chosen.sum(axis=1))


def _get_best(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike,
    pauli: PauliXZ,
    weight_slack: int,
    logicals: Literal["dressed", "bare"],
) -> tuple[int, int, int]:
    """The best rounds, then largest load, then total weight, over every choice of operators."""
    k, n = code.dimension, len(code)
    ops = np.asarray(logical_ops, dtype=int)
    ops = ops[:k, :n] if pauli is Pauli.X else ops[k:, n:]
    words = get_class_words(code, pauli, logicals)
    candidates = []
    for op in ops:
        class_words = words[get_class(code, pauli, op)]
        min_weight = min(int(word.sum()) for word in class_words)
        candidates.append([w for w in class_words if w.sum() <= min_weight + weight_slack])
    return min(_get_value(np.array(choice)) for choice in itertools.product(*candidates))


def _get_value(ops: npt.NDArray[np.int_]) -> tuple[int, int, int]:
    """The rounds (by brute-force coloring), the largest load, and the total weight."""
    k = len(ops)
    overlap = ops @ ops.T > 0
    num_rounds = next(
        r
        for r in range(1, k + 1)
        if any(
            all(
                not overlap[i, j] or colors[i] != colors[j]
                for i, j in itertools.combinations(range(k), 2)
            )
            for colors in itertools.product(range(r), repeat=k)
        )
    )
    return num_rounds, int(ops.sum(axis=0).max()), int(ops.sum())
