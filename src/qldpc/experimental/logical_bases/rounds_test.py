# SPDX-License-Identifier: Apache-2.0

"""Unit tests for choosing logical operators, one per class of a basis, for parallel rounds."""

import itertools
import types
from typing import Any, Literal
from unittest import mock

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from . import rounds
from .evaluation import evaluate_logical_ops
from .rounds import (
    LogicalOpsRounds,
    OptimizedLogicalRounds,
    _check_choice,
    _count_candidates,
    _list_candidates,
    optimize_logical_rounds,
)

# a [[7, 4]] code whose choice of X-type operators at minimum weight needs 2 to 4 rounds
_CHOICE_CHECKS = ([[1, 0, 1, 1, 1, 0, 0]], [[0, 1, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0]])

# a [[6, 3]] code whose X-type operators need 3 rounds at minimum weight, and 2 with slack 2
_SLACK_CHECKS = ([[1, 1, 1, 1, 0, 1]], [[1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1]])

# an [[8, 4]] code whose X-type candidates of the first information set all share qubits; the
# candidates of a 2-round choice avoid those qubits
_SHARED_CHECKS = (
    [[1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1, 0, 0]],
    [[0, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1]],
)

# an [[8, 4]] code whose choices of X-type operators with the fewest rounds differ in largest load
_LOAD_CHECKS = (
    [[0, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 0, 0]],
    [[0, 0, 0, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0]],
)

# a [[5, 3]] code whose lightest X-type choice at slack 2 does not have the smallest largest load
_ORDER_CHECKS = ([[1, 1, 1, 1, 1]], [[0, 1, 1, 1, 1]])

# a [[10, 6]] code with an integer program that HiGHS (in SciPy 1.18) fails to solve with presolve
_PRESOLVE_CHECKS = (
    [[1, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]],
)


@pytest.mark.parametrize(
    "checks, weight_slack",
    [
        (_CHOICE_CHECKS, 0),
        (_SLACK_CHECKS, 0),
        (_SLACK_CHECKS, 2),
        (_LOAD_CHECKS, 0),
        (_ORDER_CHECKS, 2),
        (_PRESOLVE_CHECKS, 2),
    ],
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


def test_time_limit_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time limit covers solving for both Pauli types together, but not listing candidates."""
    now = [0.0]

    def tick() -> float:
        now[0] += 1
        return now[0]

    def slow_count(*args: Any) -> int:
        now[0] += 50
        return count(*args)

    def recorded_milp(*args: Any, **kwargs: Any) -> Any:
        limits.append(kwargs["options"]["time_limit"])
        return milp(*args, **kwargs)

    # each reading of the clock takes one tick, and each count takes 50
    count, milp = rounds._count_candidates, scipy.optimize.milp
    limits: list[float] = []
    monkeypatch.setattr(rounds, "time", types.SimpleNamespace(monotonic=tick))
    monkeypatch.setattr(rounds, "_count_candidates", slow_count)
    with mock.patch.object(scipy.optimize, "milp", recorded_milp):
        result = optimize_logical_rounds(codes.CSSCode(*_CHOICE_CHECKS), time_limit=100)
    assert result.optimal
    assert len(limits) > 2 and limits[0] == 100
    assert all(0 < limit - next_limit < 50 for limit, next_limit in itertools.pairwise(limits))


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

    # a class with no candidate found still gets one of minimum weight
    code = codes.CSSCode([[1, 1, 0, 1, 1, 0]], [[1, 0, 1, 1, 0, 0]])
    result = optimize_logical_rounds(code)
    _check_result(code, code.get_logical_ops(), result, 0, "dressed")


def test_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Random information sets find candidates that combine many rows of the first one."""
    monkeypatch.setattr(rounds, "_MAX_ROWS_PER_SAMPLE", 1)
    code = codes.CSSCode(*_CHOICE_CHECKS)
    trivial = code.get_matrix(Pauli.X).view(np.ndarray).astype(np.uint8)
    for op in code.get_logical_ops(Pauli.X).view(np.ndarray).astype(np.uint8):
        words = _get_class_words(code, Pauli.X, "dressed")[_get_class(code, Pauli.X, op)]
        min_weight = min(int(word.sum()) for word in words)
        assert _list_candidates(op, trivial, min_weight, 2)[1]


def test_solver_failures() -> None:
    """Solver failures are retried without presolve, and results that break constraints raise."""
    code = codes.CSSCode(*_CHOICE_CHECKS)
    milp = scipy.optimize.milp

    def failing_presolve(*args: Any, **kwargs: Any) -> Any:
        if kwargs["options"].get("presolve", True):
            return scipy.optimize.OptimizeResult(status=4, x=None, message="error")
        return milp(*args, **kwargs)

    with mock.patch.object(scipy.optimize, "milp", failing_presolve):
        result = optimize_logical_rounds(code, time_limit=100)
    _check_result(code, code.get_logical_ops(), result, 0, "dressed")
    assert result.optimal

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


def test_candidates() -> None:
    """Candidates are listed and counted exactly, lightest first."""
    code = codes.CSSCode(*_SLACK_CHECKS)
    trivial = code.get_matrix(Pauli.X).view(np.ndarray).astype(np.uint8)
    for op in code.get_logical_ops(Pauli.X).view(np.ndarray).astype(np.uint8):
        words = _get_class_words(code, Pauli.X, "dressed")[_get_class(code, Pauli.X, op)]
        min_weight = min(int(word.sum()) for word in words)
        expected = {
            word.astype(np.uint8).tobytes() for word in words if word.sum() <= min_weight + 2
        }
        listed, complete = _list_candidates(op, trivial, min_weight, 2)
        assert complete
        assert {word.astype(np.uint8).tobytes() for word in listed} == expected
        assert [int(word.sum()) for word in listed] == sorted(int(word.sum()) for word in listed)
        assert _count_candidates(op, trivial, min_weight, min_weight + 2) == len(expected)


def test_special_inputs() -> None:
    """A code without logical qubits needs no rounds, and arguments are validated."""
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


def _check_result(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike,
    result: rounds.OptimizedLogicalRounds,
    weight_slack: int,
    logicals: Literal["dressed", "bare"],
) -> None:
    """Check a result against brute force over every choice of operators."""
    _check_valid(code, logical_ops, result, weight_slack, logicals)
    k, n = code.dimension, len(code)
    paulis: list[tuple[PauliXZ, Any, slice, slice]] = [
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
    result: rounds.OptimizedLogicalRounds,
    weight_slack: int,
    logicals: Literal["dressed", "bare"],
) -> None:
    """Check that a result is a basis with the given classes, weights, and valid rounds."""
    evaluate_logical_ops(code, result.logical_ops, logicals=logicals)
    k, n = code.dimension, len(code)
    ops = np.asarray(logical_ops, dtype=int)
    paulis: list[tuple[PauliXZ, Any, slice, slice]] = [
        (Pauli.X, result.x, slice(None, k), slice(None, n)),
        (Pauli.Z, result.z, slice(k, None), slice(n, None)),
    ]
    for pauli, rounds_of_type, rows, cols in paulis:
        words = _get_class_words(code, pauli, logicals)
        chosen = result.logical_ops[rows, cols].astype(int)
        for op, new_op in zip(ops[rows, cols], chosen):
            class_words = words[_get_class(code, pauli, op)]
            assert _get_class(code, pauli, new_op) == _get_class(code, pauli, op)
            assert new_op.sum() <= min(int(word.sum()) for word in class_words) + weight_slack
        for r in range(rounds_of_type.num_rounds):
            assert chosen[rounds_of_type.rounds == r].sum(axis=0).max(initial=0) <= 1
        assert rounds_of_type.lower_bound <= _get_value(chosen)[0]
        np.testing.assert_array_equal(
            getattr(result.metrics, str(pauli).lower()).weights, chosen.sum(axis=1)
        )


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
    words = _get_class_words(code, pauli, logicals)
    candidates = []
    for op in ops:
        class_words = words[_get_class(code, pauli, op)]
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


def _get_class(code: codes.CSSCode, pauli: PauliXZ, op: npt.ArrayLike) -> tuple[int, ...]:
    """The class of an operator: its pairing with each of the code's opposite-type operators."""
    dual_ops = np.asarray(code.get_logical_ops(pauli.swap_xz()), dtype=int)
    return tuple(int(bit) for bit in dual_ops @ np.asarray(op, dtype=int) % 2)


def _get_class_words(
    code: codes.CSSCode, pauli: PauliXZ, logicals: Literal["dressed", "bare"]
) -> dict[tuple[int, ...], list[npt.NDArray[np.int_]]]:
    """Every nontrivial logical operator of one Pauli type, by class, by enumeration."""
    other = pauli.swap_xz()
    checks = code.get_stabilizer_ops(other) if logicals == "dressed" else code.get_matrix(other)
    space = np.asarray(galois.GF2(np.asarray(checks)).null_space(), dtype=int)
    words: dict[tuple[int, ...], list[npt.NDArray[np.int_]]] = {}
    for combination in itertools.product([0, 1], repeat=len(space)):
        word = np.array(combination, dtype=int) @ space % 2
        if any(bits := _get_class(code, pauli, word)):
            words.setdefault(bits, []).append(word)
    return words
