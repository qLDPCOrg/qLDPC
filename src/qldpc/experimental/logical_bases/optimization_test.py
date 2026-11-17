# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the search for complete bases of logical operators with low weights."""

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal
from unittest import mock

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from . import optimization
from .evaluation import evaluate_logical_ops
from .lightest_basis import LightestLogicalBasis, get_lightest_logical_basis
from .optimization import _ClassSearch, optimize_logical_basis

# a [[7, 3]] code whose optimal complete basis is heavier than its lightest bases
_PAIRED_CHECKS = (
    [[0, 1, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1]],
    [[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1, 1]],
)


@pytest.fixture
def paired_code() -> codes.CSSCode:
    """A [[7, 3]] code whose optimal complete basis is heavier than its lightest bases."""
    return codes.CSSCode(*_PAIRED_CHECKS)


def _max_weight(weights_x: Sequence[int], weights_z: Sequence[int]) -> tuple[int, ...]:
    """All weights in decreasing order."""
    return tuple(sorted([*weights_x, *weights_z], reverse=True))


def _total_weight(weights_x: Sequence[int], weights_z: Sequence[int]) -> tuple[int, ...]:
    """The total weight, then all weights in decreasing order."""
    return (sum(weights_x) + sum(weights_z), *_max_weight(weights_x, weights_z))


def _z_weight(weights_x: Sequence[int], weights_z: Sequence[int]) -> int:
    """The total weight of the Z-type operators only."""
    return sum(weights_z)


@pytest.mark.parametrize(
    "objective, function",
    [("max_weight", _max_weight), ("total_weight", _total_weight), (_z_weight, _z_weight)],
)
def test_optimum(
    paired_code: codes.CSSCode,
    objective: Literal["max_weight", "total_weight"] | Callable[..., Any],
    function: Callable[..., Any],
) -> None:
    """The search finds an optimal basis, which pairing makes heavier than the lightest bases."""
    result = optimize_logical_basis(paired_code, objective=objective, seed=0)
    metrics = evaluate_logical_ops(paired_code, result.logical_ops)
    value = function(metrics.x.weights.tolist(), metrics.z.weights.tolist())
    assert value == _get_brute_force_optimum(paired_code, "dressed", function)
    np.testing.assert_array_equal(result.metrics.x.weights, metrics.x.weights)
    np.testing.assert_array_equal(result.metrics.z.weights, metrics.z.weights)

    if callable(objective):
        assert result.lower_bound is None
    else:
        assert result.lower_bound == _get_lower_bound(paired_code, "dressed", function)
        assert value > result.lower_bound
    assert not result.optimal
    assert result.stop_reason == "patience"


def test_certified_optimum() -> None:
    """The search stops at the lower bound, which proves its result optimal, whatever its budget."""
    code = codes.ToricCode(2)
    with (
        mock.patch.object(
            _ClassSearch, "_improves", autospec=True, side_effect=_ClassSearch._improves
        ) as improves,
        mock.patch.object(
            _ClassSearch, "_perturb", autospec=True, side_effect=_ClassSearch._perturb
        ) as perturb,
    ):
        result = optimize_logical_basis(code)
    assert not improves.called  # the start meets the bound, so the descent tries no move
    assert not perturb.called
    assert result.optimal
    assert result.stop_reason == "lower_bound"
    assert result.lower_bound == (2, 2, 2, 2)
    metrics = evaluate_logical_ops(code, result.logical_ops)
    assert (metrics.x.max_weight, metrics.z.max_weight) == (2, 2)
    assert optimize_logical_basis(code, max_searches=0).optimal


@pytest.mark.parametrize("logicals", ["dressed", "bare"])
def test_subsystem_code(
    subsystem_code: codes.CSSCode, logicals: Literal["dressed", "bare"]
) -> None:
    """Dressed and bare complete bases of a subsystem code are optimal, and bare ones can be set."""
    result = optimize_logical_basis(subsystem_code, logicals=logicals, seed=0)
    metrics = evaluate_logical_ops(subsystem_code, result.logical_ops, logicals=logicals)
    value = _max_weight(metrics.x.weights.tolist(), metrics.z.weights.tolist())
    assert value == _get_brute_force_optimum(subsystem_code, logicals, _max_weight)
    assert result.lower_bound == _get_lower_bound(subsystem_code, logicals, _max_weight)
    if logicals == "bare":
        subsystem_code.set_logical_ops(result.logical_ops.astype(int))


@pytest.mark.parametrize("function", [_max_weight, _total_weight])
def test_local_optima(
    paired_code: codes.CSSCode, function: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descent takes only improving moves, and ends at a basis that no move improves."""
    result = optimize_logical_basis(paired_code, objective=function, patience=0)
    k, n = paired_code.dimension, len(paired_code)
    classes_x = [_get_class(paired_code, Pauli.X, op) for op in result.logical_ops[:k, :n]]
    classes_z = [_get_class(paired_code, Pauli.Z, op) for op in result.logical_ops[k:, n:]]
    assert not _get_improving_moves(paired_code, function, classes_x, classes_z)

    # this descent from the code's own basis needs more than one move, and a descent that judged
    # moves with outdated weights would take a move here that does not lower the objective
    code = codes.CSSCode([[1, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1]], [[0, 1, 1, 0, 1, 0]])
    search = _ClassSearch(code, function, "dressed", None)
    moves = []
    improves = _ClassSearch._improves

    def record_move(
        self: _ClassSearch, weights_x: Any, weights_z: Any, value: Any, i: int, j: int, *rest: int
    ) -> bool:
        moves.append((i, j, improves(self, weights_x, weights_z, value, i, j, *rest)))
        return moves[-1][2]

    monkeypatch.setattr(_ClassSearch, "_improves", record_move)
    descent_x, descent_z = search._descend([1, 2, 4], [1, 2, 4])
    bits_x = [_to_bits(c, 3) for c in descent_x]
    bits_z = [_to_bits(c, 3) for c in descent_z]
    assert not _get_improving_moves(code, function, bits_x, bits_z)

    # replay the moves taken, each of which must lower the objective
    weights_x = _get_class_weights(code, Pauli.X, "dressed")
    weights_z = _get_class_weights(code, Pauli.Z, "dressed")
    replay_x, replay_z = [1, 2, 4], [1, 2, 4]
    taken = [(i, j) for i, j, improved in moves if improved]
    assert len(taken) > 1
    for i, j in taken:
        value = function(
            [weights_x[_to_bits(c, 3)] for c in replay_x],
            [weights_z[_to_bits(c, 3)] for c in replay_z],
        )
        replay_x[i], replay_z[j] = replay_x[i] ^ replay_x[j], replay_z[j] ^ replay_z[i]
        assert value > function(
            [weights_x[_to_bits(c, 3)] for c in replay_x],
            [weights_z[_to_bits(c, 3)] for c in replay_z],
        )
    assert (replay_x, replay_z) == (descent_x, descent_z)


def test_perturbations_escape_local_optima() -> None:
    """Perturbations lead from a basis that no move improves to an optimal one."""
    code = codes.CSSCode(
        [[1, 1, 0, 1, 1, 1, 1, 1]], [[0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 1, 0, 1]]
    )
    assert not optimize_logical_basis(code, patience=0, seed=0).optimal
    assert optimize_logical_basis(code, seed=0).optimal


def test_patience(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search stops after ``patience`` rounds in a row without improvement."""
    code = codes.CSSCode(
        [[0, 1, 0, 0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1, 1, 1, 0]],
        [[0, 1, 0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 0, 1]],
    )
    descents = []
    descend = _ClassSearch._descend

    def record_descent(self: _ClassSearch, *args: Any) -> tuple[list[int], list[int]]:
        descents.append(descend(self, *args))
        return descents[-1]

    monkeypatch.setattr(_ClassSearch, "_descend", record_descent)
    patience = 2
    optimize_logical_basis(code, objective=_max_weight, patience=patience, seed=1)

    # score each descent by brute force, and count the rounds since the last improvement
    weights_x = _get_class_weights(code, Pauli.X, "dressed")
    weights_z = _get_class_weights(code, Pauli.Z, "dressed")
    k = code.dimension
    values = [
        _max_weight(
            [weights_x[_to_bits(c, k)] for c in classes_x],
            [weights_z[_to_bits(c, k)] for c in classes_z],
        )
        for classes_x, classes_z in descents
    ]
    stale_counts = []
    best, stale = values[0], 0
    for value in values[1:]:
        stale = 0 if value < best else stale + 1
        stale_counts.append(stale)
        best = min(best, value)
    assert stale_counts[-1] == patience
    assert max(stale_counts[:-1]) < patience
    assert any(count == 0 for count in stale_counts[1:])  # an improvement after a stale round


def test_budget(paired_code: codes.CSSCode, monkeypatch: pytest.MonkeyPatch) -> None:
    """A budget of searches stops the search early, with the best basis that it found."""
    # the first descent alone runs at least two fewer searches than the whole search, so a budget of
    # one more than the first descent stops the search early
    first_descent = optimize_logical_basis(paired_code, patience=0, seed=0)
    unbounded = optimize_logical_basis(paired_code, seed=0)
    assert first_descent.searches + 1 < unbounded.searches

    budget = first_descent.searches + 1
    searches_before_rounds = []
    perturb = _ClassSearch._perturb

    def record_round(self: _ClassSearch, *args: Any) -> tuple[list[int], list[int]]:
        searches_before_rounds.append(self.searches)
        return perturb(self, *args)

    monkeypatch.setattr(_ClassSearch, "_perturb", record_round)
    result = optimize_logical_basis(paired_code, max_searches=budget, seed=0)
    assert result.stop_reason == "max_searches"
    assert searches_before_rounds and max(searches_before_rounds) < budget
    assert result.searches < unbounded.searches
    values = []
    for basis in [result, first_descent]:
        metrics = evaluate_logical_ops(paired_code, basis.logical_ops)
        values.append(_max_weight(metrics.x.weights.tolist(), metrics.z.weights.tolist()))
    assert values[0] <= values[1]

    # the start is found whatever the budget
    result = optimize_logical_basis(paired_code, max_searches=0, seed=0)
    assert result.stop_reason == "max_searches"
    assert result.searches > 0
    evaluate_logical_ops(paired_code, result.logical_ops)


def test_searches_and_seed(paired_code: codes.CSSCode, pytestconfig: pytest.Config) -> None:
    """The number of searches is counted exactly, and a seed makes the search reproducible."""
    seed = pytestconfig.getoption("randomly_seed")
    with mock.patch.object(
        optimization,
        "get_distance_from_generators",
        side_effect=optimization.get_distance_from_generators,
    ) as search:
        result = optimize_logical_basis(paired_code, seed=seed)
    assert result.searches == search.call_count

    repeat = optimize_logical_basis(paired_code, seed=seed)
    np.testing.assert_array_equal(repeat.logical_ops, result.logical_ops)
    assert repeat.searches == result.searches


@pytest.mark.parametrize("function", [_max_weight, _total_weight, _z_weight])
def test_moves(paired_code: codes.CSSCode, function: Callable[..., Any]) -> None:
    """A move is taken exactly when it improves the objective, by the brute-force class weights.

    Starting from the code's own basis, some moves give an operator a class heavier than both
    classes that the move replaces.
    """
    weights_x = _get_class_weights(paired_code, Pauli.X, "dressed")
    weights_z = _get_class_weights(paired_code, Pauli.Z, "dressed")
    search = _ClassSearch(paired_code, function, "dressed", None)
    classes_x, classes_z = [1, 2, 4], [1, 2, 4]
    current_x = [weights_x[_to_bits(c, 3)] for c in classes_x]
    current_z = [weights_z[_to_bits(c, 3)] for c in classes_z]
    value = function(current_x, current_z)
    for i, j in itertools.permutations(range(3), 2):
        new_x, new_z = classes_x[i] ^ classes_x[j], classes_z[j] ^ classes_z[i]
        trial_x, trial_z = list(current_x), list(current_z)
        trial_x[i], trial_z[j] = weights_x[_to_bits(new_x, 3)], weights_z[_to_bits(new_z, 3)]
        improves = search._improves(current_x, current_z, value, i, j, new_x, new_z)
        assert improves == (function(trial_x, trial_z) < value)


def test_descent_stops_when_out_of_searches(paired_code: codes.CSSCode) -> None:
    """A descent tries no move once the search has used up its budget of distance searches."""
    unbudgeted = _record_descent(paired_code, budgeted=False)
    budgeted = _record_descent(paired_code, budgeted=True)
    assert budgeted and not any(budgeted)
    assert len(budgeted) < len(unbudgeted)  # the budget cut the descent short


def test_perturbation_stops_when_out_of_searches(paired_code: codes.CSSCode) -> None:
    """A perturbation tries no move once the search has used up its budget of distance searches.

    Each random move searches at most the two classes that it would change, so a perturbation that
    checks the budget before each move runs at most one search more than its budget.
    """
    unbudgeted, budget = _count_perturbation_searches(paired_code, budgeted=False)
    budgeted, _ = _count_perturbation_searches(paired_code, budgeted=True)
    assert budgeted <= budget + 1
    assert unbudgeted > budget + 1  # without the budget, the perturbation runs further


@pytest.mark.parametrize("swap", [False, True])
def test_start(swap: bool) -> None:
    """Of the three starts, the search takes the one with the smallest maximum weight.

    The starts are the code's own basis and the lightest basis of either Pauli type, with the
    classes that complete them.  Swapping the checks of the code swaps which lightest basis wins.
    """
    code = codes.CSSCode(*(_PAIRED_CHECKS[::-1] if swap else _PAIRED_CHECKS))
    weights_x = _get_class_weights(code, Pauli.X, "dressed")
    weights_z = _get_class_weights(code, Pauli.Z, "dressed")
    k = code.dimension

    # the classes of the starts, with the classes that complete them
    starts = [(np.eye(k, dtype=int), np.eye(k, dtype=int))]
    paulis: list[PauliXZ] = [Pauli.X, Pauli.Z]
    for pauli in paulis:
        lightest = get_lightest_logical_basis(code, pauli).logical_ops.astype(int)
        classes = lightest @ np.asarray(code.get_logical_ops(pauli.swap_xz()), dtype=int).T % 2
        duals = np.asarray(np.linalg.inv(galois.GF2(classes)).T, dtype=int)
        starts.append((classes, duals) if pauli is Pauli.X else (duals, classes))
    values = [
        _max_weight(
            [weights_x[tuple(int(bit) for bit in c)] for c in classes_x],
            [weights_z[tuple(int(bit) for bit in c)] for c in classes_z],
        )
        for classes_x, classes_z in starts
    ]
    assert values.index(min(values)) == (2 if swap else 1)

    search = _ClassSearch(code, _max_weight, "dressed", None)
    classes_x, classes_z = search._get_start(*_get_lightest_bases(code))
    start_x = [weights_x[_to_bits(c, k)] for c in classes_x]
    start_z = [weights_z[_to_bits(c, k)] for c in classes_z]
    assert _max_weight(start_x, start_z) == min(values)


def test_start_by_max_weight() -> None:
    """The start has the smallest maximum weight, even if another start has a better objective."""
    # a [[6, 4]] code whose own basis has maximum weight 2 and total Z-type weight 8, while its
    # Z-type lightest basis, completed by the classes that pair with it, has a larger maximum weight
    # and a smaller total Z-type weight
    code = codes.CSSCode([[0, 1, 1, 1, 1, 1]], [[1, 0, 1, 1, 1, 1]])
    weights_x = _get_class_weights(code, Pauli.X, "dressed")
    weights_z = _get_class_weights(code, Pauli.Z, "dressed")
    k = code.dimension
    own = [tuple(int(bit) for bit in row) for row in np.eye(k, dtype=int)]
    assert [weights_x[c] for c in own] == [weights_z[c] for c in own] == [2] * k

    # the Z-type lightest basis, completed by the X-type classes that pair with it
    lightest_x, lightest_z = _get_lightest_bases(code)
    dual_ops = np.asarray(code.get_logical_ops(Pauli.X), dtype=int)
    classes = lightest_z.logical_ops.astype(int) @ dual_ops.T % 2
    duals = np.asarray(np.linalg.inv(galois.GF2(classes)).T, dtype=int)
    other_x = [weights_x[tuple(int(bit) for bit in row)] for row in duals]
    other_z = [weights_z[tuple(int(bit) for bit in row)] for row in classes]
    assert max(other_x + other_z) > 2 and sum(other_z) < 2 * k

    search = _ClassSearch(code, _z_weight, "dressed", None)
    classes_x, classes_z = search._get_start(lightest_x, lightest_z)
    assert classes_x == classes_z == [1, 2, 4, 8]


def test_special_inputs() -> None:
    """A code without logical qubits has an empty optimal basis, and arguments are validated."""
    code = codes.CSSCode([[1, 1]], [[1, 1]])
    result = optimize_logical_basis(code, objective=_z_weight)
    assert result.logical_ops.shape == (0, 4)
    assert result.optimal
    assert result.lower_bound == 0

    with pytest.raises(ValueError, match="objective"):
        optimize_logical_basis(code, objective="load")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        optimize_logical_basis(code, patience=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        optimize_logical_basis(code, max_searches=-1)


def _record_descent(code: codes.CSSCode, budgeted: bool) -> list[bool]:
    """Descend from the code's basis, and record before each move whether a budget is used up."""
    search, budget = _get_search_with_budget(code, budgeted)

    # pass moves through, recording before each one whether the budget is used up
    out_of_searches = []
    improves = search._improves

    def record_move(*args: Any) -> bool:
        out_of_searches.append(search.searches >= budget)
        return improves(*args)

    with mock.patch.object(search, "_improves", side_effect=record_move):
        search._descend([1, 2, 4], [1, 2, 4])
    return out_of_searches


def _count_perturbation_searches(code: codes.CSSCode, budgeted: bool) -> tuple[int, int]:
    """Perturb the code's basis, and return the total number of searches run, and a budget."""
    search, budget = _get_search_with_budget(code, budgeted)
    search._perturb([1, 2, 4], [1, 2, 4], np.random.default_rng(0))
    return search.searches, budget


def _get_search_with_budget(code: codes.CSSCode, budgeted: bool) -> tuple[_ClassSearch, int]:
    """A search that knows the weights of the code's basis, and a budget for it.

    The budget allows one search more than finding those weights took, and the search has it only if
    ``budgeted``.
    """
    search = _ClassSearch(code, _max_weight, "dressed", None)
    search._get_value([1, 2, 4], [1, 2, 4])
    budget = search.searches + 1
    if budgeted:
        search.max_searches = budget
    return search, budget


def _get_lightest_bases(code: codes.CSSCode) -> tuple[LightestLogicalBasis, LightestLogicalBasis]:
    """The lightest bases of the two Pauli types."""
    return get_lightest_logical_basis(code, Pauli.X), get_lightest_logical_basis(code, Pauli.Z)


def _get_lower_bound(
    code: codes.CSSCode, logicals: Literal["dressed", "bare"], function: Callable[..., Any]
) -> Any:
    """The objective of the lightest bases of the two Pauli types."""
    lightest_x = get_lightest_logical_basis(code, Pauli.X, logicals=logicals)
    lightest_z = get_lightest_logical_basis(code, Pauli.Z, logicals=logicals)
    return function(lightest_x.weights.tolist(), lightest_z.weights.tolist())


def _get_brute_force_optimum(
    code: codes.CSSCode, logicals: Literal["dressed", "bare"], function: Callable[..., Any]
) -> Any:
    """The best value of an objective over all complete bases, by enumerating class bases."""
    weights_x = _get_class_weights(code, Pauli.X, logicals)
    weights_z = _get_class_weights(code, Pauli.Z, logicals)
    k = code.dimension
    best = None
    for classes in itertools.product(list(weights_x), repeat=k):
        matrix = galois.GF2(np.array(classes, dtype=int).reshape(k, k))
        if np.linalg.matrix_rank(matrix) < k:
            continue
        duals = [tuple(int(bit) for bit in column) for column in np.linalg.inv(matrix).T]
        value = function([weights_x[c] for c in classes], [weights_z[c] for c in duals])
        best = value if best is None or value < best else best
    return best


def _get_class(code: codes.CSSCode, pauli: PauliXZ, operator: npt.ArrayLike) -> tuple[int, ...]:
    """The class of an operator: its pairing with each of the code's opposite-type operators."""
    dual_ops = np.asarray(code.get_logical_ops(pauli.swap_xz()), dtype=int)
    return tuple(int(bit) for bit in dual_ops @ np.asarray(operator, dtype=int) % 2)


def _get_improving_moves(
    code: codes.CSSCode,
    function: Callable[..., Any],
    classes_x: Sequence[tuple[int, ...]],
    classes_z: Sequence[tuple[int, ...]],
) -> list[tuple[int, int]]:
    """The moves that improve a class basis, by the brute-force class weights."""
    weights_x = _get_class_weights(code, Pauli.X, "dressed")
    weights_z = _get_class_weights(code, Pauli.Z, "dressed")
    current_x = [weights_x[c] for c in classes_x]
    current_z = [weights_z[c] for c in classes_z]
    value = function(current_x, current_z)
    moves = []
    for i, j in itertools.permutations(range(len(classes_x)), 2):
        new_x = tuple(a ^ b for a, b in zip(classes_x[i], classes_x[j]))
        new_z = tuple(a ^ b for a, b in zip(classes_z[j], classes_z[i]))
        trial_x, trial_z = list(current_x), list(current_z)
        trial_x[i], trial_z[j] = weights_x[new_x], weights_z[new_z]
        moves.append((i, j, function(trial_x, trial_z) < value))
    return [(i, j) for i, j, improves in moves if improves]


def _to_bits(c: int, k: int) -> tuple[int, ...]:
    """The bits of a class given as an integer."""
    return tuple(c >> m & 1 for m in range(k))


def _get_class_weights(
    code: codes.CSSCode, pauli: PauliXZ, logicals: Literal["dressed", "bare"]
) -> dict[tuple[int, ...], int]:
    """The minimum weight of each nontrivial logical class of one Pauli type, by enumeration.

    The class of an operator is its pairing with each of the code's logical operators of the
    opposite type.  Dressed operators commute with the stabilizers of the opposite type, and bare
    ones with its gauge operators.
    """
    other = pauli.swap_xz()
    checks = code.get_stabilizer_ops(other) if logicals == "dressed" else code.get_matrix(other)
    space: npt.NDArray[np.int_] = np.asarray(galois.GF2(np.asarray(checks)).null_space(), dtype=int)
    dual_ops = np.asarray(code.get_logical_ops(other), dtype=int)
    weights: dict[tuple[int, ...], int] = {}
    for combination in itertools.product([0, 1], repeat=len(space)):
        word = np.array(combination, dtype=int) @ space % 2
        bits = tuple(int(bit) for bit in dual_ops @ word % 2)
        if any(bits):
            weights[bits] = min(weights.get(bits, len(word)), int(word.sum()))
    return weights
