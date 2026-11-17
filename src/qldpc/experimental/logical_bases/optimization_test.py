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

from .evaluation import evaluate_logical_ops
from .lightest_basis import get_lightest_logical_basis
from .optimization import _ClassSearch, optimize_logical_basis


@pytest.fixture
def paired_code() -> codes.CSSCode:
    """A [[7, 3]] code whose optimal complete basis is heavier than its lightest bases."""
    return codes.CSSCode(
        [[0, 1, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1]],
        [[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1, 1]],
    )


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
    """The search finds an optimal complete basis, which pairing makes heavier than the bound."""
    result = optimize_logical_basis(paired_code, objective=objective, seed=0)
    metrics_x, metrics_z = evaluate_logical_ops(paired_code, result.logical_ops)
    value = function(metrics_x.weights.tolist(), metrics_z.weights.tolist())
    assert value == _get_brute_force_optimum(paired_code, "dressed", function)

    if callable(objective):
        assert result.lower_bound is None
    else:
        assert value > result.lower_bound
    assert not result.optimal
    assert result.stop_reason == "patience"
    np.testing.assert_array_equal(result.metrics[0].weights, metrics_x.weights)


def test_certified_optimum() -> None:
    """The search stops at the lower bound, which proves its result optimal."""
    code = codes.ToricCode(4)
    with mock.patch.object(
        _ClassSearch, "_perturb", autospec=True, side_effect=_ClassSearch._perturb
    ) as perturb:
        result = optimize_logical_basis(code, seed=0)
    assert not perturb.called
    assert result.optimal
    assert result.stop_reason == "lower_bound"
    assert result.lower_bound == (4, 4, 4, 4)
    metrics_x, metrics_z = evaluate_logical_ops(code, result.logical_ops)
    assert (metrics_x.max_weight, metrics_z.max_weight) == (4, 4)


@pytest.mark.parametrize("logicals", ["dressed", "bare"])
def test_subsystem_code(
    subsystem_code: codes.CSSCode, logicals: Literal["dressed", "bare"]
) -> None:
    """Dressed and bare complete bases of a subsystem code are optimal, and bare ones can be set."""
    result = optimize_logical_basis(subsystem_code, logicals=logicals, seed=0)
    metrics_x, metrics_z = evaluate_logical_ops(
        subsystem_code, result.logical_ops, logicals=logicals
    )
    value = _max_weight(metrics_x.weights.tolist(), metrics_z.weights.tolist())
    assert value == _get_brute_force_optimum(subsystem_code, logicals, _max_weight)
    if logicals == "bare":
        subsystem_code.set_logical_ops(result.logical_ops.astype(int))


def test_perturbations_escape_local_optima() -> None:
    """Perturbations lead from a basis that no move improves to an optimal one."""
    code = codes.CSSCode(
        [[1, 1, 0, 1, 1, 1, 1, 1]], [[0, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 1, 0, 1]]
    )
    assert not optimize_logical_basis(code, patience=0, seed=0).optimal
    assert optimize_logical_basis(code, seed=0).optimal


def test_stopping(paired_code: codes.CSSCode) -> None:
    """The search stops when it runs out of searches, or after one descent without patience."""
    result = optimize_logical_basis(paired_code, max_searches=1, seed=0)
    assert result.stop_reason == "max_searches"
    assert result.searches >= 1
    evaluate_logical_ops(paired_code, result.logical_ops)

    result = optimize_logical_basis(paired_code, patience=0, seed=0)
    assert result.stop_reason == "patience"


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


def test_descent_stops_when_out_of_searches(
    paired_code: codes.CSSCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descent tries no move once the search has run its maximum number of distance searches."""
    search = _ClassSearch(paired_code, _max_weight, "dressed", None)
    classes_x, classes_z = search._get_start()
    search.max_searches = search.searches + 1
    out_of_searches = []
    improves = _ClassSearch._improves

    def record_move(self: _ClassSearch, *args: Any) -> bool:
        out_of_searches.append(self._is_out_of_searches())
        return improves(self, *args)

    monkeypatch.setattr(_ClassSearch, "_improves", record_move)
    search._descend(classes_x, classes_z)
    assert search._is_out_of_searches()
    assert out_of_searches and not any(out_of_searches)


def test_start(paired_code: codes.CSSCode) -> None:
    """The search starts from the best of the code's basis and the lightest bases, with duals."""
    weights_x = _get_class_weights(paired_code, Pauli.X, "dressed")
    weights_z = _get_class_weights(paired_code, Pauli.Z, "dressed")
    k = paired_code.dimension

    # the classes of the candidates, with the classes that complete them
    candidates = [(np.eye(k, dtype=int), np.eye(k, dtype=int))]
    paulis: list[PauliXZ] = [Pauli.X, Pauli.Z]
    for pauli in paulis:
        lightest = get_lightest_logical_basis(paired_code, pauli).logical_ops.astype(int)
        classes = (
            lightest @ np.asarray(paired_code.get_logical_ops(pauli.swap_xz()), dtype=int).T % 2
        )
        duals = np.asarray(np.linalg.inv(galois.GF2(classes)).T, dtype=int)
        candidates.append((classes, duals) if pauli is Pauli.X else (duals, classes))
    values = [
        _max_weight(
            [weights_x[tuple(int(bit) for bit in c)] for c in classes_x],
            [weights_z[tuple(int(bit) for bit in c)] for c in classes_z],
        )
        for classes_x, classes_z in candidates
    ]
    assert min(values) < values[0]

    search = _ClassSearch(paired_code, _max_weight, "dressed", None)
    classes_x, classes_z = search._get_start()
    start_x = [weights_x[_to_bits(c, k)] for c in classes_x]
    start_z = [weights_z[_to_bits(c, k)] for c in classes_z]
    assert _max_weight(start_x, start_z) == min(values)


def test_special_inputs() -> None:
    """A code without logical qubits has an empty optimal basis, and objectives are validated."""
    code = codes.CSSCode([[1, 1]], [[1, 1]])
    result = optimize_logical_basis(code)
    assert result.logical_ops.shape == (0, 4)
    assert result.optimal

    with pytest.raises(ValueError, match="objective"):
        optimize_logical_basis(code, objective="load")  # type: ignore[arg-type]


def _get_brute_force_optimum(
    code: codes.CSSCode, logicals: str, function: Callable[..., Any]
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


def _to_bits(c: int, k: int) -> tuple[int, ...]:
    """The bits of a class given as an integer."""
    return tuple(c >> m & 1 for m in range(k))


def _get_class_weights(
    code: codes.CSSCode, pauli: PauliXZ, logicals: str
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
