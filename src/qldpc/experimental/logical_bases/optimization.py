# SPDX-License-Identifier: Apache-2.0

"""Search for complete bases of logical operators of binary CSS codes with low weights."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from qldpc import codes
from qldpc._binary import get_inverse
from qldpc.codes.distance import NO_DISTANCE, get_distance_from_generators
from qldpc.objects import Pauli

from ._sectors import Sector, _as_uint8, get_sector
from .evaluation import LogicalOpsMetrics
from .lightest_basis import LightestLogicalBasis, _get_min_weight_word, get_lightest_logical_basis

# Perturbations of the iterated local search keep every class within this much of the best maximum
# weight, so that they never have to resolve the weights of heavy classes, which are expensive.
_PERTURBATION_SLACK = 2

# Each perturbation applies between 2 and this many random moves (fewer if there are fewer logical
# qubits), and tries up to this many random moves per logical qubit to find each one.
_MAX_PERTURBATION_MOVES = 6
_ATTEMPTS_PER_MOVE = 4

####################################################################################################
# optimization


ClassObjective = Callable[[Sequence[int], Sequence[int]], Any]
"""A function of the weights of the X-type and Z-type operators of a basis; lower is better.

Its values must be totally ordered, such as numbers or tuples of numbers, and it must be monotone:
raising any one weight never lowers it.  The search relies on monotonicity to rule out moves from
lower bounds on weights, so a function that is not monotone can make it miss better bases.
"""


def optimize_logical_basis(
    code: codes.CSSCode,
    *,
    objective: Literal["max_weight", "total_weight"] | ClassObjective = "max_weight",
    logicals: Literal["dressed", "bare"] = "dressed",
    patience: int = 20,
    max_searches: int | None = None,
    seed: int | None = None,
) -> OptimizedLogicalBasis:
    """Search for a complete basis of logical operators of a CSS code with low weights.

    A complete basis has ``k`` X-type and ``k`` Z-type logical operators, with X-type operator ``j``
    anticommuting with Z-type operator ``j`` and commuting with all others.  For dressed operators
    of a subsystem code, these conditions hold for their logical classes, as
    :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops` describes.

    The smallest weights that a basis can have depend only on the logical classes of its operators,
    so the search runs over class bases: lists of the classes of the X-type and of the Z-type
    operators, paired as above.  Each class gets an operator of minimum weight.  The search starts
    from the code's own basis or the lightest basis of either Pauli type
    (:func:`~qldpc.experimental.logical_bases.get_lightest_logical_basis`), with the classes that
    complete it: whichever has the smallest maximum weight, with ties broken by the objective.

    A descent then takes moves while they lower the objective.  A move replaces the class of X-type
    operator ``i`` by its sum with the class of X-type operator ``j``, and the class of Z-type
    operator ``j`` by its sum with the class of Z-type operator ``i``, which keeps the classes
    paired.  Random moves, chosen with ``seed``, then perturb the best basis found, and descents
    from there continue the search.

    The weights of the lightest bases of the two Pauli types bound the objective from below.  The
    search stops once it reaches this lower bound, which proves the basis optimal, once ``patience``
    rounds of perturbation and descent in a row bring no improvement, or once it has run
    ``max_searches`` distance searches.  Pairing can force a basis to be heavier than the lightest
    bases, so a basis that does not reach the lower bound may still be optimal.

    Each class weight comes from an exact distance search (see :mod:`qldpc.codes.distance`).  Most
    searches only ask whether a class has an operator of at most some weight, so their cost depends
    on that weight rather than on the weight of the class.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        objective: What to minimize.  ``"max_weight"`` (the default) minimizes the largest of the
            ``2k`` weights, then the next largest, and so on.  ``"total_weight"`` minimizes the
            total weight, and breaks ties as ``"max_weight"`` does.  A function of the lists of the
            weights of the X-type and Z-type operators sets another objective (see
            :data:`ClassObjective`), for which there is no lower bound.
        logicals: For a subsystem code, ``"dressed"`` (the default) uses logical operators that
            commute with the stabilizers, while ``"bare"`` uses those that commute with every gauge
            operator.  The two are the same for a stabilizer code.  Only a basis of bare operators
            can be passed to :meth:`qldpc.codes.CSSCode.set_logical_ops`.
        patience: The number of rounds of perturbation and descent in a row without improvement
            after which the search stops.  With zero, the search stops after the first descent.
        max_searches: If given, a budget of distance searches.  The search always finishes finding
            its start, and checks the budget between moves and between rounds, so it can run more
            searches than the budget.  It then returns the best basis found so far.
        seed: A seed for the random perturbations, which makes the search reproducible.

    Returns:
        The basis found, with the reason why the search stopped.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, if ``objective`` or ``logicals`` is invalid, or if
            ``patience`` or ``max_searches`` is negative.
    """
    if patience < 0 or (max_searches is not None and max_searches < 0):
        raise ValueError("patience and max_searches must be nonnegative")
    function = _get_objective_function(objective)
    lightest_x = get_lightest_logical_basis(code, Pauli.X, logicals=logicals)
    lightest_z = get_lightest_logical_basis(code, Pauli.Z, logicals=logicals)
    lower_bound = (
        function(lightest_x.weights.tolist(), lightest_z.weights.tolist())
        if not callable(objective) or code.dimension == 0
        else None
    )

    search = _ClassSearch(code, function, logicals, max_searches)
    classes_x, classes_z, stop_reason = search.run(
        lightest_x, lightest_z, lower_bound, patience, np.random.default_rng(seed)
    )
    logical_ops = search.get_logical_ops(classes_x, classes_z)
    return OptimizedLogicalBasis(logical_ops, lower_bound, stop_reason, search.searches)


@dataclasses.dataclass(frozen=True, eq=False)
class OptimizedLogicalBasis:
    """A complete basis of logical operators found by a search for low weights.

    Attributes:
        logical_ops: The basis, with shape ``(2k, 2n)``, laid out as by
            :meth:`qldpc.codes.CSSCode.get_logical_ops`.
        lower_bound: The value of the objective for the lightest bases of the two Pauli types,
            which no complete basis can beat, or None for an objective given as a function.
        stop_reason: Why the search stopped: ``"lower_bound"`` if it reached the lower bound, which
            proves the basis optimal, ``"patience"`` if it stopped improving, or ``"max_searches"``
            if it used up its budget of distance searches.
        searches: The number of distance searches run to find the weights of classes.  The searches
            that find the lightest bases and recover operators of minimum weight are not counted.
    """

    logical_ops: npt.NDArray[np.uint8]
    lower_bound: Any
    stop_reason: Literal["lower_bound", "patience", "max_searches"]
    searches: int

    @property
    def optimal(self) -> bool:
        """Whether the basis is proven optimal, by reaching the lower bound."""
        return self.stop_reason == "lower_bound"

    @property
    def metrics(self) -> tuple[LogicalOpsMetrics, LogicalOpsMetrics]:
        """The metrics of the X-type and Z-type operators."""
        k, n = len(self.logical_ops) // 2, self.logical_ops.shape[1] // 2
        ops_x, ops_z = self.logical_ops[:k, :n], self.logical_ops[k:, n:]
        return LogicalOpsMetrics._from_ops(ops_x), LogicalOpsMetrics._from_ops(ops_z)


def _get_objective_function(
    objective: Literal["max_weight", "total_weight"] | ClassObjective,
) -> ClassObjective:
    """The objective as a function of the lists of weights of X-type and Z-type operators."""
    if callable(objective):
        return objective
    if objective == "max_weight":
        return _get_max_weight_objective
    if objective == "total_weight":
        return _get_total_weight_objective
    raise ValueError(
        f"objective must be 'max_weight', 'total_weight', or a function, not {objective!r}"
    )


def _get_max_weight_objective(
    weights_x: Sequence[int], weights_z: Sequence[int]
) -> tuple[int, ...]:
    """All ``2k`` weights in decreasing order, so that the largest is minimized first."""
    return tuple(sorted([*weights_x, *weights_z], reverse=True))


def _get_total_weight_objective(
    weights_x: Sequence[int], weights_z: Sequence[int]
) -> tuple[int, ...]:
    """The total of the ``2k`` weights, followed by the weights in decreasing order."""
    return (sum(weights_x) + sum(weights_z), *_get_max_weight_objective(weights_x, weights_z))


####################################################################################################
# class search


class _ClassTable:
    """Minimum weights of the logical classes of one Pauli type, and operators that have them.

    A class is an integer whose bit ``m`` is the pairing with the code's ``m``-th logical operator
    of the opposite type.  Weights come from exact distance searches, which are memoized, or from
    operators of known minimum weight, such as those of a lightest basis.  Lower bounds on the
    weights of classes are kept from searches limited to some weight.
    """

    def __init__(self, code: codes.CSSCode, sector: Sector) -> None:
        self.sector = sector
        self.reference_ops = _as_uint8(code.get_logical_ops(sector.pauli))
        self.floor = 1  # a lower bound on the weight of every nontrivial operator
        self.searches = 0
        self.weights: dict[int, int] = {}
        self.lower_bounds: dict[int, int] = {}
        self.operators: dict[int, npt.NDArray[np.uint8]] = {}

    def lower_bound(self, c: int) -> int:
        """The weight of class ``c`` if known, and otherwise a lower bound on it."""
        if c in self.weights:
            return self.weights[c]
        return max(self.floor, self.lower_bounds.get(c, 0))

    def weight_at_most(self, c: int, limit: int) -> int | None:
        """The weight of class ``c`` if it is at most ``limit``, and otherwise None."""
        if c not in self.weights and self.lower_bound(c) <= limit:
            self.searches += 1
            space, trivial = self._get_coset_space(c)
            weight = get_distance_from_generators(space, trivial, max_weight=limit)
            assert weight is not NO_DISTANCE
            if weight <= limit:
                self.weights[c] = int(weight)
            else:
                self.lower_bounds[c] = limit + 1
        known = self.weights.get(c)
        return known if known is not None and known <= limit else None

    def weight(self, c: int) -> int:
        """The weight of class ``c``."""
        weight = self.weight_at_most(c, self.sector.num_qubits)
        assert weight is not None
        return weight

    def record(self, lightest: LightestLogicalBasis) -> list[int]:
        """Record the operators of a lightest basis, which have minimum weight in their classes.

        Each operator of a lightest basis is a lightest operator outside the span of the classes of
        the operators before it, which contains no operator of its class, so it is a lightest
        operator of its class.  Returns the classes of the operators.
        """
        classes = [_from_bits(bits) for bits in self.sector.get_classes(lightest.logical_ops)]
        for c, operator in zip(classes, lightest.logical_ops, strict=True):
            self.weights[c] = int(operator.sum())
            self.operators[c] = operator
        self.floor = int(lightest.weights[0])
        return classes

    def get_operator(self, c: int) -> npt.NDArray[np.uint8]:
        """An operator of minimum weight in class ``c``."""
        if c not in self.operators:
            space, trivial = self._get_coset_space(c)
            self.operators[c] = _get_min_weight_word(space, trivial, self.weight(c))
        return self.operators[c]

    def _get_coset_space(self, c: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Rows whose span, outside span(trivial), is exactly the operators of class ``c``."""
        bits = _to_bits(c, self.sector.dimension).astype(np.int64)
        representative = (bits @ self.reference_ops.astype(np.int64) % 2).astype(np.uint8)
        return np.vstack([self.sector.trivial, representative]), self.sector.trivial


class _ClassSearch:
    """A local search over class bases, for a monotone objective of their weights.

    The X-type classes ``a`` and the Z-type classes ``b`` of a basis are lists of ``k`` integers,
    with ``a[i]`` and ``b[j]`` pairing to 1 exactly when ``i == j``.
    """

    def __init__(
        self,
        code: codes.CSSCode,
        objective: ClassObjective,
        logicals: Literal["dressed", "bare"],
        max_searches: int | None,
    ) -> None:
        self.objective = objective
        self.max_searches = max_searches
        self.table_x = _ClassTable(code, get_sector(code, Pauli.X, logicals))
        self.table_z = _ClassTable(code, get_sector(code, Pauli.Z, logicals))
        self.dimension = code.dimension

    @property
    def searches(self) -> int:
        """The number of distance searches run so far."""
        return self.table_x.searches + self.table_z.searches

    def run(
        self,
        lightest_x: LightestLogicalBasis,
        lightest_z: LightestLogicalBasis,
        lower_bound: Any,
        patience: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int], Literal["lower_bound", "patience", "max_searches"]]:
        """Search from the best start, and return the best classes and why the search stopped."""
        if self.dimension == 0:
            return [], [], "lower_bound"

        # descend from the best start
        best_x, best_z = self._descend(*self._get_start(lightest_x, lightest_z))
        best_value = self._get_value(best_x, best_z)

        # perturb the best basis and descend again, until rounds in a row stop improving it
        stale = 0
        while not self._is_out_of_searches() and best_value != lower_bound and stale < patience:
            classes_x, classes_z = self._descend(*self._perturb(best_x, best_z, rng))
            value = self._get_value(classes_x, classes_z)
            stale = 0 if value < best_value else stale + 1
            if value <= best_value:
                best_x, best_z, best_value = classes_x, classes_z, value

        # report why the search stopped, preferring the proof of optimality
        if best_value == lower_bound:
            return best_x, best_z, "lower_bound"
        stop_reason: Literal["patience", "max_searches"] = (
            "max_searches" if self._is_out_of_searches() else "patience"
        )
        return best_x, best_z, stop_reason

    def get_logical_ops(
        self, classes_x: Sequence[int], classes_z: Sequence[int]
    ) -> npt.NDArray[np.uint8]:
        """Operators of minimum weight in the given classes, as a basis with shape ``(2k, 2n)``."""
        n = self.table_x.sector.num_qubits
        logical_ops = np.zeros((2 * self.dimension, 2 * n), dtype=np.uint8)
        for i, (c_x, c_z) in enumerate(zip(classes_x, classes_z, strict=True)):
            logical_ops[i, :n] = self.table_x.get_operator(c_x)
            logical_ops[self.dimension + i, n:] = self.table_z.get_operator(c_z)
        return logical_ops

    def _get_start(
        self, lightest_x: LightestLogicalBasis, lightest_z: LightestLogicalBasis
    ) -> tuple[list[int], list[int]]:
        """The start with the smallest maximum weight, with ties broken by the objective.

        The candidates are the code's own basis and the lightest basis of either Pauli type, each
        with the classes that complete it.  A common limit on the weights rises from the smallest
        weight of a nontrivial operator until some candidate has every class within it, so the
        classes of the other candidates are never resolved above that limit.
        """
        classes_x = self.table_x.record(lightest_x)
        classes_z = self.table_z.record(lightest_z)
        code_classes = [1 << m for m in range(self.dimension)]
        candidates = [
            (code_classes, code_classes),
            (classes_x, _get_dual_classes(classes_x)),
            (_get_dual_classes(classes_z), classes_z),
        ]

        limit = min(self.table_x.floor, self.table_z.floor)
        while True:
            fitting = [
                (candidate_x, candidate_z)
                for candidate_x, candidate_z in candidates
                if all(self.table_x.weight_at_most(c, limit) is not None for c in candidate_x)
                and all(self.table_z.weight_at_most(c, limit) is not None for c in candidate_z)
            ]
            if fitting:
                return min(fitting, key=lambda candidate: self._get_value(*candidate))
            limit += 1

    def _get_value(self, classes_x: Sequence[int], classes_z: Sequence[int]) -> Any:
        """The objective of a class basis."""
        weights_x = [self.table_x.weight(c) for c in classes_x]
        weights_z = [self.table_z.weight(c) for c in classes_z]
        return self.objective(weights_x, weights_z)

    def _descend(self, classes_x: list[int], classes_z: list[int]) -> tuple[list[int], list[int]]:
        """Take improving moves, heaviest pairs first, until none improves or searches run out."""
        classes_x, classes_z = list(classes_x), list(classes_z)
        improved = True
        while improved and not self._is_out_of_searches():
            improved = False
            weights_x = [self.table_x.weight(c) for c in classes_x]
            weights_z = [self.table_z.weight(c) for c in classes_z]
            value = self.objective(weights_x, weights_z)
            moves = sorted(
                ((i, j) for i in range(self.dimension) for j in range(self.dimension) if i != j),
                key=lambda move: -max(weights_x[move[0]], weights_z[move[1]]),
            )
            for i, j in moves:
                if self._is_out_of_searches():
                    break
                new_x, new_z = classes_x[i] ^ classes_x[j], classes_z[j] ^ classes_z[i]
                if self._improves(weights_x, weights_z, value, i, j, new_x, new_z):
                    classes_x[i], classes_z[j] = new_x, new_z
                    improved = True
                    break
        return classes_x, classes_z

    def _improves(
        self,
        weights_x: list[int],
        weights_z: list[int],
        value: Any,
        i: int,
        j: int,
        new_x: int,
        new_z: int,
    ) -> bool:
        """Whether a move lowers the objective below ``value``.

        The move gives X-type operator ``i`` class ``new_x``, and Z-type operator ``j`` class
        ``new_z``.  Unknown weights are replaced by lower bounds, and a monotone objective that does
        not improve even then cannot improve at all.  Otherwise, the classes are searched up to a
        limit that doubles until their weights are known or the bounds rule out an improvement.
        """
        limit = max(weights_x[i], weights_z[j])
        trial_x, trial_z = list(weights_x), list(weights_z)
        while True:
            # check the bounds again before each search, which can make the next search unneeded
            for table, c in [(self.table_x, new_x), (self.table_z, new_z)]:
                trial_x[i] = self.table_x.lower_bound(new_x)
                trial_z[j] = self.table_z.lower_bound(new_z)
                if not self.objective(trial_x, trial_z) < value:
                    return False
                table.weight_at_most(c, limit)
            if new_x in self.table_x.weights and new_z in self.table_z.weights:
                trial_x[i], trial_z[j] = self.table_x.weights[new_x], self.table_z.weights[new_z]
                return bool(self.objective(trial_x, trial_z) < value)
            limit = min(2 * limit, self.table_x.sector.num_qubits)

    def _perturb(
        self, classes_x: list[int], classes_z: list[int], rng: np.random.Generator
    ) -> tuple[list[int], list[int]]:
        """Apply a few random moves that keep every class near the best maximum weight."""
        classes_x, classes_z = list(classes_x), list(classes_z)
        weights = [self.table_x.weight(c) for c in classes_x]
        weights += [self.table_z.weight(c) for c in classes_z]
        limit = max(weights) + _PERTURBATION_SLACK
        num_moves = 0
        if self.dimension > 1:
            num_moves = int(rng.integers(2, min(self.dimension, _MAX_PERTURBATION_MOVES) + 1))
        for _ in range(num_moves):
            for _ in range(_ATTEMPTS_PER_MOVE * self.dimension):
                i, j = (int(index) for index in rng.choice(self.dimension, size=2, replace=False))
                new_x, new_z = classes_x[i] ^ classes_x[j], classes_z[j] ^ classes_z[i]
                if (
                    self.table_x.weight_at_most(new_x, limit) is not None
                    and self.table_z.weight_at_most(new_z, limit) is not None
                ):
                    classes_x[i], classes_z[j] = new_x, new_z
                    break
        return classes_x, classes_z

    def _is_out_of_searches(self) -> bool:
        """Whether the search has used up its budget of distance searches."""
        return self.max_searches is not None and self.searches >= self.max_searches


def _get_dual_classes(classes: Sequence[int]) -> list[int]:
    """Classes ``d`` with ``classes[i]`` and ``d[j]`` pairing to 1 exactly when ``i == j``."""
    matrix = np.array([_to_bits(c, len(classes)) for c in classes], dtype=np.uint8)
    inverse = get_inverse(matrix)
    return [_from_bits(column) for column in inverse.T]


def _to_bits(c: int, k: int) -> npt.NDArray[np.uint8]:
    """The ``k`` bits of a class, least significant first."""
    return np.array([c >> m & 1 for m in range(k)], dtype=np.uint8)


def _from_bits(bits: npt.ArrayLike) -> int:
    """The class whose bits, least significant first, are given."""
    return sum(int(bit) << m for m, bit in enumerate(np.asarray(bits)))
