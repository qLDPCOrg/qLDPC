# SPDX-License-Identifier: Apache-2.0

"""Exact CSS/SWEL code annealing: the chain, its energy, and how proposals are scored.

A Markov chain of codes minimizes the energy E of arXiv:2607.19574, the normalized log of the rate
of undetected logical errors, over local Clifford conjugations of a binary CSS stabilizer code.
Every proposal is scored exactly by its distance and its number of minimum-weight logical
operators, so there are no approximate or cached scores.  By default the operators are the X-only
and Z-only ones; the ``pauli`` objective instead counts every physical Pauli representative,
including mixed and Y-only ones.

Before the first epoch, a pilot scores one epoch's worth of proposals from the initial code to fit
the first temperature, without moving the chain.  Independently of the chain, the search records the
lowest-energy code and the greatest-distance code (ties broken by the lowest multiplicity) among
*all* evaluated proposals, including pilot and rejected ones.

Scoring takes most of a step, and it has two exact methods that give identical scores, so the
method changes only the running time.  Brouwer-Zimmermann counts at a guess of the distance, and
verifies it; the MacWilliams method walks every word of the check spaces once.  With the ``auto``
method, the search races the two at the start of each epoch and keeps the faster, so the choice
depends on timing, and no result depends on it.  See ``_score_proposal``, ``_score_by_guessing``
and ``_select_method`` for the details.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from qldpc.codes import CSSLogicalCounter, CSSLogicalCounts, LogicalCountMethod

from .proposals import CODE_TYPES, CodeType, propose

# Exact counting methods of the library, and ``auto``, which times the two exact methods on a few
# proposals at the start of each epoch and keeps the faster.
SearchMethod = Literal["auto", "brouwer_zimmermann", "macwilliams", "brute_force"]
METHODS = get_args(SearchMethod)

# The method of the one-off counts that do not belong to a chain, and the independent checks.
_DEFAULT_LIBRARY_METHOD: LogicalCountMethod = "brouwer_zimmermann"

# Quantities that the multiplicity of the energy can count.
Objective = Literal["homogeneous", "pauli"]
OBJECTIVES = get_args(Objective)

# Number of recent proposals whose distances vote on the distance of the next one.
_GUESS_WINDOW = 16

# Limit on the information sets of a count.  Scoring a proposal is a short count on a large code,
# where the library default of 16 sets costs more in setup than it saves.
_MAX_INFORMATION_SETS = 8

# Bounds on the fitted log-temperature; a search with no uphill proposals uses the upper bound.
LOG_TEMPERATURE_BOUNDS = (-10.0, 0.0)

# Choosing a method.  The MacWilliams method walks all words of the check spaces, so it is only
# timed for check ranks up to _RACE_MAX_RANK.  Each method scores _RACE_PROPOSALS proposals of the
# current code (fewer in a short epoch).  The race stops early after _RACE_SECONDS, or once a method
# has taken _RACE_DECISIVE_RATIO times as long as the other for _RACE_MIN_PROPOSALS proposals.  The
# search keeps its method unless the other one takes less than _RACE_SWITCH_FRACTION of its time.
_RACE_MAX_RANK = 26
_RACE_PROPOSALS = 16
_RACE_SECONDS = 0.5
_RACE_MIN_PROPOSALS = 3
_RACE_DECISIVE_RATIO = 3.0
_RACE_SWITCH_FRACTION = 0.9
_RACE_STREAM = 0x52414345  # the ASCII bytes of "RACE"

_clock = time.process_time

####################################################################################################
# configuration and chain state


@dataclass(frozen=True)
class SearchConfig:
    """Search policy; the seed and codes of a chain live in ``SearchState`` and its checkpoint.

    The homogeneous objective counts the minimum-weight X-only and Z-only logical operators, and the
    pauli objective counts every minimum physical Pauli representative.  Neither counts logical
    equivalence classes modulo stabilizers.  The objective is recorded in every checkpoint so that
    saved results state exactly what was optimized.

    Attributes:
        code_type: ``css`` searches general CSS codes with CNOT or CSS gates; ``swel`` searches
            SWEL codes with the SWEL gate, which keeps every code self-dual.
        gate_size: Number of qubits that a proposed gate acts on.
        epochs: Number of epochs of the chain.  An epoch has one temperature.
        steps_per_epoch: Number of proposals in an epoch, and in the pilot before the first one.
        uphill_acceptance_start: Target mean acceptance probability of the uphill proposals (those
            that raise the energy) in the first epoch.  The temperature is fit to the proposals of
            the previous epoch (or of the pilot) and bounded by ``LOG_TEMPERATURE_BOUNDS``, so the
            acceptance of an epoch only approaches its target; ``EpochStatistics`` records what
            it was.
        uphill_acceptance_stop: The same target in the last epoch.  The targets in between decrease
            geometrically, which cools the chain.
        temperature_memory: Weight of the previous temperature in the temperature of the next
            epoch, which is otherwise the temperature that gives the target acceptance to the
            proposals of the last epoch.
        objective: What the multiplicity of the energy counts (see above).
        physical_error_rate: The physical error rate p of the energy.
        method: How proposals are scored (see the module docstring).
    """

    code_type: CodeType = "css"
    gate_size: int = 2
    objective: Objective = "homogeneous"
    epochs: int = 100
    steps_per_epoch: int = 10_000
    uphill_acceptance_start: float = 0.01
    uphill_acceptance_stop: float = 0.001
    temperature_memory: float = 0.5
    physical_error_rate: float = 1e-3
    method: SearchMethod = "auto"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.gate_size, self.epochs, self.steps_per_epoch)
        ):
            raise ValueError("Gate size, epochs, and steps_per_epoch must be integers")
        if self.code_type not in CODE_TYPES or self.objective not in OBJECTIVES:
            raise ValueError("Unrecognized code type or objective")
        if self.gate_size not in (2, 3, 4) or (self.code_type == "swel" and self.gate_size != 4):
            raise ValueError("CSS gates have size 2/3/4; SWEL searches require size 4")
        if self.epochs < 1 or self.steps_per_epoch < 1:
            raise ValueError("epochs and steps_per_epoch must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.uphill_acceptance_start,
                self.uphill_acceptance_stop,
                self.temperature_memory,
                self.physical_error_rate,
            )
        ):
            raise ValueError("Search parameters must be finite")
        if not 0 < self.uphill_acceptance_start < 1 or not 0 < self.uphill_acceptance_stop < 1:
            raise ValueError("Target acceptance rates must be strictly between zero and one")
        if not 0 <= self.temperature_memory < 1 or not 0 < self.physical_error_rate < 1:
            raise ValueError("temperature_memory must be in [0,1) and physical_error_rate in (0,1)")
        if self.method not in METHODS:
            raise ValueError(f"Unknown counting method {self.method!r}; choose from {METHODS}")
        if self.method == "macwilliams" and self.objective == "pauli" and self.code_type == "css":
            raise ValueError("The MacWilliams method counts Paulis only of self-dual codes")

    def validate_code(self, counter: CSSLogicalCounter) -> None:
        """Check code-dependent policy invariants at initialization and checkpoint import."""
        if counter.code.is_subsystem_code:
            raise ValueError("CSS stabilizer checks must commute; subsystem gauges are unsupported")
        if counter.dimension <= 0 or counter.num_qubits < self.gate_size:
            raise ValueError(
                "The initial code needs logical qubits and enough physical gate targets"
            )
        if self.code_type == "swel":
            if not counter.is_self_dual:
                raise ValueError("A SWEL search requires a self-dual initial code")
            if not counter.code.is_swel:
                raise ValueError("The self-dual initial code does not have the SWEL property")


@dataclass(frozen=True)
class EpochStatistics:
    """Completed-epoch observables, excluding the initial pilot evaluations.

    The uphill acceptance rate is 1.0 if no proposal of the epoch raised the energy.
    """

    epoch: int
    temperature: float
    acceptance_rate: float
    uphill_acceptance_rate: float
    current_energy: float
    lowest_energy: float


@dataclass(frozen=True)
class Score:
    """Exact distance and number of minimum-weight logical operators of a code.

    The counts are those of nontrivial logical operators whose weight is the distance: X-only,
    Z-only, and, for the pauli objective only, physical Pauli operators including mixed and Y-only
    ones.  Without that objective the Pauli count is None.
    """

    distance: int
    count_x: int
    count_z: int
    count_pauli: int | None

    @property
    def count_homogeneous(self) -> int:
        """The number of minimum-weight X-only plus Z-only logical operators."""
        return self.count_x + self.count_z

    @classmethod
    def from_counts(cls, counts: CSSLogicalCounts, config: SearchConfig) -> Score:
        """The score of a code for an objective, from library counts at its distance."""
        count_pauli = counts.count_pauli if config.objective == "pauli" else None
        return cls(counts.weight, counts.count_x, counts.count_z, count_pauli)


@dataclass(frozen=True)
class ScoredCode:
    """A code of the chain, as the counter that scores it, with its exact score."""

    counter: CSSLogicalCounter
    score: Score


@dataclass
class SearchState:
    """Owned chain state, with separate energy and distance optima.

    ``current`` is the code of the chain.  ``lowest_energy`` minimizes the selected energy, and
    ``greatest_distance`` maximizes distance and then minimizes energy at that distance; neither is
    restricted to accepted proposals.  Checkpoints are taken at completed epoch boundaries.
    """

    config: SearchConfig
    seed: int
    current: ScoredCode
    lowest_energy: ScoredCode
    greatest_distance: ScoredCode
    rng: np.random.Generator
    log_temperature: float
    energy_jumps: npt.NDArray[np.float64]
    completed_epochs: int = 0
    proposed: int = 0
    accepted: int = 0
    history: list[EpochStatistics] = field(default_factory=list)

    # Distances of the latest proposals, which speed up scoring.  They are neither saved nor
    # compared, so a restored chain starts with an empty window.
    recent_distances: deque[int] = field(
        default_factory=lambda: deque(maxlen=_GUESS_WINDOW), repr=False, compare=False
    )

    # The library method that scores the proposals of the current epoch, which is set when the
    # epoch starts and is not saved.
    epoch_method: LogicalCountMethod | None = field(default=None, repr=False, compare=False)


####################################################################################################
# search entry points


def start_search(initial: CSSLogicalCounter, config: SearchConfig, *, seed: int = 0) -> SearchState:
    """Create a chain at an initial code, and run its pilot.

    The pilot scores one epoch's worth of neighbors of the initial code, each proposed from that
    code, and fits the first temperature to their energy jumps.  It never moves the chain, but its
    proposals can set the optimum records.

    Args:
        initial: Owned code counter, with at least one logical qubit.
        config: Exact scoring and proposal policy.
        seed: NumPy search seed.
    """
    config.validate_code(initial)
    start = ScoredCode(initial, score_code(initial, config))
    state = SearchState(
        config=config,
        seed=seed,
        current=start,
        lowest_energy=start,
        greatest_distance=start,
        rng=np.random.default_rng(seed),
        log_temperature=0.0,
        energy_jumps=np.empty(config.steps_per_epoch, dtype=float),
    )
    _start_epoch(state)
    initial_energy = energy(start.score, config)
    for step in range(config.steps_per_epoch):
        candidate = propose(
            initial, state.rng, code_type=config.code_type, gate_size=config.gate_size
        )
        proposal = ScoredCode(candidate, _score_proposal(state, candidate))
        state.energy_jumps[step] = energy(proposal.score, config) - initial_energy
        _update_optima(state, proposal)
    state.log_temperature = _fit_log_temperature(state.energy_jumps, config.uphill_acceptance_start)
    return state


def run_epochs(
    state: SearchState,
    *,
    stop_after_epochs: int | None = None,
    on_epoch: Callable[[SearchState], None] | None = None,
    on_lowest_energy: Callable[[SearchState], None] | None = None,
    on_greatest_distance: Callable[[SearchState], None] | None = None,
) -> SearchState:
    """Continue a chain, optionally stopping at a reproducible completed-epoch boundary.

    Every proposal is scored exactly, whether or not it is accepted, and its energy jump enters the
    temperature fit of the next epoch.

    Args:
        state: The chain, which is updated in place and returned.
        stop_after_epochs: Number of further epochs to run, or None to run the remaining epochs of
            the policy.
        on_epoch: Called after every completed epoch, such as to save a restartable checkpoint.
        on_lowest_energy: Called as soon as a proposal sets a new lowest-energy code, such as to
            preserve that code before its epoch completes.
        on_greatest_distance: The same for a new greatest-distance code, even if its energy is not
            the lowest.
    """
    if stop_after_epochs is not None and stop_after_epochs < 0:
        raise ValueError("stop_after_epochs must be nonnegative")
    config = state.config
    stop = config.epochs
    if stop_after_epochs is not None:
        stop = min(stop, state.completed_epochs + stop_after_epochs)
    targets = np.geomspace(
        config.uphill_acceptance_start, config.uphill_acceptance_stop, config.epochs
    )
    for epoch in range(state.completed_epochs, stop):
        _run_epoch(state, float(targets[epoch]), on_lowest_energy, on_greatest_distance)
        if on_epoch is not None:
            on_epoch(state)
    return state


####################################################################################################
# energy, temperature and acceptance


def get_library_method(config: SearchConfig) -> LogicalCountMethod:
    """The library method of a count outside a chain, where ``auto`` means the default."""
    return _DEFAULT_LIBRARY_METHOD if config.method == "auto" else config.method


def score_code(
    counter: CSSLogicalCounter, config: SearchConfig, method: LogicalCountMethod | None = None
) -> Score:
    """Exact distance and multiplicity of the minimum-weight logicals for the objective.

    A Brouwer--Zimmermann count leaves state in its counter that the codes deformed from it inherit
    (see ``_start_epoch``), which helps only if Brouwer--Zimmermann also scores the chain.  The
    ``auto`` method can choose either method in each epoch, so its default count runs on a copy of
    the code and leaves the given counter as it is.

    Args:
        counter: The code to score.
        config: Search policy, whose objective is scored.
        method: Library counting method, by default that of ``get_library_method``.
    """
    if method is None:
        method = get_library_method(config)
        if config.method == "auto":
            counter = CSSLogicalCounter(counter.code)
    counts = counter.count_logical_ops(
        method=method,
        count_pauli=config.objective == "pauli",
        max_information_sets=_MAX_INFORMATION_SETS,
    )
    return Score.from_counts(counts, config)


def energy(score: Score, config: SearchConfig) -> float:
    """The energy E of arXiv:2607.19574: the normalized log of the undetected logical error rate.

    An error is undetected and changes the logical state when it is a nontrivial logical operator,
    which at leading order happens with probability m * p**d, where m is the multiplicity that the
    objective counts, d is the distance, and p is the physical error rate.  Return the log of this
    probability over log(1 / p), which is log(m) / log(1 / p) - d.  A greater distance lowers the
    energy unless it comes with more than 1 / p times as many minimum-weight logical operators.
    """
    count = _get_multiplicity(score, config)
    return math.log(count) / -math.log(config.physical_error_rate) - score.distance


def get_lowest_energy_code(
    counters: Iterable[CSSLogicalCounter], config: SearchConfig
) -> CSSLogicalCounter:
    """The code of the lowest energy among codes that are scored exactly, the first if tied.

    Starting a chain from the lowest-energy code of many independent random codes puts it in a good
    region, much like a first epoch at infinite temperature, which accepts every proposal, but with
    uncorrelated codes.  The codes are consumed one at a time, so they can be an endless stream cut
    short by ``itertools.islice``.

    Raises:
        ValueError: If there are no codes.
    """
    lowest, lowest_energy = None, math.inf
    for counter in counters:
        config.validate_code(counter)
        value = energy(score_code(counter, config), config)
        if lowest is None or value < lowest_energy:
            lowest, lowest_energy = counter, value
    if lowest is None:
        raise ValueError("There are no codes to choose the lowest-energy code from")
    return lowest


def distance_rank(distance: int, count: int) -> tuple[int, int]:
    """The rank of a code among the codes of its size, which is greater for a better code.

    A greater distance ranks higher, and then fewer minimum-weight logical operators.
    """
    return distance, -count


def score_rank(score: Score, config: SearchConfig) -> tuple[int, int]:
    """The ``distance_rank`` of a score, compared as exact integers instead of rounded energies.

    The energy increases strictly with the multiplicity at a fixed distance, so among codes of one
    distance this rank orders codes as the energy does.
    """
    return distance_rank(score.distance, _get_multiplicity(score, config))


####################################################################################################
# helper methods


def _run_epoch(
    state: SearchState,
    target_rate: float,
    on_lowest_energy: Callable[[SearchState], None] | None,
    on_greatest_distance: Callable[[SearchState], None] | None,
) -> None:
    """Run one epoch of proposals, then record its statistics at the epoch boundary."""
    config = state.config
    _start_epoch(state)

    # Smooth the temperature toward the fit to the previous epoch's proposal jumps.
    state.log_temperature = config.temperature_memory * state.log_temperature + (
        1 - config.temperature_memory
    ) * _fit_log_temperature(state.energy_jumps, target_rate)
    temperature = math.exp(state.log_temperature)

    # Propose from the current code.  Every proposal, accepted or not, can set an optimum record,
    # and its actual jump (not just accepted ones) feeds the next temperature fit.
    current_energy = energy(state.current.score, config)
    accepted = uphill = accepted_uphill = 0
    for step in range(config.steps_per_epoch):
        candidate = propose(
            state.current.counter, state.rng, code_type=config.code_type, gate_size=config.gate_size
        )
        proposal = ScoredCode(candidate, _score_proposal(state, candidate))
        new_energy = energy(proposal.score, config)
        jump = new_energy - current_energy
        state.energy_jumps[step] = jump
        uphill += jump > 0
        if _accept_move(jump, temperature, float(state.rng.random())):
            state.current = proposal
            current_energy = new_energy
            accepted += 1
            accepted_uphill += jump > 0

        # Both records are consistent before either persistence callback sees the state.
        improved_energy, improved_distance = _update_optima(state, proposal)
        if improved_energy and on_lowest_energy is not None:
            on_lowest_energy(state)
        if improved_distance and on_greatest_distance is not None:
            on_greatest_distance(state)

    # The epoch is complete: advance the counters and log its observables.
    state.completed_epochs += 1
    state.proposed += config.steps_per_epoch
    state.accepted += accepted
    state.history.append(
        EpochStatistics(
            state.completed_epochs,
            temperature,
            accepted / config.steps_per_epoch,
            accepted_uphill / uphill if uphill else 1.0,
            current_energy,
            energy(state.lowest_energy.score, config),
        )
    )


def _score_proposal(state: SearchState, candidate: CSSLogicalCounter) -> Score:
    """Score a proposal exactly with the method of the epoch.

    The MacWilliams method gets every weight from one enumeration of the checks, so it counts
    at the distance directly.  Any other method first tries to count at the distance that is likely
    (see ``_score_by_guessing``).  The result is exact either way.
    """
    method = state.epoch_method
    assert method is not None, "_start_epoch sets the method of an epoch before it scores"
    if method == "macwilliams":
        score = score_code(candidate, state.config, method)
        state.recent_distances.append(score.distance)
        return score
    return _score_by_guessing(state, candidate, method)


def _score_by_guessing(
    state: SearchState, candidate: CSSLogicalCounter, method: LogicalCountMethod
) -> Score:
    """Score a proposal exactly, first trying to count at the distance that is likely.

    Most proposals keep the distance of their neighbors, and counting at a known weight skips the
    distance search.  A proposal is one gate away from the current code, so its distance lies within
    a small range of the current one.  Each failed guess says whether the distance is lower or
    higher, so the next guess is the adjacent weight on that side, which takes at most one guess per
    weight of the range.  The result is exact either way: only a positive count at the guessed
    weight is accepted, and the standard computation provides the score if the range runs out.
    """
    reach = _get_distance_reach(state.config)
    lowest = max(1, state.current.score.distance - reach)
    highest = min(candidate.num_qubits, state.current.score.distance + reach)
    guess = min(max(_guess_distance(state), lowest), highest)

    # Each failed guess removes itself and everything on the wrong side from the possible range.
    score = None
    while score is None and lowest <= highest:
        outcome = _score_at_guess(candidate, guess, state.config, method)
        if isinstance(outcome, Score):
            score = outcome
        elif outcome == "below":
            highest = guess = guess - 1
        else:
            lowest = guess = guess + 1
    if score is None:
        score = score_code(candidate, state.config, method)
    state.recent_distances.append(score.distance)
    return score


def _get_distance_reach(config: SearchConfig) -> int:
    """The most that one proposal can change the distance of a code.

    A gate acts on ``gate_size`` qubits, and an invertible action maps a nonzero restriction of an
    operator to a nonzero one, so the weight of every logical operator changes by at most
    ``gate_size - 1``.  The SWEL gate maps an even-weight restriction to itself and an odd-weight
    one to its complement, which limits the change to 2.  This only speeds up the search, because
    scores never rely on it.
    """
    return 2 if config.code_type == "swel" else config.gate_size - 1


def _guess_distance(state: SearchState) -> int:
    """The most common distance among recent proposals, else that of the current code."""
    if state.recent_distances:
        return Counter(state.recent_distances).most_common(1)[0][0]
    return state.current.score.distance


def _score_at_guess(
    counter: CSSLogicalCounter,
    guess: int,
    config: SearchConfig,
    method: LogicalCountMethod,
) -> Score | Literal["below", "above"]:
    """The score of a code if its distance is the guess, else which side of the guess it is on.

    Counting within the distance rejects a weight above the distance, and finds no X-only or Z-only
    logical operator below it.  Hence the code has distance ``guess`` exactly when the count
    succeeds and finds some operator; "below" means that the distance is lower than the guess, and
    "above" that it is higher.
    """
    try:  # the library raises a ValueError for a weight above the distance
        counts = counter.count_logical_ops(
            guess,
            method=method,
            count_pauli=config.objective == "pauli",
            within_distance=True,
            max_information_sets=_MAX_INFORMATION_SETS,
        )
    except ValueError:
        return "below"
    return Score.from_counts(counts, config) if counts.count_x or counts.count_z else "above"


def _start_epoch(state: SearchState) -> None:
    """Choose the method of the epoch, and prepare the current code for it.

    The children of a code inherit the Brouwer--Zimmermann state of their parent, and update it at
    every deformation.  With the ``auto`` method, the current code gets that state when the epoch
    uses Brouwer--Zimmermann, so that its children start prepared.  When the epoch uses the
    MacWilliams method instead, which has no use for that state, the chain drops it by continuing
    from a copy of the current code.
    """
    previous = state.epoch_method
    method = _select_method(state)
    if state.config.method == "auto":
        if method == "brouwer_zimmermann":
            state.current.counter.count_logical_ops(
                state.current.score.distance,
                method=method,
                max_information_sets=_MAX_INFORMATION_SETS,
            )
        elif previous == "brouwer_zimmermann":
            fresh = CSSLogicalCounter(state.current.counter.code)
            state.current = replace(state.current, counter=fresh)
    state.epoch_method = method


def _select_method(state: SearchState) -> LogicalCountMethod:
    """The method that scores the proposals of the next epoch.

    A configured method is used as it is.  For ``auto``, the two exact methods score the same
    proposals of the current code, and the faster one wins.  The MacWilliams method costs the
    same for every proposal of a chain, because a local gate keeps the check ranks, while the cost
    of Brouwer--Zimmermann changes with the distance, so the race is repeated at every epoch.  The
    proposals come from their own random generator, so that the race never moves the chain, and
    each method scores its own copy of every proposal.
    """
    config = state.config
    if config.method != "auto":
        return config.method
    counter = state.current.counter
    # Physical Pauli counts of the MacWilliams method need self-dual codes, which only the SWEL gate
    # keeps.
    if max(len(counter.checks_x), len(counter.checks_z)) > _RACE_MAX_RANK or (
        config.objective == "pauli" and config.code_type != "swel"
    ):
        return "brouwer_zimmermann"
    methods: tuple[LogicalCountMethod, ...] = ("brouwer_zimmermann", "macwilliams")

    # Both methods score children of a copy that Brouwer--Zimmermann has prepared, as they would be
    # during an epoch that uses it, so the chain itself is left as it is.
    counter = CSSLogicalCounter(counter.code)
    counter.count_logical_ops(
        state.current.score.distance,
        method="brouwer_zimmermann",
        max_information_sets=_MAX_INFORMATION_SETS,
    )
    shadows = {
        method: replace(
            state,
            epoch_method=method,
            recent_distances=deque(state.recent_distances, maxlen=_GUESS_WINDOW),
        )
        for method in methods
    }
    spent = dict.fromkeys(methods, 0.0)
    proposals = min(_RACE_PROPOSALS, max(2, config.steps_per_epoch // 8))
    for index in range(proposals + 1):
        # The first proposal only compiles kernels.  The order alternates, against drifting clocks.
        for method in methods if index % 2 == 0 else methods[::-1]:
            rng = np.random.default_rng([state.seed, state.completed_epochs, _RACE_STREAM, index])
            candidate = propose(
                counter, rng, code_type=config.code_type, gate_size=config.gate_size
            )
            start = _clock()
            _score_proposal(shadows[method], candidate)
            if index:
                spent[method] += _clock() - start
        if index >= _RACE_MIN_PROPOSALS and (
            max(spent.values()) > _RACE_SECONDS
            or max(spent.values()) > _RACE_DECISIVE_RATIO * min(spent.values())
        ):
            break
    fastest = min(methods, key=spent.__getitem__)
    if (
        state.epoch_method in spent
        and spent[fastest] > _RACE_SWITCH_FRACTION * spent[state.epoch_method]
    ):
        return state.epoch_method
    return fastest


def _update_optima(state: SearchState, proposal: ScoredCode) -> tuple[bool, bool]:
    """Let any evaluated proposal replace the lowest-energy or greatest-distance code."""
    config = state.config
    improved_energy = energy(proposal.score, config) < energy(state.lowest_energy.score, config)
    if improved_energy:
        state.lowest_energy = proposal
    improved_distance = score_rank(proposal.score, config) > score_rank(
        state.greatest_distance.score, config
    )
    if improved_distance:
        state.greatest_distance = proposal
    return improved_energy, improved_distance


def _get_multiplicity(score: Score, config: SearchConfig) -> int:
    """The number of minimum-weight logical operators that the objective counts.

    Raises:
        ValueError: If the score has no positive distance or no positive count, so that the code
            has no energy or rank.
    """
    if config.objective == "pauli":
        count = score.count_pauli
    else:
        count = score.count_homogeneous
    if score.distance < 1 or count is None or count <= 0:
        raise ValueError(
            "The search requires a positive distance and multiplicity of the objective"
        )
    return count


def _fit_log_temperature(jumps: npt.ArrayLike, target: float) -> float:
    """Fit log T so that the mean uphill acceptance probability equals ``target``.

    The fit is bounded by ``LOG_TEMPERATURE_BOUNDS``.  Jumps that do not raise the energy are
    ignored.  If there are no uphill jumps, return log T=0.  The jumps of rejected uphill proposals
    must still be supplied.
    """
    values = np.asarray(jumps, dtype=float)
    uphill = values[values > 0]
    if not len(uphill):
        return 0.0

    def cost(log_temperature: float) -> float:
        mean = np.exp(-uphill / math.exp(log_temperature)).mean()
        return float((mean - target) ** 2)

    result = minimize_scalar(
        cost, bounds=LOG_TEMPERATURE_BOUNDS, method="bounded", options={"xatol": 1e-10}
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(result.x)


def _accept_move(jump: float, temperature: float, uniform: float) -> bool:
    """Stable Metropolis decision; zero uniforms accept even an underflow-sized probability."""
    return uniform == 0 or math.log(uniform) < min(0, -jump / temperature)
