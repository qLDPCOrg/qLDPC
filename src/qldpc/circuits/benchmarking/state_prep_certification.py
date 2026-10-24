# SPDX-License-Identifier: Apache-2.0

"""Exact certification of bounded fault spread in state-preparation circuits.

A *fault* is one error mechanism of the circuit's detector error model (DEM), and a *pattern* is a
set ``f`` of faults, of size ``|f|``.  Write ``P f`` for the post-selected detectors that the
pattern flips, and ``S f`` for the syndrome that it leaves on the prepared state, measured against
all stabilizers of that state, logical ones included.  Let ``mu(s)`` be the smallest weight of an
error with syndrome ``s``.  The certificate proves ``mu(S f) <= |f|`` for every pattern with
``P f = 0`` and ``1 <= |f| <= max_faults``.  The argument rests on these facts:

- Syndromes identify equivalent errors.  Two errors with the same syndrome differ by an operator
  that commutes with all ``n`` independent stabilizers of the ``n``-qubit state, and every such
  operator is itself a stabilizer (up to a phase).  So the two errors leave the same state, and
  ``mu(S f)`` is the weight of the lightest error equivalent to the one that the faults cause.
- Weight is subadditive: a product of two errors is no heavier than the sum of their weights, so
  ``mu(s + t) <= mu(s) + mu(t)``.  For physical weight, the product acts on no more qubits than the
  two errors together.  For CSS weight, ``max(a+c, b+d) <= max(a, b) + max(c, d)``.
- It is enough to check minimal patterns.  A pattern with ``P f = 0`` is *accepted*.  Removing an
  accepted subset from an accepted pattern leaves an accepted pattern, so every accepted pattern
  splits into disjoint *minimal* ones, which contain no smaller accepted pattern.  If each minimal
  pattern meets the bound, subadditivity gives the bound for their union.  A pattern on the faults
  ``J`` is minimal exactly when ``P[:, J]`` has a null space of dimension one, so
  ``|J| <= rank(P) + 1``.  A fault that flips no post-selected detector is minimal only by itself.
- Faults with identical (flag, syndrome) columns cancel in pairs, and a fault with no flags and no
  syndrome does nothing.  Dropping them keeps the symptoms and lowers the fault count, so it cannot
  hide a violation, and every remaining fault still names an original DEM mechanism.
- If the distinct nonzero flag columns are linearly independent, an accepted pattern uses each flag
  column an even number of times.  The minimal flagged patterns are then pairs of faults with equal
  flags, for any rank and any budget.
- If every fault alone leaves an error that is equivalent to one of weight at most one, then
  subadditivity gives ``mu(S f) <= |f|`` for every pattern, so no pattern needs checking.

Two complete searches enumerate the remaining minimal patterns, one size at a time from two faults
up.  Brouwer--Zimmermann (BZ) coverage
visits every vector in the kernel of ``P`` up to a weight bound, whatever test is applied to it
afterwards, and a rank check then discards the non-minimal ones.  Independent-prefix search builds
sets of independent columns in increasing order.  Sorted by index, a minimal pattern is an
independent prefix plus a last column whose flags equal the XOR of the prefix's flags, and every
such pair is a minimal pattern.  Requiring the closing column to have the largest index lists each
minimal pattern once, and a prefix that already contains a dependency never extends to one.  A cost
heuristic chooses between the two searches and affects speed only, not the result.  A violating
pattern with the fewest faults is minimal, since otherwise one of the minimal patterns that it
splits into would violate the bound with fewer faults.  Checking sizes in increasing order therefore
finds such a pattern first.  One shared exact oracle, ``PauliCosetSearch``, computes ``mu`` for
every pattern.
"""

from __future__ import annotations

import math
import numbers
from bisect import bisect_left
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import stim

from qldpc.codes.distance._bounded_weight_search import BoundedKernelSearch, PauliCosetSearch

from ..bookkeeping import DetectorRecord
from ..common import get_pauli_product_measurements, get_unaddressed_measurements
from ..encoding import get_state_stabilizers
from .state_prep_diagnostics import _get_selected_detectors

# Avoid switching searches for small BZ jobs or marginal estimated savings.
_MIN_COEFFICIENT_SUPPORTS = 1024
_PREFIX_ADVANTAGE = 2

####################################################################################################
# state-preparation certification


def certify_state_prep_spread(
    circuit: stim.Circuit,
    *,
    max_faults: int,
    output_qubits: Sequence[int] | None = None,
    add_flags: bool = False,
    post_select: bool | Collection[int] | Literal["flags"] = False,
    css_weight: bool | None = None,
) -> StatePrepSpreadCertificate:
    r"""Certify bounded error spread of a noisy Clifford state preparation.

    Check that, for every set of one through ``max_faults`` faults that passes post-selection, the
    error left on the output is equivalent to an error of weight at most the number of faults.  Two
    errors are equivalent if they leave the prepared state unchanged, that is, if they differ by a
    stabilizer of that state (logical stabilizers included).  The weight is either the physical
    weight, the number of qubits that an error acts on, or the CSS weight ``max(|x|, |z|)``, where
    ``x`` and ``z`` are the X and Z parts of the error.  The minimum is over whole errors, not
    separately over the X and Z parts.

    A fault is one error mechanism of the circuit's detector error model (DEM), and every fault
    costs one.  Mechanisms are treated as independent, even when they come from one noise channel
    and could not physically occur together.  A mechanism that flips several detectors at once stays
    one fault.  Stim converts the circuit with ``approximate_disjoint_errors=True``; probabilities
    play no role, and nothing is claimed about the sampling distribution.

    The output qubits must end in one definite stabilizer state, the same for every measurement
    outcome.  Every detector and observable in the circuit, including those you do not post-select
    on and any added flag detectors, must be deterministic.  Branches that depend on random
    measurement outcomes are not supported.  Stabilizer signs and qubit labels are kept as given,
    and the diagnostics that the function appends to read out the output state are noiseless.

    With ``css_weight=None``, CSS weight is chosen when the prepared state is a CSS state in the
    physical Pauli basis, that is, when its stabilizer group is generated by pure-X and pure-Z
    operators.  A state that becomes CSS only after local Clifford gates counts as non-CSS.  A CSS
    code can also prepare a non-CSS state, such as a logical Y eigenstate; if you use that state
    sector by sector, pass ``css_weight=True`` yourself.

    Passing is a sufficient condition on the prepared state, not a proof that a protocol using it is
    fault tolerant.  It helps only if the next gadget still works when an input error of weight
    ``r`` is combined with further faults, as long as ``r`` plus the number of further faults is
    within the gadget's limit.  A result for CSS weight also needs that gadget to keep X and Z
    errors separate, which you must justify when passing ``css_weight=True``.  The certificate says
    nothing about the acceptance probability, and it is not a necessary condition for fault
    tolerance.  The search is exact and can take exponential time.  See the module docstring for the
    proof.

    Args:
        circuit: Noisy Stim Clifford circuit starting in Stim's all-zero state.

    Keyword args:
        max_faults: Largest number of faults to consider, a positive integer.
        output_qubits: Distinct indices of the output qubits, at least one, in the desired Pauli
            frame.  By default, every qubit that the circuit acts on and never measures, in
            increasing order.
        add_flags: Whether to add a flag detector for each measurement that no detector or
            observable addresses, as in get_state_prep_diagnostic_circuit.
        post_select: The detectors to post-select on, out of the selectable detectors: those
            already in the circuit followed by any added flag detectors.  True selects all of
            them, False (the default) selects none, a collection of integers selects the
            detectors with those indices (possibly negative, to count back from the last of them),
            and "flags" selects exactly the added flag detectors, which requires ``add_flags=True``.
            A pattern is accepted only if none of these detectors fire, relative to their ideal
            values.  The diagnostics appended to read out the output state never reject.
        css_weight: True for CSS weight, False for physical weight, or None to choose CSS weight
            when the prepared state is a CSS state.

    Returns:
        A certificate with the decision, the DEM, and the weight that was used.  A failure comes
        with a counterexample: a pattern that passes post-selection, and a proved lower bound on
        the weight of the output error that exceeds the number of faults.  No violating pattern has
        fewer faults.  Fault indices count the error instructions of the returned DEM.

    Raises:
        ValueError: For invalid inputs, an output that is not one definite pure state,
            nondeterministic detectors or observables, or a circuit that Stim cannot convert to a
            DEM.
    """
    if css_weight is not None and not isinstance(css_weight, bool):
        raise ValueError("css_weight must be a bool or None")
    if (
        isinstance(max_faults, bool)
        or not isinstance(max_faults, numbers.Integral)
        or max_faults < 1
    ):
        raise ValueError("max_faults must be a positive integer")
    if isinstance(post_select, str) and not (post_select == "flags" and add_flags):
        raise ValueError('post_select="flags" selects the added flag detectors: use add_flags=True')
    if output_qubits is None:
        output_qubits = _get_unmeasured_qubits(circuit)
        if not output_qubits:
            raise ValueError("The circuit has no unmeasured qubits to use as output_qubits")
    qubits = tuple(output_qubits)
    _validate_indices(qubits, circuit.num_qubits, "output_qubits", nonempty=True)

    # Flag detectors come after the detectors that the circuit already has.
    detector_record = DetectorRecord({"prep": range(circuit.num_detectors)})
    flag_detectors = stim.Circuit()
    if add_flags:
        for measurement in get_unaddressed_measurements(circuit):
            target = stim.target_rec(measurement - circuit.num_measurements)
            flag_detectors.append("DETECTOR", [target])
    detector_record.append({"flags": range(flag_detectors.num_detectors)})
    selected = _resolve_post_selection(post_select, detector_record)
    circuit = circuit + flag_detectors

    # Complete state stabilizers, including logical stabilizers, identify output-error classes.
    stabilizers = get_state_stabilizers(circuit, qubits)
    if len(stabilizers) != len(qubits):
        raise ValueError("Selected outputs must have a definite pure, complete stabilizer state")
    matrix = np.array(
        [np.concatenate(stabilizer.to_numpy()) for stabilizer in stabilizers], dtype=np.uint8
    )
    resolved_css_weight = _is_css_state(matrix) if css_weight is None else css_weight
    dem, flags, syndromes = _get_fault_maps(circuit, stabilizers, qubits, selected)

    statistics = StatePrepSpreadStatistics(mechanisms=len(flags))
    counterexample = _certify_maps(
        flags, syndromes, matrix, resolved_css_weight, int(max_faults), statistics
    )
    return StatePrepSpreadCertificate(
        counterexample is None,
        int(max_faults),
        tuple(map(int, qubits)),
        tuple(map(str, stabilizers)),
        dem,
        counterexample,
        statistics,
        resolved_css_weight,
    )


####################################################################################################
# certificate results and search statistics


@dataclass(frozen=True)
class StatePrepSpreadCounterexample:
    """A set of faults that passes post-selection but spreads into a heavier output error.

    ``fault_indices`` number the error instructions of the certificate's DEM, counting only error
    instructions.  ``output_syndrome`` lists the syndrome bits in the order of
    ``target_stabilizers``.  ``effective_weight_lower_bound`` is a proved lower bound on the weight
    of any equivalent error, and it exceeds the number of faults.  ``output_pauli`` is one Pauli
    with that syndrome, written as a Stim string over the output qubits in the caller's order.  It
    is not necessarily the lightest such Pauli or the error that the faults actually cause.  The set
    has the fewest faults of any violating set.
    """

    fault_indices: tuple[int, ...]
    output_syndrome: tuple[int, ...]
    effective_weight_lower_bound: int
    output_pauli: str

    @property
    def fault_count(self) -> int:
        """Number of original DEM mechanisms in the witness."""
        return len(self.fault_indices)


@dataclass
class StatePrepSpreadStatistics:
    """Deterministic search-effort counters for one certification."""

    mechanisms: int = 0
    """Unit-cost error mechanisms in the diagnostic DEM."""

    flag_rank: int = 0
    """GF(2) rank of the selected rejection flags."""

    effective_budget: int = 0
    """Largest pattern size searched.

    This is the fault budget capped at ``flag_rank + 1``, at 2 for independent flag types, and at 1
    when no single fault has output weight above one.
    """

    search_method: Literal["singletons", "brouwer_zimmermann", "independent_prefix"] = "singletons"
    """Flagged-pattern search, or ``singletons`` if no flagged patterns are searched."""

    generated: int = 0
    """Flag-free singletons plus search-dependent work, not all fault combinations.

    BZ counts coefficient supports; prefix search counts attempted extensions (including dependent
    ones) and emitted closing supports.  Both accumulate over the sizes searched.
    """

    minimal_candidates: int = 0
    """Minimal accepted patterns checked for output weight, counting repeats."""


@dataclass(frozen=True)
class StatePrepSpreadCertificate:
    """Completed spread decision with its model and witness frame.

    Failure always includes a counterexample; incomplete searches return no result.
    """

    certified: bool
    """Whether accepted patterns of one through ``max_faults`` faults meet the spread bound."""

    max_faults: int
    """Requested unit-cost fault budget."""

    output_qubits: tuple[int, ...]
    """Physical output indices in target-stabilizer and witness-Pauli order."""

    target_stabilizers: tuple[str, ...]
    """Signed Stim strings for the complete state stabilizers, one per output qubit."""

    detector_error_model: stim.DetectorErrorModel
    """Flattened, undecomposed DEM including any added flags and noiseless output diagnostics."""

    counterexample: StatePrepSpreadCounterexample | None
    """Accepted violating pattern on failure, or None on success."""

    statistics: StatePrepSpreadStatistics
    """Deterministic search-effort counters."""

    css_weight: bool
    """Resolved metric: True for CSS-sector weight, False for physical Pauli support."""


####################################################################################################
# helper methods


def _get_fault_maps(
    circuit: stim.Circuit,
    stabilizers: Sequence[stim.PauliString],
    qubits: Sequence[int],
    selected: Sequence[int],
) -> tuple[stim.DetectorErrorModel, list[int], list[int]]:
    """Compile the undecomposed DEM with packed flag and syndrome columns.

    Lists follow DEM error-instruction order; bits follow selected-detector and stabilizer order.
    """
    # Appended detectors diagnose the output state; only selected original detectors reject.
    diagnostics = get_pauli_product_measurements(stabilizers, qubits=qubits)
    for jj in range(len(stabilizers)):
        diagnostics.append("DETECTOR", [stim.target_rec(jj - len(stabilizers))])
    diagnostic_circuit = circuit + diagnostics

    # Exact anticommutation analysis validates determinism without sampling.
    diagnostic_circuit.without_noise().detector_error_model()
    dem = diagnostic_circuit.detector_error_model(approximate_disjoint_errors=True).flattened()

    selected_map = {detector: jj for jj, detector in enumerate(selected)}
    flags: list[int] = []
    syndromes: list[int] = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        flag = syndrome = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detector = target.val
                if detector in selected_map:
                    flag ^= 1 << selected_map[detector]
                if detector >= circuit.num_detectors:
                    syndrome ^= 1 << (detector - circuit.num_detectors)
        flags.append(flag)
        syndromes.append(syndrome)
    return dem, flags, syndromes


def _get_unmeasured_qubits(circuit: stim.Circuit) -> tuple[int, ...]:
    """Qubits that some instruction acts on and no measurement targets, in increasing order."""
    touched: set[int] = set()
    measured: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS"):
            continue
        qubits = {tt.qubit_value for tt in instruction.targets_copy() if tt.qubit_value is not None}
        touched |= qubits
        if stim.gate_data(instruction.name).produces_measurements:
            measured |= qubits
    return tuple(sorted(touched - measured))


def _resolve_post_selection(
    post_select: bool | Collection[int] | Literal["flags"], detector_record: DetectorRecord
) -> tuple[int, ...]:
    """Distinct detector indices to post-select on, with the semantics of the diagnostic tasks."""
    if not isinstance(post_select, (bool, str)):
        post_select = tuple(post_select)
        if any(isinstance(jj, bool) or not isinstance(jj, numbers.Integral) for jj in post_select):
            raise ValueError("post_select must contain integer detector indices")
    selected = tuple(int(jj) for jj in _get_selected_detectors(post_select, detector_record))
    if len(set(selected)) != len(selected):
        raise ValueError("post_select must not repeat a detector")
    return selected


def _is_css_state(stabilizers: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> bool:
    """Test whether a complete rank-n stabilizer space is CSS in the physical basis."""
    n = len(stabilizers)
    # The pure-X/Z intersections have dimensions n-rank(W_Z) and n-rank(W_X).  They span W exactly
    # when rank(W_X)+rank(W_Z)=n, regardless of the generator basis.
    ranks = [
        _binary_rank([int.from_bytes(np.packbits(row).tobytes(), "little") for row in sector])
        for sector in (stabilizers[:, :n], stabilizers[:, n:])
    ]
    return sum(ranks) == n


def _validate_indices(
    indices: tuple[int, ...], size: int, name: str, *, nonempty: bool = False
) -> None:
    if (
        (nonempty and not indices)
        or any(isinstance(jj, bool) or not isinstance(jj, numbers.Integral) for jj in indices)
        or any(not 0 <= jj < size for jj in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(f"{name} must contain distinct in-range integer indices")


def _certify_maps(
    flags: Sequence[int],
    syndromes: Sequence[int],
    stabilizers: npt.ArrayLike,
    css_weight: bool,
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Merge joint columns and search minimal accepted supports."""
    # Equal joint columns cancel without changing acceptance or syndrome; removing pairs cannot hide
    # violations.  Keep original DEM indices for witnesses.
    representatives = {
        (flag, syndrome): jj
        for jj, (flag, syndrome) in enumerate(zip(flags, syndromes, strict=True))
        if flag or syndrome
    }
    statistics.flag_rank = _binary_rank([flag for flag, _ in representatives])

    # A minimal accepted support has nullity one, hence at most rank(P)+1 mechanisms.  Independent
    # flag types require even multiplicities: only equal-flag pairs are minimal.
    limit = min(budget, statistics.flag_rank + 1)
    independent = len({flag for flag, _ in representatives if flag}) == statistics.flag_rank
    statistics.effective_budget = min(limit, 2) if independent else limit

    # No pattern is checked against a weight above the effective budget.
    oracle = PauliCosetSearch(
        stabilizers,
        metric="css" if css_weight else "pauli",
        max_weight=statistics.effective_budget,
    )
    return _search_minimal_patterns(representatives, oracle, independent, statistics)


def _search_minimal_patterns(
    representatives: dict[tuple[int, int], int],
    oracle: PauliCosetSearch,
    independent: bool,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Check minimal patterns by increasing size, so that any violation has the fewest faults."""
    flagged: list[tuple[int, int, int]] = []

    # A flag-free mechanism is minimal by itself and cannot belong to a larger minimal pattern.
    for (flag, syndrome), original in representatives.items():
        if flag:
            flagged.append((flag, syndrome, original))
        else:
            statistics.generated += 1
            failure = _check_fault_pattern((original,), syndrome, oracle, statistics)
            if failure is not None:
                return failure
    limit = statistics.effective_budget
    if limit < 2 or not flagged:
        return None

    # By subadditivity, patterns of faults that each have weight at most one meet the bound.
    if all(oracle.decide(syndrome, 1).within_bound for _, syndrome, _ in flagged):
        statistics.effective_budget = 1
        return None

    use_prefix = independent or _use_prefix_search(len(flagged), statistics.flag_rank, limit)
    statistics.search_method = "independent_prefix" if use_prefix else "brouwer_zimmermann"
    iterate = _iter_minimal_patterns if use_prefix else _iter_kernel_patterns
    for size in range(2, limit + 1):
        for indices, syndrome in iterate(flagged, size, statistics):
            failure = _check_fault_pattern(indices, syndrome, oracle, statistics)
            if failure is not None:
                return failure
    return None


def _check_fault_pattern(
    indices: tuple[int, ...],
    syndrome: int,
    oracle: PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Check a pattern or return its violation in the original DEM and output frames."""
    statistics.minimal_candidates += 1
    # Compare with this pattern's own fault count, not the maximum allowed search budget.
    decision = oracle.decide(syndrome, len(indices))
    if decision.within_bound:
        return None

    n = oracle.num_qubits
    pauli = stim.PauliString.from_numpy(
        xs=np.array([(decision.representative >> jj) & 1 for jj in range(n)], dtype=bool),
        zs=np.array([(decision.representative >> (jj + n)) & 1 for jj in range(n)], dtype=bool),
    )
    return StatePrepSpreadCounterexample(
        indices,
        tuple((syndrome >> jj) & 1 for jj in range(n)),
        decision.lower_bound,
        str(pauli),
    )


def _use_prefix_search(num_columns: int, rank: int, budget: int) -> bool:
    """Choose between complete searches by estimated work, not correctness."""
    dimension = num_columns - rank
    if dimension <= rank:
        return False

    # Prefix XOR determines the closing column, saving one support layer.
    coefficient_count = sum(
        math.comb(dimension, depth) for depth in range(1, min(dimension, budget) + 1)
    )
    prefix_count = sum(
        math.comb(num_columns, depth) for depth in range(1, min(num_columns, budget - 1) + 1)
    )
    return (
        coefficient_count >= _MIN_COEFFICIENT_SUPPORTS
        and coefficient_count >= _PREFIX_ADVANTAGE * prefix_count
    )


def _iter_kernel_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    size: int,
    statistics: StatePrepSpreadStatistics,
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield the minimal patterns of a given size from BZ coverage of the rejection kernel.

    Each pattern is a tuple of original DEM indices and the XOR of the syndromes of its columns.  A
    pattern that several information sets reach is yielded once for each of them.
    """
    checks = np.array(
        [
            [(flag >> jj) & 1 for flag, _, _ in mechanisms]
            for jj in range(max(flag for flag, _, _ in mechanisms).bit_length())
        ],
        dtype=np.uint8,
    )
    outer = BoundedKernelSearch(checks, size)
    counted = 0
    for word in outer.words():
        support = tuple(jj for jj in range(len(mechanisms)) if (word >> jj) & 1)

        # Smaller words were checked at smaller sizes.  Nullity one excludes words that split into
        # smaller accepted patterns.
        if len(support) != size or _binary_rank([mechanisms[jj][0] for jj in support]) != size - 1:
            continue
        syndrome = 0
        for jj in support:
            syndrome ^= mechanisms[jj][1]
        statistics.generated += outer.generated - counted
        counted = outer.generated
        yield tuple(mechanisms[jj][2] for jj in support), syndrome
    statistics.generated += outer.generated - counted


def _iter_minimal_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    size: int,
    statistics: StatePrepSpreadStatistics,
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield minimal patterns of a given size of nonzero-flag (flag, syndrome, original) columns.

    Prefixes are independent and increase in internal, not original DEM, order.  The largest
    internal index uniquely closes the pattern; carrying XORs avoids reconstructing full words.
    """
    groups: dict[int, list[int]] = {}
    for index, (flag, _, _) in enumerate(mechanisms):
        groups.setdefault(flag, []).append(index)

    def visit(
        support: tuple[int, ...],
        start: int,
        flag_sum: int,
        syndrome: int,
        basis: tuple[int, ...],
    ) -> Iterator[tuple[tuple[int, ...], int]]:
        if len(support) == size - 1:
            matches = groups.get(flag_sum, [])
            for position in range(bisect_left(matches, start), len(matches)):
                _, closing_syndrome, original = mechanisms[matches[position]]
                statistics.generated += 1
                yield support + (original,), syndrome ^ closing_syndrome
            return
        for index in range(start, len(mechanisms)):
            statistics.generated += 1
            flag, column_syndrome, original = mechanisms[index]
            # Descending leading bits give an echelon basis; a zero remainder means dependence,
            # which cannot extend to a minimal accepted support.
            reduced = flag
            for row in basis:
                reduced = min(reduced, reduced ^ row)
            if reduced:
                yield from visit(
                    support + (original,),
                    index + 1,
                    flag_sum ^ flag,
                    syndrome ^ column_syndrome,
                    tuple(sorted((*basis, reduced), reverse=True)),
                )

    yield from visit((), 0, 0, 0, ())


def _binary_rank(columns: Sequence[int]) -> int:
    """Find the rank of packed binary columns by leading-bit elimination."""
    pivots: dict[int, int] = {}
    for column in columns:
        while column:
            pivot = column.bit_length()
            if pivot in pivots:
                column ^= pivots[pivot]
            else:
                pivots[pivot] = column
                break
    return len(pivots)
