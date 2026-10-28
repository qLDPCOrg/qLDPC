# SPDX-License-Identifier: Apache-2.0

"""Exact search for an accepted fault of the smallest fault weight that violates the bound.

The search works on the flags and syndrome of each primitive fault, and never sees a circuit.
``fault_propagation.md`` proves that it is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy.typing as npt

from qldpc.codes.distance._bounded_weight_search import MinWeightBound, PauliCosetSearch

from ._fault_enumeration import FaultLister

####################################################################################################
# exact search for a violating fault


class Violation(NamedTuple):
    """An accepted fault that violates the bound.

    Attributes:
        fault_indices: The DEM indices of the primitive faults that make up the fault.
        syndrome: The packed syndrome of the fault's output error.
        bound: The oracle's answer for that syndrome at the fault's fault weight, which gives an
            error with the syndrome and a lower bound above the fault weight.
    """

    fault_indices: tuple[int, ...]
    syndrome: int
    bound: MinWeightBound


def find_smallest_violation(
    flags: Sequence[int],
    syndromes: Sequence[int],
    stabilizers: npt.ArrayLike,
    css_weight: bool,
    max_weight: int | None,
    memory_limit: int = 2**30,  # 1 GiB
) -> Violation | None:
    """Find a violating accepted fault of the smallest fault weight, or None if there is none.

    The search keeps one primitive fault for each distinct nonzero signature, and limits the fault
    weight to ``max_weight`` and to what an irreducible accepted fault can have.  Then it runs
    ``_search_irreducible_faults``.
    """
    # A primitive fault with zero flags and zero syndrome does nothing, and two primitive faults
    # with equal (flags, syndrome) signatures cancel.  Keeping one primitive fault for each distinct
    # nonzero signature cannot hide a violation, and each signature keeps the index of its first
    # primitive fault.
    signatures: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(zip(flags, syndromes, strict=True)):
        if any(signature):
            signatures.setdefault(signature, index)

    # An irreducible accepted fault has fault weight at most one more than the rank of the flags.
    flag_rank = _get_binary_rank([flag for flag, _ in signatures])
    max_fault_weight = flag_rank + 1 if max_weight is None else min(max_weight, flag_rank + 1)

    oracle = PauliCosetSearch(stabilizers, css_weight=css_weight)
    return _search_irreducible_faults(signatures, max_fault_weight, oracle, memory_limit)


def _search_irreducible_faults(
    signatures: dict[tuple[int, int], int],
    max_fault_weight: int,
    oracle: PauliCosetSearch,
    memory_limit: int = 2**30,  # 1 GiB
) -> Violation | None:
    """Check the irreducible accepted faults that can violate the bound, by increasing fault weight.

    The signatures map the (flags, syndrome) of each kept primitive fault to its DEM index, and
    ``max_fault_weight`` is the largest fault weight to check.  Return a violating fault of the
    smallest fault weight, or None if there is none.
    """
    # A primitive fault that fires no post-selection detector is an irreducible accepted fault by
    # itself, and no larger irreducible accepted fault contains it.  Check these first.
    flagged: list[tuple[int, int, int]] = []
    for (flag, syndrome), index in signatures.items():
        if flag:
            flagged.append((flag, syndrome, index))
            continue
        failure = _check_fault((index,), syndrome, oracle)
        if failure is not None:
            return failure

    # A primitive fault is heavy if its own error has no equivalent error of Pauli weight at most
    # one, and light otherwise.  A fault that contains only light primitive faults cannot violate
    # the bound, so without a heavy primitive fault there is nothing more to check.
    heavy: list[tuple[int, int, int, int]] = []
    light: list[tuple[int, int, int, int]] = []
    for flag, syndrome, index in flagged:
        bound = oracle.get_min_weight_bound(syndrome, 1)
        primitive = (flag, syndrome, index, bound.representative)
        (light if bound.within_max_weight else heavy).append(primitive)
    if not heavy:
        return None

    # Give each primitive fault an error with its syndrome, of low Pauli weight so that the search
    # can skip more faults.  The error of a light primitive fault has Pauli weight at most one.  A
    # heavy primitive fault whose syndrome is the sum of those of two light primitive faults gets
    # the product of their errors, which has Pauli weight at most two.
    light_errors = {syndrome: error for _, syndrome, _, error in light}
    for jj, (flag, syndrome, index, _) in enumerate(heavy):
        for other, error in light_errors.items():
            if syndrome ^ other in light_errors:
                heavy[jj] = (flag, syndrome, index, error ^ light_errors[syndrome ^ other])
                break

    # Order the primitive faults with the heavy ones first, and each group in DEM order, so that the
    # faults to check are the ones that start with a heavy primitive fault.
    primitives = heavy + light

    # The oracle needs only small tables, so the lister may use the whole memory limit.
    lister = FaultLister(
        [flag for flag, _, _, _ in primitives],
        [syndrome for _, syndrome, _, _ in primitives],
        [representative for _, _, _, representative in primitives],
        oracle.num_qubits,
        oracle.css_weight,
        len(heavy),
        memory_limit,
    )

    # Check the faults by increasing fault weight.
    for fault_weight in range(2, max_fault_weight + 1):
        violation = lister.find_first(fault_weight, oracle.find_first_not_within_max_weight)
        if violation is not None:
            positions, syndrome = violation
            indices = tuple(primitives[jj][2] for jj in positions)
            bound = oracle.get_min_weight_bound(syndrome, fault_weight)
            return Violation(indices, syndrome, bound)
    return None


def _check_fault(
    indices: tuple[int, ...], syndrome: int, oracle: PauliCosetSearch
) -> Violation | None:
    """Return the fault with these indices and syndrome if it violates the bound."""
    bound = oracle.get_min_weight_bound(syndrome, len(indices))
    if bound.within_max_weight:
        return None
    return Violation(indices, syndrome, bound)


def _get_binary_rank(columns: Sequence[int]) -> int:
    """Get the GF(2) rank of binary columns that are packed into integers."""
    # The basis holds columns with distinct leading bits, largest first.  Reducing a column by each
    # basis column in turn clears these leading bits from it, so a nonzero result is independent of
    # the basis.
    basis: list[int] = []
    for column in columns:
        for row in basis:
            column = min(column, column ^ row)
        if column:
            basis.append(column)
            basis.sort(reverse=True)
    return len(basis)
