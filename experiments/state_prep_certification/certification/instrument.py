# SPDX-License-Identifier: Apache-2.0

"""Instrumentation of qLDPC's certification of bounded fault propagation, in this process only.

``Instrumentation`` replaces names in the modules of ``qldpc.circuits.certification`` that use
them, without editing qLDPC, and restores on exit the values that it found on entry:

* ``fault_propagation.find_smallest_violation`` is timed, giving the search time T_search; the
  setup time is T_setup = T - T_search.
* ``_fault_search.CosetLeaderSearch`` becomes a subclass of the class that it names on entry,
  whose ``get_min_weight_bound`` and ``find_first_not_within_max_weight`` time and count the
  outermost calls only, the calls of the fault search; a depth guard ignores calls that one query
  makes to another.  Calls are classified by their place in the search: threshold-1 checks of
  single zero-flag mechanisms (made through ``_check_fault``), threshold-1 classification of
  flagged mechanisms, and batches at threshold m >= 2, whose submitted rows are counted.
* ``FaultLister.__init__`` records the room of the table of syndromes (``_max_syndromes``), and
  ``FaultLister.find_first`` times each fault weight and reports a progress record after it.

With ``shadow=True``, every outermost query is also answered by a connected-cluster
``CosetLeaderSearch`` built once on the same stabilizers, timed separately.  BZ answers each query
first, then CC.  The two answers must agree (``within_max_weight``, or the index of the first
failing row); a disagreement raises ``ShadowDisagreement``.  Each mirrored call is appended to a
JSON-lines file.  The shadow record also names the method to which qLDPC resolves
``method="auto"`` (``auto``), which is the same for every query, so that the ``bz`` and ``cc``
times of the calls file compare auto's choice with the other oracle.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from qldpc.circuits.certification import _fault_search, fault_propagation
from qldpc.codes.distance import CosetLeaderSearch, MinWeightBound, coset_leaders


class ShadowDisagreement(Exception):
    """The BZ and CC oracles answered a query differently."""


class Instrumentation:
    """Counters and times of one certification, filled by the wrappers of ``install``.

    Attributes:
        search_seconds: T_search, the time of ``find_smallest_violation``.
        oracle_setup_seconds: The time to construct the oracle of the fault search (the last one,
            if it constructs several).
        zero_flag_checks: Threshold-1 checks of single zero-flag mechanisms, and their time.
        classification: Threshold-1 classification queries of flagged mechanisms, and their time.
        batches: For each threshold m, the batch calls, submitted rows, and oracle time.
        weights: Per-weight records of ``find_first``: m, T_m, rows, oracle time, cumulative T.
        lister: ``max_syndromes`` of the fault lister.
        shadow: Totals of the CC oracle by threshold, its setup time, and the method to which
            ``method="auto"`` resolves, with ``shadow=True``.
    """

    def __init__(
        self,
        *,
        shadow: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float | None] | None = None,
        calls_file: pathlib.Path | None = None,
    ) -> None:
        self.shadow_enabled = shadow
        self.progress = progress
        self.clock = clock
        self.calls_file = calls_file
        self.search_seconds: float | None = None
        self.zero_flag_checks = {"calls": 0, "seconds": 0.0}
        self.classification = {"calls": 0, "seconds": 0.0}
        self.batches: dict[int, dict[str, Any]] = {}
        self.weights: list[dict[str, Any]] = []
        self.lister: dict[str, Any] = {}
        self.oracle_setup_seconds: float | None = None
        self.shadow: dict[str, Any] = {
            "setup_seconds": None,
            "by_threshold": {},
            "auto": coset_leaders._get_coset_search_method("auto"),
        }
        self._depth = 0
        self._in_check = False
        self._weight_rows = 0
        self._weight_oracle = 0.0

    def summary(self) -> dict[str, Any]:
        """The recorded fields, for a result row."""
        return {
            "T_search": self.search_seconds,
            "oracle_setup_seconds": self.oracle_setup_seconds,
            "zero_flag_checks": self.zero_flag_checks,
            "classification": self.classification,
            "batches": {str(m): value for m, value in sorted(self.batches.items())},
            "per_weight": self.weights,
            "max_syndromes": self.lister.get("max_syndromes"),
            "shadow": self.shadow if self.shadow_enabled else None,
        }

    @contextmanager
    def install(self) -> Iterator[Instrumentation]:
        """Replace the wrapped names for the duration of the block."""
        instrumentation = self
        original_find = fault_propagation.find_smallest_violation
        original_check = _fault_search._check_fault
        original_init = _fault_search.FaultLister.__init__
        original_first = _fault_search.FaultLister.find_first
        original_search = _fault_search.CosetLeaderSearch

        def find_smallest_violation(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_find(*args, **kwargs)
            finally:
                instrumentation.search_seconds = time.perf_counter() - start

        def check_fault(*args: Any, **kwargs: Any) -> Any:
            instrumentation._in_check = True
            try:
                return original_check(*args, **kwargs)
            finally:
                instrumentation._in_check = False

        def lister_init(lister: Any, *args: Any, **kwargs: Any) -> None:
            original_init(lister, *args, **kwargs)
            instrumentation.lister = {"max_syndromes": int(lister._max_syndromes)}

        def find_first(lister: Any, fault_weight: int, *args: Any, **kwargs: Any) -> Any:
            instrumentation._weight_rows = 0
            instrumentation._weight_oracle = 0.0
            start = time.perf_counter()
            result = original_first(lister, fault_weight, *args, **kwargs)
            record = {
                "m": fault_weight,
                "T_m": time.perf_counter() - start,
                "rows": instrumentation._weight_rows,
                "oracle_seconds": instrumentation._weight_oracle,
                "violation": result is not None,
                "T_cumulative": instrumentation.clock() if instrumentation.clock else None,
            }
            instrumentation.weights.append(record)
            if instrumentation.progress is not None:
                instrumentation.progress(record)
            return result

        fault_propagation.find_smallest_violation = find_smallest_violation
        _fault_search._check_fault = check_fault
        _fault_search.FaultLister.__init__ = lister_init  # type: ignore[method-assign, assignment]
        _fault_search.FaultLister.find_first = find_first  # type: ignore[method-assign, assignment]
        _fault_search.CosetLeaderSearch = _instrumented_search(  # type: ignore[misc]
            self, original_search
        )
        try:
            yield self
        finally:
            fault_propagation.find_smallest_violation = original_find
            _fault_search._check_fault = original_check
            _fault_search.FaultLister.__init__ = original_init  # type: ignore[method-assign]
            _fault_search.FaultLister.find_first = original_first  # type: ignore[method-assign]
            _fault_search.CosetLeaderSearch = original_search  # type: ignore[misc]

    def _record_call(self, kind: str, threshold: int, rows: int, seconds: float) -> None:
        """Add one outermost oracle call to the counters."""
        if kind == "zero_flag":
            self.zero_flag_checks["calls"] += 1
            self.zero_flag_checks["seconds"] += seconds
        elif kind == "classification":
            self.classification["calls"] += 1
            self.classification["seconds"] += seconds
        else:
            batch = self.batches.setdefault(threshold, {"calls": 0, "rows": 0, "seconds": 0.0})
            batch["calls"] += 1
            batch["rows"] += rows
            batch["seconds"] += seconds
            self._weight_rows += rows
            self._weight_oracle += seconds

    def _record_shadow(
        self, kind: str, threshold: int, rows: int, bz_seconds: float, cc_seconds: float
    ) -> None:
        """Add one mirrored call to the shadow totals and to the calls file."""
        key = f"{kind}:{threshold}"
        totals = self.shadow["by_threshold"].setdefault(
            key, {"calls": 0, "rows": 0, "bz_seconds": 0.0, "cc_seconds": 0.0}
        )
        totals["calls"] += 1
        totals["rows"] += rows
        totals["bz_seconds"] += bz_seconds
        totals["cc_seconds"] += cc_seconds
        if self.calls_file is not None:
            with open(self.calls_file, "a") as file:
                file.write(
                    json.dumps(
                        {
                            "kind": kind,
                            "m": threshold,
                            "rows": rows,
                            "bz": bz_seconds,
                            "cc": cc_seconds,
                        }
                    )
                    + "\n"
                )


def _instrumented_search(
    instrumentation: Instrumentation, base: type[CosetLeaderSearch]
) -> type[CosetLeaderSearch]:
    """A subclass of an oracle class that reports its outermost calls."""

    class InstrumentedSearch(base):  # type: ignore[valid-type, misc]
        def __init__(self, stabilizers: npt.ArrayLike, **kwargs: Any) -> None:
            start = time.perf_counter()
            super().__init__(stabilizers, **kwargs)
            instrumentation.oracle_setup_seconds = time.perf_counter() - start
            self._shadow: CosetLeaderSearch | None = None
            if instrumentation.shadow_enabled:
                start = time.perf_counter()
                self._shadow = CosetLeaderSearch(
                    stabilizers,
                    css_weight=kwargs.get("css_weight", False),
                    method="connected_cluster",
                )
                instrumentation.shadow["setup_seconds"] = time.perf_counter() - start

        def get_min_weight_bound(self, syndrome: Any, max_weight: int) -> MinWeightBound:
            if instrumentation._depth:
                return super().get_min_weight_bound(syndrome, max_weight)
            kind = "zero_flag" if instrumentation._in_check else "classification"
            instrumentation._depth += 1
            try:
                start = time.perf_counter()
                bound = super().get_min_weight_bound(syndrome, max_weight)
                seconds = time.perf_counter() - start
            finally:
                instrumentation._depth -= 1
            instrumentation._record_call(kind, max_weight, 1, seconds)
            if self._shadow is not None:
                start = time.perf_counter()
                other = self._shadow.get_min_weight_bound(syndrome, max_weight)
                cc_seconds = time.perf_counter() - start
                if bool(other.within_max_weight) != bool(bound.within_max_weight):
                    raise ShadowDisagreement(
                        f"{kind} query at threshold {max_weight}: BZ within "
                        f"{bound.within_max_weight}, CC within {other.within_max_weight}"
                    )
                instrumentation._record_shadow(kind, max_weight, 1, seconds, cc_seconds)
            return bound

        def find_first_not_within_max_weight(
            self, syndromes: npt.NDArray[np.uint64], max_weight: int
        ) -> tuple[int, MinWeightBound | None]:
            if instrumentation._depth:
                return super().find_first_not_within_max_weight(syndromes, max_weight)
            instrumentation._depth += 1
            try:
                start = time.perf_counter()
                answer = super().find_first_not_within_max_weight(syndromes, max_weight)
                seconds = time.perf_counter() - start
            finally:
                instrumentation._depth -= 1
            rows = len(syndromes)
            instrumentation._record_call("batch", max_weight, rows, seconds)
            if self._shadow is not None:
                start = time.perf_counter()
                other = self._shadow.find_first_not_within_max_weight(syndromes, max_weight)
                cc_seconds = time.perf_counter() - start
                if int(other[0]) != int(answer[0]):
                    raise ShadowDisagreement(
                        f"batch at threshold {max_weight} of {rows} rows: BZ first failing "
                        f"row {answer[0]}, CC {other[0]}"
                    )
                instrumentation._record_shadow("batch", max_weight, rows, seconds, cc_seconds)
            return answer

    return InstrumentedSearch
