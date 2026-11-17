# SPDX-License-Identifier: Apache-2.0

"""Trace the decisions and stage times of the exact searches of a workload, or compare two traces.

The script wraps functions of ``qldpc.codes.distance`` in this process only, warms up as
``trial.py`` does, and runs a ``distance``, ``hadamard``, ``count``, ``ccdistance``, or ``mcount``
workload once under the given tuning, on one thread unless the tuning sets ``num_threads``.  It
cannot trace ``ccount`` workloads, whose compiled setup calls the stages past the wrappers.  For a
Brouwer-Zimmermann (BZ) search, it records each family of information sets, certificate, linear
program, consideration of a switch to automorphism images, prediction of the cost of completion
tables, comparison of a predicted cost with ``completion_bound_max_cost_percent``, and layer, and
the first test of each number of rows and weight for tables.  For a connected-cluster search, it
records each weight limit, and for a count by the MacWilliams method, each walk and test.  The
wrapper of each event says what it records.  Each layer and weight limit numbers its sector, such as
the X-type logical operators of a CSS code, in a ``sector`` field (see ``wrap_sectors``).  The
script prints the events, and with ``--output`` saves them as JSON with the tuning and a fingerprint
of the qLDPC source.  With ``--setup-only``, it ends the workload before the first layer of a BZ
distance search, after its setup; workloads of other kinds run in full.

With ``--compare``, the script instead says whether two saved traces of one workload, such as traces
under two tunings or of two checkouts, make the same decisions, and prints the first that differs,
unless one trace ends before it.  If the tunings differ only in thresholds and every decision is the
same, so is the search, apart from the time of the decisions themselves.  The comparison sees the
number of sets of each family but not the sets, not the states that a connected-cluster search
visits within a weight limit, and not the sectors of layers and weight limits.

The script refuses to run if this process imports qLDPC from elsewhere; to trace another checkout,
run the script of that checkout with its own Python.

Examples:

    python experiments/distance_tuning/trace.py distance/qt7/X --output a.json
    python experiments/distance_tuning/trace.py distance/qt7/X --tuning '{"lp_minimum_work": 0}' \\
        --output b.json
    python experiments/distance_tuning/trace.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qldpc.codes.distance import CSSLogicalCounter, _tuning, core, logical_counts
from qldpc.codes.distance._brouwer_zimmermann import affine_search, distance
from qldpc.codes.distance._connected_cluster import search as support_search
from qldpc.codes.distance._counting import macwilliams
from qldpc.codes.distance._information_sets import (
    automorphism_families,
    basis_families,
    certificates,
    completion_bounds,
    matroid_partition,
)

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import trial" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import sweep, trial, workloads
else:
    import sweep
    import trial
    import workloads

EVENTS: list[dict[str, Any]] = []

# The statistics of the layer that is being scanned, or None outside a layer.
CURRENT_LAYER: dict[str, Any] | None = None

# The time at which the traced workload started.
START = 0.0

# Whether to stop before the first layer of the first search.
SETUP_ONLY = False

# The rows and weight of the latest prediction of the cost of completion tables.
LAST_PREDICTION: dict[str, Any] = {"rows": None, "weight": None}

# The seconds of asking for automorphisms in the current weight limit of a connected-cluster search.
CURRENT_ASK: dict[str, Any] = {}

# The index of the sector of each search, by the id of the search (see wrap_sectors).
SECTOR_INDICES: dict[int, int | None] = {}

# The index of the sector whose search is prepared next, within a call that wrap_sectors wraps, or
# None outside such calls.
NEXT_SECTOR_INDEX: int | None = None

# The coverages that matroid partitioning tries within a call of get_partitioned_family, or None
# outside such calls.
PARTITION_COVERAGES: list[int] | None = None

# The kinds of workloads that the script can trace.
TRACED_KINDS = ("distance", "hadamard", "count", "ccdistance", "mcount")

# The stage that an overlapping family event records for each check of the library.
OVERLAPPING_STAGES = {
    basis_families._OverlappingCheck.NOTHING_TO_EXTEND: "no open sets",
    basis_families._OverlappingCheck.BELOW_MINIMUM_WORK: "below minimum work",
    basis_families._OverlappingCheck.CANNOT_DO_LESS_WORK: "cannot do less work",
    basis_families._OverlappingCheck.NONE: "built",
}


class SetupDone(Exception):
    """Raised before the first layer of a search whose trace stops after its setup."""


def main() -> None:
    """Trace a workload, and print and save the trace, or compare two traces."""
    args = parse_args()
    if args.compare is not None:
        compare(*(json.loads(path.read_text(encoding="utf-8")) for path in args.compare))
        return
    if args.workload.split("/")[0] not in TRACED_KINDS:
        raise ValueError(f"Only workloads of the kinds {', '.join(TRACED_KINDS)} can be traced")
    sweep.check_local_import()
    trial.check_workload(args.workload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    source = sweep.get_source_fingerprint()
    with _tuning.use_tuning(**{"num_threads": 1} | json.loads(args.tuning)) as tuning:
        trial.warm_up(args.workload.split("/")[0])
        install_wrappers()
        EVENTS.clear()
        global START, SETUP_ONLY
        SETUP_ONLY = args.setup_only
        start = START = time.perf_counter()
        try:
            result = workloads.run_workload(args.workload)
        except SetupDone:
            result = "setup only"
        seconds = time.perf_counter() - start
    print_trace(result, seconds)
    if args.output is not None:
        record = {
            "workload": args.workload,
            "tuning": json.loads(args.tuning),
            "full_tuning": dataclasses.asdict(tuning),
            "source": source,
            "result": result,
            "seconds": seconds,
            "events": EVENTS,
        }
        args.output.write_text(json.dumps(record, indent=1), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "workload", nargs="?", help=f"a workload of the kinds {', '.join(TRACED_KINDS)}"
    )
    parser.add_argument(
        "--tuning",
        default="{}",
        help="a JSON object of changes to the fields of SearchTuning (default: no changes)",
    )
    parser.add_argument("--output", type=pathlib.Path, help="a JSON file for the events")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="stop a BZ distance search before its first layer, after its setup",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=pathlib.Path,
        metavar=("FIRST", "SECOND"),
        help="compare the decisions of two saved traces of one workload, instead of tracing",
    )
    args = parser.parse_args()
    if (args.workload is None) == (args.compare is None):
        parser.error("give either a workload or --compare")
    return args


####################################################################################################
# wrappers


def install_wrappers() -> None:
    """Wrap the stages of the exact searches, in this process, to record events."""
    search_class = distance._SectorSearch
    tables_class = completion_bounds.CompletionTables
    patches: list[tuple[object, str, Callable[[Any], Any]]] = [
        (basis_families, f"get_{kind}_family", functools.partial(wrap_family, kind))
        for kind in ("disjoint", "overlapping", "partitioned")
    ]
    patches += [
        (matroid_partition, "get_information_sets_with_coverage", wrap_partition),
        (basis_families.BasisFamily, "get_search_certificate", wrap_certificate),
        (search_class, "advance", wrap_advance),
        (search_class, "_switch_to_automorphism_family", wrap_switch),
        (search_class, "_scan_set", wrap_scan_set_method),
        (completion_bounds, "select_completion_tables", wrap_select),
        (completion_bounds, "predict_completion_cost", wrap_predict),
        (tables_class, "build", lambda build: staticmethod(wrap_build(build))),
        (affine_search, "scan_set", wrap_scan_set),
        (completion_bounds, "considers_tables", wrap_considers),
        (completion_bounds, "is_within_cost_limit", wrap_limit),
        (certificates, "get_lp_lower_bound", wrap_linear_program),
        (core, "get_distance_brouwer_zimmermann", wrap_sectors),
        (core, "get_distance_connected_cluster", wrap_sectors),
        (logical_counts, "get_distance_brouwer_zimmermann", wrap_sectors),
        (distance, "_prepare_sector_search", wrap_prepare),
        (support_search.SupportSearch, "__init__", wrap_support_search),
        (support_search.SupportSearch, "find_witness", wrap_find_witness),
        (support_search.SupportSearch, "_use_automorphisms", wrap_use_automorphisms),
        (macwilliams, "get_span_weights", wrap_walk),
        (CSSLogicalCounter, "is_self_dual", wrap_self_dual_test),
    ]
    for owner, name, wrap in patches:
        setattr(owner, name, wrap(getattr(owner, name)))


def wrap_sectors(function: Callable[..., Any]) -> Callable[..., Any]:
    """Number the sectors of each call of the wrapped distance search from zero.

    The wrapped function, ``get_distance_brouwer_zimmermann`` or
    ``get_distance_connected_cluster``, prepares a search of each of its sectors in the order in
    which it receives them, and ``wrap_prepare`` and ``wrap_support_search`` record the index of the
    sector of each search.  So a call with one sector has only sector 0.  ``core._search_css``
    puts the X sector, of the X-type logical operators, before the Z sector, and so does the
    distance search of a count.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global NEXT_SECTOR_INDEX
        NEXT_SECTOR_INDEX = 0
        try:
            return function(*args, **kwargs)
        finally:
            NEXT_SECTOR_INDEX = None

    return wrapper


def take_sector_index() -> int | None:
    """The index of the sector whose search is being prepared, or None outside a wrapped call.

    Within a wrapped call, it also advances the index for the next search.
    """
    global NEXT_SECTOR_INDEX
    index = NEXT_SECTOR_INDEX
    if index is not None:
        NEXT_SECTOR_INDEX = index + 1
    return index


def wrap_prepare(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record, for each BZ distance search that a call prepares, the index of its sector.

    A sector that setup settles takes its index but has no search, and so no layers.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        index = take_sector_index()
        prepared = function(*args, **kwargs)
        if isinstance(prepared, distance._SectorSearch):
            SECTOR_INDICES[id(prepared)] = index
        return prepared

    return wrapper


def wrap_certificate(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each certificate of a search, its estimated work, and its seconds.

    A family of overlapping sets whose estimated work reaches ``lp_minimum_work`` also bounds
    unvisited words by linear programming (see ``BasisFamily.get_search_certificate``).
    """

    @functools.wraps(function)
    def wrapper(
        family: Any, upper_bound: int, lp_minimum_work: int, *args: Any, **kwargs: Any
    ) -> Any:
        start = time.perf_counter()
        result = function(family, upper_bound, lp_minimum_work, *args, **kwargs)
        seconds = time.perf_counter() - start
        overlapping = family.max_coverage != 1
        work = int(family.certificate.work(upper_bound)) if overlapping else None
        lp = work is not None and work >= lp_minimum_work
        event = {"event": "certificate", "overlapping": overlapping, "work": work, "lp": lp}
        EVENTS.append(event | {"seconds": seconds})
        return result

    return wrapper


def wrap_family(kind: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the seconds of building a family, and its number of sets.

    Partitioning of a search with automorphisms may first build their family, to compare its work
    (see ``get_partition_gate``); the seconds of partitioning then include building that family.
    For partitioning, the record also holds the coverages that partitioning tries (``coverages``),
    and ``tried_unit``, the sum of ``b**2 * r**2 * N`` over them.  It can be less than ``unit``,
    since partitioning skips a coverage at which no family could replace the best so far.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global PARTITION_COVERAGES
        competing: dict[str, Any] = {}
        get_competing_work = kwargs.get("get_competing_work")
        if get_competing_work is not None:
            kwargs["get_competing_work"] = functools.partial(
                record_competing_work, get_competing_work, competing
            )
        if kind == "partitioned":
            PARTITION_COVERAGES = []
        start = time.perf_counter()
        try:
            family = function(*args, **kwargs)
        finally:
            coverages, PARTITION_COVERAGES = PARTITION_COVERAGES, None
        seconds = time.perf_counter() - start
        event: dict[str, Any] = {
            "event": "family",
            "kind": kind,
            "sets": len(family.ranks),
            "seconds": seconds,
        }
        if kind == "partitioned" and len(args) == 5:
            event |= get_partition_gate(*args) | competing
            if competing and competing["competing_work"] <= event["optimistic_work"]:
                event["partitioned"] = False
            assert coverages is not None
            event["coverages"] = coverages
            event["tried_unit"] = (
                sum(coverage**2 for coverage in coverages) * event["rows"] ** 2 * event["columns"]
            )
        if kind == "overlapping" and len(args) == 5:
            event |= get_overlapping_gate(*args)
        EVENTS.append(event)
        return family

    return wrapper


def record_competing_work(get_competing_work: Callable[[], int], record: dict[str, Any]) -> int:
    """Record the work of the family that would rule partitioning out, and the seconds it took."""
    start = time.perf_counter()
    work = get_competing_work()
    record |= {"competing_work": int(work), "competing_seconds": time.perf_counter() - start}
    return work


def wrap_partition(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the coverage of each matroid partitioning in a call of ``get_partitioned_family``."""

    @functools.wraps(function)
    def wrapper(basis: Any, coverage: int, *args: Any, **kwargs: Any) -> Any:
        if PARTITION_COVERAGES is not None:
            PARTITION_COVERAGES.append(coverage)
        return function(basis, coverage, *args, **kwargs)

    return wrapper


def get_partition_gate(
    basis: Any, _labels: Any, basis_family: Any, upper_bound: int, tuning: Any
) -> dict[str, Any]:
    """The quantities that ``basis_families.get_partitioned_family`` compares to skip it.

    These are, in candidate/set evaluations where they are work:

    - the estimated work of the family that partitioning would replace;
    - the largest coverage that partitioning may try, or 0 if it may try none;
    - the unit that the gate of partitioning multiplies by ``partition_work_ratio``: the sum of
      ``b**2 * r**2 * N`` over the coverages ``b`` that it may try, for ``r`` rows and ``N``
      nonzero columns;
    - the least work that any partitioned family could do, which a search with automorphisms
      compares with the work of their family;
    - for each coverage ``b`` from two that partitioning may try, in order, the least work that a
      family found at ``b`` could do (``coverage_optimistic_work``).

    Partitioning skips a coverage ``b`` if its bound is at least the work of the best family so far.
    At a ``partition_max_coverage`` of 1, partitioning may try only coverage one, and the least work
    is over disjoint families.  It skips coverage one if the given family is disjoint and has the
    pruned disjoint rank bound (Section 4.4 of ``brouwer_zimmermann.md``).  The record also says
    whether these quantities let partitioning run (``partitioned``), even if it then tried no
    coverage or found no family; ``wrap_family`` clears ``partitioned`` when a competing family
    rules partitioning out.
    """
    plan = basis_families._get_partition_plan(basis, tuning)
    work = basis_family.certificate.work(upper_bound)
    divisor = basis_family.weight_divisor
    optimistic_work = plan.estimate_optimistic_work(divisor, upper_bound)
    coverage_optimistic_work = [
        int(plan.estimate_optimistic_coverage_work(coverage, divisor, upper_bound))
        for coverage in plan.coverages
        if coverage > 1
    ]
    return {
        "rows": plan.dimension,
        "columns": plan.num_columns,
        "estimated_work": int(work),
        "max_coverage": max(plan.coverages, default=0),
        "unit": plan.unit,
        "ratio": tuning.partition_work_ratio,
        "optimistic_work": int(optimistic_work),
        "coverage_optimistic_work": coverage_optimistic_work,
        "partitioned": bool(
            plan.coverages
            and work >= tuning.partition_work_ratio * plan.unit
            and work > optimistic_work
        ),
    }


def get_overlapping_gate(
    basis: Any, _labels: Any, basis_family: Any, upper_bound: int, policy: Any
) -> dict[str, Any]:
    """The check that ends a call of ``basis_families.get_overlapping_family``, and its work.

    The record's ``stage`` names the check of ``basis_families._check_overlapping_family`` that
    keeps the given family, or is ``built`` if none does, in which case the library builds sets.
    Its ``estimated_work`` is the work of the given family, or None if the stage is
    ``no open sets``.
    """
    check, work, _ = basis_families._check_overlapping_family(
        basis, basis_family, upper_bound, policy
    )
    nothing_to_extend = check == basis_families._OverlappingCheck.NOTHING_TO_EXTEND
    return {
        "rows": basis.shape[0],
        "minimum_work": int(policy.minimum_work),
        "estimated_work": None if nothing_to_extend else int(work),
        "stage": OVERLAPPING_STAGES[check],
    }


def wrap_linear_program(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each linear program that a certificate solves, its bound, and its seconds."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> int:
        start = time.perf_counter()
        bound = function(*args, **kwargs)
        weight = None if CURRENT_LAYER is None else CURRENT_LAYER["weight"]
        event = {"event": "linear program", "weight": weight, "bound": int(bound)}
        EVENTS.append(event | {"seconds": time.perf_counter() - start})
        return bound

    return wrapper


def wrap_considers(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record, for each number of rows and weight, the first test of whether to consider tables."""
    seen = set()

    @functools.wraps(function)
    def wrapper(num_rows: int, weight: int, tuning: Any) -> bool:
        result = function(num_rows, weight, tuning)
        if (num_rows, weight) not in seen and 2 <= weight <= num_rows:
            seen.add((num_rows, weight))
            layer_work = completion_bounds._get_layer_work(num_rows, weight)
            EVENTS.append(
                {
                    "event": "considers",
                    "rows": num_rows,
                    "weight": weight,
                    "layer_work": int(layer_work),
                    "considers": result,
                }
            )
        return result

    return wrapper


def wrap_limit(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each comparison of a predicted cost with ``completion_bound_max_cost_percent``."""

    @functools.wraps(function)
    def wrapper(cost: float, tuning: Any) -> bool:
        result = function(cost, tuning)
        EVENTS.append({"event": "limit", "cost": cost, **LAST_PREDICTION, "within": result})
        return result

    return wrapper


def wrap_switch(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each consideration of a switch to automorphism images, and its outcome.

    The works are those that ``_switch_to_automorphism_family`` compares, at the search's best
    weight after it checks the rows of the automorphism family: the work of that family from layer
    one, and the work left in the current family from the layer before which it considers the
    switch.
    """

    @functools.wraps(function)
    def wrapper(search: Any) -> None:
        weight, certificate = search.next_weight, search.instance.certificate
        start = time.perf_counter()
        function(search)
        seconds = time.perf_counter() - start
        switched = search.uses_automorphism_family
        event = {"event": "automorphisms", "weight": weight, "best": search.best}
        family = search.automorphism_family
        family_certificate = (
            search.instance.certificate if switched else None if family is None else family[1]
        )
        if family_certificate is not None:
            event["automorphism_work"] = int(
                automorphism_families.estimate_automorphism_family_work(
                    family_certificate, search.best
                )
            )
            event["remaining_work"] = int(certificate.work(search.best, weight))
        EVENTS.append(event | {"switched": switched, "seconds": seconds})

    return wrapper


def wrap_advance(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each layer that a search advances by, with the index of the sector of the search."""

    @functools.wraps(function)
    def wrapper(search: Any) -> int:
        global CURRENT_LAYER
        if search.finished or CURRENT_LAYER is not None:
            return function(search)
        if SETUP_ONLY:
            raise SetupDone
        weight, dimension = search.next_weight, search.dimension
        uses_automorphism_family = search.uses_automorphism_family
        can_switch = search._considers_automorphism_family
        start = time.perf_counter()
        CURRENT_LAYER = {
            "event": "layer",
            "search": id(search),
            "sector": SECTOR_INDICES.get(id(search)),
            "weight": weight,
            "dimension": dimension,
            "supports": math.comb(dimension, weight),
            "can_switch": can_switch,
            "switch_work": None,
            "scanned_sets": 0,
            "evaluated": 0,
            "lookups": 0,
            "skipped": 0,
            "tables": [],
            "table_bytes": 0,
            "blocks": 0,
            "builds": 0,
            "choose_seconds": 0.0,
            "build_seconds": 0.0,
        }
        # The work of the layer that the switch to automorphism images compares with its threshold,
        # as the source of ``advance`` computes it.  The certificate of the search may solve linear
        # programs, which the search would solve in this layer anyway.
        if can_switch:
            num_scanned = search.instance.certificate.get_num_sets_to_certify(weight, search.best)
            switch_work = min(num_scanned * math.comb(dimension, weight), certificates.WORK_CAP)
            CURRENT_LAYER["switch_work"] = switch_work
        try:
            best = function(search)
        finally:
            layer, CURRENT_LAYER = CURRENT_LAYER, None
        layer["seconds"] = time.perf_counter() - start
        layer["sets"] = len(search.instance.certificate.ranks)
        # The width of the blocks of the tables that the layer chose, or of those of each set that
        # chose, or None if it chose none.  A set of deficient rank chooses no tables.
        choices = layer["tables"]
        layer["tables"] = None if not any(choices) else choices[0] if len(choices) == 1 else choices
        # A search that switches to automorphism images scans its set again from layer two.
        if search.uses_automorphism_family and not uses_automorphism_family:
            layer["weight"] = 2
            layer["supports"] = math.comb(dimension, 2)
        layer |= {"best": search.best, "lower_bound": search.lower_bound}
        layer["automorphism_family"] = search.uses_automorphism_family
        EVENTS.append(layer)
        return best

    return wrapper


def wrap_select(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the tables that a set of a layer chooses, and the seconds spent choosing them."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        tables = function(*args, **kwargs)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["choose_seconds"] += time.perf_counter() - start
            CURRENT_LAYER["tables"].append(0 if tables is None else int(tables.block_bits))
            if tables is not None:
                record_tables(tables)
        return tables

    return wrapper


def wrap_predict(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each prediction of the cost of completion tables.

    The cost is a share of the cost of a scan without tables.  ``at`` is the seconds since the
    workload started, which also time the steps of counts.
    """

    @functools.wraps(function)
    def wrapper(rows: Any, weight: int, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        cost, block_bits = function(rows, weight, *args, **kwargs)
        seconds = time.perf_counter() - start
        LAST_PREDICTION.update(rows=len(rows), weight=weight)
        EVENTS.append(
            {
                "event": "predict",
                "rows": len(rows),
                "weight": weight,
                "supports": math.comb(len(rows), weight),
                "cost": cost,
                "block_bits": block_bits,
                "seconds": seconds,
                "at": time.perf_counter() - START,
            }
        )
        return cost, block_bits

    return wrapper


def wrap_build(function: Callable[..., Any]) -> Callable[..., Any]:
    """Add the seconds of building completion tables to the current layer."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = function(*args, **kwargs)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["build_seconds"] += time.perf_counter() - start
            CURRENT_LAYER["builds"] += 1
            record_tables(result)
        return result

    return wrapper


def record_tables(tables: completion_bounds.CompletionTables) -> None:
    """Record the bytes and the number of blocks of the largest tables of the current layer."""
    assert CURRENT_LAYER is not None
    CURRENT_LAYER["table_bytes"] = max(CURRENT_LAYER["table_bytes"], len(tables.tables))
    CURRENT_LAYER["blocks"] = max(CURRENT_LAYER["blocks"], len(tables.offsets) - 1)


def wrap_scan_set_method(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``_scan_set`` of a distance search to count the sets that the current layer scans."""

    @functools.wraps(function)
    def wrapper(search: Any, *args: Any) -> Any:
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["scanned_sets"] += 1
        return function(search, *args)

    return wrapper


def wrap_scan_set(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``affine_search.scan_set`` to add the counts of each scan to the current layer.

    The counts are the supports that the scan weighs, its lookups, and the supports that it skips.
    """

    @functools.wraps(function)
    def wrapper(*args: Any) -> Any:
        result = function(*args)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["evaluated"] += int(result[2])
            CURRENT_LAYER["lookups"] += int(result[3])
            CURRENT_LAYER["skipped"] += int(result[4])
        return result

    return wrapper


def wrap_support_search(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the index of the sector of each connected-cluster search, as ``wrap_prepare`` does.

    Also time the function that the search calls to find automorphisms.
    """

    @functools.wraps(function)
    def wrapper(search: Any, *args: Any, get_automorphisms: Any = None, **kwargs: Any) -> None:
        SECTOR_INDICES[id(search)] = take_sector_index()
        if get_automorphisms is not None:
            get_automorphisms = functools.partial(time_finding, get_automorphisms)
        function(search, *args, get_automorphisms=get_automorphisms, **kwargs)

    return wrapper


def time_finding(get_automorphisms: Callable[[], Any]) -> Any:
    """Find the automorphisms of a connected-cluster search, and record the seconds it took."""
    start = time.perf_counter()
    automorphisms = get_automorphisms()
    CURRENT_ASK["find_seconds"] = time.perf_counter() - start
    return automorphisms


def wrap_use_automorphisms(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the seconds of verifying automorphisms and ordering their orbits, and the roots."""

    @functools.wraps(function)
    def wrapper(search: Any, automorphisms: Any) -> None:
        start = time.perf_counter()
        function(search, automorphisms)
        CURRENT_ASK["verify_seconds"] = time.perf_counter() - start
        CURRENT_ASK["roots"] = len(search._roots)

    return wrapper


def wrap_find_witness(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each weight limit of a connected-cluster search, and any request for automorphisms.

    Each event names the sector of the search (see ``wrap_sectors``).  The predicted states are
    those that ``find_witness`` adds to the states visited so far, and compares with
    ``automorphism_minimum_states``, while the search has not yet asked.
    """

    @functools.wraps(function)
    def wrapper(search: Any, max_weight: int) -> Any:
        may_ask = search._get_automorphisms is not None
        states_before = search.num_states
        predicted = int(search._predict_states(max_weight)) if may_ask else None
        CURRENT_ASK.clear()
        start = time.perf_counter()
        witness = function(search, max_weight)
        seconds = time.perf_counter() - start
        event = {
            "event": "weight limit",
            "search": id(search),
            "sector": SECTOR_INDICES.get(id(search)),
            "max_weight": max_weight,
            "states_before": states_before,
            "predicted_states": predicted,
            "states": search.num_states - states_before,
            "asked": may_ask and search._get_automorphisms is None,
            "witness_weight": None if witness is None else len(witness),
        }
        if event["asked"]:
            event |= {
                "find_seconds": CURRENT_ASK.get("find_seconds", 0.0),
                "verify_seconds": CURRENT_ASK.get("verify_seconds", 0.0),
                "roots": CURRENT_ASK.get("roots"),
            }
        EVENTS.append(event | {"seconds": seconds})
        return witness

    return wrapper


def wrap_walk(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each walk of a check space by the MacWilliams method."""

    @functools.wraps(function)
    def wrapper(generators: Any, chunk_size: int) -> Any:
        start = time.perf_counter()
        weights = function(generators, chunk_size)
        seconds = time.perf_counter() - start
        rank, num_bits = generators.shape
        EVENTS.append(
            {"event": "walk", "rank": rank, "bits": num_bits, "steps": 2**rank, "seconds": seconds}
        )
        return weights

    return wrapper


def wrap_self_dual_test(prop: property) -> property:
    """Record each test of whether the X and Z checks of a counter span the same space.

    A counter built as self-dual answers without a test, and is not recorded.
    """
    assert prop.fget is not None
    get_self_dual = prop.fget

    def getter(counter: CSSLogicalCounter) -> bool:
        if counter._self_dual:
            return get_self_dual(counter)
        cached = counter._canonical is not None
        start = time.perf_counter()
        result = get_self_dual(counter)
        seconds = time.perf_counter() - start
        rank, num_bits = counter._gauge_x.shape
        EVENTS.append(
            {
                "event": "self-dual test",
                "rank": rank,
                "bits": num_bits,
                "cached": cached,
                "self_dual": result,
                "seconds": seconds,
            }
        )
        return result

    return property(getter, doc=prop.__doc__)


####################################################################################################
# output


def print_trace(result: str, seconds: float) -> None:
    """Print the events of a trace."""
    print(f"result {result} in {seconds:.3f} s")
    for event in EVENTS:
        if event["event"] != "layer":
            details = ", ".join(f"{key}={value}" for key, value in event.items() if key != "event")
            print(f"  {event['event']}: {details}")
            continue
        tables = "" if event["tables"] is None else f" tables={event['tables']}"
        print(
            f"  layer sector={event['sector']} w={event['weight']:<2d}"
            f" rows={event['dimension']:<4d} sets={event['sets']:<2d}"
            f" scanned={event['scanned_sets']:<2d}"
            f" supports={event['supports']:.2e} evaluated={event['evaluated']:.2e}"
            f" skipped={event['skipped']:.2e} lookups={event['lookups']:.2e}{tables}"
            f" best={event['best']} lower={event['lower_bound']}"
            f"{' aut' if event['automorphism_family'] else ''}"
            f" choose={event['choose_seconds']:.3f}s build={event['build_seconds']:.3f}s"
            f" {event['seconds']:.3f}s"
        )


####################################################################################################
# comparison


def compare(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Print whether two traces of one workload make the same decisions.

    Layers that only one tuning considers for tables add comparisons with the limit to its trace;
    the comparison also reports the decisions without those.
    """
    if first["workload"] != second["workload"]:
        raise ValueError("Compare traces of one workload")
    for index, trace in enumerate((first, second), 1):
        print(
            f"trace {index}: result {trace['result']}, changes {trace['tuning']},"
            f" source {trace.get('source')}"
        )
    decisions = [get_decisions(first), get_decisions(second)]
    same = decisions[0] == decisions[1]
    print(
        f"{first['workload']}: {len(decisions[0])} and {len(decisions[1])} decisions, same: {same}"
    )
    considered = [
        {
            (e["rows"], e["weight"])
            for e in trace["events"]
            if e["event"] == "considers" and e["considers"]
        }
        for trace in (first, second)
    ]
    for index, other in ((0, 1), (1, 0)):
        extra = considered[index] - considered[other]
        if not extra:
            continue
        trimmed = [d for d in decisions[index] if not (d[0] == "limit" and (d[1], d[2]) in extra)]
        print(f"    layers that only trace {index + 1} considers: {sorted(extra)}")
        print(f"    same decisions without their predictions: {trimmed == decisions[other]}")
    if not same:
        for index, (one, two) in enumerate(zip(*decisions)):
            if one != two:
                print(f"    first difference, at decision {index}: {one} and {two}")
                break


def get_decisions(trace: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The decisions of a trace, in order, with predicted costs rounded to 9 digits."""
    decisions: list[tuple[Any, ...]] = []
    for event in trace["events"]:
        kind = event["event"]
        if kind == "family":
            decisions.append(
                (kind, event["kind"], event["sets"], event.get("stage"), event.get("partitioned"))
            )
        elif kind == "certificate":
            decisions.append((kind, event["overlapping"], event["lp"]))
        elif kind == "automorphisms":
            decisions.append((kind, event["switched"]))
        elif kind == "layer":
            decisions.append(
                (kind, event["weight"], event["dimension"], event["sets"], event["tables"])
            )
        elif kind == "limit" and not math.isinf(event["cost"]):
            cost = round(event["cost"], 9)
            decisions.append((kind, event["rows"], event["weight"], cost, event["within"]))
        elif kind == "weight limit":
            decisions.append((kind, event["max_weight"], event["asked"]))
        elif kind == "self-dual test":
            decisions.append((kind, event["rank"], event["self_dual"]))
        elif kind == "walk":
            decisions.append((kind, event["rank"]))
    return decisions


if __name__ == "__main__":
    main()
