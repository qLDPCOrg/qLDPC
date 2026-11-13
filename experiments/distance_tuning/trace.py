# SPDX-License-Identifier: Apache-2.0

"""Trace the decisions and stage times of BZ searches of a distance, hadamard, or count workload.

The script wraps functions of ``qldpc.codes.distance._brouwer_zimmermann`` and
``qldpc.codes.distance._information_sets`` in this process only, warms up as ``trial.py`` does, and
runs a ``distance``, ``hadamard``, or ``count`` workload once under the given tuning.  It cannot
trace ``ccount`` workloads: their compiled setup calls the stages directly, past the wrappers.  It
records:

- each test of whether to enumerate every codeword instead of searching information sets: by size,
  with the dimension ``k``, the length ``n``, and ``2**k / (k * n)``, and by work, with the
  candidates of each route;
- each family of information sets that the search builds (disjoint, overlapping, or by matroid
  partitioning), with its number of sets and the seconds that building it took, and the quantities
  that the gates of overlapping families and of partitioning compare: for overlapping families, the
  check that stopped the attempt and the estimated work, and for partitioning, the work of the
  search and the cost of partitioning, in candidate/set evaluations, the least work that any
  partitioned family could do, and whether the gates let it try (``partitioned``), even if it then
  found no family.  Before trying, a search with automorphisms asks for them, verifies them, and
  builds their family, and skips partitioning if that family does no more work than the least work
  of a partitioned family.  The trace records that family's work (``competing_work``) and the
  seconds those steps took (``competing_seconds``), which the seconds of partitioning include;
- each certificate of a search, which may solve linear programs, with its estimated work, which
  decides whether it may, and the seconds that took, and each linear program that it solves, with
  its bound and seconds;
- each attempt to switch to a family of automorphism images, the seconds it took, which exclude
  finding the automorphisms and building their family if setup already did, and whether the
  search switched;
- the first test of each number of rows ``k`` and weight ``w`` for whether a layer or step
  considers completion tables, with its ``C(k, w)`` supports, and each comparison of a predicted
  cost with the limit, with the rows and weight of the prediction before it;
- each prediction of the cost of completion tables, as a share of the cost of a scan without them,
  with the number of rows and the weight of its layer or step, the seconds it took, and the seconds
  since the workload started, which also time the steps of counts;
- each coefficient layer that a distance search advances by: its weight, number of basis rows and
  sets (of the family of automorphism images, for the layer that switches to it), its supports,
  whether the search can still switch to automorphism images and, if so, the work of the layer that
  the switch compares (scanned sets times supports), the best weight and lower bound after it, the
  supports that it evaluated and skipped, its candidate/set evaluations (each support of a scan in
  lockstep counts once for each of its sets), and the lookups of completion tables that it made, the
  width of the blocks of its tables if any, with their bytes and number of blocks, the seconds spent
  choosing tables (which, for a layer of one set, include building them) and building them, and
  its seconds in all.

It prints the events, and with ``--output``, saves them as JSON, with the workload, the changes of
the tuning, the whole tuning, and the source (see ``sweep.get_source_fingerprint``).  With
``--setup-only``, it stops before the first layer of the first distance search, after its
families, or before the first enumeration of codewords; it does not stop counts.  The thresholds of
``SearchTuning`` decide where these stages start, so a trace shows which decisions a threshold can
flip, and what each stage costs, before timing the tunings that flip them with ``sweep.py``.

Example:

    python experiments/distance_tuning/trace.py distance/qt7/X --tuning '{"lp_minimum_work": 0}'
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

import numpy as np

from qldpc.codes.distance import _tuning
from qldpc.codes.distance._brouwer_zimmermann import affine_search, distance
from qldpc.codes.distance._information_sets import basis_families, certificates, completion_bounds

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


class SetupDone(Exception):
    """Raised before the first layer of a search whose trace stops after its setup."""


def main() -> None:
    """Trace a workload, and print and save the trace."""
    args = parse_args()
    if args.workload.split("/")[0] not in ("distance", "hadamard", "count"):
        raise ValueError("Only distance, hadamard, and count workloads can be traced")
    trial.check_workload(args.workload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    source = sweep.get_source_fingerprint()
    with _tuning.use_tuning(**json.loads(args.tuning)) as tuning:
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
        "workload", help="a distance, hadamard, or count workload name (see workloads.py)"
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
        help="stop a distance search before its first layer or enumeration, after its setup",
    )
    return parser.parse_args()


####################################################################################################
# wrappers


def install_wrappers() -> None:
    """Wrap the stages of BZ searches, in this process, to record events."""
    search_class = distance._BrouwerZimmermannSearch
    tables_class = completion_bounds.CompletionTables
    patches: list[tuple[object, str, Callable[[Callable[..., Any]], Callable[..., Any]]]] = [
        (basis_families, f"get_{kind}_family", functools.partial(wrap_family, kind))
        for kind in ("disjoint", "overlapping", "partitioned")
    ]
    patches += [
        (basis_families.BasisFamily, "get_search_certificate", wrap_certificate),
        (search_class, "advance", wrap_advance),
        (search_class, "_switch_to_automorphism_family", wrap_switch),
        (search_class, "_choose_tables_of_sets", wrap_choose),
        (completion_bounds, "select_completion_tables", wrap_select),
        (completion_bounds, "predict_completion_cost", wrap_predict),
        (tables_class, "build", lambda build: staticmethod(wrap_build(build))),
        (affine_search, "scan_information_set", wrap_scan),
        (affine_search, "scan_family", wrap_scan),
        (completion_bounds, "considers_tables", wrap_considers),
        (completion_bounds, "is_within_cost_limit", wrap_limit),
        (distance, "enumeration_is_cheaper_than_setup", wrap_enumeration_by_size),
        (distance, "_enumeration_is_cheaper_than_search", wrap_enumeration_by_work),
        (distance, "get_nested_distance_brute_force", wrap_enumerate),
        (certificates, "get_lp_lower_bound", wrap_linear_program),
    ]
    for owner, name, wrap in patches:
        setattr(owner, name, wrap(getattr(owner, name)))


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
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        competing: dict[str, Any] = {}
        get_competing_work = kwargs.get("get_competing_work")
        if get_competing_work is not None:
            kwargs["get_competing_work"] = functools.partial(
                record_competing_work, get_competing_work, competing
            )
        start = time.perf_counter()
        family = function(*args, **kwargs)
        seconds = time.perf_counter() - start
        event = {"event": "family", "kind": kind, "sets": len(family.ranks), "seconds": seconds}
        if kind == "partitioned" and len(args) == 5:
            event |= get_partition_gate(*args) | competing
            if competing and competing["competing_work"] <= event["optimistic_work"]:
                event["partitioned"] = False
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


def get_partition_gate(
    basis: Any, _labels: Any, basis_family: Any, upper_bound: int, tuning: Any
) -> dict[str, Any]:
    """The quantities that ``basis_families.get_partitioned_family`` compares to skip it.

    These are the estimated work of the family that partitioning would replace, in candidate/set
    evaluations, and the estimated cost of partitioning in the same unit, which the search
    multiplies by ``partition_work_ratio`` (see the source of ``get_partitioned_family``), the
    unit ``b * k**2 * n``, summed over coverages ``b``, that ``partition_cost_factor`` multiplies,
    and the least work that any partitioned family could do, which a search with automorphisms
    compares with the work of their family.
    """
    dimension = len(basis)
    num_columns = int(np.count_nonzero(np.any(basis, axis=0)))
    total_coverage = sum(range(2, tuning.max_partition_coverage + 1))
    cost = tuning.partition_cost_factor * total_coverage * dimension**2 * num_columns
    work = basis_family.certificate.work(upper_bound)
    optimistic_work = basis_families._estimate_optimistic_work(
        dimension,
        num_columns,
        basis_family.weight_divisor,
        upper_bound,
        tuning.max_information_sets,
        basis_families.get_capped_binomials(dimension),
    )
    return {
        "rows": dimension,
        "columns": num_columns,
        "estimated_work": int(work),
        "estimated_cost": int(cost),
        "unit": total_coverage * dimension**2 * num_columns,
        "ratio": tuning.partition_work_ratio,
        "optimistic_work": int(optimistic_work),
        "partitioned": bool(
            tuning.max_partition_coverage >= 2
            and work >= tuning.partition_work_ratio * cost
            and work > optimistic_work
        ),
    }


def get_overlapping_gate(
    basis: Any, _labels: Any, basis_family: Any, upper_bound: int, policy: Any
) -> dict[str, Any]:
    """The check that ends an attempt of ``basis_families.get_overlapping_family``, and its work.

    The checks follow the source of ``get_overlapping_family``.  It keeps a family with no full
    sets, with as many as the limit, or whose full sets pack every column; then one whose estimated
    work is below ``minimum_work`` of the policy; and then one that no family of at most the limit
    could beat.  Past these checks, it builds sets.
    """
    max_sets, minimum_work, _, binomials, _ = policy
    dimension = basis.shape[0]
    num_columns = int(np.count_nonzero(basis.sum(axis=0)))
    ranks = basis_family.ranks
    num_full = int(np.count_nonzero(ranks == dimension))
    gate: dict[str, Any] = {"rows": dimension, "minimum_work": int(minimum_work)}
    if num_full == 0 or num_full >= max_sets or num_full * dimension == num_columns:
        return gate | {"stage": "no open sets", "estimated_work": None}
    divisor = basis_family.weight_divisor
    work = int(
        basis_families.estimate_certificate_work(
            dimension, ranks, divisor, basis_family.max_coverage, upper_bound, binomials
        )
    )
    gate["estimated_work"] = work
    if work < minimum_work:
        return gate | {"stage": "below minimum work"}
    optimistic_work = basis_families._estimate_optimistic_work(
        dimension, num_columns, divisor, upper_bound, max_sets, binomials
    )
    if optimistic_work >= work:
        return gate | {"stage": "cannot do less work"}
    return gate | {"stage": "built"}


def wrap_enumeration_by_size(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each test of whether to enumerate codewords instead of building information sets."""

    @functools.wraps(function)
    def wrapper(dimension: int, length: int, max_dimension: int, factor: int) -> bool:
        result = function(dimension, length, max_dimension, factor)
        EVENTS.append(
            {
                "event": "enumeration",
                "test": "size",
                "dimension": dimension,
                "length": length,
                "words_per_setup_unit": (
                    2**dimension / (dimension * length) if dimension < 63 else math.inf
                ),
                "enumerate": result,
            }
        )
        return result

    return wrapper


def wrap_enumeration_by_work(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each test of whether to enumerate codewords instead of searching information sets."""

    @functools.wraps(function)
    def wrapper(*, num_logical_rows: int, certificate: Any, upper_bound: int) -> bool:
        result = function(
            num_logical_rows=num_logical_rows, certificate=certificate, upper_bound=upper_bound
        )
        dimension = certificate.dimension
        EVENTS.append(
            {
                "event": "enumeration",
                "test": "work",
                "dimension": dimension,
                "exhaustive": ((1 << num_logical_rows) - 1) << (dimension - num_logical_rows),
                "estimated_work": int(certificate.work(upper_bound)),
                "enumerate": result,
            }
        )
        return result

    return wrapper


def wrap_enumerate(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the seconds of each enumeration of codewords, or stop before it after the setup."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if SETUP_ONLY:
            raise SetupDone
        start = time.perf_counter()
        result = function(*args, **kwargs)
        EVENTS.append({"event": "enumerate", "seconds": time.perf_counter() - start})
        return result

    return wrapper


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
    """Record the first test of each layer shape for whether it considers completion tables."""
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
    """Record each comparison of a predicted cost of completion tables with the limit."""

    @functools.wraps(function)
    def wrapper(cost: float, tuning: Any) -> bool:
        result = function(cost, tuning)
        EVENTS.append({"event": "limit", "cost": cost, **LAST_PREDICTION, "within": result})
        return result

    return wrapper


def wrap_switch(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the seconds of an attempt to switch to automorphism images, and its outcome."""

    @functools.wraps(function)
    def wrapper(search: Any) -> None:
        start = time.perf_counter()
        function(search)
        seconds = time.perf_counter() - start
        switched = search.uses_automorphism_family
        EVENTS.append({"event": "automorphisms", "switched": switched, "seconds": seconds})

    return wrapper


def wrap_advance(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each layer that a search advances by."""

    @functools.wraps(function)
    def wrapper(search: Any) -> int:
        global CURRENT_LAYER
        if search.finished or CURRENT_LAYER is not None:
            return function(search)
        if SETUP_ONLY:
            raise SetupDone
        weight, dimension = search.next_weight, search.dimension
        uses_automorphism_family = search.uses_automorphism_family
        can_switch = search.get_automorphism_family is not None and not uses_automorphism_family
        start = time.perf_counter()
        CURRENT_LAYER = {
            "event": "layer",
            "search": id(search),
            "weight": weight,
            "dimension": dimension,
            "supports": math.comb(dimension, weight),
            "can_switch": can_switch,
            "switch_work": None,
            "evaluated": 0,
            "set_evaluations": 0,
            "lookups": 0,
            "skipped": 0,
            "tables": None,
            "table_bytes": 0,
            "blocks": 0,
            "choose_seconds": 0.0,
            "build_seconds": 0.0,
        }
        # The work of the layer that the switch to automorphism images compares with its threshold,
        # as the source of ``advance`` computes it.  Its certificate may solve linear programs,
        # which the search would solve in this layer anyway.
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
    """Record the tables of a layer of one set, and the seconds spent choosing them."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        tables = function(*args, **kwargs)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["choose_seconds"] += time.perf_counter() - start
            CURRENT_LAYER["tables"] = None if tables is None else int(tables.block_bits)
            if tables is not None:
                record_tables(tables)
        return tables

    return wrapper


def wrap_predict(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record each prediction of the cost of completion tables."""

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


def wrap_choose(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record the tables of a layer of several sets, and the seconds spent choosing them."""

    @functools.wraps(function)
    def wrapper(search: Any, *args: Any) -> Any:
        start = time.perf_counter()
        block_bits = function(search, *args)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["choose_seconds"] += time.perf_counter() - start
            CURRENT_LAYER["tables"] = block_bits
        return block_bits

    return wrapper


def wrap_build(function: Callable[..., Any]) -> Callable[..., Any]:
    """Add the seconds of building completion tables to the current layer."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = function(*args, **kwargs)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["build_seconds"] += time.perf_counter() - start
            record_tables(result)
        return result

    return wrapper


def record_tables(tables: completion_bounds.CompletionTables) -> None:
    """Record the bytes and the number of blocks of the largest tables of the current layer."""
    assert CURRENT_LAYER is not None
    CURRENT_LAYER["table_bytes"] = max(CURRENT_LAYER["table_bytes"], len(tables.tables))
    CURRENT_LAYER["blocks"] = max(CURRENT_LAYER["blocks"], len(tables.offsets) - 1)


def wrap_scan(function: Callable[..., Any]) -> Callable[..., Any]:
    """Add the evaluated and skipped supports and the lookups of a scan to the current layer."""

    @functools.wraps(function)
    def wrapper(*args: Any) -> Any:
        result = function(*args)
        if CURRENT_LAYER is not None:
            CURRENT_LAYER["evaluated"] += int(result[2])
            # A scan in lockstep weighs each support in each of its sets.
            num_sets = 1 if len(result) == 5 else len(args[0])
            CURRENT_LAYER["set_evaluations"] += int(result[2]) * num_sets
            if len(result) == 5:
                CURRENT_LAYER["lookups"] += int(result[3])
                CURRENT_LAYER["skipped"] += int(result[4])
        return result

    return wrapper


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
            f"  layer w={event['weight']:<2d} rows={event['dimension']:<4d}"
            f" sets={event['sets']:<2d}"
            f" supports={event['supports']:.2e} evaluated={event['evaluated']:.2e}"
            f" skipped={event['skipped']:.2e} lookups={event['lookups']:.2e}{tables}"
            f" best={event['best']} lower={event['lower_bound']}"
            f"{' aut' if event['automorphism_family'] else ''}"
            f" choose={event['choose_seconds']:.3f}s build={event['build_seconds']:.3f}s"
            f" {event['seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
