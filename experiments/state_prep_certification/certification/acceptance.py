# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests of the certification experiment.

    python acceptance.py [--local] [--only AT1 AT3 ...] [--results-dir R]

* **AT1:** C-goto certifies at K = 2; its weakened variant, which deletes the flag, violates at
  weight 1; on E-chen, BZ and CC give the same verdict and smallest violating weight at K = 3.
* **AT2:** the listing counter (up to K') and the accepted-combination counter (up to K) equal
  brute force on the ground-truth circuits of the d = 3 A-ft codes and on 20 random DEMs
  (N_0 <= 25, p <= 8, K <= 4), including DEMs where a light class precedes a heavy one in DEM
  order; the listing counter gives the same counts when its compiled search returns one class
  codeword at a time; and the rows submitted to the oracle never exceed the listed faults on
  completed weights.
* **AT3:** instrumented and uninstrumented certification give identical certificates on C-goto,
  its variant, and A-ft steane; every wrapped name exists; on a CSS circuit with a violation, each
  query of the fault search is counted once, against an independent spy of the calls that the
  fault search makes.  On E-chen, the shadow oracle answers each query that the counters count,
  with one threshold-1 query per reduced mechanism, and an oracle whose answers are shifted raises
  ``ShadowDisagreement``.
* **AT4:** ``cert_bz`` with a limit of 5 s, on the first of the B-cat circuits of largest K and
  size that completes a weight in that time, ends as a timeout with T >= 5 s, reports its completed
  weights as consecutive weights from 2 without a violation, and is not rerun when its task is run
  again.  A ``cert_bz`` run on the weakened C-goto, which violates, gets a post-processing allowance
  of 1 s, and its description is delayed by 5 s (``post_delay_s``).  The run keeps status ok, T, and
  its verdict, and its ``post_status`` is ``timeout``.  A fast ``cert_bz`` run on C-goto gets
  exactly ``repeats.count`` repeats (``repeat_tasks``), each with a limit of 600 s and no
  ``post_s``, and a repeat ends without post-processing or certificate text.
* **AT5:** every input matches its SHA-256, its recomputed DEM parameters equal those of
  ``circuits.json``, and the D-synth states are the codes' |0...0> (``circuits.py --check``).
  Every noisy input has only the expected noise, with every measurement noisy.
* **AT6:** neither ``mqt`` nor ``qiskit`` is loaded by the node's modules.
* **AT7:** the brute-force baseline, on the ground-truth circuits and on C-goto, its variant, and
  E-chen: it receives the arguments of ``cert_bz``'s fault search and builds one BZ oracle by the
  fault search's name; its verdict and smallest violating weight equal those of ``cert_bz`` and the
  ground truth; and on each completed weight, the combinations enumerated and the rows queried equal
  A_m of the counter and of the ground truth.  The ``brute_bz`` arm runs through the task runner on
  E-chen.  On the 20 random DEMs of AT2, with syndromes of Steane's |0>,
  ``brute_force_smallest_violation`` finds a violation of the same weight as qLDPC's
  ``find_smallest_violation`` and the ground truth, also with small batches.  Each circuit's U,
  recomputed from N_0 and K, and whether it is within the cap, equal those of ``circuits.json``.
* **AT8:** the analysis (``analyze.py``) of synthetic rows on four B-cat circuits.  A
  ``count_listing`` run that timed out after one completed weight gives Pi, E, Phi, q, q_h, and h
  from its partial results.  A ground-truth count that differs from the counters', and a baseline
  whose enumerated combinations differ from A_m, each make their circuit a disagreement, as does a
  baseline stopped after a violation at weight m against a finished certification whose smallest
  violation is not at m.  An expected task without a row is reported as ``missing``.

The expected values of AT1 come from qLDPC's notebook and tests; a failure is investigated and
reported, never fixed by changing a rule.  Each test prints PASS or FAIL; the script exits with
status 1 if any failed.  The tests are the same on the node and on a workstation: ``--local`` and
the ``node`` argument of each test exist because the plug-in interface passes whether the tests
run on the node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from qldpc.codes.distance import CosetLeaderSearch, DistanceMethod, MinWeightBound

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import (
    analyze,
    baseline,
    circuits,
    counting,
    experiment,
    instrument,
)
from experiments.state_prep_certification.common import local_runs

# The failures that a test reports as FAIL, after which the other tests still run.
FAILURES = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    SystemExit,
    TypeError,
    ValueError,
    instrument.ShadowDisagreement,
    numba.core.errors.NumbaError,
)

TESTS: dict[str, Callable[[pathlib.Path, bool], None]] = {}


def main() -> None:
    """Run the selected tests."""
    args = parse_args()
    results_dir = args.results_dir or pathlib.Path(tempfile.mkdtemp(prefix="spc-cert-"))
    passed = run_all(node=not args.local, results_dir=results_dir, only=args.only)
    sys.exit(0 if passed else 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="mark the run as off the node, which runs the same tests",
    )
    parser.add_argument(
        "--only", nargs="*", choices=sorted(TESTS), help="run only these tests, such as AT1"
    )
    parser.add_argument("--results-dir", type=pathlib.Path, help="where the tests' runs go")
    return parser.parse_args()


def run_all(node: bool, results_dir: pathlib.Path, only: list[str] | None = None) -> bool:
    """Run the tests, print PASS or FAIL for each, and return whether all passed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, test in TESTS.items():
        if only and name not in only:
            continue
        start = time.monotonic()
        try:
            test(results_dir / name, node)
            print(f"PASS {name} ({time.monotonic() - start:.1f} s)", flush=True)
        except FAILURES:
            failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print(f"{len(failed)} failed: {failed}" if failed else "all passed", flush=True)
    return not failed


def test(function: Callable[[pathlib.Path, bool], None]) -> Callable[[pathlib.Path, bool], None]:
    """Register a test by the ID at the start of its name, as in ``at1_...`` for AT1."""
    TESTS[function.__name__.split("_")[0].upper()] = function
    return function


def _certify(circuit_id: str, method: DistanceMethod, max_weight: int | None = None) -> Any:
    """Certify a circuit of ``circuits.json`` with its options, at its K unless one is given."""
    from qldpc import circuits as qldpc_circuits

    entry = experiment._entry(circuit_id)
    options = experiment.certify_options(entry)
    if max_weight is not None:
        options["max_weight"] = max_weight
    return qldpc_circuits.certify_bounded_fault_propagation(
        experiment._noisy_input(entry), method=method, **options
    )


####################################################################################################
# tests


@test
def at1_known_circuits(directory: pathlib.Path, node: bool) -> None:
    """C-goto certifies at K = 2, its variant violates at 1, and BZ and CC agree on E-chen."""
    for method in experiment.METHODS.values():
        goto = _certify("cgoto", method, max_weight=2)
        assert goto.certified, (method, str(goto))
        weakened = _certify("w.cgoto", method, max_weight=2)
        assert not weakened.certified, (method, str(weakened))
        assert len(weakened.counterexample.fault_indices) == 1
    verdicts = []
    for method in experiment.METHODS.values():
        chen = _certify("echen", method, max_weight=3)
        verdicts.append(
            (chen.certified, None if chen.certified else len(chen.counterexample.fault_indices))
        )
    assert verdicts[0] == verdicts[1], verdicts
    print(f"  E-chen at K = 3: certified {verdicts[0][0]}, smallest violation {verdicts[0][1]}")


@test
def at2_counters(directory: pathlib.Path, node: bool) -> None:
    """The counters equal brute force on small circuits and random DEMs."""
    # random DEMs, some with a light class before a heavy one in DEM order
    light_first = 0
    for flags, heavy, max_weight in _random_dems():
        light_first += bool(heavy and not heavy[0] and any(heavy))
        _compare_counts(flags, heavy, max_weight)
    assert light_first >= 3, light_first

    # the ground-truth circuits of the d = 3 A-ft codes
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    small = [
        e
        for e in entries
        if e["family"] == "A-ft" and e["status"] == "ok" and e["ground_truth"] and e["code"][2] == 3
    ]
    assert small, "no ground-truth A-ft circuit with d = 3"
    for entry in small:
        reference = _certify(entry["id"], "brouwer_zimmermann", max_weight=1)
        flags, syndromes = counting.dem_signatures(
            reference.detector_error_model,
            reference.post_selection_detectors,
            len(reference.state_stabilizers),
        )
        reduced = counting.reduce_signatures(flags, syndromes)
        stabilizers = experiment.stabilizer_matrix(reference.state_stabilizers)
        truth = counting.brute_force(reduced, stabilizers, entry["options"]["max_weight"])
        _compare_counts(reduced.flags, truth["heavy"], entry["options"]["max_weight"], truth)

        # the BZ oracle classifies the flagged mechanisms as brute force does
        oracle = CosetLeaderSearch(
            stabilizers, css_weight=reference.css_weight, method="brouwer_zimmermann"
        )
        heavy = [
            not oracle.get_min_weight_bound(syndrome, 1).within_max_weight
            for flag, syndrome in zip(reduced.flags, reduced.syndromes)
            if flag
        ]
        assert heavy == truth["heavy"], entry["id"]

        # the oracle's submitted rows never exceed the listed faults on completed weights
        instrumentation = instrument.Instrumentation()
        with instrumentation.install():
            certificate = _certify(entry["id"], "brouwer_zimmermann")
        listed = truth["Phi"]
        for record in instrumentation.weights:
            if not record["violation"]:
                assert record["rows"] <= listed[str(record["m"])], (entry["id"], record)
        expected = truth["smallest_violation"]
        found = None if certificate.certified else len(certificate.counterexample.fault_indices)
        assert found == expected, (entry["id"], found, expected)
    print(f"  {len(small)} circuits; 20 random DEMs, {light_first} with a light class first")


@test
def at3_instrumentation(directory: pathlib.Path, node: bool) -> None:
    """Instrumentation changes no certificate, counts each query once, and shadows each query."""
    from qldpc.circuits.certification import _fault_search, fault_propagation

    assert hasattr(fault_propagation, "find_smallest_violation")
    for name in ("CosetLeaderSearch", "FaultLister", "_check_fault"):
        assert hasattr(_fault_search, name), name
    for name in ("_should_build_closing_pairs", "_max_syndromes"):
        assert name in _fault_search.FaultLister.__init__.__code__.co_names, name

    ids = ["cgoto", "w.cgoto"] + [
        e["id"]
        for e in json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
        if e["id"].startswith("aft.steane.") and e["status"] == "ok"
    ]
    for circuit_id in ids:
        plain = _certify(circuit_id, "brouwer_zimmermann")
        instrumentation = instrument.Instrumentation()
        with instrumentation.install():
            measured = _certify(circuit_id, "brouwer_zimmermann")
        assert str(plain) == str(measured), circuit_id
        assert _fault_search.CosetLeaderSearch is instrument._ORIGINAL_SEARCH

    # a CSS circuit with a violation: the weakened A-ft circuit of largest smallest violation, so
    # that batches, and calls nested in them, occur
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    violating = []
    for entry in entries:
        if entry["family"] != "W" or entry["status"] != "ok" or "aft." not in entry["id"]:
            continue
        certificate = _certify(entry["id"], "brouwer_zimmermann")
        if not certificate.certified:
            weight = len(certificate.counterexample.fault_indices)
            violating.append((weight, entry["id"]))
    assert violating, "no weakened A-ft circuit with a violation"
    weight, circuit_id = max(violating)
    spy_counts = _spy_calls(circuit_id)
    instrumentation = instrument.Instrumentation()
    with instrumentation.install():
        _certify(circuit_id, "brouwer_zimmermann")
    batches = sum(batch["calls"] for batch in instrumentation.batches.values())
    counted = (
        instrumentation.zero_flag_checks["calls"]
        + instrumentation.classification["calls"]
        + batches
    )
    assert counted == spy_counts, (circuit_id, counted, spy_counts)
    print(
        f"  {circuit_id} (violation at {weight}): {counted} queries of the fault search, "
        f"{batches} of them batches, as the spy counts"
    )

    # the shadow oracle answers every counted query of E-chen's search, which reaches weight 3
    instrumentation = instrument.Instrumentation(shadow=True)
    with instrumentation.install():
        certificate = _certify("echen", "brouwer_zimmermann")
    counted_by_kind = {
        "zero_flag:1": instrumentation.zero_flag_checks["calls"],
        "classification:1": instrumentation.classification["calls"],
    } | {f"batch:{m}": batch["calls"] for m, batch in instrumentation.batches.items()}
    shadowed = {
        key: totals["calls"] for key, totals in instrumentation.shadow["by_threshold"].items()
    }
    assert shadowed == {key: calls for key, calls in counted_by_kind.items() if calls}, (
        shadowed,
        counted_by_kind,
    )
    flags, syndromes = counting.dem_signatures(
        certificate.detector_error_model,
        certificate.post_selection_detectors,
        len(certificate.state_stabilizers),
    )
    reduced = counting.reduce_signatures(flags, syndromes)
    assert counted_by_kind["zero_flag:1"] == reduced.num_zero_flag, counted_by_kind
    num_flagged = reduced.num_reduced - reduced.num_zero_flag
    assert counted_by_kind["classification:1"] == num_flagged, counted_by_kind
    assert any(key.startswith("batch:") for key in shadowed), shadowed
    assert instrumentation.shadow["auto"] in ("brouwer_zimmermann", "connected_cluster")

    # a shadow oracle that shifts its answer to each batch is caught
    try:
        with _ShiftedShadow(shadow=True).install():
            _certify("echen", "brouwer_zimmermann")
    except instrument.ShadowDisagreement:
        pass
    else:
        raise AssertionError("a shifted shadow oracle raised no ShadowDisagreement")
    print(f"  echen: the shadow answered {sum(shadowed.values())} queries, {shadowed}")


@test
def at4_limits_and_resume(directory: pathlib.Path, node: bool) -> None:
    """Limits, the post-processing allowance, repeats, worst cases, and resume."""
    # the first of the five cat circuits of largest K and size that completes a weight but not the
    # search within 5 s
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    candidates = [e for e in entries if e["family"] == "B-cat" and e["status"] == "ok"]
    candidates.sort(key=lambda e: (e["options"]["max_weight"], e["num_mechanisms"]), reverse=True)
    for circuit in candidates[:5]:
        task = experiment._task("cert_bz", circuit, 0, 5.0)
        row = local_runs.run_locally(directory / circuit["id"], [task])[task["key"]]
        if row["status"] == "timeout" and row["partial"]:
            break
    else:
        raise AssertionError("no cat circuit completes a weight but not the search in 5 s")
    assert row["T"] >= 5.0, row

    # the partial results are the completed weights, in order from 2, each without a violation
    weights = [record["m"] for record in row["partial"]]
    assert weights == list(range(2, 2 + len(weights))), weights
    assert all(not record["violation"] for record in row["partial"]), row["partial"]
    assert all(record["T_cumulative"] <= row["T"] for record in row["partial"]), row["partial"]
    assert len(weights) < circuit["options"]["max_weight"], weights
    print(f"  {circuit['id']}: timeout at T = {row['T']:.1f} s, completed weights {weights}")

    before = (directory / circuit["id"] / "runs.jsonl").read_bytes()
    local_runs.run_locally(directory / circuit["id"], [task])
    assert (directory / circuit["id"] / "runs.jsonl").read_bytes() == before

    # a short post-processing allowance, which a delayed description exceeds, marks only the
    # post-processing
    entry = experiment._entry("w.cgoto")
    task = experiment._task("cert_bz", entry, 0, experiment._limit("cert_bz"))
    task |= {"post_s": 1.0, "post_delay_s": 5.0}
    row = local_runs.run_locally(directory / "short_post", [task])[task["key"]]
    result = row["result"]
    assert row["status"] == "ok" and row["T"] is not None, row
    assert row["post_status"] == "timeout", row
    assert result["certified"] is False and result["violation_weight"], result
    assert "certificate_text" not in result and "residual_weight" not in result, result
    print(f"  {entry['id']}: violation at {result['violation_weight']}, post-processing timeout")

    # a fast run gets the configured repeats, each without post-processing
    task = experiment._task("cert_bz", experiment._entry("cgoto"), 0, experiment._limit("cert_bz"))
    row = local_runs.run_locally(directory / "fast", [task])[task["key"]]
    assert row["status"] == "ok" and row["T"] < experiment.CONFIG["repeats"]["below_s"], row
    assert row["post_status"] == "ok" and "certificate_text" in row["result"], row
    repeats = experiment.repeat_tasks(task, row)
    assert len(repeats) == experiment.CONFIG["repeats"]["count"], repeats
    for repeat in repeats:
        assert repeat["limit_s"] == 600.0 and "post_s" not in repeat, repeat
    row = local_runs.run_locally(directory / "repeat", repeats[:1])[repeats[0]["key"]]
    assert row["status"] == "ok" and row["post_status"] is None, row
    assert "certificate_text" not in row["result"], row["result"]
    print(f"  cgoto: {len(repeats)} repeats of limit 600 s; a repeat has no post-processing")


@test
def at5_circuit_data(directory: pathlib.Path, node: bool) -> None:
    """Inputs, their hashes, states, and noise."""
    import stim

    circuits.check_inputs()
    allowed = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    checked = [entry for entry in entries if entry["status"] == "ok"]
    for entry in checked:
        circuit_id = entry["id"]
        p = entry["options"]["noise"]["p"]
        noisy = experiment._noisy_input(entry)
        for instruction in noisy.flattened():
            data = stim.gate_data(instruction.name)
            if data.produces_measurements:
                assert instruction.gate_args_copy() == [p], (circuit_id, str(instruction))
            elif data.is_noisy_gate:
                assert instruction.name in allowed, (circuit_id, instruction.name)
                assert instruction.gate_args_copy() == [p], (circuit_id, str(instruction))
    print(f"  {len(checked)} noisy inputs")


@test
def at6_node_imports(directory: pathlib.Path, node: bool) -> None:
    """The node's modules, all imported by this script, load neither MQT nor qiskit."""
    loaded = [name for name in sys.modules if name.split(".")[0] in ("mqt", "qiskit")]
    assert not loaded, loaded


@test
def at7_brute_force(directory: pathlib.Path, node: bool) -> None:
    """The baseline agrees with the certifier, the counters, and the ground truth."""
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    ids = {e["id"] for e in entries if e["status"] == "ok" and e["ground_truth"]}
    ids |= {"cgoto", "w.cgoto", "echen"}
    violations = 0
    for circuit_id in sorted(ids):
        violations += _check_brute_bz(circuit_id)
    print(f"  brute_bz on {len(ids)} circuits, {violations} with a violation")

    # the arm itself, through the task runner, on E-chen
    task = experiment._task(
        "brute_bz", experiment._entry("echen"), 0, experiment._limit("brute_bz")
    )
    row = local_runs.run_locally(directory / "arm", [task])[task["key"]]
    assert row["status"] == "ok" and row["post_status"] is None, row
    certificate = _certify("echen", "brouwer_zimmermann")
    expected = None if certificate.certified else len(certificate.counterexample.fault_indices)
    assert row["result"]["violation_weight"] == expected, (row["result"], expected)
    assert row["result"]["per_weight"] == row["partial"], (row["result"], row["partial"])
    assert row["result"]["rows"] == sum(record["rows"] for record in row["partial"])

    # brute_force_smallest_violation against find_smallest_violation on the random DEMs of AT2
    outcomes = _check_random_dems()
    assert outcomes["certified"] and outcomes["violation"], outcomes
    print(f"  20 random DEMs: {outcomes}")

    # U, and whether it is within the cap, as circuits.json records them
    cap = experiment.CONFIG["brute_force"]["max_combinations"]
    ok = [entry for entry in entries if entry["status"] == "ok"]
    for entry in ok:
        size = baseline.enumeration_size(entry["N0"], entry["options"]["max_weight"])
        assert entry["U"] == size and entry["brute_force"] == (size <= cap), entry["id"]


@test
def at8_analysis(directory: pathlib.Path, node: bool) -> None:
    """The analysis of synthetic rows: partial counts, disagreements, and missing tasks."""
    # four B-cat circuits with t = 2 and K = 3: the first violates at 2 and its listing count
    # times out; the ground truth of the second and the baseline of the third disagree on a count,
    # and both violate at 3, so the box would prefer either of them over the first if they agreed;
    # the fourth certifies, and its cert_cc task has no row
    first, second, third, fourth = (experiment._entry(f"bcat.w{w}.t2") for w in (10, 11, 12, 13))
    rows: list[dict[str, Any]] = []
    expected: dict[str, dict[str, Any]] = {}

    def add(
        entry: dict[str, Any],
        arm: str,
        status: str,
        T: float | None = None,
        result: dict[str, Any] | None = None,
        partial: list[dict[str, Any]] | None = None,
    ) -> None:
        task = experiment._task(arm, entry, 0, experiment._limit(arm))
        expected[task["key"]] = task
        if status == "missing":
            return
        post_status = "ok" if arm in experiment.METHODS else None
        rows.append(
            {
                "task_key": task["key"],
                "arm": arm,
                "repeat": 0,
                "status": status,
                "post_status": post_status,
                "T": T,
                "peak_rss_gib": 0.5,
                "partial": partial or [],
                "result": result or {},
            }
        )

    def certification(violation_weight: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        weights = [
            {"m": m, "rows": 30, "violation": m == violation_weight, "T_m": 1.0}
            for m in range(2, (violation_weight or 3) + 1)
        ]
        result = {
            "certified": violation_weight is None,
            "violation_weight": violation_weight,
            "zero_flag_checks": {"calls": 10},
            "classification": {"calls": 20},
            "batches": {str(record["m"]): {"calls": 1, "rows": 30} for record in weights},
        }
        return result, weights

    # the certifications and counts of the four circuits
    accepted = {"1": 30, "2": 400, "3": 3000}
    listing: dict[str, Any] = {"Pi": {"1": 1, "2": 5}, "E": {"2": 0, "3": 2}, "Phi": {"2": 80}}
    for entry, violation_weight, T in ((first, 2, 100.0), (second, 3, 1000.0), (third, 3, 2000.0)):
        add(entry, "cert_bz", "ok", T, *certification(violation_weight))
    add(fourth, "cert_bz", "ok", 200.0, *certification(None))
    add(fourth, "cert_cc", "missing")
    for entry in (first, second, third, fourth):
        combinations = {m: math.comb(entry["N0"], int(m)) for m in accepted}
        add(entry, "count_accepted", "ok", 1.0, {"A": accepted, "C": combinations})
    class_parameters = {"q": 4, "q_h": 2, "h": 5}
    add(
        first,
        "count_listing",
        "timeout",
        1800.0,
        partial=[class_parameters, {"m": 2, "Pi": 1, "E": 0, "Phi": 40}],
    )

    # Phi_2 = A_2 on the second and third circuits would raise the listing table's maximum
    for entry, phi in ((second, 400), (third, 400), (fourth, 80)):
        counts = listing | {"Phi": listing["Phi"] | {"2": phi}}
        add(entry, "count_listing", "ok", 1.0, counts | class_parameters | {"K_prime": 3})

    # the ground truth of the second circuit counts one accepted combination more at weight 2
    truth = listing | {"Phi": listing["Phi"] | {"2": 400}, "A": accepted | {"2": 401}}
    add(second, "ground_truth", "ok", 1.0, truth | {"smallest_violation": 3})

    # the baseline of the third circuit enumerates one combination more at weight 1
    for entry, smallest, enumerated, T in ((third, 3, 31, 3000.0), (fourth, None, 30, 300.0)):
        enumerations = [
            {"m": int(m), "combinations": count, "rows": count, "violation": False}
            for m, count in (accepted | {"1": enumerated}).items()
        ]
        enumerations[-1]["violation"] = smallest is not None
        result = {"certified": smallest is None, "violation_weight": smallest, "rows": 3430}
        add(entry, "brute_bz", "ok", T, result, enumerations)

    # analyze the rows, and read the tables
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    by_id = {entry["id"]: entry for entry in entries}
    out = directory / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    analyze.write_outputs(analyze.collect(rows, by_id, expected), by_id, out)
    tables = {}
    for name in ("summary", "per_weight"):
        with open(out / f"{name}.csv") as file:
            tables[name] = list(csv.DictReader(file))

    # the partial listing counts of the first circuit
    summary = {row["id"]: row for row in tables["summary"]}
    row = summary[first["id"]]
    assert (row["q"], row["q_h"], row["h"]) == ("4", "2", "5"), row
    by_weight = {int(row["m"]): row for row in tables["per_weight"] if row["id"] == first["id"]}
    completed = by_weight[2]
    assert (completed["Pi_m_minus_1"], completed["E_m"], completed["Phi_m"]) == ("1", "0", "40")
    assert by_weight[3]["Phi_m"] == "", by_weight[3]

    # the baseline disagrees with any finished certification, here cert_cc alone, and its counts
    # disagree with A_m on its completed weights even if it and every certification timed out
    accepted_counts = {"A": accepted}
    brute = {"status": "ok", "result": {"certified": True, "violation_weight": None}, "partial": []}
    violating = {"status": "ok", "result": {"certified": False, "violation_weight": 3}}
    assert analyze._agree_baseline(brute, [violating], accepted_counts) is False
    assert analyze._agree_baseline(brute, [brute], accepted_counts) is True
    miscounted = {
        "status": "timeout",
        "result": {},
        "partial": [{"m": 1, "combinations": 31, "violation": False}],
    }
    assert analyze._agree_baseline(miscounted, [], accepted_counts) is False
    assert analyze._agree_baseline(dict(miscounted, partial=[]), [], accepted_counts) is None
    matched = dict(miscounted, partial=[{"m": 1, "combinations": 30, "violation": False}])
    assert analyze._agree_baseline(matched, [], accepted_counts) is True
    violating_at_1 = {"status": "ok", "result": {"certified": False, "violation_weight": 1}}
    assert analyze._agree_baseline(matched, [violating_at_1], accepted_counts) is False
    assert analyze._agree_baseline(matched, [violating], accepted_counts) is True
    # a baseline stopped after it recorded a violation, at weight 2, its smallest
    stopped = dict(
        matched,
        partial=[*matched["partial"], {"m": 2, "combinations": 5, "violation": True}],
    )
    violating_at_2 = {"status": "ok", "result": {"certified": False, "violation_weight": 2}}
    assert analyze._agree_baseline(stopped, [violating_at_2], accepted_counts) is True
    assert analyze._agree_baseline(stopped, [violating], accepted_counts) is False
    assert analyze._agree_baseline(stopped, [brute], accepted_counts) is False
    assert analyze._agree_baseline(stopped, [], accepted_counts) is True

    # the disagreements
    assert [summary[e["id"]]["verdict"] for e in (second, third)] == ["disagreement"] * 2, summary
    assert summary[second["id"]]["agree_ground_truth"] == "False", summary[second["id"]]
    assert summary[third["id"]]["agree_baseline"] == "False", summary[third["id"]]

    # the missing task
    assert summary[fourth["id"]]["status_cert_cc"] == "missing", summary[fourth["id"]]

    print("  4 B-cat circuits, 2 disagreements, 1 missing task")


def _check_brute_bz(circuit_id: str) -> bool:
    """Compare brute_bz with cert_bz, the counters, and the ground truth on one circuit.

    The baseline receives the arguments of cert_bz's fault search, and builds one BZ oracle from
    them by the fault search's name.  Return whether the circuit has a violation.
    """
    from qldpc.circuits.certification import fault_propagation

    entry = experiment._entry(circuit_id)
    max_weight = entry["options"]["max_weight"]
    recorded: dict[str, Any] = {}
    original = fault_propagation.find_smallest_violation

    def recorder(*args: Any, **kwargs: Any) -> Any:
        names = ("flags", "syndromes", "stabilizers", "max_weight", "css_weight")
        recorded.update(zip(names + ("memory_limit", "method"), args, strict=True), **kwargs)
        return original(*args, **kwargs)

    fault_propagation.find_smallest_violation = recorder
    try:
        certified = _certify(circuit_id, "brouwer_zimmermann")
    finally:
        fault_propagation.find_smallest_violation = original

    brute = baseline.BruteForce()
    with brute.install(), _OracleSpy() as constructions:
        found = _certify(circuit_id, "brouwer_zimmermann")
    arguments = brute.arguments
    assert arguments is not None
    for name, value in recorded.items():
        assert np.array_equal(arguments[name], value), (circuit_id, name)
    assert constructions == [(recorded["css_weight"], "brouwer_zimmermann")], constructions

    # the verdict and the smallest violating weight
    weights = [None if c.certified else c.counterexample.fault_weight for c in (certified, found)]
    assert weights[0] == weights[1], (circuit_id, weights)
    reduced = counting.reduce_signatures(recorded["flags"], recorded["syndromes"])
    stabilizers = np.array(recorded["stabilizers"], dtype=np.uint8)
    if entry["ground_truth"]:
        truth = counting.brute_force(reduced, stabilizers, max_weight)
        assert truth["smallest_violation"] == weights[1], (circuit_id, truth, weights)
    if not found.certified:
        example = found.counterexample
        flag_sum = syndrome_sum = 0
        for index in example.fault_indices:
            flag_sum ^= recorded["flags"][index]
            syndrome_sum ^= recorded["syndromes"][index]
        assert flag_sum == 0, circuit_id
        assert syndrome_sum == sum(bit << j for j, bit in enumerate(example.output_syndrome))
        assert example.residual_weight_lower_bound > example.fault_weight, circuit_id

    # on each completed weight, every accepted combination is enumerated and queried
    accepted = counting.accepted_counts(reduced.flags, max_weight, 2**22)["A"]
    truth_accepted = truth["A"] if entry["ground_truth"] else accepted
    for record in brute.weights:
        assert record["distinct_syndromes"] <= record["rows"] == record["combinations"], record
        if not record["violation"]:
            m = str(record["m"])
            assert record["combinations"] == accepted[m] == truth_accepted[m], (circuit_id, record)
    expected_weights = list(range(1, (weights[1] or max_weight) + 1))
    assert [record["m"] for record in brute.weights] == expected_weights, brute.weights
    return not found.certified


def _check_random_dems() -> dict[str, int]:
    """Compare the baseline with ``find_smallest_violation`` on the random DEMs of AT2.

    Each DEM gets syndromes of the |0> state of C-goto (Steane's code), mostly of residual weight
    at most 1, and the baseline also runs with batches of 7 rows and with a cap of 3 distinct
    syndromes, which must find the same violation and enumerate the same completed weights.
    Return the number of DEMs certified and with a violation.
    """
    from qldpc.circuits.certification import _fault_search

    reference = _certify("cgoto", "brouwer_zimmermann", max_weight=1)
    stabilizers = experiment.stabilizer_matrix(reference.state_stabilizers)
    table = counting.residual_weight_table(stabilizers)
    light = np.flatnonzero(table <= 1)
    rng = np.random.default_rng(20261116)
    outcomes = {"certified": 0, "violation": 0}
    for flags, _, max_weight in _random_dems():
        syndromes = [
            int(rng.choice(light)) if rng.random() < 0.8 else int(rng.integers(0, len(table)))
            for _ in flags
        ]
        arguments = (flags, syndromes, stabilizers, max_weight, True)
        expected = _fault_search.find_smallest_violation(*arguments, method="brouwer_zimmermann")
        brute = baseline.BruteForce()
        found = brute.find_smallest_violation(*arguments, method="brouwer_zimmermann")
        assert (
            baseline.brute_force_smallest_violation(*arguments, method="brouwer_zimmermann")
            == found
        )
        reduced = counting.reduce_signatures(flags, syndromes)
        truth = counting.brute_force(reduced, stabilizers, max_weight)
        weights = [None if v is None else len(v.fault_indices) for v in (expected, found)]
        assert weights[0] == weights[1] == truth["smallest_violation"], (weights, truth)
        outcomes["certified" if found is None else "violation"] += 1
        if found is not None:
            flag_sum = syndrome_sum = 0
            for index in found.fault_indices:
                flag_sum ^= flags[index]
                syndrome_sum ^= syndromes[index]
            assert flag_sum == 0 and syndrome_sum == found.syndrome, found
            assert table[found.syndrome] > len(found.fault_indices) < found.bound.lower_bound

        accepted = counting.accepted_counts(reduced.flags, max_weight, 2**22)["A"]
        for record in brute.weights:
            if not record["violation"]:
                assert record["combinations"] == record["rows"] == accepted[str(record["m"])]

        # small batches resume the enumeration, and the cap on distinct syndromes holds
        small = baseline.BruteForce(batch_size=7, max_distinct=3)
        assert small.find_smallest_violation(*arguments, method="brouwer_zimmermann") == found
        for record, other in zip(brute.weights, small.weights, strict=True):
            if record["violation"]:
                continue
            assert record["combinations"] == other["combinations"], (record, other)
            if record["distinct_syndromes"] > 3:
                assert other["distinct_syndromes"] is None, other
            else:
                assert other["distinct_syndromes"] == record["distinct_syndromes"], other
    return outcomes


def _random_dems() -> list[tuple[list[int], list[bool], int]]:
    """The 20 random DEMs of AT2 and AT7: flags, heavy flagged mechanisms, and K.

    Each has N_0 <= 25 mechanisms, p <= 8 flag bits, and K <= 4, with repeated flags so that
    classes have several members.
    """
    rng = np.random.default_rng(20261115)
    dems = []
    for _ in range(20):
        num_flag_bits = int(rng.integers(2, 9))
        size = int(rng.integers(5, 26))
        max_weight = int(rng.integers(2, 5))
        flags = [
            int(rng.integers(1, 2**num_flag_bits)) if rng.random() > 0.15 else 0
            for _ in range(size)
        ]
        for i in range(1, size):
            if rng.random() < 0.4:
                flags[i] = flags[int(rng.integers(0, i))]
        flagged = [flag for flag in flags if flag]
        heavy = [bool(rng.random() < 0.35) for _ in flagged]
        dems.append((flags, heavy, max_weight))
    return dems


def _compare_counts(
    flags: list[int], heavy: list[bool], max_weight: int, truth: dict[str, Any] | None = None
) -> None:
    """Compare the counters with brute force on one DEM, and with a buffer of one codeword.

    The listing counter runs up to K' = min(K, r + 1), and the accepted counter up to K.
    """
    flagged = [flag for flag in flags if flag]
    classes = counting.get_classes(flagged, heavy)
    weight_limit = min(max_weight, counting.gf2_rank(flags) + 1)
    listing = counting.listing_counts(classes, weight_limit)
    accepted = counting.accepted_counts(flags, max_weight, 2**22)
    truth = truth or counting.brute_force_counts(flags, heavy, max_weight)
    for key in ("Pi", "E", "Phi"):
        assert listing[key] == truth[key], (key, listing[key], truth[key])
    assert accepted["A"] == truth["A"], (accepted["A"], truth["A"])
    assert set(accepted["A"]) == {str(m) for m in range(1, max_weight + 1)}, accepted["A"]
    assert counting.listing_counts(classes, weight_limit, buffer_size=1) == listing


def _spy_calls(circuit_id: str) -> int:
    """The oracle calls made directly by the fault search, counted by inspecting the caller."""
    from qldpc.circuits.certification import _fault_search

    original = _fault_search.CosetLeaderSearch
    calls = [0]

    class Spy(original):  # type: ignore[valid-type, misc]
        def get_min_weight_bound(self, *args: Any, **kwargs: Any) -> Any:
            calls[0] += _called_from_search()
            return super().get_min_weight_bound(*args, **kwargs)

        def find_first_not_within_max_weight(self, *args: Any, **kwargs: Any) -> Any:
            calls[0] += _called_from_search()
            return super().find_first_not_within_max_weight(*args, **kwargs)

    _fault_search.CosetLeaderSearch = Spy  # type: ignore[misc]
    try:
        _certify(circuit_id, "brouwer_zimmermann")
    finally:
        _fault_search.CosetLeaderSearch = original  # type: ignore[misc]
    return calls[0]


def _called_from_search() -> int:
    """1 if the caller of the spied method is a function of the fault search, else 0."""
    frame = sys._getframe(2)
    return int(frame.f_globals.get("__name__") == "qldpc.circuits.certification._fault_search")


class _OracleSpy:
    """Record (css_weight, method) of every oracle that the fault search's name constructs."""

    def __enter__(self) -> list[tuple[bool, str]]:
        from qldpc.circuits.certification import _fault_search

        self.original = _fault_search.CosetLeaderSearch
        constructions: list[tuple[bool, str]] = []

        class Spy(self.original):  # type: ignore[name-defined]
            def __init__(self, stabilizers: Any, **kwargs: Any) -> None:
                constructions.append((kwargs["css_weight"], kwargs["method"]))
                super().__init__(stabilizers, **kwargs)

        _fault_search.CosetLeaderSearch = Spy  # type: ignore[misc]
        return constructions

    def __exit__(self, *args: object) -> None:
        from qldpc.circuits.certification import _fault_search

        _fault_search.CosetLeaderSearch = self.original  # type: ignore[misc]


class _ShiftedSearch(CosetLeaderSearch):
    """An oracle whose answer to each batch is one row later than the true one."""

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        index, bound = super().find_first_not_within_max_weight(syndromes, max_weight)
        return index + 1, bound


class _ShiftedShadow(instrument.Instrumentation):
    """Instrumentation whose shadow oracle shifts its answers."""

    shadow_search = _ShiftedSearch


if __name__ == "__main__":
    main()
