# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests of the certification experiment.

    python acceptance.py [--local] [--only AT1 AT3 ...] [--results-dir R]

* **AT1:** C-goto certifies at K = 2; its weakened variant, which deletes the flag, violates at
  weight 1; on E-chen, BZ and CC give the same verdict and smallest violating weight at K = 3.
* **AT2:** the listing counter (up to K') and the accepted-combination counter (up to K) equal brute
  force on the ground-truth circuits of the d = 3 A-ft codes and on 20 random DEMs (N_0 <= 25,
  p <= 8, K <= 4), including DEMs where a light class precedes a heavy one in DEM order; the listing
  counter gives the same counts when its compiled search returns one class codeword at a time; the
  accepted-combination counter gives no A_m with a cap of no entries, even without flagged
  mechanisms; and the rows submitted to the oracle never exceed the listed faults on completed
  weights.  Through the task runner, ``count_listing`` on E-chen and ``ground_truth`` on W(A-ft
  Shor) and C-goto finish without ``compiled_in_timing``.  The compiled functions of ``counting.py``
  and ``baseline.py`` call by name no compiled function of another file.
* **AT3:** instrumented and uninstrumented certification give identical certificates on C-goto,
  its variant, and A-ft steane; every wrapped name exists; an instrumentation installed around
  another wrapper of the oracle class subclasses it and restores it; on a CSS circuit with a
  violation, each query of the fault search is counted once, against an independent spy of the
  calls that the fault search makes.  On E-chen, the shadow oracle answers each query that the
  counters count, with one threshold-1 query per reduced zero-flag mechanism and one per distinct
  syndrome of the flagged ones, and an oracle whose answers are shifted raises
  ``ShadowDisagreement``.
* **AT4:** ``cert_bz`` is stopped by its limit after it completes a fault weight.  It runs on a
  B-cat circuit with t = 2 and K = 3, and sleeps 2 s after it reports each completed weight
  (``weight_delay_s``).  The limit falls 1 s after the end of the next-to-last weight of a full run,
  which compiles nothing in its measured call.  The stopped run ends as a timeout with T at least
  the limit, reports its completed weights as consecutive weights from 2 without a violation, and is
  not rerun when its task is run again.
  A ``cert_bz`` run on the weakened C-goto, which violates, gets a post-processing allowance of
  1 s, and its description is delayed by 5 s (``post_delay_s``).  The run keeps status ok, T, and
  its verdict, and its ``post_status`` is ``timeout``.  A fast ``cert_bz`` run on C-goto gets
  exactly ``repeats.count`` repeats (``repeat_tasks``), each with a limit of 600 s and no
  ``post_s``, and a repeat ends without post-processing or certificate text.
* **AT5:** every input matches its SHA-256 and prepares a CSS state, its recomputed DEM
  parameters, prepared state, and output qubits equal those of ``circuits.json``, and each code
  file passes ``code_files.check_code_file``: the checks of ``circuits.py --check``.  A copy of
  E-chen's record with another state, other output qubits, ``css_weight`` false, or ``css_weight``
  set in its options fails exactly the check of that field.  A copy of the record of an A-ft
  circuit with another t, n, and noise fails exactly the checks of K = t + 1, n, t against the
  distance of its code file, and the noise.
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
* **AT8:** the analysis (``analyze.py``) of synthetic rows on circuits of ``circuits.json``: six
  B-cat circuits through every output, and others for the cases below.  A ``count_listing`` run that
  timed out after one completed weight gives Pi, E, Phi, q, q_h, and h from its partial results.  A
  ground-truth count that differs from the counters', and a baseline whose enumerated combinations
  differ from A_m, or exceed it on a weight where it found a violation, in a first run or a repeat,
  and of ``count_accepted`` or else of the ground truth, each make their circuit a disagreement, as
  does a baseline stopped after a violation at weight m against a finished certification whose
  smallest violation is not at m.  An expected task without a row is reported as ``missing``.  Rows
  with ``compiled_in_timing`` set give the statuses, verdict, counts of ``families.csv``, and times
  that ``analyze.py``'s docstring gives them: the ``cert_bz`` and ``brute_bz`` timeouts of one
  circuit, whose ``cert_cc`` timeout is unflagged, against another circuit's plain timeouts; a
  repeat; and a first run that finished.  With two of the synthetic rows marked as rows of
  ``--local`` runs, ``analyze.py`` refuses the label without ``--allow-local``; with it, every
  output gives the number of these rows, and the name of the refused environment variable that a
  start, here a dry run, accepted with ``--allow-env``.  The analysis of the label's archive
  requires ``--out``, and leaves no extraction of the archive, or of the archive of ``check``,
  behind, whether it succeeds or refuses the rows.  ``analyze.py`` warns of a label whose hash does
  not match its ``circuits.json``, the working tree's other frozen files, and the commit and
  requirements of its manifest, and of no other.  W(C-goto) and a D-synth (MQT 1.9.0) circuit give
  the example counterexamples of their groups.  ``listing_table.csv`` takes C(N_0, m) from the
  circuit's N_0 in place of an A_m that was not computed where z >= m, and gives no ratio where z <
  m.  ``shadow.csv`` gives the totals of a finished shadow run by threshold, and the calls that a
  timed-out run wrote, without a last line cut off, and leaves out a run with ``compiled_in_timing``
  set and a disagreement.  A calls file with a malformed line before its last is an error.  Each
  kind of contradiction that ``analyze.check_consistency`` lists makes its circuit a disagreement: a
  shadow run that ended in ``ShadowDisagreement``, also one that the runner discarded because it
  compiled, a finished shadow run, a repeat, a deletion parent, the final m* of a deletion task with
  no accepted deletion, the ground truth, the baseline, a violation recorded by an unfinished
  certification, or the result of a certification that exceeded its memory cap after it returned,
  against another exact result; a final circuit whose m* is at most t, above the weight of a
  violation that the search of its m* reported, or other than that of another deletion task that
  ends at the same final circuit; Phi_m of the listing or the ground truth above A_m; Omega_m above
  Phi_m, of the listing or the ground truth, on a completed weight or one where the run found a
  violation; the ground truth's heavy mechanisms against h, also where the ground truth exceeded its
  memory cap after it returned; and a counterexample's residual weight against its fault weight or
  lower bound.  A circuit whose claims of every kind agree gets their m*.  Every case of the
  combined margin of ``analyze_applications.py`` gives its value, its bound, or a disagreement, and
  a margin of at most t keeps the arm and sector of its counterexample.  A margin or a deletion
  parent that disagrees with ``cert_bz``, or a final circuit that CC does not certify, makes its
  circuit a disagreement, and agreeing deletion tasks do not.  A disagreeing parent does so also
  when its verdict is only in the partial results of a killed task.  The margin counts of one family
  are checked, and ``margin_violations.csv`` lists m* = t, but not m* = t + 1 or a disagreement.
  The deletion outputs rebuild a reduced circuit from its input, report a wrong final SHA-256, leave
  the rebuild unchecked for an input that is not the label's, exclude a disagreement from
  ``deletion_table.csv`` but list its task, list an uncertified parent, give a timeout and a flagged
  timeout their statuses, and keep the rows of two tasks killed during the search of their final m*:
  with m* > k for the largest fault weight k that the search completed, and with m* where it found a
  violation.  ``deletion_table.csv`` lists its circuits in the order of their ids, and an analysis's
  marks head each circuit of ``pruned/``, which still parses.
* **AT9:** the deletion orders and gate counts of a small circuit equal those written by hand.  An
  acceptance estimate is within five standard errors of the exact probability from the DEM.
  Candidate circuits whose prepared state changed, whether a detector breaks or the certifier
  accepts their detectors (a flipped sign, a rotated qubit), and one with an added random
  measurement, are rejected for those reasons.  ``prune`` runs twice through the task runner on
  C-goto, steane, and ``cc_4_8_8.heuristic_opt``, with the same steps, in its order, sent as partial
  results, and then its final checks, with the final circuit, and the fault weights of the search of
  its final m*.  Each step and each final circuit, rebuilt, has its recorded outcome: by Stim's
  simulator, by certification, and, where small enough, by the enumeration of the ground truth.  The
  final m* agrees with ``cert_bz`` at K = t + 1, and the acceptance estimates of parent and final
  circuit agree with the exact probabilities from their DEMs.  The enumeration also checks every
  step on the d = 3 A-ft circuits that pass at K = t.  C-goto keeps its flag.  A run whose steps are
  delayed past its 5 s limit (``step_delay_s``) is a timeout that keeps its steps.
* **AT10:** the margin gives "all" on C-goto and steane, 3 on E-chen, and 1 on W(C-goto).  On these
  and every ground-truth circuit with r > t, it agrees with ``cert_bz``, with the enumeration up to
  min(m*, r + 1) where small enough, and its sectors with an enumeration of the stabilizer group.
  A run on E-chen delayed past its 5 s limit is a timeout that gives only a lower bound.
* **AT11:** altered copies of code files (an Lx that anticommutes with a check, a check as Lz, no
  Lx, another k, and the d = 5 surface code without its exchange of X and Z) each fail the check
  that names it, and the code file of a one-qubit |0>, which has no X checks, passes.

The expected values of AT1 on C-goto and its weakened variant come from qLDPC: the notebook
``examples/logical_error_rates/5_fault_tolerance_certificates.ipynb`` certifies C-goto's circuit up
to any fault weight when it post-selects on the flag, and finds a violation at fault weight 1 when
it does not, and ``test_steane_preparation_meets_css_weight_but_not_physical_weight`` of
``src/qldpc/circuits/certification/fault_propagation_test.py`` certifies the circuit with its flag.
A failure of AT1 means that the experiment disagrees with these results: find the cause, rather than
change a rule of the experiment until AT1 passes.  Each test prints PASS or FAIL; the script exits
with status 1 if any failed.  The tests are the same on the node and on a workstation: ``--local``
and the ``node`` argument of each test exist because the plug-in interface passes whether the tests
run on the node.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np
import numpy.typing as npt

from qldpc import circuits as qldpc_circuits
from qldpc.codes.distance import CosetLeaderSearch, CosetMethod, MinWeightBound

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import (
    analyze,
    analyze_applications,
    applications,
    baseline,
    circuits,
    code_files,
    counting,
    experiment,
    instrument,
)
from experiments.state_prep_certification.common import acceptance, local_runs, manifest, store

SUITE = acceptance.Suite("spc-cert-")
test = SUITE.test
run_all = SUITE.run_all


def _certify(circuit_id: str, method: CosetMethod, max_weight: int | None = None) -> Any:
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
    """Counters equal brute force; two arms run through the runner without compiled_in_timing."""
    # the compiled functions of counting.py and baseline.py call only compiled functions of their
    # own file, as the cache of their compiled code requires (see counting.py)
    for module in (counting, baseline):
        compiled = [
            value
            for value in vars(module).values()
            if getattr(getattr(value, "py_func", None), "__module__", None) == module.__name__
        ]
        assert compiled, module.__name__
        for function in compiled:
            for name in function.py_func.__code__.co_names:
                called = getattr(function.py_func.__globals__.get(name), "py_func", None)
                assert called is None or called.__module__ == module.__name__, (function, name)

    # random DEMs, some with a light class before a heavy one in DEM order
    light_first = 0
    for flags, heavy, max_weight in _random_dems():
        light_first += bool(heavy and not heavy[0] and any(heavy))
        _compare_counts(flags, heavy, max_weight)
    assert light_first >= 3, light_first

    # the accepted counter honors a cap that its initial map exceeds, without flagged mechanisms
    assert counting.accepted_counts([0], 1, max_entries=0)["A"] is None

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

    # three tasks through the runner, whose rows run_locally checks for compiled_in_timing.  The
    # listing count on E-chen reaches the disk-cached search of counting.py.  The ground truth uses
    # galois's arithmetic, which counting._compile_galois compiles first, and on C-goto it also
    # reaches, at fault weight 2, more of counting.py's disk-cached functions.
    for arm, circuit_id in (
        ("count_listing", "echen"),
        ("ground_truth", "w.aft.shor.opt_opt"),
        ("ground_truth", "cgoto"),
    ):
        task = experiment._task(arm, experiment._entry(circuit_id), 0, experiment._limit(arm))
        row = local_runs.run_locally(directory / task["key"], [task])[task["key"]]
        assert row["status"] == "ok", (task["key"], row["status"])


@test
def at3_instrumentation(directory: pathlib.Path, node: bool) -> None:
    """Instrumentation changes no certificate, counts each query once, and shadows each query."""
    from qldpc.circuits.certification import _fault_search, fault_propagation

    assert hasattr(fault_propagation, "find_smallest_violation")
    for name in ("CosetLeaderSearch", "FaultLister", "_check_fault"):
        assert hasattr(_fault_search, name), name
    assert "_max_syndromes" in _fault_search.FaultLister.__init__.__code__.co_names

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
        assert _fault_search.CosetLeaderSearch is CosetLeaderSearch

    # an instrumentation around another wrapper of the oracle keeps that wrapper, and restores it
    with _OracleSpy():
        wrapper = _fault_search.CosetLeaderSearch
        with instrument.Instrumentation().install():
            assert issubclass(_fault_search.CosetLeaderSearch, wrapper)
        assert _fault_search.CosetLeaderSearch is wrapper

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
    # the search classifies each distinct syndrome of the flagged mechanisms once, and E-chen's
    # flagged mechanisms repeat syndromes, so this count differs from one query per mechanism
    flagged_syndromes = [
        syndrome for flag, syndrome in zip(reduced.flags, reduced.syndromes, strict=True) if flag
    ]
    assert len(set(flagged_syndromes)) < len(flagged_syndromes), len(flagged_syndromes)
    assert counted_by_kind["classification:1"] == len(set(flagged_syndromes)), counted_by_kind
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
    """Limits, the post-processing allowance, repeats, and resume."""
    # a circuit with t = 2, so that weight 2 has no violation, and K = 3, so that the full run
    # reports two weights.  Each run sleeps after reporting each weight, and the limit falls in the
    # middle of the sleep after the next-to-last weight.  The runner checks the limit every 0.2 s or
    # so, a small part of the 1 s of sleep that remains after the limit, so that it kills the run
    # before the run reports another weight.  The full run compiles nothing in its measured call,
    # since run_locally fails if it does.
    circuit, delay = experiment._entry("bcat.w13.t2"), 2.0
    task = experiment._task("cert_bz", circuit, 0, experiment._limit("cert_bz"))
    task["weight_delay_s"] = delay
    full = local_runs.run_locally(directory / "full", [task])[task["key"]]
    assert full["status"] == "ok" and full["T"] is not None, full
    ends = [record["T_cumulative"] for record in full["partial"]]
    assert len(ends) >= 2 and None not in ends, full["partial"]
    limit = ends[-2] + delay / 2
    task = experiment._task("cert_bz", circuit, 0, limit)
    task["weight_delay_s"] = delay
    row = local_runs.run_locally(directory / circuit["id"], [task])[task["key"]]
    weights = [record["m"] for record in row["partial"]]
    assert row["status"] == "timeout" and row["T"] >= limit, row
    assert weights and weights[-1] < full["partial"][-1]["m"], (weights, full["partial"])

    # the partial results are the completed weights, in order from 2, each without a violation
    assert weights == list(range(2, 2 + len(weights))), weights
    assert all(not record["violation"] for record in row["partial"]), row["partial"]
    assert all(record["T_cumulative"] <= row["T"] for record in row["partial"]), row["partial"]
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

    # a record with one field changed fails that field's check alone; E-chen has no code file, whose
    # check would also fail on another state
    entry = next(entry for entry in entries if entry["id"] == "echen")
    assert circuits._check_input(entry) == [], entry
    changes = {
        "other than the recorded": {
            "stabilizers": ["-" + entry["stabilizers"][0][1:], *entry["stabilizers"][1:]]
        },
        "output_qubits is": {"output_qubits": entry["output_qubits"][::-1]},
        "css_weight is": {"css_weight": False},
        "options set css_weight": {"options": entry["options"] | {"css_weight": True}},
    }
    for message, change in changes.items():
        problems = circuits._check_input(entry | change)
        assert len(problems) == 1 and message in problems[0], (message, problems)

    # a record with a code file whose K, n, t, and noise disagree fails exactly these four checks
    steane = next(entry for entry in entries if entry["id"] == "aft.steane.heuristic_opt")
    altered = steane | {
        "t": 2,
        "code": [8, *steane["code"][1:]],
        "options": steane["options"] | {"noise": steane["options"]["noise"] | {"p": 0.002}},
    }
    problems = circuits._check_input(altered)
    messages = ("max_weight is 2, not t + 1", "noise is not", "code's n is 8", "t is 2, not")
    assert len(problems) == len(messages), problems
    assert all(any(text in problem for problem in problems) for text in messages), problems
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
    """Analysis of synthetic rows: partial counts, disagreements, missing tasks, flagged rows."""
    # six B-cat circuits with t = 2 and K = 3: the first violates at 2 and its listing count
    # times out; the ground truth of the second and the baseline of the third disagree on a count,
    # and both violate at 3, so the example counterexample of their group would be either of them,
    # not the first, if they agreed; the fourth certifies, and its cert_cc task has no row.  The
    # certifications and baselines of the fifth and sixth time out, and the rows of the fifth's
    # cert_bz and brute_bz runs have compiled_in_timing set (rows of retries that compiled too).
    first, second, third, fourth, fifth, sixth = (
        experiment._entry(f"bcat.w{w}.t2") for w in (10, 11, 12, 13, 14, 15)
    )
    rows: list[dict[str, Any]] = []
    expected: dict[str, dict[str, Any]] = {}

    def add(
        entry: dict[str, Any],
        arm: str,
        status: str,
        T: float | None = None,
        result: dict[str, Any] | None = None,
        partial: list[dict[str, Any]] | None = None,
        compiled: bool = False,
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
                "compiled_in_timing": compiled,
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

    # the certifications and counts of the first four circuits
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

    # Phi_2 = A_2 on the second and third circuits would raise listing_table.csv's maximum
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

    # the timeouts of the fifth and sixth circuits
    for arm in ("cert_bz", "cert_cc", "brute_bz"):
        add(fifth, arm, "timeout", 1800.0, compiled=arm != "cert_cc")
        add(sixth, arm, "timeout", 1800.0)

    # the shadow runs: finished on the first circuit, a disagreement, and the fifth, whose row has
    # compiled_in_timing set, and timed out on the fourth after it wrote two calls and part of a
    # third
    seconds = {"bz_seconds": 1.0, "cc_seconds": 2.0}
    by_threshold = {
        "zero_flag:1": {"calls": 10, "rows": 10} | seconds,
        "classification:1": {"calls": 20, "rows": 20} | seconds,
        "batch:2": {"calls": 1, "rows": 30} | seconds,
    }
    shadow_runs: list[tuple[dict[str, Any], int | None]] = [(first, 2), (second, 3), (fifth, None)]
    for entry, shadow_weight in shadow_runs:
        result = certification(shadow_weight)[0] | {"shadow": {"by_threshold": by_threshold}}
        add(entry, "shadow", "ok", 10.0, result, compiled=entry is fifth)
    add(fourth, "shadow", "timeout", 3600.0)
    calls_dir = directory / "oracle_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    call = json.dumps({"kind": "batch", "m": 2, "rows": 7, "bz": 0.5, "cc": 0.25})
    shadow_key = experiment._task("shadow", fourth, 0, experiment._limit("shadow"))["key"]
    (calls_dir / f"{shadow_key}.jsonl").write_text(f"{call}\n{call}\n{call[:20]}")

    # analyze the rows, and read summary.csv, per_weight.csv, and families.csv
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    by_id = {entry["id"]: entry for entry in entries}
    out = directory / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    analyze.write_outputs(analyze.collect(rows, by_id, expected), by_id, out, calls_dir=calls_dir)
    tables = {}
    for name in ("summary", "per_weight", "families"):
        with open(out / f"{name}.csv") as file:
            tables[name] = list(csv.DictReader(file))

    # the partial listing counts of the first circuit
    summary = {row["id"]: row for row in tables["summary"]}
    row = summary[first["id"]]
    assert (row["q"], row["q_h"], row["h"]) == ("4", "2", "5"), row
    by_weight = {int(row["m"]): row for row in tables["per_weight"] if row["id"] == first["id"]}
    completed = by_weight[2]
    assert (completed["naive_prefixes_m_minus_1"], completed["E_m"], completed["Phi_m"]) == (
        "1",
        "0",
        "40",
    )
    assert by_weight[3]["Phi_m"] == "", by_weight[3]

    # the baseline disagrees with any finished certification, here cert_cc alone, and its counts
    # disagree with A_m on its completed weights even if it and every certification timed out
    brute = {"status": "ok", "result": {"certified": True, "violation_weight": None}, "partial": []}
    violating = {"status": "ok", "result": {"certified": False, "violation_weight": 3}}
    assert analyze._agree_baseline(brute, [violating], accepted) is False
    assert analyze._agree_baseline(brute, [brute], accepted) is True
    miscounted = {
        "status": "timeout",
        "result": {},
        "partial": [{"m": 1, "combinations": 31, "violation": False}],
    }
    assert analyze._agree_baseline(miscounted, [], accepted) is False
    assert analyze._agree_baseline(dict(miscounted, partial=[]), [], accepted) is None
    matched = dict(miscounted, partial=[{"m": 1, "combinations": 30, "violation": False}])
    assert analyze._agree_baseline(matched, [], accepted) is True
    violating_at_1 = {"status": "ok", "result": {"certified": False, "violation_weight": 1}}
    assert analyze._agree_baseline(matched, [violating_at_1], accepted) is False
    assert analyze._agree_baseline(matched, [violating], accepted) is True
    # a baseline stopped after it recorded a violation, at weight 2, its smallest
    stopped = dict(
        matched,
        partial=[*matched["partial"], {"m": 2, "combinations": 5, "violation": True}],
    )
    violating_at_2 = {"status": "ok", "result": {"certified": False, "violation_weight": 2}}
    assert analyze._agree_baseline(stopped, [violating_at_2], accepted) is True
    assert analyze._agree_baseline(stopped, [violating], accepted) is False
    assert analyze._agree_baseline(stopped, [brute], accepted) is False
    assert analyze._agree_baseline(stopped, [], accepted) is True

    # the disagreements, whose counterexamples are listed but are not the group's example
    assert [summary[e["id"]]["verdict"] for e in (second, third)] == ["disagreement"] * 2, summary
    counterexamples = json.loads((out / "counterexamples.json").read_text())
    assert counterexamples["examples"] == {"B-cat": first["id"]}, counterexamples["examples"]
    listed = [(entry["id"], entry["disagreement"]) for entry in counterexamples["all"]]
    assert listed == [(first["id"], False), (second["id"], True), (third["id"], True)], listed

    # the example counterexamples of W(C-goto) and of a D-synth (MQT 1.9.0) circuit
    violation = {"certified": False, "violation_weight": 1}
    example_run = {"repeat": 0, "status": "ok", "post_status": "ok", "T": 1.0, "peak_rss_gib": 0.5}
    examples = {
        circuit_id: {
            "circuit": by_id[circuit_id],
            "arms": {"cert_bz": [example_run | {"partial": [], "result": violation}]},
        }
        for circuit_id in ("w.cgoto", "dsynth190.bb72.naive")
    }
    verdicts = dict.fromkeys(examples, "violation at 1")
    (directory / "examples").mkdir()
    path = analyze.write_counterexamples(examples, verdicts, by_id, directory / "examples")
    assert json.loads(path.read_text())["examples"] == {
        "A-unv/W": "w.cgoto",
        "D-synth/W": "dsynth190.bb72.naive",
    }
    assert summary[second["id"]]["agree_ground_truth"] == "False", summary[second["id"]]
    assert summary[third["id"]]["agree_baseline"] == "False", summary[third["id"]]

    # shadow.csv, without the disagreement and the run with compiled_in_timing set
    with open(out / "shadow.csv") as file:
        columns = ("id", "status", "m", "calls", "rows", "bz_seconds", "cc_seconds")
        shadow = [tuple(row[column] for column in columns) for row in csv.DictReader(file)]
    assert shadow == [
        (first["id"], "ok", "1", "30", "30", "2.0", "4.0"),
        (first["id"], "ok", "2", "1", "30", "1.0", "2.0"),
        (fourth["id"], "timeout", "2", "2", "14", "1.0", "0.5"),
    ], shadow
    # only the last line of a calls file may be cut off; a malformed line before it is an error
    (calls_dir / "interior.jsonl").write_text(f"{call}\n{call[:20]}\n{call}\n")
    try:
        analyze._read_calls(calls_dir, "interior")
    except ValueError as error:
        assert "interior.jsonl" in str(error) and "line 2" in str(error), error
    else:
        raise AssertionError("a calls file with a malformed line before its last was read")

    # the missing task
    assert summary[fourth["id"]]["status_cert_cc"] == "missing", summary[fourth["id"]]

    # the fifth circuit's flagged timeouts and the sixth's plain ones (see analyze.py's docstring)
    row = summary[fifth["id"]]
    statuses = (row["status_cert_bz"], row["status_brute_bz"], row["verdict_bz"], row["verdict"])
    assert statuses == ("timeout_compiled",) * 4, row
    assert (summary[sixth["id"]]["status_cert_bz"], summary[sixth["id"]]["verdict"]) == (
        "timeout",
        "undecided",
    )
    (family,) = tables["families"]
    keys = ("timeout_compiled_cert_bz", "timeout_cert_bz", "timeout_compiled", "undecided")
    assert [family[key] for key in (*keys, "brute_timed_out")] == ["1"] * 5, family

    # a flagged repeat, and a flagged first run that finished
    result, weights = certification(None)
    repeat_flagged, first_flagged = (
        analyze.summarize_certification(
            [
                {"repeat": r, "status": "ok", "T": 1.0, "peak_rss_gib": 0.5, "partial": weights}
                | {"result": result, "compiled_in_timing": r == compiled}
                for r in range(analyze.REPEATS["count"] + 1)
            ]
        )
        for compiled in (1, 0)
    )
    assert repeat_flagged["T"] is None and repeat_flagged["repeats_not_ok"] == ["r1:compiled"]
    assert first_flagged["T"] is None and first_flagged["status"] == "ok", first_flagged
    (_, weight) = analyze._per_weight(
        "x", {"options": {"max_weight": 2}}, first_flagged, repeat_flagged, first_flagged, {}, {}
    )
    assert weight["T_m_bz"] is None and weight["T_m_cc"] == 1.0, weight

    # listing_table.csv: C(N_0, m) from the circuit's N_0 in place of an A_m that count_accepted
    # did not compute, where z >= m, and no ratio where z < m, since A_m may then be 0
    circuit = {"options": {"max_weight": 2}, "N0": 10}
    runs = (first_flagged, first_flagged, first_flagged)
    (_, weight) = analyze._per_weight("x", circuit, *runs, {"Phi": {"2": 9}}, {})
    assert weight["C_N0_m"] == 45, weight
    parameters = {"family": "F", "verdict": "certified", "N0": 10, "h": 1, "q_h": 1, "r": 1}
    circuit_rows = [{"id": "x", "z": 2} | parameters, {"id": "y", "z": 1} | parameters]
    (family_row,) = analyze.listing_table(circuit_rows, [weight, weight | {"id": "y"}])
    ratios = (family_row["circuits"], family_row["ratio_unknown"], family_row["C_in_place_of_A"])
    assert ratios == (1, 1, 1) and family_row["phi_over_A_max"] == 0.2, family_row

    # with two of the synthetic rows marked as rows of --local runs, the analysis refuses the label
    # unless given --allow-local, and then every output gives their number; it also names, in
    # every output, the refused environment variable that a start accepted with --allow-env, here a
    # dry run, which can run the acceptance tests whose pass later starts reuse, and the recorded
    # variable that a resume changed
    label_dir = directory / "label"
    label_dir.mkdir()
    (label_dir / "circuits.json").write_bytes(circuits.CIRCUITS_JSON.read_bytes())
    for number, row in enumerate(rows):
        store.append_row(label_dir, dict(row, local=number < 2))
    start = {
        "allowed_environment": ["PYTHONPATH"],
        "environment": {"PYTHONPATH": "/elsewhere"},
        "dry_run": True,
    }
    resume = {"environment": {"PYTHONPATH": "/elsewhere", "NUMBA_OPT": "0"}}
    (label_dir / store.STARTS_FILE).write_text(json.dumps(start) + "\n" + json.dumps(resume) + "\n")

    # the label check: a label made with the working tree's frozen files, of which the label's
    # circuits.json is a copy, and a bundle's requirements matches; any other label warns
    frozen = [experiment.HERE / name for name in experiment.FROZEN_FILES]
    commit, requirements = "c" * 40, "0" * 64
    label = manifest.compute_label(
        experiment.NAME, manifest.label_files(frozen), commit, requirements
    )
    for name, warns in ((label[:-1] + ("1" if label[-1] == "0" else "0"), True), (label, False)):
        record = {
            "label": f"{name}-local",
            "harness": {"commit": commit},
            "bundle": {"requirements_sha256": requirements},
        }
        (label_dir / "manifest.json").write_text(json.dumps(record))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            analyze.check_frozen_files(label_dir)
        assert ("does not match" in output.getvalue()) == warns, (name, output.getvalue())
    earlier = label_dir / "analysis" / "earlier"
    earlier.mkdir(parents=True)
    (earlier / "families.csv").write_text("id\nx\n")
    (earlier / "notes.json").write_text("[]\n")
    command = [sys.executable, analyze.__file__, "--label-dir", str(label_dir)]
    analysis = subprocess.run(command, capture_output=True, text=True, check=False)
    assert analysis.returncode != 0 and "--allow-local" in analysis.stderr, analysis.stderr
    analysis = subprocess.run(
        command + ["--allow-local"], capture_output=True, text=True, check=False
    )
    assert analysis.returncode == 0, analysis.stderr
    assert "settings that a node start refuses: PYTHONPATH;" in analysis.stdout, analysis.stdout
    assert "different NUMBA_OPT" in analysis.stdout, analysis.stdout
    assert "warning" not in analysis.stdout, analysis.stdout
    # the analysis marks only the files that it wrote, not those of an earlier analysis beside them
    assert (earlier / "families.csv").read_text() == "id\nx\n"
    assert (earlier / "notes.json").read_text() == "[]\n"
    shutil.rmtree(earlier)
    marks = {
        "local_rows": 2,
        "allowed_environment": "PYTHONPATH",
        "environment_changed": "NUMBA_OPT",
    }
    others = acceptance.check_marked(label_dir / "analysis", marks)
    assert not others, others  # no deletion run, so no circuit in pruned/

    # an archive needs --out; its extraction, and that of the archive of check, is removed whether
    # or not the analysis succeeds
    archive = directory / "label.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(label_dir, arcname=label_dir.name)
    scratch = directory / "tmp"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    command = [sys.executable, analyze.__file__, "--archive", str(archive)]
    analysis = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    assert analysis.returncode != 0 and "--out" in analysis.stderr, analysis.stderr
    command += ["--check-archive", str(archive), "--out", str(directory / "from_archive")]
    for allow_local in ([], ["--allow-local"]):
        analysis = subprocess.run(
            command + allow_local, capture_output=True, text=True, check=False, env=environment
        )
        assert (analysis.returncode == 0) == bool(allow_local), analysis.stderr
        assert not any(scratch.iterdir()), list(scratch.iterdir())

    _check_consistency()
    _check_application_analysis(directory / "applications")
    print("  6 B-cat circuits, 2 disagreements, 1 missing task, 1 circuit of flagged timeouts")


@test
def at9_prune(directory: pathlib.Path, node: bool) -> None:
    """The deletion arm against hand counts, the DEM, and the enumeration, and its timeout."""
    import stim

    # both orders and the counts of a circuit with a repeated and a two-target measurement, whose
    # tie is broken against target order, and one encoder and one verification gate, whose measured
    # qubit is its second target
    circuit = stim.Circuit("R 0 1 2 3 4 5\nCX 1 5 1 3\nM 3\nMX 4 2\nM 0 2")
    assert applications.measured_qubits(circuit, "last_measured_first") == [2, 0, 4, 3]
    assert applications.measured_qubits(circuit, "first_measured_first") == [3, 4, 2, 0]
    assert applications.count_gates(circuit) == {
        "measured_qubits": 4,
        "measurements": 5,
        "encoder_gates": 1,
        "verification_gates": 1,
    }

    # an acceptance estimate against the exact probability from the DEM, with five flags
    certificate = _certify("aft.carbon.heuristic_opt", "brouwer_zimmermann")
    estimate = applications.estimate_acceptance(certificate, 200_000, 1)
    exact = _exact_acceptance(certificate)
    assert abs(estimate["probability"] - exact) <= 5 * estimate["standard_error"], (estimate, exact)

    # a deletion that changes the prepared state, and one that adds a random measurement
    goto = experiment._entry("cgoto")
    options = experiment.certify_options(goto) | {
        "output_qubits": goto["output_qubits"],
        "max_weight": goto["t"],
    }
    noise = goto["options"]["noise"]
    text = applications.circuit_text(_input("cgoto"))
    changed = stim.Circuit(text.replace("CX 0 3 2 1", "CX 2 1"))
    random = stim.Circuit(text + "H 8\nM 8\n")
    assert (
        applications._check_deletion(changed, goto, noise, options)["outcome"] == "target changed"
    )
    assert not _prepares(changed, goto)
    # two changes of the prepared state, a flipped sign and a rotated qubit, whose detectors the
    # certifier accepts, so that its stabilizers show the change.  The rotated child violates the
    # bound, so it guards that the state is checked before the bound, and the flipped child
    # certifies, so it guards the check of the state itself.
    for appended in ("X 0\n", "H 0\n"):
        moved = stim.Circuit(text + appended)
        applications._certify(moved, noise, options, "brouwer_zimmermann")
        outcome = applications._check_deletion(moved, goto, noise, options)["outcome"]
        assert outcome == "target changed" and not _prepares(moved, goto), (appended, outcome)
    outcome = applications._check_deletion(random, goto, noise, options)["outcome"]
    assert outcome == "detectors rejected" and _prepares(random, goto), outcome

    # through the task runner, twice each, with the same steps, in the order of the arm, each
    # step and the final circuit checked by rebuilding it, and the final checks, with the final
    # circuit, sent before the fault weights of the final m*'s search
    accepted = 0
    for circuit_id in ("cgoto", "aft.steane.heuristic_opt", "aft.cc_4_8_8.heuristic_opt"):
        entry = experiment._entry(circuit_id)
        task = experiment._task("prune", entry, 0, experiment._limit("prune"))
        runs = [
            local_runs.run_locally(directory / f"{circuit_id}.{run}", [task])[task["key"]]
            for run in range(2)
        ]
        assert all(row["status"] == "ok" for row in runs), runs
        first, second = (row["result"] for row in runs)
        assert (first["steps"], first["final_sha256"]) == (second["steps"], second["final_sha256"])
        assert [record for record in runs[0]["partial"] if "step" in record] == first["steps"]
        order = applications.measured_qubits(_input(circuit_id), "last_measured_first")
        assert [record["qubit"] for record in first["steps"]] == order, circuit_id
        accepted += _check_prune(entry, first, _enumerable(entry))
        keys = (
            "parent_counts",
            "final_counts",
            "final_certified_cc",
            "acceptance",
            "final_circuit",
            "final_sha256",
        )
        partial = runs[0]["partial"]
        (index,) = [i for i, record in enumerate(partial) if "final_checks" in record]
        assert partial[index] == {"final_checks": {k: first[k] for k in keys}}, circuit_id
        searched = [record["m"] for record in partial[index + 1 :]]
        assert searched == list(range(2, 2 + len(searched))), (circuit_id, searched)
        kept = sum(record["outcome"] == "accepted" for record in first["steps"])
        print(f"  {circuit_id}: {kept} of {len(first['steps'])} deletions accepted")

        if circuit_id == "cgoto":
            # C-goto keeps its flag, whose deletion gives W(C-goto), which violates at weight 1
            (step,) = first["steps"]
            assert step["outcome"] == "violation" and step["violation_weight"] == 1, step

    # every step of the d = 3 A-ft circuits that pass at K = t, against the enumeration
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    checked = 0
    for entry in entries:
        steane = entry["id"] == "aft.steane.heuristic_opt"
        if (
            entry["family"] == "A-ft"
            and entry["code"][2] == 3
            and _enumerable(entry)
            and not steane
        ):
            result = applications.prune(_input(entry["id"]), entry, "last_measured_first", 0)
            if result["parent_certified"]:
                accepted += _check_prune(entry, result, True)
                checked += 1
    assert checked and accepted, (checked, accepted)
    print(f"  {checked} d = 3 circuits and {accepted} accepted deletions agree with enumeration")

    # a run whose steps are delayed past its limit keeps its completed steps, and no result
    entry = experiment._entry("aft.cc_4_8_8.heuristic_opt")
    task = experiment._task("prune", entry, 0, 5.0) | {"step_delay_s": 10.0}
    row = local_runs.run_locally(directory / "delayed", [task])[task["key"]]
    steps = [record for record in row["partial"] if "step" in record]
    assert row["status"] == "timeout" and steps and "steps" not in row["result"], row


@test
def at10_margin(directory: pathlib.Path, node: bool) -> None:
    """The margin arm against cert_bz, the enumeration, and its sectors, and its timeout."""
    expected: dict[str, int | str] = {
        "cgoto": "all",
        "aft.steane.heuristic_opt": "all",
        "echen": 3,
        "w.cgoto": 1,
    }
    entries = json.loads(circuits.CIRCUITS_JSON.read_text())["circuits"]
    ids = list(expected) + [
        e["id"] for e in entries if e["status"] == "ok" and e["ground_truth"] and e["r"] > e["t"]
    ]
    enumerated = sectors_enumerated = 0
    for circuit_id in dict.fromkeys(ids):
        entry = experiment._entry(circuit_id)
        options = experiment.certify_options(entry)
        result = applications.find_margin(experiment._noisy_input(entry), options)
        margin = result["margin"]
        assert margin == expected.get(circuit_id, margin), (circuit_id, margin)

        # cert_bz at K = t + 1 gives the same m*, or "all" past the cutoff, or a smaller bound
        certificate = _certify(circuit_id, "brouwer_zimmermann")
        if not certificate.certified:
            assert margin == certificate.counterexample.fault_weight, (circuit_id, margin)
        elif entry["r"] + 1 <= options["max_weight"]:
            assert margin == "all", (circuit_id, margin)
        else:
            assert margin == "all" or margin > options["max_weight"], (circuit_id, margin)

        # the sectors: their larger weight is the residual weight, the violating parts exceed the
        # fault weight, and each weight is that of an enumeration of the stabilizer group
        if margin != "all":
            sectors = (result["sector_weight_x"], result["sector_weight_z"])
            assert max(sectors) == result["residual_weight"], (circuit_id, result)
            parts = [name for name, w in zip("XZ", sectors) if w > len(result["fault_indices"])]
            assert result["violating_parts"] == " and ".join(parts), (circuit_id, result)
            if len(entry["output_qubits"]) <= 15:
                assert sectors == _enumerated_sectors(entry, result["output_pauli"]), circuit_id
                sectors_enumerated += 1

        # the enumeration to min(m*, r + 1), where it is small enough
        limit = entry["r"] + 1 if margin == "all" else min(margin, entry["r"] + 1)
        size = sum(math.comb(entry["N0"], m) for m in range(1, limit + 1))
        if size <= experiment.CONFIG["ground_truth"]["max_combinations"]:
            truth = _enumerated_violation(_input(circuit_id), entry, options, limit)
            assert truth == (None if margin == "all" else margin), (circuit_id, truth, margin)
            enumerated += 1
    assert enumerated and sectors_enumerated, (enumerated, sectors_enumerated)
    print(f"  {len(set(ids))} circuits: {enumerated} enumerated, {sectors_enumerated} sectors")

    # "all" reaches the cutoff r + 1 only if the search completed it
    early = {"margin": "all", "per_weight": [{"m": 2, "violation": False}]}
    assert applications.cutoff_reached(early, 1) and not applications.cutoff_reached(early, 5)
    violation = {"margin": 6, "per_weight": [{"m": 6, "violation": True}]}
    assert not applications.cutoff_reached(violation, 5)

    # a run on E-chen delayed past its limit keeps its completed weights, and gives no m*
    entry = experiment._entry("echen")
    task = experiment._task("margin", entry, 0, 5.0) | {"step_delay_s": 10.0}
    row = local_runs.run_locally(directory / "delayed", [task])[task["key"]]
    completed = [record["m"] for record in row["partial"] if not record["violation"]]
    assert row["status"] == "timeout" and completed and max(completed) < 3, row
    assert "margin" not in row["result"], row["result"]


@test
def at11_code_files(directory: pathlib.Path, node: bool) -> None:
    """Altered copies of code files fail the check that they break; AT5 checks the files."""
    entry = experiment._entry("aft.steane.heuristic_opt")
    code = json.loads((code_files.HERE / entry["code_file"]).read_text())
    surface = experiment._entry("aft.rotated_surface_d5.heuristic_opt")
    exchanged = json.loads((code_files.HERE / surface["code_file"]).read_text())
    swap = {"Hx": "Hz", "Hz": "Hx", "Lx": "Lz", "Lz": "Lx"}
    alterations: tuple[tuple[dict[str, Any], dict[str, Any], str], ...] = (
        (
            entry,
            code | {"Lx": [[1 - bit if j == 3 else bit for j, bit in enumerate(code["Lx"][0])]]},
            "Hz Lx^T",
        ),
        (entry, code | {"Lz": [code["Hz"][0]]}, "Lx Lz^T"),
        (entry, code | {"Lx": []}, "Lx has 0 rows"),
        (entry | {"code": [7, 2, 3]}, code, "k = n - rank(Hx) - rank(Hz)"),
        (surface, exchanged | {key: exchanged[other] for key, other in swap.items()}, "state"),
    )
    for altered_entry, altered_code, check in alterations:
        problems = code_files.check_code(altered_entry, altered_code)
        assert any(check in problem for problem in problems), (check, problems)

    # a code without X checks: the |0> state of one qubit
    zero = {
        "id": "zero",
        "family": "A-ft",
        "t": 0,
        "output_qubits": [0],
        "stabilizers": ["+Z"],
        "code": [1, 1, 1],
    }
    one_qubit = {"distance": 1, "Hx": [], "Hz": [], "Lx": [[1]], "Lz": [[1]]}
    assert code_files.check_code(zero, one_qubit) == []
    print(f"  {len(alterations)} altered code files fail, and a code without X checks passes")


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


def _check_consistency() -> None:
    """Each class of contradiction between exact results makes a disagreement (part of AT8)."""
    calls = {"calls": 1}
    certified: dict[str, Any] = {"certified": True, "violation_weight": None, "K": 3, "batches": {}}
    certified |= {"zero_flag_checks": calls, "classification": calls}

    def violating(weight: int, **description: Any) -> dict[str, Any]:
        return certified | {"certified": False, "violation_weight": weight} | description

    def parent(**final: Any) -> dict[str, Any]:
        return {"parent_certified": True, "parent_violation_weight": None} | final

    accepted_step = {"step": 0, "qubit": 1, "outcome": "accepted"}
    rejected_step = {"step": 0, "qubit": 1, "outcome": "violation", "violation_weight": 1}
    disagreement = "ShadowDisagreement: batch at threshold 2 of 30 rows: BZ first failing row 3"

    # one circuit per class with t = 2, K = 3, and r + 1 > K, so that certification at K gives
    # m* > K, as {arm: [(status, result, partial, other fields of the row), one per run]}, and a
    # consistent one last
    cases: list[dict[str, list[tuple[str, Any, Any, Any]]]] = [
        # a shadow run that ended in ShadowDisagreement, and a kept shadow run whose discarded run,
        # which compiled, ended in it
        {
            "cert_bz": [("ok", certified, [], None)],
            "shadow": [("error", {}, [], {"error": disagreement})],
        },
        {
            "cert_bz": [("ok", certified, [], None)],
            "shadow": [
                (
                    "ok",
                    certified,
                    [],
                    {"discarded_run": {"status": "error", "error": disagreement}},
                )
            ],
        },
        # a finished shadow run against cert_bz
        {"cert_bz": [("ok", violating(3), [], None)], "shadow": [("ok", certified, [], None)]},
        # a repeat against its first run
        {"cert_bz": [("ok", result, [], None) for result in (certified, violating(2), certified)]},
        # a deletion parent's violation at K = t against cert_cc, where cert_bz did not finish
        {
            "cert_bz": [("timeout", {}, [], None)],
            "cert_cc": [("ok", certified, [], None)],
            "prune": [
                ("timeout", {}, [parent(parent_certified=False, parent_violation_weight=2)], None)
            ],
        },
        # a final circuit whose m* is at most t, which CC certifies at t
        {
            "cert_bz": [("ok", certified, [], None)],
            "prune": [
                (
                    "ok",
                    parent(
                        steps=[accepted_step], final_certified_cc=True, final_margin={"margin": 2}
                    ),
                    [],
                    None,
                )
            ],
        },
        # the final m* of a task that accepted no deletion, against the circuit's m*
        {
            "cert_bz": [("ok", violating(3), [], None)],
            "prune": [
                (
                    "ok",
                    parent(
                        steps=[rejected_step], final_certified_cc=True, final_margin={"margin": 4}
                    ),
                    [],
                    None,
                )
            ],
        },
        # a final circuit's m* against a violation that the search of its m* reported
        {
            "cert_bz": [("ok", certified, [], None)],
            "prune": [
                (
                    "ok",
                    parent(
                        steps=[accepted_step], final_certified_cc=True, final_margin={"margin": 5}
                    ),
                    [{"final_checks": {}}, {"m": 4, "violation": True}],
                    None,
                )
            ],
        },
        # the final m* of two deletion tasks that end at the same final circuit
        {
            "cert_bz": [("ok", certified, [], None)],
            **{
                arm: [
                    (
                        "ok",
                        parent(
                            steps=[accepted_step],
                            final_certified_cc=True,
                            final_sha256="a" * 64,
                            final_margin={"margin": margin},
                        ),
                        [],
                        None,
                    )
                ]
                for arm, margin in (("prune", 4), ("prune_sensitivity", 5))
            },
        },
        # the ground truth against a finished margin
        {
            "margin": [("ok", {"margin": 4}, [], None)],
            "ground_truth": [("ok", {"smallest_violation": 3}, [], None)],
        },
        # a repeat of the baseline whose enumerated combinations differ from A_m
        {
            "brute_bz": [
                ("ok", certified, [{"m": 1, "combinations": count, "violation": False}], None)
                for count in (3, 2)
            ],
            "count_accepted": [("ok", {"A": {"1": 3}}, [], None)],
        },
        # the baseline's enumerated combinations against the ground truth's A_m, where
        # count_accepted did not compute it
        {
            "brute_bz": [
                ("ok", certified, [{"m": 1, "combinations": 2, "violation": False}], None)
            ],
            "ground_truth": [("ok", {"smallest_violation": None, "A": {"1": 3}}, [], None)],
        },
        # more combinations than A_m that the baseline enumerated on a weight where it found a
        # violation
        {
            "brute_bz": [
                ("ok", violating(2), [{"m": 2, "combinations": 5, "violation": True}], None)
            ],
            "count_accepted": [("ok", {"A": {"2": 4}}, [], None)],
        },
        # more listed faults than accepted combinations
        {
            "count_listing": [("ok", {"Phi": {"2": 5}}, [], None)],
            "count_accepted": [("ok", {"A": {"2": 4}}, [], None)],
        },
        # more listed faults of the ground truth than accepted combinations
        {
            "ground_truth": [
                ("ok", {"smallest_violation": None, "Phi": {"2": 5}, "A": {"2": 4}}, [], None)
            ]
        },
        # more rows submitted to the oracle on a completed weight than listed faults, and on a
        # weight where cert_cc found a violation than listed faults of the ground truth
        {
            "cert_bz": [("ok", certified, [{"m": 2, "rows": 6, "violation": False}], None)],
            "count_listing": [("ok", {"Phi": {"2": 5}}, [], None)],
        },
        {
            "cert_cc": [("ok", violating(2), [{"m": 2, "rows": 6, "violation": True}], None)],
            "ground_truth": [("ok", {"smallest_violation": 2, "Phi": {"2": 5}}, [], None)],
        },
        # the ground truth against the baseline, where no certification finished
        {
            "cert_bz": [("timeout", {}, [], None)],
            "brute_bz": [("ok", certified, [], None)],
            "ground_truth": [("ok", {"smallest_violation": 3}, [], None)],
        },
        # a violation that an unfinished certification recorded, against a finished one
        {
            "cert_bz": [("ok", certified, [], None)],
            "cert_cc": [("timeout", {}, [{"m": 2, "violation": True}], None)],
        },
        # the result of a certification that exceeded its memory cap after it returned, against a
        # finished one
        {
            "cert_bz": [("ok", certified, [], None)],
            "cert_cc": [("memout", violating(1), [], None)],
        },
        # the ground truth's heavy mechanisms against h of the listing, also from a ground truth
        # that exceeded its memory cap after it returned
        {
            "ground_truth": [("ok", {"smallest_violation": None, "heavy": [True, True]}, [], None)],
            "count_listing": [("ok", {"h": 1}, [], None)],
        },
        {
            "ground_truth": [
                ("memout", {"smallest_violation": None, "heavy": [True, True]}, [], None)
            ],
            "count_listing": [("ok", {"h": 1}, [], None)],
        },
        # counterexamples whose residual weight does not exceed the fault weight, or is below
        # the certifier's lower bound
        {"cert_bz": [("ok", violating(3, residual_exceeds_fault_weight=False), [], None)]},
        {
            "cert_bz": [
                ("ok", violating(3, residual_weight=4, residual_weight_lower_bound=5), [], None)
            ]
        },
        # every kind of claim, consistent: m* = 4
        {
            "cert_bz": [("ok", certified, [], None)] * 3,
            "cert_cc": [("timeout", {}, [{"m": 2, "violation": False}], None)],
            "shadow": [("ok", certified, [], None)],
            "brute_bz": [("ok", certified, [], None)],
            "ground_truth": [
                ("ok", {"smallest_violation": None, "heavy": [True, False]}, [], None)
            ],
            "count_listing": [("ok", {"h": 1}, [], None)],
            "margin": [("ok", {"margin": 4}, [], None)],
            "prune": [
                (
                    "ok",
                    parent(
                        steps=[rejected_step], final_certified_cc=True, final_margin={"margin": 4}
                    ),
                    [],
                    None,
                )
            ],
        },
    ]
    entries = {}
    rows = []
    for width, arms in enumerate(cases, start=10):
        entry = experiment._entry(f"bcat.w{width}.t2")
        assert entry["r"] + 1 > entry["options"]["max_weight"] == 3, entry
        entries[entry["id"]] = entry
        for arm, runs in arms.items():
            for repeat, (status, result, partial, fields) in enumerate(runs):
                task = experiment._task(arm, entry, repeat, experiment._limit(arm))
                rows.append(
                    {"task_key": task["key"], "arm": arm, "repeat": repeat, "post_status": None}
                    | {"status": status, "result": result, "partial": partial}
                    | {"T": 1.0, "peak_rss_gib": 0.5}
                    | (fields or {})
                )
    summary, _ = analyze.summarize(analyze.collect(rows, entries))
    verdicts = {row["id"]: row["verdict"] for row in summary}
    *contradicting, consistent = entries
    assert [verdicts[circuit_id] for circuit_id in contradicting] == ["disagreement"] * 24, verdicts
    (row,) = [row for row in summary if row["id"] == consistent]
    assert row["verdict"] == "certified" and row["margin"] == 4, row
    assert row["contradictions"] is None, row


def _check_application_analysis(directory: pathlib.Path) -> None:
    """The combined margin, and the deletion outputs, on synthetic rows (part of AT8)."""
    import stim

    def run(status: str, result: dict[str, Any] | None = None, partial: Any = None) -> Any:
        return {"status": status, "repeat": 0, "result": result or {}, "partial": partial or []}

    def bz(weight: int | None) -> dict[str, Any]:
        return run("ok", {"certified": weight is None, "violation_weight": weight, "K": 3})

    # every case of the combined margin for t = 2 and K = 3, as (cert_bz, cert_cc, margin arm,
    # flag rank r, expected margin or "> k" or "disagreement")
    not_run = run("not run")
    weights = [{"m": m, "violation": False} for m in (2, 3, 4)]
    sector = {"sector_weight_x": 0, "sector_weight_z": 2, "violating_parts": "Z"}
    cases: list[tuple[Any, Any, Any, int, Any]] = [
        (run("ok", bz(1)["result"] | sector), not_run, None, 7, 1),
        (bz(3), bz(3), None, 7, 3),
        (run("timeout"), bz(3), None, 7, 3),
        (bz(None), not_run, None, 2, "all"),
        (bz(None), not_run, [run("ok", {"margin": 5})], 7, 5),
        (bz(None), not_run, [run("ok", {"margin": "all"})], 7, "all"),
        (bz(None), not_run, [run("timeout", None, weights)], 7, "> 4"),
        (bz(None), not_run, [run("timeout", None, weights[:1])], 7, "> 3"),
        (run("timeout", None, weights[:1]), not_run, None, 7, "> 2"),
        (
            bz(None),
            not_run,
            [run("timeout", None, [*weights[:2], {"m": 4, "violation": True}])],
            7,
            4,
        ),
        (not_run, not_run, None, 7, None),
        (bz(3), not_run, [run("ok", {"margin": 4})], 7, "disagreement"),
        (bz(None), not_run, [run("ok", {"margin": 3})], 7, "disagreement"),
    ]
    for bz_run, cc_run, margin_rows, r, expected in cases:
        circuit = {"t": 2, "r": r, "options": {"max_weight": 3}}
        arms = {"cert_bz": [bz_run], "cert_cc": [cc_run], "margin": margin_rows or []}
        arms = {arm: rows for arm, rows in arms.items() if rows and rows[0] is not not_run}
        claims = analyze_applications.margin_claims(circuit, arms)
        fields = analyze_applications.margin_fields(arms, claims)
        if expected == "disagreement":
            assert fields["agree_margin"] is False and fields["margin"] is None, fields
        elif isinstance(expected, str) and expected.startswith(">"):
            assert fields["margin"] is None, fields
            assert fields["margin_lower_bound"] == int(expected[2:]), (expected, fields)
        else:
            assert fields["margin"] == expected and fields["agree_margin"] is not False, fields
        if expected == 1:
            copied = {key: fields[f"counterexample_{key.replace('_weight', '')}"] for key in sector}
            assert copied == sector and fields["counterexample_arm"] == "cert_bz", fields

    # in summarize, a margin and a deletion parent that disagree with cert_bz, and a final circuit
    # that CC does not certify, each make a disagreement; a consistent deletion task does not
    def certification(weight: int | None, K: int) -> dict[str, Any]:
        calls = {"calls": 1}
        return bz(weight)["result"] | {"K": K, "zero_flag_checks": calls, "classification": calls}

    parent = {"parent_certified": True, "parent_violation_weight": None}
    synthetic = {
        "echen": {"cert_bz": certification(3, 3) | {"batches": {}}, "margin": {"margin": 4}},
        "cgoto": {
            "cert_bz": certification(None, 2) | {"batches": {}},
            "prune": {"parent_certified": False, "parent_violation_weight": 1},
        },
        "aft.steane.heuristic_opt": {
            "cert_bz": certification(None, 2) | {"batches": {}},
            "prune": parent | {"final_certified_cc": False},
        },
        "aft.steane.opt_opt": {
            "cert_bz": certification(2, 2) | {"batches": {}},
            "prune": parent | {"final_certified_cc": True},
        },
        "aft.steane.heuristic_heuristic": {
            "cert_bz": certification(None, 2) | {"batches": {}},
            "prune": parent | {"final_certified_cc": True},
        },
        "aft.shor.opt_opt": {
            "cert_bz": certification(1, 2) | {"batches": {}},
            "prune": {"parent_certified": False, "parent_violation_weight": 1},
        },
        # a killed task whose certified parent, known from its partial results, cert_bz refutes
        "aft.shor.heuristic_opt": {
            "cert_bz": certification(1, 2) | {"batches": {}},
            "prune": run("timeout", None, [parent]),
        },
    }
    rows = []
    for circuit_id, results in synthetic.items():
        for arm, result in results.items():
            task = experiment._task(arm, experiment._entry(circuit_id), 0, experiment._limit(arm))
            rows.append(
                {"task_key": task["key"], "arm": arm, "repeat": 0, "post_status": None}
                | {"T": 1.0, "peak_rss_gib": 0.5}
                | (result if "status" in result else run("ok", result))
            )
    entries = {circuit_id: experiment._entry(circuit_id) for circuit_id in synthetic}
    summary, _ = analyze.summarize(analyze.collect(rows, entries))
    disagreements = {row["id"] for row in summary if row["verdict"] == "disagreement"}
    expected = {"echen", "cgoto", "aft.steane.heuristic_opt", "aft.shor.heuristic_opt"}
    assert disagreements == expected, disagreements

    # the margin counts of one family with t = 2, as (m*, lower bound, verdict)
    margin_cases = [
        (2, None, "certified"),
        (3, None, "certified"),
        (4, None, "certified"),
        ("all", None, "certified"),
        (None, 5, "certified"),
        (None, None, "certified"),
        (1, None, "disagreement"),
    ]
    keys = ("margin", "margin_lower_bound", "verdict")
    margin_rows = [{"t": 2} | dict(zip(keys, row)) for row in margin_cases]
    assert analyze_applications.margin_counts(margin_rows) == {
        "margin_at_most_t": 1,
        "margin_t_plus_1": 1,
        "margin_at_least_t_plus_2": 1,
        "margin_at_least_t_plus_2_values": "4/2: 1",
        "margin_all": 1,
        "margin_unresolved": 1,
        "margin_unresolved_k_range": "5-5",
        "margin_unknown": 1,
    }

    # the deletion outputs: a reduced circuit rebuilt from its input, a wrong final SHA-256, an
    # input that is not the one the label ran on, a disagreement, a parent not certified at K = t,
    # a timeout with final checks that do not hold the final circuit, and two timeouts during the
    # search of the final m*, one after it found a violation; deletion_table.csv is ordered by
    # circuit id, so the finished task of BB [[72,12,6]], whose input is missing, comes last
    reduced_id = "aft.cc_4_8_8.heuristic_opt"
    reduced = applications.prune(
        _input(reduced_id), experiment._entry(reduced_id), "last_measured_first", 0
    )
    assert any(record["outcome"] == "accepted" for record in reduced["steps"]), reduced
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for circuit_id in (reduced_id, "aft.steane.heuristic_opt"):
        (inputs / f"{circuit_id}.stim").write_text(applications.circuit_text(_input(circuit_id)))
    (inputs / "aft.shor.heuristic_opt.stim").write_text(
        applications.circuit_text(_input(reduced_id))
    )
    checks = {"final_certified_cc": True}
    final_keys = ("parent_counts", "final_counts", "final_certified_cc", "acceptance")
    final_checks = {key: reduced[key] for key in (*final_keys, "final_circuit", "final_sha256")}
    searched = [{"m": m, "violation": False} for m in (2, 3)]
    killed = [parent, *reduced["steps"], {"final_checks": final_checks}, *searched]
    found = [*killed[:-1], {"m": 3, "violation": True}]
    arms = {
        reduced_id: [run("ok", reduced, reduced["steps"])],
        "aft.steane.heuristic_opt": [run("ok", reduced | {"final_sha256": "0" * 64})],
        "aft.shor.heuristic_opt": [run("ok", reduced)],
        "aft.steane.opt_opt": [run("ok", reduced)],
        "dsynth190.bb72.naive": [run("ok", reduced)],
        "cgoto": [run("ok", {"parent_certified": False, "parent_violation_weight": 1})],
        "echen": [run("timeout", None, [parent, {"final_checks": checks}])],
        "w.cgoto": [run("timeout", None, [parent]) | {"compiled_in_timing": True}],
        "aft.steane.heuristic_heuristic": [run("timeout", None, killed)],
        "aft.carbon.heuristic_opt": [run("timeout", None, found)],
    }
    per_circuit = {
        circuit_id: {"circuit": experiment._entry(circuit_id), "arms": {"prune": arm_rows}}
        for circuit_id, arm_rows in arms.items()
    }
    # margin_violations.csv lists m* = t, and leaves out m* = t + 1 and a disagreement
    margins = {reduced_id: (2, "certified"), "aft.steane.heuristic_opt": (3, "certified")}
    margins["aft.steane.opt_opt"] = (1, "disagreement")
    summary_rows = [
        {"id": circuit_id, "family": "F", "t": 2, "counterexample_violating_parts": "Z"}
        | dict(zip(("margin", "verdict"), margins.get(circuit_id, (None, "certified"))))
        for circuit_id in arms
    ]
    out = directory / "analysis"
    (out / "pruned").mkdir(parents=True, exist_ok=True)
    (out / "pruned" / "prune.stale.stim").write_text("H 0\n")  # of an earlier analysis
    paths = analyze_applications.write_applications(per_circuit, summary_rows, out, inputs)
    assert not (out / "pruned" / "prune.stale.stim").exists(), "pruned/ was not written whole"
    with open(out / "deletion_table.csv") as file:
        table_rows = list(csv.DictReader(file))
    ids = [row["id"] for row in table_rows]
    assert ids == sorted(ids), ids
    table = {row["id"]: row["rebuild_matches"] for row in table_rows}
    expected_table = {
        reduced_id: "True",
        "aft.steane.heuristic_opt": "False",
        "aft.shor.heuristic_opt": "",
        "aft.steane.heuristic_heuristic": "",
        "aft.carbon.heuristic_opt": "",
        "dsynth190.bb72.naive": "",
    }
    assert table == expected_table, table
    # the tasks killed during the search of their final m* give m* > 3, and m* = 3 where the
    # search found a violation at 3
    killed_rows = {
        row["id"]: (row["final_margin"], row["final_margin_lower_bound"], row["attempted"])
        for row in table_rows
        if row["status"] == "timeout"
    }
    attempted = str(len(reduced["steps"]))
    assert killed_rows == {
        "aft.steane.heuristic_heuristic": ("", "3", attempted),
        "aft.carbon.heuristic_opt": ("3", "", attempted),
    }, killed_rows
    with open(out / "deletion_tasks.csv") as file:
        tasks = {row["id"]: row["disagreement"] for row in csv.DictReader(file)}
    assert tasks == dict.fromkeys(table, "False") | {"aft.steane.opt_opt": "True"}, tasks
    notes = json.loads((out / "deletion_notes.json").read_text())
    assert [entry["id"] for entry in notes["not_certified_at_t"]] == ["cgoto"], notes
    assert notes["unfinished"][0]["final_checks"] == checks, notes
    # a deletion run's plain and flagged timeouts
    statuses = {entry["id"]: entry["status"] for entry in notes["unfinished"]}
    assert statuses == {"echen": "timeout", "w.cgoto": "timeout_compiled"}, notes
    assert notes["disagreements"] == [{"id": "aft.steane.opt_opt", "arm": "prune"}], notes
    mismatches = notes["rebuild_mismatches"]
    assert mismatches == [{"id": "aft.steane.heuristic_opt", "arm": "prune"}], notes
    with open(out / "margin_violations.csv") as file:
        assert [row["id"] for row in csv.DictReader(file)] == [reduced_id]
    stim.Circuit((out / "pruned" / f"prune.{reduced_id}.stim").read_text())

    # an analysis's marks head each circuit of pruned/, which still parses
    store.mark_outputs(paths, {"local_rows": 2})
    pruned = (out / "pruned" / f"prune.{reduced_id}.stim").read_text()
    assert pruned.startswith("# local_rows: 2\n"), pruned[:80]
    stim.Circuit(pruned)


def _check_prune(entry: dict[str, Any], result: dict[str, Any], enumerate_steps: bool) -> int:
    """Check each step of a deletion task by rebuilding its circuit, and its final circuit.

    Each accepted circuit prepares the parent's signed state, by Stim's simulator, and is certified
    at K = t.  Each rejection has the outcome that the rebuilt circuit gives.  With
    ``enumerate_steps``, every verdict is also that of the enumeration of the ground truth.  The
    final circuit is the recorded one, both oracles certify it, its final m* agrees with cert_bz at
    K = t + 1, and the acceptance estimates of parent and final circuit agree with the exact
    probabilities from their DEMs.  Return the number of accepted steps that the enumeration
    checked.
    """
    import stim

    accepted_enumerated = 0

    noise = entry["options"]["noise"]
    options = experiment.certify_options(entry) | {
        "output_qubits": entry["output_qubits"],
        "max_weight": entry["t"],
    }
    current = _input(entry["id"])
    for record in result["steps"]:
        child, _, _ = circuits._delete_qubit(current, record["qubit"])
        outcome = record["outcome"]
        if outcome == "target changed":
            assert not _prepares(child, entry), record
            continue
        assert _prepares(child, entry), record
        noisy = experiment.noisy_circuit(child, noise)
        if outcome == "detectors rejected":
            try:
                qldpc_circuits.certify_bounded_fault_propagation(
                    noisy, method="brouwer_zimmermann", **options
                )
            except ValueError:
                continue
            raise AssertionError(f"qLDPC accepts the detectors of {record}")
        certificate = qldpc_circuits.certify_bounded_fault_propagation(
            noisy, method="brouwer_zimmermann", **options
        )
        example = certificate.counterexample
        weight = None if example is None else example.fault_weight
        assert weight == record.get("violation_weight"), (record, weight)
        assert (outcome == "accepted") == (weight is None), record
        if enumerate_steps:
            truth = _enumerated_violation(child, entry, options, entry["t"])
            assert truth == weight, (record, truth)
            accepted_enumerated += outcome == "accepted"
        if outcome == "accepted":
            current = child

    # the final circuit, rebuilt from the parent and the accepted deletions, by both oracles
    text = applications.circuit_text(current)
    assert text == result["final_circuit"], entry["id"]
    assert hashlib.sha256(text.encode()).hexdigest() == result["final_sha256"], entry["id"]
    noisy = experiment.noisy_circuit(stim.Circuit(text), noise)
    final = qldpc_circuits.certify_bounded_fault_propagation(
        noisy, method="brouwer_zimmermann", **options
    )
    cc = qldpc_circuits.certify_bounded_fault_propagation(
        noisy, method="connected_cluster", **options
    )
    assert final.certified and cc.certified and result["final_certified_cc"] is True, (
        entry["id"],
        final.certified,
        cc.certified,
        result["final_certified_cc"],
    )

    # the final m*, against cert_bz at K = t + 1
    beyond = qldpc_circuits.certify_bounded_fault_propagation(
        noisy, method="brouwer_zimmermann", **(options | {"max_weight": entry["t"] + 1})
    )
    margin, example = result["final_margin"]["margin"], beyond.counterexample
    if example is None:
        assert margin == "all" or margin > entry["t"] + 1, (entry["id"], margin)
    else:
        assert margin == example.fault_weight, (entry["id"], margin)

    # the acceptance estimates of parent and final circuit, against the exact probabilities
    parent = qldpc_circuits.certify_bounded_fault_propagation(
        experiment.noisy_circuit(_input(entry["id"]), noise),
        method="brouwer_zimmermann",
        **(options | {"max_weight": 1}),
    )
    for which, certificate in (("parent", parent), ("final", final)):
        estimate, exact = result["acceptance"][which], _exact_acceptance(certificate)
        assert abs(estimate["probability"] - exact) <= 5 * estimate["standard_error"], which
    return accepted_enumerated


def _enumerable(entry: dict[str, Any]) -> bool:
    """Whether the ground-truth enumeration up to K = t is within its rule for a circuit.

    The sum over 1 <= m <= t of C(N_0, m) must be within the ground-truth limit, and the flags must
    fit in 64 bits, as ``counting.brute_force`` requires.
    """
    size = sum(math.comb(entry["N0"], m) for m in range(1, entry["t"] + 1))
    limit = experiment.CONFIG["ground_truth"]["max_combinations"]
    return size <= limit and entry["num_post_selection_detectors"] <= 64


def _prepares(circuit: Any, entry: dict[str, Any]) -> bool:
    """Whether a noiseless circuit leaves the parent's signed state on its output qubits.

    Stim's simulator runs the circuit with a fixed seed, and every stabilizer of the parent's state,
    placed on the output qubits, must then have expectation +1.
    """
    import stim

    outputs = entry["output_qubits"]
    simulator = stim.TableauSimulator(seed=0)
    simulator.do(circuit)
    size = max(simulator.num_qubits, max(outputs) + 1)
    for text in entry["stabilizers"]:
        stabilizer = stim.PauliString(text)
        placed = stim.PauliString(size)
        for position, qubit in enumerate(outputs):
            placed[qubit] = stabilizer[position]
        placed.sign = stabilizer.sign
        if simulator.peek_observable_expectation(placed) != 1:
            return False
    return True


def _enumerated_violation(
    circuit: Any, entry: dict[str, Any], options: dict[str, Any], max_weight: int
) -> int | None:
    """The smallest violating fault weight up to ``max_weight`` by the ground-truth enumeration.

    The DEM comes from an untimed certification at weight 1, as for the arm ``ground_truth``.
    """
    noisy = experiment.noisy_circuit(circuit, entry["options"]["noise"])
    reference = qldpc_circuits.certify_bounded_fault_propagation(
        noisy, method="brouwer_zimmermann", **(options | {"max_weight": 1})
    )
    flags, syndromes = counting.dem_signatures(
        reference.detector_error_model,
        reference.post_selection_detectors,
        len(reference.state_stabilizers),
    )
    reduced = counting.reduce_signatures(flags, syndromes)
    stabilizers = experiment.stabilizer_matrix(reference.state_stabilizers)
    smallest: int | None = counting.brute_force(reduced, stabilizers, max_weight)[
        "smallest_violation"
    ]
    return smallest


def _exact_acceptance(certificate: Any) -> float:
    """The probability that no post-selection detector of a certificate fires, from its DEM.

    The distribution of the parities of the post-selection detectors is built mechanism by
    mechanism, each firing independently with its probability.
    """
    position = {detector: j for j, detector in enumerate(certificate.post_selection_detectors)}
    distribution = {0: 1.0}
    for instruction in certificate.detector_error_model.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id() and target.val in position:
                mask ^= 1 << position[target.val]
        if not mask:
            continue
        updated: dict[int, float] = {}
        for state, weight in distribution.items():
            updated[state] = updated.get(state, 0.0) + weight * (1 - probability)
            updated[state ^ mask] = updated.get(state ^ mask, 0.0) + weight * probability
        distribution = updated
    return distribution.get(0, 0.0)


def _enumerated_sectors(entry: dict[str, Any], output_pauli: str) -> tuple[int, int]:
    """The residual weights of the X and Z parts of a Pauli error, by enumeration.

    The X part is minimized over the X parts of the elements of the state's stabilizer group that
    have no Z part, and the Z part likewise, which enumerates the whole group.
    """
    import stim

    stabilizers = experiment.stabilizer_matrix([stim.PauliString(s) for s in entry["stabilizers"]])
    n = len(entry["output_qubits"])
    x_part, z_part = (part.astype(np.uint8) for part in stim.PauliString(output_pauli).to_numpy())
    group = np.zeros((1, 2 * n), dtype=np.uint8)
    for row in stabilizers:
        group = np.vstack([group, group ^ row])
    x_type = group[~group[:, n:].any(axis=1), :n]
    z_type = group[~group[:, :n].any(axis=1), n:]
    return (
        int(((x_type ^ x_part).sum(axis=1)).min()),
        int(((z_type ^ z_part).sum(axis=1)).min()),
    )


def _input(circuit_id: str) -> Any:
    """The noiseless certification input of a circuit."""
    import stim

    return stim.Circuit((experiment.INPUTS / f"{circuit_id}.stim").read_text())


if __name__ == "__main__":
    SUITE.main(__doc__)
