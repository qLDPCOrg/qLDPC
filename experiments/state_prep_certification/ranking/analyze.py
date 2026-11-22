# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the ranking experiment: admission, scores, agreement, the gate, and costs.

    python analyze.py (--archive PATH | --label-dir DIR) --certification-archive PATH [--out DIR]

The analysis is descriptive: tables with first-order standard errors, which are not confidence
bounds, and the outcome of the Phi_cal gate.  It writes CSV and JSON files only.

**Provenance, a hard failure.**  Before it joins any circuit id, the analysis checks the
certification archive, and stops unless every check passes:

* the archive matches its ``.sha256`` file, and its files match its inner ``SHA256SUMS``;
* its manifest records the subset ``main``, a run from a bundle, and no uncommitted changes;
* its label recomputes, as ``manifest.compute_label`` computes it, from ``pins.json`` and
  ``certification/config.json`` at the harness commit that the manifest records, from the label's
  copy of ``circuits.json``, and from the bundle's requirements;
* that copy of ``circuits.json`` equals the one that the ranking label froze.

**Admission.**  A circuit is admitted if certification's analysis of the archive gives it the
verdict "certified" or "violation at t + 1" at K = t + 1, which is certification at K = t, rather
than "disagreement" (the certification arms or checks disagree), "timeout_compiled" (an arm's
timeout may be due only to a compilation; see certification's ``analyze.py``), or no verdict.  A
group is in scope if it has at least two admitted circuits.  Every circuit not admitted, and every
group dropped, is reported.

**Scores** (``experiment.scores``), per admitted circuit and rate: Phi_unc, Phi_cal with the
histogram and calibration parts sigma_hist and sigma_cal of its standard error
(``ConsumerCalibration.get_standard_error_parts``), the acceptance alpha, and Pr[R > 0 | A], the
probability that an accepted shot has a nonzero residual weight.  Lower values predict a smaller
Phi_true, except for alpha, whose higher values do.  Phi_cal uses a histogram of each Pauli type of
the calibration, X and Z.  A calibration cell (s, m, g) holds the excess of a block error of Pauli
type s and residual weight m, with g faults of the consumer.  It is unsampled if no draw reached it,
and qLDPC then counts it as certain failure.  A circuit's ``unsampled_cells`` in ``candidates.csv``
are the unsampled cells that its score uses: its histogram of type s gives residual weight m a
nonzero probability, and Pr[g] > 0.

**Pairs**, within a group at one rate, are resolved if |Delta| > 2 sigma_Delta + tau_i + tau_j, with
Delta the difference of the two references, sigma_Delta their errors in quadrature, and tau_i and
tau_j their truncation allowances (``direct.py``), and neither reference has an inadequate stratum.
A score's call on a resolved pair is the sign of its difference, and equal values are a tie, counted
as wrong.  A call is uncertain if the difference is within twice its standard error.  For Phi_cal,
that standard error is qLDPC's error of the difference
(``ConsumerCalibration.get_excess_failure_probability_difference``).  It adds both circuits'
histogram parts in quadrature to a calibration part, which propagates each cell's and baseline's
error through the difference of the two circuits' derivatives of Phi_cal with respect to it, so that
part cancels to the extent that their residual-weight distributions agree.  A Phi_cal call is also
uncertain if either circuit has unsampled cells.  A call without a score, as after a failed task,
counts as wrong and is counted.

**Agreement and the gate.**  A rate is eligible with at least ``ELIGIBLE_PAIRS`` resolved pairs,
and a score agrees at an eligible rate if it calls at least ``AGREE_ACCURACY`` of them correctly.
The gate keeps Phi_cal if at least two rates are eligible, Phi_cal agrees at each, and, pooled over
them, Phi_cal calls strictly more pairs correctly than each of Phi_unc, alpha, and Pr[R > 0 | A].
Phi_cal's uncertain calls count as wrong here.  The ``agrees`` column of ``agreement.csv`` counts
each call by its sign, uncertain or not, as ``correct`` does, and ``correct_certain``, which the
gate uses, gives the correct calls that are not uncertain.
The gate is inconclusive with fewer than two eligible rates.

**Costs** are CPU seconds, projected to matched first-order standard errors, with the achieved costs
and errors beside them.  For a group at a rate, the target is sigma* = max(0.1 max_i |Phi_true,i|,
0.01 p^2).  The reference of each circuit targets sigma*, Phi_unc targets sigma*, and Phi_cal splits
it: sigma_hist <= sigma* / sqrt(2) for each circuit and sigma_cal <= sigma* / sqrt(2) for the worst.
Setup and exactly listed strata are fixed costs.  The sampled cost of a component is multiplied by
max(s_min, (sigma / target)^2), with s_min = ``MIN_DRAWS`` over the fewest draws of its sampled
strata, rows, or cells.

This projection assumes that sigma falls as the inverse square root of the draws.  Call a sampled
stratum, histogram row, or calibration cell or baseline a part of its component.  A part with fewer
than ``MIN_NONZERO`` draws of nonzero value fails, since its sigma may not fall that way.
Histograms do not store their draws, so for a row, the Kish effective number of its draws of nonzero
value, which is at most their number, stands in for that number.  A component's projection is made
only if its failing parts can change the factor max(s_min, (sigma / target)^2) by at most a share
``EXEMPT_SHARE`` (``exempted``): the sum of their bounds on sigma^2 is at most that share of
max(sigma^2_pass, s_min target^2), with sigma^2_pass the estimated variance of its passing parts.
Otherwise its projected cost is left empty.  Each bound (``direct_terms``, ``histogram_terms``,
``calibration_terms``) holds whatever the part's own draws are, given the ranges of its values and
its number of draws.  The calibration must pass this test for the histograms of every circuit of the
group, and Phi_cal's histograms of both Pauli types are one component.  The estimated sigma reported
beside a projection includes the exempted parts' own estimates, which few draws may understate, so
the sum of their bounds on sigma^2 is reported beside it (``exempted_bound_*``; in ``groups.csv``,
the largest over the group's circuits).  The calibration's baseline f(0, 0), the consumer's failure
probability with no block error and no fault of its own, is exact if it was estimated as 0
(``exact_no_fault_baseline``), and then has no error and no draws to count.

The two methods cost A + k b for k circuits: A_score is the consumer's setup and the projected
calibration, b_score the mean over the circuits of the projected cost of the histograms of both
Pauli types (setup, exactly listed strata, and scaled sampling), the time to rebuild them, and the
time to evaluate Phi_cal; A_direct is the consumer's setup for the reference, and b_direct the mean
of the preparation's setup and the projected reference.  With delta_a = A_score - A_direct and
delta_b = b_direct - b_score, the scores cost no more for k circuits exactly when delta_a <= k
delta_b; ``break_even`` gives the outcome, k*, in each case of the sign of delta_b.  k* is computed
only where every component of the group's costs is projected.  b_unc, the per-circuit cost of
Phi_unc, is reported beside them.  A row of the ranking label with ``compiled_in_timing`` set (see
``common/runner.py``) gives no CPU seconds.  A group gets no costs at a rate where its calibration,
or the histogram or direct reference of one of its admitted circuits, has such a row.

Outputs, in ``--out`` (by default ``analysis/`` in the label directory): ``admission.csv``,
``candidates.csv`` (with each circuit's projected and achieved costs), ``pairs.csv``,
``agreement.csv``, ``kendall.csv``, ``gate.json``, ``groups.csv``, ``cost_table.csv``,
``leading_order.csv``, ``checks.json`` (RT6, the ``mc_check`` estimates against the reference
within three standard errors plus tau; RT7, whether the two copies of each ``repro`` task agree;
and every task without a final ``ok`` row or whose row has ``compiled_in_timing`` set), and
``consumers/``, the consumer circuit and block preparation of each group and rate.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import pathlib
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import analyze as certification_analyze
from experiments.state_prep_certification.certification import experiment as certification
from experiments.state_prep_certification.common import manifest, store
from experiments.state_prep_certification.common import tasks as task_store
from experiments.state_prep_certification.ranking import direct, experiment
from qldpc.experimental.circuits.benchmarking import (
    ConsumerCalibration,
    ResidualWeightHistogram,
    residual_weights,
)

CONFIG = experiment.CONFIG
M = experiment.MAX_RESIDUAL_WEIGHT
SCORES = ("phi_cal", "phi_unc", "alpha", "residual")
GATE_RIVALS = ("phi_unc", "alpha", "residual")
ELIGIBLE_PAIRS = 8
AGREE_ACCURACY = 0.8
MIN_NONZERO = direct.MIN_NONZERO
EXEMPT_SHARE = 0.01
MIN_DRAWS = 1000
PRECISION_RELATIVE = 0.1
PRECISION_FLOOR = 0.01


class ProvenanceError(SystemExit):
    """The certification archive is not the one that the ranking label requires."""


def main() -> None:
    """Analyze the run, and write the outputs."""
    args = parse_args()
    label_dir = args.label_dir or certification_analyze._unpack(
        args.archive, pathlib.Path(tempfile.mkdtemp(prefix="spc-analysis-"))
    )
    out = args.out or label_dir / "analysis"
    analyze(label_dir, args.certification_archive, out)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=pathlib.Path, help="the ranking archive from the node")
    source.add_argument("--label-dir", type=pathlib.Path, help="a ranking label directory")
    parser.add_argument(
        "--certification-archive",
        type=pathlib.Path,
        required=True,
        help="the certification archive of subset main, from the node",
    )
    parser.add_argument("--out", type=pathlib.Path, help="the output directory")
    return parser.parse_args()


def analyze(
    label_dir: pathlib.Path, certification_archive: pathlib.Path, out: pathlib.Path
) -> None:
    """Check provenance, admit the circuits, and write every output."""
    # the analysis reads this tree's configuration, and the label's circuits
    out.mkdir(parents=True, exist_ok=True)
    if (label_dir / "config.json").read_bytes() != (experiment.HERE / "config.json").read_bytes():
        raise SystemExit(f"{label_dir.name} ran with another config.json than this tree's")
    circuits_bytes = (label_dir / "circuits.json").read_bytes()

    # provenance and admission, then the scores, pairs, agreement, and costs
    certification_dir = check_provenance(certification_archive, circuits_bytes)
    grouped = experiment.groups(experiment.candidates(json.loads(circuits_bytes)["circuits"]))
    admission = admit(certification_dir, grouped)
    rows, missing = _ranking_rows(label_dir)
    candidates, groups = evaluate(rows, grouped, admission)
    pairs = compare(candidates)
    agreement, gate = agree(pairs, candidates)
    group_rows = costs(candidates, groups)

    # the CSV files, gate.json, checks.json, and the consumer circuits
    _write_csv(out / "admission.csv", admission)
    _write_csv(out / "candidates.csv", [_flatten(row) for row in candidates])
    _write_csv(out / "pairs.csv", pairs)
    _write_csv(out / "agreement.csv", agreement)
    _write_csv(out / "kendall.csv", kendall_rows(candidates))
    (out / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    _write_csv(out / "groups.csv", group_rows)
    _write_csv(out / "cost_table.csv", cost_table(group_rows))
    _write_csv(out / "leading_order.csv", leading_order_rows(rows, candidates))
    checks = {"rt6": rt6(rows, candidates), "rt7": rt7(rows), "not_ok": missing}
    (out / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    consumers_dir = out / "consumers"
    consumers_dir.mkdir(exist_ok=True)
    for group in groups:
        name = f"{group['group']}.p{group['rate']:g}"
        if group.get("consumer_circuit"):
            (consumers_dir / f"{name}.consumer.stim").write_text(group["consumer_circuit"])
            (consumers_dir / f"{name}.block_preparation.stim").write_text(
                group["block_preparation"]
            )


####################################################################################################
# provenance and admission


def check_provenance(archive: pathlib.Path, circuits_bytes: bytes) -> pathlib.Path:
    """Check the certification archive as the module docstring says; return its label directory.

    Raises:
        ProvenanceError: If any check fails.
    """
    sums_file = archive.with_name(archive.name + ".sha256")
    if not sums_file.exists():
        raise ProvenanceError(f"{sums_file} is missing")
    recorded = sums_file.read_text().split()
    if recorded[:2] != [store.sha256_file(archive), archive.name]:
        raise ProvenanceError(f"{archive.name} does not match {sums_file.name}")
    # the files of the archive against its inner SHA256SUMS
    label_dir = certification_analyze._unpack(
        archive, pathlib.Path(tempfile.mkdtemp(prefix="spc-analysis-"))
    )
    sums = (label_dir / "SHA256SUMS").read_text().splitlines()
    listed = {}
    for line in sums:
        digest, name = line.split("  ", 1)
        listed[name] = digest
    present = {
        path.relative_to(label_dir).as_posix()
        for path in label_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if present != set(listed) or any(
        store.sha256_file(label_dir / name) != digest for name, digest in listed.items()
    ):
        raise ProvenanceError(f"{archive.name} does not match its inner SHA256SUMS")

    # the run that the manifest records, and its label
    record = json.loads((label_dir / "manifest.json").read_text())
    if record.get("subset") != "main":
        raise ProvenanceError(f"{archive.name} is of subset {record.get('subset')}, not main")
    if not record.get("bundle"):
        raise ProvenanceError(f"{archive.name} was not run from a bundle")
    if record["harness"].get("dirty") or "-dirty-" in record["label"]:
        raise ProvenanceError(f"{archive.name} ran with uncommitted changes")
    commit = record["harness"]["commit"]
    files = [
        _git_show(commit, f"experiments/state_prep_certification/{name}")
        for name in ("pins.json", "certification/config.json")
    ]
    label_circuits = (label_dir / "circuits.json").read_bytes()
    expected = manifest.compute_label(
        certification.NAME,
        [("pins.json", files[0]), ("config.json", files[1]), ("circuits.json", label_circuits)],
        commit,
        record["bundle"]["requirements_sha256"],
    )
    if record["label"] != expected:
        raise ProvenanceError(f"{archive.name}: the label does not recompute ({expected})")
    if label_circuits != circuits_bytes:
        raise ProvenanceError(
            f"{archive.name} ran with another circuits.json than the ranking label froze"
        )
    return label_dir


def admit(
    certification_dir: pathlib.Path, grouped: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Each candidate's certification verdict and admission, and whether its group is in scope."""
    circuits = {
        e["id"]: e
        for e in json.loads((certification_dir / "circuits.json").read_text())["circuits"]
    }
    rows = certification_analyze._final_rows(certification_dir)
    expected = certification_analyze._expected_tasks(certification_dir)
    summary, _ = certification_analyze.summarize(
        certification_analyze.collect(rows, circuits, expected)
    )
    verdicts = {row["id"]: row["verdict"] for row in summary}
    admission = []
    for group, members in grouped.items():
        group_rows = []
        for entry in members:
            t = entry["t"]
            verdict = verdicts.get(entry["id"], "no row")
            admitted = entry["options"]["max_weight"] == t + 1 and verdict in (
                "certified",
                f"violation at {t + 1}",
            )
            group_rows.append(
                {"id": entry["id"], "group": group, "verdict": verdict, "admitted": admitted}
            )
        in_scope = sum(row["admitted"] for row in group_rows) >= 2
        admission += [row | {"group_in_scope": in_scope} for row in group_rows]
    return admission


####################################################################################################
# scores


def evaluate(
    rows: dict[str, dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    admission: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The scores and reference of every candidate at every rate, and each group's calibration."""
    admitted = {row["id"] for row in admission if row["admitted"] and row["group_in_scope"]}
    candidates, groups = [], []
    for group, members in grouped.items():
        for rate in CONFIG["rates"]:
            # a calib task that did not finish keeps the fields of its partial result
            calib_row = rows.get(f"calib.{group}.p{rate:g}")
            fields = _fields(calib_row)
            # a row with compiled_in_timing set gives no CPU seconds (see the module docstring)
            if _compiled(calib_row):
                fields = {
                    key: value
                    for key, value in fields.items()
                    if not key.startswith("cpu_") and key != "score_timing"
                }
            group_row: dict[str, Any] = {
                "group": group,
                "rate": rate,
                "status_calib": _status(calib_row),
                "compiled_in_timing_calib": _compiled(calib_row),
                **{key: value for key, value in fields.items() if key not in ("group", "rate")},
            }
            if "calibration" in fields:
                group_row["calibration"] = exact_no_fault_baseline(
                    ConsumerCalibration.from_dict(fields["calibration"])
                )
            groups.append(group_row)
            for entry in members:
                candidates.append(
                    _candidate(rows, entry, group, rate, entry["id"] in admitted, group_row)
                )
    return candidates, groups


def exact_no_fault_baseline(calibration: ConsumerCalibration) -> ConsumerCalibration:
    """The calibration with the baseline f(0, 0) exact, if it was estimated as 0.

    With no consumer fault and no block error, every shot gives the same detectors and
    observables, and the decoder is deterministic, so f(0, 0) is exactly 0 or 1, and an estimate
    of 0 from any number of shots means f(0, 0) = 0 exactly.  ``Consumer.calibrate`` still gives it
    the Jeffreys error of its zero failures.  The value of every score is unchanged.
    """
    baseline = np.asarray(calibration.baseline_failure_probabilities)
    errors = np.array(calibration.baseline_standard_errors, dtype=float)
    if baseline[0] != 0 or not errors[0]:
        return calibration
    errors[0] = 0.0
    return dataclasses.replace(calibration, baseline_standard_errors=errors)


def unsampled_cells(
    calibration: ConsumerCalibration, histograms: list[ResidualWeightHistogram]
) -> list[list[Any]]:
    """The unsampled cells (s, m, g) of a calibration that a circuit's score uses.

    ``histograms`` are the circuit's histograms of the Pauli types that the calibration covers.  The
    score uses a cell if the histogram of type s gives residual weight m a nonzero probability, and
    Pr[g] > 0.
    """
    distribution = calibration.fault_weight_probabilities
    cells = []
    for histogram in histograms:
        counts = _table(calibration.excess_num_samples, histogram.pauli)
        rows = histogram.get_residual_weight_distribution()[1 : len(counts) + 1]
        weighted = (rows[:, None] * distribution > 0) & (counts == 0)
        cells += [[str(histogram.pauli), int(m) + 1, int(g)] for m, g in zip(*np.nonzero(weighted))]
    return cells


def _candidate(
    rows: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    group: str,
    rate: float,
    admitted: bool,
    group_row: dict[str, Any],
) -> dict[str, Any]:
    """The scores, reference, and costs of one circuit at one rate."""
    name = f"{entry['id']}.p{rate:g}"
    histogram_row, direct_row = rows.get(f"histogram.{name}"), rows.get(f"direct.{name}")
    candidate: dict[str, Any] = {
        "id": entry["id"],
        "group": group,
        "rate": rate,
        "admitted": admitted,
        "status_histogram": _status(histogram_row),
        "status_direct": _status(direct_row),
        "compiled_arms": " ".join(
            name
            for name, row in (("histogram", histogram_row), ("direct", direct_row))
            if _compiled(row)
        )
        or None,
    }
    # the baselines need only the total-weight histogram, Phi_unc also the calibration, and Phi_cal
    # also the histograms of the Pauli types
    fields = _fields(histogram_row)
    calibration = group_row.get("calibration")
    if "total" in fields:
        total = experiment.histogram_from_fields(fields["total"])
        values = experiment.baselines(total)
        if calibration is not None:
            profile = experiment.unc_profile(calibration)
            values["phi_unc"] = total.get_excess_failure_probability(profile)
        candidate |= {score: value for score, (value, _) in values.items()}
        candidate |= {f"sigma_{score}": error for score, (_, error) in values.items()}
    names = ("total", *experiment.PAULI_TYPES)
    if all(name in fields for name in names) and calibration is not None:
        histograms = {name: experiment.histogram_from_fields(fields[name]) for name in names}
        typed = {name: histograms[name] for name in experiment.PAULI_TYPES}
        calibrated = experiment.calibrated_histograms(typed, calibration)
        values = experiment.scores(histograms["total"], typed, calibration)
        sigma_hist, sigma_cal = calibration.get_standard_error_parts(*calibrated)
        candidate |= {
            "histograms": histograms,
            "calibrated": calibrated,
            "calibration": calibration,
            **{score: value for score, (value, _) in values.items()},
            **{f"sigma_{score}": error for score, (_, error) in values.items()},
            "sigma_hist": sigma_hist,
            "sigma_cal": sigma_cal,
            "unsampled_cells": unsampled_cells(calibration, calibrated),
            "truncation_histogram": histograms["total"].truncation_error_bound,
            "truncation_calibration": calibration.truncated_probability,
        }
        if not _compiled(histogram_row):
            candidate["cpu_histogram"] = {name: _cpu(fields[name]) for name in names}
    if direct_row is not None and direct_row["status"] == "ok":
        result = direct_row["result"]
        candidate |= {
            "phi_true": result["phi"],
            "sigma_true": result["sigma"],
            "tau": result["tau"],
            "alpha_reference": result["alpha"],
            "alpha_exact": result["alpha_exact"],
            "pi": result["pi"],
            "eta": result["eta"],
            "inadequate": result["inadequate"],
            "strata": result["strata"],
            "num_direct_samples": result["num_samples"],
        }
        if not _compiled(direct_row):
            candidate["cpu_direct"] = {
                key: result[key]
                for key in (
                    "cpu_consumer_setup_s",
                    "cpu_prep_setup_s",
                    "cpu_exact_s",
                    "cpu_sampled_s",
                )
            }
    return candidate


####################################################################################################
# agreement


def compare(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every pair of admitted circuits of a group at a rate, with the reference and each call."""
    pairs = []
    keyed = itertools.groupby(
        sorted(
            (c for c in candidates if c["admitted"]),
            key=lambda c: (c["group"], c["rate"], c["id"]),
        ),
        key=lambda c: (c["group"], c["rate"]),
    )
    for (group, rate), members in keyed:
        for first, second in itertools.combinations(list(members), 2):
            pairs.append(_pair(group, rate, first, second))
    return pairs


def agree(
    pairs: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The agreement of each score at each rate, and the outcome of the Phi_cal gate."""
    rows = []
    for score in SCORES:
        for rate in CONFIG["rates"]:
            resolved = [p for p in pairs if p["rate"] == rate and p["resolved"]]
            calls = [p.get(f"call_{score}") for p in resolved]
            uncertain = [p.get(f"uncertain_{score}", False) for p in resolved]
            correct = sum(call == "correct" for call in calls)
            certain = sum(c == "correct" and not u for c, u in zip(calls, uncertain))
            eligible = len(resolved) >= ELIGIBLE_PAIRS
            accuracy = correct / len(resolved) if resolved else None
            taus = [tau for tau in _kendall(candidates, score, rate).values() if tau is not None]
            rows.append(
                {
                    "score": score,
                    "rate": rate,
                    "eligible": eligible,
                    "resolved_pairs": len(resolved),
                    "unresolved_pairs": sum(p["rate"] == rate and not p["resolved"] for p in pairs),
                    "correct": correct,
                    "correct_certain": certain,
                    "ties": sum(call == "tie" for call in calls),
                    "no_score": sum(call == "no score" for call in calls),
                    "uncertain": sum(uncertain),
                    "accuracy": accuracy,
                    "agrees": eligible and accuracy is not None and accuracy >= AGREE_ACCURACY,
                    "kendall_tau_b_median": statistics.median(taus) if taus else None,
                    "kendall_groups": len(taus),
                }
            )
    return rows, gate(rows)


def gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The outcome of the Phi_cal gate, with Phi_cal's uncertain calls counted as wrong."""
    by_score = {(row["score"], row["rate"]): row for row in rows}
    eligible = [rate for rate in CONFIG["rates"] if by_score[("phi_cal", rate)]["eligible"]]
    outcome: dict[str, Any] = {"eligible_rates": eligible}
    if len(eligible) < 2:
        return outcome | {"outcome": "inconclusive", "reason": "fewer than two eligible rates"}
    agrees = {
        rate: by_score[("phi_cal", rate)]["correct_certain"]
        >= AGREE_ACCURACY * by_score[("phi_cal", rate)]["resolved_pairs"]
        for rate in eligible
    }
    pooled = {
        score: sum(
            by_score[(score, rate)]["correct_certain" if score == "phi_cal" else "correct"]
            for rate in eligible
        )
        for score in ("phi_cal", *GATE_RIVALS)
    }
    beats = all(pooled["phi_cal"] > pooled[score] for score in GATE_RIVALS)
    kept = all(agrees.values()) and beats
    return outcome | {
        "outcome": "kept" if kept else "not kept",
        "phi_cal_agrees": {f"{rate:g}": value for rate, value in agrees.items()},
        "pooled_correct": pooled,
    }


def kendall_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kendall's tau_b of each score against Phi_true in each group at each rate."""
    return [
        {"score": score, "rate": rate, "group": group, "kendall_tau_b": tau}
        for score in SCORES
        for rate in CONFIG["rates"]
        for group, tau in _kendall(candidates, score, rate).items()
    ]


def _pair(group: str, rate: float, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """The reference difference of a pair, whether it is resolved, and each score's call."""
    pair: dict[str, Any] = {
        "group": group,
        "rate": rate,
        "first": first["id"],
        "second": second["id"],
    }
    if "phi_true" not in first or "phi_true" not in second:
        return pair | {"resolved": False, "reason": "no reference"}
    delta = first["phi_true"] - second["phi_true"]
    sigma = math.hypot(first["sigma_true"], second["sigma_true"])
    margin = 2 * sigma + first["tau"] + second["tau"]
    pair |= {"delta_true": delta, "sigma_delta": sigma, "margin": margin}
    if first["inadequate"] or second["inadequate"]:
        return pair | {"resolved": False, "reason": "inadequate stratum"}
    if abs(delta) <= margin:
        return pair | {"resolved": False, "reason": "within margin"}
    pair |= {"resolved": True, "reason": None}
    truth = 1 if delta > 0 else -1
    for score in SCORES:
        if score not in first or score not in second:
            pair[f"call_{score}"] = "no score"
            continue
        difference = first[score] - second[score]
        sign = 0 if difference == 0 else (1 if difference > 0 else -1)
        if score == "alpha":
            sign = -sign
        if score == "phi_cal":
            _, error = first["calibration"].get_excess_failure_probability_difference(
                first["calibrated"], second["calibrated"]
            )
        else:
            error = math.hypot(first[f"sigma_{score}"], second[f"sigma_{score}"])
        uncertain = abs(difference) <= 2 * error
        if score == "phi_cal" and (first["unsampled_cells"] or second["unsampled_cells"]):
            uncertain = True
        pair[f"call_{score}"] = "tie" if sign == 0 else ("correct" if sign == truth else "wrong")
        pair[f"uncertain_{score}"] = uncertain
    return pair


def _kendall(candidates: list[dict[str, Any]], score: str, rate: float) -> dict[str, float | None]:
    """Kendall's tau_b between a score and Phi_true in each group at a rate.

    It is None where it is undefined, as when every value of the score is equal.
    """
    taus: dict[str, float | None] = {}
    members = [
        c
        for c in candidates
        if c["admitted"] and c["rate"] == rate and "phi_true" in c and score in c
    ]
    for name, group in itertools.groupby(
        sorted(members, key=lambda c: c["group"]), key=lambda c: c["group"]
    ):
        group_members = list(group)
        if len(group_members) < 2:
            continue
        sign = -1 if score == "alpha" else 1
        tau = scipy.stats.kendalltau(
            [sign * c[score] for c in group_members], [c["phi_true"] for c in group_members]
        ).statistic
        taus[name] = None if math.isnan(tau) else float(tau)
    return taus


####################################################################################################
# costs


def costs(candidates: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The cost lines of each group at each rate, as the module docstring defines them.

    A group needs at least two admitted circuits, its calibration and score timing, and every
    circuit's reference and scores, none of them from a row with ``compiled_in_timing`` set.  Each
    circuit's own projected and achieved costs are added to its row of ``candidates``.
    """
    rows = []
    for group in groups:
        members = [
            c
            for c in candidates
            if c["group"] == group["group"] and c["rate"] == group["rate"] and c["admitted"]
        ]
        row = {
            key: value
            for key, value in group.items()
            if key not in ("calibration", "consumer_circuit", "block_preparation", "score_timing")
        }
        row["admitted"] = len(members)
        taus = [c["tau"] for c in members if "tau" in c]
        row |= {"tau_min": min(taus, default=None), "tau_max": max(taus, default=None)}
        compiled = group.get("compiled_in_timing_calib") or any(
            c.get("compiled_arms") for c in members
        )
        complete = (
            members
            and "calibration" in group
            and "score_timing" in group
            and all("phi_true" in c and "phi_cal" in c for c in members)
        )
        if len(members) >= 2 and compiled:
            reason = "a row has compiled_in_timing set"
            rows.append(row | {"k_star": "not computed", "reason": reason})
            continue
        if len(members) < 2 or not complete:
            rows.append(row | {"k_star": "not computed", "reason": "incomplete or out of scope"})
            continue
        rows.append(row | _cost_lines(members, group))
    return rows


def project(
    fixed: float, sampled: float, sigma: float, target: float, draws: int, valid: bool
) -> float | None:
    """A projected cost: the fixed cost and the sampled cost scaled to the target error.

    The sampled cost is multiplied by max(s_min, (sigma / target)^2), with s_min = ``MIN_DRAWS``
    / ``draws``, the fewest draws of a sampled stratum, row, or cell of the component.  It is None
    where the inverse-sample law is not justified (``valid`` is False).
    """
    if not valid:
        return None
    if not draws:
        return fixed
    return fixed + max(MIN_DRAWS / draws, (sigma / target) ** 2) * sampled


def break_even(delta_a: float, delta_b: float) -> str:
    """The break-even outcome of the inequality delta_a <= k delta_b, for k >= 1 circuits.

    With delta_b > 0, it is the smallest such k; with delta_b = 0, every k or none; with
    delta_b < 0, the finite range of k for which it holds, or none.
    """
    if delta_b > 0:
        if delta_a <= 0:
            return "1 (already no costlier)"
        return str(max(1, math.ceil(delta_a / delta_b)))
    if delta_b == 0:
        return "already no costlier for every k" if delta_a <= 0 else "none"
    if delta_a <= 0 and delta_a / delta_b >= 1:
        return f"only 1 <= k <= {math.floor(delta_a / delta_b)}"
    return "none"


def cost_table(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per rate, the medians and ranges over groups of the cost lines and achieved errors.

    Each projected line is taken over the groups where it is projected, and the number of these
    groups is given.  Also given are the median of the numeric k*, every group's k* outcome, and
    the range of the truncation allowance tau over the admitted circuits with a reference.
    """
    rows = []
    for rate in CONFIG["rates"]:
        computed = [r for r in group_rows if r["rate"] == rate and "A_direct" in r]
        row: dict[str, Any] = {"rate": rate, "groups": len(computed)}
        for key in (
            "A_score",
            "b_score",
            "A_direct",
            "b_direct",
            "b_unc",
            "A_score_achieved",
            "b_score_achieved",
            "b_direct_achieved",
            "b_unc_achieved",
            "sigma_true_max",
            "sigma_hist_max",
            "sigma_cal_max",
            "sigma_unc_max",
        ):
            values = [r[key] for r in computed if r[key] is not None]
            row[f"{key}_groups"] = len(values)
            row[f"{key}_median"] = statistics.median(values) if values else None
            row[f"{key}_min"] = min(values, default=None)
            row[f"{key}_max"] = max(values, default=None)
        numeric = [_k_star_number(r["k_star"]) for r in computed]
        numbers = [number for number in numeric if number is not None]
        row["k_star_median"] = statistics.median(numbers) if numbers else None
        row["k_star"] = "; ".join(f"{r['group']}: {r['k_star']}" for r in computed)
        with_tau = [r for r in group_rows if r["rate"] == rate and r["tau_min"] is not None]
        row["tau_min"] = min((r["tau_min"] for r in with_tau), default=None)
        row["tau_max"] = max((r["tau_max"] for r in with_tau), default=None)
        rows.append(row)
    return rows


def nonzero_changes(mean: float, error: float, count: int) -> int:
    """The increases plus decreases of a calibration cell, from its mean and Jeffreys error.

    With a = increases + 1/2, b = decreases + 1/2, and A = count + 3/2, the variance is
    ((a + b) A - (a - b)^2) / (A^2 (A + 1)), and a - b = mean count.
    """
    total = count + 1.5
    plus_minus = (error**2 * total**2 * (total + 1) + (mean * count) ** 2) / total
    return round(plus_minus - 1)


def exempted(passing: list[float], failing: list[float], floor: float) -> bool:
    """Whether a projection may be made despite the parts that fail the nonzero-draw test.

    The projection scales a sampled cost by s = max(s_min, sigma^2 / sigma_target^2), so parts
    that add at most b to sigma^2 change s by a factor of at most 1 + b / (s sigma_target^2).  They
    may be exempted if that factor is at most 1 + ``EXEMPT_SHARE``, with s taken from the passing
    parts: their bounds sum to at most ``EXEMPT_SHARE`` of max(sigma^2_pass, s_min
    sigma_target^2).

    Args:
        passing: The estimated contributions to sigma^2 of the sampled parts that pass.
        failing: The bounds on the contributions of those that fail.
        floor: s_min sigma_target^2 of the projection.

    Returns:
        Whether the failing parts may be exempted; with no failing part, True.
    """
    return sum(failing) <= EXEMPT_SHARE * max(sum(passing), floor)


def check_projection(
    terms: tuple[list[float], list[float]], draws: int, target: float
) -> tuple[bool, float]:
    """Whether a component's projection may be made, and the summed bound of its exempted parts.

    Args:
        terms: The passing parts' estimates and the failing parts' bounds, as ``direct_terms``,
            ``histogram_terms``, and ``calibration_terms`` give them.
        draws: The fewest draws of a sampled part, which sets s_min = ``MIN_DRAWS`` / draws.
        target: The component's target error sigma_target.

    Returns:
        Whether the projection may be made, and, if so, the summed bound of the failing parts that
        it exempts; 0.0 if it may not.
    """
    passing, failing = terms
    floor = MIN_DRAWS / draws * target**2 if draws else 0.0
    valid = exempted(passing, failing, floor)
    return valid, sum(failing) if valid else 0.0


def direct_terms(candidate: dict[str, Any]) -> tuple[list[float], list[float]]:
    """The passing estimates and failing bounds of a reference's sampled strata.

    A stratum (f, g) passes with ``MIN_NONZERO`` nonzero changes, and, for g = 0, which also
    estimates a_f, as many accepted draws.  Its contribution to sigma^2 is that of
    ``direct.combine_strata``.  Its bound uses that each draw's weighted change and weighted
    acceptance lie in [-W, W], for the largest importance weight W that the proposal allows, so
    that each mean has a variance of at most W^2 / n.  With the reference's acceptance alpha
    (``alpha_reference``, estimated by ``combine_strata``) at least alpha_exact, the contribution
    is at most ((pi_f eta_g + max(1, |Phi|) pi_f) W)^2 / (n alpha_exact^2) for g = 0, and
    (pi_f eta_g W)^2 / (n alpha_exact^2) otherwise; |Phi| <= 1, but its estimate can exceed 1.
    A stratum (f, 0) whose draws are nearly all accepted can still pass with an unreliable
    estimate of a_f's variance; that term carries a factor Phi^2.
    """
    pi, eta = candidate["pi"], candidate["eta"]
    alpha, phi = candidate["alpha_reference"], candidate["phi_true"]
    passing, failing = [], []
    for stratum in candidate["strata"]:
        if stratum["exact"]:
            continue
        f, g = stratum["f"], stratum["g"]
        weight = pi[f] * eta[g]
        estimate = weight**2 * stratum["variance"]
        if g == 0:
            estimate += (
                -2 * phi * weight * pi[f] * stratum["covariance"]
                + phi**2 * pi[f] ** 2 * stratum["acceptance_variance"]
            )
        passes = stratum["nonzero"] >= MIN_NONZERO and (
            g or stratum["accepted_draws"] >= MIN_NONZERO
        )
        if passes:
            passing.append(estimate / alpha**2)
        else:
            acceptance_range = max(1.0, abs(phi)) * pi[f] if g == 0 else 0.0
            draw_range = (weight + acceptance_range) * stratum["importance_weight_bound"]
            failing.append(draw_range**2 / (stratum["draws"] * candidate["alpha_exact"] ** 2))
    return passing, failing


def histogram_terms(
    histogram: ResidualWeightHistogram, values: npt.NDArray[np.float64]
) -> tuple[list[float], list[float]]:
    """The passing estimates and failing bounds of a score's sampled histogram rows.

    ``values`` are the score's values of the outcomes, in the column order of the histogram's
    ``weight_sums``.  Each row's estimate is its term in the first-order variance of
    ``ResidualWeightHistogram._get_conditional_mean``, so that the estimates of all rows sum to
    that variance.  A row passes if the Kish effective number of its draws of nonzero value
    (accepted with a residual weight of at least 1) is at least ``MIN_NONZERO``.

    A row's bound uses that each draw's deviation lies within D = (max v - min v) + max |v|, over
    the values v with 0 for a discarded draw; that within row f, the ratio of any two importance
    weights of qLDPC's proposal is at most R^f, with R the inverse of its
    ``_FLAGGED_ODDS_FACTOR``, so that sum w^2 / (sum w)^2 <= min(1, R^f / n_f); and that the
    acceptance is at least that of the exact rows and the truncated mass.  So the row contributes
    at most pi_f^2 D^2 min(1, R^f / n_f) / alpha_low^2.
    """
    # each outcome's value, and each row's mean, as qLDPC computes them
    values = np.array(values, dtype=float)
    values[0] = 0.0
    accepted = np.ones(len(values))
    accepted[0] = 0.0
    fractions = histogram._get_outcome_fractions()
    probabilities = histogram.fault_weight_probabilities
    truncated = histogram.truncated_probability
    numerators, acceptances = fractions @ values, fractions @ accepted
    denominator = probabilities @ acceptances + truncated
    mean = (probabilities @ numerators + truncated * values[-1]) / denominator

    # each outcome's deviation in the first-order variance, and the bound's constants
    deviations = (values[None, :] - numerators[:, None]) - mean * (
        accepted[None, :] - acceptances[:, None]
    )
    totals = histogram.weight_sums.sum(axis=1)
    acceptance_low = probabilities[histogram.exact] @ acceptances[histogram.exact] + truncated
    span = values.max() - values.min() + np.abs(values).max()
    ratio = 1 / residual_weights._FLAGGED_ODDS_FACTOR

    # each sampled row's estimate, if it passes, or bound, if it fails
    passing, failing = [], []
    for row in np.flatnonzero((totals > 0) & ~histogram.exact):
        squares = histogram.squared_weight_sums[row]
        estimate = (
            probabilities[row] ** 2 * float(squares @ deviations[row] ** 2) / totals[row] ** 2
        )
        nonzero_weights = histogram.weight_sums[row, 2:].sum()
        nonzero_squares = squares[2:].sum()
        if nonzero_weights and nonzero_weights**2 / nonzero_squares >= MIN_NONZERO:
            passing.append(estimate / denominator**2)
        else:
            spread = min(1.0, ratio ** float(row) / histogram.num_samples[row])
            failing.append(probabilities[row] ** 2 * span**2 * spread / acceptance_low**2)
    return passing, failing


def calibration_terms(
    calibration: ConsumerCalibration, histograms: list[ResidualWeightHistogram], num_samples: int
) -> tuple[list[float], list[float]]:
    """The passing estimates and failing bounds of a calibration, for one circuit's histograms.

    ``histograms`` are the circuit's histograms of the S Pauli types that the calibration covers.
    The parts are the cells and baselines with a nonzero term in qLDPC's calibration part of the
    error of Phi_cal (``ConsumerCalibration._get_error_terms``).  A part's term there is the
    derivative of Phi_cal with respect to the part, times the part's standard error.  A part's
    estimate is its term squared, so that the estimates sum to sigma_cal^2.  A cell passes with
    ``MIN_NONZERO`` increases and decreases (``nonzero_changes``), and a baseline with as many
    failures; an exact baseline has no error and is not a part.  ``num_samples`` is the
    calibration's draws for each Pauli type, fault weight, and physical error weight.
    ``Consumer.calibrate`` pools a baseline's shots over the Pauli types, so they are
    ``num_samples`` S min(M + 1, n), for M the calibration's largest residual weight and n the
    block's qubits.  The derivative's size is Pr[g] times, for a cell, the probability of its
    residual weight m in its type's histogram, which is at most 1, or, for a baseline, the
    probability summed over the types of the residual weights that count as certain failure, which
    is at most S.  A cell's Jeffreys variance is at most 1 / count, and a baseline's at most
    1 / shots, so a failing cell at fault weight g contributes at most Pr[g]^2 / count, and a
    failing baseline at most (S Pr[g])^2 / shots.
    """
    _, _, terms = calibration._get_error_terms(histograms)
    distribution = calibration.fault_weight_probabilities
    tables = calibration.excess_failure_probabilities
    baseline = np.asarray(calibration.baseline_failure_probabilities)
    num_qubits = len(calibration.state_stabilizers or ())
    shots = num_samples * len(tables) * min(M + 1, num_qubits)
    passing, failing = [], []

    # the cells of each table, row by row, and then the baselines, as qLDPC orders the terms
    offset = 0
    for pauli in tables:
        table = _table(tables, pauli)
        errors = _table(calibration.excess_standard_errors, pauli)
        counts = _table(calibration.excess_num_samples, pauli)
        for (m, g), term in zip(np.ndindex(table.shape), terms[offset : offset + table.size]):
            if not term:
                continue
            count = int(counts[m, g])
            if nonzero_changes(float(table[m, g]), float(errors[m, g]), count) >= MIN_NONZERO:
                passing.append(float(term) ** 2)
            else:
                failing.append(float(distribution[g]) ** 2 / count)
        offset += table.size
    for g, term in enumerate(terms[offset:]):
        if not term:
            continue
        if round(baseline[g] * shots) >= MIN_NONZERO:
            passing.append(float(term) ** 2)
        else:
            failing.append(float(len(tables) * distribution[g]) ** 2 / shots)
    return passing, failing


def phi_cal_values(
    calibration: ConsumerCalibration, histogram: ResidualWeightHistogram
) -> npt.NDArray[np.float64]:
    """The values of Phi_cal's outcomes for a histogram of a Pauli type, as qLDPC scores them.

    Phi_cal is the sum, over the Pauli types that the calibration covers, of the mean of these
    values in the histogram of each type.
    """
    assert histogram.pauli is not None
    failures = calibration.get_failure_probabilities(histogram.pauli)
    heavier = histogram.max_residual_weight + 2 - len(failures)
    return np.concatenate([[0.0], failures - failures[0], np.full(heavier, 1 - failures[0])])


def phi_unc_values(
    calibration: ConsumerCalibration, histogram: ResidualWeightHistogram
) -> npt.NDArray[np.float64]:
    """The values of Phi_unc's outcomes for a total-weight histogram, as qLDPC scores them."""
    profile = experiment.unc_profile(calibration)
    failures = [
        profile.get_failure_probability(weight)
        for weight in range(histogram.max_residual_weight + 2)
    ]
    values = np.array([0.0, *failures]) - failures[0]
    values[0] = 0.0
    return values


def _cost_lines(members: list[dict[str, Any]], group: dict[str, Any]) -> dict[str, Any]:
    """The projected and achieved costs of both methods for one group at one rate, and k*."""
    target = max(
        PRECISION_RELATIVE * max(abs(c["phi_true"]) for c in members),
        PRECISION_FLOOR * group["rate"] ** 2,
    )
    timing = group["score_timing"]["mean_cpu_s"]
    calibration = group["calibration"]

    # direct simulation: fixed setup and exact strata, and the scaled sampled strata
    for c in members:
        cpu = c["cpu_direct"]
        valid, c["exempted_bound_direct"] = check_projection(
            direct_terms(c), c["num_direct_samples"], target
        )
        fixed = cpu["cpu_prep_setup_s"] + cpu["cpu_exact_s"]
        c["cost_direct_achieved"] = fixed + cpu["cpu_sampled_s"]
        c["cost_direct_projected"] = project(
            fixed, cpu["cpu_sampled_s"], c["sigma_true"], target, c["num_direct_samples"], valid
        )

    # Phi_cal: the calibration once per group, against the error of its worst circuit
    worst_cal = max(c["sigma_cal"] for c in members)
    counts = np.concatenate([np.ravel(table) for table in calibration.excess_num_samples.values()])
    cal_draws = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
    checks = [
        check_projection(
            calibration_terms(calibration, c["calibrated"], group["num_samples"]),
            cal_draws,
            target / math.sqrt(2),
        )
        for c in members
    ]
    cal_valid = all(valid for valid, _ in checks)
    cal_bound = max(bound for _, bound in checks) if cal_valid else 0.0
    a_score_achieved = group["cpu_consumer_setup_s"] + group["cpu_sampled_s"]
    a_score = project(
        group["cpu_consumer_setup_s"],
        group["cpu_sampled_s"],
        worst_cal,
        target / math.sqrt(2),
        cal_draws,
        cal_valid,
    )

    # each circuit's histograms of a score, with the time to rebuild them and evaluate the score;
    # Phi_cal's histograms, one of each Pauli type, are projected together
    for c in members:
        for score, names, sigma, sigma_target, score_values in (
            (
                "phi_cal",
                experiment.PAULI_TYPES,
                c["sigma_hist"],
                target / math.sqrt(2),
                phi_cal_values,
            ),
            ("phi_unc", ("total",), c["sigma_phi_unc"], target, phi_unc_values),
        ):
            passing, failing, row_draws = [], [], []
            fixed, sampled = timing[score], 0.0
            for name in names:
                cpu, histogram = c["cpu_histogram"][name], c["histograms"][name]
                terms = histogram_terms(histogram, score_values(calibration, histogram))
                passing += terms[0]
                failing += terms[1]
                rows = histogram.num_samples[~histogram.exact & (histogram.num_samples > 0)]
                row_draws += rows.tolist()
                fixed += cpu["cpu_setup_s"] + cpu["cpu_exact_s"] + timing["rebuild_histogram"]
                sampled += cpu["cpu_sampled_s"]
            draws = int(min(row_draws, default=0))
            valid, c[f"exempted_bound_{score}"] = check_projection(
                (passing, failing), draws, sigma_target
            )
            c[f"cost_{score}_achieved"] = fixed + sampled
            c[f"cost_{score}_projected"] = project(
                fixed, sampled, sigma, sigma_target, draws, valid
            )

    # the cost lines, and k* where every component is projected
    a_direct = statistics.mean(c["cpu_direct"]["cpu_consumer_setup_s"] for c in members)
    lines = {
        "b_direct": _mean_or_none([c["cost_direct_projected"] for c in members]),
        "b_score": _mean_or_none([c["cost_phi_cal_projected"] for c in members]),
        "b_unc": _mean_or_none([c["cost_phi_unc_projected"] for c in members]),
    }
    b_direct, b_score = lines["b_direct"], lines["b_score"]
    k_star = "not computed"
    if a_score is not None and b_direct is not None and b_score is not None:
        k_star = break_even(a_score - a_direct, b_direct - b_score)
    return {
        "sigma_target": target,
        "A_score": a_score,
        "A_direct": a_direct,
        **lines,
        "A_score_achieved": a_score_achieved,
        "b_score_achieved": statistics.mean(c["cost_phi_cal_achieved"] for c in members),
        "b_direct_achieved": statistics.mean(c["cost_direct_achieved"] for c in members),
        "b_unc_achieved": statistics.mean(c["cost_phi_unc_achieved"] for c in members),
        "sigma_true_max": max(c["sigma_true"] for c in members),
        "sigma_hist_max": max(c["sigma_hist"] for c in members),
        "sigma_cal_max": worst_cal,
        "sigma_unc_max": max(c["sigma_phi_unc"] for c in members),
        "exempted_bound_calibration": cal_bound,
        **{
            f"exempted_bound_{name}": max(c[f"exempted_bound_{name}"] for c in members)
            for name in ("direct", "phi_cal", "phi_unc")
        },
        "projected_calibration": a_score is not None,
        "projected_direct": lines["b_direct"] is not None,
        "projected_histograms": lines["b_score"] is not None,
        "projected_unc": lines["b_unc"] is not None,
        "k_star": k_star,
        "reason": None if k_star != "not computed" else "the inverse-sample law is not justified",
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    """The mean of values, or None if any is None."""
    if any(value is None for value in values):
        return None
    return statistics.mean(value for value in values if value is not None)


def _k_star_number(outcome: str) -> int | None:
    """The number k* of a break-even outcome, if it has one."""
    first = outcome.split()[0]
    return int(first) if first.isdigit() else None


####################################################################################################
# checks and the leading order


def rt6(rows: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RT6: plain paired Monte Carlo against the reference, within 3 sigma plus tau."""
    results = []
    for circuit_id in CONFIG["mc_check"]["inputs"]:
        rate = CONFIG["mc_check"]["rate"]
        row = rows.get(f"mc_check.{circuit_id}.p{rate:g}")
        reference = next((c for c in candidates if c["id"] == circuit_id and c["rate"] == rate), {})
        if row is None or row["status"] != "ok" or "phi_true" not in reference:
            results.append({"id": circuit_id, "rate": rate, "passed": None, "status": _status(row)})
            continue
        mc = row["result"]
        bound = 3 * math.hypot(mc["sigma"], reference["sigma_true"]) + reference["tau"]
        difference = abs(mc["phi"] - reference["phi_true"])
        results.append(
            {
                "id": circuit_id,
                "rate": rate,
                "mc": mc["phi"],
                "mc_sigma": mc["sigma"],
                "direct": reference["phi_true"],
                "difference": difference,
                "bound": bound,
                "passed": difference <= bound,
            }
        )
    return results


def rt7(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """RT7: the two copies of each ``repro`` task give identical outputs, apart from CPU times."""
    results = []
    repro = CONFIG["repro"]
    for kind in ("histogram", "direct"):
        name = f"repro.{kind}.{repro['input']}.p{repro['rate']:g}"
        first, second = rows.get(f"{name}.a"), rows.get(f"{name}.b")
        if first is None or second is None or first["status"] != "ok" or second["status"] != "ok":
            results.append({"task": name, "identical": None})
            continue
        results.append(
            {"task": name, "identical": _strip_cpu(first["result"]) == _strip_cpu(second["result"])}
        )
    return results


def leading_order_rows(
    rows: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The leading-order coefficient of Phi_true of each circuit of ``exact.group``.

    Beside it are the circuit's reference and the leading term at ``exact.frozen_rate``.
    """
    frozen = CONFIG["exact"]["frozen_rate"]
    results = []
    for key, row in sorted(rows.items()):
        if row["arm"] != "exact":
            continue
        result = row["result"] if row["status"] == "ok" else {}
        circuit_id = key.split(".", 1)[1]
        reference = next(
            (c for c in candidates if c["id"] == circuit_id and c["rate"] == frozen), {}
        )
        line: dict[str, Any] = {
            "id": circuit_id,
            "status": row["status"],
            "order": result.get("order"),
            "coefficient": result.get("coefficient"),
            "note": result.get("note"),
        }
        if result.get("order"):
            line["leading_term"] = result["value"] * frozen ** result["order"]
        line["phi_true_direct"] = reference.get("phi_true")
        line["sigma_true_direct"] = reference.get("sigma_true")
        results.append(line)
    return results


####################################################################################################
# helpers


def _ranking_rows(label_dir: pathlib.Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The final rows of the label by key, and every expected task without an ``ok`` row.

    A task whose row has ``compiled_in_timing`` set is listed too.
    """
    final, problems = store.final_rows(label_dir)
    for problem in problems:
        print(f"note: {label_dir.name}: {problem}")
    expected = (
        task_store.load_tasks(label_dir) if (label_dir / task_store.TASKS_FILE).exists() else {}
    )
    not_ok = [
        f"{key}: {final[key]['status'] if key in final else 'missing'}"
        for key in sorted(expected)
        if key not in final or final[key]["status"] != "ok"
    ]
    not_ok += [f"{key}: compiled_in_timing" for key in sorted(final) if _compiled(final[key])]
    return final, not_ok


def _fields(row: dict[str, Any] | None) -> dict[str, Any]:
    """The result of a row, or, if its run did not finish, the fields of its partial results."""
    if row is None:
        return {}
    if row["status"] == "ok":
        result: dict[str, Any] = row["result"]
        return result
    fields: dict[str, Any] = {}
    for record in row.get("partial", []):
        fields |= record
    return fields


def _table(tables: Any, pauli: Any) -> np.ndarray:
    """The table of a Pauli type in a calibration's tables, which ``Consumer.calibrate`` fills."""
    assert tables is not None
    return np.asarray(tables[pauli])


def _compiled(row: dict[str, Any] | None) -> bool:
    """Whether a row has ``compiled_in_timing`` set, so that its times are incomplete."""
    return row is not None and bool(row.get("compiled_in_timing"))


def _status(row: dict[str, Any] | None) -> str:
    """The status of a row, or ``missing``."""
    return "missing" if row is None else str(row["status"])


def _cpu(fields: dict[str, Any]) -> dict[str, float]:
    """The CPU seconds of a histogram's parts."""
    return {key: fields[key] for key in ("cpu_setup_s", "cpu_exact_s", "cpu_sampled_s")}


def _strip_cpu(value: Any) -> Any:
    """A result without its CPU times."""
    if isinstance(value, dict):
        return {k: _strip_cpu(v) for k, v in value.items() if not k.startswith("cpu_")}
    if isinstance(value, list):
        return [_strip_cpu(item) for item in value]
    return value


def _flatten(candidate: dict[str, Any]) -> dict[str, Any]:
    """A candidate's row of candidates.csv, without its objects and its laws pi and eta."""
    row = {
        k: v
        for k, v in candidate.items()
        if k
        not in (
            "histograms",
            "calibrated",
            "calibration",
            "strata",
            "cpu_histogram",
            "cpu_direct",
            "pi",
            "eta",
        )
    }
    for name, cpu in candidate.get("cpu_histogram", {}).items():
        row |= {f"{key}_{name}": value for key, value in cpu.items()}
    row |= candidate.get("cpu_direct", {})
    return row


def _git_show(commit: str, path: str) -> bytes:
    """A file of the repository at a commit; a provenance error if Git cannot show it."""
    shown = subprocess.run(
        ["git", "-C", str(manifest.REPOSITORY_ROOT), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if shown.returncode:
        raise ProvenanceError(f"cannot read {path} at the harness commit {commit}")
    return shown.stdout


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV, with the union of their keys as the header, or ``empty``."""
    if not rows:
        path.write_text("empty\n")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )


if __name__ == "__main__":
    main()
