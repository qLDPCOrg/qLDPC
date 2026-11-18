# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the ranking experiment: admission, scores, agreement, the gate, and costs.

    python analyze.py (--archive PATH | --label-dir DIR) --certification-archive PATH [--out DIR]

The analysis is descriptive: tables with first-order standard errors, which are not confidence
bounds, and the outcome of the Phi_cal gate, as the plan of the experiment fixes them.  It writes
CSV and JSON files only.

**Provenance, a hard failure.**  Before it joins any circuit id, the analysis stops unless the
certification archive matches its ``.sha256`` file and its inner ``SHA256SUMS``; its manifest
records the subset ``main``, a run from a bundle, and no uncommitted changes; its label recomputes
from ``pins.json`` and ``certification/config.json`` at the harness commit that the manifest
records, from the label's copy of ``circuits.json``, and from the bundle's requirements; and that
copy equals the ``circuits.json`` that the ranking label froze.

**Admission.**  A circuit is admitted if certification's analysis of the archive gives it the
verdict "certified" or "violation at t + 1" at K = t + 1, which is certification at K = t, and no
disagreement.  A group is in scope if it has at least two admitted circuits.  Every circuit not
admitted, and every group dropped, is reported.

**Scores** (``experiment.scores``), per admitted circuit and rate: Phi_unc, Phi_unc of the X-type
histogram, Phi_cal with its histogram and calibration contributions sigma_hist and sigma_cal, the
acceptance alpha, and Pr[R > 0 | A].  Lower values predict a smaller Phi_true, except for alpha,
whose higher values do.  A calibration cell (m, g) is unreachable if no X error of weight m has X
residual weight m on the group's state, and unsampled if it is reachable and no draw reached it.

**Pairs**, within a group at one rate: resolved if |Delta| > 2 sigma_Delta + tau_i + tau_j and
neither reference has an inadequate stratum.  A score's call on a resolved pair is the sign of its
difference; equal values are a tie, counted as wrong; a call is uncertain if the difference is
within twice its standard error, with Phi_cal's shared calibration error propagated through the
difference of the two circuits' cell weights, or if a reachable unsampled cell has weight in
either histogram.  A rate is eligible with at least ``ELIGIBLE_PAIRS`` resolved pairs, and a score
agrees at an eligible rate if it calls at least ``AGREE_ACCURACY`` of them correctly.  The gate
keeps Phi_cal if at least two rates are eligible, Phi_cal agrees at each, and, pooled over them,
Phi_cal calls strictly more pairs correctly than each of Phi_unc, alpha, and Pr[R > 0 | A], with
Phi_cal's uncertain calls counted as wrong; it is inconclusive with fewer than two eligible rates.

**Costs** are projected sampling costs at matched first-order standard error, with the achieved
costs and errors beside them (plan section 7).  The sampled cost of a component is scaled by
max(s_min, (sigma / sigma_target)^2), but only where every sampled stratum, row, or cell that
contributes to sigma has at least ``MIN_NONZERO`` draws of nonzero value; for histogram rows, whose
draws are not stored, the Kish effective number of the draws of nonzero value stands in for their
count.  k* follows the four cases of the break-even inequality.

Outputs, in ``--out`` (by default ``analysis/`` in the label directory): ``admission.csv``,
``candidates.csv`` (with each circuit's projected and achieved costs), ``pairs.csv``,
``agreement.csv``, ``kendall.csv``, ``gate.json``, ``groups.csv``, ``cost_table.csv``,
``leading_order.csv``, ``checks.json`` (RT6 and RT7, and every task without a final ``ok`` row), and
``consumers/``, the consumer circuit and block preparation of each group and rate.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import statistics
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import numpy as np
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
from qldpc.objects import Pauli

CONFIG = experiment.CONFIG
M = experiment.MAX_RESIDUAL_WEIGHT
SCORES = ("phi_cal", "phi_unc", "phi_unc_x", "alpha", "residual")
GATE_RIVALS = ("phi_unc", "alpha", "residual")
ELIGIBLE_PAIRS = 8
AGREE_ACCURACY = 0.8
MIN_NONZERO = direct.MIN_NONZERO
MIN_DRAWS = 1000
PRECISION_RELATIVE = 0.1
PRECISION_FLOOR = 0.01


class ProvenanceError(SystemExit):
    """The certification archive is not the one that the ranking label requires."""


def main() -> None:
    """Analyze the run, and write the outputs."""
    args = parse_args()
    label_dir = args.label_dir or _unpack(args.archive)
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
    out.mkdir(parents=True, exist_ok=True)
    if (label_dir / "config.json").read_bytes() != (experiment.HERE / "config.json").read_bytes():
        raise SystemExit(f"{label_dir.name} ran with another config.json than this tree's")
    circuits_bytes = (label_dir / "circuits.json").read_bytes()
    certification_dir = check_provenance(certification_archive, circuits_bytes)
    entries = {e["id"]: e for e in json.loads(circuits_bytes)["circuits"]}
    grouped = experiment.groups(
        sorted(
            (
                e
                for e in entries.values()
                if e["status"] == "ok"
                and e["family"] in CONFIG["families"]
                and e["t"] == CONFIG["t"]
            ),
            key=lambda e: e["id"],
        )
    )
    admission = admit(certification_dir, grouped)
    rows, missing = _ranking_rows(label_dir)
    candidates, groups = evaluate(rows, grouped, admission)
    pairs = compare(candidates)
    agreement, gate = agree(pairs, candidates)
    group_rows = costs(candidates, groups)

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
    label_dir = _unpack(archive)
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

    record = json.loads((label_dir / "manifest.json").read_text())
    if record.get("subset") != "main":
        raise ProvenanceError(f"{archive.name} is of subset {record.get('subset')}, not main")
    if not record.get("bundle"):
        raise ProvenanceError(f"{archive.name} was not run from a bundle")
    if record["harness"].get("dirty") or record["label"].endswith("-dirty"):
        raise ProvenanceError(f"{archive.name} ran with uncommitted changes")
    commit = record["harness"]["commit"]
    files = [
        _git_show(commit, f"experiments/state_prep_certification/{name}")
        for name in ("pins.json", "certification/config.json")
    ]
    label_circuits = (label_dir / "circuits.json").read_bytes()
    expected = certification_label(
        [("pins.json", files[0]), ("config.json", files[1]), ("circuits.json", label_circuits)],
        commit,
        record["bundle"]["requirements_sha256"],
    )
    if record["label"] != expected or label_dir.name != expected:
        raise ProvenanceError(f"{archive.name}: the label does not recompute ({expected})")
    if label_circuits != circuits_bytes:
        raise ProvenanceError(
            f"{archive.name} ran with another circuits.json than the ranking label froze"
        )
    return label_dir


def certification_label(files: list[tuple[str, bytes]], commit: str, requirements: str) -> str:
    """The label that ``manifest.compute_label`` gives a bundle's certification run.

    ``files`` are the names and contents of ``pins.json`` and the frozen files, in order.
    """
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode() + b"\0" + data + b"\0")
    digest.update(commit.encode())
    digest.update(requirements.encode())
    return f"{certification.NAME}-{digest.hexdigest()[:12]}"


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
        cells = reachable_cells(members[0])
        for rate in CONFIG["rates"]:
            calib_row = rows.get(f"calib.{group}.p{rate:g}")
            calibration_fields = _field(calib_row, "calibration")
            calibration = (
                None
                if calibration_fields is None
                else ConsumerCalibration.from_dict(calibration_fields)
            )
            group_row: dict[str, Any] = {
                "group": group,
                "rate": rate,
                "status_calib": _status(calib_row),
                "unreachable_cells": [m for m in range(1, M + 1) if not cells[m]],
            }
            if calibration is not None:
                counts = np.asarray(calibration.excess_num_samples[Pauli.X])  # type: ignore[index]
                group_row["unsampled_cells"] = [
                    [m, g]
                    for m in range(1, M + 1)
                    for g in range(counts.shape[1])
                    if cells[m] and counts[m - 1, g] == 0
                ]
                group_row["calibration"] = calibration
                if calib_row is not None and calib_row["status"] == "ok":
                    group_row |= {
                        key: calib_row["result"][key]
                        for key in (
                            "consumer_circuit",
                            "consumer_circuit_sha256",
                            "block_preparation",
                            "block_preparation_sha256",
                            "fault_distance",
                            "num_consumer_mechanisms",
                            "max_fault_weight",
                            "num_samples",
                            "cpu_consumer_setup_s",
                            "cpu_sampled_s",
                            "score_timing",
                        )
                    }
            groups.append(group_row)
            for entry in members:
                candidates.append(
                    _candidate(rows, entry, group, rate, entry["id"] in admitted, group_row)
                )
    return candidates, groups


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
        "unsampled_cells": group_row.get("unsampled_cells", []),
    }
    total, x_type = _field(histogram_row, "total"), _field(histogram_row, "x")
    calibration = group_row.get("calibration")
    if total is not None and x_type is not None and calibration is not None:
        histograms = (
            experiment.histogram_from_fields(total),
            experiment.histogram_from_fields(x_type),
        )
        values = experiment.scores(*histograms, calibration)
        without_errors = dataclasses.replace(
            calibration, excess_standard_errors=None, baseline_standard_errors=None
        )
        sigma_hist = without_errors.get_excess_failure_probability(histograms[1])[1]
        sigma = values["phi_cal"][1]
        candidate |= {
            "histograms": histograms,
            "calibration": calibration,
            **{f"{score}": value for score, (value, _) in values.items()},
            **{f"sigma_{score}": error for score, (_, error) in values.items()},
            "sigma_hist": sigma_hist,
            "sigma_cal": math.sqrt(max(sigma**2 - sigma_hist**2, 0.0)),
            "truncation_histogram": histograms[0].truncation_error_bound,
            "truncation_calibration": calibration.truncated_probability,
            "cpu_histogram": {
                name: _cpu(fields) for name, fields in (("total", total), ("x", x_type))
            },
        }
    if direct_row is not None and direct_row["status"] == "ok":
        result = direct_row["result"]
        candidate |= {
            "phi_true": result["phi"],
            "sigma_true": result["sigma"],
            "tau": result["tau"],
            "inadequate": result["inadequate"],
            "strata": result["strata"],
            "cpu_direct": {
                key: result[key]
                for key in (
                    "cpu_consumer_setup_s",
                    "cpu_prep_setup_s",
                    "cpu_exact_s",
                    "cpu_sampled_s",
                )
            },
            "num_direct_samples": result["num_samples"],
        }
    return candidate


def reachable_cells(entry: dict[str, Any]) -> dict[int, bool]:
    """Whether some X error of weight m has X residual weight m on a circuit's state, for m <= M."""
    matrix = certification.stabilizer_matrix(entry["stabilizers"])
    setup = residual_weights._ResidualWeightSetup(matrix, True, Pauli.X, M)
    num_qubits = matrix.shape[1] // 2
    reachable = {}
    for m in range(1, M + 1):
        supports = residual_weights._get_combinations(num_qubits, m)
        errors = np.zeros((len(supports), 2 * num_qubits), dtype=np.uint8)
        np.put_along_axis(errors, supports, 1, axis=1)
        reachable[m] = bool(np.any(setup.get_error_weights(errors) == m))
    return reachable


def calibration_difference_error(
    calibration: ConsumerCalibration,
    first: ResidualWeightHistogram,
    second: ResidualWeightHistogram | None = None,
) -> float:
    """The calibration's contribution to the error of Phi_cal(first) - Phi_cal(second).

    This is the calibration part of the error of
    ``ConsumerCalibration.get_excess_failure_probability``, with each cell's and each baseline's
    weight replaced by the difference of the two histograms' weights.  Without ``second``, it is
    the calibration contribution sigma_cal of ``first``.
    """
    distribution, _ = calibration._get_distribution()
    errors = np.asarray(calibration.excess_standard_errors[Pauli.X])  # type: ignore[index]
    counts = np.asarray(calibration.excess_num_samples[Pauli.X])  # type: ignore[index]
    baseline_errors = np.asarray(calibration.baseline_standard_errors)

    def weights(histogram: ResidualWeightHistogram | None) -> tuple[np.ndarray, np.ndarray]:
        """The weight of each cell's row and of each baseline in a histogram's score."""
        if histogram is None:
            return np.zeros(len(errors)), np.zeros(len(distribution))
        probabilities = histogram.get_residual_weight_distribution()
        rows = probabilities[1 : len(errors) + 1]
        return rows, probabilities[len(errors) + 1 :].sum() + rows @ (counts == 0)

    (rows_1, baseline_1), (rows_2, baseline_2) = weights(first), weights(second)
    cell_terms = ((rows_1 - rows_2)[:, None] * errors * distribution) ** 2
    baseline_terms = ((baseline_1 - baseline_2) * baseline_errors * distribution) ** 2
    return math.sqrt(float(cell_terms.sum() + baseline_terms.sum()))


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
            error = math.sqrt(
                first["sigma_hist"] ** 2
                + second["sigma_hist"] ** 2
                + calibration_difference_error(
                    first["calibration"], first["histograms"][1], second["histograms"][1]
                )
                ** 2
            )
        else:
            error = math.hypot(first[f"sigma_{score}"], second[f"sigma_{score}"])
        uncertain = abs(difference) <= 2 * error
        if score == "phi_cal" and _weights_unsampled(first, second):
            uncertain = True
        pair[f"call_{score}"] = "tie" if sign == 0 else ("correct" if sign == truth else "wrong")
        pair[f"uncertain_{score}"] = uncertain
    return pair


def _weights_unsampled(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether a reachable unsampled cell has weight in either circuit's X-type histogram."""
    for candidate in (first, second):
        probabilities = candidate["histograms"][1].get_residual_weight_distribution()
        if any(probabilities[m] > 0 for m, _ in candidate["unsampled_cells"]):
            return True
    return False


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
            taus = list(_kendall(candidates, score, rate).values())
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


def _kendall(candidates: list[dict[str, Any]], score: str, rate: float) -> dict[str, float]:
    """Kendall's tau_b between a score and Phi_true in each group at a rate, where defined."""
    taus = {}
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
        if not math.isnan(tau):
            taus[name] = float(tau)
    return taus


####################################################################################################
# costs


def costs(candidates: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The cost lines of each group at each rate (plan section 7), with k* where computed."""
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
        complete = (
            members
            and "calibration" in group
            and "score_timing" in group
            and all("phi_true" in c and "phi_cal" in c for c in members)
        )
        if len(members) < 2 or not complete:
            rows.append(row | {"k_star": "not computed", "reason": "incomplete or out of scope"})
            continue
        rows.append(row | _cost_lines(members, group))
    return rows


def _cost_lines(members: list[dict[str, Any]], group: dict[str, Any]) -> dict[str, Any]:
    """The projected and achieved costs of both methods for one group at one rate, and k*."""
    target = max(
        PRECISION_RELATIVE * max(abs(c["phi_true"]) for c in members),
        PRECISION_FLOOR * group["rate"] ** 2,
    )
    timing = group["score_timing"]["mean_cpu_s"]
    calibration = group["calibration"]

    # direct simulation: fixed setup and exact strata, and the scaled sampled strata
    direct_lines = []
    for c in members:
        cpu = c["cpu_direct"]
        sampled = [s for s in c["strata"] if not s["exact"]]
        valid = all(
            s["nonzero"] >= MIN_NONZERO and (s["g"] or s["accepted_draws"] >= MIN_NONZERO)
            for s in sampled
        )
        scale = _scale(c["sigma_true"], target, c["num_direct_samples"])
        direct_lines.append(
            (
                cpu["cpu_prep_setup_s"] + cpu["cpu_exact_s"] + scale * cpu["cpu_sampled_s"],
                cpu["cpu_prep_setup_s"] + cpu["cpu_exact_s"] + cpu["cpu_sampled_s"],
                valid,
            )
        )
    a_direct = statistics.mean(c["cpu_direct"]["cpu_consumer_setup_s"] for c in members)

    # scores: the calibration once per group, and each circuit's histograms and score time
    worst_cal = max(c["sigma_cal"] for c in members)
    cal_scale = _scale(worst_cal, target / math.sqrt(2), group["num_samples"])
    cal_valid = _calibration_valid(calibration, members, group["num_samples"])
    a_score = group["cpu_consumer_setup_s"] + cal_scale * group["cpu_sampled_s"]
    a_score_achieved = group["cpu_consumer_setup_s"] + group["cpu_sampled_s"]
    score_lines: list[tuple[float, float, bool]] = []
    unc_lines: list[tuple[float, float, bool]] = []
    for c in members:
        for name, sigma, sigma_target, score, lines in (
            ("x", c["sigma_hist"], target / math.sqrt(2), "phi_cal", score_lines),
            ("total", c["sigma_phi_unc"], target, "phi_unc", unc_lines),
        ):
            cpu = c["cpu_histogram"][name]
            histogram = c["histograms"][0 if name == "total" else 1]
            sampled_rows = histogram.num_samples[~histogram.exact & (histogram.num_samples > 0)]
            scale = _scale(sigma, sigma_target, int(sampled_rows.min(initial=0)))
            fixed = cpu["cpu_setup_s"] + cpu["cpu_exact_s"] + timing[score]
            fixed += timing["rebuild_histogram"]
            lines.append(
                (
                    fixed + scale * cpu["cpu_sampled_s"],
                    fixed + cpu["cpu_sampled_s"],
                    _histogram_valid(histogram),
                )
            )
    for c, direct_line, score_line, unc_line in zip(members, direct_lines, score_lines, unc_lines):
        c |= {
            "cost_direct_projected": direct_line[0],
            "cost_direct_achieved": direct_line[1],
            "cost_phi_cal_projected": score_line[0],
            "cost_phi_cal_achieved": score_line[1],
            "cost_phi_unc_projected": unc_line[0],
            "cost_phi_unc_achieved": unc_line[1],
        }
    b_direct = statistics.mean(line[0] for line in direct_lines)
    b_score = statistics.mean(line[0] for line in score_lines)
    valid = (
        cal_valid and all(line[2] for line in direct_lines) and all(line[2] for line in score_lines)
    )
    return {
        "sigma_target": target,
        "A_score": a_score,
        "b_score": b_score,
        "A_direct": a_direct,
        "b_direct": b_direct,
        "b_unc": statistics.mean(line[0] for line in unc_lines),
        "A_score_achieved": a_score_achieved,
        "b_score_achieved": statistics.mean(line[1] for line in score_lines),
        "b_direct_achieved": statistics.mean(line[1] for line in direct_lines),
        "b_unc_achieved": statistics.mean(line[1] for line in unc_lines),
        "sigma_true_max": max(c["sigma_true"] for c in members),
        "sigma_hist_max": max(c["sigma_hist"] for c in members),
        "sigma_cal_max": worst_cal,
        "sigma_unc_max": max(c["sigma_phi_unc"] for c in members),
        "scaling_valid_direct": all(line[2] for line in direct_lines),
        "scaling_valid_calibration": cal_valid,
        "scaling_valid_histograms": all(line[2] for line in score_lines),
        "scaling_valid_unc": all(line[2] for line in unc_lines),
        "tau_min": min(c["tau"] for c in members),
        "tau_max": max(c["tau"] for c in members),
        "k_star": break_even(a_score - a_direct, b_direct - b_score) if valid else "not computed",
        "reason": None if valid else "the inverse-sample law is not justified",
    }


def break_even(delta_a: float, delta_b: float) -> str:
    """The break-even outcome of the inequality delta_a <= k delta_b (plan section 7)."""
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

    Also the median of the numeric k*, every group's k* outcome, and the range of tau.
    """
    rows = []
    for rate in CONFIG["rates"]:
        computed = [r for r in group_rows if r["rate"] == rate and "A_score" in r]
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
            values = [r[key] for r in computed]
            row[f"{key}_median"] = statistics.median(values) if values else None
            row[f"{key}_min"] = min(values, default=None)
            row[f"{key}_max"] = max(values, default=None)
        numeric = [int(r["k_star"]) for r in computed if str(r["k_star"]).isdigit()]
        row["k_star_median"] = statistics.median(numeric) if numeric else None
        row["tau_min"] = min((r["tau_min"] for r in computed), default=None)
        row["tau_max"] = max((r["tau_max"] for r in computed), default=None)
        row["k_star"] = "; ".join(f"{r['group']}: {r['k_star']}" for r in computed)
        rows.append(row)
    return rows


def _scale(sigma: float, target: float, num_samples: int) -> float:
    """The factor max(s_min, (sigma / target)^2) of a sampled cost, with s_min = 1000 / n.

    n is the smallest number of draws of a sampled stratum, row, or cell of the component, so that
    each gets at least 1000 draws after scaling.
    """
    return max(MIN_DRAWS / num_samples, (sigma / target) ** 2) if num_samples else 0.0


def _histogram_valid(histogram: ResidualWeightHistogram) -> bool:
    """Whether every sampled row of a histogram has enough draws of nonzero value.

    A draw has a nonzero value if it is accepted with a residual weight of at least 1.  The draws
    are not stored, so the Kish effective number of these draws, which is at most their number,
    stands in for it.
    """
    for row in np.flatnonzero(~histogram.exact & (histogram.num_samples > 0)):
        weights = histogram.weight_sums[row, 2:].sum()
        squares = histogram.squared_weight_sums[row, 2:].sum()
        if not weights or weights**2 / squares < MIN_NONZERO:
            return False
    return True


def _calibration_valid(
    calibration: ConsumerCalibration, members: list[dict[str, Any]], num_samples: int
) -> bool:
    """Whether every cell and baseline that contributes to sigma_cal has enough nonzero draws.

    A cell contributes if it was reached and has weight in some circuit's X-type histogram; its
    nonzero draws are its increases and decreases, which its mean and Jeffreys standard error
    determine.  A baseline contributes if it has weight through overflow or an unsampled cell; its
    nonzero draws are its failures.
    """
    distribution, _ = calibration._get_distribution()
    table = np.asarray(calibration.excess_failure_probabilities[Pauli.X])
    errors = np.asarray(calibration.excess_standard_errors[Pauli.X])  # type: ignore[index]
    counts = np.asarray(calibration.excess_num_samples[Pauli.X])  # type: ignore[index]
    baseline = np.asarray(calibration.baseline_failure_probabilities)
    num_qubits = len(calibration.state_stabilizers or ())
    shots = num_samples * min(M + 1, num_qubits)
    for c in members:
        probabilities = c["histograms"][1].get_residual_weight_distribution()
        rows = probabilities[1 : len(table) + 1]
        for m, g in zip(*np.nonzero((rows[:, None] * distribution > 0) & (counts > 0))):
            if nonzero_changes(table[m, g], errors[m, g], counts[m, g]) < MIN_NONZERO:
                return False
        baseline_weights = probabilities[len(table) + 1 :].sum() + rows @ (counts == 0)
        for g in np.flatnonzero(baseline_weights * distribution > 0):
            if round(baseline[g] * shots) < MIN_NONZERO:
                return False
    return True


def nonzero_changes(mean: float, error: float, count: int) -> int:
    """The increases plus decreases of a cell, from its mean and Jeffreys standard error.

    With a = increases + 1/2, b = decreases + 1/2, and A = count + 3/2, the variance is
    ((a + b) A - (a - b)^2) / (A^2 (A + 1)), and a - b = mean count.
    """
    total = count + 1.5
    plus_minus = (error**2 * total**2 * (total + 1) + (mean * count) ** 2) / total
    return round(plus_minus - 1)


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
        if first is None or second is None or "ok" != first["status"] or "ok" != second["status"]:
            results.append({"task": name, "identical": None})
            continue
        results.append(
            {"task": name, "identical": _strip_cpu(first["result"]) == _strip_cpu(second["result"])}
        )
    return results


def leading_order_rows(
    rows: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The leading-order coefficients of each circuit of the exact group, beside Phi_true."""
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
        line: dict[str, Any] = {"id": circuit_id, "status": row["status"]}
        for name in ("true", "cal"):
            entry = result.get(name, {})
            line |= {
                f"order_{name}": entry.get("order"),
                f"coefficient_{name}": entry.get("coefficient"),
                f"note_{name}": entry.get("note"),
            }
            if entry.get("order"):
                line[f"leading_term_{name}"] = entry["value"] * frozen ** entry["order"]
        line["phi_true_direct"] = reference.get("phi_true")
        line["sigma_true_direct"] = reference.get("sigma_true")
        results.append(line)
    return results


####################################################################################################
# helpers


def _ranking_rows(label_dir: pathlib.Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The final rows of the label by key, and every expected task without an ``ok`` row."""
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
    return final, not_ok


def _field(row: dict[str, Any] | None, name: str) -> Any:
    """A field of a row's result, or of its partial results if the run did not finish."""
    if row is None:
        return None
    if row["status"] == "ok":
        return row["result"].get(name)
    for record in row.get("partial", []):
        if name in record:
            return record[name]
    return None


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
    """A candidate's row of candidates.csv, without its objects."""
    row = {
        k: v
        for k, v in candidate.items()
        if k
        not in (
            "histograms",
            "calibration",
            "strata",
            "cpu_histogram",
            "cpu_direct",
            "unsampled_cells",
        )
    }
    for name, cpu in candidate.get("cpu_histogram", {}).items():
        row |= {f"{key}_{name}": value for key, value in cpu.items()}
    row |= candidate.get("cpu_direct", {})
    return row


def _git_show(commit: str, path: str) -> bytes:
    """A file of the repository at a commit."""
    return subprocess.run(
        ["git", "-C", str(manifest.REPOSITORY_ROOT), "show", f"{commit}:{path}"],
        capture_output=True,
        check=True,
    ).stdout


def _unpack(archive: pathlib.Path) -> pathlib.Path:
    """Unpack an archive into a temporary directory, and return its label directory."""
    target = pathlib.Path(tempfile.mkdtemp(prefix="spc-ranking-"))
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    (label_dir,) = [path for path in target.iterdir() if path.is_dir()]
    return label_dir


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
