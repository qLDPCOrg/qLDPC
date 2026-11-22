# SPDX-License-Identifier: Apache-2.0

"""The analysis of the margins and the deletions, for ``analyze.py``.

**Combined margin.**  m* is the smallest violating fault weight of a circuit, or "all" if none
exists at any fault weight.  Each run of an exact arm makes claims on m* (``margin_claims``): a
violation found at fault weight w determines m* = w, since every arm checks the fault weights in
increasing order, and no violation up to fault weight k gives the lower bound m* > k, or determines
"all" if the cutoff r + 1 is at most k.  The claims come from the result of each run of
``cert_bz``, ``cert_cc``, ``shadow``, ``brute_bz``, ``margin``, and ``ground_truth``, first runs and
repeats, finished or not, and from each fault weight that it reported as completed or violating;
from the parent verdict of each deletion task at K = t; and from the final m* of a deletion task
that accepted no deletion, whose final circuit is the parent.  All values that determine m* must be
equal and exceed every lower bound; otherwise the circuit is a disagreement.  The combined margin
is the determined value, in the column ``margin``, or else the largest lower bound k, in the column
``margin_lower_bound``, meaning m* > k.

**Deletions.**  Each final circuit must be certified at K = t by the connected-cluster oracle, by
the task's result or, for a run killed during the search of its final m*, by the final checks that
it sent as a partial result.  Its final m* must exceed t, as that certification and BZ's of its last
accepted deletion require.  A failure of either check makes the circuit a disagreement
(``final_contradictions``).  The final circuit is rebuilt from the parent's input in
``circuit_data/inputs/`` and the accepted deletions, and compared with its recorded SHA-256, if
that input is the one that the label ran on.

Outputs, besides the margin fields of ``summary.csv`` and the margin counts of ``families.csv``:

* ``margin_violations.csv``: each circuit with m* <= t, with the residual weights of the X and Z
  parts of its counterexample and the parts that exceed its fault weight.
* ``deletion.csv``: every step of every deletion task, with its outcome.
* ``deletion_tasks.csv``: the fields of ``deletion_table.csv`` for every finished deletion task
  whose parent is certified at K = t, with whether its circuit is a disagreement.
* ``deletion_table.csv``: one row per finished deletion task whose parent is certified at K = t, on
  a circuit without a disagreement, in the order of circuit ids: the family, with the MQT QECC
  release of a synthesized circuit, gate counts (encoder and verification), measured qubits,
  measurements, and acceptance probabilities of parent and final circuit, the final m*, and the
  deletions attempted and accepted.
* ``deletion_notes.json``: each arm's deletion order, the parents not certified at K = t, the tasks
  that did not finish with their completed steps and any final checks, the disagreements, and the
  final circuits whose rebuild does not match.
* ``pruned/<arm>.<id>.stim``: each final circuit of ``deletion_table.csv``.  The analysis deletes
  ``pruned/`` with everything in it before it writes them, so that it holds only these circuits.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
INPUTS = HERE / "circuit_data" / "inputs"

# The arms whose runs make claims on a circuit's m* by their results and their fault weights.
MARGIN_ARMS = ("cert_bz", "cert_cc", "shadow", "brute_bz", "margin", "ground_truth")

# A claim of one source on m*: the values that it determines, and its largest lower bound.
Claim = dict[str, Any]


####################################################################################################
# margins


def margin_claims(
    circuit: dict[str, Any], arms: dict[str, list[dict[str, Any]]]
) -> dict[str, Claim]:
    """Every claim of a circuit's runs on its m*, by source, as the module docstring lists.

    A source is a run, as ``cert_bz.r1``, or a part of a deletion task, as ``prune parent``.  Its
    claim holds the values ``exact`` that it determines, an integer or "all", and the largest lower
    bound ``bound`` k, meaning m* > k, or None.
    """
    from experiments.state_prep_certification.certification import applications

    claims: dict[str, Claim] = {}

    def determine(source: str, value: int | str) -> None:
        claims.setdefault(source, {"exact": set(), "bound": None})["exact"].add(value)

    def no_violation_up_to(source: str, weight: int) -> None:
        if circuit["r"] + 1 <= weight:
            determine(source, "all")
            return
        claim = claims.setdefault(source, {"exact": set(), "bound": None})
        claim["bound"] = max(weight, claim["bound"] or 0)

    # each run's result, and each fault weight that it reported
    max_weight = circuit["options"]["max_weight"]
    for arm in MARGIN_ARMS:
        for row in arms.get(arm) or []:
            source = f"{arm}.r{row['repeat']}"
            result = row.get("result") or {}
            if row["status"] == "ok":
                if arm == "margin":
                    determine(source, result["margin"])
                elif arm == "ground_truth" and result["smallest_violation"] is not None:
                    determine(source, result["smallest_violation"])
                elif arm != "ground_truth" and not result.get("certified"):
                    determine(source, result["violation_weight"])
                else:
                    no_violation_up_to(source, max_weight)
            for record in row.get("partial") or []:
                if "m" in record and "violation" in record:
                    if record["violation"]:
                        determine(source, record["m"])
                    else:
                        no_violation_up_to(source, record["m"])

    # each deletion task's parent verdict at K = t, and the final m* of a task that accepted no
    # deletion, whose final circuit is the parent
    for arm in applications.PRUNE_ORDERS:
        if not arms.get(arm):
            continue
        run = arms[arm][0]
        parent = _parent_verdict(run)
        if parent is None:
            continue
        if parent["parent_certified"]:
            no_violation_up_to(f"{arm} parent", circuit["t"])
        else:
            determine(f"{arm} parent", parent["parent_violation_weight"])
        steps = _steps(run)
        final = (run.get("result") or {}).get("final_margin")
        none_accepted = steps is not None and not any(
            record["outcome"] == "accepted" for record in steps
        )
        if final is not None and none_accepted:
            determine(f"{arm} final", final["margin"])
    return claims


def combine_claims(claims: dict[str, Claim]) -> tuple[int | str | None, int | None, list[str]]:
    """The m* that claims determine, else their largest lower bound, and their conflicts.

    Two claims conflict if they determine different values, or one determines a value that is at
    most the other's lower bound; "all" exceeds every bound.  A claim can conflict with itself.
    The value and the bound are None if there are conflicts.
    """
    items = sorted(claims.items())
    conflicts = [
        f"{source} ({_claim_text(claim)})"
        if source == other
        else f"{source} ({_claim_text(claim)}) against {other} ({_claim_text(other_claim)})"
        for index, (source, claim) in enumerate(items)
        for other, other_claim in items[index:]
        if not _compatible(claim, other_claim)
    ]
    if conflicts:
        return None, None, conflicts
    values = {value for claim in claims.values() for value in claim["exact"]}
    if values:
        return values.pop(), None, []
    bounds = [claim["bound"] for claim in claims.values() if claim["bound"] is not None]
    return None, max(bounds, default=None), []


def margin_fields(
    arms: dict[str, list[dict[str, Any]]], claims: dict[str, Claim]
) -> dict[str, Any]:
    """The combined margin of a circuit, its agreement, and the sector of its counterexample.

    ``claims`` are those of ``margin_claims``.  ``margin`` is an integer m*, "all", or None if
    unresolved or a disagreement, and ``margin_lower_bound`` is k where m* > k is all that is known.
    ``agree_margin`` is whether the claims agree, and None if they come from fewer than two
    sources and agree.
    """
    value, bound, conflicts = combine_claims(claims)
    fields: dict[str, Any] = {
        "margin": value,
        "margin_lower_bound": bound,
        "agree_margin": False if conflicts else (True if len(claims) > 1 else None),
    }

    # the sector of the smallest violation, from the first run that describes it
    if isinstance(value, int):
        for arm in ("cert_bz", "cert_cc", "margin"):
            result = (arms.get(arm) or [{}])[0].get("result") or {}
            found = result.get("margin" if arm == "margin" else "violation_weight")
            if found == value and "sector_weight_x" in result:
                fields |= {
                    "margin_sector_x": result["sector_weight_x"],
                    "margin_sector_z": result["sector_weight_z"],
                    "margin_violating_parts": result["violating_parts"],
                }
                break
    return fields


def margin_counts(members: list[dict[str, Any]]) -> dict[str, Any]:
    """The margin counts of one family in families.csv, over its circuits without disagreement.

    It counts the circuits with m* <= t (``margin_at_most_t``, listed in margin_violations.csv),
    m* = t + 1, m* >= t + 2 (with the values that occur, as "m*/t: count"), "all", m* > k (with the
    range of k), and no information on m*.
    """
    kept = [row for row in members if row["verdict"] != "disagreement"]
    exact = [row for row in kept if isinstance(row["margin"], int)]
    beyond: dict[str, int] = {}
    for row in exact:
        if row["margin"] >= row["t"] + 2:
            key = f"{row['margin']}/{row['t']}"
            beyond[key] = beyond.get(key, 0) + 1
    bounds = [row["margin_lower_bound"] for row in kept if row["margin_lower_bound"] is not None]
    return {
        "margin_at_most_t": sum(row["margin"] <= row["t"] for row in exact),
        "margin_t_plus_1": sum(row["margin"] == row["t"] + 1 for row in exact),
        "margin_at_least_t_plus_2": sum(beyond.values()),
        "margin_at_least_t_plus_2_values": (
            " ".join(f"{key}: {count}" for key, count in sorted(beyond.items())) or None
        ),
        "margin_all": sum(row["margin"] == "all" for row in kept),
        "margin_unresolved": len(bounds),
        "margin_unresolved_k_range": f"{min(bounds)}-{max(bounds)}" if bounds else None,
        "margin_unknown": sum(
            row["margin"] is None and row["margin_lower_bound"] is None for row in kept
        ),
    }


def _compatible(claim: Claim, other: Claim) -> bool:
    """Whether two claims on m* can both hold."""
    values = claim["exact"] | other["exact"]
    bound = max(claim["bound"] or 0, other["bound"] or 0)
    return len(values) <= 1 and all(value == "all" or value > bound for value in values)


def _claim_text(claim: Claim) -> str:
    """A claim as text, as "m* = 3" or "m* > 2"."""
    parts = [f"m* = {value}" for value in sorted(claim["exact"], key=str)]
    if claim["bound"] is not None:
        parts.append(f"m* > {claim['bound']}")
    return ", ".join(parts)


####################################################################################################
# deletions and outputs


def agree_prune(
    bz: dict[str, Any], cc: dict[str, Any], arms: dict[str, list[dict[str, Any]]]
) -> bool | None:
    """Whether the deletion tasks of a circuit agree with its certification and the CC oracle.

    Each parent verdict at K = t is compared with that of ``cert_bz`` at K = t + 1, or of
    ``cert_cc`` where ``cert_bz`` did not finish: the parent is certified at t exactly when the
    certification certifies or violates at t + 1, and otherwise both violate at the same weight.
    Each final circuit must be certified by the CC oracle.  None if nothing was compared.
    """
    from experiments.state_prep_certification.certification import applications

    certification = bz if bz["status"] == "ok" else cc
    compared = []
    for arm in applications.PRUNE_ORDERS:
        if not arms.get(arm):
            continue
        run = arms[arm][0]
        parent = _parent_verdict(run)
        if parent is not None and certification["status"] == "ok":
            result = certification["result"]
            certified_at_t = result.get("certified") or (
                result.get("violation_weight") == result.get("K")
            )
            if parent["parent_certified"]:
                compared.append(bool(certified_at_t))
            else:
                compared.append(result.get("violation_weight") == parent["parent_violation_weight"])
        checks = _final_checks(run)
        if checks is not None:
            compared.append(checks["final_certified_cc"] is True)
    return all(compared) if compared else None


def final_contradictions(
    circuit: dict[str, Any], arms: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """The contradictions of the final circuits of a circuit's deletion tasks.

    A final circuit must be certified at K = t by the CC oracle, and its final m* must exceed t, as
    that certification and BZ's certification of its last accepted deletion require.
    """
    from experiments.state_prep_certification.certification import applications

    contradictions = []
    for arm in applications.PRUNE_ORDERS:
        if not arms.get(arm):
            continue
        run = arms[arm][0]
        checks = _final_checks(run)
        if checks is None:
            continue
        if checks["final_certified_cc"] is not True:
            contradictions.append(f"{arm} final: not certified by CC at t")
        final = (run.get("result") or {}).get("final_margin")
        if final is not None and final["margin"] != "all" and final["margin"] <= circuit["t"]:
            contradictions.append(
                f"{arm} final (m* = {final['margin']}) against its certification at t"
            )
    return contradictions


def write_applications(
    per_circuit: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    out: pathlib.Path,
    inputs: pathlib.Path = INPUTS,
) -> list[pathlib.Path]:
    """Write margin_violations.csv and the deletion outputs, and return their paths.

    The module docstring describes the outputs.
    """
    from experiments.state_prep_certification.certification import applications
    from experiments.state_prep_certification.certification.analyze import _status, _write_csv

    # the circuits with m* <= t
    verdicts = {row["id"]: row["verdict"] for row in summary_rows}
    violations = [
        {
            key: row.get(key)
            for key in (
                "id",
                "family",
                "t",
                "margin",
                "margin_sector_x",
                "margin_sector_z",
                "margin_violating_parts",
            )
        }
        for row in summary_rows
        if row["verdict"] != "disagreement"
        and isinstance(row["margin"], int)
        and row["margin"] <= row["t"]
    ]
    paths = [_write_csv(out / "margin_violations.csv", violations)]

    # each deletion task's steps, its row of deletion_tasks.csv, and its row of deletion_table.csv
    # or its entry in deletion_notes.json
    steps, table, tasks = [], [], []
    notes: dict[str, Any] = {
        "order": applications.PRUNE_ORDERS,
        "not_certified_at_t": [],
        "unfinished": [],
        "disagreements": [],
        "rebuild_mismatches": [],
    }
    shutil.rmtree(out / "pruned", ignore_errors=True)
    (out / "pruned").mkdir()
    for circuit_id, entry in sorted(per_circuit.items()):
        circuit = entry["circuit"]
        for arm in applications.PRUNE_ORDERS:
            if not entry["arms"].get(arm):
                continue
            run = entry["arms"][arm][0]
            result = run["result"]
            records = [record for record in run.get("partial") or [] if "step" in record]
            status = _status([run])
            steps += [{"id": circuit_id, "arm": arm, "status": status} | r for r in records]
            parent = _parent_verdict(run)
            disagreement = verdicts[circuit_id] == "disagreement"
            row = None
            if run["status"] == "ok" and result.get("parent_certified"):
                rebuilt = _rebuild_matches(circuit, result, inputs)
                row = _table_row(circuit_id, circuit, arm, result, rebuilt)
                tasks.append(row | {"disagreement": disagreement})
            if disagreement:
                notes["disagreements"].append({"id": circuit_id, "arm": arm})
            elif parent is not None and not parent["parent_certified"]:
                notes["not_certified_at_t"].append(
                    {
                        "id": circuit_id,
                        "arm": arm,
                        "violation_weight": parent["parent_violation_weight"],
                    }
                )
            elif row is None:
                unfinished = {
                    "id": circuit_id,
                    "arm": arm,
                    "status": status,
                    "steps": len(records),
                }
                checks = _final_checks(run)
                if checks is not None:
                    unfinished["final_checks"] = checks
                notes["unfinished"].append(unfinished)
            else:
                if row["rebuild_matches"] is False:
                    notes["rebuild_mismatches"].append({"id": circuit_id, "arm": arm})
                pruned = out / "pruned" / f"{arm}.{circuit_id}.stim"
                pruned.write_text(result["final_circuit"])
                paths.append(pruned)
                table.append(row)
    table.sort(key=lambda row: row["id"])
    paths.append(_write_csv(out / "deletion.csv", steps))
    paths.append(_write_csv(out / "deletion_table.csv", table))
    paths.append(_write_csv(out / "deletion_tasks.csv", tasks))
    (out / "deletion_notes.json").write_text(json.dumps(notes, indent=1) + "\n")
    return [*paths, out / "deletion_notes.json"]


def _final_checks(run: dict[str, Any]) -> dict[str, Any] | None:
    """The final checks of a deletion task: those of its result, or of its partial results.

    None if the task did not reach them, as when its parent is not certified at K = t.
    """
    if run["status"] == "ok" and run["result"].get("parent_certified"):
        result: dict[str, Any] = run["result"]
        return result
    for record in run.get("partial") or []:
        if "final_checks" in record:
            checks: dict[str, Any] = record["final_checks"]
            return checks
    return None


def _parent_verdict(run: dict[str, Any]) -> dict[str, Any] | None:
    """The parent verdict of a deletion task, from its result or its partial results."""
    for record in [run.get("result") or {}] + list(run.get("partial") or []):
        if "parent_certified" in record:
            return record
    return None


def _steps(run: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The steps of a finished deletion task, or None if its result has none."""
    if run["status"] != "ok":
        return None
    steps: list[dict[str, Any]] | None = run["result"].get("steps")
    return steps


def _table_row(
    circuit_id: str,
    circuit: dict[str, Any],
    arm: str,
    result: dict[str, Any],
    rebuilt: bool | None,
) -> dict[str, Any]:
    """The row of deletion_table.csv of one finished deletion task."""
    row: dict[str, Any] = {
        "id": circuit_id,
        "arm": arm,
        "order": result["order"],
        "family": circuit["family"],
        "mqt_qecc_release": circuit.get("mqt_qecc_release"),
        "code": circuit.get("code"),
        "t": circuit["t"],
    }
    for which in ("parent", "final"):
        counts = result[f"{which}_counts"]
        acceptance = result["acceptance"][which]
        row |= {
            f"{which}_two_qubit_gates": counts["encoder_gates"] + counts["verification_gates"],
            f"{which}_encoder_gates": counts["encoder_gates"],
            f"{which}_verification_gates": counts["verification_gates"],
            f"{which}_measured_qubits": counts["measured_qubits"],
            f"{which}_measurements": counts["measurements"],
            f"{which}_acceptance": acceptance["probability"],
            f"{which}_acceptance_se": acceptance["standard_error"],
        }
    final_margin = result["final_margin"]
    return row | {
        "acceptance_p": result["acceptance"]["p"],
        "final_margin": final_margin["margin"],
        "final_violating_parts": final_margin.get("violating_parts"),
        "attempted": len(result["steps"]),
        "accepted": sum(record["outcome"] == "accepted" for record in result["steps"]),
        "final_certified_cc": result["final_certified_cc"],
        "final_sha256": result["final_sha256"],
        "rebuild_matches": rebuilt,
    }


def _rebuild_matches(
    circuit: dict[str, Any], result: dict[str, Any], inputs: pathlib.Path
) -> bool | None:
    """Whether the parent with the accepted deletions is the final circuit.

    None if the parent's input is missing, or is not the one that the label ran on, by its SHA-256
    in the label's ``circuits.json``.
    """
    import stim

    from experiments.state_prep_certification.certification import applications, circuits

    path = inputs / f"{circuit['id']}.stim"
    if (
        not path.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != circuit["input_sha256"]
    ):
        return None
    parent = stim.Circuit(path.read_text())
    for record in result["steps"]:
        if record["outcome"] == "accepted":
            parent, _, _ = circuits._delete_qubit(parent, record["qubit"])
    text = applications.circuit_text(parent)
    return hashlib.sha256(text.encode()).hexdigest() == result["final_sha256"]
