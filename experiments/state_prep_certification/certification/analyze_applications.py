# SPDX-License-Identifier: Apache-2.0

"""The analysis of the margins and the deletions, for ``analyze.py``.

**Combined margin.**  m* is the smallest violating fault weight of a circuit, or "all" if none
exists at any fault weight.  Each run of an exact arm makes claims on m* (``margin_claims``): a
violation found at fault weight w determines m* = w, since every arm checks the fault weights in
increasing order, and no violation up to fault weight k gives the lower bound m* > k, or determines
"all" if the cutoff r + 1 is at most k.  The claims come from the result of each run of
``cert_bz``, ``cert_cc``, ``shadow``, ``brute_bz``, ``margin``, and ``ground_truth``, first runs and
repeats, finished or not, and from each fault weight that it reported as completed or violating;
from the parent verdict of each deletion task at K = t; and from the claim on the final circuit of a
deletion task that accepted no deletion, which is the parent (``final_claim``).  All values that
determine m* must be equal and exceed every lower bound; otherwise the circuit is a disagreement.
The combined margin is the determined value, in the column ``margin``, or else the largest lower
bound k, in the column ``margin_lower_bound``, meaning m* > k.

**Deletions.**  Each final circuit must be certified at K = t by the connected-cluster oracle, by
the task's result or, for a run killed during the search of its final m*, by the final checks that
it sent as a partial result.  The claim on its m* (``final_claim``) holds m* > t, since BZ
certified it at K = t, the m* of the task's result, and each fault weight that the search of its
final m* reported, and its parts must agree.  A completed weight of a final circuit other than the
parent gives only a lower bound, since its flag rank is not recorded.  A failure of either check
makes the circuit a disagreement (``final_contradictions``).  The final circuit is rebuilt from the
parent's input in ``circuit_data/inputs/`` and the accepted deletions, and compared with its
recorded SHA-256, if that input is the one that the label ran on.

Outputs, besides the margin fields of ``summary.csv`` and the margin counts of ``families.csv``:

* ``margin_violations.csv``: each circuit with m* <= t, with the ``counterexample_*`` fields of
  ``margin_fields``, which describe one counterexample of weight m*.
* ``deletion.csv``: every step of every deletion task, with its outcome.
* ``deletion_tasks.csv``: the fields of ``deletion_table.csv`` for every deletion task that
  reached its final checks, with whether its circuit is a disagreement.
* ``deletion_table.csv``: one row per deletion task that reached its final checks, finished or
  killed during the search of its final m*, on a circuit without a disagreement, in the order of
  circuit ids: the task's status, the family, with the MQT QECC release of a synthesized circuit,
  gate counts (encoder and verification), measured qubits, measurements, and acceptance
  probabilities of parent and final circuit, the final m*, or else the lower bound m* > k, for k
  the largest of t and the fault weights that its search completed, and the deletions attempted
  and accepted.
* ``deletion_notes.json``: each arm's deletion order, the parents not certified at K = t, the other
  tasks without a row of ``deletion_table.csv``, with their completed steps and any final checks,
  the disagreements, and the final circuits whose rebuild does not match.
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
    cutoff = circuit["r"] + 1

    # each run's result, and each fault weight that it reported
    max_weight = circuit["options"]["max_weight"]
    for arm in MARGIN_ARMS:
        for row in arms.get(arm) or []:
            source = f"{arm}.r{row['repeat']}"
            result = row.get("result") or {}
            if row["status"] == "ok":
                if arm == "margin":
                    _determine(claims, source, result["margin"])
                elif arm == "ground_truth" and result["smallest_violation"] is not None:
                    _determine(claims, source, result["smallest_violation"])
                elif arm != "ground_truth" and not result.get("certified"):
                    _determine(claims, source, result["violation_weight"])
                else:
                    _no_violation_up_to(claims, source, max_weight, cutoff)
            _weight_claims(claims, source, row.get("partial") or [], cutoff)

    # each deletion task's parent verdict at K = t, and the claim of the final circuit of a task
    # that accepted no deletion, which is the parent
    for arm in applications.PRUNE_ORDERS:
        if not arms.get(arm):
            continue
        run = arms[arm][0]
        parent = _parent_verdict(run)
        if parent is None:
            continue
        if parent["parent_certified"]:
            _no_violation_up_to(claims, f"{arm} parent", circuit["t"], cutoff)
        else:
            _determine(claims, f"{arm} parent", parent["parent_violation_weight"])
        final = final_claim(circuit, run)
        if final is not None and not _accepted_deletion(run):
            claims[f"{arm} final"] = final
    return claims


def final_claim(circuit: dict[str, Any], run: dict[str, Any]) -> Claim | None:
    """The claim on the m* of a deletion task's final circuit, or None if it has no final checks.

    BZ certified the final circuit at K = t, as the parent or as its last accepted deletion, so
    m* > t.  The m* of the task's result and each fault weight that the search of the final m*
    reported after the final checks add to the claim, as in ``margin_claims``.  A completed weight
    determines "all" by the cutoff r + 1 only where the task accepted no deletion, so that the
    final circuit is the parent, whose flag rank r is known.
    """
    if _steps(run) is None:
        return None
    cutoff = None if _accepted_deletion(run) else circuit["r"] + 1
    claims: dict[str, Claim] = {}
    _no_violation_up_to(claims, "final", circuit["t"], cutoff)
    final = (run.get("result") or {}).get("final_margin")
    if final is not None:
        _determine(claims, "final", final["margin"])
    _weight_claims(claims, "final", _final_search(run), cutoff)
    return claims["final"]


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
    """The combined margin of a circuit, its agreement, and the sector of one counterexample.

    ``claims`` are those of ``margin_claims``.  ``margin`` is an integer m*, "all", or None if
    unresolved or a disagreement, and ``margin_lower_bound`` is k where m* > k is all that is known.
    ``agree_margin`` is whether the claims agree, and None if they come from fewer than two
    sources and agree.  For an integer m*, the ``counterexample_*`` fields describe the
    counterexample of the first of ``cert_bz``, ``cert_cc``, and ``margin`` that described one of
    weight m*, named by ``counterexample_arm``: the residual weights of its X and Z parts, and the
    parts whose own residual weight exceeds its fault weight.
    """
    value, bound, conflicts = combine_claims(claims)
    fields: dict[str, Any] = {
        "margin": value,
        "margin_lower_bound": bound,
        "agree_margin": False if conflicts else (True if len(claims) > 1 else None),
    }

    # the sector of one counterexample of weight m*, from the first run that describes one
    if isinstance(value, int):
        for arm in ("cert_bz", "cert_cc", "margin"):
            result = (arms.get(arm) or [{}])[0].get("result") or {}
            found = result.get("margin" if arm == "margin" else "violation_weight")
            if found == value and "sector_weight_x" in result:
                fields |= {
                    "counterexample_arm": arm,
                    "counterexample_sector_x": result["sector_weight_x"],
                    "counterexample_sector_z": result["sector_weight_z"],
                    "counterexample_violating_parts": result["violating_parts"],
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


def _determine(claims: dict[str, Claim], source: str, value: int | str) -> None:
    """Add to a source's claim that m* is this value."""
    claims.setdefault(source, {"exact": set(), "bound": None})["exact"].add(value)


def _no_violation_up_to(
    claims: dict[str, Claim], source: str, weight: int, cutoff: int | None
) -> None:
    """Add to a source's claim that no fault weight up to this one violates.

    That determines "all" if the weight reaches the cutoff r + 1, if known, and else gives m* > k
    for k the weight.
    """
    if cutoff is not None and cutoff <= weight:
        _determine(claims, source, "all")
        return
    claim = claims.setdefault(source, {"exact": set(), "bound": None})
    claim["bound"] = max(weight, claim["bound"] or 0)


def _weight_claims(
    claims: dict[str, Claim], source: str, records: list[dict[str, Any]], cutoff: int | None
) -> None:
    """Add to a source's claim each fault weight that its records report completed or violating."""
    for record in records:
        if "m" in record and "violation" in record:
            if record["violation"]:
                _determine(claims, source, record["m"])
            else:
                _no_violation_up_to(claims, source, record["m"], cutoff)


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

    A final circuit must be certified at K = t by the CC oracle, and the parts of the claim on its
    m* (``final_claim``) must agree.  The final circuit of a task that accepted no deletion is the
    parent, whose claims ``margin_claims`` compares instead.
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
        claim = final_claim(circuit, run)
        if claim is not None and _accepted_deletion(run):
            contradictions += combine_claims({f"{arm} final": claim})[2]
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
                "counterexample_arm",
                "counterexample_sector_x",
                "counterexample_sector_z",
                "counterexample_violating_parts",
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
            records = [record for record in run.get("partial") or [] if "step" in record]
            status = _status([run])
            steps += [{"id": circuit_id, "arm": arm, "status": status} | r for r in records]
            parent = _parent_verdict(run)
            disagreement = verdicts[circuit_id] == "disagreement"
            row, final_circuit = None, ""
            checks = _final_checks(run)
            if checks is not None and "final_sha256" in checks:
                rebuilt = _rebuild_matches(circuit, checks, _steps(run) or [], inputs)
                row = {"status": status} | _table_row(circuit_id, circuit, arm, run, rebuilt)
                tasks.append(row | {"disagreement": disagreement})
                final_circuit = checks["final_circuit"]
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
                if checks is not None:
                    unfinished["final_checks"] = checks
                notes["unfinished"].append(unfinished)
            else:
                if row["rebuild_matches"] is False:
                    notes["rebuild_mismatches"].append({"id": circuit_id, "arm": arm})
                pruned = out / "pruned" / f"{arm}.{circuit_id}.stim"
                pruned.write_text(final_circuit)
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
    """The steps of a deletion task that reached its final checks, or None.

    A task that sent its final checks had sent all its steps as partial results before them.
    """
    if run["status"] == "ok":
        steps: list[dict[str, Any]] | None = run["result"].get("steps")
        return steps
    if _final_checks(run) is None:
        return None
    return [record for record in run.get("partial") or [] if "step" in record]


def _accepted_deletion(run: dict[str, Any]) -> bool:
    """Whether a deletion task that reached its final checks accepted a deletion."""
    return any(record["outcome"] == "accepted" for record in _steps(run) or [])


def _final_search(run: dict[str, Any]) -> list[dict[str, Any]]:
    """The records of a deletion task's partial results after its final checks.

    These are the fault weights that the search of the final m* completed or found a violation at.
    """
    partial = run.get("partial") or []
    start = next((i for i, record in enumerate(partial) if "final_checks" in record), len(partial))
    return partial[start + 1 :]


def _final_margin(
    circuit: dict[str, Any], run: dict[str, Any]
) -> tuple[int | str | None, int | None]:
    """The final m* of a deletion task, or else the lower bound k of m* > k, from ``final_claim``.

    Both are None if the parts of the claim disagree, or if the task has no final checks.
    """
    claim = final_claim(circuit, run)
    if claim is None:
        return None, None
    value, bound, _ = combine_claims({"final": claim})
    return value, bound


def _table_row(
    circuit_id: str,
    circuit: dict[str, Any],
    arm: str,
    run: dict[str, Any],
    rebuilt: bool | None,
) -> dict[str, Any]:
    """The row of deletion_table.csv of one deletion task that reached its final checks.

    Where the claim on the final m* (``final_claim``) does not determine it, as when its search did
    not finish, ``final_margin_lower_bound`` is the largest fault weight k that the search
    completed, or else t, at which the final circuit is certified, meaning m* > k.
    """
    from experiments.state_prep_certification.certification import applications

    checks = _final_checks(run) or {}
    steps = _steps(run) or []
    row: dict[str, Any] = {
        "id": circuit_id,
        "arm": arm,
        "order": applications.PRUNE_ORDERS[arm],
        "family": circuit["family"],
        "mqt_qecc_release": circuit.get("mqt_qecc_release"),
        "code": circuit.get("code"),
        "t": circuit["t"],
    }
    for which in ("parent", "final"):
        counts = checks[f"{which}_counts"]
        acceptance = checks["acceptance"][which]
        row |= {
            f"{which}_two_qubit_gates": counts["encoder_gates"] + counts["verification_gates"],
            f"{which}_encoder_gates": counts["encoder_gates"],
            f"{which}_verification_gates": counts["verification_gates"],
            f"{which}_measured_qubits": counts["measured_qubits"],
            f"{which}_measurements": counts["measurements"],
            f"{which}_acceptance": acceptance["probability"],
            f"{which}_acceptance_se": acceptance["standard_error"],
        }
    margin, bound = _final_margin(circuit, run)
    return row | {
        "acceptance_p": checks["acceptance"]["p"],
        "final_margin": margin,
        "final_margin_lower_bound": bound,
        "final_violating_parts": (run["result"].get("final_margin") or {}).get("violating_parts"),
        "attempted": len(steps),
        "accepted": sum(record["outcome"] == "accepted" for record in steps),
        "final_certified_cc": checks["final_certified_cc"],
        "final_sha256": checks["final_sha256"],
        "rebuild_matches": rebuilt,
    }


def _rebuild_matches(
    circuit: dict[str, Any],
    checks: dict[str, Any],
    steps: list[dict[str, Any]],
    inputs: pathlib.Path,
) -> bool | None:
    """Whether the parent with the accepted deletions is the final circuit of the final checks.

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
    for record in steps:
        if record["outcome"] == "accepted":
            parent, _, _ = circuits._delete_qubit(parent, record["qubit"])
    text = applications.circuit_text(parent)
    return hashlib.sha256(text.encode()).hexdigest() == checks["final_sha256"]
