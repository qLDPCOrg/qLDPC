# SPDX-License-Identifier: Apache-2.0

"""The analysis of the margins and the deletions, for ``analyze.py``.

**Combined margin.**  m* is the smallest violating fault weight of a circuit, or "all" if none
exists at any fault weight.  ``cert_bz`` and ``cert_cc`` at K = t + 1 determine m* if they find a
violation, or "all" if they certify a circuit whose cutoff r + 1 is at most K.  A certification at a
K below the cutoff r + 1, or the completed weights without a violation of a run that did not finish,
give only a lower bound m* > k.  The ``margin`` arm determines m* or "all" if it finishes, and gives
a lower bound by its completed weights otherwise.  All values that determine m* must be equal and
exceed every lower bound; otherwise the circuit is a disagreement.  The combined margin is the
determined value, in the column ``margin``, or else the largest lower bound k, in the column
``margin_lower_bound``, meaning m* > k.

**Deletions.**  The parent verdict of each ``prune`` and ``prune_sensitivity`` task at K = t must
agree with ``cert_bz``'s at K = t + 1: the parent is certified at t exactly when ``cert_bz``
certifies or violates at t + 1, and otherwise both violate at the same weight.  Each final circuit
must also be certified by the connected-cluster oracle, by the task's result or, for a run killed
during the search of its final m*, by the final checks that it sent as a partial result.  A failure
of either check, the parent verdict or the final circuit's certification, makes the circuit a
disagreement.  The final circuit is rebuilt from the parent's input in ``circuit_data/inputs/`` and
the accepted deletions, and compared with its recorded SHA-256, if that input is the one that the
label ran on.

Outputs, besides the margin fields of ``summary.csv`` and the margin column of ``main_table.csv``:

* ``margin_footnote.csv``: each circuit with m* <= t, with the residual weights of the X and Z parts
  of its counterexample and the parts that exceed its fault weight, for the footnote of the margin
  column.
* ``deletion.csv``: every step of every deletion task, with its outcome.
* ``deletion_tasks.csv``: the fields of the deletion table for every finished deletion task whose
  parent is certified at K = t, with whether its circuit is a disagreement.
* ``deletion_table.csv``: one row per finished deletion task whose parent is certified at K = t, on
  a circuit without a disagreement, the BB [[72,12,6]] rows first: the family, with the MQT QECC
  release of a synthesized circuit, gate counts (encoder and verification), measured qubits,
  measurements, and acceptance probabilities of parent and final circuit, the final m*, and the
  deletions attempted and accepted.
* ``deletion_caption.json``: each arm's deletion order, the BB [[72,12,6]] inputs of the table
  (empty if the table has no BB [[72,12,6]] row), the parents not certified at K = t, the tasks that
  did not finish with their completed steps and any final checks, the disagreements, and the final
  circuits whose rebuild does not match.
* ``pruned/<arm>.<id>.stim``: each final circuit of the deletion table.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
INPUTS = HERE / "circuit_data" / "inputs"


####################################################################################################
# margins


def margin_fields(
    circuit: dict[str, Any],
    bz: dict[str, Any],
    cc: dict[str, Any],
    margin_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """The combined margin of a circuit, its agreement, and the sector of its counterexample.

    ``bz`` and ``cc`` are summaries of the certification runs (``analyze.summarize_certification``),
    and ``margin_rows`` the rows of the ``margin`` arm, if any.  ``margin`` is an integer m*, "all",
    or None if unresolved or a disagreement, and ``margin_lower_bound`` is k where m* > k is all
    that is known.  ``agree_margin`` is None if fewer than two determinations or bounds exist.
    """
    exact: dict[str, Any] = {}
    bounds: dict[str, int] = {}
    max_weight = circuit["options"]["max_weight"]
    margin_run = margin_rows[0] if margin_rows else None
    runs = [("cert_bz", bz), ("cert_cc", cc)]
    if margin_run is not None:
        runs.append(("margin", margin_run))
    # each run gives a value that determines m*, a lower bound on it, or nothing
    for name, run in runs:
        result = run.get("result") or {}
        if run["status"] == "ok" and name == "margin":
            exact[name] = result["margin"]
        elif run["status"] == "ok" and not result.get("certified"):
            exact[name] = result["violation_weight"]
        elif run["status"] == "ok" and circuit["r"] + 1 <= max_weight:
            exact[name] = "all"
        elif run["status"] == "ok":
            bounds[name] = max_weight
        else:
            completed = [
                record["m"]
                for record in run.get("partial") or []
                if "m" in record and not record["violation"]
            ]
            if completed:
                bounds[name] = max(completed)

    # every determined value must be the same, and exceed every lower bound
    values = set(exact.values())
    agree = len(values) <= 1 and all(
        value == "all" or value > bound for value in values for bound in bounds.values()
    )
    fields: dict[str, Any] = {
        "margin": None,
        "margin_lower_bound": None,
        "agree_margin": agree if len(exact) + len(bounds) > 1 else None,
    }
    if not agree:
        return fields
    if values:
        fields["margin"] = values.pop()
    elif bounds:
        fields["margin_lower_bound"] = max(bounds.values())

    # the sector of the smallest violation, from the first run that describes it
    if isinstance(fields["margin"], int):
        for name, run in runs:
            result = run.get("result") or {}
            if exact.get(name) == fields["margin"] and "sector_weight_x" in result:
                fields |= {
                    "margin_sector_x": result["sector_weight_x"],
                    "margin_sector_z": result["sector_weight_z"],
                    "margin_violating_parts": result["violating_parts"],
                }
                break
    return fields


def margin_column(members: list[dict[str, Any]]) -> dict[str, Any]:
    """The margin column of one family of the main table, over its circuits without disagreement.

    It counts the circuits with m* <= t (``margin_at_most_t``, listed in margin_footnote.csv),
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


####################################################################################################
# deletions and outputs


def agree_prune(bz: dict[str, Any], arms: dict[str, list[dict[str, Any]]]) -> bool | None:
    """Whether the deletion tasks of a circuit agree with ``cert_bz`` and with the CC oracle.

    None if nothing was compared.
    """
    from experiments.state_prep_certification.certification import applications

    compared = []
    for arm in applications.PRUNE_ORDERS:
        if not arms.get(arm):
            continue
        run = arms[arm][0]
        parent = _parent_verdict(run)
        if parent is not None and bz["status"] == "ok":
            certified_at_t = bz["result"].get("certified") or (
                bz["result"].get("violation_weight") == bz["result"].get("K")
            )
            if parent["parent_certified"]:
                compared.append(bool(certified_at_t))
            else:
                compared.append(
                    bz["result"].get("violation_weight") == parent["parent_violation_weight"]
                )
        checks = _final_checks(run)
        if checks is not None:
            compared.append(checks["final_certified_cc"] is True)
    return all(compared) if compared else None


def write_applications(
    per_circuit: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    out: pathlib.Path,
    inputs: pathlib.Path = INPUTS,
) -> None:
    """Write margin_footnote.csv and the deletion outputs; see the module docstring."""
    from experiments.state_prep_certification.certification import applications
    from experiments.state_prep_certification.certification.analyze import _status, _write_csv

    # the circuits of the footnote of the margin column
    verdicts = {row["id"]: row["verdict"] for row in summary_rows}
    footnote = [
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
    _write_csv(out / "margin_footnote.csv", footnote)

    # each deletion task's steps, its row of deletion_tasks.csv, and its row of the table or its
    # entry in the caption
    steps, table, tasks = [], [], []
    caption: dict[str, Any] = {
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
                caption["disagreements"].append({"id": circuit_id, "arm": arm})
            elif parent is not None and not parent["parent_certified"]:
                caption["not_certified_at_t"].append(
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
                caption["unfinished"].append(unfinished)
            else:
                if row["rebuild_matches"] is False:
                    caption["rebuild_mismatches"].append({"id": circuit_id, "arm": arm})
                (out / "pruned" / f"{arm}.{circuit_id}.stim").write_text(result["final_circuit"])
                table.append(row)
    table.sort(key=lambda row: ("bb72" not in row["id"], row["id"], row["arm"]))
    caption["bb72_inputs"] = sorted({row["id"] for row in table if "bb72" in row["id"]})
    _write_csv(out / "deletion.csv", steps)
    _write_csv(out / "deletion_table.csv", table)
    _write_csv(out / "deletion_tasks.csv", tasks)
    (out / "deletion_caption.json").write_text(json.dumps(caption, indent=1) + "\n")


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


def _table_row(
    circuit_id: str,
    circuit: dict[str, Any],
    arm: str,
    result: dict[str, Any],
    rebuilt: bool | None,
) -> dict[str, Any]:
    """The row of the deletion table of one finished deletion task."""
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
