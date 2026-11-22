# SPDX-License-Identifier: Apache-2.0

"""The plug-in of the certification experiment (see ``common/plugin.py`` for the interface).

Each task is one arm on one circuit of ``circuits.json``.  K is the circuit's ``max_weight``,
t + 1 for design t:

* ``cert_bz``, ``cert_cc``: ``certify_bounded_fault_propagation`` with the circuit's options, by
  the Brouwer-Zimmermann or connected-cluster oracle, instrumented (``instrument.py``).  T is the
  wall time of the call, DEM construction included.  After T, within the post-processing
  allowance, the first run describes the certificate: its text and, for a counterexample, its
  fault indices, output Pauli, lower bound, and exact residual weight.  Repeats skip this
  description.
* ``brute_bz``: the same BZ certification with qLDPC's fault search replaced by the brute-force
  baseline (``baseline.py``), which queries every accepted fault combination of weight at most K
  with the certifier's own oracle.  It runs only on the circuits whose enumeration size U is
  within the cap of ``config.json``, as ``circuits.json`` records; it has no post-processing.
* ``shadow``: BZ certification in which every oracle query is also answered by a CC oracle.
* ``count_listing``: the counts of the listing (``counting.py``) on the reduced DEM.
* ``count_accepted``: A_m and C(N_0, m) for m <= K.
* ``ground_truth``: brute force on the small circuits that ``circuits.json`` marks, in the subset
  ``check``.
* ``margin``: the smallest violating fault weight m* of a circuit whose flag rank r exceeds its
  design t, by BZ certification with no limit on the fault weight (``applications.py``).
* ``prune`` and ``prune_sensitivity``: deletion of the measured qubits of a circuit of
  ``settings.VERIFIED_FAMILIES`` under exact certification at K = t, in two orders
  (``applications.py``);
  ``prune_sensitivity`` runs only on the circuits of ``prune.sensitivity_inputs``.

The counts use these symbols: N error mechanisms of the DEM; N_0 reduced mechanisms, one per
distinct nonzero signature (flags, syndrome); z of them with zero flags; p post-selection
detectors, the bits of the flags; r the GF(2) rank of the flags; K' = min(K, r + 1), above which
no flagged fault is accepted and irreducible; h heavy flagged mechanisms, whose syndrome has no
error of weight 1; q flag classes, the flagged mechanisms grouped by equal flags; q_h start
classes, those with a heavy member; Pi (the prefixes of a naive enumeration of class codewords,
not qLDPC's work), E, and Phi by weight, as ``counting.py`` defines them; and U, the sum over
1 <= m <= K of C(N_0, m).

The count and ground-truth arms take the DEM from an untimed certification with ``max_weight=1``,
and their measured call is the counting itself, so that the runner's limit applies to it.  The
runner retries any run whose measured call compiled (``common/runner.py``).  No run warms up before
timing, apart from ``ground_truth``, which first compiles galois's arithmetic, untimed
(``counting._compile_galois``).  A run of ``cert_bz``, ``cert_cc``, or ``brute_bz`` that finishes in
under ``repeats.below_s`` gets ``repeats.count`` more runs (``repeat_tasks``).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from experiments.state_prep_certification.certification import (
    applications,
    baseline,
    circuits,
    counting,
    instrument,
)
from experiments.state_prep_certification.certification.settings import (
    CIRCUITS_JSON,
    CONFIG,
    VERIFIED_FAMILIES,
    certify_options,
    noisy_input,
)
from experiments.state_prep_certification.certification.states import stabilizer_matrix
from experiments.state_prep_certification.common import runner
from experiments.state_prep_certification.common.tasks import repeated_worst_case_s
from qldpc import circuits as qldpc_circuits
from qldpc.codes.distance import CosetLeaderSearch, CosetMethod

NAME = "certification"
EXTRA_WHEELS: list[str] = []
DATA = [
    "circuit_data/inputs",
    "circuit_data/synthesized",
    "circuit_data/synthesized_mqt_1.9.0",
    "circuit_data/codes",
]
FROZEN_FILES = ["config.json", "circuit_data/inputs/circuits.json"]
LIMITS = CONFIG["limits"]
JOB_MEM_GIB = float(CONFIG["mem_cap_gib"])

ARMS: tuple[str, ...] = (
    "cert_bz",
    "cert_cc",
    "brute_bz",
    "shadow",
    "count_accepted",
    "count_listing",
    "ground_truth",
    "margin",
    "prune",
    "prune_sensitivity",
)
METHODS: dict[str, CosetMethod] = {
    "cert_bz": "brouwer_zimmermann",
    "cert_cc": "connected_cluster",
}
MODES = {
    "cert_bz": "timed",
    "cert_cc": "timed",
    "brute_bz": "brute",
    "shadow": "shadow",
    "count_accepted": "count",
    "count_listing": "count",
    "ground_truth": "ground_truth",
    "margin": "margin",
    "prune": "prune",
    "prune_sensitivity": "prune",
}

# The arms whose fast runs are repeated (``repeat_tasks``).
REPEATED_ARMS = ("cert_bz", "cert_cc", "brute_bz")

# The circuits of the ``smoke`` subset.
SMOKE_FAMILIES = ("C-goto", "E-chen")
SMOKE_PREFIX = "aft.steane."


def prepare(label_dir: pathlib.Path) -> None:
    """Check every input against ``circuits.json``, and copy ``circuits.json`` into the label.

    See ``circuits.check_inputs``, which stops the run if any input differs.  Any ``circuits.json``
    that ``circuits.py --build`` wrote may be run; it enters the label's hash, and its copy in the
    label directory travels in the label's archive, from which ``analyze.py`` reads it.
    """
    circuits.check_inputs()
    shutil.copyfile(CIRCUITS_JSON, label_dir / CIRCUITS_JSON.name)


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """The tasks of a subset: ``main``, ``check``, or ``smoke``.

    ``main`` has certification, the brute-force baseline, the shadow run, the counts, the margins,
    and the deletions; ``check`` the ground truth; and ``smoke`` every arm that applies to C-goto,
    E-chen, and A-ft steane.  In every subset, ``brute_bz`` runs only where ``circuits.json`` puts U
    within the cap, ``ground_truth`` only where its rule applies, ``margin`` only where r > t,
    ``prune`` only on the families of ``settings.VERIFIED_FAMILIES``, and ``prune_sensitivity`` only
    on the circuits of ``prune.sensitivity_inputs`` of ``config.json``.
    """
    entries = [e for e in json.loads(CIRCUITS_JSON.read_text())["circuits"] if e["status"] == "ok"]
    if subset == "smoke":
        entries = [
            e for e in entries if e["family"] in SMOKE_FAMILIES or e["id"].startswith(SMOKE_PREFIX)
        ]
        arms = ARMS
    elif subset == "main":
        arms = tuple(arm for arm in ARMS if arm != "ground_truth")
    elif subset == "check":
        arms = ("ground_truth",)
    else:
        raise ValueError(f"Unknown subset {subset!r}; choose main, check, or smoke")
    task_list = []
    for entry in entries:
        for arm in arms:
            if arm == "ground_truth" and not entry["ground_truth"]:
                continue
            if arm == "brute_bz" and not entry["brute_force"]:
                continue
            if arm == "margin" and entry["r"] <= entry["t"]:
                continue
            if arm == "prune" and entry["family"] not in VERIFIED_FAMILIES:
                continue
            if (
                arm == "prune_sensitivity"
                and entry["id"] not in CONFIG["prune"]["sensitivity_inputs"]
            ):
                continue
            task_list.append(_task(arm, entry, 0, _limit(arm)))
    return task_list


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst-case seconds of a task of pass 1 with its repeats (``repeat_tasks``).

    See ``tasks.repeated_worst_case_s``.  Only the first run of a certification has a
    post-processing allowance ``post_s``.
    """
    return repeated_worst_case_s(task, repeat_tasks, float(CONFIG["repeats"]["below_s"]))


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """The repeats of a run of ``REPEATED_ARMS`` that finished in under ``repeats.below_s``.

    There are ``repeats.count`` repeats, each with the limit max(``limit_factor`` T1,
    ``limit_floor_s``) of ``repeats`` for the first run's T1, which is 600 s at the configured
    values.
    """
    below, count = CONFIG["repeats"]["below_s"], CONFIG["repeats"]["count"]
    if task["arm"] not in REPEATED_ARMS or row["status"] != "ok" or row["T"] >= below:
        return []
    entry = {"id": task["input"]["id"]}
    limit = _repeat_limit(row["T"])
    return [_task(task["arm"], entry, repeat, limit) for repeat in range(1, count + 1)]


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """Run the acceptance tests of ``acceptance.py``."""
    from experiments.state_prep_certification.certification import acceptance as tests

    return tests.run_all(node=node, results_dir=results_dir)


def _limit(arm: str) -> float:
    """The shared allowance of an arm's task."""
    if arm in REPEATED_ARMS:
        return float(LIMITS["cert_s"])
    if arm == "shadow":
        return float(LIMITS["shadow_s"])
    if arm == "ground_truth":
        return float(LIMITS["ground_truth_s"])
    if arm == "margin":
        return float(LIMITS["margin_s"])
    if arm in ("prune", "prune_sensitivity"):
        return float(LIMITS["prune_s"])
    return float(LIMITS["count_s"])


def _task(arm: str, entry: dict[str, Any], repeat: int, limit: float) -> dict[str, Any]:
    """A task of one arm on one circuit; only the first run of a certification has ``post_s``."""
    task = {
        "key": f"{arm}.{entry['id']}.r{repeat}",
        "experiment": NAME,
        "arm": arm,
        "mode": MODES[arm],
        "repeat": repeat,
        "limit_s": limit,
        "mem_cap_gib": JOB_MEM_GIB,
        "startup_cap_s": float(LIMITS["startup_cap_s"]),
        "grace_s": float(LIMITS["grace_s"]),
        "input": {"id": entry["id"]},
    }
    if arm in METHODS and repeat == 0:
        task["post_s"] = float(LIMITS["post_s"])
    return task


def _repeat_limit(first_seconds: float) -> float:
    """The limit of each repeat of a certification whose first run took T1 seconds."""
    repeats = CONFIG["repeats"]
    return max(float(repeats["limit_factor"]) * first_seconds, float(repeats["limit_floor_s"]))


####################################################################################################
# running a task (child process)


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Run one task of the certification experiment.

    A task's ``post_delay_s``, which only the acceptance tests set, delays the description of a
    certificate by that many seconds, within the post-processing allowance.  Its ``weight_delay_s``,
    which only AT4 sets, makes a certification sleep that many seconds after it reports each
    completed fault weight, so that a limit can stop it between weights on a machine of any speed.
    The tasks of ``applications.ARMS`` run in ``applications.run``.
    """
    entry = _entry(task["input"]["id"])
    if task["arm"] in applications.ARMS:
        return applications.run(task, entry)
    noisy = noisy_input(entry)
    arm = task["arm"]
    options = certify_options(entry["options"])
    max_weight = options["max_weight"]
    fields: dict[str, Any] = {"id": entry["id"], "family": entry["family"], "K": max_weight}

    if arm in METHODS or arm == "shadow":
        method: CosetMethod = METHODS.get(arm, "brouwer_zimmermann")
        calls_file = None
        if arm == "shadow":
            (label_dir / "oracle_calls").mkdir(exist_ok=True)
            calls_file = label_dir / "oracle_calls" / f"{task['key']}.jsonl"
            calls_file.unlink(missing_ok=True)
        delay = float(task.get("weight_delay_s", 0.0))
        instrumentation = instrument.Instrumentation(
            shadow=arm == "shadow",
            progress=_delayed(runner.send_partial, delay) if delay else runner.send_partial,
            clock=runner.elapsed,
            calls_file=calls_file,
        )
        with instrumentation.install():
            certificate = runner.timed(
                qldpc_circuits.certify_bounded_fault_propagation, noisy, method=method, **options
            )
        measured = runner.measured_seconds()
        fields |= {
            "method": method,
            "css_weight": certificate.css_weight,
            "certified": certificate.certified,
            "violation_weight": (
                None
                if certificate.counterexample is None
                else len(certificate.counterexample.fault_indices)
            ),
            "T_setup": None
            if instrumentation.search_seconds is None
            else (measured - instrumentation.search_seconds),
            **instrumentation.summary(),
        }
        if arm == "shadow" or task["repeat"]:
            return fields
        runner.send_result(fields)
        # AT4 delays the description, so that it overruns a short post-processing allowance however
        # fast the description itself is
        time.sleep(float(task.get("post_delay_s", 0.0)))
        return applications.describe_certificate(certificate)

    if arm == "brute_bz":
        brute = baseline.BruteForce(progress=runner.send_partial, clock=runner.elapsed)
        with brute.install():
            certificate = runner.timed(
                qldpc_circuits.certify_bounded_fault_propagation,
                noisy,
                method="brouwer_zimmermann",
                **options,
            )
        measured = runner.measured_seconds()
        return fields | {
            "method": "brouwer_zimmermann",
            "css_weight": certificate.css_weight,
            "certified": certificate.certified,
            "violation_weight": (
                None
                if certificate.counterexample is None
                else len(certificate.counterexample.fault_indices)
            ),
            "U": entry["U"],
            "T_setup": None if brute.search_seconds is None else measured - brute.search_seconds,
            **brute.summary(),
        }

    # the counts and the ground truth start from the DEM of an untimed certification at weight 1
    reference = qldpc_circuits.certify_bounded_fault_propagation(
        noisy, method="brouwer_zimmermann", **(options | {"max_weight": 1})
    )
    flags, syndromes = counting.dem_signatures(
        reference.detector_error_model,
        reference.post_selection_detectors,
        len(reference.state_stabilizers),
    )
    reduced = counting.reduce_signatures(flags, syndromes)
    weight_limit = reduced.max_fault_weight(max_weight)
    stabilizers = stabilizer_matrix(reference.state_stabilizers)
    fields |= {
        "N": reduced.num_mechanisms,
        "N0": reduced.num_reduced,
        "z": reduced.num_zero_flag,
        "p": len(reference.post_selection_detectors),
        "r": reduced.rank,
        "K_prime": weight_limit,
    }
    if arm == "count_accepted":
        fields |= runner.timed(
            counting.accepted_counts,
            reduced.flags,
            max_weight,
            int(CONFIG["accepted_dp_max_entries"]),
        )
    elif arm == "count_listing":
        fields |= runner.timed(_listing, reduced, stabilizers, reference.css_weight, weight_limit)
    else:
        # the ground truth uses galois's GF(2) arithmetic, which this compiles first
        counting._compile_galois()
        fields |= runner.timed(counting.ground_truth, reduced, stabilizers, max_weight)
    return fields


def _listing(
    reduced: counting.ReducedDem,
    stabilizers: npt.NDArray[np.uint8],
    css_weight: bool,
    weight_limit: int,
) -> dict[str, Any]:
    """Classify the flagged mechanisms with the BZ oracle, then count the listing by weight.

    The class parameters q, q_h, and h, and then the counts of each completed weight, are also
    sent as partial results.
    """
    oracle = CosetLeaderSearch(stabilizers, css_weight=css_weight, method="brouwer_zimmermann")
    flagged = [i for i, flag in enumerate(reduced.flags) if flag]
    heavy = [
        not oracle.get_min_weight_bound(reduced.syndromes[i], 1).within_max_weight for i in flagged
    ]
    classes = counting.get_classes([reduced.flags[i] for i in flagged], heavy)
    fields = {
        "q": len(classes.flags),
        "q_h": classes.num_start,
        "h": sum(heavy),
        "threshold_one_queries": {
            "zero_flag": reduced.num_zero_flag,
            "flagged": len({reduced.syndromes[i] for i in flagged}),
        },
    }

    # the class parameters go first as a partial result, so that a run killed while listing keeps
    # them beside the weights that it completed
    runner.send_partial(fields)
    return fields | counting.listing_counts(classes, weight_limit, progress=runner.send_partial)


def _delayed(
    send: Callable[[dict[str, Any]], None], delay: float
) -> Callable[[dict[str, Any]], None]:
    """``send``, followed by a sleep of ``delay`` seconds."""

    def send_then_sleep(record: dict[str, Any]) -> None:
        send(record)
        time.sleep(delay)

    return send_then_sleep


def _entry(circuit_id: str) -> dict[str, Any]:
    """The record of a circuit in ``circuits.json``."""
    for entry in json.loads(CIRCUITS_JSON.read_text())["circuits"]:
        if entry["id"] == circuit_id:
            return entry
    raise KeyError(circuit_id)
