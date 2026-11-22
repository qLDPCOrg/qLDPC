# SPDX-License-Identifier: Apache-2.0

"""The plug-in of the certification experiment (see ``common/plugin.py`` for the interface).

Each task is one arm on one circuit of ``circuits.json``, with K the circuit's ``max_weight``,
t + 1 for design t.  The README's Arms table says what each arm does, and its Terms define the
symbols.  Each arm times one call (``runner.timed``), whose wall time T includes:

* ``cert_bz``, ``cert_cc``, ``shadow``, and ``brute_bz``: the certification, DEM construction
  included; for ``shadow``, with the CC oracle's answers, and for ``brute_bz``, with the
  baseline's count of distinct syndromes (``baseline.py``).  After T, within the post-processing
  allowance, the first run of ``cert_bz`` or ``cert_cc`` describes the certificate
  (``applications.describe_certificate``): its text and, for a counterexample, its fault indices,
  output Pauli, lower bound, and exact residual weight.  Repeats, ``shadow``, and ``brute_bz``
  have no post-processing.
* ``count_accepted``, ``count_listing``, and ``ground_truth``: the counting itself, so that the
  runner's limit applies to it.  These arms take the DEM from an untimed certification with
  ``max_weight=1``.  That certification is preparation, not a warm-up, but it also compiles, or
  loads from the cache, the code that their measured calls share with it.
* ``margin``: the whole margin search (``applications.find_margin``), DEM construction and the
  description of its counterexample included.
* ``prune`` and ``prune_sensitivity``: the whole deletion task (``applications.prune``): the
  parent's certification, every trial deletion, the final circuit's CC certification and
  acceptance estimates, and its margin search with the description of its counterexample.

No run warms up before timing, apart from ``ground_truth``, whose task process first compiles
galois's GF(2) arithmetic, untimed (``counting._compile_galois``).  The runner retries any run
whose measured call compiled (``common/runner.py``).  A run of ``cert_bz``, ``cert_cc``, or
``brute_bz`` that finishes in under ``repeats.below_s`` gets ``repeats.count`` more runs
(``repeat_tasks``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import time
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


@dataclasses.dataclass(frozen=True)
class Arm:
    """How the tasks of an arm run.

    Attributes:
        mode: The ``mode`` of the arm's tasks.
        limit_key: The key of the arm's limit in ``limits`` of ``config.json``.
        repeated: Whether a run that finishes in under ``repeats.below_s`` gets repeats
            (``repeat_tasks``).
        method: The oracle of a certification arm, ``cert_bz`` or ``cert_cc``, whose first run
            describes its certificate within the post-processing allowance ``post_s``; None for the
            other arms.
    """

    mode: str
    limit_key: str
    repeated: bool = False
    method: CosetMethod | None = None


# The arms, in the order of a circuit's tasks.
ARMS = {
    "cert_bz": Arm("timed", "cert_s", repeated=True, method="brouwer_zimmermann"),
    "cert_cc": Arm("timed", "cert_s", repeated=True, method="connected_cluster"),
    "brute_bz": Arm("brute", "cert_s", repeated=True),
    "shadow": Arm("shadow", "shadow_s"),
    "count_accepted": Arm("count", "count_s"),
    "count_listing": Arm("count", "count_s"),
    "ground_truth": Arm("ground_truth", "ground_truth_s"),
    "margin": Arm("margin", "margin_s"),
    "prune": Arm("prune", "prune_s"),
    "prune_sensitivity": Arm("prune", "prune_s"),
}

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
    E-chen, and the A-ft circuits of the Steane code.  In every subset, ``brute_bz`` runs only where
    ``circuits.json`` puts U within the cap, ``ground_truth`` only where its rule applies,
    ``margin`` only where r > t, ``prune`` only on the families of ``settings.VERIFIED_FAMILIES``,
    and ``prune_sensitivity`` only on the circuits of ``prune.sensitivity_inputs`` of
    ``config.json``.
    """
    entries = [e for e in json.loads(CIRCUITS_JSON.read_text())["circuits"] if e["status"] == "ok"]
    if subset == "smoke":
        entries = [
            e for e in entries if e["family"] in SMOKE_FAMILIES or e["id"].startswith(SMOKE_PREFIX)
        ]
        arms = tuple(ARMS)
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
            task_list.append(_task(arm, entry))
    return task_list


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst-case seconds of a task of pass 1 with its repeats (``repeat_tasks``).

    See ``tasks.repeated_worst_case_s``.  Only the first run of a certification has a
    post-processing allowance ``post_s``.
    """
    return repeated_worst_case_s(task, repeat_tasks, float(CONFIG["repeats"]["below_s"]))


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """The repeats of a run of a repeated arm that finished in under ``repeats.below_s``.

    There are ``repeats.count`` repeats, each with the limit max(``limit_factor`` T1,
    ``limit_floor_s``) of ``repeats`` for the first run's T1, which is ``limit_floor_s`` whenever
    ``limit_factor`` times ``below_s`` is at most ``limit_floor_s``, as in ``config.json``.
    """
    below, count = CONFIG["repeats"]["below_s"], CONFIG["repeats"]["count"]
    if not ARMS[task["arm"]].repeated or row["status"] != "ok" or row["T"] >= below:
        return []
    entry = {"id": task["input"]["id"]}
    limit = _repeat_limit(row["T"])
    return [_task(task["arm"], entry, repeat, limit) for repeat in range(1, count + 1)]


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """Run the acceptance tests of ``acceptance.py``."""
    from experiments.state_prep_certification.certification import acceptance as tests

    return tests.run_all(node=node, results_dir=results_dir)


def _task(
    arm: str, entry: dict[str, Any], repeat: int = 0, limit: float | None = None
) -> dict[str, Any]:
    """A task of one arm on one circuit; only the first run of a certification has ``post_s``.

    Its limit is ``limit``, if given, and otherwise the arm's shared allowance in ``config.json``.
    """
    task = {
        "key": f"{arm}.{entry['id']}.r{repeat}",
        "experiment": NAME,
        "arm": arm,
        "mode": ARMS[arm].mode,
        "repeat": repeat,
        "limit_s": float(LIMITS[ARMS[arm].limit_key]) if limit is None else limit,
        "mem_cap_gib": JOB_MEM_GIB,
        "startup_cap_s": float(LIMITS["startup_cap_s"]),
        "grace_s": float(LIMITS["grace_s"]),
        "input": {"id": entry["id"]},
    }
    if ARMS[arm].method is not None and repeat == 0:
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
    The tasks of the modes ``margin`` and ``prune`` run in ``applications.run``.
    """
    entry = _entry(task["input"]["id"])
    arm = task["arm"]
    mode = ARMS[arm].mode
    if mode in ("margin", "prune"):
        return applications.run(task, entry)
    fields: dict[str, Any] = {
        "id": entry["id"],
        "family": entry["family"],
        "K": entry["options"]["max_weight"],
    }
    if mode in ("timed", "shadow"):
        return _run_certification(task, entry, fields, label_dir)
    if mode == "brute":
        return fields | _run_baseline(entry)
    return fields | _run_counts(arm, entry)


def _run_certification(
    task: dict[str, Any], entry: dict[str, Any], fields: dict[str, Any], label_dir: pathlib.Path
) -> dict[str, Any]:
    """Run ``cert_bz``, ``cert_cc``, or ``shadow``: the instrumented certification of a circuit.

    The first run of ``cert_bz`` or ``cert_cc`` sends ``fields`` with its verdict as its result, and
    returns the description of its certificate, within the post-processing allowance.
    """
    arm = task["arm"]
    noisy = noisy_input(entry)
    options = certify_options(entry["options"])
    method: CosetMethod = ARMS[arm].method or "brouwer_zimmermann"
    calls_file = None
    if arm == "shadow":
        (label_dir / "oracle_calls").mkdir(exist_ok=True)
        calls_file = label_dir / "oracle_calls" / f"{task['key']}.jsonl"
        calls_file.unlink(missing_ok=True)
    instrumentation = instrument.Instrumentation(
        shadow=arm == "shadow",
        progress=applications.delayed_progress(float(task.get("weight_delay_s", 0.0))),
        clock=runner.elapsed,
        calls_file=calls_file,
    )
    with instrumentation.install():
        certificate = runner.timed(
            qldpc_circuits.certify_bounded_fault_propagation, noisy, method=method, **options
        )
    fields = {
        **fields,
        **_verdict(method, certificate),
        "T_setup": _setup_seconds(instrumentation.search_seconds),
        **instrumentation.summary(),
    }
    if arm == "shadow" or task["repeat"]:
        return fields
    runner.send_result(fields)
    # AT4 delays the description, so that it overruns a short post-processing allowance however
    # fast the description itself is
    time.sleep(float(task.get("post_delay_s", 0.0)))
    return applications.describe_certificate(certificate)


def _run_baseline(entry: dict[str, Any]) -> dict[str, Any]:
    """Run ``brute_bz``: the BZ certification of a circuit with the brute-force fault search."""
    noisy = noisy_input(entry)
    options = certify_options(entry["options"])
    brute = baseline.BruteForce(progress=runner.send_partial, clock=runner.elapsed)
    with brute.install():
        certificate = runner.timed(
            qldpc_circuits.certify_bounded_fault_propagation,
            noisy,
            method="brouwer_zimmermann",
            **options,
        )
    return {
        **_verdict("brouwer_zimmermann", certificate),
        "U": entry["U"],
        "T_setup": _setup_seconds(brute.search_seconds),
        **brute.summary(),
    }


def _run_counts(arm: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Run ``count_accepted``, ``count_listing``, or ``ground_truth`` on a circuit.

    The counts and the ground truth start from the DEM of an untimed certification at weight 1.
    """
    noisy = noisy_input(entry)
    options = certify_options(entry["options"])
    max_weight = options["max_weight"]
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
    fields = {
        "N": reduced.num_mechanisms,
        "N0": reduced.num_reduced,
        "z": reduced.num_zero_flag,
        "p": len(reference.post_selection_detectors),
        "r": reduced.rank,
        "K_prime": weight_limit,
    }
    if arm == "count_accepted":
        return fields | runner.timed(
            counting.accepted_counts,
            reduced.flags,
            max_weight,
            int(CONFIG["accepted_dp_max_entries"]),
        )
    if arm == "count_listing":
        return fields | runner.timed(
            _listing, reduced, stabilizers, reference.css_weight, weight_limit
        )
    # the ground truth uses galois's GF(2) arithmetic, which this compiles first
    counting._compile_galois()
    return fields | runner.timed(counting.ground_truth, reduced, stabilizers, max_weight)


def _verdict(method: CosetMethod, certificate: Any) -> dict[str, Any]:
    """The oracle and the verdict of a certification, with the weight of its counterexample."""
    example = certificate.counterexample
    return {
        "method": method,
        "css_weight": certificate.css_weight,
        "certified": certificate.certified,
        "violation_weight": None if example is None else len(example.fault_indices),
    }


def _setup_seconds(search_seconds: float | None) -> float | None:
    """T_setup = T - T_search of the measured certification, or None if it did not search."""
    return None if search_seconds is None else runner.measured_seconds() - search_seconds


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


def _entry(circuit_id: str) -> dict[str, Any]:
    """The record of a circuit in ``circuits.json``."""
    for entry in json.loads(CIRCUITS_JSON.read_text())["circuits"]:
        if entry["id"] == circuit_id:
            return entry
    raise KeyError(circuit_id)
