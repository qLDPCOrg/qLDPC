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
  ``circuits.VERIFIED_FAMILIES`` under exact certification at K = t, in two orders
  (``applications.py``);
  ``prune_sensitivity`` runs only on the circuits of ``prune.sensitivity_inputs``.

The counts use these symbols: N error mechanisms of the DEM; N_0 reduced mechanisms, one per
distinct nonzero signature (flags, syndrome); z of them with zero flags; p post-selection
detectors, the bits of the flags; r the GF(2) rank of the flags; K' = min(K, r + 1), above which
no flagged fault is accepted and irreducible; h heavy flagged mechanisms, whose syndrome has no
error of weight 1; q flag classes, the flagged mechanisms grouped by equal flags; q_h start
classes, those with a heavy member; Pi, E, and Phi by weight, as ``counting.py`` defines them; and
U, the sum over 1 <= m <= K of C(N_0, m).

The count and ground-truth arms take the DEM from an untimed certification with
``max_weight=1``, and their measured call is the counting itself, so that the runner's limit
applies to it.  Before timing, every run except those of ``applications.ARMS`` certifies C-goto
and E-chen with both methods, untimed, and a run of ``brute_bz`` also runs the baseline on C-goto.
A run of ``cert_bz``, ``cert_cc``, or ``brute_bz`` that finishes in under ``repeats.below_s`` gets
``repeats.count`` more runs (``repeat_tasks``).
"""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from experiments.state_prep_certification.certification import baseline, counting, instrument
from experiments.state_prep_certification.common import runner
from qldpc import circuits as qldpc_circuits
from qldpc import math as qldpc_math
from qldpc.codes.distance import CosetLeaderSearch, DistanceMethod

NAME = "certification"
EXTRA_WHEELS: list[str] = []
DATA = [
    "circuit_data/inputs",
    "circuit_data/synthesized",
    "circuit_data/synthesized_mqt_1.9.0",
    "circuit_data/codes",
]
FROZEN_FILES = ["config.json", "circuit_data/inputs/circuits.json"]
HERE = pathlib.Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
LIMITS = CONFIG["limits"]
JOB_MEM_GIB = float(CONFIG["mem_cap_gib"])
INPUTS = HERE / "circuit_data" / "inputs"
CIRCUITS_JSON = INPUTS / "circuits.json"

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
METHODS: dict[str, DistanceMethod] = {
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

# The circuits that every run certifies before timing.
WARM_UP_CIRCUITS = ("cgoto", "echen")


def prepare(label_dir: pathlib.Path) -> None:
    """Check every input against ``circuits.json``, and copy ``circuits.json`` into the label.

    See ``circuits.check_inputs``, which stops the run if any input differs.  Any ``circuits.json``
    that ``circuits.py --build`` wrote may be run; it enters the label's hash, and its copy in the
    label directory travels in the label's archive, from which ``analyze.py`` reads it.
    """
    from experiments.state_prep_certification.certification import circuits

    circuits.check_inputs()
    shutil.copyfile(CIRCUITS_JSON, label_dir / CIRCUITS_JSON.name)


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """The tasks of a subset: ``main``, ``check``, or ``smoke``.

    ``main`` has certification, the brute-force baseline, the shadow run, the counts, the margins,
    and the deletions; ``check`` the ground truth; and ``smoke`` every arm that applies to C-goto,
    E-chen, and A-ft steane.  In every subset, ``brute_bz`` runs only where ``circuits.json`` puts U
    within the cap, ``ground_truth`` only where its rule applies, ``margin`` only where r > t,
    ``prune`` only on the families of ``circuits.VERIFIED_FAMILIES``, and ``prune_sensitivity`` only
    on the circuits of ``prune.sensitivity_inputs`` of ``config.json``.
    """
    from experiments.state_prep_certification.certification import circuits

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
            if arm == "prune" and entry["family"] not in circuits.VERIFIED_FAMILIES:
                continue
            if (
                arm == "prune_sensitivity"
                and entry["id"] not in CONFIG["prune"]["sensitivity_inputs"]
            ):
                continue
            task_list.append(_task(arm, entry, 0, _limit(arm)))
    return task_list


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst-case seconds of a task of pass 1 with its repeats.

    Each run costs at most start-up cap + limit + grace + post-processing allowance, where the
    allowance is ``post_s`` for the first run of a certification and ``runner.FINISH_S`` for every
    other run, including those of ``brute_bz``.  An arm other than those of ``REPEATED_ARMS`` runs
    once.  A run of ``REPEATED_ARMS`` costs the larger of a slow run, which uses its whole limit,
    and a fast run, which takes at most ``repeats.below_s``, with its ``repeats.count`` repeats at
    the largest limit of ``repeat_tasks``.
    """
    first_post = float(task.get("post_s", runner.FINISH_S))
    slow = _run_cost(float(task["limit_s"]), first_post)
    if task["arm"] not in REPEATED_ARMS:
        return slow
    below, count = float(CONFIG["repeats"]["below_s"]), int(CONFIG["repeats"]["count"])
    fast = _run_cost(below, first_post) + count * _run_cost(_repeat_limit(below), runner.FINISH_S)
    return max(slow, fast)


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


def _run_cost(limit: float, post: float) -> float:
    """The worst-case seconds of one run with this limit and post-processing allowance."""
    return float(LIMITS["startup_cap_s"]) + limit + float(LIMITS["grace_s"]) + post


def _repeat_limit(first_seconds: float) -> float:
    """The limit of each repeat of a certification whose first run took T1 seconds."""
    repeats = CONFIG["repeats"]
    return max(float(repeats["limit_factor"]) * first_seconds, float(repeats["limit_floor_s"]))


####################################################################################################
# running a task (child process)


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Run one task of the certification experiment.

    A task's ``post_delay_s``, which only the acceptance tests set, delays the description of a
    certificate by that many seconds, within the post-processing allowance.  The tasks of
    ``applications.ARMS`` run in ``applications.run``, without warm-up.
    """
    from experiments.state_prep_certification.certification import applications

    entry = _entry(task["input"]["id"])
    if task["arm"] in applications.ARMS:
        return applications.run(task, entry)
    noisy = _noisy_input(entry)
    _warm_up()
    arm = task["arm"]
    options = certify_options(entry)
    max_weight = options["max_weight"]
    fields: dict[str, Any] = {"id": entry["id"], "family": entry["family"], "K": max_weight}

    if arm in METHODS or arm == "shadow":
        method: DistanceMethod = METHODS.get(arm, "brouwer_zimmermann")
        calls_file = None
        if arm == "shadow":
            (label_dir / "oracle_calls").mkdir(exist_ok=True)
            calls_file = label_dir / "oracle_calls" / f"{task['key']}.jsonl"
            calls_file.unlink(missing_ok=True)
        instrumentation = instrument.Instrumentation(
            shadow=arm == "shadow",
            progress=runner.send_partial,
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
        # every description takes well under a second, so that AT4 delays one to exceed a short
        # post-processing allowance
        time.sleep(float(task.get("post_delay_s", 0.0)))
        return _describe(certificate)

    if arm == "brute_bz":
        _warm_up_baseline()
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
        fields |= runner.timed(counting.brute_force, reduced, stabilizers, max_weight)
    return fields


def certify_options(entry: dict[str, Any]) -> dict[str, Any]:
    """The keyword options of ``certify_bounded_fault_propagation`` for a circuit, K included."""
    options = entry["options"]
    keys = ("add_flags", "post_select", "output_qubits", "css_weight", "max_weight")
    return {key: options[key] for key in keys}


def noisy_circuit(circuit: stim.Circuit, noise: dict[str, Any]) -> stim.Circuit:
    """The given noiseless circuit with the noise of ``noise``.

    ``noise["model"]`` is the import path of a noise model of ``qldpc.circuits``, such as
    ``qldpc.circuits.DepolarizingNoiseModel``, and ``noise["p"]`` its probability.
    """
    module, name = noise["model"].rsplit(".", 1)
    model = getattr(importlib.import_module(module), name)
    return model(noise["p"]).noisy_circuit(circuit)


def stabilizer_matrix(stabilizers: Any) -> npt.NDArray[np.uint8]:
    """State stabilizers, as ``stim.PauliString`` objects or their text, as rows ``[x | z]``.

    Text is parsed by stim first, since ``string_to_op`` would take a leading sign for a qubit.
    """
    return np.array(
        [qldpc_math.string_to_op(stim.PauliString(s)) for s in stabilizers], dtype=np.uint8
    )


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


def _describe(certificate: Any) -> dict[str, Any]:
    """The untimed description of a certificate, with the exact residual of a counterexample.

    The description of a counterexample also gives its sector (``applications.sector_fields``).
    """
    from experiments.state_prep_certification.certification import applications

    fields: dict[str, Any] = {"certificate_text": str(certificate)}
    example = certificate.counterexample
    if example is None:
        return fields
    num_qubits = len(certificate.output_qubits)
    fields |= {
        "fault_indices": list(example.fault_indices),
        "output_pauli": str(example.output_pauli),
        "residual_weight_lower_bound": example.residual_weight_lower_bound,
    }
    oracle = CosetLeaderSearch(
        stabilizer_matrix(certificate.state_stabilizers),
        css_weight=certificate.css_weight,
        method="brouwer_zimmermann",
    )
    batch = oracle.pack_syndromes(np.array(example.output_syndrome, dtype=np.uint8))
    # with a cap of the number of output qubits, the returned weight is exact
    weight = int(oracle.get_min_weights(batch, num_qubits)[0])
    fields["residual_weight"] = weight
    fields["residual_exceeds_fault_weight"] = weight > len(example.fault_indices)
    return fields | applications.sector_fields(certificate)


def _entry(circuit_id: str) -> dict[str, Any]:
    """The record of a circuit in ``circuits.json``."""
    for entry in json.loads(CIRCUITS_JSON.read_text())["circuits"]:
        if entry["id"] == circuit_id:
            return entry
    raise KeyError(circuit_id)


def _noisy_input(entry: dict[str, Any]) -> stim.Circuit:
    """The circuit's frozen noiseless input, with the noise of ``circuits.json``."""
    circuit = stim.Circuit((INPUTS / f"{entry['id']}.stim").read_text())
    return noisy_circuit(circuit, entry["options"]["noise"])


def _warm_up() -> None:
    """Certify C-goto and E-chen with both methods and their own options, untimed.

    C-goto's search ends at weight 2, and E-chen's reaches weight 3, so that the code paths of
    weights 3 and more are compiled before timing too.
    """
    for circuit_id in WARM_UP_CIRCUITS:
        entry = _entry(circuit_id)
        noisy = _noisy_input(entry)
        for method in METHODS.values():
            qldpc_circuits.certify_bounded_fault_propagation(
                noisy, method=method, **certify_options(entry)
            )


def _warm_up_baseline() -> None:
    """Run the brute-force baseline on C-goto, untimed, which compiles its enumeration."""
    entry = _entry("cgoto")
    with baseline.BruteForce().install():
        qldpc_circuits.certify_bounded_fault_propagation(
            _noisy_input(entry), method="brouwer_zimmermann", **certify_options(entry)
        )
