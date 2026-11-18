# SPDX-License-Identifier: Apache-2.0

"""The plug-in of the ranking experiment (see ``common/plugin.py`` for the interface).

The candidates are the circuits of certification's ``circuits.json`` of the families of
``config.json`` with design t = 1, in groups of circuits that prepare the same signed state of the
same code.  Every task runs at each rate p of ``config.json``, with the noise model of the
circuit's options at rate p:

* ``histogram``: the total-weight and X-type residual-weight histograms of a circuit, with
  ``histogram_samples`` samples and the circuit's preparation options.
* ``calib``: the X-type calibration of the group's consumer (``consumer.py``), with
  ``calibration_samples`` samples, the consumer's fault distance, and the time of the two scores
  and two baselines on the group's first circuit, whose histograms it builds with
  ``score_timing.histogram_samples`` samples.  The row keeps the noiseless consumer and the block
  preparation, with their SHA-256.
* ``direct``: the stratified reference of Phi_true (``direct.py``), with ``direct.samples`` draws
  per sampled stratum.
* ``exact``: the exact leading-order coefficients of Phi_true and Phi_cal of each circuit of the
  group ``exact.group``, with the decoder frozen at ``exact.frozen_rate``.
* ``mc_check``: plain paired Monte Carlo of Phi_true on the circuits of ``mc_check``, by
  ``Consumer.estimate_excess_failure_probability``.
* ``repro``: one ``histogram`` and one ``direct`` task of ``repro``, each twice, with the same seed.

The seed of a task is the first 8 bytes of the SHA-256 of its key, and the two copies of a
``repro`` task share the seed of their key without the copy.  Each run calls ``runner.timed``
once, around the whole task.  The ``histogram``, ``calib``, and ``direct`` tasks also record the
CPU seconds of their parts for the cost comparison: setup (detector error models, decoders, and
mechanism lists), exactly listed strata, and sampled strata or cells.  Each sends its finished
parts as partial results, which a run killed at its limit keeps.  Admission of candidates by
their certification verdicts is left to ``analyze.py``, so every task runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shutil
import time
from typing import Any

import numpy as np
import stim

from experiments.state_prep_certification.common import runner
from experiments.state_prep_certification.ranking import consumer as ranking_consumer
from experiments.state_prep_certification.ranking import direct
from qldpc import circuits as qldpc_circuits
from qldpc.experimental.circuits.benchmarking import (
    ConsumerCalibration,
    ConsumerProfile,
    ResidualWeightHistogram,
    consumers,
    estimate_residual_weight_histogram,
    residual_weights,
)
from qldpc.objects import Pauli, PauliXZ

NAME = "ranking"
EXTRA_WHEELS: list[str] = []
# certification's bundle entry ships its inputs, so a ranking bundle also names certification
DATA: list[str] = []
FROZEN_FILES = ["config.json", "../certification/circuit_data/inputs/circuits.json"]
HERE = pathlib.Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
LIMITS = CONFIG["limits"]
JOB_MEM_GIB = float(CONFIG["mem_cap_gib"])
CIRCUITS_JSON = HERE.parent / "certification" / "circuit_data" / "inputs" / "circuits.json"
MAX_RESIDUAL_WEIGHT = int(CONFIG["max_residual_weight"])


def candidates(entries: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The circuits of the experiment's families and design t, by id.

    They are taken from ``entries``, the circuits of a ``circuits.json``, by default this tree's.
    """
    if entries is None:
        entries = json.loads(CIRCUITS_JSON.read_text())["circuits"]
    return sorted(
        (
            entry
            for entry in entries
            if entry["status"] == "ok"
            and entry["family"] in CONFIG["families"]
            and entry["t"] == CONFIG["t"]
        ),
        key=lambda entry: entry["id"],
    )


def groups(entries: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The circuits by group: same code file and signed stabilizers, named by the code file."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in candidates() if entries is None else entries:
        name = pathlib.Path(entry["code_file"]).name.removesuffix("_code.json")
        members = grouped.setdefault(name, [])
        if members and members[0]["stabilizers"] != entry["stabilizers"]:
            raise ValueError(f"{entry['id']} prepares another state than {members[0]['id']}")
        members.append(entry)
    return dict(sorted(grouped.items()))


def prepare(label_dir: pathlib.Path) -> None:
    """Check certification's inputs and code files, and copy the frozen files into the label.

    See ``circuits.check_inputs``, which checks every input's SHA-256 and every code file against
    its ``code_sha256``.  The copies of ``config.json`` and ``circuits.json`` in the label directory
    travel in the label's archive, where ``analyze.py`` checks them against the working tree and
    the certification label.
    """
    if not CIRCUITS_JSON.exists():
        raise SystemExit(
            f"{CIRCUITS_JSON} is missing: link certification's circuit_data from the data "
            "repository, or build a bundle that also names certification"
        )
    from experiments.state_prep_certification.certification import circuits

    circuits.check_inputs()
    groups()
    shutil.copyfile(CIRCUITS_JSON, label_dir / CIRCUITS_JSON.name)
    shutil.copyfile(HERE / "config.json", label_dir / "config.json")


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """The tasks of a subset: ``main``, or ``smoke``, the group and rate of ``smoke``.

    ``smoke`` has the ``histogram``, ``direct``, and ``calib`` tasks of its group and rate, the
    ``exact`` tasks, and the ``repro`` tasks.
    """
    grouped = groups()
    rates = CONFIG["rates"]
    if subset == "smoke":
        grouped = {CONFIG["smoke"]["group"]: grouped[CONFIG["smoke"]["group"]]}
        rates = [CONFIG["smoke"]["rate"]]
    elif subset != "main":
        raise ValueError(f"Unknown subset {subset!r}; choose main or smoke")
    task_list = []
    for group, members in grouped.items():
        for rate in rates:
            task_list.append(_task("calib", f"{group}.p{rate:g}", {"group": group, "rate": rate}))
            for entry in members:
                for arm in ("histogram", "direct"):
                    inputs = {"id": entry["id"], "rate": rate}
                    task_list.append(_task(arm, f"{entry['id']}.p{rate:g}", inputs))
    for entry in groups()[CONFIG["exact"]["group"]]:
        task_list.append(_task("exact", entry["id"], {"id": entry["id"]}))
    if subset == "main":
        check = CONFIG["mc_check"]
        for circuit_id in check["inputs"]:
            name = f"{circuit_id}.p{check['rate']:g}"
            task_list.append(_task("mc_check", name, {"id": circuit_id, "rate": check["rate"]}))
    repro = CONFIG["repro"]
    for kind in ("histogram", "direct"):
        for copy in ("a", "b"):
            name = f"{kind}.{repro['input']}.p{repro['rate']:g}"
            inputs = {"kind": kind, "id": repro["input"], "rate": repro["rate"], "copy": copy}
            task_list.append(_task("repro", f"{name}.{copy}", inputs, seed_name=f"repro.{name}"))
    return task_list


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst case of a task: start-up cap, limit, grace, and the allowance to finish."""
    return (
        float(task["startup_cap_s"])
        + float(task["limit_s"])
        + float(task["grace_s"])
        + runner.FINISH_S
    )


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """No task is repeated."""
    return []


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """Run the acceptance tests of ``acceptance.py``: on the node, those that precede the run."""
    from experiments.state_prep_certification.ranking import acceptance as tests

    return tests.run_all(node=node, results_dir=results_dir)


def _task(
    arm: str, name: str, inputs: dict[str, Any], seed_name: str | None = None
) -> dict[str, Any]:
    """A task of an arm; its seed is that of its key, or of ``seed_name``."""
    key = f"{arm}.{name}"
    digest = hashlib.sha256((seed_name or key).encode()).digest()
    return {
        "key": key,
        "experiment": NAME,
        "arm": arm,
        "mode": arm,
        "repeat": 0,
        "limit_s": float(LIMITS[f"{arm}_s"]),
        "mem_cap_gib": JOB_MEM_GIB,
        "startup_cap_s": float(LIMITS["startup_cap_s"]),
        "grace_s": float(LIMITS["grace_s"]),
        "seed": int.from_bytes(digest[:8], "big"),
        "input": inputs,
    }


####################################################################################################
# scores


def unc_profile(calibration: ConsumerCalibration) -> ConsumerProfile:
    """The profile of Phi_unc: the consumer's exact fault-weight law, with t = 1."""
    probabilities = np.asarray(calibration.mechanism_probabilities)
    law, _ = residual_weights._get_fault_weight_probabilities(probabilities, CONFIG["t"])
    return ConsumerProfile.from_downstream_fault_weights(law, CONFIG["t"])


def scores(
    total: ResidualWeightHistogram,
    x_type: ResidualWeightHistogram,
    calibration: ConsumerCalibration,
) -> dict[str, tuple[float, float]]:
    """The scores and baselines of a candidate, each with its first-order standard error.

    ``phi_unc`` uses the total-weight histogram, and ``phi_unc_x`` the X-type histogram, which is
    reported beside it.  The baselines are those of ``baselines``.
    """
    profile = unc_profile(calibration)
    return {
        "phi_unc": total.get_excess_failure_probability(profile),
        "phi_unc_x": x_type.get_excess_failure_probability(profile),
        "phi_cal": calibration.get_excess_failure_probability(x_type),
        **baselines(total),
    }


def baselines(total: ResidualWeightHistogram) -> dict[str, tuple[float, float]]:
    """The acceptance ``alpha`` and ``residual``, Pr[R > 0 | A], of a total-weight histogram."""
    return {
        "alpha": (total.acceptance_probability, acceptance_error(total)),
        "residual": total.get_excess_failure_probability(ConsumerProfile.hard(0)),
    }


def acceptance_error(histogram: ResidualWeightHistogram) -> float:
    """The first-order standard error of a histogram's acceptance probability."""
    totals = histogram.weight_sums.sum(axis=1)
    sampled = (totals > 0) & ~histogram.exact
    fractions = histogram.weight_sums[sampled, 1:].sum(axis=1) / totals[sampled]
    # a sample deviates from its row's acceptance estimate by 1_A - fraction
    squares = histogram.squared_weight_sums[sampled]
    variances = (
        squares[:, 0] * fractions**2 + squares[:, 1:].sum(axis=1) * (1 - fractions) ** 2
    ) / totals[sampled] ** 2
    return float(np.sqrt(histogram.fault_weight_probabilities[sampled] ** 2 @ variances))


def histogram_fields(histogram: ResidualWeightHistogram) -> dict[str, Any]:
    """A histogram's fields as JSON values."""
    fields = {}
    for field in dataclasses.fields(histogram):
        value = getattr(histogram, field.name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif field.name == "pauli":
            value = None if value is None else str(value)
        fields[field.name] = value
    return fields


def histogram_from_fields(fields: dict[str, Any]) -> ResidualWeightHistogram:
    """The histogram of ``histogram_fields``."""
    return ResidualWeightHistogram(
        max_residual_weight=fields["max_residual_weight"],
        css_weight=fields["css_weight"],
        fault_weight_probabilities=np.array(fields["fault_weight_probabilities"], dtype=float),
        truncated_probability=fields["truncated_probability"],
        weight_sums=np.array(fields["weight_sums"], dtype=float),
        squared_weight_sums=np.array(fields["squared_weight_sums"], dtype=float),
        num_samples=np.array(fields["num_samples"], dtype=int),
        exact=np.array(fields["exact"], dtype=bool),
        pauli=fields["pauli"],
    )


####################################################################################################
# running a task (child process)


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Run one task of the ranking experiment."""
    inputs, seed = task["input"], int(task["seed"])
    arm = task["arm"]
    if arm == "repro":
        samples = int(CONFIG["repro"]["samples"])
        entry = entry_of(inputs["id"])
        if inputs["kind"] == "histogram":
            return runner.timed(histograms, entry, inputs["rate"], samples, seed)
        return runner.timed(reference, entry, inputs["rate"], samples, seed)
    if arm == "histogram":
        samples = int(CONFIG["histogram_samples"])
        return runner.timed(histograms, entry_of(inputs["id"]), inputs["rate"], samples, seed)
    if arm == "direct":
        samples = int(CONFIG["direct"]["samples"])
        return runner.timed(reference, entry_of(inputs["id"]), inputs["rate"], samples, seed)
    if arm == "calib":
        return runner.timed(calibrate, inputs["group"], inputs["rate"], seed)
    if arm == "exact":
        return runner.timed(exact_leading_order, entry_of(inputs["id"]))
    if arm == "mc_check":
        return runner.timed(monte_carlo, entry_of(inputs["id"]), inputs["rate"], seed)
    raise ValueError(f"Unknown arm {arm!r}")


def histograms(entry: dict[str, Any], rate: float, num_samples: int, seed: int) -> dict[str, Any]:
    """The total-weight and X-type histograms of a circuit at a rate, with their CPU seconds.

    Setup is the mechanism list, and the exactly listed strata cost a run without samples, less
    setup; the sampled strata cost a run with samples, less the run without them.  Each histogram
    is also sent as a partial result.
    """
    noisy = noisy_input(entry, rate)
    options = histogram_options(entry)
    fields: dict[str, Any] = {"id": entry["id"], "rate": rate, "num_samples": num_samples}
    paulis: tuple[PauliXZ | None, ...] = (None, Pauli.X)
    for name, pauli, child in zip(("total", "x"), paulis, _child_seeds(seed, 2)):
        start = time.process_time()
        full, qubits, selected = residual_weights._get_selection(
            noisy, options["output_qubits"], options["add_flags"], options["post_select"]
        )
        residual_weights._get_mechanisms(
            full,
            qubits,
            selected,
            css_weight=options["css_weight"],
            pauli=pauli,
            max_residual_weight=MAX_RESIDUAL_WEIGHT,
            method="brouwer_zimmermann",
        )
        setup = time.process_time()
        estimate_residual_weight_histogram(
            noisy, MAX_RESIDUAL_WEIGHT, 0, pauli=pauli, method="brouwer_zimmermann", **options
        )
        exact = time.process_time()
        histogram = estimate_residual_weight_histogram(
            noisy,
            MAX_RESIDUAL_WEIGHT,
            num_samples,
            pauli=pauli,
            seed=child,
            method="brouwer_zimmermann",
            **options,
        )
        end = time.process_time()
        fields[name] = histogram_fields(histogram) | {
            "cpu_setup_s": setup - start,
            "cpu_exact_s": max(exact - setup - (setup - start), 0.0),
            # the full run repeats the setup and the exact listing before it samples
            "cpu_sampled_s": max((end - exact) - (exact - setup), 0.0),
        }
        runner.send_partial({name: fields[name]})
    return fields


def reference(entry: dict[str, Any], rate: float, num_samples: int, seed: int) -> dict[str, Any]:
    """The stratified reference of a circuit at a rate, with its CPU seconds.

    Setup is split into the consumer's (its detector error model, decoder, and mechanism list) and
    the preparation's (its mechanism list).  Each finished stratum is sent as a partial result.
    """
    settings = CONFIG["direct"]
    rng = np.random.default_rng(seed)
    start = time.process_time()
    consumer = ranking_consumer.build_consumer(entry, rate, CONFIG["decoder"])
    simulation = consumers._ConsumerSimulation(consumer, rng)
    consumer_setup = time.process_time()
    preparation = direct.get_preparation(
        noisy_input(entry, rate), histogram_options(entry), MAX_RESIDUAL_WEIGHT
    )
    prep_setup = time.process_time()
    result = direct.stratified_reference(
        direct.Pairing(preparation, simulation),
        MAX_RESIDUAL_WEIGHT,
        consumers._FAULT_WEIGHT_TOLERANCE,
        int(settings["max_exact_faults"]),
        num_samples,
        float(settings["flagged_odds_factor"]),
        rng,
        progress=runner.send_partial,
    )
    return result | {
        "id": entry["id"],
        "rate": rate,
        "num_samples": num_samples,
        "cpu_consumer_setup_s": consumer_setup - start,
        "cpu_prep_setup_s": prep_setup - consumer_setup,
    }


def calibrate(group: str, rate: float, seed: int) -> dict[str, Any]:
    """The X-type calibration of a group's consumer at a rate, and the time of the scores.

    The consumer's setup is timed as building the consumer and its simulation once before the
    calibration, which builds the simulation again; the calibration's sampled cells cost its time
    less that simulation.  The fields are sent as a partial result before the fault distance and
    the scores are timed.
    """
    members = groups()[group]
    first = members[0]
    calibration_seed, timing_seed = _child_seeds(seed, 2)
    block_preparation, noiseless, _ = ranking_consumer.consumer_circuits(first, rate)
    start = time.process_time()
    consumer = ranking_consumer.build_consumer(first, rate, CONFIG["decoder"])
    built = time.process_time()
    consumers._ConsumerSimulation(consumer, np.random.default_rng(calibration_seed))
    setup = time.process_time()
    calibration = consumer.calibrate(
        MAX_RESIDUAL_WEIGHT,
        int(CONFIG["calibration_samples"]),
        paulis=("X",),
        seed=calibration_seed,
    )
    end = time.process_time()
    fields: dict[str, Any] = {
        "group": group,
        "rate": rate,
        "ids": [entry["id"] for entry in members],
        "num_samples": int(CONFIG["calibration_samples"]),
        "calibration": calibration.to_dict(),
        "max_fault_weight": len(np.asarray(calibration.baseline_failure_probabilities)) - 1,
        "num_consumer_mechanisms": len(np.asarray(calibration.mechanism_probabilities)),
        "cpu_consumer_setup_s": setup - start,
        "cpu_sampled_s": max((end - setup) - (setup - built), 0.0),
        "consumer_circuit": str(noiseless),
        "consumer_circuit_sha256": ranking_consumer.sha256_text(str(noiseless)),
        "block_preparation": str(block_preparation),
        "block_preparation_sha256": ranking_consumer.sha256_text(str(block_preparation)),
    }
    runner.send_partial(fields)
    fields["fault_distance"] = int(
        qldpc_circuits.get_fault_distance(
            consumer.block_preparation + consumer.circuit, method="brouwer_zimmermann"
        )
    )
    fields["score_timing"] = time_scores(first, rate, calibration, timing_seed)
    return fields


def time_scores(
    entry: dict[str, Any], rate: float, calibration: ConsumerCalibration, seed: int
) -> dict[str, Any]:
    """The CPU seconds of rebuilding the stored objects and evaluating each score of a circuit.

    The histograms have ``score_timing.histogram_samples`` samples.  Each of at most
    ``score_timing.evaluations`` rounds, within ``score_timing.max_s``, rebuilds the calibration and
    the histograms by a round trip through their stored fields (``to_dict`` and ``from_dict``, and
    ``histogram_fields`` and ``histogram_from_fields``), and evaluates each score once.
    """
    settings = CONFIG["score_timing"]
    noisy = noisy_input(entry, rate)
    options = histogram_options(entry)
    total_seed, x_seed = _child_seeds(seed, 2)
    histograms = {
        "total": estimate_residual_weight_histogram(
            noisy,
            MAX_RESIDUAL_WEIGHT,
            settings["histogram_samples"],
            seed=total_seed,
            method="brouwer_zimmermann",
            **options,
        ),
        "x": estimate_residual_weight_histogram(
            noisy,
            MAX_RESIDUAL_WEIGHT,
            settings["histogram_samples"],
            pauli="X",
            seed=x_seed,
            method="brouwer_zimmermann",
            **options,
        ),
    }
    seconds = {"rebuild_calibration": 0.0, "rebuild_histogram": 0.0}
    evaluations = 0
    deadline = time.monotonic() + float(settings["max_s"])
    while evaluations < int(settings["evaluations"]) and time.monotonic() < deadline:
        start = time.process_time()
        rebuilt = ConsumerCalibration.from_dict(calibration.to_dict())
        middle = time.process_time()
        total = histogram_from_fields(histogram_fields(histograms["total"]))
        x_type = histogram_from_fields(histogram_fields(histograms["x"]))
        end = time.process_time()
        seconds["rebuild_calibration"] += middle - start
        seconds["rebuild_histogram"] += (end - middle) / 2
        for name in ("phi_unc", "phi_cal", "alpha", "residual"):
            start = time.process_time()
            _evaluate(name, total, x_type, rebuilt)
            seconds[name] = seconds.get(name, 0.0) + time.process_time() - start
        evaluations += 1
    return {
        "id": entry["id"],
        "histogram_samples": settings["histogram_samples"],
        "evaluations": evaluations,
        "mean_cpu_s": {name: value / evaluations for name, value in seconds.items()},
    }


def exact_leading_order(entry: dict[str, Any]) -> dict[str, Any]:
    """The exact leading-order coefficients of Phi_true and Phi_cal of a circuit.

    The decoder is the consumer's at ``exact.frozen_rate``, and the coefficients of p in the
    probabilities of the mechanisms come from both mechanism lists at ``exact.probe_rate``.
    """
    settings = CONFIG["exact"]
    frozen, probe = float(settings["frozen_rate"]), float(settings["probe_rate"])
    options = histogram_options(entry)
    preparation = direct.get_preparation(noisy_input(entry, frozen), options, MAX_RESIDUAL_WEIGHT)
    probe_preparation = direct.get_preparation(
        noisy_input(entry, probe), options, MAX_RESIDUAL_WEIGHT
    )
    if not (
        np.array_equal(preparation.flag_words, probe_preparation.flag_words)
        and np.array_equal(preparation.errors, probe_preparation.errors)
    ):
        raise ValueError("The preparation's mechanisms differ between the two rates")
    rng = np.random.default_rng(0)
    simulation = consumers._ConsumerSimulation(
        ranking_consumer.build_consumer(entry, frozen, CONFIG["decoder"]), rng
    )
    probe_simulation = consumers._ConsumerSimulation(
        ranking_consumer.build_consumer(entry, probe, CONFIG["decoder"]), rng
    )
    if (simulation._signatures != probe_simulation._signatures).nnz:
        raise ValueError("The consumer's mechanisms differ between the two rates")
    prep_coefficients = direct.first_order_coefficients(
        preparation.probabilities, probe_preparation.probabilities, probe
    )
    consumer_coefficients = direct.first_order_coefficients(
        simulation.probabilities, probe_simulation.probabilities, probe
    )
    result = direct.leading_order(
        direct.Pairing(preparation, simulation),
        prep_coefficients,
        consumer_coefficients,
        MAX_RESIDUAL_WEIGHT,
    )
    return {"id": entry["id"], "frozen_rate": frozen, **result}


def monte_carlo(entry: dict[str, Any], rate: float, seed: int) -> dict[str, Any]:
    """Phi_true of a circuit at a rate by qLDPC's plain paired Monte Carlo."""
    options = histogram_options(entry)
    shots = int(CONFIG["mc_check"]["shots"])
    consumer = ranking_consumer.build_consumer(entry, rate, CONFIG["decoder"])
    phi, sigma = consumer.estimate_excess_failure_probability(
        noisy_input(entry, rate),
        shots,
        output_qubits=options["output_qubits"],
        add_flags=options["add_flags"],
        post_select=options["post_select"],
        seed=seed,
    )
    return {"id": entry["id"], "rate": rate, "shots": shots, "phi": phi, "sigma": sigma}


def entry_of(circuit_id: str) -> dict[str, Any]:
    """The record of a circuit in ``circuits.json``."""
    for entry in json.loads(CIRCUITS_JSON.read_text())["circuits"]:
        if entry["id"] == circuit_id:
            return entry
    raise KeyError(circuit_id)


def noisy_input(entry: dict[str, Any], rate: float) -> stim.Circuit:
    """A circuit's frozen noiseless input, with the noise model of its options at a rate."""
    from experiments.state_prep_certification.certification import experiment as certification

    circuit = stim.Circuit((CIRCUITS_JSON.parent / f"{entry['id']}.stim").read_text())
    return certification.noisy_circuit(circuit, entry["options"]["noise"] | {"p": rate})


def histogram_options(entry: dict[str, Any]) -> dict[str, Any]:
    """The preparation options of a circuit, as ``estimate_residual_weight_histogram`` takes them.

    The output qubits are those of ``circuits.json``, in their order, which the consumer's block
    qubits follow.
    """
    options = entry["options"]
    return {
        "output_qubits": entry["output_qubits"],
        "add_flags": options["add_flags"],
        "post_select": options["post_select"],
        "css_weight": options["css_weight"],
    }


def _evaluate(
    name: str,
    total: ResidualWeightHistogram,
    x_type: ResidualWeightHistogram,
    calibration: ConsumerCalibration,
) -> object:
    """Evaluate one score, as ``scores`` does, without its standard error where it has none."""
    if name == "phi_unc":
        return total.get_excess_failure_probability(unc_profile(calibration))
    if name == "phi_cal":
        return calibration.get_excess_failure_probability(x_type)
    if name == "alpha":
        return total.acceptance_probability
    return total.get_excess_failure_probability(ConsumerProfile.hard(0))


def _child_seeds(seed: int, count: int) -> list[int]:
    """Independent seeds derived from a task's seed."""
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(count)
    ]
