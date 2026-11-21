# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests of the ranking experiment.

    python acceptance.py [--local] [--only RT1 RT3 ...] [--results-dir R]

On the node, before the run, only RT1, RT2, and RT8 run; with ``--local``, every test runs.  Each
test is stopped after ``acceptance_limit_s`` of ``config.json``.  RT6 and RT7 are evaluated by
``analyze.py`` on the node's ``mc_check`` and ``repro`` tasks.

* **RT1:** certification's inputs and code files pass ``circuits.check_inputs``, and every circuit
  of a group prepares the group's signed state on its output qubits, in order.
* **RT2:** at each rate, Stim builds the detector error model of each group's consumer, and every
  single-qubit X or Z error on the block, whose qubits are the output qubits, flips a detector.
  ``build_consumer`` refuses a group whose signed state differs from the block's in one sign.
  Whether the decoder corrects every single-qubit X and Z error on the block, with an ideal
  consumer, is recorded, with the number of the consumer's mechanisms and its fault distance, in
  ``RT2/consumers.json``; it is not a failure.
* **RT3:** on ``cgoto`` and ``aft.carbon.opt_opt`` at p = 10^-3, the reference's mechanism list
  gives, on each exactly listed fault weight, the acceptance and the probabilities of each residual
  weight of qLDPC's histogram of each Pauli type, to 10^-12; the consumer's fault-weight law and its
  largest fault weight equal those of the calibration.  On ``cgoto``, sampled strata of weights
  that are also listed exactly agree with the listing within four standard errors, with weights
  of 1; and at qLDPC's flagged-odds factor, each importance weight and the bound on a stratum's
  weights equal those of a brute-force listing on six of its mechanisms.
* **RT4:** on ``cgoto`` at p = 10^-3, with every fault weight up to 4 listed and a calibration with
  exact cells of both Pauli types, Phi_unc and Phi_cal from qLDPC equal a computation from the
  listed faults, to 10^-12, with Phi_unc's profile worked out by hand.  The score values of the cost
  projection give qLDPC's Phi_unc with its error; for Phi_cal, the means of those values in the
  histograms of the two Pauli types sum to qLDPC's Phi_cal, and their errors add in quadrature to
  its sigma_hist.  The standard error of the acceptance of a sampled fault weight with equal
  importance weights is the binomial one.
* **RT5:** the ``exact`` task's coefficient on the four steane circuits equals that of a plain
  enumeration over the same mechanism lists and decoder, exactly; and in every stratum with
  f + g <= 2, at each rate, the reference's numbers of paired changes +1 and -1 equal those of the
  plain enumeration.
* **RT8:** no ``mqt`` or ``qiskit`` module is loaded.
* **RT9:** on a synthetic certification archive, named and labelled as a node's, each verdict and
  a disagreement admit or drop their circuits as the rule says, a circuit without a row is not
  admitted, and a group with one admitted circuit is out of scope; a row with ``compiled_in_timing``
  set changes no admission.  An archive that breaks any one provenance check fails it.  A ``direct``
  task that sleeps after each stratum (``stratum_delay_s``), stopped at its limit during the first
  sleep, is a timeout that keeps the laws of both fault weights and its one finished stratum.  A
  ``histogram``, a ``direct``, and a ``calib`` task finish through the task runner without
  ``compiled_in_timing``, and the histograms and the calibration cover both Pauli types.
* **RT10:** the reference's ratio, error, allowance, and inadequate strata from hand-made strata;
  resolved and unresolved pairs, and correct, wrong, tied, uncertain, and missing calls, with
  Phi_cal's uncertainty from qLDPC's error of the difference; a circuit's unsampled cells; the
  gate's outcomes; a pair on each side of the margin; a group's cost lines and k*, and each validity
  input of the cost projection on each side of its threshold, including a failing cell that only
  the circuit with the smaller sigma_cal uses and an exempted stratum; the break-even outcomes at
  their boundaries; projected costs; the exemption of parts with few draws, at its boundary and
  with the floor s_min sigma_target^2; each part's estimate, against qLDPC's or the reference's
  variance, and bound, with a calibration of both Pauli types; the exact baseline f(0, 0); the
  nonzero draws of a calibration cell through qLDPC's Jeffreys error; and a row of
  ``leading_order.csv``.  Expected values are worked out by hand, unless said otherwise.  A
  candidate keeps its histogram's alpha as its score, and its sigma_hist and sigma_cal are qLDPC's;
  a histogram task that stopped after its total-weight histogram still gives Phi_unc.  A
  histogram's, the direct reference's, or the calibration's row with ``compiled_in_timing`` set
  gives no CPU seconds but keeps the scores and the reference.  Such a row of an admitted circuit,
  in a group of two, leaves the group without costs, and ``checks.json`` lists it.

Each test prints PASS or FAIL; the script exits with status 1 if any failed.
"""

from __future__ import annotations

import dataclasses
import fractions
import functools
import io
import itertools
import json
import math
import pathlib
import subprocess
import sys
import tarfile
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import circuits as certification_circuits
from experiments.state_prep_certification.common import acceptance, local_runs, manifest, store
from experiments.state_prep_certification.ranking import analyze, direct, experiment
from experiments.state_prep_certification.ranking import consumer as ranking_consumer
from qldpc import circuits
from qldpc.experimental.circuits.benchmarking import (
    ConsumerCalibration,
    ResidualWeightHistogram,
    consumers,
    estimate_residual_weight_histogram,
    residual_weights,
)
from qldpc.objects import Pauli, PauliXZ

M = experiment.MAX_RESIDUAL_WEIGHT
SUITE = acceptance.Suite(
    "spc-ranking-",
    node_tests=("RT1", "RT2", "RT8"),
    limit_s=float(experiment.CONFIG["acceptance_limit_s"]),
)
run_all = SUITE.run_all


def test(function: Callable[[pathlib.Path], None]) -> Callable[[pathlib.Path], None]:
    """Register a test, which takes only its directory, by the ID at the start of its name."""

    @functools.wraps(function)
    def run(directory: pathlib.Path, node: bool) -> None:
        function(directory)

    SUITE.test(run)
    return function


####################################################################################################
# tests


@test
def rt1_inputs(directory: pathlib.Path) -> None:
    """Inputs, code files, and the state of each group."""
    certification_circuits.check_inputs()
    for group, members in experiment.groups().items():
        for entry in members:
            noiseless = experiment.noisy_input(entry, 0.001).without_noise()
            prepared = circuits.get_state_stabilizers(noiseless, entry["output_qubits"])
            assert len(prepared) == len(entry["output_qubits"]), entry["id"]
            assert certification_circuits.same_state(
                [str(op) for op in prepared], members[0]["stabilizers"]
            ), (group, entry["id"])
        print(f"  {group}: {[entry['id'] for entry in members]}")


@test
def rt2_consumers(directory: pathlib.Path) -> None:
    """Each group's consumer at each rate: its DEM, and single X and Z errors on the block."""
    records = []
    for group, members in experiment.groups().items():
        entry = members[0]
        num_qubits = len(entry["output_qubits"])
        for rate in experiment.CONFIG["rates"]:
            consumer = ranking_consumer.build_consumer(entry, rate)
            simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))

            # by Stim, with a certain error between the noiseless block preparation and consumer:
            # every single-qubit X or Z error on the block reaches a detector
            assert len(consumer.block_qubits or ()) == num_qubits, group
            for qubit in consumer.block_qubits or ():
                for pauli in ("X", "Z"):
                    circuit = consumer.block_preparation.copy()
                    circuit.append(f"{pauli}_ERROR", [qubit], 1.0)
                    circuit += consumer.circuit.without_noise()
                    detectors = circuit.compile_detector_sampler().sample(1)
                    assert detectors.any(), (group, pauli, qubit)

            # whether the decoder corrects each single-qubit X and Z error, with no consumer fault
            uncorrected = {}
            flips = simulation.get_flips(np.zeros((num_qubits, 0), dtype=np.intp))
            for offset, name in ((0, "x"), (num_qubits, "z")):
                errors = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
                errors[range(num_qubits), offset + np.arange(num_qubits)] = 1
                failures = simulation.get_failures(*flips, errors)
                uncorrected[f"uncorrected_single_{name}_errors"] = np.flatnonzero(failures).tolist()
            records.append(
                {
                    "group": group,
                    "rate": rate,
                    "num_consumer_mechanisms": len(simulation.probabilities),
                    "fault_distance": int(
                        circuits.get_fault_distance(
                            consumer.block_preparation + consumer.circuit,
                            method="brouwer_zimmermann",
                        )
                    ),
                    **uncorrected,
                }
            )
            print(f"  {records[-1]}")
    (directory / "consumers.json").write_text(json.dumps(records, indent=1) + "\n")

    # a group whose signed state differs from the consumer's block, in one sign, is refused
    entry = experiment.groups()["steane"][0]
    stabilizers = list(entry["stabilizers"])
    stabilizers[0] = ("-" if stabilizers[0][0] == "+" else "+") + stabilizers[0][1:]
    try:
        ranking_consumer.build_consumer(entry | {"stabilizers": stabilizers}, 0.001)
    except ValueError:
        pass
    else:
        raise AssertionError("build_consumer accepted a block in another state")


@test
def rt3_mechanisms(directory: pathlib.Path) -> None:
    """The reference's mechanism lists against qLDPC's histograms and calibration."""
    rate = 0.001
    for circuit_id in ("cgoto", "aft.carbon.opt_opt"):
        entry = experiment.entry_of(circuit_id)
        noisy = experiment.noisy_input(entry, rate)
        options = experiment.histogram_options(entry)
        preparation = direct.get_preparation(noisy, options, M)
        pi, u = direct.fault_weight_law(
            preparation.probabilities, M + 1, residual_weights._TRUNCATION_TOLERANCE
        )
        for pauli in experiment.PAULI_TYPES.values():
            histogram = estimate_residual_weight_histogram(noisy, M, 0, pauli=pauli, **options)
            assert len(pi) == len(histogram.fault_weight_probabilities), circuit_id
            assert np.allclose(pi, histogram.fault_weight_probabilities, rtol=0, atol=1e-15)
            assert abs(u - histogram.truncated_probability) <= 1e-15, circuit_id
            for f in np.flatnonzero(histogram.exact)[1:]:
                faults = residual_weights._get_combinations(len(preparation), int(f))
                probabilities = direct._conditional_probabilities(preparation.probabilities, faults)
                accepted = preparation.accepted(faults)
                errors = preparation.fault_errors(faults)
                weights = preparation.residual_weights(errors, pauli, M)
                outcomes = np.where(accepted, 1 + weights, 0)
                expected = np.bincount(outcomes, weights=probabilities, minlength=M + 3)
                assert np.allclose(expected, histogram.weight_sums[f], rtol=0, atol=1e-12), (
                    circuit_id,
                    pauli,
                    f,
                )
            exact_weights = np.flatnonzero(histogram.exact)[-1]
            print(f"  {circuit_id}, {pauli}: fault weights 1-{exact_weights} agree")

        consumer = ranking_consumer.build_consumer(entry, rate)
        simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))
        calibration = consumer.calibrate(M, 10, paulis=("X",), seed=0)
        eta, _ = direct.fault_weight_law(
            simulation.probabilities, M, consumers._FAULT_WEIGHT_TOLERANCE
        )
        law = calibration.fault_weight_probabilities
        assert len(eta) == len(law) and np.allclose(eta, law, rtol=0, atol=1e-12), circuit_id

    # sampled strata against their exact listings
    entry = experiment.entry_of("cgoto")
    consumer = ranking_consumer.build_consumer(entry, rate)
    pairing = direct.build_pairing(
        experiment.noisy_input(entry, rate),
        experiment.histogram_options(entry),
        consumer,
        M,
        np.random.default_rng(0),
    )
    for f, g in ((2, 0), (2, 1), (3, 0)):
        exact = direct._list_stratum(pairing, f, g)
        assert exact.plus + exact.minus > 0, (f, g)
        sampled = direct._sample_stratum(
            # factor 1, config.json's value
            pairing,
            f,
            g,
            20_000,
            1.0,
            np.random.default_rng(f + g),
        )
        assert abs(sampled.mean - exact.mean) <= 4 * math.sqrt(sampled.variance), (f, g)
        if g == 0:
            assert exact.acceptance is not None and sampled.acceptance is not None
            gap = abs(sampled.acceptance - exact.acceptance)
            assert gap <= 4 * math.sqrt(sampled.acceptance_variance), (f, g)
        print(f"  cgoto ({f}, {g}): sampled {sampled.mean:.4f}, exact {exact.mean:.4f}")
        assert sampled.importance_weight_bound == 1.0 == sampled.max_importance_weight, sampled

    # at qLDPC's factor, each weight is P_f(F) / q_f(F), and the weight bound is its largest value
    # over every fault F of weight f, by brute force on six mechanisms, three of them flagged
    preparation = pairing.preparation
    flagged = np.flatnonzero(preparation.flagged)[:3]
    unflagged = np.flatnonzero(~preparation.flagged)[:3]
    chosen = np.concatenate([flagged, unflagged])
    small = direct.Preparation(
        preparation.probabilities[chosen],
        preparation.flag_words[chosen],
        preparation.errors[chosen],
        preparation.stabilizers,
        preparation.css_weight,
    )
    odds = small.probabilities / (1 - small.probabilities)
    qldpc_factor = residual_weights._FLAGGED_ODDS_FACTOR
    proposal = odds * np.where(small.flagged, qldpc_factor, 1.0)
    # f = 2 tests a bound reached with f flagged mechanisms, and f = 4 one capped at the three
    # flagged
    for f in (2, 4):
        subsets = list(itertools.combinations(range(6), f))
        noise = np.array([np.prod(odds[list(subset)]) for subset in subsets])
        tilted = np.array([np.prod(proposal[list(subset)]) for subset in subsets])
        ratios = dict(zip(subsets, (noise / noise.sum()) / (tilted / tilted.sum())))
        faults, importance, bound = direct.sample_faults(
            small, f, 20, qldpc_factor, np.random.default_rng(f)
        )
        assert math.isclose(bound, max(ratios.values()), rel_tol=1e-9), (f, bound)
        for fault, weight in zip(faults, importance):
            assert math.isclose(weight, ratios[tuple(sorted(fault))], rel_tol=1e-9), (f, fault)


@test
def rt4_scores(directory: pathlib.Path) -> None:
    """Phi_unc and Phi_cal against a computation from listed faults, and the error formulas."""
    rate = 0.001
    entry = experiment.entry_of("cgoto")
    noisy = experiment.noisy_input(entry, rate)
    options = experiment.histogram_options(entry)
    max_fault_weight = 4
    total = estimate_residual_weight_histogram(
        noisy, M, 0, max_fault_weight=max_fault_weight, **options
    )
    typed = {
        name: estimate_residual_weight_histogram(
            noisy, M, 0, max_fault_weight=max_fault_weight, pauli=pauli, **options
        )
        for name, pauli in experiment.PAULI_TYPES.items()
    }
    assert total.exact.all() and all(histogram.exact.all() for histogram in typed.values())
    consumer = ranking_consumer.build_consumer(entry, rate)
    simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))
    tables: dict[Any, npt.NDArray[np.float64]] = {
        Pauli.X: np.array([[0.1, 0.3, 0.5, 0.6], [0.7, 0.6, 0.4, 0.2]]),
        Pauli.Z: np.array([[0.2, 0.1, 0.3, 0.4], [0.5, 0.8, 0.6, 0.3]]),
    }
    baseline = np.array([0.0, 0.05, 0.2, 0.3])
    calibration = ConsumerCalibration(
        tables, baseline, simulation.probabilities, description="exact cells"
    )
    distribution = calibration.fault_weight_probabilities
    no_block_error = float(baseline @ distribution) + calibration.truncated_probability
    # Phi_unc's consumer fails if m plus its fault weight g exceeds t = 1
    t = int(experiment.CONFIG["t"])
    failures = [1 - float(distribution[: t - m + 1].sum()) if m <= t else 1.0 for m in range(M + 2)]
    values_unc = [failure - failures[0] for failure in failures]

    # the listed faults, each weighed by its probability, with heavier fault weights accepted with
    # overflow; Phi_cal sums the means of the two Pauli types
    preparation = direct.get_preparation(noisy, options, M)
    pi, u = residual_weights._get_fault_weight_probabilities(
        preparation.probabilities, max_fault_weight
    )

    def listed_mean(values: list[float], pauli: PauliXZ | None) -> float:
        """The mean of the values of the residual weights of type pauli over accepted faults."""
        numerator, acceptance = u * values[-1], u
        for f in range(max_fault_weight + 1):
            faults = residual_weights._get_combinations(len(preparation), f)
            weights = direct._conditional_probabilities(preparation.probabilities, faults)
            accepted = preparation.accepted(faults)
            residual = preparation.residual_weights(preparation.fault_errors(faults), pauli, M)
            acceptance += pi[f] * float(weights @ accepted)
            numerator += pi[f] * float(weights[accepted] @ np.array(values)[residual[accepted]])
        return numerator / acceptance

    computed = experiment.scores(total, typed, calibration)
    listed = {
        "phi_unc": listed_mean(values_unc, None),
        "phi_cal": math.fsum(
            listed_mean([0.0, *(table @ distribution), 1 - no_block_error], pauli)
            for pauli, table in tables.items()
        ),
    }
    for score, value in listed.items():
        assert abs(computed[score][0] - value) <= 1e-12, (score, computed[score][0], value)
        print(f"  {score}: {value:.6e} from qLDPC and from the listed faults")

    # the score values that the cost projection uses give qLDPC's scores and histogram errors
    calibrated = consumer.calibrate(M, 300, seed=1)
    histograms = [
        estimate_residual_weight_histogram(noisy, M, 2000, pauli=pauli, seed=seed, **options)
        for seed, pauli in enumerate(calibrated.excess_failure_probabilities, start=2)
    ]
    phi_cal = calibrated.get_excess_failure_probability(*histograms)[0]
    sigma_hist, _ = calibrated.get_standard_error_parts(*histograms)
    means, errors = zip(
        *(h._get_conditional_mean(analyze.phi_cal_values(calibrated, h)) for h in histograms)
    )
    assert math.isclose(sum(means), phi_cal, rel_tol=1e-12), (means, phi_cal)
    assert math.isclose(math.hypot(*errors), sigma_hist, rel_tol=1e-12), (errors, sigma_hist)
    total_sampled = estimate_residual_weight_histogram(noisy, M, 2000, seed=4, **options)
    phi_unc = total_sampled.get_excess_failure_probability(experiment.unc_profile(calibrated))
    unc_values = analyze.phi_unc_values(calibrated, total_sampled)
    assert np.allclose(total_sampled._get_conditional_mean(unc_values), phi_unc, rtol=1e-12, atol=0)

    # the standard error of the acceptance, where equal importance weights make it binomial
    accepted_draws, draws = 30, 200
    weight_sums = np.zeros((2, M + 3))
    weight_sums[0, 1] = 1.0
    weight_sums[1, 0], weight_sums[1, 2] = draws - accepted_draws, accepted_draws
    binomial = ResidualWeightHistogram(
        max_residual_weight=M,
        css_weight=True,
        fault_weight_probabilities=np.array([0.6, 0.3]),
        truncated_probability=0.1,
        weight_sums=weight_sums,
        squared_weight_sums=weight_sums * [[0], [1]],
        num_samples=np.array([0, draws]),
        exact=np.array([True, False]),
    )
    fraction = accepted_draws / draws
    expected = 0.3 * math.sqrt(fraction * (1 - fraction) / draws)
    assert abs(experiment.acceptance_error(binomial) - expected) <= 1e-15, expected


@test
def rt5_leading_order(directory: pathlib.Path) -> None:
    """The exact task and the exact strata against a plain enumeration."""
    group = experiment.groups()[experiment.CONFIG["exact"]["group"]]
    frozen = float(experiment.CONFIG["exact"]["frozen_rate"])
    for entry in group:
        result = experiment.exact_leading_order(entry)
        pairing, prep_c, consumer_c = _frozen_pairing(entry, frozen)
        plain = _plain_leading_order(pairing, prep_c, consumer_c)
        assert result["coefficient"] == plain, (entry["id"], result, plain)
        print(f"  {entry['id']}: {result['coefficient']}")

        for rate in experiment.CONFIG["rates"]:
            consumer = ranking_consumer.build_consumer(entry, rate)
            pairing = direct.build_pairing(
                experiment.noisy_input(entry, rate),
                experiment.histogram_options(entry),
                consumer,
                M,
                np.random.default_rng(0),
            )
            for f, g in ((1, 0), (2, 0), (1, 1)):
                stratum = direct._list_stratum(pairing, f, g)
                assert (stratum.plus, stratum.minus) == _plain_counts(pairing, f, g), (rate, f, g)


@test
def rt8_imports(directory: pathlib.Path) -> None:
    """The node's modules, all imported by this script, load neither MQT nor qiskit."""
    loaded = [name for name in sys.modules if name.split(".")[0] in ("mqt", "qiskit")]
    assert not loaded, loaded


@test
def rt9_analysis(directory: pathlib.Path) -> None:
    """Admission, provenance, a timeout that keeps its strata, and unflagged runs of three arms."""
    verdicts = {
        "aft.steane.heuristic_heuristic": ("certified", "certified"),
        "aft.steane.heuristic_opt": ("violation at 2", "violation at 2"),
        "aft.steane.opt_opt": ("violation at 1", "violation at 1"),
        "cgoto": ("certified", "violation at 2"),
        "aft.shor.opt_opt": ("certified", "certified"),
    }
    circuits_bytes = experiment.CIRCUITS_JSON.read_bytes()
    archive = synthetic_certification_archive(directory / "certification", verdicts)
    label_dir = analyze.check_provenance(archive, circuits_bytes)
    admission = {row["id"]: row for row in analyze.admit(label_dir, experiment.groups())}
    assert admission["aft.steane.heuristic_heuristic"]["admitted"]
    assert admission["aft.steane.heuristic_opt"]["admitted"]
    assert not admission["aft.steane.opt_opt"]["admitted"]
    assert admission["cgoto"]["verdict"] == "disagreement" and not admission["cgoto"]["admitted"]
    assert admission["aft.shor.heuristic_opt"]["verdict"] == "no row"
    assert admission["aft.steane.heuristic_opt"]["group_in_scope"]
    assert admission["aft.shor.opt_opt"]["admitted"]
    assert not admission["aft.shor.opt_opt"]["group_in_scope"]

    # the admission reads the certification rows' verdicts whatever their compiled_in_timing: here
    # every row has it set
    runs = label_dir / "runs.jsonl"
    rows = [
        json.loads(line) | {"compiled_in_timing": True} for line in runs.read_text().splitlines()
    ]
    runs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert {row["id"]: row for row in analyze.admit(label_dir, experiment.groups())} == admission

    # each provenance check fails on an archive that breaks only it
    other = json.loads(circuits_bytes)
    other["circuits"] = other["circuits"][:-1]
    broken = {
        "another circuits.json": (archive, json.dumps(other).encode()),
        "subset": (_variant(directory, "subset", verdicts, subset="smoke"), circuits_bytes),
        "no bundle": (_variant(directory, "bundle", verdicts, bundle=False), circuits_bytes),
        "dirty": (_variant(directory, "dirty", verdicts, dirty=True), circuits_bytes),
        "label": (_variant(directory, "label", verdicts, label="certification-0"), circuits_bytes),
        "outer SHA-256": (_variant(directory, "outer", verdicts, outer=True), circuits_bytes),
        "inner SHA256SUMS": (_variant(directory, "inner", verdicts, inner=True), circuits_bytes),
    }
    for case, (path, frozen) in broken.items():
        try:
            analyze.check_provenance(path, frozen)
        except analyze.ProvenanceError:
            continue
        raise AssertionError(f"an archive with a broken {case} passed the provenance check")

    # a direct task stopped at its limit keeps the laws of the fault weights and its strata: it
    # sleeps 30 s after each stratum, so that a limit of 10 s stops it during the first sleep on any
    # machine on which the setup and the first stratum take under 10 s
    task = experiment._task("direct", "cgoto.p0.001", {"id": "cgoto", "rate": 0.001})
    task |= {"limit_s": 10.0, "stratum_delay_s": 30.0}
    row = local_runs.run_locally(directory / "timeout", [task])[task["key"]]
    strata = [record["stratum"] for record in row["partial"] if "stratum" in record]
    first = row["partial"][0]
    assert row["status"] == "timeout" and "pi" in first and "eta" in first, row["status"]
    assert len(strata) == 1, len(strata)
    print(f"  a direct task stopped at its limit keeps {len(strata)} stratum")

    # a histogram, a direct, and a calib task finish through the runner without
    # compiled_in_timing, which run_locally checks; the histograms and the calibration cover both
    # Pauli types
    results = {}
    for arm, name, inputs in (
        ("histogram", "cgoto.p0.001", {"id": "cgoto", "rate": 0.001}),
        ("direct", "cgoto.p0.001", {"id": "cgoto", "rate": 0.001}),
        ("calib", "steane.p0.001", {"group": "steane", "rate": 0.001}),
    ):
        task = experiment._task(arm, name, inputs) | {"limit_s": 600.0}
        row = local_runs.run_locally(directory / task["key"], [task])[task["key"]]
        assert row["status"] == "ok", (task["key"], row["status"])
        results[arm] = row["result"]
    paulis = {name: results["histogram"][name]["pauli"] for name in ("total", "x", "z")}
    assert paulis == {"total": None, "x": "X", "z": "Z"}, paulis
    tables = results["calib"]["calibration"]["excess_failure_probabilities"]
    assert set(tables) == {"X", "Z"}, set(tables)


@test
def rt10_rules(directory: pathlib.Path) -> None:
    """The reference's ratio, error, and allowance, and the rules of pairs, the gate, and costs."""
    # the ratio estimate, its first-order error, and tau, from hand-made strata
    strata = [
        direct.Stratum(1, 0, exact=True, mean=0.2, acceptance=0.5),
        direct.Stratum(
            2,
            0,
            exact=False,
            mean=0.3,
            variance=0.01,
            acceptance=0.4,
            acceptance_variance=0.004,
            covariance=0.002,
            draws=100,
            ess=5.0,
        ),
        direct.Stratum(1, 1, exact=True, mean=0.1),
    ]
    pi, u, eta, v = np.array([0.5, 0.3, 0.15]), 0.05, np.array([0.9, 0.08]), 0.02
    result = direct.combine_strata(strata, pi, u, eta, v)
    numerator = 0.3 * 0.9 * 0.2 + 0.15 * 0.9 * 0.3 + 0.3 * 0.08 * 0.1
    alpha = 0.5 + 0.3 * 0.5 + 0.15 * 0.4 + 0.05
    phi = numerator / alpha
    variance = (
        (0.15 * 0.9) ** 2 * 0.01 - 2 * phi * (0.15 * 0.9) * 0.15 * 0.002 + phi**2 * 0.15**2 * 0.004
    ) / alpha**2
    assert math.isclose(result["phi"], phi, rel_tol=1e-12), result["phi"]
    assert math.isclose(result["sigma"], math.sqrt(variance), rel_tol=1e-12), result["sigma"]
    assert math.isclose(result["tau"], 0.05 / (0.5 + 0.3 * 0.5) + 0.02, rel_tol=1e-12)
    assert result["inadequate"] == [[2, 0]], result["inadequate"]

    # pairs: resolved, within the margin, and with an inadequate stratum; each kind of call
    rate = experiment.CONFIG["rates"][0]
    calibration = ConsumerCalibration(
        {"X": [[0.1], [0.2]]},
        [0.0],
        excess_standard_errors={"X": [[0.0], [0.0]]},
        baseline_standard_errors=[0.0],
        excess_num_samples={"X": [[10], [10]]},
    )
    weight_sums = np.zeros((1, M + 3))
    weight_sums[0, 1:4] = [0.8, 0.1, 0.1]
    histogram = ResidualWeightHistogram(
        max_residual_weight=M,
        css_weight=True,
        fault_weight_probabilities=np.array([1.0]),
        truncated_probability=0.0,
        weight_sums=weight_sums,
        squared_weight_sums=np.zeros_like(weight_sums),
        num_samples=np.array([0]),
        exact=np.array([True]),
        pauli="X",
    )

    def candidate(name: str, true: float, scores: dict[str, float], **extra: Any) -> dict[str, Any]:
        """A candidate with a reference, its scores and errors, and its calibration."""
        values = {"phi_cal": 0.0, "phi_unc": 0.0, "alpha": 0.5, "residual": 0.0} | scores
        return {
            "id": name, "group": "g", "rate": rate, "admitted": True, "phi_true": true,
            "sigma_true": 0.01, "tau": 0.001, "inadequate": [], "calibration": calibration,
            "calibrated": [dataclasses.replace(histogram)], "unsampled_cells": [],
            **values, **{f"sigma_{score}": 0.0 for score in values}, **extra,
        }  # fmt: skip

    candidates = [
        candidate("a", 1.0, {"phi_cal": 1.0, "phi_unc": 1.0, "alpha": 0.9, "residual": 2.0},
                  sigma_residual=5.0),
        candidate("b", 2.0, {"phi_cal": 2.0, "phi_unc": 2.0, "alpha": 0.8, "residual": 1.0},
                  unsampled_cells=[["X", 1, 0]]),
        candidate("c", 2.01, {"phi_cal": 3.0, "phi_unc": 3.0, "alpha": 0.9, "residual": 3.0}),
        candidate("d", 9.0, {}, inadequate=[[2, 0]]),
    ]  # fmt: skip
    del candidates[2]["residual"]
    pairs = {(p["first"], p["second"]): p for p in analyze.compare(candidates)}
    assert pairs[("b", "c")]["reason"] == "within margin"
    assert pairs[("a", "d")]["reason"] == "inadequate stratum"
    ab, ac = pairs[("a", "b")], pairs[("a", "c")]
    assert ab["resolved"] and ac["resolved"]
    assert ab["call_phi_unc"] == "correct" and ab["call_alpha"] == "correct"
    assert ac["call_alpha"] == "tie" and ab["call_residual"] == "wrong"
    assert ab["uncertain_residual"] and not ac["uncertain_phi_unc"]
    assert ab["uncertain_phi_cal"] and not ac["uncertain_phi_cal"]
    assert ac["call_residual"] == "no score"
    rows, _ = analyze.agree(list(pairs.values()), candidates)
    row = next(r for r in rows if r["score"] == "phi_cal" and r["rate"] == rate)
    assert (row["resolved_pairs"], row["correct"], row["correct_certain"]) == (2, 2, 1), row
    assert not row["eligible"]

    # Phi_cal's call uses qLDPC's error of the difference, not the two errors in quadrature: with
    # equal histograms it is 0, whatever sigma_phi_cal; with histograms whose probabilities of
    # residual weights 1 and 2 differ by 0.1, for cells with errors of 0.5, it is 0.05 sqrt(2), more
    # than half the difference of 0.1
    noisy_calibration = dataclasses.replace(
        calibration, excess_standard_errors={"X": [[0.5], [0.5]]}
    )
    shifted = np.array(weight_sums)
    shifted[0, 1:4] = [0.8, 0.0, 0.2]
    pairs_cal: dict[str, list[tuple[ResidualWeightHistogram, float, float]]] = {
        "equal": [(histogram, 1.0, 5.0), (histogram, 1.1, 5.0)],
        "different": [
            (histogram, 1.0, 0.0),
            (dataclasses.replace(histogram, weight_sums=shifted), 1.1, 0.0),
        ],
    }
    for case, cal_members in pairs_cal.items():
        near = [
            candidate(name, true, {"phi_cal": phi}, sigma_phi_cal=sigma, calibration=noisy_calibration,
                      calibrated=[dataclasses.replace(member)])
            for name, true, (member, phi, sigma) in zip("ef", (1.0, 2.0), cal_members)
        ]  # fmt: skip
        uncertain = analyze._pair("g", rate, *near)["uncertain_phi_cal"]
        assert uncertain == (case == "different"), case

    # a circuit's unsampled cells: the cells of each type that no draw reached, whose residual
    # weight has a nonzero probability in its histogram of that type; here Z cell (1, 0), which
    # z_shifted, giving residual weight 1 no probability, does not use
    sparse = ConsumerCalibration(
        {"X": [[0.1], [0.2]], "Z": [[1.0], [0.2]]},
        [0.0],
        excess_standard_errors={"X": [[0.01], [0.01]], "Z": [[0.0], [0.01]]},
        excess_num_samples={"X": [[10], [10]], "Z": [[0], [10]]},
    )
    z_type = dataclasses.replace(histogram, pauli="Z")
    assert analyze.unsampled_cells(sparse, [histogram, z_type]) == [["Z", 1, 0]]
    z_shifted = dataclasses.replace(z_type, weight_sums=shifted)
    assert analyze.unsampled_cells(sparse, [histogram, z_shifted]) == []

    # the gate, from agreement rows: kept, not kept, and inconclusive
    def gate(
        phi_cal: tuple[int, int], rival: int, eligible: int = 2, best: str | None = None
    ) -> str:
        """The gate when Phi_cal's calls at each eligible rate are (correct, certain and correct).

        Each eligible rate has 10 resolved pairs.

        Every other score makes ``rival`` correct calls, none of them certain, except ``best``,
        which makes as many correct and certain calls as Phi_cal's certain ones.  At an ineligible
        rate, Phi_cal calls none correctly and every other score all 10.
        """
        rows = []
        for score in analyze.SCORES:
            calls = phi_cal if score == "phi_cal" else (rival, 0)
            calls = (phi_cal[1], phi_cal[1]) if score == best else calls
            for index, r in enumerate(experiment.CONFIG["rates"]):
                rate_calls = (
                    calls if index < eligible else ((0, 0) if score == "phi_cal" else (10, 10))
                )
                rows.append(
                    {
                        "score": score,
                        "rate": r,
                        "eligible": index < eligible,
                        "resolved_pairs": 10,
                        "correct": rate_calls[0],
                        "correct_certain": rate_calls[1],
                    }
                )
        return str(analyze.gate(rows)["outcome"])

    assert gate((8, 8), 7) == "kept"
    assert gate((9, 7), 6) == "not kept"
    assert gate((10, 8), 9) == "not kept"
    for best in ("phi_unc", "alpha", "residual"):
        assert gate((9, 9), 8, best=best) == "not kept", best
    assert gate((9, 9), 8, eligible=1) == "inconclusive"

    # the margin of a pair, 2 sigma_Delta + tau_i + tau_j = 0.04828..., between two differences
    for difference, resolved in ((0.047, False), (0.050, True)):
        near = [
            candidate(name, true, {}, tau=0.01)
            for name, true in (("x", 1.0), ("y", 1.0 + difference))
        ]
        assert analyze._pair("g", rate, *near)["resolved"] == resolved, difference

    # the cost lines of one group, against values worked out by hand; every validity input is at
    # its threshold of 10 nonzero draws, and the reached cell with the fewest draws sets s_min
    target = 0.1 * 2e-3

    def cost_lines(
        direct_nonzero: int = 10,
        accepted: int = 10,
        row: tuple[int, ...] = (100, 1890, 6, 4, 0),
        cells: tuple[tuple[int, int, int], ...] = ((6, 4, 1000), (8, 2, 500)),
        worst_row: tuple[int, ...] | None = None,
        eta_1: float = 0.1,
    ) -> dict[str, Any]:
        """The cost lines of two hand-made circuits.

        ``direct_nonzero`` and ``accepted`` are the nonzero and accepted draws of two sampled
        strata of the reference, ``row`` the summed weights of the histograms' sampled row (whose
        draws all have weight 1), ``worst_row`` that of the circuit with the larger sigma_cal if
        it differs, ``cells`` the increases, decreases, and draws of the calibration's two cells of
        each Pauli type, and ``eta_1`` the consumer's probability of fault weight 1.
        """

        def histogram(
            sampled_row: tuple[int, ...], pauli: PauliXZ | None
        ) -> ResidualWeightHistogram:
            """A histogram with an exact row of fault weight 0 and one sampled row."""
            weights = np.zeros((2, M + 3))
            weights[0, 1] = 1.0
            weights[1] = sampled_row
            return ResidualWeightHistogram(
                max_residual_weight=M,
                css_weight=True,
                fault_weight_probabilities=np.array([0.9, 0.1]),
                truncated_probability=0.0,
                weight_sums=weights,
                squared_weight_sums=weights,
                num_samples=np.array([0, 2000]),
                exact=np.array([True, False]),
                pauli=pauli,
            )

        # both Pauli types have the same cells
        jeffreys = [[float(consumers._get_jeffreys_deviations(i, d, n))] for i, d, n in cells]
        calibration = analyze.exact_no_fault_baseline(
            ConsumerCalibration(
                {pauli: [[(i - d) / n] for i, d, n in cells] for pauli in (Pauli.X, Pauli.Z)},
                [0.0],
                excess_standard_errors={pauli: jeffreys for pauli in (Pauli.X, Pauli.Z)},
                baseline_standard_errors=[0.01],
                excess_num_samples={
                    pauli: [[n] for *_, n in cells] for pauli in (Pauli.X, Pauli.Z)
                },
            )
        )

        def histograms(sampled_row: tuple[int, ...]) -> dict[str, ResidualWeightHistogram]:
            """The total-weight histogram and one of each Pauli type, with the same rows."""
            kinds: dict[str, PauliXZ | None] = {"total": None, **experiment.PAULI_TYPES}
            return {name: histogram(sampled_row, pauli) for name, pauli in kinds.items()}

        stratum = {
            "exact": False,
            "variance": 1e-12,
            "acceptance_variance": 1e-12,
            "covariance": 0.0,
            "draws": 10_000,
            "importance_weight_bound": 1.0,
        }
        members: list[dict[str, Any]] = [
            {
                "phi_true": phi,
                "sigma_true": 2 * target,
                "tau": 0.0,
                "alpha_reference": 0.9,
                "alpha_exact": 0.8,
                "pi": [0.5, 0.3, 0.2],
                "eta": [0.9, eta_1],
                "num_direct_samples": 10_000,
                "strata": [
                    stratum | {"f": 1, "g": 1, "nonzero": direct_nonzero, "accepted_draws": 50},
                    stratum | {"f": 2, "g": 0, "nonzero": 10, "accepted_draws": accepted},
                ],
                "cpu_direct": {
                    "cpu_consumer_setup_s": 0.5,
                    "cpu_prep_setup_s": 1.0,
                    "cpu_exact_s": 2.0,
                    "cpu_sampled_s": 3.0,
                },
                "sigma_hist": 2 * target / math.sqrt(2),
                "sigma_cal": share * target / math.sqrt(2),
                "sigma_phi_unc": target,
                "histograms": histograms(member_row),
                "cpu_histogram": {
                    name: {"cpu_setup_s": 0.1, "cpu_exact_s": 0.2, "cpu_sampled_s": 0.3}
                    for name in ("total", "x", "z")
                },
            }
            for phi, share, member_row in ((1e-3, 1.0, worst_row or row), (2e-3, 0.5, row))
        ]
        for member in members:
            member["calibrated"] = [member["histograms"][name] for name in ("x", "z")]
        group = {
            "rate": 1e-3,
            "calibration": calibration,
            "num_samples": 10_000,
            "cpu_consumer_setup_s": 0.4,
            "cpu_sampled_s": 5.0,
            "score_timing": {
                "mean_cpu_s": {"rebuild_histogram": 0.01, "phi_cal": 0.02, "phi_unc": 0.03}
            },
        }
        return analyze._cost_lines(members, group)

    lines = cost_lines()
    expected = {
        "sigma_target": target,
        "A_direct": 0.5,
        "b_direct": 1.0 + 2.0 + 4 * 3.0,
        # s_min = MIN_DRAWS / 500 = 2 exceeds (sigma_cal / (sigma* / sqrt 2))^2 = 1 of the worst
        "A_score": 0.4 + (1000 / 500) * 5.0,
        "sigma_cal_max": target / math.sqrt(2),
        # both histograms of Phi_cal have fixed costs, and their sampled costs scale together
        "b_score": 2 * (0.1 + 0.2 + 0.01) + 0.02 + 4 * (2 * 0.3),
        "b_unc": 0.1 + 0.2 + 0.01 + 0.03 + max(1000 / 2000, 1.0) * 0.3,
    }
    for key, value in expected.items():
        assert math.isclose(lines[key], value, rel_tol=1e-12), (key, lines[key], value)
    delta_a = expected["A_score"] - expected["A_direct"]
    delta_b = expected["b_direct"] - expected["b_score"]
    assert lines["k_star"] == analyze.break_even(delta_a, delta_b) == "1", lines["k_star"]

    # one input below its threshold, whose bound is not small, leaves its line unprojected
    for line, broken in (
        ("b_direct", cost_lines(direct_nonzero=9)),
        ("b_direct", cost_lines(accepted=9)),
        ("b_score", cost_lines(row=(100, 1891, 5, 4, 0))),
        ("A_score", cost_lines(cells=((5, 4, 1000), (8, 2, 500)))),
    ):
        assert broken[line] is None and broken["k_star"] == "not computed", (line, broken)
        others = {"b_direct", "b_score", "A_score"} - {line}
        assert all(broken[other] is not None for other in others), (line, broken)

    # residual weights above the calibration's cells give weight to the baseline f(0, 0), which is
    # exact and blocks nothing
    assert cost_lines(row=(100, 1890, 6, 3, 1))["k_star"] != "not computed"

    # a failing cell that only the circuit with the smaller sigma_cal reaches still blocks
    assert (
        cost_lines(cells=((6, 4, 1000), (8, 1, 500)), worst_row=(100, 1890, 10, 0, 0))["A_score"]
        is None
    )
    assert (
        cost_lines(cells=((6, 4, 1000), (8, 1, 500)), row=(100, 1890, 10, 0, 0))["A_score"]
        is not None
    )

    # a failing stratum (1, 1) with eta_1 = 1e-9 is exempted, with its bound reported:
    # (pi_1 eta_1 W)^2 / (n alpha_exact^2) is far below 1% of s_min sigma*^2 = 0.1 sigma*^2
    exempting = cost_lines(direct_nonzero=9, eta_1=1e-9)
    assert exempting["b_direct"] is not None and exempting["k_star"] != "not computed"
    assert math.isclose(
        exempting["exempted_bound_direct"], (0.3e-9) ** 2 / (10_000 * 0.8**2), rel_tol=1e-12
    )

    # the 1% rule, at its boundary, and with the floor s_min sigma_target^2
    assert analyze.exempted([1.0], [0.0099], 0.0) and not analyze.exempted([1.0], [0.0101], 0.0)
    assert analyze.exempted([], [], 0.0) and not analyze.exempted([], [1e-30], 0.0)
    assert not analyze.exempted([1.0], [0.0101], 0.5)
    # s_min = 1000 / 4000 and sigma_target = 2 give a floor of 1, and an exempted bound is reported
    assert analyze.check_projection(([1e-30], [0.0099]), 4000, 2.0) == (True, 0.0099)
    assert analyze.check_projection(([1e-30], [0.0101]), 4000, 2.0) == (False, 0.0)
    assert analyze.check_projection(([1.0], [0.0099]), 0, 2.0) == (True, 0.0099)

    # a reference's terms: estimates that sum to combine_strata's variance, skipping exact strata
    passing_strata = [
        direct.Stratum(1, 0, exact=True, mean=0.2, acceptance=0.5, plus=1),
        direct.Stratum(
            2, 0, exact=False, mean=0.3, variance=0.01, acceptance=0.4,
            acceptance_variance=0.004, covariance=0.002, draws=100, nonzero=20,
            accepted_draws=20, ess=50.0, importance_weight_bound=1.0,
        ),
        direct.Stratum(2, 1, exact=False, mean=0.1, variance=0.02, draws=100, nonzero=30,
                       accepted_draws=40, ess=50.0, importance_weight_bound=1.0),
    ]  # fmt: skip
    combined = direct.combine_strata(passing_strata, pi, u, eta, v)
    reference = {
        "pi": pi.tolist(), "eta": eta.tolist(), "alpha_reference": combined["alpha"],
        "alpha_exact": combined["alpha_exact"], "phi_true": combined["phi"],
        "strata": combined["strata"],
    }  # fmt: skip
    passing, failing = analyze.direct_terms(reference)
    assert not failing and math.isclose(sum(passing), combined["sigma"] ** 2, rel_tol=1e-12)

    # and the bounds of failing strata, the second also estimating a_1 with |Phi| <= 1
    base = {"exact": False, "draws": 100, "importance_weight_bound": 2.0}
    _, failing = analyze.direct_terms(
        {
            "pi": [0.5, 0.3, 0.2], "eta": [0.9, 0.1], "alpha_reference": 0.8,
            "alpha_exact": 0.7,
            "phi_true": 0.01,
            "strata": [
                base | {"f": 1, "g": 1, "variance": 0.5, "nonzero": 3, "accepted_draws": 50},
                base | {"f": 1, "g": 0, "variance": 0.5, "covariance": 0.0,
                        "acceptance_variance": 0.1, "nonzero": 30, "accepted_draws": 5},
            ],
        }
    )  # fmt: skip
    assert math.isclose(failing[0], (0.03 * 2.0) ** 2 / (100 * 0.49), rel_tol=1e-12), failing
    assert math.isclose(failing[1], ((0.27 + 0.3) * 2.0) ** 2 / (100 * 0.49), rel_tol=1e-12)

    # a histogram's terms: estimates that sum to qLDPC's variance, and a failing row's bound
    values = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    row_sums = np.zeros((3, M + 3))
    row_sums[0, 1] = 1.0
    row_sums[1] = [50, 900, 30, 15, 5]
    row_sums[2] = [40, 940, 12, 6, 2]
    passing_histogram = ResidualWeightHistogram(
        max_residual_weight=M,
        css_weight=True,
        fault_weight_probabilities=np.array([0.8, 0.15, 0.04]),
        truncated_probability=0.01,
        weight_sums=row_sums.copy(),
        squared_weight_sums=row_sums * [[1.0], [1.0], [1.5]],
        num_samples=np.array([0, 1000, 1000]),
        exact=np.array([True, False, False]),
        pauli="X",
    )
    passing, failing = analyze.histogram_terms(passing_histogram, values)
    variance = passing_histogram._get_conditional_mean(values)[1] ** 2
    assert not failing and math.isclose(sum(passing), variance, rel_tol=1e-12), (passing, variance)
    row_sums[2] = [40, 952, 4, 3, 1]
    failing_histogram = dataclasses.replace(
        passing_histogram, weight_sums=row_sums, squared_weight_sums=row_sums.copy()
    )
    # row 2 fails with 8 draws of nonzero value; D = (1 - 0) + 1, R^2 / n = 400 / 1000, and the
    # acceptance of the exact row and the truncated mass is 0.8 + 0.01
    _, failing = analyze.histogram_terms(failing_histogram, values)
    assert math.isclose(failing[0], 0.04**2 * 2**2 * 0.4 / 0.81**2, rel_tol=1e-12), failing

    # a candidate with a histogram and a reference keeps the histogram's alpha as its score
    total_histogram = dataclasses.replace(passing_histogram, pauli=None)
    fields = experiment.histogram_fields(total_histogram)
    direct_result = {
        "phi": 0.1, "sigma": 0.01, "tau": 0.0, "alpha": 0.5, "alpha_exact": 0.4,
        "pi": [0.5, 0.5], "eta": [1.0], "inadequate": [], "strata": [], "num_samples": 100,
        "cpu_consumer_setup_s": 0.0, "cpu_prep_setup_s": 0.0, "cpu_exact_s": 0.0,
        "cpu_sampled_s": 0.0,
    }  # fmt: skip
    candidate_rows = {
        f"histogram.z.p{rate:g}": {"status": "ok", "result": {"total": fields, "x": fields}},
        f"direct.z.p{rate:g}": {"status": "ok", "result": direct_result},
    }
    scored = analyze._candidate(candidate_rows, {"id": "z"}, "g", rate, admitted=True, group_row={})
    assert scored["alpha"] == total_histogram.acceptance_probability != 0.5, scored["alpha"]
    assert scored["alpha_reference"] == 0.5, scored

    # a calibration's terms: estimates that sum to sigma_cal^2, and the bounds of failing parts
    def terms_calibration(
        z_cells: tuple[tuple[int, int, int], ...], baseline_1: float
    ) -> ConsumerCalibration:
        """A calibration with two fault weights, X and Z cells at g = 0, and f(0, 1)."""
        cells: dict[Any, tuple[tuple[int, int, int], ...]] = {
            "X": ((60, 10, 1000), (30, 5, 500)),
            "Z": z_cells,
        }
        return ConsumerCalibration(
            {s: [[(i - d) / n, 0.1] for i, d, n in c] for s, c in cells.items()},
            [0.0, baseline_1],
            [0.01],
            excess_standard_errors={
                s: [[float(consumers._get_jeffreys_deviations(i, d, n)), 0.01] for i, d, n in c]
                for s, c in cells.items()
            },
            baseline_standard_errors=[0.0, 0.01],
            excess_num_samples={s: [[n, 1000] for *_, n in c] for s, c in cells.items()},
            state_stabilizers=["+ZIII", "+IZII", "+IIZI", "+IIIZ"],
        )  # fmt: skip

    typed_histograms = [dataclasses.replace(passing_histogram, pauli=s) for s in ("X", "Z")]
    passing_calibration = terms_calibration(((60, 10, 1000), (30, 5, 500)), 0.1)
    passing, failing = analyze.calibration_terms(passing_calibration, typed_histograms, 1000)
    _, sigma_cal = passing_calibration.get_standard_error_parts(*typed_histograms)
    assert not failing and math.isclose(sum(passing), sigma_cal**2, rel_tol=1e-12), passing
    # Z cell (2, 0) has 9 increases and decreases, and the baseline at g = 1 has 9 failures in 6000
    # shots, 1000 for each Pauli type and physical error weight 1 to min(M + 1, 4) = 3; their bounds
    # are Pr[0]^2 / 500 and (2 Pr[1])^2 / 6000, and with 10 failures the baseline passes
    failing_calibration = terms_calibration(((60, 10, 1000), (8, 1, 500)), 9 / 6000)
    _, failing = analyze.calibration_terms(failing_calibration, typed_histograms, 1000)
    pr_0, pr_1 = failing_calibration.fault_weight_probabilities
    expected_bounds = [pr_0**2 / 500, (2 * pr_1) ** 2 / 6000]
    assert np.allclose(failing, expected_bounds, rtol=1e-12, atol=0), failing
    passing_baseline = terms_calibration(((60, 10, 1000), (8, 1, 500)), 10 / 6000)
    _, failing = analyze.calibration_terms(passing_baseline, typed_histograms, 1000)
    assert len(failing) == 1, failing

    # f(0, 0) estimated as 0 is made exact, and any other baseline is left as it is
    errors = np.asarray(analyze.exact_no_fault_baseline(
        dataclasses.replace(passing_calibration, baseline_standard_errors=[0.01, 0.01])
    ).baseline_standard_errors)  # fmt: skip
    assert list(errors) == [0.0, 0.01], errors
    nonzero_baseline = dataclasses.replace(
        passing_calibration,
        baseline_failure_probabilities=[0.1, 0.1],
        baseline_standard_errors=[0.01, 0.01],
    )
    assert analyze.exact_no_fault_baseline(nonzero_baseline) is nonzero_baseline

    # the break-even outcomes, at and near their boundaries
    for delta_a, delta_b, outcome in (
        (5.0, 2.0, "3"),
        (4.0, 2.0, "2"),
        (-1.0, 2.0, "1 (already no costlier)"),
        (-1.0, 0.0, "already no costlier for every k"),
        (1.0, 0.0, "none"),
        (-5.0, -2.0, "only 1 <= k <= 2"),
        (-2.0, -2.0, "only 1 <= k <= 1"),
        (-1.0, -2.0, "none"),
    ):
        assert analyze.break_even(delta_a, delta_b) == outcome, (delta_a, delta_b)

    # projected costs: the minimum allocation, the scaling, and no projection where invalid
    assert analyze.project(1.0, 2.0, 0.1, 1.0, 10_000, True) == 1.0 + 0.1 * 2.0
    assert analyze.project(1.0, 2.0, 3.0, 1.0, 10_000, True) == 1.0 + 9.0 * 2.0
    assert analyze.project(1.0, 2.0, 3.0, 1.0, 10_000, False) is None

    # the nonzero draws of a calibration cell, through qLDPC's Jeffreys error
    for increases, decreases, count in ((3, 1, 50), (40, 12, 1000), (0, 0, 10)):
        error = float(consumers._get_jeffreys_deviations(increases, decreases, count))
        mean = (increases - decreases) / count
        assert analyze.nonzero_changes(mean, error, count) == increases + decreases

    # a row with compiled_in_timing set (the row of a retry that compiled too) gives no CPU seconds:
    # evaluate drops those of that row alone, whether a histogram's, the direct reference's, or the
    # calibration's
    calib_result = {
        "calibration": passing_calibration.to_dict(),
        "num_samples": 1000,
        "cpu_consumer_setup_s": 0.4,
        "cpu_sampled_s": 5.0,
        "score_timing": {"mean_cpu_s": {"rebuild_histogram": 0.01, "phi_cal": 0.02}},
    }
    cpu = {"cpu_setup_s": 0.1, "cpu_exact_s": 0.2, "cpu_sampled_s": 0.3}
    typed_fields = {
        name: experiment.histogram_fields(dataclasses.replace(passing_histogram, pauli=pauli)) | cpu
        for name, pauli in experiment.PAULI_TYPES.items()
    }
    timed_rows = {
        "histogram": (
            f"histogram.z.p{rate:g}",
            {"status": "ok", "result": {"total": fields | cpu, **typed_fields}},
        ),
        "direct": (f"direct.z.p{rate:g}", candidate_rows[f"direct.z.p{rate:g}"]),
        "calib": (f"calib.g.p{rate:g}", {"status": "ok", "result": calib_result}),
    }
    state = {"id": "z", "stabilizers": passing_calibration.state_stabilizers}
    admission = [{"id": "z", "admitted": True, "group_in_scope": True}]
    for flagged in (None, *timed_rows):
        flagged_rows = {
            key: row | {"compiled_in_timing": name == flagged}
            for name, (key, row) in timed_rows.items()
        }
        evaluated, groups = analyze.evaluate(flagged_rows, {"g": [state]}, admission)
        (scored,) = [c for c in evaluated if c["rate"] == rate]
        (group_row,) = [g for g in groups if g["rate"] == rate]
        assert "phi_cal" in scored and "phi_true" in scored, (flagged, scored)
        parts = analyze.exact_no_fault_baseline(passing_calibration).get_standard_error_parts(
            *typed_histograms
        )
        assert np.allclose((scored["sigma_hist"], scored["sigma_cal"]), parts, rtol=1e-12, atol=0)
        present = {
            "histogram": "cpu_histogram" in scored,
            "direct": "cpu_direct" in scored,
            "calib": "score_timing" in group_row or "cpu_sampled_s" in group_row,
        }
        assert present == {name: name != flagged for name in present}, (flagged, present)
        expected_arms = flagged if flagged in ("histogram", "direct") else None
        assert scored["compiled_arms"] == expected_arms, (flagged, scored)
        assert group_row["compiled_in_timing_calib"] is (flagged == "calib"), group_row

    # a histogram task that stopped after its total-weight histogram still gives Phi_unc
    total_only = {f"histogram.z.p{rate:g}": {"status": "timeout", "partial": [{"total": fields}]}}
    scored = analyze._candidate(
        total_only,
        {"id": "z"},
        "g",
        rate,
        admitted=True,
        group_row={"calibration": passing_calibration},
    )
    assert "phi_unc" in scored and "phi_cal" not in scored, scored

    # such a row of an admitted circuit leaves its group without costs, and checks.json lists it
    members = [{"group": "g", "rate": rate, "admitted": True, "compiled_arms": None}]
    members += [members[0] | {"compiled_arms": "direct"}]
    (cost_line,) = analyze.costs(members, [{"group": "g", "rate": rate}])
    assert cost_line["reason"] == "a row has compiled_in_timing set", cost_line
    label_dir = directory / "compiled"
    label_dir.mkdir(parents=True, exist_ok=True)
    store.append_row(
        label_dir, {"task_key": "direct.c", "status": "ok", "compiled_in_timing": True}
    )
    assert analyze._ranking_rows(label_dir)[1] == ["direct.c: compiled_in_timing"]

    # a row of leading_order.csv: the coefficient, its leading term at the frozen rate, and the
    # reference there
    frozen = float(experiment.CONFIG["exact"]["frozen_rate"])
    exact_rows = {
        "exact.y": {
            "arm": "exact",
            "status": "ok",
            "result": {"order": 2, "coefficient": "3/2", "value": 1.5, "note": None},
        }
    }
    reference = {"id": "y", "rate": frozen, "phi_true": 1e-8, "sigma_true": 1e-9}
    elsewhere = {"id": "y", "rate": 2 * frozen, "phi_true": 1.0, "sigma_true": 1.0}
    (leading,) = analyze.leading_order_rows(exact_rows, [elsewhere, reference])
    assert leading == {
        "id": "y", "status": "ok", "order": 2, "coefficient": "3/2", "note": None,
        "leading_term": 1.5 * frozen**2, "phi_true_direct": 1e-8, "sigma_true_direct": 1e-9,
    }, leading  # fmt: skip


####################################################################################################
# helpers


def synthetic_certification_archive(
    directory: pathlib.Path,
    verdicts: dict[str, tuple[str, str]],
    subset: str = "main",
    bundle: bool = True,
    dirty: bool = False,
    label: str | None = None,
) -> pathlib.Path:
    """A certification archive of a bundle's run, as ``run_node.py`` names it, with given verdicts.

    Each circuit gets one ``cert_bz`` and one ``cert_cc`` row, with the verdicts "certified" or
    "violation at m".  The label is that of ``manifest.compute_label`` for the harness commit
    HEAD, this tree's ``pins.json``, certification's ``config.json`` and ``circuits.json``, and a
    stand-in for the bundle's requirements; the other arguments change the manifest.
    """
    commit = subprocess.run(
        ["git", "-C", str(manifest.REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    requirements = "0" * 64
    frozen = [
        experiment.CIRCUITS_JSON.parents[2] / "config.json",
        experiment.CIRCUITS_JSON,
    ]
    label = label or manifest.compute_label(
        "certification", manifest.label_files(frozen), commit, requirements
    )
    label_dir = directory / label
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "circuits.json").write_bytes(experiment.CIRCUITS_JSON.read_bytes())
    record = {
        "label": label,
        "subset": subset,
        "harness": {"commit": commit, "dirty": dirty, "source": "bundle.json"},
        "bundle": {"requirements_sha256": requirements} if bundle else None,
    }
    (label_dir / "manifest.json").write_text(json.dumps(record) + "\n")
    for circuit_id, (bz, cc) in verdicts.items():
        for arm, verdict in (("cert_bz", bz), ("cert_cc", cc)):
            certified = verdict == "certified"
            store.append_row(
                label_dir,
                {
                    "task_key": f"{arm}.{circuit_id}.r0",
                    "arm": arm,
                    "repeat": 0,
                    "status": "ok",
                    "post_status": "ok",
                    "T": 1.0,
                    "peak_rss_gib": 0.1,
                    "partial": [],
                    "result": {
                        "certified": certified,
                        "violation_weight": None if certified else int(verdict.split()[-1]),
                        "zero_flag_checks": {"calls": 0},
                        "classification": {"calls": 0},
                        "batches": {},
                    },
                },
            )
    return store.make_archive(label_dir, f"{label}-node-20261118T000000Z")


def _variant(
    directory: pathlib.Path,
    name: str,
    verdicts: dict[str, tuple[str, str]],
    outer: bool = False,
    inner: bool = False,
    **changes: Any,
) -> pathlib.Path:
    """A synthetic certification archive with one change to its manifest or its checksums.

    ``outer`` changes the archive's ``.sha256`` file, and ``inner`` replaces a file of the archive
    without updating its ``SHA256SUMS``.
    """
    archive = synthetic_certification_archive(directory / name, verdicts, **changes)
    if outer:
        sums = archive.with_name(archive.name + ".sha256")
        sums.write_text(f"{'0' * 64}  {archive.name}\n")
    if inner:
        with tarfile.open(archive) as tar:
            members = [(member, tar.extractfile(member)) for member in tar.getmembers()]
            contents = {m.name: (f.read() if f else b"") for m, f in members}
        with tarfile.open(archive, "w:gz") as tar:
            for name_, data in contents.items():
                data = data + b"\n" if name_.endswith("runs.jsonl") else data
                info = tarfile.TarInfo(name_)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        sums = archive.with_name(archive.name + ".sha256")
        sums.write_text(f"{store.sha256_file(archive)}  {archive.name}\n")
    return archive


def _frozen_pairing(
    entry: dict[str, Any], frozen: float
) -> tuple[direct.Pairing, list[fractions.Fraction], list[fractions.Fraction]]:
    """The pairing at the frozen rate and the coefficients of p, as the ``exact`` task has them."""
    probe = float(experiment.CONFIG["exact"]["probe_rate"])
    options = experiment.histogram_options(entry)
    preparation = direct.get_preparation(experiment.noisy_input(entry, frozen), options, M)
    probe_preparation = direct.get_preparation(experiment.noisy_input(entry, probe), options, M)
    rng = np.random.default_rng(0)
    simulation = consumers._ConsumerSimulation(ranking_consumer.build_consumer(entry, frozen), rng)
    probe_simulation = consumers._ConsumerSimulation(
        ranking_consumer.build_consumer(entry, probe), rng
    )
    return (
        direct.Pairing(preparation, simulation),
        direct.first_order_coefficients(
            preparation.probabilities, probe_preparation.probabilities, probe
        ),
        direct.first_order_coefficients(
            simulation.probabilities, probe_simulation.probabilities, probe
        ),
    )


def _plain_failure(
    pairing: direct.Pairing, error: list[int], consumer_fault: tuple[int, ...]
) -> int:
    """Whether the consumer fails with one block error and one consumer fault, decoded alone."""
    simulation = pairing.simulation
    detectors, observables = simulation.get_flips(
        np.array([consumer_fault], dtype=np.intp).reshape(1, -1)
    )
    return int(
        simulation.get_failures(detectors, observables, np.array([error], dtype=np.uint8))[0]
    )


def _plain_change(
    pairing: direct.Pairing, fault: tuple[int, ...], consumer_fault: tuple[int, ...]
) -> int:
    """delta(F, T), from Python integers: the flags and output error of F by exclusive or."""
    preparation = pairing.preparation
    flags = 0
    error = [0] * preparation.errors.shape[1]
    for mechanism in fault:
        for word in preparation.flag_words[mechanism]:
            flags ^= int(word)
        error = [a ^ int(b) for a, b in zip(error, preparation.errors[mechanism])]
    if flags:
        return 0
    zero = [0] * len(error)
    return _plain_failure(pairing, error, consumer_fault) - _plain_failure(
        pairing, zero, consumer_fault
    )


def _plain_counts(pairing: direct.Pairing, f: int, g: int) -> tuple[int, int]:
    """The numbers of pairs (F, T) of weights f and g with changes +1 and -1."""
    plus = minus = 0
    num_consumer = len(pairing.simulation.probabilities)
    for fault in itertools.combinations(range(len(pairing.preparation)), f):
        for consumer_fault in itertools.combinations(range(num_consumer), g):
            change = _plain_change(pairing, fault, consumer_fault)
            plus += change > 0
            minus += change < 0
    return plus, minus


def _plain_leading_order(
    pairing: direct.Pairing,
    prep_c: list[fractions.Fraction],
    consumer_c: list[fractions.Fraction],
) -> str | None:
    """The leading coefficient of Phi_true, by plain enumeration."""
    num_mechanisms = len(pairing.preparation)
    singles = [(j,) for j in range(num_mechanisms)]
    terms = [fractions.Fraction(_plain_change(pairing, fault, ())) for fault in singles]
    first = sum((c * term for c, term in zip(prep_c, terms)), fractions.Fraction())
    if first:
        return str(first)
    if any(terms):
        return None
    total = fractions.Fraction()
    for fault in itertools.combinations(range(num_mechanisms), 2):
        total += prep_c[fault[0]] * prep_c[fault[1]] * _plain_change(pairing, fault, ())
    for j in range(num_mechanisms):
        for k in range(len(consumer_c)):
            total += prep_c[j] * consumer_c[k] * _plain_change(pairing, (j,), (k,))
    return str(total)


if __name__ == "__main__":
    SUITE.main(__doc__)
