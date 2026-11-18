# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests of the ranking experiment.

    python acceptance.py [--local] [--only RT1 RT3 ...] [--results-dir R]

On the node, before the run, only RT1, RT2, and RT8 run; with ``--local``, every test runs.  Each
test is stopped after ``acceptance_limit_s`` of ``config.json``.  RT6 and RT7 are evaluated by
``analyze.py`` on the node's ``mc_check`` and ``repro`` tasks.

* **RT1:** certification's inputs and code files pass ``circuits.check_inputs``, and every circuit
  of a group prepares the group's signed state on its output qubits, in order.
* **RT2:** at each rate, Stim builds the detector error model of each group's consumer, and every
  single-qubit Z error on the block flips no detector or observable.  Whether the decoder corrects
  every single-qubit X error on the block, with an ideal consumer, is recorded, with the number of
  the consumer's mechanisms and its fault distance, in ``RT2/consumers.json``; it is not a failure.
* **RT3:** on ``cgoto`` and ``aft.carbon.opt_opt`` at p = 10^-3, the reference's mechanism list
  gives, on each exactly listed fault weight, the acceptance and the probabilities of each X
  residual weight of qLDPC's X-type histogram, to 10^-12; the consumer's fault-weight law and its
  largest fault weight equal those of the calibration.  On ``cgoto``, sampled strata of weights
  that are also listed exactly agree with the listing within four standard errors.
* **RT4:** on ``cgoto`` at p = 10^-3, with every fault weight up to 4 listed and a calibration
  with exact cells, Phi_unc and Phi_cal from qLDPC equal a computation from the listed faults, to
  10^-12.  The calibration contribution of ``analyze.calibration_difference_error`` equals
  sqrt(sigma^2 - sigma_hist^2) of qLDPC's errors, and vanishes for a difference of a circuit with
  itself; the standard error of the acceptance of a sampled fault weight with equal importance
  weights is the binomial one.
* **RT5:** the ``exact`` task's coefficients on the four steane circuits equal those of a plain
  enumeration over the same mechanism lists and frozen decoder, exactly; and in every stratum with
  f + g <= 2, at each rate, the reference's numbers of paired changes +1 and -1 equal those of the
  plain enumeration.
* **RT8:** no ``mqt`` or ``qiskit`` module is loaded.
* **RT9:** the analysis (plan section 9): on a synthetic certification archive, each verdict and a
  disagreement admit or drop their circuits as the rule says, a circuit without a row is not
  admitted, and an archive whose ``circuits.json`` differs from the ranking label's fails the
  provenance check.  The four cases of the break-even number give their outcomes.  A ``direct``
  task stopped at a limit of 3 s is a timeout that keeps its finished strata.

Each test prints PASS or FAIL; the script exits with status 1 if any failed.
"""

from __future__ import annotations

import argparse
import fractions
import functools
import itertools
import json
import math
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import circuits as certification_circuits
from experiments.state_prep_certification.common import local_runs, manifest, store
from experiments.state_prep_certification.ranking import analyze, direct, experiment
from experiments.state_prep_certification.ranking import consumer as consumer_module
from qldpc import circuits
from qldpc.experimental.circuits.benchmarking import (
    ConsumerCalibration,
    ResidualWeightHistogram,
    consumers,
    estimate_residual_weight_histogram,
    residual_weights,
)
from qldpc.objects import Pauli, PauliXZ

FAILURES = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SystemExit,
    TypeError,
    ValueError,
)
NODE_TESTS = ("RT1", "RT2", "RT8")
M = experiment.MAX_RESIDUAL_WEIGHT
DECODER = experiment.CONFIG["decoder"]
TESTS: dict[str, Callable[[pathlib.Path], None]] = {}


def main() -> None:
    """Run the selected tests."""
    args = parse_args()
    results_dir = args.results_dir or pathlib.Path(tempfile.mkdtemp(prefix="spc-ranking-"))
    passed = run_all(node=not args.local, results_dir=results_dir, only=args.only)
    sys.exit(0 if passed else 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--local", action="store_true", help="run every test, off the node")
    parser.add_argument("--only", nargs="*", choices=sorted(TESTS), help="run only these tests")
    parser.add_argument("--results-dir", type=pathlib.Path, help="where the tests' outputs go")
    return parser.parse_args()


def run_all(node: bool, results_dir: pathlib.Path, only: list[str] | None = None) -> bool:
    """Run the tests, print PASS or FAIL for each, and return whether all passed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    limit = float(experiment.CONFIG["acceptance_limit_s"])
    failed = []
    for name, test in TESTS.items():
        if (only and name not in only) or (node and name not in NODE_TESTS):
            continue
        start = time.monotonic()
        directory = results_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGALRM, _stop)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            test(directory)
            print(f"PASS {name} ({time.monotonic() - start:.1f} s)", flush=True)
        except FAILURES:
            failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    print(f"{len(failed)} failed: {failed}" if failed else "all passed", flush=True)
    return not failed


def test(function: Callable[[pathlib.Path], None]) -> Callable[[pathlib.Path], None]:
    """Register a test by the ID at the start of its name, as in ``rt1_...`` for RT1."""
    TESTS[function.__name__.split("_")[0].upper()] = function
    return function


def _stop(signum: int, frame: Any) -> None:
    """Stop a test that exceeds its limit."""
    raise TimeoutError("the test exceeded its limit")


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
    """Each group's consumer at each rate: its DEM, Z errors, and single X errors."""
    records = []
    for group, members in experiment.groups().items():
        entry = members[0]
        num_qubits = len(entry["output_qubits"])
        for rate in experiment.CONFIG["rates"]:
            consumer = consumer_module.build_consumer(entry, rate, DECODER)
            simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))
            assert not np.any(simulation._responses[num_qubits:]), (group, rate)
            errors = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
            errors[range(num_qubits), range(num_qubits)] = 1
            flips = simulation.get_flips(np.zeros((num_qubits, 0), dtype=np.intp))
            failures = simulation.get_failures(*flips, errors)
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
                    "uncorrected_single_x_errors": np.flatnonzero(failures).tolist(),
                }
            )
            print(f"  {records[-1]}")
    (directory / "consumers.json").write_text(json.dumps(records, indent=1) + "\n")


@test
def rt3_mechanisms(directory: pathlib.Path) -> None:
    """The reference's mechanism lists against qLDPC's histograms and calibration."""
    rate = 0.001
    for circuit_id in ("cgoto", "aft.carbon.opt_opt"):
        entry = experiment.entry_of(circuit_id)
        noisy = experiment.noisy_input(entry, rate)
        options = experiment.histogram_options(entry)
        preparation = direct.get_preparation(noisy, options, M)
        histogram = estimate_residual_weight_histogram(noisy, M, 0, pauli="X", **options)
        for f in np.flatnonzero(histogram.exact)[1:]:
            faults = residual_weights._get_combinations(len(preparation), int(f))
            probabilities = direct._conditional_probabilities(preparation.probabilities, faults)
            accepted = preparation.accepted(faults)
            weights = preparation.residual_weights(preparation.fault_errors(faults), Pauli.X, M)
            outcomes = np.where(accepted, 1 + weights, 0)
            expected = np.bincount(outcomes, weights=probabilities, minlength=M + 3)
            assert np.allclose(expected, histogram.weight_sums[f], rtol=0, atol=1e-12), (
                circuit_id,
                f,
            )
        print(f"  {circuit_id}: fault weights 1-{np.flatnonzero(histogram.exact)[-1]} agree")

        consumer = consumer_module.build_consumer(entry, rate, DECODER)
        simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))
        calibration = consumer.calibrate(M, 10, paulis=("X",), seed=0)
        eta, _ = direct.fault_weight_law(
            simulation.probabilities, M, consumers._FAULT_WEIGHT_TOLERANCE
        )
        law, _ = calibration._get_distribution()
        assert len(eta) == len(law) and np.allclose(eta, law, rtol=0, atol=1e-12), circuit_id

    # sampled strata against their exact listings
    entry = experiment.entry_of("cgoto")
    consumer = consumer_module.build_consumer(entry, rate, DECODER)
    pairing = direct.build_pairing(
        experiment.noisy_input(entry, rate),
        experiment.histogram_options(entry),
        consumer,
        M,
        np.random.default_rng(0),
    )
    factor = float(experiment.CONFIG["direct"]["flagged_odds_factor"])
    for f, g in ((2, 0), (2, 1), (3, 0)):
        exact = direct._list_stratum(pairing, f, g)
        sampled = direct._sample_stratum(
            pairing, f, g, 20_000, factor, np.random.default_rng(f + g)
        )
        assert abs(sampled.mean - exact.mean) <= 4 * math.sqrt(sampled.variance), (f, g)
        if g == 0:
            assert exact.acceptance is not None and sampled.acceptance is not None
            gap = abs(sampled.acceptance - exact.acceptance)
            assert gap <= 4 * math.sqrt(sampled.acceptance_variance), (f, g)
        print(f"  cgoto ({f}, {g}): sampled {sampled.mean:.4f}, exact {exact.mean:.4f}")


@test
def rt4_scores(directory: pathlib.Path) -> None:
    """Phi_unc and Phi_cal against a computation from listed faults, and the error formulas."""
    rate = 0.001
    entry = experiment.entry_of("cgoto")
    noisy = experiment.noisy_input(entry, rate)
    options = experiment.histogram_options(entry)
    total = estimate_residual_weight_histogram(noisy, M, 0, max_fault_weight=4, **options)
    x_type = estimate_residual_weight_histogram(
        noisy, M, 0, max_fault_weight=4, pauli="X", **options
    )
    assert total.exact.all() and x_type.exact.all()
    consumer = consumer_module.build_consumer(entry, rate, DECODER)
    simulation = consumers._ConsumerSimulation(consumer, np.random.default_rng(0))
    num_columns = 4
    table = np.array([[0.1, 0.3, 0.5, 0.6], [0.7, 0.6, 0.4, 0.2]])
    baseline = np.array([0.0, 0.05, 0.2, 0.3])
    calibration = ConsumerCalibration(
        {"X": table}, baseline, simulation.probabilities, description="exact cells"
    )
    distribution, truncated = calibration._get_distribution()
    no_block_error = float(baseline @ distribution) + truncated
    values_cal = [0.0, *(table @ distribution), 1 - no_block_error]
    profile = experiment.unc_profile(calibration)
    failures = [profile.get_failure_probability(m) for m in range(M + 2)]
    values_unc = [failure - failures[0] for failure in failures]

    # the listed faults, each weighed by its probability, with heavier fault weights accepted with
    # overflow
    preparation = direct.get_preparation(noisy, options, M)
    pi, u = residual_weights._get_fault_weight_probabilities(preparation.probabilities, 4)
    cases: list[tuple[list[float], PauliXZ | None, ResidualWeightHistogram, float]] = [
        (values_unc, None, total, experiment.scores(total, x_type, calibration)["phi_unc"][0]),
        (values_cal, Pauli.X, x_type, calibration.get_excess_failure_probability(x_type)[0]),
    ]
    for values, pauli, histogram, score in cases:
        numerator, acceptance = u * values[-1], u
        for f in range(num_columns + 1):
            faults = residual_weights._get_combinations(len(preparation), f)
            weights = direct._conditional_probabilities(preparation.probabilities, faults)
            accepted = preparation.accepted(faults)
            residual = preparation.residual_weights(preparation.fault_errors(faults), pauli, M)
            acceptance += pi[f] * float(weights @ accepted)
            numerator += pi[f] * float(weights[accepted] @ np.array(values)[residual[accepted]])
        assert histogram.exact.all()
        assert abs(score - numerator / acceptance) <= 1e-12, (pauli, score, numerator / acceptance)
        print(f"  {pauli or 'total'}: {score:.6e} from qLDPC and from the listed faults")

    # the calibration contribution, by the difference of cell weights
    calibrated = consumer.calibrate(M, 300, paulis=("X",), seed=1)
    histogram = estimate_residual_weight_histogram(noisy, M, 2000, pauli="X", seed=2, **options)
    other = estimate_residual_weight_histogram(noisy, M, 2000, pauli="X", seed=3, **options)
    sigma = calibrated.get_excess_failure_probability(histogram)[1]
    without = ConsumerCalibration.from_dict(
        calibrated.to_dict() | {"excess_standard_errors": None, "baseline_standard_errors": None}
    )
    sigma_hist = without.get_excess_failure_probability(histogram)[1]
    sigma_cal = analyze.calibration_difference_error(calibrated, histogram)
    assert abs(sigma_cal**2 - (sigma**2 - sigma_hist**2)) <= 1e-12 * sigma**2, (sigma, sigma_cal)
    assert analyze.calibration_difference_error(calibrated, histogram, histogram) == 0
    assert analyze.calibration_difference_error(calibrated, histogram, other) > 0

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
        for name in ("true", "cal"):
            assert result[name]["coefficient"] == plain[name], (entry["id"], name, result, plain)
        print(
            f"  {entry['id']}: {result['true']['coefficient']} and {result['cal']['coefficient']}"
        )

        for rate in experiment.CONFIG["rates"]:
            consumer = consumer_module.build_consumer(entry, rate, DECODER)
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
    """Admission, provenance, the break-even cases, and a timeout that keeps its strata."""
    verdicts = {
        "aft.steane.heuristic_heuristic": ("certified", "certified"),
        "aft.steane.heuristic_opt": ("violation at 2", "violation at 2"),
        "aft.steane.opt_opt": ("violation at 1", "violation at 1"),
        "cgoto": ("certified", "violation at 2"),
    }
    circuits_bytes = experiment.CIRCUITS_JSON.read_bytes()
    archive = synthetic_certification_archive(directory / "certification", verdicts)
    label_dir = analyze.check_provenance(archive, circuits_bytes)
    admission = {row["id"]: row for row in analyze.admit(label_dir, experiment.groups())}
    assert admission["aft.steane.heuristic_heuristic"]["admitted"]
    assert admission["aft.steane.heuristic_opt"]["admitted"]
    assert not admission["aft.steane.opt_opt"]["admitted"]
    assert admission["cgoto"]["verdict"] == "disagreement" and not admission["cgoto"]["admitted"]
    assert admission["aft.shor.opt_opt"]["verdict"] == "no row"
    assert admission["aft.steane.heuristic_opt"]["group_in_scope"]
    assert not admission["aft.shor.opt_opt"]["group_in_scope"]
    other = json.loads(circuits_bytes)
    other["circuits"] = other["circuits"][:-1]
    try:
        analyze.check_provenance(archive, json.dumps(other).encode())
    except analyze.ProvenanceError:
        pass
    else:
        raise AssertionError("a different circuits.json passed the provenance check")

    assert analyze.break_even(5.0, 2.0) == "3"
    assert analyze.break_even(-1.0, 2.0) == "1 (already no costlier)"
    assert analyze.break_even(-1.0, 0.0) == "already no costlier for every k"
    assert analyze.break_even(1.0, 0.0) == "none"
    assert analyze.break_even(-5.0, -2.0) == "only 1 <= k <= 2"
    assert analyze.break_even(-1.0, -2.0) == "none"

    task = experiment._task(
        "direct", "aft.hamming.opt_opt.p0.005", {"id": "aft.hamming.opt_opt", "rate": 0.005}
    ) | {"limit_s": 3.0}
    row = local_runs.run_locally(directory / "timeout", [task])[task["key"]]
    strata = [record["stratum"] for record in row["partial"]]
    assert row["status"] == "timeout" and strata, row["status"]
    assert all(stratum["f"] >= 1 for stratum in strata), strata
    print(f"  a direct task stopped at 3 s keeps {len(strata)} strata")


####################################################################################################
# helpers


def synthetic_certification_archive(
    directory: pathlib.Path, verdicts: dict[str, tuple[str, str]]
) -> pathlib.Path:
    """A certification archive of subset ``main`` from a bundle, with the given verdicts.

    Each circuit gets one ``cert_bz`` and one ``cert_cc`` row, with the verdicts "certified" or
    "violation at m".  The label is that of the harness commit HEAD, the current ``circuits.json``,
    and a stand-in for the bundle's requirements.
    """
    commit = subprocess.run(
        ["git", "-C", str(manifest.REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    requirements = "0" * 64
    files = [
        ("pins.json", analyze._git_show(commit, "experiments/state_prep_certification/pins.json")),
        (
            "config.json",
            analyze._git_show(
                commit, "experiments/state_prep_certification/certification/config.json"
            ),
        ),
        ("circuits.json", experiment.CIRCUITS_JSON.read_bytes()),
    ]
    label = analyze.certification_label(files, commit, requirements)
    label_dir = directory / label
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "circuits.json").write_bytes(files[2][1])
    record = {
        "label": label,
        "subset": "main",
        "harness": {"commit": commit, "dirty": False, "source": "bundle.json"},
        "bundle": {"requirements_sha256": requirements},
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
    return store.make_archive(label_dir, label)


def _frozen_pairing(
    entry: dict[str, Any], frozen: float
) -> tuple[direct.Pairing, list[fractions.Fraction], list[fractions.Fraction]]:
    """The pairing at the frozen rate and the coefficients of p, as the ``exact`` task has them."""
    probe = float(experiment.CONFIG["exact"]["probe_rate"])
    options = experiment.histogram_options(entry)
    preparation = direct.get_preparation(experiment.noisy_input(entry, frozen), options, M)
    probe_preparation = direct.get_preparation(experiment.noisy_input(entry, probe), options, M)
    rng = np.random.default_rng(0)
    simulation = consumers._ConsumerSimulation(
        consumer_module.build_consumer(entry, frozen, DECODER), rng
    )
    probe_simulation = consumers._ConsumerSimulation(
        consumer_module.build_consumer(entry, probe, DECODER), rng
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
) -> dict[str, str | None]:
    """The leading coefficients of Phi_true and Phi_cal, by plain enumeration."""
    preparation = pairing.preparation
    num_qubits = preparation.stabilizers.shape[1] // 2
    num_mechanisms = len(preparation)
    num_consumer = len(consumer_c)

    def residual_x(fault: tuple[int, ...]) -> int:
        """The X residual weight of a fault's output error."""
        error = np.zeros(2 * num_qubits, dtype=np.uint8)
        for mechanism in fault:
            error ^= preparation.errors[mechanism]
        return int(preparation.residual_weights(error[None, :], Pauli.X, M)[0])

    @functools.cache
    def cell(m: int, consumer_fault: tuple[int, ...]) -> fractions.Fraction:
        """The exact excess of the ensemble of residual weight m, with a fixed consumer fault."""
        if m == 0:
            return fractions.Fraction()
        zero = [0] * (2 * num_qubits)
        if m > M:
            return 1 - fractions.Fraction(_plain_failure(pairing, zero, consumer_fault))
        mass = changed = fractions.Fraction()
        for weight in range(1, min(M + 1, num_qubits) + 1):
            for support in itertools.combinations(range(num_qubits), weight):
                error = [int(q in support) for q in range(num_qubits)] + [0] * num_qubits
                errors = np.array([error], dtype=np.uint8)
                if preparation.residual_weights(errors, Pauli.X, M)[0] != m:
                    continue
                share = fractions.Fraction(1, math.comb(num_qubits, weight))
                mass += share
                changed += share * (
                    _plain_failure(pairing, error, consumer_fault)
                    - _plain_failure(pairing, zero, consumer_fault)
                )
        return changed / mass if mass else fractions.Fraction()

    def accepted(fault: tuple[int, ...]) -> bool:
        """Whether post-selection accepts a fault."""
        flags = 0
        for mechanism in fault:
            for word in preparation.flag_words[mechanism]:
                flags ^= int(word)
        return not flags

    result: dict[str, str | None] = {}
    singles = [(j,) for j in range(num_mechanisms)]
    true_terms = [fractions.Fraction(_plain_change(pairing, fault, ())) for fault in singles]
    cal_terms = [cell(residual_x(fault), ()) if accepted(fault) else 0 for fault in singles]
    for name, terms in (("true", true_terms), ("cal", cal_terms)):
        first = sum((c * term for c, term in zip(prep_c, terms)), fractions.Fraction())
        if first:
            result[name] = str(first)
            continue
        if any(terms):
            result[name] = None
            continue
        total = fractions.Fraction()
        for fault in itertools.combinations(range(num_mechanisms), 2):
            value = (
                _plain_change(pairing, fault, ())
                if name == "true"
                else (cell(residual_x(fault), ()) if accepted(fault) else 0)
            )
            total += prep_c[fault[0]] * prep_c[fault[1]] * value
        for j in range(num_mechanisms):
            for k in range(num_consumer):
                value = (
                    _plain_change(pairing, (j,), (k,))
                    if name == "true"
                    else (cell(residual_x((j,)), (k,)) if accepted((j,)) else 0)
                )
                total += prep_c[j] * consumer_c[k] * value
        result[name] = str(total)
    return result


if __name__ == "__main__":
    main()
