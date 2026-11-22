# SPDX-License-Identifier: Apache-2.0

"""The two applications of the certifier: exact margins and the deletion of verification.

Both arms are tasks of the certification plug-in that make several certification calls within one
limit, and send each completed step to the runner as a partial result, so that a run killed at its
limit keeps its completed steps.  A task's ``step_delay_s``, which only the acceptance tests set,
sleeps that many seconds after each step, so that a short limit ends the run between steps.

* ``margin``: BZ certification with no limit on the fault weight, on a circuit whose flag rank r
  exceeds its design t.  A step is a completed fault weight.  The run gives m*, the smallest fault
  weight of an accepted fault whose residual weight exceeds its fault weight.  It gives "all" if no
  such fault exists at any weight: the search proves this when it completes the cutoff r + 1, or
  earlier when no flagged mechanism is heavy, and ``cutoff_reached`` says which.  A run killed at
  its limit gives its completed weights without a violation, so that m* exceeds the largest of
  them, and the weight at which it found a violation, if it found one, which is m*.
* ``prune`` (and ``prune_sensitivity``, the same in another order): delete the measured qubits of a
  verified circuit one at a time, and keep a deletion only if the circuit then still prepares the
  parent's signed state on the parent's output qubits, qLDPC accepts its detectors, and BZ certifies
  it at K = t.  A step is one tried qubit.  The search of the final m* sends each completed fault
  weight too, and ``step_delay_s`` delays each of these as it does a step.  ``prune`` tries the
  measured qubits in decreasing order of their last measurement ("last measured first"), and
  ``prune_sensitivity`` in increasing order of their first measurement ("first measured first").
  The final circuit is then certified at K = t by the connected-cluster oracle, the acceptance
  probability of parent and final circuit is estimated by sampling, and these are sent as a partial
  result, with the final circuit, before the final circuit gets its m*, whose search sends each
  completed fault weight as a partial result.  A parent that is not certified at K = t gets no step.

Deleting a qubit removes every instruction target group that involves it, as for the weakened
variants (``circuits._delete_qubit``).  The sector of a counterexample is the residual weight of the
X part and of the Z part of its output error, each minimized modulo the stabilizers of the prepared
state; for CSS weight, the larger of the two is the residual weight.  ``describe_certificate``
describes a certificate, with the sector of its counterexample, for the margins and for the first
runs of ``cert_bz`` and ``cert_cc`` (``experiment.py``).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import stim

from experiments.state_prep_certification.certification import circuits, instrument
from experiments.state_prep_certification.certification.settings import (
    CONFIG,
    INPUTS,
    certify_options,
    noisy_circuit,
    noisy_input,
)
from experiments.state_prep_certification.certification.states import same_state, stabilizer_matrix
from experiments.state_prep_certification.common import runner
from qldpc import circuits as qldpc_circuits
from qldpc.codes.distance import CosetLeaderSearch

ARMS = ("margin", "prune", "prune_sensitivity")
PRUNE_ORDERS = {"prune": "last_measured_first", "prune_sensitivity": "first_measured_first"}

# The number of shots that each batch of the acceptance estimate samples, which bounds its memory.
_SHOTS_PER_BATCH = 100_000


def run(task: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Run a task of ``margin``, ``prune``, or ``prune_sensitivity`` on a circuit of the list."""
    fields: dict[str, Any] = {"id": entry["id"], "family": entry["family"], "t": entry["t"]}

    def report(record: dict[str, Any]) -> None:
        runner.send_partial(record)
        time.sleep(float(task.get("step_delay_s", 0.0)))

    if task["arm"] == "margin":
        noisy = noisy_input(entry)
        options = certify_options(entry)
        result = runner.timed(find_margin, noisy, options, report)
        return fields | result | {"cutoff_reached": cutoff_reached(result, entry["r"])}
    circuit = stim.Circuit((INPUTS / f"{entry['id']}.stim").read_text())
    seed = int(hashlib.sha256(task["key"].encode()).hexdigest()[:8], 16)
    order = PRUNE_ORDERS[task["arm"]]
    return fields | runner.timed(prune, circuit, entry, order, seed, report)


####################################################################################################
# margins


def find_margin(
    noisy: stim.Circuit,
    options: dict[str, Any],
    report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """BZ certification with no limit on the fault weight: m* or "all", with its counterexample.

    ``options`` are the circuit's certify options, whose ``max_weight`` is ignored.  ``report``, if
    given, receives the record of each completed fault weight (``instrument.py``).
    """
    instrumentation = instrument.Instrumentation(progress=report, clock=runner.elapsed)
    with instrumentation.install():
        certificate = qldpc_circuits.certify_bounded_fault_propagation(
            noisy, method="brouwer_zimmermann", **(options | {"max_weight": None})
        )
    example = certificate.counterexample
    fields: dict[str, Any] = {
        "method": "brouwer_zimmermann",
        "margin": "all" if example is None else len(example.fault_indices),
        "per_weight": instrumentation.weights,
    }
    if example is not None:
        fields |= describe_certificate(certificate)
    return fields


def cutoff_reached(result: dict[str, Any], flag_rank: int) -> bool:
    """Whether a margin of "all" from ``find_margin`` completed the cutoff r + 1.

    r is ``flag_rank``, the flag rank of the circuit.  "All" can also come earlier, when no flagged
    mechanism is heavy; a violation never reaches it.  The weights from 2 up are the records of
    ``per_weight``; weight 1 is complete if "all".
    """
    searched = max((record["m"] for record in result["per_weight"]), default=1)
    return result["margin"] == "all" and searched == flag_rank + 1


def describe_certificate(certificate: Any) -> dict[str, Any]:
    """The untimed description of a certificate, with the exact residual of a counterexample.

    The description of a counterexample also gives its sector (``sector_fields``).
    """
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
    return fields | sector_fields(certificate)


def sector_fields(certificate: Any) -> dict[str, Any]:
    """The sector of a certificate's counterexample, for CSS weight; empty otherwise.

    ``sector_weight_x`` and ``sector_weight_z`` are the residual weights of the X and Z parts of the
    output error, and ``violating_parts`` names the parts whose own residual weight exceeds the
    fault weight: "X", "Z", or "X and Z".
    """
    example = certificate.counterexample
    if example is None or not certificate.css_weight:
        return {}
    stabilizers = stabilizer_matrix(certificate.state_stabilizers)
    num_qubits = len(certificate.output_qubits)
    x_part, z_part = (part.astype(np.uint8) for part in example.output_pauli.to_numpy())
    # a Pauli (x, z) anticommutes with a stabilizer (s_x, s_z) if s_x.z + s_z.x is odd
    syndromes = np.array(
        [stabilizers[:, num_qubits:] @ x_part % 2, stabilizers[:, :num_qubits] @ z_part % 2],
        dtype=np.uint8,
    )

    # within a CSS stabilizer group, the CSS weight of a pure X or Z error is the residual weight
    # of that part alone; with a cap of the number of qubits, the weights are exact
    oracle = CosetLeaderSearch(stabilizers, css_weight=True, method="brouwer_zimmermann")
    weight_x, weight_z = (
        int(w) for w in oracle.get_min_weights(oracle.pack_syndromes(syndromes), num_qubits)
    )
    fault_weight = len(example.fault_indices)
    parts = [name for name, weight in (("X", weight_x), ("Z", weight_z)) if weight > fault_weight]
    return {
        "sector_weight_x": weight_x,
        "sector_weight_z": weight_z,
        "violating_parts": " and ".join(parts) or None,
    }


####################################################################################################
# deletion of verification measurements


def prune(
    circuit: stim.Circuit,
    entry: dict[str, Any],
    order: str,
    seed: int,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Delete measured qubits from a circuit in the given order, keeping certified deletions.

    Args:
        circuit: The noiseless certification input of the circuit, the parent.
        entry: The circuit's record in ``circuits.json``.
        order: ``"last_measured_first"`` or ``"first_measured_first"``.
        seed: The seed of the acceptance estimates of parent and final circuit.
        report: If given, receives the record of each step, and then that of each fault weight
            that the search of the final m* completes (``instrument.py``).  The parent's verdict,
            before the first step, and the final checks, before the final m*, go to the runner
            directly as partial results, without the delay of a step.
    """
    report = report or (lambda record: None)
    options = certify_options(entry) | {
        "output_qubits": entry["output_qubits"],
        "max_weight": entry["t"],
    }
    noise = entry["options"]["noise"]
    parent = _certify(circuit, noise, options, "brouwer_zimmermann")
    fields: dict[str, Any] = {"order": order}
    example = parent.counterexample
    verdict = {
        "parent_certified": example is None,
        "parent_violation_weight": None if example is None else len(example.fault_indices),
    }
    runner.send_partial(verdict)
    fields |= verdict
    if example is not None:
        return fields

    # one pass over the measured qubits of the parent, each tried once
    current, steps = circuit, []
    for step, qubit in enumerate(measured_qubits(circuit, order)):
        child, _, _ = circuits._delete_qubit(current, qubit)
        record = {"step": step, "qubit": qubit} | _check_deletion(child, entry, noise, options)
        report(record)
        steps.append(record)
        if record["outcome"] == "accepted":
            current = child

    # the checks of the final circuit, sent with the circuit before its m*, whose search is
    # unbounded and reports each completed fault weight
    text = circuit_text(current)
    final_cc = _certify(current, noise, options, "connected_cluster")
    shots = int(CONFIG["prune"]["acceptance_shots"])
    checks = {
        "parent_counts": count_gates(circuit),
        "final_counts": count_gates(current),
        "final_certified_cc": final_cc.certified,
        "acceptance": {
            "p": noise["p"],
            "parent": estimate_acceptance(parent, shots, seed),
            "final": estimate_acceptance(final_cc, shots, seed),
        },
        "final_circuit": text,
        "final_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    runner.send_partial({"final_checks": checks})
    final_margin = find_margin(noisy_circuit(current, noise), options, report)
    final_margin.pop("per_weight")
    final_margin.pop("certificate_text", None)
    return fields | checks | {"steps": steps, "final_margin": final_margin}


def measured_qubits(circuit: stim.Circuit, order: str) -> list[int]:
    """The measured qubits of a circuit in a deletion order.

    ``"last_measured_first"`` orders them by decreasing index of the instruction of their last
    measurement, and ``"first_measured_first"`` by increasing index of the instruction of their
    first measurement.  Qubits tied by one instruction keep the order of their first measurement.
    """
    if order not in ("last_measured_first", "first_measured_first"):
        raise ValueError(f"Unknown deletion order {order!r}")
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for index, instruction in enumerate(circuit.flattened()):
        if stim.gate_data(instruction.name).produces_measurements:
            for target in instruction.targets_copy():
                first.setdefault(target.value, index)
                last[target.value] = index
    if order == "first_measured_first":
        return list(first)
    return sorted(first, key=lambda qubit: -last[qubit])


def count_gates(circuit: stim.Circuit) -> dict[str, int]:
    """Measured qubits, measurements, and two-qubit gates split into encoder and verification.

    Encoder gates act on no measured qubit, and verification gates on at least one.
    """
    measured = set(measured_qubits(circuit, "first_measured_first"))
    encoder = verification = measurements = 0
    for instruction in circuit.flattened():
        data = stim.gate_data(instruction.name)
        if data.produces_measurements:
            measurements += len(instruction.targets_copy())
        if data.is_two_qubit_gate:
            for group in instruction.target_groups():
                if any(target.value in measured for target in group):
                    verification += 1
                else:
                    encoder += 1
    return {
        "measured_qubits": len(measured),
        "measurements": measurements,
        "encoder_gates": encoder,
        "verification_gates": verification,
    }


def estimate_acceptance(certificate: Any, shots: int, seed: int) -> dict[str, Any]:
    """The acceptance probability of a certified circuit, from Stim shots of its noisy circuit.

    A shot is accepted if none of the certificate's post-selection detectors fires.  The standard
    error is binomial.
    """
    sampler = certificate.diagnostic_circuit.compile_detector_sampler(seed=seed)
    selected = np.array(certificate.post_selection_detectors, dtype=int)
    accepted, remaining = 0, shots
    while remaining:
        batch = min(remaining, _SHOTS_PER_BATCH)
        detectors = sampler.sample(batch)
        accepted += int(np.count_nonzero(~detectors[:, selected].any(axis=1)))
        remaining -= batch
    probability = accepted / shots
    return {
        "probability": probability,
        "standard_error": math.sqrt(probability * (1 - probability) / shots),
        "shots": shots,
        "seed": seed,
    }


def circuit_text(circuit: stim.Circuit) -> str:
    """The text of a circuit as written to a file, from which its SHA-256 is computed."""
    return str(circuit) + "\n"


def _check_deletion(
    child: stim.Circuit, entry: dict[str, Any], noise: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any]:
    """The outcome of one deletion: accepted, or the reason for its rejection.

    The reason is the first check that fails: of the prepared state, the detectors, then the bound.
    The state is checked from the certificate's stabilizers, so that they are not found twice.
    They equal the child's own: the certifier finds them with any flag detectors added, and accepts
    only deterministic detectors, which add no stabilizers.  If the certifier raises a ValueError,
    the state is checked from the child.
    """
    try:
        certificate = _certify(child, noise, options, "brouwer_zimmermann")
    except ValueError as error:
        stabilizers = qldpc_circuits.get_state_stabilizers(child, entry["output_qubits"])
        if not _prepares_target(stabilizers, entry):
            return {"outcome": "target changed"}
        return {"outcome": "detectors rejected", "reason": str(error)[:300]}
    if not _prepares_target(certificate.state_stabilizers, entry):
        return {"outcome": "target changed"}
    example = certificate.counterexample
    if example is None:
        return {"outcome": "accepted"}
    return {"outcome": "violation", "violation_weight": len(example.fault_indices)}


def _prepares_target(stabilizers: Any, entry: dict[str, Any]) -> bool:
    """Whether stabilizers of a prepared state give the parent's signed state, ``entry``'s."""
    texts = [str(stabilizer) for stabilizer in stabilizers]
    return len(texts) == len(entry["output_qubits"]) and same_state(entry["stabilizers"], texts)


def _certify(
    circuit: stim.Circuit, noise: dict[str, Any], options: dict[str, Any], method: Any
) -> Any:
    """Certify a noiseless circuit with the given noise, options, and method."""
    noisy = noisy_circuit(circuit, noise)
    return qldpc_circuits.certify_bounded_fault_propagation(noisy, method=method, **options)
