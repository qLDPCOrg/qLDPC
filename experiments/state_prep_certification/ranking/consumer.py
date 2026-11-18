# SPDX-License-Identifier: Apache-2.0

"""The consumer of each group: one noisy round of syndrome measurement, then a Z-basis readout.

The consumer is qLDPC's Z-basis memory experiment with one round, built from the group's code file
(``Hx``, ``Hz``, ``Lx``, and ``Lz`` in output-qubit order), with the resets of the data qubits
removed.  A noiseless block preparation, Stim's circuit for the group's signed state, puts data
qubit i in the state of output qubit i, and the observables are the code's logical Z operators
``Lz``.  The decoder is BP-OSD, built by ``Consumer`` from the detector error model at each rate.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import galois
import numpy as np
import stim

from experiments.state_prep_certification.common import store
from qldpc import circuits, codes, decoders
from qldpc.experimental.circuits.benchmarking import Consumer
from qldpc.objects import Pauli

CERTIFICATION = pathlib.Path(__file__).resolve().parents[1] / "certification"


def code_matrices(entry: dict[str, Any]) -> dict[str, Any]:
    """The code file of a circuit of ``circuits.json``, checked against its ``code_sha256``."""
    path = CERTIFICATION / entry["code_file"]
    if store.sha256_file(path) != entry["code_sha256"]:
        raise ValueError(f"{entry['id']}: {entry['code_file']} does not match code_sha256")
    code: dict[str, Any] = json.loads(path.read_text())
    return code


def consumer_circuits(
    entry: dict[str, Any], p: float
) -> tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
    """The block preparation, the noiseless consumer, and the consumer with noise at rate p.

    The logical X operators are changed by a basis change within their span so that
    ``Lx Lz^T = I``, which qLDPC requires; the observables are the rows of ``Lz`` as given.
    """
    matrices = code_matrices(entry)
    hx, hz = np.array(matrices["Hx"]), np.array(matrices["Hz"])
    lx, lz = galois.GF2(np.array(matrices["Lx"])), galois.GF2(np.array(matrices["Lz"]))
    code = codes.CSSCode(hx, hz)
    code.set_logical_ops_xz(np.linalg.inv(lx @ lz.T) @ lx, lz)
    num_qubits = len(code)

    # data qubit i is the qubit with coordinates (0, i), which qLDPC numbers i
    memory = circuits.get_memory_experiment(code, basis=Pauli.Z, num_rounds=1)
    coordinates = memory.get_final_qubit_coordinates()
    if any(coordinates[qubit] != [0, qubit] for qubit in range(num_qubits)):
        raise ValueError("The data qubits of the memory experiment are not qubits 0, ..., n - 1")
    noiseless = stim.Circuit()
    for instruction in memory:
        if instruction.name == "R":
            targets = [t.value for t in instruction.targets_copy() if t.value >= num_qubits]
            if targets:
                noiseless.append("R", targets)
        else:
            noiseless.append(instruction)

    tableau = stim.Tableau.from_stabilizers([stim.PauliString(s) for s in entry["stabilizers"]])
    block_preparation = tableau.to_circuit()
    noisy = circuits.DepolarizingNoiseModel(p).noisy_circuit(noiseless)
    return block_preparation, noiseless, noisy


def build_consumer(entry: dict[str, Any], p: float, decoder: dict[str, Any]) -> Consumer:
    """The consumer of a circuit's group at rate p, with BP-OSD of the given options."""
    block_preparation, _, noisy = consumer_circuits(entry, p)
    return Consumer(
        block_preparation,
        noisy,
        decoders.bp_osd(**decoder),
        block_qubits=range(len(entry["output_qubits"])),
        correctable_weight=int(entry["t"]),
        description=f"Z-basis memory, one round, without data resets; code {entry['code_file']}",
    )


def sha256_text(text: str) -> str:
    """The SHA-256 of a text, in hexadecimal."""
    return hashlib.sha256(text.encode()).hexdigest()
