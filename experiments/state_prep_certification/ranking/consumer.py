# SPDX-License-Identifier: Apache-2.0

"""The consumer of each group: qLDPC's Steane-swap consumer of the group's code.

The consumer is qLDPC's ``get_steane_swap_consumer`` of the group's code, built from its code file
(``Hx``, ``Hz``, ``Lx``, and ``Lz`` in output-qubit order), with the correctable weight t of the
group's circuits.  The block under test is a fresh logical ``|0>`` block (``basis=Pauli.Z``), and
the noise model at each rate p is ``DepolarizingNoiseModel(p)``.  The consumer's decoder is qLDPC's
default, which decodes the syndrome of each step on its own with a table of lightest errors.  Both X
and Z errors on the block reach its detectors.  ``build_consumer`` checks that the ideal state of
the block is the group's signed state, with the block qubits in output-qubit order.
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
from qldpc import circuits, codes
from qldpc.experimental.circuits.benchmarking import Consumer, get_steane_swap_consumer
from qldpc.objects import Pauli

CERTIFICATION = pathlib.Path(__file__).resolve().parents[1] / "certification"


def code_matrices(entry: dict[str, Any]) -> dict[str, Any]:
    """The code file of a circuit of ``circuits.json``, checked against its ``code_sha256``."""
    path = CERTIFICATION / entry["code_file"]
    if store.sha256_file(path) != entry["code_sha256"]:
        raise ValueError(f"{entry['id']}: {entry['code_file']} does not match code_sha256")
    code: dict[str, Any] = json.loads(path.read_text())
    return code


def group_code(entry: dict[str, Any]) -> codes.CSSCode:
    """The code of a circuit's group, with the logical operators of its code file.

    The logical X operators are changed by a basis change within their span so that
    ``Lx Lz^T = I``, which qLDPC requires; the logical Z operators are the rows of ``Lz`` as given.
    """
    matrices = code_matrices(entry)
    lx, lz = galois.GF2(np.array(matrices["Lx"])), galois.GF2(np.array(matrices["Lz"]))
    code = codes.CSSCode(np.array(matrices["Hx"]), np.array(matrices["Hz"]))
    code.set_logical_ops_xz(np.linalg.inv(lx @ lz.T) @ lx, lz)
    return code


def build_consumer(entry: dict[str, Any], p: float) -> Consumer:
    """The consumer of a circuit's group at rate p.

    Raises:
        ValueError: If the ideal state of the consumer's block is not the group's signed state.
    """
    consumer = get_steane_swap_consumer(
        group_code(entry),
        circuits.DepolarizingNoiseModel(p),
        basis=Pauli.Z,
        correctable_weight=int(entry["t"]),
    )
    simulator = stim.TableauSimulator()
    simulator.do_circuit(consumer.block_preparation)
    block_qubits = list(consumer.block_qubits or ())
    for stabilizer in entry["stabilizers"]:
        operator = stim.PauliString(simulator.num_qubits)
        for qubit, pauli in zip(block_qubits, stim.PauliString(stabilizer)):
            operator[qubit] = pauli
        operator.sign = stim.PauliString(stabilizer).sign
        if simulator.peek_observable_expectation(operator) != 1:
            raise ValueError(f"{entry['id']}: the consumer's block is not in the group's state")
    return consumer


def sha256_text(text: str) -> str:
    """The SHA-256 of a text, in hexadecimal."""
    return hashlib.sha256(text.encode()).hexdigest()
