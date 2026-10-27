# SPDX-License-Identifier: Apache-2.0

"""Tests for state-preparation diagnostics and Sinter task construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import sinter
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_state_prep_benchmarks() -> None:
    """State preparation circuit benchmarks."""

    # construct a state prep circuit for the Steane code
    code = codes.SteaneCode()
    circuit = stim.Circuit("""
        # non-fault-tolerant |0> state prep
        H 0 2 4
        CX 0 3 2 1 4 5
        CX 0 1 2 6 4 3
        CX 2 5 3 6
        # flag a logical Z representative
        H 7
        CZ 7 1 7 3 7 5
        MX 7
    """)

    noise_model_family = circuits.DepolarizingNoiseModel
    error_rates = np.logspace(-3, -1, 5)

    # observables can be specified by either a symplectic matrix or list of Pauli strings
    observables = code.get_logical_ops(Pauli.Z, symplectic=True)
    string_observables = [math.op_to_string(obs) for obs in observables]

    # build sinter tasks
    tasks = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates,
        noise_model_family,
        observables=string_observables,
    )
    for error_rate, task in zip(error_rates, tasks):
        assert task.json_metadata["p"] == error_rate
    float32_rates = np.array([0.01], dtype=np.float32)
    assert circuits.get_state_prep_diagnostic_tasks(code, circuit, float32_rates)[0].circuit
    assert (
        circuits.get_state_prep_diagnostic_tasks(code, circuit, [0.123], metadata={"p": 999})[
            0
        ].json_metadata["p"]
        == 0.123
    )

    # find observables automatically, and add flags
    task = circuits.get_state_prep_diagnostic_tasks(
        code, circuit, error_rates[:1], noise_model_family, add_flags=True
    )[0]
    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(
        code, circuit, add_flags=True
    )
    assert task.circuit == noise_model_family(float(error_rates[0])).noisy_circuit(
        diagnostic_circuit
    )


def test_state_prep_post_selection() -> None:
    """Post-select on prep and flag detectors and observables, but never on final stabilizers."""
    # a prep detector (index 0), and one unaddressed measurement for a flag (index 1)
    code = codes.SteaneCode()
    circuit = circuits.get_encoding_circuit(code, only_zero=True)
    circuit += stim.Circuit("R 7 8\nM 7 8\nDETECTOR rec[-1]")
    num_detectors = 2 + len(code.get_stabilizer_ops())

    def get_task(**kwargs: Any) -> sinter.Task:
        return circuits.get_state_prep_diagnostic_tasks(code, circuit, [0.01], **kwargs)[0]

    task = get_task(add_flags=True)
    assert task.postselection_mask is None and task.postselected_observables_mask is None
    assert np.array_equal(_selected(get_task(post_select=True), num_detectors - 1), [1] + [0] * 6)
    assert np.array_equal(
        _selected(get_task(add_flags=True, post_select=True), num_detectors), [1, 1] + [0] * 6
    )
    assert np.array_equal(
        _selected(get_task(add_flags=True, post_select="flags"), num_detectors), [0, 1] + [0] * 6
    )
    assert np.array_equal(
        _selected(get_task(add_flags=True, post_select=[-2]), num_detectors), [1, 0] + [0] * 6
    )
    # repeated and aliased indices select a detector once
    assert np.array_equal(
        _selected(get_task(add_flags=True, post_select=[1, -2, 0]), num_detectors),
        [1, 1] + [0] * 6,
    )
    with pytest.raises(ValueError, match="selectable prep/flag detectors"):
        get_task(add_flags=True, post_select=[2])
    with pytest.raises(ValueError, match="add_flags=True"):
        get_task(post_select="flags")
    with pytest.raises(ValueError, match="post_select"):
        get_task(add_flags=True, post_select="all")
    for post_select in (0, [0.5], [True]):
        with pytest.raises(TypeError, match="post_select"):
            get_task(post_select=post_select)

    task = get_task(post_select_observables=True)
    assert task.postselected_observables_mask is not None
    selected = np.unpackbits(task.postselected_observables_mask, bitorder="little")
    assert np.array_equal(selected[: task.circuit.num_observables], [1] * code.dimension)


def test_state_prep_benchmark_remaps_data_qubits() -> None:
    """Diagnostic validation and measurements honor a non-identity data-qubit frame."""
    code = codes.SteaneCode()
    qubit_ids = circuits.QubitIDs(
        tuple(range(3, 3 + len(code))),
        tuple(range(20, 20 + code.num_checks)),
    )
    prep = circuits.with_remapped_qubits(
        circuits.get_encoding_circuit(code, only_zero=True), qubit_ids.data
    )
    diagnostic, _ = circuits.get_state_prep_diagnostic_circuit(code, prep, qubit_ids=qubit_ids)
    assert diagnostic.num_observables > 0
    assert "Z4" in str(diagnostic)
    assert "Z0" not in str(diagnostic)


def test_pure_logical_state(pytestconfig: pytest.Config) -> None:
    """Identify invalid pure logical states."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))
    encoder = circuits.get_encoding_circuit(code)

    with pytest.raises(ValueError, match="does not prepare"):
        circuits.get_state_prep_diagnostic_circuit(
            code, stim.Circuit(f"X {len(code) - 1}") + encoder
        )

    with pytest.raises(ValueError, match="pure"):
        circuits.get_state_prep_diagnostic_circuit(
            code, stim.Circuit(f"H 0\nCX 0 {len(code)}") + encoder
        )

    # data qubits that the circuit never touches are in |0>
    zero_code = codes.QuditCode([[0, 0, 1, 0], [0, 0, 0, 1]])
    assert circuits.get_state_prep_diagnostic_circuit(zero_code, stim.Circuit("X 0\nX 0"))


def _selected(task: sinter.Task, num_detectors: int) -> npt.NDArray[np.uint8]:
    """The post-selection bits of the first detectors of a Sinter task."""
    assert task.postselection_mask is not None
    return np.unpackbits(task.postselection_mask, bitorder="little")[:num_detectors]
