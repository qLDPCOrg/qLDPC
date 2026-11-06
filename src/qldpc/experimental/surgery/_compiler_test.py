# SPDX-License-Identifier: Apache-2.0

"""Tests for the private lattice-surgery compiler slice."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from ._compiler import (
    _compile_interblock_pair_measurement,
    _CompiledPairMeasurement,
    _LogicalBlock,
    _LogicalPauliRef,
    _PauliPairMeasurement,
    _SurgeryBackendUnsupportedError,
)


def _pair_request(
    basis: Pauli = Pauli.X,
    *,
    left_index: int = 0,
    right_index: int = 0,
) -> _PauliPairMeasurement:
    """Build the standard two-block compiler request."""
    return _PauliPairMeasurement(
        "parity",
        (
            _LogicalPauliRef("left", left_index, basis),
            _LogicalPauliRef("right", right_index, basis),
        ),
    )


def _raw_observable_samples(circuit: stim.Circuit, *, shots: int) -> dict[int, np.ndarray]:
    """Reconstruct raw observable bits from a circuit's measurement records."""
    flattened = circuit.flattened()
    measurements = flattened.compile_sampler().sample(shots=shots).astype(np.uint8)
    observable_samples: dict[int, np.ndarray] = {}
    for instruction in flattened:
        if instruction.name != "OBSERVABLE_INCLUDE":
            continue
        observable_index = int(instruction.gate_args_copy()[0])
        columns = [measurements.shape[1] + target.value for target in instruction.targets_copy()]
        observable_samples[observable_index] = np.bitwise_xor.reduce(
            measurements[:, columns],
            axis=1,
        )
    return observable_samples


@pytest.mark.parametrize(
    ("basis", "states", "expected"),
    [
        (Pauli.X, ("+", "+"), 0),
        (Pauli.X, ("-", "+"), 1),
        (Pauli.Z, ("0", "0"), 0),
        (Pauli.Z, ("0", "1"), 1),
    ],
)
def test_compile_interblock_pair_steane_truth_table(
    basis: PauliXZ,
    states: tuple[str, str],
    expected: int,
) -> None:
    """Raw measurement records reproduce both logical parity truth tables."""
    code = codes.SteaneCode()
    compiled = _compile_interblock_pair_measurement(
        (_LogicalBlock("left", code), _LogicalBlock("right", code)),
        _pair_request(basis),
        rounds=2,
        initial_states={"left": states[0], "right": states[1]},
    )

    raw_observables = _raw_observable_samples(compiled.diagnostic_circuit, shots=8)
    assert set(raw_observables) == {0, 1}
    assert np.all(raw_observables[0] == expected)
    assert np.array_equal(raw_observables[0], raw_observables[1])
    assert not compiled.diagnostic_circuit.compile_detector_sampler().sample(shots=8).any()

    logical = tuple(int(value) for value in code.get_logical_ops(basis)[0].tolist())
    assert compiled.resolved_operators == (logical, logical)
    assert compiled.resources.syndrome_rounds == 2


def test_compile_interblock_pair_matches_existing_public_builder() -> None:
    """Explicit block semantics preserve the existing distinct-code circuit.

    The compiled record also names its request, observables, backend, block allocation and
    resources consistently.
    """
    from qldpc.experimental.surgery import build_bridge, build_gadget, build_joint_ppm_circuit

    shared_code = codes.SteaneCode()
    request = _pair_request(Pauli.X)
    compiled = _compile_interblock_pair_measurement(
        (_LogicalBlock("left", shared_code), _LogicalBlock("right", shared_code)),
        request,
        rounds=1,
    )

    assert isinstance(compiled, _CompiledPairMeasurement)
    assert compiled.request is request
    assert compiled.result_observables == (("parity", 0),)
    assert compiled.diagnostic_observable_indices == (1,)
    assert compiled.circuit.num_observables == 1
    assert compiled.diagnostic_circuit.num_observables == 2
    assert compiled.backend.name == "webster-universal-adapter"
    assert compiled.backend.cellulate_max_len is None
    assert not compiled.backend.boosted

    allocation = dict(compiled.block_data_qubits)
    assert allocation["left"] == tuple(range(7))
    assert allocation["right"] == tuple(range(7, 14))

    resources = compiled.resources
    assert resources.source_data_qubits == 14
    assert resources.logical_outcomes == 1
    assert resources.syndrome_rounds == 1
    assert resources.adapter_data_qubits == compiled.backend.bridge_width
    assert (
        resources.source_data_qubits
        + resources.gadget_ancilla_qubits
        + resources.adapter_data_qubits
        == compiled.merged_code.num_qudits
    )
    assert (
        compiled.merged_code.num_qudits + resources.syndrome_ancilla_qubits
        == resources.allocated_physical_qubits
    )

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    logical_l = np.asarray(code_l.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    logical_r = np.asarray(code_r.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget_l = build_gadget(code_l, logical_l, basis=Pauli.X)
    gadget_r = build_gadget(code_r, logical_r, basis=Pauli.X)
    expected, expected_code = build_joint_ppm_circuit(
        gadget_l,
        gadget_r,
        build_bridge(gadget_l, gadget_r),
        rounds=1,
    )

    assert compiled.diagnostic_circuit == expected
    assert np.array_equal(compiled.merged_code.matrix_x, expected_code.matrix_x)
    assert np.array_equal(compiled.merged_code.matrix_z, expected_code.matrix_z)


@pytest.mark.parametrize(
    ("basis", "states"),
    [(Pauli.X, ("-", "+")), (Pauli.Z, ("0", "1"))],
)
def test_compile_interblock_pair_hgp_initialization_binds_logical_index(
    basis: PauliXZ,
    states: tuple[str, str],
) -> None:
    """Initial states flip the requested nonzero logicals of a multi-logical HGP code.

    The X case flips the left block and the Z case the right one, so together they bind both
    requested logical indices; either flip gives odd raw parity.
    """
    code = codes.HGPCode(codes.HammingCode(3))
    compiled = _compile_interblock_pair_measurement(
        (_LogicalBlock("left", code), _LogicalBlock("right", code)),
        _pair_request(basis, left_index=3, right_index=5),
        rounds=1,
        initial_states={"left": states[0], "right": states[1]},
    )

    assert code.dimension == 16
    assert compiled.merged_code.dimension == 31
    assert compiled.resources.source_data_qubits == 116
    assert not compiled.circuit.compile_detector_sampler().sample(shots=2).any()
    logicals = code.get_logical_ops(basis)
    expected_left = tuple(int(value) for value in logicals[3].tolist())
    expected_right = tuple(int(value) for value in logicals[5].tolist())
    zero_index = tuple(int(value) for value in logicals[0].tolist())
    assert compiled.resolved_operators == (expected_left, expected_right)
    assert expected_left != zero_index
    assert expected_right != zero_index

    raw_observables = _raw_observable_samples(compiled.diagnostic_circuit, shots=8)
    assert np.all(raw_observables[0] == 1)
    assert np.array_equal(raw_observables[0], raw_observables[1])


def test_compiler_rejects_invalid_rounds_and_block_collections() -> None:
    """Round and block-registry errors are rejected before construction."""
    code = codes.SteaneCode()
    blocks = (_LogicalBlock("left", code), _LogicalBlock("right", code))
    request = _pair_request()

    with pytest.raises(ValueError, match="rounds must be >= 1"):
        _compile_interblock_pair_measurement(blocks, request, rounds=0)
    with pytest.raises(ValueError, match="exactly 2 blocks"):
        _compile_interblock_pair_measurement(blocks[:1], request, rounds=1)


def test_compiler_rejects_invalid_logical_blocks() -> None:
    """Block identifiers and CSS-code requirements fail at the compiler boundary."""
    code = codes.SteaneCode()
    request = _pair_request()

    with pytest.raises(ValueError, match="must not be empty"):
        _compile_interblock_pair_measurement(
            (_LogicalBlock(" ", code), _LogicalBlock("right", code)),
            request,
            rounds=1,
        )
    with pytest.raises(ValueError, match="duplicate logical block"):
        _compile_interblock_pair_measurement(
            (_LogicalBlock("left", code), _LogicalBlock("left", code)),
            request,
            rounds=1,
        )

    nonbinary = codes.CSSCode([[1, 0]], [[0, 1]], field=3, is_subsystem_code=False)
    with pytest.raises(ValueError, match="binary CSS code"):
        _compile_interblock_pair_measurement(
            (_LogicalBlock("left", nonbinary), _LogicalBlock("right", code)),
            request,
            rounds=1,
        )

    subsystem = codes.CSSCode([[1, 0]], [[1, 0]], is_subsystem_code=True)
    with pytest.raises(ValueError, match="not a subsystem code"):
        _compile_interblock_pair_measurement(
            (_LogicalBlock("left", subsystem), _LogicalBlock("right", code)),
            request,
            rounds=1,
        )


def test_compiler_rejects_invalid_pair_requests() -> None:
    """Result-key, block, basis, and index errors in a request are explicit."""
    code = codes.SteaneCode()
    blocks = (_LogicalBlock("left", code), _LogicalBlock("right", code))

    with pytest.raises(ValueError, match="result_key must not be empty"):
        _compile_interblock_pair_measurement(
            blocks,
            _PauliPairMeasurement(" ", _pair_request().factors),
            rounds=1,
        )
    with pytest.raises(ValueError, match="unknown logical block"):
        _compile_interblock_pair_measurement(
            blocks,
            _PauliPairMeasurement(
                "parity",
                (
                    _LogicalPauliRef("missing", 0, Pauli.X),
                    _LogicalPauliRef("right", 0, Pauli.X),
                ),
            ),
            rounds=1,
        )
    with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
        _compile_interblock_pair_measurement(
            blocks,
            _pair_request(Pauli.Y),
            rounds=1,
        )
    with pytest.raises(IndexError, match="out of range"):
        _compile_interblock_pair_measurement(
            blocks,
            _pair_request(Pauli.X, left_index=1),
            rounds=1,
        )
    with pytest.raises(ValueError, match="two distinct block"):
        _compile_interblock_pair_measurement(
            blocks,
            _PauliPairMeasurement(
                "parity",
                (
                    _LogicalPauliRef("left", 0, Pauli.X),
                    _LogicalPauliRef("left", 0, Pauli.X),
                ),
            ),
            rounds=1,
        )
    with pytest.raises(ValueError, match="same-basis factors"):
        _compile_interblock_pair_measurement(
            blocks,
            _PauliPairMeasurement(
                "parity",
                (
                    _LogicalPauliRef("left", 0, Pauli.X),
                    _LogicalPauliRef("right", 0, Pauli.Z),
                ),
            ),
            rounds=1,
        )


def test_compiler_rejects_invalid_initial_states() -> None:
    """Logical initialization mappings are exact and use state labels."""
    code = codes.SteaneCode()
    blocks = (_LogicalBlock("left", code), _LogicalBlock("right", code))
    request = _pair_request()

    with pytest.raises(ValueError, match=r"missing=.*right"):
        _compile_interblock_pair_measurement(
            blocks,
            request,
            rounds=1,
            initial_states={"left": "+"},
        )
    with pytest.raises(ValueError, match=r"extra=.*third"):
        _compile_interblock_pair_measurement(
            blocks,
            request,
            rounds=1,
            initial_states={"left": "+", "right": "+", "third": "+"},
        )
    with pytest.raises(ValueError, match="must be one of"):
        _compile_interblock_pair_measurement(
            blocks,
            request,
            rounds=1,
            initial_states={"left": "x", "right": "+"},
        )


def test_compiler_reports_backend_capability_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid requests distinguish fallback limitations from malformed input."""
    import qldpc.experimental.surgery._compiler as compiler

    one_qubit = codes.CSSCode(
        np.zeros((0, 1), dtype=int),
        np.zeros((0, 1), dtype=int),
        is_subsystem_code=False,
    )
    with pytest.raises(_SurgeryBackendUnsupportedError, match="support weight >= 2"):
        _compile_interblock_pair_measurement(
            (_LogicalBlock("left", one_qubit), _LogicalBlock("right", one_qubit)),
            _pair_request(),
            rounds=1,
        )

    code = codes.SteaneCode()

    def reject_bridge(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("synthetic backend rejection")

    monkeypatch.setattr(compiler, "build_bridge", reject_bridge)
    with pytest.raises(_SurgeryBackendUnsupportedError, match="cannot realize") as caught:
        _compile_interblock_pair_measurement(
            (_LogicalBlock("left", code), _LogicalBlock("right", code)),
            _pair_request(),
            rounds=1,
        )
    assert isinstance(caught.value.__cause__, ValueError)
