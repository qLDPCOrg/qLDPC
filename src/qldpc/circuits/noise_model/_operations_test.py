# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _operations.py."""

import pytest
import stim

from qldpc.circuits.noise_model import _operations


def test_op_type() -> None:
    """Stim instructions are classified by how noise applies to them."""
    assert _operations.op_type("H") == _operations.CLIFFORD_1Q
    assert _operations.op_type("CNOT") == _operations.CLIFFORD_2Q
    assert _operations.op_type("SPP") == _operations.CLIFFORD_PP
    assert _operations.op_type("MX") == _operations.JUST_MEASURE_1Q
    assert _operations.op_type("MXX") == _operations.JUST_MEASURE_2Q
    assert _operations.op_type("MPP") == _operations.JUST_MEASURE_PP
    assert _operations.op_type("RY") == _operations.JUST_RESET_1Q
    assert _operations.op_type("MR") == _operations.MEASURE_RESET_1Q
    assert _operations.op_type("DETECTOR") == _operations.ANNOTATION
    assert _operations.op_type("X_ERROR") == _operations.NOISE
    assert _operations.op_type("NOT_A_GATE") == _operations.UNSUPPORTED
    assert {"M", "R", "MR"} <= _operations.COLLAPSING_OPS and "H" not in _operations.COLLAPSING_OPS
    assert {"M", "MZ", "MPP", "MXX"} <= _operations.JUST_MEASURE_OPS


def test_involves_classical_bits() -> None:
    """Classically controlled instructions are identified by their targets."""
    assert _operations.involves_classical_bits(stim.Circuit("M 0\nCX rec[-1] 1")[1])
    assert _operations.involves_classical_bits(stim.Circuit("CX sweep[0] 1")[0])
    assert not _operations.involves_classical_bits(stim.Circuit("CX 0 1")[0])


def test_gate_aliases() -> None:
    """Instructions can be matched by any of their names, and Pauli products by their bases."""
    assert {"CX", "CNOT", "ZCX"} <= set(
        _operations.get_gate_aliases(stim.CircuitInstruction("CX", [0, 1]))
    )
    for instruction, name in [
        ("MPP X0*Y1*Z2", "MXYZ"),
        ("MPP Z0", "M"),
        ("SPP X0*Y1", "SXY"),
        ("SPP_DAG X0*Y1", "SXY_DAG"),
    ]:
        aliases = _operations.get_gate_aliases(stim.Circuit(instruction)[0])
        assert aliases[0] == name
        assert instruction.split()[0] in aliases
    with pytest.raises(ValueError, match="single Pauli product"):
        _operations.get_gate_aliases(stim.Circuit("MPP X0 Z1")[0])


def test_canonical_rule_key() -> None:
    """Rule keys are canonicalized and validated."""
    assert (
        _operations.canonical_rule_key("cnot")
        == _operations.canonical_rule_key("ZCX")
        in ("CX", "CNOT")
    )
    assert _operations.canonical_rule_key("mz") == "M"
    assert _operations.canonical_rule_key("mxyz") == "MXYZ"
    assert _operations.canonical_rule_key("SXY_DAG") == "SXY_DAG"
    with pytest.raises(TypeError, match="must be strings"):
        _operations.canonical_rule_key(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unrecognized"):
        _operations.canonical_rule_key("NOT_A_GATE")
    with pytest.raises(ValueError, match="no _DAG form"):
        _operations.canonical_rule_key("MXY_DAG")
    with pytest.raises(ValueError, match="cannot target"):
        _operations.canonical_rule_key("DETECTOR")


def test_known_gate_shape() -> None:
    """Rule keys with a fixed shape report their arity and whether they measure or reset."""
    assert _operations.known_gate_shape("H") == (1, False, False)
    assert _operations.known_gate_shape("MR") == (1, True, True)
    assert _operations.known_gate_shape("CX") == (2, False, False)
    assert _operations.known_gate_shape("MXX") == (2, True, False)
    assert _operations.known_gate_shape("MXYZ") == (3, True, False)
    assert _operations.known_gate_shape("SXY_DAG") == (2, False, False)
    assert _operations.known_gate_shape("MPP") is None
