# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rules.py."""

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent
from qldpc.circuits.noise_model import rules


def test_noise_rule_errors() -> None:
    """Cover various NoiseRule errors."""
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(readout_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(reset_error=1.1)

    # `after` as a stim.Circuit rejects non-noise instructions and repeat blocks.
    with pytest.raises(ValueError, match="non-noise instruction"):
        circuits.NoiseRule(after=stim.Circuit("H 0"))
    fragment_with_repeat = stim.Circuit()
    fragment_with_repeat.append(stim.CircuitRepeatBlock(2, stim.Circuit("X_ERROR(0.1) 0")))
    with pytest.raises(TypeError, match="may contain only noise instructions"):
        circuits.NoiseRule(after=fragment_with_repeat)
    with pytest.raises(ValueError, match="at least one qubit"):
        circuits.NoiseRule(after=stim.Circuit())

    # An empty PauliChannel with an explicitly-declared arity is honored — a mismatch against the
    # rule's expected arity is flagged rather than silently dropped.
    with pytest.raises(ValueError, match="arity 3; expected 2"):
        circuits.NoiseModel(clifford_nq_error={2: circuits.PauliChannel({}, num_qubits=3)})
    # Same policy for idle_error: an explicitly-declared arity is validated BEFORE trivializing, so
    # a shape-wrong empty channel is flagged rather than silently accepted.
    with pytest.raises(ValueError, match=r"idle_error.*multi-qubit"):
        circuits.NoiseModel(
            idle_error=circuits.NoiseRule(after=circuits.PauliChannel({}, num_qubits=3))
        )

    # get_noise_rule requires pre-split (single Pauli product) MPP/SPP ops.
    with pytest.raises(ValueError, match="split into a single Pauli product"):
        # Two products: X0*Y1 and Z2*X3 — targets at odd indices include non-combiners (Z2, X3).
        mpp = stim.Circuit("MPP X0*Y1 Z2*X3")[0]
        rule = circuits.NoiseRule(readout_error=0.1)
        circuits.NoiseModel(rules={"MXY": rule}).get_noise_rule(mpp)

    # NoiseRule cannot combine `after`-noise with readout_error / reset_error — those should be
    # separate rules (or handled via NoiseModel-level defaults).
    with pytest.raises(ValueError, match=r"after.*readout_error"):
        circuits.NoiseRule(after={"X": 0.01}, readout_error=0.1)
    with pytest.raises(ValueError, match=r"after.*readout_error"):
        circuits.NoiseRule(after=circuits.PauliChannel({"X": 0.01}), reset_error=0.1)


def test_after_stim_circuit_form() -> None:
    """`after=stim.Circuit(...)`: verbatim-fragment emission and immunity filtering.

    Fragments emit verbatim per k-qubit block (with remapped qubits).  Immunity drops the whole
    fragment under immunize_gates=True and raises under immunize_gates=False.
    """

    # Multiple noise ops in one fragment (combining DEPOLARIZE2 + PAULI_CHANNEL_1 — not expressible
    # as a single PauliChannel — is the canonical reason to use the stim.Circuit escape hatch).
    noise_rule = circuits.NoiseRule(
        after=stim.Circuit("DEPOLARIZE2(0.2) 0 1\nPAULI_CHANNEL_1(0, 0.1, 0.1) 0 1")
    )
    assert circuits_are_equivalent(
        stim.Circuit("CX 0 1\nDEPOLARIZE2(0.2) 0 1\nPAULI_CHANNEL_1(0, 0.1, 0.1) 0 1"),
        circuits.NoiseModel(rules={"CX": noise_rule}).noisy_circuit(stim.Circuit("CX 0 1")),
    )

    circuit = stim.Circuit("SPP X0*Y1*Z2")

    # A raw CE chain touching an immune qubit is dropped under immunize_gates=True (default).
    ce_rule = circuits.NoiseRule(after=stim.Circuit("CORRELATED_ERROR(0.02) X0 Y1 Z2"))
    assert circuits_are_equivalent(
        circuit,
        circuits.NoiseModel(rules={"SPP": ce_rule}).noisy_circuit(circuit, immune_qubits=[1]),
    )
    # Under immunize_gates=False, the same rule raises on partial immunity.
    with pytest.raises(ValueError, match="partial immunity"):
        circuits.NoiseModel(rules={"SPP": ce_rule}).noisy_circuit(
            circuit, immune_qubits=[1], immunize_gates=False
        )

    # A 3q PauliChannel whose strings are non-I on a single position emits natively as
    # PAULI_CHANNEL_1 (via `PauliChannel.to_circuit`'s 1-active-position branch).
    channel = circuits.PauliChannel({"IXI": 0.1, "IYI": 0.05})
    sparse_rule = circuits.NoiseRule(after=channel)
    assert circuits_are_equivalent(
        stim.Circuit("SPP X0*Y1*Z2\nPAULI_CHANNEL_1(0.1, 0.05, 0) 1"),
        circuits.NoiseModel(rules={"SPP": sparse_rule}).noisy_circuit(circuit),
    )

    # Measurement-producing noise is rejected because it would shift later rec[] targets.
    with pytest.raises(ValueError, match="measurement-producing"):
        circuits.NoiseRule(after=stim.Circuit("HERALDED_ERASE(0.05) 0 1"))

    # A Mapping `after` whose entries all have zero probabilities normalizes to a trivial rule.
    zero_rule = circuits.NoiseRule(after={"X": 0.0, "Y": 0.0, "Z": 0.0})
    assert not bool(zero_rule)

    # A mixed 1q+2q stim.Circuit fragment must emit per-k-qubit-block, NOT native-broadcast
    # wholesale across every qubit_target (the Mapping form rejects this shape; the stim.Circuit
    # escape hatch is expected to preserve per-block semantics).
    mixed_rule = circuits.NoiseRule(after=stim.Circuit("X_ERROR(0.01) 0\nDEPOLARIZE2(0.02) 0 1"))
    circuit = stim.Circuit("CX 0 1 2 3")
    expected = circuit + stim.Circuit("""
        X_ERROR(0.01) 0
        DEPOLARIZE2(0.02) 0 1
        X_ERROR(0.01) 2
        DEPOLARIZE2(0.02) 2 3
        """)
    assert circuits_are_equivalent(
        expected,
        circuits.NoiseModel(rules={"CX": mixed_rule}).noisy_circuit(circuit),
    )


def test_noise_rule_repr() -> None:
    """NoiseRule has a repr that lists only its set fields."""
    assert repr(circuits.NoiseRule()) == "NoiseRule()"
    assert repr(circuits.NoiseRule(readout_error=0.1)) == "NoiseRule(readout_error=0.1)"
    assert (
        repr(circuits.NoiseRule(readout_error=0.1, reset_error=0.2))
        == "NoiseRule(readout_error=0.1, reset_error=0.2)"
    )
    assert (
        repr(circuits.NoiseRule(after={"XX": 0.1})) == "NoiseRule(after=PauliChannel({'XX': 0.1}))"
    )
    # The stim.Circuit `after` form defers to stim's own repr.
    circuit_rule = circuits.NoiseRule(after=stim.Circuit("X_ERROR(0.1) 0"))
    assert repr(circuit_rule).startswith("NoiseRule(after=")


def test_noisy_operation() -> None:
    """NoiseRule.noisy_operation adds readout, reset, and `after` noise to an operation."""
    op = stim.CircuitInstruction("MR", [0, 1], [0.1])
    noisy_op, after = circuits.NoiseRule(readout_error=0.2, reset_error=0.3).noisy_operation(op)
    assert noisy_op == stim.CircuitInstruction("MR", [0, 1], [0.1 + 0.2 - 2 * 0.1 * 0.2])
    assert after == stim.Circuit("X_ERROR(0.3) 0 1")
    noisy_op, after = circuits.NoiseRule(reset_error=0.3).noisy_operation(
        stim.CircuitInstruction("RX", [2])
    )
    assert noisy_op == stim.CircuitInstruction("RX", [2]) and after == stim.Circuit(
        "Z_ERROR(0.3) 2"
    )
    noisy_op, after = circuits.NoiseRule(readout_error=0.2).noisy_operation(
        stim.CircuitInstruction("M", [0])
    )
    assert noisy_op == stim.CircuitInstruction("M", [0], [0.2]) and after == stim.Circuit()

    with pytest.raises(ValueError, match="readout_error is not valid"):
        circuits.NoiseRule(readout_error=0.1).noisy_operation(stim.CircuitInstruction("H", [0]))
    with pytest.raises(ValueError, match="reset_error is not valid"):
        circuits.NoiseRule(reset_error=0.1).noisy_operation(stim.CircuitInstruction("M", [0]))

    class MalformedMeasurement:
        name = "M"
        tag = ""

        @staticmethod
        def targets_copy() -> list[stim.GateTarget]:
            return [stim.target_rec(-1)]

        @staticmethod
        def gate_args_copy() -> list[float]:
            return [0.1, 0.2]

    with pytest.raises(ValueError, match="one gate argument"):
        circuits.NoiseRule(readout_error=0.1).noisy_operation(MalformedMeasurement())


def test_emit_after() -> None:
    """NoiseRule.emit_after emits `after` noise per gate, and checks the number of targets."""
    circuit = stim.Circuit()
    circuits.NoiseRule().emit_after(circuit, [0, 1, 2])
    assert circuit == stim.Circuit()
    with pytest.raises(ValueError, match="multiple of 2 qubits"):
        circuits.NoiseRule(after={"XX": 0.1}).emit_after(circuit, [0, 1, 2])


def test_as_noise_rule() -> None:
    """Error specifications are normalized to noise rules."""
    rule = circuits.NoiseRule(after={"X": 0.1})
    assert rules._as_noise_rule(rule, 1) is rule
    assert rules._as_noise_rule(None, 1) is None
    assert rules._as_noise_rule(False, 1) is None
    assert repr(rules._as_noise_rule(circuits.PauliChannel({"X": 0.1}), 1)) == repr(rule)
    assert repr(rules._as_noise_rule({"X": 0.1}, 1)) == repr(rule)
    assert repr(rules._as_noise_rule(0.1, 2)) == repr(
        circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.1))
    )
    with pytest.raises(TypeError, match="expected a float"):
        rules._as_noise_rule(True, 1)


def test_normalize_clifford_nq_error() -> None:
    """The clifford_nq_error argument is normalized to a dictionary of noise rules."""
    rule = circuits.NoiseRule(after={"XXX": 0.1})
    assert rules._normalize_clifford_nq_error(None) == {}
    normalized = rules._normalize_clifford_nq_error(
        {1: 0.1, 2: circuits.PauliChannel({"XX": 0.1}), 3: rule, 4: {"XXXX": 0.1}, 5: 0}
    )
    assert {weight: rule.after for weight, rule in normalized.items()} == {
        1: circuits.PauliChannel.depolarizing(1, 0.1),
        2: circuits.PauliChannel({"XX": 0.1}),
        3: rule.after,
        4: circuits.PauliChannel({"XXXX": 0.1}),
    }
    with pytest.raises(ValueError, match="must be >= 1"):
        rules._normalize_clifford_nq_error({0: 0.1})
    with pytest.raises(TypeError, match="unsupported type"):
        rules._normalize_clifford_nq_error({1: "X"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"is not in \[0, 1\]"):
        rules._normalize_clifford_nq_error({1: 1.5})


def test_rule_validation() -> None:
    """Rules are validated against the shape of the operations they apply to."""
    rule = circuits.NoiseRule(after={"X": 0.1})
    rules._validate_rule_for_arity(rule, 1, "context")
    with pytest.raises(ValueError, match="arity 1; expected 2"):
        rules._validate_rule_for_arity(rule, 2, "context")
    with pytest.raises(ValueError, match="readout_error"):
        rules._validate_rule_for_arity(circuits.NoiseRule(readout_error=0.1), 1, "context")
    with pytest.raises(ValueError, match="reset_error"):
        rules._validate_rule_for_arity(circuits.NoiseRule(reset_error=0.1), 1, "context")

    rules._validate_custom_rule(
        circuits.NoiseRule(readout_error=0.1), stim.CircuitInstruction("MR", [0])
    )
    with pytest.raises(ValueError, match="rule_func returned a rule for 'CX'"):
        rules._validate_custom_rule(rule, stim.CircuitInstruction("CX", [0, 1]))


def test_rule_with_immunity() -> None:
    """Noise rules are dropped or conditioned for operations that touch immune qubits."""
    rule = circuits.NoiseRule(after={"XX": 0.1, "ZZ": 0.2, "XZ": 0.3})
    op = stim.CircuitInstruction("CX", [0, 1])
    assert rules._rule_with_immunity(rule, op, frozenset(), immunize_gates=True) is rule
    assert rules._rule_with_immunity(rule, op, frozenset({2}), immunize_gates=True) is rule
    assert rules._rule_with_immunity(rule, op, frozenset({0, 1}), immunize_gates=False) is None
    assert rules._rule_with_immunity(rule, op, frozenset({1}), immunize_gates=True) is None
    rule = circuits.NoiseRule(after={"XI": 0.1, "XX": 0.2})
    conditioned = rules._rule_with_immunity(rule, op, frozenset({1}), immunize_gates=False)
    assert conditioned is not None and conditioned.after == circuits.PauliChannel({"XI": 0.1})
    with pytest.raises(ValueError, match="partial immunity"):
        rules._rule_with_immunity(
            circuits.NoiseRule(readout_error=0.1),
            stim.CircuitInstruction("MXX", [0, 1]),
            frozenset({1}),
            immunize_gates=False,
        )
