"""Unit tests for noise_model.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import pickle

import pytest
import stim
import tsim

from qldpc import circuits


def _circuits_are_equivalent(
    circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10
) -> bool:
    """Test equivalence between circuits after some standardization."""
    trivial_noise_model = circuits.NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def test_gate_errors() -> None:
    """Add gate errors to a circuit."""

    # ordinary gate errors
    circuit = stim.Circuit("""
        H 0
        CX 0 1 1 2
        TICK
        M 0
        RX 1
        MR 2
    """)
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0.1, clifford_2q_error=0.2, readout_error=0.3, reset_error=0.4
    )
    noisy_circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.1) 0
        CX 0 1
        DEPOLARIZE2(0.2) 0 1
        CX 1 2
        DEPOLARIZE2(0.2) 1 2
        TICK
        MZ(0.3) 0
        RX 1
        MR(0.3) 2
        Z_ERROR(0.4) 1
        X_ERROR(0.4) 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # compose gate errors
    p_m = 0.1
    double_p_m = 1 - (1 - p_m) ** 2
    noise_model = circuits.NoiseModel(readout_error=p_m)
    circuit = stim.Circuit("""
        H 0
        M 0
    """)
    noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({p_m}) 0
    """)
    double_noisy_circuit = stim.Circuit(f"""
        H 0
        MZ({double_p_m}) 0
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert _circuits_are_equivalent(double_noisy_circuit, noise_model.noisy_circuit(noisy_circuit))

    # reusing a qubit in the same moment raises an error
    circuit = stim.Circuit("""
        CX 0 1 1 2
    """)
    noise_model = circuits.SI1000NoiseModel(0.1)
    with pytest.raises(ValueError, match="multiple uses"):
        noise_model.noisy_circuit(circuit, insert_ticks=False)


def test_idle_errors() -> None:
    """Add idling errors to a circuit."""

    circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M 0
        DETECTOR rec[-1]
    """)
    noise_model = circuits.NoiseModel(
        readout_error=0.1, idle_error=0.2, additional_error_waiting_for_m_or_r=0.3
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        DEPOLARIZE1(0.2) 2
        DEPOLARIZE1(0.3) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # user-defined idling noise via NoiseRule (overrides the default DEPOLARIZE1 channel)
    idle_rule = circuits.NoiseRule(after={"X": 0.05, "Z": 0.1})
    m_or_r_rule = circuits.NoiseRule(after=circuits.PauliChannel({"X": 0.2, "Z": 0.3}))
    noise_model = circuits.NoiseModel(
        readout_error=0.1, idle_error=idle_rule, additional_error_waiting_for_m_or_r=m_or_r_rule
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        PAULI_CHANNEL_1(0.05, 0.0, 0.1) 2
        PAULI_CHANNEL_1(0.2, 0, 0.3) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # zero/None errors normalize to None for all NoiseRule-valued fields
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0,
        clifford_2q_error=0,
        idle_error=0,
        additional_error_waiting_for_m_or_r=None,
    )
    assert noise_model.clifford_1q_error is None
    assert noise_model.clifford_2q_error is None
    assert noise_model.idle_error is None
    assert noise_model.additional_error_waiting_for_m_or_r is None
    assert not bool(noise_model)


def test_immunity() -> None:
    """Qubits and operations can be immune to errors."""

    # qubits can be immune to errors.  The input reuses qubit 1 across ``H 0 1`` and ``CNOT 1 2``,
    # so a TICK must separate them (either manually with ``insert_ticks=False`` or automatically).
    circuit = stim.Circuit("""
        H 0 1
        CNOT 1 2
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    expected = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.1) 1
        TICK
        CNOT 1 2
        DEPOLARIZE2(0.1) 1 2
    """)
    # Automatic TICK insertion:
    assert _circuits_are_equivalent(
        expected, noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=True)
    )
    # Explicit TICK with ``insert_ticks=False``:
    assert _circuits_are_equivalent(
        expected,
        noise_model.noisy_circuit(
            stim.Circuit("H 0 1\nTICK\nCNOT 1 2"),
            immune_qubits=[0],
            insert_ticks=False,
        ),
    )
    # Without a TICK between reusing gates, ``insert_ticks=False`` raises loudly.
    with pytest.raises(ValueError, match="multiple times without a TICK"):
        noise_model.noisy_circuit(circuit, immune_qubits=[0], insert_ticks=False)

    # operations can be immune to errors
    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        CX 0 1
        H['{immune_op_tag}'] 0
        X 1
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=True)
    noisy_circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        CX 0 1
        DEPOLARIZE2(0.1) 0 1
        H['{immune_op_tag}'] 0
        X 1
        DEPOLARIZE1(0.1) 1
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag)
    )

    # circuits can be made immune to errors
    tableau = stim.Tableau.random(5)
    noiseless_circuit = circuits.as_noiseless_circuit(tableau.to_circuit())
    assert noise_model.noisy_circuit(noiseless_circuit).to_tableau() == tableau


def test_immune_qubits() -> None:
    """immune_qubits + immunize_gates: noise on partially-immune gates and measurements.

    Under immunize_gates=True (the default), partial-immune noise is dropped per broadcast atom.
    Under immunize_gates=False, PauliChannel after-noise is conditioned via
    ``PauliChannel.conditioned_on``; stim.Circuit fragments and readout_error on partial immunity
    raise instead.
    """

    # 2-qubit broadcast pairs: (0, 1) both immune → drop, (1, 2) partial → drop under True /
    # condition to DEPOLARIZE1(p/5) on surviving qubit under False, (3, 4) neither immune → keep
    # full 2q noise.  DEPOLARIZE2(p) conditioned on one immune position gives a uniform 1q
    # depolarizing channel of total probability p/5 on the surviving qubit.
    circuit = stim.Circuit("""
        CNOT 0 1 1 2 3 4
    """)
    dm = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    assert _circuits_are_equivalent(
        stim.Circuit("""
            CNOT 0 1
            TICK
            CNOT 1 2
            CNOT 3 4
            DEPOLARIZE2(0.1) 3 4
        """),
        dm.noisy_circuit(circuit, immune_qubits=[0, 1], immunize_gates=True),
    )
    assert _circuits_are_equivalent(
        stim.Circuit("""
            CNOT 0 1
            TICK
            CNOT 1 2
            CNOT 3 4
            DEPOLARIZE1(0.02) 2
            DEPOLARIZE2(0.1) 3 4
        """),
        dm.noisy_circuit(circuit, immune_qubits=[0, 1], immunize_gates=False),
    )

    # 3q PauliChannel with an immune middle position: whole gate's noise drops under True; under
    # False the channel is projected via ``PauliChannel.conditioned_on`` — surviving strings
    # (those with I at the immune position) emit as native PAULI_CHANNEL_2 on the outer qubits.
    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03})
    spp_circuit = stim.Circuit("SPP X0*Y1*Z2")
    spp_model = circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(after=channel)})
    assert _circuits_are_equivalent(
        spp_circuit,
        spp_model.noisy_circuit(spp_circuit, immune_qubits=[1], immunize_gates=True),
    )
    assert _circuits_are_equivalent(
        stim.Circuit(
            "SPP X0*Y1*Z2\nPAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0) 0 2"
        ),
        spp_model.noisy_circuit(spp_circuit, immune_qubits=[1], immunize_gates=False),
    )

    # reset_error broadcasts X_ERROR; immune targets are stripped per atom under True.
    assert _circuits_are_equivalent(
        stim.Circuit("R 0 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.1) 2"),
        circuits.NoiseModel(reset_error=0.1).noisy_circuit(
            stim.Circuit("R 0 1 2"), immune_qubits=[1]
        ),
    )

    # Under immunize_gates=True, any measurement touching an immune qubit is dropped entirely —
    # readout_error included.  `M 0` keeps its readout error; `M 1` (fully immune) drops it;
    # `MPP X0*Y1` (partial immunity) also drops it, matching the dead-simple policy.
    assert _circuits_are_equivalent(
        stim.Circuit("M(0.1) 0\nTICK\nM 1\nTICK\nMPP X0*Y1"),
        circuits.NoiseModel(readout_error=0.1).noisy_circuit(
            stim.Circuit("M 0\nTICK\nM 1\nTICK\nMPP X0*Y1"), immune_qubits=[1]
        ),
    )
    # Under immunize_gates=False, the same MPP with partial immunity has no clean projection
    # (readout_error can't be conditioned), so we raise.
    with pytest.raises(ValueError, match="partial immunity"):
        circuits.NoiseModel(readout_error=0.1).noisy_circuit(
            stim.Circuit("MPP X0*Y1"), immune_qubits=[1], immunize_gates=False
        )


def test_classical_controls() -> None:
    """Classically controled gates get special treatment."""
    noise_model: circuits.NoiseModel

    # classically controls are immune to noise, but the qubits still pick up idling errors
    circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
    """)
    noise_model = circuits.SI1000NoiseModel(0.1)
    noisy_circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
        DEPOLARIZE2(0.1) 0 1
        DEPOLARIZE1(0.01) 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # qubits addressed by classical controls pick up idling errors by default
    circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        TICK
        H 0 1 2
    """)
    noise_model = circuits.NoiseModel(idle_error=0.1)
    noisy_circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        DEPOLARIZE1(0.1) 1 2
        TICK
        H 0 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_measurements() -> None:
    """Pauli product measurements get special treatment."""

    circuit = stim.Circuit("""
        MPP X1*Y2*Z3
    """)
    noise_model = circuits.NoiseModel(readout_error=0.1, idle_error=0.2)
    noisy_circuit = stim.Circuit("""
        MPP(0.1) X1*Y2*Z3
        DEPOLARIZE1(0.2) 0
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # override the default MPP rule for specific Pauli products
    circuit = stim.Circuit("""
        MPP Z0*Z1*Z2
        MPP X0*Y1*Z2
    """)
    noise_rule = circuits.NoiseRule(readout_error=0.2)
    noise_model = circuits.NoiseModel(readout_error=0.1, rules={"MXYZ": noise_rule})
    noisy_circuit = stim.Circuit("""
        MPP(0.1) Z0*Z1*Z2
        MPP(0.2) X0*Y1*Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_cliffords() -> None:
    """SPP gates on 1 or 2 qubits get, respectively, 1q or 2q noise."""

    # SPP on one qubit -> clifford_1q_error; SPP on two qubits -> clifford_2q_error;
    # SPP on three or more qubits is ignored by default.
    circuit = stim.Circuit("""
        SPP X0 X0*Y1
        SPP_DAG X0*Y1*Z2
    """)
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, clifford_2q_error=0.2)
    noisy_circuit = stim.Circuit("""
        SPP X0
        DEPOLARIZE1(0.1) 0
        TICK
        SPP X0*Y1
        DEPOLARIZE2(0.2) 0 1
        TICK
        SPP_DAG X0*Y1*Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # multi-product SPP is split so each Pauli product is treated independently
    circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
    """)
    noisy_circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
        DEPOLARIZE1(0.1) 0
        DEPOLARIZE2(0.2) 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # explicit rules dict overrides the default weight-based dispatch, including for weight >= 3
    noise_rule = circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.3))
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, rules={"SPP": noise_rule})
    circuit = stim.Circuit("""
        SPP X0 X0*Y1*Z2
    """)
    noisy_circuit = stim.Circuit("""
        SPP X0
        DEPOLARIZE1(0.3) 0
        TICK
        SPP X0*Y1*Z2
        DEPOLARIZE1(0.3) 0 1 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_rule_func() -> None:
    """A user-provided rule_func assigns noise per gate application, with top priority."""

    # Broadcast gates are decomposed into individual applications before the callback is invoked;
    # it is consulted once per application (per qubit for a 1q gate).  Returning None falls back to
    # the ordinary clifford_1q_error.
    seen: list[tuple[str, str, tuple[int, ...]]] = []

    def per_qubit(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        targets = [t for t in op.targets_copy() if not t.is_combiner]
        seen.append((op.name, op.tag, tuple(t.qubit_value for t in targets)))
        if op.name == "H" and targets[0].qubit_value == 1:
            return circuits.NoiseRule(after={"X": 0.5})
        return None

    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, rule_func=per_qubit)
    circuit = stim.Circuit("H 0 1 2")
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        DEPOLARIZE1(0.1) 0
        X_ERROR(0.5) 1
        DEPOLARIZE1(0.1) 2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert seen == [("H", "", (0,)), ("H", "", (1,)), ("H", "", (2,))]

    # A two-qubit gate is decomposed per pair; the callback sees both targets and may return a
    # two-qubit rule.  Other pairs fall back to clifford_2q_error.
    def per_pair(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        targets = op.targets_copy()
        if op.name == "CX" and {t.qubit_value for t in targets} == {2, 3}:
            return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.9))
        return None

    noise_model = circuits.NoiseModel(clifford_2q_error=0.1, rule_func=per_pair)
    circuit = stim.Circuit("CX 0 1 2 3")
    noisy_circuit = stim.Circuit("""
        CX 0 1 2 3
        DEPOLARIZE2(0.1) 0 1
        DEPOLARIZE2(0.9) 2 3
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # The callback takes top priority over `rules`, and receives the instruction's tag.
    def tagged(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        assert op.tag == "mytag"
        return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.7))

    noise_model = circuits.NoiseModel(
        rules={"H": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.2))},
        rule_func=tagged,
    )
    circuit = stim.Circuit()
    circuit.append("H", [0], tag="mytag")
    noisy_circuit = stim.Circuit()
    noisy_circuit.append("H", [0], tag="mytag")
    noisy_circuit.append("DEPOLARIZE1", [0], [0.7])
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # Measurement and reset errors can be assigned per application too.  The callback receives the
    # raw op.name (`M`, `MZ`, `R`, etc.); if the user wants alias-agnostic matching they can either
    # check every alias or use the same key that `rules` uses (stim's canonical name).
    def measure_reset(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        if op.name in ("R", "RZ"):
            return circuits.NoiseRule(reset_error=0.3)
        if op.name in ("M", "MZ") and op.targets_copy()[0].qubit_value == 0:
            return circuits.NoiseRule(readout_error=0.25)
        return None

    noise_model = circuits.NoiseModel(readout_error=0.01, rule_func=measure_reset)
    circuit = stim.Circuit("R 2\nTICK\nM 0 1")
    noisy_circuit = stim.Circuit("""
        R 2
        X_ERROR(0.3) 2
        TICK
        M(0.25) 0
        M(0.01) 1
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # Noise-immune qubits still take precedence over the callback.
    def kick(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.5))

    noise_model = circuits.NoiseModel(rule_func=kick)
    circuit = stim.Circuit("H 0 1")
    noisy_circuit = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.5) 0
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits={1})
    )

    # The callback is not consulted for annotations or classically-controlled operations.
    consulted: list[str] = []

    def record(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        consulted.append(op.name)
        return circuits.NoiseRule(after={"X": 0.5})

    noise_model = circuits.NoiseModel(rule_func=record)
    noise_model.noisy_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nM 0\nCX rec[-1] 1"))
    assert consulted == ["M"]

    # A returned rule's readout_error/reset_error must match the gate it is assigned to.
    bad_readout = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(readout_error=0.1))
    with pytest.raises(ValueError, match="rule for 'H'.*readout_error.*measurement gates"):
        bad_readout.noisy_circuit(stim.Circuit("H 0"))
    bad_reset = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(reset_error=0.1))
    with pytest.raises(ValueError, match="rule for 'H'.*reset_error.*reset gates"):
        bad_reset.noisy_circuit(stim.Circuit("H 0"))

    # A returned rule's `after` arity must match the gate application's qubit count.
    bad_arity = circuits.NoiseModel(
        rule_func=lambda op: circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.1))
    )
    with pytest.raises(ValueError, match="rule for 'CX'.*`after` has arity 1"):
        bad_arity.noisy_circuit(stim.Circuit("CX 0 1"))

    # SXYZ can be used as a `rules` key just like MXYZ.
    noise_model = circuits.NoiseModel(
        rules={"SXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(3, 0.1))}
    )
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    assert "PAULI_CHANNEL_3" in str(noisy_circuit) or "CORRELATED_ERROR" in str(noisy_circuit)

    # The recursive noisy_circuit call inside a REPEAT block forwards immune_op_tag.
    kick_all = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(after={"X": 0.5}))
    body = stim.Circuit()
    body.append("H", [0], tag="__skip__")
    body.append("TICK")
    repeat_circuit = stim.Circuit()
    repeat_circuit.append(stim.CircuitRepeatBlock(repeat_count=2, body=body))
    result = kick_all.noisy_circuit(repeat_circuit, immune_op_tag="__skip__")
    assert "X_ERROR" not in str(result)


def test_pauli_channel_class() -> None:
    """PauliChannel construction, validation, and helpers."""

    ch = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch.num_qubits == 3
    assert dict(ch.probabilities) == {"XYZ": 0.01, "ZZZ": 0.02}
    assert bool(ch)
    assert ch == circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    assert ch != circuits.PauliChannel({"XYZ": 0.01})
    assert ch != "foo"  # __eq__ returns NotImplemented for non-PauliChannel operands
    assert repr(ch).startswith("PauliChannel(")

    # zero-prob channel is falsy but still constructible
    assert not bool(circuits.PauliChannel({"X": 0.0}))

    # depolarizing(n, p): 4**n - 1 entries, probabilities sum to p, first key in lex order is "IX"
    dp = circuits.PauliChannel.depolarizing(2, 0.15)
    assert dp.num_qubits == 2
    assert len(dp.probabilities) == 15
    assert abs(sum(dp.probabilities.values()) - 0.15) < 1e-12
    assert next(iter(dp.probabilities.keys())) == "IX"

    # empty channel is the trivial 0-qubit channel
    empty = circuits.PauliChannel({})
    assert empty.num_qubits == 0
    assert dict(empty.probabilities) == {}
    assert not bool(empty)

    # validation errors
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"": 0.1})
    with pytest.raises(ValueError, match="must have length"):
        circuits.PauliChannel({"XY": 0.1, "X": 0.1})
    with pytest.raises(ValueError, match="invalid characters"):
        circuits.PauliChannel({"W": 0.1})
    with pytest.raises(ValueError, match="Identity string"):
        circuits.PauliChannel({"II": 0.1})
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel({"X": 1.5})
    with pytest.raises(ValueError, match="Sum of Pauli channel"):
        circuits.PauliChannel({"X": 0.7, "Y": 0.7})
    with pytest.raises(ValueError, match="num_qubits=0"):
        circuits.PauliChannel.depolarizing(0, 0.1)
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.PauliChannel.depolarizing(2, 1.5)
    with pytest.raises(ValueError, match="disagrees with Pauli string length"):
        circuits.PauliChannel({"XY": 0.1}, num_qubits=3)
    with pytest.raises(ValueError, match="must be >= 0"):
        circuits.PauliChannel({}, num_qubits=-1)

    # An empty channel with a nontrivial arity is a distinct object and reproduces via repr.
    empty_3q = circuits.PauliChannel({}, num_qubits=3)
    assert empty_3q.num_qubits == 3
    assert repr(empty_3q) == "PauliChannel({}, num_qubits=3)"
    assert empty_3q != circuits.PauliChannel({})


def test_pauli_channel_conditioned_on() -> None:
    """PauliChannel.conditioned_on returns the identity-on-immune-qubits sub-channel."""

    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03, "XII": 0.04})

    # Marginalize position 1 (middle): keep strings with I at pos 1.  Arity is preserved, so the
    # surviving strings still have an I at position 1.
    assert channel.conditioned_on([1]) == circuits.PauliChannel({"XIZ": 0.02, "XII": 0.04})

    # Marginalize positions 0 and 2: only "IZI" has I at both.
    assert channel.conditioned_on([0, 2]) == circuits.PauliChannel({"IZI": 0.03})

    # Empty index list is a no-op (returns an equal channel).
    assert channel.conditioned_on([]) == channel

    # Marginalizing every position yields an empty channel on the same number of qubits.
    empty_3q = channel.conditioned_on([0, 1, 2])
    assert empty_3q == circuits.PauliChannel({}, num_qubits=3)
    assert empty_3q.num_qubits == 3
    assert not bool(empty_3q)

    # If nothing survives, the result is empty but arity is preserved.
    all_non_id = circuits.PauliChannel({"XY": 0.1, "YX": 0.1})
    assert all_non_id.conditioned_on([0]) == circuits.PauliChannel({}, num_qubits=2)

    # Out-of-range indices raise.
    with pytest.raises(ValueError, match="not in"):
        channel.conditioned_on([3])
    with pytest.raises(ValueError, match="not in"):
        channel.conditioned_on([-1])


def test_pauli_channel_to_circuit() -> None:
    """PauliChannel.to_circuit emits noise, optionally appending to a given circuit and qubits."""

    channel = circuits.PauliChannel({"X": 0.1, "Z": 0.2})

    # With no arguments, a fresh circuit is created acting on range(num_qubits).
    expected = stim.Circuit()
    expected.append("PAULI_CHANNEL_1", [0], [0.1, 0.0, 0.2])
    assert channel.to_circuit() == expected

    # Explicit qubits are used as the targets.
    assert channel.to_circuit(qubits=[3]) == stim.Circuit("PAULI_CHANNEL_1(0.1, 0, 0.2) 3")

    # An existing circuit is appended to and returned (same object).
    circuit = stim.Circuit("H 0")
    returned = channel.to_circuit(append_to=circuit, qubits=[2])
    assert returned is circuit
    assert circuit == stim.Circuit("H 0\nPAULI_CHANNEL_1(0.1, 0, 0.2) 2")

    # A tag is applied to the emitted instruction.
    tagged = channel.to_circuit(qubits=[0], tag="foo")
    assert tagged == stim.Circuit("PAULI_CHANNEL_1[foo](0.1, 0, 0.2) 0")

    # simplify=False always emits a CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain instead of a
    # native PAULI_CHANNEL form, even for a 1-qubit channel that would otherwise collapse.
    expected = stim.Circuit()
    expected.append("CORRELATED_ERROR", [stim.target_x(0)], [0.1])
    expected.append("ELSE_CORRELATED_ERROR", [stim.target_z(0)], [0.2 / 0.9])
    assert channel.to_circuit(simplify=False) == expected

    # A two-qubit depolarizing channel collapses to a native DEPOLARIZE2.
    depol2 = circuits.PauliChannel.depolarizing(2, 0.15)
    assert depol2.to_circuit() == stim.Circuit("DEPOLARIZE2(0.15) 0 1")

    # An empty channel emits nothing, returning the (possibly provided) circuit unchanged.
    assert circuits.PauliChannel({}, num_qubits=3).to_circuit() == stim.Circuit()
    prefilled = stim.Circuit("H 0")
    assert circuits.PauliChannel({}).to_circuit(append_to=prefilled) is prefilled
    assert prefilled == stim.Circuit("H 0")

    # Raise an error when provided qubit count that disagrees with the channel arity.
    with pytest.raises(ValueError, match="Provided 2 qubits for a 1-qubit channel"):
        channel.to_circuit(qubits=[0, 1])


def test_multi_qubit_pauli_channel_after_gate() -> None:
    """NoiseRule(after=PauliChannel(...)) emits a CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain."""

    # A sparse 3-qubit Pauli channel applied via rules={"SPP": ...}
    channel = circuits.PauliChannel({"XYZ": 0.01, "ZZZ": 0.02})
    rule = circuits.NoiseRule(after=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    circuit = stim.Circuit("""
        SPP X0*Y1*Z2
    """)
    # Marginals: XYZ at 0.01, ZZZ at 0.02.  ELSE renormalized: 0.02 / (1 - 0.01).
    noisy_circuit = stim.Circuit("""
        SPP X0*Y1*Z2
        CORRELATED_ERROR(0.01) X0 Y1 Z2
        ELSE_CORRELATED_ERROR(0.020202020202020204) Z0 Z1 Z2
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # 2-qubit channels emit a native PAULI_CHANNEL_2 (args in stim's order:
    # IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ)
    channel = circuits.PauliChannel({"XI": 0.0, "IX": 0.05, "XX": 0.05})
    rule = circuits.NoiseRule(after=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    circuit = stim.Circuit("SPP X0*Y1")
    noisy_circuit = stim.Circuit("""
        SPP X0*Y1
        PAULI_CHANNEL_2(0.05, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1
    """)
    assert _circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_clifford_nq_error() -> None:
    """clifford_nq_error dispatches by qubit count k."""

    # A float value produces a uniform depolarizing channel of that weight
    noise_model = circuits.NoiseModel(clifford_nq_error={3: 0.001})
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    noisy = noise_model.noisy_circuit(circuit)
    # 63 non-identity 3q Paulis: 1 CORRELATED_ERROR (canonicalized by stim to "E") plus
    # 62 ELSE_CORRELATED_ERROR
    ce_count = sum(1 for op in noisy if op.name == "E")
    else_ce_count = sum(1 for op in noisy if op.name == "ELSE_CORRELATED_ERROR")
    assert ce_count == 1 and else_ce_count == 62

    # multi-product SPP: each product is routed to its weight's rule.  Weight-1 and weight-2
    # channels emit native PAULI_CHANNEL_1 / PAULI_CHANNEL_2; weight-3 and above use a
    # CORRELATED_ERROR chain.  Values may be a raw Mapping (auto-wrapped) or a PauliChannel.
    noise_model = circuits.NoiseModel(
        clifford_nq_error={
            1: {"X": 0.01},  # raw Mapping — auto-wrapped as PauliChannel
            2: circuits.PauliChannel({"XY": 0.02}),
            3: circuits.PauliChannel({"XYZ": 0.03}),
        }
    )
    circuit = stim.Circuit("SPP X0 Y1*Z2 X3*Y4*Z5")
    noisy = noise_model.noisy_circuit(circuit)
    expected = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
        X_ERROR(0.01) 0
        PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0) 1 2
        CORRELATED_ERROR(0.03) X3 Y4 Z5
    """)
    assert _circuits_are_equivalent(expected, noisy)

    # The stim.Circuit form of `after` is an escape hatch: broadcast noise + a joint CE chain
    # can be combined by spelling out targets in one fragment.
    rule = circuits.NoiseRule(
        after=stim.Circuit("""
            X_ERROR(0.01) 0 1 2
            CORRELATED_ERROR(0.02) X0 Y1 Z2
        """)
    )
    noise_model = circuits.NoiseModel(clifford_nq_error={3: rule})
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    noisy = noise_model.noisy_circuit(circuit)
    expected = stim.Circuit("""
        SPP X0*Y1*Z2
        X_ERROR(0.01) 0 1 2
        CORRELATED_ERROR(0.02) X0 Y1 Z2
    """)
    assert _circuits_are_equivalent(expected, noisy)

    # Falsy values (zero float, empty NoiseRule) are dropped from the normalized dict
    assert circuits.NoiseModel(clifford_nq_error={3: 0.0}).clifford_nq_error == {}
    assert circuits.NoiseModel(clifford_nq_error={3: circuits.NoiseRule()}).clifford_nq_error == {}

    # clifford_1q_error / clifford_2q_error also accept a PauliChannel directly, or a raw
    # Pauli-string Mapping (auto-wrapped as PauliChannel).
    noise_model = circuits.NoiseModel(
        clifford_1q_error=circuits.PauliChannel({"X": 0.01}),
        clifford_2q_error={"XY": 0.02},
    )
    assert _circuits_are_equivalent(
        stim.Circuit(
            "H 0\nX_ERROR(0.01) 0\nTICK\nCX 0 1\nPAULI_CHANNEL_2("
            + "0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1"
        ),
        noise_model.noisy_circuit(stim.Circuit("H 0\nCX 0 1")),
    )


def test_clifford_nq_error_errors() -> None:
    """Validation errors around clifford_nq_error and related rules."""

    # Ambiguity: pp[k] and clifford_kq_error both user-specified (symmetric — either side raises,
    # even when either value is a no-op zero).
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.02})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_2q_error=0.01, clifford_nq_error={2: 0.02})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.0})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.0, clifford_nq_error={1: 0.01})

    # Invalid key / out-of-range probability / bool rejection
    with pytest.raises(ValueError, match="must be >= 1"):
        circuits.NoiseModel(clifford_nq_error={0: 0.01})
    with pytest.raises(ValueError, match="not in \\[0, 1\\]"):
        circuits.NoiseModel(clifford_nq_error={3: 1.5})
    with pytest.raises(TypeError, match="unsupported type bool"):
        circuits.NoiseModel(clifford_nq_error={2: True})

    # num_qubits mismatch between key and PauliChannel value (checked at construction).
    with pytest.raises(ValueError, match="arity 2; expected 3"):
        circuits.NoiseModel(clifford_nq_error={3: circuits.PauliChannel.depolarizing(2, 0.01)})

    # num_qubits mismatch via `rules=` is not caught at construction (arity varies), but is caught
    # at emission by `emit_after`.
    bad_rule = circuits.NoiseRule(after=circuits.PauliChannel({"XY": 0.01}))
    noise_model = circuits.NoiseModel(rules={"SPP": bad_rule})
    with pytest.raises(ValueError, match="expects a multiple of 2 qubits"):
        noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))

    # A 2-qubit `after` broadcast (via a 2q PauliChannel) applied to a wrong-arity gate raises
    # at construction time via `_validate_rule_for_arity`, both for fixed-arity gate names and
    # for basis-suffixed rule keys ("MXYZ" is arity 3 — a 2q `after` is caught up front).
    with pytest.raises(ValueError, match="arity 2; expected 1"):
        circuits.NoiseModel(
            rules={"H": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        )
    with pytest.raises(ValueError, match="arity 2; expected 3"):
        circuits.NoiseModel(
            rules={"MXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        )
    # Bare "MPP" / "SPP" remain variable-arity; wrong-arity `after` is caught at emission.
    with pytest.raises(ValueError, match="expects a multiple of 2 qubits"):
        circuits.NoiseModel(
            rules={"MPP": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        ).noisy_circuit(stim.Circuit("MPP X0*Y1*Z2"))

    # readout_error / reset_error are rejected on rules for gates that can't measure/reset.
    with pytest.raises(ValueError, match="readout_error.*only valid on measurement"):
        circuits.NoiseModel(rules={"H": circuits.NoiseRule(readout_error=0.1)})
    with pytest.raises(ValueError, match="reset_error.*only valid on reset"):
        circuits.NoiseModel(rules={"M": circuits.NoiseRule(reset_error=0.1)})


def test_pauli_channel_idle_error_rejection() -> None:
    """Multi-qubit `after` rules are not accepted on idle-error rules (all shapes rejected)."""
    channel = circuits.PauliChannel({"XY": 0.01})
    with pytest.raises(ValueError, match="idle_error.*multi-qubit"):
        circuits.NoiseModel(idle_error=circuits.NoiseRule(after=channel))
    with pytest.raises(ValueError, match="additional_error_waiting_for_m_or_r.*multi-qubit"):
        circuits.NoiseModel(additional_error_waiting_for_m_or_r=circuits.NoiseRule(after=channel))
    # stim.Circuit form is rejected on the same grounds
    with pytest.raises(ValueError, match="idle_error.*multi-qubit"):
        circuits.NoiseModel(
            idle_error=circuits.NoiseRule(after=stim.Circuit("DEPOLARIZE2(0.1) 0 1"))
        )


def test_pauli_channel_drops_zeros() -> None:
    """PauliChannel drops zero-prob entries but preserves arity."""
    ch = circuits.PauliChannel({"XI": 0.0, "IX": 0.05, "XX": 0.05})
    assert list(ch.probabilities.keys()) == ["IX", "XX"]
    # An all-zero PauliChannel drops its entries but keeps the arity derived from the strings,
    # and normalizes to a trivial `after` when attached to a NoiseRule.
    all_zero = circuits.PauliChannel({"XY": 0.0})
    assert all_zero.num_qubits == 2 and dict(all_zero.probabilities) == {} and not bool(all_zero)
    all_zero_rule = circuits.NoiseRule(after=circuits.PauliChannel({"XY": 0.0}))
    assert not all_zero_rule.after and not bool(all_zero_rule)


def test_pauli_channel_hashable() -> None:
    """PauliChannel is hashable (equal channels hash equal), regardless of insertion order."""
    ch1 = circuits.PauliChannel({"XI": 0.1, "IX": 0.2})
    ch2 = circuits.PauliChannel({"IX": 0.2, "XI": 0.1})  # same channel, different insertion order
    assert ch1 == ch2 and hash(ch1) == hash(ch2)
    # Usable as a dict key / set member
    assert {ch1: "a", ch2: "b"} == {ch1: "b"}


def test_pauli_channel_float_drift_clamped() -> None:
    """Chain emission clamps and short-circuits when FP drift makes remaining <= prob."""
    # 3-qubit channel with keys chosen so lex order gives probs {0.1, 0.2, 0.3, 0.4}.
    # Sequential subtraction leaves remaining = 0.39999999999999997 < 0.4 at the final step, so
    # the last ELSE_CORRELATED_ERROR must clamp to 1.0 (rather than emit a value > 1.0) and stop.
    channel = circuits.PauliChannel({"XXX": 0.1, "XXY": 0.2, "XXZ": 0.3, "YYY": 0.4})
    rule = circuits.NoiseRule(after=channel)
    noise_model = circuits.NoiseModel(rules={"SPP": rule})
    noisy = noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    expected = stim.Circuit("""
        SPP X0*Y1*Z2
        CORRELATED_ERROR(0.1) X0 X1 X2
        ELSE_CORRELATED_ERROR(0.2222222222222222) X0 X1 Y2
        ELSE_CORRELATED_ERROR(0.42857142857142855) X0 X1 Z2
        ELSE_CORRELATED_ERROR(1.0) Y0 Y1 Y2
    """)
    assert _circuits_are_equivalent(expected, noisy)


def test_repeat_blocks() -> None:
    """Repeat blocks get special treatment."""

    circuit = stim.Circuit("""
        H 0
        REPEAT 3 {
            CX 0 1
        }
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT 3 {
            CX 0 1
            DEPOLARIZE2(0.1) 0 1
        }
    """)
    assert _circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, insert_ticks=False)
    )

    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    assert _circuits_are_equivalent(
        noisy_circuit,
        noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag),
    )


def test_noise_rule_errors() -> None:
    """Cover various NoiseRule errors."""
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(readout_error=1.1)
    with pytest.raises(ValueError, match="not between 0 and 1"):
        circuits.NoiseRule(reset_error=1.1)

    # Passing a non-numeric, non-Mapping, non-NoiseRule type for a scalar noise field raises.
    with pytest.raises(TypeError, match="expected a float"):
        circuits.NoiseModel(clifford_1q_error="DEPOLARIZE1")  # type: ignore[arg-type]

    # `after` as a stim.Circuit rejects non-noise instructions and repeat blocks.
    with pytest.raises(ValueError, match="non-noise instruction"):
        circuits.NoiseRule(after=stim.Circuit("H 0"))
    fragment_with_repeat = stim.Circuit()
    fragment_with_repeat.append(stim.CircuitRepeatBlock(2, stim.Circuit("X_ERROR(0.1) 0")))
    with pytest.raises(ValueError, match="may contain only noise instructions"):
        circuits.NoiseRule(after=fragment_with_repeat)

    # An empty PauliChannel with an explicitly-declared arity is honored — a mismatch against the
    # rule's expected arity is flagged rather than silently dropped.
    with pytest.raises(ValueError, match="arity 3; expected 2"):
        circuits.NoiseModel(clifford_nq_error={2: circuits.PauliChannel({}, num_qubits=3)})
    # Same policy for idle_error: an explicitly-declared arity is validated BEFORE trivializing,
    # so a shape-wrong empty channel is flagged rather than silently accepted.
    with pytest.raises(ValueError, match="idle_error.*multi-qubit"):
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
    with pytest.raises(ValueError, match="after.*readout_error"):
        circuits.NoiseRule(after={"X": 0.01}, readout_error=0.1)
    with pytest.raises(ValueError, match="after.*readout_error"):
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
    assert _circuits_are_equivalent(
        stim.Circuit("CX 0 1\nDEPOLARIZE2(0.2) 0 1\nPAULI_CHANNEL_1(0, 0.1, 0.1) 0 1"),
        circuits.NoiseModel(rules={"CX": noise_rule}).noisy_circuit(stim.Circuit("CX 0 1")),
    )

    circuit = stim.Circuit("SPP X0*Y1*Z2")

    # A raw CE chain touching an immune qubit is dropped under immunize_gates=True (default).
    ce_rule = circuits.NoiseRule(after=stim.Circuit("CORRELATED_ERROR(0.02) X0 Y1 Z2"))
    assert _circuits_are_equivalent(
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
    assert _circuits_are_equivalent(
        stim.Circuit("SPP X0*Y1*Z2\nPAULI_CHANNEL_1(0.1, 0.05, 0) 1"),
        circuits.NoiseModel(rules={"SPP": sparse_rule}).noisy_circuit(circuit),
    )

    # HERALDED_ERASE with partial immunity: under immunize_gates=True the whole fragment drops
    # (no partial-atom filtering — stim.Circuit fragments are all-or-nothing); under
    # immunize_gates=False the fragment can't be conditioned, so raise.
    heralded_rule = circuits.NoiseRule(after=stim.Circuit("HERALDED_ERASE(0.05) 0 1"))
    cx_circuit = stim.Circuit("CX 0 1")
    assert _circuits_are_equivalent(
        cx_circuit,
        circuits.NoiseModel(rules={"CX": heralded_rule}).noisy_circuit(
            cx_circuit, immune_qubits=[1]
        ),
    )
    with pytest.raises(ValueError, match="partial immunity"):
        circuits.NoiseModel(rules={"CX": heralded_rule}).noisy_circuit(
            cx_circuit, immune_qubits=[1], immunize_gates=False
        )

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
    assert _circuits_are_equivalent(
        expected,
        circuits.NoiseModel(rules={"CX": mixed_rule}).noisy_circuit(circuit),
    )


def test_trivial_noise() -> None:
    """A NoiseModel with only a rule_func is truthy even before the func is consulted."""
    assert not bool(circuits.NoiseRule())
    assert not bool(circuits.NoiseModel())
    assert bool(circuits.NoiseModel(rule_func=lambda op: None))


def test_tsim_circuits() -> None:
    """noisy_circuit and as_noiseless_circuit accept and return tsim.Circuit."""
    noise_model = circuits.DepolarizingNoiseModel(0.01)

    tsim_circuit = tsim.Circuit("H 0\nCX 0 1\nM 0 1")
    stim_circuit = tsim_circuit.stim_circuit

    stim_noisy = noise_model.noisy_circuit(stim_circuit)
    tsim_noisy = noise_model.noisy_circuit(tsim_circuit)
    assert isinstance(stim_noisy, stim.Circuit)
    assert isinstance(tsim_noisy, tsim.Circuit)
    assert stim_noisy == tsim_noisy.stim_circuit

    stim_noiseless = circuits.as_noiseless_circuit(stim_circuit)
    tsim_noiseless = circuits.as_noiseless_circuit(tsim_circuit)
    assert isinstance(stim_noiseless, stim.Circuit)
    assert isinstance(tsim_noiseless, tsim.Circuit)


####################################################################################################
# serialization, str, and repr capabilities


def test_noise_model_serialization() -> None:
    """NoiseModel survives pickling / deep-copying and reconstructs an equivalent model.

    This transitively exercises PauliChannel and NoiseRule pickling (both appear inside the model),
    so no separate per-class serialization tests are needed.
    """
    model = circuits.NoiseModel(
        clifford_1q_error=0.1,
        clifford_2q_error=0.2,
        readout_error=0.3,
        reset_error=0.4,
        idle_error=0.01,
        additional_error_waiting_for_m_or_r=0.02,
        rules={"SXY": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.05))},
    )
    circuit = stim.Circuit("R 0 1\nH 0\nCX 0 1\nTICK\nM 0 1")
    for restored in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        assert _circuits_are_equivalent(
            model.noisy_circuit(circuit), restored.noisy_circuit(circuit)
        )

    # The one caveat: a rule_func that isn't importable at module scope (a local closure or a
    # lambda) can't be pickled.  That limitation is inherent to pickle, not to NoiseModel — a
    # module-level function would pickle fine.
    func_model = circuits.NoiseModel(clifford_1q_error=0.1, rule_func=lambda op: None)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(func_model)


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


def test_noise_model_str() -> None:
    """NoiseModel.__str__ reads like a constructor call listing only the set fields."""
    assert str(circuits.NoiseModel()) == "NoiseModel()"

    model = circuits.NoiseModel(
        clifford_1q_error=0.1,
        readout_error=0.3,
        reset_error=0.4,
        idle_error=0.01,
        additional_error_waiting_for_m_or_r=0.02,
        rules={"M": circuits.NoiseRule(readout_error=0.05)},
    )
    text = str(model)
    assert text.startswith("NoiseModel(") and text.endswith(")")
    for field in (
        "clifford_nq_error=",
        "readout_error=0.3",
        "reset_error=0.4",
        "idle_error=",
        "additional_error_waiting_for_m_or_r=",
        "rules={'M': NoiseRule(readout_error=0.05)}",
    ):
        assert field in text
    assert "rule_func" not in text

    # With no rule_func, __repr__ is the (eval-able) __str__.
    assert repr(circuits.NoiseModel()) == "NoiseModel()"
    assert repr(model) == text


def test_noise_model_str_rule_func() -> None:
    """__str__ describes a user-provided rule_func by name, falling back to its type."""
    circuit = stim.Circuit("H 0")

    # A named, module-level function: its qualified name and module appear.
    named = circuits.NoiseModel(rule_func=_example_rule_func)
    named.noisy_circuit(circuit)  # exercises (and thus covers) the rule_func body
    assert "rule_func=<" in str(named) and "_example_rule_func" in str(named)

    # A lambda reports as <lambda>.
    lam = circuits.NoiseModel(rule_func=lambda op: None)
    lam.noisy_circuit(circuit)
    assert "<lambda>" in str(lam)

    # A callable object without __qualname__ falls back to its class name.
    inst = circuits.NoiseModel(rule_func=_CallableRule())
    inst.noisy_circuit(circuit)
    assert "_CallableRule" in str(inst)

    # A rule_func can't be reproduced faithfully, so __repr__ falls back to the default object repr
    # (rather than the non-eval-able __str__).
    assert repr(named) != str(named)
    assert repr(named).startswith("<") and "NoiseModel object at " in repr(named)


def _example_rule_func(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
    """A named, module-level rule_func used to check NoiseModel.__str__ prints its name."""
    return None


class _CallableRule:
    """A callable object (no __qualname__) used to test the rule_func description fallback."""

    def __call__(self, op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        return None
