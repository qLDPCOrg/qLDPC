# SPDX-License-Identifier: Apache-2.0

"""Unit tests for noise_model.py."""

import copy
import math
import pickle
import random

import pytest
import stim

from qldpc import circuits, decoders
from qldpc.circuits.conftest import (
    StimCircuitWrapper,
    circuits_are_equivalent,
    flatten_repeat_blocks,
)

# keep the moments of circuits, as written
TICKS = circuits.Scheduler.preserve_ticks(single_qubit_gates_take_time=True)


def test_gate_errors() -> None:
    """Add gate errors to a circuit."""
    assert circuits.DEFAULT_IMMUNE_OP_TAG == "__IMMUNE_TO_NOISE__"
    assert circuits.DEFAULT_IMMUNE_QUBIT_TAG == "__IMMUNE_QUBIT_TO_NOISE__"

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
        scheduler=TICKS,
        clifford_1q_error=0.1,
        clifford_2q_error=0.2,
        readout_error=0.3,
        reset_error=0.4,
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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # compose gate errors
    p_m = 0.1
    double_p_m = 2 * p_m - 2 * p_m**2
    noise_model = circuits.NoiseModel(scheduler=TICKS, readout_error=p_m)
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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert circuits_are_equivalent(double_noisy_circuit, noise_model.noisy_circuit(noisy_circuit))

    noise_model = circuits.SI1000NoiseModel(0.1)
    with pytest.raises(ValueError, match="immune_op_tag"):
        noise_model.noisy_circuit(stim.Circuit("H 0"), immune_op_tag="")
    with pytest.raises(ValueError, match="immune_qubit_tag"):
        noise_model.noisy_circuit(stim.Circuit("H 0"), immune_qubit_tag="")
    with pytest.raises(ValueError, match="max_gate_size"):
        circuits.DepolarizingNoiseModel(0.1, max_gate_size=0)
    wrapped = circuits.as_noiseless_circuit(stim.Circuit("H 0"))
    assert circuits.as_noiseless_circuit(wrapped) == wrapped

    with pytest.raises(ValueError, match="between 0 and 1"):
        circuits.NoiseModel(scheduler=TICKS, readout_error=2)
    with pytest.raises(ValueError, match="Duplicate noise rules"):
        circuits.NoiseModel(
            scheduler=TICKS, rules={"CX": circuits.NoiseRule(), "cnot": circuits.NoiseRule()}
        )
    with pytest.raises(ValueError, match="Duplicate noise rules"):
        circuits.NoiseModel(
            scheduler=TICKS,
            rules={
                "M": circuits.NoiseRule(readout_error=0.1),
                "MZ": circuits.NoiseRule(readout_error=0.2),
            },
        )
    measurement_model = circuits.NoiseModel(
        scheduler=TICKS, rules={"M": circuits.NoiseRule(readout_error=0.1)}
    )
    assert measurement_model.noisy_circuit(stim.Circuit("M 0\nMPP Z1")) == stim.Circuit(
        "M(0.1) 0\nMPP(0.1) Z1"
    )
    mz_model = circuits.NoiseModel(
        scheduler=TICKS, rules={"MZ": circuits.NoiseRule(readout_error=0.2)}
    )
    assert mz_model.noisy_circuit(stim.Circuit("MZ 0\nMPP Z1")) == stim.Circuit(
        "M(0.2) 0\nMPP(0.2) Z1"
    )
    with pytest.raises(ValueError, match="Unrecognized noise rule key"):
        circuits.NoiseModel(scheduler=TICKS, rules={"CNOTT": circuits.NoiseRule()})


def test_identity_gates() -> None:
    """Identity gates get the noise of ordinary one- and two-qubit Clifford gates."""
    noise_model = circuits.NoiseModel(
        scheduler=TICKS, clifford_1q_error=0.1, clifford_2q_error=0.2, idle_error=0.3
    )
    circuit = stim.Circuit("I 0\nII 1 2\nH 3")
    noisy_circuit = stim.Circuit("""
        I 0
        II 1 2
        H 3
        DEPOLARIZE1(0.1) 0
        DEPOLARIZE2(0.2) 1 2
        DEPOLARIZE1(0.1) 3
        DEPOLARIZE1(0.3) 4
    """)
    assert circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, system_qubits=range(5))
    )


def test_idle_errors() -> None:
    """Add idling errors to a circuit."""

    circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M 0
        DETECTOR rec[-1]
    """)
    noise_model = circuits.NoiseModel(
        scheduler=TICKS, readout_error=0.1, idle_error=0.2, additional_error_waiting_for_m_or_r=0.3
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        DEPOLARIZE1(0.3) 1 2
        DEPOLARIZE1(0.2) 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    immune_coords = stim.Circuit()
    immune_coords.append("QUBIT_COORDS", 0, (0,), tag=circuits.DEFAULT_IMMUNE_QUBIT_TAG)
    annotated = stim.Circuit()
    annotated.append(stim.CircuitRepeatBlock(1, immune_coords))
    annotated.append("H", 1)
    annotated_noise = circuits.NoiseModel(scheduler=TICKS, clifford_1q_error=0.2, idle_error=0.1)
    noisy_annotated = annotated_noise.noisy_circuit(annotated)
    assert not any(
        "DEPOLARIZE" in line and line.endswith(" 0") for line in str(noisy_annotated).splitlines()
    )
    assert "DEPOLARIZE1(0.2) 1" in str(noisy_annotated)
    unprotected = annotated_noise.noisy_circuit(annotated, immune_qubit_tag=None)
    assert "DEPOLARIZE1(0.1) 0" in str(unprotected)

    moment_model = circuits.NoiseModel(scheduler=TICKS, idle_error=0.01)
    assert moment_model.noisy_circuit(stim.Circuit("QUBIT_COORDS(0) 0")) == stim.Circuit(
        "QUBIT_COORDS(0) 0"
    )
    assert moment_model.noisy_circuit(stim.Circuit("X_ERROR(0.1) 0")) == stim.Circuit(
        "X_ERROR(0.1) 0"
    )
    classical = moment_model.noisy_circuit(stim.Circuit("M 0\nCX rec[-1] 1"), system_qubits=[0, 1])
    assert "CX rec[-1] 1" in str(classical)
    sweep = moment_model.noisy_circuit(stim.Circuit("CX sweep[3] 2"), system_qubits=[3])
    assert "DEPOLARIZE1(0.01) 3" in str(sweep)
    immune_only = moment_model.noisy_circuit(
        stim.Circuit("H[__IMMUNE_TO_NOISE__] 0"), system_qubits=[0, 1]
    )
    assert "DEPOLARIZE1(0.01) 1" in str(immune_only)

    # Repeated targets inside one Pauli product are legal Stim and remain one moment atom.
    repeated_product = circuits.NoiseModel(scheduler=TICKS, idle_error=0.01).noisy_circuit(
        stim.Circuit("MPP X0*Z0")
    )
    assert repeated_product == stim.Circuit("MPP X0*Z0")

    # user-defined idling noise via NoiseRule (overrides the default DEPOLARIZE1 channel)
    idle_rule = circuits.NoiseRule(after={"X": 0.05, "Z": 0.1})
    m_or_r_rule = circuits.NoiseRule(after=circuits.PauliChannel({"X": 0.2, "Z": 0.3}))
    noise_model = circuits.NoiseModel(
        scheduler=TICKS,
        readout_error=0.1,
        idle_error=idle_rule,
        additional_error_waiting_for_m_or_r=m_or_r_rule,
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        PAULI_CHANNEL_1(0.2, 0, 0.3) 1 2
        PAULI_CHANNEL_1(0.05, 0.0, 0.1) 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # zero/None errors normalize to None for all NoiseRule-valued fields
    noise_model = circuits.NoiseModel(
        scheduler=TICKS,
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


@pytest.mark.parametrize(("probability", "moments"), [(0.01, 5), (0.75, 4), (0.1, 100_000)])
def test_aggregated_idle_depolarization(probability: float, moments: int) -> None:
    """An idle run on one qubit becomes one error, before the qubit is next used."""
    circuit = stim.Circuit("R 0\nTICK")
    circuit.append(stim.CircuitRepeatBlock(moments, stim.Circuit("X 1\nTICK")))
    circuit += stim.Circuit("M 0")
    noisy = circuits.NoiseModel(idle_error=probability).noisy_circuit(circuit, system_qubits=[0])

    expected = 0.75 * (1 - (1 - 4 * probability / 3) ** moments)
    expected_circuit = stim.Circuit(f"""
        R 0
        REPEAT {moments} {{
            TICK
            X 1
        }}
        DEPOLARIZE1({expected}) 0
        TICK
        M 0
    """)
    assert noisy.approx_equals(expected_circuit, atol=1e-12)


def test_quadratic_idle_errors() -> None:
    """Quadratic idle errors grow over each run of idle moments, and restart after an operation.

    Over two idle moments, a flip probability ``p`` per moment becomes that of four moments of an
    independent error, ``(1 - (1 - 2 * p)**4) / 2``.
    """
    circuit = stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0\nTICK\nH 1\nTICK\nM 0")
    noisy = circuits.NoiseModel(scheduler=TICKS, quadratic_idle_error={"Z": 0.01}).noisy_circuit(
        circuit
    )
    expected = stim.Circuit("""
        H 0
        Z_ERROR(0.01) 1
        TICK
        H 1
        TICK
        H 1
        Z_ERROR(0.03881592) 0
        TICK
        H 0
        Z_ERROR(0.01) 1
        TICK
        H 1
        Z_ERROR(0.01) 0
        TICK
        M 0
        Z_ERROR(0.01) 1
    """)
    assert noisy.approx_equals(expected, atol=1e-12)

    # Custom idle fragments are still applied once per moment, and a float is a uniform channel.
    model = circuits.NoiseModel(
        scheduler=TICKS,
        idle_error=circuits.NoiseRule(after=stim.Circuit("X_ERROR(0.1) 0")),
        quadratic_idle_error=0.03,
    )
    noisy = model.noisy_circuit(
        stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0"), system_qubits=[0]
    )
    expected = stim.Circuit("""
        H 0
        TICK
        H 1
        X_ERROR(0.1) 0
        TICK
        H 1
        X_ERROR(0.1) 0
        DEPOLARIZE1(0.10962552) 0
        TICK
        H 0
    """)
    assert noisy.approx_equals(expected, atol=1e-12)

    # Over a long idle run, the flip probability saturates: it is (1 - exp(-2)) / 2 rather than
    # 1000**2 * 1e-6 = 1.  It is combined with the independent idle_error into one channel.
    circuit = stim.Circuit("R 0\nTICK")
    circuit.append(stim.CircuitRepeatBlock(1000, stim.Circuit("X 1\nTICK")))
    circuit += stim.Circuit("M 0")
    model = circuits.NoiseModel(
        scheduler=TICKS, idle_error={"X": 0.01}, quadratic_idle_error={"Z": 1e-6}
    )
    p_x, p_z = (1 - 0.98**1000) / 2, (1 - math.exp(-2)) / 2
    expected = stim.Circuit(f"""
        R 0
        REPEAT 1000 {{
            TICK
            X 1
        }}
        PAULI_CHANNEL_1({p_x * (1 - p_z)}, {p_x * p_z}, {(1 - p_x) * p_z}) 0
        TICK
        M 0
    """)
    assert model.noisy_circuit(circuit, system_qubits=[0]).approx_equals(expected, atol=1e-6)


def test_noise_commutes_with_flattening() -> None:
    """Adding noise to a circuit is equivalent to adding noise to its flattened circuit.

    Moreover, combining idle errors does not change the detector error model.
    """
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0.01,
        clifford_2q_error=0.02,
        readout_error=0.03,
        idle_error={"X": 0.005, "Z": 0.007},
        quadratic_idle_error={"Z": 0.003},
        additional_error_waiting_for_m_or_r=0.04,
        scheduler=TICKS,
    )
    combined_idle_noise = circuits.NoiseModel(
        idle_error=circuits.PauliChannelSequence({"X": 0.05, "Z": 0.1}), scheduler=TICKS
    )
    per_moment_idle_noise = circuits.NoiseModel(
        idle_error=circuits.NoiseRule(after=stim.Circuit("E(0.05) X0\nE(0.1) Z0")),
        scheduler=TICKS,
    )
    rng = random.Random(0)
    for _ in range(100):
        circuit = _random_circuit(rng)
        flattened_circuit = flatten_repeat_blocks(circuit, circuits.DEFAULT_IMMUNE_OP_TAG)
        expected = noise_model.noisy_circuit(flattened_circuit)
        assert noise_model.noisy_circuit(circuit).flattened() == expected.flattened()

        combined = combined_idle_noise.noisy_circuit(circuit)
        per_moment = per_moment_idle_noise.noisy_circuit(circuit)
        assert _dem_errors(combined) == pytest.approx(_dem_errors(per_moment), abs=1e-12)


def _random_circuit(rng: random.Random) -> stim.Circuit:
    """A random circuit on three qubits with nested (immune, gate-free, and empty) repeat blocks."""

    def random_body(depth: int) -> stim.Circuit:
        body = stim.Circuit()
        for _ in range(rng.randint(0, 4)):
            qubit, choice = rng.randrange(3), rng.random()
            if choice < 0.2:
                body.append("TICK")
            elif choice < 0.5:
                gate = rng.choice(["H", "CX", "MR"])
                body.append(gate, [qubit, (qubit + 1) % 3] if gate == "CX" else [qubit])
            elif choice < 0.6:
                body.append("X_ERROR", [qubit], 0.1)
            elif choice < 0.7:
                body.append("H", [qubit], tag=circuits.DEFAULT_IMMUNE_OP_TAG)
            elif depth < 3:
                tag = rng.choice(["", "", circuits.DEFAULT_IMMUNE_OP_TAG])
                body.append(
                    stim.CircuitRepeatBlock(rng.randint(1, 5), random_body(depth + 1), tag=tag)
                )
        return body

    measurements = stim.Circuit("M 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]")
    return stim.Circuit("R 0 1 2\nTICK") + random_body(0) + measurements


def _dem_errors(circuit: stim.Circuit) -> dict[str, float]:
    """The probability of each set of flipped detectors and observables."""
    dem = circuit.detector_error_model(allow_gauge_detectors=True, flatten_loops=True)
    errors: dict[str, float] = {}
    for error in dem.flattened():
        if error.type == "error":
            key, probability = str(error.targets_copy()), error.args_copy()[0]
            old = errors.get(key, 0)
            errors[key] = old + probability - 2 * old * probability
    return errors


def test_aggregated_idle_respects_observable_barrier() -> None:
    """Idle errors are not deferred past an observable that depends on the idle qubit."""
    circuit = stim.Circuit("""
        R 0 1
        TICK
        X 1
        TICK
        OBSERVABLE_INCLUDE(0) Z0
        X 1
        TICK
        M 0 1
    """)
    noisy = circuits.NoiseModel(scheduler=TICKS, idle_error={"X": 0.1}).noisy_circuit(circuit)
    assert noisy.detector_error_model() == stim.DetectorErrorModel("error(0.1) L0")


def test_immunity() -> None:
    """Qubits and operations can be immune to errors."""

    # qubits can be immune to errors
    circuit = stim.Circuit("""
        H 0 1
        CNOT 1 2
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False, scheduler=TICKS)
    expected = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.1) 1
        TICK
        CNOT 1 2
        DEPOLARIZE2(0.1) 1 2
    """)
    assert circuits_are_equivalent(expected, noise_model.noisy_circuit(circuit, immune_qubits=[0]))

    # operations can be immune to errors
    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        CX 0 1
        H['{immune_op_tag}'] 0
        X 1
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=True, scheduler=TICKS)
    noisy_circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        DEPOLARIZE1(0.1) 1
        TICK
        CX 0 1
        DEPOLARIZE2(0.1) 0 1
        H['{immune_op_tag}'] 0
        X 1
        DEPOLARIZE1(0.1) 1
    """)
    assert circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag)
    )
    tagged_gate = stim.Circuit()
    tagged_gate.append("H", 0, tag=immune_op_tag)
    assert circuits_are_equivalent(
        stim.Circuit(f"H[{immune_op_tag}] 0\nDEPOLARIZE1(0.1) 0"),
        noise_model.noisy_circuit(tagged_gate, immune_op_tag=None),
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
    dm = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False, scheduler=TICKS)
    assert circuits_are_equivalent(
        stim.Circuit("""
            CNOT 0 1
            TICK
            CNOT 1 2
            CNOT 3 4
            DEPOLARIZE2(0.1) 3 4
        """),
        dm.noisy_circuit(circuit, immune_qubits=[0, 1], immunize_gates=True),
    )
    assert circuits_are_equivalent(
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
    # False the channel is projected via ``PauliChannel.conditioned_on`` — surviving strings (those
    # with I at the immune position) emit as native PAULI_CHANNEL_2 on the outer qubits.
    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03})
    spp_circuit = stim.Circuit("SPP X0*Y1*Z2")
    spp_model = circuits.NoiseModel(
        scheduler=TICKS, rules={"SPP": circuits.NoiseRule(after=channel)}
    )
    assert circuits_are_equivalent(
        spp_circuit,
        spp_model.noisy_circuit(spp_circuit, immune_qubits=[1], immunize_gates=True),
    )
    assert circuits_are_equivalent(
        stim.Circuit(
            "SPP X0*Y1*Z2\nPAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0) 0 2"
        ),
        spp_model.noisy_circuit(spp_circuit, immune_qubits=[1], immunize_gates=False),
    )

    # reset_error broadcasts X_ERROR; immune targets are stripped per atom under True.
    assert circuits_are_equivalent(
        stim.Circuit("R 0 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.1) 2"),
        circuits.NoiseModel(scheduler=TICKS, reset_error=0.1).noisy_circuit(
            stim.Circuit("R 0 1 2"), immune_qubits=[1]
        ),
    )

    # Under immunize_gates=True, any measurement touching an immune qubit is dropped entirely —
    # readout_error included.  `M 0` keeps its readout error; `M 1` (fully immune) drops it;
    # `MPP X0*Y1` (partial immunity) also drops it, matching the dead-simple policy.
    assert circuits_are_equivalent(
        stim.Circuit("M(0.1) 0\nTICK\nM 1\nTICK\nMPP X0*Y1"),
        circuits.NoiseModel(scheduler=TICKS, readout_error=0.1).noisy_circuit(
            stim.Circuit("M 0\nTICK\nM 1\nTICK\nMPP X0*Y1"), immune_qubits=[1]
        ),
    )
    # Under immunize_gates=False, the same MPP with partial immunity has no clean projection
    # (readout_error can't be conditioned), so we raise.
    with pytest.raises(ValueError, match="partial immunity"):
        circuits.NoiseModel(scheduler=TICKS, readout_error=0.1).noisy_circuit(
            stim.Circuit("MPP X0*Y1"), immune_qubits=[1], immunize_gates=False
        )


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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # multi-product SPP is split so each Pauli product is treated independently
    circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
    """)
    noisy_circuit = stim.Circuit("""
        SPP X0 Y1*Z2 X3*Y4*Z5
        DEPOLARIZE1(0.1) 0
        DEPOLARIZE2(0.2) 1 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # Explicit rules must match a single Pauli-product width; they are not broadcast over product
    # factors.
    noise_rule = circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.3))
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, rules={"SPP": noise_rule})
    circuit = stim.Circuit("""
        SPP X0 X0*Y1*Z2
    """)
    with pytest.raises(ValueError, match="has arity 1, but"):
        noise_model.noisy_circuit(circuit)

    measurement_rule = circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.1))
    measurement_model = circuits.NoiseModel(
        rules={"MXX": measurement_rule, "MYY": measurement_rule, "MZZ": measurement_rule}
    )
    measured = measurement_model.noisy_circuit(stim.Circuit("MXX 0 1\nMYY 2 3\nMZZ 4 5"))
    assert str(measured).count("DEPOLARIZE2") == 1
    assert all(str(qubit) in str(measured) for qubit in range(6))
    with pytest.raises(ValueError, match="readout_error"):
        circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(readout_error=0.1)}).noisy_circuit(
            stim.Circuit("SPP X0")
        )
    with pytest.raises(ValueError, match="reset_error"):
        circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(reset_error=0.1)}).noisy_circuit(
            stim.Circuit("SPP X0")
        )


def test_correlated_error_chains_are_atomic_when_re_noised() -> None:
    """TICK insertion never splits an E/ELSE_CORRELATED_ERROR mechanism."""
    model = circuits.NoiseModel(clifford_nq_error={3: 0.01})
    first = model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    second = model.noisy_circuit(first)
    instruction_names = [instruction.name for instruction in second]
    assert "ELSE_CORRELATED_ERROR" in instruction_names
    assert all(
        instruction_names[index - 1] in ("E", "ELSE_CORRELATED_ERROR")
        for index, name in enumerate(instruction_names)
        if name == "ELSE_CORRELATED_ERROR"
    )
    second.detector_error_model(approximate_disjoint_errors=True)
    decoders.DetectorErrorModelArrays(first)


def test_rule_func() -> None:
    """A user-provided rule_func assigns noise per gate application, with top priority."""

    # Broadcast gates are decomposed into individual applications before the callback is invoked; it
    # is consulted once per application (per qubit for a 1q gate).  Returning None falls back to the
    # ordinary clifford_1q_error.
    seen: list[tuple[str, str, tuple[int, ...]]] = []

    def per_qubit(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        targets = [t for t in op.targets_copy() if not t.is_combiner]
        seen.append((op.name, op.tag, tuple(t.qubit_value for t in targets)))
        if op.name == "H" and targets[0].qubit_value == 1:
            return circuits.NoiseRule(after={"X": 0.5})
        return None

    noise_model = circuits.NoiseModel(scheduler=TICKS, clifford_1q_error=0.1, rule_func=per_qubit)
    circuit = stim.Circuit("H 0 1 2")
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        DEPOLARIZE1(0.1) 0
        X_ERROR(0.5) 1
        DEPOLARIZE1(0.1) 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    assert seen == [("H", "", (0,)), ("H", "", (1,)), ("H", "", (2,))]

    # A two-qubit gate is decomposed per pair; the callback sees both targets and may return a
    # two-qubit rule.  Other pairs fall back to clifford_2q_error.
    def per_pair(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        targets = op.targets_copy()
        if op.name == "CX" and {t.qubit_value for t in targets} == {2, 3}:
            return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.9))
        return None

    noise_model = circuits.NoiseModel(scheduler=TICKS, clifford_2q_error=0.1, rule_func=per_pair)
    circuit = stim.Circuit("CX 0 1 2 3")
    noisy_circuit = stim.Circuit("""
        CX 0 1 2 3
        DEPOLARIZE2(0.1) 0 1
        DEPOLARIZE2(0.9) 2 3
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # The callback takes top priority over `rules`, and receives the instruction's tag.
    def tagged(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        assert op.tag == "mytag"
        return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.7))

    noise_model = circuits.NoiseModel(
        scheduler=TICKS,
        rules={"H": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.2))},
        rule_func=tagged,
    )
    circuit = stim.Circuit()
    circuit.append("H", [0], tag="mytag")
    noisy_circuit = stim.Circuit()
    noisy_circuit.append("H", [0], tag="mytag")
    noisy_circuit.append("DEPOLARIZE1", [0], [0.7])
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # Measurement and reset errors can be assigned per application too.  The callback receives the
    # raw op.name (`M`, `MZ`, `R`, etc.); if the user wants alias-agnostic matching they can either
    # check every alias or use the same key that `rules` uses (stim's canonical name).
    def measure_reset(op: stim.CircuitInstruction) -> circuits.NoiseRule | None:
        if op.name in ("R", "RZ"):
            return circuits.NoiseRule(reset_error=0.3)
        if op.name in ("M", "MZ") and op.targets_copy()[0].qubit_value == 0:
            return circuits.NoiseRule(readout_error=0.25)
        return None

    noise_model = circuits.NoiseModel(scheduler=TICKS, readout_error=0.01, rule_func=measure_reset)
    circuit = stim.Circuit("R 2\nTICK\nM 0 1")
    noisy_circuit = stim.Circuit("""
        R 2
        X_ERROR(0.3) 2
        TICK
        M(0.25) 0
        M(0.01) 1
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # Noise-immune qubits still take precedence over the callback.
    def kick(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.5))

    noise_model = circuits.NoiseModel(scheduler=TICKS, rule_func=kick)
    circuit = stim.Circuit("H 0 1")
    noisy_circuit = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.5) 0
    """)
    assert circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits={1})
    )

    # The callback sees gates, including identities, but not annotations or classically-controlled
    # ops.
    consulted: list[str] = []

    def record(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        consulted.append(op.name)
        return circuits.NoiseRule(after={"X": 0.5})

    noise_model = circuits.NoiseModel(scheduler=TICKS, rule_func=record)
    noise_model.noisy_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nI 2\nM 0\nCX rec[-1] 1"))
    assert consulted == ["I", "M"]

    # An identity's after-noise precedes a subsequent operation on the same qubit.
    identity_rule = circuits.NoiseModel(
        scheduler=TICKS,
        rule_func=lambda op: circuits.NoiseRule(after={"X": 1.0}) if op.name == "I" else None,
    )
    circuit = stim.Circuit("I 0\nM 0")
    noisy_circuit = stim.Circuit("I 0\nX_ERROR(1) 0\nTICK\nM 0")
    assert circuits_are_equivalent(noisy_circuit, identity_rule.noisy_circuit(circuit))

    # A returned rule's readout_error/reset_error must match the gate it is assigned to.
    bad_readout = circuits.NoiseModel(
        scheduler=TICKS, rule_func=lambda op: circuits.NoiseRule(readout_error=0.1)
    )
    with pytest.raises(ValueError, match=r"rule for 'H'.*readout_error.*measurement gates"):
        bad_readout.noisy_circuit(stim.Circuit("H 0"))
    bad_reset = circuits.NoiseModel(
        scheduler=TICKS, rule_func=lambda op: circuits.NoiseRule(reset_error=0.1)
    )
    with pytest.raises(ValueError, match=r"rule for 'H'.*reset_error.*reset gates"):
        bad_reset.noisy_circuit(stim.Circuit("H 0"))

    # A returned rule's `after` arity must match the gate application's qubit count.
    bad_arity = circuits.NoiseModel(
        scheduler=TICKS,
        rule_func=lambda op: circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.1)),
    )
    with pytest.raises(ValueError, match=r"rule for 'CX'.*`after` has arity 1"):
        bad_arity.noisy_circuit(stim.Circuit("CX 0 1"))

    # SXYZ can be used as a `rules` key just like MXYZ.
    noise_model = circuits.NoiseModel(
        scheduler=TICKS,
        rules={"SXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(3, 0.1))},
    )
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    assert "CORRELATED_ERROR" in str(noisy_circuit)
    assert "ELSE_CORRELATED_ERROR" in str(noisy_circuit)

    # The recursive noisy_circuit call inside a REPEAT block forwards immune_op_tag.
    kick_all = circuits.NoiseModel(
        scheduler=TICKS, rule_func=lambda op: circuits.NoiseRule(after={"X": 0.5})
    )
    body = stim.Circuit()
    body.append("H", [0], tag="__skip__")
    body.append("TICK")
    repeat_circuit = stim.Circuit()
    repeat_circuit.append(stim.CircuitRepeatBlock(repeat_count=2, body=body))
    result = kick_all.noisy_circuit(repeat_circuit, immune_op_tag="__skip__")
    assert "X_ERROR" not in str(result)


def test_clifford_nq_error() -> None:
    """clifford_nq_error dispatches by qubit count k."""

    # A float value produces a uniform depolarizing channel of that weight
    noise_model = circuits.NoiseModel(clifford_nq_error={3: 0.001})
    circuit = stim.Circuit("SPP X0*Y1*Z2")
    noisy = noise_model.noisy_circuit(circuit)
    # 63 non-identity 3q Paulis: 1 CORRELATED_ERROR (canonicalized by stim to "E") plus 62
    # ELSE_CORRELATED_ERROR
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
    assert circuits_are_equivalent(expected, noisy)

    # The stim.Circuit form of `after` is an escape hatch: broadcast noise + a joint CE chain can be
    # combined by spelling out targets in one fragment.
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
    assert circuits_are_equivalent(expected, noisy)

    # Falsy values (zero float, empty NoiseRule) are dropped from the normalized dict
    assert circuits.NoiseModel(clifford_nq_error={3: 0.0}).clifford_nq_error == {}
    assert circuits.NoiseModel(clifford_nq_error={3: circuits.NoiseRule()}).clifford_nq_error == {}

    # clifford_1q_error / clifford_2q_error also accept a PauliChannel directly, or a raw
    # Pauli-string Mapping (auto-wrapped as PauliChannel).
    noise_model = circuits.NoiseModel(
        clifford_1q_error=circuits.PauliChannel({"X": 0.01}),
        clifford_2q_error={"XY": 0.02},
    )
    assert circuits_are_equivalent(
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
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.0})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.0, clifford_nq_error={1: 0.01})
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_2q_error=0.01, clifford_nq_error={2: 0.02})

    # bool rejection
    with pytest.raises(TypeError, match="unsupported type bool"):
        circuits.NoiseModel(clifford_nq_error={2: True})

    # num_qubits mismatch between key and PauliChannel value (checked at construction).
    with pytest.raises(ValueError, match="arity 2; expected 3"):
        circuits.NoiseModel(clifford_nq_error={3: circuits.PauliChannel.depolarizing(2, 0.01)})

    # num_qubits mismatch via `rules=` is not caught at construction (arity varies), but is caught
    # at emission by `emit_after`.
    bad_rule = circuits.NoiseRule(after=circuits.PauliChannel({"XY": 0.01}))
    noise_model = circuits.NoiseModel(rules={"SPP": bad_rule})
    with pytest.raises(ValueError, match="has arity 2, but"):
        noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))

    # A 2-qubit `after` broadcast (via a 2q PauliChannel) applied to a wrong-arity basis-suffixed
    # rule key raises at construction time ("MXYZ" is arity 3 — a 2q `after` is caught up front).
    with pytest.raises(ValueError, match="arity 2; expected 3"):
        circuits.NoiseModel(
            rules={"MXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        )
    # Bare "MPP" / "SPP" remain variable-arity; wrong-arity `after` is caught at emission.
    with pytest.raises(ValueError, match="has arity 2, but"):
        circuits.NoiseModel(
            rules={"MPP": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        ).noisy_circuit(stim.Circuit("MPP X0*Y1*Z2"))


def test_pauli_channel_idle_error_rejection() -> None:
    """Multi-qubit `after` rules are not accepted on idle-error rules (all shapes rejected)."""
    channel = circuits.PauliChannel({"XY": 0.01})
    with pytest.raises(ValueError, match=r"idle_error.*multi-qubit"):
        circuits.NoiseModel(idle_error=circuits.NoiseRule(after=channel))
    with pytest.raises(ValueError, match=r"additional_error_waiting_for_m_or_r.*multi-qubit"):
        circuits.NoiseModel(additional_error_waiting_for_m_or_r=circuits.NoiseRule(after=channel))
    # stim.Circuit form is rejected on the same grounds
    with pytest.raises(ValueError, match=r"idle_error.*multi-qubit"):
        circuits.NoiseModel(
            idle_error=circuits.NoiseRule(after=stim.Circuit("DEPOLARIZE2(0.1) 0 1"))
        )


def test_quadratic_idle_error_runs() -> None:
    """An idle run continues past instructions that commute with Pauli errors.

    Noise channels and classically controlled Paulis on an idle qubit do not depend on when a Pauli
    error occurs, so the error of the qubit's whole run is the same as if they were absent.  A Pauli
    annotation depends on that, so it ends the run.
    """
    noise_model = circuits.NoiseModel(scheduler=TICKS, quadratic_idle_error={"Z": 0.01})
    step = "H 1\nTICK\n"

    def get_error(interruption: str) -> float:
        circuit = stim.Circuit(f"R 0\nTICK\n{step * 2}{interruption}\n{step * 3}M 0")
        return _get_z_error(noise_model.noisy_circuit(circuit), 0)

    for commuting in ["X_ERROR(0.2) 0", "M 2\nCX rec[-1] 0"]:
        assert get_error(commuting) == pytest.approx((1 - 0.98**25) / 2)
    q_2, q_3 = (1 - 0.98**4) / 2, (1 - 0.98**9) / 2
    assert get_error("OBSERVABLE_INCLUDE(0) Z0") == pytest.approx(q_2 + q_3 - 2 * q_2 * q_3)

    # a repeat block that applies noise to an idle qubit stays compact
    circuit = stim.Circuit("R 0")
    circuit.append(stim.CircuitRepeatBlock(10**4, stim.Circuit("TICK\nX_ERROR(0) 0\nH 1")))
    noisy = noise_model.noisy_circuit(circuit)
    assert len(str(noisy).splitlines()) < 20
    assert _get_z_error(noisy, 0) == pytest.approx(0.5)


def _get_z_error(circuit: stim.Circuit, qubit: int) -> float:
    """The net probability of independent Z errors on a qubit."""
    probability = 0.0
    for instruction in circuit.flattened():
        if instruction.name == "Z_ERROR" and stim.GateTarget(qubit) in instruction.targets_copy():
            new = instruction.gate_args_copy()[0]
            probability += new - 2 * probability * new
    return probability


def test_quadratic_idle_error_specifications() -> None:
    """A quadratic idle error is a one-qubit Pauli channel that flips a qubit at most half the time.

    Equivalent specifications give the same noise, including totals of 1/2 up to rounding.
    """
    circuit = stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0")
    noisy = circuits.NoiseModel(quadratic_idle_error={"Z": 0.5}).noisy_circuit(circuit)
    model = circuits.NoiseModel(quadratic_idle_error=circuits.PauliChannel({"Z": 0.5}))
    assert model.noisy_circuit(circuit) == noisy
    uniform = circuits.NoiseModel(quadratic_idle_error={"X": 1 / 6, "Y": 1 / 6, "Z": 1 / 6})
    assert uniform.noisy_circuit(circuit) == circuits.NoiseModel(
        quadratic_idle_error=0.5
    ).noisy_circuit(circuit)
    assert circuits.NoiseModel(quadratic_idle_error={"X": 0.25, "Z": 0.25 + 1e-16})
    assert circuits.NoiseModel(quadratic_idle_error=False).quadratic_idle_error is None

    sequence = circuits.PauliChannelSequence({"Z": 0.01})
    for invalid in ({"X": 0.3, "Z": 0.3}, {"XY": 0.01}, sequence):
        with pytest.raises(ValueError, match="quadratic_idle_error"):
            circuits.NoiseModel(quadratic_idle_error=invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        circuits.NoiseModel(quadratic_idle_error=True)


def test_trivial_noise() -> None:
    """A NoiseModel with only a rule_func is truthy even before the func is consulted."""
    assert not bool(circuits.NoiseRule())
    assert not bool(circuits.NoiseModel())
    assert not bool(circuits.NoiseModel(quadratic_idle_error=0))
    assert bool(circuits.NoiseModel(quadratic_idle_error=0.01))
    assert bool(circuits.NoiseModel(rule_func=lambda op: None))


def test_wrapping_circuits() -> None:
    """noisy_circuit and as_noiseless_circuit accept and return stim-wrapping circuits."""
    noise_model = circuits.DepolarizingNoiseModel(0.01)

    stim_circuit = stim.Circuit("H 0\nCX 0 1\nM 0 1")
    wrapped_circuit = StimCircuitWrapper(stim_circuit)

    stim_noisy = noise_model.noisy_circuit(stim_circuit)
    wrapped_noisy = noise_model.noisy_circuit(wrapped_circuit)
    assert isinstance(stim_noisy, stim.Circuit)
    assert isinstance(wrapped_noisy, StimCircuitWrapper)
    assert stim_noisy == wrapped_noisy.stim_circuit

    stim_noiseless = circuits.as_noiseless_circuit(stim_circuit)
    wrapped_noiseless = circuits.as_noiseless_circuit(wrapped_circuit)
    assert isinstance(stim_noiseless, stim.Circuit)
    assert isinstance(wrapped_noiseless, StimCircuitWrapper)
    assert stim_noiseless == wrapped_noiseless.stim_circuit


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
        quadratic_idle_error={"Z": 0.001},
        additional_error_waiting_for_m_or_r=0.02,
        rules={"SXY": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.05))},
    )
    circuit = stim.Circuit("R 0 1\nH 0\nCX 0 1\nTICK\nM 0 1")
    # This deserializes only the in-memory object constructed above, never untrusted external data.
    round_tripped = pickle.loads(pickle.dumps(model))  # noqa: S301
    for restored in (round_tripped, copy.deepcopy(model)):
        assert circuits_are_equivalent(
            model.noisy_circuit(circuit), restored.noisy_circuit(circuit)
        )

    # The one caveat: a rule_func that isn't importable at module scope (a local closure or a
    # lambda) can't be pickled.  That limitation is inherent to pickle, not to NoiseModel — a
    # module-level function would pickle fine.
    func_model = circuits.NoiseModel(clifford_1q_error=0.1, rule_func=lambda op: None)
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(func_model)


def test_scheduler() -> None:
    """A scheduler sets the moments in which qubits idle.

    By default, operations are as early as possible and resets as late as possible, so qubit 1 does
    not idle before its first gate.  Keeping the moments of the circuit, it idles for a moment.
    """
    circuit = stim.Circuit("R 0 1\nTICK\nH 0\nTICK\nCX 0 1\nTICK\nM 0 1")
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1})
    assert noise_model.noisy_circuit(circuit) == stim.Circuit("""
        R 0 1
        H 0
        TICK
        CX 0 1
        TICK
        M 0 1
    """)
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=TICKS)
    assert noise_model.noisy_circuit(circuit) == stim.Circuit("""
        R 0 1
        TICK
        H 0
        Z_ERROR(0.1) 1
        TICK
        CX 0 1
        TICK
        M 0 1
    """)
    assert circuits.SI1000NoiseModel(0.1).scheduler == TICKS


def test_custom_scheduler() -> None:
    """A custom scheduler sets moments, and by default classically controlled gates take no time."""
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=_OneOperationPerMoment())
    circuit = stim.Circuit("H 0\nM 0\nCX rec[-1] 1\nH 1")
    expected = stim.Circuit("""
        H 0
        TICK
        M 0
        TICK
        CX rec[-1] 1
        Z_ERROR(0.244) 1
        TICK
        H 1
        Z_ERROR(0.18) 0
    """)
    assert noise_model.noisy_circuit(circuit).approx_equals(expected, atol=1e-12)


class _OneOperationPerMoment(circuits.Scheduler):
    """A scheduler that puts every operation in a moment of its own."""

    def schedule(self, circuit: circuits.StimCircuitLike) -> circuits.StimCircuitLike:
        scheduled = type(circuit)()
        for index in range(len(circuit)):
            scheduled.append(circuit[index])
            scheduled.append("TICK")
        return scheduled


def test_untimed_gates() -> None:
    """Gates that take no time share a moment with another operation, and leave a qubit idle.

    The noise of each gate precedes the next gate on its qubits.
    """
    noise_model = circuits.NoiseModel(
        clifford_1q_error={"X": 0.01},
        clifford_2q_error={"ZZ": 0.02},
        idle_error={"Z": 0.1},
        scheduler=circuits.Scheduler.preserve_ticks(),
    )
    circuit = stim.Circuit("H 0\nCX 0 1\nH 0 2\nTICK\nCX 1 2")
    assert noise_model.noisy_circuit(circuit) == stim.Circuit("""
        H 0
        X_ERROR(0.01) 0
        CX 0 1
        PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02) 0 1
        H 0 2
        X_ERROR(0.01) 0 2
        Z_ERROR(0.1) 2
        TICK
        CX 1 2
        PAULI_CHANNEL_2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02) 1 2
        Z_ERROR(0.1) 0
    """)

    # repeated gates on a qubit get noise after each of them
    noise_model = circuits.NoiseModel(
        clifford_1q_error={"Z": 0.01}, scheduler=circuits.Scheduler.preserve_ticks()
    )
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("H 0 0"))
    assert noisy_circuit == stim.Circuit("H 0\nZ_ERROR(0.01) 0\nH 0\nZ_ERROR(0.01) 0")

    # I takes no time, II takes a moment, and a moment between TICKs lasts a moment regardless of
    # its operations, so qubits idle only in the second moment
    noise_model = circuits.NoiseModel(
        idle_error={"Z": 0.1}, scheduler=circuits.Scheduler.preserve_ticks()
    )
    circuit = stim.Circuit("I 0\nCX 0 1\nTICK\nI 0\nTICK\nII 0 1\nCX 0 1")
    assert noise_model.noisy_circuit(circuit) == stim.Circuit("""
        I 0
        CX 0 1
        TICK
        I 0
        Z_ERROR(0.1) 0 1
        TICK
        II 0 1
        TICK
        CX 0 1
    """)

    # a classically controlled gate takes no time
    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nCZ 1 2")
    assert noise_model.noisy_circuit(circuit) == circuit


def test_noise_model_str() -> None:
    """NoiseModel.__str__ reads like a constructor call listing only the set fields."""
    assert str(circuits.NoiseModel()) == "NoiseModel()"

    model = circuits.NoiseModel(
        clifford_1q_error=0.1,
        readout_error=0.3,
        reset_error=0.4,
        idle_error=0.01,
        quadratic_idle_error={"Z": 0.001},
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
        "quadratic_idle_error=",
        "additional_error_waiting_for_m_or_r=",
        "rules={'M': NoiseRule(readout_error=0.05)}",
    ):
        assert field in text
    assert "rule_func" not in text and "scheduler" not in text
    scheduler = circuits.Scheduler.alap()
    assert f"scheduler={scheduler!r}" in str(circuits.NoiseModel(scheduler=scheduler))

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
