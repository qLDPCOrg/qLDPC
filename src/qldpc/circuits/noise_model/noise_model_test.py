# SPDX-License-Identifier: Apache-2.0

"""Unit tests for noise_model.py."""

import copy
import math
import pickle
import warnings

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import (
    StimCircuitWrapper,
    circuits_are_equivalent,
)


def test_gate_errors() -> None:
    """Add gate errors to a circuit, before the next use of their qubits.

    Gate errors compose with the noise of a noisy circuit, and invalid arguments and unknown rule
    keys are rejected.
    """
    assert circuits.DEFAULT_IMMUNE_OP_TAG == "__IMMUNE_TO_NOISE__"
    assert circuits.DEFAULT_IMMUNE_QUBIT_TAG == "__IMMUNE_QUBIT_TO_NOISE__"

    # gate noise is inserted without TICKs, splitting a broadcast gate only where a qubit is reused
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
        TICK
        M(0.3) 0
        DEPOLARIZE2(0.2) 1 2
        RX 1
        MR(0.3) 2
        Z_ERROR(0.4) 1
        X_ERROR(0.4) 2
    """)
    assert noise_model.noisy_circuit(circuit) == noisy_circuit

    # compose gate errors
    p_m = 0.1
    double_p_m = 2 * p_m - 2 * p_m**2
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
        circuits.NoiseModel(readout_error=2)
    measurement_model = circuits.NoiseModel(rules={"M": circuits.NoiseRule(readout_error=0.1)})
    assert measurement_model.noisy_circuit(stim.Circuit("M 0\nMPP Z1")) == stim.Circuit(
        "M(0.1) 0\nMPP(0.1) Z1"
    )
    with pytest.raises(ValueError, match="Unrecognized noise rule key"):
        circuits.NoiseModel(rules={"CNOTT": circuits.NoiseRule()})


def test_identity_gates() -> None:
    """Identity gates get the noise and timing of ordinary one- and two-qubit Clifford gates."""
    noise_model = circuits.NoiseModel(
        clifford_1q_error=0.1,
        clifford_2q_error=0.2,
        idle_error=0.3,
        scheduler=circuits.TickScheduler(),
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
    assert noise_model.noisy_circuit(circuit, system_qubits=range(5)) == noisy_circuit


def test_idle_errors() -> None:
    """A qubit gets the idle noise of the moments between its operations before the next one.

    A qubit also waits in each moment in which another qubit is measured or reset, including while
    it undergoes a gate.  Single-qubit gates take no time by default, idle errors may be custom
    rules, tagged coordinate annotations make qubits immune, annotations and noise channels join the
    moment of a gate, classically controlled gates take no time, and zero errors are no noise.
    """
    ticks = circuits.TickScheduler()

    # qubit 1 waits while qubit 0 is measured, and qubit 2 also idles
    circuit = stim.Circuit("H 0 1 2\nTICK\nH 1\nM 0\nDETECTOR rec[-1]")
    noise_model = circuits.NoiseModel(
        readout_error=0.1, idle_error=0.2, additional_error_waiting_for_m_or_r=0.3, scheduler=ticks
    )
    noisy_circuit = stim.Circuit("""
        H 0 1 2
        TICK
        H 1
        M(0.1) 0
        DETECTOR rec[-1]
        DEPOLARIZE1(0.3) 1
        DEPOLARIZE1(0.42) 2
    """)
    assert noise_model.noisy_circuit(circuit).approx_equals(noisy_circuit, atol=1e-12)

    # by default, single-qubit gates take no time, so a qubit idles while it undergoes one
    noise_model = circuits.NoiseModel(idle_error=0.1)
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("H 0\nCX 1 2"))
    assert noisy_circuit == stim.Circuit("H 0\nCX 1 2\nDEPOLARIZE1(0.1) 0")

    # a user-defined idle error
    noise_model = circuits.NoiseModel(
        idle_error=circuits.NoiseRule(after={"X": 0.05, "Z": 0.1}), scheduler=ticks
    )
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("H 0\nTICK\nH 1"))
    assert noisy_circuit == stim.Circuit(
        "H 0\nTICK\nPAULI_CHANNEL_1(0.05, 0, 0.1) 1\nH 1\nPAULI_CHANNEL_1(0.05, 0, 0.1) 0"
    )

    # qubits of tagged coordinate annotations are immune to idle noise
    annotated = stim.Circuit()
    annotated.append("QUBIT_COORDS", 0, (0,), tag=circuits.DEFAULT_IMMUNE_QUBIT_TAG)
    annotated.append("H", 1)
    annotated_noise = circuits.NoiseModel(clifford_1q_error=0.2, idle_error=0.1)
    gate_noise = stim.Circuit("DEPOLARIZE1(0.2) 1")
    assert annotated_noise.noisy_circuit(annotated) == annotated + gate_noise + stim.Circuit(
        "DEPOLARIZE1(0.1) 1"
    )
    assert annotated_noise.noisy_circuit(
        annotated, immune_qubit_tag=None
    ) == annotated + gate_noise + stim.Circuit("DEPOLARIZE1(0.1) 0 1")

    # annotations and noise channels join the moment of a gate, while classically controlled gates
    # take no time
    moment_model = circuits.NoiseModel(idle_error=0.01)
    circuit = stim.Circuit("CX 0 1\nX_ERROR(0.1) 2\nQUBIT_COORDS(0) 2")
    assert moment_model.noisy_circuit(circuit) == circuit + stim.Circuit("DEPOLARIZE1(0.01) 2")
    sweep = moment_model.noisy_circuit(stim.Circuit("CX sweep[3] 2"), system_qubits=[3])
    assert sweep == stim.Circuit("CX sweep[3] 2\nDEPOLARIZE1(0.01) 3")
    immune_only = moment_model.noisy_circuit(
        stim.Circuit("CX[__IMMUNE_TO_NOISE__] 0 1"), system_qubits=[0, 1, 2]
    )
    assert immune_only == stim.Circuit("CX[__IMMUNE_TO_NOISE__] 0 1\nDEPOLARIZE1(0.01) 2")

    # Repeated targets inside one Pauli product are legal Stim and remain one operation.
    repeated_product = circuits.NoiseModel(idle_error=0.01).noisy_circuit(stim.Circuit("MPP X0*Z0"))
    assert repeated_product == stim.Circuit("MPP X0*Z0")

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


def test_classical_controls() -> None:
    """Classically controlled gates are noiseless, and their qubits idle."""
    noise_model: circuits.NoiseModel = circuits.SI1000NoiseModel(0.1)
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("CX 0 1 rec[-1] 2"))
    assert noisy_circuit == stim.Circuit(
        "CX 0 1 rec[-1] 2\nDEPOLARIZE2(0.1) 0 1\nDEPOLARIZE1(0.01) 2"
    )

    noise_model = circuits.NoiseModel(
        idle_error=0.1,
        scheduler=circuits.TickScheduler(),
    )
    circuit = stim.Circuit("H 0\nCX rec[-1] 1\nTICK\nH 0 1 2")
    noisy_circuit = stim.Circuit("H 0\nCX rec[-1] 1\nTICK\nDEPOLARIZE1(0.1) 1 2\nH 0 1 2")
    assert noise_model.noisy_circuit(circuit) == noisy_circuit


def test_pauli_product_measurements() -> None:
    """Pauli product measurements get readout errors, and rules for specific Pauli products."""
    noise_model = circuits.NoiseModel(readout_error=0.1, idle_error=0.2)
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("MPP X1*Y2*Z3"))
    assert noisy_circuit == stim.Circuit("MPP(0.1) X1*Y2*Z3\nDEPOLARIZE1(0.2) 0")

    noise_rule = circuits.NoiseRule(readout_error=0.2)
    noise_model = circuits.NoiseModel(readout_error=0.1, rules={"MXYZ": noise_rule})
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("MPP Z0*Z1*Z2\nMPP X0*Y1*Z2"))
    assert noisy_circuit == stim.Circuit("MPP(0.1) Z0*Z1*Z2\nMPP(0.2) X0*Y1*Z2")


def test_aggregated_idle_depolarization() -> None:
    """An idle run on one qubit becomes one error, before the qubit is next used."""
    probability, moments = 0.01, 5
    circuit = stim.Circuit("R 0\nTICK\n" + "X 1\nTICK\n" * moments + "M 0")
    noise_model = circuits.NoiseModel(idle_error=probability, scheduler=circuits.TickScheduler())
    noisy = noise_model.noisy_circuit(circuit, system_qubits=[0])
    expected = 0.75 * (1 - (1 - 4 * probability / 3) ** moments)
    assert noisy.approx_equals(
        circuit[:-1] + stim.Circuit(f"DEPOLARIZE1({expected}) 0\nM 0"), atol=1e-12
    )


def test_empty_moments() -> None:
    """Every moment of a schedule lasts one unit of time, including a moment without operations.

    A custom scheduler may schedule a circuit by laying it out with TICKs.
    """
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=_Spaced())
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("H 0\nS 1"))
    two_moments = 2 * 0.1 * 0.9
    expected = stim.Circuit(f"H 0\nZ_ERROR({two_moments}) 1\nS 1\nZ_ERROR({two_moments}) 0")
    assert noisy_circuit.approx_equals(expected, atol=1e-12)


class _Spaced(circuits.Scheduler):
    """A scheduler that leaves an empty moment between consecutive instructions."""

    def get_schedule(self, circuit: circuits.StimCircuitLike) -> circuits.Schedule:
        timed_circuit = stim.Circuit()
        for index in range(len(circuit)):
            timed_circuit.append(circuit[index])
            timed_circuit += stim.Circuit("TICK\nTICK")
        return circuits.TickScheduler().get_schedule(timed_circuit)


def test_moments_at_repeat_blocks() -> None:
    """The boundaries of REPEAT blocks are boundaries between moments, also for idle noise.

    With a TICK at each boundary, as in qLDPC's memory experiments, the moments are those of the
    flattened circuit.
    """
    noise_model = circuits.NoiseModel(
        idle_error={"Z": 0.1},
        scheduler=circuits.TickScheduler(),
    )
    circuit = stim.Circuit("H 0\nREPEAT 1 {\nH 1\n}")
    noisy_block = stim.Circuit("Z_ERROR(0.1) 1\nREPEAT 1 {\nH 1\nZ_ERROR(0.1) 0\n}")
    assert noise_model.noisy_circuit(circuit) == circuit[:1] + noisy_block
    assert noise_model.noisy_circuit(circuit.flattened()) == stim.Circuit("H 0 1")

    circuit = stim.Circuit("H 0\nTICK\nREPEAT 1 {\nH 1\n}")
    assert noise_model.noisy_circuit(circuit) == circuit[:2] + noisy_block
    assert noise_model.noisy_circuit(circuit.flattened()) == stim.Circuit(
        "H 0\nTICK\nZ_ERROR(0.1) 1\nH 1\nZ_ERROR(0.1) 0"
    )


def test_repeat_blocks_without_gates() -> None:
    """A REPEAT body with only noise channels has no moments, so qubits do not idle during it."""
    noise_model = circuits.NoiseModel(
        idle_error={"Z": 0.1},
        scheduler=circuits.TickScheduler(),
    )
    circuit = stim.Circuit("H 0 1\nTICK\nREPEAT 5 {\nX_ERROR(0.01) 0\nTICK\n}\nH 0 1")
    assert noise_model.noisy_circuit(circuit) == circuit


def test_idle_runs_at_repeat_blocks() -> None:
    """Gate noise is emitted by the end of each iteration of a REPEAT block.

    Qubits that are busy in every moment do not idle, also in nested REPEAT blocks.  A noiseless
    REPEAT block gets no warning about quadratic idle errors.
    """
    ticks = circuits.TickScheduler()
    circuit = stim.Circuit("R 0\nTICK\nREPEAT 3 {\nM 1\nTICK\n}\nM 0\nDETECTOR rec[-1]")
    noise_model = circuits.NoiseModel(quadratic_idle_error={"Z": 0.01}, scheduler=ticks)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        noise_model.noisy_circuit(circuits.as_noiseless_circuit(circuit))

    # qubits that are busy in every moment do not idle, also in nested REPEAT blocks
    noise_model = circuits.NoiseModel(idle_error={"Z": 0.1}, scheduler=ticks)
    circuit = stim.Circuit("H 0\nTICK\nREPEAT 4 {\nREPEAT 1 {\nH 0\n}\nTICK\n}\nH 0")
    assert noise_model.noisy_circuit(circuit) == circuit

    noise_model = circuits.DepolarizingNoiseModel(0.1)
    circuit = stim.Circuit("H 0\nREPEAT 3 {\nCX 0 1\n}")
    noisy_circuit = stim.Circuit(
        "H 0\nDEPOLARIZE1(0.1) 0\nREPEAT 3 {\nCX 0 1\nDEPOLARIZE2(0.1) 0 1\n}"
    )
    assert noise_model.noisy_circuit(circuit) == noisy_circuit


def test_quadratic_idle_errors() -> None:
    """Quadratic idle errors grow over each run of idle moments, and restart after an operation.

    Over two idle moments, a flip probability ``p`` per moment becomes that of four moments of an
    independent error, ``(1 - (1 - 2 * p)**4) / 2``.
    """
    ticks = circuits.TickScheduler()
    circuit = stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0\nTICK\nH 1\nTICK\nM 0")
    model = circuits.NoiseModel(quadratic_idle_error={"Z": 0.01}, scheduler=ticks)
    expected = stim.Circuit("""
        H 0
        TICK
        Z_ERROR(0.01) 1
        H 1
        TICK
        H 1
        TICK
        Z_ERROR(0.03881592) 0
        H 0
        TICK
        Z_ERROR(0.01) 1
        H 1
        TICK
        Z_ERROR(0.01) 0
        M 0
        Z_ERROR(0.01) 1
    """)
    assert model.noisy_circuit(circuit).approx_equals(expected, atol=1e-12)

    # Custom idle fragments are still applied once per moment, and a float is a uniform channel.
    model = circuits.NoiseModel(
        idle_error=circuits.NoiseRule(after=stim.Circuit("X_ERROR(0.1) 0")),
        quadratic_idle_error=0.03,
        scheduler=ticks,
    )
    noisy = model.noisy_circuit(
        stim.Circuit("H 0\nTICK\nH 1\nTICK\nH 1\nTICK\nH 0"), system_qubits=[0]
    )
    expected = stim.Circuit("""
        H 0
        TICK
        H 1
        TICK
        H 1
        TICK
        X_ERROR(0.1) 0 0
        DEPOLARIZE1(0.10962552) 0
        H 0
    """)
    assert noisy.approx_equals(expected, atol=1e-12)

    # Over a long idle run, the flip probability saturates: it is about (1 - exp(-2)) / 2 rather
    # than 100**2 * 1e-4 = 1.  It is combined with the independent idle_error into one channel.
    circuit = stim.Circuit("R 0\nTICK\n" + "X 1\nTICK\n" * 100 + "M 0")
    model = circuits.NoiseModel(
        idle_error={"X": 0.01}, quadratic_idle_error={"Z": 1e-4}, scheduler=ticks
    )
    p_x, p_z = (1 - 0.98**100) / 2, (1 - math.exp(-2)) / 2
    expected = circuit[:-1] + stim.Circuit(
        f"PAULI_CHANNEL_1({p_x * (1 - p_z)}, {p_x * p_z}, {(1 - p_x) * p_z}) 0\nM 0"
    )
    assert model.noisy_circuit(circuit, system_qubits=[0]).approx_equals(expected, atol=1e-4)


def test_idle_runs_in_nested_repeat_blocks() -> None:
    """Idle noise over nested REPEAT blocks counts every moment, with runs split at their edges.

    Qubit 0 idles from its reset to its measurement, through all iterations of both blocks, for
    ``2 * (1 + 3) = 8`` moments, of which ``2 * 3 = 6`` measure qubit 1.  Each idle moment has an
    X error with probability ``p``, which multiplies the expectation of ``M 0`` by ``1 - 2 * p``.
    Each moment that measures qubit 1 also has an X error with probability ``w``, which multiplies
    it by ``1 - 2 * w``.  The quadratic idle error is split into 8 runs of one moment, each of which
    multiplies it by ``1 - 2 * q``, so the detector flips with probability
    ``(1 - (1 - 2 * p)**8 * (1 - 2 * w)**6 * (1 - 2 * q)**8) / 2``.
    """
    p, w, q = 0.01, 0.02, 0.001
    noise_model = circuits.NoiseModel(
        idle_error={"X": p},
        additional_error_waiting_for_m_or_r={"X": w},
        quadratic_idle_error={"X": q},
        scheduler=circuits.TickScheduler(),
    )
    circuit = stim.Circuit("""
        R 0 1
        TICK
        REPEAT 2 {
            H 1
            TICK
            REPEAT 3 {
                MR 1
                TICK
            }
        }
        M 0
        DETECTOR rec[-1]
    """)
    expected = (1 - (1 - 2 * p) ** 8 * (1 - 2 * w) ** 6 * (1 - 2 * q) ** 8) / 2
    with pytest.warns(UserWarning, match="REPEAT"):
        errors = _dem_errors(noise_model.noisy_circuit(circuit))
    assert errors == pytest.approx({str([stim.DemTarget.relative_detector_id(0)]): expected})


def _dem_errors(circuit: stim.Circuit) -> dict[str, float]:
    """The probability of each set of flipped detectors and observables."""
    dem = circuit.detector_error_model(approximate_disjoint_errors=True, flatten_loops=True)
    errors: dict[str, float] = {}
    for error in dem.flattened():
        if error.type == "error":
            key, probability = str(error.targets_copy()), error.args_copy()[0]
            old = errors.get(key, 0)
            errors[key] = old + probability - 2 * old * probability
    return errors


def test_observables() -> None:
    """An observable with Pauli targets joins the moment of the operation before it on its qubits.

    So it sees the noise of that operation and of the idle moments before it, but not of the idle
    moments after it, whether or not it also reads measurements.
    """
    noise_model = circuits.NoiseModel(reset_error=0.1)
    for observable in ["OBSERVABLE_INCLUDE(0) Z0", "M 1\nOBSERVABLE_INCLUDE(0) rec[-1] Z0"]:
        noisy = noise_model.noisy_circuit(stim.Circuit(f"R 0\n{observable}"))
        assert noisy.detector_error_model() == stim.DetectorErrorModel("error(0.1) L0")

    circuit = stim.Circuit("""
        R 0 1
        TICK
        X 1
        TICK
        X 0
        TICK
        X 1
        TICK
        OBSERVABLE_INCLUDE(0) Z0
        M 0 1
    """)
    noise_model = circuits.NoiseModel(
        idle_error={"X": 0.1},
        scheduler=circuits.TickScheduler(),
    )
    noisy = noise_model.noisy_circuit(circuit)
    assert noisy.detector_error_model() == stim.DetectorErrorModel("error(0.1) L0")


def test_schedulers() -> None:
    """Noise models schedule circuits only for idle noise."""
    assert circuits.NoiseModel().scheduler == circuits.ASAPScheduler()
    assert circuits.DepolarizingNoiseModel(0.1).scheduler == circuits.ASAPScheduler()
    assert circuits.SI1000NoiseModel(0.1).scheduler == circuits.ASAPScheduler()

    circuit = stim.Circuit("CX 0 1 1 2")
    noise_model = circuits.NoiseModel(clifford_2q_error=0.1, scheduler=_Unavailable())
    assert noise_model.noisy_circuit(circuit) == stim.Circuit(
        "CX 0 1\nDEPOLARIZE2(0.1) 0 1\nCX 1 2\nDEPOLARIZE2(0.1) 1 2"
    )
    noise_model = circuits.NoiseModel(idle_error=0.1, scheduler=_Unavailable())
    with pytest.raises(NotImplementedError):
        noise_model.noisy_circuit(circuit)


class _Unavailable(circuits.Scheduler):
    """A scheduler that cannot schedule circuits."""

    def get_schedule(self, circuit: circuits.StimCircuitLike) -> circuits.Schedule:
        raise NotImplementedError


def test_deprecated_insert_ticks() -> None:
    """insert_ticks=True lays a circuit out in moments as released, and adds noise in them.

    A TICK layer is split where gates share a qubit, and every gate except a classically controlled
    one takes time, whatever the scheduler of the model.  So CX 1 2 starts a new moment, with H 3,
    and qubits 2, 3, and 0 idle once each.  H 1 starts a new moment after CX rec[-1] 1, in whose
    moment qubit 1 idles.  The deprecated insert_ticks=False requests no layout, so the noise is the
    same as without it.
    """
    noise_model = circuits.NoiseModel(clifford_2q_error=0.1, idle_error={"Z": 0.2})
    circuit = stim.Circuit("CX 0 1 1 2\nH 3")
    with pytest.warns(DeprecationWarning, match="insert_ticks"):
        noisy_circuit = noise_model.noisy_circuit(circuit, insert_ticks=True)
    assert noisy_circuit == stim.Circuit("""
        CX 0 1
        TICK
        DEPOLARIZE2(0.1) 0 1
        Z_ERROR(0.2) 2
        CX 1 2
        Z_ERROR(0.2) 3
        H 3
        DEPOLARIZE2(0.1) 1 2
        Z_ERROR(0.2) 0
    """)
    circuit = stim.Circuit("M 0\nCX rec[-1] 1\nH 1")
    with pytest.warns(DeprecationWarning, match="insert_ticks"):
        noisy_circuit = noise_model.noisy_circuit(circuit, insert_ticks=True)
    assert noisy_circuit == stim.Circuit("""
        M 0
        CX rec[-1] 1
        TICK
        Z_ERROR(0.2) 1
        H 1
        Z_ERROR(0.2) 0
    """)
    with pytest.warns(DeprecationWarning, match="insert_ticks") as record:
        noisy_circuit = noise_model.noisy_circuit(circuit, insert_ticks=False)
    assert record[0].filename == __file__
    assert noisy_circuit == noise_model.noisy_circuit(circuit)


def test_immunity() -> None:
    """Operations, REPEAT blocks, and circuits can be immune to errors."""
    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        TICK
        CX 0 1
        TICK
        H['{immune_op_tag}'] 0
        X 1
    """)
    noise_model: circuits.NoiseModel = circuits.DepolarizingNoiseModel(
        0.1,
        include_idling_error=True,
        scheduler=circuits.TickScheduler(),
    )
    noisy_circuit = stim.Circuit(f"""
        H['{immune_op_tag}'] 0
        TICK
        DEPOLARIZE1(0.1) 1
        CX 0 1
        TICK
        DEPOLARIZE2(0.1) 0 1
        H['{immune_op_tag}'] 0
        X 1
        DEPOLARIZE1(0.1) 1
    """)
    assert noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag) == noisy_circuit
    tagged_gate = stim.Circuit()
    tagged_gate.append("H", 0, tag=immune_op_tag)
    assert circuits_are_equivalent(
        stim.Circuit(f"H[{immune_op_tag}] 0\nDEPOLARIZE1(0.1) 0"),
        noise_model.noisy_circuit(tagged_gate, immune_op_tag=None),
    )

    # a noiseless REPEAT block takes time, in which other qubits idle
    circuit = stim.Circuit("H 1\nTICK")
    circuit += circuits.as_noiseless_circuit(stim.Circuit("H 0"))
    circuit += stim.Circuit("TICK\nH 1")
    noise_model = circuits.NoiseModel(
        idle_error={"Z": 0.1},
        scheduler=circuits.TickScheduler(),
    )
    noisy_block = circuits.as_noiseless_circuit(stim.Circuit("H 0\nZ_ERROR(0.1) 1"))
    assert noise_model.noisy_circuit(circuit, system_qubits=[1]) == (
        circuit[:2] + noisy_block + circuit[3:]
    )

    # a noiseless REPEAT block ends the idle runs of its qubits, at any depth
    circuit = stim.Circuit("R 0 1\nTICK\nH 1\nTICK")
    circuit += circuits.as_noiseless_circuit(stim.Circuit("REPEAT 1 {\nH 0\n}"))
    circuit += stim.Circuit("H 1\nTICK\nMX 0\nDETECTOR rec[-1]")
    noise_model = circuits.NoiseModel(
        idle_error={"X": 0.1},
        scheduler=circuits.TickScheduler(),
    )
    dem = noise_model.noisy_circuit(circuit).detector_error_model()
    assert dem == stim.DetectorErrorModel("error(0.1) D0")

    # circuits can be made immune to errors
    circuit = stim.Circuit("H 0\nCX 0 1\nS 2\nCZ 2 3\nSQRT_X 4")
    noiseless_circuit = circuits.as_noiseless_circuit(circuit)
    assert noise_model.noisy_circuit(noiseless_circuit).to_tableau() == circuit.to_tableau()


def test_immune_qubits() -> None:
    """Gates on immune qubits get no noise, or only noise that acts as the identity on them.

    The noise of a rule is dropped for gates on immune qubits, or with immunize_gates=False, it
    keeps only the Pauli strings that act as the identity on immune qubits.  Noise that is not a
    Pauli channel raises an error under immunize_gates=False if a gate acts on both immune and other
    qubits.
    """
    # CX 0 1 acts on immune qubits only, and gets no noise.  CX 1 2 acts on an immune and another
    # qubit, so it gets no noise with immunize_gates=True.  With immunize_gates=False, its
    # DEPOLARIZE2(p) keeps the 3 of 15 Pauli strings that are I on qubit 1, which make
    # DEPOLARIZE1(p/5) on qubit 2.  CX 3 4 keeps its noise.
    circuit = stim.Circuit("""
        CNOT 0 1 1 2 3 4
    """)
    dm = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    assert dm.noisy_circuit(circuit, immune_qubits=[0, 1], immunize_gates=True) == stim.Circuit(
        "CNOT 0 1 1 2 3 4\nDEPOLARIZE2(0.1) 3 4"
    )
    assert dm.noisy_circuit(circuit, immune_qubits=[0, 1], immunize_gates=False) == stim.Circuit(
        "CNOT 0 1 1 2 3 4\nDEPOLARIZE1(0.02) 2\nDEPOLARIZE2(0.1) 3 4"
    )

    # with immunize_gates=False, the Pauli strings of the channel that are I on qubit 1 become a
    # PAULI_CHANNEL_2 on qubits 0 and 2
    channel = circuits.PauliChannel({"XYZ": 0.01, "XIZ": 0.02, "IZI": 0.03})
    spp_circuit = stim.Circuit("SPP X0*Y1*Z2")
    spp_model = circuits.NoiseModel(rules={"SPP": circuits.NoiseRule(after=channel)})
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

    # a reset gets no error on an immune qubit
    assert circuits_are_equivalent(
        stim.Circuit("R 0 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.1) 2"),
        circuits.NoiseModel(reset_error=0.1).noisy_circuit(
            stim.Circuit("R 0 1 2"), immune_qubits=[1]
        ),
    )

    # with immunize_gates=True, a measurement on an immune qubit gets no noise, so M 0 keeps its
    # readout error, while M 1 and MPP X0*Y1 get none
    assert circuits_are_equivalent(
        stim.Circuit("M(0.1) 0\nTICK\nM 1\nTICK\nMPP X0*Y1"),
        circuits.NoiseModel(readout_error=0.1).noisy_circuit(
            stim.Circuit("M 0\nTICK\nM 1\nTICK\nMPP X0*Y1"), immune_qubits=[1]
        ),
    )
    # with immunize_gates=False, an MPP on immune and other qubits raises an error, because readout
    # errors cannot be conditioned on immune qubits
    with pytest.raises(ValueError, match="partial immunity"):
        circuits.NoiseModel(readout_error=0.1).noisy_circuit(
            stim.Circuit("MPP X0*Y1"), immune_qubits=[1], immunize_gates=False
        )


def test_pauli_product_cliffords() -> None:
    """SPP gates on 1 or 2 qubits get, respectively, 1q or 2q noise.

    A multi-product SPP gate gets noise per product, explicit rules must fit the width of a product,
    and rules for two-qubit measurements apply to each measurement.
    """

    # SPP gates get the default noise for their number of qubits, and none on three or more qubits
    circuit = stim.Circuit("""
        SPP X0 X0*Y1
        SPP_DAG X0*Y1*Z2
    """)
    noise_model = circuits.NoiseModel(clifford_1q_error=0.1, clifford_2q_error=0.2)
    noisy_circuit = stim.Circuit("""
        SPP X0
        DEPOLARIZE1(0.1) 0
        SPP X0*Y1
        DEPOLARIZE2(0.2) 0 1
        SPP_DAG X0*Y1*Z2
    """)
    assert noise_model.noisy_circuit(circuit) == noisy_circuit

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
    with pytest.raises(ValueError, match=r"rules\['SPP'\]: `after` has arity 1; expected 3"):
        noise_model.noisy_circuit(circuit)

    measurement_rule = circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.1))
    measurement_model = circuits.NoiseModel(rules={"MZZ": measurement_rule})
    measured = measurement_model.noisy_circuit(stim.Circuit("MZZ 0 1 2 3"))
    assert measured == stim.Circuit("MZZ 0 1 2 3\nDEPOLARIZE2(0.1) 0 1 2 3")
    readout_rule = circuits.NoiseRule(readout_error=0.1)
    with pytest.raises(ValueError, match="readout_error"):
        circuits.NoiseModel(rules={"SPP": readout_rule}).noisy_circuit(stim.Circuit("SPP X0"))
    measured = circuits.NoiseModel(rules={"MPP": readout_rule}).noisy_circuit(
        stim.Circuit("MPP X0")
    )
    assert measured == stim.Circuit("MPP(0.1) X0")


def test_correlated_error_chains_stay_whole() -> None:
    """Noise that is pending on a qubit is emitted after, not inside, a chain of correlated errors.

    The noise of ``X 0`` is pending when a later link of the chain acts on qubit 0.  The chain flips
    qubit 1 with probability 0.01, or else qubit 0 with probability 0.02, and the gate noise flips
    qubit 0 with probability 0.1.
    """
    model = circuits.NoiseModel(clifford_1q_error={"X": 0.1})
    circuit = stim.Circuit("""
        X 0
        E(0.01) X1
        ELSE_CORRELATED_ERROR(0.02) X0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    detectors = [str([stim.DemTarget.relative_detector_id(index)]) for index in range(2)]
    flip_0 = 0.99 * 0.02
    assert _dem_errors(model.noisy_circuit(circuit)) == pytest.approx(
        {detectors[0]: 0.1 + flip_0 - 2 * 0.1 * flip_0, detectors[1]: 0.01}
    )


def test_rule_func() -> None:
    """A user-provided rule_func assigns noise per gate application, with top priority.

    The rules that it returns must fit their gates, and noise-immune qubits and operations still
    take precedence over them.
    """

    # rule_func gets one gate application at a time, which is one qubit of a one-qubit gate, and
    # None falls back on clifford_1q_error
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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

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
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # rule_func can also give measurement and reset errors.  It gets the name of the gate as
    # written, such as M or MZ, while rules match any of its aliases.
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
        TICK
        M(0.25) 0
        M(0.01) 1
        X_ERROR(0.3) 2
    """)
    assert noise_model.noisy_circuit(circuit) == noisy_circuit

    # Noise-immune qubits still take precedence over the callback.
    def kick(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        return circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.5))

    noise_model = circuits.NoiseModel(rule_func=kick)
    circuit = stim.Circuit("H 0 1")
    noisy_circuit = stim.Circuit("""
        H 0 1
        DEPOLARIZE1(0.5) 0
    """)
    assert circuits_are_equivalent(
        noisy_circuit, noise_model.noisy_circuit(circuit, immune_qubits={1})
    )

    # The callback sees gates, including identities, but not annotations, classically-controlled
    # ops, or the gates of noiseless REPEAT blocks.
    consulted: list[str] = []

    def record(op: stim.CircuitInstruction) -> circuits.NoiseRule:
        consulted.append(op.name)
        return circuits.NoiseRule(after={"X": 0.5})

    noise_model = circuits.NoiseModel(rule_func=record)
    circuit = stim.Circuit("QUBIT_COORDS(0, 0) 0\nI 2\nM 0\nCX rec[-1] 1")
    noise_model.noisy_circuit(circuit + circuits.as_noiseless_circuit(stim.Circuit("H 3")))
    assert sorted(consulted) == ["I", "M"]

    # An identity's after-noise precedes a subsequent operation on the same qubit.
    identity_rule = circuits.NoiseModel(
        rule_func=lambda op: circuits.NoiseRule(after={"X": 1.0}) if op.name == "I" else None
    )
    circuit = stim.Circuit("I 0\nM 0")
    assert identity_rule.noisy_circuit(circuit) == stim.Circuit("I 0\nX_ERROR(1) 0\nM 0")

    # A returned rule's readout_error/reset_error must match the gate it is assigned to.
    bad_readout = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(readout_error=0.1))
    with pytest.raises(ValueError, match=r"rule for 'H'.*readout_error.*measurement gates"):
        bad_readout.noisy_circuit(stim.Circuit("H 0"))
    bad_reset = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(reset_error=0.1))
    with pytest.raises(ValueError, match=r"rule for 'H'.*reset_error.*reset gates"):
        bad_reset.noisy_circuit(stim.Circuit("H 0"))

    # A returned rule's `after` arity must match the gate application's qubit count.
    bad_arity = circuits.NoiseModel(
        rule_func=lambda op: circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(1, 0.1))
    )
    with pytest.raises(ValueError, match=r"rule for 'CX'.*`after` has arity 1"):
        bad_arity.noisy_circuit(stim.Circuit("CX 0 1"))

    # SXYZ can be used as a `rules` key just like MXYZ.
    noise_model = circuits.NoiseModel(
        rules={"SXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(3, 0.1))}
    )
    noisy_circuit = noise_model.noisy_circuit(stim.Circuit("SPP X0*Y1*Z2"))
    assert "CORRELATED_ERROR" in str(noisy_circuit)
    assert "ELSE_CORRELATED_ERROR" in str(noisy_circuit)

    # immune_op_tag also applies to a tagged gate inside a REPEAT block
    kick_all = circuits.NoiseModel(rule_func=lambda op: circuits.NoiseRule(after={"X": 0.5}))
    body = stim.Circuit()
    body.append("H", [0], tag="__skip__")
    body.append("TICK")
    repeat_circuit = stim.Circuit()
    repeat_circuit.append(stim.CircuitRepeatBlock(repeat_count=2, body=body))
    result = kick_all.noisy_circuit(repeat_circuit, immune_op_tag="__skip__")
    assert "X_ERROR" not in str(result)


def test_noise_rule_subclass() -> None:
    """A noise model adds noise with the noisy_operation method of a subclass of NoiseRule."""

    class DephasingRule(circuits.NoiseRule):
        def noisy_operation(
            self, op: stim.CircuitInstruction
        ) -> tuple[stim.CircuitInstruction, stim.Circuit]:
            noisy_op, after = super().noisy_operation(op)
            after.append("Z_ERROR", [target.value for target in op.targets_copy()], [0.5])
            return noisy_op, after

    model = circuits.NoiseModel(rules={"H": DephasingRule(after={"X": 0.1})})
    assert model.noisy_circuit(stim.Circuit("H 0 1")) == stim.Circuit(
        "H 0 1\nX_ERROR(0.1) 0 1\nZ_ERROR(0.5) 0 1"
    )


def test_clifford_nq_error() -> None:
    """clifford_nq_error dispatches by qubit count k."""

    # a float is a uniform depolarizing channel on k qubits: each of the 7 nonempty sets of flipped
    # Z measurements comes from 8 of the 63 non-identity Paulis.
    p = 0.001
    noise_model = circuits.NoiseModel(clifford_nq_error={3: p})
    circuit = stim.Circuit("""
        R 0 1 2
        SPP Z0*Z1*Z2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    errors = _dem_errors(noise_model.noisy_circuit(circuit))
    assert len(errors) == 7
    assert all(
        probability == pytest.approx(8 * p / 63, rel=1e-3) for probability in errors.values()
    )

    # each Pauli product of an SPP gate gets the entry for its number of qubits, which may be a
    # Mapping or a PauliChannel
    noise_model = circuits.NoiseModel(
        clifford_nq_error={
            1: {"X": 0.01},  # a Mapping of Pauli strings
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

    # a stim.Circuit fragment can combine broadcast noise with a CORRELATED_ERROR on all qubits of
    # the gate
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

    # entries without noise are dropped
    assert circuits.NoiseModel(clifford_nq_error={3: circuits.NoiseRule()}).clifford_nq_error == {}

    # clifford_1q_error and clifford_2q_error also take a PauliChannel or a Mapping of Pauli strings
    noise_model = circuits.NoiseModel(
        clifford_1q_error=circuits.PauliChannel({"X": 0.01}),
        clifford_2q_error={"XY": 0.02},
    )
    assert circuits_are_equivalent(
        stim.Circuit(
            "H 0\nX_ERROR(0.01) 0\nCX 0 1\nPAULI_CHANNEL_2("
            + "0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1"
        ),
        noise_model.noisy_circuit(stim.Circuit("H 0\nCX 0 1")),
    )


def test_clifford_nq_error_errors() -> None:
    """Validation errors around clifford_nq_error and related rules."""

    # giving both clifford_nq_error[k] and clifford_kq_error raises, even if one of them is zero
    with pytest.raises(ValueError, match="Ambiguous"):
        circuits.NoiseModel(clifford_1q_error=0.01, clifford_nq_error={1: 0.0})

    # bool rejection
    with pytest.raises(TypeError, match="unsupported type bool"):
        circuits.NoiseModel(clifford_nq_error={2: True})

    # a rule whose key fixes its number of qubits is checked when the noise model is made
    with pytest.raises(ValueError, match="arity 2; expected 3"):
        circuits.NoiseModel(
            rules={"MXYZ": circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.01))}
        )


def test_empty_mapping_is_no_noise() -> None:
    """An empty Mapping of Pauli strings adds no noise, but other Mappings must fit their gates."""
    assert not circuits.NoiseModel(idle_error={})
    with pytest.raises(ValueError, match="arity 2; expected 1"):
        circuits.NoiseModel(idle_error={"XX": 0.0})
    with pytest.raises(ValueError, match="arity 1; expected 2"):
        circuits.NoiseModel(clifford_nq_error={2: {"X": 0.0}})


def test_idle_error_rule_validation() -> None:
    """Idle errors reject rules with readout or reset errors, which they cannot apply."""
    with pytest.raises(ValueError, match=r"idle_error: `readout_error` is only valid"):
        circuits.NoiseModel(idle_error=circuits.NoiseRule(readout_error=0.1))
    with pytest.raises(ValueError, match=r"additional_error_waiting_for_m_or_r: `reset_error`"):
        circuits.NoiseModel(additional_error_waiting_for_m_or_r=circuits.NoiseRule(reset_error=0.1))


def test_quadratic_idle_error_runs() -> None:
    """An idle run continues past noise channels, classically controlled Paulis, and observables.

    The error of a qubit's whole run is the same as if they were absent: over ``n`` idle moments, a
    flip probability ``p`` per moment becomes ``(1 - (1 - 2 * p)**(n**2)) / 2``.  This also holds
    for an observable in a later moment than the last gate on its qubit, because it reads a
    measurement.
    """
    noise_model = circuits.NoiseModel(
        quadratic_idle_error={"Z": 0.01},
        scheduler=circuits.TickScheduler(),
    )
    step = "H 1\nTICK\n"

    def get_error(interruption: str) -> float:
        circuit = stim.Circuit(f"R 0\nTICK\n{step * 2}{interruption}\n{step * 3}M 0")
        return _get_z_error(noise_model.noisy_circuit(circuit), 0)

    for interruption in ["X_ERROR(0.2) 0", "M 2\nCX rec[-1] 0", "OBSERVABLE_INCLUDE(0) Z0"]:
        assert get_error(interruption) == pytest.approx((1 - 0.98**25) / 2)

    # qubit 0 gets one depolarizing error for its run of 5 moments
    p = 0.001
    uniform_model = circuits.NoiseModel(
        quadratic_idle_error=p,
        scheduler=circuits.TickScheduler(),
    )
    circuit = stim.Circuit(
        f"R 0 1\nTICK\n{step * 2}M 1\nOBSERVABLE_INCLUDE(0) rec[-1] X0\nTICK\n{step * 2}M 0"
    )
    errors = [
        instruction.gate_args_copy()[0]
        for instruction in uniform_model.noisy_circuit(circuit)
        if instruction.name == "DEPOLARIZE1" and stim.GateTarget(0) in instruction.targets_copy()
    ]
    assert errors == pytest.approx([(1 - (1 - 2 * p) ** 25) / 2])

    # a repeat block that applies noise to an idle qubit stays compact
    circuit = stim.Circuit("R 0")
    circuit.append(stim.CircuitRepeatBlock(10**4, stim.Circuit("TICK\nX_ERROR(0) 0\nH 1")))
    with pytest.warns(UserWarning, match="REPEAT"):
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

    sequence = circuits.PauliChannelSequence({"Z": 0.01})
    for invalid in ({"X": 0.3, "Z": 0.3}, {"XY": 0.01}, sequence):
        with pytest.raises(ValueError, match="quadratic_idle_error"):
            circuits.NoiseModel(quadratic_idle_error=invalid)  # type: ignore[arg-type]


def test_trivial_noise() -> None:
    """Noise models and rules without noise are falsy, and a rule_func makes a model truthy."""
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
    """NoiseModel survives pickling / deep-copying and reconstructs an equivalent model."""
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
    round_tripped = pickle.loads(pickle.dumps(model))  # noqa: S301 - trusted in-memory round trip
    for restored in (round_tripped, copy.deepcopy(model)):
        assert circuits_are_equivalent(
            model.noisy_circuit(circuit), restored.noisy_circuit(circuit)
        )


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
    assert "rule_func" not in text

    # A scheduler other than the default is listed.
    si1000 = circuits.SI1000NoiseModel(0.1, scheduler=circuits.TickScheduler())
    assert "scheduler=TickScheduler()" in str(si1000)
    schedule = circuits.TickScheduler().get_schedule(stim.Circuit("H 0\nTICK\nH 0"))
    with_schedule = circuits.SI1000NoiseModel(0.1, scheduler=schedule)
    assert "scheduler=<Schedule with 2 moments>" in str(with_schedule)

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
