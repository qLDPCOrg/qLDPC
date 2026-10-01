# SPDX-License-Identifier: Apache-2.0

"""Unit tests for syndrome_measurement.py."""

import random

import numpy as np
import pytest
import stim
import sympy

from qldpc import circuits, codes, math
from qldpc.objects import Pauli

# default strategy used by the validity-checking helper below
DEFAULT_STRATEGY = circuits.EdgeColoring()


def test_syndrome_measurement(pytestconfig: pytest.Config) -> None:
    """Verify that syndromes are read out correctly."""
    seed = pytestconfig.getoption("randomly_seed")

    # default strategies for non-CSS and CSS codes
    assert syndrome_measurement_is_valid(codes.FiveQubitCode())
    assert syndrome_measurement_is_valid(codes.SteaneCode())

    # special strategies for toric and surface codes
    assert syndrome_measurement_is_valid(codes.ToricCode(2, rotated=True))
    assert syndrome_measurement_is_valid(codes.SurfaceCode(2, rotated=True))

    # special strategy for HGPCodes
    code_a = codes.ClassicalCode.random(5, 3, seed=seed)
    code_b = codes.ClassicalCode.random(3, 2, seed=seed + 1)
    assert syndrome_measurement_is_valid(codes.HGPCode(code_a, code_b))

    # special strategy for QCCodes
    np.random.seed(seed)
    symbols = [sympy.Symbol(f"x_{ss}") for ss in range(3)]
    orders = [np.random.randint(2, 6) for _ in range(len(symbols))]
    term_indices_a = np.random.choice(range(np.prod(orders)), replace=False, size=4)
    term_indices_b = np.random.choice(range(np.prod(orders)), replace=False, size=3)
    term_exponents_a = [np.unravel_index(index, orders) for index in term_indices_a]
    term_exponents_b = [np.unravel_index(index, orders) for index in term_indices_b]
    poly_a = sum(
        np.prod([symbol**exponent for symbol, exponent in zip(symbols, exponents_a)])
        for exponents_a in term_exponents_a
    )
    poly_b = sum(
        np.prod([symbol**exponent for symbol, exponent in zip(symbols, exponents_b)])
        for exponents_b in term_exponents_b
    )
    assert syndrome_measurement_is_valid(codes.QCCode(orders, poly_a, poly_b))

    # EdgeColoringXZ strategy
    assert syndrome_measurement_is_valid(codes.SteaneCode(), circuits.EdgeColoringXZ())
    with pytest.raises(TypeError, match="only supports CSS codes"):
        circuits.EdgeColoringXZ().get_circuit(codes.FiveQubitCode())
    with pytest.raises(ValueError, match="subsystem"):
        circuits.EdgeColoring().get_circuit(codes.BaconShorCode(2))
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.EdgeColoring().get_circuit(code=codes.SurfaceCode(2, field=3))


def test_validate_syndrome_qubit_ids() -> None:
    """Validate default, explicit, and unsupported syndrome-measurement layouts."""
    code = codes.SurfaceCode(2)
    qubit_ids = circuits.QubitIDs.from_code(code)
    assert circuits.validate_syndrome_qubit_ids(code) == qubit_ids
    assert circuits.validate_syndrome_qubit_ids(code, qubit_ids) is not qubit_ids

    with pytest.raises(ValueError, match="subsystem"):
        circuits.validate_syndrome_qubit_ids(codes.BaconShorCode(2))


def test_gauge_layers() -> None:
    """Partition the checks of a code into layers of commuting checks."""
    strategy = circuits.EdgeColoring()

    # CSS codes measure all X-type checks and then all Z-type checks, skipping empty layers
    bacon_shor = codes.BaconShorCode(2, 3)
    assert strategy.get_gauge_layers(bacon_shor) == (
        tuple(range(bacon_shor.num_checks_x)),
        tuple(range(bacon_shor.num_checks_x, bacon_shor.num_checks)),
    )
    repetition = codes.CSSCode.classical(codes.RepetitionCode(3), Pauli.Z)
    assert strategy.get_gauge_layers(repetition) == ((0, 1),)

    # the checks of non-CSS codes are colored such that each layer consists of commuting checks
    code = codes.QuditCode(codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    layers = strategy.get_gauge_layers(code)
    assert len(layers) == 2
    circuits.validate_gauge_layers(code, layers)
    assert strategy.get_gauge_layers(codes.FiveQubitCode()) == (tuple(range(4)),)

    # layers must partition the checks of a code into sets of commuting checks
    with pytest.raises(ValueError, match="partition"):
        circuits.validate_gauge_layers(code, layers[:1])
    with pytest.raises(ValueError, match="partition"):
        circuits.validate_gauge_layers(code, layers + ((0,),))
    with pytest.raises(ValueError, match="do not commute"):
        circuits.validate_gauge_layers(code, (tuple(range(code.num_checks)),))
    with pytest.raises(ValueError, match="do not commute"):
        strategy.get_subsystem_circuit(code, layers=(tuple(range(code.num_checks)),))


def test_subsystem_circuit(pytestconfig: pytest.Config) -> None:
    """Measure the checks of a code one commuting layer at a time."""
    seed = pytestconfig.getoption("randomly_seed")

    # the layers of stabilizer codes are measured correctly
    for code in [codes.FiveQubitCode(), codes.SteaneCode(), codes.SurfaceCode(3)]:
        assert syndrome_measurement_is_valid(code, subsystem=True)
    code_a = codes.ClassicalCode.random(5, 3, seed=seed)
    code_b = codes.ClassicalCode.random(3, 2, seed=seed + 1)
    assert syndrome_measurement_is_valid(
        codes.HGPCode(code_a, code_b), circuits.EdgeColoringXZ(), subsystem=True
    )

    # layers are measured sequentially, one layer of checks at a time
    code = codes.BaconShorCode(2, 3)
    qubit_ids = circuits.QubitIDs.from_code(code, shift=5)
    circuit, record = circuits.EdgeColoring().get_subsystem_circuit(code, qubit_ids)
    measured_qubits = [
        target.qubit_value
        for instruction in circuit.flattened()
        if instruction.name == "MX"
        for target in instruction.targets_copy()
    ]
    assert measured_qubits == list(qubit_ids.checks_x + qubit_ids.checks_z)
    names = [instruction.name for instruction in circuit if instruction.name != "TICK"]
    assert [name for nn, name in enumerate(names) if name not in names[nn + 1 : nn + 2]] == [
        "RX",
        "CX",
        "MX",
        "RX",
        "CZ",
        "MX",
    ]
    assert record == circuits.MeasurementRecord(
        {check_id: [mm] for mm, check_id in enumerate(measured_qubits)}
    )
    assert circuit.num_measurements == record.num_events == code.num_checks

    # EdgeColoringXZ only supports CSS codes
    code = codes.QuditCode(codes.BaconShorCode(2).matrix, is_subsystem_code=True)
    with pytest.raises(TypeError, match="only supports CSS codes"):
        circuits.EdgeColoringXZ().get_subsystem_circuit(code)

    # layer circuits must be synchronized with their measurement records
    class BadStrategy(circuits.SyndromeMeasurementStrategy):
        def get_circuit(
            self, code: codes.QuditCode, qubit_ids: circuits.QubitIDs | None = None
        ) -> tuple[stim.Circuit, circuits.MeasurementRecord]:
            circuit, record = circuits.EdgeColoring().get_circuit(code, qubit_ids)
            circuit.append("M", 0)
            return circuit, record

    with pytest.raises(ValueError, match="record contains"):
        BadStrategy().get_subsystem_circuit(codes.BaconShorCode(2))


def syndrome_measurement_is_valid(
    code: codes.QuditCode,
    strategy: circuits.SyndromeMeasurementStrategy = DEFAULT_STRATEGY,
    *,
    subsystem: bool = False,
) -> bool:
    """Check the validity of syndrome measurement in a given code."""
    # prepare a logical |0> state
    state_prep = circuits.get_encoding_circuit(code)

    # apply random Pauli errors to the data qubits
    errors = random.choices([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], k=len(code))
    error_ops = stim.Circuit()
    for qubit, pauli in enumerate(errors):
        error_ops.append(f"{pauli}_error", [qubit], [1])

    # measure syndromes
    syndrome_extraction, record = (
        strategy.get_subsystem_circuit(code) if subsystem else strategy.get_circuit(code)
    )
    for check in range(len(code), len(code) + code.num_checks):
        syndrome_extraction.append("DETECTOR", record.get_target_rec(check))

    # sample the circuit to obtain a syndrome vector
    circuit = state_prep + error_ops + syndrome_extraction
    syndrome = circuit.compile_detector_sampler().sample(1).ravel()

    # compare against the expected syndrome
    error_xz = code.field([pauli.value for pauli in errors]).T.ravel()
    expected_syndrome = code.matrix @ math.symplectic_conjugate(error_xz)

    return np.array_equal(expected_syndrome, syndrome)


@pytest.mark.parametrize("code", [codes.SurfaceCode(3)])
def test_syndrome_measurement_scheduling(code: codes.CSSCode) -> None:
    """Verify valid gate scheduling in some syndrome measurement circuits."""
    for strategy in [circuits.EdgeColoring(), circuits.EdgeColoringXZ()]:
        circuit, _ = strategy.get_circuit(code)
        circuit_without_ticks = stim.Circuit(str(circuit).replace("TICK", ""))
        assert gate_schedule_is_valid(circuit)
        assert not gate_schedule_is_valid(circuit_without_ticks)
        assert circuit[-1].name == "TICK"

    edge = next(iter(code.graph.edges))
    single_layer = circuits.EdgeColoring.graph_to_circuit(
        code.graph.edge_subgraph([edge]), circuits.QubitIDs.from_code(code), "smallest_last"
    )
    assert single_layer.num_ticks == 1


def test_edge_coloring_xz_distance_tradeoff() -> None:
    """The documented EdgeColoringXZ distance tradeoff remains observable."""
    circuit = circuits.get_memory_experiment(
        codes.SurfaceCode(5, rotated=True),
        basis=Pauli.Z,
        num_rounds=1,
        noise_model=circuits.DepolarizingNoiseModel(1e-3),
        syndrome_measurement_strategy=circuits.EdgeColoringXZ(),
    )
    assert len(circuit.shortest_graphlike_error()) == 4


def gate_schedule_is_valid(circuit: stim.Circuit) -> bool:
    """Check that no qubit is addressed twice between TICKS in a circuit."""
    tick = stim.CircuitInstruction("TICK")
    is_addressed_in_current_moment: dict[int, bool] = {}
    for instruction in circuit:
        if instruction == tick:
            # reset the record of qubits that are addressed in the current moment
            is_addressed_in_current_moment = {}
        else:
            # update the record of qubits that have been addressed in the current moment
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    if is_addressed_in_current_moment.get(target.qubit_value, False):
                        return False
                    is_addressed_in_current_moment[target.qubit_value] = True
    return True
