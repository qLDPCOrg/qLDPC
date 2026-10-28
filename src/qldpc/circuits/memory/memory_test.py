# SPDX-License-Identifier: Apache-2.0

"""Unit tests for memory.py."""

import collections
import itertools
import random

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.circuits.memory import memory
from qldpc.objects import PAULIS_XZ, Pauli, PauliXZ


def test_memory_experiment() -> None:
    """Stim circuits for memory experiments."""
    num_rounds, shots = 5, 10
    noise_model = circuits.DepolarizingNoiseModel(1e-2)

    # try out a classical error correcting code
    rep_code = codes.RepetitionCode(3)
    circuit = circuits.get_memory_experiment(
        rep_code, basis=Pauli.Z, num_rounds=num_rounds, noise_model=noise_model
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == rep_code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == rep_code.dimension

    # try tracking both operators in a quantum code
    code = codes.RepetitionCode(2)
    circuit = circuits.get_memory_experiment(
        code, basis=None, num_rounds=num_rounds, noise_model=noise_model
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == code.dimension * 2

    # we can also ask for a noiseless circuit, and inject noise afterwards
    noiseless_circuit = circuits.get_memory_experiment(code, basis=None, num_rounds=num_rounds)
    dem_1 = circuit.detector_error_model()
    dem_2 = noise_model.noisy_circuit(noiseless_circuit).detector_error_model()
    assert dem_1 == dem_2
    parts = circuits.get_memory_experiment_parts(rep_code, basis=Pauli.X, num_rounds=1)
    assembled = parts.initialization + parts.qec_cycle + parts.readout
    assert circuits.get_memory_experiment(
        rep_code, basis=Pauli.X, num_rounds=1, noise_model=noise_model
    ) == noise_model.noisy_circuit(assembled)
    surface_code = codes.SurfaceCode(2)

    class AncillaStrategy(circuits.SyndromeMeasurementStrategy):
        def __init__(self, *, use_reference: bool = False) -> None:
            self.use_reference = use_reference

        def get_circuit(
            self, code: codes.QuditCode, qubit_ids: circuits.QubitIDs | None = None
        ) -> tuple[stim.Circuit, circuits.MeasurementRecord]:
            assert qubit_ids is not None
            circuit, record = circuits.EdgeColoring().get_circuit(code, qubit_ids)
            target = qubit_ids.reference[0] if self.use_reference else qubit_ids.ancilla[0]
            circuit.append("H", target)
            circuit.append("TICK")
            return circuit, record

    qubit_ids = circuits.QubitIDs(
        range(1, 5),
        range(10, 10 + surface_code.num_checks),
        [9],
        reference=[0],
    )
    combined = circuits.get_memory_experiment(
        surface_code,
        basis=None,
        noise_model=circuits.NoiseModel(idle_error=0.01),
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=AncillaStrategy(),
    )
    assert not any("DEPOLARIZE" in line and " 0" in line for line in str(combined).splitlines())
    assert any("DEPOLARIZE" in line and " 9" in line for line in str(combined).splitlines())
    noiseless_combined = circuits.get_memory_experiment(
        surface_code,
        basis=None,
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=AncillaStrategy(),
    )
    deferred = circuits.NoiseModel(idle_error=0.01).noisy_circuit(noiseless_combined)
    assert not any("DEPOLARIZE" in line and " 0" in line for line in str(deferred).splitlines())
    assert any("DEPOLARIZE" in line and " 9" in line for line in str(deferred).splitlines())
    with pytest.raises(ValueError, match="Bell-reference ancillas"):
        circuits.get_memory_experiment_parts(
            surface_code,
            basis=None,
            qubit_ids=qubit_ids,
            syndrome_measurement_strategy=AncillaStrategy(use_reference=True),
        )

    # Pauli.Y basis measurements are not supported
    with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
        circuits.get_memory_experiment(rep_code, basis=Pauli.Y)  # type:ignore[arg-type]

    # non-CSS codes are not supported by fixed-basis memory experiments
    with pytest.raises(TypeError, match=r"only support CSS codes"):
        circuits.get_memory_experiment(codes.FiveQubitCode())


def test_memory_experiment_circuit_distance() -> None:
    """Default memory experiments preserve the distance of a distance-3 surface code.

    Building a detector error model requires deterministic detectors and observables, and an
    observable that is a stabilizer would have no undetectable error at all.
    """
    code = codes.SurfaceCode(3, rotated=True)
    noise_model = circuits.DepolarizingNoiseModel(1e-3)
    for basis in PAULIS_XZ + [None]:
        circuit = circuits.get_memory_experiment(
            code, basis=basis, num_rounds=2, noise_model=noise_model
        )
        assert circuit.num_observables == (2 if basis is None else 1)
        assert len(circuit.shortest_graphlike_error()) == 3


def test_qubit_ids(pytestconfig: pytest.Config) -> None:
    """We can construct memory experiments with different qubit IDs."""
    random.seed(pytestconfig.getoption("randomly_seed"))
    assert circuits.get_qubit_coordinates([0], [1], [2]) == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
    """)

    for code in [codes.SurfaceCode(2, rotated=True), codes.BaconShorCode(2, 3)]:
        check_qubit_ids(code)


def check_qubit_ids(code: codes.CSSCode) -> None:
    """Memory experiments with random qubit IDs agree with remapped default experiments."""
    # pick a number of "extra" unused qubits and a number of QEC rounds
    num_unused_qubits = 3
    num_qec_rounds = 3

    # assign random qubit indices
    qubits = list(range(len(code) + code.num_checks + code.dimension + num_unused_qubits))
    random.shuffle(qubits)
    qubit_ids = circuits.QubitIDs(
        data=qubits[: len(code)],
        check=qubits[len(code) : len(code) + code.num_checks],
        ancilla=qubits[len(code) + code.num_checks + code.dimension :],
        reference=qubits[
            len(code) + code.num_checks : len(code) + code.num_checks + code.dimension
        ],
    )

    for basis in PAULIS_XZ + [None]:
        # produce a memory experiment with the requested qubit IDs
        init, cycle, readout, *_, qubit_ids = circuits.get_memory_experiment_parts(
            code, basis=basis, num_rounds=num_qec_rounds, qubit_ids=qubit_ids
        )
        circuit_a = init + cycle + readout

        # produces a memory experiment with the default qubit IDs and remap manually
        qubit_map = (
            qubit_ids.data + qubit_ids.check + (qubit_ids.reference if basis is None else ())
        )
        circuit_b = circuits.with_remapped_qubits(
            circuits.get_memory_experiment(code, basis=basis, num_rounds=num_qec_rounds),
            qubit_map,
        )

        if code.is_subsystem_code:
            # layer codes may order the commuting gates within a moment by qubit index
            assert sort_gate_targets(circuit_a) == sort_gate_targets(circuit_b)
        else:
            assert circuit_a.flattened() == circuit_b.flattened()


def sort_gate_targets(circuit: stim.Circuit) -> stim.Circuit:
    """Flatten a circuit and sort the (commuting) target pairs of each two-qubit gate instruction.

    Syndrome measurement strategies may order the gates within a single moment by qubit index.
    """
    sorted_circuit = stim.Circuit()
    for instruction in circuit.flattened():
        targets = instruction.targets_copy()
        if stim.gate_data(instruction.name).is_two_qubit_gate:
            pairs = sorted(zip(targets[::2], targets[1::2]), key=lambda pair: str(pair))
            targets = [target for pair in pairs for target in pair]
        sorted_circuit.append(
            instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag
        )
    return sorted_circuit


def test_errors() -> None:
    """Cover invalid options for observable annotations."""
    with pytest.raises(ValueError, match="CSS codes"):
        circuits.get_observables(codes.FiveQubitCode(), basis=Pauli.X, on_measurements=True)
    with pytest.raises(ValueError, match="fixed measurement basis"):
        circuits.get_observables(codes.SteaneCode(), basis=None, on_measurements=True)
    with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
        circuits.get_observables(codes.SteaneCode(), basis="y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_rounds"):
        circuits.get_memory_experiment_parts(codes.RepetitionCode(3), Pauli.X, num_rounds=0)
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_observables(codes.SteaneCode(), data_qubits=[0], basis=Pauli.X)
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_observables(
            codes.SteaneCode(), basis=Pauli.X, on_measurements=[stim.target_rec(-1)]
        )
    with pytest.raises(ValueError, match="observable indices"):
        circuits.get_observables(codes.SteaneCode(), basis=Pauli.X, observable_indices=[0, 1])
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_logical_bell_prep(codes.SteaneCode(), data_qubits=[0])
    with pytest.raises(ValueError, match="one target per logical qubit"):
        circuits.get_logical_bell_prep(codes.SteaneCode(), reference_qubits=[0, 1])
    toric_code = codes.ToricCode(2)
    with pytest.raises(ValueError, match="either no reference qubits or exactly one reference"):
        circuits.get_memory_experiment_parts(
            toric_code,
            basis=None,
            qubit_ids=circuits.QubitIDs.from_code(toric_code, num_references=1),
        )


def test_memory_rejects_unsynchronized_strategy_records() -> None:
    """A strategy that emits an unrecorded measurement fails at the first lookback."""

    class BadStrategy(circuits.SyndromeMeasurementStrategy):
        def get_circuit(
            self, code: codes.QuditCode, qubit_ids: circuits.QubitIDs | None = None
        ) -> tuple[stim.Circuit, circuits.MeasurementRecord]:
            circuit, record = circuits.EdgeColoring().get_circuit(code, qubit_ids)
            circuit.append("M", 0)
            return circuit, record

    with pytest.raises(ValueError, match="record contains"):
        circuits.get_memory_experiment_parts(
            codes.SurfaceCode(2),
            basis=Pauli.X,
            syndrome_measurement_strategy=BadStrategy(),
        )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z, None])
def test_subsystem_memory_experiment(basis: PauliXZ | None) -> None:
    """Memory experiments for subsystem codes have deterministic and complete detectors."""
    code = codes.BaconShorCode(3, 4)
    for num_rounds in [1, 3]:
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds)
        circuit = parts.initialization + parts.qec_cycle + parts.readout
        assert_detectors_track_stabilizers(code, basis, parts.detector_record, num_rounds)

        # stim rejects detectors and observables that are not deterministic
        circuit.detector_error_model()

    # the circuit-level distance matches the code distance for the tracked logical operators
    noise_model = circuits.DepolarizingNoiseModel(1e-3)
    circuit = circuits.get_memory_experiment(code, basis, num_rounds=3, noise_model=noise_model)
    distance = code.get_distance(None if basis is None else basis.swap_xz())
    assert get_circuit_distance(circuit) == distance

    # measuring gauge checks individually does not determine any stabilizers
    class SingletonLayers(circuits.EdgeColoring):
        def get_gauge_layers(self, code: codes.QuditCode) -> tuple[tuple[int, ...], ...]:
            return tuple((check,) for check in range(code.num_checks))

    with pytest.raises(ValueError, match="does not determine all"):
        circuits.get_memory_experiment(code, basis, syndrome_measurement_strategy=SingletonLayers())

    # The circuit and its detectors share one schedule, even if layers are not reproducible.  After
    # the default layers, this strategy alternates single Z-type and X-type checks, which determine
    # no stabilizers, and which make the detectors of the default layers nondeterministic.
    checks_x, checks_z = circuits.EdgeColoring().get_gauge_layers(code)
    alternating_layers = tuple(
        (check,)
        for pair in itertools.zip_longest(checks_z, checks_x)
        for check in pair
        if check is not None
    )

    class NonReproducibleLayers(circuits.EdgeColoring):
        num_calls = 0

        def get_gauge_layers(self, code: codes.QuditCode) -> tuple[tuple[int, ...], ...]:
            self.num_calls += 1
            return super().get_gauge_layers(code) if self.num_calls == 1 else alternating_layers

    circuits.get_memory_experiment(
        code, basis, num_rounds=2, syndrome_measurement_strategy=NonReproducibleLayers()
    ).detector_error_model()


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z, None])
def test_subsystem_product_memory_experiments(basis: PauliXZ | None) -> None:
    """Memory experiments for other subsystem hypergraph product codes."""
    code_a = codes.ClassicalCode.random(5, 3, seed=0)
    code_b = codes.ClassicalCode.random(4, 2, seed=1)
    for code in [codes.SHYPSCode(3), codes.SHPCode(code_a, code_b)]:
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds=2)
        assert_detectors_track_stabilizers(code, basis, parts.detector_record, num_rounds=2)
        circuit = parts.initialization + parts.qec_cycle + parts.readout
        circuit.detector_error_model()
        assert not np.any(circuit.compile_detector_sampler().sample(4, append_observables=True))


def test_subsystem_memory_redundant_generators() -> None:
    """Redundant gauge and stabilizer generators get low-weight detectors.

    The [[49, 9, 4]] SHYPS code has 49 Z-type gauge generators of rank 28, and 21 Z-type stabilizer
    generators of rank 12, each of which is a product of four Z-type gauge generators.  Detectors
    should track all of these stabilizers, and complete the 49 - 28 = 21 relations among Z-type
    gauge generators with 21 - 12 = 9 relations implied by these stabilizers, plus 12 others.
    """
    code = codes.SHYPSCode(3)
    num_rounds = 2
    parts = circuits.get_memory_experiment_parts(code, Pauli.Z, num_rounds=num_rounds)
    assert_detectors_track_stabilizers(code, Pauli.Z, parts.detector_record, num_rounds)
    detector_types = collections.Counter(
        (len(key), len(parts.detector_record[key]))
        for key in parts.detector_record
        if isinstance(key, tuple)
    )
    assert detector_types == {(4, num_rounds + 1): 21, (4, num_rounds): 12}

    # relation detectors only multiply outcomes from a single round
    detector_sizes = collections.Counter(
        len(instruction.targets_copy())
        for instruction in parts.qec_cycle.flattened()
        if instruction.name == "DETECTOR"
    )
    assert detector_sizes == {4: 21 + 12 * num_rounds, 8: 21 * (num_rounds - 1)}


def test_low_weight_products() -> None:
    """Greedy weight reduction preserves spans and does not increase Hamming weights."""
    # reducing a basis preserves its span, and reduces the weight of some basis vectors
    basis = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 1]], dtype=np.uint8)
    reduced = memory._get_low_weight_basis(basis)
    assert memory._get_rank(galois.GF2(reduced)) == 3
    assert memory._get_rank(galois.GF2(np.vstack([basis, reduced]))) == 3
    assert np.all(np.count_nonzero(reduced, axis=1) <= np.count_nonzero(basis, axis=1))
    assert np.count_nonzero(reduced) < np.count_nonzero(basis)

    # reducing a vector adds an element of the span of other vectors, and does not increase weight
    vector = np.ones(5, dtype=np.uint8)
    result = memory._reduce_weight(vector, reduced)
    assert np.count_nonzero(result) <= np.count_nonzero(vector)
    assert memory._get_rank(galois.GF2(np.vstack([reduced, result ^ vector]))) == 3
    assert memory._reduce_weight(vector, basis[:1]).tolist() == [0, 0, 0, 0, 1]  # unique minimum
    assert memory._reduce_weight(vector, basis[:0]).tolist() == vector.tolist()


def test_non_css_subsystem_memory_experiment() -> None:
    """Combined-basis memory experiments support non-CSS subsystem codes."""
    code = codes.QuditCode(codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    code = code.deformed(stim.Circuit("H_YZ 0 3 5 \n SQRT_X 1 7"))
    assert code.is_subsystem_code
    support_x, support_z = code.matrix.reshape(-1, 2, len(code)).transpose(1, 0, 2)
    assert np.any(np.any(support_x, axis=1) & np.any(support_z, axis=1))  # a check is not CSS

    parts = circuits.get_memory_experiment_parts(code, None, num_rounds=3)
    assert_detectors_track_stabilizers(
        code, None, parts.detector_record, num_rounds=3, generators_in_layers=False
    )
    circuit = parts.initialization + parts.qec_cycle + parts.readout
    circuit.detector_error_model()

    noise_model = circuits.DepolarizingNoiseModel(1e-3)
    circuit = circuits.get_memory_experiment(code, None, num_rounds=3, noise_model=noise_model)
    assert get_circuit_distance(circuit) == 3


def test_subsystem_memory_non_generating_stabilizers() -> None:
    """Detectors are complete even if stabilizer generators mix gauge layers."""
    bacon_shor = codes.BaconShorCode(3)
    code = codes.QuditCode(bacon_shor.matrix, is_subsystem_code=True)
    stabilizer_ops = bacon_shor.get_stabilizer_ops(symplectic=True)
    stabilizer_ops[0] += stabilizer_ops[-1]  # mix X-type and Z-type stabilizers
    code._stabilizer_ops = stabilizer_ops

    parts = circuits.get_memory_experiment_parts(code, None, num_rounds=2)
    assert_detectors_track_stabilizers(
        code, None, parts.detector_record, num_rounds=2, generators_in_layers=False
    )
    circuit = parts.initialization + parts.qec_cycle + parts.readout
    circuit.detector_error_model()


def assert_detectors_track_stabilizers(
    code: codes.QuditCode,
    basis: PauliXZ | None,
    detector_record: circuits.DetectorRecord,
    num_rounds: int,
    *,
    generators_in_layers: bool = True,
) -> None:
    """Assert that detectors of a subsystem-code memory experiment are complete.

    Detectors should track a generating set of stabilizers of the tracked type, including all
    tracked generators in code.get_stabilizer_ops() if these are products of checks in single gauge
    layers.  Together with the stabilizer detectors in each default gauge layer, relation detectors
    should generate all relations among the checks (of the tracked type) in that layer.
    """
    qubit_ids = circuits.QubitIDs.from_code(code)
    check_to_row = {check_id: row for row, check_id in enumerate(qubit_ids.check)}
    gauge_ops = code.matrix

    keys = [key for key in detector_record if isinstance(key, tuple)]
    assert len(keys) == len(detector_record)
    products = code.field(
        [np.sum(gauge_ops[[check_to_row[check_id] for check_id in key]], axis=0) for key in keys]
    )
    is_relation = ~np.any(products, axis=1)
    stabilizer_keys = [key for key, relation in zip(keys, is_relation) if not relation]
    relation_keys = [key for key, relation in zip(keys, is_relation) if relation]
    stabilizers = products[~is_relation]

    # stabilizers are compared with readout, but relations are not
    assert all(len(detector_record[key]) == num_rounds + 1 for key in stabilizer_keys)
    assert all(len(detector_record[key]) == num_rounds for key in relation_keys)

    # every detector tracks a stabilizer, which commutes with all gauge operators
    assert not np.any(stabilizers @ math.symplectic_conjugate(gauge_ops).T)

    # the tracked stabilizers generate all stabilizers of the tracked type
    is_tracked = np.ones(len(gauge_ops), dtype=bool)
    target = code.get_stabilizer_ops(symplectic=True)
    if basis is not None:
        other_support = target.reshape(-1, 2, len(code))[:, basis.swap_xz(), :]
        target = target[~np.any(other_support, axis=1)]
        is_tracked = ~np.any(gauge_ops.reshape(-1, 2, len(code))[:, basis.swap_xz(), :], axis=1)
    rank = np.linalg.matrix_rank(target)
    assert rank == np.linalg.matrix_rank(stabilizers)
    assert rank == np.linalg.matrix_rank(np.vstack([stabilizers, target]))
    if generators_in_layers:
        assert {tuple(row.tolist()) for row in target} <= {
            tuple(row.tolist()) for row in stabilizers
        }

    # in each layer, relation detectors complete the relations implied by stabilizer detectors
    layers = circuits.EdgeColoring().get_gauge_layers(code)
    num_keys_in_layers = 0
    for layer in layers:
        layer_check_ids = {qubit_ids.check[cc] for cc in layer}
        layer_vectors: dict[bool, list[npt.NDArray[np.int_]]] = {False: [], True: []}
        for key, relation in zip(keys, is_relation):
            if set(key) <= layer_check_ids:
                vector = np.zeros(len(gauge_ops), dtype=int)
                vector[[check_to_row[check_id] for check_id in key]] = 1
                layer_vectors[bool(relation)].append(vector)
                num_keys_in_layers += 1
        stabilizer_vectors = code.field(
            np.array(layer_vectors[False], dtype=int).reshape(-1, len(gauge_ops))
        )
        relation_vectors = code.field(
            np.array(layer_vectors[True], dtype=int).reshape(-1, len(gauge_ops))
        )
        implied = code.field.Zeros((0, len(gauge_ops)))
        if len(stabilizer_vectors):
            implied = (stabilizer_vectors @ gauge_ops).T.null_space() @ stabilizer_vectors
        tracked_layer = [cc for cc in layer if is_tracked[cc]]
        num_relations = len(tracked_layer) - memory._get_rank(gauge_ops[tracked_layer])
        assert memory._get_rank(code.field(np.vstack([implied, relation_vectors]))) == num_relations
        assert memory._get_rank(implied) + len(relation_vectors) == num_relations
    assert num_keys_in_layers == len(keys)


def get_circuit_distance(circuit: stim.Circuit) -> int:
    """Weight of the smallest set of circuit faults that flips an observable undetectably."""
    return len(
        circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=4,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    )
