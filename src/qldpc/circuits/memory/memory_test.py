# SPDX-License-Identifier: Apache-2.0

"""Unit tests for memory.py."""

import collections
import copy
import itertools
import random
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.circuits import CheckKey, CheckProductKey, ReadoutKey
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
    parts = circuits.get_memory_experiment_parts(
        rep_code, basis=Pauli.X, num_rounds=1, readout_keys=True
    )
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
            readout_keys=True,
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
        assert_relabeled_qubit_ids_agree(code)


def assert_relabeled_qubit_ids_agree(code: codes.CSSCode) -> None:
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
            code, basis=basis, num_rounds=num_qec_rounds, qubit_ids=qubit_ids, readout_keys=True
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
            # subsystem codes, measured in gauge layers, may order the commuting gates within a
            # moment, and the measurements that a detector multiplies, by qubit index
            assert sort_targets(circuit_a) == sort_targets(circuit_b)
        else:
            assert circuit_a.flattened() == circuit_b.flattened()


def sort_targets(circuit: stim.Circuit) -> stim.Circuit:
    """Flatten a circuit and sort the commuting targets of two-qubit gates and detectors.

    Syndrome measurement strategies may order the gates within a single moment by qubit index, and
    detectors of subsystem codes multiply measurements in the order of their check qubit indices.
    """
    sorted_circuit = stim.Circuit()
    for instruction in circuit.flattened():
        targets = instruction.targets_copy()
        if stim.gate_data(instruction.name).is_two_qubit_gate:
            pairs = sorted(zip(targets[::2], targets[1::2]), key=lambda pair: str(pair))
            targets = [target for pair in pairs for target in pair]
        elif instruction.name == "DETECTOR":
            targets = sorted(targets, key=lambda target: target.value)
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
        circuits.get_memory_experiment_parts(
            codes.RepetitionCode(3), Pauli.X, num_rounds=0, readout_keys=True
        )
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
            readout_keys=True,
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
            readout_keys=True,
        )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z, None])
def test_subsystem_memory_experiment(basis: PauliXZ | None) -> None:
    """Memory experiments for subsystem codes have deterministic and complete detectors."""
    code = codes.BaconShorCode(3, 4)
    for num_rounds in [1, 3]:
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds, readout_keys=True)
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
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds=2, readout_keys=True)
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
    parts = circuits.get_memory_experiment_parts(
        code, Pauli.Z, num_rounds=num_rounds, readout_keys=True
    )
    assert_detectors_track_stabilizers(code, Pauli.Z, parts.detector_record, num_rounds)
    detector_types = collections.Counter(
        (len(key.check_ids), key.is_relation, len(parts.detector_record[key]))
        for key in parts.detector_record
        if isinstance(key, CheckProductKey)
    )
    assert detector_types == {(4, False, num_rounds): 21, (4, True, num_rounds): 12}
    assert sum(isinstance(key, ReadoutKey) for key in parts.detector_record) == 21

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


def test_future_change_readout_keys() -> None:
    """Readout detectors are appended to the entries of their checks, with a warning by default."""
    num_rounds = 3
    cases: list[tuple[codes.CSSCode, PauliXZ | None]] = [
        (codes.SurfaceCode(2), Pauli.Z),
        (codes.SurfaceCode(2), None),
        (get_bacon_shor_code_with_relations(), Pauli.X),
        (get_bacon_shor_code_with_relations(), None),
    ]
    for code, basis in cases:
        # if readout_keys is not specified, warn that the default will change
        with pytest.warns(FutureWarning, match="readout_keys=True"):
            default_parts = circuits.get_memory_experiment_parts(code, basis, num_rounds)
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds, readout_keys=False)
        assert default_parts.detector_record == parts.detector_record
        assert default_parts.measurement_record == parts.measurement_record

        # readout_keys=True moves the readout detector (and final stabilizer measurement) at the end
        # of the entries of a key to the entries of a ReadoutKey
        new_parts = circuits.get_memory_experiment_parts(code, basis, num_rounds, readout_keys=True)
        assert new_parts.qec_cycle + new_parts.readout == parts.qec_cycle + parts.readout
        for record, new_record in [
            (parts.detector_record, new_parts.detector_record),
            (parts.measurement_record, new_parts.measurement_record),
        ]:
            assert sum(map(len, record.values())) == sum(map(len, new_record.values()))
            for key in record:
                assert isinstance(key, int | CheckProductKey)
                assert record[key] == new_record.get(key, []) + new_record.get(ReadoutKey(key), [])

        # stabilizer codes are keyed by plain check indices, as in earlier releases
        if not code.is_subsystem_code:
            assert all(type(key) is CheckKey for key in parts.detector_record)
            for check_id in parts.qubit_ids.checks_z:
                assert len(parts.detector_record[int(check_id)]) == num_rounds + 1


def test_detector_keys() -> None:
    """Keys of the detectors that track checks, products of checks, and readouts."""
    # a CheckKey is equal to an integer, and carries the attributes of a key
    key = CheckKey(5)
    assert key == 5 and hash(key) == hash(5) and isinstance(key, int) and repr(key) == "5"
    assert key.check_ids == (5,) and not key.is_relation
    assert type(key + 1) is int
    with pytest.raises(AttributeError, match="immutable"):
        key.is_relation = True  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del key._is_relation
    record = circuits.DetectorRecord({key: [0, 1], ReadoutKey(key): [2]})
    assert record[5] == [0, 1] and 5 in record and record[ReadoutKey(5)] == [2]
    relation_keys: list[CheckKey | CheckProductKey] = [
        CheckKey(7, is_relation=True),
        CheckProductKey((1, 2), is_relation=True),
    ]
    for relation_key in relation_keys:
        copied_key = copy.deepcopy(relation_key)
        assert copied_key == relation_key and type(copied_key) is type(relation_key)
        assert copied_key.check_ids == relation_key.check_ids and copied_key.is_relation

    # a CheckProductKey is determined by the set of checks that it multiplies
    product_key = CheckProductKey(tuple(np.array([3, 1])), is_relation=True)
    assert product_key.check_ids == (1, 3) and all(type(cc) is int for cc in product_key.check_ids)
    assert product_key == CheckProductKey((1, 3)) and hash(product_key) == hash(
        CheckProductKey((1, 3))
    )
    for check_ids in [(1,), (1, 1)]:
        with pytest.raises(ValueError, match="at least two distinct"):
            CheckProductKey(check_ids)

    # a subsystem check that is a stabilizer gets a CheckKey, and other detectors get product keys
    subsystem_code = get_bacon_shor_code_with_relations()
    parts = circuits.get_memory_experiment_parts(subsystem_code, None, 3, readout_keys=True)
    assert_detectors_track_stabilizers(subsystem_code, None, parts.detector_record, 3)
    keys = [key for key in parts.detector_record if isinstance(key, CheckKey | CheckProductKey)]
    assert {(type(key), key.is_relation) for key in keys} == {
        (CheckKey, False),
        (CheckProductKey, False),
        (CheckProductKey, True),
    }
    assert type(next(key for key in keys if key == parts.qubit_ids.checks_x[-1])) is CheckKey


def test_detectors_read_check_measurements() -> None:
    """The detectors of each key read the measurements of its checks, and the final readout."""
    num_rounds = 3
    cases: list[tuple[codes.CSSCode, PauliXZ | None]] = [
        (codes.SurfaceCode(2), Pauli.Z),
        (codes.SurfaceCode(2), None),
        (get_bacon_shor_code_with_relations(), Pauli.X),
        (get_bacon_shor_code_with_relations(), None),
    ]
    for code, basis in cases:
        parts = circuits.get_memory_experiment_parts(code, basis, num_rounds, readout_keys=True)
        circuit = parts.initialization + parts.qec_cycle + parts.readout
        detectors, measured_qubits, measured_products = resolve_measurements(
            circuit, parts.qubit_ids.data
        )
        check_to_row = {check_id: row for row, check_id in enumerate(parts.qubit_ids.check)}
        check_measurements = {
            check_id: [mm for mm, qubit in enumerate(measured_qubits) if qubit == check_id]
            for check_id in parts.qubit_ids.check
        }
        assert all(len(measurements) == num_rounds for measurements in check_measurements.values())
        data_measurements = {
            qubit: mm for mm, qubit in enumerate(measured_qubits) if qubit in parts.qubit_ids.data
        }

        num_detectors = 0
        for key in parts.detector_record:
            if not isinstance(key, CheckKey | CheckProductKey):
                continue
            round_detectors = parts.detector_record[key]
            readout_detectors = parts.detector_record.get(ReadoutKey(key), [])
            assert len(round_detectors) == num_rounds

            # each round compares the outcomes of the checks to those of the previous round, unless
            # the checks multiply to a relation
            rounds = [
                {check_measurements[cc][rr] for cc in key.check_ids} for rr in range(num_rounds)
            ]
            for rr, detector in enumerate(round_detectors):
                expected = rounds[rr] ^ (rounds[rr - 1] if rr and not key.is_relation else set())
                assert detectors[detector] == expected

            # the readout detector compares the last round to the final readout
            stabilizer = np.sum(code.matrix[[check_to_row[cc] for cc in key.check_ids]], axis=0)
            assert len(readout_detectors) == (0 if key.is_relation else 1)
            num_detectors += len(round_detectors) + len(readout_detectors)
            if key.is_relation:
                continue
            if basis is None:
                readout = {
                    mm
                    for mm, product in measured_products.items()
                    if np.array_equal(product, stabilizer)
                }
                assert len(readout) == 1
                assert set(parts.measurement_record[ReadoutKey(key)]) == readout
            else:
                support = stabilizer.reshape(2, -1)[basis]
                readout = {
                    data_measurements[parts.qubit_ids.data[qq]] for qq in np.flatnonzero(support)
                }
            assert detectors[readout_detectors[0]] == rounds[-1] | readout
        assert num_detectors == circuit.num_detectors


def resolve_measurements(
    circuit: stim.Circuit, data_qubits: Sequence[int]
) -> tuple[list[set[int]], list[int | None], dict[int, npt.NDArray[np.int_]]]:
    """The measurements read by each detector, and the qubit or product measured by each.

    Returns:
        detectors: The set of (absolute) measurement indices whose parity each detector reads.
        measured_qubits: The qubit of each single-qubit measurement, or None for a product.
        measured_products: The symplectic vector, on the data qubits, of each MPP product, by
            measurement index.
    """
    detectors: list[set[int]] = []
    measured_qubits: list[int | None] = []
    measured_products: dict[int, npt.NDArray[np.int_]] = {}
    num_qubits = len(data_qubits)
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            num_measurements = len(measured_qubits)
            detector: set[int] = set()
            for target in instruction.targets_copy():
                detector ^= {num_measurements + target.value}
            detectors.append(detector)
        elif instruction.name == "MPP":
            for group in instruction.target_groups():
                product = np.zeros(2 * num_qubits, dtype=int)
                for target in group:
                    qq = data_qubits.index(target.qubit_value)
                    product[qq] = target.is_x_target or target.is_y_target
                    product[num_qubits + qq] = target.is_z_target or target.is_y_target
                measured_products[len(measured_qubits)] = product
                measured_qubits.append(None)
        elif stim.gate_data(instruction.name).produces_measurements:
            assert stim.gate_data(instruction.name).is_single_qubit_gate
            measured_qubits.extend(target.value for target in instruction.targets_copy())
    return detectors, measured_qubits, measured_products


def get_bacon_shor_code_with_relations() -> codes.CSSCode:
    """A subsystem code with a stabilizer check, products of checks, and relations in a layer."""
    bacon_shor = codes.BaconShorCode(2)
    matrix_x = np.vstack([bacon_shor.matrix_x, bacon_shor.get_stabilizer_ops(Pauli.X)])
    return codes.CSSCode(matrix_x, bacon_shor.matrix_z, is_subsystem_code=True)


def test_non_css_subsystem_memory_experiment() -> None:
    """Combined-basis memory experiments support non-CSS subsystem codes."""
    code = codes.QuditCode(codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    code = code.deformed(stim.Circuit("H_YZ 0 3 5 \n SQRT_X 1 7"))
    assert code.is_subsystem_code
    support_x, support_z = code.matrix.reshape(-1, 2, len(code)).transpose(1, 0, 2)
    assert np.any(np.any(support_x, axis=1) & np.any(support_z, axis=1))  # a check is not CSS

    parts = circuits.get_memory_experiment_parts(code, None, num_rounds=3, readout_keys=True)
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

    parts = circuits.get_memory_experiment_parts(code, None, num_rounds=2, readout_keys=True)
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

    The memory experiment should use readout_keys=True.  Detectors should track a generating set of
    stabilizers of the tracked type, including all tracked generators in code.get_stabilizer_ops()
    if these are products of checks in single gauge layers.  Together with the stabilizer detectors
    in each default gauge layer, relation detectors should generate all relations among the checks
    (of the tracked type) in that layer.  Every stabilizer, but no relation, should have a readout
    detector keyed by a ReadoutKey.
    """
    qubit_ids = circuits.QubitIDs.from_code(code)
    check_to_row = {check_id: row for row, check_id in enumerate(qubit_ids.check)}
    gauge_ops = code.matrix

    keys = [key for key in detector_record if isinstance(key, CheckKey | CheckProductKey)]
    readout_record_keys = [key for key in detector_record if isinstance(key, ReadoutKey)]
    assert len(keys) + len(readout_record_keys) == len(detector_record)
    assert all(isinstance(key, CheckKey) == (len(key.check_ids) == 1) for key in keys)
    products = code.field(
        [
            np.sum(gauge_ops[[check_to_row[check_id] for check_id in key.check_ids]], axis=0)
            for key in keys
        ]
    )
    is_relation = ~np.any(products, axis=1)
    assert [key.is_relation for key in keys] == is_relation.tolist()
    stabilizer_keys = [key for key, relation in zip(keys, is_relation) if not relation]
    stabilizers = products[~is_relation]

    # every key has one detector for each round, and stabilizers (but not relations) are compared
    # with the final readout by a readout detector of their own
    assert all(len(detector_record[key]) == num_rounds for key in keys)
    assert {readout_key.key for readout_key in readout_record_keys} == set(stabilizer_keys)
    assert all(len(detector_record[key]) == 1 for key in readout_record_keys)

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
            if set(key.check_ids) <= layer_check_ids:
                vector = np.zeros(len(gauge_ops), dtype=int)
                vector[[check_to_row[check_id] for check_id in key.check_ids]] = 1
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
