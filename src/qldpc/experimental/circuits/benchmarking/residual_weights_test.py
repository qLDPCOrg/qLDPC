# SPDX-License-Identifier: Apache-2.0

"""Unit tests for residual_weights.py."""

from __future__ import annotations

import dataclasses
import itertools
import math

import galois
import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.experimental.circuits.benchmarking import residual_weights
from qldpc.experimental.circuits.benchmarking.consumer_profiles import ConsumerProfile
from qldpc.experimental.circuits.benchmarking.residual_weights import (
    ResidualWeightHistogram,
    estimate_residual_weight_histogram,
)
from qldpc.objects import Pauli, PauliXZLike

# A flagged preparation of the CSS state with stabilizers Z0*Z1, Z1*Z2, and X0*X1*X2, given as
# layers with (layer, Pauli, probability) error mechanisms after them.  Qubit 3 checks Z0*Z2 as a
# flag.
_CSS_LAYERS = ["H 0", "CX 0 1", "CX 1 2", "CX 0 3 2 3", "M 3\nDETECTOR rec[-1]"]
_CSS_MECHANISMS = [
    (0, "X0", 1e-3),
    (0, "Z0", 3e-4),
    (1, "X1", 7e-4),
    (1, "Z0*X1", 4e-4),
    (2, "Y2", 3e-4),
    (2, "X0*X2", 2e-4),
    (3, "X3", 5e-4),
    (3, "Z2", 6e-4),
]
# A preparation of a state that is not CSS: its stabilizers include Y0*X1*X2.
_NON_CSS_LAYERS = ["H 0\nS 0", "CX 0 1", "CX 1 2"]
_NON_CSS_MECHANISMS = [
    (0, "X0", 1e-3),
    (0, "Z0", 3e-4),
    (1, "X1*Y0", 7e-4),
    (2, "Y2", 3e-4),
    (2, "Z1*Z2", 2e-4),
    (2, "X2", 9e-4),
]


@pytest.mark.parametrize(
    "layers, mechanisms, num_data_qubits, css_weight, post_select, pauli",
    [
        (_CSS_LAYERS, _CSS_MECHANISMS, 3, None, True, None),
        (_CSS_LAYERS, _CSS_MECHANISMS, 3, False, True, None),
        (_CSS_LAYERS, _CSS_MECHANISMS, 3, False, True, "X"),
        (_CSS_LAYERS, _CSS_MECHANISMS, 3, None, True, "Z"),
        (_NON_CSS_LAYERS, _NON_CSS_MECHANISMS, 3, None, False, None),
        (_NON_CSS_LAYERS, _NON_CSS_MECHANISMS, 3, True, False, None),
    ],
)
def test_matches_brute_force(
    layers: list[str],
    mechanisms: list[tuple[int, str, float]],
    num_data_qubits: int,
    css_weight: bool | None,
    post_select: bool,
    pauli: PauliXZLike | None,
) -> None:
    """Exact histograms match a brute-force simulation of every fault.

    The residual weight of one Pauli type is the least weight of that type of any Pauli error that
    maps the ideal state to the actual one, whatever ``css_weight`` is.  For a CSS state, the Pauli
    errors that do so are the output error times X-type and Z-type stabilizers, so its two parts
    reduce independently.
    """
    circuit = _build_circuit(layers, mechanisms)
    max_residual_weight = 1
    histogram = estimate_residual_weight_histogram(
        circuit,
        max_residual_weight,
        output_qubits=range(num_data_qubits),
        css_weight=css_weight,
        post_select=post_select,
        pauli=pauli,
    )
    assert np.all(histogram.exact)
    use_css_weight = bool(css_weight) if css_weight is not None else layers is _CSS_LAYERS
    assert histogram.css_weight == use_css_weight
    assert histogram.pauli is (None if pauli is None else Pauli.coerce_xz(pauli))

    acceptance, distribution = _brute_force(
        layers, mechanisms, num_data_qubits, use_css_weight, max_residual_weight, pauli
    )
    assert histogram.acceptance_probability == pytest.approx(acceptance, rel=1e-12)
    assert histogram.discard_probability == pytest.approx(1 - acceptance, abs=1e-15)
    assert np.allclose(histogram.get_residual_weight_distribution(), distribution, rtol=1e-9)
    tails = histogram.get_tail_probabilities()
    assert np.allclose(tails, np.cumsum(distribution[::-1])[::-1][1:], rtol=1e-9)


def test_sampling_matches_exact_histogram() -> None:
    """Sampled strata agree with exact enumeration within their standard errors."""
    circuit = _build_circuit(_CSS_LAYERS, _CSS_MECHANISMS)
    exact = estimate_residual_weight_histogram(circuit, 1, output_qubits=range(3), post_select=True)
    sampled = estimate_residual_weight_histogram(
        circuit, 1, 40_000, output_qubits=range(3), post_select=True, max_exact_faults=0, seed=0
    )
    assert not np.any(sampled.exact[1:]) and np.all(sampled.num_samples[1:] > 0)
    assert sampled.acceptance_probability == pytest.approx(exact.acceptance_probability, rel=1e-3)
    for threshold in range(2):
        expected = exact.get_tail_probabilities()[threshold]
        value, error = sampled.get_excess_failure_probability(ConsumerProfile.hard(threshold))
        assert 0 < error < expected
        assert abs(value - expected) < 5 * error


def test_logical_error_blind_fault() -> None:
    """A Z-type Pauli error on logical |0> flips no logical observable, but has residual weight 2.

    In the rotated distance-3 surface code, the weight-two Z-type Pauli error on qubits 0 and 2 is
    equivalent to no Z-type Pauli error of weight one modulo the stabilizers of logical |0>, yet it
    commutes with logical Z.
    """
    code = codes.SurfaceCode(3, rotated=True)
    probability = 1e-3
    circuit = circuits.get_encoding_circuit(code, only_zero=True)
    circuit.append("CORRELATED_ERROR", [stim.target_z(0), stim.target_z(2)], probability)

    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(code, circuit)
    dem = diagnostic_circuit.detector_error_model()
    assert not any(
        target.is_logical_observable_id()
        for instruction in dem.flattened()
        if instruction.type == "error"
        for target in instruction.targets_copy()
    )

    histogram = estimate_residual_weight_histogram(circuit, 2, output_qubits=range(len(code)))
    assert np.allclose(histogram.get_tail_probabilities(), [probability, probability, 0])


def test_fault_weight_counts_error_instructions() -> None:
    """Each error instruction is an error mechanism, even if another has the same effect here.

    Both error instructions flip the output qubit, and only the first also fires a detector that is
    not post-selected, so a fault with both has fault weight two and residual weight zero.
    """
    probability = 1e-3
    circuit = stim.Circuit(
        f"CORRELATED_ERROR({probability}) X0 X1\nX_ERROR({probability}) 0\nM 1\nDETECTOR rec[-1]"
    )
    histogram = estimate_residual_weight_histogram(circuit, 0, output_qubits=[0])
    expected = [(1 - probability) ** 2, 2 * probability * (1 - probability), probability**2]
    assert np.allclose(histogram.fault_weight_probabilities, expected)
    assert histogram.get_tail_probabilities()[0] == pytest.approx(expected[1])


def test_added_flags() -> None:
    """Post-selection on added flags discards the faults that a flag detects.

    The circuit prepares a Bell state on qubits 0 and 1 and measures Z0*Z1 on qubit 2 as a flag.
    An X0 error before the flag fires it, while a Z1 error after it leaves residual weight one.
    """
    flagged, unflagged = 1e-3, 2e-4
    circuit = stim.Circuit(
        f"H 0\nCX 0 1\nX_ERROR({flagged}) 0\nCX 0 2 1 2\nM 2\nZ_ERROR({unflagged}) 1"
    )
    histogram = estimate_residual_weight_histogram(circuit, 1, add_flags=True)
    assert histogram.discard_probability == pytest.approx(flagged)
    assert np.allclose(histogram.get_residual_weight_distribution(), [1 - unflagged, unflagged, 0])


def test_noiseless_and_truncated_histograms() -> None:
    """Noiseless circuits leave no error, and neglected or unsampled shots count as overflow."""
    circuit = _build_circuit(_CSS_LAYERS, [])
    noiseless = estimate_residual_weight_histogram(circuit, 1, output_qubits=range(3))
    assert noiseless.max_fault_weight == 0
    assert np.allclose(noiseless.get_residual_weight_distribution(), [1, 0, 0])

    # two error mechanisms, each leaving residual weight one on |00>
    first, second = 1e-3, 2e-4
    circuit = stim.Circuit(f"X_ERROR({first}) 0\nX_ERROR({second}) 1")
    no_error = (1 - first) * (1 - second)

    truncated = estimate_residual_weight_histogram(
        circuit, 0, 100, max_fault_weight=1, max_exact_faults=0, seed=0
    )
    assert truncated.max_fault_weight == 1 and truncated.num_samples[1] > 0
    assert truncated.truncated_probability == pytest.approx(first * second)
    # every shot is accepted, so the bound is the truncated probability itself
    assert truncated.truncation_error_bound == pytest.approx(first * second)
    assert np.allclose(truncated.get_residual_weight_distribution(), [no_error, 1 - no_error])

    unsampled = estimate_residual_weight_histogram(circuit, 0, 0, max_exact_faults=0)
    assert np.all(unsampled.num_samples == 0) and not np.any(unsampled.exact[1:])
    assert unsampled.acceptance_probability == pytest.approx(1)
    assert np.allclose(unsampled.get_residual_weight_distribution(), [no_error, 1 - no_error])


def test_standard_error() -> None:
    """The standard error of a sampled stratum is the binomial error of its accepted samples.

    Six unit-weight samples of fault weight one are discarded twice, accepted three times with
    residual weight zero, and accepted once with overflow, so a quarter of the four accepted
    samples overflow.
    """
    counts = np.array([2.0, 3.0, 1.0])
    histogram = ResidualWeightHistogram(
        max_residual_weight=0,
        css_weight=True,
        fault_weight_probabilities=np.array([0.0, 1.0]),
        truncated_probability=0.0,
        weight_sums=np.array([[0.0, 1.0, 0.0], counts]),
        squared_weight_sums=np.array([[0.0, 0.0, 0.0], counts]),
        num_samples=np.array([0, 6]),
        exact=np.array([True, False]),
    )
    value, error = histogram.get_excess_failure_probability(ConsumerProfile.hard(0))
    assert value == pytest.approx(1 / 4)
    assert error == pytest.approx(np.sqrt(1 / 4 * 3 / 4 / 4))


def test_more_than_64_qubits() -> None:
    """States on more than 64 qubits use the general residual-weight search.

    Qubit 0 is in |+> and the others are in |0>, so an X error on qubit 64 and the X part of a Y
    error on qubit 1 leave X-type residual errors, and a Z error on qubit 0 leaves a Z-type one.
    """
    x_probability, y_probability, z_probability = 1e-3, 2e-3, 3e-3
    circuit = stim.Circuit(
        f"H 0\nX_ERROR({x_probability}) 64\nY_ERROR({y_probability}) 1\nZ_ERROR({z_probability}) 0"
    )
    no_x_error = (1 - x_probability) * (1 - y_probability)
    overflow = x_probability * y_probability
    cases: list[tuple[PauliXZLike | None, float]] = [
        (None, no_x_error * (1 - z_probability)),
        ("X", no_x_error),
    ]
    for pauli, no_error in cases:
        histogram = estimate_residual_weight_histogram(
            circuit, 1, output_qubits=range(65), pauli=pauli
        )
        expected = [no_error, 1 - no_error - overflow, overflow]
        assert np.allclose(histogram.get_residual_weight_distribution(), expected)


def test_excess_failure_probability() -> None:
    """Hard profiles give tail probabilities, and fault budgets combine tails with fault weights."""
    circuit = _build_circuit(_CSS_LAYERS, _CSS_MECHANISMS)
    histogram = estimate_residual_weight_histogram(
        circuit, 2, output_qubits=range(3), post_select=True
    )
    tails = histogram.get_tail_probabilities()
    for threshold in range(3):
        value, error = histogram.get_excess_failure_probability(ConsumerProfile.hard(threshold))
        assert value == pytest.approx(tails[threshold]) and error == 0

    # with a fault weight g ~ Poisson(nu), the guarantee m + g <= t fails with m + g > t and g <= t
    expected_fault_weight, correctable_weight = 0.3, 2
    profile = ConsumerProfile.poisson(expected_fault_weight, correctable_weight)
    value, _ = histogram.get_excess_failure_probability(profile)
    expected = sum(
        math.exp(-expected_fault_weight)
        * expected_fault_weight**fault_weight
        / math.factorial(fault_weight)
        * tails[correctable_weight - fault_weight]
        for fault_weight in range(correctable_weight + 1)
    )
    assert value == pytest.approx(expected)

    with pytest.raises(ValueError, match="resolves residual weights up to 3"):
        histogram.get_excess_failure_probability(ConsumerProfile.hard(3))


def test_invalid_arguments() -> None:
    """Invalid arguments and states raise errors."""
    circuit = _build_circuit(_CSS_LAYERS, _CSS_MECHANISMS)
    with pytest.raises(TypeError, match="max_residual_weight must be an integer"):
        estimate_residual_weight_histogram(circuit, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_samples must be at least 0"):
        estimate_residual_weight_histogram(circuit, 1, -1)
    with pytest.raises(ValueError, match="max_fault_weight must be at least 0"):
        estimate_residual_weight_histogram(circuit, 1, max_fault_weight=-1)
    with pytest.raises(TypeError, match="css_weight must be a bool"):
        estimate_residual_weight_histogram(circuit, 1, css_weight=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no unmeasured qubits"):
        estimate_residual_weight_histogram(stim.Circuit("H 0\nM 0"), 1)
    with pytest.raises(ValueError, match="definite pure stabilizer state"):
        estimate_residual_weight_histogram(stim.Circuit("H 0\nCX 0 1"), 1, output_qubits=[0])
    with pytest.raises(ValueError, match="probability one"):
        estimate_residual_weight_histogram(stim.Circuit("X_ERROR(1) 0"), 1)
    with pytest.raises(ValueError, match="CSS"):
        estimate_residual_weight_histogram(_build_circuit(_NON_CSS_LAYERS, []), 1, pauli="X")


def test_invalid_histograms() -> None:
    """Histograms with inconsistent shapes, invalid data, or invalid Pauli types are rejected."""
    histogram = ResidualWeightHistogram(
        max_residual_weight=0,
        css_weight=True,
        fault_weight_probabilities=np.array([1.0]),
        truncated_probability=0.0,
        weight_sums=np.array([[0.0, 1.0, 0.0]]),
        squared_weight_sums=np.zeros((1, 3)),
        num_samples=np.zeros(1, dtype=int),
        exact=np.ones(1, dtype=bool),
    )
    with pytest.raises(ValueError, match="inconsistent shapes"):
        dataclasses.replace(histogram, weight_sums=np.ones((1, 2)))
    with pytest.raises(ValueError, match="must be nonnegative"):
        dataclasses.replace(histogram, truncated_probability=-0.1)
    with pytest.raises(ValueError, match="must be finite"):
        dataclasses.replace(histogram, weight_sums=np.array([[0.0, np.nan, 0.0]]))
    assert dataclasses.replace(histogram, pauli="x").pauli is Pauli.X
    with pytest.raises(ValueError, match="Pauli"):
        dataclasses.replace(histogram, pauli=Pauli.Y)  # type: ignore[arg-type]


def test_weight_oracles(pytestconfig: pytest.Config) -> None:
    """Both residual-weight oracles agree with brute force on every syndrome of the Steane code."""
    code = codes.SteaneCode()
    x_supports = code.matrix_x.view(np.ndarray).astype(np.uint8)
    logical_z = code.get_logical_ops(Pauli.Z)
    z_supports = np.vstack([code.matrix_z, logical_z]).view(np.ndarray).astype(np.uint8)
    max_residual_weight = 2

    # mix the generators, so that the supports must be recovered from a general basis
    num_qubits = len(code)
    stabilizers = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    stabilizers[: len(x_supports), :num_qubits] = x_supports
    stabilizers[len(x_supports) :, num_qubits:] = z_supports
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    mixed = _random_invertible(num_qubits, rng) @ galois.GF2(stabilizers)
    recovered_x, recovered_z = residual_weights._get_css_supports(np.asarray(mixed, dtype=np.uint8))
    for recovered, original in ((recovered_x, x_supports), (recovered_z, z_supports)):
        rank = np.linalg.matrix_rank(galois.GF2(original))
        assert len(recovered) == rank
        assert np.linalg.matrix_rank(galois.GF2(np.vstack([recovered, original]))) == rank

    tables = residual_weights._CSSWeightTables(x_supports, z_supports, max_residual_weight)
    search = residual_weights._CosetSearchWeights(stabilizers, True, max_residual_weight)
    keys = []
    expected = []
    for z_syndrome in range(2 ** len(x_supports)):
        for x_syndrome in range(2 ** len(z_supports)):
            syndrome = z_syndrome | x_syndrome << len(x_supports)
            weight = max(
                _brute_force_weight(x_supports, z_syndrome),
                _brute_force_weight(z_supports, x_syndrome),
            )
            keys.append(syndrome)
            expected.append(min(weight, max_residual_weight + 1))
    for oracle in (tables, search):
        packed = np.array([oracle.get_key_words(key) for key in keys], dtype=np.uint64)
        assert np.array_equal(oracle.get_weights(packed), expected)


def test_combinations_and_proposals() -> None:
    """Listed combinations and proposal probabilities have their defining properties."""
    for num_items, size in [(5, 0), (5, 2), (4, 4), (3, 5)]:
        combinations = residual_weights._get_combinations(num_items, size)
        expected = list(itertools.combinations(range(num_items), size))
        assert combinations.shape == (len(expected), size)
        assert [tuple(row) for row in combinations] == expected

    odds = np.array([1e-3, 2e-4, 5e-3, 1e-3])
    probabilities = residual_weights._get_proposal_probabilities(odds, 1.5)
    assert probabilities.sum() == pytest.approx(1.5)
    scaled_odds = probabilities / (1 - probabilities)
    assert np.allclose(scaled_odds / odds, scaled_odds[0] / odds[0])


def _build_circuit(layers: list[str], mechanisms: list[tuple[int, str, float]]) -> stim.Circuit:
    """A circuit with each error mechanism as an independent correlated error after its layer."""
    circuit = stim.Circuit()
    for index, layer in enumerate(layers):
        circuit += stim.Circuit(layer)
        for mechanism_layer, pauli, probability in mechanisms:
            if mechanism_layer == index:
                targets = [_pauli_target(term) for term in pauli.split("*")]
                circuit.append("CORRELATED_ERROR", targets, probability)
    return circuit


def _pauli_target(term: str) -> stim.GateTarget:
    """A Stim Pauli target, such as X0, from its text."""
    return {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[term[0]](int(term[1:]))


def _brute_force(
    layers: list[str],
    mechanisms: list[tuple[int, str, float]],
    num_data_qubits: int,
    css_weight: bool,
    max_residual_weight: int,
    pauli_type: PauliXZLike | None = None,
) -> tuple[float, np.ndarray]:
    """The acceptance probability and residual-weight distribution, by simulating every fault.

    A fault, a set of the given error mechanisms, is accepted if every measurement agrees with the
    noiseless run.  Its residual weight is the least Pauli weight (or weight of one Pauli type) of a
    Pauli error on the data qubits that maps the noiseless output state to the actual one, up to a
    phase; measured qubits are reset before the states are compared.
    """
    ideal = _simulate(layers, [])
    ideal_record = ideal.current_measurement_record()
    paulis = sorted(
        itertools.product("IXYZ", repeat=num_data_qubits),
        key=lambda pauli: _pauli_weight(pauli, css_weight, pauli_type),
    )
    accepted = 0.0
    masses = np.zeros(max_residual_weight + 2)
    for occurs in itertools.product([False, True], repeat=len(mechanisms)):
        probability = math.prod(
            mechanism[2] if occurred else 1 - mechanism[2]
            for mechanism, occurred in zip(mechanisms, occurs)
        )
        simulator = _simulate(
            layers, [mechanism for mechanism, occurred in zip(mechanisms, occurs) if occurred]
        )
        if simulator.current_measurement_record() != ideal_record:
            continue
        state = simulator.state_vector()
        for pauli in paulis:
            moved = ideal.copy()
            moved.do(stim.PauliString("".join(pauli).replace("I", "_")))
            if abs(np.vdot(moved.state_vector(), state)) > 1 - 1e-4:
                weight = _pauli_weight(pauli, css_weight, pauli_type)
                break
        accepted += probability
        masses[min(weight, max_residual_weight + 1)] += probability
    return accepted, masses / accepted


def _simulate(layers: list[str], mechanisms: list[tuple[int, str, float]]) -> stim.TableauSimulator:
    """Run the layers with the given error mechanisms applied, and reset the measured qubits."""
    simulator = stim.TableauSimulator()
    measured: set[int] = set()
    for index, layer in enumerate(layers):
        layer_circuit = stim.Circuit(layer)
        for instruction in layer_circuit:
            if instruction.name != "DETECTOR":
                simulator.do(instruction)
            if instruction.name == "M":
                measured.update(target.value for target in instruction.targets_copy())
        for mechanism_layer, pauli, _ in mechanisms:
            if mechanism_layer == index:
                for term in pauli.split("*"):
                    simulator.do(stim.Circuit(f"{term[0]} {term[1:]}"))
    for qubit in measured:
        simulator.reset(qubit)
    return simulator


def _pauli_weight(
    pauli: tuple[str, ...], css_weight: bool, pauli_type: PauliXZLike | None = None
) -> int:
    """The CSS or physical weight of a Pauli, or its weight of one type, as one letter per qubit."""
    if pauli_type is not None:
        return sum(letter in str(pauli_type).upper() + "Y" for letter in pauli)
    if not css_weight:
        return sum(letter != "I" for letter in pauli)
    return max(sum(letter in "XY" for letter in pauli), sum(letter in "ZY" for letter in pauli))


def _brute_force_weight(supports: np.ndarray, syndrome: int) -> int:
    """The least weight of a binary vector whose parities on the given supports are a syndrome."""
    num_qubits = supports.shape[1]
    best = num_qubits + 1
    for vector in itertools.product([0, 1], repeat=num_qubits):
        parities = (supports @ np.array(vector)) % 2
        if sum(int(bit) << index for index, bit in enumerate(parities)) == syndrome:
            best = min(best, sum(vector))
    return best


def _random_invertible(size: int, rng: np.random.Generator) -> galois.FieldArray:
    """A uniformly random invertible binary matrix."""
    while True:
        matrix = galois.GF2(rng.integers(2, size=(size, size)))
        if np.linalg.matrix_rank(matrix) == size:
            return matrix
