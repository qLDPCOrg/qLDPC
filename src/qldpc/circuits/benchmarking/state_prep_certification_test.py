# SPDX-License-Identifier: Apache-2.0

"""Tests for exact bounded-spread certification of noisy Clifford state preparation."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits
from qldpc.codes.distance._bounded_weight_search import PauliCosetSearch

from . import state_prep_certification


def _z_state(num_qubits: int) -> npt.NDArray[np.int_]:
    """Stabilizers of |0...0>, for which an error's effective weight is its syndrome's weight."""
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _pack_pauli(pauli: str) -> int:
    """Pack a Stim Pauli string as an integer, with X in the low bits and Z in the next block."""
    xs, zs = stim.PauliString(pauli).to_numpy()
    n = len(xs)
    return sum(int(x) << jj for jj, x in enumerate(xs)) | sum(
        int(z) << (n + jj) for jj, z in enumerate(zs)
    )


def _smallest_violation(flags: Sequence[int], syndromes: Sequence[int], budget: int) -> int | None:
    """Fewest faults in an accepted pattern that violates the spread bound on the |0...0> state."""
    for size in range(1, min(budget, len(flags)) + 1):
        for support in itertools.combinations(range(len(flags)), size):
            flag = syndrome = 0
            for jj in support:
                flag ^= flags[jj]
                syndrome ^= syndromes[jj]
            if flag == 0 and syndrome.bit_count() > size:
                return size
    return None


def _certify_map(
    flags: Sequence[int],
    syndromes: Sequence[int],
    budget: int,
    num_qubits: int,
    statistics: circuits.FaultSpreadStatistics | None = None,
) -> circuits.FaultSpreadCounterexample | None:
    """Certify a flag/syndrome map on |0...0> and check the result against brute force."""
    failure = state_prep_certification._certify_maps(
        flags,
        syndromes,
        _z_state(num_qubits),
        False,
        budget,
        statistics or circuits.FaultSpreadStatistics(),
    )
    smallest = _smallest_violation(flags, syndromes, budget)
    assert (failure is None) == (smallest is None)
    if failure is not None:
        assert failure.fault_count == smallest
        flag = syndrome = 0
        for index in failure.fault_indices:
            flag ^= flags[index]
            syndrome ^= syndromes[index]
        assert len(set(failure.fault_indices)) == failure.fault_count <= budget
        assert flag == 0
        assert failure.fault_count < failure.effective_weight_lower_bound <= syndrome.bit_count()
        assert failure.output_syndrome == tuple((syndrome >> jj) & 1 for jj in range(num_qubits))
        oracle = PauliCosetSearch(_z_state(num_qubits))
        assert oracle.syndrome(_pack_pauli(failure.output_pauli)) == syndrome
    return failure


def _true_minimum(stabilizers: Sequence[str], syndrome: Sequence[int], metric_css: bool) -> int:
    """Minimum weight over all Paulis with this syndrome, by enumerating every Pauli."""
    n = len(stabilizers)
    rows = np.array([np.concatenate(stim.PauliString(ss).to_numpy()) for ss in stabilizers])
    paulis = (np.arange(4**n)[:, None] >> np.arange(2 * n)) & 1
    x, z = paulis[:, :n], paulis[:, n:]
    syndromes = (x @ rows[:, n:].T + z @ rows[:, :n].T) % 2
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if metric_css else (x | z).sum(axis=1)
    return int(weights[np.all(syndromes == np.array(syndrome), axis=1)].min())


def _assert_valid_counterexample(
    certificate: circuits.FaultSpreadCertificate,
    circuit: stim.Circuit,
    post_select: Sequence[int],
) -> None:
    """Check a witness against the returned DEM and a brute-force minimum, not the search."""
    failure = certificate.counterexample
    assert failure is not None and not certificate.certified
    errors = [inst for inst in certificate.detector_error_model if inst.type == "error"]
    events: set[int] = set()
    for index in failure.fault_indices:
        events ^= {tt.val for tt in errors[index].targets_copy() if tt.is_relative_detector_id()}
    assert not events & set(post_select)
    syndrome = tuple(
        int(circuit.num_detectors + jj in events)
        for jj in range(len(certificate.target_stabilizers))
    )
    assert syndrome == failure.output_syndrome
    stabilizers = [stim.PauliString(ss) for ss in certificate.target_stabilizers]
    pauli = stim.PauliString(failure.output_pauli)
    assert tuple(int(not pauli.commutes(ss)) for ss in stabilizers) == failure.output_syndrome
    assert failure.effective_weight_lower_bound > failure.fault_count
    if len(stabilizers) <= 8:
        minimum = _true_minimum(certificate.target_stabilizers, syndrome, certificate.css_weight)
        assert failure.effective_weight_lower_bound <= minimum


def test_certify_maps_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """Random maps with repeated columns and flag rank above the budget match brute force."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    maps: list[tuple[int, npt.NDArray[np.int_], npt.NDArray[np.int_], Sequence[int]]] = []
    for n, rank in itertools.product((3, 5), range(5)):
        for _ in range(12):
            flag_array, syndrome_array = (
                rng.integers(1 << rank, size=8),
                rng.integers(1 << n, size=8),
            )
            maps.append((n, flag_array, syndrome_array, (1, 2, 3, 5)))
    # With 24 columns of flag rank 5, the independent-prefix search is the cheaper complete search.
    for _ in range(6):
        flag_array, syndrome_array = rng.integers(1, 32, size=24), rng.integers(1, 8, size=24)
        maps.append((3, flag_array, syndrome_array, (2, 3)))
    # One fault of weight zero certifies and one of weight three on three qubits does not, so both
    # outcomes occur whatever the random maps are.
    maps.append((3, np.array([0]), np.array([0]), (1,)))
    maps.append((3, np.array([0]), np.array([7]), (1,)))
    outcomes = {True: 0, False: 0}
    for n, flag_array, syndrome_array, budgets in maps:
        flags, syndromes = list(map(int, flag_array)), list(map(int, syndrome_array))
        for budget in budgets:
            outcomes[_certify_map(flags, syndromes, budget, n) is None] += 1
    assert all(outcomes.values())


def test_hand_derived_dependency_cases() -> None:
    """Triples, decomposable pairs, duplicate joint symptoms, zero columns, and budget limits."""
    for flags, syndromes, budget, certified in (
        ([1, 2, 3], [1, 2, 12], 2, True),  # the triple of weight 4 needs three faults
        ([1, 2, 3], [1, 2, 12], 3, False),
        ([1, 1], [1, 14], 3, False),  # an accepted pair of weight 4
        ([0, 0, 0], [1, 2, 4], 4, True),  # flag-free faults of weight one
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),  # pairs of weight two
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),  # identical columns cancel in pairs
        ([0, 0], [0, 0], 4, True),  # faults with no effect
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 3, True),
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 4, False),  # four faults of weight six
    ):
        failure = _certify_map(flags, syndromes, budget, 6)
        assert (failure is None) == certified


def test_faults_of_weight_at_most_one_certify_every_budget() -> None:
    """Without a fault that spreads on its own, no pattern can exceed its fault count."""
    flags = [3, 5, 6, 9, 10, 12, 1, 2, 4, 8, 7, 11]
    syndromes = [1, 2, 4, 8, 0, 1, 2, 4, 8, 0, 1, 2]
    statistics = circuits.FaultSpreadStatistics()
    assert _certify_map(flags, syndromes, 30, 4, statistics) is None
    assert statistics.effective_budget == 1 and statistics.minimal_candidates == 0

    # One fault of weight three brings back the search, and it fails with the equal-flag fault.
    statistics = circuits.FaultSpreadStatistics()
    failure = _certify_map(flags + [3], syndromes + [14], 13, 4, statistics)
    assert failure is not None and failure.fault_count == 2
    assert statistics.effective_budget > 1


def test_failures_report_the_fewest_faults() -> None:
    """A violation of two faults is reported although one of three faults is also present."""
    # Columns 0-2 form an accepted triple of weight four, and columns 3 and 4 an accepted pair of
    # weight three.  The fillers add flags 8 to 56, whose accepted patterns have weight at most
    # three, and make the independent-prefix search the cheaper one.
    flags, syndromes = [1, 2, 3, 4, 4], [1, 2, 12, 1, 6]
    fillers = list(itertools.product(range(8, 64, 8), (0, 1, 2, 4, 8, 16, 32)))
    methods = set()
    for extra in ([], fillers):
        statistics = circuits.FaultSpreadStatistics()
        failure = _certify_map(
            flags + [flag for flag, _ in extra],
            syndromes + [syndrome for _, syndrome in extra],
            3,
            6,
            statistics,
        )
        assert failure is not None
        assert failure.fault_count == 2 and set(failure.fault_indices) == {3, 4}
        methods.add(statistics.search_method)
    assert methods == {"brouwer_zimmermann", "independent_prefix"}


def test_independent_flag_types_need_only_equal_flag_pairs() -> None:
    """For independent flag columns, only equal-flag pairs matter, at any column count."""
    outcomes = set()
    # Flags 3, 5, 8, 16, and 2**130 are independent; three copies exceed twice the flag rank.
    for copies in (2, 3):
        flags = [flag for flag in (3, 5, 8, 16, 1 << 130) for _ in range(copies)]
        passing = [3, 5, 6] * 5  # each pair has weight two
        failing = [3, 12, 0] * 5
        for syndromes in (passing[: len(flags)], failing[: len(flags)]):
            for budget in (2, 3, 30):
                statistics = circuits.FaultSpreadStatistics()
                failure = _certify_map(flags, syndromes, budget, 4, statistics)
                assert statistics.effective_budget == 2  # only pairs of equal flags matter
                outcomes.add(failure is None)
    assert outcomes == {True, False}


def test_minimal_patterns_match_exhaustive_enumeration(pytestconfig: pytest.Config) -> None:
    """Both complete searches yield every minimal accepted pattern of a given size."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for width in range(1, 10):
        for flag_bits in range(1, 5):
            flags = list(map(int, rng.integers(1, 1 << flag_bits, size=width)))
            syndromes = list(map(int, rng.integers(16, size=width)))
            originals = list(map(int, rng.permutation(np.arange(width) * 7 + 100)))
            mechanisms = list(zip(flags, syndromes, originals, strict=True))
            expected: dict[int, dict[tuple[int, ...], int]] = {}
            accepted: list[int] = []
            for size in range(1, 6):
                expected[size] = {}
                for support in itertools.combinations(range(width), size):
                    flag = syndrome = 0
                    for jj in support:
                        flag ^= flags[jj]
                        syndrome ^= syndromes[jj]
                    word = sum(1 << jj for jj in support)
                    if flag == 0:
                        if not any(previous & word == previous for previous in accepted):
                            expected[size][tuple(originals[jj] for jj in support)] = syndrome
                        accepted.append(word)
            for size in range(2, 6):
                statistics = circuits.FaultSpreadStatistics()
                prefix = list(
                    state_prep_certification._iter_minimal_patterns(mechanisms, size, statistics)
                )
                assert len(prefix) == len(expected[size])
                assert dict(prefix) == expected[size]
                # Kernel coverage may reach a word from more than one information set.
                kernel = state_prep_certification._iter_kernel_patterns(
                    mechanisms, size, statistics
                )
                assert dict(kernel) == expected[size]


def test_minimal_patterns_with_flags_wider_than_a_machine_word() -> None:
    """Flag columns of 80 and 130 bits combine correctly."""
    mechanisms = [(1 << 80, 1, 90), (1 << 130, 2, 10), ((1 << 80) ^ (1 << 130), 4, 50)]
    statistics = circuits.FaultSpreadStatistics()
    prefix = state_prep_certification._iter_minimal_patterns(mechanisms, 3, statistics)
    assert list(prefix) == [((90, 10, 50), 7)]
    kernel = state_prep_certification._iter_kernel_patterns(mechanisms, 3, statistics)
    assert set(kernel) == {((90, 10, 50), 7)}


def test_flagged_circuit_with_permuted_signed_outputs() -> None:
    """Many flags, a signed Y stabilizer, and permuted outputs; witnesses name DEM mechanisms."""
    for damaging in (False, True):
        circuit = stim.Circuit()
        circuit.append("I", range(9))
        for flag in range(1, 16):
            for syndrome in (3, 5, 6):
                targets = [stim.target_x(jj) for jj in range(3) if (syndrome >> jj) & 1]
                targets += [stim.target_x(5 + jj) for jj in range(4) if (flag >> jj) & 1]
                circuit.append("CORRELATED_ERROR", targets, 0.001)
        if damaging:  # also flips the outputs on qubits 3 and 4, with the flag on qubit 5
            circuit.append("CORRELATED_ERROR", [stim.target_x(jj) for jj in (3, 4, 5)], 0.001)
        circuit += stim.Circuit("H 0 1 2\nS 0 2\nX 0")
        circuit.append("M", range(5, 9))
        for jj in range(4):
            circuit.append("DETECTOR", [stim.target_rec(jj - 4)])
        result = circuits.certify_state_prep_spread(
            circuit, output_qubits=[2, 0, 4, 1, 3], post_select=range(4), max_faults=3
        )
        assert any("Y" in ss for ss in result.target_stabilizers)
        assert any(ss.startswith("-") for ss in result.target_stabilizers)
        assert result.certified != damaging
        if damaging:
            _assert_valid_counterexample(result, circuit, range(4))


def test_steane_preparation_meets_css_spread_but_not_physical_spread() -> None:
    """Steane preparation: a single Y0 Z1 fault has CSS weight one but physical weight two."""
    circuit = circuits.DepolarizingNoiseModel(1e-3).noisy_circuit(
        stim.Circuit("""
            H 0 2 4
            CX 0 3 2 1 4 5
            CX 0 1 2 6 4 3
            CX 2 5 3 6
            H 7
            CZ 7 1 7 3 7 5
            MX 7
            DETECTOR rec[-1]
        """)
    )

    def certify(css_weight: bool | None) -> circuits.FaultSpreadCertificate:
        return circuits.certify_state_prep_spread(
            circuit, output_qubits=range(7), max_faults=3, post_select=[0], css_weight=css_weight
        )

    css = certify(None)
    assert css.certified and css.css_weight is True
    # The default outputs are qubits 0-6: the flag qubit 7 is measured.
    assert circuits.certify_state_prep_spread(circuit, max_faults=3, post_select=[0]) == css
    physical = certify(False)
    assert physical.css_weight is False
    _assert_valid_counterexample(physical, circuit, [0])
    failure = physical.counterexample
    assert failure is not None
    # Every Pauli of weight at most one has a different syndrome, so its class is heavier.
    stabilizers = [stim.PauliString(ss) for ss in physical.target_stabilizers]
    for qubit, pauli in itertools.product(range(7), "IXYZ"):
        candidate = stim.PauliString(7)
        candidate[qubit] = pauli
        syndrome = tuple(int(not candidate.commutes(ss)) for ss in stabilizers)
        assert syndrome != failure.output_syndrome


@pytest.mark.parametrize("css_weight", [None, True, False])
def test_state_selection_and_overrides(css_weight: bool | None) -> None:
    """Automatic selection picks CSS weight for |0+> only, and an explicit value overrides it."""
    css_state = stim.Circuit("I 0\nH 1\nCORRELATED_ERROR(0.01) X0 Z1")  # X0 Z1 has CSS weight one
    result = circuits.certify_state_prep_spread(
        css_state, output_qubits=[0, 1], max_faults=3, css_weight=css_weight
    )
    assert result.css_weight is (css_weight is not False)
    assert result.certified is (css_weight is not False)
    graph_state = stim.Circuit("H 0 1\nCZ 0 1\nCORRELATED_ERROR(0.001) X0 X1")
    result = circuits.certify_state_prep_spread(
        graph_state, output_qubits=[0, 1], max_faults=3, css_weight=css_weight
    )
    assert result.certified and result.css_weight is (css_weight is True)


def test_css_detection_depends_on_the_stabilizer_row_space() -> None:
    """Generators that mix X and Z can still span a CSS state, and a logical Y state is not CSS."""

    def matrix(generators: Sequence[str]) -> npt.NDArray[np.uint8]:
        return np.array(
            [np.concatenate(stim.PauliString(gg).to_numpy()) for gg in generators], dtype=np.uint8
        )

    def state(circuit: str) -> npt.NDArray[np.uint8]:
        tableau = stim.Tableau.from_circuit(stim.Circuit(circuit))
        return matrix([str(gg) for gg in tableau.to_stabilizers()])

    # These products of the stabilizers X0, Z1, and Z2 of |+00> each mix X and Z.
    assert state_prep_certification._is_css_state(matrix(["XZ_", "X_Z", "XZZ"]))
    assert state_prep_certification._is_css_state(state("H 0\nI 1 2"))
    assert not state_prep_certification._is_css_state(state("H 0 1\nCZ 0 1"))
    # A CSS repetition code can prepare logical Y, whose stabilizers are not pure X or Z.
    assert not state_prep_certification._is_css_state(state("H 0\nS 0\nCX 0 1 0 2"))


def test_css_override_minimizes_jointly_on_a_non_css_state() -> None:
    """On a path graph state, Z^4 is equivalent to pure X and to pure Z, but not to both at once."""
    result = circuits.certify_state_prep_spread(
        stim.Circuit("H 0 1 2 3\nCZ 0 1 1 2 2 3\nCORRELATED_ERROR(0.001) Z0 Z1 Z2 Z3"),
        output_qubits=range(4),
        max_faults=1,
        css_weight=True,
    )
    assert result.counterexample is not None
    assert result.counterexample.effective_weight_lower_bound == 2


@pytest.mark.parametrize("css_weight", [True, False])
def test_flag_rank_above_the_fault_budget(css_weight: bool) -> None:
    """With 65 flags, one accepted triple has CSS weight three but physical weight six."""
    rank, order = 65, 3
    n = 2 * order
    circuit = stim.Circuit()
    circuit.append("H", range(order, n))  # |0>^3 |+>^3
    for bit in range(rank):  # one fault per flag
        circuit.append("CORRELATED_ERROR", [stim.target_x(n + bit)], 0.001)
    targets = [stim.target_x(jj) for jj in range(order)]
    targets += [stim.target_z(jj) for jj in range(order, n)]
    targets += [stim.target_x(n + bit) for bit in range(order - 1)]  # cancels flags 0 and 1
    circuit.append("CORRELATED_ERROR", targets, 0.001)
    circuit.append("M", range(n, n + rank))
    for bit in range(rank):
        circuit.append("DETECTOR", [stim.target_rec(bit - rank)])
    for budget in (order - 1, order):
        result = circuits.certify_state_prep_spread(
            circuit,
            output_qubits=range(n),
            post_select=range(rank),
            max_faults=budget,
            css_weight=css_weight,
        )
        assert result.certified is (budget < order or css_weight)
        assert result.css_weight is css_weight
        if not result.certified:
            _assert_valid_counterexample(result, circuit, range(rank))
            assert result.counterexample is not None
            assert result.counterexample.fault_count == order


@pytest.mark.parametrize("css_weight", [None, False])
def test_four_ghz_blocks_have_global_lower_bound_four(css_weight: bool | None) -> None:
    """Four GHZ blocks hit by one accepted triple need weight four, not five."""
    circuit = stim.Circuit()
    for block in range(0, 12, 3):
        circuit.append("H", [block])
        circuit.append("CX", [block, block + 1, block, block + 2])
    circuit.append("CORRELATED_ERROR", [stim.target_x(12)], 0.001)
    circuit.append("CORRELATED_ERROR", [stim.target_x(13)], 0.001)
    circuit.append("CORRELATED_ERROR", [stim.target_x(jj) for jj in (2, 5, 8, 11, 12, 13)], 0.001)
    circuit.append("M", [12, 13])
    circuit.append("DETECTOR", [stim.target_rec(-2)])
    circuit.append("DETECTOR", [stim.target_rec(-1)])

    def certify(max_faults: int) -> circuits.FaultSpreadCertificate:
        return circuits.certify_state_prep_spread(
            circuit,
            output_qubits=range(12),
            post_select=[0, 1],
            max_faults=max_faults,
            css_weight=css_weight,
        )

    assert certify(2).certified
    certificate = certify(3)
    assert certificate.css_weight is (css_weight is None)
    _assert_valid_counterexample(certificate, circuit, [0, 1])
    failure = certificate.counterexample
    assert failure is not None
    # X on each block's last qubit flips two Z-type stabilizers there, so every equivalent error
    # has an X component on all four blocks.  The minimum is four in both metrics.
    assert failure.fault_indices == (0, 1, 2)
    assert failure.effective_weight_lower_bound == 4


def test_stabilizers_keep_signs_and_the_output_frame() -> None:
    """Reported stabilizers are +1 eigenoperators of the ideal state on the chosen output qubits."""
    circuit = stim.Circuit("H 4\nS 4\nCX 4 1\nX 1\nCORRELATED_ERROR(0.01) Y4")
    result = circuits.certify_state_prep_spread(circuit, output_qubits=[4, 1], max_faults=3)
    assert result.certified and result.output_qubits == (4, 1)
    assert any(ss.startswith("-") for ss in result.target_stabilizers)
    simulator = stim.TableauSimulator()
    simulator.do(circuit.without_noise())
    for text in result.target_stabilizers:
        physical = stim.PauliString(circuit.num_qubits)
        for qubit, pauli in zip(result.output_qubits, text[1:], strict=True):
            physical[qubit] = pauli
        assert simulator.peek_observable_expectation(-physical if text[0] == "-" else physical) == 1


def test_default_outputs_skip_measured_and_untouched_qubits() -> None:
    """Outputs default to the qubits that are acted on and never measured, in increasing order."""
    # Qubits 0, 2, and 3 are untouched, and qubit 5 is measured.
    circuit = stim.Circuit("H 4\nS 4\nCX 4 1\nX 1\nCORRELATED_ERROR(0.01) Y4\nCX 5 1\nM 5")
    default = circuits.certify_state_prep_spread(circuit, max_faults=3)
    assert default.output_qubits == (1, 4)
    assert default == circuits.certify_state_prep_spread(
        circuit, max_faults=3, output_qubits=[1, 4]
    )
    assert default != circuits.certify_state_prep_spread(
        circuit, max_faults=3, output_qubits=[4, 1]
    )


def test_unmeasured_qubits_account_for_every_kind_of_measurement() -> None:
    """Resets, inverted and Pauli-product measurements, repeats, and noise-only qubits."""
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 9
        H 0 1 2 3 4 5 6 7
        MR 1
        M !2
        MPP X3*Z4
        MXX 5 6
        REPEAT 2 {
            MX 7
        }
        DETECTOR rec[-1]
        CORRELATED_ERROR(0.1) X8
    """)
    assert state_prep_certification._get_unmeasured_qubits(circuit) == (0, 8)


def test_added_flags_match_explicit_flag_detectors() -> None:
    """Bare measurements become flag detectors, and addressed measurements do not."""
    body = """
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
    """

    def certify(
        circuit: stim.Circuit,
        post_select: bool | Sequence[int] | Literal["flags"],
        add_flags: bool,
    ) -> circuits.FaultSpreadCertificate:
        return circuits.certify_state_prep_spread(
            circuit, max_faults=3, post_select=post_select, add_flags=add_flags
        )

    explicit = stim.Circuit(body + "DETECTOR rec[-1]")
    reference = certify(explicit, [0], add_flags=False)
    assert not reference.certified and reference.counterexample is not None
    assert reference.counterexample.fault_count == 2
    # The added flag is detector 0, so these all select it.
    for post_select in ("flags", True, [0], [-1]):
        assert certify(stim.Circuit(body), post_select, add_flags=True) == reference
    # The measurement is already addressed, so there is no flag to select and a single fault fails.
    unflagged = certify(explicit, "flags", add_flags=True)
    assert unflagged.counterexample is not None and unflagged.counterexample.fault_count == 1


def test_invalid_post_selection_is_rejected() -> None:
    """Flags need add_flags, and selections are distinct integer indices of selectable detectors."""
    circuit = stim.Circuit("I 0\nCORRELATED_ERROR(0.01) X0\nM 1\nDETECTOR rec[-1]")
    invalid: list[Any] = ["flags", "all", [1], [-2], [0, -1], [0.5], [False]]
    for post_select in invalid:
        with pytest.raises(ValueError):
            circuits.certify_state_prep_spread(
                circuit, max_faults=1, output_qubits=[0], post_select=post_select
            )


def test_correlated_mechanism_costs_one_fault() -> None:
    """A two-qubit error is one fault, reported in the order of the requested output qubits."""
    circuit = stim.Circuit("I 0 1\nCORRELATED_ERROR(0.01) X0 Y1")
    result = circuits.certify_state_prep_spread(circuit, output_qubits=[1, 0], max_faults=3)
    _assert_valid_counterexample(result, circuit, [])
    failure = result.counterexample
    assert failure is not None
    assert failure.fault_indices == (0,) and failure.effective_weight_lower_bound == 2
    assert result.output_qubits == (1, 0)


def test_post_selection_can_accept_a_damaging_pair() -> None:
    """Two faults that each trip the detector cancel it together and damage four outputs."""
    circuit = stim.Circuit("""
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
        DETECTOR rec[-1]
    """)

    def certify(
        max_faults: int, post_select: bool | Sequence[int] = (0,)
    ) -> circuits.FaultSpreadCertificate:
        return circuits.certify_state_prep_spread(
            circuit, output_qubits=[0, 1, 2, 3], post_select=post_select, max_faults=max_faults
        )

    assert certify(1).certified
    result = certify(3)
    _assert_valid_counterexample(result, circuit, [0])
    assert result.counterexample is not None and result.counterexample.fault_count == 2

    # True selects every detector of the circuit, here detector 0, and False selects none.
    assert certify(3, True) == result
    unselected = certify(1, False)
    assert not unselected.certified
    assert unselected == circuits.certify_state_prep_spread(
        circuit, output_qubits=[0, 1, 2, 3], max_faults=1
    )


def test_circuit_without_faults_certifies() -> None:
    """Deterministic nonzero measurements, repeats, and observables need no search."""
    circuit = stim.Circuit("""
        X 0
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    result = circuits.certify_state_prep_spread(
        circuit, output_qubits=[0], post_select=[0, 1], max_faults=4
    )
    assert result.certified and result.target_stabilizers == ("-Z",)


@pytest.mark.parametrize("css_weight", ["css", np.bool_(True)])
def test_invalid_css_weight_is_rejected(css_weight: bool) -> None:
    """Only bool and None are accepted, and numpy booleans are not bool."""
    with pytest.raises(ValueError, match="css_weight"):
        circuits.certify_state_prep_spread(
            stim.Circuit("I 0"), output_qubits=[0], max_faults=1, css_weight=css_weight
        )


@pytest.mark.parametrize("budget", [0, True, 1.5])
def test_invalid_fault_budget_is_rejected(budget: int) -> None:
    """The budget must be a positive integer, and a bool is not an integer here."""
    with pytest.raises(ValueError, match="max_faults"):
        circuits.certify_state_prep_spread(
            stim.Circuit("I 0"), output_qubits=[0], max_faults=budget
        )


@pytest.mark.parametrize("outputs", [[], [0, 0], [1], [True], [0.5]])
def test_invalid_output_qubits_are_rejected(outputs: list[int]) -> None:
    """Output qubits must be a nonempty list of distinct in-range integers."""
    with pytest.raises(ValueError, match="output_qubits"):
        circuits.certify_state_prep_spread(stim.Circuit("I 0"), output_qubits=outputs, max_faults=1)


def test_unsupported_circuits_and_detectors_are_rejected() -> None:
    """Mixed or random outputs, nondeterministic annotations, and unknown detectors raise."""
    # An output entangled with an unselected qubit, and an output that was measured.
    for text in ("H 0\nCX 0 1", "H 0\nM 0"):
        with pytest.raises(ValueError):
            circuits.certify_state_prep_spread(stim.Circuit(text), output_qubits=[0], max_faults=1)
    # Random detectors and observables, even when selected for post-selection.
    for text in (
        "H 1\nM 1\nDETECTOR rec[-1]",
        "H 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ):
        with pytest.raises(ValueError):
            circuits.certify_state_prep_spread(stim.Circuit(text), output_qubits=[0], max_faults=1)
    with pytest.raises(ValueError):
        circuits.certify_state_prep_spread(
            stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]"),
            output_qubits=[0],
            post_select=[0],
            max_faults=1,
        )
    with pytest.raises(ValueError):
        circuits.certify_state_prep_spread(
            stim.Circuit("I 0"), output_qubits=[0], post_select=[0], max_faults=1
        )
    # There are no unmeasured qubits to default to.
    with pytest.raises(ValueError):
        circuits.certify_state_prep_spread(stim.Circuit("H 0\nM 0"), max_faults=1)
