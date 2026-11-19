# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary distances of plain arrays and distance method validation."""

from __future__ import annotations

import copy
import functools
import math
import pickle
from collections.abc import Callable
from typing import Any, cast, get_args
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
import scipy.sparse
from sympy.abc import x

import qldpc
from qldpc._binary import get_independent_rows
from qldpc.external import _bliss
from qldpc.objects import Pauli

from ..conftest import Tune
from . import _brute_force, core
from ._binary import get_css_stabilizers
from ._connected_cluster.search import SupportSearch
from ._tuning import use_tuning
from .conftest import get_pauli_weight, get_random_full_rank_matrix, get_span

# "auto" enumerates the small codes of the tests that take every method, unless its tuning makes
# enumeration costly (test_auto_without_enumeration_gives_the_same_distances).
METHODS: list[core.DistanceMethod] = [
    "auto",
    "brouwer_zimmermann",
    "connected_cluster",
    "brute_force",
]


@pytest.mark.parametrize("method", METHODS)
def test_css_distances(method: core.DistanceMethod) -> None:
    """CSS distances of sectors, of codes, of subsystem codes, and of codes that encode nothing."""
    checks_x = [[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0, 0]]
    checks_z = [[1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0, 0]]
    assert core.get_distance_css(checks_x, checks_z, "X", method=method) == 3
    assert core.get_distance_css(checks_x, checks_z, Pauli.Z, method=method) == 2
    assert core.get_distance_css(checks_x, checks_z, method=method) == 2
    assert core.get_distance_css(checks_x, checks_z, "x", method=method, max_weight=1) == 2
    assert core.get_distance_css(checks_x, checks_z, method=method, max_weight=0) == 1

    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    distance = core.get_distance_css(
        bacon_shor_code.matrix_x, bacon_shor_code.matrix_z, method=method
    )
    assert distance == 3
    assert core.get_distance_css([[1, 1]], [[1, 1]], method=method) is core.NO_DISTANCE

    # These gauge generators leave no stabilizers, so X on one qubit is a dressed logical operator,
    # although the bare logical X operator, X on every qubit, has weight 3.
    gauge_x, gauge_z = [[0, 1, 1], [1, 1, 0]], [[1, 0, 1], [1, 1, 0]]
    assert core.get_distance_css(gauge_x, gauge_z, "X", method=method) == 1

    # Empty X checks take the width of the Z checks, so the X logical operators are the words of the
    # Hamming code, and Z on any qubit is a logical operator.
    hamming_checks = qldpc.codes.HammingCode(3).matrix
    assert core.get_distance_css([], hamming_checks, "X", method=method) == 3
    assert core.get_distance_css(np.zeros((0, 0)), hamming_checks, method=method) == 1
    assert core.get_distance_css(np.zeros((0, 0)), [], method=method) is core.NO_DISTANCE


@pytest.mark.parametrize("method", METHODS)
def test_distances_from_checks(method: core.DistanceMethod) -> None:
    """Checks give classical, CSS, and stabilizer distances, and excluded rows must satisfy them."""
    hamming_code = qldpc.codes.HammingCode(3)
    steane_code = qldpc.codes.SteaneCode()
    five_qubit_code = qldpc.codes.FiveQubitCode()
    assert core.get_distance_from_checks(hamming_code.matrix, method=method) == 3
    assert core.get_distance_from_checks(hamming_code.matrix, method=method, max_weight=1) == 2
    distance = core.get_distance_from_checks(
        steane_code.matrix_z, steane_code.matrix_x, method=method
    )
    assert distance == 3
    distance = core.get_distance_from_checks(
        five_qubit_code.matrix, five_qubit_code.matrix, symplectic=True, method=method
    )
    assert distance == 3

    # Excluded rows that span every word that satisfies the checks leave no distance, even when they
    # are redundant, as the repeated row makes them here.
    generator = np.asarray(hamming_code.generator)
    distance = core.get_distance_from_checks(
        hamming_code.matrix, np.vstack([generator, generator[:1]]), method=method
    )
    assert distance is core.NO_DISTANCE
    with pytest.raises(ValueError, match="satisfy the checks"):
        core.get_distance_from_checks([[1, 1, 0]], [[1, 0, 0]], method=method)

    # Empty checks take the width of the excluded rows, and every word satisfies them.
    assert core.get_distance_from_checks([], [[1, 1, 0]], method=method) == 1


@pytest.mark.parametrize("method", METHODS)
def test_distances_of_stabilizer_and_subsystem_codes(method: core.DistanceMethod) -> None:
    """Pauli checks give the distances of stabilizer codes, and dressed ones of subsystem codes."""
    five_qubit_code = qldpc.codes.FiveQubitCode()
    assert core.get_distance_pauli(np.vstack([five_qubit_code.matrix] * 2), method=method) == 3
    assert core.get_distance_pauli(five_qubit_code.matrix, method=method, max_weight=1) == 2
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    assert core.get_distance_pauli(bacon_shor_code.matrix, method=method) == 3
    assert core.get_distance_pauli(bacon_shor_code.matrix, method=method, max_weight=1) == 2
    stabilizer_state = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
    assert core.get_distance_pauli(stabilizer_state, method=method) is core.NO_DISTANCE

    # The CSS gauge generators [[0, 1, 1], [1, 1, 0]] and [[1, 0, 1], [1, 1, 0]] leave no
    # stabilizers.  An S gate on qubit 1 makes them non-CSS, and the last row of gauge is redundant.
    # A single-qubit Pauli is a dressed logical operator, but every bare one weighs 3.
    gauge = [
        [0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0],
    ]
    assert core.get_distance_pauli(gauge, method=method) == 1
    no_stabilizers = np.zeros((0, 6), dtype=np.uint8)
    assert core.get_distance_from_checks(gauge, no_stabilizers, symplectic=True, method=method) == 3


def test_auto_without_enumeration_gives_the_same_distances() -> None:
    """With enumeration never cheap, "auto" searches otherwise, and finds the same distances."""
    with use_tuning(auto_brute_force_max_words=0):
        test_css_distances("auto")
        test_distances_from_checks("auto")
        test_distances_of_stabilizer_and_subsystem_codes("auto")
        test_bare_and_dressed_distances("auto")


def test_auto_takes_the_method_of_its_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default, a search enumerates if that is cheap, and otherwise takes the method of its kind.

    Words that only generators give, and binary words that exclude nothing nonzero, take the BZ
    method, and other words a search by cost, including Pauli strings that exclude nothing and CSS
    sectors of which only some exclude nothing.
    """
    taken: list[str] = []
    for name in ("get_nested_distance_brute_force", "get_distance_brouwer_zimmermann"):
        monkeypatch.setattr(core, name, _record_call(taken, name, getattr(core, name)))
    monkeypatch.setattr(core, "search_by_cost", _record_call(taken, "by cost", core.search_by_cost))

    hamming_code = qldpc.codes.HammingCode(3)
    steane_code = qldpc.codes.SteaneCode()
    five_qubit_code = qldpc.codes.FiveQubitCode()
    # The X checks of the last two searches are empty.  So their X sector excludes nothing, and
    # every Z-type string satisfies the checks, so the Z sector has words of weight 1.
    searches: list[tuple[Callable[[], Any], str, int]] = [
        (
            lambda: core.get_distance_from_checks(hamming_code.matrix),
            "get_distance_brouwer_zimmermann",
            3,
        ),
        (
            lambda: core.get_distance_from_generators(hamming_code.generator),
            "get_distance_brouwer_zimmermann",
            3,
        ),
        (
            lambda: core.get_distance_from_generators(
                steane_code.get_logical_ops(Pauli.X), steane_code.get_stabilizer_ops(Pauli.X)
            ),
            "get_distance_brouwer_zimmermann",
            3,
        ),
        (
            lambda: core.get_distance_from_checks(steane_code.matrix_z, steane_code.matrix_x),
            "by cost",
            3,
        ),
        (lambda: core.get_distance_css(steane_code.matrix_x, steane_code.matrix_z), "by cost", 3),
        (lambda: core.get_distance_pauli(five_qubit_code.matrix), "by cost", 3),
        (
            lambda: core.get_distance_from_checks(
                five_qubit_code.matrix, np.zeros((0, 10), int), symplectic=True
            ),
            "by cost",
            3,
        ),
        (
            lambda: core.get_distance_css(np.zeros((0, 7), int), hamming_code.matrix, "X"),
            "get_distance_brouwer_zimmermann",
            3,
        ),
        (
            lambda: core.get_distance_css(np.zeros((0, 7), int), hamming_code.matrix),
            "by cost",
            1,
        ),
    ]
    for search, method, distance in searches:
        taken.clear()
        assert search() == distance
        assert taken[:1] == ["get_nested_distance_brute_force"]
        taken.clear()
        with use_tuning(auto_brute_force_max_words=0):
            assert search() == distance
        assert taken[:1] == [method]


def _record_call(taken: list[str], name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to record its name in ``taken`` whenever it is called."""

    def recording(*args: Any, **kwargs: Any) -> Any:
        taken.append(name)
        return function(*args, **kwargs)

    return recording


def test_pauli_checks_of_css_codes_are_searched_as_css_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pauli checks that generate a CSS group are searched as CSS, even if they mix X and Z."""
    # Adding the Z-type checks of the Steane code to its X-type checks mixes X and Z in the rows,
    # but keeps the group that they generate.
    steane_code = qldpc.codes.SteaneCode()
    checks_x = steane_code.matrix_x.view(np.ndarray)
    checks_z = steane_code.matrix_z.view(np.ndarray)
    checks = np.block([[checks_x, checks_z], [np.zeros_like(checks_z), checks_z]])

    searched: list[tuple[list[int], bool]] = []

    def record(sectors: Any, **kwargs: Any) -> Any:
        widths = [np.shape(sector_checks)[1] for sector_checks, _ in sectors]
        searched.append((widths, kwargs.get("symplectic", False)))
        return get_distance_connected_cluster(sectors, **kwargs)

    get_distance_connected_cluster = core.get_distance_connected_cluster
    monkeypatch.setattr(core, "get_distance_connected_cluster", record)
    # The identity maps one sector of the Steane code onto the other, so turning off sector_maps
    # keeps both sectors.
    with use_tuning(sector_maps=False):
        assert core.get_distance_pauli(checks, method="connected_cluster") == 3
    assert searched == [([7, 7], False)]


@pytest.mark.parametrize("method", ["auto", "brouwer_zimmermann", "connected_cluster"])
def test_css_searches_search_one_sector_if_a_permutation_maps_it_onto_the_other(
    monkeypatch: pytest.MonkeyPatch, method: core.DistanceMethod
) -> None:
    """A search takes the X sector alone if a qubit permutation maps it onto the Z sector.

    The X-type and Z-type checks of the Steane code have the same row space, so the identity maps
    each sector onto the other.  Bliss finds a permutation for the surface code of distance 3 and
    for the Bacon-Shor code of 3 x 3 qubits, a subsystem code.  The rectangular surface and
    Bacon-Shor codes have none, since their X and Z distances differ.  With the switch
    ``sector_maps`` off, a search never looks.  Every search finds the distance that brute force,
    which searches both sectors, finds.
    """
    searched: list[int] = []

    def record(name: str) -> None:
        function = getattr(core, name)

        def recording(sectors: Any, *args: Any, **kwargs: Any) -> Any:
            searched.append(len(sectors))
            return function(sectors, *args, **kwargs)

        monkeypatch.setattr(core, name, recording)

    for name in ("get_distance_brouwer_zimmermann", "get_distance_connected_cluster"):
        record(name)
    record("search_by_cost")
    looked: list[str] = []
    monkeypatch.setattr(core, "get_sector_map", _record_call(looked, "", core.get_sector_map))

    num_sectors_with_bliss = 1 if _bliss.is_installed() else 2
    cases = [
        (qldpc.codes.SteaneCode(), 1),
        (qldpc.codes.SurfaceCode(3), num_sectors_with_bliss),
        (qldpc.codes.BaconShorCode(3), num_sectors_with_bliss),
        (qldpc.codes.SurfaceCode(3, 4), 2),
        (qldpc.codes.BaconShorCode(3, 4), 2),
    ]
    for code, num_sectors in cases:
        checks_x, checks_z = code.matrix_x.view(np.ndarray), code.matrix_z.view(np.ndarray)
        distance = core.get_distance_css(checks_x, checks_z, method="brute_force")
        bounded = core.get_distance_css(checks_x, checks_z, method="brute_force", max_weight=2)
        for switch, expected in ((True, num_sectors), (False, 2)):
            searched.clear()
            looked.clear()
            with use_tuning(sector_maps=switch, auto_brute_force_max_words=0):
                assert core.get_distance_css(checks_x, checks_z, method=method) == distance
                assert (
                    core.get_distance_css(checks_x, checks_z, method=method, max_weight=2)
                    == bounded
                )
            assert searched[:1] == [expected] and searched[-1] == expected
            assert len(looked) == (2 if switch else 0)


def test_brute_force_never_looks_for_sector_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Brute force, also when "auto" takes it, searches both sectors without looking for a map."""
    looked: list[str] = []
    monkeypatch.setattr(core, "get_sector_map", _record_call(looked, "", core.get_sector_map))
    searched: list[str] = []
    monkeypatch.setattr(
        core, "_search_checks", _record_call(searched, "sector", core._search_checks)
    )
    steane_code = qldpc.codes.SteaneCode()
    for method in ("brute_force", "auto"):
        with use_tuning(sector_maps=True, auto_brute_force_max_words=2**40):
            distance = core.get_distance_css(
                steane_code.matrix_x, steane_code.matrix_z, method=method
            )
        assert distance == 3
    assert not looked and len(searched) == 4


@pytest.mark.parametrize("method", METHODS)
def test_bare_and_dressed_distances(method: core.DistanceMethod) -> None:
    """The recipes of get_distance_from_checks give the bare and dressed distances of a code."""
    # The only stabilizer of this subsystem code is Z on its last qubit, and X on one of its first
    # three qubits is a dressed logical operator, but every bare one weighs 3.
    code = qldpc.codes.CSSCode(
        [[0, 1, 1, 0], [1, 1, 0, 0]], [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]
    )
    stabilizers = code.get_stabilizer_ops()
    assert (
        core.get_distance_from_checks(stabilizers, code.matrix, symplectic=True, method=method) == 1
    )
    assert (
        core.get_distance_from_checks(code.matrix, stabilizers, symplectic=True, method=method) == 3
    )
    stabilizers_x, stabilizers_z = (
        code.get_stabilizer_ops(Pauli.X),
        code.get_stabilizer_ops(Pauli.Z),
    )
    assert core.get_distance_from_checks(code.matrix_z, stabilizers_x, method=method) == 3
    assert core.get_distance_from_checks(code.matrix_x, stabilizers_z, method=method) == 3

    # The dressed recipe excludes all of the gauge generators of the Bacon-Shor code, as Pauli
    # strings and as X-type rows.  Excluding only its stabilizers counts its X-type gauge operators
    # of weight 2 as logical operators.
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    distance = core.get_distance_from_checks(
        bacon_shor_code.get_stabilizer_ops(), bacon_shor_code.matrix, symplectic=True, method=method
    )
    assert distance == 3
    stabilizers_z = bacon_shor_code.get_stabilizer_ops(Pauli.Z)
    distance = core.get_distance_from_checks(stabilizers_z, bacon_shor_code.matrix_x, method=method)
    assert distance == 3
    distance = core.get_distance_from_checks(
        stabilizers_z, bacon_shor_code.get_stabilizer_ops(Pauli.X), method=method
    )
    assert distance == 2

    # Bare logical operators as generators give the dressed distance with the gauge generators
    # excluded, and the bare distance with only the stabilizers excluded.
    if method != "connected_cluster":
        logical_ops = code.get_logical_ops()
        for excluded, expected in ((code.matrix, 1), (stabilizers, 3)):
            assert (
                core.get_distance_from_generators(
                    logical_ops, excluded, symplectic=True, method=method
                )
                == expected
            )


def test_distance_functions_check_their_input() -> None:
    """The distance functions take sparse matrices, and reject invalid matrices and methods."""
    hamming_code = qldpc.codes.HammingCode(3)
    sparse = scipy.sparse.csr_array(np.asarray(hamming_code.matrix, dtype=int))
    assert core.get_distance_from_checks(sparse) == 3

    checks = [[1, 1, 1, 1]]
    for search in (
        functools.partial(core.get_distance_from_generators, checks),
        functools.partial(core.get_distance_from_checks, checks),
        functools.partial(core.get_distance_css, checks, checks),
        functools.partial(core.get_distance_pauli, checks),
    ):
        for unknown in ("other", ["brouwer_zimmermann"]):
            with pytest.raises(ValueError, match="Unknown distance method"):
                search(method=cast(Any, unknown))
    with pytest.raises(TypeError, match="checks must be an array of binary entries, not a"):
        core.get_distance_from_checks(cast(Any, hamming_code))

    with pytest.raises(ValueError, match="satisfy the checks"):
        core.get_distance_from_checks([[1, 0, 0, 0]], [[0, 0, 1, 0]], symplectic=True)
    with pytest.raises(ValueError, match="need excluded rows"):
        core.get_distance_from_checks(qldpc.codes.FiveQubitCode().matrix, symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_checks([[1, 0, 0]], [[0, 0, 0]], symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_pauli([[1, 0, 1]])
    with pytest.raises(ValueError, match=r"Expected Pauli\.X or Pauli\.Z"):
        core.get_distance_css(checks, checks, cast(Any, "Y"))
    with pytest.raises(ValueError, match="checks_z has 6 columns"):
        core.get_distance_css(checks, [[1, 1, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match="excluded has 0 columns"):
        core.get_distance_from_generators(checks, np.zeros((2, 0), dtype=int))
    with pytest.raises(ValueError, match="checks_z has 5 columns"):
        core.get_distance_css(np.zeros((0, 3)), np.zeros((0, 5)))


def test_automorphisms_are_checked_and_asked_for_once(tune: Tune) -> None:
    """Automorphisms are checked, and a function returning them is called at most once.

    They must be permutations, and they must preserve the code, except that brute force ignores
    them.
    """
    # With automorphism_minimum_states=0, a connected-cluster search asks for automorphisms before
    # its first weight limit, so both sectors of the bicycle code below would ask.  The code words
    # of the checks below are 000 and 111, so the cyclic shift [1, 2, 0] is an automorphism.
    tune(automorphism_minimum_states=0)
    checks = [[1, 1, 0], [0, 1, 1]]
    for method in ("brouwer_zimmermann", "connected_cluster"):
        assert core.get_distance_from_checks(checks, method=method, automorphisms=[[1, 2, 0]]) == 3
    get_automorphisms = mock.Mock(return_value=[[1, 2, 0]])
    distance = core.get_distance_from_checks(
        checks, method="connected_cluster", automorphisms=get_automorphisms
    )
    assert distance == 3
    get_automorphisms.assert_called_once()

    # The searches of both sectors of a CSS code share one call, also when "auto" searches by cost
    # and its connected-cluster search asks.  A transposition is not an automorphism.
    code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    transposition = np.arange(14)
    transposition[[0, 1]] = [1, 0]
    tune(auto_brute_force_max_words=0)
    for css_method in ("connected_cluster", "auto"):
        get_automorphisms = mock.Mock(return_value=code._get_permutation_automorphisms())
        distance = core.get_distance_css(
            code.matrix_x, code.matrix_z, method=css_method, automorphisms=get_automorphisms
        )
        assert distance == 3
        get_automorphisms.assert_called_once()
        with pytest.raises(ValueError, match="not an automorphism"):
            core.get_distance_css(
                code.matrix_x, code.matrix_z, method=css_method, automorphisms=transposition[None]
            )

    # A search of Pauli strings by cost asks once too, from the connected-cluster search of its free
    # phase.  Hadamards on the second block of the code keep the automorphisms above.
    pauli_checks = code.conjugated(range(7, 14)).matrix
    get_automorphisms = mock.Mock(return_value=code._get_permutation_automorphisms())
    distance = core.get_distance_pauli(pauli_checks, method="auto", automorphisms=get_automorphisms)
    assert distance == 3
    get_automorphisms.assert_called_once()
    with pytest.raises(ValueError, match="not an automorphism"):
        core.get_distance_pauli(pauli_checks, method="auto", automorphisms=transposition[None])

    # Brute force neither asks for automorphisms nor verifies them.
    get_automorphisms = mock.Mock(return_value=transposition[None])
    for search in (
        functools.partial(core.get_distance_css, code.matrix_x, code.matrix_z),
        functools.partial(core.get_distance_from_checks, code.matrix_z, code.matrix_x),
    ):
        assert search(method="brute_force", automorphisms=get_automorphisms) == 3
    get_automorphisms.assert_not_called()

    for automorphisms in ([[1, 0]], [[0, 0, 1]], [[[1, 2, 0]]]):
        with pytest.raises(ValueError, match="permutations of range"):
            core.get_distance_from_checks(checks, automorphisms=automorphisms)
    with pytest.raises(ValueError, match="permutations of range"):
        core.get_distance_from_checks(
            checks, method="connected_cluster", automorphisms=lambda: [[0, 0, 1]]
        )


@pytest.mark.parametrize(
    ("method", "searcher"),
    [("connected_cluster", "get_distance_connected_cluster"), ("auto", "search_by_cost")],
)
def test_connected_cluster_searches_sparse_stabilizers(
    monkeypatch: pytest.MonkeyPatch, tune: Tune, method: core.DistanceMethod, searcher: str
) -> None:
    """A connected-cluster search of a subsystem code searches sparse checks of its stabilizers.

    So does a search by cost, which "auto" runs when enumeration is not cheap.
    """
    tune(auto_brute_force_max_words=0)
    # With X and Z on a bulk qubit of the [[16, 1, 4]] surface code as gauge generators, the two
    # checks of each type on that qubit merge into one stabilizer of weight 6, the dressed distance
    # drops to 3, and the row-reduced stabilizers have rows of weight 8.  The other checks are
    # stabilizers, but they do not span the stabilizer group.
    code = qldpc.codes.SurfaceCode(4)
    qubit = 5
    gauge_qubit = np.zeros((1, len(code)), dtype=np.uint8)
    gauge_qubit[0, qubit] = 1
    checks_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
    checks_z = code.matrix_z.view(np.ndarray).astype(np.uint8)
    gauge_x, gauge_z = np.vstack([checks_x, gauge_qubit]), np.vstack([checks_z, gauge_qubit])
    derived_x, derived_z = get_css_stabilizers(gauge_x, gauge_z)
    assert derived_x.sum(axis=1).max() == derived_z.sum(axis=1).max() == 8

    # The stabilizers are the checks off the gauge qubit, and the product of those on it.
    def get_stabilizers(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        on_qubit = checks[:, qubit] == 1
        return np.vstack([checks[~on_qubit], checks[on_qubit].sum(axis=0) % 2])

    searched: list[npt.NDArray[np.uint8]] = []

    def record(sectors: Any, *args: Any, **kwargs: Any) -> Any:
        searched.extend(np.asarray(checks) for checks, _ in sectors)
        return search(sectors, *args, **kwargs)

    search = getattr(core, searcher)
    monkeypatch.setattr(core, searcher, record)
    distance = core.get_distance_css(gauge_x, gauge_z, method=method)
    assert distance == core.get_distance_css(gauge_x, gauge_z, method="brute_force") == 3
    for checks, stabilizers in zip(
        searched, (get_stabilizers(checks_z), get_stabilizers(checks_x)), strict=True
    ):
        assert checks.sum(axis=1).max() <= 6
        assert set(map(tuple, stabilizers[:-1])) <= set(map(tuple, checks))
        assert np.array_equal(get_independent_rows(checks), get_independent_rows(stabilizers))
    assert core.get_distance_css(gauge_x, gauge_z, Pauli.Z, method=method) == 3

    searched.clear()
    assert core.get_distance_css(checks_x, checks_z, method=method) == 4
    assert [checks.tolist() for checks in searched] == [checks_z.tolist(), checks_x.tolist()]

    # Hadamards on half of the qubits make a Pauli form that is not CSS.
    searched.clear()
    qubits = np.arange(0, len(code), 2)
    columns, swapped = np.r_[qubits, qubits + len(code)], np.r_[qubits + len(code), qubits]

    def apply_hadamards(rows: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rows = rows.copy()
        rows[:, columns] = rows[:, swapped]
        return rows

    gauge = apply_hadamards(scipy.linalg.block_diag(gauge_x, gauge_z))
    assert core.get_distance_pauli(gauge, method=method) == 3
    (checks,) = searched
    assert (checks[:, : len(code)] | checks[:, len(code) :]).sum(axis=1).max() <= 6
    stabilizers = scipy.linalg.block_diag(get_stabilizers(checks_x), get_stabilizers(checks_z))
    assert np.array_equal(
        get_independent_rows(checks), get_independent_rows(apply_hadamards(stabilizers))
    )
    kept = scipy.linalg.block_diag(get_stabilizers(checks_x)[:-1], get_stabilizers(checks_z)[:-1])
    assert set(map(tuple, apply_hadamards(kept))) <= set(map(tuple, checks))


def test_connected_cluster_searches_take_the_active_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSS and Pauli distances search by connected clusters with the active tuning.

    ``get_distance_css`` searches through ``_search_css``, and ``get_distance_pauli``, whose
    searches are symplectic, through ``_search_checks``.  Each search takes both of its switches
    from the active tuning, whether they are on or off.
    """
    switches: list[tuple[bool, bool, bool]] = []
    initialize = SupportSearch.__init__

    def record(search: SupportSearch, *args: Any, **kwargs: Any) -> None:
        symplectic = kwargs.get("symplectic", False)
        switches.append((symplectic, kwargs["restrict_viable_symbols"], kwargs["pair_sum_checks"]))
        initialize(search, *args, **kwargs)

    monkeypatch.setattr(SupportSearch, "__init__", record)
    steane_code = qldpc.codes.SteaneCode()
    for switch in (True, False):
        switches.clear()
        with use_tuning(restrict_viable_symbols=switch, pair_sum_checks=switch):
            core.get_distance_css(
                steane_code.matrix_x, steane_code.matrix_z, method="connected_cluster"
            )
            core.get_distance_pauli(qldpc.codes.FiveQubitCode().matrix, method="connected_cluster")
        assert {symplectic for symplectic, _, _ in switches} == {False, True}
        assert all(restrict == pair_sums == switch for _, restrict, pair_sums in switches)


def test_connected_cluster_needs_parity_checks() -> None:
    """get_distance_from_generators rejects the connected-cluster method, which needs checks."""
    with pytest.raises(ValueError, match="use get_distance_from_checks"):
        core.get_distance_from_generators([[1, 1, 0]], method="connected_cluster")


def test_special_inputs_of_generators() -> None:
    """Matrix shapes are checked, and inputs without nontrivial words have no distance."""
    no_distance = qldpc.codes.NO_DISTANCE
    assert (
        core.get_distance_from_generators(np.empty((0, 4), dtype=np.uint8), cutoff=0) is no_distance
    )
    for method in ("brouwer_zimmermann", "brute_force"):
        distance = core.get_distance_from_generators([[1, 0]], [[1, 0]], method=method)
        assert distance is no_distance
    assert core.get_distance_from_generators([], [[1, 0]]) is no_distance
    no_logical_ops = np.empty((0, 4), dtype=np.uint8)
    assert (
        core.get_distance_from_generators(no_logical_ops, [[1, 0, 0, 0]], symplectic=True)
        is no_distance
    )
    assert (
        core.get_distance_from_generators([[1, 0, 0, 0]], [[1, 0, 0, 0]], symplectic=True)
        is no_distance
    )

    with pytest.raises(ValueError, match="excluded has 2 columns"):
        core.get_distance_from_generators([[1, 0, 0]], [[1, 0]])
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_generators([[1, 0, 0]], [], symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_generators(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            symplectic=True,
        )
    with pytest.raises(ValueError, match="generators must be a matrix or a single row"):
        core.get_distance_from_generators(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width of generators"):
        core.get_distance_from_generators([])
    with pytest.raises(ValueError, match="excluded must be a matrix or a single row"):
        core.get_distance_from_generators(
            [[1, 0]], np.zeros((1, 1, 2), dtype=np.uint8), symplectic=True
        )


def test_dependent_logical_rows_give_the_distance_of_their_span() -> None:
    """Repeated logical rows are reduced to a basis of the quotient before any search."""
    distance = core.get_distance_from_generators(
        [[1, 1, 1], [1, 1, 1]], [], cutoff=0, method="brouwer_zimmermann"
    )
    assert distance == 3


def test_brouwer_zimmermann_searches_never_enumerate_every_word() -> None:
    """A BZ search scans information sets, even where enumerating every word would cost less.

    The generators span only ``2**5`` words, few enough that ``method="auto"`` enumerates them.
    """
    logicals = [[1, 0, 0, 0, 1, 0, 1, 1, 0]]
    stabilizers = [
        [0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1],
    ]
    expected = core.get_distance_from_generators(logicals, stabilizers, method="brute_force")
    with mock.patch.object(
        _brute_force, "_get_distance_brute_force", wraps=_brute_force._get_distance_brute_force
    ) as enumerate_words:
        distance = core.get_distance_from_generators(
            logicals, stabilizers, method="brouwer_zimmermann"
        )
    assert distance == expected
    enumerate_words.assert_not_called()


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_distances_of_known_classical_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    """Classical distances match known values for Hamming and ring codes."""
    method: core.DistanceMethod = "brouwer_zimmermann"
    assert core.get_distance_from_generators(code.generator, method=method) == expected_distance
    assert core.get_distance_from_checks(code.matrix, method=method) == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.C4Code(), 2),
        (qldpc.codes.quantum.C6Code(), 2),
        (qldpc.codes.quantum.SteaneCode(), 3),
        (qldpc.codes.quantum.SurfaceCode(3), 3),
        (qldpc.codes.quantum.SurfaceCode(4), 4),
        (qldpc.codes.quantum.ToricCode(4), 4),
    ],
)
def test_distances_of_known_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    """Quantum distances match known values for C4, C6, Steane, and small surface/toric codes."""
    method: core.DistanceMethod = "brouwer_zimmermann"
    distance_x = core.get_distance_from_generators(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
        method=method,
    )
    distance_z = core.get_distance_from_generators(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
        method=method,
    )
    distance_all = core.get_distance_from_generators(
        code.get_logical_ops(), code.get_stabilizer_ops(), symplectic=True, method=method
    )
    distance_css = core.get_distance_css(code.matrix_x, code.matrix_z, method=method)
    assert distance_x == distance_z == distance_all == distance_css == expected_distance


def test_quantum_distances_use_qubit_automorphisms(tune: Tune) -> None:
    """Searches use qubit permutations, lifted for Pauli strings, and reject non-automorphisms."""
    # Hadamards on the second block of this bicycle code make a code that is not CSS, which the
    # translations of both blocks still preserve.
    tune(automorphism_minimum_layer_work=0)
    bicycle_code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    translations = bicycle_code._get_permutation_automorphisms()
    transposition = np.arange(len(bicycle_code))
    transposition[[0, 1]] = [1, 0]
    code = bicycle_code.conjugated(range(7, 14))
    problems = [
        (code.get_logical_ops(), code.get_stabilizer_ops(), True),
        (bicycle_code.get_logical_ops("X"), bicycle_code.get_stabilizer_ops("X"), False),
    ]
    for logical_ops, stabilizers, symplectic in problems:
        search = functools.partial(
            core.get_distance_from_generators, logical_ops, stabilizers, symplectic=symplectic
        )
        expected = search(method="brute_force")
        for automorphisms in (translations, None):
            get_automorphisms = mock.Mock(return_value=automorphisms)
            distance = search(automorphisms=get_automorphisms, method="brouwer_zimmermann")
            assert distance == expected == 3
            get_automorphisms.assert_called_once()
        with pytest.raises(ValueError, match="not an automorphism"):
            search(automorphisms=transposition[None], method="brouwer_zimmermann")


def test_brouwer_zimmermann_random_cross_checks() -> None:
    """Information-set search agrees with enumerated spans of random nested codes.

    This integration check covers what ``core.py`` adds to a BZ search: the nested bases of words
    with excluded rows, and the map of Pauli strings to binary words.  The expected minimum weights
    come from enumerating the spans, independently of ``core.py``.
    """
    BZ: core.DistanceMethod = "brouwer_zimmermann"
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        basis = get_random_full_rank_matrix(rng, min(5, length - 1), length)
        expected = min(word.bit_count() for word in get_span(basis) if word)
        assert core.get_distance_from_generators(basis, cutoff=0, method=BZ) == expected

        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = min(word.bit_count() for word in get_span(basis) - get_span(stabilizers))
        distance = core.get_distance_from_generators(logical_ops, stabilizers, cutoff=0, method=BZ)
        assert distance == expected

    for num_qubits in range(3, 7):
        basis = get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = min(
            get_pauli_weight(word, num_qubits, css_weight=False)
            for word in get_span(basis) - get_span(stabilizers)
        )
        distance = core.get_distance_from_generators(
            logical_ops, stabilizers, symplectic=True, cutoff=0, method=BZ
        )
        assert distance == expected


def test_max_weight() -> None:
    """A weight limit returns the smaller of the distance and max_weight + 1, on every route."""
    rng = np.random.default_rng(7)
    basis = get_random_full_rank_matrix(rng, 4, 7)
    classical = functools.partial(core.get_distance_from_generators, basis)
    classical_distance = min(word.bit_count() for word in get_span(basis) if word)
    css = functools.partial(core.get_distance_from_generators, basis[2:], basis[:2])
    css_distance = min(word.bit_count() for word in get_span(basis) - get_span(basis[:2]))
    basis = get_random_full_rank_matrix(rng, 5, 8)
    pauli = functools.partial(
        core.get_distance_from_generators, basis[2:], basis[:2], symplectic=True
    )
    pauli_distance = min(
        ((word | word >> 4) & 0b1111).bit_count() for word in get_span(basis) - get_span(basis[:2])
    )

    def check(distance_with_limit: Callable[..., float], distance: int, method: str) -> None:
        assert distance >= 2
        for max_weight in [distance - 2, distance]:
            result = distance_with_limit(cutoff=0, max_weight=max_weight, method=method)
            assert result == min(distance, max_weight + 1)

    for distance_with_limit, distance in [(classical, classical_distance), (css, css_distance)]:
        for method in ["brouwer_zimmermann", "brute_force"]:
            check(distance_with_limit, distance, method)

    # The Pauli case checks that the BZ route, which searches binary words of twice the weight of
    # the Pauli strings, doubles the bound.
    check(pauli, pauli_distance, "brouwer_zimmermann")


def test_max_weight_caps_cutoff() -> None:
    """A large cutoff cannot make the search give up on a word within the limit."""
    # Both rows weigh more than the limit, but their sum weighs 3, which the BZ search finds after
    # the rows.  The test pins BZ, since "auto" would enumerate so small a code.
    basis = [[1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]]
    distance = core.get_distance_from_generators(
        basis, cutoff=6, max_weight=3, method="brouwer_zimmermann"
    )
    assert distance == 3


def test_max_weight_conventions_and_validation() -> None:
    """The limit leaves NO_DISTANCE unchanged, and must be a nonnegative integer."""
    no_distance = qldpc.codes.NO_DISTANCE
    assert core.get_distance_from_generators([[0, 0, 0]], max_weight=1) is no_distance
    with pytest.raises(ValueError, match="max_weight must be nonnegative"):
        core.get_distance_from_generators([[1, 1]], max_weight=-1)

    for not_an_integer in (2.5, True):
        with pytest.raises(TypeError):
            core.get_distance_from_generators([[1, 1]], max_weight=cast(int, not_an_integer))
    # A NumPy limit gives a Python integer, even on a route that clamps its result.
    distance = core.get_distance_from_generators(
        [[1, 1, 1, 1]], max_weight=np.int64(1), method="brute_force"
    )
    assert distance == 2 and type(distance) is int


def test_cutoff_must_be_an_integer() -> None:
    """The distance functions reject a cutoff that is not an integer."""
    with pytest.raises(TypeError, match="cutoff must be an integer"):
        core.get_distance_from_generators([[1, 1]], cutoff=cast(int, 2.5))


def test_distance_method_validation() -> None:
    """Each method name resolves to its canonical name, and the types list every accepted name."""
    for method in METHODS:
        assert core._validate_distance_method(method) == method
    assert core._validate_distance_method("information_sets") == "brouwer_zimmermann"

    assert set(get_args(core.DistanceMethod)) == set(core._DISTANCE_METHODS)
    assert set(get_args(core._CanonicalDistanceMethod)) == set(core._DISTANCE_METHODS.values())


def test_no_distance_behaves_as_nan_and_survives_copies() -> None:
    """NO_DISTANCE is a float nan, and copies and pickles of it are the same member."""
    no_distance = core.NO_DISTANCE
    assert qldpc.codes.NoDistance is qldpc.codes.distance.NoDistance is core.NoDistance
    assert qldpc.codes.NO_DISTANCE is qldpc.codes.distance.NO_DISTANCE is core.NoDistance.NAN
    assert isinstance(no_distance, float) and math.isnan(no_distance)
    assert str(no_distance) == f"{no_distance}" == repr(no_distance) == "nan"
    assert repr((2, 0, no_distance)) == "(2, 0, nan)"
    assert not no_distance >= 3 and no_distance != core.NO_DISTANCE
    restored = pickle.loads(pickle.dumps(no_distance))  # noqa: S301 - trusted in-memory round trip
    assert restored is no_distance
    assert copy.copy(no_distance) is copy.deepcopy(no_distance) is no_distance
    assert core.NoDistance(math.nan) is no_distance
    with pytest.raises(ValueError, match="not a valid NoDistance"):
        core.NoDistance(1.0)
