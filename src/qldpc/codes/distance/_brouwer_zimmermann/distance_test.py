# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary Brouwer-Zimmermann distance searches."""

from __future__ import annotations

import collections
import dataclasses
import functools
import gc
import itertools
import math
import threading
import weakref
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc
from qldpc._binary import get_independent_rows, row_reduce_binary
from qldpc.objects import PAULIS_XZ, Pauli

from ...conftest import Tune
from .. import distances
from .._information_sets import basis_families, certificates, completion_bounds, matroid_partition
from .._tuning import DEFAULT_TUNING, MAX_VALUE, use_tuning
from ..conftest import (
    MISSED_SET_CODE,
    THREAD_TIMEOUT,
    CostlyStages,
    check_shared_arrays_on_threads,
    get_costly_basis,
    get_golay_checks,
    get_random_css_code,
    get_random_full_rank_matrix,
    get_sector,
)
from . import affine_instance, affine_search
from . import distance as bz

# A tuning that never uses completion tables, for searches built directly.
TABLES_DISABLED = dataclasses.replace(
    DEFAULT_TUNING,
    completion_bound_minimum_layer_work=MAX_VALUE,
    completion_bound_max_cost_percent=0,
    completion_bound_max_table_bytes=0,
)

# ``TABLES_DISABLED``, with scans that return after each support.
ONE_SUPPORT_CHUNKS = dataclasses.replace(TABLES_DISABLED, search_chunk_exponent=0)

# Changes to the tuning that make searches use completion tables in every layer of two or more rows,
# whatever their predicted cost.  Eight-column blocks, and one sample for the cost model, keep the
# tables and predictions quick to build as plain Python.
FORCED_TABLES: dict[str, Any] = {
    "completion_bound_minimum_layer_work": 0,
    "completion_bound_max_cost_percent": MAX_VALUE,
    "completion_bound_block_bits": 8,
    "completion_bound_samples": 1,
    "completion_bound_min_samples": 1,
}


def test_brouwer_zimmermann_search_state() -> None:
    """Only completed layers advance the certificate, not interrupted ones or ones cut off.

    A later call restarts an interrupted layer from its first support.
    """
    # One set of rank zero, whose two eligible rows weigh one and two.
    search = bz._SectorSearch(
        affine_instance.AffineInstance(
            generators=np.array([[[2], [3]]], dtype=np.uint64),
            labels=np.ones((1, 2, 1), dtype=np.uint64),
            widths=np.array([1], dtype=np.int64),
            pivots=np.full((1, 2), -1, dtype=np.int64),
            certificate=certificates.Certificate(2, (0,)),
        ),
        cutoff=1,
        tuning=ONE_SUPPORT_CHUNKS,
        best=3,
        lower_bound=0,
    )

    # Interrupt a layer after its first chunk, which weighs the first row, before the search takes
    # the result of the chunk.
    scan_set = affine_search.scan_set

    def interrupted_scan(*args: Any) -> tuple[int, int, int, int, int]:
        scan_set(*args)
        raise KeyboardInterrupt

    with (
        mock.patch.object(affine_search, "scan_set", interrupted_scan),
        pytest.raises(KeyboardInterrupt),
    ):
        search.advance()
    assert (search.best, search.lower_bound, search.next_weight) == (3, 0, 1)

    # Reach the cutoff at the first row, which the search weighs again because it restarts the
    # interrupted layer.
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 1)
    assert search.advance() == 1

    # Complete the layers.
    search.cutoff = -(2**100)
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 2)
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (np.iinfo(np.int64).max, 3)


# Each case reaches one early exit from search setup or a shared search with real inputs.  The test
# forces whether setup builds overlapping families, and keeps it from partitioning, but a change to
# how setup builds its families can still move a case to another exit.  If the test or the coverage
# gate then fails, find a new natural input here rather than stubbing internals.
SETUP_EXIT_CASES = [
    # A shared search reaches the cutoff inside a coefficient-weight scan.
    pytest.param(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        ],
        [],
        2,
        False,
        id="scan-cutoff",
    ),
    # Rows of an overlapping replacement family reach the cutoff.
    pytest.param(
        [[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]],
        [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        ],
        4,
        True,
        id="overlap-cutoff",
    ),
    # The certificate of an overlapping replacement family closes the gap to the best witness.
    pytest.param(
        [[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]],
        [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        ],
        0,
        True,
        id="overlap-certificate",
    ),
]


@pytest.mark.parametrize("logicals, stabilizers, cutoff, overlap", SETUP_EXIT_CASES)
def test_brouwer_zimmermann_setup_exits(
    logicals: list[list[int]],
    stabilizers: list[list[int]],
    cutoff: int,
    overlap: bool,
    tune: Tune,
    monkeypatch: pytest.MonkeyPatch,
    prepared: list[bz._SectorSearch],
) -> None:
    """Each case settles in setup exactly when an overlapping family replaces its disjoint one.

    Each case also agrees with a brute-force oracle.
    """
    expected = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=0, method="brute_force"
    )

    # Record the overlapping families that replace disjoint ones.
    replacements: list[basis_families.BasisFamily] = []
    get_overlapping_family = basis_families.get_overlapping_family

    def recording_get_overlapping_family(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        family = get_overlapping_family(*args, **kwargs)
        if len(family.ranks):
            replacements.append(family)
        return family

    monkeypatch.setattr(basis_families, "get_overlapping_family", recording_get_overlapping_family)
    tune(
        overlapping_minimum_work=0 if overlap else MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )
    distance = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=cutoff, method="brouwer_zimmermann"
    )
    assert expected <= distance <= max(cutoff, expected)

    # Setup settles a case exactly if an overlapping family replaces the disjoint one.  The cases
    # with a cutoff end at or below it, and the case without one settles in setup, which without a
    # cutoff only its certificate can do.
    assert bool(prepared) == (not replacements) == (not overlap)
    assert (distance <= cutoff) == (cutoff > 0)


def test_brouwer_zimmermann_setup_cutoffs() -> None:
    """Setup can reach the cutoff from initial rows or a newly systematic row."""
    # Each row weighs three, which bounds the distance, and their sum weighs two.
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    assert (
        bz._prepare_sector_search(
            basis,
            _no_labels(basis),
            cutoff=3,
            weight_divisor=1,
            upper_bound=3,
            tuning=DEFAULT_TUNING,
        )
        == 3
    )
    assert (
        bz._prepare_sector_search(
            basis,
            _no_labels(basis),
            cutoff=2,
            weight_divisor=1,
            upper_bound=3,
            tuning=DEFAULT_TUNING,
        )
        == 2
    )


def test_an_upper_bound_acts_as_a_known_witness() -> None:
    """A search returns the smaller of its upper bound and the distance."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(3).generator, dtype=np.uint8))
    for upper_bound, expected in ((2, 2), (3, 3), (5, 3)):
        distance = bz.get_distance_brouwer_zimmermann(
            [(basis, _no_labels(basis))], cutoff=0, upper_bound=upper_bound
        )
        assert distance == expected


def test_brouwer_zimmermann_shared_search() -> None:
    """A shared Brouwer-Zimmermann search returns the minimum distance across nested codes."""
    rng = np.random.default_rng(119)
    basis_a = get_random_full_rank_matrix(rng, 5, 8)
    basis_b = get_random_full_rank_matrix(rng, 5, 9)
    labels_b = np.vstack(
        [
            np.zeros((2, 3), dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
        ]
    )

    expected_a = qldpc.codes.distance.get_distance_from_generators(
        basis_a,
        cutoff=0,
        method="brute_force",
    )
    expected_b = qldpc.codes.distance.get_distance_from_generators(
        basis_b[2:],
        basis_b[:2],
        cutoff=0,
        method="brute_force",
    )
    actual = bz.get_distance_brouwer_zimmermann(
        [(basis_a, _no_labels(basis_a)), (basis_b, labels_b)],
        cutoff=0,
    )
    assert actual == min(expected_a, expected_b)


def test_labels_of_more_than_one_machine_word() -> None:
    """A search reads every machine word of labels wider than 64 columns."""
    # Only row 0 has a label in the first word, and the lightest words do not use row 0, so a
    # search that read only that word would return a larger distance.
    rng = np.random.default_rng(28)
    basis = (rng.random((11, 30)) < 0.35).astype(np.uint8)
    labels = np.zeros((11, 76), dtype=np.uint8)
    labels[0, 0] = 1
    labels[np.arange(1, 11), 64 + np.arange(1, 11)] = 1
    expected = qldpc.codes.distance.get_distance_from_generators(
        basis, cutoff=0, method="brute_force"
    )
    assert bz.get_distance_brouwer_zimmermann([(basis, labels)], cutoff=0) == expected


def test_a_given_disjoint_basis_family_replaces_the_first_stage() -> None:
    """A given function replaces a search's first stage, and the search calls it only if needed."""
    rng = np.random.default_rng(121)
    basis = get_random_full_rank_matrix(rng, 5, 9)
    labels = np.vstack([np.zeros((2, 3), dtype=np.uint8), np.eye(3, dtype=np.uint8)])
    expected = qldpc.codes.distance.get_distance_from_generators(
        basis[2:], basis[:2], cutoff=0, method="brute_force"
    )
    get_basis_family = mock.Mock(
        return_value=basis_families.get_disjoint_family(
            basis, labels, random_column_orders=DEFAULT_TUNING.disjoint_random_column_orders
        )
    )
    search = functools.partial(
        bz.get_distance_brouwer_zimmermann,
        [(basis, labels)],
        get_disjoint_basis_families=[get_basis_family],
    )
    with mock.patch.object(
        basis_families, "get_disjoint_family", wraps=basis_families.get_disjoint_family
    ) as first_stage:
        assert search(cutoff=0) == expected
        get_basis_family.assert_called_once_with()
        search(cutoff=basis.shape[1])
        get_basis_family.assert_called_once_with()
    first_stage.assert_not_called()


def test_brouwer_zimmermann_sector_certificates() -> None:
    """A completed sector's certificate must not finish a different sector's search."""
    generators = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis_a = get_independent_rows(
        np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    )
    basis_b = get_independent_rows(
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    expected = [
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        for basis in (basis_a, basis_b)
    ]
    assert expected == [4, 2]

    searches: list[bz._SectorSearch] = []
    advanced: set[int] = set()
    original_prepare = bz._prepare_sector_search
    original_advance = bz._SectorSearch.advance

    def prepare(*args: Any, **kwargs: Any) -> int | bz._SectorSearch:
        result = original_prepare(*args, **kwargs)
        if isinstance(result, bz._SectorSearch):
            searches.append(result)
        return result

    def advance(search: bz._SectorSearch) -> int:
        advanced.add(id(search))
        return original_advance(search)

    # Exercise real layer advancement, rather than two results certified during setup.
    with (
        mock.patch.object(bz, "_prepare_sector_search", side_effect=prepare),
        mock.patch.object(bz._SectorSearch, "advance", advance),
    ):
        actual = bz.get_distance_brouwer_zimmermann(
            [(basis_a, _no_labels(basis_a)), (basis_b, _no_labels(basis_b))], cutoff=0
        )
    assert actual == min(expected)
    assert len(searches) == 2
    assert {id(search) for search in searches} <= advanced


def test_brouwer_zimmermann_subsystem_code() -> None:
    """A search treats the gauge operators of the [[9, 1, 3]] Bacon-Shor code as trivial.

    See https://arxiv.org/abs/quant-ph/0506023.
    """
    code = qldpc.codes.QuditCode(qldpc.codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    assert code.get_distance_exact(method="brouwer_zimmermann") == 3


@pytest.mark.parametrize("length", [64, 65, 257])
def test_brouwer_zimmermann_packed_widths(length: int) -> None:
    """The weights of rows are exact across uint64 words and above 255."""
    generators = np.ones((1, length), dtype=np.uint8)
    assert (
        qldpc.codes.distance.get_distance_from_generators(
            generators, cutoff=0, method="brouwer_zimmermann"
        )
        == length
    )


def test_the_lightest_eligible_row_excludes_stabilizers() -> None:
    """Upper bounds from rows come from eligible rows, not lighter stabilizers."""
    basis = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    assert bz._get_lightest_eligible_row_weight(basis, labels) == 2


def test_a_search_of_an_mb_code_takes_overlapping_sets(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """The X-sector search of a 64-qubit MB code takes overlapping sets, and finds the distance.

    Section 3.8 of ``brouwer_zimmermann.md`` works this example: overlapping sets do far less work
    than the one set of a disjoint family.
    """
    # Remove the work gate of overlapping families, so that setup chooses between the families by
    # their predicted work alone.  Keep setup from partitioning, since a partitioned family can also
    # cover a column more than once.
    tune(overlapping_minimum_work=0, partition_work_ratio=MAX_VALUE)
    code = qldpc.codes.MBCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    distance = code.forget_distance().get_distance_exact("X", cutoff=0, method="brouwer_zimmermann")
    assert distance == 8
    assert any(family.max_coverage > 1 for family in costly_stages.search_families)


def test_brouwer_zimmermann_overlap_improves_quotient_witness(tune: Tune) -> None:
    """A search whose overlapping family has no row as light as the distance still finds it."""

    # Shorten a systematic basis of the cyclic [31, 21] BCH code to [26, 16], and isolate a one-bit
    # quotient.  Building the basis from the binary polynomial 1 + x + x^2 + x^4 + x^5 + x^7 + x^10
    # that generates the code needs no field GF(32), which ``qldpc.codes.BCHCode`` builds.
    basis = get_independent_rows(_get_cyclic_basis("11101101001", 31))[5:, 5:]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=0, method="brute_force"
    )

    # Lower only the work threshold, so that this small search builds an overlapping family, and
    # check that the family has no witness as light as the distance.
    tune(overlapping_minimum_work=0)
    original = basis_families.get_overlapping_family

    def select(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        selected = original(*args, **kwargs)
        assert len(selected.ranks) and selected.max_coverage > 1
        assert (
            min(
                bz._get_lightest_eligible_row_weight(generators, labels)
                for generators, labels in zip(selected.generators, selected.labels, strict=True)
            )
            > expected
        )
        return selected

    with mock.patch.object(
        basis_families, "get_overlapping_family", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.distance.get_distance_from_generators(
                logicals, stabilizers, cutoff=0, method="brouwer_zimmermann"
            )
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.mark.usefixtures("completion_tables")
def test_a_search_scans_every_set_that_its_certificate_counts() -> None:
    """A search scans every set that its certificate counts in the layer that holds the distance."""
    # Both disjoint sets certify weight six once they complete layer two, and the first word of
    # weight five is in layer two of the second set.
    rows = [
        "1000100010001101001110",
        "0100100010000111110000",
        "0010000010000100101010",
        "0001100010000000011111",
        "0000010110000001011001",
        "0000001100000010001011",
        "0000000000101110101110",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    assert (
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        == 5
    )
    distance = qldpc.codes.distance.get_distance_from_generators(
        basis, cutoff=0, method="brouwer_zimmermann"
    )
    assert distance == 5


def test_a_search_through_the_costly_stages_finds_the_distance(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """A search with a partitioned family finds the distance."""
    tune(overlapping_minimum_work=0, partition_work_ratio=0, max_information_sets=16)
    basis = get_costly_basis()
    assert (
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        == 6
    )
    distance = qldpc.codes.distance.get_distance_from_generators(
        basis, cutoff=0, method="brouwer_zimmermann"
    )
    assert distance == 6
    assert any(
        len(family.ranks) and family.max_coverage > 1
        for family in costly_stages.partitioned_families
    )


def test_a_coverage_limit_of_one_partitions_into_disjoint_sets(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """A search with a coverage limit of one, and no overlapping families, takes disjoint sets.

    The search packs the first six columns of ``MISSED_SET_CODE`` and a partial set, and its
    partitioned family holds two disjoint information sets.
    """
    tune(
        overlapping_minimum_work=MAX_VALUE,
        disjoint_random_column_orders=0,
        partition_max_coverage=1,
        partition_work_ratio=0,
    )
    distance = qldpc.codes.distance.get_distance_from_generators(
        MISSED_SET_CODE, cutoff=0, method="brouwer_zimmermann"
    )
    assert (
        distance
        == qldpc.codes.distance.get_distance_from_generators(MISSED_SET_CODE, method="brute_force")
        == 4
    )
    (family,) = costly_stages.partitioned_families
    assert family.ranks.tolist() == [6, 6] and family.max_coverage == 1


@pytest.fixture
def switch_before_any_layer(tune: Tune) -> None:
    """Let searches consider automorphism families before their first layer, however cheap it is."""
    tune(automorphism_minimum_layer_work=0)


@dataclasses.dataclass
class CompletionRecord:
    """Whether searches had to use completion tables, the tables that they chose, and their work.

    Each entry of ``choices`` holds the tables that a search held before a layer and the tables that
    it chose for the layer, or None.
    """

    forced: bool
    choices: list[tuple[completion_bounds.CompletionTables | None, ...]]
    lookups: int = 0
    skipped: int = 0


@pytest.fixture(params=["default", "forced"])
def completion_tables(
    request: pytest.FixtureRequest, tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CompletionRecord]:
    """Run a test with the default choice of completion tables, or with tables in every layer.

    Record the tables that searches choose and the bounds that their scans look up, and check that a
    forced search used tables.
    """
    record = CompletionRecord(request.param == "forced", [])
    if record.forced:
        tune(**FORCED_TABLES)
    select = completion_bounds.select_completion_tables
    scan_set = affine_search.scan_set

    def recording_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        tables = select(*args, **kwargs)
        record.choices.append((kwargs["current"], tables))
        return tables

    def counting_scan(*args: Any) -> tuple[int, int, int, int, int]:
        result = scan_set(*args)
        record.lookups += result[3]
        record.skipped += result[4]
        return result

    monkeypatch.setattr(completion_bounds, "select_completion_tables", recording_select)
    monkeypatch.setattr(affine_search, "scan_set", counting_scan)
    yield record
    if record.forced:
        assert record.lookups


@pytest.mark.usefixtures("switch_before_any_layer")
def test_automorphism_family_searches_scan_every_layer_they_certify(
    completion_tables: CompletionRecord,
) -> None:
    """A search of an automorphism family finds a distance that only its last layer reaches."""
    # The lightest rows of the scanned set weigh ten, and the first word of weight eight that the
    # search finds sums four rows of that set.  Completing layer four certifies weight ten, so a
    # search that certified that layer before scanning it would return ten.
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    sector = get_sector(code, Pauli.X)
    expected = bz.get_distance_brouwer_zimmermann([sector], cutoff=0)
    assert expected == 8
    distance = bz.get_distance_brouwer_zimmermann(
        [sector], cutoff=0, get_automorphisms=code._get_permutation_automorphisms
    )
    assert distance == expected

    # With tables in every layer, layers after the first that uses them reuse the same tables.
    if completion_tables.forced:
        assert any(
            held is not None and tables is held for held, tables in completion_tables.choices
        )


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch) -> list[bz._SectorSearch]:
    """The searches that setup prepares, rather than settles, in the order of their sectors."""
    searches: list[bz._SectorSearch] = []
    prepare = bz._prepare_sector_search

    def recording_prepare(*args: Any, **kwargs: Any) -> Any:
        result = prepare(*args, **kwargs)
        if isinstance(result, bz._SectorSearch):
            searches.append(result)
        return result

    monkeypatch.setattr(bz, "_prepare_sector_search", recording_prepare)
    return searches


def test_stepped_searches_advance_one_unfinished_search_at_a_time_in_turn(
    monkeypatch: pytest.MonkeyPatch, tune: Tune, prepared: list[bz._SectorSearch]
) -> None:
    """Each advance scans one layer of the next unfinished search, in turn, until all finish.

    Once built, the shared bound is at most the weight of the lightest eligible row.
    """
    # Record the index of each search that advances, and which searches had finished by then.
    visits: list[tuple[int, list[bool]]] = []
    advance = bz._SectorSearch.advance

    def recording_advance(search: bz._SectorSearch) -> int:
        visits.append((prepared.index(search), [other.finished for other in prepared]))
        return advance(search)

    monkeypatch.setattr(bz._SectorSearch, "advance", recording_advance)
    # This code has distance 5, as the connected-cluster method finds.  Setup builds overlapping
    # families and does not partition, so that one sector's search finishes while the other's goes
    # on, and the turns skip it.
    tune(overlapping_minimum_work=0, partition_work_ratio=MAX_VALUE)
    code = get_random_css_code(np.random.default_rng(6), 60, 20, 25)
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    search = bz.BrouwerZimmermannSearch(sectors, cutoff=0)
    num_searches = len(prepared)
    assert num_searches == 2 and not search.finished
    lightest_rows = [basis[np.any(labels, axis=1)].sum(axis=1).min() for basis, labels in sectors]
    assert search.best <= min(lightest_rows)
    bests = [search.best]
    while not search.finished:
        num_visits = len(visits)
        search.advance()
        assert len(visits) == num_visits + 1
        bests.append(search.best)
    assert bests == sorted(bests, reverse=True) and bests[-1] == 5

    # Each search that advances is the first unfinished one after the last, in cyclic order.
    previous = -1
    for index, finished in visits:
        candidates = [(previous + offset) % num_searches for offset in range(1, num_searches + 1)]
        assert index == next(candidate for candidate in candidates if not finished[candidate])
        previous = index
    assert any(
        index == previous_index for (previous_index, _), (index, _) in itertools.pairwise(visits)
    )

    # Once finished, the search does nothing.
    num_visits = len(visits)
    search.advance()
    assert len(visits) == num_visits


def test_searches_that_setup_settles_are_finished_at_once(
    tune: Tune, prepared: list[bz._SectorSearch]
) -> None:
    """A search is finished when built if setup settles it.

    Setup settles the sector of the Hamming code directly.  In the random code, it settles one
    sector, whose witness finishes the search of the other.
    """
    hamming_code = qldpc.codes.HammingCode(3)
    sector = (np.asarray(hamming_code.generator, dtype=np.uint8), np.zeros((4, 0), np.uint8))
    search = bz.BrouwerZimmermannSearch([sector], cutoff=7)
    assert search.finished and search.best == 3 and not prepared

    # Setup keeps the disjoint family of each sector.
    tune(overlapping_minimum_work=MAX_VALUE, partition_work_ratio=MAX_VALUE)
    code = get_random_css_code(np.random.default_rng(153), 16, 5, 5)
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    search = bz.BrouwerZimmermannSearch(sectors, cutoff=0)
    assert len(prepared) == 1 and search.finished
    assert search.best == code.get_distance(method="brute_force") == 2


def test_a_witness_from_setup_lowers_the_shared_bound_once_its_search_advances(
    prepared: list[bz._SectorSearch],
) -> None:
    """A witness that setup finds in the rows of a family lowers ``best`` once its search advances.

    The rows of the basis weigh four or more, and a row of the family's one set, on the first three
    columns, weighs three, the distance.  The words of layer two of that set weigh more.
    """
    basis = np.array(
        [
            [0, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert basis.sum(axis=1).min() == 4
    family, _ = _get_family(basis, [[0, 1, 2]])
    tuning = dataclasses.replace(
        TABLES_DISABLED, overlapping_minimum_work=MAX_VALUE, partition_work_ratio=MAX_VALUE
    )
    search = bz.BrouwerZimmermannSearch(
        [(basis, _no_labels(basis))],
        cutoff=0,
        get_disjoint_basis_families=[lambda: family],
        tuning=tuning,
    )
    (sector_search,) = prepared
    # The shared bound may already hold the witness before the search advances.
    assert search.best >= sector_search.best == 3
    search.advance()
    assert (
        search.best
        == qldpc.codes.distance.get_distance_from_generators(basis, method="brute_force")
        == 3
    )


@pytest.mark.usefixtures("switch_before_any_layer", "completion_tables")
def test_automorphism_family_searches_find_exact_distances() -> None:
    """Searches given automorphisms agree with brute force, alone or sharing witnesses."""
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    identity = np.arange(len(code))[None]
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    for sector, pauli in zip(sectors, PAULIS_XZ, strict=True):
        expected = qldpc.codes.distance.get_distance_from_generators(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
            cutoff=0,
            method="brute_force",
        )
        for get_automorphisms in (
            code._get_permutation_automorphisms,
            lambda: identity,
            lambda: None,
        ):
            distance = bz.get_distance_brouwer_zimmermann(
                [sector], cutoff=0, get_automorphisms=get_automorphisms
            )
            assert distance == expected
    assert bz.get_distance_brouwer_zimmermann(
        sectors, cutoff=0, get_automorphisms=code._get_permutation_automorphisms
    ) == code.get_distance(method="brute_force")


def test_stepped_searches_use_the_tuning_that_they_are_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Searches built with a tuning prepare every sector with it, whatever the active tuning.

    The function that builds the search passes its tuning on, so calling it checks both.
    """
    tunings: list[Any] = []
    prepare = bz._prepare_sector_search

    def recording_prepare(*args: Any, **kwargs: Any) -> Any:
        tunings.append(kwargs["tuning"])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(bz, "_prepare_sector_search", recording_prepare)
    code = get_random_css_code(np.random.default_rng(153), 16, 5, 5)
    tuning = dataclasses.replace(DEFAULT_TUNING, search_chunk_exponent=5)
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    with use_tuning(search_chunk_exponent=6):
        bz.get_distance_brouwer_zimmermann(sectors, cutoff=0, tuning=tuning)
    assert tunings == [tuning, tuning]


def test_the_next_layer_work_is_that_of_the_layer_that_advance_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Searches report the work of the layer that ``advance`` scans next, and none at the end.

    The work counts the supports of the sets that the next unfinished search plans to scan, an
    upper bound on the supports that it scans.  A layer of an automorphism family scans its one
    set.  The surface code scans several sets per layer, the random code skips the search of a
    finished sector, and the bivariate bicycle code switches to automorphism families.
    """
    asked: list[bz._SectorSearch] = []
    scanned: list[tuple[bz._SectorSearch, int]] = []
    get_layer_work = bz._SectorSearch.get_layer_work
    scan_set = bz._SectorSearch._scan_set

    def recording_get_layer_work(search: bz._SectorSearch) -> int:
        asked.append(search)
        return get_layer_work(search)

    def recording_scan_set(search: bz._SectorSearch, index: int, weight: int) -> bool:
        scanned.append((search, math.comb(search.dimension, weight)))
        return scan_set(search, index, weight)

    monkeypatch.setattr(bz._SectorSearch, "get_layer_work", recording_get_layer_work)
    monkeypatch.setattr(bz._SectorSearch, "_scan_set", recording_scan_set)
    layers: list[tuple[int, int, int, bool]] = []
    for code, minimum_layer_work, distance in (
        (qldpc.codes.SurfaceCode(5), MAX_VALUE, 5),
        (get_random_css_code(np.random.default_rng(4), 60, 20, 25), MAX_VALUE, 4),
        (qldpc.codes.BBCode({x: 9, y: 3}, 1 + x + x**6 * y, 1 + y + x**2 * y), 0, None),
    ):
        sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
        with use_tuning(automorphism_minimum_layer_work=minimum_layer_work):
            search = bz.BrouwerZimmermannSearch(
                sectors, cutoff=0, get_automorphisms=code._get_permutation_automorphisms
            )
            while not search.finished:
                asked.clear()
                scanned.clear()
                work = search.get_next_layer_work()
                (sector_search,) = asked
                uses_family = sector_search.uses_automorphism_family
                search.advance()
                assert scanned and all(scanner is sector_search for scanner, _ in scanned)
                layers.append(
                    (work, sum(supports for _, supports in scanned), len(scanned), uses_family)
                )
                # Stop the bivariate bicycle code at its first family layer.
                if uses_family:
                    break
        if distance is not None:
            assert search.best == distance and search.get_next_layer_work() == 0
    assert all(supports <= work for work, supports, _, _ in layers)
    assert any(supports == work and num_sets > 1 for work, supports, num_sets, _ in layers)
    family_layers = [(work, supports) for work, supports, _, uses_family in layers if uses_family]
    assert family_layers and all(work == supports for work, supports in family_layers)


def test_automorphism_family_searches_scan_their_new_set_from_layer_two(
    tune: Tune, completion_tables: CompletionRecord
) -> None:
    """A search that switches to automorphism images before layer three rescans from layer two."""
    size, word = 8, [1, 1, 0, 1, 1, 0, 0, 0]
    basis = np.hstack(
        [np.eye(size, dtype=np.uint8), np.array([np.roll(word, i) for i in range(size)], np.uint8)]
    )
    labels = _no_labels(basis)
    shift = np.roll(np.arange(size), -1)
    automorphisms = np.concatenate([shift, shift + size])[None]
    first_family = basis_families.BasisFamily(
        basis[None],
        labels[None],
        np.arange(size)[None],
        np.array([size]),
        basis_families.infer_weight_divisor(basis),
        1,
    )
    # The search switches before layer three, whose work, unlike that of layer two, reaches the
    # threshold.
    tune(automorphism_minimum_layer_work=(math.comb(size, 2) + math.comb(size, 3)) // 2)
    distance = bz.get_distance_brouwer_zimmermann(
        [(basis, labels)],
        cutoff=0,
        get_disjoint_basis_families=[lambda: first_family],
        get_automorphisms=lambda: automorphisms,
    )
    assert (
        distance
        == qldpc.codes.distance.get_distance_from_generators(basis, method="brute_force")
        == 4
    )

    # With tables in every layer, the search builds tables for its first set, and again for the
    # set of its automorphism family, after it drops the tables of the first set.
    if completion_tables.forced:
        built = [(held, tables) for held, tables in completion_tables.choices if tables is not held]
        assert len(built) == 2 and built[1][0] is None


def test_searches_ask_for_automorphisms_only_before_costly_layers() -> None:
    """The search of a sector asks for automorphisms at most once.

    It asks only before a costly layer or partitioning.
    """
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    sector = get_sector(code, Pauli.X)
    for minimum_layer_work, expected_calls in ((10**9, 0), (0, 1)):
        get_automorphisms = mock.Mock(wraps=code._get_permutation_automorphisms)
        with use_tuning(automorphism_minimum_layer_work=minimum_layer_work):
            bz.get_distance_brouwer_zimmermann(
                [sector], cutoff=0, get_automorphisms=get_automorphisms
            )
        assert get_automorphisms.call_count == expected_calls


@pytest.fixture
def switch_decisions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
    """Record each consideration of a switch to automorphism images: the best weight and outcome."""
    decisions: list[tuple[int, bool]] = []
    switch = bz._SectorSearch._switch_to_automorphism_family

    def recording_switch(search: bz._SectorSearch) -> None:
        switch(search)
        decisions.append((search.best, search.uses_automorphism_family))

    monkeypatch.setattr(bz._SectorSearch, "_switch_to_automorphism_family", recording_switch)
    return decisions


def test_a_declined_switch_is_reconsidered_once_the_best_weight_drops(
    switch_decisions: list[tuple[int, bool]],
) -> None:
    """A search reconsiders a declined switch to automorphism images only at a lower best weight."""
    tuning = dataclasses.replace(
        TABLES_DISABLED,
        automorphism_minimum_layer_work=0,
        overlapping_minimum_work=MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )
    for coefficients, num_columns, step, expected in (
        ("1001111001000111111110111", 45, 5, [(10, False), (8, True)]),
        ("1111000101011", 21, 7, [(8, False)]),
    ):
        basis = _get_cyclic_basis(coefficients, num_columns)
        labels = _no_labels(basis)
        get_automorphisms = mock.Mock(return_value=np.roll(np.arange(num_columns), -step)[None])
        switch_decisions.clear()
        search = bz._prepare_sector_search(
            basis,
            labels,
            cutoff=0,
            weight_divisor=1,
            upper_bound=bz._get_lightest_eligible_row_weight(basis, labels),
            get_automorphisms=get_automorphisms,
            tuning=tuning,
        )
        assert isinstance(search, bz._SectorSearch)
        get_family = search.get_automorphism_family = mock.Mock(
            wraps=search.get_automorphism_family
        )
        while not search.finished:
            search.advance()
        assert search.best == bz.get_distance_brouwer_zimmermann([(basis, labels)], cutoff=0) == 8
        assert switch_decisions == expected
        assert get_automorphisms.call_count == get_family.call_count == 1


def test_a_decline_at_the_end_of_setup_is_reconsidered_before_a_costly_layer(
    switch_decisions: list[tuple[int, bool]],
) -> None:
    """A switch declined in setup is reconsidered before a costly layer at a lower best weight."""
    rows = [
        "110100010001010110111011100110110110011100111",
        "011011101011000011001011111000101110001010101",
        "100101000101010111001110101000010011111011111",
        "111101001100110000100000001100010011011111000",
        "110010001101111111000101101001000100101100000",
        "100010011000101010110100010011101000011101011",
        "111111011100010010010001000110001000101010110",
        "000100000000111111010000111010011111011010000",
        "101011110111000101110101000110111101011000111",
        "110100101100111101110010110000001000010110100",
        "111011000100110101100001001000000001011011101",
        "011101111001000011000000010010011001001011010",
        "101011100001101010101101111000111010101101101",
        "010011010011011001010101011111101000010000001",
        "110101000111000111001100101110000000010101011",
        "101111100100100000001101000000010111010110000",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = _no_labels(basis)
    identity = np.arange(basis.shape[1])[None]
    expected_distance = qldpc.codes.distance.get_distance_from_generators(
        basis, method="brute_force"
    )
    get_information_sets = matroid_partition.get_information_sets_with_coverage
    for minimum_layer_work, expected in ((0, [(10, False), (9, False)]), (10**9, [(10, False)])):
        switch_decisions.clear()
        with (
            use_tuning(
                partition_work_ratio=0,
                max_information_sets=16,
                automorphism_minimum_layer_work=minimum_layer_work,
            ),
            mock.patch.object(
                matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
            ) as partition,
        ):
            distance = bz.get_distance_brouwer_zimmermann(
                [(basis, labels)], cutoff=0, get_automorphisms=lambda: identity
            )
        assert distance == expected_distance == 9
        assert partition.called
        assert switch_decisions == expected


def test_a_late_switch_weighs_the_layers_left_in_the_current_family(
    tune: Tune, switch_decisions: list[tuple[int, bool]]
) -> None:
    """A search declines a switch that would cost more than the layers that its family has left."""
    # The threshold lies between the work of layer two, at most 210 for the two sets, and that of
    # layer three of one set, 455.
    tune(
        automorphism_minimum_layer_work=300,
        overlapping_minimum_work=MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )
    # The search considers the switch before layer three, after its two disjoint sets have found a
    # word of weight eight.  The images would cost 575 evaluations, more than the 455 of layer three
    # of one set, which certifies eight, though the whole search of the disjoint sets predicts 695.
    basis = _get_cyclic_basis("111100101101100010011", 35)
    labels = _no_labels(basis)
    automorphisms = np.roll(np.arange(35), -7)[None]
    distance = bz.get_distance_brouwer_zimmermann(
        [(basis, labels)], cutoff=0, get_automorphisms=lambda: automorphisms
    )
    assert distance == bz.get_distance_brouwer_zimmermann([(basis, labels)], cutoff=0) == 8
    assert switch_decisions == [(8, False)]


def test_searches_switch_to_automorphism_images_instead_of_partitioning(tune: Tune) -> None:
    """A search skips partitioning, and switches at once, if automorphism images do no more work."""
    tune(partition_work_ratio=0, max_information_sets=16)
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    sector = get_sector(code, Pauli.X)
    identity = np.arange(len(code))[None]
    expected = qldpc.codes.distance.get_distance_from_generators(
        code.get_logical_ops(Pauli.X), code.get_stabilizer_ops(Pauli.X), method="brute_force"
    )
    get_information_sets = matroid_partition.get_information_sets_with_coverage
    switch = bz._SectorSearch._switch_to_automorphism_family

    # The switch spy records whether each search that considers a switch makes it.  A search that
    # partitions in setup may finish there, and consider none.
    switched: list[bool] = []

    def _spy_switch(search: bz._SectorSearch) -> None:
        switch(search)
        switched.append(search.uses_automorphism_family)

    # With the code's translations, the search switches at once.  With the identity alone, no
    # automorphisms, or the largest threshold, which asks for none, it partitions instead.
    for get_automorphisms, minimum_layer_work, partitions in (
        (code._get_permutation_automorphisms, 10**9, False),
        (code._get_permutation_automorphisms, MAX_VALUE, True),
        (lambda: identity, 0, True),
        (lambda: None, 0, True),
    ):
        recorder = mock.Mock(wraps=get_automorphisms)
        switched.clear()
        with (
            use_tuning(automorphism_minimum_layer_work=minimum_layer_work),
            mock.patch.object(
                matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
            ) as partition,
            mock.patch.object(bz._SectorSearch, "_switch_to_automorphism_family", _spy_switch),
        ):
            distance = bz.get_distance_brouwer_zimmermann(
                [sector], cutoff=0, get_automorphisms=recorder
            )
        assert distance == expected
        assert partition.called == partitions
        assert recorder.call_count == (minimum_layer_work != MAX_VALUE)
        assert any(switched) != partitions

    # The identity gives this [21, 8, 6] basis no cheaper family, so its search partitions, and then
    # considers and declines the switch at the end of setup.
    switched.clear()
    with (
        use_tuning(automorphism_minimum_layer_work=10**9),
        mock.patch.object(
            matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
        ) as partition,
        mock.patch.object(bz._SectorSearch, "_switch_to_automorphism_family", _spy_switch),
    ):
        assert (
            qldpc.codes.distance.get_distance_from_generators(
                get_costly_basis(),
                cutoff=0,
                automorphisms=np.arange(21)[None],
                method="brouwer_zimmermann",
            )
            == 6
        )
    assert partition.called
    assert switched == [False]


def test_a_switch_at_the_end_of_setup_can_certify_the_distance(
    tune: Tune, prepared: list[bz._SectorSearch], switch_decisions: list[tuple[int, bool]]
) -> None:
    """A switch at the end of setup can certify the distance, so that the search ends in setup."""
    tune(overlapping_minimum_work=MAX_VALUE, partition_work_ratio=0)
    code = qldpc.codes.BBCode({x: 6, y: 2}, x**3 * y + x**2 * y + x, x**5 + x**3 * y + x)
    expected = qldpc.codes.distance.get_distance_from_generators(
        code.get_logical_ops(Pauli.X), code.get_stabilizer_ops(Pauli.X), method="brute_force"
    )
    distance = bz.get_distance_brouwer_zimmermann(
        [get_sector(code, Pauli.X)],
        cutoff=0,
        get_automorphisms=code._get_permutation_automorphisms,
    )
    assert distance == expected
    assert switch_decisions and not prepared


@pytest.mark.parametrize("partition_work_ratio", [MAX_VALUE, 0])
@pytest.mark.usefixtures("switch_before_any_layer")
def test_rows_of_an_automorphism_family_can_reach_the_cutoff(
    tune: Tune, partition_work_ratio: int
) -> None:
    """The rows of a scanned set with images can end a search at the cutoff."""
    # The rows of the disjoint family weigh more than six, and a row of the scanned set weighs six.
    # If partitioning may try some coverage and passes its gate, and some partitioned family could
    # do less work, the search asks for automorphisms to decide whether to partition, and switches
    # at the end of setup.  Otherwise it switches before its first layer.
    tune(overlapping_minimum_work=MAX_VALUE, partition_work_ratio=partition_work_ratio)
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    sector = get_sector(code, Pauli.Z)
    assert bz.get_distance_brouwer_zimmermann([sector], cutoff=0) == 4
    distance = bz.get_distance_brouwer_zimmermann(
        [sector], cutoff=6, get_automorphisms=code._get_permutation_automorphisms
    )
    assert distance == 6


@pytest.mark.usefixtures("switch_before_any_layer")
@pytest.mark.parametrize("partitions", [False, True])
def test_searches_reject_wrong_automorphisms(tune: Tune, partitions: bool) -> None:
    """A permutation that does not preserve a sector fails loudly when a search would use it.

    The search asks for automorphisms before its first layer, or, if it may partition, when it is
    built, to weigh partitioning against them (Section 3.7 of ``brouwer_zimmermann.md``).
    """
    tune(partition_work_ratio=0 if partitions else MAX_VALUE)
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    transposition = np.arange(len(code))
    transposition[[0, 1]] = [1, 0]
    build = functools.partial(
        bz.BrouwerZimmermannSearch,
        [get_sector(code, Pauli.X)],
        cutoff=0,
        get_automorphisms=lambda: transposition[None],
    )
    if partitions:
        with pytest.raises(ValueError, match="not an automorphism"):
            build()
    else:
        search = build()
        with pytest.raises(ValueError, match="not an automorphism"):
            search.advance()


@pytest.mark.usefixtures("switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_completion_tables_keep_distances_exact(completion_tables: CompletionRecord) -> None:
    """Searches that skip completions by tables agree with independent oracles."""

    def assert_skips(distance: int) -> int:
        assert completion_tables.skipped > skipped
        return distance

    # A fixed seed, since whether the tables of a search skip anything depends on its columns.
    rng = np.random.default_rng(0)
    code = qldpc.codes.MBCode({x: 12}, x**3 + x**8 + x**10 + x**11, x**5 + x**6 + x**7 + x**8)
    for pauli in PAULIS_XZ:
        expected = qldpc.codes.distance.get_distance_from_generators(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
            method="brute_force",
        )
        assert expected is not qldpc.codes.NO_DISTANCE
        for upper_bound in (None, expected + 1):
            skipped = completion_tables.skipped
            # The checks of each sector of this [[24, 4, 4]] code have rank ten, so its search basis
            # has ten tail columns, which one sixteen-column block covers.  That makes the bounds
            # exact, so a bound that is too high changes the distance.
            with use_tuning(completion_bound_block_bits=16):
                distance = bz.get_distance_brouwer_zimmermann(
                    [get_sector(code, pauli)],
                    cutoff=0,
                    upper_bound=upper_bound,
                    get_automorphisms=code._get_permutation_automorphisms,
                )
            assert assert_skips(distance) == expected

    # Relabel the columns, and the automorphisms alike: column ``j`` is column ``columns[j]`` of
    # the given sector, so a generator ``g`` maps it to ``inverse[g[columns[j]]]``, for the inverse
    # permutation ``np.argsort(columns)``.  The connected-cluster method also finds the distance 8.
    # The relabeled basis is C-ordered, as the searches pass it, so that numba compiles no version
    # for another layout.
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    basis, labels = get_sector(code, Pauli.X)
    columns = rng.permutation(len(code))
    automorphisms = np.asarray(code._get_permutation_automorphisms())
    skipped = completion_tables.skipped
    distance = bz.get_distance_brouwer_zimmermann(
        [(np.ascontiguousarray(basis[:, columns]), labels)],
        cutoff=0,
        get_automorphisms=functools.partial(
            np.array, np.argsort(columns)[automorphisms[:, columns]]
        ),
    )
    assert assert_skips(distance) == 8

    # A general Pauli problem checks that searched words weigh twice their nontrivial qubits.
    five_qubit_code = qldpc.codes.FiveQubitCode()
    logicals, stabilizers = five_qubit_code.get_logical_ops(), five_qubit_code.get_stabilizer_ops()
    skipped = completion_tables.skipped
    quantum_distance = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, symplectic=True, cutoff=0, method="brouwer_zimmermann"
    )
    assert quantum_distance is not qldpc.codes.NO_DISTANCE
    assert assert_skips(quantum_distance) == qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, symplectic=True, method="brute_force"
    )


def test_sets_that_are_not_information_sets_count_pivot_ones_only_below_their_rank() -> None:
    """A set of rank zero has no pivot ones, so its words weigh only their packed rows."""
    search = bz._SectorSearch(
        affine_instance.AffineInstance(
            generators=np.array([[[7], [6]]], dtype=np.uint64),
            labels=np.array([[[1], [2]]], dtype=np.uint64),
            widths=np.array([1], dtype=np.int64),
            pivots=np.full((1, 2), -1, dtype=np.int64),
            certificate=certificates.Certificate(2, (0,)),
        ),
        cutoff=-1,
        tuning=ONE_SUPPORT_CHUNKS,
        best=10,
        lower_bound=0,
    )
    assert search.advance() == 2
    assert search.advance() == 1


@pytest.mark.usefixtures("switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_joint_searches_with_completion_tables(
    completion_tables: CompletionRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joint searches that use tables in every layer are exact, and free their tables."""
    searches: list[bz._SectorSearch] = []
    skipped: list[int] = []
    advance = bz._SectorSearch.advance

    def counting_advance(search: bz._SectorSearch) -> int:
        if not any(known is search for known in searches):
            searches.append(search)
            skipped.append(0)
        skipped_before = completion_tables.skipped
        best = advance(search)
        index = [id(known) for known in searches].index(id(search))
        skipped[index] += completion_tables.skipped - skipped_before
        return best

    monkeypatch.setattr(bz._SectorSearch, "advance", counting_advance)
    code = qldpc.codes.MBCode({x: 15}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    for upper_bound, expected in ((None, code.get_distance(method="brute_force")), (5, 5)):
        searches.clear()
        skipped.clear()
        distance = bz.get_distance_brouwer_zimmermann(
            sectors,
            cutoff=0,
            upper_bound=upper_bound,
            get_automorphisms=code._get_permutation_automorphisms,
        )
        assert distance == expected
        assert len(searches) == 2 and all(skipped)
        assert all(search.completion_tables is None for search in searches)


@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
@pytest.mark.parametrize("mixed", [False, True])
def test_layers_of_several_sets_scan_each_set_with_its_tables(
    completion_tables: CompletionRecord, monkeypatch: pytest.MonkeyPatch, mixed: bool
) -> None:
    """Searches that scan several sets of a layer, each with its own tables or none, are exact."""
    select = completion_bounds.select_completion_tables
    scan_set_method = bz._SectorSearch._scan_set
    scan_set = affine_search.scan_set
    num_choices, scanned_index, returned_to_first_set = 0, 0, False
    later_scans = {"with tables": 0, "without tables": 0}
    returns_with_tables = 0

    def mixing_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        nonlocal num_choices
        num_choices += 1
        tables = select(*args, **kwargs)
        return None if mixed and num_choices % 2 else tables

    def recording_scan_set(search: bz._SectorSearch, index: int, weight: int) -> bool:
        nonlocal scanned_index, returned_to_first_set
        returned_to_first_set = index == 0 and scanned_index > 0
        scanned_index = index
        return scan_set_method(search, index, weight)

    def checking_scan(*args: Any) -> tuple[int, int, int, int, int]:
        nonlocal returns_with_tables
        rows, width, tables, block_bits = args[0], args[3], args[10], args[12]
        if scanned_index:
            later_scans["with tables" if len(tables) else "without tables"] += 1
        if len(tables):
            returns_with_tables += returned_to_first_set
            own = completion_bounds.CompletionTables.build(rows[:, :width], block_bits)
            assert np.array_equal(tables, own.tables)
        return scan_set(*args)

    monkeypatch.setattr(completion_bounds, "select_completion_tables", mixing_select)
    monkeypatch.setattr(bz._SectorSearch, "_scan_set", recording_scan_set)
    monkeypatch.setattr(affine_search, "scan_set", checking_scan)
    generators = get_random_full_rank_matrix(np.random.default_rng(4), 10, 36)
    distance = bz.get_distance_brouwer_zimmermann([(generators, _no_labels(generators))], cutoff=0)
    assert distance == qldpc.codes.distance.get_distance_from_generators(
        generators, cutoff=0, method="brute_force"
    )
    assert completion_tables.skipped
    assert later_scans["with tables"]
    assert bool(later_scans["without tables"]) == mixed
    assert returns_with_tables


@pytest.mark.parametrize("cutoff", [0, 5])
def test_a_layer_of_several_sets_stops_once_its_sets_certify_the_best_weight(cutoff: int) -> None:
    """A search stops a layer after the set whose completion of it certifies the best weight."""
    generators = np.array(
        [
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    # Layer one certifies weight four, and in layer two set 0 finds a word of weight five, which its
    # completion of the layer certifies.  A cutoff of five stops the scan at that word, before the
    # search certifies more than four.
    family, certificate = _get_family(generators, [[0, 1, 2], [3, 4, 5]])
    search = bz._SectorSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=cutoff,
        tuning=dataclasses.replace(TABLES_DISABLED, search_chunk_exponent=20),
        best=6,
        lower_bound=certificate.lower_bound(1),
        next_weight=2,
    )
    scanned: list[int] = []
    scan_set_method = bz._SectorSearch._scan_set

    def recording_scan_set(search: bz._SectorSearch, index: int, weight: int) -> bool:
        scanned.append(index)
        return scan_set_method(search, index, weight)

    with mock.patch.object(bz._SectorSearch, "_scan_set", recording_scan_set):
        assert search.advance() == 5
    assert search.finished and search.lower_bound == (4 if cutoff else 5)
    assert scanned == [0]


def test_a_layer_with_a_set_that_is_not_an_information_set_is_exact() -> None:
    """A layer that scans a full set and a set of deficient rank finds the lightest word."""
    generators = np.array(
        [
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # The second set has rank two, and finds the lightest word, of weight four, in layer two.  The
    # tuning forces tables, so the second set must scan without those of the first set.
    family, certificate = _get_family(generators, [[0, 1, 2], [3, 4, 5]])
    search = bz._SectorSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=0,
        tuning=dataclasses.replace(
            DEFAULT_TUNING,
            search_chunk_exponent=20,
            completion_bound_minimum_layer_work=0,
            completion_bound_max_cost_percent=MAX_VALUE,
        ),
        best=5,
        lower_bound=certificate.lower_bound(1),
        next_weight=2,
    )
    while not search.finished:
        search.advance()
    assert (
        search.best
        == qldpc.codes.distance.get_distance_from_generators(
            generators, cutoff=0, method="brute_force"
        )
        == 4
    )


def test_searches_hold_the_tables_of_one_set_at_a_time(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search frees the tables of one set before it builds the tables of another."""
    tune(**FORCED_TABLES)
    build = completion_bounds.CompletionTables.build
    built: list[weakref.ref[completion_bounds.CompletionTables]] = []

    def checking_build(
        rows: npt.NDArray[np.uint64], block_bits: int
    ) -> completion_bounds.CompletionTables:
        gc.collect()
        assert all(ref() is None for ref in built)
        tables = build(rows, block_bits)
        built.append(weakref.ref(tables))
        return tables

    monkeypatch.setattr(completion_bounds.CompletionTables, "build", staticmethod(checking_build))
    generators = get_random_full_rank_matrix(np.random.default_rng(4), 10, 36)
    distance = bz.get_distance_brouwer_zimmermann([(generators, _no_labels(generators))], cutoff=0)
    assert distance == qldpc.codes.distance.get_distance_from_generators(
        generators, cutoff=0, method="brute_force"
    )
    assert len(built) > 1


@pytest.mark.usefixtures("switch_before_any_layer")
def test_searches_take_completion_limits_from_their_tuning(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searches use completion tables within the limits of their tuning."""
    block_bits: set[int] = set()
    select = completion_bounds.select_completion_tables

    def recording_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        tables = select(*args, **kwargs)
        if tables is not None:
            block_bits.add(tables.block_bits)
        return tables

    monkeypatch.setattr(completion_bounds, "select_completion_tables", recording_select)
    code = qldpc.codes.MBCode({x: 12}, x**3 + x**8 + x**10 + x**11, x**5 + x**6 + x**7 + x**8)
    sectors = [get_sector(code, pauli) for pauli in PAULIS_XZ]
    tune(completion_bound_minimum_layer_work=0, completion_bound_max_cost_percent=MAX_VALUE)
    # Each set of either sector has ten tail columns, which the default tables cover with one block.
    # Each of the other limits is far from what any tables of these sets need.
    for limits, expected_block_bits in (
        ({}, {16}),
        ({"completion_bound_max_table_bytes": 1}, set()),
        ({"completion_bound_max_cost_percent": 0}, set()),
        ({"completion_bound_minimum_layer_work": MAX_VALUE}, set()),
    ):
        block_bits.clear()
        with use_tuning(**limits):
            assert bz.get_distance_brouwer_zimmermann(
                sectors,
                cutoff=0,
                get_automorphisms=code._get_permutation_automorphisms,
            ) == code.get_distance(method="brute_force")
        assert block_bits == expected_block_bits


def test_searches_need_a_sector_or_an_upper_bound() -> None:
    """Without sectors, the result is the known upper bound, which must then be given."""
    with pytest.raises(ValueError, match="at least one sector or an upper bound"):
        bz.get_distance_brouwer_zimmermann([], cutoff=0)
    assert bz.get_distance_brouwer_zimmermann([], cutoff=0, upper_bound=3) == 3


class ItemScan(NamedTuple):
    """A scan of a chunk of work items: its thread, bytes of tables, counts, and finds.

    The counts are those of supports weighed, of lookups, and of supports skipped.  ``lowered`` says
    whether the chunk found a word lighter than the best weight that it started from.
    """

    thread: int
    table_bytes: int
    counts: tuple[int, int, int]
    lowered: bool


@pytest.fixture
def item_scans(monkeypatch: pytest.MonkeyPatch) -> list[ItemScan]:
    """Record each scan of a chunk of work items, which only a layer split across threads runs."""
    scans: list[ItemScan] = []
    scan_set_items = affine_search.scan_set_items

    def recording_scan(*args: Any) -> tuple[int, int, int, int, int]:
        result = scan_set_items(*args)
        counts = (int(result[2]), int(result[3]), int(result[4]))
        lowered = bool(result[0] < args[7])
        scans.append(ItemScan(threading.get_ident(), len(args[10]), counts, lowered))
        return result

    monkeypatch.setattr(affine_search, "scan_set_items", recording_scan)
    return scans


@pytest.fixture
def small_batches(tune: Tune) -> None:
    """Make small searches scan their layers of three or more rows on threads, if they have any.

    Small batches open the gate of splitting a layer, and searches skip the costly stages of setup,
    which would end them before such layers.
    """
    tune(
        search_chunk_exponent=3,
        overlapping_minimum_work=MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )


@pytest.mark.usefixtures("small_batches")
@pytest.mark.parametrize("num_threads", [2, 3, 8])
def test_searches_on_threads_find_the_distances_that_one_thread_finds(
    item_scans: list[ItemScan], num_threads: int
) -> None:
    """Searches that split layers across threads find the distances of searches on one thread.

    The searches are of:

    - the [23, 11, 8] even-weight subcode of the [23, 12, 7] Golay code, a classical code;
    - the X sector, and both sectors, of the [[25, 1, 5]] rotated surface code;
    - a [[16, 1, 4]] rotated surface code with Hadamards on half of its qubits, which is not CSS;
    - the [[25, 1, 5]] Bacon-Shor code, a subsystem code (see
      https://arxiv.org/abs/quant-ph/0506023).

    On one thread they call no ``scan_set_items``, and on more threads they do.
    """
    surface = qldpc.codes.SurfaceCode(5)
    small_surface = qldpc.codes.SurfaceCode(4)
    hadamard = small_surface.conjugated(range(len(small_surface) // 2)).matrix
    method: qldpc.codes.DistanceMethod = "brouwer_zimmermann"
    searches: list[tuple[int, Callable[[], object]]] = [
        (
            8,
            functools.partial(
                distances.get_distance_from_generators, get_golay_checks(), method=method
            ),
        ),
        (
            5,
            functools.partial(
                distances.get_distance_css,
                surface.matrix_x,
                surface.matrix_z,
                Pauli.X,
                method=method,
            ),
        ),
        (
            5,
            functools.partial(
                distances.get_distance_css, surface.matrix_x, surface.matrix_z, method=method
            ),
        ),
        (4, functools.partial(distances.get_distance_pauli, hadamard, method=method)),
        (
            5,
            functools.partial(
                distances.get_distance_pauli, qldpc.codes.BaconShorCode(5).matrix, method=method
            ),
        ),
    ]
    for expected, search in searches:
        item_scans.clear()
        with use_tuning(num_threads=1):
            assert search() == expected
        assert not item_scans
        with use_tuning(num_threads=num_threads):
            assert search() == expected
        assert item_scans


@pytest.mark.usefixtures("small_batches")
def test_threads_find_the_lightest_words_of_layers_split_across_them(
    item_scans: list[ItemScan],
) -> None:
    """Threads find the lightest words of searches whose lightest words lie in split layers.

    The searches are of a random [50, 30] classical code and of the X sector of a [[54, 4]]
    bivariate bicycle code, whose X distance is eight.  Each finds the distance that a search on one
    thread finds.  The stabilizers of the bivariate bicycle code weigh six, so only its search tests
    on threads the labels that tell its logical operators from its stabilizers.
    """
    random_code = get_random_full_rank_matrix(np.random.default_rng(1), 30, 50)
    bicycle = qldpc.codes.BBCode({x: 9, y: 3}, 1 + x + x**6 * y, 1 + y + x**2 * y)
    method: qldpc.codes.DistanceMethod = "brouwer_zimmermann"
    searches: list[tuple[int, Callable[[], object]]] = [
        (5, functools.partial(distances.get_distance_from_generators, random_code, method=method)),
        (
            8,
            functools.partial(
                distances.get_distance_css,
                bicycle.matrix_x,
                bicycle.matrix_z,
                Pauli.X,
                method=method,
            ),
        ),
    ]
    for expected, search in searches:
        with use_tuning(num_threads=1):
            assert search() == expected
        item_scans.clear()
        with use_tuning(num_threads=3):
            assert search() == expected
        assert any(scan.lowered for scan in item_scans)


@pytest.mark.parametrize(
    ("num_rows", "weight", "num_threads", "length"),
    [(12, 3, 8, 1), (12, 5, 8, 2), (70, 4, 1, 1), (40, 4, 4, 2)],
)
def test_work_items_fix_the_first_rows_of_supports_in_lexicographic_order(
    num_rows: int, weight: int, num_threads: int, length: int
) -> None:
    """The items of a layer fix one row, or two if fixing one leaves fewer than 32 for each thread.

    Their fixed rows are the first rows of the supports of the layer, in lexicographic order.  Their
    weights count their supports.
    """
    items = bz._WorkItems(num_rows, weight, num_threads)
    supports = collections.Counter(
        support[:length] for support in itertools.combinations(range(num_rows), weight)
    )
    expected = sorted(supports)
    assert items.length == length and items.num_items == len(expected)
    assert [tuple(items.get_item(index)) for index in range(items.num_items)] == expected
    np.testing.assert_array_equal(
        items.cumulative_weights, np.cumsum([supports[item] for item in expected])
    )


@pytest.mark.usefixtures("switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_searches_on_threads_advance_as_searches_on_one_thread(
    tune: Tune,
    completion_tables: CompletionRecord,
    prepared: list[bz._SectorSearch],
    item_scans: list[ItemScan],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After each layer, a search on threads has the bounds and best weight of one on one thread.

    So the decisions that follow, such as the choice of completion tables and the switch to an
    automorphism family, are those of one thread.  The search uses both.  The threads run with
    every array of the search, such as its completion tables, read-only.
    """
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    sector = get_sector(code, Pauli.X)
    histories = []
    # The threads scan slices of the arrays of the sector search, which it takes before they start.
    checked = check_shared_arrays_on_threads(
        monkeypatch,
        lambda sector_search, index, weight, tables: [search, sector_search, tables],
        bz._SectorSearch,
        "_scan_set_on_threads",
    )
    for num_threads in (1, 3):
        prepared.clear()
        tune(num_threads=num_threads, search_chunk_exponent=4)
        search = bz.BrouwerZimmermannSearch(
            [sector], cutoff=0, get_automorphisms=code._get_permutation_automorphisms
        )
        history = []
        while not search.finished:
            search.advance()
            history.append(
                [
                    (each.next_weight, each.lower_bound, each.best, each.uses_automorphism_family)
                    for each in prepared
                ]
            )
        histories.append(history)
    assert histories[0] == histories[1]
    assert histories[0][-1][0][2:] == (8, True)
    assert any(scan.table_bytes for scan in item_scans)
    assert checked and min(checked) > 0


@pytest.mark.usefixtures("small_batches", "switch_before_any_layer")
@pytest.mark.parametrize(("cutoff", "num_threads"), [(8, 2), (9, 3), (8, 8)])
def test_searches_on_threads_stop_at_the_cutoff_within_a_layer(
    tune: Tune,
    prepared: list[bz._SectorSearch],
    item_scans: list[ItemScan],
    monkeypatch: pytest.MonkeyPatch,
    cutoff: int,
    num_threads: int,
) -> None:
    """A stop at the cutoff, in a layer split across threads, leaves the bounds as before the layer.

    The search returns the weight of a word between the distance, eight, and the cutoff.  It scans
    one set of an automorphism family, whose best weight is ten before layer four.  A search on one
    thread finds the item of that layer that holds the first word within the cutoff.  On threads,
    ranges that start after that item wait for the stop event before they scan, so the threads
    stop before they claim the last item, and they scan no chunk of those ranges.  The range of
    that item waits until another thread holds such a range.  The ranges are sized by the weights
    of the items of the layer.
    """
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    sector = get_sector(code, Pauli.X)
    dimension, weight = 38, 4
    items = bz._WorkItems(dimension, weight, num_threads)

    # Find the fixed rows of the first word within the cutoff, at which a scan on one thread stops.
    witness_items: list[tuple[int, ...]] = []
    scan_set = affine_search.scan_set

    def recording_scan(*args: Any) -> tuple[int, int, int, int, int]:
        result = scan_set(*args)
        if result[0] <= cutoff and not witness_items:
            witness_items.append(tuple(args[4][: items.length].tolist()))
        return result

    with monkeypatch.context() as patch, use_tuning(num_threads=1):
        patch.setattr(affine_search, "scan_set", recording_scan)
        bz.get_distance_brouwer_zimmermann(
            [sector], cutoff=cutoff, get_automorphisms=code._get_permutation_automorphisms
        )
    item_rows = [tuple(items.get_item(index).tolist()) for index in range(items.num_items)]
    (witness_item,) = witness_items
    witness_index = item_rows.index(witness_item)
    prepared.clear()

    tune(num_threads=num_threads)
    gated = False
    ranges: list[tuple[int, int, int]] = []
    late_scans: list[int] = []
    later_claimed = threading.Event()
    weights: list[npt.NDArray[np.float64]] = []
    run_work_items = bz._threads.run_work_items

    def gated_run(
        num_items: int,
        threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **kwargs: Any,
    ) -> None:
        def gated_start(stop: threading.Event) -> Callable[[int, int], None]:
            run_range = start_worker(stop)

            def gated_range(start: int, end: int) -> None:
                ranges.append((start, end, num_items))
                if not (gated and start > witness_index):
                    # The range of the witness waits until another thread holds a later range.
                    if gated and witness_index < end:
                        assert later_claimed.wait(timeout=THREAD_TIMEOUT)
                    run_range(start, end)
                    return
                later_claimed.set()
                assert stop.wait(timeout=THREAD_TIMEOUT)
                thread = threading.get_ident()
                scans_before = sum(scan.thread == thread for scan in item_scans)
                run_range(start, end)
                late_scans.append(sum(scan.thread == thread for scan in item_scans) - scans_before)

            return gated_range

        if gated:
            weights.append(kwargs["cumulative_weights"])
        run_work_items(num_items, threads, gated_start, **kwargs)

    monkeypatch.setattr(bz._threads, "run_work_items", gated_run)
    search = bz.BrouwerZimmermannSearch(
        [sector], cutoff=cutoff, get_automorphisms=code._get_permutation_automorphisms
    )
    (sector_search,) = prepared
    while not search.finished:
        bounds = (sector_search.lower_bound, sector_search.next_weight)
        gated = bounds[1] == 4
        ranges.clear()
        search.advance()
    assert 8 <= search.best <= cutoff
    assert sector_search.uses_automorphism_family and bounds[1] == 4
    assert (sector_search.lower_bound, sector_search.next_weight) == bounds
    assert sector_search.dimension == dimension and ranges[0][2] == items.num_items
    assert all(end < num_items for _, end, num_items in ranges)
    assert late_scans and not any(late_scans)
    (layer_weights,) = weights
    np.testing.assert_array_equal(layer_weights, items.cumulative_weights)


@pytest.mark.usefixtures("small_batches")
@pytest.mark.parametrize("tables", [False, True])
def test_scans_on_threads_weigh_look_up_and_skip_what_one_thread_does(
    tune: Tune, item_scans: list[ItemScan], monkeypatch: pytest.MonkeyPatch, tables: bool
) -> None:
    """The scans of work items count, in total, the supports and lookups of a scan on one thread.

    The search, of the X sector of the [[25, 1, 5]] rotated surface code, starts at its distance,
    five, which it never lowers, so every chunk on any thread starts from the same best weight.
    With tables in every layer, the scans skip completions.
    """
    if tables:
        tune(**FORCED_TABLES)
    totals: list[npt.NDArray[np.int64]] = []
    scan_set = affine_search.scan_set

    def counting_scan(*args: Any) -> tuple[int, int, int, int, int]:
        result = scan_set(*args)
        totals[-1] += result[2:]
        return result

    monkeypatch.setattr(affine_search, "scan_set", counting_scan)
    sector = get_sector(qldpc.codes.SurfaceCode(5), Pauli.X)
    for num_threads in (1, 3):
        item_scans.clear()
        totals.append(np.zeros(3, dtype=np.int64))
        tune(num_threads=num_threads)
        assert bz.get_distance_brouwer_zimmermann([sector], cutoff=0, upper_bound=5) == 5
        totals[-1] += np.sum([scan.counts for scan in item_scans], axis=0, dtype=np.int64)
    assert item_scans
    assert totals[0][2] > 0 if tables else totals[0][1] == 0
    np.testing.assert_array_equal(totals[0], totals[1])


@pytest.mark.usefixtures("small_batches")
def test_an_error_on_a_thread_stops_a_search_on_threads(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error on a thread that scans work items ends the search with that error.

    No thread of the search is left running.  The other thread fails only once the calling thread
    has started to scan, and the calling thread scans its chunk only after the other thread has
    failed.  Each holds one range of items, and the layer has more, so neither can run out of items
    or see the stop event first.
    """
    tune(num_threads=2)
    calling_thread = threading.get_ident()
    scanning, failed = threading.Event(), threading.Event()
    scan_set_items = affine_search.scan_set_items

    def failing_scan(*args: Any) -> tuple[int, int, int, int, int]:
        if threading.get_ident() != calling_thread:
            assert scanning.wait(timeout=THREAD_TIMEOUT)
            failed.set()
            raise ValueError("failed")
        # The scan from depth -1 before the threads start returns at once.
        if args[6] >= 0:
            scanning.set()
            assert failed.wait(timeout=THREAD_TIMEOUT)
        return scan_set_items(*args)

    monkeypatch.setattr(affine_search, "scan_set_items", failing_scan)
    threads = set(threading.enumerate())
    surface = qldpc.codes.SurfaceCode(5)
    with pytest.raises(ValueError, match="failed"):
        bz.get_distance_brouwer_zimmermann([get_sector(surface, Pauli.X)], cutoff=0)
    assert set(threading.enumerate()) <= threads


def test_layers_of_a_set_of_deficient_rank_are_exact_on_threads(
    item_scans: list[ItemScan],
) -> None:
    """Threads scan the layers of a set whose block has deficient rank, and find the distance.

    The set of rank at most three on eight rows is the whole family, so the search scans every
    layer and finds the lightest word.  Its words weigh their packed rows plus a pivot one for
    each of their rows below the rank, which a scan that took every row for a pivot row would
    overcount.
    """
    generators = get_random_full_rank_matrix(np.random.default_rng(0), 8, 20)
    family, certificate = _get_family(generators, [[0, 1, 2]])
    search = bz._SectorSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=0,
        tuning=ONE_SUPPORT_CHUNKS,
        best=generators.shape[1] + 1,
        lower_bound=0,
        num_threads=3,
    )
    while not search.finished:
        search.advance()
    assert search.best == qldpc.codes.distance.get_distance_from_generators(
        generators, cutoff=0, method="brute_force"
    )
    assert any(scan.lowered for scan in item_scans)


@pytest.mark.usefixtures("small_batches")
def test_searches_resolve_their_number_of_threads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search under the default tuning keeps the thread count that numba had when it started."""
    thread_counts: list[int] = []
    run_work_items = bz._threads.run_work_items

    def recording_run(num_items: int, num_threads: int, *args: Any, **kwargs: Any) -> None:
        thread_counts.append(num_threads)
        run_work_items(num_items, num_threads, *args, **kwargs)

    monkeypatch.setattr(bz._threads, "run_work_items", recording_run)
    monkeypatch.setattr(bz._threads._num_threads, "NUM_THREADS", None)
    monkeypatch.setattr(bz._threads.numba.config, "NUMBA_NUM_THREADS", 3)
    code = qldpc.codes.SurfaceCode(5)
    search = bz.BrouwerZimmermannSearch([get_sector(code, Pauli.X)], cutoff=0)
    monkeypatch.setattr(bz._threads.numba.config, "NUMBA_NUM_THREADS", 5)
    while not search.finished:
        search.advance()
    assert thread_counts and set(thread_counts) == {3}


def _no_labels(basis: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Quotient labels with no columns, which exclude no subcode."""
    return np.empty((len(basis), 0), dtype=np.uint8)


def _get_cyclic_basis(coefficients: str, num_columns: int) -> npt.NDArray[np.uint8]:
    """A basis of a binary cyclic code: the shifts of the coefficients of its generator polynomial.

    ``coefficients`` lists them from the constant term up, and the polynomial must divide
    ``x**num_columns - 1``, so that shifting the columns preserves the code.
    """
    row = np.zeros(num_columns, dtype=np.uint8)
    row[: len(coefficients)] = [int(bit) for bit in coefficients]
    return np.array([np.roll(row, shift) for shift in range(num_columns - len(coefficients) + 1)])


def _get_family(
    generators: npt.NDArray[np.uint8], column_sets: list[list[int]]
) -> tuple[basis_families.BasisFamily, certificates.Certificate]:
    """A basis family without labels, of generators row reduced on each set of columns.

    Returns the family and its certificate, with the ranks that the row reduction finds.
    """
    num_rows = len(generators)
    reduced, pivots = [], np.full((len(column_sets), num_rows), -1, dtype=np.int64)
    for index, columns in enumerate(column_sets):
        rows, _, found = row_reduce_binary(generators, None, columns=np.array(columns))
        reduced.append(rows)
        pivots[index, : len(found)] = found
    ranks = np.count_nonzero(pivots >= 0, axis=1)
    family = basis_families.BasisFamily(
        np.array(reduced),
        np.empty((len(column_sets), num_rows, 0), dtype=np.uint8),
        pivots,
        ranks,
        1,
        1,
    )
    return family, certificates.Certificate(num_rows, tuple(ranks.tolist()))
