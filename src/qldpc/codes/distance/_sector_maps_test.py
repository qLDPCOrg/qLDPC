# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the qubit permutations that map the X sector of a CSS code onto its Z sector."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc._binary import get_independent_rows
from qldpc.external import _bliss

from . import _sector_maps
from .conftest import get_span


def test_sector_maps_swap_the_row_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found maps swap the row spaces, equal row spaces take the identity, and some codes have none.

    Codes whose X and Z checks have different ranks, such as rectangular surface and Bacon-Shor
    codes, have no map.  Without bliss, only the identity is found.
    """
    steane_x, steane_z = _get_checks(codes.SteaneCode())
    assert np.array_equal(np.asarray(_find(steane_x, steane_z)), np.arange(7))
    assert _find(*_get_checks(codes.SurfaceCode(3, 4))) is None
    assert _find(*_get_checks(codes.BaconShorCode(3, 4))) is None

    monkeypatch.setattr(_bliss, "is_installed", lambda: False)
    assert np.array_equal(np.asarray(_find(steane_x, steane_z)), np.arange(7))
    assert _find(*_get_checks(codes.SurfaceCode(3))) is None


@pytest.mark.skipif(not _bliss.is_installed(), reason="requires pybliss")
def test_bliss_finds_sector_maps() -> None:
    """Bliss finds maps of surface, toric, and Bacon-Shor codes, which swap their row spaces."""
    for code in [codes.SurfaceCode(3), codes.ToricCode(4), codes.BaconShorCode(3)]:
        checks_x, checks_z = _get_checks(code)
        permutation = _find(checks_x, checks_z)
        assert permutation is not None
        assert _swaps_row_spaces(checks_x, checks_z, permutation)


def test_equal_row_spaces_are_tested_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only equal row spaces take the identity, also when the random words do not tell them apart.

    The row space of the Z-type checks of the last pair lies strictly inside that of the X-type
    checks, so no permutation swaps them.
    """
    monkeypatch.setattr(_bliss, "is_installed", lambda: False)
    monkeypatch.setattr(_sector_maps, "_NUM_RANDOM_WORDS", 0)
    assert np.array_equal(np.asarray(_find(*_get_checks(codes.SteaneCode()))), np.arange(7))
    assert _find(*_get_checks(codes.SurfaceCode(3))) is None
    gauge_x = np.array([[0, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]], dtype=np.uint8)
    gauge_z = np.array([[1, 0, 0, 1, 1, 1]], dtype=np.uint8)
    assert _find(gauge_x, gauge_z) is None


def test_cheap_tests_reject_before_bliss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks of different weights, or qubits of unmatched degrees, rule out a map without bliss.

    Each pair of check matrices has equal ranks and different row spaces.  In the first, every
    qubit has X-degree 1 and Z-degree 1, but the X-type checks weigh 3 and 1 and the Z-type checks
    2 and 2.  In the second, every check weighs 2, but the qubits have X-degrees 1, 1, 1, 1 and
    Z-degrees 1, 1, 2, 0, so no qubit can map to qubit 2, whose Z-degree 2 would have to be the
    X-degree of its preimage.
    """
    calls: list[object] = []
    monkeypatch.setattr(_bliss, "is_installed", lambda: True)
    monkeypatch.setattr(_bliss, "get_matrix_isomorphism", lambda *args: calls.append(args))
    pairs = [
        ([[1, 1, 1, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [0, 0, 1, 1]]),
        ([[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 0, 1, 0], [0, 1, 1, 0]]),
    ]
    for checks_x, checks_z in pairs:
        gauge_x, gauge_z = np.array(checks_x, dtype=np.uint8), np.array(checks_z, dtype=np.uint8)
        assert _find(gauge_x, gauge_z) is None
        assert not any(
            _swaps_row_spaces(gauge_x, gauge_z, np.array(perm))
            for perm in itertools.permutations(range(4))
        )
    assert not calls


def test_candidates_are_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permutation that does not swap the row spaces is rejected, also when bliss returns it.

    A quarter turn of the 3 x 3 grid of qubits of the rotated surface code of distance 3, which
    maps the qubit in row r and column c to row 2 - c and column r, swaps its row spaces, and the
    identity does not.  A permutation that maps one row space onto the other, but not the other
    way around, is also rejected, in either order of the two matrices.
    """
    checks_x, checks_z = _get_checks(codes.SurfaceCode(3))
    rows, cols = np.indices((3, 3)).reshape(2, 9)
    turn = 3 * (2 - cols) + rows
    assert _sector_maps.maps_sectors(checks_x, checks_z, turn)
    assert _swaps_row_spaces(checks_x, checks_z, turn)
    assert not _sector_maps.maps_sectors(checks_x, checks_z, np.arange(9))
    swapped = turn.copy()
    swapped[[0, 4]] = swapped[[4, 0]]
    assert not _sector_maps.maps_sectors(checks_x, checks_z, swapped)
    assert not _swaps_row_spaces(checks_x, checks_z, swapped)
    one_way = np.array([2, 3, 4, 5, 0, 1])
    gauge_x = np.array([[1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    gauge_z = np.array([[0, 0, 1, 1, 0, 0]], dtype=np.uint8)
    assert not _sector_maps.maps_sectors(gauge_x, gauge_z, one_way)
    assert not _sector_maps.maps_sectors(gauge_z, gauge_x, one_way)

    # bliss stands in with no isomorphism, and then with one that fixes every row and qubit
    num_rows = len(checks_x) + len(checks_z)
    monkeypatch.setattr(_bliss, "is_installed", lambda: True)
    monkeypatch.setattr(_bliss, "get_matrix_isomorphism", lambda *_: None)
    assert _find(checks_x, checks_z) is None
    monkeypatch.setattr(_bliss, "get_matrix_isomorphism", lambda *_: list(range(num_rows + 9)))
    assert _find(checks_x, checks_z) is None
    monkeypatch.setattr(
        _bliss,
        "get_matrix_isomorphism",
        lambda *_: list(range(num_rows)) + [num_rows + int(qubit) for qubit in turn],
    )
    assert np.array_equal(np.asarray(_find(checks_x, checks_z)), turn)


def _find(
    gauge_x: npt.NDArray[np.uint8], gauge_z: npt.NDArray[np.uint8]
) -> npt.NDArray[np.int_] | None:
    """The result of get_sector_map, given the basis that _search_css passes it."""
    return _sector_maps.get_sector_map(gauge_x, gauge_z, get_independent_rows(gauge_x))


def _get_checks(code: codes.CSSCode) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The X-type and Z-type checks of a CSS code, as binary arrays."""
    return code.matrix_x.view(np.ndarray).astype(np.uint8), code.matrix_z.view(np.ndarray).astype(
        np.uint8
    )


def _swaps_row_spaces(
    checks_x: npt.NDArray[np.uint8],
    checks_z: npt.NDArray[np.uint8],
    permutation: npt.NDArray[np.int_],
) -> bool:
    """Whether a permutation swaps the row spaces of two matrices, by enumerating both spans."""
    images_x = np.zeros_like(checks_x)
    images_x[:, permutation] = checks_x
    images_z = np.zeros_like(checks_z)
    images_z[:, permutation] = checks_z
    return get_span(images_x) == get_span(checks_z) and get_span(images_z) == get_span(checks_x)
