# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded-weight error enumeration in lookup decoders."""

from __future__ import annotations

import itertools

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import math
from qldpc.decoders.custom._lookup_enumeration import _iter_errors_and_syndromes


def _brute_force(
    matrix: galois.FieldArray,
    max_weight: int,
    syndrome_mask: npt.NDArray[np.bool_] | None,
    symplectic: bool,
    error_channel: npt.NDArray[np.floating] | None = None,
    probability_cutoff: float = 0,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    """Map every admissible error to its syndrome by checking every vector over the field."""
    field = type(matrix)
    repeat = 2 if symplectic else 1
    num_sites = matrix.shape[1] // repeat
    expected = {}
    for vector in itertools.product(range(field.order), repeat=matrix.shape[1]):
        error = field(vector)
        sites = error.reshape(repeat, num_sites).view(np.ndarray).any(axis=0)
        if np.count_nonzero(sites) > max_weight:
            continue
        if probability_cutoff:
            assert error_channel is not None
            active = error.view(np.ndarray).astype(bool)
            probabilities = np.where(active, error_channel / (field.order - 1), 1 - error_channel)
            if np.prod(probabilities) < probability_cutoff:
                continue
        product = math.symplectic_conjugate(matrix) @ error if symplectic else matrix @ error
        syndrome = product.view(np.ndarray)
        if syndrome_mask is not None:
            if np.any(syndrome[~syndrome_mask]):
                continue
            syndrome = syndrome[syndrome_mask]
        expected[tuple(vector)] = tuple(syndrome.tolist())
    return expected


def _enumerate(
    matrix: galois.FieldArray,
    max_weight: int,
    syndrome_mask: npt.NDArray[np.bool_] | None,
    symplectic: bool,
    **kwargs: object,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [
        (tuple(error.tolist()), tuple(syndrome.tolist()))
        for error, syndrome in _iter_errors_and_syndromes(
            matrix,
            max_weight,
            syndrome_mask,
            symplectic,
            **kwargs,  # type: ignore[arg-type]
        )
    ]


def _weights(errors: list[tuple[int, ...]], symplectic: bool) -> list[int]:
    repeat = 2 if symplectic else 1
    return [int(np.count_nonzero(np.reshape(error, (repeat, -1)).any(axis=0))) for error in errors]


@pytest.mark.parametrize(
    "matrix, symplectic",
    [
        (galois.GF2([[1, 1, 0], [0, 1, 1]]), False),
        (galois.GF(3)([[1, 2, 0], [0, 1, 1]]), False),
        (galois.GF2([[1, 0, 0, 1], [0, 1, 1, 0]]), True),
    ],
)
@pytest.mark.parametrize("max_weight", [0, 1, 2])
def test_exhaustive_enumeration(
    matrix: galois.FieldArray, symplectic: bool, max_weight: int
) -> None:
    """Exhaustive enumeration yields each admissible error once, heaviest first."""
    enumerated = _enumerate(matrix, max_weight, None, symplectic)
    errors = [error for error, _ in enumerated]
    assert len(errors) == len(set(errors))
    assert dict(enumerated) == _brute_force(matrix, max_weight, None, symplectic)
    weights = _weights(errors, symplectic)
    assert weights == sorted(weights, reverse=True)

    # post-selection drops errors that flip masked syndrome bits, and those bits themselves
    mask = np.array([True, False])
    masked = _enumerate(matrix, max_weight, mask, symplectic)
    assert dict(masked) == _brute_force(matrix, max_weight, mask, symplectic)
    assert all(len(syndrome) == 1 for _, syndrome in masked)


@pytest.mark.parametrize(
    "matrix, symplectic, error_channel, probability_cutoff",
    [
        (galois.GF2([[1, 1, 0, 1], [0, 1, 1, 1]]), False, [0.4, 0.1, 0.02, 0.3], 1e-2),
        (galois.GF(3)([[1, 2, 1], [0, 1, 2]]), False, [0.3, 0.05, 0.2], 1e-2),
        (galois.GF2([[1, 0, 1, 1], [0, 1, 1, 0]]), True, [0.2, 0.05, 0.3, 0.1], 5e-3),
        # a site that is never active, and one that always is
        (galois.GF2([[1, 1, 1], [0, 1, 1]]), False, [0.0, 1.0, 0.2], 1e-3),
        # every error has probability exactly equal to the (inclusive) cutoff
        (galois.GF2([[1, 1]]), False, [0.5, 0.5], 0.25),
    ],
)
def test_probability_cutoff_enumeration(
    matrix: galois.FieldArray,
    symplectic: bool,
    error_channel: list[float],
    probability_cutoff: float,
) -> None:
    """A probability cutoff yields exactly the admissible errors at or above the cutoff."""
    channel = np.asarray(error_channel)
    max_weight = matrix.shape[1]
    enumerated = _enumerate(
        matrix,
        max_weight,
        None,
        symplectic,
        error_channel=channel,
        probability_cutoff=probability_cutoff,
    )
    errors = [error for error, _ in enumerated]
    assert len(errors) == len(set(errors))
    assert dict(enumerated) == _brute_force(
        matrix, max_weight, None, symplectic, channel, probability_cutoff
    )
    weights = _weights(errors, symplectic)
    assert weights == sorted(weights, reverse=True)


def test_probability_cutoff_prunes_without_enumerating() -> None:
    """A cutoff avoids enumerating the vast number of unlikely errors."""
    error_channel = np.full(60, 1e-3)
    error_channel[:6] = 0.4
    errors = np.array(
        [
            error
            for error, _ in _iter_errors_and_syndromes(
                galois.GF2.Zeros((1, 60)),
                30,
                None,
                False,
                error_channel=error_channel,
                probability_cutoff=1e-3,
            )
        ]
    )
    assert len(errors) == 2**6
    assert not np.any(errors[:, 6:])
