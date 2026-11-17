# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the logical operators of one Pauli type of a binary CSS code."""

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from ._sectors import get_sector


def test_get_sector_validation() -> None:
    """Only binary CSS codes and the two kinds of logical operators are supported."""
    with pytest.raises(TypeError, match="CSS code"):
        get_sector(codes.FiveQubitCode(), "X", "dressed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="binary"):
        get_sector(codes.CSSCode([[1, 1]], [[1, 2]], field=3), "X", "dressed")
    with pytest.raises(ValueError, match="'dressed' or 'bare'"):
        get_sector(codes.SteaneCode(), "X", "gauge")  # type: ignore[arg-type]


@pytest.mark.parametrize("pauli", [Pauli.X, Pauli.Z])
def test_sectors(
    pauli: PauliXZ, stabilizer_code: codes.CSSCode, subsystem_code: codes.CSSCode
) -> None:
    """Each sector has k independent classes, and dressed operators extend bare ones."""
    for code in [stabilizer_code, subsystem_code]:
        dressed = get_sector(code, pauli, "dressed")
        bare = get_sector(code, pauli, "bare")
        for sector in [dressed, bare]:
            assert (sector.num_qubits, sector.dimension) == (len(code), code.dimension)
            space = sector.get_logical_space()
            assert not np.any(sector.checks.astype(int) @ space.T.astype(int) % 2)
            assert _rank(space) == _rank(sector.trivial) + code.dimension
            assert _rank(np.vstack([space, sector.trivial])) == _rank(space)
            reference_ops = code.get_logical_ops(pauli).view(np.ndarray)
            np.testing.assert_array_equal(sector.get_classes(reference_ops), np.eye(code.dimension))
            assert not np.any(sector.get_classes(sector.trivial))

        # Every bare operator is a dressed operator, and every stabilizer is a gauge operator.
        assert _rank(np.vstack([dressed.get_logical_space(), bare.get_logical_space()])) == _rank(
            dressed.get_logical_space()
        )
        assert _rank(np.vstack([dressed.trivial, bare.trivial])) == _rank(dressed.trivial)
        if code.is_subsystem_code:
            assert _rank(bare.get_logical_space()) < _rank(dressed.get_logical_space())
        else:
            assert _rank(bare.get_logical_space()) == _rank(dressed.get_logical_space())


def _rank(matrix: npt.NDArray[np.uint8]) -> int:
    """The rank of a binary matrix."""
    return int(np.linalg.matrix_rank(galois.GF2(matrix))) if matrix.size else 0
