"""Tests for the internal CSS surgery construction seam."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qldpc import codes
from qldpc.experimental.surgery.construction import _CSSConeMaps
from qldpc.objects import Pauli, PauliXZ


def _cone_maps_for_logical(
    code: codes.CSSCode, logical: np.ndarray[Any, Any], basis: PauliXZ
) -> _CSSConeMaps:
    """Build current Webster maps without using its assembly function."""
    from qldpc.experimental.surgery.gadget import (
        _compute_gauge_basis,
        _restrict_checks_to_support,
    )

    support, data_checks, incidence = _restrict_checks_to_support(code, logical, basis=basis)
    gauge = _compute_gauge_basis(incidence)
    measurement_to_data = np.zeros((len(support), len(code)), dtype=np.uint8)
    measurement_to_data[np.arange(len(support)), np.asarray(support)] = 1
    num_complement = code.matrix_z.shape[0] if basis is Pauli.X else code.matrix_x.shape[0]
    complement_from_data = np.zeros((num_complement, incidence.shape[0]), dtype=np.uint8)
    for column, check in enumerate(data_checks):
        complement_from_data[check, column] = 1
    return _CSSConeMaps(
        basis=basis,
        measurement_to_data=measurement_to_data,
        measurement_boundary=incidence.T,
        complement_from_data=complement_from_data,
        complement_boundary=gauge,
        measurement_groups=np.ones((1, len(support)), dtype=np.uint8),
    )


def _steane_cone_maps(
    basis: PauliXZ,
) -> tuple[codes.CSSCode, _CSSConeMaps, np.ndarray[Any, Any]]:
    """Build the current Steane Webster construction as private cone maps."""
    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(basis)[0], dtype=np.uint8)
    maps = _cone_maps_for_logical(code, logical, basis)
    return code, maps, logical.reshape(1, -1)


def _block_diagonal(*matrices: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Block-diagonalize binary matrices."""
    output = np.zeros(
        (sum(matrix.shape[0] for matrix in matrices), sum(matrix.shape[1] for matrix in matrices)),
        dtype=np.uint8,
    )
    row = 0
    column = 0
    for matrix in matrices:
        height, width = matrix.shape
        output[row : row + height, column : column + width] = matrix
        row += height
        column += width
    return output


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_css_cone_maps_build_current_webster_layout(basis: PauliXZ) -> None:
    """The private seam assembles both CSS-dual orientations and retains regions."""
    code, maps_object, logical = _steane_cone_maps(basis)
    result = maps_object.build(code)

    assert result.code.dimension == 0
    assert result.expected_logical_loss == 1
    assert maps_object.measures_exact_span(code, logical)
    assert result.regions.data_qubits == tuple(range(7))
    assert len(result.regions.ancilla_qubits) == maps_object.measurement_boundary.shape[1]
    assert len(result.regions.measurement_checks) == maps_object.measurement_to_data.shape[0]
    assert len(result.regions.complementary_checks) == maps_object.complement_boundary.shape[0]
    assert result.regions.basis is basis


def test_css_cone_maps_reject_invalid_basis() -> None:
    """The cone seam remains CSS X/Z-only."""
    from qldpc.experimental.surgery.construction import _CSSConeMaps

    code, maps, _ = _steane_cone_maps(Pauli.X)
    invalid = _CSSConeMaps(
        basis=Pauli.Y,  # type: ignore[arg-type]
        measurement_to_data=maps.measurement_to_data,
        measurement_boundary=maps.measurement_boundary,
        complement_from_data=maps.complement_from_data,
        complement_boundary=maps.complement_boundary,
        measurement_groups=maps.measurement_groups,
    )
    with pytest.raises(ValueError, match="basis"):
        invalid.build(code)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measurement_to_data", np.zeros(3, dtype=np.uint8), "two-dimensional"),
        (
            "measurement_to_data",
            np.zeros((2, 7), dtype=np.uint8),
            "measurement_to_data shape",
        ),
        ("measurement_boundary", np.array([[2]], dtype=np.uint8), "binary"),
        (
            "complement_from_data",
            np.zeros((2, 3), dtype=np.uint8),
            "complement_from_data shape",
        ),
        (
            "complement_boundary",
            np.zeros((1, 2), dtype=np.uint8),
            "address every ancilla",
        ),
        (
            "measurement_groups",
            np.zeros((1, 2), dtype=np.uint8),
            "select the new measurement checks",
        ),
    ],
)
def test_css_cone_maps_reject_invalid_blocks(
    field: str, value: np.ndarray[Any, Any], message: str
) -> None:
    """Every construction block is shape- and field-checked before assembly."""
    code, maps, _ = _steane_cone_maps(Pauli.X)
    invalid = _CSSConeMaps(
        basis=maps.basis,
        measurement_to_data=value if field == "measurement_to_data" else maps.measurement_to_data,
        measurement_boundary=(
            value if field == "measurement_boundary" else maps.measurement_boundary
        ),
        complement_from_data=(
            value if field == "complement_from_data" else maps.complement_from_data
        ),
        complement_boundary=(value if field == "complement_boundary" else maps.complement_boundary),
        measurement_groups=value if field == "measurement_groups" else maps.measurement_groups,
    )
    with pytest.raises(ValueError, match=message):
        invalid.build(code)


def test_css_cone_maps_reject_noncommuting_square() -> None:
    """A local block edit cannot silently produce anticommuting checks."""
    import dataclasses

    code, maps, _ = _steane_cone_maps(Pauli.X)
    bad = maps.measurement_to_data.copy()
    bad[0, 0] ^= 1
    with pytest.raises(ValueError, match="commuting square"):
        dataclasses.replace(maps, measurement_to_data=bad).build(code)


def test_css_cone_maps_reject_invalid_complement_chain() -> None:
    """Ancilla-only complementary checks must commute with measurement checks."""
    import dataclasses

    code, maps, _ = _steane_cone_maps(Pauli.X)
    bad = maps.complement_boundary.copy()
    bad[0, 0] ^= 1
    with pytest.raises(ValueError, match="chain complex"):
        dataclasses.replace(maps, complement_boundary=bad).build(code)


def test_css_cone_maps_reject_open_measurement_group() -> None:
    """An outcome group must cancel every ancilla contribution."""
    import dataclasses

    code, maps, _ = _steane_cone_maps(Pauli.X)
    bad = np.eye(maps.measurement_boundary.shape[0], dtype=np.uint8)[:1]
    with pytest.raises(ValueError, match="cancel"):
        dataclasses.replace(maps, measurement_groups=bad).build(code)


def test_css_cone_maps_reject_wrong_requested_width() -> None:
    """Requested logical supports must address the data code."""
    code, maps, _ = _steane_cone_maps(Pauli.X)

    with pytest.raises(ValueError, match="address the data"):
        maps.measures_exact_span(code, np.zeros((1, len(code) + 1), dtype=np.uint8))


def test_css_cone_maps_support_two_independent_outcomes() -> None:
    """The internal seam is rank-aware even though public builders remain single-outcome."""
    component = codes.SteaneCode()
    code = codes.CSSCode.stack([component, component])
    logical = np.asarray(component.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    zeros = np.zeros(len(component), dtype=np.uint8)
    requested = np.vstack([np.hstack([logical, zeros]), np.hstack([zeros, logical])])
    left = _cone_maps_for_logical(code, requested[0], Pauli.X)
    right = _cone_maps_for_logical(code, requested[1], Pauli.X)
    maps = _CSSConeMaps(
        basis=Pauli.X,
        measurement_to_data=np.vstack([left.measurement_to_data, right.measurement_to_data]),
        measurement_boundary=_block_diagonal(left.measurement_boundary, right.measurement_boundary),
        complement_from_data=np.hstack([left.complement_from_data, right.complement_from_data]),
        complement_boundary=_block_diagonal(left.complement_boundary, right.complement_boundary),
        measurement_groups=_block_diagonal(left.measurement_groups, right.measurement_groups),
    )

    result = maps.build(code)
    assert result.expected_logical_loss == 2
    assert maps.measures_exact_span(code, requested)
    assert result.code.dimension == code.dimension - 2


@pytest.mark.parametrize(
    ("input_dimension", "merged_dimension", "expected", "message"),
    [
        (2, 0, 2, None),
        (2, 1, 2, "exactly 2"),
        (2, 0, 1, "exactly one"),
        (2, 1, 0, "at least one"),
    ],
)
def test_validate_logical_loss(
    input_dimension: int,
    merged_dimension: int,
    expected: int,
    message: str | None,
) -> None:
    """Rank-aware validation supports future multi-outcome constructions."""
    from qldpc.experimental.surgery.construction import _validate_logical_loss

    merged = codes.TrivialCode(merged_dimension)
    if message is None:
        _validate_logical_loss(
            merged,
            input_dimension=input_dimension,
            expected_logical_loss=expected,
            operation="test",
        )
    else:
        with pytest.raises(ValueError, match=message):
            _validate_logical_loss(
                merged,
                input_dimension=input_dimension,
                expected_logical_loss=expected,
                operation="test",
            )
