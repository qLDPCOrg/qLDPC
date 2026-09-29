"""Internal algebraic seam for CSS surgery constructions."""

from __future__ import annotations

import dataclasses
from typing import Any

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ


def _binary_matrix(name: str, matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a copied binary matrix."""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must be binary")
    return array.astype(np.uint8, copy=True)


def _rank(matrix: np.ndarray[Any, Any]) -> int:
    """Compute matrix rank over GF(2)."""
    return int(np.linalg.matrix_rank(galois.GF2(np.asarray(matrix, dtype=np.uint8))))


@dataclasses.dataclass(frozen=True)
class _CSSConeRegions:
    """Qubit and check-row regions of an assembled CSS cone."""

    basis: PauliXZ
    data_qubits: tuple[int, ...]
    ancilla_qubits: tuple[int, ...]
    data_checks_x: tuple[int, ...]
    interface_checks_x: tuple[int, ...]
    data_checks_z: tuple[int, ...]
    interface_checks_z: tuple[int, ...]

    @property
    def measurement_checks(self) -> tuple[int, ...]:
        """Rows of the merged check matrix whose products encode measured outcomes."""
        return self.interface_checks_x if self.basis is Pauli.X else self.interface_checks_z

    @property
    def complementary_checks(self) -> tuple[int, ...]:
        """New complementary-basis check rows of the cone."""
        return self.interface_checks_z if self.basis is Pauli.X else self.interface_checks_x


@dataclasses.dataclass(frozen=True)
class _CSSConeResult:
    """Validated merged code and its construction provenance."""

    code: CSSCode
    maps: _CSSConeMaps
    regions: _CSSConeRegions
    expected_logical_loss: int


@dataclasses.dataclass(frozen=True)
class _CSSConeMaps:
    """Four-map CSS cone in a measurement-basis-neutral convention.

    ``measurement_to_data`` and ``measurement_boundary`` form the new checks in the measured
    basis. ``complement_from_data`` extends the original complementary checks onto the ancillas,
    and ``complement_boundary`` forms new ancilla-only complementary checks.
    """

    basis: PauliXZ
    measurement_to_data: np.ndarray[Any, Any]
    measurement_boundary: np.ndarray[Any, Any]
    complement_from_data: np.ndarray[Any, Any]
    complement_boundary: np.ndarray[Any, Any]
    measurement_groups: np.ndarray[Any, Any]

    def _validated_arrays(
        self, data_code: CSSCode
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        """Validate and return normalized construction blocks."""
        if self.basis not in (Pauli.X, Pauli.Z):
            raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {self.basis!r}")
        measurement_to_data = _binary_matrix("measurement_to_data", self.measurement_to_data)
        measurement_boundary = _binary_matrix("measurement_boundary", self.measurement_boundary)
        complement_from_data = _binary_matrix("complement_from_data", self.complement_from_data)
        complement_boundary = _binary_matrix("complement_boundary", self.complement_boundary)
        measurement_groups = _binary_matrix("measurement_groups", self.measurement_groups)

        num_measurements, num_ancillas = measurement_boundary.shape
        complement_matrix = (
            np.asarray(data_code.matrix_z, dtype=np.uint8)
            if self.basis is Pauli.X
            else np.asarray(data_code.matrix_x, dtype=np.uint8)
        )
        if measurement_to_data.shape != (num_measurements, data_code.num_qudits):
            raise ValueError(
                "measurement_to_data shape is incompatible with measurement_boundary and the "
                "data code"
            )
        if complement_from_data.shape != (complement_matrix.shape[0], num_ancillas):
            raise ValueError(
                "complement_from_data shape is incompatible with the complementary data checks "
                "and ancillas"
            )
        if complement_boundary.shape[1] != num_ancillas:
            raise ValueError("complement_boundary must address every ancilla column")
        if measurement_groups.shape[1] != num_measurements:
            raise ValueError("measurement_groups must select the new measurement checks")

        # check that the cone maps produce commuting CSS checks before building the code
        commuting_square = (
            measurement_to_data @ complement_matrix.T
            + measurement_boundary @ complement_from_data.T
        ) % 2
        if np.any(commuting_square):
            raise ValueError("the four construction maps do not form a commuting square")
        if np.any(measurement_boundary @ complement_boundary.T % 2):
            raise ValueError(
                "measurement_boundary and complement_boundary do not form a chain complex"
            )
        if np.any(measurement_groups @ measurement_boundary % 2):
            raise ValueError("measurement groups do not cancel their ancilla support")
        return (
            measurement_to_data,
            measurement_boundary,
            complement_from_data,
            complement_boundary,
            measurement_groups,
        )

    def measured_operators(self, data_code: CSSCode) -> np.ndarray[Any, Any]:
        """Data-qubit supports measured by the independent check groups."""
        measurement_to_data, _, _, _, measurement_groups = self._validated_arrays(data_code)
        return np.asarray(measurement_groups @ measurement_to_data % 2, dtype=np.uint8)

    def expected_logical_loss(self, data_code: CSSCode) -> int:
        """Rank of the measured space modulo same-basis data stabilizers."""
        measured = self.measured_operators(data_code)
        same_basis = (
            np.asarray(data_code.matrix_x, dtype=np.uint8)
            if self.basis is Pauli.X
            else np.asarray(data_code.matrix_z, dtype=np.uint8)
        )
        return _rank(np.vstack([same_basis, measured])) - _rank(same_basis)

    def measures_exact_span(self, data_code: CSSCode, requested: np.ndarray[Any, Any]) -> bool:
        """Whether measured and requested operators span the same logical cosets."""
        requested_array = _binary_matrix("requested", requested)
        if requested_array.shape[1] != data_code.num_qudits:
            raise ValueError("requested operators must address the data qubits")
        measured = self.measured_operators(data_code)
        same_basis = (
            np.asarray(data_code.matrix_x, dtype=np.uint8)
            if self.basis is Pauli.X
            else np.asarray(data_code.matrix_z, dtype=np.uint8)
        )
        rank_measured = _rank(np.vstack([same_basis, measured]))
        rank_requested = _rank(np.vstack([same_basis, requested_array]))
        rank_union = _rank(np.vstack([same_basis, measured, requested_array]))
        return rank_measured == rank_requested == rank_union

    def build(self, data_code: CSSCode) -> _CSSConeResult:
        """Assemble and validate the merged CSS code and its row regions."""
        (
            measurement_to_data,
            measurement_boundary,
            complement_from_data,
            complement_boundary,
            _,
        ) = self._validated_arrays(data_code)
        matrix_x = np.asarray(data_code.matrix_x, dtype=np.uint8)
        matrix_z = np.asarray(data_code.matrix_z, dtype=np.uint8)
        num_ancillas = measurement_boundary.shape[1]
        zero_x = np.zeros((matrix_x.shape[0], num_ancillas), dtype=np.uint8)
        zero_z = np.zeros((matrix_z.shape[0], num_ancillas), dtype=np.uint8)
        zero_comp = np.zeros((complement_boundary.shape[0], data_code.num_qudits), dtype=np.uint8)

        if self.basis is Pauli.X:
            merged_x = np.block([[matrix_x, zero_x], [measurement_to_data, measurement_boundary]])
            merged_z = np.block(
                [[matrix_z, complement_from_data], [zero_comp, complement_boundary]]
            )
            num_interface_x = measurement_to_data.shape[0]
            num_interface_z = complement_boundary.shape[0]
        else:
            merged_x = np.block(
                [[matrix_x, complement_from_data], [zero_comp, complement_boundary]]
            )
            merged_z = np.block([[matrix_z, zero_z], [measurement_to_data, measurement_boundary]])
            num_interface_x = complement_boundary.shape[0]
            num_interface_z = measurement_to_data.shape[0]

        merged_code = CSSCode(
            merged_x.astype(np.int_),
            merged_z.astype(np.int_),
            is_subsystem_code=False,
        )
        regions = _CSSConeRegions(
            basis=self.basis,
            data_qubits=tuple(range(data_code.num_qudits)),
            ancilla_qubits=tuple(range(data_code.num_qudits, data_code.num_qudits + num_ancillas)),
            data_checks_x=tuple(range(matrix_x.shape[0])),
            interface_checks_x=tuple(range(matrix_x.shape[0], matrix_x.shape[0] + num_interface_x)),
            data_checks_z=tuple(range(matrix_z.shape[0])),
            interface_checks_z=tuple(range(matrix_z.shape[0], matrix_z.shape[0] + num_interface_z)),
        )
        return _CSSConeResult(
            code=merged_code,
            maps=self,
            regions=regions,
            expected_logical_loss=self.expected_logical_loss(data_code),
        )


def _validate_logical_loss(
    merged_code: CSSCode,
    *,
    input_dimension: int,
    expected_logical_loss: int,
    operation: str,
) -> None:
    """Require the merged-code dimension to match the measured-space rank."""
    if expected_logical_loss < 1:
        raise ValueError(
            f"{operation} must request at least one independent logical measurement, got "
            f"{expected_logical_loss}."
        )
    logical_loss = input_dimension - merged_code.dimension
    if logical_loss == expected_logical_loss:
        return
    expected = (
        "one logical degree of freedom"
        if expected_logical_loss == 1
        else f"{expected_logical_loss} logical degrees of freedom"
    )
    guidance = (
        "A reducible logical support can fix its factors separately; boost or replace the gadget "
        "before compiling the circuit."
        if expected_logical_loss == 1
        else "The construction's measurement-group rank and merged-code dimension disagree."
    )
    raise ValueError(
        f"{operation} must fix exactly {expected}, but the input encodes {input_dimension} and the "
        f"merged code encodes {merged_code.dimension} (logical loss {logical_loss}). {guidance}"
    )
