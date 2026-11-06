# SPDX-License-Identifier: Apache-2.0

"""Pauli operators on qubits and qudits."""

from __future__ import annotations

import dataclasses
import enum
from typing import Literal


class Pauli(enum.Enum):
    """Pauli operators."""

    I = (0, 0)
    Z = (0, 1)
    X = (1, 0)
    Y = (1, 1)

    def __mul__(self, other: Pauli) -> Pauli:
        """Product of two Pauli operators."""
        val_x = (self.value[0] + other.value[0]) % 2
        val_z = (self.value[1] + other.value[1]) % 2
        return Pauli((val_x, val_z))

    def __invert__(self) -> Pauli:
        """Hadamard-transform this Pauli operator."""
        return Pauli(self.value[::-1])

    def swap_xz(self) -> PauliXZ:
        """Convert between Pauli.X and Pauli.Z, promising a PauliXZ return type (or an error)."""
        if self is Pauli.X:
            return Pauli.Z
        if self is Pauli.Z:
            return Pauli.X
        raise ValueError(
            f"Pauli.swap_xz only converts between Pauli.X and Pauli.Z (provided: {self})"
        )

    def __str__(self) -> str:
        if self is Pauli.I:
            return "I"
        elif self is Pauli.Z:
            return "Z"
        elif self is Pauli.X:
            return "X"
        return "Y"

    @staticmethod
    def from_string(string: str) -> Pauli:
        """Build a Pauli operator from a string."""
        if string == "I":
            return Pauli.I
        elif string == "Z":
            return Pauli.Z
        elif string == "X":
            return Pauli.X
        elif string == "Y":
            return Pauli.Y
        raise ValueError(f"Invalid Pauli operator: {string}")

    @staticmethod
    def coerce_xz(value: PauliXZLike) -> PauliXZ:
        """Convert Pauli.X, Pauli.Z, or a case-insensitive "X" or "Z" string into a PauliXZ.

        Raises:
            ValueError: If the value does not identify Pauli.X or Pauli.Z.
        """
        if isinstance(value, str):
            if value.upper() == "X":
                return Pauli.X
            if value.upper() == "Z":
                return Pauli.Z
        elif value is Pauli.X or value is Pauli.Z:
            return value
        raise ValueError(
            f"Expected Pauli.X or Pauli.Z, or a case-insensitive 'X' or 'Z' string (got {value!r})"
        )

    @property
    def index(self) -> int:
        """Numerical index for Pauli operators."""
        return self.__index__()

    def __index__(self) -> int:
        """Allow indexing arrays with Pauli operators."""
        if self == Pauli.X:
            return 0
        if self == Pauli.Z:
            return 1
        raise TypeError(f"Only Pauli.X and Pauli.Z have an index (got {self})")


PauliXZ = Literal[Pauli.X, Pauli.Z]
PAULIS_XZ: list[PauliXZ] = [Pauli.X, Pauli.Z]

# inputs that Pauli.coerce_xz converts into Pauli.X or Pauli.Z
PauliXZLike = PauliXZ | Literal["X", "Z", "x", "z"]


@dataclasses.dataclass(frozen=True)
class QuditPauli:
    """A qudit operator of the form ``X(val_x)*Z(val_z)``."""

    value: tuple[int, int] = (0, 0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuditPauli) and self.value == other.value

    def __invert__(self) -> QuditPauli:
        """Fourier-transform this qudit operator."""
        return QuditPauli(self.value[::-1])

    def __neg__(self) -> QuditPauli:
        """Invert the shifts and phases on this qudit operator."""
        return QuditPauli((-self.value[0], -self.value[1]))

    def __str__(self) -> str:
        val_x, val_z = self.value
        if not val_x and not val_z:
            return "I"
        if val_x == val_z:
            return f"Y({val_z})"
        ops = []
        if val_x:
            ops.append(f"X({val_x})")
        if val_z:
            ops.append(f"Z({val_z})")
        return "*".join(ops)

    @staticmethod
    def from_string(string: str) -> QuditPauli:
        """Build a qudit operator from its string representation."""
        if string == "I":
            return QuditPauli((0, 0))

        invalid_op = f"Invalid qudit operator: {string}"

        val_x, val_z = 0, 0
        factors = string.split("*")
        if len(factors) > 2:
            raise ValueError(invalid_op)

        for factor in factors:
            pauli = factor[:1]
            val_str = factor[2:-1]
            _factor = f"{pauli}({val_str})"
            if pauli not in "XYZ" or not val_str.isdecimal() or factor != _factor:
                raise ValueError(invalid_op)

            val = int(val_str)
            if pauli == "X":
                val_x = val
            elif pauli == "Z":
                val_z = val
            else:  # pauli == "Y"
                val_x = val_z = val

        return QuditPauli((val_x, val_z))
