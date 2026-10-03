# SPDX-License-Identifier: Apache-2.0

"""Explicit input contracts for custom decoder factories."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import stim

from qldpc.math import IntegerArray

from ..protocols import ErrorDecoder, ObservableDecoder, SupportsDecode


@dataclasses.dataclass(frozen=True, slots=True)
class _MatrixDecoderFactory:
    factory: Callable[..., ErrorDecoder | SupportsDecode]

    def build(self, matrix: IntegerArray) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder for a parity-check matrix."""
        return self.factory(matrix)


@dataclasses.dataclass(frozen=True, slots=True)
class _DEMDecoderFactory:
    factory: Callable[..., ObservableDecoder]

    def build(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build an observable decoder for a detector error model."""
        return self.factory(dem)


def from_matrix(factory: Callable[..., ErrorDecoder | SupportsDecode]) -> _MatrixDecoderFactory:
    """Declare that a custom factory builds an error decoder from a parity-check matrix.

    Pass the result as ``decoder=``, ``decoder_x=``, or ``decoder_z=`` where an error decoder is
    accepted.  A detector error model is not converted into a matrix for this factory.
    """
    if not callable(factory):
        raise TypeError("from_matrix requires a callable factory")
    return _MatrixDecoderFactory(factory)


def from_dem(factory: Callable[..., ObservableDecoder]) -> _DEMDecoderFactory:
    """Declare that a custom factory builds an observable decoder from a binary Stim DEM.

    Pass the result where observable predictions are used, such as to a code-capacity estimator
    or a SinterDecoder.  A code-capacity estimator builds a DEM for the code's checks and logicals;
    it rejects this factory for nonbinary codes.
    """
    if not callable(factory):
        raise TypeError("from_dem requires a callable factory")
    return _DEMDecoderFactory(factory)
