# SPDX-License-Identifier: Apache-2.0

"""Explicit input contracts for custom decoder factories."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import stim

from qldpc.math import IntegerArray

from ..protocols import ErrorDecoder, ObservableDecoder, SupportsDecode


@dataclasses.dataclass(frozen=True, slots=True)
class MatrixDecoderFactory:
    """A custom factory that builds an error decoder from a parity-check matrix.

    Create one with ``decoders.from_matrix(factory)``, and pass it as ``decoder=``.  A workflow that
    only has a detector error model rejects it.
    """

    factory: Callable[..., ErrorDecoder | SupportsDecode]

    def build(self, matrix: IntegerArray) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder for a parity-check matrix."""
        return self.factory(matrix)


@dataclasses.dataclass(frozen=True, slots=True)
class DEMDecoderFactory:
    """A custom factory that builds an observable decoder from a binary detector error model.

    Create one with ``decoders.from_dem(factory)``, and pass it as ``decoder=``.  A workflow that
    needs inferred errors, such as a window decoder, rejects it.
    """

    factory: Callable[..., ObservableDecoder]

    def build(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build an observable decoder for a detector error model."""
        return self.factory(dem)


def from_matrix(factory: Callable[..., ErrorDecoder | SupportsDecode]) -> MatrixDecoderFactory:
    """Declare a parity-check matrix input for a custom error-decoder factory."""
    if not callable(factory):
        raise TypeError("from_matrix requires a callable factory")
    return MatrixDecoderFactory(factory)


def from_dem(factory: Callable[..., ObservableDecoder]) -> DEMDecoderFactory:
    """Declare a binary detector error model input for a custom observable-decoder factory."""
    if not callable(factory):
        raise TypeError("from_dem requires a callable factory")
    return DEMDecoderFactory(factory)
