# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit custom decoder factories."""

from __future__ import annotations

import dataclasses
import functools
import pickle

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.construction.factories import (
    DEMDecoderFactory,
    MatrixDecoderFactory,
    from_dem,
    from_matrix,
)
from qldpc.decoders.custom.lookup import LookupDecoder, ObservableLookupDecoder


def test_factories_pass_their_input_to_the_wrapped_factory() -> None:
    """A factory builds by calling the wrapped callable on exactly the input that it is given."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]])
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1")
    seen: list[object] = []

    def build_errors(matrix: npt.NDArray[np.int_]) -> LookupDecoder:
        seen.append(matrix)
        return LookupDecoder(matrix, max_weight=1)

    def build_observables(dem: stim.DetectorErrorModel) -> ObservableLookupDecoder:
        seen.append(dem)
        return ObservableLookupDecoder(dem, max_weight=1)

    matrix_factory = from_matrix(build_errors)
    dem_factory = from_dem(build_observables)
    assert isinstance(matrix_factory, MatrixDecoderFactory)
    assert isinstance(dem_factory, DEMDecoderFactory)
    assert matrix_factory.factory is build_errors
    assert dem_factory.factory is build_observables

    error_decoder = matrix_factory.build(matrix)
    assert isinstance(error_decoder, LookupDecoder)
    assert np.array_equal(matrix @ error_decoder.decode_errors(np.array([1, 0])) % 2, [1, 0])
    observable_decoder = dem_factory.build(dem)
    assert isinstance(observable_decoder, ObservableLookupDecoder)
    assert observable_decoder.decode_observables(np.array([1, 0])).tolist() == [1]
    assert seen == [matrix, dem]

    # the public helpers and types are the same objects
    assert decoders.from_matrix is from_matrix and decoders.from_dem is from_dem
    assert decoders.MatrixDecoderFactory is MatrixDecoderFactory
    assert decoders.DEMDecoderFactory is DEMDecoderFactory


def test_factories_validate_and_are_immutable() -> None:
    """Factories require a callable and cannot be reassigned after construction."""
    with pytest.raises(TypeError, match="from_matrix requires a callable factory"):
        from_matrix(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="from_dem requires a callable factory"):
        from_dem("lookup")  # type: ignore[arg-type]

    factory = from_matrix(functools.partial(LookupDecoder, max_weight=1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        factory.factory = LookupDecoder  # type: ignore[misc]


def test_factories_pickle() -> None:
    """Factories of importable callables survive pickling, as Sinter worker processes require."""
    matrix = np.array([[1, 1]])
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    matrix_factory = from_matrix(functools.partial(LookupDecoder, max_weight=1))
    dem_factory = from_dem(functools.partial(ObservableLookupDecoder, max_weight=1))

    restored_matrix_factory = pickle.loads(pickle.dumps(matrix_factory))  # noqa: S301
    restored_dem_factory = pickle.loads(pickle.dumps(dem_factory))  # noqa: S301
    assert isinstance(restored_matrix_factory, MatrixDecoderFactory)
    assert isinstance(restored_dem_factory, DEMDecoderFactory)
    assert restored_matrix_factory.build(matrix).decode(np.array([1])).shape == (2,)
    assert restored_dem_factory.build(dem).decode_observables(np.array([1])).tolist() == [1]
