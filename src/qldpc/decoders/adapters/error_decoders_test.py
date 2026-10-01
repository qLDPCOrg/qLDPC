# SPDX-License-Identifier: Apache-2.0

"""Tests for error-decoder adapters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.adapters import error_decoders


class _FixedDecoder(decoders.ErrorDecoder):
    def __init__(self, output: npt.ArrayLike, *, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output.copy()


def test_errors_to_observables_decoder() -> None:
    """Inferred errors are aligned with a DEM and projected onto its observables."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    decoder = error_decoders.ErrorsToObservablesDecoder(_FixedDecoder([1, 0]), dem)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [1, 0])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), [[1, 0], [1, 0]])

    erasing = error_decoders.ErrorsToObservablesDecoder(
        _FixedDecoder([0, 1, 1], has_erasure_bit=True), dem
    )
    assert erasing.has_erasure_bit
    assert np.array_equal(erasing.decode_observables(np.array([0, 1])), [0, 1, 1])


def test_expanded_error_decoder() -> None:
    """Merged errors are expanded to the original DEM width, preserving erasure bits."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    for add_erasure_bit in [False, True]:
        merging_decoder = decoders.GUFDecoder(
            decoders.DetectorErrorModelArrays(dem).detector_flip_matrix.toarray(),
            add_erasure_bit=add_erasure_bit,
        )
        decoder = error_decoders.ExpandedErrorDecoder(merging_decoder, dem)
        errors = decoder.decode_errors_batch(syndromes)
        assert errors.shape == (2, 3 + add_erasure_bit)
        assert np.array_equal(errors, [decoder.decode_errors(syndrome) for syndrome in syndromes])
        assert np.array_equal(errors[:, [0, 1]].sum(axis=1), [1, 0])
        assert decoder.decode_errors_batch(syndromes[:0]).shape == (0, 3 + add_erasure_bit)


def test_match_error_decoder_to_dem() -> None:
    """Output validation returns, expands, or rejects a decoder as appropriate."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    full_width = _FixedDecoder([0, 0, 0])
    assert error_decoders.match_error_decoder_to_dem(full_width, dem) is full_width

    merged = error_decoders.match_error_decoder_to_dem(_FixedDecoder([0, 0]), dem)
    assert isinstance(merged, error_decoders.ExpandedErrorDecoder)

    with pytest.raises(ValueError, match="inferred an error of length 1"):
        error_decoders.match_error_decoder_to_dem(_FixedDecoder([0]), dem)

    decomposed = _FixedDecoder([0, 0])
    decomposed._infers_decomposed_errors = True  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="components of decomposed error mechanisms"):
        error_decoders.match_error_decoder_to_dem(decomposed, dem)
