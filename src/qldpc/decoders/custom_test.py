"""Unit tests for custom.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
import itertools
import random
import unittest.mock

import galois
import numpy as np
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders, math


@functools.cache
def get_toy_problem() -> tuple[galois.FieldArray, galois.FieldArray, galois.FieldArray]:
    """Get a toy decoding problem."""
    field = galois.GF(2)
    matrix = np.eye(3, 2, dtype=int).view(field)
    error = np.array([1, 1], dtype=int).view(field)
    syndrome = matrix @ error
    return matrix, error, syndrome


def test_relay_bp() -> None:
    """The Relay-BP decoder needs a custom wrapper class."""
    matrix, error, syndrome = get_toy_problem()
    errors = np.array([error, error])
    syndromes = np.array([syndrome, syndrome])

    decoder = decoders.get_decoder_RBP(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert np.array_equal(errors, decoder.decode_batch(syndromes))

    # decode from a sparse parity check matrix
    decoder = decoders.get_decoder_RBP(scipy.sparse.dok_matrix(matrix))
    assert np.array_equal(error, decoder.decode_detailed(syndrome).decoding)

    # decode from a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    decoder = decoders.get_decoder_RBP(dem)
    assert np.array_equal(error, decoder.decode(syndrome))

    # fail to initialize a relay-bp decoder because relay-bp is not installed
    with (
        unittest.mock.patch.dict("sys.modules", {"relay_bp": None}),
        pytest.raises(ImportError, match="Failed to import relay-bp"),
    ):
        decoders.get_decoder(np.array([[]]), with_RBP=True)

    # fail to initialize a relay-bp decoder from an unrecognized name
    with pytest.raises(ValueError, match="name not recognized"):
        decoders.get_decoder(np.array([[]]), with_RBP=True, name="invalid_name")

    # fail when a decoder name string is passed where the matrix should be
    with pytest.raises(ValueError, match="breaking change"):
        decoders.RelayBPDecoder("MinSumBPDecoderF32")

    # passing explicit error_priors alongside a DEM emits a warning
    with pytest.warns(UserWarning, match="will override"):
        decoders.RelayBPDecoder(dem, error_priors=[0.1, 0.1])


def test_lookup() -> None:
    """Lookup decoding should be straightforward."""
    matrix, error, syndrome = get_toy_problem()

    decoder = decoders.get_decoder_lookup(matrix, max_weight=2)
    assert np.array_equal(error, decoder.decode(syndrome))

    # decode with a detector error model
    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    decoder = decoders.get_decoder_lookup(dem, max_weight=2)
    assert np.array_equal(error, decoder.decode(syndrome))


def test_lookup_observable() -> None:
    """Lookup decoding that targets the most likely observable flip for each syndrome.

    The decoder groups errors by their observable flip value, sums probabilities within each group,
    and returns an error from the group with the highest total probability.  This gives a different
    answer than simply picking the single most likely error when degenerate errors (sharing the same
    observable flip) collectively outweigh a more probable individual error with a different flip.
    """

    def get_obs_flip(da: decoders.DetectorErrorModelArrays, error: np.ndarray) -> int:
        return int(np.asarray(da.observable_flip_matrix @ error, dtype=int).ravel()[0] % 2)

    # --- Part 1: explicit PCM ---
    # Three mechanisms all give syndrome (1,): p=0.10 (obs_flip=0), p=0.09 (obs_flip=1),
    # p=0.06 (obs_flip=1).  The decoder without observable awareness picks the most likely
    # individual error (p=0.10, obs_flip=0).  The observable-aware decoder sums per class:
    # obs_flip=0 total=0.10, obs_flip=1 total=0.09+0.06=0.15 → obs_flip=1 wins.
    pcm = np.array([[1, 1, 1]], dtype=int)
    obs_matrix = np.array([[0, 1, 1]], dtype=int)
    error_channel = np.array([0.10, 0.09, 0.06])
    syndrome = np.array([1], dtype=int)

    standard = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_channel)
    assert int((obs_matrix @ standard.decode(syndrome) % 2)[0]) == 0  # most likely error

    aware = decoders.LookupDecoder(
        pcm, max_weight=1, error_channel=error_channel, observable_flip_matrix=obs_matrix
    )
    assert int((obs_matrix @ aware.decode(syndrome) % 2)[0]) == 1  # most likely observable flip

    # --- Part 2: same scenario via DEM + simplify=False ---
    # simplify=False keeps the two "D0 L0" mechanisms separate, reproducing the three-mechanism PCM.
    dem = stim.DetectorErrorModel("""
        error(0.10) D0
        error(0.09) D0 L0
        error(0.06) D0 L0
    """)
    da = decoders.DetectorErrorModelArrays(dem, simplify=False)

    standard_dem = decoders.LookupDecoder(
        da.detector_flip_matrix, max_weight=1, error_channel=da.error_probs
    )
    assert get_obs_flip(da, standard_dem.decode(syndrome)) == 0  # same result as Part 1

    aware_dem = decoders.LookupDecoder(dem, max_weight=1, simplify=False)
    assert get_obs_flip(da, aware_dem.decode(syndrome)) == 1  # same result as Part 1

    # --- Part 3: same DEM with default simplify=True ---
    # simplify=True merges the two "D0 L0" mechanisms into one with combined p≈0.145 > 0.10,
    # so the merged mechanism is simply the most likely error; a standard decoder initialized
    # from the DEM already returns the correct obs_flip=1 without explicit observable tracking.
    da_simplified = decoders.DetectorErrorModelArrays(dem)
    decoder_simplified = decoders.LookupDecoder(dem, max_weight=1)
    assert get_obs_flip(da_simplified, decoder_simplified.decode(syndrome)) == 1

    # --- Part 4: max_weight=2 DEM where simplification is not enough ---
    # Five mechanisms, all with unique (detector, observable) patterns so simplify changes nothing:
    #   error(0.05) D0 D1   — M0: syndrome (1,1), obs_flip=0
    #   error(0.25) D0      — M1: syndrome (1,0), obs_flip=0
    #   error(0.10) D1      — M2: syndrome (0,1), obs_flip=0
    #   error(0.10) D0 L0   — M3: syndrome (1,0), obs_flip=1
    #   error(0.25) D1 L0   — M4: syndrome (0,1), obs_flip=1
    # For syndrome (1,1), the most likely individual combination is M1+M4 (P≈0.048, obs_flip=1).
    # But summing per class: obs_flip=0 = P(M0)+P(M1+M2)+P(M3+M4) ≈ 0.055
    #                         obs_flip=1 = P(M1+M4)+P(M3+M2)       ≈ 0.053
    # A decoder without observable awareness returns obs_flip=1; the observable-aware decoder
    # correctly returns obs_flip=0.
    dem4 = stim.DetectorErrorModel("""
        error(0.05) D0 D1
        error(0.25) D0
        error(0.10) D1
        error(0.10) D0 L0
        error(0.25) D1 L0
    """)
    da4 = decoders.DetectorErrorModelArrays(dem4)
    syndrome4 = np.array([1, 1], dtype=int)

    standard4 = decoders.LookupDecoder(
        da4.detector_flip_matrix, max_weight=2, error_channel=da4.error_probs
    )
    assert get_obs_flip(da4, standard4.decode(syndrome4)) == 1  # most likely individual combination

    aware4 = decoders.LookupDecoder(dem4, max_weight=2)
    assert get_obs_flip(da4, aware4.decode(syndrome4)) == 0  # most likely observable flip

    # Providing an observable_flip_matrix without error probabilities raises an error.
    with pytest.raises(ValueError, match="stim.DetectorErrorModel, error_channel, or penalty_func"):
        decoders.LookupDecoder(pcm, max_weight=1, observable_flip_matrix=obs_matrix)

    # Providing both error_channel and penalty_func raises an error.
    with pytest.raises(ValueError, match="both an error_channel and a penalty_func"):
        decoders.LookupDecoder(
            pcm, max_weight=1, error_channel=error_channel, penalty_func=lambda v: 0.0
        )

    # Providing a DEM alongside conflicting arguments raises an error.
    with pytest.raises(ValueError, match="Cannot specify"):
        decoders.LookupDecoder(dem4, max_weight=1, error_channel=da4.error_probs)

    # The FieldArray branch: observable_flip_matrix is a galois FieldArray (quantum / non-binary use).
    field = galois.GF(2)
    pcm_gf2 = pcm.view(field)
    obs_gf2 = obs_matrix.view(field)
    aware_gf2 = decoders.LookupDecoder(
        pcm_gf2, max_weight=1, error_channel=error_channel, observable_flip_matrix=obs_gf2
    )
    assert int((obs_matrix @ aware_gf2.decode(syndrome) % 2)[0]) == 1


def test_ilp_decoder() -> None:
    """Decode using an integer linear program."""
    matrix, error, syndrome = get_toy_problem()
    decoder = decoders.ILPDecoder(scipy.sparse.csc_matrix(matrix))
    assert np.array_equal(error, decoder.decode(syndrome))

    # decode over the trinary field
    field = galois.GF(3)
    matrix = -matrix.view(field)
    error = -error.view(field)
    decoder = decoders.ILPDecoder(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))


def test_invalid_ilp() -> None:
    """Fail to solve an invalid integer linear programming problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="could not be found"):
        decoders.decode(matrix, syndrome, with_ILP=True)

    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        decoders.decode(galois.GF(4)(matrix), syndrome, with_ILP=True)


def test_generalized_union_find() -> None:
    """Generalized Union-Find."""
    base_code: codes.CSSCode = codes.C4Code()
    code = functools.reduce(codes.CSSCode.concatenate, [base_code] * 3)
    error = code.field.Zeros(len(code))
    error[[3, 4]] = 1
    matrix = code.matrix_z
    syndrome = matrix @ error
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True)) > 2
    assert np.count_nonzero(decoders.decode(matrix, syndrome, with_GUF=True, max_weight=2)) == 2

    # cover the trivial syndrome with the generalized Union-Find decoer
    assert np.array_equal(
        np.zeros_like(error), decoders.decode(matrix, np.zeros_like(syndrome), with_GUF=True)
    )


def test_augmented_decoders() -> None:
    """Composite and direct decoders, built from other decoders."""
    matrix, error, syndrome = get_toy_problem()
    decoder = decoders.get_decoder(matrix, with_MWPM=True)

    # decode corrupted code words directly
    direct_decoder = decoders.DirectDecoder.from_indirect(decoder, matrix)

    assert np.array_equal(np.zeros_like(error), direct_decoder.decode(error))

    errors = np.array([error] * 3)
    assert np.array_equal(np.zeros_like(errors), direct_decoder.decode_batch(errors))

    # decode composite syndromes
    composite_decoder = decoders.CompositeDecoder.from_copies(decoder, syndrome.size, 2)

    composite_error = np.concatenate([error] * 2)
    composite_syndrome = np.concatenate([syndrome] * 2)
    assert np.array_equal(composite_error, composite_decoder.decode(composite_syndrome))

    composite_errors = np.array([composite_error] * 3)
    composite_syndromes = np.array([composite_syndrome] * 3)
    assert np.array_equal(composite_errors, composite_decoder.decode_batch(composite_syndromes))


def test_quantum_decoding(pytestconfig: pytest.Config) -> None:
    """Decode random weight-2 errors in a GF(3) surface code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SurfaceCode(4, field=3)
    local_errors = tuple(itertools.product(range(code.field.order), repeat=2))[1:]
    qubit_a, qubit_b = np.random.choice(range(len(code)), size=2, replace=False)
    pauli_a, pauli_b = random.choices(local_errors, k=2)
    error = code.field.Zeros(2 * len(code))
    error[[qubit_a, qubit_a + len(code)]] = pauli_a
    error[[qubit_b, qubit_b + len(code)]] = pauli_b
    syndrome = code.matrix @ math.symplectic_conjugate(error)

    decoder: decoders.Decoder
    decoder = decoders.GUFDecoder(code.matrix, symplectic=True)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))

    decoder = decoders.LookupDecoder(code.matrix, symplectic=True, max_weight=2)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))

    decoder = decoders.LookupDecoder(
        code.matrix,
        symplectic=True,
        add_erasure_bit=True,
        max_weight=2,
        penalty_func=lambda vec: int(np.count_nonzero(vec)),
    )
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert decoded_error[-1] == 0
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error[:-1]))
    assert decoder.decode(np.ones_like(syndrome))[-1] == 1

    decoder = decoders.WeightedLookupDecoder(
        code.matrix, symplectic=True, add_erasure_bit=True, max_weight=2
    )
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert decoded_error[-1] == 0
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error[:-1]))
    assert decoder.decode(np.ones_like(syndrome))[-1] == 1


def test_penalty_func() -> None:
    """Lookup tables can build penalty functions that penalize unlikely errors."""
    error_channel = [0.2, 0.1]
    penalty_func = decoders.LookupDecoder.build_penalty_func(error_channel)
    assert penalty_func([0, 0]) < penalty_func([1, 0]) < penalty_func([0, 1]) < penalty_func([1, 1])
