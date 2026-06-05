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


def test_observable_lookup_decoding() -> None:
    """Lookup decoding can identify the most likely observable flip for each syndrome."""
    obs_matrix: math.IntegerArray

    # toy detector error model and error syndrome
    dem = stim.DetectorErrorModel("""
        error(0.10) D0
        error(0.09) D0 L0
        error(0.06) D0 L0
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=False)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    syndrome = np.array([1], dtype=int)

    # given only the parity check matrix, a LookupDecoder will return the most likely error
    decoder = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_probs)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [0])

    # provided a DEM, the LookupDecoder will simplify and predict the most likely observable flip
    decoder = decoders.LookupDecoder(dem, max_weight=1)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])

    # The above example is "trivial" in the sense that simplifying the DEM is sufficient to predict
    # the correct observable flips....
    dem_arrays = decoders.DetectorErrorModelArrays(dem, simplify=True)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    decoder = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_probs)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])

    # However, sometimes simplifying is not enough.  Consider th following DEM, in which each error
    # has a unique (detector, observable) patterns, so simplifying changes nothing:
    dem = stim.DetectorErrorModel("""
        error(0.04) D0 D1  # E0: syndrome (1, 1), obs_flip=0
        error(0.25) D0     # E1: syndrome (1, 0), obs_flip=0
        error(0.10) D1     # E2: syndrome (0, 1), obs_flip=0
        error(0.10) D0 L0  # E3: syndrome (1, 0), obs_flip=1
        error(0.25) D1 L0  # E4: syndrome (0, 1), obs_flip=1
    """)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    pcm, obs_matrix, error_probs = dem_arrays.get_arrays()
    syndrome = np.array([1, 1], dtype=int)

    # without knowing about observables, the most likely error is E0, with obs_flip=0
    decoder = decoders.LookupDecoder(pcm, max_weight=2)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [0])

    # however, it is more likely that either (E1 + E4) XOR (E2 + E3) occurred, which have obs_flip=1
    decoder = decoders.LookupDecoder(dem, max_weight=2)
    assert np.array_equal(obs_matrix @ decoder.decode(syndrome), [1])


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
    local_errors = tuple(itertools.product(code.field.elements, repeat=2))[1:]
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
