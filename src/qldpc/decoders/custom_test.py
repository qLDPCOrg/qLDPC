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

    # passing an explicit penalty_func alongside a DEM emits a warning
    with pytest.warns(UserWarning, match="will override"):
        decoders.LookupDecoder(dem, max_weight=2, penalty_func=lambda v: float(np.count_nonzero(v)))


def test_lookup_observable() -> None:
    """Observable-flip-aware lookup decoding (MAP obs_flip vs MAP error)."""
    # Toy problem: 1 detector, 3 error mechanisms.
    # Mechanism 0: flips detector only (obs_flip 0), p=0.10
    # Mechanism 1: flips detector and observable (obs_flip 1), p=0.09
    # Mechanism 2: flips detector and observable (obs_flip 1), p=0.06
    # For syndrome (1,), MAP error picks mechanism 0 (highest individual p),
    # but MAP obs_flip picks obs_flip=1 because P(F=1|S) = 0.09+0.06 = 0.15 > P(F=0|S) = 0.10.
    pcm = np.array([[1, 1, 1]], dtype=int)
    obs_matrix = np.array([[0, 1, 1]], dtype=int)
    error_channel = [0.10, 0.09, 0.06]
    syndrome = np.array([1], dtype=int)

    # MAP error decoder (no observable_flip_matrix): returns mechanism-0 error → obs_flip 0
    map_error_decoder = decoders.LookupDecoder(pcm, max_weight=1, error_channel=error_channel)
    map_error_result = map_error_decoder.decode(syndrome)
    assert int((obs_matrix @ map_error_result % 2)[0]) == 0

    # MAP obs_flip decoder: returns an error whose obs_flip is 1
    map_obs_decoder = decoders.LookupDecoder(
        pcm, max_weight=1, error_channel=error_channel, observable_flip_matrix=obs_matrix
    )
    map_obs_result = map_obs_decoder.decode(syndrome)
    assert int((obs_matrix @ map_obs_result % 2)[0]) == 1

    # Auto-extraction from DEM: when simplify=True, mechanisms with identical detector+observable
    # patterns are merged, so the resulting DEM has fewer mechanisms.  Check obs_flip via the
    # DEM's own observable_flip_matrix rather than the original (3-column) obs_matrix.
    error_channel_arr = np.array(error_channel)
    dem = decoders.DetectorErrorModelArrays.from_arrays(pcm, obs_matrix, error_channel_arr).to_dem()
    dem_decoder = decoders.LookupDecoder(dem, max_weight=1)
    dem_result = dem_decoder.decode(syndrome)
    dem_arrays = decoders.DetectorErrorModelArrays(dem)
    obs_flip_via_dem = int(
        np.asarray(dem_arrays.observable_flip_matrix @ dem_result, dtype=int).ravel()[0] % 2
    )
    assert obs_flip_via_dem == 1

    # DEM with no observables: observable_flip_matrix is not auto-extracted; decoder is backward
    # compatible.  All three mechanisms share the same detector pattern so they get merged into one.
    dem_no_obs = decoders.DetectorErrorModelArrays.from_arrays(pcm, None, error_channel_arr).to_dem()
    no_obs_decoder = decoders.LookupDecoder(dem_no_obs, max_weight=1)
    assert decoders.DetectorErrorModelArrays(dem_no_obs).num_observables == 0
    assert np.any(no_obs_decoder.decode(syndrome) != 0)

    # No penalty_func: observable_flip_matrix is a no-op (can't determine "most likely" without probs)
    decoder_no_prob = decoders.LookupDecoder(pcm, max_weight=1)
    decoder_with_obs_no_prob = decoders.LookupDecoder(
        pcm, max_weight=1, observable_flip_matrix=obs_matrix
    )
    assert np.array_equal(decoder_no_prob.decode(syndrome), decoder_with_obs_no_prob.decode(syndrome))


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
