# SPDX-License-Identifier: Apache-2.0

"""Tests for the Relay-BP decoder adapter and builders."""

from __future__ import annotations

import copy
import functools
import subprocess
import sys
import unittest.mock
import warnings
from collections.abc import Callable
from typing import cast

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import stim

from qldpc import decoders
from qldpc.decoders.conftest import ToyProblem
from qldpc.decoders.construction.resolution import _get_error_decoder
from qldpc.decoders.external.relay_bp import (
    _get_decoder_min_sum_bp,
    _get_decoder_rbp,
    _get_decoder_relay_bp,
)


def test_relay_bp(toy_problem: ToyProblem) -> None:
    """The Relay-BP decoder wraps matrices, sparse matrices, and detector error models."""
    matrix, error, syndrome = toy_problem
    errors = np.array([error, error])
    syndromes = np.array([syndrome, syndrome])

    decoder = _get_decoder_rbp(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert np.array_equal(errors, decoder.decode_batch(syndromes))
    detailed = decoder.decode_errors_detailed(syndrome)
    assert np.array_equal(detailed.error, error)
    assert detailed.diagnostics["relay_bp.success"]
    iterations = cast(int, detailed.diagnostics["relay_bp.iterations"])
    assert iterations <= cast(int, detailed.diagnostics["relay_bp.max_iterations"])
    # erasure is flagged for a syndrome that no error reproduces, even without an erasure bit
    unreproducible = _get_decoder_rbp(np.ones((2, 1), dtype=int))
    assert unreproducible.decode_errors_detailed(np.array([1, 0])).erasure
    decoder = _get_decoder_rbp(matrix)
    assert np.array_equal(error, copy.copy(decoder).decode(syndrome))

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        decoder.compute_observables()

    decoder = _get_decoder_rbp(scipy.sparse.dok_matrix(matrix))
    assert np.array_equal(error, decoder.decode_detailed(syndrome).decoding)

    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    decoder = _get_decoder_rbp(dem)
    assert np.array_equal(error, decoder.decode(syndrome))

    with (
        unittest.mock.patch.dict("sys.modules", {"relay_bp": None}),
        pytest.raises(ImportError, match="Failed to import relay-bp"),
    ):
        _get_error_decoder(np.array([[]]), decoder=decoders.relay_bp())

    with pytest.raises(ValueError, match="name not recognized"):
        _get_decoder_rbp(np.array([[]]), name="invalid_name")

    with pytest.raises(TypeError, match="breaking change"):
        decoders.external.RelayBPDecoder("MinSumBPDecoderF32")

    with pytest.warns(UserWarning, match="will override"):
        decoders.external.RelayBPDecoder(dem, error_priors=[0.1, 0.1])

    with pytest.raises(ValueError, match="Cannot specify an observable_error_matrix"):
        decoders.external.RelayBPDecoder(dem, observable_error_matrix=np.eye(2, dtype=np.uint8))

    builders: list[Callable[..., decoders.external.RelayBPDecoder]] = [
        _get_decoder_relay_bp,
        _get_decoder_min_sum_bp,
    ]
    for builder in builders:
        relay_decoder = builder(matrix, precision="F32")
        assert np.array_equal(np.asarray(matrix) @ relay_decoder.decode(syndrome) % 2, syndrome)


def test_relay_bp_observables() -> None:
    """A RelayBPDecoder predicts observable flips, with or without an erasure bit."""
    # noise at which Relay-BP's relay legs, and hence its persistent random state, affect results
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.05
    )
    dem = circuit.detector_error_model()
    syndromes = circuit.compile_detector_sampler(seed=0).sample(100).astype(int)
    observable_flip_matrix = decoders.DetectorErrorModelArrays(dem).observable_flip_matrix

    for add_erasure_bit in [False, True]:
        # Relay-BP draws random relay parameters from a generator that persists across calls, so
        # compare fresh decoders that have decoded the same syndromes in the same order.
        get_decoder = functools.partial(_get_decoder_rbp, dem, add_erasure_bit=add_erasure_bit)
        predicted_flips = get_decoder().decode_observables_batch(syndromes, progress_bar=False)
        assert predicted_flips.shape == (len(syndromes), dem.num_observables + add_erasure_bit)
        decoder = get_decoder()
        assert np.array_equal(
            predicted_flips, [decoder.decode_observables(syndrome) for syndrome in syndromes]
        )
        detailed = get_decoder().decode_observables_detailed_batch(syndromes)
        assert np.array_equal([result.observable_flips for result in detailed], predicted_flips)
        assert np.array_equal(
            get_decoder().decode_observables_detailed(syndromes[0]).observable_flips,
            predicted_flips[0],
        )

        errors = get_decoder().decode_batch(syndromes, progress_bar=False)
        if add_erasure_bit:
            assert np.array_equal(predicted_flips[:, -1], errors[:, -1])
            errors = errors[:, :-1]
        expected_flips = np.asarray(errors @ observable_flip_matrix.T) % 2
        assert np.array_equal(predicted_flips[:, : dem.num_observables], expected_flips)

        no_syndromes = syndromes[:0]
        assert decoder.decode_batch(no_syndromes).shape == (0, errors.shape[1] + add_erasure_bit)
        assert decoder.decode_observables_batch(no_syndromes).shape == (
            0,
            dem.num_observables + add_erasure_bit,
        )
        assert decoder.decode_observables_detailed_batch(no_syndromes) == ()

    with pytest.raises(ValueError, match="requires an observable_error_matrix"):
        _get_decoder_rbp(np.eye(2, dtype=int)).decode_observables(np.zeros(2, dtype=int))


def test_erasure_bit_marks_an_unexplained_syndrome(pytestconfig: pytest.Config) -> None:
    """Generalized Union-Find and Relay-BP mark exactly the syndromes their errors miss."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    def check_erasure_bits(
        matrix: npt.NDArray[np.int_],
        syndromes: npt.NDArray[np.int_],
        decoded_errors: npt.NDArray[np.int_],
    ) -> None:
        for decoded, syndrome in zip(decoded_errors, syndromes):
            explained = np.array_equal(matrix @ decoded[:-1] % 2, syndrome)
            assert bool(decoded[-1]) == (not explained)

    num_erasures = 0
    for _ in range(4):
        num_checks, num_bits = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        matrix = rng.integers(2, size=(num_checks, num_bits))
        matrix[0] = 0
        syndromes = np.array(
            [[(bits >> cc) & 1 for cc in range(num_checks)] for bits in range(2**num_checks)],
            dtype=int,
        )

        guf_decoder = decoders.custom.GUFDecoder(galois.GF(2)(matrix), add_erasure_bit=True)
        relay_bp_decoder = decoders.external.RelayBPDecoder(matrix, add_erasure_bit=True)

        guf_errors = np.array([guf_decoder.decode(syndrome) for syndrome in syndromes])
        relay_bp_errors = np.array([relay_bp_decoder.decode(syndrome) for syndrome in syndromes])
        check_erasure_bits(matrix, syndromes, guf_errors)
        check_erasure_bits(matrix, syndromes, relay_bp_errors)

        decoded_batch = relay_bp_decoder.decode_batch(syndromes)
        check_erasure_bits(matrix, syndromes, decoded_batch)

        num_erasures += int(guf_errors[:, -1].sum()) + int(relay_bp_errors[:, -1].sum())

    assert num_erasures


@pytest.mark.parametrize(
    ("builder", "helper"),
    [
        (_get_decoder_relay_bp, decoders.relay_bp),
        (_get_decoder_min_sum_bp, decoders.min_sum_bp),
    ],
)
def test_relay_backend_options(
    builder: Callable[..., decoders.external.RelayBPDecoder],
    helper: Callable[..., decoders.DecoderSpec[decoders.external.RelayBPDecoder]],
) -> None:
    """Named options and backend_options survive deferred construction."""
    matrix = np.eye(2, dtype=int)
    backend_options = {"backend_extension": 12}
    with unittest.mock.patch("qldpc.decoders.external.relay_bp._get_relay_decoder") as backend:
        builder(matrix, backend_options=backend_options)
        assert backend.call_args.kwargs["backend_extension"] == 12
        spec = helper(backend_options=backend_options)
        assert spec.options["backend_options"] == backend_options
        spec.build(matrix)
        assert backend.call_args.kwargs["backend_extension"] == 12

    # the backend rejects unsupported names when the decoder is built
    with pytest.raises(TypeError, match="backend_extension"):
        helper(backend_options=backend_options).build(matrix)


def test_relay_bp_import_is_lazy() -> None:
    """Importing the integration does not import relay-bp until a decoder is built."""
    code = """
import sys
import qldpc.decoders.external.relay_bp
assert "relay_bp" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_relay_bp_unsorted_observables() -> None:
    """Relay-BP decodes a detector error model in which an error flips several observables."""
    # the observable flip matrix of this model stores the observables of error 0 as (8, 1)
    dem = stim.DetectorErrorModel("error(0.1) D0 L1 L8\nerror(0.1) D0 D1\nerror(0.1) D1 L0")
    expected_flips = np.zeros(dem.num_observables, dtype=int)
    expected_flips[[1, 8]] = 1
    decoder = _get_decoder_rbp(dem)
    assert np.array_equal(decoder.decode_observables(np.array([1, 0])), expected_flips)


def test_relay_bp_noncanonical_sparse_matrices() -> None:
    """Relay-BP accepts sparse matrices with unsorted or duplicate indices, and leaves them be."""
    # error 1 lists its checks out of order, and error 2 lists check 0 twice, which cancels
    matrix = scipy.sparse.csc_matrix(([1] * 6, [0, 1, 0, 1, 0, 0], [0, 1, 3, 6]), shape=(2, 3))
    # error 0 lists its observables out of order, and error 2 lists observable 0 twice
    observable_error_matrix = scipy.sparse.csc_matrix(
        ([1] * 4, [1, 0, 0, 0], [0, 2, 2, 4]), shape=(2, 3)
    )
    matrix_indices = matrix.indices.copy()
    observable_indices = observable_error_matrix.indices.copy()

    with pytest.warns(UserWarning, match="Reducing these entries mod 2"):
        decoder = _get_decoder_rbp(matrix, observable_error_matrix=observable_error_matrix)
    syndromes = np.array([[1, 0], [1, 1], [0, 1]])
    assert np.array_equal(decoder.decode_batch(syndromes), np.eye(3, dtype=int))
    assert np.array_equal(decoder.decode_observables_batch(syndromes), [[1, 1], [0, 0], [0, 0]])
    assert np.array_equal(matrix.indices, matrix_indices)
    assert np.array_equal(observable_error_matrix.indices, observable_indices)


def test_relay_bp_reads_entries_mod_2() -> None:
    """Relay-BP reads dense and sparse matrices mod 2 alike, and warns if that changes an entry."""
    unsorted_binary = scipy.sparse.csc_matrix(([1, 1, 1], [0, 1, 0], [0, 1, 3]), shape=(2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # reordering indices alone does not warn
        _get_decoder_rbp(unsorted_binary)

    # error 1 has entry 2 in check 0, which is 0 over GF(2), and entry -1 in check 1, which is 1
    dense = np.array([[1, 2], [0, -1]])
    syndromes = np.array([[1, 0], [0, 1], [1, 1]])
    decodes = []
    for matrix in (
        dense,
        dense.astype(float),
        scipy.sparse.csc_matrix(dense),
        scipy.sparse.csr_matrix(dense),
    ):
        with pytest.warns(UserWarning, match="Reducing these entries mod 2"):
            decoder = _get_decoder_rbp(matrix, error_priors=[0.1, 0.1])
        decodes.append(decoder.decode_batch(syndromes))
    # over GF(2) the matrix is the identity, so each syndrome is its own error
    for decode in decodes:
        assert np.array_equal(decode, syndromes)
    assert np.array_equal(dense, [[1, 2], [0, -1]])


def test_relay_bp_rejects_invalid_matrices() -> None:
    """Relay-BP rejects non-integer entries and fields other than GF(2), but accepts GF(2)."""
    for matrix in (np.array([[0.5, 1]]), scipy.sparse.csc_matrix([[0.5, 1]])):
        with pytest.raises(ValueError, match="integer entries"):
            _get_decoder_rbp(matrix)
    with pytest.raises(ValueError, match="requires a binary matrix"):
        _get_decoder_rbp(galois.GF(3)([[1, 2], [0, 1]]))
    with pytest.raises(ValueError, match="requires a binary matrix"):
        _get_decoder_rbp(
            galois.GF(2)([[1, 0], [0, 1]]), observable_error_matrix=galois.GF(4)([[1, 2]])
        )
    decoder = _get_decoder_rbp(galois.GF(2)([[1, 0], [0, 1]]))
    assert np.array_equal(decoder.decode(np.array([1, 0])), [1, 0])
