# SPDX-License-Identifier: Apache-2.0

"""Tests for the Relay-BP decoder adapter and builders."""

from __future__ import annotations

import copy
import functools
import subprocess
import sys
import unittest.mock
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
        decoders.RelayBPDecoder("MinSumBPDecoderF32")

    with pytest.warns(UserWarning, match="will override"):
        decoders.RelayBPDecoder(dem, error_priors=[0.1, 0.1])

    with pytest.raises(ValueError, match="Cannot specify an observable_error_matrix"):
        decoders.RelayBPDecoder(dem, observable_error_matrix=np.eye(2, dtype=np.uint8))

    builders: list[Callable[..., decoders.RelayBPDecoder]] = [
        _get_decoder_relay_bp,
        _get_decoder_min_sum_bp,
    ]
    for builder in builders:
        relay_decoder = builder(matrix, precision="F32")
        assert np.array_equal(np.asarray(matrix) @ relay_decoder.decode(syndrome) % 2, syndrome)


def test_relay_bp_observables() -> None:
    """A RelayBPDecoder predicts observable flips, with or without an erasure bit."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.02
    )
    dem = circuit.detector_error_model()
    syndromes = circuit.compile_detector_sampler(seed=0).sample(100).astype(int)
    observable_flip_matrix = decoders.DetectorErrorModelArrays(dem).observable_flip_matrix

    for add_erasure_bit in [False, True]:
        get_decoder = functools.partial(_get_decoder_rbp, dem, add_erasure_bit=add_erasure_bit)
        predicted_flips = get_decoder().decode_observables_batch(syndromes, progress_bar=False)
        assert predicted_flips.shape == (len(syndromes), dem.num_observables + add_erasure_bit)
        decoder = get_decoder()
        assert np.array_equal(
            predicted_flips, [decoder.decode_observables(syndrome) for syndrome in syndromes]
        )
        detailed = decoder.decode_observables_detailed_batch(syndromes)
        assert np.array_equal([result.observable_flips for result in detailed], predicted_flips)
        assert np.array_equal(
            decoder.decode_observables_detailed(syndromes[0]).observable_flips, predicted_flips[0]
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

        guf_decoder = decoders.GUFDecoder(galois.GF(2)(matrix), add_erasure_bit=True)
        relay_bp_decoder = decoders.RelayBPDecoder(matrix, add_erasure_bit=True)

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
    builder: Callable[..., decoders.RelayBPDecoder],
    helper: Callable[..., decoders.DecoderSpec[decoders.RelayBPDecoder]],
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
