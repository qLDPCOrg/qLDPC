# SPDX-License-Identifier: Apache-2.0

"""Tests for the Relay-BP decoder adapter and builders."""

from __future__ import annotations

import copy
import functools
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
from qldpc.conftest import run_python
from qldpc.decoders.conftest import ToyProblem
from qldpc.decoders.construction._resolution import build_error_decoder
from qldpc.decoders.external.relay_bp import (
    RelayBPDecoder,
    _get_decoder_min_sum_bp,
    _get_decoder_relay_bp,
    _get_relay_bp,
)


def test_relay_bp(toy_problem: ToyProblem) -> None:
    """The Relay-BP decoder wraps matrices, sparse matrices, and detector error models."""
    matrix, error, syndrome = toy_problem
    errors = np.array([error, error])
    syndromes = np.array([syndrome, syndrome])

    decoder = RelayBPDecoder(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert np.array_equal(errors, decoder.decode_batch(syndromes))
    detailed = decoder.decode_errors_detailed(syndrome)
    assert np.array_equal(detailed.error, error)
    assert detailed.diagnostics["relay_bp.success"]
    iterations = cast(int, detailed.diagnostics["relay_bp.iterations"])
    assert iterations <= cast(int, detailed.diagnostics["relay_bp.max_iterations"])
    # erasure is flagged for a syndrome that no error reproduces, even without an erasure bit
    unreproducible = RelayBPDecoder(np.ones((2, 1), dtype=int))
    assert unreproducible.decode_errors_detailed(np.array([1, 0])).erasure
    decoder = RelayBPDecoder(matrix)
    assert np.array_equal(error, copy.copy(decoder).decode(syndrome))

    # batch methods show no progress bar by default
    with unittest.mock.patch.object(decoder, "decoder") as runner:
        runner.decode_batch.return_value = errors
        decoder.decode_errors_batch(syndromes)
        decoder.decode_batch(syndromes)
    assert [call.args[2] for call in runner.decode_batch.call_args_list] == [False, False]

    decoder = RelayBPDecoder(scipy.sparse.dok_matrix(matrix))
    assert np.array_equal(error, decoder.decode_detailed(syndrome).decoding)

    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 1e-3).to_dem()
    decoder = RelayBPDecoder(dem)
    assert np.array_equal(error, decoder.decode(syndrome))

    with (
        unittest.mock.patch.dict("sys.modules", {"relay_bp": None}),
        pytest.raises(ModuleNotFoundError, match=r"qldpc\[relay-bp\]") as exc_info,
    ):
        build_error_decoder(np.array([[]]), decoder=decoders.relay_bp())
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)

    with pytest.raises(ValueError, match="name not recognized"):
        RelayBPDecoder(np.array([[]]), name="invalid_name")

    with pytest.raises(TypeError, match="not a string"):
        decoders.external.RelayBPDecoder("MinSumBPDecoderF32")

    with pytest.warns(UserWarning, match="will override") as caught:
        decoders.relay_bp(error_priors=[0.1, 0.1]).build(dem)
    assert caught[0].filename == __file__

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
        get_decoder = functools.partial(RelayBPDecoder, dem, add_erasure_bit=add_erasure_bit)
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
        RelayBPDecoder(np.eye(2, dtype=int)).decode_observables(np.zeros(2, dtype=int))


def test_relay_bp_erasure_bit_marks_an_unexplained_syndrome(pytestconfig: pytest.Config) -> None:
    """Relay-BP marks exactly the syndromes that its errors miss."""
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

        relay_bp_decoder = decoders.external.RelayBPDecoder(matrix, add_erasure_bit=True)
        relay_bp_errors = np.array([relay_bp_decoder.decode(syndrome) for syndrome in syndromes])
        check_erasure_bits(matrix, syndromes, relay_bp_errors)

        decoded_batch = relay_bp_decoder.decode_batch(syndromes)
        check_erasure_bits(matrix, syndromes, decoded_batch)

        num_erasures += int(relay_bp_errors[:, -1].sum())

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
    """Immediate and deferred builders forward backend_options to the Relay-BP backend."""
    matrix = np.eye(2, dtype=int)
    backend_options = {"backend_extension": 12}
    # the backend receives backend_options, and rejects unsupported names
    with pytest.raises(TypeError, match="backend_extension"):
        builder(matrix, backend_options=backend_options)
    with pytest.raises(TypeError, match="backend_extension"):
        helper(backend_options=backend_options).build(matrix)


def test_relay_bp_import_is_lazy() -> None:
    """Importing the integration does not import relay-bp until a decoder is built."""
    code = """
import sys
import qldpc.decoders.external.relay_bp
assert "relay_bp" not in sys.modules
"""
    run_python(code)


def test_relay_bp_nested_import_failure() -> None:
    """A module missing inside relay-bp is reported as is, rather than as a missing relay-bp."""
    nested_error = ModuleNotFoundError("No module named 'nested'", name="nested")
    with (
        pytest.raises(ModuleNotFoundError, match="nested"),
        unittest.mock.patch("builtins.__import__", side_effect=nested_error),
    ):
        _get_relay_bp()


def test_relay_bp_unsorted_observables() -> None:
    """Relay-BP accepts an observable matrix with unsorted indices, which a DEM can produce."""
    # the only error lists the observables it flips as (1, 0)
    observable_matrix = scipy.sparse.csc_matrix(([1, 1], [1, 0], [0, 2]), shape=(2, 1))
    with warnings.catch_warnings():
        # reordering indices alone does not warn
        warnings.filterwarnings("error", ".*Reducing these entries mod 2")
        decoder = RelayBPDecoder(np.array([[1]]), observable_error_matrix=observable_matrix)
    assert np.array_equal(decoder.decode_observables(np.array([1])), [1, 1])


def test_relay_bp_noncanonical_sparse_matrices() -> None:
    """Relay-BP accepts sparse matrices with unsorted or duplicate indices, and leaves them be."""
    # error 0 lists its checks as (1, 0), and error 1 lists check 0 twice, which cancels
    matrix = scipy.sparse.csc_matrix(
        (np.ones(4, dtype=np.uint8), [1, 0, 0, 0], [0, 2, 4]), shape=(2, 2)
    )
    indices = matrix.indices.copy()
    with pytest.warns(UserWarning, match="Reducing these entries mod 2"):
        decoder = RelayBPDecoder(matrix)
    assert np.array_equal(decoder.decode(np.array([1, 1])), [1, 0])
    assert np.array_equal(matrix.indices, indices)


def test_relay_bp_reads_entries_mod_2() -> None:
    """Relay-BP reads a dense matrix mod 2, and warns if that changes an entry."""
    matrix = np.array([[2, 1]])
    with pytest.warns(UserWarning, match="Reducing these entries mod 2"):
        decoder = RelayBPDecoder(matrix)
    assert np.array_equal(decoder.decode(np.array([1])), [0, 1])
    assert np.array_equal(matrix, [[2, 1]])


def test_relay_bp_rejects_invalid_matrices() -> None:
    """Relay-BP rejects non-integer entries and fields other than GF(2)."""
    for matrix in (np.array([[0.5]]), scipy.sparse.csc_matrix([[0.5]])):
        with pytest.raises(ValueError, match="integer entries"):
            RelayBPDecoder(matrix)
    with pytest.raises(ValueError, match="requires a binary matrix"):
        RelayBPDecoder(galois.GF(3)([[1]]))
