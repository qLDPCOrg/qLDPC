# SPDX-License-Identifier: Apache-2.0

"""Unit tests for custom.py."""

from __future__ import annotations

import copy
import functools
import itertools
import unittest.mock

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import stim

from qldpc import codes, decoders, math
from qldpc.decoders.conftest import SurfaceCodeProblem, ToyProblem


def test_batch_decoding_by_alias() -> None:
    """An error decoder that only implements decode and decode_batch decodes batches."""
    matrix = np.eye(2, dtype=int)
    syndromes = np.eye(2, dtype=int)

    class OldDecoder(decoders.ErrorDecoder):
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndrome

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndromes

    old_decoder = OldDecoder()
    assert np.array_equal(old_decoder.decode_errors(syndromes[0]), syndromes[0])
    assert np.array_equal(decoders.batch_decode_errors(old_decoder, syndromes), syndromes)

    composite_decoder = decoders.CompositeDecoder((old_decoder, 1), (old_decoder, 1))
    assert composite_decoder.decoders == (old_decoder, old_decoder)
    assert np.array_equal(composite_decoder.decode_batch(syndromes), syndromes)

    direct_decoder = decoders.DirectDecoder.from_indirect(old_decoder, matrix)
    assert np.array_equal(direct_decoder.decode_batch(syndromes), np.zeros_like(syndromes))


def test_relay_bp(toy_problem: ToyProblem) -> None:
    """The Relay-BP decoder needs a custom wrapper class."""
    matrix, error, syndrome = toy_problem
    errors = np.array([error, error])
    syndromes = np.array([syndrome, syndrome])

    decoder = decoders.get_decoder_RBP(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))
    assert np.array_equal(errors, decoder.decode_batch(syndromes))

    # copying a decoder does not recurse looking for the inner decoder
    assert np.array_equal(error, copy.copy(decoder).decode(syndrome))

    # a call with no arguments is forwarded to the inner decoder
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        decoder.compute_observables()

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
        decoders.get_error_decoder(np.array([[]]), decoder=decoders.relay_bp())

    # fail to initialize a relay-bp decoder from an unrecognized name
    with pytest.raises(ValueError, match="name not recognized"):
        decoders.get_decoder_RBP(np.array([[]]), name="invalid_name")

    # fail when a decoder name string is passed where the matrix should be
    with pytest.raises(TypeError, match="breaking change"):
        decoders.RelayBPDecoder("MinSumBPDecoderF32")

    # passing explicit error_priors alongside a DEM emits a warning
    with pytest.warns(UserWarning, match="will override"):
        decoders.RelayBPDecoder(dem, error_priors=[0.1, 0.1])

    # an observable_error_matrix conflicts with the observables of a detector error model, which
    # must be rejected under `python -O` as well
    with pytest.raises(ValueError, match="Cannot specify an observable_error_matrix"):
        decoders.RelayBPDecoder(dem, observable_error_matrix=np.eye(2, dtype=np.uint8))


def test_relay_bp_observables() -> None:
    """A RelayBPDecoder predicts observable flips, with or without an erasure bit."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.02
    )
    dem = circuit.detector_error_model()
    syndromes = circuit.compile_detector_sampler(seed=0).sample(100).astype(int)
    observable_flip_matrix = decoders.DetectorErrorModelArrays(dem).observable_flip_matrix

    for add_erasure_bit in [False, True]:
        # relay_bp advances a seeded random number generator with every decode, so compare the
        # outputs of freshly built decoders
        get_decoder = functools.partial(
            decoders.get_decoder_RBP, dem, add_erasure_bit=add_erasure_bit
        )
        predicted_flips = get_decoder().decode_observables_batch(syndromes, progress_bar=False)
        assert predicted_flips.shape == (len(syndromes), dem.num_observables + add_erasure_bit)
        decoder = get_decoder()
        assert np.array_equal(
            predicted_flips, [decoder.decode_observables(syndrome) for syndrome in syndromes]
        )

        # the predicted flips are those of the inferred errors
        errors = get_decoder().decode_batch(syndromes, progress_bar=False)
        if add_erasure_bit:
            assert np.array_equal(predicted_flips[:, -1], errors[:, -1])
            errors = errors[:, :-1]
        expected_flips = np.asarray(errors @ observable_flip_matrix.T) % 2
        assert np.array_equal(predicted_flips[:, : dem.num_observables], expected_flips)

        # an empty batch, which relay_bp cannot decode, yields empty predictions
        no_syndromes = syndromes[:0]
        assert decoder.decode_batch(no_syndromes).shape == (0, errors.shape[1] + add_erasure_bit)
        assert decoder.decode_observables_batch(no_syndromes).shape == (
            0,
            dem.num_observables + add_erasure_bit,
        )

    # predicting observable flips requires observables
    with pytest.raises(ValueError, match="requires an observable_error_matrix"):
        decoders.get_decoder_RBP(np.eye(2, dtype=int)).decode_observables(np.zeros(2, dtype=int))


def test_ilp_decoder(toy_problem: ToyProblem) -> None:
    """Decode using an integer linear program."""
    matrix, error, syndrome = toy_problem
    decoder = decoders.ILPDecoder(scipy.sparse.csc_matrix(matrix))
    assert np.array_equal(error, decoder.decode(syndrome))

    # decode over the trinary field
    field = galois.GF(3)
    matrix = -matrix.view(field)
    error = -error.view(field)
    decoder = decoders.ILPDecoder(matrix)
    assert np.array_equal(error, decoder.decode(syndrome))


def test_ilp_decoder_minimum_weight(pytestconfig: pytest.Config) -> None:
    """An integer linear program returns an error of minimum weight that reproduces the syndrome.

    Both properties are checked against exhaustive search.  The particular minimum-weight error
    that gets returned is up to the solver, so it is not checked.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    for order in [2, 5]:
        field = galois.GF(order)
        for _ in range(4):
            num_checks, num_bits = rng.integers(2, 4), rng.integers(2, 4)
            matrix = field(rng.integers(order, size=(num_checks, num_bits)))
            error = field(rng.integers(order, size=num_bits))
            syndrome = matrix @ error

            candidates = [
                field(vector)
                for vector in itertools.product(range(order), repeat=int(num_bits))
                if np.array_equal(matrix @ field(vector), syndrome)
            ]
            min_weight = min(np.count_nonzero(candidate) for candidate in candidates)

            decoded = decoders.ILPDecoder(matrix).decode(np.asarray(syndrome, dtype=int))
            assert np.array_equal(matrix @ field(decoded), syndrome)
            assert np.count_nonzero(decoded) == min_weight


def test_ilp_decoder_early_termination() -> None:
    """A HiGHS integer linear program that stops early does not return an unusable error.

    A solver told to give up immediately can report a finite objective for a point that reproduces
    no syndrome at all.  With no way to report that, such a point is rejected; given an erasure bit,
    it is reported as an erasure instead, which is what the decoders that infer errors heuristically
    already do.
    """
    pytest.importorskip("highspy")
    matrix = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0]])
    syndrome = np.array([1, 0, 1])

    decoder = decoders.ILPDecoder(matrix, solver="HIGHS", time_limit=1e-9)
    with (
        pytest.warns(UserWarning, match="inaccurate"),
        pytest.raises(ValueError, match="does not reproduce the syndrome"),
    ):
        decoder.decode(syndrome)

    # the same solver, asked for an erasure bit, erases the shot rather than refusing it
    decoder = decoders.ILPDecoder(matrix, add_erasure_bit=True, solver="HIGHS", time_limit=1e-9)
    with pytest.warns(UserWarning, match="inaccurate"):
        decoded = decoder.decode(syndrome)
    assert len(decoded) == matrix.shape[1] + 1
    assert decoded[-1] == 1

    # without the time limit, the same problem is solved, and the erasure bit reports no erasure
    decoded = decoders.ILPDecoder(matrix, add_erasure_bit=True).decode(syndrome)
    assert decoded[-1] == 0
    assert np.array_equal(matrix @ decoded[:-1] % 2, syndrome)


def test_ilp_decoder_unreproducible_syndrome() -> None:
    """A syndrome that no error reproduces is erased with a warning, rather than refused.

    The one column below spans only the all-zero and all-one vectors, so the program can prove that
    nothing reproduces a syndrome on a single check.
    """
    matrix = np.array([[1], [1]])
    syndrome = np.array([0, 1])

    with pytest.raises(ValueError, match="could not be found"):
        decoders.ILPDecoder(matrix).decode(syndrome)

    decoder = decoders.ILPDecoder(matrix, add_erasure_bit=True)
    with pytest.warns(UserWarning, match="could not be found"):
        decoded = decoder.decode(syndrome)
    assert len(decoded) == matrix.shape[1] + 1
    assert decoded[-1] == 1


def test_ilp_decoder_near_integral_values() -> None:
    """A mixed integer solver's near-integral values are rounded, not truncated toward zero."""
    import cvxpy

    matrix = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0]])
    syndrome = np.array([1, 0, 1])
    decoder = decoders.ILPDecoder(matrix)
    expected = decoder.decode(syndrome)

    solve = cvxpy.Problem.solve

    def solve_then_perturb(problem: cvxpy.Problem, **kwargs: object) -> float:
        """Solve, then report the solution the way a solver at its tolerance would."""
        result = solve(problem, **kwargs)
        decoder.variables.value = np.asarray(decoder.variables.value) - 4e-16
        return float(result)

    with unittest.mock.patch.object(cvxpy.Problem, "solve", solve_then_perturb):
        assert np.array_equal(expected, decoder.decode(syndrome))

    # The returned integer error must not be narrowed to a syndrome dtype that cannot represent it.
    field = galois.GF(3)
    decoder = decoders.ILPDecoder(field([[1, 1], [0, 1]]), add_erasure_bit=True)
    decoded = decoder.decode(np.array([0, 1], dtype=bool))
    assert np.array_equal(decoded, [2, 1, 0])
    assert np.array_equal(field([[1, 1], [0, 1]]) @ field(decoded[:-1]), [0, 1])


def test_invalid_ilp() -> None:
    """Fail to solve an invalid integer linear programming problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="could not be found"):
        decoders.get_error_decoder(matrix, decoder=decoders.ilp()).decode(syndrome)

    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        decoders.get_error_decoder(galois.GF(4)(matrix), decoder=decoders.ilp()).decode(syndrome)


def test_generalized_union_find() -> None:
    """Generalized Union-Find."""
    base_code: codes.CSSCode = codes.C4Code()
    code = functools.reduce(codes.CSSCode.concatenate, [base_code] * 3)
    error = code.field.Zeros(len(code))
    error[[3, 4]] = 1
    matrix = code.matrix_z
    syndrome = matrix @ error
    assert (
        np.count_nonzero(
            decoders.get_error_decoder(matrix, decoder=decoders.guf()).decode(syndrome)
        )
        > 2
    )
    assert (
        np.count_nonzero(
            decoders.get_error_decoder(matrix, decoder=decoders.guf(max_weight=2)).decode(syndrome)
        )
        == 2
    )

    # cover the trivial syndrome with the generalized Union-Find decoer
    assert np.array_equal(
        np.zeros_like(error),
        decoders.get_error_decoder(matrix, decoder=decoders.guf()).decode(np.zeros_like(syndrome)),
    )


def test_erasure_bit_marks_an_unexplained_syndrome(pytestconfig: pytest.Config) -> None:
    """Generalized Union-Find and Relay-BP set an erasure bit iff their error misses the syndrome.

    Both decoders answer a syndrome that no error explains, so without the bit the answer is
    indistinguishable from the error inferred for a syndrome that is explained.  Every syndrome of
    every matrix is checked, so a trivial syndrome -- which the all-zero error does explain -- has
    to come back unerased.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))

    def check_erasure_bits(
        matrix: npt.NDArray[np.int_],
        syndromes: npt.NDArray[np.int_],
        decoded_errors: npt.NDArray[np.int_],
    ) -> None:
        """Assert that each erasure bit is set exactly when its error misses its syndrome."""
        for decoded, syndrome in zip(decoded_errors, syndromes):
            explained = np.array_equal(matrix @ decoded[:-1] % 2, syndrome)
            assert bool(decoded[-1]) == (not explained)

    num_erasures = 0
    for _ in range(4):
        num_checks, num_bits = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        matrix = rng.integers(2, size=(num_checks, num_bits))
        matrix[0] = 0  # a trivial check row, so that some syndrome is always unexplainable
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

        # decoding a batch appends one erasure bit per shot, under the same rule
        decoded_batch = relay_bp_decoder.decode_batch(syndromes)
        check_erasure_bits(matrix, syndromes, decoded_batch)

        num_erasures += int(guf_errors[:, -1].sum()) + int(relay_bp_errors[:, -1].sum())

    assert num_erasures  # a run in which nothing is erased checks nothing


def test_symplectic_erasure() -> None:
    """A qudit syndrome that no error can induce is erased rather than answered.

    A trivial row of a parity check matrix witnesses no error at all, so a syndrome bit on it is
    exactly what an erasure bit reports.  Reaching that conclusion requires the Tanner graph to
    carry a node for the trivial check.
    """
    code = codes.FiveQubitCode()
    matrix = np.vstack([np.asarray(code.matrix, dtype=int), np.zeros(2 * len(code), dtype=int)])
    decoder = decoders.GUFDecoder(matrix, symplectic=True, add_erasure_bit=True)

    # only the trivial check fires, which no error can do
    syndrome = np.zeros(matrix.shape[0], dtype=int)
    syndrome[-1] = 1
    decoded = decoder.decode(syndrome)
    assert len(decoded) == 2 * len(code) + 1
    assert decoded[-1] == 1
    assert not np.any(decoded[:-1])


def test_composite_erasure() -> None:
    """A CompositeDecoder is erased when any of its code blocks is erased."""
    # row 1 of this matrix is trivial, so no error explains a syndrome that is nonzero there
    matrix = np.array([[1, 1, 0], [0, 0, 0]])
    block_decoder = decoders.RelayBPDecoder(matrix, add_erasure_bit=True)
    composite_decoder = decoders.CompositeDecoder.from_copies(block_decoder, 2, 2)

    # one erasure bit for the composite, not one per block
    syndromes = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    expected_erasures = [0, 1, 1, 1]
    for syndrome, erased in zip(syndromes, expected_erasures):
        decoded = composite_decoder.decode(syndrome)
        assert len(decoded) == 2 * matrix.shape[1] + 1
        assert decoded[-1] == erased

    decoded_batch = composite_decoder.decode_batch(syndromes)
    assert np.array_equal(decoded_batch[:, -1], expected_erasures)

    # a block that cannot erase contributes all of its entries, and none of them is an erasure bit
    plain_decoder = decoders.RelayBPDecoder(matrix)

    # the second bit of a block's syndrome is the one its trivial check row cannot explain, so only
    # the erasing block's half of each composite syndrome can erase the composite
    for blocks, expected in [
        (((plain_decoder, 2), (block_decoder, 2)), syndromes[:, 3]),
        (((block_decoder, 2), (plain_decoder, 2)), syndromes[:, 1]),
    ]:
        mixed_decoder = decoders.CompositeDecoder(*blocks)
        assert mixed_decoder.has_erasure_bit
        for syndrome, erased in zip(syndromes, expected):
            decoded = mixed_decoder.decode(syndrome)
            assert len(decoded) == 2 * matrix.shape[1] + 1
            assert decoded[-1] == erased


def test_augmented_decoders(toy_problem: ToyProblem) -> None:
    """Composite and direct decoders, built from other decoders."""
    matrix, error, syndrome = toy_problem
    decoder = decoders.get_error_decoder(matrix, decoder=decoders.mwpm())

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

    # a decoder whose output is wider than the code word cannot correct that word, in either
    # single-shot or batch form
    class WideDecoder:
        """A decoder that appends an extra entry to every error it infers."""

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.zeros(error.size + 1, dtype=int)

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.zeros((len(syndromes), error.size + 1), dtype=int)

    direct_decoder = decoders.DirectDecoder.from_indirect(WideDecoder(), matrix)
    with pytest.raises(ValueError, match="cannot be subtracted"):
        direct_decoder.decode(error)
    with pytest.raises(ValueError, match="cannot be subtracted"):
        direct_decoder.decode_batch(errors)


def test_quantum_decoding_from_plain_matrix() -> None:
    """A parity check matrix that is not a FieldArray is interpreted over GF(2)."""
    code = codes.FiveQubitCode()
    error = code.field.Zeros(2 * len(code))
    error[2] = 1
    syndrome = np.asarray(code.matrix @ math.symplectic_conjugate(error), dtype=int)

    decoder = decoders.GUFDecoder(np.asarray(code.matrix, dtype=int), symplectic=True)
    decoded_error = code.field(decoder.decode(syndrome))
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))


def test_quantum_decoding(surface_code_problem: SurfaceCodeProblem) -> None:
    """Decode random weight-2 errors in a GF(3) surface code."""
    code, _error, syndrome = surface_code_problem
    decoder = decoders.GUFDecoder(code.matrix, symplectic=True)
    decoded_error = decoder.decode(syndrome).view(code.field)
    assert np.array_equal(syndrome, code.matrix @ math.symplectic_conjugate(decoded_error))
