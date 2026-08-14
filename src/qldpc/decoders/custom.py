"""Custom decoder classes.

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
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc import codes, math
from qldpc.math import IntegerArray
from qldpc.objects import Node

from .dems import DetectorErrorModelArrays

if TYPE_CHECKING:
    import cvxpy

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods


class Decoder(Protocol):
    """Template class for a decoder."""

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""


class BatchDecoder(Protocol):
    """Template class for a decoder that can decode in batches."""

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes and return inferred errors."""


class RelayBPDecoder:
    """Wrapper class for Relay-BP decoders, introduced in arXiv:2506.01779.

    Requires relay_bp to be installed, for example via "pip install 'qldpc[relay-bp]'".

    This class first constructs a relay_bp.decoder.DynDecoder decoder by class name, such as
    "RelayDecoderF32"; see help(relay_bp) for more options.  To enable parallelized decoding, which
    as of relay-bp==0.2.1 is only implemented for the relay_bp.ObservableDecoderRunner class,
    RelayBPDecoder wraps the relay_bp.decoder.DynDecoder in a relay_bp.ObservableDecoderRunner at
    initialization time.

    IMPORTANT POINTS TO NOTE:
    -------------------------

    1. relay_bp.ObservableDecoderRunner expects to be passed an observable_error_matrix when
        initialized.  If a RelayBPDecoder is initialized without an observable_error_matrix, this
        matrix is set to np.empty((0, 0), dtype=np.uint8).  All observable-related methods of the
        decoder will subsequently fail.
    2. RelayBPDecoder "wants" to be a subclass of relay_bp.ObservableDecoderRunner.  However, the
        latter does not allow subclassing because it is implemented in rust and exposed to Python
        via bindings.  As a hack, if a decoder: RelayBPDecoder is asked for a method or attribute it
        does not recognize, such as decoder.decode_observables_batch(detectors, parallel=True) or
        decoder.decode_detailed(detectors), it tries to pass all arguments to an identically-named
        method of relay_bp.ObservableDecoderRunner.  A consequence of this hack is that most of the
        methods that are recognized by RelayBPDecoder in practice do not appear in its
        documentation.
        See help(relay_bp.ObservableDecoderRunner) for a list of all RelayBPDecoder methods.

    For details about Relay-BP decoders, see:

    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """

    def __init__(
        self,
        pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
        error_priors: npt.NDArray[np.floating] | Sequence[float] | None = None,
        *,
        name: str = "RelayDecoderF32",
        observable_error_matrix: IntegerArray | None = None,
        include_decode_result: bool = False,
        **decoder_args: object,
    ) -> None:
        """Initialize a RelayBP decoder from the relay_bp package.

        Args:
            pcm_or_dem: A parity check matrix or detector error model (DEM).
            error_priors: Prior probabilities for each error, or None.  If error_priors is None and
                pcm_or_dem is a DEM, these are set to the error probabilities in the DEM by default.
            name: The name of the RelayBP decoder to instantiate.  Must be one of the classes listed
                under help(relay_bp.bp).
            observable_error_matrix: A binary matrix whose rows specify which error mechanisms flip
                which observables, or None.  If pcm_or_dem is a DEM, this matrix is extracted from
                the DEM.  If pcm_or_dem is a matrix and observable_error_matrix is None, the
                constructed RelayBPDecoder will not be able to predict observable flips (or logical
                error rates).
            include_decode_result: Argument passed to relay_bp.ObservableDecoderRunner.
            **decoder_args: Arguments passed to the "inner" (syndrome -> error) decoder from
                relay_bp.  See help(relay_bp.RelayDecoderF32) or https://pypi.org/project/relay-bp/
                for the options (alpha, alpha_iteration_scaling_factor, gamma0, etc.).
        """
        try:
            import relay_bp
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Failed to import relay-bp.  Try installing 'qldpc[relay-bp]'"
            )
        if not isinstance(name, str) or not hasattr(relay_bp, name):
            raise ValueError(
                f"Relay-BP decoder name not recognized: {name}\n"
                "See 'import relay_bp; help(relay_bp.bp)' for available Relay-BP decoders"
            )
        if isinstance(pcm_or_dem, str):
            raise TypeError(
                "I think you provided a Relay-BP decoder decoder name in place of a parity check"
                " matrix.  There was breaking change to this API.  See"
                " help(qldpc.decoders.RelayBPDecoder)"
            )

        # extract relevant data from a detector error model
        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            assert observable_error_matrix is None, (
                "Cannot specify an observable_error_matrix when providing a detector error model"
            )
            dem_arrays = DetectorErrorModelArrays(pcm_or_dem)
            pcm = dem_arrays.detector_flip_matrix
            observable_error_matrix = dem_arrays.observable_flip_matrix
            if error_priors is None:
                error_priors = dem_arrays.error_probs
            else:
                warnings.warn(
                    "Explicitly provided error_priors will override the error probabilities of the "
                    "provided detector error model",
                    stacklevel=2,
                )
        else:
            pcm = pcm_or_dem
            if error_priors is None:
                error_priors = [PLACEHOLDER_ERROR_RATE] * pcm.shape[1]

        # sanitize inputs
        if isinstance(pcm, galois.FieldArray):
            pcm = pcm.view(np.ndarray)
        elif isinstance(pcm, scipy.sparse.spmatrix):
            pcm = pcm.tocsc()
            pcm.sort_indices()
        if observable_error_matrix is None:
            observable_error_matrix = np.empty((0, 0), dtype=np.uint8)

        # build the decoder
        self.decoder = relay_bp.ObservableDecoderRunner(
            getattr(relay_bp, name)(pcm, np.asarray(error_priors), **decoder_args),
            observable_error_matrix,
            include_decode_result,
        )

    def decode(self, /, detectors: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error.

        Typecast detectors to np.uint8 for compatibility with the relay_bp package.
        """
        return self.decoder.decode(np.asarray(detectors, dtype=np.uint8))

    def decode_batch(
        self,
        /,
        detectors: npt.NDArray[np.int_],
        parallel: bool = False,
        progress_bar: bool = True,
        leave_progress_bar_on_finish: bool = False,
    ) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes and return inferred errors.

        Typecast detectors to np.uint8 for compatibility with the relay_bp package.
        """
        return self.decoder.decode_batch(
            np.asarray(detectors, dtype=np.uint8),
            parallel,
            progress_bar,
            leave_progress_bar_on_finish,
        )

    def __getattr__(self, name: str) -> Any:
        """Inherit all methods of self.decoder: relay_bp.ObservableDecoderRunner.

        Always typecast the first argument to np.uint8 for compatibility with the relay_bp package.
        """
        inner_func = getattr(self.decoder, name)

        @functools.wraps(inner_func)
        def outer_func(*args: object, **kwargs: object) -> Any:
            return inner_func(np.asarray(args[0], dtype=np.uint8), *args[1:], **kwargs)

        return outer_func


class ILPDecoder:
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """

    def __init__(self, matrix: IntegerArray, **decoder_args: object) -> None:
        import cvxpy

        self.modulus = type(matrix).order if isinstance(matrix, galois.FieldArray) else 2
        if not galois.is_prime(self.modulus):
            raise ValueError("ILP decoding only supports prime number fields")

        # convert the input matrix into a dense array
        if isinstance(matrix, galois.FieldArray):
            matrix = matrix.view(np.ndarray)
        elif isinstance(matrix, scipy.sparse.spmatrix):
            matrix = matrix.todense()

        self.matrix = np.asarray(matrix, dtype=int) % self.modulus
        _num_checks, num_variables = self.matrix.shape

        # variables, their constraints, and the objective (minimizing number of nonzero variables)
        self.variable_constraints = []
        if self.modulus == 2:
            self.variables = cvxpy.Variable(num_variables, boolean=True)
            self.objective = cvxpy.Minimize(cvxpy.norm(self.variables, 1))
        else:
            self.variables = cvxpy.Variable(num_variables, integer=True)
            nonzero_variable_flags = cvxpy.Variable(num_variables, boolean=True)
            self.variable_constraints += [var >= 0 for var in iter(self.variables)]
            self.variable_constraints += [var <= self.modulus - 1 for var in iter(self.variables)]
            self.variable_constraints += [self.modulus * nonzero_variable_flags >= self.variables]
            self.objective = cvxpy.Minimize(cvxpy.norm(nonzero_variable_flags, 1))

        self.decoder_args = decoder_args

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        import cvxpy

        # identify all constraints
        constraints = self.variable_constraints + self.cvxpy_constraints_for_syndrome(syndrome)

        # solve the optimization problem!
        problem = cvxpy.Problem(self.objective, constraints)
        result = problem.solve(**self.decoder_args)

        # raise error if the optimization failed
        if not isinstance(result, float) or not np.isfinite(result) or self.variables.value is None:
            message = "Optimal solution to integer linear program could not be found!"
            raise ValueError(message + f"\nSolver output: {result}")

        # return solution to the problem variables
        return self.variables.value.astype(syndrome.dtype)

    def cvxpy_constraints_for_syndrome(
        self, syndrome: npt.NDArray[np.int_]
    ) -> list[cvxpy.Constraint]:
        """Build cvxpy constraints of the form ``matrix @ variables == syndrome (mod q)``.

        This method uses boolean slack variables {s_j} to relax each constraint of the form
        ``expression = val mod q``
        to
        ``expression = val + sum_j q^j s_j``.
        """
        import cvxpy

        syndrome = np.asarray(syndrome, dtype=int) % self.modulus

        constraints = []
        for idx, (check, syndrome_bit) in enumerate(zip(self.matrix, syndrome)):
            # identify the largest power of q needed for the relaxation
            max_zero = int(sum(check) * (self.modulus - 1) - syndrome_bit)
            if max_zero == 0 or self.modulus == 2:
                max_power_of_q = max_zero.bit_length() - 1
            else:
                max_power_of_q = int(np.log2(max_zero) / np.log2(self.modulus))

            if max_power_of_q > 0:
                powers_of_q = [self.modulus**jj for jj in range(1, max_power_of_q + 1)]
                slack_variables = cvxpy.Variable(max_power_of_q, boolean=True)
                zero_mod_q = powers_of_q @ slack_variables
            else:
                zero_mod_q = 0

            constraint = check @ self.variables == syndrome_bit + zero_mod_q
            constraints.append(constraint)

        return constraints


class GUFDecoder:
    """The generalized Union-Find (GUF) decoder in https://arxiv.org/abs/2103.08049.

    If passed a max_weight argument, this decoder tries to find an error with
    ``weight <= max_weight``, and returns the first such error that it finds.  If no such error is
    found, this decoder returns the minimum-weight error that it found while trying.  Be warned that
    passing a max_weight makes this decoder have worst-case exponential runtime.

    If initialized with ``symplectic=True``, this decoder treats the provided parity check matrix as
    that of a QuditCode, with the first and last half of the columns denoting, respectively, the X
    and Z support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.

    Warning: this implementation of the generalized Union-Find decoder is highly unoptimized.  For
    one, it is written entirely in Python.  Moreover, this implementation does not factor an error
    set into connected components.
    """

    def __init__(
        self,
        matrix: IntegerArray,
        *,
        max_weight: int | None = None,
        symplectic: bool = False,
    ) -> None:
        matrix = np.asanyarray(matrix)

        self.default_max_weight = max_weight
        self.symplectic = symplectic

        # get_weight returns the weight of one error vector; the concrete type depends on the
        # backend: np.count_nonzero yields a scalar np.intp, while math.symplectic_weight yields an
        # ndarray (0-dimensional for a single vector).
        self.get_weight: Callable[[npt.NDArray[np.int_]], np.intp | npt.NDArray[np.int_]]
        self.code: codes.AbstractCode
        if not symplectic:
            # "ordinary" decoding of a classical code
            self.get_weight = np.count_nonzero  # Hamming weight (of an error vector)
            self.code = codes.ClassicalCode(matrix)

        else:
            # decoding a quantum code: the "weight" of an error vector is its symplectic weight
            self.get_weight = math.symplectic_weight
            self.code = codes.QuditCode(-math.symplectic_conjugate(matrix))

        self.graph = self.code.graph.to_undirected()

    def decode(
        self, syndrome: npt.NDArray[np.int_], *, max_weight: int | None = None
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        max_weight = max_weight if max_weight is not None else self.default_max_weight
        syndrome = syndrome.view(self.code.field)
        syndrome_bits = np.flatnonzero(syndrome)

        # construct an "error set", within which we look for solutions to the decoding problem
        error_set = {Node(int(index), is_data=False) for index in syndrome_bits}
        solutions = np.zeros((0, len(self.code)), dtype=int)
        last_error_set_size = 0
        while solutions.size == 0:
            # grow the error set by one step on the Tanner graph
            error_set |= {neighbor for node in error_set for neighbor in self.graph.neighbors(node)}

            # if the error set has not grown, there is no valid solution, so exit now
            if len(error_set) == last_error_set_size:
                return np.zeros(
                    len(self.code) * (2 if self.symplectic else 1),
                    dtype=syndrome.dtype,
                )
            last_error_set_size = len(error_set)

            # check whether the syndrome can be induced by errors in the interior of the error_set
            checks, bits = self.get_sub_problem_indices(syndrome, error_set)
            sub_matrix = self.code.matrix[np.ix_(checks, bits)]
            sub_syndrome = syndrome[checks]

            """
            Try to identify errors in the interior of the error_set that reproduce the syndrome,
            looking for solutions x to H @ x = s, or solutions [y,c] to [H|-s] @ [y,c].T = 0.
            """
            augmented_matrix = np.column_stack([sub_matrix, -sub_syndrome]).view(self.code.field)
            candidate_solutions = augmented_matrix.null_space()
            solutions = candidate_solutions[np.where(candidate_solutions[:, -1])]

        # convert solutions [y,c] --> [y/c,1] --> y
        if self.code.field is galois.GF2:
            converted_solutions = solutions[:, :-1]
        else:
            converted_solutions = solutions[:, :-1] / solutions[:, -1][:, None]

        # identify the minimum-weight solution found so far
        min_weight_solution = min(converted_solutions, key=self.get_weight)
        weight = self.get_weight(min_weight_solution)

        if max_weight is not None and weight > max_weight:
            # identify null-syndrome vectors
            null_vectors = sub_matrix.null_space()

            # minimize the weight of the solution over additions of null-syndrome vectors
            min_weight = weight
            one_solution = min_weight_solution.copy()
            null_vector_coefficients = itertools.product(
                self.code.field.elements, repeat=len(null_vectors)
            )
            next(null_vector_coefficients)  # skip the all-0 vector of coefficients
            for coefficients in null_vector_coefficients:
                solution = one_solution + self.code.field(coefficients) @ null_vectors
                weight = self.get_weight(solution)
                if weight < min_weight:
                    min_weight = weight
                    min_weight_solution = solution
                    if weight <= max_weight:
                        break

        # construct the full error
        error = self.code.field.Zeros(len(self.code) * (2 if self.symplectic else 1))
        error[bits] = min_weight_solution
        return error.view(np.ndarray).astype(syndrome.dtype)

    def get_sub_problem_indices(
        self, syndrome: npt.NDArray[np.int_], error_set: set[Node]
    ) -> tuple[list[int], list[int]]:
        """Syndrome and data bit indices for decoding on the interior of the given error set."""
        # identify the "interior" of error set: nodes whose neighbors are contained in the set
        interior_nodes = [
            node for node in error_set if error_set.issuperset(self.graph.neighbors(node))
        ]
        # identify interior data bit nodes, and their neighbors
        interior_data_nodes = [node for node in interior_nodes if node.is_data]
        check_nodes = {node for node in error_set if not node.is_data} | {
            neighbor for node in interior_data_nodes for neighbor in self.graph.neighbors(node)
        }
        checks = [node.index for node in check_nodes]
        bits = [node.index for node in interior_data_nodes]

        if self.symplectic:
            # add classical bits to account for the support of Z-type operators in the error vector
            bits += [bit + len(self.code) for bit in bits]

        # the order of checks, bits is technically arbitrary, but according to unofficial empirical
        # tests, reverse-sorted order works better for concatenated codes
        return sorted(checks, reverse=True), sorted(bits, reverse=True)


class CompositeDecoder:
    """Decoder for a composite syndrome from multiple independent code blocks.

    A CompositeDecoder is instantiated from a sequence of tuples, where each tuple contains

    (a) the decoder for a one code block
    (b) the length of a syndrome vector for that code block.

    When asked to decode a syndrome, a CompositeDecoder splits the syndrome into segments of
    appropriate lengths, and decodes these segments independently with their corresponding decoders.
    """

    def __init__(self, *decoders_and_syndrome_lengths: tuple[Decoder, int]) -> None:
        self.decoders, syndrome_lengths = zip(*decoders_and_syndrome_lengths)
        self.slices = tuple(
            slice(sum(syndrome_lengths[:ss]), sum(syndrome_lengths[: ss + 1]))
            for ss in range(len(syndrome_lengths))
        )

        self.decode_batch_implemented = all(
            hasattr(decoder, "decode_batch") for decoder in self.decoders
        )
        if self.decode_batch_implemented:
            self.decode_batch = self._decode_batch

    @staticmethod
    def from_copies(decoder: Decoder, syndrome_length: int, num_copies: int) -> CompositeDecoder:
        """Initialize a CompositeDecoder from copies of a given decoder and syndrome_length."""
        return CompositeDecoder(*[(decoder, syndrome_length)] * num_copies)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome by parts."""
        return np.hstack(
            [decoder.decode(syndrome[slice]) for decoder, slice in zip(self.decoders, self.slices)]
        )

    def _decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes by parts."""
        return (
            np.hstack(
                [
                    decoder.decode_batch(syndromes[:, slice])
                    for decoder, slice in zip(self.decoders, self.slices)
                ]
            )
            if self.decode_batch_implemented
            else NotImplemented
        )


class DirectDecoder:
    """Decoder that maps corrupted code words to corrected code words.

    In contrast, an "indirect" decoder maps a syndrome to an error.

    A DirectDecoder can be instantiated from:

    - an indirect decoder, and
    - a parity check matrix.

    When asked to decode a candidate code word, a DirectDecoder first computes a syndrome, decodes
    the syndrome with an indirect decoder to infer an error, and then subtracts the error from the
    candidate word.
    """

    def __init__(
        self,
        decode_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]],
        decode_batch_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]] | None = None,
    ) -> None:
        self.decode_func = decode_func
        self.decode_batch_func = decode_batch_func
        if decode_batch_func is not None:
            self.decode_batch = self._decode_batch

    def decode(self, word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a corrupted code word and return a corrected code word."""
        return self.decode_func(word)

    def _decode_batch(self, words: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of corrupted code words and return a batch of corrected code words."""
        return (
            self.decode_batch_func(words) if self.decode_batch_func is not None else NotImplemented
        )

    @staticmethod
    def from_indirect(decoder: Decoder, matrix: IntegerArray) -> DirectDecoder:
        """Instantiate a DirectDecoder from an indirect decoder and a parity check matrix."""
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF2
        field_matrix = matrix.view(field)

        def decode_func(candidate_word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            candidate_word = candidate_word.view(field)
            syndrome = field_matrix @ candidate_word
            error = decoder.decode(syndrome.view(np.ndarray)).view(field)
            return (candidate_word - error).view(np.ndarray)

        decode_batch_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]] | None = None

        if hasattr(decoder, "decode_batch"):

            def decode_batch_func(candidate_words: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
                candidate_words = candidate_words.view(field)
                syndromes = candidate_words @ field_matrix.T
                errors = decoder.decode_batch(syndromes.view(np.ndarray)).view(field)
                return (candidate_words - errors).view(np.ndarray)

        return DirectDecoder(decode_func, decode_batch_func)
