"""Custom decoder classes

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

import collections
import functools
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

import cvxpy
import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc import codes, math
from qldpc.abstract import DEFAULT_FIELD_ORDER
from qldpc.math import IntegerArray
from qldpc.objects import Node

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


class RelayBPDecoder(BatchDecoder):
    """Wrapper class for Relay-BP decoders, introduced in arXiv:2506.01779.

    Requires relay_bp to be installed, for example via "pip install 'qldpc[relay-bp]'".

    This class first constructs a relay_bp.decoder.DynDecoder decoder by class name, such as
    "RelayDecoderF32"; see help(relay_bp) for more options.  To enable parallelized decoding, which
    which as of relay-bp==0.1.0 is only implemented for the relay_bp.ObservableDecoderRunner class,
    RelayBPDecoder wraps the relay_bp.decoder.DynDecoder in a relay_bp.ObservableDecoderRunner at
    initialization time.

    IMPORTANT POINTS TO NOTE:
    -------------------------
    1. relay_bp.ObservableDecoderRunner expects to be passed an observable_error_matrix when
        initialized.  If a RelayBPDecoder is initialized without an observable_error_matrix, this
        matrix is set to np.empty((0, 0), dtype=int).  All observable-related methods of the decoder
        will subsequently fail.
    2. RelayBPDecoder "wants" to be a subclass of relay_bp.ObservableDecoderRunner.  However, the
        latter does not allow subclassing because it is implemented in rust and exposed to Python
        via bindings.  As a hack, if a decoder: RelayBPDecoder is asked for a method or attribute it
        does not recognize, such as decoder.decode_observables_batch(detectors, parallel=True) or
        decoder.decode_detailed(detectors), it tries to pass all arguments to an identically-named
        method of relay_bp.ObservableDecoderRunner.  A consequence of this hack is that most of the
        methods that are recognized by RelayBPDecoder in practice do not appear in its documentation.
        See help(relay_bp.ObservableDecoderRunner) for a list of all RelayBPDecoder methods.

    For details about Relay-BP decoders, see:
    - Documentation: https://pypi.org/project/relay-bp
    - Reference: https://arxiv.org/abs/2506.01779
    """

    def __init__(
        self,
        name: str,
        matrix: IntegerArray,
        error_priors: npt.NDArray[np.float64] | Sequence[float] | None,
        *,
        observable_error_matrix: IntegerArray | None = None,
        include_decode_result: bool = False,
    ) -> None:
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

        # sanitize inputs
        if isinstance(matrix, galois.FieldArray):
            matrix = matrix.view(np.ndarray)
        elif isinstance(matrix, scipy.sparse.spmatrix):
            matrix = matrix.tocsc()
            matrix.sort_indices()  # type:ignore[union-attr]
        if error_priors is None:
            error_priors = [PLACEHOLDER_ERROR_RATE] * matrix.shape[1]
        if observable_error_matrix is None:
            observable_error_matrix = np.empty((0, 0), dtype=int)

        self.decoder = relay_bp.ObservableDecoderRunner(
            getattr(relay_bp, name)(matrix, np.asarray(error_priors)),
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


class LookupDecoder(Decoder):
    """Decoder based on a lookup table that maps syndromes to errors.

    In addition to a parity check matrix, this decoder needs to be initialized with a max_weight.
    The decoder consider then enumerates all errors with weight <= max_weight, in order of
    decreasing weight.  For each error ee, it computes the corresponding syndrome ss, and assigns
    syndrome ss the "correction" ee, overriding any previously assigned correction if present.

    If provided a penalty_func that maps an error to a real number (i.e., a penalty), the decoder
    only assigns correction ee to syndrome ss if (a) ss has no assigned correction, or (b) the
    penalty of ee is smaller than the penalty of the correction currently assigned to ss.

    If provided an error_channel of independent probabilities for each "error mechanism" (associated
    with one column of the parity check matrix), construct a penalty_func that penalizes unlikely
    errors.

    If initialized with symplectic=True, this decoder treats the provided parity check matrix as that
    of a QuditCode, with the first and last half of the columns denoting, respectively, the X and Z
    support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.
    """

    def __init__(
        self,
        matrix: IntegerArray,
        max_weight: int,
        *,
        symplectic: bool = False,
        error_channel: npt.NDArray[np.float64] | Sequence[float] | None = None,
        penalty_func: Callable[[npt.NDArray[np.int_] | Sequence[int]], float] | None = None,
    ) -> None:
        assert error_channel is None or penalty_func is None, (
            "Cannot specify both an error_channel and a penalty_func"
        )
        penalty_func = penalty_func or (
            self.build_penalty_func(error_channel) if error_channel is not None else None
        )

        self.shape: tuple[int, ...] = matrix.shape
        self.syndrome_to_correction = {}

        error_weights: dict[tuple[int, ...], float] = {}
        for error, syndrome in LookupDecoder.iter_errors_and_syndomes(
            matrix, max_weight, symplectic
        ):
            if penalty_func is None:
                self.syndrome_to_correction[syndrome] = error
            elif (error_weight := penalty_func(error)) <= error_weights.get(syndrome, np.inf):
                error_weights[syndrome] = error_weight
                self.syndrome_to_correction[syndrome] = error

    @staticmethod
    def iter_errors_and_syndomes(
        matrix: IntegerArray, max_weight: int, symplectic: bool
    ) -> Iterator[tuple[npt.NDArray[np.int_], tuple[int, ...]]]:
        """Iterate over all errors that this decoder considers, and their associated syndromes.

        Errors are sorted in decreasing weight (number of bits/qudits addressed nontrivially).
        """
        code = codes.ClassicalCode(matrix) if not symplectic else codes.QuditCode(matrix)
        matrix = code.matrix if not symplectic else math.symplectic_conjugate(code.matrix)

        # identify the set of local errors that can occur
        repeat = 2 if symplectic else 1
        error_ops = tuple(itertools.product(range(code.field.order), repeat=repeat))[1:]

        block_length = matrix.shape[1] // repeat
        for weight in range(max_weight, -1, -1):
            for error_sites in itertools.combinations(range(block_length), weight):
                error_site_indices = list(error_sites)
                for local_errors in itertools.product(error_ops, repeat=weight):
                    error = code.field.Zeros((repeat, block_length))
                    error[:, error_site_indices] = np.asarray(local_errors, dtype=int).T
                    error = error.ravel()
                    syndrome = matrix @ error
                    yield error.view(np.ndarray), tuple(syndrome.view(np.ndarray))

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        return self.syndrome_to_correction.get(
            tuple(syndrome.view(np.ndarray)), np.zeros(self.shape[1], dtype=int)
        )

    @staticmethod
    def build_penalty_func(
        error_channel: npt.NDArray[np.float64] | Sequence[float],
    ) -> Callable[[npt.NDArray[np.int_] | Sequence[int]], float]:
        """Construct a penalty function from independent probabilities of individual errors."""
        error_channel = np.asarray(error_channel)
        log_probs = np.log(error_channel)
        log_non_probs = np.log(1 - error_channel)

        def penalty_func(error: npt.NDArray[np.int_] | Sequence[int]) -> float:
            """Penalize unlikely combinations of errors."""
            events = np.asarray(error).astype(bool)
            log_probability_of_error = np.sum(log_probs[events]) + np.sum(log_non_probs[~events])
            return -float(log_probability_of_error)

        return penalty_func


class WeightedLookupDecoder(LookupDecoder):
    """Decoder based on a lookup table that maps syndromes to errors.

    A WeightedLookupDecoder is a LookupDecoder that, when initialized, records *all* errors that are
    consistent with a given syndrome.  The WeightedLookupDecoder then minimizes a penalty function
    that is provided to the .decode method.  A WeightedLookupDecoder can thereby be initialized
    once, and subsequently asked to decode with different penalty functions.
    """

    def __init__(
        self,
        matrix: IntegerArray,
        max_weight: int,
        *,
        symplectic: bool = False,
    ) -> None:
        self.shape: tuple[int, ...] = matrix.shape
        self.syndrome_to_candidates: dict[tuple[int, ...], list[npt.NDArray[np.int_]]] = (
            collections.defaultdict(list)
        )
        for error, syndrome in LookupDecoder.iter_errors_and_syndomes(
            matrix, max_weight, symplectic
        ):
            self.syndrome_to_candidates[syndrome].append(error)

    def decode(
        self,
        syndrome: npt.NDArray[np.int_],
        penalty_func: Callable[[npt.NDArray[np.int_]], float] = lambda vec: int(
            np.count_nonzero(vec)
        ),
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        errors = self.syndrome_to_candidates.get(
            tuple(syndrome.view(np.ndarray)), [np.zeros(self.shape[1], dtype=int)]
        )
        return min(errors, key=penalty_func) if penalty_func is not None else errors[-1]


class ILPDecoder(Decoder):
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """

    def __init__(self, matrix: IntegerArray, **decoder_args: object) -> None:
        self.modulus = type(matrix).order if isinstance(matrix, galois.FieldArray) else 2
        if not galois.is_prime(self.modulus):
            raise ValueError("ILP decoding only supports prime number fields")

        self.matrix = np.array(matrix, dtype=int) % self.modulus
        num_checks, num_variables = self.matrix.shape

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
        return self.variables.value.astype(int)

    def cvxpy_constraints_for_syndrome(
        self, syndrome: npt.NDArray[np.int_]
    ) -> list[cvxpy.Constraint]:
        """Build cvxpy constraints of the form `matrix @ variables == syndrome (mod q)`.

        This method uses boolean slack variables {s_j} to relax each constraint of the form
        `expression = val mod q`
        to
        `expression = val + sum_j q^j s_j`.
        """
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


class GUFDecoder(Decoder):
    """The generalized Union-Find (GUF) decoder in https://arxiv.org/abs/2103.08049.

    If passed a max_weight argument, this decoder tries to find an error with weight <= max_weight,
    and returns the first such error that it finds.  If no such error is found, this decoder returns
    the minimum-weight error that it found while trying.  Be warned that passing a max_weight makes
    this decoder have worst-case exponential runtime.

    If initialized with symplectic=True, this decoder treats the provided parity check matrix as that
    of a QuditCode, with the first and last half of the columns denoting, respectively, the X and Z
    support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.

    Warning: this implementation of the generalized Union-Find decoder is highly unoptimized.  For
    one, it is written entirely in Python.  Moreover, this implementation does not factor an error
    set into connected componenents.
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

        self.get_weight: Callable[[npt.NDArray[np.int_]], int]
        self.code: codes.AbstractCode
        if not symplectic:
            # "ordinary" decoding of a classical code
            self.get_weight = np.count_nonzero  # Hamming weight (of an error vector)
            self.code = codes.ClassicalCode(matrix)

        else:
            # decoding a quantum code: the "weight" of an error vector is its symplectic weight
            self.get_weight = math.symplectic_weight
            self.code = codes.QuditCode(math.symplectic_conjugate(matrix))

        self.graph = self.code.graph.to_undirected()

    def decode(
        self, syndrome: npt.NDArray[np.int_], *, max_weight: int | None = None
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        max_weight = max_weight if max_weight is not None else self.default_max_weight
        syndrome = np.asarray(syndrome, dtype=int).view(self.code.field)
        syndrome_bits = np.where(syndrome)[0]

        # construct an "error set", within which we look for solutions to the decoding problem
        error_set = set(Node(index, is_data=False) for index in syndrome_bits)
        solutions = np.zeros((0, len(self.code)), dtype=int)
        last_error_set_size = 0
        while solutions.size == 0:
            # grow the error set by one step on the Tanner graph
            error_set |= set(
                neighbor for node in error_set for neighbor in self.graph.neighbors(node)
            )

            # if the error set has not grown, there is no valid solution, so exit now
            if len(error_set) == last_error_set_size:
                return np.zeros(len(self.code) * (2 if self.symplectic else 1), dtype=int)
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
        if self.code.field.order == 2:
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
        return error.view(np.ndarray)

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
        check_nodes = set(node for node in error_set if not node.is_data) | set(
            neighbor for node in interior_data_nodes for neighbor in self.graph.neighbors(node)
        )
        checks = [node.index for node in check_nodes]
        bits = [node.index for node in interior_data_nodes]

        if self.symplectic:
            # add classical bits to account for the support of Z-type operators in the error vector
            bits += [bit + len(self.code) for bit in bits]

        # the order of checks, bits is technically arbitrary, but according to unofficial empirical
        # tests, reverse-sorted order works better for concatenated codes
        return sorted(checks, reverse=True), sorted(bits, reverse=True)


class CompositeDecoder(Decoder):
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


class DirectDecoder(Decoder):
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
        field = (
            type(matrix)
            if isinstance(matrix, galois.FieldArray)
            else galois.GF(DEFAULT_FIELD_ORDER)
        )
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
