# SPDX-License-Identifier: Apache-2.0

"""Generalized Union-Find decoder."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import galois
import numpy as np
import numpy.typing as npt
import stim

from qldpc import math
from qldpc.math import IntegerArray
from qldpc.objects import Node

from ..common import _erasure_bit_support, _to_pcm, with_erasure_bits
from ..construction.specs import decoder_spec
from ..protocols import ErrorDecoder

if TYPE_CHECKING:
    from qldpc import codes


class GUFDecoder(ErrorDecoder):
    """The generalized Union-Find (GUF) decoder in https://arxiv.org/abs/2103.08049.

    If passed a max_weight argument, this decoder tries to find an error with
    ``weight <= max_weight``, and returns the first such error that it finds.  If no such error is
    found, this decoder returns the minimum-weight error that it found while trying.  Be warned that
    passing a max_weight makes this decoder have worst-case exponential runtime.

    If initialized with ``symplectic=True``, this decoder treats the provided parity check matrix as
    that of a QuditCode, with the first and last half of the columns denoting, respectively, the X
    and Z support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.

    If initialized with ``add_erasure_bit=True``, this decoder appends a bit to all decoded errors,
    set to 1 when its search exhausts without finding an error that reproduces the syndrome, and to
    0 otherwise.  Without that bit, an exhausted search is reported as the all-zero error, which is
    indistinguishable from the error inferred for a trivial syndrome.

    .. warning::
        This implementation of the generalized Union-Find decoder is highly unoptimized.  For one,
        it is written entirely in Python.  Moreover, this implementation does not factor an error
        set into connected components.
    """

    def __init__(
        self,
        matrix: IntegerArray,
        *,
        max_weight: int | None = None,
        symplectic: bool = False,
        add_erasure_bit: bool = False,
    ) -> None:
        from qldpc import codes

        matrix = np.asanyarray(matrix)

        self.default_max_weight = max_weight
        self.symplectic = symplectic
        self.has_erasure_bit = add_erasure_bit

        self.get_weight: Callable[[npt.NDArray[np.int_]], np.intp | npt.NDArray[np.int_]]
        self.code: codes.AbstractCode
        if not symplectic:
            self.get_weight = np.count_nonzero
            self.code = codes.ClassicalCode(matrix)
        else:
            self.get_weight = math.symplectic_weight
            field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF2
            self.code = codes.QuditCode(-math.symplectic_conjugate(matrix.view(field)))

        self.graph = self.code.graph.to_undirected()

    def decode_errors(
        self, syndrome: npt.NDArray[np.int_], *, max_weight: int | None = None
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error.

        If the search exhausts without finding an error that reproduces the given syndrome, return
        the all-zero error, whose appended erasure bit is set if this decoder tracks one.
        """
        max_weight = max_weight if max_weight is not None else self.default_max_weight
        syndrome = syndrome.view(self.code.field)
        syndrome_bits = np.flatnonzero(syndrome)

        error_set = {Node(int(index), is_data=False) for index in syndrome_bits}
        solutions = np.zeros((0, len(self.code)), dtype=int)
        last_error_set_size = 0
        while solutions.size == 0:
            error_set |= {neighbor for node in error_set for neighbor in self.graph.neighbors(node)}

            if len(error_set) == last_error_set_size:
                exhausted = np.zeros(
                    len(self.code) * (2 if self.symplectic else 1) + self.has_erasure_bit,
                    dtype=syndrome.dtype,
                )
                if self.has_erasure_bit and syndrome_bits.size:
                    exhausted[-1] = 1
                return exhausted
            last_error_set_size = len(error_set)

            checks, bits = self.get_sub_problem_indices(syndrome, error_set)
            sub_matrix = self.code.matrix[np.ix_(checks, bits)]
            sub_syndrome = syndrome[checks]

            augmented_matrix = np.column_stack([sub_matrix, -sub_syndrome]).view(self.code.field)
            candidate_solutions = augmented_matrix.null_space()
            solutions = candidate_solutions[np.where(candidate_solutions[:, -1])]

        if self.code.field is galois.GF2:
            converted_solutions = solutions[:, :-1]
        else:
            converted_solutions = solutions[:, :-1] / solutions[:, -1][:, None]

        min_weight_solution = min(converted_solutions, key=self.get_weight)
        weight = self.get_weight(min_weight_solution)

        if max_weight is not None and weight > max_weight:
            null_vectors = sub_matrix.null_space()

            min_weight = weight
            one_solution = min_weight_solution.copy()
            null_vector_coefficients = itertools.product(
                self.code.field.elements, repeat=len(null_vectors)
            )
            next(null_vector_coefficients)
            for coefficients in null_vector_coefficients:
                solution = one_solution + self.code.field(coefficients) @ null_vectors
                weight = self.get_weight(solution)
                if weight < min_weight:
                    min_weight = weight
                    min_weight_solution = solution
                    if weight <= max_weight:
                        break

        error = self.code.field.Zeros(len(self.code) * (2 if self.symplectic else 1))
        error[bits] = min_weight_solution
        decoded_error = error.view(np.ndarray).astype(syndrome.dtype)
        if self.has_erasure_bit:
            decoded_error = with_erasure_bits(decoded_error, False)
        return decoded_error

    def decode(
        self, syndrome: npt.NDArray[np.int_], *, max_weight: int | None = None
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error (alias for decode_errors)."""
        return self.decode_errors(syndrome, max_weight=max_weight)

    def get_sub_problem_indices(
        self, syndrome: npt.NDArray[np.int_], error_set: set[Node]
    ) -> tuple[list[int], list[int]]:
        """Syndrome and data bit indices for decoding on the interior of the given error set."""
        interior_nodes = [
            node for node in error_set if error_set.issuperset(self.graph.neighbors(node))
        ]
        interior_data_nodes = [node for node in interior_nodes if node.is_data]
        check_nodes = {node for node in error_set if not node.is_data} | {
            neighbor for node in interior_data_nodes for neighbor in self.graph.neighbors(node)
        }
        checks = [node.index for node in check_nodes]
        bits = [node.index for node in interior_data_nodes]

        if self.symplectic:
            bits += [bit + len(self.code) for bit in bits]

        return sorted(checks, reverse=True), sorted(bits, reverse=True)


@_erasure_bit_support("GUF", supported=True)
def _get_decoder_guf(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel, **decoder_args: object
) -> GUFDecoder:
    """Build a generalized union-find (GUF) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model to decode.  A DEM is converted to
            its dense detector-flip matrix.
        max_weight: Maximum weight of a candidate error, or None for no limit.
        symplectic: Whether to treat the parity-check matrix as that of a QuditCode, whose first
            and last halves of columns denote the X and Z support of a stabilizer.
        add_erasure_bit: Whether to append a flag when the search is exhausted without finding an
            error that reproduces the syndrome.
        **decoder_args: The options above, passed to
            :class:`~qldpc.decoders.custom.guf.GUFDecoder`.

    Returns:
        A :class:`~qldpc.decoders.custom.guf.GUFDecoder`.

    Supplying ``max_weight`` can make the search exponential.  See
    :class:`~qldpc.decoders.custom.guf.GUFDecoder` and
    `arXiv:2103.08049 <https://arxiv.org/abs/2103.08049>`_.
    """
    return GUFDecoder(_to_pcm(pcm_or_dem), **decoder_args)  # type: ignore[arg-type]


_GUF_SETTINGS_RETURNS = (
    "Decoder settings.  Their ``build(pcm_or_dem)`` method takes a parity-check matrix or "
    "detector error model (DEM), whose dense detector-flip matrix is decoded, and returns a "
    ":class:`~qldpc.decoders.custom.guf.GUFDecoder`."
)


guf = decoder_spec(
    "guf", _get_decoder_guf, signature_source=GUFDecoder, returns=_GUF_SETTINGS_RETURNS
)
