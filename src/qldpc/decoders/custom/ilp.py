# SPDX-License-Identifier: Apache-2.0

"""Integer-linear-program decoder."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import stim

from qldpc.math import IntegerArray

from ..common import _erasure_bit_support, _to_pcm, with_erasure_bits
from ..construction.specs import decoder_spec
from ..protocols import ErrorDecoder, ErrorDecodeResult

if TYPE_CHECKING:
    import cvxpy


class ILPDecoder(ErrorDecoder):
    """Decoder based on solving an integer linear program (ILP).

    An integer program that is allowed to run to completion either finds an error of minimum weight
    that reproduces the syndrome or proves that no error reproduces it, so an inferred error that
    does not reproduce the syndrome means the solver stopped early, at a point it never proved
    feasible.  ``time_limit`` and ``mip_max_nodes`` are the arguments that ask it to stop early.

    If initialized with ``add_erasure_bit=True``, this decoder appends a bit to all decoded errors,
    set to 1 for a syndrome that it cannot explain and to 0 otherwise.  Without that bit there is no
    way to report such a syndrome, so it is rejected instead.

    A syndrome goes unexplained either because the inferred error does not reproduce it, or because
    the program reports no solution for it at all.  The second case also warns, since a program
    reports no solution both when it proves that no error reproduces the syndrome and when it fails.

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """

    def __init__(
        self, matrix: IntegerArray, *, add_erasure_bit: bool = False, **decoder_args: object
    ) -> None:
        import cvxpy

        self.has_erasure_bit = add_erasure_bit

        self.modulus = type(matrix).order if isinstance(matrix, galois.FieldArray) else 2
        if not galois.is_prime(self.modulus):
            raise ValueError("ILP decoding only supports prime number fields")

        if isinstance(matrix, galois.FieldArray):
            matrix = matrix.view(np.ndarray)
        elif isinstance(matrix, scipy.sparse.spmatrix):
            matrix = matrix.todense()

        self.matrix = np.asarray(matrix, dtype=int) % self.modulus
        _num_checks, num_variables = self.matrix.shape

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

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        error, erased, _ = self._solve(syndrome, self.has_erasure_bit)
        return with_erasure_bits(error, erased) if self.has_erasure_bit else error

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        """Decode one syndrome and report the solver status and objective value.

        The erasure flag is set if the inferred error does not reproduce the syndrome, or (with a
        warning) if no optimal solution is found.
        """
        error, erased, problem = self._solve(syndrome, signal_erasure=True)
        if self.has_erasure_bit:
            error = with_erasure_bits(error, erased)
        diagnostics: dict[str, object] = {"ilp.status": str(problem.status)}
        if isinstance(problem.value, float) and np.isfinite(problem.value):
            diagnostics["ilp.objective_value"] = problem.value
        return ErrorDecodeResult(error, erased, diagnostics)

    def cvxpy_constraints_for_syndrome(
        self, syndrome: npt.NDArray[np.int_]
    ) -> list[cvxpy.Constraint]:
        """Build cvxpy constraints of the form ``matrix @ variables == syndrome (mod q)``.

        This method relaxes each constraint of the form
        ``expression = val mod q``
        to
        ``expression = val + q t``,
        where t is a nonnegative integer built out of boolean variables {b_j} as
        ``t = sum_j 2^j b_j``.

        Since the variables are nonnegative and val is reduced mod q, ``expression - val`` is a
        nonnegative multiple of q, so t is nonnegative, and it is bounded above by the largest value
        that ``expression`` can take, less val, in units of q.  Enough bits to reach that bound
        reach every value below it too, since a binary expansion represents every integer in range.
        """
        import cvxpy

        syndrome = np.asarray(syndrome, dtype=int) % self.modulus

        constraints = []
        for check, syndrome_bit in zip(self.matrix, syndrome):
            max_offset = int(sum(check) * (self.modulus - 1) - syndrome_bit)

            num_bits = (max_offset // self.modulus).bit_length() if max_offset > 0 else 0
            if not num_bits:
                zero_mod_q: Any = 0
            else:
                slack_bits = cvxpy.Variable(num_bits, boolean=True)
                zero_mod_q = [self.modulus * 2**jj for jj in range(num_bits)] @ slack_bits

            constraint = check @ self.variables == syndrome_bit + zero_mod_q
            constraints.append(constraint)

        return constraints

    def _solve(
        self, syndrome: npt.NDArray[np.int_], signal_erasure: bool
    ) -> tuple[npt.NDArray[np.int_], bool, cvxpy.Problem]:
        """Solve the integer linear program, and return its error, erasure flag, and problem.

        If not signaling erasure, raise an error instead of returning an erased result.
        """
        import cvxpy

        constraints = self.variable_constraints + self.cvxpy_constraints_for_syndrome(syndrome)

        problem = cvxpy.Problem(self.objective, constraints)
        result = problem.solve(**self.decoder_args)

        if not isinstance(result, float) or not np.isfinite(result) or self.variables.value is None:
            message = (
                "Optimal solution to integer linear program could not be found!"
                f"\nSolver output: {result}"
            )
            if not signal_erasure:
                raise ValueError(message)
            warnings.warn(message, stacklevel=3)
            return np.zeros(self.matrix.shape[1], dtype=syndrome.dtype), True, problem

        error = (np.rint(self.variables.value) % self.modulus).astype(int)

        reproduces_syndrome = np.array_equal(
            self.matrix @ error % self.modulus,
            np.asarray(syndrome, dtype=int) % self.modulus,
        )
        if not signal_erasure and not reproduces_syndrome:
            raise ValueError(
                "Integer linear program returned an error that does not reproduce the syndrome!"
                f"\nSolver status: {problem.status}"
            )
        return error, not reproduces_syndrome, problem


@_erasure_bit_support("ILP", supported=True)
def _get_decoder_ilp(
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    *,
    add_erasure_bit: bool = False,
    **decoder_args: object,
) -> ILPDecoder:
    """Build an integer-linear-program (ILP) decoder.

    Args:
        pcm_or_dem: A parity-check matrix or detector error model to decode.  A DEM is converted to
            its dense detector-flip matrix.
        add_erasure_bit: Whether to append a flag when the solver cannot produce an error that
            reproduces the syndrome.
        **decoder_args: Arguments passed to ``cvxpy.Problem.solve``.

    Returns:
        An :class:`~qldpc.decoders.custom.ilp.ILPDecoder`.

    ILP decoding supports prime fields.  Without an erasure bit, an unexplained syndrome is rejected
    rather than returned as an ordinary inferred error.
    """
    return ILPDecoder(_to_pcm(pcm_or_dem), add_erasure_bit=add_erasure_bit, **decoder_args)


_ILP_SPEC_RETURNS = (
    "A decoder specification.  Its ``build(pcm_or_dem)`` method takes a parity-check matrix or "
    "detector error model (DEM), whose dense detector-flip matrix is decoded, and returns an "
    ":class:`~qldpc.decoders.custom.ilp.ILPDecoder`."
)

ilp = decoder_spec("ilp", _get_decoder_ilp, signature_source=ILPDecoder, returns=_ILP_SPEC_RETURNS)
