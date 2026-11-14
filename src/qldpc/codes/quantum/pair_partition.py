# SPDX-License-Identifier: Apache-2.0

"""Pair-partition codes: quasi-cyclic CSS codes built from circulant permutation matrices."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt
import sympy

from qldpc import abstract

from ..common import CSSCode, _mark_read_only
from .two_block import _get_block_translations

####################################################################################################
# pair-partition codes


class PairPartitionCode(CSSCode):
    """Pair-partition code: a quasi-cyclic CSS code built from circulant permutation matrices.

    A pair-partition code is a binary CSS code whose X-type and Z-type parity check matrices are
    ``J x L`` arrays of ``P x P`` circulant permutation matrices (CPMs),

    - ``matrix_x = [[C(e_{il})]]``, and
    - ``matrix_z = [[C(d_{jl})]]``.

    Here ``e_{il}`` and ``d_{jl}`` are the entries of the integer arrays ``exponents_x`` and
    ``exponents_z``, and ``P`` is ``lift_size``.  The CPM ``C(s)`` has the nonzero entry of its row
    ``r`` in column ``r - s`` (mod ``P``).  It is therefore the lift of ``x**s`` in the group
    algebra ``F_2[Z_P]``, where ``x`` generates the cyclic group ``Z_P``.  The code has ``L * P``
    qubits and ``J * P`` parity checks of each type.  Every parity check has weight ``L``, and every
    qubit is addressed by ``J`` parity checks of each type.  Qubit ``l * P + r`` is qubit ``r`` of
    block column ``l``.  The repositories in the references below interleave block columns instead,
    which permutes qubits.

    The parity check matrices are CPM lifts of the all-ones ``J x L`` base matrix, as in the
    standard construction of quasi-cyclic LDPC codes.  The block of ``matrix_x @ matrix_z.T`` in
    block row ``i`` and block column ``j`` is ``sum_l C(e_{il} - d_{jl})``.  Distinct CPMs have
    disjoint supports, so this block vanishes over ``F_2`` if and only if the ``L`` block columns
    of the parity check matrices split into pairs ``{u, v}`` that satisfy the paired-difference
    equation ``d_{ju} - e_{iu} = d_{jv} - e_{iv}`` (mod ``P``).  The two CPMs of each pair then
    coincide and cancel.  A ``J x J`` array of such pair partitions, one for each pair of block rows
    ``(i, j)``, is called a pair-partition array.  PairPartitionCode.from_pair_partition_array
    builds codes from a pair-partition array.

    The relabelings ``e_{il} -> e_{il} + tau_l + alpha_i`` and ``d_{jl} -> d_{jl} + tau_l + beta_j``
    preserve the paired-difference equations, and change the code only by permuting qubits and
    parity checks.  Two further properties of the exponents are heuristics for better
    belief-propagation decoding.  The first is the absence of 4-cycles in the Tanner graphs of
    ``matrix_x`` and ``matrix_z`` (see PairPartitionCode.has_4_cycles).  The second is
    mixed-distinctness, which makes every X-type check overlap every Z-type check on zero or two
    qubits (see PairPartitionCode.is_mixed_distinct).

    Translating every block column of qubits by the same cyclic shift maps the checks of each type
    onto themselves.  Exact distance searches use these translations to search less if the optional
    pybliss package is not installed, or finds no symmetries of the code.

    PairPartitionCode does not record a code distance.  The ``[[276, 98, 14]]`` code below has
    ``(J, L, P) = (4, 12, 23)``, and is instance ``qc_276_98_14`` in the repository of
    arXiv:2607.14091.  Its distance of 14 is taken from the references, rather than computed::

        from qldpc import codes

        exponents_x = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 10, 16, 12, 20, 9, 6, 11, 2, 5, 21, 22],
            [0, 7, 20, 11, 3, 12, 2, 22, 14, 19, 13, 6],
            [0, 19, 17, 4, 2, 21, 12, 10, 18, 3, 11, 9],
        ]
        exponents_z = [
            [0, 11, 18, 6, 13, 1, 0, 11, 18, 6, 13, 1],
            [0, 12, 7, 22, 19, 10, 7, 10, 12, 19, 0, 22],
            [0, 5, 6, 1, 4, 11, 1, 0, 4, 5, 11, 6],
            [0, 8, 22, 21, 12, 20, 12, 22, 0, 20, 21, 8],
        ]
        code = codes.PairPartitionCode(exponents_x, exponents_z, lift_size=23)
        assert (code.num_qubits, code.dimension) == (276, 98)

    References:

    - https://arxiv.org/abs/2609.30069, which subsumes https://arxiv.org/abs/2607.14091
    - https://errorcorrectionzoo.org/c/quasi_cyclic_qldpc
    - https://github.com/ultra-high-rate-qec/design-principles-data (instances of arXiv:2609.30069)
    - https://github.com/kasaikenta/pair-partition-cpm-css-codes (instances of arXiv:2607.14091)

    Args:
        exponents_x: A ``J x L`` array of integer exponents ``e_{il}`` for the X-type checks.
        exponents_z: A ``J x L`` array of integer exponents ``d_{jl}`` for the Z-type checks.
        lift_size: The size ``P`` of the circulant permutation matrices, a positive integer.
            Exponents are stored modulo ``P``.

    Keyword args:
        promise_equal_distance_xz: Passed to CSSCode: a promise, which is not verified, that the
            X-distance and Z-distance of this code are equal.  Default: False.
        skip_validation: If True, skip the check that ``matrix_x @ matrix_z.T == 0``.  If the
            exponents then violate CSS orthogonality, the result is a subsystem code.  Structural
            input validation is performed regardless.  Default: False.
    """

    exponents_x: npt.NDArray[np.int_]
    exponents_z: npt.NDArray[np.int_]
    lift_size: int

    def __init__(
        self,
        exponents_x: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        exponents_z: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        lift_size: int,
        *,
        promise_equal_distance_xz: bool = False,
        skip_validation: bool = False,
    ) -> None:
        """Construct a pair-partition code from two exponent arrays and a cyclic lift size."""
        if lift_size < 1:
            raise ValueError(f"The lift size must be positive (got {lift_size})")
        self.lift_size = lift_size
        self.exponents_x = _mark_read_only(self._validate_exponents(exponents_x, "X") % lift_size)
        self.exponents_z = _mark_read_only(self._validate_exponents(exponents_z, "Z") % lift_size)
        if self.exponents_x.shape != self.exponents_z.shape:
            raise ValueError(
                "The X-type and Z-type exponent arrays of a PairPartitionCode must have the same"
                f" shape (got shapes {self.exponents_x.shape} and {self.exponents_z.shape})"
            )

        matrix_x = self.ring_array_x.lift()
        matrix_z = self.ring_array_z.lift()
        if not skip_validation and np.any(matrix_x @ matrix_z.T):
            raise ValueError(
                "The exponent arrays provided for this PairPartitionCode violate CSS orthogonality"
                " (matrix_x @ matrix_z.T != 0); see help(qldpc.codes.PairPartitionCode)"
            )
        super().__init__(matrix_x, matrix_z, promise_equal_distance_xz=promise_equal_distance_xz)
        if not skip_validation:
            self._is_subsystem_code = False  # verified above

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = f"{self.name} on {self.num_qubits} qubits"
        text += f", with cyclic lift size {self.lift_size} and exponent arrays"
        text += f"\n  X: {self.exponents_x.tolist()}"
        text += f"\n  Z: {self.exponents_z.tolist()}"
        return text

    @property
    def ring_array_x(self) -> abstract.RingArray:
        """The X-type parity check matrix as an array of monomials in F_2[Z_P]."""
        return self._get_ring_array(self.exponents_x)

    @property
    def ring_array_z(self) -> abstract.RingArray:
        """The Z-type parity check matrix as an array of monomials in F_2[Z_P]."""
        return self._get_ring_array(self.exponents_z)

    def has_4_cycles(self) -> bool:
        """Does the Tanner graph of matrix_x or matrix_z contain a 4-cycle?

        The Tanner graph of ``matrix_x`` has a 4-cycle if and only if, for some two block rows
        ``i != i'``, the differences ``e_{il} - e_{i'l}`` (mod ``P``) take the same value at two
        block columns ``l != l'``.  The same holds for ``matrix_z``.  See
        https://doi.org/10.1109/TIT.2004.831841.
        """
        for exponents in [self.exponents_x, self.exponents_z]:
            for row_a, row_b in itertools.combinations(exponents, 2):
                diffs = (row_a - row_b) % self.lift_size
                if np.bincount(diffs).max() > 1:
                    return True
        return False

    def is_mixed_distinct(self) -> bool:
        """Does every X-type check overlap every Z-type check on at most two qubits?

        The X-type check in row ``r`` of block row ``i`` and the Z-type check in row ``r'`` of block
        row ``j`` overlap in block column ``l`` exactly when ``d_{jl} - e_{il} = r' - r`` (mod
        ``P``).  If the code is CSS-orthogonal, every such overlap has an even size, so this method
        is equivalent to the mixed-distinctness condition of arXiv:2609.30069: distinct pairs
        ``{u, v}`` in each pair partition of a pair-partition array have distinct differences
        ``d_{ju} - e_{iu}``.  Mixed-distinctness reduces the number of 4-cycles that alternate
        between X-type and Z-type checks.
        """
        for row_x in self.exponents_x:
            for row_z in self.exponents_z:
                diffs = (row_z - row_x) % self.lift_size
                if np.bincount(diffs).max() > 2:
                    return False
        return True

    @staticmethod
    def solve_pair_partition_array(
        matchings: Sequence[Sequence[Sequence[Sequence[int]]]],
        lift_size: int,
        *,
        fix_gauge: bool = False,
    ) -> galois.FieldArray:
        """Solve the paired-difference equations of a pair-partition array.

        A pair-partition array is a ``J x J`` array whose entry ``matchings[i][j]`` is a perfect
        matching of ``range(L)``: a sequence of ``L / 2`` disjoint pairs ``(u, v)`` that together
        cover ``0, 1, ..., L - 1``.  Each pair imposes the linear equation
        ``d_{ju} - e_{iu} = d_{jv} - e_{iv}`` over ``F_P`` on the exponents of a PairPartitionCode,
        and every solution defines a valid CSS code (see help(qldpc.codes.PairPartitionCode)).

        As in arXiv:2609.30069, this method requires a prime lift size ``P``, so that it can solve
        the equations over the field ``F_P``.  Codes with a composite lift size can be constructed
        by passing exponents to PairPartitionCode directly.

        The relabelings in help(qldpc.codes.PairPartitionCode) account for ``2 * J + L - 1``
        dimensions of the solution space.  If ``fix_gauge`` is True, this method removes them by
        imposing ``e_{0l} = 0`` for all ``l`` (as in arXiv:2609.30069), ``e_{i0} = 0`` for all
        ``i``, and ``d_{j0} = 0`` for all ``j``.  Every solution is then related by a relabeling to
        exactly one gauge-fixed solution.

        Args:
            matchings: A ``J x J`` pair-partition array.
            lift_size: The size ``P`` of the circulant permutation matrices, a prime number.
            fix_gauge: If True, restrict the solution space by the gauge-fixing conditions above.
                Default: False.

        Returns:
            A basis for the solution space over ``F_P``, as an array with shape ``(K, 2, J, L)``,
            where ``K`` is the dimension of the solution space.  Basis vector ``basis[m]`` stacks
            the exponent arrays ``exponents_x = basis[m, 0]`` and ``exponents_z = basis[m, 1]``.
        """
        if not sympy.isprime(lift_size):
            raise ValueError(
                f"The lift size of a pair-partition array must be prime (got {lift_size})"
            )
        num_rows = len(matchings)  # J
        if num_rows == 0 or any(len(matching_row) != num_rows for matching_row in matchings):
            raise ValueError("A pair-partition array must be nonempty and square")
        num_cols = 2 * len(matchings[0][0])  # L

        # each pair imposes -e_{iu} + e_{iv} + d_{ju} - d_{jv} = 0 on the exponents (e, d)
        equations = []
        for i, matching_row in enumerate(matchings):
            for j, matching in enumerate(matching_row):
                cols = sorted(col for pair in matching for col in pair)
                if (
                    num_cols == 0
                    or any(len(pair) != 2 for pair in matching)
                    or cols != list(range(num_cols))
                ):
                    raise ValueError(
                        f"matchings[{i}][{j}] is not a perfect matching of range({num_cols}) into"
                        " pairs"
                    )
                for uu, vv in matching:
                    equation = np.zeros((2, num_rows, num_cols), dtype=int)
                    equation[0, i, [uu, vv]] = -1, 1
                    equation[1, j, [uu, vv]] = 1, -1
                    equations.append(equation.ravel())

        if fix_gauge:
            fixed = np.zeros((2, num_rows, num_cols), dtype=bool)
            fixed[0, 0, :] = fixed[:, :, 0] = True
            equations.extend(np.eye(fixed.size, dtype=int)[fixed.ravel()])

        field = galois.GF(lift_size)
        null_space = field(np.array(equations) % lift_size).null_space()
        return null_space.reshape(-1, 2, num_rows, num_cols).view(field)

    @classmethod
    def from_pair_partition_array(
        cls,
        matchings: Sequence[Sequence[Sequence[Sequence[int]]]],
        lift_size: int,
        coefficients: Sequence[int] | None = None,
        *,
        fix_gauge: bool = False,
        seed: int | np.random.Generator | None = None,
    ) -> PairPartitionCode:
        """Construct a pair-partition code from a pair-partition array.

        The solutions to the paired-difference equations of a pair-partition array form a linear
        family of codes (see PairPartitionCode.solve_pair_partition_array).  This method builds the
        member of that family whose coefficients in the basis returned by
        solve_pair_partition_array are ``coefficients``.  If ``coefficients`` is None, this method
        builds a uniformly random member instead.

        Every member of a family is CSS-orthogonal, but its girth, dimension, and distance depend on
        the member.  The code search of arXiv:2609.30069 samples random members, and rejects those
        with 4-cycles (see PairPartitionCode.has_4_cycles) or without mixed-distinctness (see
        PairPartitionCode.is_mixed_distinct).  Often most members are rejected, especially when
        ``P`` is small.

        Args:
            matchings: A ``J x J`` pair-partition array (see solve_pair_partition_array).
            lift_size: The size ``P`` of the circulant permutation matrices, a prime number.
            coefficients: The coefficients of the chosen member in the basis returned by
                solve_pair_partition_array, or None to choose a random member.  Default: None.
            fix_gauge: Passed to solve_pair_partition_array.  Default: False.
            seed: A seed or random number generator for choosing a random member.  Default: None.
        """
        basis = cls.solve_pair_partition_array(matchings, lift_size, fix_gauge=fix_gauge)
        field = type(basis)
        if coefficients is None:
            combination = field(np.random.default_rng(seed).integers(lift_size, size=len(basis)))
        elif len(coefficients) != len(basis):
            raise ValueError(
                f"Expected {len(basis)} coefficients for this pair-partition array, got"
                f" {len(coefficients)}"
            )
        else:
            combination = field(np.array(coefficients, dtype=int) % lift_size)
        exponents = combination @ basis.reshape(len(basis), math.prod(basis.shape[1:]))
        exponents_x, exponents_z = np.array(exponents.reshape(basis.shape[1:]), dtype=int)
        return cls(exponents_x, exponents_z, lift_size)

    def _get_qudit_automorphisms(self) -> npt.NDArray[np.int_]:
        """Translations of every block column of qubits by the generator of the cyclic group Z_P.

        The X-type check in row ``r`` of block row ``i`` addresses qubit ``r - e_{il}`` of each
        block column ``l``.  Translating every qubit ``t`` of every block column to ``t + 1``
        therefore maps this check to the X-type check in row ``r + 1`` of the same block row, so it
        maps the rows of ``matrix_x`` onto themselves.  The same argument applies to ``matrix_z``.
        """
        num_blocks = self.exponents_x.shape[1]
        return _get_block_translations(abstract.CyclicGroup(self.lift_size), num_blocks)

    @staticmethod
    def _validate_exponents(
        exponents: npt.NDArray[np.int_] | Sequence[Sequence[int]], pauli: str
    ) -> npt.NDArray[np.int_]:
        """Check that an exponent array is a nonempty 2D array of integers, and return it."""
        array = np.asarray(exponents)
        if array.ndim != 2 or array.size == 0 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"The {pauli}-type exponent array of a PairPartitionCode must be a nonempty"
                f" two-dimensional array of integers (got shape {array.shape} and dtype"
                f" {array.dtype})"
            )
        return array.astype(int)

    def _get_ring_array(self, exponents: npt.NDArray[np.int_]) -> abstract.RingArray:
        """Build the array of monomials x**e in F_2[Z_P] for an array of exponents e."""
        ring = abstract.GroupRing(abstract.CyclicGroup(self.lift_size))
        generator = ring.group.generators[0]
        return abstract.RingArray.build(
            [[generator ** int(exp) for exp in row] for row in exponents], ring
        )
