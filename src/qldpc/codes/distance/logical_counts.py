# SPDX-License-Identifier: Apache-2.0

"""Exact counts of the logical operators of a given weight in binary CSS codes.

See :class:`CSSLogicalCounter` for the conventions.

Each X/Z sector uses a normalizer basis whose quotient labels mark nontrivial words (Section 3.4 of
``brouwer_zimmermann.md``).  Words are enumerated in layers over Brouwer--Zimmermann (BZ)
information sets, whose family and certificate come from the stages in ``_information_sets.py`` that
the distance search shares.  Unlike a distance search, the scan stops only when the BZ lower bound
exceeds ``w`` (Section 5 of ``brouwer_zimmermann.md``), and a word visited by several information
sets is counted by only one of them.  Full Pauli counts are derived from the homogeneous words of
weight ``d``, with one masked rank per word.  Self-dual codes scan only one sector, and have
``count_pauli = 3 * count_x``.  Counters for codes deformed by local CSS-preserving Cliffords
transform the stored sector bases and information-set families rather than recomputing them.

The default method is ``"brouwer_zimmermann"``.  The alternative ``"macwilliams"`` counts from
check-space weight distributions; see ``_macwilliams.py`` for the algorithm.

References:

- https://arxiv.org/abs/1603.06757 (Brouwer--Zimmermann information sets)
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from qldpc._utils import format_docstring

from . import _information_sets
from ._binary import (
    get_independent_rows,
    get_kernel,
    get_nested_code_basis,
    pack_bits,
    popcount_word,
    row_reduce_binary,
)
from ._brouwer_zimmermann import get_distance_brouwer_zimmermann_many
from ._brute_force import get_distance_quantum_brute_force
from ._macwilliams import CheckEnumerators
from .core import _DISTANCE_METHODS, DistanceMethod

# codes.common imports this package, so CSSCode is imported only where it is used.
if TYPE_CHECKING:
    from ..common import CSSCode

_BinaryMatrix = npt.NDArray[np.uint8]
_PackedMatrix = npt.NDArray[np.uint64]

# Coefficient combinations visited per native call, which bounds the time between interruptions.
_CHUNK_SIZE = 1 << 16

# Batch size exponent passed to the distance searches, as in get_distance_quantum.
_BLOCK_SIZE = 15

# Check rank from which the MacWilliams method row-reduces both checks of a counter not known to be
# self-dual, to share one walk if they span the same space.  Below it, a second walk over the
# 2**rank words costs less than the row reduction.
_MIN_SHARED_WALK_RANK = 14

# The basis of a homogeneous sector, and its quotient labels.
_Sector = tuple[_BinaryMatrix, _BinaryMatrix]

LogicalCountMethod = Literal[
    "brouwer_zimmermann", "macwilliams", "brute_force", "information_sets", "weight_enumerator"
]

_CanonicalLogicalCountMethod = Literal["brouwer_zimmermann", "macwilliams", "brute_force"]

# The canonical name of every accepted logical-count method name, including aliases.
_LOGICAL_COUNT_METHODS: dict[str, _CanonicalLogicalCountMethod] = {
    **_DISTANCE_METHODS,
    "macwilliams": "macwilliams",
    "weight_enumerator": "macwilliams",
}


####################################################################################################
# results and entry points


@dataclass(frozen=True)
class CSSLogicalCounts:
    """Counts of the nontrivial logical operators of one weight in a binary CSS code.

    Attributes:
        weight: The weight of the counted operators.  If no weight was requested, this is the
            distance of the code, or the number of qubits if the code encodes no qubits, as with
            :func:`qldpc.codes.distance.get_distance_quantum`.
        count_x: The number of X-only nontrivial logical operators of weight ``weight``.
        count_z: The number of Z-only nontrivial logical operators of weight ``weight``.
        count_pauli: The number of nontrivial logical Pauli operators of weight ``weight``,
            ignoring phase, or None if it was not requested.  It includes the X-only and Z-only
            operators counted by ``count_x`` and ``count_z``, as well as Y and mixed operators.
    """

    weight: int
    count_x: int
    count_z: int
    count_pauli: int | None


class CSSLogicalCounter:
    """Exact counter of the logical operators of a given weight in a binary CSS code.

    For a CSS stabilizer code with checks ``H_X`` and ``H_Z``, the nontrivial X logicals are the
    words in ``ker(H_Z)`` outside ``row(H_X)``, and likewise for Z with X and Z exchanged.  At a
    weight ``w`` that defaults to the distance ``d``, ``count_logical_ops`` counts:

    - ``count_x`` and ``count_z``: the X-only and Z-only nontrivial logical operators of weight
      ``w``,
    - optionally ``count_pauli``: all nontrivial logical operators of weight ``w``, including Y and
      mixed operators.  This count is supported only for ``w <= d``, and is zero for ``w < d``.

    Counts are of physical operators ignoring phase, not of logical classes.  At the default weight,
    ``d`` is the minimum over both sectors, so ``count_x`` or ``count_z`` may be zero.  Counting
    generally takes an exponentially long time.

    In a subsystem code, the checks ``G_X`` and ``G_Z`` are gauge generators, and the stabilizers
    ``S_X`` and ``S_Z`` are the gauge operators that commute with all gauge operators of the other
    type.  Nontrivial logicals are those outside the gauge group (dressed logicals), so the X sector
    is ``ker(S_Z)`` modulo ``row(G_X)``, and likewise for Z.  The Brouwer-Zimmermann and brute-force
    methods support subsystem codes, but the MacWilliams method does not.

    Counters for codes deformed by local CSS-preserving Cliffords (see :meth:`deformed`) reuse the
    work of this counter.

    Args:
        code: A binary CSS code, which may be a subsystem code and may have redundant checks.

    Raises:
        TypeError: If the code is not a :class:`~qldpc.codes.CSSCode`.
        ValueError: If the code is not binary.
    """

    def __init__(self, code: CSSCode) -> None:
        from ..common import CSSCode, QuditCode

        if not isinstance(code, CSSCode):
            hint = "; convert it with code.to_css()" if isinstance(code, QuditCode) else ""
            raise TypeError(f"Expected a CSSCode, not {type(code).__name__}{hint}")
        if code.field.order != 2:
            raise ValueError(f"Logical counts require a binary code, not GF({code.field.order})")

        # Reduce the checks to independent bases in reduced row echelon form.
        self._code: CSSCode | None = code
        checks_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
        checks_z = code.matrix_z.view(np.ndarray).astype(np.uint8)
        self._num_qubits = checks_x.shape[1]
        self._gauge_x = get_independent_rows(checks_x)
        self._gauge_z = get_independent_rows(checks_z)
        # Reduced row echelon form is unique, so the X and Z checks span the same space exactly when
        # these arrays are equal.  A self-dual code then shares one array for each X/Z pair.
        self._self_dual = np.array_equal(self._gauge_x, self._gauge_z)
        if self._self_dual:
            self._gauge_z = self._gauge_x

        # Identify the stabilizers among the gauge operators, which fixes the number of logicals.
        self._stabilizers_x, self._stabilizers_z = _get_stabilizers(self._gauge_x, self._gauge_z)
        if self._self_dual:
            self._stabilizers_z = self._stabilizers_x
        self._dimension = self._num_qubits - len(self._gauge_x) - len(self._stabilizers_z)

        # Work that is done on demand and kept for later counts.
        self._sectors: tuple[_Sector, _Sector] | None = None
        self._disjoint_families: dict[int, _information_sets.BasisFamily] = {}
        self._enumerators: CheckEnumerators | None = None
        self._canonical: tuple[_BinaryMatrix, _BinaryMatrix] | None = (self._gauge_x, self._gauge_z)

    @property
    def num_qubits(self) -> int:
        """The number of physical qubits of the code."""
        return self._num_qubits

    @property
    def dimension(self) -> int:
        """The number of logical qubits that the code encodes."""
        return self._dimension

    @property
    def code(self) -> CSSCode:
        """The code whose logical operators are counted.

        For a counter built by a deformation method, this is a new :class:`~qldpc.codes.CSSCode`
        with the counter's independent checks, built on first access.
        """
        if self._code is None:
            from ..common import CSSCode

            self._code = CSSCode(self._gauge_x.astype(np.int_), self._gauge_z.astype(np.int_))
        return self._code

    @property
    def checks_x(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the X checks, in reduced row echelon form."""
        return self._get_canonical_checks()[0].copy()

    @property
    def checks_z(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the Z checks, in reduced row echelon form."""
        return self._get_canonical_checks()[1].copy()

    @property
    def is_self_dual(self) -> bool:
        """Whether the X and Z checks span the same row space."""
        return self._self_dual or np.array_equal(*self._get_canonical_checks())

    @format_docstring(MAX_SETS=_information_sets.MAX_SETS)
    def count_logical_ops(
        self,
        weight: int | None = None,
        *,
        method: LogicalCountMethod = "brouwer_zimmermann",
        count_pauli: bool = False,
        within_distance: bool = False,
        max_information_sets: int = _information_sets.MAX_SETS,
    ) -> CSSLogicalCounts:
        """Count the nontrivial logical operators of a given weight.

        Args:
            weight: Weight of the operators to count, between zero and the number of qubits.  If
                None (the default), count the operators whose weight is the distance of the code.
            method: ``"brouwer_zimmermann"`` (the default), ``"macwilliams"``, or
                ``"brute_force"``.  MacWilliams uses check-space weight distributions, cached on the
                counter but not transferred through deformation.  It does not support subsystem
                codes, full Pauli counts of non-self-dual codes, or check ranks above 63.  Brute
                force is a reference for small codes.  The aliases ``"information_sets"`` and
                ``"weight_enumerator"`` select ``"brouwer_zimmermann"`` and ``"macwilliams"``,
                respectively.
            count_pauli: If True, also count full Pauli operators, including Y and mixed operators.
                For codes that are not self-dual, each homogeneous word of the counted weight costs
                one rank calculation.  These counts are exact up to the distance and zero below it,
                and are not supported above it.  Defaults to False.
            within_distance: If True, raise an error when the weight is above the distance, that
                is, when some nontrivial X-only or Z-only logical operator is lighter than the
                weight.  Full Pauli counts always do this.  Defaults to False.
            max_information_sets: Limit on the number of information sets that a Brouwer-Zimmermann
                count may use when it replaces its first, disjoint sets by overlapping ones.  More
                sets can skip costly layers of a hard count, but each set costs setup work, which
                dominates a cheap count.  A small limit suits many cheap counts on a large code.
                This never changes the counts, and the other methods ignore it.  Defaults to
                {MAX_SETS}.

        Returns:
            The counted weight and the counts.  If no weight is given and the code encodes no
            qubits, the returned weight is the number of qubits and every count is zero.

        Raises:
            ValueError: If the weight is not an integer between zero and the number of qubits, if
                the method is unknown, if the weight is above the distance and ``count_pauli`` or
                ``within_distance`` is set, or if the MacWilliams method does not support the code
                or the request.
        """
        # Validate the inputs, resolving method aliases.
        method = _validate_logical_count_method(method)
        if weight is not None and (
            not isinstance(weight, (int, np.integer))
            or isinstance(weight, bool)
            or not 0 <= weight <= self.num_qubits
        ):
            raise ValueError("The weight must be an integer between zero and the number of qubits")
        if method == "macwilliams":
            self._validate_macwilliams_request(count_pauli)

        # A code without logicals needs no scan, and the MacWilliams method needs no sectors.
        if not self.dimension:
            empty_pauli = 0 if count_pauli else None
            return CSSLogicalCounts(
                self.num_qubits if weight is None else int(weight), 0, 0, empty_pauli
            )
        if method == "macwilliams":
            return self._count_with_macwilliams(
                weight, count_pauli=count_pauli, reject_lighter=count_pauli or within_distance
            )

        sector_x, sector_z = self._get_sectors()
        # The two sectors of a self-dual code are the same, so one of them suffices.
        sectors = [sector_x] if self._self_dual else [sector_x, sector_z]
        target = int(weight) if weight is not None else _get_distance(sectors, method)

        # A nontrivial logical Pauli (x, z) of weight d has one of two disjoint forms.  Either x is
        # a nontrivial X logical with support T of size d, and z is one of 2**(d - rank(S_X[:, T]))
        # Z words supported in T that commute with S_X, or x is an X gauge operator, and z is a
        # nontrivial Z logical of weight d whose support T contains that of x, which leaves
        # 2**(rank(G_X) - rank(G_X[:, ~T])) choices of x.  Each support T of a self-dual code
        # carries exactly three operators, so self-dual codes need no ranks.  Above d, a logical
        # Pauli need not have the support of a homogeneous logical, so the scans reject any
        # nontrivial word lighter than the target.
        weigh = count_pauli and not self._self_dual
        reject_above = count_pauli or within_distance
        no_rows = self._gauge_x[:0]

        # Scan each sector, or only the X sector of a self-dual code.
        counts_x = _count_words(
            *sector_x,
            target,
            method,
            pack_bits(self._stabilizers_x if weigh else no_rows),
            complement=False,
            max_exponent=target if weigh else 0,
            reject_lighter=reject_above,
            get_disjoint=functools.partial(self._get_disjoint_family, 0),
            max_sets=max_information_sets,
        )
        if self._self_dual or counts_x is None:
            counts_z = counts_x
        else:
            counts_z = _count_words(
                *sector_z,
                target,
                method,
                pack_bits(self._gauge_x if weigh else no_rows),
                complement=True,
                max_exponent=len(self._gauge_x) if weigh else 0,
                reject_lighter=reject_above,
                get_disjoint=functools.partial(self._get_disjoint_family, 1),
                max_sets=max_information_sets,
            )
        if counts_x is None or counts_z is None:
            raise _get_above_distance_error(target)

        # Sum the histograms, and derive the full Pauli count as described above.
        count_x, count_z = int(counts_x.sum()), int(counts_z.sum())
        num_pauli = None
        if count_pauli and self._self_dual:
            num_pauli = 3 * count_x
        elif count_pauli:
            num_pauli = sum(
                int(count) << exponent
                for counts in (counts_x, counts_z)
                for exponent, count in enumerate(counts)
            )
        return CSSLogicalCounts(target, count_x, count_z, num_pauli)

    def deformed(self, targets: Sequence[int], matrix_x: npt.ArrayLike) -> CSSLogicalCounter:
        """Return a counter for this code deformed by a local CSS-preserving Clifford.

        The Clifford maps an X support ``x`` on the targets, as a column vector, to ``matrix_x @ x``
        and a Z support ``z`` to ``matrix_z @ z``, where ``matrix_z = (matrix_x^-1)^T`` over GF(2)
        preserves commutation.  The packed matrices are cached by ``matrix_x``, so reusing a gate is
        cheaper than its first use.

        Args:
            targets: Between one and 64 distinct qubit indices on which the Clifford acts, in order.
            matrix_x: Invertible binary matrix whose rows and columns are indexed by the targets.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the targets are invalid, or if ``matrix_x`` is not an invertible binary
                matrix that matches the number of targets.
        """
        positions = self._get_positions(targets)
        local_x, local_z = _get_local_matrices(matrix_x, len(positions))
        return self._deformed(positions, local_x, local_z)

    def deformed_cnot(self, control: int, target: int) -> CSSLogicalCounter:
        """Return a counter for this code deformed by a CNOT on zero-based qubit indices.

        The CNOT acts as Stim's ``CX control target``: it maps X on the control to X on the control
        and target, and Z on the target to Z on the control and target.

        Args:
            control: The control qubit.
            target: The target qubit, distinct from the control.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the qubits are not distinct, in-range indices.
        """
        return self._deformed(self._get_positions([control, target]), *_CNOT)

    def deformed_swel(
        self, qubit_0: int, qubit_1: int, qubit_2: int, qubit_3: int
    ) -> CSSLogicalCounter:
        """Return a counter for this code deformed by a four-qubit SWEL-preserving Clifford.

        The Clifford applies the same binary matrix, whose rows have weight three, to X and Z
        supports on the four targets.  It therefore maps self-dual codes to self-dual codes, and
        preserves the self-dual-with-equivalent-logicals (SWEL) property of
        :meth:`qldpc.codes.CSSCode.is_swel`.  It is valid for any CSS code.

        Args:
            qubit_0: The qubit of the first matrix row.
            qubit_1: The qubit of the second matrix row.
            qubit_2: The qubit of the third matrix row.
            qubit_3: The qubit of the fourth matrix row.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the qubits are not distinct, in-range indices.
        """
        positions = self._get_positions([qubit_0, qubit_1, qubit_2, qubit_3])
        return self._deformed(positions, *_SWEL)

    def _get_positions(self, targets: Sequence[int]) -> npt.NDArray[np.int64]:
        """Validate deformation targets."""
        positions = list(targets)
        if (
            not 1 <= len(positions) <= 64
            or any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or not 0 <= index < self.num_qubits
                for index in positions
            )
            or len(set(positions)) != len(positions)
        ):
            raise ValueError(
                "Clifford targets must be 1 to 64 distinct qubit indices in"
                f" range({self.num_qubits}) (got {positions})"
            )
        return np.array(positions, dtype=np.int64)

    def _deformed(
        self,
        positions: npt.NDArray[np.int64],
        local_x: npt.NDArray[np.uint64],
        local_z: npt.NDArray[np.uint64],
    ) -> CSSLogicalCounter:
        """Apply validated, packed local matrices to every stored X-type and Z-type matrix.

        The Clifford maps each gauge, stabilizer and sector space onto that of the new code, and
        keeps the quotient labels, so nothing is recomputed.  The checks are no longer in reduced
        row echelon form, which is recomputed only if needed.
        """
        counter = CSSLogicalCounter.__new__(CSSLogicalCounter)
        counter._num_qubits, counter._dimension = self._num_qubits, self._dimension
        counter._code, counter._canonical = None, None

        # A gate that acts identically on X and Z keeps a self-dual code self-dual.
        shared = self._self_dual and (local_x is local_z or np.array_equal(local_x, local_z))
        counter._self_dual = shared
        counter._gauge_x = _apply_local(self._gauge_x, positions, local_x)
        counter._gauge_z = (
            counter._gauge_x if shared else _apply_local(self._gauge_z, positions, local_z)
        )
        # Stabilizer codes share their gauge and stabilizer arrays.
        counter._stabilizers_x = (
            counter._gauge_x
            if self._stabilizers_x is self._gauge_x
            else _apply_local(self._stabilizers_x, positions, local_x)
        )
        if self._stabilizers_z is self._gauge_z:
            counter._stabilizers_z = counter._gauge_z
        elif shared:
            counter._stabilizers_z = counter._stabilizers_x
        else:
            counter._stabilizers_z = _apply_local(self._stabilizers_z, positions, local_z)

        # Transform the kept work where the gate allows it, and drop the rest.
        counter._sectors = None
        counter._disjoint_families = {}
        # A gate changes the weight of every check word that touches its qubits, so the weight
        # enumerators are not kept.
        counter._enumerators = None
        if self._sectors is not None:
            (basis_x, labels_x), (basis_z, labels_z) = self._sectors
            sector_x = (_apply_local(basis_x, positions, local_x), labels_x)
            sector_z = sector_x if shared else (_apply_local(basis_z, positions, local_z), labels_z)
            counter._sectors = (sector_x, sector_z)

            # The kept families follow the gate.  If one cannot, the sector is prepared again when
            # it is next counted.  A self-dual code keeps one family, for both of its sectors.
            for index, local in ((0, local_x), (1, local_z)):
                family = self._disjoint_families.get(0 if self._self_dual else index)
                if family is not None and not (shared and index == 1):
                    moved = _transport_family(family, positions, local, counter._sectors[index][0])
                    if moved is not None:
                        counter._disjoint_families[index] = moved
        return counter

    def _validate_macwilliams_request(self, count_pauli: bool) -> None:
        """Reject codes and requests that the MacWilliams method does not support.

        The X stabilizers are the X gauge operators that commute with every Z gauge operator, so
        they span fewer dimensions than the X gauge operators exactly when some X and Z checks
        anticommute, that is, in a subsystem code.
        """
        if len(self._stabilizers_x) < len(self._gauge_x):
            raise ValueError(
                "The MacWilliams method does not support subsystem codes; use"
                " 'brouwer_zimmermann' or 'brute_force'"
            )
        if count_pauli and not self.is_self_dual:
            raise ValueError(
                "The MacWilliams method counts full Pauli operators only for self-dual codes;"
                " use 'brouwer_zimmermann' or 'brute_force'"
            )

    def _count_with_macwilliams(
        self, weight: int | None, *, count_pauli: bool, reject_lighter: bool
    ) -> CSSLogicalCounts:
        """Count logical operators by the weight enumerators of the checks of a stabilizer code.

        Each support of a nontrivial logical Pauli operator of weight ``d`` in a self-dual code
        carries exactly three such operators: one X-only, one Z-only, and one Y-only.  The checks of
        a self-dual code may share one weight distribution, even if a deformation gave them
        different bases.  Equal check spaces have equal distributions, so whether they share one
        changes only the number of walks, never the counts.  The enumerators are kept, so that later
        counts at any weight, with any options, do not walk again.
        """
        # Walk the check spaces once per counter.
        if self._enumerators is None:
            rank = len(self._gauge_x)
            # A full Pauli count has already been validated as self-dual.
            share = (
                count_pauli
                or self._self_dual
                or (rank == len(self._gauge_z) >= _MIN_SHARED_WALK_RANK and self.is_self_dual)
            )
            self._enumerators = CheckEnumerators(
                self._gauge_x, self._gauge_x if share else self._gauge_z
            )
        enumerators = self._enumerators

        # Find the distance, or make sure that no nontrivial logical is lighter than the weight.
        if weight is None:
            target = next(
                candidate
                for candidate in range(1, self.num_qubits + 1)
                if any(enumerators.count_logical_ops(candidate))
            )
        else:
            target = int(weight)
            if reject_lighter and any(
                any(enumerators.count_logical_ops(lighter)) for lighter in range(1, target)
            ):
                raise _get_above_distance_error(target)

        count_x, count_z = enumerators.count_logical_ops(target)
        return CSSLogicalCounts(target, count_x, count_z, 3 * count_x if count_pauli else None)

    def _get_sectors(self) -> tuple[_Sector, _Sector]:
        """The X and Z sectors: normalizer bases whose first rows span the gauge operators."""
        if self._sectors is None:
            sector_x = get_nested_code_basis(get_kernel(self._stabilizers_z), self._gauge_x)
            sector_z = (
                sector_x
                if self._self_dual
                else get_nested_code_basis(get_kernel(self._stabilizers_x), self._gauge_z)
            )
            self._sectors = (sector_x, sector_z)
        return self._sectors

    def _get_disjoint_family(self, index: int) -> _information_sets.BasisFamily:
        """The first stage of the information-set family of a sector, which is kept.

        This stage does not depend on the counted weight, so counting at another weight, as when a
        guessed weight fails, does not repeat it.
        """
        if index not in self._disjoint_families:
            self._disjoint_families[index] = _information_sets.get_disjoint_family(
                *self._get_sectors()[index]
            )
        return self._disjoint_families[index]

    def _get_canonical_checks(self) -> tuple[_BinaryMatrix, _BinaryMatrix]:
        """The checks in reduced row echelon form, which deformation does not preserve."""
        if self._canonical is None:
            self._canonical = (
                get_independent_rows(self._gauge_x),
                get_independent_rows(self._gauge_z),
            )
        return self._canonical


def count_logical_ops_css(
    code: CSSCode,
    weight: int | None = None,
    *,
    method: LogicalCountMethod = "brouwer_zimmermann",
    count_pauli: bool = False,
    within_distance: bool = False,
    max_information_sets: int = _information_sets.MAX_SETS,
) -> CSSLogicalCounts:
    """Count the nontrivial logical operators of a given weight in a binary CSS code.

    This is shorthand for ``CSSLogicalCounter(code).count_logical_ops(...)``; see
    :class:`CSSLogicalCounter` and :meth:`CSSLogicalCounter.count_logical_ops` for the arguments and
    conventions.

    Returns:
        The counted weight and the counts.

    Raises:
        TypeError: If the code is not a :class:`~qldpc.codes.CSSCode`.
        ValueError: If the code is not binary, or for invalid arguments as described in
            :meth:`CSSLogicalCounter.count_logical_ops`.
    """
    return CSSLogicalCounter(code).count_logical_ops(
        weight,
        method=method,
        count_pauli=count_pauli,
        within_distance=within_distance,
        max_information_sets=max_information_sets,
    )


def _validate_logical_count_method(
    method: LogicalCountMethod,
) -> _CanonicalLogicalCountMethod:
    """Validate a logical-count method selector, and return its canonical name.

    The aliases ``"information_sets"`` and ``"weight_enumerator"`` resolve to
    ``"brouwer_zimmermann"`` and ``"macwilliams"``, respectively.
    """
    if method not in _LOGICAL_COUNT_METHODS:
        raise ValueError(
            f"Unknown logical-count method {method!r}; choose from 'brouwer_zimmermann',"
            " 'macwilliams', or 'brute_force'."
        )
    return _LOGICAL_COUNT_METHODS[method]


def _get_above_distance_error(weight: int) -> ValueError:
    """The error for counts that the distance makes unsupported."""
    return ValueError(
        "Counts above the distance are not supported here: the distance is less than"
        f" weight {weight}"
    )


####################################################################################################
# counting words of one weight


def _get_distance(sectors: Sequence[_Sector], method: DistanceMethod) -> int:
    """Minimum weight of a nontrivial word in any of the given sectors."""
    if method == "brute_force":
        return min(_get_brute_force_distance(sector) for sector in sectors)
    return get_distance_brouwer_zimmermann_many(sectors, cutoff=0, block_size=_BLOCK_SIZE)


def _get_brute_force_distance(sector: _Sector) -> int:
    """Minimum weight of a nontrivial word of one sector, by exhaustive enumeration."""
    basis, labels = sector
    eligible = np.any(labels, axis=1)
    return get_distance_quantum_brute_force(
        basis[eligible], basis[~eligible], cutoff=0, block_size=_BLOCK_SIZE, homogeneous=True
    )


def _count_words(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    method: DistanceMethod,
    rank_rows: _PackedMatrix,
    *,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
    get_disjoint: Callable[[], _information_sets.BasisFamily],
    max_sets: int,
) -> npt.NDArray[np.int64] | None:
    """Visit each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over exponents ``max_exponent - rank(rank_rows[:, mask])``, where the mask
    is the support of the word, or its complement.  The histogram sums to the number of words, and
    the caller weighs each exponent by ``2**exponent``, which could overflow a native integer.  The
    scan also visits every lighter word, so with ``reject_lighter`` it returns None as soon as it
    finds a nontrivial one.
    """
    # Select the information sets, or scan the whole basis as one set for brute force.
    certificate: _information_sets.Certificate | None
    if method == "brute_force":
        sets, certificate = [(basis, labels)], None
    else:
        sets, certificate = _get_counting_information_sets(
            basis, labels, weight, get_disjoint(), max_sets
        )

    # Pack the sets for the native scan.
    rows = [pack_bits(generators) for generators, _ in sets]
    tags = [pack_bits(set_labels) for _, set_labels in sets]
    # Ownership compares sets, so a single set needs no coefficient extractor.
    extractors = (
        np.stack([_get_coefficient_extractor(generators) for generators, _ in sets])
        if len(sets) > 1
        else np.zeros((1, 0, rows[0].shape[1]), dtype=np.uint64)
    )

    # Scan the sets layer by layer, until the certificate rules out unvisited words of the weight.
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    for layer in range(1, len(basis) + 1):
        for index in range(len(sets)):
            support = np.zeros(layer, dtype=np.int64)
            depth = 0
            while depth >= 0:
                depth, lighter = _count_chunk(
                    rows[index],
                    tags[index],
                    extractors,
                    index,
                    support,
                    depth,
                    weight,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    _CHUNK_SIZE,
                )
                if lighter:
                    return None
        # Equality would still allow unvisited words of the target weight.
        if certificate is not None and certificate.lower_bound(layer) > weight:
            break
    return histogram


def _get_counting_information_sets(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    disjoint: _information_sets.BasisFamily,
    max_sets: int,
) -> tuple[list[_Sector], _information_sets.Certificate]:
    """Select BZ information sets that certify every word of the given weight.

    Returns the sets and the certificate that bounds the weight of unvisited words after each layer.
    The family stages aim at the counting stop ``L(t) >= weight + 1``; any family gives an exact
    count.
    """
    family = _information_sets.prepare_family(
        basis, labels, upper_bound=weight + 1, disjoint=disjoint, max_sets=max_sets
    )
    sets = []
    for information_set in family.information_sets:
        assert information_set.labels is not None
        sets.append((information_set.generators, information_set.labels))
    return sets, family.certificate


def _transport_family(
    family: _information_sets.BasisFamily,
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
    basis: _BinaryMatrix,
) -> _information_sets.BasisFamily | None:
    """The family of a sector after a local gate, or None if one of its sets stops being a set.

    Each set is mapped through the gate and reduced again on its own pivot columns, so the pivots
    stay disjoint and the labels move with their rows.  A set stops being one if its pivot columns
    are no longer independent.
    """
    sets = []
    for information_set in family.information_sets:
        generators = _apply_local(information_set.generators, positions, local)
        reduced, labels, pivots = row_reduce_binary(
            generators, information_set.labels, columns=information_set.pivots
        )
        if len(pivots) != information_set.rank:
            return None
        sets.append(_information_sets.InformationSet(reduced, labels, len(pivots), pivots))
    return _information_sets.BasisFamily.from_information_sets(
        sets,
        dimension=family.certificate.dimension,
        weight_divisor=_information_sets.infer_weight_divisor(basis),
    )


def _get_coefficient_extractor(generators: _BinaryMatrix) -> _PackedMatrix:
    """Packed rows ``E.T`` such that ``generators @ E`` is the identity over GF(2).

    Bit ``b`` of the coefficient vector of a word ``w`` is then the parity of ``w & E.T[b]``.
    """
    # Row reduction finds T with T @ generators = R, where R is the identity on its pivot columns P,
    # so that w[P] @ T recovers the coefficients of w.
    size = len(generators)
    _, transform, pivots = row_reduce_binary(generators, np.eye(size, dtype=np.uint8))
    assert transform is not None
    extractor = np.zeros((generators.shape[1], size), dtype=np.uint8)
    extractor[pivots] = transform
    return pack_bits(extractor.T)


####################################################################################################
# stabilizers of subsystem codes


def _get_stabilizers(
    gauge_x: _BinaryMatrix, gauge_z: _BinaryMatrix
) -> tuple[_BinaryMatrix, _BinaryMatrix]:
    """Independent bases of the stabilizers among the given gauge generators.

    The stabilizers are the gauge operators that commute with every gauge operator of the other
    type.  If all gauge generators commute, the given arrays are returned.
    """
    overlaps = (gauge_x.astype(int) @ gauge_z.T.astype(int) % 2).astype(np.uint8)
    if not np.any(overlaps):
        return gauge_x, gauge_z
    # A combination c @ gauge_x commutes with every Z gauge generator exactly when c @ overlaps = 0.
    stabilizers_x = get_kernel(overlaps.T).astype(int) @ gauge_x % 2
    stabilizers_z = get_kernel(overlaps).astype(int) @ gauge_z % 2
    return (
        get_independent_rows(stabilizers_x.astype(np.uint8)),
        get_independent_rows(stabilizers_z.astype(np.uint8)),
    )


####################################################################################################
# local Cliffords

# Packed local matrices, in which bit j of row i is entry (i, j).  The CNOT, with its control first,
# acts by [[1, 0], [1, 1]] on X supports and by [[1, 1], [0, 1]] on Z supports.  The SWEL-preserving
# Clifford acts by [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]] on both.
_CNOT = (np.array([1, 3], dtype=np.uint64), np.array([3, 2], dtype=np.uint64))
_SWEL = (np.array([7, 11, 13, 14], dtype=np.uint64),) * 2


def _get_local_matrices(
    matrix_x: npt.ArrayLike, size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Packed local matrices for X and Z supports, from a cache keyed by ``matrix_x``."""
    order = getattr(type(matrix_x), "order", None)
    local = np.asarray(matrix_x)
    if local.dtype.kind not in "biuf":
        # The bytes of other dtypes, such as objects, do not identify their entries.
        local, order = _as_local_matrix(matrix_x, size), None
    return _get_cached_local_matrices(size, order, local.shape, local.dtype.str, local.tobytes())


@functools.lru_cache(maxsize=1024)
def _get_cached_local_matrices(
    size: int, order: int | None, shape: tuple[int, ...], dtype: str, data: bytes
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Validate and pack ``matrix_x`` given by its bytes, and the matrix for Z supports.

    Searches apply the same local Cliffords repeatedly, so results are cached.  The key identifies
    the input completely, including its shape and field, so invalid input never matches a cached
    entry.  Results are read-only so that callers cannot alter the cache.
    """
    if order is not None and order != 2:
        raise ValueError("Local Clifford matrices must be binary")
    matrix = _as_local_matrix(np.frombuffer(data, dtype=dtype).reshape(shape), size)
    local_x, local_z = _pack_local(matrix), _pack_local(_get_dual_local_matrix(matrix))
    local_x.setflags(write=False)
    local_z.setflags(write=False)
    return local_x, local_z


def _as_local_matrix(matrix: npt.ArrayLike, size: int) -> _BinaryMatrix:
    """Validate a binary local Clifford matrix that acts on ``size`` targets."""
    order = getattr(type(matrix), "order", None)
    local = np.asarray(matrix)
    if local.shape != (size, size):
        raise ValueError("Local Clifford matrices must be square and match the target count")
    if (order is not None and order != 2) or not np.all((local == 0) | (local == 1)):
        raise ValueError("Local Clifford matrices must be binary")
    return local.astype(np.uint8)


def _get_dual_local_matrix(matrix_x: _BinaryMatrix) -> _BinaryMatrix:
    """Return the matrix for Z supports, ``matrix_z``, with ``matrix_x @ matrix_z.T = I``."""
    size = len(matrix_x)
    # Row reduction finds T with T @ matrix_x = I, so that matrix_z = T.T.
    _, inverse, pivots = row_reduce_binary(matrix_x, np.eye(size, dtype=np.uint8))
    if len(pivots) < size:
        raise ValueError("Local Clifford matrix_x must be invertible over GF(2)")
    assert inverse is not None
    return inverse.T.copy()


def _pack_local(matrix: _BinaryMatrix) -> npt.NDArray[np.uint64]:
    """Pack each row of a local matrix into one word, whose bit j is column j."""
    # There are at most 64 targets, so each row fits in a single word.
    return pack_bits(matrix)[:, 0]


####################################################################################################
# native kernels


@njit(cache=True)
def _apply_local(
    rows: _BinaryMatrix, positions: npt.NDArray[np.int64], local: npt.NDArray[np.uint64]
) -> _BinaryMatrix:
    """Apply a packed local matrix to the target columns of a copy of the given rows.

    Column ``positions[i]`` of a row becomes the parity of ``local[i]`` and the word whose bit ``j``
    is the row's entry at ``positions[j]``.
    """
    result = rows.copy()
    for row in range(rows.shape[0]):
        word = np.uint64(0)
        for index in range(len(positions)):
            word |= np.uint64(rows[row, positions[index]]) << np.uint64(index)
        if word:
            for index in range(len(positions)):
                result[row, positions[index]] = popcount_word(local[index] & word) & 1
    return result


@njit(cache=True)
def _count_chunk(
    rows: _PackedMatrix,
    tags: _PackedMatrix,
    extractors: npt.NDArray[np.uint64],
    set_index: int,
    support: npt.NDArray[np.int64],
    depth: int,
    target: int,
    rank_rows: _PackedMatrix,
    complement: bool,
    max_exponent: int,
    histogram: npt.NDArray[np.int64],
    reject_lighter: bool,
    chunk_size: int,
) -> tuple[int, bool]:
    """Visit up to ``chunk_size`` combinations of ``len(support)`` rows, resuming at ``depth``.

    ``support`` holds the increasing row indices of the current combination; ``depth`` is the
    position being advanced, or -1 once every combination has been visited.  Each owned nontrivial
    word of weight ``target`` adds one to ``histogram`` (see ``_count_words``).  Returns the depth
    from which to resume, and whether the scan stopped at a nontrivial word lighter than ``target``,
    which is checked only with ``reject_lighter``.
    """
    num_rows, width = rows.shape
    layer = len(support)
    # Prefixes are recomputed on entry, so only support and depth carry state between calls.
    prefixes = np.zeros((layer, width), dtype=np.uint64)
    tag_prefixes = np.zeros((layer, tags.shape[1]), dtype=np.uint64)
    for position in range(1, depth + 1):
        row = support[position - 1]
        for index in range(width):
            prefixes[position, index] = prefixes[position - 1, index] ^ rows[row, index]
        for index in range(tags.shape[1]):
            tag_prefixes[position, index] = tag_prefixes[position - 1, index] ^ tags[row, index]

    # Advance through the combinations until the chunk is used up.
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    visited = 0
    while depth >= 0 and visited < chunk_size:
        # Backtrack when no room remains for the rows after this position.
        if support[depth] > num_rows - layer + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue

        # Extend the prefix by the row at this position, and advance to the next position.
        if depth + 1 < layer:
            row = support[depth]
            for index in range(width):
                prefixes[depth + 1, index] = prefixes[depth, index] ^ rows[row, index]
            for index in range(tags.shape[1]):
                tag_prefixes[depth + 1, index] = tag_prefixes[depth, index] ^ tags[row, index]
            depth += 1
            support[depth] = row + 1
            continue

        # Visit the remaining combinations that differ only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        for row in range(support[depth], stop):
            weight = np.int64(0)
            for index in range(width):
                word[index] = prefixes[depth, index] ^ rows[row, index]
                weight += popcount_word(word[index])
                # A heavier word is rejected below, so its unwritten words are never read.
                if weight > target:
                    break
            if weight > target or (weight < target and not reject_lighter):
                continue

            # A nonzero quotient label marks a nontrivial logical rather than a stabilizer.
            nontrivial = False
            for index in range(tags.shape[1]):
                if tag_prefixes[depth, index] != tags[row, index]:
                    nontrivial = True
                    break
            if nontrivial and weight < target:
                return depth, True
            if nontrivial and _is_owned(word, extractors, set_index, layer):
                rank = _get_masked_rank(rank_rows, word, complement, pivots)
                histogram[max_exponent - rank] += 1

        # Record the visited rows, and resume after them.
        visited += stop - support[depth]
        support[depth] = stop
    return depth, False


@njit(cache=True)
def _is_owned(
    word: npt.NDArray[np.uint64], extractors: npt.NDArray[np.uint64], set_index: int, layer: int
) -> bool:
    """Whether set ``set_index`` minimizes (layer, set index) among the sets that visit ``word``.

    Each set visits a word in the layer given by the weight of its coefficient vector in that set,
    so exactly one set owns each word, and it visits the word before any other set does.
    """
    for other in range(len(extractors)):
        if other == set_index:
            continue
        # A set that visits the word in an earlier layer, or in the same layer with a smaller index,
        # owns it.
        limit = layer if other < set_index else layer - 1
        size = 0
        for extractor_row in extractors[other]:
            parity = np.int64(0)
            for index in range(len(word)):
                parity += popcount_word(word[index] & extractor_row[index])
            size += int(parity & 1)
            if size > limit:
                break
        if size <= limit:
            return False
    return True


@njit(cache=True)
def _get_masked_rank(
    rows: _PackedMatrix,
    mask: npt.NDArray[np.uint64],
    complement: bool,
    pivots: _PackedMatrix,
) -> int:
    """Rank of ``rows`` restricted to the support of ``mask``, or to its complement.

    ``pivots`` is scratch space with at least as many rows as ``rows``.
    """
    pivot_words = np.empty(len(rows), dtype=np.int64)
    pivot_bits = np.empty(len(rows), dtype=np.uint64)
    rank = 0
    for row in rows:
        current = row & ~mask if complement else row & mask
        # Pivot rows are reduced when stored, so eliminating them in order clears every pivot bit.
        for previous in range(rank):
            if current[pivot_words[previous]] & pivot_bits[previous]:
                current ^= pivots[previous]

        # A nonzero remainder becomes a new pivot row, whose pivot is its lowest set bit.
        for index in range(len(current)):
            if current[index]:
                pivots[rank] = current
                pivot_words[rank] = index
                pivot_bits[rank] = current[index] & (~current[index] + np.uint64(1))
                rank += 1
                break
    return rank
