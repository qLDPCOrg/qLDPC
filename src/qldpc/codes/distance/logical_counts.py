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

The default method is ``"brouwer_zimmermann"``, whose scan is in ``_counting/scan.py``.  The
alternative ``"macwilliams"`` counts from check-space weight distributions; see
``_counting/macwilliams.py`` for the algorithm.

References:

- https://arxiv.org/abs/1603.06757 (Brouwer--Zimmermann information sets)
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, SupportsIndex

import numpy as np
import numpy.typing as npt

from qldpc._compilation import set_on_request_functions_compiled
from qldpc._utils import format_docstring

from . import _information_sets
from ._binary import (
    get_independent_rows,
    get_kernel,
    get_nested_code_basis,
    pack_bits,
)
from ._brouwer_zimmermann import get_distance_brouwer_zimmermann_many
from ._brute_force import get_distance_quantum_brute_force
from ._counting import local_cliffords, macwilliams, scan
from .core import _DISTANCE_METHODS, DistanceMethod

# codes.common imports this package, and codes.quantum imports codes.common, so code classes are
# imported only where they are used.
if TYPE_CHECKING:
    from ..common import CSSCode

_BinaryMatrix = npt.NDArray[np.uint8]


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
        self._disjoint_basis_families: dict[int, _information_sets.BasisFamily] = {}
        self._enumerators: macwilliams.CheckEnumerators | None = None
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
        max_information_sets: SupportsIndex = _information_sets.MAX_SETS,
    ) -> CSSLogicalCounts:
        """Count the nontrivial logical operators of a given weight.

        A program that makes very many counts can call
        :func:`~qldpc.codes.distance.compile_search_setup` first.

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
                This never changes the counts, and the other methods do not use it.  Defaults to
                {MAX_SETS}.

        Returns:
            The counted weight and the counts.  If no weight is given and the code encodes no
            qubits, the returned weight is the number of qubits and every count is zero.

        Raises:
            TypeError: If ``max_information_sets`` is not an integer.
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
        # Compiled code specializes on the type of each argument, so fix it to a Python integer.
        max_sets = operator.index(max_information_sets)

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
        # The distance search and the scans start from the disjoint basis families that this
        # counter keeps.
        get_disjoint_basis_families = [
            functools.partial(self._get_disjoint_basis_family, index) for index in range(2)
        ]
        target = (
            int(weight)
            if weight is not None
            else _get_distance(sectors, method, get_disjoint_basis_families)
        )

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
        counts_x = scan.count_words(
            *sector_x,
            target,
            method,
            pack_bits(self._stabilizers_x if weigh else no_rows),
            complement=False,
            max_exponent=target if weigh else 0,
            reject_lighter=reject_above,
            get_disjoint_basis_family=get_disjoint_basis_families[0],
            max_sets=max_sets,
        )
        if self._self_dual or counts_x is None:
            counts_z = counts_x
        else:
            counts_z = scan.count_words(
                *sector_z,
                target,
                method,
                pack_bits(self._gauge_x if weigh else no_rows),
                complement=True,
                max_exponent=len(self._gauge_x) if weigh else 0,
                reject_lighter=reject_above,
                get_disjoint_basis_family=get_disjoint_basis_families[1],
                max_sets=max_sets,
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
        local_x, local_z = local_cliffords.get_local_matrices(matrix_x, len(positions))
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
        return self._deformed(self._get_positions([control, target]), *local_cliffords.CNOT)

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
        return self._deformed(positions, *local_cliffords.SWEL)

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
        counter._gauge_x = local_cliffords.apply_local(self._gauge_x, positions, local_x)
        counter._gauge_z = (
            counter._gauge_x
            if shared
            else local_cliffords.apply_local(self._gauge_z, positions, local_z)
        )
        # Stabilizer codes share their gauge and stabilizer arrays.
        counter._stabilizers_x = (
            counter._gauge_x
            if self._stabilizers_x is self._gauge_x
            else local_cliffords.apply_local(self._stabilizers_x, positions, local_x)
        )
        if self._stabilizers_z is self._gauge_z:
            counter._stabilizers_z = counter._gauge_z
        elif shared:
            counter._stabilizers_z = counter._stabilizers_x
        else:
            counter._stabilizers_z = local_cliffords.apply_local(
                self._stabilizers_z, positions, local_z
            )

        # Transform the kept work where the gate allows it, and drop the rest.
        counter._sectors = None
        counter._disjoint_basis_families = {}
        # A gate changes the weight of every check word that touches its qubits, so the weight
        # enumerators are not kept.
        counter._enumerators = None
        if self._sectors is not None:
            (basis_x, labels_x), (basis_z, labels_z) = self._sectors
            sector_x = (local_cliffords.apply_local(basis_x, positions, local_x), labels_x)
            sector_z = (
                sector_x
                if shared
                else (local_cliffords.apply_local(basis_z, positions, local_z), labels_z)
            )
            counter._sectors = (sector_x, sector_z)

            # The kept families follow the gate.  If one cannot, the sector is prepared again when
            # it is next counted.  A self-dual code keeps one family, for both of its sectors.
            for index, local in ((0, local_x), (1, local_z)):
                basis_family = self._disjoint_basis_families.get(0 if self._self_dual else index)
                if basis_family is not None and not (shared and index == 1):
                    moved = scan.transport_basis_family(
                        basis_family, positions, local, counter._sectors[index][0]
                    )
                    if moved is not None:
                        counter._disjoint_basis_families[index] = moved
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
            self._enumerators = macwilliams.CheckEnumerators(
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

    def _get_disjoint_basis_family(self, index: int) -> _information_sets.BasisFamily:
        """The disjoint basis family of a sector, the first stage of its basis families, kept.

        This stage does not depend on the counted weight, so counting at another weight, as when a
        guessed weight fails, does not repeat it.
        """
        if index not in self._disjoint_basis_families:
            self._disjoint_basis_families[index] = _information_sets.get_disjoint_family(
                *self._get_sectors()[index]
            )
        return self._disjoint_basis_families[index]

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
    max_information_sets: SupportsIndex = _information_sets.MAX_SETS,
) -> CSSLogicalCounts:
    """Count the nontrivial logical operators of a given weight in a binary CSS code.

    This is shorthand for ``CSSLogicalCounter(code).count_logical_ops(...)``; see
    :class:`CSSLogicalCounter` and :meth:`CSSLogicalCounter.count_logical_ops` for the arguments and
    conventions.

    Returns:
        The counted weight and the counts.

    Raises:
        TypeError: If the code is not a :class:`~qldpc.codes.CSSCode`, or if
            ``max_information_sets`` is not an integer.
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


def compile_search_setup() -> None:
    """Compile the code that sets up exact distance and logical-operator count searches.

    By default, the inner loops of these searches are compiled by numba, and cached on disk, the
    first time that they are used.  The code that sets those loops up runs as ordinary Python.  It
    selects and bounds the information sets of a code, packs them for the scan of its words, and
    moves them through local gates.  This function compiles that code instead, for the rest of this
    process.  It does not change any result, and it does not affect the MacWilliams method, which
    does not use that code.

    Call it once, at the start of a program that makes very many counts of similar codes, such as a
    search over many deformed codes.  Every process that calls it compiles the code again, so it
    does not pay off for a few counts.  Counters may be created before or after the call.  The
    compiled code is never written to disk, so it cannot become outdated.
    """
    set_on_request_functions_compiled(True)
    try:
        _warm_up_compiled_functions()
    except BaseException:
        # Do not leave a process that was refused, or interrupted, partly compiled.
        set_on_request_functions_compiled(False)
        raise


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


def _warm_up_compiled_functions() -> None:
    """Compile every function now, by counting on a small code, so that no later call compiles."""
    from ..quantum import SteaneCode

    counter = CSSLogicalCounter(SteaneCode())
    counter.count_logical_ops(method="brouwer_zimmermann")
    counter.deformed_cnot(0, 1).count_logical_ops(method="brouwer_zimmermann")

    # The distance search of so small a code is exhaustive, so call the stages that it uses here.
    basis, labels = counter._get_sectors()[0]
    basis_family = _information_sets.prune_family(
        _information_sets.get_disjoint_family(basis, labels), 4
    )
    _information_sets.get_overlapping_family(
        basis,
        labels,
        basis_family,
        4,
        _information_sets.MAX_SETS,
        *_information_sets.get_overlapping_family_policy(len(basis)),
    )


####################################################################################################
# distances of the sectors


def _get_distance(
    sectors: Sequence[_Sector],
    method: DistanceMethod,
    get_disjoint_basis_families: Sequence[Callable[[], _information_sets.BasisFamily]],
) -> int:
    """Minimum weight of a nontrivial word in any of the given sectors.

    ``get_disjoint_basis_families`` holds a function for each sector that returns its disjoint
    basis family, for the BZ search.
    """
    if method == "brute_force":
        return min(_get_brute_force_distance(sector) for sector in sectors)
    return get_distance_brouwer_zimmermann_many(
        sectors,
        cutoff=0,
        block_size=_BLOCK_SIZE,
        get_disjoint_basis_families=get_disjoint_basis_families[: len(sectors)],
    )


def _get_brute_force_distance(sector: _Sector) -> int:
    """Minimum weight of a nontrivial word of one sector, by exhaustive enumeration."""
    basis, labels = sector
    eligible = np.any(labels, axis=1)
    return get_distance_quantum_brute_force(
        basis[eligible], basis[~eligible], cutoff=0, block_size=_BLOCK_SIZE, homogeneous=True
    )


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
