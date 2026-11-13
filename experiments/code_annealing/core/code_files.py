# SPDX-License-Identifier: Apache-2.0

"""Text files of codes, one block per code, and low-weight generators for them.

A block starts with a comment line ``# (n, k, d) m min-weight logicals`` and continues with one
line per stabilizer generator: X-type generators first, then Z-type generators, with ``_`` for
the identity and ``X`` or ``Z`` on the support.  Blocks are separated by blank lines::

    # (7, 1, 3) 14 min-weight logicals
    XXXX___
    _XX_XX_
    XX__X_X
    ZZZZ___
    _ZZ_ZZ_
    ZZ__Z_Z

``m`` is the number of minimum-weight X-only plus Z-only logical operators.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from qldpc.codes import CSSCode
from qldpc.codes.distance import NO_DISTANCE, CSSLogicalCounter, CSSLogicalCounts

from .atomic import write_text_atomic

# The parameters (n, k, d) of a code: its numbers of qubits and logical qubits, and its distance.
CodeParams: TypeAlias = tuple[int, int, int]

_HEADER = re.compile(r"#\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*(\d+)\s*min-weight logicals")


@dataclass(frozen=True)
class CodeEntry:
    """A code from a text file, with the parameters that its header claims."""

    path: Path
    """The file that holds the code."""

    code: CSSCode
    distance: int
    """The distance that the header claims."""

    multiplicity: int
    """The number of minimum-weight X-only plus Z-only logical operators that the header claims."""

    @property
    def num_qubits(self) -> int:
        return self.code.num_qubits

    @property
    def num_logicals(self) -> int:
        return self.code.dimension

    @property
    def params(self) -> CodeParams:
        """The parameters ``(n, k, d)`` that the header claims."""
        return self.num_qubits, self.num_logicals, self.distance


def format_code(checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, distance: int, count: int) -> str:
    """The text block of a code with the given binary X and Z generators, under its header.

    Args:
        checks_x: Binary X-type generators, one row per generator.
        checks_z: Binary Z-type generators, one row per generator.
        distance: Distance d of the code, for the header.
        count: Number m of minimum-weight X-only plus Z-only logical operators, for the header.
    """
    rows_x, rows_z = (np.asarray(checks, dtype=np.uint8) for checks in (checks_x, checks_z))
    num_qubits = (rows_x if len(rows_x) else rows_z).shape[1]
    num_logicals = num_qubits - len(rows_x) - len(rows_z)
    header = f"# ({num_qubits}, {num_logicals}, {distance}) {count} min-weight logicals"
    return header + "\n" + format_generators(rows_x, rows_z)


def format_generators(checks_x: npt.ArrayLike, checks_z: npt.ArrayLike) -> str:
    """The lines of binary X and Z generators, such as ``XX_X`` and ``_ZZ_``, one per generator."""
    lines: list[str] = []
    for pauli, checks in (("X", checks_x), ("Z", checks_z)):
        rows = np.asarray(checks, dtype=np.uint8)
        lines.extend("".join(pauli if bit else "_" for bit in row) for row in rows)
    return "\n".join(lines) + "\n"


def write_code_file(path: str | Path, blocks: Iterable[str]) -> None:
    """Write text blocks from ``format_code``, separated by blank lines, to a file, atomically."""
    write_text_atomic(path, "\n".join(blocks))


def read_code_file(path: str | Path) -> list[CodeEntry]:
    """Read the codes of a text file.

    The header of each block claims the parameters of its code.  Reading checks that the generators
    have the claimed number of qubits and encode the claimed number of logical qubits, but trusts
    the claimed distance and count, which only an exact recount can check.
    """
    entries = []
    for block in split_blocks(Path(path).read_text()):
        # The header, which claims the parameters of the code.
        number, header = block[0]
        match = _HEADER.fullmatch(header)
        if match is None:
            raise ValueError(
                f"{path}, line {number}: a code block must start with a header such as"
                f" '# (7, 1, 3) 14 min-weight logicals', not {header!r}"
            )
        num_qubits, num_logicals, distance, count = map(int, match.groups())

        # The generators, each of one Pauli type on the claimed number of qubits.
        rows_x: list[list[bool]] = []
        rows_z: list[list[bool]] = []
        for number, line in block[1:]:
            paulis = set(line) - {"_"}
            if len(line) != num_qubits or paulis not in ({"X"}, {"Z"}):
                raise ValueError(
                    f"{path}, line {number}: a generator must be X-type or Z-type with"
                    f" {num_qubits} qubits, not {line!r}"
                )
            (rows_x if paulis == {"X"} else rows_z).append([char != "_" for char in line])

        # The code, which must encode the claimed number of logical qubits.
        code = CSSCode(_as_matrix(rows_x, num_qubits), _as_matrix(rows_z, num_qubits))
        if code.dimension != num_logicals:
            raise ValueError(
                f"{path}, line {block[0][0]}: a header claims {num_logicals} logical qubits, but"
                f" its code has {code.dimension}"
            )
        entries.append(CodeEntry(Path(path), code, distance, count))
    return entries


def recount(entry: CodeEntry) -> CSSLogicalCounts:
    """Count the minimum-weight logical operators of a code exactly, and check its header.

    Raises:
        ValueError: If the code encodes no logical qubits, or if the distance or count differs from
            the claims of the header.
    """
    counts = CSSLogicalCounter.from_code(entry.code).count_logical_ops()
    if counts.weight is NO_DISTANCE:
        raise ValueError(f"{entry.path}: the code {entry.params} encodes no logical qubits")
    check_header(entry, counts.weight, counts.count_x + counts.count_z)
    return counts


def check_header(entry: CodeEntry, distance: int, count: int) -> None:
    """Check a distance and count of a code, found independently, against those of its header.

    Raises:
        ValueError: If either differs from the claims of the header.
    """
    if (distance, count) != (entry.distance, entry.multiplicity):
        raise ValueError(
            f"{entry.path}: the header of the code [[{entry.num_qubits}, {entry.num_logicals}]]"
            f" claims (d, count) = {(entry.distance, entry.multiplicity)}, but the code has"
            f" {(distance, count)}, so the header is wrong"
        )


def split_blocks(text: str) -> list[list[tuple[int, str]]]:
    """The lines of each block of a text, where blank lines, or lines of spaces, separate blocks.

    Returns:
        For each block, its nonblank lines, stripped, with their line numbers from one.
    """
    blocks: list[list[tuple[int, str]]] = [[]]
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            blocks[-1].append((number, line.strip()))
        elif blocks[-1]:
            blocks.append([])
    return [block for block in blocks if block]


def get_reduced_weights(code: CSSCode) -> list[int]:
    """The weights of the X-type and then the Z-type generators of a code, after reduce_weights."""
    return [
        int(row.sum())
        for checks in (code.matrix_x, code.matrix_z)
        for row in reduce_weights(checks.view(np.ndarray))
    ]


def reduce_weights(matrix: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Lower the weights of independent generators by adding rows to each other.

    A row is replaced by its sum with another row whenever that is lighter, until no sum is.  The
    result spans the same space, with greedily low weights; it need not be the lowest possible.
    """
    rows = np.array(matrix, dtype=np.uint8)
    improved = True
    while improved:
        improved = False
        for ii in range(len(rows)):
            sums = rows ^ rows[ii]
            sums[ii] = rows[ii]
            weights = sums.sum(axis=1, dtype=int)
            jj = int(np.argmin(weights))
            if weights[jj] < rows[ii].sum(dtype=int):
                rows[ii] = sums[jj]
                improved = True
    return rows


####################################################################################################
# helper methods


def _as_matrix(rows: Sequence[Sequence[bool]], width: int) -> npt.NDArray[np.int_]:
    return np.array(rows, dtype=np.int_).reshape(-1, width)
