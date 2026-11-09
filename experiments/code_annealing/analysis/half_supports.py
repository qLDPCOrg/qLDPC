# SPDX-License-Identifier: Apache-2.0

"""Count the half and majority supports of the minimum-weight logical operators of codes.

These counts estimate the logical error rates of the annealed codes at low physical error rates (see
``figures/plot_error_estimates.py``).  The simulations decode each sector by a lookup table of the
errors of weight up to ``(d - 1) // 2`` and discard all other syndromes, so the decoder completes a
minimum-weight logical operator when ``d // 2 + 1`` qubits of its support err.  Call the subsets of
``(d + 1) // 2`` qubits of a support its half supports, and those of ``d // 2 + 1`` qubits its
majority supports.  For odd ``d``, they are the same.  For even ``d``, they count different events,
so both are counted: errors on all qubits of a half support are discarded.  Errors on a majority
support are logical errors, which the estimates of logical error rates use.

Run with ``python -m experiments.code_annealing.analysis.half_supports [--max-qubits N]``.  For
every code of the two text files of annealed codes, the script prints one line, and saves the same
text to ``data/half_supports.txt`` of the results directory (with ``--max-qubits``, which selects
only some of the codes, it only prints)::

    [n, k, d] mx/mz hx/hz

where ``d`` is the distance, and ``mx`` and ``mz`` are the numbers of minimum-weight X-only and
Z-only logical operators, which are the numbers of their distinct supports.  The numbers ``hx`` and
``hz`` are those of the half supports, the distinct subsets of ``(d + 1) // 2`` qubits that lie in
the supports of the X-only and Z-only operators, respectively.  For even ``d``, the numbers ``gx``
and ``gz`` of majority supports, of ``d // 2 + 1`` qubits, follow in parentheses.
Everything is recomputed exactly, and a code whose distance or number of minimum-weight logical
operators differs from the claims of its code file is an error.  The text has one block for each
code file: a line with the file name, a line for each code, and a blank line.  For example::

    annealed_codes.txt
    [7, 1, 3] 7/7 21/21
    [20, 4, 4] 5/5 30/30 (20/20)

``read_half_supports`` reads the text back for the figure scripts.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TypeAlias

from ..core.atomic import write_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeParams, check_header, read_code_file, split_blocks
from ..core.paths import find_code_files, get_half_supports_file
from .min_weight_supports import MinWeightSupports, count_support_subsets, get_min_weight_supports

# A line of the file: [n, k, d] mx/mz hx/hz, and for even d (gx/gz) after that.
_CODE_LINE = re.compile(
    r"\[(\d+),\s*(\d+),\s*(\d+)\]\s+(\d+)/(\d+)\s+(\d+)/(\d+)(?:\s+\((\d+)/(\d+)\))?"
)


def main(argv: Sequence[str] | None = None) -> None:
    """Print the line of every code, and write the same text to a file unless codes are skipped."""
    parser = make_parser(__doc__)
    parser.add_argument("--max-qubits", type=int, help="skip the codes with more qubits")
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    files = find_code_files(args.results_dir)
    output = get_half_supports_file(args.results_dir)

    def print_lines(stream: IO[str] | None) -> None:
        for line in _get_lines(files.values(), args.max_qubits):
            print(line, flush=True)
            if stream is not None:
                stream.write(line + "\n")

    if args.max_qubits is None:
        write_atomic(output, print_lines)
    else:
        print_lines(None)
        print(f"Wrote nothing to {output}, because --max-qubits skips codes", file=sys.stderr)


def _get_lines(paths: Iterable[Path], max_qubits: int | None) -> Iterator[str]:
    """The lines to print: a file name, a line for each code of the file, and a blank line."""
    for path in paths:
        yield path.name
        for entry in read_code_file(path):
            if max_qubits is None or entry.num_qubits <= max_qubits:
                supports = get_min_weight_supports(entry.code)
                # The header claims come from another count, by collect_annealed_codes or the
                # paper's files, so this enumeration checks them independently.
                count = len(supports.supports_x) + len(supports.supports_z)
                check_header(entry, supports.distance, count)
                yield _format_line(entry.num_qubits, entry.num_logicals, supports)
        yield ""


def _format_line(num_qubits: int, num_logicals: int, supports: MinWeightSupports) -> str:
    """The line of a code with the given supports of its minimum-weight logical operators."""
    distance = supports.distance
    sizes = [(distance + 1) // 2] if distance % 2 else [distance // 2, distance // 2 + 1]
    counts = [
        f"{count_support_subsets(supports.supports_x, size)}"
        f"/{count_support_subsets(supports.supports_z, size)}"
        for size in sizes
    ]
    line = (
        f"[{num_qubits}, {num_logicals}, {distance}]"
        f" {len(supports.supports_x)}/{len(supports.supports_z)} {counts[0]}"
    )
    return line if len(counts) == 1 else f"{line} ({counts[1]})"


####################################################################################################
# reading the file


@dataclass(frozen=True)
class HalfSupportCounts:
    """The numbers of minimum-weight logical operators of a code and of subsets of their supports.

    There are ``count_x`` X-only and ``count_z`` Z-only minimum-weight logical operators.  For a
    code of distance ``d``, the half supports are the distinct subsets of size ``(d + 1) // 2`` of
    their supports, so that a subset that several operators share counts once.  The majority
    supports are the distinct subsets of size ``d // 2 + 1``, which are the half supports if ``d``
    is odd.
    """

    num_qubits: int
    num_logicals: int
    distance: int
    count_x: int
    count_z: int
    num_half_supports_x: int
    num_half_supports_z: int
    num_majority_supports_x: int
    num_majority_supports_z: int


# counts by the name of the code file and the code parameters (n, k, d)
HalfSupportCountsByFile: TypeAlias = dict[str, dict[CodeParams, HalfSupportCounts]]


def read_half_supports(path: str | Path) -> HalfSupportCountsByFile:
    """Read the counts of a text file that this script writes.

    A line with a file name starts a block, each following line is a code, and a blank line ends
    the block.  A code line is ``[n, k, d] mx/mz hx/hz``, followed by ``(gx/gz)`` if and only if
    d is even.  See the module docstring for the meaning of these numbers.

    Args:
        path: The text file to read.

    Returns:
        The counts of each code, by the name of the code file and then by the code parameters
        ``(n, k, d)``.

    Raises:
        ValueError: If a code line is malformed, or a code file or code is listed twice.
    """
    counts: HalfSupportCountsByFile = {}
    for (number, name), *lines in split_blocks(Path(path).read_text()):
        if name in counts:
            raise ValueError(f"line {number}: the file {name!r} is listed twice")
        block = counts[name] = {}
        for number, line in lines:
            code = _parse_code_line(line, number)
            params = (code.num_qubits, code.num_logicals, code.distance)
            if params in block:
                raise ValueError(f"line {number}: the code {list(params)} is listed twice")
            block[params] = code
    return counts


def _parse_code_line(line: str, number: int) -> HalfSupportCounts:
    """Parse a line ``[n, k, d] mx/mz hx/hz`` with ``(gx/gz)`` added for even d."""
    match = _CODE_LINE.fullmatch(line)
    if match is None:
        raise ValueError(f"line {number}: expected '[n, k, d] mx/mz hx/hz', not {line!r}")
    *values, majority_x, majority_z = match.groups()
    num_qubits, num_logicals, distance, count_x, count_z, half_x, half_z = map(int, values)
    if (majority_x is None) != (distance % 2 == 1):
        raise ValueError(
            f"line {number}: the counts of subsets of size {distance // 2 + 1} in parentheses"
            f" must follow an even distance, and only an even distance: {line!r}"
        )
    return HalfSupportCounts(
        num_qubits,
        num_logicals,
        distance,
        count_x,
        count_z,
        half_x,
        half_z,
        half_x if majority_x is None else int(majority_x),
        half_z if majority_z is None else int(majority_z),
    )


if __name__ == "__main__":
    main()
