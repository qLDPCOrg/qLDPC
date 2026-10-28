# SPDX-License-Identifier: Apache-2.0

"""Find the best known codes of the qecdb.org database next to each annealed code.

Run with ``python -m experiments.code_annealing.analysis.scrape_qecdb``, which needs network access.

For every annealed code ``[[n, k, d]]`` of the two code files, this finds the best known
neighbors in the database: the smallest ``n`` with at least the same ``k`` and ``d``, the largest
``k`` with at most the same ``n`` and at least the same ``d``, and the largest ``d`` with at most
the same ``n`` and at least the same ``k``.  CSS codes are compared with the CSS codes of the
database, and SWEL codes with its self-dual CSS codes.  The script prints, and saves to the data
file ``data/scrape_qecdb_YYYY-MM-DD.txt`` of the results directory, one line
``(n, k, d) n_best k_best d_best`` per annealed code, in one block per code file, with ``None`` for
a neighbor that the database does not have.  The distance of an annealed code is the one that its
code file claims, and that of a database code is the one that the database lists.

The database grows, and may come to list the annealed codes themselves, so a data file records the
database on the date of its scrape (in UTC), which its name carries.  ``latex_tables.py`` reads the
newest data file, and earlier ones are kept, so that a table can be rebuilt from the snapshot behind
it.  The snapshot behind the paper's table is in ``paper_data/``.

Scraping needs no packages beyond the standard library.  For each annealed code, it asks the JSON
API of the database three questions, one for each neighbor, whereas
``qldpc.codes.QuditCode.from_qecdb_id`` retrieves the code of one ID.  Each question restricts the
codes by parameter bounds and sorts them by the neighbor's parameter, so the database does the
search and one code of the answer is enough.  The requests are paced by ``_REQUEST_DELAY`` seconds,
and the data file is written only after all of them have succeeded.
"""

from __future__ import annotations

import datetime
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.atomic import write_text_atomic
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeParams, read_code_file
from ..core.paths import find_code_files, get_data_dir
from ..core.proposals import CODE_TYPES, CodeType

_DATA_FILE_PATTERN = "scrape_qecdb_????-??-??.txt"
_QECDB_API_URL = "https://qecdb.org/api/codes"

_REQUEST_TIMEOUT = 30
"""Seconds to wait for the database to answer."""

_REQUEST_DELAY = 0.25
"""Seconds to wait before every request, to be polite to the database."""


_SUMMARY_PATTERN = re.compile(r"\((\d+), (\d+), (\d+)\) (\d+|None) (\d+|None) (\d+|None)")


def main(argv: Sequence[str] | None = None) -> None:
    """Ask the database about every annealed code, and save the data file of today."""
    parser = make_parser(__doc__)
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    today = datetime.datetime.now(datetime.UTC).date()
    data_file = get_data_dir(args.results_dir) / f"scrape_qecdb_{today.isoformat()}.txt"
    try:
        _scrape_data_file(find_code_files(args.results_dir), data_file)
    except RuntimeError as error:
        raise SystemExit(f"Scraping failed, and nothing was written: {error}") from error


def find_data_file(results_dir: Path) -> Path | None:
    """The newest data file of the results directory, by the date in its name, or None."""
    data_files = sorted(get_data_dir(results_dir).glob(_DATA_FILE_PATTERN))
    return data_files[-1] if data_files else None


@dataclass(frozen=True)
class BestKnown:
    """The parameters of an annealed code and of its best known neighbors in the database."""

    params: CodeParams
    n_best: int | None
    """The smallest ``n`` with at least the same ``k`` and ``d``."""
    k_best: int | None
    """The largest ``k`` with at most the same ``n`` and at least the same ``d``."""
    d_best: int | None
    """The largest ``d`` with at most the same ``n`` and at least the same ``k``."""

    def __str__(self) -> str:
        """The line of the data file."""
        return f"{self.params} {self.n_best} {self.k_best} {self.d_best}"

    @property
    def wins(self) -> tuple[bool, bool, bool]:
        """Whether the annealed code has a smaller ``n``, a larger ``k`` and a larger ``d``.

        Each compares with its neighbor, and a neighbor that the database lacks is beaten.
        """
        nn, kk, dd = self.params
        return (
            self.n_best is None or nn < self.n_best,
            self.k_best is None or kk > self.k_best,
            self.d_best is None or dd > self.d_best,
        )

    @staticmethod
    def parse(line: str) -> BestKnown:
        """Read a line of the data file."""
        match = _SUMMARY_PATTERN.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"Not a line of the form '(n, k, d) n_best k_best d_best': {line!r}")
        nn, kk, dd = (int(text) for text in match.groups()[:3])
        n_best, k_best, d_best = (
            None if text == "None" else int(text) for text in match.groups()[3:]
        )
        return BestKnown((nn, kk, dd), n_best, k_best, d_best)


def is_up_to_date(code_files: Mapping[CodeType, Path], data_file: Path) -> bool:
    """Does a data file describe exactly the codes of the code files, in the same order?"""
    return [
        [result.params for result in block] for block in _read_blocks(data_file)
    ] == _get_expected_params(code_files)


def read_best_known(
    code_files: Mapping[CodeType, Path], data_file: Path
) -> dict[CodeType, list[BestKnown]]:
    """The neighbors of every code of the code files, by code type, in the order of the files.

    Raises:
        ValueError: If the data file does not describe exactly the codes of the code files.
    """
    if not is_up_to_date(code_files, data_file):
        raise ValueError(
            f"{data_file} is out of date, because the codes changed: run collect_data --scrape"
        )
    return dict(zip(_sorted_code_types(code_files), _read_blocks(data_file), strict=True))


def query_best_known(params: CodeParams, code_type: CodeType) -> BestKnown:
    """The best known neighbors of a code ``(n, k, d)`` of a code type in the database.

    The ``css`` code type holds all of the CSS codes of the database, and the ``swel`` code type
    those that are self-dual.  Each neighbor is one request.

    Raises:
        RuntimeError: If the database cannot be reached or answers in an unexpected format.
    """
    nn, kk, dd = params
    return BestKnown(
        params,
        _query_extreme("n", "asc", code_type, kMin=kk, dMin=dd),
        _query_extreme("k", "desc", code_type, nMax=nn, dMin=dd),
        _query_extreme("d", "desc", code_type, nMax=nn, kMin=kk),
    )


####################################################################################################
# helper methods


def _sorted_code_types(code_files: Mapping[CodeType, Path]) -> list[CodeType]:
    """The code types of the code files, in the order of the blocks of the data file."""
    return sorted(code_files, key=CODE_TYPES.index)


def _get_expected_params(code_files: Mapping[CodeType, Path]) -> list[list[CodeParams]]:
    """The parameters of the codes of the code files, in the blocks of the data file."""
    return [
        [
            (entry.num_qubits, entry.num_logicals, entry.distance)
            for entry in read_code_file(code_files[code_type])
        ]
        for code_type in _sorted_code_types(code_files)
    ]


def _read_blocks(data_file: Path) -> list[list[BestKnown]]:
    """The blocks of a data file, which are separated by blank lines.

    Raises:
        ValueError: If a line is not a line of the data file, naming the file and the line.
    """
    blocks: list[list[BestKnown]] = [[]]
    for number, line in enumerate(data_file.read_text().splitlines(), start=1):
        if not line.strip():
            if blocks[-1]:
                blocks.append([])
            continue
        try:
            blocks[-1].append(BestKnown.parse(line))
        except ValueError as error:
            raise ValueError(f"{data_file}, line {number}: {error}") from error
    return [block for block in blocks if block]


def _scrape_data_file(code_files: Mapping[CodeType, Path], data_file: Path) -> None:
    """Ask the database about every annealed code, print the neighbors, and save them to a file.

    The code files are read first, and the file is written after the last request, so a failure
    keeps the previous file.
    """
    entries = {code_type: read_code_file(code_file) for code_type, code_file in code_files.items()}
    blocks: list[list[BestKnown]] = []
    for code_type in _sorted_code_types(code_files):
        if blocks:
            print()
        block = []
        for entry in entries[code_type]:
            params = (entry.num_qubits, entry.num_logicals, entry.distance)
            block.append(query_best_known(params, code_type))
            print(block[-1], flush=True)
        blocks.append(block)
    write_text_atomic(
        data_file, "\n".join("".join(f"{result}\n" for result in block) for block in blocks)
    )


def _query_extreme(field: str, direction: str, code_type: CodeType, **bounds: int) -> int | None:
    """The first value of a parameter of the codes within bounds, sorted by that parameter.

    The bounds are query parameters of the database such as ``nMax`` or ``kMin``, and ``None`` means
    that no code matches.

    Raises:
        RuntimeError: If the database cannot be reached, answers in an unexpected format, or does
            not list the parameter of the first code that it returns.
    """
    query: dict[str, str | int] = {"cssOnly": 1, "sort": field, "dir": direction, **bounds}
    if code_type == "swel":
        query["selfdual"] = 1
    time.sleep(_REQUEST_DELAY)
    data = _fetch_json(
        f"{_QECDB_API_URL}?{urllib.parse.urlencode({**query, 'page': 0, 'pageSize': 1})}"
    )
    try:
        items = data["items"]
        value = items[0].get(field) if items else None
    except (AttributeError, KeyError, TypeError) as exception:
        raise RuntimeError("qecdb.org answered in an unexpected format") from exception
    if not items:
        return None
    # an integer, or the decimal text of one; anything else, even a float, is a malformed answer
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise RuntimeError(
        f"Could not read the {field} of the first code that qecdb.org lists: {value!r}"
    )


def _fetch_json(url: str) -> Any:
    """Retrieve JSON data from a URL."""
    try:
        with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exception:
        raise RuntimeError(f"Cannot access {url}") from exception
    except ValueError as exception:
        raise RuntimeError(
            f"qecdb.org answered {url} with something other than JSON"
        ) from exception


if __name__ == "__main__":
    main()
